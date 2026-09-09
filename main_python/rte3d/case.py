"""Read and solve an arbitrary case supplied as a NetCDF file.

The file layout is the one the test executable of rte-rrtmgp-cpp reads, so cases
written for it run here unchanged:

    dimensions   x, y, lay, lev            (or col, lay, lev for a flat column list)
    profiles     p_lay, p_lev, t_lay, t_lev            (lay|lev, y, x)
    gases        vmr_<name>                            scalar, (lay), or (lay, y, x)
    clouds       lwp, iwp, rel, dei                    (lay, y, x)
    longwave     t_sfc (y, x), emis_sfc (y, x, band_lw)
    shortwave    mu0 (y, x), sfc_alb_dir, sfc_alb_dif  (y, x, band_sw)
                 tsi (y, x) or tsi_scaling             (scalar)
    optional     col_dry                               (lay, y, x)

Columns are flattened as x fastest, which is what `(lay, y, x)` already is in memory,
so the reshape is free and the output can be folded back the same way.

Ice size is `dei`, an effective *diameter*, as RRTMGP's current coefficient files
document it; `rei` is accepted as a fallback name for the same quantity and passed
through unchanged. Aerosols are not read: rte3d has no aerosol optics yet.
"""
import numpy as np
import xarray as xr

# The single-angle longwave quadrature, gauss_Ds and gauss_wts in mo_rte_lw.F90.
LW_SECANT = 1.0/0.6096748751


def read_case(path):
    """Read one case file into the arrays rte3d takes.

    Everything is (nlay, ncol) or (nlev, ncol); the boundary conditions are (ncol,) or
    (nbnd, ncol). Only what the file provides is present in the result, so a clear-sky
    or longwave-only case is simply a file with fewer variables.
    """
    d = xr.open_dataset(path)

    gridded = 'col' not in d.sizes
    if gridded:
        nx = int(d.sizes.get('x', 1))
        ny = int(d.sizes.get('y', 1))
    else:
        nx, ny = int(d.sizes['col']), 1
    ncol = nx*ny

    def field(name):
        """A (lay|lev, y, x) field as (nlay|nlev, ncol)."""
        return np.ascontiguousarray(d[name].values.astype(np.float64).reshape(-1, ncol))

    def surface(name):
        return np.ascontiguousarray(d[name].values.astype(np.float64).reshape(ncol))

    def by_band(name):
        """A (y, x, band) field as (nbnd, ncol). The file has the band fastest."""
        v = d[name].values.astype(np.float64).reshape(ncol, -1)
        return np.ascontiguousarray(v.T)

    # How a column index folds back to the file's own horizontal layout, so the
    # output can be written the way the input was read.
    atm = dict(nx=nx, ny=ny, ncol=ncol,
               dims=('y', 'x') if gridded else ('col',),
               shape=(ny, nx) if gridded else (ncol,),
               play=field('p_lay'), plev=field('p_lev'),
               tlay=field('t_lay'), tlev=field('t_lev'))

    atm['nlay'] = atm['play'].shape[0]
    atm['nlev'] = atm['plev'].shape[0]

    # Which end of the profile is the top. Both orientations are supported throughout,
    # so this is read from the data rather than assumed.
    atm['top_at_1'] = bool(atm['play'][0, 0] < atm['play'][-1, 0])

    if 'col_dry' in d:
        atm['col_dry'] = field('col_dry')

    atm['gases'] = {}
    for name, var in d.variables.items():
        if not name.startswith('vmr_'):
            continue
        gas = name[len('vmr_'):]
        values = var.values.astype(np.float64)
        if values.ndim == 0:
            atm['gases'][gas] = float(values)
        elif values.ndim == 1:
            atm['gases'][gas] = np.ascontiguousarray(values)
        else:
            atm['gases'][gas] = field(name)

    ice = 'dei' if 'dei' in d else 'rei'
    if 'lwp' in d:
        atm.update(lwp=field('lwp'), iwp=field('iwp'),
                   rel=field('rel'), dei=field(ice))

    if 't_sfc' in d:
        atm['tsfc'] = surface('t_sfc')
    if 'emis_sfc' in d:
        atm['sfc_emis'] = by_band('emis_sfc')

    if 'mu0' in d:
        atm['mu0'] = surface('mu0')
    if 'sfc_alb_dir' in d:
        atm['sfc_alb_dir'] = by_band('sfc_alb_dir')
        atm['sfc_alb_dif'] = by_band(
            'sfc_alb_dif' if 'sfc_alb_dif' in d else 'sfc_alb_dir')

    # The incoming solar flux is given either as an absolute irradiance per column or
    # as a factor on the k-distribution's own; both end up as a per-column factor once
    # the k-distribution is known.
    if 'tsi' in d:
        atm['tsi'] = surface('tsi')
    elif 'tsi_scaling' in d:
        atm['tsi_scaling'] = np.full(ncol, float(d['tsi_scaling'].values))

    return atm


def make_gas_concs(rte3d, atm, gas_names=None):
    """The gases the file provides, optionally filtered to those a k-distribution knows.

    A k-distribution rejects gases it has no coefficients for, and the longwave and
    shortwave ones do not carry the same list, so pass the list from the file dict.
    """
    g = rte3d.Gas_concs()
    for name, value in atm['gases'].items():
        if gas_names is None or name in gas_names:
            g.set_vmr(name, value)

    return g


def gpoint_bands(f):
    """0-based band index for every g-point, from a k-distribution file dict.

    The file's band2gpt is 1-based and inclusive on both ends, as stored.
    """
    band2gpt = f['band2gpt']
    out = np.empty(int(band2gpt.max()), dtype=np.intp)
    for band, (first, last) in enumerate(band2gpt):
        out[first-1:last] = band

    return out


def expand_bands(per_band, gpt_band):
    """(nbnd, ncol) boundary condition -> the (ngpt, ncol) the solvers take.

    The band count is checked here: a file whose surface fields are resolved on the
    other band's grid indexes cleanly and would otherwise be used as it stands.
    """
    nbnd = int(gpt_band[-1]) + 1
    if per_band.shape[0] != nbnd:
        raise ValueError(
            f'A surface field is given on {per_band.shape[0]} bands where this '
            f'k-distribution has {nbnd}.')

    return np.ascontiguousarray(per_band[gpt_band])


def cloud_props(rte3d, cloud_optics, atm, two_stream, delta_scale=True):
    """Cloud optics for the case, band-resolved.

    The solvers take the cloud properties by band, so the coefficient file has to be a
    -bnd one. Longwave clouds are absorption only; shortwave ones are delta-scaled
    before they are added, as the reference driver does.
    """
    if not two_stream:
        return dict(cloud_tau=rte3d.cloud_optics(
            cloud_optics, clwp=atm['lwp'], ciwp=atm['iwp'],
            reliq=atm['rel'], reice=atm['dei'], two_stream=False))

    tau, ssa, g = rte3d.cloud_optics(
        cloud_optics, clwp=atm['lwp'], ciwp=atm['iwp'],
        reliq=atm['rel'], reice=atm['dei'])

    if delta_scale:
        tau, ssa, g = rte3d.delta_scale_2str(tau, ssa, g)

    return dict(cloud_tau=tau, cloud_ssa=ssa, cloud_g=g)


def solve_lw(rte3d, kdist, gas_concs, atm, gpt_band,
             cloud_optics=None, byband=False):
    """Longwave fluxes for the case. Returns the solver's dict as it stands."""
    clouds = cloud_props(rte3d, cloud_optics, atm, False) if cloud_optics else {}

    return rte3d.solve_lw(
        kdist, gas_concs, atm['top_at_1'],
        atm['play'], atm['plev'], atm['tlay'], atm['tlev'], atm['tsfc'],
        secants=np.full((1, atm['ncol']), LW_SECANT),
        weights=np.array([1.0]),
        sfc_emis=expand_bands(atm['sfc_emis'], gpt_band),
        col_dry=atm.get('col_dry'), byband=byband, **clouds)


def solve_sw(rte3d, kdist, gas_concs, atm, gpt_band,
             cloud_optics=None, byband=False, delta_cloud=True):
    """Shortwave fluxes for the case.

    Night-time columns -- those with mu0 at or below zero -- are solved with mu0 = 1
    and zeroed afterwards, as the reference drivers do; their fluxes are meaningless
    rather than wrong.
    """
    clouds = (cloud_props(rte3d, cloud_optics, atm, True, delta_cloud)
              if cloud_optics else {})

    ngpt, ncol, nlay = kdist.ngpt, atm['ncol'], atm['nlay']

    daytime = atm['mu0'] > 0.0
    mu0 = np.where(daytime, atm['mu0'], 1.0)

    solar = np.array(kdist.solar_source, dtype=np.float64)
    toa = np.broadcast_to(solar[:, None], (ngpt, ncol))*scaling(kdist, atm)[None, :]

    out = rte3d.solve_sw(
        kdist, gas_concs, atm['top_at_1'],
        atm['play'], atm['plev'], atm['tlay'],
        mu0=np.ascontiguousarray(np.broadcast_to(mu0[None, :], (nlay, ncol))),
        sfc_alb_dir=expand_bands(atm['sfc_alb_dir'], gpt_band),
        sfc_alb_dif=expand_bands(atm['sfc_alb_dif'], gpt_band),
        inc_flux_dir=np.ascontiguousarray(toa),
        col_dry=atm.get('col_dry'), byband=byband, **clouds)

    for name, flux in out.items():
        flux[..., ~daytime] = 0.0

    return out


def scaling(kdist, atm):
    """Per-column factor on the k-distribution's own solar source.

    A case that gives `tsi` sets the absolute irradiance and the factor follows from
    the k-distribution's total; one that gives `tsi_scaling` sets the factor directly;
    a case that gives neither uses the k-distribution unscaled.
    """
    if 'tsi' in atm:
        return atm['tsi']/np.array(kdist.solar_source, dtype=np.float64).sum()
    if 'tsi_scaling' in atm:
        return atm['tsi_scaling']

    return np.ones(atm['ncol'])
