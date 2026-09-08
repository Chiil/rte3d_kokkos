"""The RRTMGP all-sky case, without aerosols.

Unlike RFMIP this is a cloudy calculation, so it exercises cloud optics and the
by-band increments. It is also stored surface-first -- `top_at_1` is False -- which is
the opposite orientation to RFMIP, so between them the two cases cover both branches
of every solver.

The reference files in extern/rrtmgp-data carry both the inputs and the fluxes
RRTMGP produced from them, in (layer, column) order, which is already rte3d's.
"""
import numpy as np
import xarray as xr

# Fixed concentrations from examples/all-sky/rrtmgp_allsky.F90. Water vapour and ozone
# come from the profile instead.
FIXED_GASES = dict(co2=348e-6, ch4=1650e-9, n2o=306e-9, n2=0.7808, o2=0.2095, co=0.0)

# Boundary conditions, also from the driver.
SFC_EMIS = 0.98
SFC_ALBEDO = 0.06
MU0 = 0.86

# The driver calls set_ice_roughness(2), which is 1-based; the middle of the three ice
# roughness types, not the default first one. Getting this wrong moves the shortwave
# fluxes by several W/m2, so it is not a detail.
ICERGH = 1


def read_allsky(path):
    """Read one all-sky reference file: inputs and the fluxes to compare against."""
    d = xr.open_dataset(path)

    def f8(name):
        return np.ascontiguousarray(d[name].values.astype(np.float64))

    out = dict(
        play=f8('p_lay'), plev=f8('p_lev'),
        tlay=f8('t_lay'), tlev=f8('t_lev'),
        lwp=f8('lwp'), iwp=f8('iwp'), rel=f8('rel'), rei=f8('rei'),
        h2o=f8('h2o'), o3=f8('o3'),
    )

    # Layer 0 is at the surface here, so the surface temperature is the first level.
    out['tsfc'] = np.ascontiguousarray(out['tlev'][0].copy())

    for name in ('lw_flux_up', 'lw_flux_dn', 'sw_flux_up', 'sw_flux_dn', 'sw_flux_dir'):
        if name in d:
            out[name] = f8(name)

    return out


def make_gas_concs(rte3d, atm):
    g = rte3d.Gas_concs()
    g.set_vmr('h2o', atm['h2o'])
    g.set_vmr('o3', atm['o3'])
    for name, value in FIXED_GASES.items():
        g.set_vmr(name, value)

    return g


def solve_lw(rte3d, kdist, cloud_optics, gas_concs, atm):
    """Longwave all-sky fluxes.

    Clouds are band-resolved and absorption-only here, matching the driver: the
    longwave solver used is the no-scattering one, so the cloud contribution enters as
    an absorption optical depth added by band.
    """
    cloud_tau = rte3d.cloud_optics(
        cloud_optics, clwp=atm['lwp'], ciwp=atm['iwp'],
        reliq=atm['rel'], reice=atm['rei'], two_stream=False)

    ngpt = kdist.ngpt
    ncol = atm['play'].shape[1]

    out = rte3d.solve_lw(
        kdist, gas_concs,
        False,  # layer 0 is at the surface
        atm['play'], atm['plev'], atm['tlay'], atm['tlev'], atm['tsfc'],
        secants=np.full((1, ncol), 1.0/0.6096748751),
        weights=np.array([1.0]),
        sfc_emis=np.full((ngpt, ncol), SFC_EMIS),
        cloud_tau=cloud_tau)

    return out['flux_up'], out['flux_dn']


def solve_sw(rte3d, kdist, cloud_optics, gas_concs, atm):
    """Shortwave all-sky fluxes.

    Cloud properties are delta-scaled before being added, as the driver does. They stay
    band-resolved: the solve takes the slice for each g-point's own band.
    """
    cloud_tau, cloud_ssa, cloud_g = rte3d.cloud_optics(
        cloud_optics, clwp=atm['lwp'], ciwp=atm['iwp'],
        reliq=atm['rel'], reice=atm['rei'])

    cloud_tau, cloud_ssa, cloud_g = rte3d.delta_scale_2str(cloud_tau, cloud_ssa, cloud_g)

    ngpt = kdist.ngpt
    nlay, ncol = atm['play'].shape

    # The k-distribution's solar source is used as it stands: the all-sky case does not
    # renormalise to a per-column total solar irradiance the way RFMIP does.
    toa = np.ascontiguousarray(np.broadcast_to(kdist.solar_source[:, None], (ngpt, ncol)))
    albedo = np.full((ngpt, ncol), SFC_ALBEDO)

    out = rte3d.solve_sw(
        kdist, gas_concs, False,
        atm['play'], atm['plev'], atm['tlay'],
        mu0=np.full((nlay, ncol), MU0),
        sfc_alb_dir=albedo, sfc_alb_dif=albedo, inc_flux_dir=toa,
        cloud_tau=cloud_tau, cloud_ssa=cloud_ssa, cloud_g=cloud_g)

    return out['flux_up'], out['flux_dn'], out['flux_dir']
