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


def _band_lims(kdist):
    """Band g-point limits, (nbnd, 2) and 0-based inclusive, for the by-band increments."""
    return np.ascontiguousarray(kdist.arrays()['band_lims_gpt'])


def solve_lw(rte3d, kdist, cloud_optics, gas_concs, atm):
    """Longwave all-sky fluxes.

    Clouds are band-resolved and absorption-only here, matching the driver: the
    longwave solver used is the no-scattering one, so the cloud contribution enters as
    an absorption optical depth added by band.
    """
    out = rte3d.gas_optics_lw(
        kdist, gas_concs, atm['play'], atm['plev'], atm['tlay'], atm['tlev'], atm['tsfc'])

    cloud_tau = rte3d.cloud_optics(
        cloud_optics, clwp=atm['lwp'], ciwp=atm['iwp'],
        reliq=atm['rel'], reice=atm['rei'], two_stream=False)

    tau = rte3d.increment_1scalar_by_1scalar(
        out['tau'], cloud_tau, gpt_lims=_band_lims(kdist))

    ngpt, _, ncol = tau.shape

    flux_up, flux_dn, _ = rte3d.lw_solver_noscat(
        False,  # layer 0 is at the surface
        secants=np.full((1, ngpt, ncol), 1.0/0.6096748751),
        weights=np.array([1.0]),
        tau=tau,
        lay_source=out['lay_source'], lev_source=out['lev_source'],
        sfc_emis=np.full((ngpt, ncol), SFC_EMIS),
        sfc_source=out['sfc_source'],
        inc_flux=np.zeros((ngpt, ncol)))

    return flux_up.sum(axis=0), flux_dn.sum(axis=0)


def solve_sw(rte3d, kdist, cloud_optics, gas_concs, atm):
    """Shortwave all-sky fluxes.

    Cloud properties are delta-scaled before being added, as the driver does.
    """
    tau, ssa = rte3d.gas_optics_sw(
        kdist, gas_concs, atm['play'], atm['plev'], atm['tlay'])

    cloud_tau, cloud_ssa, cloud_g = rte3d.cloud_optics(
        cloud_optics, clwp=atm['lwp'], ciwp=atm['iwp'],
        reliq=atm['rel'], reice=atm['rei'])

    cloud_tau, cloud_ssa, cloud_g = rte3d.delta_scale_2str(cloud_tau, cloud_ssa, cloud_g)

    g = np.zeros_like(tau)
    tau, ssa, g = rte3d.increment_2stream_by_2stream(
        tau, ssa, g, cloud_tau, cloud_ssa, cloud_g, gpt_lims=_band_lims(kdist))

    ngpt, nlay, ncol = tau.shape

    # The k-distribution's solar source is used as it stands: the all-sky case does not
    # renormalise to a per-column total solar irradiance the way RFMIP does.
    toa = np.ascontiguousarray(np.broadcast_to(kdist.solar_source[:, None], (ngpt, ncol)))
    albedo = np.full((ngpt, ncol), SFC_ALBEDO)

    flux_up, flux_dn, flux_dir = rte3d.sw_solver_2stream(
        False, tau, ssa, g,
        mu0=np.full((nlay, ncol), MU0),
        sfc_alb_dir=albedo, sfc_alb_dif=albedo, inc_flux_dir=toa)

    return flux_up.sum(axis=0), flux_dn.sum(axis=0), flux_dir.sum(axis=0)
