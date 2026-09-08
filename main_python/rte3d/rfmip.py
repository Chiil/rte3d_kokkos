"""Read the RFMIP clear-sky case into the arrays rte3d takes.

RFMIP is 100 sites x 18 experiments x 60 layers, with reference fluxes produced by
RRTMGP itself, so it serves as an end-to-end oracle for the whole chain.

Two things need translating. RFMIP names gases after the molecule
("carbon_dioxide") where RRTMGP uses a formula ("co2"), and it stores concentrations
in scaled units given by each variable's `units` attribute. Both follow
examples/rfmip-clear-sky/mo_rfmip_io.F90.
"""
import numpy as np
import xarray as xr

# RRTMGP gas name -> RFMIP variable name, where they differ. Everything not listed
# uses the RRTMGP name unchanged. From determine_gas_names with forcing_index = 1.
_RENAME = {
    'co': 'carbon_monoxide',
    'ch4': 'methane',
    'o2': 'oxygen',
    'n2o': 'nitrous_oxide',
    'n2': 'nitrogen',
    'co2': 'carbon_dioxide',
    'ccl4': 'carbon_tetrachloride',
    'ch3br': 'methyl_bromide',
    'ch3cl': 'methyl_chloride',
    'cfc22': 'hcfc22',
}

# Read from a full three-dimensional field rather than a global-mean scalar.
_PROFILE = {'h2o': 'water_vapor', 'o3': 'ozone'}


def _scaling(da):
    """RFMIP puts the unit scaling in the `units` attribute, as a literal like 1.e-6."""
    return float(da.attrs.get('units', 1.0))


def read_rfmip(path, gas_names, expt=0):
    """Read one experiment. Columns are the 100 sites.

    gas_names is the k-distribution's list, so the concentrations come back keyed the
    way rte3d expects. Arrays are (nlay, ncol) or (nlev, ncol), with index 0 at the top
    of the atmosphere, matching RFMIP's own ordering.
    """
    d = xr.open_dataset(path)

    # RFMIP is (site, layer); rte3d is (layer, site).
    play = d['pres_layer'].values.T.astype(np.float64)
    plev = d['pres_level'].values.T.astype(np.float64)
    tlay = d['temp_layer'].isel(expt=expt).values.T.astype(np.float64)
    tlev = d['temp_level'].isel(expt=expt).values.T.astype(np.float64)

    gases = {}
    for name in gas_names:
        if name == 'no2':
            # RFMIP provides no NO2; the reference driver sets it to zero.
            gases[name] = 0.0
        elif name in _PROFILE:
            var = d[_PROFILE[name]]
            gases[name] = (var.isel(expt=expt).values.T.astype(np.float64)
                           * _scaling(var))
        else:
            var = d[_RENAME.get(name, name) + '_GM']
            gases[name] = float(var.values[expt]) * _scaling(var)

    return dict(
        play=np.ascontiguousarray(play),
        plev=np.ascontiguousarray(plev),
        tlay=np.ascontiguousarray(tlay),
        tlev=np.ascontiguousarray(tlev),
        tsfc=np.ascontiguousarray(
            d['surface_temperature'].isel(expt=expt).values.astype(np.float64)),
        sfc_emis=np.ascontiguousarray(d['surface_emissivity'].values.astype(np.float64)),
        sfc_alb=np.ascontiguousarray(d['surface_albedo'].values.astype(np.float64)),
        solar_zenith_angle=np.ascontiguousarray(
            d['solar_zenith_angle'].values.astype(np.float64)),
        total_solar_irradiance=np.ascontiguousarray(
            d['total_solar_irradiance'].values.astype(np.float64)),
        gases=gases,
    )


def read_reference(path, name, expt=0):
    """One of the reference flux files, as (nlev, ncol) with index 0 at the top."""
    d = xr.open_dataset(path)
    return np.ascontiguousarray(d[name].isel(expt=expt).values.T.astype(np.float64))


def make_gas_concs(rte3d, gases):
    g = rte3d.Gas_concs()
    for name, value in gases.items():
        g.set_vmr(name, value)
    return g


def _per_gpoint(per_column, ngpt):
    """RFMIP gives one surface emissivity or albedo per column; the solvers take one
    per g-point. The reference expands per band, which is the same thing here since
    the value is spectrally constant."""
    return np.ascontiguousarray(
        np.broadcast_to(per_column[None, :], (ngpt, per_column.shape[0])))


def solve_lw(rte3d, kdist, gas_concs, atm, n_quad_angles=1):
    """Gas optics and longwave transport for one RFMIP experiment.

    Returns broadband (nlev, ncol) upward and downward fluxes. The quadrature is the
    reference's: for a single angle, secant 1/0.6096748751 with unit weight
    (gauss_Ds and gauss_wts in mo_rte_lw.F90).
    """
    if n_quad_angles != 1:
        raise NotImplementedError('only the single-angle quadrature is wired up here')

    ngpt = kdist.ngpt
    ncol = atm['play'].shape[1]

    out = rte3d.solve_lw(
        kdist, gas_concs,
        True,  # RFMIP stores the top of the atmosphere at index 0
        atm['play'], atm['plev'], atm['tlay'], atm['tlev'], atm['tsfc'],
        secants=np.full((1, ncol), 1.0/0.6096748751),
        weights=np.array([1.0]),
        sfc_emis=_per_gpoint(atm['sfc_emis'], ngpt))

    return out['flux_up'], out['flux_dn']


def solve_sw(rte3d, kdist, gas_concs, atm):
    """Gas optics and shortwave transport for one RFMIP experiment.

    Returns broadband fluxes and the daytime mask. Night-time columns are zeroed, as
    the reference driver does; their fluxes are meaningless rather than wrong.
    """
    ngpt = kdist.ngpt
    nlay, ncol = atm['play'].shape

    daytime = atm['solar_zenith_angle'] < 90.0
    mu0 = np.where(daytime, np.cos(np.radians(atm['solar_zenith_angle'])), 1.0)

    # The k-distribution's solar source is renormalised to each column's own total
    # solar irradiance, so its absolute scale does not matter.
    toa = np.broadcast_to(kdist.solar_source[:, None], (ngpt, ncol)).copy()
    toa *= atm['total_solar_irradiance'][None, :] / toa.sum(axis=0)[None, :]

    albedo = _per_gpoint(atm['sfc_alb'], ngpt)

    out = rte3d.solve_sw(
        kdist, gas_concs, True,
        atm['play'], atm['plev'], atm['tlay'],
        mu0=np.ascontiguousarray(np.broadcast_to(mu0[None, :], (nlay, ncol))),
        sfc_alb_dir=albedo, sfc_alb_dif=albedo, inc_flux_dir=toa)

    up = out['flux_up']
    dn = out['flux_dn']
    up[:, ~daytime] = 0.0
    dn[:, ~daytime] = 0.0

    return up, dn, daytime
