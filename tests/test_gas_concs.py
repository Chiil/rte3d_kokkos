"""Step 2b: gas concentrations, dry air column and column gas amounts.

get_col_dry is not part of the reference's C API, so it is checked against an
independent numpy transcription plus a physical identity that needs no reference at
all: the dry air column integrates to the hydrostatic mass of the atmosphere.
"""
import numpy as np
import pytest

# From mo_gas_optics_constants.
M_H2O = 0.018016
M_DRY = 0.028964
AVOGAD = 6.02214076e23
GRAV = 9.80665


def tolerance(rte3d):
    return 1e-5 if rte3d.runtime().precision == 'single' else 1e-12


def col_dry_numpy(vmr_h2o, plev, latitude=None):
    """Independent transcription of get_col_dry."""
    g0 = (GRAV if latitude is None
          else 9.80665 - 0.02586*np.cos(2.0*np.pi*latitude/180.0))

    delta_plev = np.abs(plev[:-1] - plev[1:])
    fact = 1.0 / (1.0 + vmr_h2o)
    m_air = (M_DRY + M_H2O*vmr_h2o) * fact

    return 10.0 * delta_plev * AVOGAD * fact / (1000.0 * m_air * 100.0 * g0)


def atmosphere(rng, nlay, ncol):
    plev = np.linspace(1.0e5, 1.0e2, nlay + 1)[:, None] * np.ones((1, ncol))
    return plev, rng.uniform(0.0, 0.02, (nlay, ncol))


def test_gas_concs_scalar_profile_and_field(rte3d):
    nlay, ncol = 6, 4
    rng = np.random.default_rng(60)

    g = rte3d.Gas_concs()
    g.set_vmr('CO2', 348e-6)                       # names are case-insensitive
    g.set_vmr('o3', rng.uniform(0, 1e-6, nlay))    # profile
    field = rng.uniform(0, 0.02, (nlay, ncol))
    g.set_vmr('h2o', field)

    assert g.contains('co2') and g.contains('Co2')
    assert not g.contains('ch4')
    assert g.names() == ['co2', 'h2o', 'o3']

    np.testing.assert_allclose(g.get_vmr('co2', nlay, ncol), 348e-6, rtol=tolerance(rte3d))
    # The field is stored at the module's precision, so compare against it cast down:
    # set_vmr/get_vmr is a plain round-trip, exact once the dtype matches.
    h2o = g.get_vmr('h2o', nlay, ncol)
    np.testing.assert_array_equal(h2o, field.astype(h2o.dtype))

    # A profile broadcasts across columns.
    o3 = g.get_vmr('o3', nlay, ncol)
    np.testing.assert_array_equal(o3, o3[:, :1]*np.ones((1, ncol)))


def test_gas_concs_rejects_unknown_gas(rte3d):
    g = rte3d.Gas_concs()
    with pytest.raises(ValueError, match='ch4'):
        g.get_vmr('ch4', 4, 3)


@pytest.mark.parametrize('with_latitude', [False, True])
@pytest.mark.parametrize('nlay,ncol', [(20, 6), (1, 4)])
def test_col_dry_matches_transcription(rte3d, nlay, ncol, with_latitude):
    rng = np.random.default_rng(61)
    plev, vmr_h2o = atmosphere(rng, nlay, ncol)
    latitude = rng.uniform(-80.0, 80.0, ncol) if with_latitude else None

    actual = rte3d.compute_col_dry(vmr_h2o, plev, latitude)
    expected = col_dry_numpy(vmr_h2o, plev, latitude)

    np.testing.assert_allclose(actual, expected, rtol=tolerance(rte3d), atol=0.0)


def test_col_dry_integrates_to_hydrostatic_mass(rte3d):
    """A check the reference cannot give us: summed over the column, the dry air
    amount times the mean molar mass must equal the dry mass the hydrostatic balance
    implies, p_sfc/g minus the water vapour."""
    nlay, ncol = 40, 3
    rng = np.random.default_rng(62)
    plev, vmr_h2o = atmosphere(rng, nlay, ncol)

    col_dry = rte3d.compute_col_dry(vmr_h2o, plev)

    # molecules/cm2 -> kg/m2 of dry air
    dry_mass = col_dry.sum(axis=0) * 1e4 * M_DRY / AVOGAD

    # Total mass from hydrostatic balance, less the water vapour it contains.
    total_mass = (plev[0] - plev[-1]) / GRAV
    water_mass = (col_dry*vmr_h2o).sum(axis=0) * 1e4 * M_H2O / AVOGAD

    np.testing.assert_allclose(dry_mass, total_mass - water_mass, rtol=tolerance(rte3d))


@pytest.mark.parametrize('supply_col_dry', [False, True])
def test_col_gas(rte3d, supply_col_dry):
    nlay, ncol = 12, 5
    rng = np.random.default_rng(63)
    plev, vmr_h2o = atmosphere(rng, nlay, ncol)

    g = rte3d.Gas_concs()
    g.set_vmr('h2o', vmr_h2o)
    g.set_vmr('co2', 348e-6)
    g.set_vmr('o3', rng.uniform(0, 1e-6, nlay))

    names = ['h2o', 'co2', 'o3']
    col_dry = col_dry_numpy(vmr_h2o, plev) if supply_col_dry else None

    col_gas = rte3d.compute_col_gas(g, names, plev, col_dry)

    assert col_gas.shape == (len(names) + 1, nlay, ncol)

    expected_dry = col_dry_numpy(vmr_h2o, plev)
    tol = tolerance(rte3d)

    # Index 0 is dry air itself; the rest are vmr * col_dry, in the given order.
    np.testing.assert_allclose(col_gas[0], expected_dry, rtol=tol, atol=0.0)
    np.testing.assert_allclose(col_gas[1], vmr_h2o*expected_dry, rtol=tol, atol=0.0)
    np.testing.assert_allclose(col_gas[2], 348e-6*expected_dry, rtol=tol, atol=0.0)


def test_col_gas_order_follows_gas_names(rte3d):
    """The k-distribution decides the gas order, not the container's own ordering."""
    nlay, ncol = 4, 2
    plev, vmr_h2o = atmosphere(np.random.default_rng(64), nlay, ncol)

    g = rte3d.Gas_concs()
    g.set_vmr('h2o', vmr_h2o)
    g.set_vmr('co2', 1e-4)
    g.set_vmr('ch4', 2e-6)

    a = rte3d.compute_col_gas(g, ['h2o', 'co2', 'ch4'], plev)
    b = rte3d.compute_col_gas(g, ['h2o', 'ch4', 'co2'], plev)

    np.testing.assert_array_equal(a[2], b[3])
    np.testing.assert_array_equal(a[3], b[2])
