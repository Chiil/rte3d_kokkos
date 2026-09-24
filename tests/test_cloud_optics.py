"""Step 3: cloud optics from lookup tables.

ty_cloud_optics_rrtmgp%cloud_optics has no bind(C) entry point, so the oracle is again
tests/shim/rte3d_shim.F90, driven with the same tables read from the real coefficient
files in extern/rrtmgp-data.
"""
import os

import numpy as np
import pytest

from compare import assert_close, rtol_for

DATA = os.path.join(os.path.dirname(__file__), '..', 'extern', 'rrtmgp-data')

requires_data = pytest.mark.skipif(
    not os.path.exists(os.path.join(DATA, 'rrtmgp-clouds-lw-bnd.nc')),
    reason='rrtmgp-data submodule not checked out')


def clouds(rng, nlay, ncol, bounds, empty_fraction=0.4):
    """Water paths and particle sizes spanning the tables, with some clear layers.

    Sizes are drawn strictly inside the table bounds: the reference rejects anything
    outside, and rte3d extrapolates rather than clamping, so the two only agree where
    the input is valid.
    """
    liq_lwr, liq_upr, ice_lwr, ice_upr = bounds

    clwp = rng.uniform(1.0, 200.0, (nlay, ncol))
    ciwp = rng.uniform(1.0, 200.0, (nlay, ncol))

    # Clear and partially clear layers, so the masking is exercised in both phases.
    clwp[rng.random((nlay, ncol)) < empty_fraction] = 0.0
    ciwp[rng.random((nlay, ncol)) < empty_fraction] = 0.0

    return dict(
        clwp=np.ascontiguousarray(clwp),
        ciwp=np.ascontiguousarray(ciwp),
        reliq=rng.uniform(liq_lwr, liq_upr, (nlay, ncol)),
        reice=rng.uniform(ice_lwr, ice_upr, (nlay, ncol)),
    )


@requires_data
@pytest.mark.parametrize('two_stream', [True, False])
@pytest.mark.parametrize('icergh', [0, 1, 2])
@pytest.mark.parametrize('coeff_file', ['rrtmgp-clouds-lw-bnd.nc', 'rrtmgp-clouds-sw-g224.nc'])
def test_cloud_optics_matches_reference(rte3d, fortran_shim, coeff_file, icergh, two_stream):
    from rte3d.kdist import read_cloud_optics

    f = read_cloud_optics(os.path.join(DATA, coeff_file))
    c = rte3d.load_cloud_optics(f, icergh)

    rng = np.random.default_rng(80 + icergh)
    a = clouds(rng, 24, 8, c.size_bounds)

    expected = fortran_shim.cloud_optics(f, icergh=icergh, two_stream=two_stream, **a)
    actual = rte3d.cloud_optics(c, two_stream=two_stream, **a)

    if two_stream:
        for name, exp, act in zip(('tau', 'ssa', 'g'), expected, actual):
            assert_close(act, exp, rtol=rtol_for(rte3d, 1e-12),
                                       err_msg=f'{name} differs from the reference')
    else:
        assert_close(actual, expected, rtol=rtol_for(rte3d, 1e-12))


@requires_data
def test_cloud_optics_is_physical(rte3d):
    from rte3d.kdist import read_cloud_optics

    f = read_cloud_optics(os.path.join(DATA, 'rrtmgp-clouds-sw-g224.nc'))
    c = rte3d.load_cloud_optics(f)

    rng = np.random.default_rng(81)
    a = clouds(rng, 30, 6, c.size_bounds, empty_fraction=0.0)

    tau, ssa, g = rte3d.cloud_optics(c, **a)

    # The bounds are physical, but ssa is a ratio of two interpolated sums and g comes
    # off the same tables, so either can land an ulp outside. Allow that much: in
    # single precision the tables put ssa at 1 + 6e-8 for the most conservative
    # droplets, which is round-off and not a table read gone wrong.
    eps = np.finfo(ssa.dtype).eps

    assert tau.min() > 0.0, 'every layer here is cloudy'
    assert -eps <= ssa.min() and ssa.max() <= 1.0 + eps
    assert -1.0 - eps <= g.min() and g.max() <= 1.0 + eps

    # In the shortwave, cloud droplets scatter far more than they absorb.
    assert ssa.mean() > 0.8


@requires_data
def test_clear_layers_have_no_cloud_optical_depth(rte3d):
    """A layer with no condensate must contribute nothing, whatever particle size is
    sitting in the array for it."""
    from rte3d.kdist import read_cloud_optics

    f = read_cloud_optics(os.path.join(DATA, 'rrtmgp-clouds-lw-bnd.nc'))
    c = rte3d.load_cloud_optics(f)

    nlay, ncol = 8, 3
    zeros = np.zeros((nlay, ncol))
    liq_lwr, liq_upr, ice_lwr, ice_upr = c.size_bounds

    tau = rte3d.cloud_optics(
        c, clwp=zeros, ciwp=zeros,
        reliq=np.full((nlay, ncol), 0.5*(liq_lwr + liq_upr)),
        reice=np.full((nlay, ncol), 0.5*(ice_lwr + ice_upr)),
        two_stream=False)

    np.testing.assert_array_equal(tau, np.zeros_like(tau))


@requires_data
def test_ice_roughness_changes_the_answer(rte3d):
    """The three roughness types are distinct tables; selecting one must matter."""
    from rte3d.kdist import read_cloud_optics

    f = read_cloud_optics(os.path.join(DATA, 'rrtmgp-clouds-sw-g224.nc'))
    rng = np.random.default_rng(82)

    results = []
    for icergh in range(3):
        c = rte3d.load_cloud_optics(f, icergh)
        a = clouds(rng if icergh == 0 else np.random.default_rng(82), 12, 4, c.size_bounds,
                   empty_fraction=0.0)
        results.append(rte3d.cloud_optics(c, **a)[2])   # asymmetry parameter

    assert not np.allclose(results[0], results[1])
    assert not np.allclose(results[1], results[2])
