"""Step 1b: the shortwave solvers.

The comparisons against the Fortran reference use randomised inputs rather than a
stored profile, because the branches that a single RFMIP case never reaches -- the
top_at_1=False orientation, the conservative-scattering limit, night-time columns --
are exactly the ones a port gets wrong.
"""
import numpy as np
import pytest

from compare import RECURRENCE_RTOL, assert_close

SHAPES = [(16, 42, 7), (1, 3, 5), (8, 1, 4), (3, 12, 1)]


def tolerance(rte3d):
    """These tests all run a solver, so they inherit the recurrence's amplification."""
    return 1e-5 if rte3d.runtime().precision == 'single' else RECURRENCE_RTOL


def random_inputs(ngpt, nlay, ncol, seed=0, conservative=False, night=False):
    """Randomised but physical optical properties and boundary conditions."""
    rng = np.random.default_rng(seed)

    # Log-uniform over a wide range: thin layers exercise the k -> 0 limit, thick ones
    # the exp(-tau) underflow guards.
    tau = 10.0**rng.uniform(-6.0, 2.0, (ngpt, nlay, ncol))
    ssa = np.ones((ngpt, nlay, ncol)) if conservative else rng.uniform(0.0, 1.0, (ngpt, nlay, ncol))
    g = rng.uniform(0.0, 1.0, (ngpt, nlay, ncol))

    mu0 = rng.uniform(0.1, 1.0, (nlay, ncol))
    if night:
        # Sun below the horizon in every other column.
        mu0[:, ::2] = -0.5

    return dict(
        tau=tau, ssa=ssa, g=g, mu0=mu0,
        sfc_alb_dir=rng.uniform(0.0, 1.0, (ngpt, ncol)),
        sfc_alb_dif=rng.uniform(0.0, 1.0, (ngpt, ncol)),
        inc_flux_dir=rng.uniform(0.0, 1400.0, (ngpt, ncol)),
    )


@pytest.mark.parametrize('top_at_1', [True, False])
@pytest.mark.parametrize('ngpt,nlay,ncol', SHAPES)
def test_noscat_matches_reference(rte3d, fortran_ref, top_at_1, ngpt, nlay, ncol):
    a = random_inputs(ngpt, nlay, ncol, seed=1)

    expected = fortran_ref.sw_solver_noscat(top_at_1, a['tau'], a['mu0'], a['inc_flux_dir'])
    actual = rte3d.sw_solver_noscat(top_at_1, a['tau'], a['mu0'], a['inc_flux_dir'])

    assert_close(actual, expected, rtol=tolerance(rte3d))


@pytest.mark.parametrize('top_at_1', [True, False])
@pytest.mark.parametrize('ngpt,nlay,ncol', SHAPES)
def test_2stream_matches_reference(rte3d, fortran_ref, top_at_1, ngpt, nlay, ncol):
    a = random_inputs(ngpt, nlay, ncol, seed=2)

    expected = fortran_ref.sw_solver_2stream(top_at_1, **a)
    actual = rte3d.sw_solver_2stream(top_at_1, **a)

    for name, exp, act in zip(('flux_up', 'flux_dn', 'flux_dir'), expected, actual):
        assert_close(
            act,
            exp,
            rtol=tolerance(rte3d),
            err_msg=f'{name} differs from the reference')


@pytest.mark.parametrize('top_at_1', [True, False])
def test_2stream_matches_reference_with_diffuse_bc(rte3d, fortran_ref, top_at_1):
    a = random_inputs(16, 42, 7, seed=3)
    a['inc_flux_dif'] = np.random.default_rng(4).uniform(0.0, 100.0, (16, 7))

    expected = fortran_ref.sw_solver_2stream(top_at_1, **a)
    actual = rte3d.sw_solver_2stream(top_at_1, **a)

    for exp, act in zip(expected, actual):
        assert_close(act, exp, rtol=tolerance(rte3d))


@pytest.mark.parametrize('top_at_1', [True, False])
def test_2stream_matches_reference_at_night(rte3d, fortran_ref, top_at_1):
    """Columns with mu0 <= 0 have no direct beam and so no source for diffuse light."""
    a = random_inputs(8, 20, 6, seed=5, night=True)

    expected = fortran_ref.sw_solver_2stream(top_at_1, **a)
    actual = rte3d.sw_solver_2stream(top_at_1, **a)

    for exp, act in zip(expected, actual):
        assert_close(act, exp, rtol=tolerance(rte3d))


@pytest.mark.parametrize('top_at_1', [True, False])
def test_transparent_atmosphere_is_exact(rte3d, top_at_1):
    """With tau = 0 and a black surface the answer is known in closed form.

    No reference needed, and an orientation or level-indexing error shows up as an
    O(1) discrepancy rather than a small one.
    """
    ngpt, nlay, ncol = 4, 10, 3
    zeros3 = np.zeros((ngpt, nlay, ncol))
    mu0 = np.full((nlay, ncol), 0.5)
    inc = np.full((ngpt, ncol), 1000.0)

    flux_up, flux_dn, flux_dir = rte3d.sw_solver_2stream(
        top_at_1, zeros3, zeros3, zeros3, mu0,
        np.zeros((ngpt, ncol)), np.zeros((ngpt, ncol)), inc)

    expected_dir = np.broadcast_to(inc[:, None, :] * 0.5, (ngpt, nlay+1, ncol))

    tol = tolerance(rte3d)
    assert_close(flux_dir, expected_dir, rtol=tol)
    assert_close(flux_dn, expected_dir, rtol=tol)
    np.testing.assert_allclose(flux_up, np.zeros_like(flux_up), rtol=0.0, atol=tol*1000.0)


@pytest.mark.parametrize('top_at_1', [True, False])
def test_conservative_scattering_conserves_energy(rte3d, top_at_1):
    """With ssa = 1 the atmosphere neither absorbs nor emits, so the net downward flux
    is independent of height.

    Only approximate: the two-stream conservative limit is reached through the min_k
    floor in sw_two_stream, which the reference documents as giving a relative error
    below 0.1%. That is still tight enough that a sign or indexing error stands out.

    Double precision only. For conservative scattering gamma1 - gamma2 is zero exactly,
    so k collapses to sqrt(min_k), and min_k = 1e4*epsilon scales with the working
    precision: sqrt(min_k) is ~1.5e-6 in double but ~0.035 in single. Over the thick
    layers in this case that single-precision floor injects real absorption -- the net
    flux is no longer constant with height at all -- so the conserved-energy property
    the test checks cannot hold in single precision. This is inherent to the reference
    algorithm (its own single-precision CI never tests the pure conservative limit),
    not an rte3d defect, so the test is meaningful only in double precision.
    """
    if rte3d.runtime().precision == 'single':
        pytest.skip('conservative limit is unresolvable at single precision (k = sqrt(min_k))')

    a = random_inputs(8, 30, 5, seed=6, conservative=True)
    a['sfc_alb_dir'] = np.zeros_like(a['sfc_alb_dir'])
    a['sfc_alb_dif'] = np.zeros_like(a['sfc_alb_dif'])

    flux_up, flux_dn, _ = rte3d.sw_solver_2stream(top_at_1, **a)

    # assert_allclose compares shapes before broadcasting, so broadcast explicitly.
    net = flux_dn - flux_up
    expected = np.broadcast_to(net[:, :1, :], net.shape)
    np.testing.assert_allclose(net, expected, rtol=2e-3, atol=1e-8)
