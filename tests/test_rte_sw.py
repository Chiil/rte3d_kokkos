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


def away_from_resonance(rte3d, a):
    """(ngpt, ncol) mask of the columns to compare against the reference.

    In single precision the reference loses the direct-beam terms near k*mu0 = 1,
    where rte3d interpolates across the singularity instead (test_2stream_near_resonance
    checks that against the exact value). So single precision leaves out the columns
    with any layer within twice rte3d's interpolation window of it; double keeps them all.
    """
    w0, g = a['ssa'], a['g']
    if rte3d.runtime().precision != 'single':
        return np.ones((w0.shape[0], w0.shape[2]), dtype=bool)
    gamma1 = (8.0 - w0*(5.0 + 3.0*g))/4.0
    gamma2 = 3.0*w0*(1.0 - g)/4.0
    k = np.sqrt((gamma1 - gamma2)*(gamma1 + gamma2))
    window = 2.0*np.cbrt(np.finfo(np.float32).eps)
    return ~(np.abs(k*a['mu0'][None] - 1.0) < window).any(axis=1)


def columns(flux, keep):
    """The kept (gpt, col) columns of a (ngpt, nlev, ncol) flux, as (n, nlev)."""
    return np.moveaxis(flux, 1, 2)[keep]


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
    keep = away_from_resonance(rte3d, a)

    for name, exp, act in zip(('flux_up', 'flux_dn', 'flux_dir'), expected, actual):
        assert_close(
            columns(act, keep),
            columns(exp, keep),
            rtol=tolerance(rte3d),
            err_msg=f'{name} differs from the reference')


@pytest.mark.parametrize('top_at_1', [True, False])
def test_2stream_matches_reference_with_diffuse_bc(rte3d, fortran_ref, top_at_1):
    a = random_inputs(16, 42, 7, seed=3)
    a['inc_flux_dif'] = np.random.default_rng(4).uniform(0.0, 100.0, (16, 7))

    expected = fortran_ref.sw_solver_2stream(top_at_1, **a)
    actual = rte3d.sw_solver_2stream(top_at_1, **a)
    keep = away_from_resonance(rte3d, a)

    for exp, act in zip(expected, actual):
        assert_close(columns(act, keep), columns(exp, keep), rtol=tolerance(rte3d))


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


def meador_weaver(tau, w0, g, mu0):
    """Rdir and Tdir of one layer, Eqs 14-15, in float64 and without the resonance
    guard. Only valid away from k*mu0 = 1."""
    gamma1 = (8.0 - w0*(5.0 + 3.0*g))/4.0
    gamma2 = 3.0*w0*(1.0 - g)/4.0
    gamma3 = (2.0 - 3.0*mu0*g)/4.0
    gamma4 = 1.0 - gamma3
    alpha1 = gamma1*gamma4 + gamma2*gamma3
    alpha2 = gamma1*gamma3 + gamma2*gamma4
    k = np.sqrt((gamma1 - gamma2)*(gamma1 + gamma2))
    e1, e2, t = np.exp(-k*tau), np.exp(-2.0*k*tau), np.exp(-tau/mu0)
    rt = w0/((1.0 - k*mu0)*(1.0 + k*mu0)*(k*(1.0 + e2) + gamma1*(1.0 - e2)))
    rdir = rt*((1.0 - k*mu0)*(alpha2 + k*gamma3) - (1.0 + k*mu0)*(alpha2 - k*gamma3)*e2
               - 2.0*(k*gamma3 - alpha2*k*mu0)*e1*t)
    tdir = -rt*((1.0 + k*mu0)*(alpha1 + k*gamma4)*t - (1.0 - k*mu0)*(alpha1 - k*gamma4)*e2*t
                - 2.0*(k*gamma4 + alpha1*k*mu0)*e1)
    return rdir, tdir


@pytest.mark.parametrize('w0,g,tau', [(0.5, 0.5, 1.0), (0.9, 0.2, 0.3), (0.99, 0.85, 0.05)])
def test_2stream_near_resonance(rte3d, w0, g, tau):
    """Rdir and Tdir stay accurate as k*mu0 -> 1, where Eqs 14-15 are 0/0.

    The reference replaces a denominator below eps with +eps; in single precision that
    loses Rdir and Tdir entirely within ~1e-7 of resonance, errors up to 0.2 of the
    incoming flux. The oracle is the float64 formula.
    """
    single = rte3d.runtime().precision == 'single'
    rnd = (lambda x: np.asarray(x, np.float32).astype(np.float64)) if single else np.asarray
    w0, g, tau = rnd(w0), rnd(g), rnd(tau)
    gamma1 = (8.0 - w0*(5.0 + 3.0*g))/4.0
    gamma2 = 3.0*w0*(1.0 - g)/4.0
    k = np.sqrt((gamma1 - gamma2)*(gamma1 + gamma2))

    delta = np.concatenate([-np.logspace(-1, -9, 33), [0.0], np.logspace(-9, -1, 33)])
    mu0 = rnd((1.0 + delta)/k)
    n = mu0.size

    flux_up, flux_dn, flux_dir = rte3d.sw_solver_2stream(
        True, tau=np.full((1, 1, n), tau), ssa=np.full((1, 1, n), w0), g=np.full((1, 1, n), g),
        mu0=mu0.reshape(1, n), sfc_alb_dir=np.zeros((1, n)), sfc_alb_dif=np.zeros((1, n)),
        inc_flux_dir=np.ones((1, n)))

    # Across resonance, where the formula is 0/0 in float64 too, interpolate linearly
    # between points h either side; that costs O(h^2).
    h = 1e-4
    x = k*mu0 - 1.0
    near = np.abs(x) < h
    rdir, tdir = meador_weaver(tau, w0, g, np.where(near, 0.5/k, mu0))
    r_lo, t_lo = meador_weaver(tau, w0, g, (1.0 - h)/k)
    r_hi, t_hi = meador_weaver(tau, w0, g, (1.0 + h)/k)
    frac = (x + h)/(2.0*h)
    rdir = np.where(near, r_lo + frac*(r_hi - r_lo), rdir)
    tdir = np.where(near, t_lo + frac*(t_hi - t_lo), tdir)

    # The kernel's energy budget clamp.
    tnoscat = np.exp(-tau/mu0)
    rdir = np.clip(rdir, 0.0, 1.0 - tnoscat)
    tdir = np.clip(tdir, 0.0, 1.0 - tnoscat - rdir)

    # The solver scales the incoming direct flux by mu0.
    rdir, tdir = mu0*rdir, mu0*tdir

    atol = 1e-3 if single else 1e-7
    np.testing.assert_allclose(flux_up[0, 0], rdir, rtol=0.0, atol=atol, err_msg='Rdir')
    np.testing.assert_allclose(flux_dn[0, 1] - flux_dir[0, 1], tdir, rtol=0.0, atol=atol,
                               err_msg='Tdir')
