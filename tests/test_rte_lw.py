"""Step 1c: the longwave no-scattering solver with multi-angle quadrature."""
import numpy as np
import pytest

from compare import RECURRENCE_RTOL, assert_close

SHAPES = [(16, 42, 7), (1, 3, 5), (8, 1, 4), (3, 12, 1)]


def tolerance(rte3d):
    """These tests all run a solver, so they inherit the recurrence's amplification."""
    return 1e-5 if rte3d.runtime().precision == 'single' else RECURRENCE_RTOL


def random_inputs(ngpt, nlay, ncol, nmus=1, seed=0, inc_flux=True):
    rng = np.random.default_rng(seed)

    # Secants of the propagation angle, around the 1.66 diffusivity factor the
    # single-angle quadrature uses. Weights need not sum to anything in particular:
    # the kernel just scales each angle's contribution by pi*weight.
    secants = rng.uniform(1.2, 2.0, (nmus, ngpt, ncol))
    weights = rng.uniform(0.1, 0.6, nmus)

    return dict(
        secants=secants,
        weights=weights,
        tau=10.0**rng.uniform(-6.0, 2.0, (ngpt, nlay, ncol)),
        lay_source=rng.uniform(0.0, 100.0, (ngpt, nlay, ncol)),
        lev_source=rng.uniform(0.0, 100.0, (ngpt, nlay+1, ncol)),
        sfc_emis=rng.uniform(0.0, 1.0, (ngpt, ncol)),
        sfc_source=rng.uniform(0.0, 100.0, (ngpt, ncol)),
        # Usually zero in the longwave, but non-zero here to pin down how the
        # boundary condition accumulates across quadrature angles.
        inc_flux=rng.uniform(0.0, 10.0, (ngpt, ncol)) if inc_flux else np.zeros((ngpt, ncol)),
    )


@pytest.mark.parametrize('top_at_1', [True, False])
@pytest.mark.parametrize('ngpt,nlay,ncol', SHAPES)
def test_noscat_matches_reference(rte3d, fortran_ref, top_at_1, ngpt, nlay, ncol):
    a = random_inputs(ngpt, nlay, ncol, seed=1)

    expected = fortran_ref.lw_solver_noscat(top_at_1, **a)
    actual = rte3d.lw_solver_noscat(top_at_1, **a)

    for name, exp, act in zip(('flux_up', 'flux_dn'), expected, actual):
        assert_close(
            act,
            exp,
            rtol=tolerance(rte3d),
            err_msg=f'{name} differs from the reference')
    assert actual[2] is None


@pytest.mark.parametrize('top_at_1', [True, False])
@pytest.mark.parametrize('nmus', [1, 2, 4])
def test_noscat_multi_angle_matches_reference(rte3d, fortran_ref, top_at_1, nmus):
    a = random_inputs(8, 20, 6, nmus=nmus, seed=2)

    expected = fortran_ref.lw_solver_noscat(top_at_1, **a)
    actual = rte3d.lw_solver_noscat(top_at_1, **a)

    for exp, act in zip(expected[:2], actual[:2]):
        assert_close(act, exp, rtol=tolerance(rte3d))


@pytest.mark.parametrize('top_at_1', [True, False])
@pytest.mark.parametrize('nmus', [1, 3])
def test_noscat_jacobian_matches_reference(rte3d, fortran_ref, top_at_1, nmus):
    """The Jacobian is spectrally integrated, so (nlev, ncol) and not (ngpt, nlev, ncol).

    rte_kernels.h documents it with a g-point dimension, but the Fortran declares it
    without one and the comment in the source says only broadband Jacobians are
    provided. The shape assertion below is what pins that down.
    """
    a = random_inputs(8, 20, 6, nmus=nmus, seed=3)
    a['sfc_source_jac'] = np.random.default_rng(9).uniform(0.0, 1.0, (8, 6))

    expected = fortran_ref.lw_solver_noscat(top_at_1, **a)
    actual = rte3d.lw_solver_noscat(top_at_1, **a)

    assert actual[2].shape == (21, 6)
    for exp, act in zip(expected, actual):
        assert_close(act, exp, rtol=tolerance(rte3d))


@pytest.mark.parametrize('top_at_1', [True, False])
@pytest.mark.parametrize('nmus', [1, 3])
def test_transparent_atmosphere_is_exact(rte3d, top_at_1, nmus):
    """With tau = 0 the layers neither absorb nor emit, so every level carries the
    boundary values. Needs no reference, and checks the quadrature accumulation.
    """
    ngpt, nlay, ncol = 4, 10, 3
    nlev = nlay + 1
    rng = np.random.default_rng(7)

    weights = rng.uniform(0.1, 0.6, nmus)
    a = dict(
        secants=rng.uniform(1.2, 2.0, (nmus, ngpt, ncol)),
        weights=weights,
        tau=np.zeros((ngpt, nlay, ncol)),
        lay_source=rng.uniform(0.0, 100.0, (ngpt, nlay, ncol)),
        lev_source=rng.uniform(0.0, 100.0, (ngpt, nlev, ncol)),
        sfc_emis=rng.uniform(0.0, 1.0, (ngpt, ncol)),
        sfc_source=rng.uniform(0.0, 100.0, (ngpt, ncol)),
        inc_flux=rng.uniform(0.0, 10.0, (ngpt, ncol)),
    )

    flux_up, flux_dn, _ = rte3d.lw_solver_noscat(top_at_1, **a)

    # Each angle converts the incident flux to intensity by dividing by pi*weight and
    # converts back by multiplying, so every angle contributes the incident flux itself.
    expected_dn = np.broadcast_to((nmus*a['inc_flux'])[:, None, :], (ngpt, nlev, ncol))

    # Upwelling is reflection of the incident diffuse flux plus surface emission,
    # weighted by the quadrature.
    rad_up = a['inc_flux']/(np.pi*weights[:, None, None])*(1.0 - a['sfc_emis']) \
        + a['sfc_emis']*a['sfc_source']
    expected_up = np.broadcast_to(
        (np.pi*weights[:, None, None]*rad_up).sum(axis=0)[:, None, :], (ngpt, nlev, ncol))

    tol = tolerance(rte3d)
    assert_close(flux_dn, expected_dn, rtol=tol)
    assert_close(flux_up, expected_up, rtol=tol)


def tau_min_2stream(rte3d):
    """The thinnest layer the two-stream comparisons against the reference draw.

    The reference's lw_source_2str forms Z = (B_bot - B_top)/(tau*(gamma1 + gamma2))
    and then differences of terms of size Z, so its rounding error grows like
    epsilon/tau: in single precision, at tau ~ 1e-6, it is tens of W/m2 off its
    double-precision answer. rte3d collects terms instead and is not, so single
    precision compares against the reference only from tau = 0.1 up; double keeps the
    whole range. test_2stream_source_thin_layers covers thin layers in single precision.
    """
    return 1e-1 if rte3d.runtime().precision == 'single' else 1e-6


def toon_source(lev_source_top, lev_source_bot, tau, w0, g):
    """The reference's two-stream source, Toon et al. Eqs 26-27, in float64."""
    c = float(np.float32(1.66))
    gamma1 = c*(1.0 - 0.5*w0*(1.0 + g))
    gamma2 = c*0.5*w0*(1.0 - g)
    k = np.sqrt((gamma1 - gamma2)*(gamma1 + gamma2))
    e1 = np.exp(-k*tau)
    rt = 1.0/(k*(1.0 + e1**2) + gamma1*(1.0 - e1**2))
    rdif, tdif = rt*gamma2*(1.0 - e1**2), rt*2.0*k*e1
    z = (lev_source_bot - lev_source_top)/(tau*(gamma1 + gamma2))
    up = (z + lev_source_top) - rdif*(-z + lev_source_top) - tdif*(z + lev_source_bot)
    dn = (-z + lev_source_bot) - rdif*(z + lev_source_bot) - tdif*(-z + lev_source_top)
    return np.pi*up, np.pi*dn


def test_2stream_source_thin_layers(rte3d):
    """One layer over a black surface with no incoming flux and no surface source: the
    fluxes leaving it are its two source terms. Checked down to tau = 1e-6 against
    toon_source in float64, whose own error there is ~1e-8 W/m2."""
    single = rte3d.runtime().precision == 'single'
    rnd = (lambda x: np.asarray(x, np.float32).astype(np.float64)) if single else np.asarray

    rng = np.random.default_rng(15)
    n = 200
    tau = rnd(10.0**rng.uniform(-6.0, 2.0, n))
    w0 = rnd(rng.uniform(0.0, 0.999, n))
    g = rnd(rng.uniform(0.0, 1.0, n))
    lev_source = rnd(rng.uniform(0.0, 100.0, (2, n)))

    flux_up, flux_dn = rte3d.lw_solver_2stream(
        True, tau=tau.reshape(1, 1, n), ssa=w0.reshape(1, 1, n), g=g.reshape(1, 1, n),
        lay_source=np.zeros((1, 1, n)), lev_source=lev_source.reshape(1, 2, n),
        sfc_emis=np.ones((1, n)), sfc_source=np.zeros((1, n)), inc_flux=np.zeros((1, n)))

    up, dn = toon_source(lev_source[0], lev_source[1], tau, w0, g)

    atol = 1e-3 if single else 1e-7
    np.testing.assert_allclose(flux_up[0, 0], up, rtol=0.0, atol=atol, err_msg='source_up')
    np.testing.assert_allclose(flux_dn[0, 1], dn, rtol=0.0, atol=atol, err_msg='source_dn')


def random_2stream_inputs(ngpt, nlay, ncol, seed=0, tau_min=1e-6):
    rng = np.random.default_rng(seed)

    return dict(
        tau=10.0**rng.uniform(np.log10(tau_min), 2.0, (ngpt, nlay, ncol)),
        ssa=rng.uniform(0.0, 1.0, (ngpt, nlay, ncol)),
        g=rng.uniform(0.0, 1.0, (ngpt, nlay, ncol)),
        lay_source=rng.uniform(0.0, 100.0, (ngpt, nlay, ncol)),
        lev_source=rng.uniform(0.0, 100.0, (ngpt, nlay+1, ncol)),
        sfc_emis=rng.uniform(0.0, 1.0, (ngpt, ncol)),
        sfc_source=rng.uniform(0.0, 100.0, (ngpt, ncol)),
        inc_flux=rng.uniform(0.0, 10.0, (ngpt, ncol)),
    )


@pytest.mark.parametrize('top_at_1', [True, False])
@pytest.mark.parametrize('ngpt,nlay,ncol', SHAPES)
def test_2stream_matches_reference(rte3d, fortran_ref, top_at_1, ngpt, nlay, ncol):
    """Compare against the reference with a g-point-independent lev_source.

    The reference's lw_solver_2stream passes `lev_source` to lw_source_2str without
    the `(:,:,igpt)` slice (mo_rte_solver_kernels.F90:422), so through Fortran sequence
    association every g-point silently uses lev_source(:,:,1). Holding lev_source
    constant across g-points makes that defect inert, so this still validates the whole
    solver -- lw_two_stream, lw_source_2str and adding -- at full ngpt.

    See test_2stream_reference_bug_gpt_indexing for the defect itself.
    """
    a = random_2stream_inputs(ngpt, nlay, ncol, seed=11, tau_min=tau_min_2stream(rte3d))
    a['lev_source'] = np.broadcast_to(a['lev_source'][0:1], a['lev_source'].shape).copy()

    expected = fortran_ref.lw_solver_2stream(top_at_1, **a)
    actual = rte3d.lw_solver_2stream(top_at_1, **a)

    for name, exp, act in zip(('flux_up', 'flux_dn'), expected, actual):
        assert_close(
            act,
            exp,
            rtol=tolerance(rte3d),
            err_msg=f'{name} differs from the reference')


@pytest.mark.parametrize('top_at_1', [True, False])
def test_2stream_gpt_varying_lev_source(rte3d, fortran_ref, top_at_1):
    """With a g-point-varying lev_source, validate against the reference run one
    g-point at a time -- the regime where its indexing defect cannot bite.

    This is the real correctness check for ngpt > 1: the oracle is still the reference,
    just used where it is trustworthy.
    """
    ngpt, nlay, ncol = 4, 6, 3
    a = random_2stream_inputs(ngpt, nlay, ncol, seed=14, tau_min=tau_min_2stream(rte3d))

    per_gpt = [fortran_ref.lw_solver_2stream(top_at_1, **{k: v[i:i+1] for k, v in a.items()})
               for i in range(ngpt)]
    expected_up = np.concatenate([u for u, _ in per_gpt])
    expected_dn = np.concatenate([d for _, d in per_gpt])

    flux_up, flux_dn = rte3d.lw_solver_2stream(top_at_1, **a)

    tol = tolerance(rte3d)
    assert_close(flux_up, expected_up, rtol=tol)
    assert_close(flux_dn, expected_dn, rtol=tol)


@pytest.mark.parametrize('top_at_1', [True, False])
def test_2stream_reference_bug_gpt_indexing(rte3d, fortran_ref, top_at_1):
    """Pin down the reference defect described above.

    Feeding rte3d a lev_source forced to g-point 0 everywhere reproduces the
    reference's full-ngpt output exactly, which is only possible if the reference is
    ignoring the g-point index. If this test ever starts failing, upstream has fixed
    mo_rte_solver_kernels.F90:422 and the sibling tests should be simplified.
    """
    ngpt = 4
    a = random_2stream_inputs(ngpt, 6, 3, seed=14, tau_min=tau_min_2stream(rte3d))

    expected = fortran_ref.lw_solver_2stream(top_at_1, **a)

    b = dict(a)
    b['lev_source'] = np.broadcast_to(a['lev_source'][0:1], a['lev_source'].shape).copy()
    actual = rte3d.lw_solver_2stream(top_at_1, **b)

    for exp, act in zip(expected, actual):
        assert_close(act, exp, rtol=tolerance(rte3d))

    # And confirm the two genuinely differ, so this test is not vacuous.
    straight = rte3d.lw_solver_2stream(top_at_1, **a)
    assert not np.allclose(straight[0], expected[0])


@pytest.mark.parametrize('top_at_1', [True, False])
def test_2stream_thin_layers_have_no_source(rte3d, top_at_1):
    """Below tau = 1e-8 the reference zeroes the layer sources outright, so a
    transparent column carries only the boundary values."""
    ngpt, nlay, ncol = 4, 8, 3
    nlev = nlay + 1
    rng = np.random.default_rng(12)

    a = random_2stream_inputs(ngpt, nlay, ncol, seed=13)
    # Exactly zero, not merely small: at tau = 1e-12 the transmittance is
    # 1 - 1.66e-12 per layer, which accumulates past the tolerance over a column.
    a['tau'] = np.zeros((ngpt, nlay, ncol))
    a['ssa'] = np.zeros((ngpt, nlay, ncol))
    a['g'] = np.zeros((ngpt, nlay, ncol))
    a['inc_flux'] = rng.uniform(0.0, 10.0, (ngpt, ncol))

    flux_up, flux_dn = rte3d.lw_solver_2stream(top_at_1, **a)

    # With ssa = 0 the layers do not reflect, and with tau below the threshold they do
    # not emit, so the downward flux is the boundary condition at every level and the
    # upward flux is the surface emission.
    expected_dn = np.broadcast_to(a['inc_flux'][:, None, :], (ngpt, nlev, ncol))
    expected_up = np.broadcast_to(
        (a['inc_flux']*(1.0 - a['sfc_emis']) + np.pi*a['sfc_emis']*a['sfc_source'])[:, None, :],
        (ngpt, nlev, ncol))

    tol = tolerance(rte3d)
    assert_close(flux_dn, expected_dn, rtol=tol)
    assert_close(flux_up, expected_up, rtol=tol)
