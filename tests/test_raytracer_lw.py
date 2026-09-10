"""Step 4b: the longwave Monte Carlo ray tracer.

There is no Fortran oracle for this one -- the reference has no ray tracer -- so the
tests are the physics instead. Unlike the shortwave tracer, though, the longwave one
has a case with an answer in closed form over the whole range of optical depths: an
isothermal slab over a black cold surface radiates pi*B*(1 - 2*E3(tau)) upward, where
E3 is the third exponential integral. That is the exact angular integral the tracer is
performing, and it pins the transport far more tightly than any conservation law can.

The rest are the laws the tracer must obey -- the energy budget, radiative equilibrium,
conservative scattering -- and a comparison against the plane-parallel solver, which
solves the same problem by quadrature.

Everything here is Monte Carlo, so the tolerances are set by the photon count rather
than by round-off, except where a quantity is scored exactly. Those are held to the
precision the module was built in.
"""
import numpy as np
import pytest

# Small enough to keep the suite quick on a CPU, large enough that the domain mean of
# a scored flux is good to a few tenths of a percent.
NX = NY = 8
NZ = 16
NCOL = NX*NY
DX = DY = DZ = 100.0
PHOTONS = 4096

# The Planck source, as a radiance: the tracer applies the pi that turns it into a flux,
# exactly as Rte_lw::solver_2stream does to the same sfc_source.
B = 100.0

# The quadrature RTE-RRTMGP uses for three angles -- Clough et al. 1992, Table 2, which
# is where mo_rte_lw.F90 takes gauss_Ds and gauss_wts from. The weights are doubled
# because rte3d scales an angle by pi*weight where the reference uses 2*pi*weight.
GAUSS_D = np.array([1.09719858, 1.69338507, 4.70941630])
GAUSS_WTS = 2*np.array([0.2009319137, 0.2292411064, 0.0698269799])


def exact_rtol(rte3d):
    """Tolerance for a quantity the tracer computes exactly, up to accumulation.

    The photon counts are summed with atomics, so the order the threads add them in is
    not fixed; what is left is the rounding of that sum, which is the precision the
    module was built in.
    """
    return 1e-4 if rte3d.runtime().precision == 'single' else 1e-9


def grid(**kwargs):
    """The keyword arguments describing the standard test box."""
    return dict(nx=NX, ny=NY, nz=NZ, dx=DX, dy=DY, dz=DZ, **kwargs)


def uniform(value, nlay=NZ):
    return np.full((nlay, NCOL), value)


def trace(rte3d, tau, lay_source, sfc_source, sfc_emis, ppp=PHOTONS, ssa=None, **kwargs):
    """One trace of a non-scattering gas, which is what a longwave g-point usually is."""
    nlay = tau.shape[0]
    return rte3d.trace_rays_lw(
        tau, uniform(0.0, nlay) if ssa is None else ssa,
        lay_source, sfc_source, sfc_emis,
        photons_per_pixel=ppp, **grid(**kwargs))


def exp_int_3(x, n=200000):
    """E3(x) = int_0^1 mu*exp(-x/mu) dmu, by the midpoint rule.

    Written out rather than taken from scipy, which rte3d does not depend on. The
    integrand is smooth on (0, 1] and vanishes at the origin, so the midpoint rule
    converges quickly; at this many points it is good to far better than the Monte
    Carlo noise it is compared against.
    """
    mu = (np.arange(n) + 0.5)/n
    return float((mu*np.exp(-x/mu)).sum()/n)


def column_net(out):
    """Column-integrated net absorption [W/m2], from the per-unit-height flux."""
    return out['flux_net'].sum(axis=0)*DZ


@pytest.mark.parametrize('tau_tot', [0.1, 0.3, 1.0, 2.4, 8.0])
def test_isothermal_slab_matches_the_exponential_integral(rte3d, tau_tot):
    """The one case with an answer in closed form, over four decades of optical depth.

    An isothermal slab at Planck source B over a black surface that emits nothing
    radiates pi*B*(1 - 2*E3(tau)) out of its top. Nothing about that is an
    approximation: it is the exact integral over zenith angle of the exact
    single-column solution, so it pins the free-path sampling, the null collisions and
    the isotropic emission all at once. The optically thin end is the demanding one --
    there the answer depends on the whole angular distribution rather than on a
    diffusivity angle.
    """
    tau = uniform(tau_tot/NZ)
    out = trace(rte3d, tau, uniform(B), np.zeros(NCOL), np.ones(NCOL), ppp=32768)

    expected = np.pi*B*(1. - 2.*exp_int_3(tau_tot))
    assert out['tod_up'].mean() == pytest.approx(expected, rel=0.01)

    # By symmetry, the slab radiates the same amount down onto the surface.
    assert out['sfc_dn'].mean() == pytest.approx(expected, rel=0.01)


def test_radiative_equilibrium_leaves_no_net_flux(rte3d):
    """An isothermal column closed at both ends neither warms nor cools.

    The atmosphere, the surface and the top boundary all radiate at the same
    temperature, so every cell absorbs exactly what it emits. The tracer gets this
    right cell by cell rather than only in the domain mean, because a photon's emission
    is scored as the weight it actually deposited: one reabsorbed where it was born
    cancels exactly instead of leaving two large numbers to cancel statistically.

    Without inc_dif the same column is *not* in equilibrium -- it radiates pi*B to
    space and gets nothing back -- which is what the next test checks.
    """
    out = trace(rte3d, uniform(1.0), uniform(B), np.full(NCOL, B), np.ones(NCOL),
                ppp=32768, inc_dif=np.pi*B)

    for key in ('tod_dn', 'tod_up', 'sfc_dn', 'sfc_up'):
        assert out[key].mean() == pytest.approx(np.pi*B, rel=0.02)

    # The residual is Monte Carlo noise on a quantity whose mean is zero, so it is
    # compared against the flux it is the small difference of.
    assert abs(column_net(out).mean()) < 0.02*np.pi*B


def test_an_open_top_cools_the_column_by_what_escapes(rte3d):
    """With nothing coming down from above, the column loses exactly what leaves it.

    The same isothermal atmosphere as above with inc_dif left at zero. It is opaque, so
    what it radiates to space is pi*B, and that is what the column has to be losing.
    """
    out = trace(rte3d, uniform(1.0), uniform(B), np.full(NCOL, B), np.ones(NCOL),
                ppp=32768)

    assert out['tod_up'].mean() == pytest.approx(np.pi*B, rel=0.02)
    assert column_net(out).mean() == pytest.approx(-np.pi*B, rel=0.03)


def test_a_transparent_atmosphere_shows_the_surface(rte3d):
    """A black surface under nothing radiates pi*B straight to space.

    Exact rather than statistical: with no atmosphere to absorb it, every photon the
    surface emits leaves through the top carrying the weight it was born with.
    """
    out = trace(rte3d, uniform(1e-12), uniform(0.0), np.full(NCOL, B), np.ones(NCOL))

    rtol = exact_rtol(rte3d)
    assert out['sfc_up'].mean() == pytest.approx(np.pi*B, rel=rtol)
    assert out['tod_up'].mean() == pytest.approx(np.pi*B, rel=rtol)
    assert out['sfc_dn'].mean() == 0.0
    assert np.abs(column_net(out)).max() < 1e-6*np.pi*B


def test_a_grey_surface_emits_what_its_emissivity_allows(rte3d):
    """Emissivity scales the surface source, and one minus it reflects the rest.

    Under a transparent atmosphere nothing comes down, so the whole upward flux is
    emission and it is scored exactly.
    """
    emis = 0.4
    out = trace(rte3d, uniform(1e-12), uniform(0.0), np.full(NCOL, B),
                np.full(NCOL, emis))

    assert out['sfc_up'].mean() == pytest.approx(emis*np.pi*B, rel=exact_rtol(rte3d))


@pytest.mark.parametrize('independent_column', [False, True])
def test_energy_budget_closes(rte3d, independent_column):
    """What enters the box leaves it or is absorbed, and exactly.

    Every photon's weight is accounted for where it is lost, so this is an identity of
    the scoring rather than a statement about the physics: it holds photon for photon,
    to the rounding of the sums. A tracer that dropped or double-counted weight
    anywhere -- at the surface, at a block face, at the top -- would fail it at once.

    Over the domain in general, and column by column when the transport is confined to
    a column. The difference is the whole point of tracing in three dimensions: a
    photon emitted over one column is absorbed over another, so a single column's books
    do not balance and the domain's still do.
    """
    rng = np.random.default_rng(0)
    tau = rng.uniform(0.02, 0.3, (NZ, NCOL))
    lay_source = rng.uniform(40., 120., (NZ, NCOL))

    out = trace(rte3d, tau, lay_source, np.full(NCOL, 125.), np.full(NCOL, 0.9),
                independent_column=independent_column)

    boundaries = out['tod_dn'] - out['tod_up'] + out['sfc_up'] - out['sfc_dn']
    rtol = exact_rtol(rte3d)

    assert boundaries.mean() == pytest.approx(column_net(out).mean(), rel=rtol)

    if independent_column:
        np.testing.assert_allclose(boundaries, column_net(out), rtol=rtol)
    else:
        # Horizontal transport really is what breaks the per-column identity, rather
        # than the columns happening to balance anyway.
        assert np.abs(boundaries - column_net(out)).max() > 0.1*np.pi*B


def test_conservative_scattering_neither_absorbs_nor_emits(rte3d):
    """A cloud that only scatters takes no part in the energy exchange.

    Its single-scattering albedo is one, so its absorption -- and with it its emission,
    which is the same coefficient -- is zero. It redirects the surface's radiation and
    nothing more, so what leaves the top and what comes back down to the surface add up
    to what the surface sent out.
    """
    tau_cld = np.zeros((NZ, NCOL))
    tau_cld[NZ//2:] = 0.5

    out = trace(rte3d, uniform(1e-12), uniform(0.0), np.full(NCOL, B), np.ones(NCOL),
                ppp=16384,
                tau_cld=tau_cld, ssa_cld=uniform(1.0), asy_cld=uniform(0.85))

    rtol = exact_rtol(rte3d)
    assert out['sfc_up'].mean() == pytest.approx(np.pi*B, rel=rtol)
    assert np.abs(out['flux_net'][NZ//2:]).max() < 1e-6*np.pi*B/DZ

    # Everything the surface emits comes back to it or leaves through the top.
    assert (out['tod_up'] + out['sfc_dn']).mean() == pytest.approx(np.pi*B, rel=rtol)


def test_both_vertical_orientations_give_the_same_answer(rte3d):
    """top_at_1 is read, not assumed: the same atmosphere either way up.

    The emitted power of every cell comes out the same either way, so the distribution
    photons are drawn from is bit for bit identical and the walks that follow are the
    same walks. An equality up to the order the threads add their photons in, then,
    rather than a statistical statement.
    """
    rng = np.random.default_rng(0)
    tau = rng.uniform(0.02, 0.3, (NZ, NCOL))
    lay_source = rng.uniform(40., 120., (NZ, NCOL))
    sfc_source = np.full(NCOL, 125.)
    sfc_emis = np.full(NCOL, 0.9)

    up = trace(rte3d, tau, lay_source, sfc_source, sfc_emis)
    down = trace(rte3d, tau[::-1].copy(), lay_source[::-1].copy(), sfc_source, sfc_emis,
                 top_at_1=True)

    rtol = exact_rtol(rte3d)
    for key in ('tod_up', 'sfc_dn', 'sfc_up'):
        np.testing.assert_allclose(down[key], up[key], rtol=rtol)
    np.testing.assert_allclose(down['flux_net'], up['flux_net'], rtol=rtol,
                               atol=1e-9*np.pi*B/DZ)


@pytest.mark.parametrize('nlay', [NZ, NZ + 8, NZ + 24])
def test_layers_above_the_box_are_lumped_into_its_top_cell(rte3d, nlay):
    """A deeper atmosphere on the same box radiates as the whole column would.

    The tracer resolves nz cells; everything above them is folded into the top one,
    optical depth and emission together. So a uniform atmosphere of a given total
    optical depth has to give the same answer however many layers it is handed as,
    and that answer is the closed-form one again.
    """
    tau_tot = 2.4
    out = trace(rte3d, uniform(tau_tot/nlay, nlay), uniform(B, nlay),
                np.zeros(NCOL), np.ones(NCOL), ppp=32768)

    expected = np.pi*B*(1. - 2.*exp_int_3(tau_tot))
    assert out['tod_up'].mean() == pytest.approx(expected, rel=0.01)


def test_uniform_atmosphere_matches_the_plane_parallel_solver(rte3d):
    """On a horizontally uniform column the two solvers solve the same problem.

    The plane-parallel one integrates over zenith angle by three-point quadrature where
    the tracer samples it, and it takes the source as linear in optical depth within a
    layer where the tracer takes it as uniform. Neither difference amounts to much on a
    profile this smooth, so they agree far more closely than the shortwave pair does.
    """
    lay_source = np.linspace(120., 40., NZ)[:, None]*np.ones((NZ, NCOL))
    lev_source = np.linspace(122.5, 37.5, NZ + 1)[:, None]*np.ones((NZ + 1, NCOL))
    sfc_source = np.full(NCOL, 125.)
    sfc_emis = np.ones(NCOL)
    tau = uniform(0.15)

    out = trace(rte3d, tau, lay_source, sfc_source, sfc_emis, ppp=32768,
                independent_column=True)

    secants = np.stack([np.full((1, NCOL), d) for d in GAUSS_D])
    flux_up, flux_dn, _ = rte3d.lw_solver_noscat(
        False, secants, GAUSS_WTS, tau[None], lay_source[None], lev_source[None],
        sfc_emis[None], sfc_source[None], np.zeros((1, NCOL)))

    assert out['tod_up'].mean() == pytest.approx(flux_up[0, -1].mean(), rel=0.01)
    assert out['sfc_dn'].mean() == pytest.approx(flux_dn[0, 0].mean(), rel=0.01)
    assert out['sfc_up'].mean() == pytest.approx(flux_up[0, 0].mean(), rel=0.01)

    # The heating rate profile, which is the difference of two large fluxes and the
    # quantity a longwave solve is really wanted for.
    net_pp = np.diff(flux_dn[0] - flux_up[0], axis=0).mean(axis=1)
    assert column_net(out).mean() == pytest.approx(net_pp.sum(), rel=0.02)


def test_a_cold_cloud_shades_the_clear_sky_beside_it(rte3d):
    """The point of tracing in three dimensions, and what independent columns cannot do.

    Half the domain carries a cold opaque lid. A photon leaving the clear half at a
    grazing angle travels a long way horizontally before it gets out, and the domain is
    periodic, so it runs into the lid over the other half and is absorbed. The
    independent-column run cannot see that: its clear half radiates to space as though
    the lid were not there.
    """
    tau = uniform(0.02)
    lay_source = uniform(B)

    cloudy = (np.arange(NCOL) % NX) < NX//2
    tau[NZ//2:, cloudy] = 3.0
    lay_source[NZ//2:, cloudy] = 5.0

    args = (tau, lay_source, np.full(NCOL, 120.), np.ones(NCOL))
    three_d = trace(rte3d, *args, ppp=16384)
    columns = trace(rte3d, *args, ppp=16384, independent_column=True)

    # The lid hides the warm air below it either way.
    assert columns['tod_up'][cloudy].mean() < 0.2*columns['tod_up'][~cloudy].mean()

    # Only the three-dimensional run lets it hide the clear half as well.
    assert three_d['tod_up'][~cloudy].mean() < 0.7*columns['tod_up'][~cloudy].mean()

    # Whatever the transport moves around, the budget still closes exactly.
    for out in (three_d, columns):
        boundaries = out['tod_dn'] - out['tod_up'] + out['sfc_up'] - out['sfc_dn']
        np.testing.assert_allclose(boundaries.mean(), column_net(out).mean(),
                                   rtol=exact_rtol(rte3d))
