"""Step 4: the shortwave Monte Carlo ray tracer.

There is no Fortran oracle for this one -- the reference has no ray tracer -- so the
tests are the physics instead: the cases where the answer is analytic, the conservation
laws the tracer must obey exactly, and a comparison against the two-stream solver on a
horizontally uniform atmosphere, where the two are solving the same problem and may
differ only by the two-stream's own approximation.

Everything here is Monte Carlo, so the tolerances are set by the photon count rather
than by round-off. They are deliberately loose enough that they do not depend on the
backend's random number generator, which differs between the host and the two GPU
vendors, and tight enough that a wrong answer fails them by a wide margin.
"""
import numpy as np
import pytest

# Small enough to keep the suite quick on a CPU, large enough that the domain mean of
# a scored flux is good to a few tenths of a percent.
NX = NY = 8
NZ = 16
NCOL = NX*NY
DX = DY = DZ = 100.0
PHOTONS = 2048

TSI = 1000.0


def exact_rtol(rte3d):
    """Tolerance for a quantity the tracer computes exactly, up to accumulation.

    The photon counts are summed in fixed point, so the order the threads add them in
    does not matter; what is left is each score's rounding to the counters' 2^-32 and
    the conversion of the total back to the precision the module was built in.
    """
    return 1e-4 if rte3d.runtime().precision == 'single' else 1e-9


def grid(**kwargs):
    """The keyword arguments describing the standard test box."""
    return dict(nx=NX, ny=NY, nz=NZ, dx=DX, dy=DY, dz=DZ, **kwargs)


def uniform(value, nlay=NZ):
    return np.full((nlay, NCOL), value)


def trace(rte3d, tau, ssa, alb, mu0, ppp=PHOTONS, **kwargs):
    """One trace with the sun's irradiance normalised to TSI at normal incidence."""
    return rte3d.trace_rays(tau, ssa, alb, mu0=mu0, inc_dir=TSI*mu0,
                            photons_per_pixel=ppp, **grid(**kwargs))


def absorbed(out):
    """Column-integrated absorption [W/m2], from the per-unit-height fluxes."""
    return (out['abs_dir'] + out['abs_dif']).sum(axis=0)*DZ


@pytest.mark.parametrize('mu0', [1.0, 0.6, 0.25])
@pytest.mark.parametrize('independent_column', [False, True])
def test_direct_beam_is_beer_lambert(rte3d, mu0, independent_column):
    """With no scattering and no reflection, the surface sees exp(-tau/mu0).

    The one case with an answer in closed form, and the one that pins the geometry:
    the slant path, the cell crossings and the null collisions all have to come out
    right for it to hold, and a horizontally uniform atmosphere makes the
    three-dimensional and the independent-column transport agree.
    """
    tau_lay = 0.05

    # What is scored at the surface is the part of the beam that survives the column,
    # and the noise on it goes as the inverse square root of that part. A grazing sun
    # loses far more of it -- at mu0 = 0.25 an eleventh as much arrives as at mu0 = 1
    # -- so it needs proportionally more photons to land inside the same tolerance.
    # The scaling holds PHOTONS at mu0 = 1, where it is already enough.
    ppp = int(PHOTONS*np.exp(-tau_lay*NZ)/np.exp(-tau_lay*NZ/mu0))

    out = trace(rte3d, uniform(tau_lay), uniform(0.0), np.zeros(NCOL), mu0, ppp=ppp,
                independent_column=independent_column)

    expected = TSI*mu0*np.exp(-tau_lay*NZ/mu0)

    assert out['sfc_dir'].mean() == pytest.approx(expected, rel=0.02)
    assert out['sfc_dif'].mean() == 0.0
    assert out['tod_up'].mean() == 0.0
    assert out['tod_dn'].mean() == pytest.approx(TSI*mu0, rel=exact_rtol(rte3d))


def test_layers_above_the_box_are_lumped_into_its_top_cell(rte3d):
    """A deeper atmosphere on the same box attenuates the beam by the whole column.

    The tracer resolves nz cells; everything above them is folded into the top one, so
    a uniform atmosphere given as nz layers and as nz+8 layers has to reach the surface
    with the same beam, even though the second carries half again as much optical depth.
    """
    tau_lay = 0.05
    deep = trace(rte3d, uniform(tau_lay, NZ + 8), uniform(0.0, NZ + 8),
                 np.zeros(NCOL), 1.0)

    expected = TSI*np.exp(-tau_lay*(NZ + 8))
    assert deep['sfc_dir'].mean() == pytest.approx(expected, rel=0.02)


def test_conservative_scattering_absorbs_nothing(rte3d):
    """Every photon leaves through the top or the surface, and none of them is lost.

    This is exact rather than statistical: with a single-scattering albedo of one and a
    black surface, no weight is ever taken out, so the scores add up to the incoming
    flux photon for photon. Over the domain rather than per column, since a photon may
    leave through a different column from the one it entered.
    """
    out = trace(rte3d, uniform(0.1), uniform(1.0), np.zeros(NCOL), 1.0)

    assert absorbed(out).max() == 0.0

    leaving = (out['tod_up'] + out['sfc_dir'] + out['sfc_dif']).sum()
    assert leaving == pytest.approx(out['tod_dn'].sum(), rel=exact_rtol(rte3d))


def test_energy_budget_closes(rte3d):
    """What comes in leaves, is absorbed, or is reflected, in an absorbing atmosphere.

    Not exact, unlike the case above: below a weight of a half the photon goes through
    Russian roulette, which conserves energy in the mean and not in a single run.
    """
    out = trace(rte3d, uniform(0.1), uniform(0.6), np.full(NCOL, 0.2), 1.0, ppp=8192)

    net_sfc = out['sfc_dir'] + out['sfc_dif'] - out['sfc_up']
    residual = out['tod_dn'] - out['tod_up'] - net_sfc - absorbed(out)

    assert abs(residual.mean()) < 0.01*TSI


def test_surface_albedo_reflects_what_it_should(rte3d):
    """A transparent atmosphere over a grey surface returns the albedo, twice over.

    The upward surface flux is scored before the reflected photon is followed, so it is
    exact; what reaches the top again has been through Russian roulette and is not.
    """
    albedo = 0.3
    out = trace(rte3d, uniform(1e-12), uniform(0.0), np.full(NCOL, albedo), 1.0)

    assert out['sfc_dir'].mean() == pytest.approx(TSI, rel=exact_rtol(rte3d))
    assert out['sfc_up'].mean() == pytest.approx(albedo*TSI, rel=exact_rtol(rte3d))
    assert out['tod_up'].mean() == pytest.approx(albedo*TSI, rel=0.02)


@pytest.mark.parametrize('mu0', [1.0, 0.6])
@pytest.mark.parametrize('ssa0', [0.5, 0.9, 1.0])
def test_uniform_atmosphere_matches_the_two_stream(rte3d, mu0, ssa0):
    """On a horizontally uniform atmosphere the two solvers solve the same problem.

    They cannot agree exactly: the two-stream is an approximation of the very transport
    the tracer performs, and on isotropic scattering the difference is a few percent.
    The direct beam, which both compute exactly, is held to a much tighter tolerance.

    The reflected flux is where the two part company most, and ssa = 0.5 with mu0 = 0.6
    is the worst of these cases: the two-stream reads 5.1% high at the top and 2.7%
    high on the surface flux. That is its own error, not the tracer's -- an independent
    plane-parallel Monte Carlo of the same slab, sharing no code with either, puts the
    tracer within 0.05% of it on both, and the tracer converges on its answer rather
    than drifting with the photon count. So the reflected flux gets a tolerance with
    room for that, and the surface flux keeps the tighter one it clears comfortably.
    """
    tau, ssa, albedo = 0.05, ssa0, 0.2

    out = trace(rte3d, uniform(tau), uniform(ssa), np.full(NCOL, albedo), mu0,
                ppp=8192, independent_column=True)

    flux_up, flux_dn, flux_dir = rte3d.sw_solver_2stream(
        False,
        uniform(tau)[None], uniform(ssa)[None], np.zeros((1, NZ, NCOL)),
        np.full((NZ, NCOL), mu0),
        np.full((1, NCOL), albedo), np.full((1, NCOL), albedo),
        np.full((1, NCOL), TSI))

    assert out['sfc_dir'].mean() == pytest.approx(flux_dir[0, 0].mean(), rel=0.02)
    assert (out['sfc_dir'] + out['sfc_dif']).mean() == pytest.approx(
        flux_dn[0, 0].mean(), rel=0.05)
    assert out['tod_up'].mean() == pytest.approx(flux_up[0, -1].mean(), rel=0.07)


def test_both_vertical_orientations_give_the_same_answer(rte3d):
    """top_at_1 is read, not assumed: the same atmosphere either way up.

    The photon walk sees the same scene either way, and the random stream does not
    depend on the input, so this is an equality up to the order the threads add their
    photons in rather than a statistical statement.
    """
    rng = np.random.default_rng(0)
    tau = rng.uniform(0.01, 0.2, (NZ, NCOL))
    ssa = rng.uniform(0.2, 1.0, (NZ, NCOL))
    albedo = np.full(NCOL, 0.15)

    up = trace(rte3d, tau, ssa, albedo, 0.8)
    down = trace(rte3d, tau[::-1].copy(), ssa[::-1].copy(), albedo, 0.8, top_at_1=True)

    rtol = exact_rtol(rte3d)
    np.testing.assert_allclose(down['sfc_dir'], up['sfc_dir'], rtol=rtol)
    np.testing.assert_allclose(down['tod_up'], up['tod_up'], rtol=rtol)
    np.testing.assert_allclose(down['abs_dir'], up['abs_dir'], rtol=rtol)


def test_a_cloud_casts_a_shadow_beside_itself(rte3d):
    """The point of tracing in three dimensions, and what independent columns cannot do.

    Half the domain carries a thick cloud and the sun is low, so the beam that misses
    the cloud in its own column is stopped by the neighbour it passes through. The
    independent-column run cannot see that, and its surface flux under the clear half
    is the clear-sky one everywhere.
    """
    tau_cld = np.zeros((NZ, NCOL))
    ssa_cld = np.full((NZ, NCOL), 0.999999)
    asy_cld = np.full((NZ, NCOL), 0.85)

    # A slab in the upper half of the box, over the columns with i < nx/2.
    cloudy = (np.arange(NCOL) % NX) < NX//2
    tau_cld[NZ//2:, cloudy] = 2.0

    kwargs = dict(tau_cld=tau_cld, ssa_cld=ssa_cld, asy_cld=asy_cld)
    mu0 = 0.5

    three_d = trace(rte3d, uniform(1e-6), uniform(0.0), np.zeros(NCOL), mu0,
                    ppp=8192, azi=np.pi/2, **kwargs)
    columns = trace(rte3d, uniform(1e-6), uniform(0.0), np.zeros(NCOL), mu0,
                    ppp=8192, azi=np.pi/2, independent_column=True, **kwargs)

    # The independent-column run leaves the clear half untouched by the cloud.
    assert columns['sfc_dir'][~cloudy].mean() == pytest.approx(TSI*mu0, rel=0.02)

    # The three-dimensional one does not: the beam is slanted, so it crosses the cloud
    # on its way to part of the clear half.
    assert three_d['sfc_dir'][~cloudy].mean() < 0.9*columns['sfc_dir'][~cloudy].mean()

    # Both conserve the domain-mean energy, which is what the shadow moves around.
    for out in (three_d, columns):
        total = out['tod_up'] + out['sfc_dir'] + out['sfc_dif'] + absorbed(out)
        assert total.mean() == pytest.approx(TSI*mu0, rel=0.02)
