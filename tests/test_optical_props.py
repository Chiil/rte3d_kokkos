"""Step 1e: delta-scaling, the increment operations, and flux reduction."""
import numpy as np
import pytest

from compare import assert_close

SHAPES = [(16, 42, 7), (1, 3, 5), (8, 1, 4)]


def tolerance(rte3d):
    return 1e-5 if rte3d.runtime().precision == 'single' else 1e-12


def props(rng, ngpt, nlay, ncol, nmom=None):
    """A set of optical properties. p has the moment dimension last, the reverse of
    the reference's (nmom, ncol, nlay, ngpt)."""
    out = dict(
        tau=10.0**rng.uniform(-6.0, 2.0, (ngpt, nlay, ncol)),
        ssa=rng.uniform(0.0, 1.0, (ngpt, nlay, ncol)),
        g=rng.uniform(0.0, 1.0, (ngpt, nlay, ncol)),
    )
    if nmom is not None:
        out['p'] = rng.uniform(-1.0, 1.0, (ngpt, nlay, ncol, nmom))
    return out


def bands(ngpt, nbnd):
    """Contiguous band limits covering all g-points.

    Returns rte3d's form -- (nbnd, 2), 0-based, inclusive -- and the reference's,
    which is 1-based. The reference declares gpt_lims as Fortran (2, nbnd); in
    column-major that is the same memory as C-order (nbnd, 2), so only the base
    changes, not the shape.
    """
    edges = np.linspace(0, ngpt, nbnd + 1).astype(int)
    lims = np.stack([edges[:-1], edges[1:] - 1], axis=1).astype(np.int32)
    return np.ascontiguousarray(lims), np.ascontiguousarray(lims + 1)


@pytest.mark.parametrize('use_f', [False, True])
@pytest.mark.parametrize('ngpt,nlay,ncol', SHAPES)
def test_delta_scale_matches_reference(rte3d, fortran_ref, use_f, ngpt, nlay, ncol):
    rng = np.random.default_rng(20)
    a = props(rng, ngpt, nlay, ncol)
    f = rng.uniform(0.0, 1.0, (ngpt, nlay, ncol)) if use_f else None

    expected = fortran_ref.delta_scale_2str(a['tau'], a['ssa'], a['g'], f)
    actual = rte3d.delta_scale_2str(a['tau'], a['ssa'], a['g'], f)

    for name, exp, act in zip(('tau', 'ssa', 'g'), expected, actual):
        assert_close(act, exp, rtol=tolerance(rte3d), err_msg=f'{name} differs')


@pytest.mark.parametrize('ngpt,nlay,ncol', SHAPES)
def test_increments_match_reference(rte3d, fortran_ref, ngpt, nlay, ncol):
    """All seven distinct increment operations, at matching spectral resolution."""
    rng = np.random.default_rng(21)
    a = props(rng, ngpt, nlay, ncol, nmom=4)
    b = props(rng, ngpt, nlay, ncol, nmom=4)
    tol = tolerance(rte3d)

    def check(expected, actual, label):
        expected = expected if isinstance(expected, tuple) else (expected,)
        actual = actual if isinstance(actual, tuple) else (actual,)
        for i, (exp, act) in enumerate(zip(expected, actual)):
            assert_close(act, exp, rtol=tol,
                                       err_msg=f'{label}, output {i}')

    check(fortran_ref.increment_1scalar_by_1scalar(a['tau'], b['tau']),
          rte3d.increment_1scalar_by_1scalar(a['tau'], b['tau']),
          '1scalar_by_1scalar')

    check(fortran_ref.increment_1scalar_by_2stream(a['tau'], b['tau'], b['ssa']),
          rte3d.increment_1scalar_by_2stream(a['tau'], b['tau'], b['ssa']),
          '1scalar_by_2stream')

    check(fortran_ref.increment_2stream_by_1scalar(a['tau'], a['ssa'], b['tau']),
          rte3d.increment_2stream_by_1scalar(a['tau'], a['ssa'], b['tau']),
          '2stream_by_1scalar')

    check(fortran_ref.increment_2stream_by_2stream(
              a['tau'], a['ssa'], a['g'], b['tau'], b['ssa'], b['g']),
          rte3d.increment_2stream_by_2stream(
              a['tau'], a['ssa'], a['g'], b['tau'], b['ssa'], b['g']),
          '2stream_by_2stream')

    check(fortran_ref.increment_2stream_by_nstream(
              a['tau'], a['ssa'], a['g'], b['tau'], b['ssa'], b['p']),
          rte3d.increment_2stream_by_nstream(
              a['tau'], a['ssa'], a['g'], b['tau'], b['ssa'], b['p']),
          '2stream_by_nstream')

    check(fortran_ref.increment_nstream_by_2stream(
              a['tau'], a['ssa'], a['p'], b['tau'], b['ssa'], b['g']),
          rte3d.increment_nstream_by_2stream(
              a['tau'], a['ssa'], a['p'], b['tau'], b['ssa'], b['g']),
          'nstream_by_2stream')

    check(fortran_ref.increment_nstream_by_nstream(
              a['tau'], a['ssa'], a['p'], b['tau'], b['ssa'], b['p']),
          rte3d.increment_nstream_by_nstream(
              a['tau'], a['ssa'], a['p'], b['tau'], b['ssa'], b['p']),
          'nstream_by_nstream')


def test_increment_1scalar_by_nstream_is_the_same_operation(rte3d, fortran_ref):
    """The reference's increment_1scalar_by_nstream has a body identical to
    increment_1scalar_by_2stream, and nstream_by_1scalar to 2stream_by_1scalar, so
    rte3d exposes one function for each pair. Confirm that against the reference."""
    rng = np.random.default_rng(22)
    a = props(rng, 8, 5, 4)
    b = props(rng, 8, 5, 4)

    ours = rte3d.increment_1scalar_by_2stream(a['tau'], b['tau'], b['ssa'])
    theirs = fortran_ref.increment_1scalar_by_2stream(a['tau'], b['tau'], b['ssa'])
    assert_close(ours, theirs, rtol=tolerance(rte3d))


@pytest.mark.parametrize('nbnd', [1, 3, 8])
def test_byband_increments_match_reference(rte3d, fortran_ref, nbnd):
    """The by-band routines add properties defined per band to properties defined per
    g-point. rte3d gets there with a g-point to band map rather than a second copy of
    every routine."""
    ngpt, nlay, ncol = 16, 6, 5
    rng = np.random.default_rng(23)
    a = props(rng, ngpt, nlay, ncol)
    b = props(rng, nbnd, nlay, ncol)

    lims, ref_lims = bands(ngpt, nbnd)
    tol = tolerance(rte3d)

    assert_close(
        rte3d.increment_1scalar_by_1scalar(a['tau'], b['tau'], gpt_lims=lims),
        fortran_ref.inc_1scalar_by_1scalar_bybnd(a['tau'], b['tau'], ref_lims),
        rtol=tol)

    assert_close(
        rte3d.increment_1scalar_by_2stream(a['tau'], b['tau'], b['ssa'], gpt_lims=lims),
        fortran_ref.inc_1scalar_by_2stream_bybnd(a['tau'], b['tau'], b['ssa'], ref_lims),
        rtol=tol)

    for exp, act in zip(
            fortran_ref.inc_2stream_by_1scalar_bybnd(a['tau'], a['ssa'], b['tau'], ref_lims),
            rte3d.increment_2stream_by_1scalar(a['tau'], a['ssa'], b['tau'], gpt_lims=lims)):
        assert_close(act, exp, rtol=tol)

    for exp, act in zip(
            fortran_ref.inc_2stream_by_2stream_bybnd(
                a['tau'], a['ssa'], a['g'], b['tau'], b['ssa'], b['g'], ref_lims),
            rte3d.increment_2stream_by_2stream(
                a['tau'], a['ssa'], a['g'], b['tau'], b['ssa'], b['g'], gpt_lims=lims)):
        assert_close(act, exp, rtol=tol)


def test_extract_subset(rte3d):
    rng = np.random.default_rng(24)
    a = props(rng, 5, 4, 10, nmom=3)

    # extract_subset is a plain slice, so it is exact once the expected values are cast
    # to the precision the module stored the input at.
    tau = rte3d.extract_subset(a['tau'], 2, 7)
    np.testing.assert_array_equal(tau, a['tau'][:, :, 2:7].astype(tau.dtype))
    p = rte3d.extract_subset_4d(a['p'], 2, 7)
    np.testing.assert_array_equal(p, a['p'][:, :, 2:7, :].astype(p.dtype))
    assert_close(
        rte3d.extract_subset_absorption_tau(a['tau'], a['ssa'], 2, 7),
        (a['tau']*(1.0 - a['ssa']))[:, :, 2:7],
        rtol=tolerance(rte3d))


@pytest.mark.parametrize('ngpt,nlev,ncol', [(16, 43, 7), (1, 4, 5)])
def test_flux_reduction_matches_reference(rte3d, fortran_ref, ngpt, nlev, ncol):
    rng = np.random.default_rng(25)
    dn = rng.uniform(0.0, 500.0, (ngpt, nlev, ncol))
    up = rng.uniform(0.0, 500.0, (ngpt, nlev, ncol))
    tol = tolerance(rte3d)

    assert_close(rte3d.sum_broadband(dn), fortran_ref.sum_broadband(dn), rtol=tol)
    assert_close(
        rte3d.net_broadband(dn, up),
        fortran_ref.net_broadband_full(dn, up),
        rtol=tol)

    dn_bb = dn.sum(axis=0)
    up_bb = up.sum(axis=0)
    assert_close(
        rte3d.net_broadband(dn_bb, up_bb),
        fortran_ref.net_broadband_precalc(dn_bb, up_bb),
        rtol=tol)


@pytest.mark.parametrize('nbnd', [1, 4, 16])
def test_byband_reduction(rte3d, nbnd):
    """Checked against numpy rather than the reference.

    rte_sum_byband and rte_net_byband_full live in extensions/mo_fluxes_byband.F90,
    which is not part of the kernels the oracle library is built from. They are plain
    band-wise sums, so an explicit numpy reduction is an equally strong check.
    """
    ngpt, nlev, ncol = 16, 11, 5
    rng = np.random.default_rng(26)
    dn = rng.uniform(0.0, 500.0, (ngpt, nlev, ncol))
    up = rng.uniform(0.0, 500.0, (ngpt, nlev, ncol))

    lims, _ = bands(ngpt, nbnd)
    tol = tolerance(rte3d)

    expected_sum = np.stack([dn[g0:g1+1].sum(axis=0) for g0, g1 in lims])
    expected_net = np.stack([(dn - up)[g0:g1+1].sum(axis=0) for g0, g1 in lims])

    assert_close(rte3d.sum_byband(lims, dn), expected_sum, rtol=tol)
    assert_close(rte3d.net_byband(lims, dn, up), expected_net, rtol=tol)
