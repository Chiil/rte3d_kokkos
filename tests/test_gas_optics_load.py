"""Step 2b: the k-distribution reduction.

The reference reduces the coefficient file to the gases the host supplies and rebuilds
every integer index array against that shorter list. None of that has a bind(C) entry
point, so tests/shim/rte3d_shim.F90 exposes it; see tests/build_reference.sh.

Index bases differ and are converted explicitly at each comparison below: the reference
is 1-based for flavours, g-point limits and kminor starts, rte3d is 0-based. The gas
indices are the exception -- the reference's 1-based position in the reduced gas list
is already rte3d's index into col_gas, whose entry 0 is dry air.
"""
import numpy as np
import pytest

FILE_GASES = ['h2o', 'co2', 'o3', 'n2o', 'co', 'ch4', 'o2', 'n2']

# Minor absorber identifiers and the gas each belongs to, as gas_minor/identifier_minor
# pair them in a real file (h2o_self is a water vapour continuum term, and so on).
MINOR = [
    ('h2o_self', 'h2o'), ('h2o_frgn', 'h2o'), ('co2_minor', 'co2'),
    ('o3_minor', 'o3'), ('n2o_minor', 'n2o'), ('ch4_minor', 'ch4'),
    ('o2_minor', 'o2'), ('n2_minor', 'n2'), ('co_minor', 'co'),
]


def coefficient_file(rng, nbnd=6, gpts_per_band=4, ntemp=8, npres=20, neta=9,
                     nplancktemp=32):
    """A synthetic coefficient file with a real file's string structure.

    Indices are 1-based, as they are in the NetCDF.
    """
    ngpt = nbnd * gpts_per_band
    ngas = len(FILE_GASES)

    band2gpt = np.stack([
        np.arange(nbnd)*gpts_per_band + 1,
        np.arange(nbnd)*gpts_per_band + gpts_per_band], axis=1).astype(np.int32)

    # Two key species per band and atmosphere half, 1-based into FILE_GASES. A zero
    # pair means "no key species" and is rewritten to (2,2) by the reference.
    key_species = rng.integers(1, ngas + 1, (nbnd, 2, 2)).astype(np.int32)
    key_species[0, 0, :] = 0
    key_species[nbnd - 1, 1, :] = 0

    def minor_set(n, seed):
        r = np.random.default_rng(seed)
        which = r.choice(len(MINOR), n, replace=False)
        bands = r.integers(0, nbnd, n)
        limits = np.ascontiguousarray(band2gpt[bands].astype(np.int32))
        widths = limits[:, 1] - limits[:, 0] + 1
        starts = (np.concatenate([[0], np.cumsum(widths)[:-1]]) + 1).astype(np.int32)

        # An empty scaling gas means no second gas scales this absorber.
        scaling = [FILE_GASES[i] if f else '' for i, f in
                   zip(r.integers(0, ngas, n), r.integers(0, 2, n))]

        return dict(
            gases=[MINOR[i][0] for i in which],
            limits=limits,
            starts=starts,
            n_contrib=int(widths.sum()),
            scaling=scaling,
            density=r.integers(0, 2, n).astype(np.int8),
            complement=r.integers(0, 2, n).astype(np.int8),
        )

    lo = minor_set(6, 100)
    up = minor_set(5, 101)

    press_ref = np.exp(np.linspace(np.log(1.1e5), np.log(1.0), npres))
    temp_ref = np.linspace(160.0, 355.0, ntemp)

    return dict(
        gas_names=FILE_GASES,
        gas_minor=[g for _, g in MINOR],
        identifier_minor=[i for i, _ in MINOR],
        minor_gases_lower=lo['gases'], minor_gases_upper=up['gases'],
        scaling_gas_lower=lo['scaling'], scaling_gas_upper=up['scaling'],
        key_species=key_species,
        band2gpt=band2gpt,
        band_lims_wavenum=np.stack([
            np.arange(nbnd)*100.0 + 10.0, np.arange(nbnd)*100.0 + 105.0], axis=1),
        press_ref=press_ref,
        temp_ref=temp_ref,
        press_ref_trop=9948.431564193395,
        temp_ref_p=160.0,
        temp_ref_t=355.0,
        vmr_ref=rng.uniform(1e-9, 1e-2, (ntemp, ngas + 1, 2)),
        kmajor=rng.uniform(0.0, 1e-2, (ngpt, npres + 1, neta, ntemp)),
        kminor_lower=rng.uniform(0.0, 1e-3, (lo['n_contrib'], neta, ntemp)),
        kminor_upper=rng.uniform(0.0, 1e-3, (up['n_contrib'], neta, ntemp)),
        minor_limits_gpt_lower=lo['limits'], minor_limits_gpt_upper=up['limits'],
        minor_scales_with_density_lower=lo['density'],
        minor_scales_with_density_upper=up['density'],
        scale_by_complement_lower=lo['complement'],
        scale_by_complement_upper=up['complement'],
        kminor_start_lower=lo['starts'], kminor_start_upper=up['starts'],
        totplnk=rng.uniform(0.0, 100.0, (nbnd, nplancktemp)),
        planck_frac=rng.uniform(0.0, 1.0, (ngpt, npres + 1, neta, ntemp)),
        optimal_angle_fit=rng.uniform(0.0, 2.0, (nbnd, 2)),
    )


def make_gas_concs(rte3d, names):
    g = rte3d.Gas_concs()
    for name in names:
        g.set_vmr(name, 1e-6)
    return g


# The key species must all be present, so every subset below keeps them.
SUBSETS = [
    FILE_GASES,                                   # nothing removed
    ['h2o', 'co2', 'o3', 'n2o', 'co', 'ch4', 'o2', 'n2'],
]


@pytest.mark.parametrize('drop', [[], ['co'], ['co', 'n2']])
def test_reduction_matches_reference(rte3d, fortran_shim, drop):
    """Compare rte3d's reduction against ty_gas_optics_rrtmgp%load.

    Dropping a gas removes its minor absorbers and renumbers every surviving block in
    kminor, which is the part most likely to go wrong.
    """
    rng = np.random.default_rng(70)
    f = coefficient_file(rng)

    # Key species must be supplied, so only drop gases that are not key species.
    key_gases = {FILE_GASES[i - 1] for i in np.unique(f['key_species']) if i > 0}
    available = [g for g in FILE_GASES if g not in drop or g in key_gases]

    expected = fortran_shim.load(f, available)
    loaded = rte3d.load_kdist(f, make_gas_concs(rte3d, available))
    actual = loaded.arrays()

    assert loaded.gas_names == available

    # Flavours: the reference's values are already col_gas indices.
    np.testing.assert_array_equal(actual['flavor'], expected['flavor'])
    np.testing.assert_array_equal(actual['gpoint_flavor'], expected['gpoint_flavor'] - 1)
    np.testing.assert_allclose(actual['vmr_ref'], expected['vmr_ref'], rtol=1e-15)

    for prefix in ('lower_', 'upper_'):
        np.testing.assert_array_equal(
            actual[prefix + 'minor_limits_gpt'], expected[prefix + 'minor_limits_gpt'] - 1)
        np.testing.assert_array_equal(
            actual[prefix + 'kminor_start'], expected[prefix + 'kminor_start'] - 1)
        np.testing.assert_array_equal(
            actual[prefix + 'idx_minor'], expected[prefix + 'idx_minor'])
        np.testing.assert_array_equal(
            actual[prefix + 'idx_minor_scaling'], expected[prefix + 'idx_minor_scaling'])
        np.testing.assert_array_equal(
            actual[prefix + 'scales_with_density'].astype(np.int32),
            expected[prefix + 'scales_with_density'])
        np.testing.assert_array_equal(
            actual[prefix + 'scale_by_complement'].astype(np.int32),
            expected[prefix + 'scale_by_complement'])
        np.testing.assert_allclose(
            actual[prefix + 'kminor'], expected[prefix + 'kminor'], rtol=1e-15)


def test_reduction_drops_absorbers_and_tiles_kminor(rte3d):
    """Structural invariants that hold without a reference.

    After reduction the surviving absorbers' blocks must tile kminor exactly, with no
    gap and no overlap. That is what the n_elim bookkeeping in reduce_minor_arrays is
    for, and it is the easiest thing to get wrong.
    """
    rng = np.random.default_rng(71)
    f = coefficient_file(rng)

    full = rte3d.load_kdist(f, make_gas_concs(rte3d, FILE_GASES)).arrays()
    key_gases = {FILE_GASES[i - 1] for i in np.unique(f['key_species']) if i > 0}
    reduced_names = [g for g in FILE_GASES if g in key_gases or g not in ('co', 'n2')]
    reduced = rte3d.load_kdist(f, make_gas_concs(rte3d, reduced_names)).arrays()

    for prefix in ('lower_', 'upper_'):
        for k in (full, reduced):
            starts = k[prefix + 'kminor_start']
            limits = k[prefix + 'minor_limits_gpt']
            widths = limits[:, 1] - limits[:, 0] + 1

            order = np.argsort(starts)
            assert starts[order][0] == 0, 'blocks must start at zero'
            np.testing.assert_array_equal(
                starts[order][1:], np.cumsum(widths[order])[:-1],
                err_msg=f'{prefix}kminor blocks do not tile')
            assert starts[order][-1] + widths[order][-1] == k[prefix + 'kminor'].shape[0]

        assert (reduced[prefix + 'kminor'].shape[0]
                <= full[prefix + 'kminor'].shape[0]), 'reduction must not grow the table'


def test_missing_key_species_is_rejected(rte3d):
    rng = np.random.default_rng(72)
    f = coefficient_file(rng)

    key_gases = sorted({FILE_GASES[i - 1] for i in np.unique(f['key_species']) if i > 0})
    without_a_key_gas = [g for g in FILE_GASES if g != key_gases[0]]

    with pytest.raises(ValueError, match='Key species'):
        rte3d.load_kdist(f, make_gas_concs(rte3d, without_a_key_gas))
