# Results

What the cases produce: timings, comparisons against the reference, and what those
comparisons can and cannot check. [`README.md`](README.md) is how to run them.

## les_cloudfield

The three-dimensional cloud field the ray tracer was validated on, described in
[`README.md`](README.md#les_cloudfield--a-three-dimensional-cloud-field). This is the
ray tracer's benchmark: single precision, which is what the tracer is run at, on an
RTX A4500, with the settings file as it stands -- 512 photons per pixel in both bands.
Best of three runs of `run_case.py`, which spread by under 3 percent:

| | time | per column |
|---|---|---|
| longwave, 62 of 128 g-points traced | 2506 ms | 153 us |
| shortwave, 112 g-points | 3646 ms | 223 us |

The runs agree bit for bit: the photon counts are summed in fixed point, so the order
the GPU's atomics land in does not matter. That cost 1 and 3 percent against the
floating-point counters, which gave 2487 and 3550 ms and differed from run to run in
the last bits of a third of the columns.

The plane-parallel solves beside them take 802 and 720 ms. An earlier measurement on
one MI250X GCD, in double precision and at 256 photons per pixel, gave 3045 and 7950
ms; it is not comparable with the above and is kept only for the record.

### The reference fluxes, and what they can check

The same record holds `rte-rrtmgp-cpp`'s own fluxes for this field.
`--reference` fetches them, as `les_cloudfield_reference.nc`:

```bash
python cases/fetch_data.py les_cloudfield --reference     # 149 MB
```

They are worth having, but they check less than they look like they do, and it is
worth knowing which half is which before reading anything into a difference.

**The longwave surface fluxes are a real check**, and they pass:

| domain mean | reference | rte3d |
|---|---|---|
| `rt_lw_flux_sfc_up` | 456.951 | 456.959 (+0.00%) |
| `rt_lw_flux_sfc_dn` | 403.273 | 401.750 (-0.38%) |

**The top-of-domain fluxes are not comparable at all.** The reference discards the
132 layers above the ray-tracing box; rte3d lumps them into one more cell on top, so
it sees air the reference does not. That is the difference the note in
`include/raytracer_lw.h` is about, and it is large -- `rt_lw_flux_tod_up` is 374.2
against 331.8 -- but it is not an error on either side. To compare these, truncate the
input to the 200 resolved layers so both codes see the same air.

**The shortwave is not comparable either, because of aerosols.** The input carries
eleven aerosol mixing ratios, the reference reads them and rte3d has no aerosol optics
at all, so it solves a cleaner atmosphere:

| domain mean | reference | rte3d |
|---|---|---|
| `rt_flux_sfc_dir` | 660.19 | 718.10 (+8.77%) |
| `rt_flux_sfc_dif` | 194.98 | 143.67 (-26.31%) |
| the two together | 855.16 | 861.78 (+0.77%) |

Which is the signature of scattering moved from the direct beam into the diffuse rather
than of two codes disagreeing: the split moves by a quarter and the total by under a
percent. It also rules out the sun being the cause -- the solar zenith angle these
fluxes were made at is the input file's own 30 degrees, since the 45 degrees that
`test.ini` ships would have cut the total by 18% rather than 0.77%.

The record holds a third file, the backward camera's output. rte3d has no backward
tracer, so it is not fetched.

---

## RCEMIP

### What `--compare-fortran` actually times

Both sides run the same sequence and the same amount of work:

```
column gas amounts -> interpolation -> absorption optical depth
  -> Planck sources (longwave) or Rayleigh (shortwave) -> transport
```

The column gas amounts come from rte3d on both sides, because `get_col_dry` has no
`bind(C)` entry point; that is a small constant added to each and does not tilt the
comparison. Everything after it is rte3d's Kokkos kernels on one side and the
reference's Fortran kernels through ctypes on the other — four calls per solve, so the
ctypes overhead is negligible at these sizes.

**The reference kernels are serial on the host.** Their OpenMP and OpenACC directives
target accelerators, not host threads. So the multi-threaded ratio compares rte3d on N
threads against serial Fortran; use `OMP_NUM_THREADS=1` for a per-thread comparison.

### Where things stand

Longwave, 256 columns x 256 layers, double precision, 15-core Apple Silicon:

| | clang | gcc-16 | reference (gfortran) |
|---|---|---|---|
| 1 thread, total | 358 ms | 429 ms | **290 ms** |
| &nbsp;&nbsp;gas optics | 310 ms | | |
| &nbsp;&nbsp;transport | 59 ms | | |

Transport used to be 555 ms of that single-threaded total, two thirds of the runtime.
Each `(g-point, column)` pair ran an entire sequential vertical recurrence, which suits
a GPU but leaves the column loop with a single iteration's worth of independent work no
matter what `ivdep` asserts. Solving one g-point at a time restores the reference's
structure -- layer outside, `ncol` innermost -- and the recurrences now vectorize over
columns: a 9x improvement. Gas optics is what remains of the gap.

Where it really pays is at scale, because a single g-point's working set stays in
cache instead of streaming ~80 MB arrays through memory. Longwave, 15 threads:

| columns | before | now |
|---|---|---|
| 256 | 131 ms | 221 ms |
| 512 | 274 ms | 281 ms |
| 1024 | 474 ms | 425 ms |
| 4096 | 9495 ms, 11 GB | **1226 ms, 1.1 GB** |

Below about 512 columns the per-g-point solve is launch-bound: each g-point issues
around ten parallel regions of only `ncol` work each, and at 256 columns on 15 threads
that overhead outweighs the cache win. The fix is to parallelize over g-points on the
host, giving each thread its own working set -- now possible, since the g-point loop
lives in C++ rather than in the caller. It has not been done.

Both builds agree with the reference to 1e-13 W/m2, so none of this is bought with
accuracy.

### On a GPU

RCEMIP at 65536 columns x 256 layers, single precision, on an RTX A4500:

| | longwave | shortwave |
|---|---|---|
| before | 1879 ms | 2167 ms |
| now | **1314 ms** | **1531 ms** |

Every one of those milliseconds was memory traffic, not arithmetic and not
parallelism: the kernels were already running at 400-650 GB/s, against a card that
peaks near 640. So the work was to move less --- the binary-species interpolation
recomputed rather than stored (4.0 GB, and 24 bytes per cell per g-point), the Planck
fraction and the Rayleigh scattering taken from the interpolation the optical depth was
already doing, and the fluxes added into the spectral totals where they are produced
instead of being written per g-point and read straight back. Last, gas optics runs a
band of g-points at a time, so that the interpolation state, the binary-species
weights and the minor-absorber scaling are done once per band rather than once per
g-point. `git log` has the measurement for each.

What is left is the column sweeps. Against `rte-rrtmgp-cpp`'s CUDA solver on its own
RCEMIP case --- the same input file, 4096 columns, plane-parallel, clear sky, both
single precision on one RTX A4500 --- that is now the whole of the gap:

| stage | rte3d | rte-rrtmgp-cpp | ratio |
|---|---|---|---|
| longwave, optical depth + Planck | 23.2 ms | 28.0 ms | 0.83 |
| longwave, transport | **71.3 ms** | **42.0 ms** | **1.70** |
| shortwave, optical depth + Rayleigh | 8.9 ms | 28.4 ms | 0.31 |
| shortwave, transport | **164 ms** | **45.2 ms** | **3.62** |
| GPU kernel time, longwave / shortwave | 95 / 173 ms | 70 / 74 ms | |
| what the driver's own timer reports | 114 / 197 ms | 73 / 74 ms | |

Neither driver times a stage on its own, so the split is GPU kernel time from an `nsys`
kernel summary, attributed by kernel name: gas optics is rte3d's `compute_tau`,
`compute_planck_source` and `interpolation` against the reference's
`interpolation_kernel`, `gas_optical_depths_*`, `Planck_source_kernel`,
`compute_tau_rayleigh_kernel`, `combine_abs_and_rayleigh_kernel` and the
`zero_array_kernel` that clears their output; transport is everything in the solver, and the reference's `sum_broadband_kernel` counts with it since rte3d
accumulates the fluxes inside its sweeps rather than afterwards. The two runs agree on
the fluxes to 6e-5 W/m2 in the longwave and 2e-2 in the shortwave, which is
single-precision round-off over two summation orders, so they are solving the same
problem. Note that `test_rte_rrtmgp` autotunes its launch configurations on first
encounter and saves them beside the case; the timed run has to be the second one.

The staged `--breakdown` above cannot be used for this. It writes the whole
`(ngpt, nlay, ncol)` spectrum between the two stages, which is the memory traffic the
fused path exists to avoid, and it reports gas optics thirty times slower as a result.

Gas optics is ahead --- a sixth cheaper in the longwave, and a third of the cost in the
shortwave, where there is no Planck source to compute per g-point. The sweeps are not,
and the launch geometry says why: it solves four column blocks with every g-point
resolved, so its `sw_adding` runs 229376 threads and its `lw_solver_noscat_step_2`
262144, where ours run `ncol` --- 4096. Same work, 64x the parallelism, and at 4096
columns ours reach about 15% of peak bandwidth and 8% occupancy where at 65536 they
reach 70%. The gap is a small-problem gap: 48.3 us/column at 4096 against 23.4 at 65536,
while a chunked full-spectrum solver is flat in problem size and pays for it in
memory --- 4 column blocks here, 64 for the case above.

Block size is not the lever it looks like. Pinning the sweeps' CUDA block to 32 instead
of the 128 Kokkos picks spreads them over all 56 SMs rather than leaving 24 idle, and
changes the time by nothing: a block moves warps between SMs but does not create any,
and a sweep has ncol/32 of them whatever the block. The 2-D launches did respond to
tiling, which is a layout question rather than an occupancy one; see the note in
`include/types.h`.

Two ways out, neither taken: fuse the direct-beam sweep with the adding sweeps, which
move five `(nlay, ncol)` arrays between them, or give the transport arrays a g-point
block dimension, which buys the parallelism directly at the cost of the property this
design is built around.
