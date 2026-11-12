# Results

What the cases produce: timings, comparisons against the reference, and what those
comparisons can and cannot check. [`README.md`](README.md) is how to run them.

## les_cloudfield

The three-dimensional cloud field the ray tracer was validated on, described in
[`README.md`](README.md#les_cloudfield--a-three-dimensional-cloud-field). This is the
ray tracer's benchmark: single precision, which is what the tracer is run at, on an
RTX A4500, with the settings file as it stands -- 512 photons per pixel in both bands.
Best of three runs of `run_case.py`, which spread by under 6 percent, against best of
three of `rte-rrtmgp-cpp`'s `test_rte_rrtmgp_rt` on the same field with the same
settings. The plane-parallel solves the case runs beside the tracers are in the table
too:

| | rte3d, solver | rte3d, wall | rte-rrtmgp-cpp | | rte3d, memory | rte-rrtmgp-cpp, memory |
|---|---|---|---|---|---|---|
| longwave, traced, 62 of 128 g-points | **1418 ms** | 1478 ms | 3579 ms | 2.5x | **1304 MiB** | 1722-2106 MiB |
| shortwave, traced, 112 g-points | **2820 ms** | 2874 ms | 5425 ms | 1.9x | **1006 MiB** | 1496 MiB |
| longwave, two-stream | **329 ms** | 392 ms | 778 ms | 2.4x | **1186 MiB** | 1754-1786 MiB |
| shortwave, two-stream | **293 ms** | 369 ms | 600 ms | 2.0x | **1174 MiB** | 1592-1656 MiB |

The solver column is what to compare. It is the `solve_time` each solver returns:
device time between two fences, with every input already uploaded and every output
still on the device -- the span `rte-rrtmgp-cpp` times with CUDA events, covering gas
optics, cloud optics and transport on both sides. The wall column is the whole Python
call and adds the copies to and from numpy, 55 to 75 ms here. It used to add much more: the
case driver computed the cloud optics on the device, copied 2 GB of it into numpy and
handed it back, until the solvers took the cloud physics and did that themselves. The
tracers' wall time was then 1870 and 3218 ms, the fluxes the same bit for bit.

Memory is the peak GPU memory of the whole process, one band and one solver per
process, as `nvidia-smi --query-compute-apps` reports it polled every 10 ms. It
includes the CUDA context and every input and output field on the device, so it
overstates what the radiation adds to a host model that already holds those fields
there. rte3d takes a quarter to a third less than the reference, on every solver. rte3d peaks at the same value in every run. The reference does not: its
stream-ordered memory pool reserves and releases in bursts that the polling catches
or misses, so it is given as the range over three runs. What rte3d holds is one
g-point's working set -- the gas optics' interpolation state and column amounts for the
whole domain, one g-point's optical properties, one band's cloud properties, and the
transport or tracer scratch -- and nothing with a spectral dimension.

The two-stream rows are the reference's `test_rte_rrtmgp_rt` too, with `raytracing =
false` and `plane_parallel = true`, which runs the two-stream solver a g-point at a time
as rte3d does. Its other driver, `test_rte_rrtmgp`, solves the whole spectrum for
blocks of 1024 columns instead; that was not run here, and it would hold a fixed
amount of memory whatever the domain.

The reference was run from a copy of its `les_cloudfield` directory, with `test.ini`
set to match `les_cloudfield.toml`: `samples = 512` in the shortwave and `23` in the
longwave (2^23 photons over 128 x 128 columns is 512 per pixel), longwave
`scattering = true`, backward tracing off, and the input file's own sun, `sza = 30.0`
and `azi = 108.8975`. The sun has to be written as a float: `sza = 30` is a TOML
integer, which the reference ignores for its default of -1, and the run then
transmits 20 percent more direct beam. With the settings matched, the shortwave
surface and top-of-domain fluxes agree to 0.04 percent and the longwave surface up to
0.01; longwave surface down differs by 0.7 percent, which is the air above the box
that the reference discards -- see below. Its timer excludes the first launch's
autotuning, since the tuned configurations are read from the case directory.

The runs agree bit for bit: the photon counts are summed in fixed point, so the order
the GPU's atomics land in does not matter. That cost 1 and 3 percent against the
floating-point counters, which differed from run to run in the last bits of a third of
the columns. Seeding each thread's generator directly, rather than skipping it ahead
to a subsequence of its own, took 10 percent off both bands; `include/random.h` says
why.

An earlier measurement of the tracers on one MI250X GCD, in double precision and at 256
photons per pixel, gave 3045 and 7950 ms; it is not comparable with the above and is
kept only for the record.

### The reference fluxes, and what they can check

The same record holds `rte-rrtmgp-cpp`'s own fluxes for this field.
`--reference` fetches them, as `les_cloudfield_reference.nc`:

```bash
python cases/fetch_data.py les_cloudfield --reference     # 149 MB
```

They are worth having, but they check less than they look like they do, and it is
worth knowing which half is which before reading anything into a difference.

**The longwave fluxes are a real check**, and they pass:

| domain mean | reference | rte3d |
|---|---|---|
| `rt_lw_flux_sfc_up` | 456.951 | 456.935 (-0.00%) |
| `rt_lw_flux_sfc_dn` | 403.273 | 402.095 (-0.29%) |
| `rt_lw_flux_tod_up` | 374.2 | 374.327 (+0.03%) |

**The top-of-domain flux is comparable because the case sets `lump-above = false`.** The
reference discards the 132 layers above the ray-tracing box, and so does rte3d with that
setting, so the two codes see the same air. With the layers lumped into one more cell on
top instead -- `lump-above = true` -- rte3d sees air the reference does not, and
`rt_lw_flux_tod_up` comes out at 331.8 against 374.2. That difference is what the note in
`include/raytracer_lw.h` is about, and it is not an error on either side.

**The shortwave is not comparable, because of aerosols.** The input carries
eleven aerosol mixing ratios, the reference reads them and rte3d has no aerosol optics
at all, so it solves a cleaner atmosphere:

| domain mean | reference | rte3d |
|---|---|---|
| `rt_flux_sfc_dir` | 660.19 | 718.09 (+8.77%) |
| `rt_flux_sfc_dif` | 194.98 | 143.66 (-26.32%) |
| the two together | 855.16 | 861.75 (+0.77%) |

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

The column gas amounts come from rte3d on both sides, because `get_layer_number` has no
`bind(C)` entry point; that is a small constant added to each and does not tilt the
comparison. Everything after it is rte3d's Kokkos kernels on one side and the
reference's Fortran kernels through ctypes on the other — four calls per solve, so the
ctypes overhead is negligible at these sizes.

**The reference kernels are serial on the host.** Their OpenMP and OpenACC directives
target accelerators, not host threads. So the multi-threaded ratio compares rte3d on N
threads against serial Fortran; use `OMP_NUM_THREADS=1` for a per-thread comparison.

### Where things stand

Longwave, RCEMIP at 256 layers, double precision, on a 16-core Intel i9-12900 (8
performance and 8 efficiency cores), g++ 13.3 against gfortran 13.3. rte3d is one
thread per physical core (`OMP_PLACES=cores OMP_PROC_BIND=close`); the reference is
serial, as above. Best of three:

| columns | rte3d, 1 thread | reference, 1 thread | ratio | rte3d, 16 threads |
|---|---|---|---|---|
| 256 | 743 ms | **554 ms** | 0.75 | 103 ms |
| 512 | 1478 ms | **1140 ms** | 0.77 | 198 ms |
| 1024 | 2969 ms | **2405 ms** | 0.81 | 390 ms |
| 4096 | 12074 ms | **10840 ms** | 0.90 | 1828 ms, 1.2 GB |

Fluxes agree to 1.1e-13 W/m2 at every size. Per thread, rte3d is 10 to 25 percent
behind the reference, closing as the problem grows. On all cores it is 5.4 to 5.9 times
the reference's serial run. The shortwave at 4096 columns takes 2081 ms on 16 threads,
and both bands together peak at 1.2 GB.

The per-thread gap is all gas optics; transport is ahead. Timed per stage, with
fences around each inside the fused g-point loop on the rte3d side and each kernel call
on its own on the reference's, per solve on one thread:

| | rte3d, 256 | reference, 256 | rte3d, 1024 | reference, 1024 |
|---|---|---|---|---|
| gas optics and Planck source | 619 ms | 381 ms | 2474 ms | 1796 ms |
| transport | **126 ms** | 145 ms | **479 ms** | 604 ms |

rte3d's Planck fraction is computed in its optical-depth kernel and the reference's in
its Planck kernel, so only the sum of the two is comparable. The reference's transport
includes writing a flux per g-point, which rte3d accumulates instead.

`--breakdown` cannot show this. At 256 columns it reports gas optics at 770 ms and
transport at 332 ms, more together than the 743 ms of the fused solve: each stage is a
separate Python call that copies the whole `(ngpt, nlay, ncol)` spectrum in or out --
the transport call alone moves some 400 MB of sources and optical depths into new
arrays and returns two flux arrays as large -- and that is most of what it times. The
same is why it cannot be used on a GPU, below. An earlier measurement on a 15-core
Apple Silicon host, built with clang, did add up: 358 ms in all, of which gas optics 310
and transport 59, against 290 ms for the reference.

Transport used to be 555 ms of that Apple single-threaded total, two thirds of the
runtime. Each `(g-point, column)` pair ran an entire sequential vertical recurrence,
which suits a GPU but leaves the column loop with a single iteration's worth of
independent work no matter what `ivdep` asserts. Solving one g-point at a time restores
the reference's structure -- layer outside, `ncol` innermost -- and the recurrences now
vectorize over columns: a 9x improvement there.

Where it really pays is at scale, because a single g-point's working set stays in cache
instead of streaming ~80 MB arrays through memory. On the Apple host, 4096 columns on 15
threads went from 9495 ms and 11 GB to 1226 ms and 1.1 GB. On this one the
multi-threaded cost per column is flat from 256 to 1024 columns, 381 to 401 us, and
rises to 446 at 4096. There is no sign here of the launch-bound regime the Apple host
showed below 512 columns, where its 256-column run took 221 ms on 15 threads.

Thread placement matters on this hybrid CPU. At 1024 columns: 16 threads one per core
390 ms, 24 threads spread over every hardware thread 360 ms, 24 threads packed onto the
cores 541 ms, 8 threads 487 ms.

### On a GPU

RCEMIP at 65536 columns x 256 layers, single precision, on an RTX A4500:

| | longwave | shortwave |
|---|---|---|
| before | 1879 ms | 2167 ms |
| now | **1272 ms** | **1406 ms** |

Every one of those milliseconds was memory traffic, not arithmetic and not
parallelism: the kernels were already running at 400-650 GB/s, against a card that
peaks near 640. So the work was to move less --- the binary-species interpolation
recomputed rather than stored (4.0 GB, and 24 bytes per cell per g-point), the Planck
fraction and the Rayleigh scattering taken from the interpolation the optical depth was
already doing, and the fluxes added into the spectral totals where they are produced
instead of being written per g-point and read straight back. `git log` has the
measurement for each.

What is left is the column sweeps. Against `rte-rrtmgp-cpp`'s CUDA solver on its own
RCEMIP case --- the same input file, 4096 columns, plane-parallel, clear sky, both
single precision on one RTX A4500 --- that is now the whole of the gap:

| stage | rte3d | rte-rrtmgp-cpp | ratio |
|---|---|---|---|
| longwave, optical depth + Planck | 36.0 ms | 28.0 ms | 1.29 |
| longwave, transport | **71.4 ms** | **42.0 ms** | **1.70** |
| shortwave, optical depth + Rayleigh | 23.8 ms | 28.4 ms | 0.84 |
| shortwave, transport | **187 ms** | **45.2 ms** | **4.13** |
| GPU kernel time, longwave / shortwave | 108 / 211 ms | 70 / 74 ms | |
| what the driver's own timer reports | 120 / 223 ms | 73 / 74 ms | |

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

Gas optics is at parity --- 29% more expensive in the longwave, 16% cheaper in the
shortwave. The sweeps are not, and the launch geometry says why: it solves four column blocks with every g-point
resolved, so its `sw_adding` runs 229376 threads and its `lw_solver_noscat_step_2`
262144, where ours run `ncol` --- 4096. Same work, 64x the parallelism, and at 4096
columns ours reach about 15% of peak bandwidth and 8% occupancy where at 65536 they
reach 70%. The gap is a small-problem gap: 54.4 us/column at 4096 against 21.5 at 65536,
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

The shortwave transport above includes a deliberate loss at this size. The adding
sweeps add the diffuse fluxes into the totals themselves, as `d789c1c` made them do
again after `bac85f3` had moved that into a separate kernel: production runs are at
large column counts, where it is 4% faster (1460 -> 1406 ms at 65536 columns) and needs
two fewer scratch arrays, but at 4096 columns it costs 13% (195 -> 222 ms).

The rte3d column of this table was remeasured at `d789c1c`; the rte-rrtmgp-cpp column is
from the earlier session, its code unchanged since. Between the two sessions the
longwave stages of rte3d read 3 to 5% higher with no change to the longwave code, which
is the spread to allow between sessions.

## All-sky against the Fortran reference with OpenACC

rte-rrtmgp's own all-sky example on the GPU, against rte3d solving the same atmosphere.
RTX A4500, NVHPC 25.3, 72 layers, clouds in two thirds of the columns, the mean of 20
loops on each side. Run with [`bench_allsky_acc.py`](README.md#allsky_rtebench_allsky_accpy--performance-against-rte-rrtmgp-with-openacc),
the reference built by `tests/build_reference_acc.sh`: the `accel` kernels, with the
flags of rte-rrtmgp's own OpenACC CI (`.gitlab/dkrz.yml`, which are ICON's),
`-g -O2 -Mrecursive -Mallocatable=03 -Mstack_arrays -acc=gpu,verystrict -gpu=cc86`.

Single precision:

| band | ncol | Fortran OpenACC | rte3d | ratio | max abs diff (W/m2) |
|---|---|---|---|---|---|
| lw | 1024 | 6.70 ms | 21.77 ms | 0.31x | 4.3e-04 |
| lw | 2048 | 12.45 ms | 30.92 ms | 0.40x | 4.3e-04 |
| lw | 4096 | 23.85 ms | 39.51 ms | 0.60x | 4.3e-04 |
| lw | 8192 | 46.10 ms | 54.70 ms | 0.84x | 4.3e-04 |
| lw | 16384 | 92.25 ms | 83.34 ms | 1.11x | 4.3e-04 |
| sw | 1024 | 6.95 ms | 36.19 ms | 0.19x | 3.2e-01 |
| sw | 2048 | 13.60 ms | 42.34 ms | 0.32x | 3.2e-01 |
| sw | 4096 | 27.05 ms | 55.93 ms | 0.48x | 3.2e-01 |
| sw | 8192 | 53.50 ms | 75.90 ms | 0.70x | 3.2e-01 |
| sw | 16384 | 107.45 ms | 116.34 ms | 0.92x | 3.2e-01 |

Double precision:

| band | ncol | Fortran OpenACC | rte3d | ratio | max abs diff (W/m2) |
|---|---|---|---|---|---|
| lw | 1024 | 17.35 ms | 93.65 ms | 0.19x | 9.1e-13 |
| lw | 2048 | 32.15 ms | 114.86 ms | 0.28x | 9.1e-13 |
| lw | 4096 | 62.45 ms | 158.54 ms | 0.39x | 9.1e-13 |
| lw | 8192 | 125.50 ms | 282.68 ms | 0.44x | 9.1e-13 |
| sw | 1024 | 28.95 ms | 158.34 ms | 0.18x | 5.0e-08 |
| sw | 2048 | 53.60 ms | 183.90 ms | 0.29x | 5.0e-08 |
| sw | 4096 | 104.85 ms | 226.98 ms | 0.46x | 5.0e-08 |
| sw | 8192 | 208.20 ms | 403.59 ms | 0.52x | 5.0e-08 |

The ratio is Fortran time over rte3d time: below 1 rte3d is slower. The single-precision
rows are with the no-scattering solver fused into its sweeps, [below](#the-longwave-solver-fused-into-its-sweeps),
and the approximate divide in the solvers, [below that](#an-approximate-divide-in-the-solvers);
the double-precision rows are from before both, and neither changes them: the divide is
exact in double precision.

**What is timed.** The same span on both sides: cloud optics, gas optics and the solver,
with every input and output already on the device. For the reference that is the
example's own loop timer. It prints only to the millisecond, so its time per loop is
`(T(21 loops) - T(1 loop))/20`, which also drops the first loop's allocations. For
rte3d it is `solve_time`, after a warm-up. The atmosphere is the one the example
generates and writes out; rte3d reads it back, so both solve identical columns, and the
last column of each table is the largest difference between their fluxes. In double
precision that is round-off. In single precision it is the deliberate differences
listed in [`CHANGES.md`](../CHANGES.md), the `k²` floor among them.

**Where the reference stops.** It solves the whole spectrum for every column at once,
and runs out of the card's 20 GB at 16384 columns in double precision and 32768 in
single: 2.4 GB for each `(ncol, nlay, ngpt)` array. rte3d holds one g-point at a time,
which is the difference in design between the two, so a reference run beyond these
sizes would have to be done in blocks of columns.

**Where rte3d stands.** rte3d is behind at every size the reference can reach, in
both precisions, with one exception: the single-precision longwave at 16384 columns,
which it now solves 11% faster. The gap narrows with column count in single precision but
much less in double precision. Two separate costs make it up:

- *A fixed cost per solve*, largest in the shortwave. Doubling from 1024 to 2048 columns
  costs rte3d only 6 ms more in single precision and 26 ms in double, against a 36 and
  158 ms solve at 1024. Part of that is the one-g-point-at-a-time design: each of 224
  g-points launches its own kernels, too small at 1024 columns to fill the card, the
  small-problem gap the RCEMIP comparison above describes. In double precision most of
  it is one kernel: the shortwave direct-beam sweep, 490 us per g-point at 1024 columns
  against 61 in single. It runs one thread per column, 1024 threads in blocks of 128,
  so 8 of the card's 46 SMs, and each of those steps through 72 layers of
  double-precision `exp`, `sqrt` and divisions in turn.
- *The cost per additional column*, from the 8192 to 16384 step (single) and the 4096
  to 8192 step (double):

  | | rte3d | Fortran OpenACC |
  |---|---|---|
  | lw, single | 3.5 us | 5.6 us |
  | sw, single | 4.9 us | 6.6 us |
  | lw, double | 30.3 us | 15.4 us |
  | sw, double | 43.1 us | 25.2 us |

  In single precision each extra column costs rte3d less than the reference: the
  longwave already passes 1 at 16384 columns, and the shortwave's slope says it would
  just beyond, though the reference cannot run there without blocking. In double
  precision each extra column costs rte3d twice as much, so the gap does not close.
  `ncu` says why: at 8192 columns `compute_tau` keeps the FP64 pipeline 88% busy and
  DRAM 11%, and the shortwave direct-beam sweep 85% and 4%. The A4500 runs double
  precision at 1/32 of the single-precision rate, and rte3d recomputes per g-point what
  it could store -- the binary-species interpolation, the two-stream coefficients inside
  the sweep, and since the change below the longwave layer sources in both sweeps -- which
  in single precision is the right trade and in double precision is not.

### The longwave solver fused into its sweeps

Profiled at 16384 columns in single precision, the longwave solver was the whole of the
gap: 44.4 ms against the reference's 32.1. It ran four kernels per g-point: one
computing the transmission and layer sources into three `(nlay, ncol)` arrays, a
transport sweep down and one up writing radiances into two more, and one adding
those radiances into the flux totals. The first and last were at 540 GB/s, near the
card's peak, so their cost was the arrays themselves, 13.5 ms each. Now each sweep
computes the transmission and sources it needs and puts its fluxes straight into the
totals, as the shortwave adding sweeps do, and five of those arrays are gone.

Two variants were measured, two rounds each, which agreed within 2%. *Fused* is the one
in the code; *accumulate only* kept the source kernel and moved only the accumulation
into the sweeps. Longwave solve time:

| | before | fused | accumulate only |
|---|---|---|---|
| RCEMIP, 65536 columns, single, GPU | 1278 ms | **1022 ms** | 1146 ms |
| all-sky, 16384 columns, single | 100.8 ms | **89.0 ms** | 95.4 ms |
| all-sky, 4096 columns, single | 38.3 ms | 45.6 ms | 42.6 ms |
| all-sky, 1024 columns, single | 17.0 ms | 26.0 ms | 23.5 ms |
| all-sky, 8192 columns, double | 232 ms | 284 ms | 229 ms |
| all-sky, 1024 columns, double | 42.2 ms | 95.3 ms | 51.6 ms |
| RCEMIP, 1024 columns, double, CPU | 354 ms | 383 ms | 346 ms |
| RCEMIP, 1024 columns, single, CPU | 327 ms | 389 ms | 335 ms |
| les_cloudfield, plane-parallel solve | 233.1 ms | **207.8 ms** | 220.4 ms |
| les_cloudfield, ray tracer | 1230 ms | **1195 ms** | 1217 ms |

It is a trade towards large column counts in single precision, which is what
production runs at. The sweeps run one thread per column, so what they gain in memory
traffic they pay for in arithmetic and in latency they have too few columns to hide:
at 1024 columns, in double precision and on the CPU. Moving only the accumulation
gains half as much at 65536 columns and still costs 38% at 1024.

The les_cloudfield rows are the case's longwave with `scattering = false`, as the
reference's own `test.ini` has it; with the case's `scattering = true` both transports
use the two-stream solver, which this does not touch. The ray tracer gains less because
only the 66 g-points opaque within a cell go through this solver. The fluxes of both
runs, traced ones included, are bit-identical to before: the same operations in the same
order.

**A bug in the reference's GPU path.** As checked out (rte-rrtmgp main at #431),
the example's fluxes are NaN on the GPU below the top level, in both bands and both
precisions, and correct when the same binary runs on the host. Since #419,
`rrtmgp/frontend/mo_gas_optics_rrtmgp.F90` computes
`col_dry_arr = get_col_dry(...) * 1.e-4_wp`. `get_col_dry` works on the device and
copies its result out, the scaling and assignment happen on the host, and the device
copy of `col_dry_arr`, which every kernel after it reads, is never written, so all
column amounts on the device are zero. `tests/build_reference_acc.sh` builds from a copy
with the missing `!$acc update device(col_dry_arr)` added; the timings above are with that
fix, and the flux columns show it is the whole of the problem.


### An approximate divide in the solvers

`-use_fast_math` on every file, rather than on the ray tracer alone, made the
single-precision all-sky case 8-16% faster, but failed two gas-optics tests: `eta` no
longer came out exactly 1 for a flavour without its second gas, and a Rayleigh `ssa`
reached 1.0000001. With `--prec-div=true` added back the tests passed and the whole gain
was gone, so the gain was the division and nothing else. `nsys` at 16384 columns put
about 70% of it in the solvers -- the adding sweeps, the longwave no-scattering sweeps
and the shortwave two-stream -- and the rest in `compute_tau`.

`fast_div` in `include/types.h` is `__fdividef` on a GPU in single precision and `a/b`
everywhere else. It is used for the reciprocals and quotients of the solvers only; gas
optics keeps IEEE division. All-sky, single precision, GPU, mean of 20 loops:

| | IEEE division | `fast_div` in the solvers | fast math everywhere |
|---|---|---|---|
| lw, 4096 columns | 43-46 ms | 39-42 ms | 40.7 ms |
| lw, 16384 columns | 89 ms | 83-85 ms | 82.3 ms |
| sw, 4096 columns | 67 ms | 56-59 ms | 56.1 ms |
| sw, 16384 columns | 125 ms | 115-118 ms | 115.0 ms |

The ranges are rebuilds of the same source, which on this card land up to 3 ms apart.
Fluxes move by at most 4e-4 W/m2 in the shortwave and 3e-5 W/m2 in the longwave, and
the CPU and double-precision builds give bit-identical fluxes.
