# Design notes

Why rte3d is built the way it is: what is ported, how it validates and performs, and the
decisions behind the kernels. [`README.md`](README.md) is how to build, test and run it.

## Why the dimension order matters

Arrays are ordered `(ngpt, nlay, ncol)` and stored row-major, so the column is the
fastest-varying index. This is byte-for-byte the layout of the reference Fortran's
`(ncol, nlay, ngpt)` column-major arrays. Two consequences:

- The column stays unit-stride: coalesced on GPU, vectorizable on CPU.
- Comparing against the reference needs no transposes. A numpy array of shape
  `(ngpt, nlay, ncol)` in C order can be handed straight to the Fortran kernels.

Fixing the order this way removes the need for the separate CPU and GPU code paths
that `rte-rrtmgp-cpp` carries.

## Status

**Step 1, the RTE core — complete.**

| header | contents |
|---|---|
| `include/types.h` | precision, array aliases, parallel-for wrappers, numpy interop |
| `include/rte_sw.h` | `sw_solver_noscat`, `sw_solver_2stream` |
| `include/rte_lw.h` | `lw_solver_noscat` with multi-angle quadrature and the surface-temperature Jacobian, `lw_solver_2stream` |
| `include/optical_props.h` | delta-scaling, the increment operations, column subsetting |
| `include/fluxes.h` | broadband and by-band flux reduction |

Two things in `rte-kernels/` are deliberately not ported: `lw_transport_1rescl`, the
approximate-scattering rescaling of Tang et al. 2018 (see the note at the top of
`src/rte_lw.cpp`), and `zero_array_*`, which `Kokkos::deep_copy` already covers. The
`rte-frontend/` layer is replaced rather than ported: its class hierarchy becomes the
plain structs above, and the driver that runs a case with them is `include/solver.h`
(`namespace Solver`), described under *One g-point at a time*.

**Step 2, gas optics — complete.**

| header | contents |
|---|---|
| `include/gas_concs.h` | volume mixing ratios by name |
| `include/gas_optics.h` | the four RRTMGP kernels, the k-distribution reduction (`Gas_optics::load`), and the `gas_optics_lw` / `gas_optics_sw` frontends |
| `main_python/rte3d/kdist.py` | the only place rte3d touches NetCDF |

Loading `rrtmgp-gas-lw-g256.nc` with eight gases reduces 19 absorbers to 8, 60 lower
minor absorbers to 45, and the lower `kminor` table from 960 contributors to 720.

**Step 3, cloud and aerosol optics — in progress.** Cloud optics
(`include/cloud_optics.h`) is done, from the lookup tables in `extern/rrtmgp-data`.
Both spectral resolutions work: `-bnd` files are resolved by band and combine with gas
optics through the by-band increments, `-g###` files by g-point. Only the lookup-table
path is implemented — the reference also offers Pade approximants, but no shipped
coefficient file contains them. Aerosol optics is not started.

**Step 4, the Monte Carlo ray tracer — shortwave done.**

| header | contents |
|---|---|
| `include/raytracer.h` | the ray-tracing box, the fluxes it produces, and `trace_rays` |
| `include_kernels/raytracer_kernels.h` | the photon walk and the sampling it needs |
| `include/random.h` | the backend's generators, behind one interface |

A forward, three-dimensional, null-collision tracer, after `raytracer_sw.cu` in
rte-rrtmgp-cpp. `Solver::solve_sw_rt` drives it the way `solve_sw` drives the
two-stream solver: same gas optics, same cloud properties, one g-point at a time, only
the transport differs. Aerosols and the Mie phase function are left out — clouds
scatter as Henyey-Greenstein with the asymmetry parameter the RRTMGP tables give,
gases as Rayleigh — and the longwave and backward-camera tracers are not started.

On the RCEMIP case with clouds, 64x64 columns and a 42 degree sun, the two solvers
agree to 1.1% on the downward surface flux, 0.2% on the direct beam and 0.4% on the
column absorption -- which is the two-stream's own error plus what the third dimension
moves around, not a discrepancy. That case is 235 million photons and takes 11.9 s on
an RTX A4500, single precision, against 179 ms for the two-stream solve beside it.

**Next:** aerosol optics, the longwave ray tracer, then MicroHH integration.

## Validation

Agreement with the reference fluxes shipped in `extern/rrtmgp-data`, which RRTMGP
itself produced:

| | longwave up / down | shortwave up / down |
|---|---|---|
| RFMIP, 100 sites | 4.8e-3 / 1.0e-2 W/m2 | 6.0e-4 / 9.3e-3 W/m2 |
| all-sky, 24 columns | 9.1e-13 / 9.1e-13 W/m2 | 7.2e-11 / 7.2e-11 W/m2 |

RFMIP's own acceptance threshold is 5.8e-2 W/m2. Its reference files are float32, whose
spacing at these magnitudes is 3e-5 W/m2, so its residual is not storage precision — it
is the difference from the k-distribution the 2018 fluxes were generated with. The
all-sky reference was generated with the data that ships beside it, which is why that
one reaches round-off.

## Performance

On performance, rte3d is within about 20% of the reference Fortran per thread, and the
remaining gap is entirely gas optics. Typical result on 256 columns x 256 layers,
longwave, from `run_rcemip_rte.py --compare-fortran --breakdown`:

```
                       rte3d      reference    ratio
  1 thread            358 ms         290 ms    0.81x
    gas optics        310 ms                             <- what is left of the gap
    transport          59 ms
  15 threads          221 ms         295 ms    1.34x
```

The reference kernels are serial on the host, so the 15-thread row compares rte3d on
15 threads against Fortran on one. Fluxes agree to 1e-13 W/m2 either way.

On a GPU, RCEMIP at 65536 columns x 256 layers runs in 1314 ms longwave and 1531 ms
shortwave on an RTX A4500, single precision, and is bound by memory bandwidth
throughout. Against rte-rrtmgp-cpp's CUDA solver, gas optics is at parity and the
transport sweeps are not. The full picture, both hosts and both bands, is in
[`cases/RESULTS.md`](cases/RESULTS.md#where-things-stand).

## Where gas names live

In C++, and only in `Gas_concs` and the k-distribution loader — never in a kernel. The
reference resolves names at two moments: once at load, where the k-distribution is
*reduced* to the gases the host actually supplies and every integer index array is
rebuilt against that shorter list, and once per call, to look up each gas's mixing
ratio. rte3d keeps both in C++ so a host model with no Python can still load a
coefficient file; Python's only job is reading the file into arrays.

The reference loops minor absorbers serially, walking a per-column layer range for
each. rte3d parallelises over (g-point, layer, column) instead, which needs the inverse
mapping — for each g-point, which minor absorbers contribute. `Minor_absorbers` holds
that as a CSR list built once by `build_map`.

## The interpolation is recomputed, not stored

The reference materialises `fmajor(2,2,2,ncol,nlay,nflav)` and
`fminor(2,2,ncol,nlay,nflav)`. Both are pure functions of `ftemp`, `fpress` and `feta`,
so rte3d rebuilds them where they are used (`Gas_optics_kernels::interp_weights`)
rather than storing twelve values per (flavour, layer, column).

The g-point loop pushed the same argument one step further. `jeta`, `feta` and
`col_mix` were stored too, `(nflav, 2, nlay, ncol)` each — 4.0 GB at
`ncol = 65536`, `nlay = 256` — of which every g-point kernel read one flavour, 24 of
the 240 bytes per cell, once per g-point. `Gas_optics_kernels::eta_interp` rebuilds
that one flavour from two `col_gas` values and a `vmr_ref` lookup small enough to stay
in cache, for two divides. `Interp_state` still has the arrays, but only the
kernel-by-kernel tests, which compare them against the reference, ask for them.

What is left in `Interp_state` is `(nlay, ncol)`: seventeen bytes a cell, read by every
kernel in the g-point loop.

rte3d keeps the column as the fastest-varying dimension in these arrays, where the
reference puts `ncol` in the middle. This is the one place where a straight dimension
reversal would not have matched, so the gas-optics tests transpose before calling the
reference. The library itself never transposes.

## One kernel per interpolation, not per quantity

Three quantities come out of the same major-species interpolation, and the kernel that
does it is the most expensive one in the solve. So each rides along with it rather than
re-reading the whole interpolation state to redo the work: the Planck fraction is
`pfracin` interpolated with the weights `kmajor` just used (`compute_tau_lw`), and
Rayleigh scattering is `krayl` with the same weights and the same flavour, combined
with the absorption on the spot (`compute_tau_sw`). `Gas_optics_kernels::interp_major`
is the shared 2x2x2 sum.

The fluxes go the same way. A whole-spectrum solve never looks at a per-g-point flux,
so the solvers do not write one: `Flux_sink` says where a flux is to go, and their last
kernel adds it straight into the running spectral totals. The only g-point flux left is
the shortwave's direct beam, which the sweep that computes it also reads back.

## One source for both vertical orientations

`Rte_kernels::Vert<top_at_1>` in `include_kernels/rte_solver_kernels.h` carries the two
level offsets of a layer. The reference writes every loop out twice, once per
orientation; templating on it keeps the explicit loops the compiler wants while writing
the physics once.

## One g-point at a time

`Solver::solve_lw` and `solve_sw` are the way to run a case. They loop g-points,
and for each one increment the clouds for that g-point's band, run transport, and
accumulate the flux. Nothing in that pipeline carries the spectrum, so the working set
is the same whether the k-distribution has 16 g-points or 256: RCEMIP's full 4096
columns needs 1.1 GB rather than the 28 GB the spectrally resolved version would have.

The exception is gas optics, which runs a band at a time — at most
`Gas_optics::max_gpt_block`, 16 g-points — into `(16, nlay, ncol)` block arrays that
the g-point loop then reads a slice at a time. Everything in the optical depth except
the table lookups is the same for every g-point of a band: the interpolation state,
the binary-species interpolation and its weights, the minor absorbers' scaling. Done
once per band instead of once per g-point, that halves the cost of gas optics on the
GPU and on one CPU thread alike, for a few hundred MB of block arrays, bounded by the
band width and not by the spectrum.

The loop body is public as `gas_optics_lw_block` / `gas_optics_sw_block` and
`solve_lw_gpt` / `solve_sw_gpt`, for callers that want a single g-point — the Monte
Carlo ray tracer, and the tests. `Solve_state` holds what does not depend on the
g-point (the column gas amounts, the table interpolation), the block arrays, and the
one working set every iteration reuses.

This driver is its own module, `namespace Solver` in `include/solver.h` and
`src/solver.cpp`. It is neither gas optics nor transport but the thing that runs both,
so it sits above `Gas_optics`, which hands it a band's optical properties, and
above `Rte_lw` / `Rte_sw`, whose kernels it drives.

Doing it this way is also why the solvers are fast on a CPU: the vertical recurrences
become sweeps over layers with the column loop inside them, which is the structure the
reference Fortran has.

The two backends want opposite nestings for such a sweep: layer-outer with columns
vectorized on CPU, column-parallel with the layer loop inside the kernel on GPU. Both
are driven from the same caller-supplied `f(ilay, icol)` by
`parallel_for_column_sweep` in `include/types.h`, beside `parallel_for_2d`, so that
choice stays in the one file allowed to know which backend is in use.

Boundary conditions ride along at the ends of the sweeps rather than in launches of
their own: the `j == 0` test is invariant in the vectorized column loop, and parallel
regions are the scarcer resource.

Every one of these recurrences reads back, at each step, a value it wrote at the step
before. `parallel_for_column_sweep_carry` hands those to the kernel as
`f(j, icol, TF carried[2])` — registers on GPU, an `(2, ncol)` scratch on CPU — so that
no sweep reads an array it has written. What that buys is not the read, which came out
of cache anyway, but the array: the shortwave's downward flux needs none, and goes
straight into the spectral totals.

`Rte_kernels::adding` is `static`, and has to stay that way. nvcc identifies an extended
device lambda's closure type by its enclosing function, so the sweeps in that template
get the same mangled type in `rte_lw.cpp` and `rte_sw.cpp`, which both instantiate it;
the linker merges them and the launch calls a closure that was never registered. It
segfaults on the host, with nothing for `compute-sanitizer` to report.

## Where the backend RNG lives

Convention 3 says the backend appears in `include/types.h` and nowhere else. The ray
tracer needs a second such file, `include/random.h`, because the generators come from
the vendors and their device APIs are not interchangeable: cuRAND on CUDA, rocRAND on
HIP, and a xorshift on the host. Above that header there are two types, `Rand::Rng` and
`Rand::Qrng_2d`, and the photon walk is written once.

The quasi-random one picks the pixel a photon starts in. Drawing those from a
low-discrepancy sequence rather than a pseudo-random one spreads the photons evenly
over the domain and is worth roughly a factor of two in photon count for the same
noise. There is no host counterpart, so the CPU build draws them pseudo-randomly and is
noisier at the same photon count; it is there to run the test suite, whose tolerances
are set by the photon count anyway.

The generators are `__device__` functions where `KOKKOS_INLINE_FUNCTION` is
`__host__ __device__`, and nvcc will not let the second call the first. So the photon
walk carries `RTE3D_DEVICE_FUNCTION` instead, declared in the same header. It costs
nothing -- none of it is ever called on the host -- but it does mean anything that
touches a generator has to be declared that way all the way up.

## Null collisions, not a march through cells

A photon's free path in an inhomogeneous medium is an integral equation. The tracer
avoids it by pretending every cell in a block of a coarse grid has that block's largest
extinction, which makes the medium homogeneous and the free path a single logarithm;
the excess is undone by collisions that do nothing. `create_knull_grid` builds those
maxima, `ngrid_x/y/z` in the case file sets how coarse the blocks are. Coarser blocks
mean more null collisions, finer ones more block faces to cross.

Absorption is not sampled either. Each collision takes its share out of the photon's
weight and scores it, which is the variance reduction of Iwabuchi (2006); Russian
roulette below a weight of a half keeps the walk finite. That is why the energy budget
closes exactly under conservative scattering and only in the mean otherwise.

## The one place fast math is allowed

`src/CMakeLists.txt` compiles `raytracer.cpp`, and only that file, with
`-use_fast_math` (`-ffast-math` under HIP). Everything else keeps IEEE semantics,
because gas optics and the transport solvers are compared against the Fortran
reference at 1e-13 and would not survive it.

The tracer is different in kind: it samples a Monte Carlo distribution, where two ulp
on a sine is a fraction of the noise the photon count already leaves behind. Without
the flag, `Kokkos::sin`, `cos` and `log` lower to the software implementations,
argument-reduction slow paths and all -- twelve double-precision instructions in a
kernel that never needs one, on a card that runs them at a sixty-fourth of the single
precision rate. With it they become the single hardware instructions rte-rrtmgp-cpp's
kernel gets, and the kernel halves: 3816 SASS instructions to 1992.

Measured on RCEMIP at 65536 columns, single precision: the tracer goes from 10.61 to
10.24 s, and the fluxes move by 0.01% per column against a Monte Carlo noise floor
far above that. The domain-mean fluxes are unchanged to five decimals.

Everywhere else the fix is to keep the transcendental off the device. Gas optics had
one, Helmert's latitude correction to gravity in `compute_col_dry`, and it is a
function of the column rather than the cell, so it is now built once per column on the
host. With that gone, a single-precision GPU build contains **no double-precision
instruction at all** -- 335 kernels, 54384 instructions, verified with `cuobjdump
-sass`. Worth re-checking after adding a kernel that calls a transcendental: the
slow path of `sinf`, `cosf` and friends is double precision, and on a consumer card it
runs at a sixty-fourth of the single-precision rate.

## Known defects in the Fortran reference

Each is reproduced or worked around deliberately, and pinned by a test.

- **`lw_solver_2stream` ignores the g-point index of `lev_source`**
  (`rte-kernels/mo_rte_solver_kernels.F90:422`): the call to `lw_source_2str` passes
  `lev_source` rather than `lev_source(:,:,igpt)`, so Fortran sequence association
  hands every g-point the first g-point's slice. `lw_solver_noscat_oneangle:189` does
  it correctly. Present since the original 2018 import. rte3d does the correct thing;
  see `test_2stream_reference_bug_gpt_indexing`.
- **`LW_diff_sec = 1.66` is a single-precision literal** promoted to double, so its
  value is 1.65999996662139893. rte3d matches it bit-for-bit rather than "fixing" it.
- **`rte_kernels.h` mis-documents `flux_upJac`** as `(ncol,nlay+1,ngpt)`; the Fortran
  declares it `(ncol,nlay+1)`, since only broadband Jacobians are provided.

## Why the test shim exists

The shim exists because `ty_gas_optics_rrtmgp%load` and `ty_cloud_optics_rrtmgp` have
no `bind(C)` entry points. `tests/build_reference.sh` builds a second library from
`tests/shim/rte3d_shim.F90` that exposes them, and Python hands the same raw arrays to
both implementations. To let the shim read the reduced arrays, the build compiles a
*copy* of `mo_gas_optics_rrtmgp.F90` with the type's `private` relaxed; the reference
source is never modified.
