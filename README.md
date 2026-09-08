# rte3d

RTE-RRTMGP in Kokkos: one frontend for CPU and GPU, row-major arrays throughout, and
the g-point as the outermost dimension.

## Quick start

```bash
git submodule update --init --recursive          # kokkos, pybind11, rrtmgp-data

mkdir build && cd build
cmake .. -DSYST=macbook                          # see config/ for other systems
cmake --build .
cd ..

export RTE3D_PYTHON_PATH=$PWD/build/main_python
pytest tests                                     # 153 tests
python cases/rfmip/run_rfmip.py --plot           # a case, end to end
```

The reference-comparison tests skip unless the Fortran oracle is built; see
[Testing](#testing).

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
plain structs above.

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

**Next:** aerosol optics, then the Monte Carlo ray tracer, then MicroHH integration.

## Cases

Three runnable end-to-end scripts live in [`cases/`](cases/README.md), which documents
every flag:

```bash
python cases/rfmip/run_rfmip.py   --plot     # clear-sky, 100 sites, vs reference
python cases/allsky/run_allsky.py --plot     # cloudy, 24 columns, vs reference
python cases/rcemip/run_rcemip.py --ncol 512 --compare-fortran   # benchmark
```

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

On performance, rte3d is within about 20% of the reference Fortran per thread, and the
remaining gap is entirely gas optics; see
[`cases/README.md`](cases/README.md#where-things-stand).

## Design notes

### Where gas names live

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

### Interpolation weights are recomputed, not stored

The reference materialises `fmajor(2,2,2,ncol,nlay,nflav)` and
`fminor(2,2,ncol,nlay,nflav)`. Both are pure functions of `ftemp`, `fpress` and `feta`,
which the same kernel already computes, so rte3d stores those instead and rebuilds the
weights where they are used (`Gas_optics_kernels::interp_weights`). That trades twelve
stored values per (flavour, layer, column) for four multiplies: at `ncol = 1e5`,
`nlay = 60`, `nflav = 10` it is roughly 2.4 GB instead of 7 GB.

rte3d also keeps the column as the fastest-varying dimension in these arrays, where the
reference puts `ncol` in the middle. This is the one place where a straight dimension
reversal would not have matched, so the gas-optics tests transpose before calling the
reference. The library itself never transposes.

### One source for both vertical orientations

`Rte_kernels::Vert<top_at_1>` in `include_kernels/rte_solver_kernels.h` carries the two
level offsets of a layer. The reference writes every loop out twice, once per
orientation; templating on it keeps the explicit loops the compiler wants while writing
the physics once.

### One g-point at a time

The solvers take `(nlay, ncol)` optical properties and return `(nlev, ncol)` fluxes: no
field they touch carries a g-point dimension, so their working set is the same whether
the k-distribution has 16 g-points or 256. The caller loops g-points and accumulates.
That is what makes the solvers fit on a GPU, and it is also why they are fast on a CPU
— the vertical recurrences become sweeps over layers with the column loop inside them,
which is the structure the reference Fortran has.

The two backends want opposite nestings for such a sweep: layer-outer with columns
vectorized on CPU, column-parallel with the layer loop inside the kernel on GPU. Both
are driven from the same caller-supplied `f(ilay, icol)` by
`parallel_for_column_sweep` in `include/types.h`, beside `parallel_for_2d`, so that
choice stays in the one file allowed to know which backend is in use.

Boundary conditions ride along at the ends of the sweeps rather than in launches of
their own: the `j == 0` test is invariant in the vectorized column loop, and parallel
regions are the scarcer resource.

### Known defects in the Fortran reference

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

## Dependencies

A C++20 compiler, and Python 3 with numpy, xarray and a NetCDF backend
(`netCDF4` or `h5netcdf`). `pytest` to run the tests, `matplotlib` only for `--plot`.

Kokkos, pybind11 and the RRTMGP coefficient data come in as git submodules; there is
deliberately **no** NetCDF, HDF5 or FFTW dependency in the C++, since all file reading
happens in Python and arrives through pybind11 as numpy arrays.

## Building

```bash
mkdir build && cd build
cmake .. -DSYST=macbook
cmake --build .
```

| switch | meaning |
|---|---|
| `-DSYST=<system>` | **required**; picks `config/<system>.cmake` — `macbook` (clang), `macbook_gcc`, `ubuntu_22lts_gcc` |
| `-DUSEGPU=1` | build for GPU; the backend (CUDA or HIP) comes from the config file |
| `-DUSESP=1` | 32-bit floats instead of 64-bit |
| `-DCMAKE_BUILD_TYPE=DEBUG` | unoptimised build |

Use a separate build directory per configuration and point `RTE3D_PYTHON_PATH` at
whichever you want; `build/` and `build_gcc/` can coexist.

## Testing

```bash
export RTE3D_PYTHON_PATH=$PWD/build/main_python
pytest tests
```

That runs everything not needing the Fortran. To include the reference comparisons —
the full suite:

```bash
./tests/build_reference.sh                       # needs gfortran; set FC to override

export RTE3D_PYTHON_PATH=$PWD/build/main_python
export RTE3D_FORTRAN_REF=$PWD/build/reference/librte_kernels.dylib
export RTE3D_FORTRAN_SHIM=$PWD/build/reference/librte3d_shim.dylib
pytest tests
```

The library extension is `.dylib` on macOS and `.so` on Linux;
`build_reference.sh` prints the two `export` lines for your platform when it finishes.

| variable | what it unlocks |
|---|---|
| `RTE3D_FORTRAN_REF` | the kernel-by-kernel comparisons, via the `bind(C)` API |
| `RTE3D_FORTRAN_SHIM` | the k-distribution reduction and cloud optics comparisons |

Tests needing either one skip when it is unset. The Fortran is a correctness oracle
only — deprecated, never linked, and not needed to build or use rte3d.

The shim exists because `ty_gas_optics_rrtmgp%load` and `ty_cloud_optics_rrtmgp` have
no `bind(C)` entry points. `tests/build_reference.sh` builds a second library from
`tests/shim/rte3d_shim.F90` that exposes them, and Python hands the same raw arrays to
both implementations. To let the shim read the reduced arrays, the build compiles a
*copy* of `mo_gas_optics_rrtmgp.F90` with the type's `private` relaxed; the reference
source is never modified.

## Benchmarking against the Fortran

Copy-paste, from the repository root:

```bash
./tests/build_reference.sh                      # once; needs gfortran

export RTE3D_PYTHON_PATH=$PWD/build/main_python
export RTE3D_FORTRAN_REF=$PWD/build/reference/librte_kernels.dylib

# like for like: one thread each. This is the number that matters.
OMP_NUM_THREADS=1 python cases/rcemip/run_rcemip.py --ncol 256 --compare-fortran --breakdown

# all cores, against serial Fortran
python cases/rcemip/run_rcemip.py --ncol 256 --compare-fortran
```

Typical result on 256 columns x 256 layers, longwave:

```
                       rte3d      reference    ratio
  1 thread            363 ms         295 ms    0.81x
    gas optics        304 ms                             <- what is left of the gap
    transport          60 ms
  15 threads          131 ms         294 ms    2.24x
```

The reference kernels are serial on the host, so the 15-thread row compares rte3d on
15 threads against Fortran on one. Fluxes agree to 1e-13 W/m2 either way.

Raise `--ncol` for a bigger problem. The solvers no longer grow with the number of
g-points, so the full 4096 columns fits; gas optics and the returned spectral fluxes
still do, which is about 14 GB there. `--band lw|sw` isolates one band. Point
`RTE3D_PYTHON_PATH` at `build_gcc/main_python` to compare compilers — the two land
within 20% of each other.
