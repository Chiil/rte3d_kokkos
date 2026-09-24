# rte3d

RTE-RRTMGP in Kokkos: one frontend for CPU and GPU, row-major arrays throughout, and
the g-point as the outermost dimension.

This file is how to build, test and run it. [`DESIGN.md`](DESIGN.md) is why it is built
this way: what is ported so far, how it validates and performs, and the design notes.

## Quick start

```bash
git submodule update --init --recursive          # kokkos, pybind11, rrtmgp-data

mkdir build && cd build
cmake .. -DSYST=macbook                          # see config/ for other systems
cmake --build .
cd ..

export RTE3D_PYTHON_PATH=$PWD/build/main_python
pytest tests                                     # 216 tests
python cases/rfmip_rte/run_rfmip_rte.py --plot           # a case, end to end
```

The reference-comparison tests skip unless the Fortran oracle is built; see
[Testing](#testing).

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
| `-DSYST=<system>` | **required**; picks `config/<system>.cmake` — `macbook` (clang), `macbook_gcc`, `ubuntu_22lts_gcc` (CUDA), `lumi` (HIP) |
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

The reference is **rte-rrtmgp v1.9**: point the `rte-rrtmgp` symlink at a checkout of
that tag (`git worktree add ../rte-rrtmgp-v1.9 v1.9` in your clone keeps it apart
from a working branch). Older versions build, but before v1.8 the longwave
quadrature weights scaled by 2π rather than π, so the no-scattering fluxes come out
at twice rte3d's.

Each precision is tested against a reference built in the same precision; the
fixtures check this and fail on a mismatch. To run the whole suite in both, with a
single-precision build in `build_cpu_sp/` (`-DUSESP=1`):

```bash
./tests/build_reference.sh dp                    # -> build/reference
./tests/build_reference.sh sp                    # -> build/reference_sp
./tests/run_all.sh                               # pytest, double then single
```

The reference builds with `-O2 -march=native`, as rte3d's configs do, so both contract
to fused multiply-adds alike; set `FFLAGS` to match another configuration.
`run_all.sh` takes `RTE3D_BUILD_DP` and `RTE3D_BUILD_SP` for other build directories.

| variable | what it unlocks |
|---|---|
| `RTE3D_FORTRAN_REF` | the kernel-by-kernel comparisons, via the `bind(C)` API |
| `RTE3D_FORTRAN_SHIM` | the k-distribution reduction and cloud optics comparisons |

Tests needing either one skip when it is unset. The Fortran is a correctness oracle
only — deprecated, never linked, and not needed to build or use rte3d.

The shim exists because two of the reference's entry points are not `bind(C)`; see
[the note in `DESIGN.md`](DESIGN.md#why-the-test-shim-exists).

One more group skips on data rather than on the Fortran: `test_all_solvers.py` runs
every solver — both plane-parallel ones and the ray tracer in one and in three
dimensions, in both bands — over the two cases in `cases/`, and the half of it that
uses the LES cloud field needs that field fetched first:

```bash
python cases/fetch_data.py les_cloudfield
```

## Cases

[`cases/`](cases/README.md) holds the runnable end-to-end cases and documents every
flag. `run_case.py` is the general way in and the shape every new case takes: it reads a
case you supply as `mycase_input.nc`, with the switches in `mycase.toml`, and writes
`mycase_output.nc`.

```bash
python cases/run_case.py mycase                        # your own case, from NetCDF

python cases/make_input.py cases/rcemip/rcemip --clouds --no-settings
python cases/run_case.py cases/rcemip/rcemip           # RCEMIP, 64x64, generated

python cases/fetch_data.py les_cloudfield              # the LES cloud field, 26 MB from Zenodo
python cases/run_case.py cases/les_cloudfield/les_cloudfield
```

The file layout is the one rte-rrtmgp-cpp's test executable reads, so cases written for
that run here unchanged; `cases/make_input.py` writes an example one, an analytic RCEMIP
sounding, so there is something to run before you have written your own.

The three `_rte` cases are the exception: each is a fixed comparison against a reference
implementation, with a script of its own that reads that case's layout and checks its
reference numbers.

```bash
python cases/rfmip_rte/run_rfmip_rte.py   --plot     # clear-sky, 100 sites, vs reference
python cases/allsky_rte/run_allsky_rte.py --plot     # cloudy, 24 columns, vs reference
python cases/rcemip_rte/run_rcemip_rte.py --ncol 512 --compare-fortran   # benchmark
```

For how they validate and how fast they run, see
[`DESIGN.md`](DESIGN.md#validation) and [`cases/RESULTS.md`](cases/RESULTS.md).

## Benchmarking against the Fortran

Copy-paste, from the repository root:

```bash
./tests/build_reference.sh                      # once; needs gfortran

export RTE3D_PYTHON_PATH=$PWD/build/main_python
export RTE3D_FORTRAN_REF=$PWD/build/reference/librte_kernels.dylib

# like for like: one thread each. This is the number that matters.
OMP_NUM_THREADS=1 python cases/rcemip_rte/run_rcemip_rte.py --ncol 256 --compare-fortran --breakdown

# all cores, against serial Fortran
python cases/rcemip_rte/run_rcemip_rte.py --ncol 256 --compare-fortran
```

Raise `--ncol` for a bigger problem: nothing in the solve grows with the number of
g-points, so the full 4096 columns runs in about 1.1 GB and is where this build is at
its best — 1226 ms against 9495 ms for the same case before the restructuring. `--band
lw|sw` isolates one band. Point `RTE3D_PYTHON_PATH` at `build_gcc/main_python` to
compare compilers — the two land within 20% of each other.

The numbers this produces, and what is left of the gap, are in
[`DESIGN.md`](DESIGN.md#performance).
