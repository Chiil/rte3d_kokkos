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
pytest tests                                     # 180 tests
python cases/rfmip/run_rfmip.py --plot           # a case, end to end
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

| variable | what it unlocks |
|---|---|
| `RTE3D_FORTRAN_REF` | the kernel-by-kernel comparisons, via the `bind(C)` API |
| `RTE3D_FORTRAN_SHIM` | the k-distribution reduction and cloud optics comparisons |

Tests needing either one skip when it is unset. The Fortran is a correctness oracle
only — deprecated, never linked, and not needed to build or use rte3d.

The shim exists because two of the reference's entry points are not `bind(C)`; see
[the note in `DESIGN.md`](DESIGN.md#why-the-test-shim-exists).

## Cases

Four runnable end-to-end scripts live in [`cases/`](cases/README.md), which documents
every flag:

```bash
python cases/rfmip/run_rfmip.py   --plot     # clear-sky, 100 sites, vs reference
python cases/allsky/run_allsky.py --plot     # cloudy, 24 columns, vs reference
python cases/rcemip/run_rcemip.py --ncol 512 --compare-fortran   # benchmark
python cases/user/run_case.py mycase         # your own case, from NetCDF

python cases/fetch_data.py les_cloudfield    # the LES cloud field, 26 MB from Zenodo
python cases/user/run_case.py cases/les_cloudfield/les_cloudfield
```

The last is the general one: it reads a case you supply as `mycase_input.nc`, with the
switches in `mycase.toml`, and writes `mycase_output.nc`. The file layout is the one
rte-rrtmgp-cpp's test executable reads, so cases written for that run here unchanged;
`cases/user/make_input.py` writes an example one, an analytic RCEMIP sounding, so there
is something to run before you have written your own.

For how they validate and how fast they run, see
[`DESIGN.md`](DESIGN.md#validation) and [`cases/RESULTS.md`](cases/RESULTS.md).

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

Raise `--ncol` for a bigger problem: nothing in the solve grows with the number of
g-points, so the full 4096 columns runs in about 1.1 GB and is where this build is at
its best — 1226 ms against 9495 ms for the same case before the restructuring. `--band
lw|sw` isolates one band. Point `RTE3D_PYTHON_PATH` at `build_gcc/main_python` to
compare compilers — the two land within 20% of each other.

The numbers this produces, and what is left of the gap, are in
[`DESIGN.md`](DESIGN.md#performance).
