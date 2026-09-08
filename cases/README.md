# Cases

Runnable end-to-end scripts, in the spirit of `rte-rrtmgp-cpp`'s `rfmip/`, `allsky/`
and `rcemip/` directories. Each needs `RTE3D_PYTHON_PATH` pointing at a build:

```bash
export RTE3D_PYTHON_PATH=$PWD/build/main_python
```

## Validation cases

```bash
python cases/rfmip/run_rfmip.py   [--expt 0] [--plot] [-o rfmip_fluxes.nc]
python cases/allsky/run_allsky.py            [--plot] [-o allsky_fluxes.nc]
```

Both write broadband fluxes to NetCDF and report the largest difference from the
reference. RFMIP is checked against its own 5.8e-2 W/m2 acceptance threshold; all-sky
against 1e-9, since its reference was generated with the coefficient data that ships
beside it.

`--plot` needs matplotlib, which is not a dependency of rte3d. Without it the scripts
still run and still write the fluxes.

## Performance

```bash
python cases/rcemip/run_rcemip.py [--ncol 4096] [--band lw|sw|both] [--breakdown]
python cases/rcemip/run_rcemip.py --ncol 512 --compare-fortran
```

RCEMIP is a single radiative-convective-equilibrium profile of 256 layers, replicated
over a 64x64 domain, so `--ncol` tiles it to any column count.

`--compare-fortran` times the same sequence against the reference kernels through
ctypes, and needs `RTE3D_FORTRAN_REF` (build it with `tests/build_reference.sh`). Both
sides do the same work: column gas amounts, interpolation, absorption optical depth,
Planck or Rayleigh, transport. The column gas amounts come from rte3d either way, since
`get_col_dry` has no `bind(C)` entry point; that is a small constant on both sides.

### Where things stand

Measured on 256 columns x 256 layers, longwave, double precision:

| | rte3d, 1 thread | reference, 1 thread |
|---|---|---|
| total | 863 ms | 329 ms |
| gas optics | 306 ms | |
| transport | 555 ms | |

**rte3d is about 2.5x slower per thread than the reference**, and roughly two thirds of
that time is in transport. The reference kernels are serial on the host -- their OpenMP
and OpenACC directives target accelerators -- so with 15 threads rte3d comes out ~2.7x
ahead of serial Fortran, which is only about 17% parallel efficiency.

The cause is known and was predicted when the solvers were written. Each `(g-point,
column)` pair runs a whole sequential vertical recurrence, which suits a GPU but leaves
the compiler nothing to vectorise across columns; clang says so directly through
`-Wpass-failed`, which `include/types.h` deliberately leaves enabled. Splitting the
per-layer algebra into a `parallel_for_3d` and keeping only the recurrences per column
is the fix, and it is not done yet.

Memory is the other limit. The two-stream solver keeps about seven
`(ngpt, nlay, ncol)` scratch arrays, so RCEMIP's full 4096 columns need roughly 15 GB
in the longwave. G-point blocking would fix that, and is the same piece of work.
