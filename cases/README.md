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

Longwave, 256 columns x 256 layers, double precision, on a 15-core Apple Silicon
machine. The reference kernels are serial on the host: their OpenMP and OpenACC
directives target accelerators, not host threads.

| | clang | gcc-16 | reference (gfortran) |
|---|---|---|---|
| 1 thread, total | 863 ms | 922 ms | **303 ms** |
| &nbsp;&nbsp;gas optics | 306 ms | 366 ms | |
| &nbsp;&nbsp;transport | 555 ms | 557 ms | |
| 15 threads, total | 117 ms | 126 ms | |

Two things to read off this.

**Per thread, rte3d is about 3x slower than the reference**, and roughly two thirds of
the time is transport. Threading recovers 7.4x of the 15 cores, about 49% efficiency,
which puts the 15-thread build ~2.5x ahead of serial Fortran.

**The compiler is not the reason.** gcc accepts `#pragma GCC ivdep` where clang reports
through `-Wpass-failed` that it cannot vectorise, but the two produce transport times
within 0.4% of each other. There is genuinely nothing to vectorise: each `(g-point,
column)` pair runs an entire sequential vertical recurrence, which suits a GPU and
leaves the column loop with a single iteration's worth of independent work no matter
what the pragma asserts.

The fix is to restructure rather than to re-flag: split the per-layer algebra into a
`parallel_for_3d` over `(g-point, layer, column)`, which is most of the arithmetic and
is fully parallel, and keep only the recurrences in the per-column kernel. That is not
done yet.

Both builds agree with the reference to 1e-13 W/m2, so none of this is bought with
accuracy.

Memory is the other limit. The two-stream solver keeps about seven
`(ngpt, nlay, ncol)` scratch arrays, so RCEMIP's full 4096 columns need roughly 15 GB
in the longwave. G-point blocking would fix that, and is the same piece of work.
