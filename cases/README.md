# Cases

Runnable end-to-end scripts, in the spirit of `rte-rrtmgp-cpp`'s `rfmip/`, `allsky/`
and `rcemip/` directories. Two of them validate against reference fluxes; the third is
a performance benchmark.

## Before you start

Every script needs `RTE3D_PYTHON_PATH` pointing at a build directory:

```bash
export RTE3D_PYTHON_PATH=$PWD/build/main_python      # or build_gcc/main_python
```

In full:

| variable | needed by | how to get it |
|---|---|---|
| `RTE3D_PYTHON_PATH` | all three, always | the directory holding `rte3d_python*.so` |
| `RTE3D_FORTRAN_REF` | `run_rcemip.py --compare-fortran` only | `./tests/build_reference.sh` |

Plotting needs `matplotlib`. It is not a dependency of rte3d: without it the scripts
still run and still write their NetCDF output, and `--plot` reports what to install.

---

## `rfmip/run_rfmip.py` — clear-sky validation

100 sites, 60 layers, no clouds. Runs longwave and shortwave, writes the broadband
fluxes, and prints the largest difference from the reference for each of the four
quantities `rlu`, `rld`, `rsu`, `rsd`.

```bash
python cases/rfmip/run_rfmip.py --plot
```

| flag | default | meaning |
|---|---|---|
| `--expt N` | `0` | RFMIP experiment index, 0–17. 0 is present-day, 1 pre-industrial, and the rest vary CO2 and the other forcing agents. |
| `--plot` | off | also write `<output>.png` |
| `-o`, `--output` | `rfmip_fluxes.nc` | where the fluxes go |

Exits non-zero if any quantity exceeds RFMIP's own acceptance threshold of
5.8e-2 W/m2, the one `compare-to-reference.py` uses in the reference repository.

---

## `allsky/run_allsky.py` — cloudy validation

24 columns, 72 layers, clouds in 16 of them. Exercises cloud optics and the by-band
increments, and is stored surface-first where RFMIP is top-first, so between them the
two cases cover both vertical orientations.

```bash
python cases/allsky/run_allsky.py --plot
```

| flag | default | meaning |
|---|---|---|
| `--plot` | off | also write `<output>.png`; the third panel shows where the clouds sit |
| `-o`, `--output` | `allsky_fluxes.nc` | where the fluxes go |

Its reference fluxes were generated with the coefficient data that ships beside them,
where RFMIP's date from 2018, so this one agrees to round-off and is checked against
1e-10 relative rather than RFMIP's threshold.

---

## `rcemip/run_rcemip.py` — performance

A single radiative-convective-equilibrium profile of 256 layers, replicated over a
64x64 domain. All 4096 columns are identical, so `--ncol` tiles the profile to any
count you want and the case scales cleanly.

```bash
python cases/rcemip/run_rcemip.py --ncol 512 --breakdown
python cases/rcemip/run_rcemip.py --ncol 512 --compare-fortran
OMP_NUM_THREADS=1 python cases/rcemip/run_rcemip.py --ncol 256 --compare-fortran
```

| flag | default | meaning |
|---|---|---|
| `--ncol N` | `4096` | columns to solve. The default is the whole domain. |
| `--band lw\|sw\|both` | `both` | which band to run |
| `--repeats N` | `3` | timed repetitions after one warm-up; the best is reported |
| `--breakdown` | off | additionally time gas optics and transport separately |
| `--compare-fortran` | off | also time the reference Fortran kernels and report the ratio. Needs `RTE3D_FORTRAN_REF`. |
| `--input PATH` | the copy in `rte-rrtmgp-cpp` | where `rcemip_input.nc` lives |

**Memory.** Nothing in the solve carries a g-point dimension, so the full `--ncol 4096`
needs about 1.1 GB for both bands. `--breakdown` is the exception: timing gas optics
and transport separately means holding the whole spectrum between them, which is
roughly 11 GB at 4096 columns. Use it at 1024 or below.

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
