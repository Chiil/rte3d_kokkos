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
| `--ncol N` | `4096` | columns to solve. Memory grows linearly — see the warning below. |
| `--band lw\|sw\|both` | `both` | which band to run |
| `--repeats N` | `3` | timed repetitions after one warm-up; the best is reported |
| `--breakdown` | off | additionally time gas optics and transport separately |
| `--compare-fortran` | off | also time the reference Fortran kernels and report the ratio. Needs `RTE3D_FORTRAN_REF`. |
| `--input PATH` | the copy in `rte-rrtmgp-cpp` | where `rcemip_input.nc` lives |

**Memory.** The two-stream solver keeps about seven `(ngpt, nlay, ncol)` scratch
arrays, so the longwave needs roughly 3.7 GB at `--ncol 1024` and about 15 GB at the
full 4096. Start small and work up.

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
is fully parallel, and keep only the recurrences in the per-column kernel. The same
change fixes the memory ceiling above. Neither is done yet.

Both builds agree with the reference to 1e-13 W/m2, so none of this is bought with
accuracy.
