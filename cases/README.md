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

**Memory.** The solvers work one g-point at a time, so their scratch is `(nlay, ncol)`
and costs nothing at spectral resolution. What is left is gas optics and the fluxes,
which still carry a g-point dimension: about 14 GB at the full `--ncol 4096` in the
longwave, roughly a third of it the returned `(ngpt, nlev, ncol)` fluxes. That goes
when gas optics moves into the same per-g-point loop.

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
| 1 thread, total | 363 ms | 434 ms | **295 ms** |
| &nbsp;&nbsp;gas optics | 304 ms | 367 ms | |
| &nbsp;&nbsp;transport | 60 ms | 67 ms | |
| 15 threads, total | 131 ms | | |

Transport used to be 555 ms of that single-threaded total, two thirds of the runtime.
Each `(g-point, column)` pair ran an entire sequential vertical recurrence, which suits
a GPU but leaves the column loop with a single iteration's worth of independent work no
matter what `ivdep` asserts. Solving one g-point at a time restores the reference's
structure -- layer outside, `ncol` innermost -- and the recurrences now vectorize over
columns: a 9x improvement, and gas optics is what remains of the gap.

Two things this has not fixed.

**Threaded runs at small column counts are launch-bound.** Each g-point issues four or
five parallel regions of only `ncol` work each, so at `--ncol 256` on 15 threads
transport is slower than it is on one thread. It pays off from a few thousand columns
up; the structural fix is to parallelize over g-points on the host, which belongs with
the same change that moves the g-point loop into the frontend.

**Large column counts are cache-bound.** At `--ncol 4096` a single g-point's working set
is already ~80 MB, so each of the sweeps re-reads it from memory. Blocking the whole
solver over columns, rather than each sweep separately, would keep it resident; it has
not been tried.

Both builds agree with the reference to 1e-13 W/m2, so none of this is bought with
accuracy.
