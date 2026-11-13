# `rcemip_rte/` — performance against the references

A single radiative-convective-equilibrium profile of 256 layers, replicated over a
64x64 domain. All 4096 columns are identical, so `--ncol` tiles the profile to any
count you want and the case scales cleanly.

## How to run

Every command runs from the repository root, in a shell where the environment is
loaded. `RTE3D_PYTHON_PATH` tells the scripts which rte3d build to import; without it
they stop at the import.

```bash
cd /path/to/rte3d                                        # the repository root
export RTE3D_PYTHON_PATH=$PWD/build_cpu_dp/main_python
export RTE3D_FORTRAN_REF=$PWD/build_ref_dp/librte_kernels.so   # only for --compare-fortran
```

`RTE3D_FORTRAN_REF` is the Fortran reference, built once with
`./tests/build_reference.sh`; it must be the precision of the rte3d build
(`build_ref_sp` with an `_sp` build). On macOS the library ends in `.dylib`. The
input is read from `rte-rrtmgp-cpp/`. Then:

```bash
python cases/rcemip_rte/run_rcemip_rte.py --ncol 512 --breakdown
python cases/rcemip_rte/run_rcemip_rte.py --ncol 512 --compare-fortran
OMP_NUM_THREADS=1 python cases/rcemip_rte/run_rcemip_rte.py --ncol 256 --compare-fortran
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
needs about 1.2 GB for both bands. `--breakdown` is the exception: timing gas optics
and transport separately means holding the whole spectrum between them, which is
roughly 11 GB at 4096 columns. Use it at 1024 or below.

For what `--compare-fortran` actually times, where rte3d stands against the reference
on a host, and where it stands on a GPU, see [`RESULTS.md`](../RESULTS.md).
