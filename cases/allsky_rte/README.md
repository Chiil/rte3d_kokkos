# `allsky_rte/` — cloudy validation against rte-rrtmgp

Run from the repository root, with `RTE3D_PYTHON_PATH` pointing at a build; see
[`../README.md`](../README.md#before-you-start).

Two scripts: [`run_allsky_rte.py`](#run_allsky_rtepy--validation) checks the fluxes,
[`bench_allsky_acc.py`](#bench_allsky_accpy--performance-against-rte-rrtmgp-with-openacc)
times them against the reference on the GPU.

## `run_allsky_rte.py` — validation

24 columns, 72 layers, clouds in 16 of them. Exercises cloud optics and the by-band
increments, and is stored surface-first where RFMIP is top-first, so between them the
two cases cover both vertical orientations.

```bash
python cases/allsky_rte/run_allsky_rte.py --plot
```

| flag | default | meaning |
|---|---|---|
| `--plot` | off | also write `<output>.png`; the third panel shows where the clouds sit |
| `-o`, `--output` | `allsky_fluxes.nc` | where the fluxes go |

Its reference fluxes were generated with the coefficient data that ships beside them,
where RFMIP's date from 2018, so this one agrees to round-off and is checked against
1e-10 relative rather than RFMIP's threshold.

---

## `bench_allsky_acc.py` — performance against rte-rrtmgp with OpenACC

The reference's own all-sky example, built for the GPU with NVHPC, timed against rte3d
on the atmosphere it generates. Build the reference once per precision, then run the
script with the rte3d build of the same precision:

```bash
./tests/build_reference_acc.sh sp          # -> build_acc_sp/; dp -> build_acc_dp/
RTE3D_PYTHON_PATH=$PWD/build_gpu_sp/main_python \
    python cases/allsky_rte/bench_allsky_acc.py --acc-build build_acc_sp
```

| flag | default | meaning |
|---|---|---|
| `--acc-build DIR` | required | the build directory of `build_reference_acc.sh` |
| `--ncol N,N,...` | `1024,...,32768` | column counts; sizes the reference cannot fit on the card are reported as failed |
| `--nlay N` | `72` | layers |
| `--nloops N` | `20` | timed loops on each side |
| `--bands` | `lw,sw` | which bands |
| `--workdir DIR` | a temporary directory | where the example writes its NetCDF |

The build needs `nvfortran` and a NetCDF-Fortran built with it (`NETCDF_DIR`, by default
`~/devel/netcdf-nvidia`), and patches a bug in the reference's GPU path in a copy of its
source. What is timed, the results, and the bug are in [`RESULTS.md`](../RESULTS.md).
