# `rfmip_rte/` — clear-sky validation against rte-rrtmgp

100 sites, 60 layers, no clouds. Runs longwave and shortwave, writes the broadband
fluxes, and prints the largest difference from the reference for each of the four
quantities `rlu`, `rld`, `rsu`, `rsd`.

## How to run

Every command runs from the repository root, in a shell where the environment is
loaded. `RTE3D_PYTHON_PATH` tells the scripts which rte3d build to import; without it
they stop at the import.

```bash
cd /path/to/rte3d                                        # the repository root
export RTE3D_PYTHON_PATH=$PWD/build_cpu_dp/main_python
```

Then run it. The input and reference fluxes are read from `extern/rrtmgp-data/`; the
output goes to the current directory unless `-o` says otherwise, so point it here:

```bash
python cases/rfmip_rte/run_rfmip_rte.py --plot -o cases/rfmip_rte/rfmip_fluxes.nc
```

| flag | default | meaning |
|---|---|---|
| `--expt N` | `0` | RFMIP experiment index, 0–17. 0 is present-day, 1 pre-industrial, and the rest vary CO2 and the other forcing agents. |
| `--plot` | off | also write `<output>.png` |
| `-o`, `--output` | `rfmip_fluxes.nc` | where the fluxes go |

Exits non-zero if any quantity exceeds RFMIP's own acceptance threshold of
5.8e-2 W/m2, the one `compare-to-reference.py` uses in the reference repository.
