# `les_cloudfield/` — a three-dimensional cloud field

The case the ray tracer was validated on: the example LES field of `rte-rrtmgp-cpp`, a
2560 x 2560 x 4000 m RICO cumulus field at 20 m resolution. 128 x 128 columns, 200
ray-tracing cells in the vertical and 332 layers. Both tracers lump the 132 above the
box into one cell on top, as below.
Cloud is liquid-only and thin: 1.3% of cells, 19% of columns.

## How to run

Every command runs from the repository root, in a shell where the environment is
loaded. `RTE3D_PYTHON_PATH` tells the scripts which rte3d build to import; without it
they stop at the import.

```bash
cd /path/to/rte3d                                        # the repository root
export RTE3D_PYTHON_PATH=$PWD/build_gpu_sp/main_python
```

`build_gpu_sp` is the build the ray tracer is benchmarked with; `build_cpu_sp`, or any
other build, runs it too.

Then fetch the input once (26 MB, checksum-verified; does nothing if it is already
here), solve it, and optionally plot:

```bash
python cases/fetch_data.py les_cloudfield                       # -> les_cloudfield_input.nc
python cases/run_case.py cases/les_cloudfield/les_cloudfield    # -> les_cloudfield_output.nc
python cases/les_cloudfield/plot_solvers.py                     # optional, needs matplotlib
```

The settings are in [`les_cloudfield.toml`](les_cloudfield.toml). It is run by the
general runner, [`run_case.py`](../README.md#run_casepy--the-general-runner), whose
section documents every setting and the output variables.

`plot_solvers.py` writes three figures beside the case: `les_solvers_profiles.png`,
the domain-mean heating and cooling the two transports give; `les_solvers_sfc_down.png`,
the downward flux at the surface in both bands, each solver's map and their difference;
and `les_solvers_tod_up.png`, the upward flux leaving the top of the tracer's domain,
laid out the same way.

The input is the same `test_input.nc` that `rte-rrtmgp-cpp`'s `test_rte_rrtmgp_rt`
reads, published at [10.5281/zenodo.18757088](https://doi.org/10.5281/zenodo.18757088)
(CC BY 4.0). Fetching it writes `les_cloudfield_input.nc` beside the settings, which is
the name `run_case.py` derives from the case prefix; the fluxes go to
`les_cloudfield_output.nc` in the input's own `(z, y, x)` layout.

Both bands are run through both transports: traced, and plane-parallel beside it. The
g-point counts are the reference case's -- 112 shortwave and 128 longwave -- so that a
run here and a run of `test_rte_rrtmgp_rt` cover the same spectrum. `plot_solvers.py`
draws the two transports against each other afterwards. What the third dimension moves
is plain in both directions: at the surface, shortwave shadows displaced from under the
cloud with the bright rim beside them, 11 W/m2 in the domain mean and 318 W/m2 per
column; at the top, a shortwave field the 1D solver gives as a bright cloud on a dark
background and the tracer gives as very nearly uniform -- 6 W/m2 of spread against 141 --
because photons reflected two kilometres below have spread across the whole domain by
the time they leave it.

In the longwave the lumped cell stands in for air 75 K warmer at its bottom than at its
top, so one source will not do. With one, it sent 199.6 W/m2 down into the box where the
air it stands in for sends 227.9, and the cells beneath it cooled far too hard, 33 K/day
against 0.3 in the topmost one. It now emits down and up at separate sources, taken from a
plane-parallel solve of that air per traced g-point, and sends 227.7 down against the
two-stream's 227.9; the cooling profile follows the two-stream all the way up, and
`plot_solvers.py` finds no cells to hatch.

[`les_cloudfield.toml`](les_cloudfield.toml) carries the settings and,
at length, what was checked against the reference and the two traps that
make the comparison come out wrong if you skip them: the sun has to be set explicitly
in the reference's `test.ini`, and `min_mfp_grid_ratio` has to match, since the ini
misspells the key and the built-in default applies instead of what it says.

For how long it takes and how it compares against `rte-rrtmgp-cpp`'s own fluxes for
this field, see [`RESULTS.md`](../RESULTS.md#les_cloudfield).
