# `rcemip/` — a radiative-convective-equilibrium sounding

The RCEMIP sounding of Wing et al. (2018) at a 300 K sea surface, with the two cloud
layers, over a 64x64 domain at 100 m spacing and 256 layers. Every column holds the
same profile, so it says nothing a single column would not, but it is a full
three-dimensional field of a realistic size and it is the case the ray tracer's
optimizations were measured on -- see
[`doc/raytracer_base_optimizations.md`](../../doc/raytracer_base_optimizations.md).

The sounding is analytic, so there is no input file to fetch: `make_input.py` writes
one, and only [`rcemip.toml`](rcemip.toml) is kept in git.

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

Then write the input file and solve it:

```bash
python cases/make_input.py cases/rcemip/rcemip --clouds --no-settings   # -> rcemip_input.nc
python cases/run_case.py cases/rcemip/rcemip                            # -> rcemip_output.nc
```

The settings are in [`rcemip.toml`](rcemip.toml). It is run by the general runner,
[`run_case.py`](../README.md#run_casepy--the-general-runner), whose section documents
every setting and the output variables.

`--no-settings` because the settings are the committed half of the case; without it
`make_input.py` would offer to write a `.toml` and leave the existing one alone anyway.
Any of its flags changes the field -- `--nx`, `--ny`, `--nlay`, `--rt-nz` -- so a run at
a different size is a different case and belongs in a folder of its own.
