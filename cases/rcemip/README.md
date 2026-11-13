# `rcemip/` — a radiative-convective-equilibrium sounding

Run from the repository root, with `RTE3D_PYTHON_PATH` pointing at a build; see
[`../README.md`](../README.md#before-you-start).

It is a case like any other: run through the general runner,
[`run_case.py`](../README.md#run_casepy--the-general-runner), whose page documents
the settings and the input layout.

The RCEMIP sounding of Wing et al. (2018) at a 300 K sea surface, with the two cloud
layers, over a 64x64 domain at 100 m spacing and 256 layers. Every column holds the
same profile, so it says nothing a single column would not, but it is a full
three-dimensional field of a realistic size and it is the case the ray tracer's
optimizations were measured on -- see
[`doc/raytracer_base_optimizations.md`](../../doc/raytracer_base_optimizations.md).

The sounding is analytic, so there is no input file to fetch: `make_input.py` writes
one, and only [`rcemip.toml`](rcemip.toml) is kept in git.

```bash
python cases/make_input.py cases/rcemip/rcemip --clouds --no-settings
python cases/run_case.py cases/rcemip/rcemip
```

`--no-settings` because the settings are the committed half of the case; without it
`make_input.py` would offer to write a `.toml` and leave the existing one alone anyway.
Any of its flags changes the field -- `--nx`, `--ny`, `--nlay`, `--rt-nz` -- so a run at
a different size is a different case and belongs in a folder of its own.
