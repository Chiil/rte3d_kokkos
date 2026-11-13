# Cases

Runnable end-to-end cases. [`run_case.py`](#run_casepy--the-general-runner) is the way
in: give it a case and it solves it, with the settings in the case's own `.toml` and the
atmosphere in its own NetCDF file. Every case written from here on is one of those --
a folder holding a `.toml` and nothing else that has to be code. Two are here already:

| case | what it is |
|---|---|
| [`rcemip/`](rcemip/README.md) | an analytic RCEMIP sounding over a 64x64 domain, generated on the spot |
| [`les_cloudfield/`](les_cloudfield/README.md) | the LES cumulus field the ray tracer was validated on |

The three `_rte` directories are the exception, and the suffix is what marks them: they
are the fixed comparisons against the reference implementations -- `rte-rrtmgp`'s own
RFMIP and all-sky examples and `rte-rrtmgp-cpp`'s RCEMIP benchmark -- so each carries a
script of its own that reads that case's file layout and checks against that case's
reference numbers. They are tests, not a template to copy.

| case | what it is |
|---|---|
| [`rfmip_rte/`](rfmip_rte/README.md) | clear-sky validation against rte-rrtmgp's RFMIP example |
| [`allsky_rte/`](allsky_rte/README.md) | cloudy validation against rte-rrtmgp's all-sky example, and its OpenACC benchmark |
| [`rcemip_rte/`](rcemip_rte/README.md) | performance against the Fortran reference and rte-rrtmgp-cpp |

Each case's own `README.md` is how to run it. This file is what they share: the setup
below and the general runner. [`RESULTS.md`](RESULTS.md) is what they produce: the
timings, the comparisons against the reference, and what those comparisons can and
cannot check.

## Before you start

Every script runs from the repository root and needs `RTE3D_PYTHON_PATH` pointing at
the `main_python/` directory of a build. Load it in each new shell:

```bash
cd /path/to/rte3d                                        # the repository root
export RTE3D_PYTHON_PATH=$PWD/build_cpu_dp/main_python   # or build_cpu_sp, build_gpu_dp, build_gpu_sp
```

Each case's README gives the build it is meant for and the full set of commands.

In full:

| variable | needed by | how to get it |
|---|---|---|
| `RTE3D_PYTHON_PATH` | everything here, always | the directory holding `rte3d_python*.so` |
| `RTE3D_FORTRAN_REF` | `run_rcemip_rte.py --compare-fortran` only | `./tests/build_reference.sh` |

One case needs an input file too large to keep in git. `fetch_data.py` downloads it
from the archive it is published in, checks it against the checksum published with it,
and does nothing if it is already here:

```bash
python cases/fetch_data.py                 # what there is to fetch
python cases/fetch_data.py les_cloudfield  # 26 MB
```

It is a step of its own rather than something a run script does behind your back.

Plotting needs `matplotlib`. It is not a dependency of rte3d: without it the two
validation scripts still run and still write their NetCDF output, and `--plot` reports what to install.

---

## `run_case.py` — the general runner

The general runner: an arbitrary atmosphere from a NetCDF file, no reference to compare
against. It is the Python counterpart of `test_rte_rrtmgp.cpp` in rte-rrtmgp-cpp and
reads the same input layout, so a case written for that executable runs here unchanged.

```bash
python cases/run_case.py mycase
```

`CASE` names the case: settings come from `CASE.toml`, the atmosphere from
`CASE_input.nc`, and the fluxes go to `CASE_output.nc`, folded back into the input's
own `(lev, y, x)` layout.

The case name is the only argument: every switch lives in `CASE.toml` and nowhere else,
so a case is reproducible from the two files that carry its name.
See [`example.toml`](example.toml) for the full set -- which bands and which
solvers to run, whether to read clouds, whether to write the fluxes by band, and the
coefficient files to read them all with.

### Two solvers per band

Each band can be solved twice over, by the plane-parallel two-stream solver and by
the three-dimensional Monte Carlo ray tracer, in one run. They share the gas optics, the
cloud properties and the boundary conditions and differ only in transport, which is what
makes running both worth it: the difference is what the third dimension is worth.
The `[shortwave]` section chooses, after the same section in rte-rrtmgp-cpp's `.ini`
files:

```toml
[shortwave]
plane-parallel = true      # the two-stream solver
raytracing = false         # the ray tracer; needs the grid below in the input file
photons-per-pixel = 256    # noise falls as one over the square root of this
independent-column = false # trace without horizontal transport
```

The tracer's output has a different shape from the two-stream's, because there is no
such thing as a flux profile per column once photons move sideways:

| variable | dimensions | meaning |
|---|---|---|
| `rt_flux_sfc_dir`, `rt_flux_sfc_dif` | `(y, x)` | downward direct and diffuse at the surface |
| `rt_flux_sfc_up` | `(y, x)` | upward at the surface |
| `rt_flux_tod_dn`, `rt_flux_tod_up` | `(y, x)` | at the top of the resolved box, the bottom of the lumped cell |
| `rt_flux_toa_dn`, `rt_flux_toa_up` | `(y, x)` | at the top of the atmosphere; the `tod` ones when the box resolves the whole column |
| `rt_flux_abs_dir`, `rt_flux_abs_dif` | `(z, y, x)` | absorbed flux per unit height, W/m3 |

The `[longwave]` section takes the same four keys, plus `scattering` and
`min-mfp-grid-ratio`, which [`example.toml`](example.toml) explains.
The longwave tracer's fluxes:

| variable | dimensions | meaning |
|---|---|---|
| `rt_lw_flux_sfc_dn`, `rt_lw_flux_sfc_up` | `(y, x)` | at the surface |
| `rt_lw_flux_tod_dn`, `rt_lw_flux_tod_up` | `(y, x)` | at the top of the resolved box, the bottom of the lumped cell |
| `rt_lw_flux_toa_dn`, `rt_lw_flux_toa_up` | `(y, x)` | at the top of the atmosphere; the `tod` ones when the box resolves the whole column |
| `rt_lw_flux_abs` | `(z, y, x)` | absorbed minus emitted flux per unit height, W/m3 |

### Making an input file

[`make_input.py`](make_input.py) writes one, so there is something to run
before you have written your own reader:

```bash
python cases/make_input.py mycase --nx 64 --ny 64 --clouds
python cases/run_case.py mycase
```

The atmosphere is the RCEMIP radiative-convective-equilibrium sounding of Wing et al.
(2018) for a 300 K sea surface, which is analytic, so the script needs no input data of
its own; `--clouds` adds a liquid and an ice cloud layer and switches cloud optics on in
the settings it writes. Every column holds the same profile. It is meant as a worked
example of the layout below -- the shortest way to see what your own writer has to
produce -- and the tests use it to build the case they solve.

| flag | default | meaning |
|---|---|---|
| `--nx N`, `--ny N` | `64` | horizontal size; the columns are `nx*ny` |
| `--dx M`, `--dy M` | `100` | horizontal grid spacing in metres, for the ray tracer |
| `--rt-nz N` | all | layers the ray tracer resolves; the rest are lumped into one cell on top |
| `--nlay N` | `256` | layers, equally spaced up to 70 km |
| `--sst K` | `300` | sea surface temperature, which sets the whole sounding |
| `--clouds` | off | add the two cloud layers |
| `--no-settings` | off | write only the input file, no `CASE.toml` |

### The input file

Columns are flattened as x fastest, which is what `(lay, y, x)` already is in memory.

| variable | dimensions | needed for |
|---|---|---|
| `p_lay`, `t_lay` | `(lay, y, x)` | always |
| `p_lev`, `t_lev` | `(lev, y, x)` | always |
| `vmr_<gas>` | scalar, `(lay)` or `(lay, y, x)` | always; every gas the k-distribution knows |
| `col_dry` | `(lay, y, x)` | optional; computed from `p_lev` and water vapour otherwise |
| `t_sfc` | `(y, x)` | longwave |
| `emis_sfc` | `(y, x, band_lw)` | longwave |
| `mu0` | `(y, x)` | shortwave; columns with `mu0 <= 0` come back zero |
| `sfc_alb_dir`, `sfc_alb_dif` | `(y, x, band_sw)` | shortwave; `dif` defaults to `dir` |
| `tsi` `(y, x)` or `tsi_scaling` | scalar | optional; the k-distribution's own solar source otherwise |
| `lwp`, `iwp`, `rel`, `dei` | `(lay, y, x)` | `cloud-optics` |
| `x`, `xh`, `y`, `yh`, `z`, `zh` | `(x)`, `(xh)`, ... | `raytracing`; the Cartesian grid, equally spaced |
| `ngrid_x`, `ngrid_y`, `ngrid_z` | scalar | optional; blocks of the null-collision grid |
| `azi` | `(y, x)` | optional; solar azimuth in radians from north, clockwise |

The ray tracer needs a box of equally spaced cells, which is what `x`/`xh`, `y`/`yh`
and `z`/`zh` describe. The `z` dimension is the part it resolves in three dimensions;
every layer above it is lumped into one more cell on top, so the tracer still sees the
whole atmosphere. The sun is one direction for the whole domain, taken from the first
column's `mu0` and `azi`.

`dei` is an effective *diameter*, as RRTMGP's current cloud coefficient files document
the quantity; `rei` is accepted as a name for the same thing. A file may use a flat
`col` dimension in place of `x` and `y`, and may be stored either surface-first or
top-first -- the orientation is read from `p_lay` rather than assumed.

Aerosols are not read: rte3d has no aerosol optics yet. Cloud coefficients must be the
band-resolved `-bnd` files, since the solvers take cloud properties by band.
