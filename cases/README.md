# Cases

Runnable end-to-end scripts, in the spirit of `rte-rrtmgp-cpp`'s `rfmip/`, `allsky/`
and `rcemip/` directories. Two of them validate against reference fluxes, the third is
a performance benchmark, and `user/` runs whatever case you give it.
[`les_cloudfield/`](#les_cloudfield--a-three-dimensional-cloud-field) is a case rather
than a script: it is the field the ray tracer was validated on, and `user/run_case.py`
runs it.

This file is how to run them. [`RESULTS.md`](RESULTS.md) is what they produce: the
timings, the comparisons against the reference, and what those comparisons can and
cannot check.

## Before you start

Every script needs `RTE3D_PYTHON_PATH` pointing at a build directory:

```bash
export RTE3D_PYTHON_PATH=$PWD/build/main_python      # or build_gcc/main_python
```

In full:

| variable | needed by | how to get it |
|---|---|---|
| `RTE3D_PYTHON_PATH` | all four, always | the directory holding `rte3d_python*.so` |
| `RTE3D_FORTRAN_REF` | `run_rcemip.py --compare-fortran` only | `./tests/build_reference.sh` |

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

## `user/run_case.py` — your own case

The general runner: an arbitrary atmosphere from a NetCDF file, no reference to compare
against. It is the Python counterpart of `test_rte_rrtmgp.cpp` in rte-rrtmgp-cpp and
reads the same input layout, so a case written for that executable runs here unchanged.

```bash
python cases/user/run_case.py mycase
```

`CASE` names the case: settings come from `CASE.toml`, the atmosphere from
`CASE_input.nc`, and the fluxes go to `CASE_output.nc`, folded back into the input's
own `(lev, y, x)` layout.

The case name is the only argument: every switch lives in `CASE.toml` and nowhere else,
so a case is reproducible from the two files that carry its name.
See [`user/example.toml`](user/example.toml) for the full set -- which bands and which
solvers to run, whether to read clouds, whether to write the fluxes by band, and the
coefficient files to read them all with.

### Two shortwave solvers

The shortwave can be solved twice over, by the plane-parallel two-stream solver and by
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
| `rt_flux_tod_dn`, `rt_flux_tod_up` | `(y, x)` | at the top of the ray-tracing domain |
| `rt_flux_abs_dir`, `rt_flux_abs_dif` | `(z, y, x)` | absorbed flux per unit height, W/m3 |

### Making an input file

[`user/make_input.py`](user/make_input.py) writes one, so there is something to run
before you have written your own reader:

```bash
python cases/user/make_input.py mycase --nx 64 --ny 64 --clouds
python cases/user/run_case.py mycase
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

---

## `les_cloudfield/` — a three-dimensional cloud field

The case the ray tracer was validated on: the example LES field of `rte-rrtmgp-cpp`, a
2560 x 2560 x 4000 m RICO cumulus field at 20 m resolution. 128 x 128 columns, 200
ray-tracing cells in the vertical and 332 layers, of which the 132 above the box are
lumped into one cell on top. Cloud is liquid-only and thin: 1.3% of cells, 19% of
columns.

```bash
python cases/fetch_data.py les_cloudfield
python cases/user/run_case.py cases/les_cloudfield/les_cloudfield
```

The input is the same `test_input.nc` that `rte-rrtmgp-cpp`'s `test_rte_rrtmgp_rt`
reads, published at [10.5281/zenodo.18757088](https://doi.org/10.5281/zenodo.18757088)
(CC BY 4.0). Fetching it writes `les_cloudfield_input.nc` beside the settings, which is
the name `run_case.py` derives from the case prefix; the fluxes go to
`les_cloudfield_output.nc` in the input's own `(z, y, x)` layout.

Both bands are traced, with the g-point counts the reference case uses -- 112 shortwave
and 128 longwave -- so that a run here and a run of `test_rte_rrtmgp_rt` cover the same
spectrum. [`les_cloudfield.toml`](les_cloudfield/les_cloudfield.toml) carries the
settings and, at length, what was checked against the reference and the two traps that
make the comparison come out wrong if you skip them: the sun has to be set explicitly
in the reference's `test.ini`, and `min_mfp_grid_ratio` has to match, since the ini
misspells the key and the built-in default applies instead of what it says.

For how long it takes and how it compares against `rte-rrtmgp-cpp`'s own fluxes for
this field, see [`RESULTS.md`](RESULTS.md#les_cloudfield).

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

For what `--compare-fortran` actually times, where rte3d stands against the reference
on a host, and where it stands on a GPU, see [`RESULTS.md`](RESULTS.md).
