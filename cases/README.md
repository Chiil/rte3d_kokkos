# Cases

Runnable end-to-end scripts, in the spirit of `rte-rrtmgp-cpp`'s `rfmip/`, `allsky/`
and `rcemip/` directories. Two of them validate against reference fluxes, the third is
a performance benchmark, and `user/` runs whatever case you give it.
[`les_cloudfield/`](#les_cloudfield--a-three-dimensional-cloud-field) is a case rather
than a script: it is the field the ray tracer was validated on, and `user/run_case.py`
runs it.

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

| flag | meaning |
|---|---|
| `--settings F`, `-i F`, `-o F` | override the three paths derived from `CASE` |
| `--longwave` / `--no-longwave` | solve the longwave (default: on) |
| `--shortwave` / `--no-shortwave` | solve the shortwave (default: on) |
| `--cloud-optics` | read `lwp`, `iwp`, `rel`, `dei` and include clouds (default: off) |
| `--no-delta-cloud` | do not delta-scale the shortwave cloud properties |
| `--output-bnd-fluxes` | also write the fluxes resolved by band |

Flags override the settings file, so a case stays reproducible from its `.toml` alone
while a single run can still be varied. See [`user/example.toml`](user/example.toml)
for the switches and the coefficient-file paths.

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

On one MI250X GCD, double precision, 256 photons per pixel, best of three runs --
the first pays for warm-up and lands about a third high:

| | time | per column |
|---|---|---|
| longwave, 62 of 128 g-points traced | 3045 ms | 186 us |
| shortwave, 112 g-points | 7950 ms | 485 us |

### The reference fluxes, and what they can check

The same record holds `rte-rrtmgp-cpp`'s own fluxes for this field.
`--reference` fetches them, as `les_cloudfield_reference.nc`:

```bash
python cases/fetch_data.py les_cloudfield --reference     # 149 MB
```

They are worth having, but they check less than they look like they do, and it is
worth knowing which half is which before reading anything into a difference.

**The longwave surface fluxes are a real check**, and they pass:

| domain mean | reference | rte3d |
|---|---|---|
| `rt_lw_flux_sfc_up` | 456.951 | 456.959 (+0.00%) |
| `rt_lw_flux_sfc_dn` | 403.273 | 401.750 (-0.38%) |

**The top-of-domain fluxes are not comparable at all.** The reference discards the
132 layers above the ray-tracing box; rte3d lumps them into one more cell on top, so
it sees air the reference does not. That is the difference the note in
`include/raytracer_lw.h` is about, and it is large -- `rt_lw_flux_tod_up` is 374.2
against 331.8 -- but it is not an error on either side. To compare these, truncate the
input to the 200 resolved layers so both codes see the same air.

**The shortwave is not comparable either, because of aerosols.** The input carries
eleven aerosol mixing ratios, the reference reads them and rte3d has no aerosol optics
at all, so it solves a cleaner atmosphere:

| domain mean | reference | rte3d |
|---|---|---|
| `rt_flux_sfc_dir` | 660.19 | 718.10 (+8.77%) |
| `rt_flux_sfc_dif` | 194.98 | 143.67 (-26.31%) |
| the two together | 855.16 | 861.78 (+0.77%) |

Which is the signature of scattering moved from the direct beam into the diffuse rather
than of two codes disagreeing: the split moves by a quarter and the total by under a
percent. It also rules out the sun being the cause -- the solar zenith angle these
fluxes were made at is the input file's own 30 degrees, since the 45 degrees that
`test.ini` ships would have cut the total by 18% rather than 0.77%.

The record holds a third file, the backward camera's output. rte3d has no backward
tracer, so it is not fetched.

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
| 1 thread, total | 358 ms | 429 ms | **290 ms** |
| &nbsp;&nbsp;gas optics | 310 ms | | |
| &nbsp;&nbsp;transport | 59 ms | | |

Transport used to be 555 ms of that single-threaded total, two thirds of the runtime.
Each `(g-point, column)` pair ran an entire sequential vertical recurrence, which suits
a GPU but leaves the column loop with a single iteration's worth of independent work no
matter what `ivdep` asserts. Solving one g-point at a time restores the reference's
structure -- layer outside, `ncol` innermost -- and the recurrences now vectorize over
columns: a 9x improvement. Gas optics is what remains of the gap.

Where it really pays is at scale, because a single g-point's working set stays in
cache instead of streaming ~80 MB arrays through memory. Longwave, 15 threads:

| columns | before | now |
|---|---|---|
| 256 | 131 ms | 221 ms |
| 512 | 274 ms | 281 ms |
| 1024 | 474 ms | 425 ms |
| 4096 | 9495 ms, 11 GB | **1226 ms, 1.1 GB** |

Below about 512 columns the per-g-point solve is launch-bound: each g-point issues
around ten parallel regions of only `ncol` work each, and at 256 columns on 15 threads
that overhead outweighs the cache win. The fix is to parallelize over g-points on the
host, giving each thread its own working set -- now possible, since the g-point loop
lives in C++ rather than in the caller. It has not been done.

Both builds agree with the reference to 1e-13 W/m2, so none of this is bought with
accuracy.

### On a GPU

RCEMIP at 65536 columns x 256 layers, single precision, on an RTX A4500:

| | longwave | shortwave |
|---|---|---|
| before | 1879 ms | 2167 ms |
| now | **1314 ms** | **1531 ms** |

Every one of those milliseconds was memory traffic, not arithmetic and not
parallelism: the kernels were already running at 400-650 GB/s, against a card that
peaks near 640. So the work was to move less --- the binary-species interpolation
recomputed rather than stored (4.0 GB, and 24 bytes per cell per g-point), the Planck
fraction and the Rayleigh scattering taken from the interpolation the optical depth was
already doing, and the fluxes added into the spectral totals where they are produced
instead of being written per g-point and read straight back. `git log` has the
measurement for each.

What is left is the column sweeps. Against `rte-rrtmgp-cpp`'s CUDA solver on its own
RCEMIP case --- the same input file, 4096 columns, both single precision --- that is
now the whole of the gap:

| stage | rte3d | rte-rrtmgp-cpp |
|---|---|---|
| longwave, optical depth + Planck | 35.3 ms | 33.3 ms |
| longwave, transport | **70.1 ms** | **38.3 ms** |
| shortwave, optical depth + Rayleigh | 22.5 ms | 19.0 ms |
| shortwave, transport | **156 ms** | **40.5 ms** |
| total, longwave / shortwave | 120 / 198 ms | 74 / 75 ms |

Gas optics is at parity. The sweeps are not, and the launch geometry says why: it
solves four column blocks with every g-point resolved, so its `sw_adding` runs 229376
threads and its `lw_solver_noscat_step_2` 262144, where ours run `ncol` --- 4096. Same
work, 64x the parallelism, and at 4096 columns ours reach about 15% of peak bandwidth
and 8% occupancy where at 65536 they reach 70%. The gap is a small-problem gap: 48.3
us/column at 4096 against 23.4 at 65536, while a chunked full-spectrum solver is flat
in problem size and pays for it in memory --- 4 column blocks here, 64 for the case
above.

Block size is not the lever it looks like. Pinning the sweeps' CUDA block to 32 instead
of the 128 Kokkos picks spreads them over all 56 SMs rather than leaving 24 idle, and
changes the time by nothing: a block moves warps between SMs but does not create any,
and a sweep has ncol/32 of them whatever the block. The 2-D launches did respond to
tiling, which is a layout question rather than an occupancy one; see the note in
`include/types.h`.

Two ways out, neither taken: fuse the direct-beam sweep with the adding sweeps, which
move five `(nlay, ncol)` arrays between them, or give the transport arrays a g-point
block dimension, which buys the parallelism directly at the cost of the property this
design is built around.
