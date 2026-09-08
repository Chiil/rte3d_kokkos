# rte3d

RTE-RRTMGP in Kokkos: one frontend for CPU and GPU, row-major arrays throughout, and
the g-point as the outermost dimension.

## Why the dimension order matters

Arrays are ordered `(ngpt, nlay, ncol)` and stored row-major, so the column is the
fastest-varying index. This is byte-for-byte the layout of the reference Fortran's
`(ncol, nlay, ngpt)` column-major arrays. Two consequences:

- The column stays unit-stride: coalesced on GPU, vectorizable on CPU.
- Comparing against the reference needs no transposes. A numpy array of shape
  `(ngpt, nlay, ncol)` in C order can be handed straight to the Fortran kernels.

Fixing the order this way removes the need for the separate CPU and GPU code paths
that `rte-rrtmgp-cpp` carries.

## Status

Step 1 complete: the RTE core.

- `include/types.h` -- precision, array aliases, parallel-for wrappers, numpy interop
- `include/rte_sw.h` -- `sw_solver_noscat`, `sw_solver_2stream`
- `include/rte_lw.h` -- `lw_solver_noscat` with multi-angle quadrature and the
  surface-temperature Jacobian, `lw_solver_2stream`
- `include/optical_props.h` -- delta-scaling, the increment operations, column subsetting
- `include/fluxes.h` -- broadband and by-band flux reduction

Everything is validated against the Fortran reference at ~1e-15 relative in double
precision.

Two things in `rte-kernels/` are deliberately not ported:

- `lw_transport_1rescl`, the approximate-scattering rescaling of Tang et al. 2018.
  See the note at the top of `src/rte_lw.cpp`.
- `zero_array_*`, which `Kokkos::deep_copy` already covers.

The `rte-frontend/` layer is replaced rather than ported: its class hierarchy
(`ty_optical_props`, `ty_source_func_lw`, `ty_fluxes`) becomes the plain structs above.
The top-level `rte_lw()` / `rte_sw()` drivers, which validate inputs and expand
boundary conditions before calling the kernels, are not yet written.

Step 2a complete: all four RRTMGP gas-optics kernels -- `interpolation`,
`compute_tau_absorption`, `compute_tau_rayleigh` and `compute_planck_source`
(`include/gas_optics.h`). The Planck kernel fills the same `Source_func_lw` the
longwave solvers already take, so the two halves compose directly.

Step 2b in progress. `Gas_concs`, `compute_col_dry` and `compute_col_gas` are done
(`include/gas_concs.h`, `include/gas_optics.h`). Next: the k-distribution reduction,
then the frontend that orchestrates the kernels, then reading coefficient files with
xarray.

### Where gas names live

In C++, and only in `Gas_concs` and the k-distribution loader -- never in a kernel.
The reference resolves names at two moments: once at load, where the k-distribution is
*reduced* to the gases the host actually supplies and every integer index array is
rebuilt against that shorter list, and once per call, to look up each gas's mixing
ratio. rte3d keeps both in C++ so that a host model with no Python can still load a
coefficient file; Python's only job is reading the file into arrays.

The reference loops minor absorbers serially, walking a per-column layer range for
each. rte3d parallelises over (g-point, layer, column) instead, which needs the inverse
mapping -- for each g-point, which minor absorbers contribute. `Minor_absorbers` holds
that as a CSR list built once by `build_map`.

### Interpolation weights are recomputed, not stored

The reference materialises `fmajor(2,2,2,ncol,nlay,nflav)` and
`fminor(2,2,ncol,nlay,nflav)`. Both are pure functions of `ftemp`, `fpress` and `feta`,
which the same kernel already computes, so rte3d stores those instead and rebuilds the
weights where they are used (`Gas_optics_kernels::interp_weights`). That trades twelve
stored values per (flavour, layer, column) for four multiplies: at `ncol = 1e5`,
`nlay = 60`, `nflav = 10` it is roughly 2.4 GB instead of 7 GB.

rte3d also keeps the column as the fastest-varying dimension in these arrays, where the
reference puts `ncol` in the middle. This is the one place where a straight dimension
reversal would not have matched, so the gas-optics tests transpose before calling the
reference. The library itself never transposes.

### Known defects in the Fortran reference

Each is reproduced or worked around deliberately, and pinned by a test.

- **`lw_solver_2stream` ignores the g-point index of `lev_source`**
  (`rte-kernels/mo_rte_solver_kernels.F90:422`): the call to `lw_source_2str` passes
  `lev_source` rather than `lev_source(:,:,igpt)`, so Fortran sequence association
  hands every g-point the first g-point's slice. `lw_solver_noscat_oneangle:189` does
  it correctly. Present since the original 2018 import. rte3d does the correct thing;
  see `test_2stream_reference_bug_gpt_indexing`.
- **`LW_diff_sec = 1.66` is a single-precision literal** promoted to double, so its
  value is 1.65999996662139893. rte3d matches it bit-for-bit rather than "fixing" it.
- **`rte_kernels.h` mis-documents `flux_upJac`** as `(ncol,nlay+1,ngpt)`; the Fortran
  declares it `(ncol,nlay+1)`, since only broadband Jacobians are provided.

The vertical orientation is handled by `Rte_kernels::Vert<top_at_1>` in
`include_kernels/rte_solver_kernels.h`. The reference writes every loop out twice, once
per orientation; templating on the orientation keeps the explicit loops the compiler
wants while writing the physics once.

## Dependencies

C++20 compiler, Python 3 with numpy and pytest. Kokkos and pybind11 come in as git
submodules. There is deliberately **no** NetCDF, HDF5, or FFTW dependency: all file
reading happens in Python and arrives through pybind11 as numpy arrays.

```bash
git submodule add https://github.com/kokkos/kokkos.git extern/kokkos
git submodule add https://github.com/pybind/pybind11.git extern/pybind11
```

## Building

```bash
mkdir build && cd build
cmake .. -DSYST=macbook
cmake --build .
```

Switches, following MicroHH:

- `-DSYST=<system>` — required, picks `config/<system>.cmake`
  (`macbook`, `macbook_gcc`, `ubuntu_22lts_gcc`)
- `-DUSEGPU=1` — build for GPU; the backend (CUDA or HIP) comes from the config file
- `-DUSESP=1` — 32-bit floats instead of 64-bit
- `-DCMAKE_BUILD_TYPE=DEBUG`

## Testing

```bash
export RTE3D_PYTHON_PATH=$PWD/build/main_python
pytest tests
```

To also run the comparisons against the reference implementation:

```bash
./tests/build_reference.sh          # needs gfortran; set FC to override
export RTE3D_FORTRAN_REF=$PWD/build/reference/librte_kernels.dylib
pytest tests
```

Those tests skip when `RTE3D_FORTRAN_REF` is unset. The Fortran is a correctness oracle only — it is
deprecated, never linked, and not needed to build or use rte3d.
