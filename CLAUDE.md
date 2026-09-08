# CLAUDE.md

## Prompts
Do NOT use subagents without asking.
Claude MAY build and run the tests in this repo: `cmake --build build` and
`pytest tests`. Run them to verify a change, not to browse. Anything else -- a fresh
cmake configure, a GPU build, network operations -- ask first.
After creating or changing files, always `git add` them so nothing is left untracked.
Commit at the end of each step, once it builds and the tests pass, and say so in the
report. Do not commit work that is unverified or half-finished.
Keep plan concise as possible, ask questions to clarify.
Use plan mode as default.
Indent preprocessor lines consistent with code.
For loops should use compact style without spaces around `=` and `<`: `for (int i=0; i<n; ++i)`
With linebreak after `(`, indent 8 and not 4, as the code has everywhere.

## Project Overview

A Kokkos port of RTE-RRTMGP. Two reference implementations are linked at the repo
root, both read-only:

- `rte-rrtmgp/` — the Fortran reference, authoritative for correctness. **Deprecated**:
  used as a test oracle while porting, never as a dependency. Every kernel gets a
  Kokkos implementation.
- `rte-rrtmgp-cpp/` — the older C++/CUDA port. Useful for reference but *not* the
  design to follow: it duplicates every class across CPU and GPU, and its
  `Array<T,N>` emulates 1-based column-major Fortran indexing.
- `microhh_simple/` — the design to follow. `include/types.h`, the `*_python.cpp`
  binding pattern, and the CMake/config layout all come from here.

## Non-negotiable conventions

1. **Dimension order is `(ngpt, nlay, ncol)`**, g-point outermost, column innermost.
2. **Row-major always** (`Kokkos::LayoutRight`). Never column-major, never 1-based.
   Together with (1) this makes memory identical to the reference's Fortran arrays,
   so comparisons need no transposes.
3. **One frontend for CPU and GPU.** No `#ifdef` in physics code, no parallel
   `src/` and `src_cuda/` trees. Backend differences live only in `include/types.h`.
4. **No NetCDF/HDF5 in C++.** Python reads files and passes numpy arrays.
5. **Plain structs and free functions.** No inheritance, no virtuals, no runtime
   polymorphism in the frontend.
6. **Tests are Python**, in `tests/`, run with pytest.

## Layout

- `include/types.h` — precision, array aliases, parallel-for wrappers, numpy interop
- `include/` — frontend structs and free-function declarations
- `include_kernels/` — header-only `KOKKOS_INLINE_FUNCTION` kernels
- `src/` — implementations; bindings live beside their class in `*_python.cpp`
- `main_python/` — the pybind11 module and the `rte3d` Python package
- `config/` — per-system CMake settings
- `tests/` — pytest suite

## Build

See README.md. `-DSYST=` is required.
