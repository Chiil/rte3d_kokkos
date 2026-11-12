# Changes from the reference rte-rrtmgp

What rte3d does differently from the Fortran reference (`rte-rrtmgp/`, followed to
main #431) and why. [`DESIGN.md`](DESIGN.md) has the full reasoning.

## Structure

| change | motivation |
|---|---|
| Kokkos C++, one source for CPU and GPU; backend only in `types.h` and `random.h` | One code path, no duplicated CPU/GPU trees as in rte-rrtmgp-cpp |
| Arrays `(ngpt, nlay, ncol)`, row-major, 0-based | Same bytes as the Fortran arrays: no transposes in tests, column unit-stride for coalescing and vectorization |
| Gas-optics state arrays keep `ncol` fastest, where the reference has it in the middle | Same unit-stride column everywhere; the tests transpose for this one case |
| Plain structs and free functions replace the class hierarchy of `rte/frontend/` and `rrtmgp/frontend/` | No inheritance or virtuals in the frontend |
| No NetCDF in C++; Python reads files, C++ reduces the k-distribution | A host model can load coefficients without Python; I/O stays out of the library |
| `namespace Solver` drives gas optics, clouds and transport | The reference has no driver; this one is the single way to run a case |

## Execution

| change | motivation |
|---|---|
| Whole solve runs one g-point at a time | Working set does not scale with the spectrum: 1.2 GB instead of 28 GB for RCEMIP at 4096 columns |
| Minor absorbers parallel over (g-point, layer, column), through a CSR map per g-point | The reference loops them serially per column range, which does not parallelize |
| Interpolation weights (`fmajor`, `fminor`, `jeta`, `feta`, `col_mix`) recomputed in the kernel instead of stored | Saves GBs at large column counts; they are cheap functions of a few stored values |
| Planck fraction and Rayleigh computed in the same kernel as the major-species optical depth | One pass over the costliest interpolation instead of three |
| Solvers add straight into the spectral flux totals; no per-g-point flux arrays | Less memory and traffic |
| Vertical recurrences as column sweeps with carried values in registers | No sweep rereads an array it wrote; fewer scratch arrays |
| `Vert<top_at_1>` template instead of writing every loop twice | The physics is written once for both orientations |
| `top_at_1` decided once, from one column | The reference decides it in several places |

## Numerics (the results differ from the reference here)

Mostly from RRTMGP.jl's precision notes. They matter in single precision; in double
precision the fluxes agree with the reference to about 1e-13.

| change | motivation |
|---|---|
| Delta-scaling uses `g/(1+g)` instead of `(g-g²)/(1-g²)` | No cancellation as g → 1 (error in g' 4e-5 → 3e-8) |
| Two-stream `k²` from `2(1-w0)` (SW) or `lw_diff_sec·(1-w0)` (LW); `1-exp(-2kτ)` via `expm1` | No cancellation as w0 → 1 or τ → 0 |
| Meador-Weaver numerators rewritten through `1-e1` and `1-T` | No O(1) cancellation as k → 0 |
| SW direct beam interpolated across the singularity at k·μ0 = 1 | The reference's `+eps` denominator loses Rdir/Tdir in single precision (errors up to 0.2 of the incoming flux) |
| Floor on `k²` is 1e-12 in both bands, not `1e4·eps` | `1e4·eps` = 1.2e-3 in single precision and biases k (up to 0.6 W/m²) |
| LW two-stream source rewritten without the divide-by-τ difference | The reference's error grows like eps/τ (37.6 W/m² single precision at τ = 1e-6) |
| `feta` relative to the clamped `jeta` | The reference jumps to the wrong node at eta = 1, exactly what an absent second gas gives |

## Reference defects

| defect | handling |
|---|---|
| `interpolation` indexes `temp_ref` out of bounds off the table (#315, open) | rte3d keeps v1.8's clamped index; tests compare only where the reference is defined |
| `LW_diff_sec = 1.66` is a single-precision literal | Matched bit-for-bit, not fixed |
| `flux_upJac` mis-documented in `rte_kernels.h` | Follows the Fortran declaration `(ncol, nlay+1)` |

## Not ported

| item | reason |
|---|---|
| `lw_transport_1rescl` (Tang et al. 2018 rescaling) | Its two orientation branches read the upward radiance at different levels; to be resolved with the reference authors before reproducing it |
| `zero_array_*` | `Kokkos::deep_copy` does the same |
| Pade cloud optics | No shipped coefficient file has Pade coefficients; lookup tables only |
| Aerosol optics | Not done yet |

## Additions

| item | motivation |
|---|---|
| 3D Monte Carlo ray tracers for SW and LW (after rte-rrtmgp-cpp) | 3D radiation for LES; use the same gas and cloud optics as the two-stream |
| Integer photon counters (fixed point) and per-thread seeding | Traces reproduce bit for bit; RNG setup is cheaper |
| `min_mfp_grid_ratio` fallback to plane-parallel for opaque g-points | No photons spent where there is no horizontal transport |
| Fast math in the ray-tracer files only | Monte Carlo noise is far above the cost of fast math; everything else stays IEEE |
| Approximate division (`fast_div`) in the SW and LW solvers, GPU single precision only | 5-16% faster solves on the all-sky case; gas optics keeps exact division for its invariants |
