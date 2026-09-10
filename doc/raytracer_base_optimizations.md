# Ray tracer base optimizations

Plan for porting the *base optimizations* of Ivor Brouwer's MSc thesis, **"GPU Thread
Data Remapping In Monte Carlo Short-wave Ray-tracing"** (LIACS, Leiden University,
26 August 2026; supervisors Ben van Werkhoven and Stijn Heldens), to the Kokkos ray
tracer in this repository.

This file is self-contained: everything needed from the thesis is reproduced below, so
it can be read and acted on without the PDF.

Written against `include_kernels/raytracer_kernels.h` and `src/raytracer.cpp` at commit
`8826bde` ("Take the divisions out of the collision, and the cancellation with them").

---

## 1. What the thesis is, and which half of it we want

The thesis asks whether *thread-data remapping* — reorganizing photon state at runtime so
that the threads of a warp all handle the same event kind — reduces branch divergence
enough to pay for itself in a history-based short-wave photon tracer (one thread = one
photon history, which is what we do too).

To make that comparison fair, the author first **optimized the history-based baseline**
(thesis §4.1, "Base Optimizations"), reaching a **2.81×–4.16× speed-up** over the original
across the evaluated workloads. Only then were the remapping variants built on top.

The remapping half **lost**. The best remapping variant (a warp-local stack) was 9.6%
slower than the optimized baseline for a clear-sky workload, 1.4% slower for a
moderate-cloud workload, and only 1.9% faster for a synthetic scattering-intensive
workload. The author's own conclusion is that improved warp coherence alone does not
guarantee better performance, and that for realistic workloads the optimized
history-based implementation remains the preferred approach.

**So: port §4.1, ignore §4.2–4.7.** The remapping machinery (warp stacks, block stacks,
link-based shared-memory structures, vote skipping) is warp-intrinsic-heavy — which
collides with our convention 3, "one frontend for CPU and GPU" — and did not win.

---

## 2. The nine base optimizations, and how each lands here

The thesis' §4.1 items, verbatim in substance, each judged against our code.

| # | Thesis §4.1 item | Verdict for us |
|---|---|---|
| 4.1.1 | Removal of the Sobol sequence | **Deferred by decision** — we keep Sobol for now |
| 4.1.2 | Better reset locality in warp, and work stealing | Partly applicable (stealing only) |
| 4.1.3 | Overall register / complexity reductions | Applicable — `fmod` is still in our hot loop |
| 4.1.4 | Simplified scatter optics | Applicable, reduced payoff (no aerosols) |
| 4.1.5 | Pre-calculated cell absorption and scattering | **Applicable — the main item** |
| 4.1.6 | Improved caching (last fine cell in registers) | **Applicable, strong** |
| 4.1.7 | Accumulation of atmospheric contributions | **Applicable, strong** |
| 4.1.8 | Float-to-int simplification and fast approximation | Half applicable |
| 4.1.9 | Mixed precision (fp16 storage) | Applicable, do last |

### 4.1.1 — Removal of the Sobol sequence *(deferred by decision)*

The thesis replaced quasi-random Sobol photon launch positions with a plain uniform draw
restricted to a per-thread-block sector of the domain. Three claimed gains: the Sobol
state variables and grid logic free up registers (better occupancy); photons launched by
one block stay in one region of the domain, so their paths are similar and cache better;
and kernel arguments needed only for Sobol (notably a global per-cell photon counter,
needed because a quasi-random draw does not guarantee an exact photon count per cell)
disappear along with their atomics. The trade-off is less low-discrepancy structure at
the start, which showed no significant effect on output.

**We are keeping Sobol.** Note for later, if that is ever revisited: our `Rand::Qrng_2d`
holds two `curand` scrambled-Sobol states (32 direction vectors each), and `reset_photon`
runs a **rejection loop** — draws that land outside a non-power-of-two grid are thrown
away and redrawn — which is both wasted work and a divergence source. Also, each thread
draws a contiguous slice of the *global* sequence, so a warp's photons are spread over the
whole domain; the locality argument in 4.1.2 below depends on fixing that.

### 4.1.2 — Reset locality and work stealing *(stealing only)*

Two parts. The locality part depends on 4.1.1 and is therefore out.

The **work-stealing** part stands alone: in the original, each thread was assigned a fixed
number of photons, so the kernel's duration was set by whichever thread drew the longest
photon paths. The thesis replaced this with a **block-wide atomic counter**: whenever a
thread initializes a new photon it atomically increments the counter, so threads that
finish early absorb work from threads that did not. The author notes the atomic is cheap
in practice because it is hit sparsely — with a 64×64 grid each block processes roughly
`256 * 6 * 6` photons (photons per cell × sector size), so contention on the counter is
limited and the overhead negligible.

Our equivalent: `launch_photons` (`src/raytracer.cpp:187`) hands thread `n` a fixed
`photons_per_thread`, with the remainder spread one each over the first `photons_extra`
threads. Adopting the counter means moving from `Kokkos::RangePolicy` to a `TeamPolicy`
with a team-scoped counter — a real structural change, hence last in the ordering below.

### 4.1.3 — Register and complexity reductions

A grab-bag in the thesis: evaluate at compile time what was evaluated at runtime; shorten
variable lifetimes; drop kernel arguments that only Sobol needed. Two concrete items
transfer directly:

- **The cyclic boundary.** The original used `fmod`, an expensive operation. Because a
  photon moves at most one grid cell at a time, the wrap can be done with a single
  subtraction or addition of the domain size instead.
- **The scatter frame.** The original built two vector structs to compute the new
  direction, keeping more registers live than necessary.

Both are still present in our code — see §3.

### 4.1.4 — Simplified scatter optics

The scattering *type* (gas / cloud / aerosol) is sampled from the individual scattering
coefficients. Doing that in the kernel means reconstructing

```
k_sca = k_sca_gas + k_sca_cld + k_sca_aer
```

and comparing a random number against normalized partial sums — three global loads, plus
the asymmetry factor if the sampled type is cloud or aerosol, so **five loads worst case**.

All three coefficients belong to the same grid position, so the probabilities can be built
once, beforehand, in an optics pre-processing step. Instead of the raw coefficients, store
two thresholds:

```
p_aer     = k_sca_aer / k_sca
p_nongas  = (k_sca_aer + k_sca_cld) / k_sca
```

Then for a uniform `r ∈ [0,1)` the type is chosen by comparison alone:

```
r >= p_nongas  ->  gas scattering
r <  p_aer     ->  aerosol scattering
otherwise      ->  cloud scattering
```

Gas scattering now needs **one** load (the non-gas cutoff) instead of three; cloud and
aerosol need **two** (threshold, then asymmetry factor), so the worst case drops from five
to three. In the thesis' test workload the commonest case was cloud scattering, then gas,
then aerosol (zero occurrences).

### 4.1.5 — Pre-calculated cell absorption and scattering *(the main item)*

The original ray-tracing kernel computes two quantities at every fine-grid evaluation:
`f_noabs`, the fraction of photon weight surviving implicit absorption, and `f_scat`, the
conditional probability that a non-absorbed candidate interaction is a *real* scattering
event rather than a null collision. Computing them needs **four** cell-dependent optical
properties from global memory, and because photons visit cells in an irregular order those
loads are non-coalesced. The arithmetic is:

```
k_sca_tot = k_sca_gas + k_sca_cld + k_sca_aer          (18)
ssa       = k_sca_tot / k_ext                          (19)
f_noabs   = 1 - (1 - ssa) * k_ext / k_ext_null         (20)
f_scat    = ssa / (ssa - 1 + k_ext_null / k_ext)       (21)
```

Both depend only on **cell-local** optical properties (plus `k_ext_null`, which is a
property of the coarse block the cell sits in), so both can be pre-computed in an **optics
pre-processing kernel**. That kernel traverses the grid **in memory order**, so its loads
and stores are coalesced — which is the whole point: the same numbers, read irregularly by
the photon walk, are instead produced once, in order.

The ray-tracing kernel then loads the two pre-computed values. Irregular global loads in
the coarse-grid path go from **four raw coefficients to two probabilities**, and the
repeated arithmetic for `k_sca_tot`, `ssa`, `f_noabs` and `f_scat` leaves the photon loop
entirely.

### 4.1.6 — Improved caching

During coarse-grid traversal, several consecutive steps can fall inside the *same* fine
cell — in particular consecutive null collisions. In that case the fine cell's optical
properties need not be reloaded. Keep the most recently used fine-grid index in registers;
if the newly computed index equals the cached one, reuse the cached `f_noabs` and
`f_scat`. Costs a few registers, saves repeated global loads and the memory stalls behind
them.

### 4.1.7 — Accumulation of atmospheric contributions

At every fine-grid evaluation the photon deposits

```
contribution = w * (1 - f_noabs)                       (22)
```

into the direct or diffuse atmospheric output for that cell. The original wrote this
immediately with a **global atomic**, on every single evaluation — null collisions
included.

Instead, keep three extra registers: the last fine-grid index, the last photon kind, and
an accumulator. Add into the accumulator while both the index and the kind are unchanged;
when either changes, flush the accumulated total with a **single** read-modify-write, reset
the accumulator, and adopt the new cell and kind as the cached state. A final flush after
the traversal loop is required to write the last accumulation.

The photon kind must be part of the key because direct and diffuse photons are written to
separate output arrays.

This turns many atomics per fine-cell crossing into one per crossing, for the price of
three registers.

### 4.1.8 — Float-to-int simplification and fast approximation

Nsight Compute showed the float position → integer cell conversion accounting for **10% of
total program cycles**. The original form:

```
integer position = static_cast<int>(f / cell_size);    (23)
```

was replaced by CUDA built-ins:

```
integer position = __float2int_rz(__fdividef(f, cell_size));   (24)
```

The trade-off is slightly reduced division accuracy, with no significant output difference
observed.

### 4.1.9 — Mixed precision

Store `k_ext_null`, `f_scat` and the optical properties in **half precision**. Halving
their footprint means more optics data fits in the same cache, increasing reuse when
neighboring photons touch nearby cells, and reduces bytes moved from global memory. The
thesis reports no significant impact on output. Beneficial when memory latency and
bandwidth are the limiter.

---

## 3. Where our code stands today

Facts about our implementation, so the plan below can be followed without re-reading the
sources.

**The photon walk** is `Rt_kernels::trace_photons` in `include_kernels/raytracer_kernels.h`,
one function, templated on `bool independent_column`. One thread walks
`photons_to_shoot` photons in sequence, reusing registers.

**Per-collision work** (the `else` branch of `dn >= d_max`, around
`raytracer_kernels.h:407-424`) currently does, every time:

- derive `i`, `j`, `k` from the position via `coord_to_index` (a divide plus a cast each);
- load `Optics_scat scat` — an AoS struct of `{k_sca_gas, k_sca_cld, asy_cld}` — and
  `k_ext`, so **four values**;
- recompute `k_sca_tot`, `k_abs`, `f_abs`, `f_no_abs`, `k_ext_no_abs`;
- `Kokkos::atomic_add` into `atmos_dir` or `atmos_dif` — **on every collision, null ones
  included**;
- Russian roulette;
- decide real vs. null collision with `rng()*k_ext_no_abs >= k_sca_tot`;
- on a real scatter, pick cloud vs. gas with `rng()*k_sca_tot < scat.k_sca_cld`, then
  sample the phase function.

**Already done** (recent commits, do not redo):

- `8826bde` — the collision test was rewritten as a product, `rng()*k_ext_no_abs >=
  k_sca_tot`, rather than the ratio it came from. This deliberately removes a division
  *and* the cancellation the ratio suffers in a cell that is its own block's maximum.
  `f_abs` is written as `k_abs*k_ext_null_inv`, not the algebraically equal
  `(k_ext_null - k_abs)/k_ext_null`, so a conservative cell leaves the weight exactly
  alone.
- `afa23e1` — the quasi-random launch uses a shift, not a division, onto the lattice.
- `1fc242b`, `02b6741` — single-precision GPU builds carry no double-precision
  instructions; fast math is enabled for the ray tracer and nothing else.
- `373b237` — rays are traced once, not four times.

**Still present, and matching a thesis item:**

- `Kokkos::fmod` for the cyclic boundary, at `raytracer_kernels.h:378-384` — thesis 4.1.3.
- The two-vector scatter frame `t1`, `t2` in `scatter()`, `raytracer_kernels.h:218-236` —
  thesis 4.1.3.
- `coord_to_index` divides by `ds` — thesis 4.1.8.
- Unconditional atomics into `atmos_dir`/`atmos_dif` — thesis 4.1.7.
- No fine-cell caching at all — thesis 4.1.6.
- Static work split in `launch_photons`, `src/raytracer.cpp:187` — thesis 4.1.2.

**The optics pipeline** already runs per g-point, inside `Raytracer::trace`:

```
src/raytracer.cpp:340   bundle_optics(...)        -> k_ext, scat  (per fine cell)
src/raytracer.cpp:342   bundle_optics_tod(...)    -> the lumped top cell
src/raytracer.cpp:345   create_knull_grid(...)    -> k_null       (per coarse block)
src/raytracer.cpp:419   launch_photons<...>(...)
```

A pre-processing kernel therefore slots in as a fourth call, **after** `create_knull_grid`
and before `launch_photons`, with no restructuring of the call sequence.

**Three differences from the thesis' code** that shape the port:

1. **No aerosols.** `Optics_scat` is `{k_sca_gas, k_sca_cld, asy_cld}`. The two thresholds
   of 4.1.4 collapse to a single `p_cld = k_sca_cld / k_sca_tot`. Smaller win than the
   thesis reports.

2. **Storage does not shrink — the hot path does.** We load four values per collision
   today. The pre-computed table is also four values per cell
   (`f_no_abs`, `f_scat`, `p_cld`, `asy_cld`), but the **null-collision path, by far the
   most common, needs only the first two**, and `asy_cld` is touched only on a cloud
   scatter, the rarest branch. The gain comes from **splitting the arrays along the right
   line**: a hot pair (8 bytes in single precision, one 64-bit load) plus a cold pair —
   rather than today's `{k_ext} + {k_sca_gas, k_sca_cld, asy_cld}` split, which forces the
   kernel to touch both.

3. **`f_scat` reintroduces the division that `8826bde` removed.** Pre-computing
   `f_scat = k_sca_tot / (k_ext_null - k_abs)` moves that division into a coalesced
   pre-pass where it can be done carefully — fine in itself — but a conservative cell that
   *is* its block's maximum should give exactly `1.0` and will instead land a rounding
   short, and that shortfall accumulates over a scattering photon's many collisions.
   Guard it (see step 2 below). `f_no_abs` is safe to pre-compute in the form we already
   use.

**One coupling to respect:** `f_no_abs` and `f_scat` depend on `k_ext_null`, i.e. on the
coarse block. Each fine cell sits in exactly one block, so the table is well defined — but
it is invalidated by any change to the `kn` grid, and the pre-pass must run *after*
`create_knull_grid`, never before. The lumped top-of-domain cell from `bundle_optics_tod`
must go through the same pre-pass.

---

## 4. The plan

Ordered so that each step is independently verifiable with `pytest tests`, and so that no
step has to be undone by a later one.

### Step 1 — Fine-cell caching and contribution accumulation *(thesis 4.1.6 + 4.1.7)*

Do these two together: they key off the same test, "did the fine cell change?".

- In `trace_photons`, hold in registers: the last fine-cell linear index, the last
  `Photon_kind`, an absorption accumulator, and the last cell's optical values.
- After computing `i`, `j`, `k`, compare the linear index with the cached one. If equal,
  skip the loads and reuse the cached values.
- Accumulate `weight*f_abs` into the accumulator instead of calling `Kokkos::atomic_add`.
  Flush with one atomic when **either** the fine index **or** the photon kind changes.
- Flush once more after the traversal loop, and — importantly for us — **on every photon
  reset**, since `reset_photon` is called from three places (surface, top-of-domain,
  roulette death) and a photon's last deposit must not leak into the next photon's cell.
- Invalidate the cached index on reset as well.

No interface change, no new arrays. Expected to be the largest single win, since it
removes atomics from the null-collision path.

### Step 2 — The optics pre-pass *(thesis 4.1.5 + 4.1.4)*

- Add a kernel — `precompute_collision_optics`, say — over `(nz, ncol)` in memory order,
  called from `Raytracer::trace` **after** `create_knull_grid` (`src/raytracer.cpp:345`).
- For each fine cell, find its coarse block, read `k_ext_null`, and write:
  - **hot pair:** `f_no_abs` (in the `8826bde` form, i.e. built from `k_abs/k_ext_null`,
    not `(k_ext_null - k_abs)/k_ext_null`) and `f_scat`;
  - **cold pair:** `p_cld = k_sca_cld/k_sca_tot` and `asy_cld`.
- Store the hot pair as one array of a 2-wide struct and the cold pair as another, so the
  common path issues one load of 8 bytes.
- Clamp `f_scat` to exactly `1` when the cell is its own block's maximum and conservative,
  so the rounding-short case from §3 point 3 cannot bias a long scattering history. Verify
  against the pre-`8826bde` behaviour, not just against the tests.
- Handle the lumped top cell from `bundle_optics_tod` in the same pass.
- In the kernel, replace the arithmetic block with two loads plus `rng() < f_scat` and
  `rng() < p_cld`.
- Decide at implementation time whether the pre-pass writes new `Scratch` arrays or
  overwrites `k_ext`/`scat` in place. New arrays are clearer and let the raw coefficients
  stay available for debugging; in-place saves memory. **Open question — ask before
  choosing.**

### Step 3 — `fmod` and the reciprocals *(thesis 4.1.3, 4.1.8)*

Small, local, independent of steps 1–2.

- Replace `Kokkos::fmod` at `raytracer_kernels.h:378-384` with a subtract/add: a photon
  crosses at most one coarse block per step, so a single conditional correction suffices.
- Put `1/grid_d` and `1/kn_grid_d` in `Scene` and have `coord_to_index` multiply.
  **Note:** `static_cast<int>` already emits a round-toward-zero convert, so the thesis'
  `__float2int_rz` buys nothing — only the division is worth attacking. And it must be
  done portably: no `__fdividef` in physics code (convention 3). `02b6741` already enabled
  fast math for this kernel, so measure before and after — part of this may already be
  had.
- Optionally simplify the `scatter()` frame construction to drop one of the two vectors.

### Step 4 — Work stealing *(thesis 4.1.2, partial)*

Structural, so it goes last among the algorithmic items.

- Move `launch_photons` from `Kokkos::RangePolicy` to a `TeamPolicy`.
- Replace the static `photons_per_thread` / `photons_extra` split with a team-scoped
  atomic counter incremented on each photon launch.
- Keep the Sobol offset correct: today `qrng_offset` is derived from the thread's static
  slice of the sequence (`src/raytracer.cpp:192-205`). With a counter, the offset must
  come from the counter value instead, so that the g-point's photons still cover a
  contiguous, non-overlapping range of the sequence. **This is the part to get right** —
  the Sobol stream, not the counter, is the constraint.

### Step 5 — Mixed precision *(thesis 4.1.9)*

Only after step 2, since step 2 decides what is stored.

- Store the hot pair, and `k_null`, as fp16.
- Needs a storage-type split in `include/types.h` (compute type stays `TF`), which is the
  one place backend differences are allowed to live.
- Validate against the Fortran reference, not only against the tests.

---

## 5. Verification

- `cmake --build build && pytest tests` after every step.
- The ray tracer is stochastic: a step that changes the random stream (step 4 certainly,
  step 2 possibly) changes results within noise, not bit-for-bit. Compare converged
  statistics, and against `rte-rrtmgp/` as the correctness oracle, not against the previous
  build's exact numbers.
- Steps 1 and 3 should be bit-for-bit identical apart from float summation order in the
  accumulator — the accumulator changes the *order* of the atomic adds, so small
  differences there are expected and benign.
- Commit at the end of each step, once it builds and the tests pass.

---

## 6. Open questions

1. **Pre-pass output:** new `Scratch` arrays, or overwrite `k_ext`/`scat` in place?
2. **`f_scat` exactness:** is the clamp in step 2 acceptable, or should the collision test
   keep the product form of `8826bde` and pre-compute only `f_no_abs` and the scatter
   thresholds? The latter is safer and gives up perhaps a third of the item's benefit.
3. **Step 4 and Sobol:** confirm the intended mapping from the team counter to
   `qrng_offset` before implementing, since it determines reproducibility.
