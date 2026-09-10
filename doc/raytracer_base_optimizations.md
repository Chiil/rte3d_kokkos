# Ray tracer base optimizations

Plan for porting the *base optimizations* of Ivor Brouwer's MSc thesis, **"GPU Thread
Data Remapping In Monte Carlo Short-wave Ray-tracing"** (LIACS, Leiden University,
26 August 2026; supervisors Ben van Werkhoven and Stijn Heldens), to the Kokkos ray
tracer in this repository — **and what measuring it actually showed**.

Written first against commit `8826bde` as a plan. Rewritten at `6e1395a`, once every
item had been built or priced on the hardware. The measurements are the point of the
file now; the thesis is the reason the questions were asked, not the authority on the
answers.

This file is self-contained: everything needed from the thesis is reproduced below.

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

**So: §4.1 only, never §4.2–4.7.** The remapping machinery is warp-intrinsic-heavy, which
collides with our convention 3, "one frontend for CPU and GPU", and did not win.

One caution the numbers below will keep repeating: the thesis' 2.81×–4.16× is over *its
own* original, on different hardware. Our starting point was not that code, and the
ordering of the wins is not the same. Ported items that the thesis calls strong came out
worth nothing here, and the largest win we found is not in the thesis at all.

---

## 2. How the measurements were made

Repeat any claim below with these; without them a number in this file means nothing.

**GPU** — NVIDIA RTX A4500 (Ampere, sm_86), single precision:

```
cmake -B build_cuda -DSYST=ubuntu_cuda -DUSEGPU=1 -DUSESP=1
export PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/25.3/compilers/bin:$PATH
```

**CPU** — `-DSYST=ubuntu_22lts_gcc`, double precision, 24 OpenMP threads.

**Two cases**, both untracked, both real fields:

| | cells | g-points | columns | photons/pixel |
|---|---|---|---|---|
| `cases/user/rcemip` | 256×256×65 | 224 | 65536 | 256 |
| `cases/les_cloudfield` | 128×128×201 | 112 | 16384 | 256 |

`cases/les_cloudfield` is rte-rrtmgp-cpp's own `les_cloudfield` — a RICO cumulus field at
20 m, cloud in 0.8 percent of cells — pointed at that tree's `test_input.nc`. It exists so
that the two codes can be compared directly; see §6.

```
python cases/user/run_case.py cases/les_cloudfield/les_cloudfield
python cases/user/run_case.py cases/user/rcemip --no-longwave --raytracing --no-plane-parallel
```

**Read the timer correctly.** `run_case.py`'s `sw ray tracer` line is `repeats=1,
warmup=0` — a **single shot**, not a best-of-three like the other solvers. Run-to-run
spread is 1–3 percent, so anything under about 3 percent needs repeating before it is
believed. Every number below that matters was taken back to back against a freshly
rebuilt comparison.

Resource use per build, which explains more than the timings do:

```
cuobjdump -res-usage build_cuda/src/CMakeFiles/rte3d_core.dir/raytracer.cpp.o \
  | grep -A1 launch_photonsILb0
```

---

## 3. Where the time goes — and it is two different places

```
ncu --kernel-name-base demangled --kernel-name regex:launch_photons \
    --launch-count 1 --launch-skip 3 --metrics <...> \
    python cases/user/run_case.py <case>
```

**The two cases are in different regimes, and a profile of one says nothing about the
other.** This is the single most important thing in this file; the first version of it was
written from the LES field alone and drew the wrong general conclusion.

At production settings, 256 photons per pixel, on `6e1395a`:

| | LES cumulus | RCEMIP |
|---|---|---|
| L1 hit rate | **93.0%** | **43.9%** |
| L2 hit rate | 99.0% | 49.6% |
| L1/TEX throughput | **98.5%** | 28.6% |
| DRAM throughput | 3.4% | **35.8%** |
| LG throttle, cycles per issue | **17.0** | 0.30 |

**LES cumulus is L1-queue bound.** 20 m cells mean a photon's free path spans few cells, so
consecutive collisions keep hitting the same lines: the optics stay resident, DRAM idles,
and what costs is the *number of memory instructions issued* — 17 of the 24.6 cycles
between issues are spent waiting for the load/store queue to drain.

**RCEMIP is DRAM-latency bound.** 100 m cells over a deep domain scatter the photons, the
working set stops fitting, and the L1 hit rate halves. DRAM sits at 36 percent of peak, so
it is not bandwidth-saturated — it is the *latency* of those misses, with only 36 resident
warps to hide it.

Two consequences worth internalizing:

- An item that removes memory *instructions* pays on the LES field. An item that removes
  memory *traffic* pays on RCEMIP. `6e1395a` does both, which is why it is the only item
  that won on both cases.
- **Photon count does not change the regime; the scene does.** The LES field profiles the
  same at 32 and 256 photons per pixel (LG throttle 18.2 and 17.0). Do not conclude from a
  cheap low-photon profile — but do not assume the difference is the photon count either.

Also note the synthetic scene in the throwaway `bench_rt.py` harness lands in the RCEMIP
regime (L1 hit 31%), not the LES one. It is not a substitute for either case.

Two secondary limits, on both cases, neither yet touched: occupancy capped at 75 percent
by registers, and warp efficiency at 83 percent — the divergence the thesis' remapping half
chased and failed to profit from.

---

## 4. The nine items, measured

Baseline is `d99169b`. Timings are the `sw ray tracer` line; percentages are against the
line above.

| # | thesis §4.1 item | outcome |
|---|---|---|
| 4.1.1 | Removal of the Sobol sequence | **Priced and rejected** — costs 1.42× photons |
| 4.1.2 | Reset locality, work stealing | Not attempted; see §5 |
| 4.1.3 | Register / complexity reductions | **Done** (`dadcda3`) — CPU 1.56×, GPU ~0 |
| 4.1.4 | Simplified scatter optics | Folded into §5 step 1, not yet done |
| 4.1.5 | Pre-calculated cell absorption/scattering | **Partly superseded** by `6e1395a` |
| 4.1.6 | Improved caching (last fine cell) | **Weak** — 6% hit rate on RCEMIP |
| 4.1.7 | Accumulation of atmospheric contributions | **Done** (`2744af2`) — 11% RCEMIP, 0% LES |
| 4.1.8 | Float-to-int and fast approximation | **Done** (`dadcda3`) — 5.5% CPU, ~0 GPU |
| 4.1.9 | Mixed precision (fp16 storage) | Not attempted; caveats in §5 |
| — | **One sector per collision, not two** *(not in the thesis)* | **Done** (`6e1395a`) — **22.9% RCEMIP** |

### The running totals

| commit | RCEMIP | LES (112 gpt) | CPU 64³ |
|---|---|---|---|
| `d99169b` baseline | 48729.7 ms | 2838.1 ms | 1234.2 ms |
| `dadcda3` fmod + reciprocal | 48655.1 ms | 2778.0 ms | 750.9 ms |
| `2744af2` accumulate deposits | 43363.9 ms | 2777.7 ms | 733.5 ms |
| `6e1395a` one wide optics load | **33446.5 ms** | **2660.9 ms** | 735.2 ms |
| | **1.46×** | **1.07×** | **1.68×** |

The CPU and GPU columns disagree about which item mattered, completely. Anything decided
from a CPU profile of this kernel will be decided wrong.

### 4.1.3 and 4.1.8 — `fmod` and the reciprocals *(done, `dadcda3`)*

The cyclic boundary used `Kokkos::fmod`; a photon crosses at most one block per step, so
the wrap is one subtraction. `coord_to_index` divided by the cell size; it now multiplies
by a reciprocal carried in `Scene`.

On the **CPU this is the whole story**: `fmod` alone was 1234.2 → 793.5 ms (1.56×), a libm
call gcc will not inline sitting on the block-transition path, and the reciprocal a further
5.5 percent (776.7 → 733.7 ms with everything else equal). On the **GPU it is worth
nothing** — 0.2 percent — because nvcc inlines the modulo and `-use_fast_math` had already
taken the divide. It stays in: free on the GPU, large on the CPU.

Neither is bit-for-bit; both move where the rounding falls. The thesis' `__float2int_rz`
buys nothing for us — `static_cast<int>` already emits a round-toward-zero convert — and
`__fdividef` is out under convention 3 anyway.

The optional `scatter()` frame simplification was not done. It changes the frame's
numerics and its argument is register pressure, which is a second-order limit here.

### 4.1.6 and 4.1.7 — caching and accumulation *(done, `2744af2`)*

Hold the last fine cell, the last photon kind, its optical properties and an accumulator
in registers; add into the accumulator and flush with one atomic when the cell or the kind
changes. A reset needs no flush of its own — the key is the cell and the kind, not the
photon, so a deposit left standing can only be added to by a photon that would have
written to the same place.

Worth **11.0 percent on RCEMIP and nothing at all on the LES field**. Instrumenting the
kernel says why:

| | same fine cell as previous collision | atomics issued |
|---|---|---|
| RCEMIP | 6.0% | 82.6% of collisions |
| LES cumulus | 29.7% | 54.9% of collisions |

Read that carefully, because it inverts the thesis' reasoning. On the LES field the
accumulator works *four times better* — it removes 45 percent of the atomics — and buys
**zero**. On RCEMIP it removes 17 percent and buys 11. So the atomics were never the
limit; §3 says the limit is the LG queue, and an atomic is one instruction in it like any
other.

Note also where the RCEMIP win really comes from: of the 17.4 points of atomics removed,
only 6.0 are same-cell merges. The other ~11 are **cells that absorb nothing** — `k_abs`
comes out exactly zero in single precision, so `flush_absorbed` issues no atomic at all.
That case is not in the thesis. It is most of our gain from the item.

The cache costs **8 registers** (48 → 56), which drops theoretical occupancy from about 42
resident warps to 36. On the LES field that cancels the item exactly.

**A trim that did not work.** Dropping the optics cache and keeping only the accumulator
recovered **no registers at all** — still 56, they come from the accumulator state itself —
and was 1.7 percent *slower* on RCEMIP. The four cached coefficients ride along free on
registers already spent. Do not retry this.

### 4.1.1 — removing the Sobol sequence *(priced, rejected)*

`Rand::Qrng_2d` holds two cuRAND scrambled-Sobol states, and they are **280 bytes of
per-thread local memory**: stack is 328 B with Sobol and 48 B without, registers 56 and 50.
That looks like the largest structural cost in the kernel, so it was worth pricing even
though the doc had ruled it out.

A throwaway build replacing the quasi-random launch with a plain pseudo-random draw, which
also deletes `reset_photon`'s rejection loop:

| | with Sobol | without |
|---|---|---|
| RCEMIP | 43363.9 ms | 40603.7 ms (−6.4%) |
| LES | 2774.3 ms | 2697.6 ms (−2.8%) |

That is the **ceiling** for the whole item. Then what it costs, against a converged
8192-photon reference on the LES field:

| launch | sfc_dir rms | sfc_dif rms | tod_up rms |
|---|---|---|---|
| Sobol | **1.68%** | 4.06% | 3.99% |
| pseudo-random | 2.01% | 4.08% | 4.01% |

Both unbiased to better than 0.07 percent. Sobol buys nothing on the diffuse or
top-of-domain fluxes — those photons wander far from where they launched and the launch
stratification washes out — but on the **direct beam** it cuts per-pixel rms by 16 percent.
Noise goes as 1/√N, so matching that without Sobol costs **1.42× more photons**: 42 percent
more work to buy back 3–6 percent. A clear loss.

The thesis reports "no significant effect on output" for this change. On our
direct-beam surface flux there is one, and that is the flux the cases care most about.

**Keep Sobol.** The local memory it costs is only ~2 percent of L1 requests (5.1M of
272M, measured on the LES field), which is exactly why removing it gains so little even
where the L1 queue is the limit.

### The item that was not in the thesis — one sector per collision *(done, `6e1395a`)*

The SASS at `2744af2` had **440 `LDG.E` and not one wide load**. `k_ext` lived in its own
array beside a twelve-byte `Optics_scat`, so every collision issued **four separate 32-bit
loads** for four contiguous numbers of one cell — the exact pattern §3 says is the limit.

The two arrays became one array of a sixteen-byte aligned `Optics_cell`. The SASS now
carries one `LDG.E.128` in each of the two `trace_photons` instantiations, which is the
collision's fetch.

**RCEMIP 43400.9 → 33446.5 ms (22.9%), LES 2774.3 → 2660.9 ms (4.1%), CPU neutral.** The
single largest win of the exercise, from a change that touches three files and alters no
arithmetic.

It is numerically inert. The largest difference against the previous commit is 1.8e-4 on a
surface flux of 700 — and two runs of the *same* build differ by exactly the same 1.8e-4,
because the deposits are atomics and the GPU does not order them twice the same way.

**Why RCEMIP gains five times what the LES field gains**, measured at 256 photons per
pixel on the real case, one g-point:

| | pre-merge | merged | |
|---|---|---|---|
| kernel duration | 102.49 ms | 76.06 ms | −25.8% |
| L1 sectors, global loads | 1190.8M | 766.7M | **−35.6%** |
| DRAM bytes read | 15.81 GB | 10.04 GB | **−36.5%** |
| L2 hit rate | 44.8% | 49.6% | +4.9 pts |

One collision used to touch **two** sectors, one in each array; it now touches **one**. The
sector count and the DRAM traffic fall together, one for one, and the runtime follows.
RCEMIP is where that matters because RCEMIP is the case that misses in L1 (§3); the LES
field already had its optics resident, so it only collected the fewer-instructions half of
the change, and got 4 percent.

So this item is not really "one instruction instead of four" — that is what it looks like
in the SASS. It is **one cache sector instead of two**, which is why it is worth five times
more where the cache is under pressure.

---

## 5. What is left

### Step 1 — the hot/cold split *(what remains of thesis 4.1.5 and 4.1.4)*

`6e1395a` took the load-width half of 4.1.5 without needing a pre-pass. What remains is
narrowing the common path: a null collision needs only `f_no_abs` and `f_scat`, so a
pre-pass writing a hot pair and a cold pair turns one 16-byte load into one 8-byte load.
With no aerosols the 4.1.4 thresholds collapse to a single `p_cld = k_sca_cld/k_sca_tot`.

**Blocked, and the LES field is why.** The pre-computed values depend on `k_ext_null`,
which is a property of the *coarse block*. `Grid::make` does not enforce `nx % kn_x == 0`,
and this case does not have it: 128/48 and 200/32. Where a coarse face cuts through a fine
cell, that cell has no single `k_ext_null` and the table is not well defined. Settle this
first — enforce divisibility in `Grid::make`, or keep the in-kernel arithmetic as a
fallback — before writing any pre-pass.

Two smaller notes for whoever writes it. The pre-pass must derive the block index the way
`coord_to_index` does on the kn grid, **not** the way `create_knull_grid`'s overlapping
`i0..i1` loop does. And the `f_scat` clamp the original plan called for is **unnecessary**:
in a conservative cell that is its own block's maximum, `k_abs` is exactly zero and
`f_scat = k_sca_tot/k_ext_null` is `x/x`, which IEEE gives as exactly 1.0. The `8826bde`
concern was a *systematic* multiplicative shortfall in `weight`, which applies to
`f_no_abs`; an `f_scat` one ulp short only causes a spurious null collision with
probability ~1e-7, which costs work, not accuracy.

### Step 2 — mixed precision *(thesis 4.1.9)*

**This is now the strongest item left, and §3 says why.** RCEMIP is bound by DRAM traffic,
and `6e1395a` bought 26 percent on that case purely by halving the sectors a collision
touches. Halving the stored width does the same thing again: an `Optics_cell` in fp16 is
8 bytes, so two cells share one 32-byte sector where one cell needs one now. It also cuts
the instruction count on the LES field's L1 queue. It is the one remaining change that
pays in both regimes.

Two corrections to the original plan, both of which would otherwise produce wrong answers:

- Store **`f_abs`, not `f_no_abs`**. fp16 spacing at 1.0 is 9.8e-4, and an optically thin
  cell has `f_abs` of order 1e-4 — storing a number just below 1 quantizes the absorption
  to nothing and `weight *= f_no_abs` accumulates the error over hundreds of collisions.
  Near zero, fp16 has ample relative precision. Reconstruct `1 - f_abs` in the kernel.
- Round **`k_null` toward +∞**. Round-to-nearest can put it below the true block maximum,
  which breaks the majorant: `k_abs > k_ext_null`, negative `f_no_abs`, and a biased result
  rather than a noisy one.

### Step 3 — work stealing *(thesis 4.1.2, partial)*

Still unmeasured, and still the item I would expect least from: with hundreds of photons
per thread the per-thread path length concentrates as 1/√N, so the tail the thesis'
block-wide atomic counter removes is small for us. Instrument max-versus-mean steps per
thread before committing to a `TeamPolicy` rewrite. If it is done, the constraint is the
Sobol stream, not the counter — `qrng_offset` must still give each photon a contiguous,
non-overlapping slice.

### Dead

- **§4.1.1, Sobol.** Priced at 3–6 percent, costs 42 percent more photons. Closed.
- **§4.1.6 as a cache.** 6 percent hit rate on RCEMIP. The registers it costs are already
  spent on the accumulator, so it stays in the code, but there is nothing more here.
- **Trimming the accumulator's registers.** Measured, slower. See §4.
- **§4.2–4.7, remapping.** Lost in the thesis, and collides with convention 3.

---

## 6. Against rte-rrtmgp-cpp

Same field, same 112 g-points, same sun, same GPU:

| | best of 3 |
|---|---|
| `rte-rrtmgp-cpp` `test_rte_rrtmgp_rt` | 3352.1 ms |
| rte3d `d99169b` | 2838.1 ms |
| rte3d `6e1395a` | ~2660.9 ms |

The port was already ~15 percent ahead of the original before this exercise and is ~26
percent ahead after — while carrying the single-frontend constraint, with no separate
`src_cuda_rt` tree.

Fluxes agree to 0.001–0.07 percent on surface direct, surface diffuse, surface up and
top-of-domain up, and the absorption profile agrees to 0.00 percent in the mean over the
200 shared cells. rte3d reports one cell more: index 200 holds the layers above the box,
lumped.

**Set `sza` and `azi` explicitly in the reference's `test.ini`** — `sza = 30`,
`azi = 108.8975` for this input. Its documented fallback, "mu0 from input file is used if
sza < 0", does the opposite in the build tested: the conditional is inverted
(`test_rte_rrtmgp_rt.cu:464`, `:467`, `:1224`, `:1227`, and `test_rte_rrtmgp_bw.cu:584`,
`:587` — all `if (input_sza < 0)` where `>= 0` is meant), so the file's `mu0` is read and
then overwritten with `cos(input_sza)` of the negative value. At the default `sza = -1`
that is `cos(-1°) = 0.9998`, a sun overhead. The whole solve then transmits 15 percent more
direct beam and runs 6 percent faster, silently. This cost a full investigation before a
single-g-point check pinned it; do not repeat it.

---

## 7. Verification

- `cmake --build build && pytest tests`, and the same for `build_cuda`, after every step.
- The tracer is stochastic and the deposits are atomics, so nothing on the GPU is
  bit-for-bit, not even a build against itself. Establish the run-to-run scatter of one
  build first, then compare against it — that is how `6e1395a` was shown to be inert.
- The pytest suite does not catch a wrong sun or wrong optics on a real field. The LES
  case against rte-rrtmgp-cpp (§6) is the only end-to-end check we have; use it whenever
  the physics could have moved.
