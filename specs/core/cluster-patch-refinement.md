# Cluster-Patch Refinement: Production Implementation

_Status: **implemented** — steps 1–5 are shipped, and the derived-pairs
helper plus the `sfm match --cluster` output migration (§1) shipped
2026-07-10 (see each section's dated status block); nothing in this spec
remains open. The implementation spec for
the operation
designed in [cluster-patches.md](cluster-patches.md); read that first for
motivation, the format sections, and the experimental calibration
(`reports/2026-07-09-exp-pairwise-sift-warp.md`). This document is written so
an independent implementer can build the production version without the
prototype: it fixes the Rust kernel's inputs/outputs, the algorithm, the PyO3
surface, the reuse of existing patch machinery, and the AVX2 plan. The
prototype (`scripts/exp_cluster_patch_clusters.py`) remains the behavioral
reference._

## Overview of the pieces

1. `matches-format`: `clusters/` + `cluster_patches/` sections, version 3,
   pairs-or-clusters backbone — **shipped**, together with the section
   threading through the `io/matches.rs` read/write bindings.
2. `sfmtool-core`: a new `patch::cluster_refine` kernel — per cluster, pick a
   reference member, refine a Gaussian-windowed-ZNCC affine warp to every
   other member (shift → similarity → affine Nelder-Mead cascade seeded from
   the SIFT affine shapes), vet, and emit member-parallel arrays.
3. `sfmtool-py`: a `refine_cluster_patches` pyfunction in
   `matching/cluster.rs` (beside `background_floor_clusters` /
   `clusters_to_pair_matches`).
4. Python: the `sfm cluster-patches` command, and a single derived-pairs
   helper the existing pairwise consumers migrate to.

## 1. `matches-format` changes

> _Status (2026-07-09): **Shipped** — `feat(matches): clusters +
> cluster_patches sections, version-3 backbone rule`. The normative
> documentation now lives in `specs/formats/matches-file-format.md` (file
> tree, section details, constraints, version 2 → 3 migration); this section
> records only the as-built surface and what remains open._

As built, in `crates/matches-format` (`types.rs` / `read.rs` / `write.rs` /
`verify.rs`):

- `MatchesData` carries `image_pairs: Option<PairsData>` (the four former
  top-level pair fields, grouped), `clusters: Option<ClustersData>`,
  `cluster_patches: Option<ClusterPatchData>`, and the existing
  `two_view_geometries` — exactly one backbone `Some`, `cluster_patches`
  requires `clusters`, TVGs require `image_pairs`.
- The member-status discriminants live in
  `matches_format::ClusterMemberStatus` (0 reference … 5 not_evaluated;
  6 rejected_unlocalizable added 2026-07-10) with
  the sentinel `CLUSTER_REFERENCE_UNREFINABLE` (`u32::MAX`) for a cluster
  with no usable reference. **The core kernel's `MemberStatus` (§2) must
  emit these same discriminants** — the binding passes the u8 array through
  untouched (`sfmtool-core` does not depend on `matches-format`).
- `MATCHES_FORMAT_VERSION = 3`; readers accept 1–3, version ≤ 2 files are
  pairwise-only and load unchanged, pairwise byte streams and hashes are
  preserved.
- Two deliberate deviations from the original sketch, kept because they make
  failures legible: `verify()` runs the backbone/flag/entry-consistency gate
  up front (structurally incoherent files report structured errors instead
  of missing-zip-entry I/O errors), and the writer *validates* the supplied
  `has_*` flags and counts against the sections rather than silently
  deriving them.
- The `io/matches.rs` bindings thread both sections through
  `read_matches` / `write_matches` as flat dict keys plus always-present
  `has_clusters` / `has_cluster_patches` flags; pairwise dict output is
  unchanged. `sfmr-colmap`'s DB export takes the grouped `PairsData` and
  rejects cluster-bearing input with a clear error until the derived-pairs
  migration below lands.

> _Status (2026-07-10): **Shipped** — `feat(match): persist cluster .matches
> as the primary artifact + derived-pairs migration`. As specified:
> `pairs_from_matches` lives in `src/sfmtool/feature_match/_pairs.py`
> (exported from the package `__init__`), and `sfm match --cluster` now
> writes the clusters-bearing file (default
> `matches/<verified stem>-clusters.matches`, `--clusters-output` override)
> before verification, alongside the unchanged verified pairwise+TVG `-o`
> output. Migration notes: `_db_populate.py`'s
> `_compute_descriptor_distances` was not reused for the cluster path — the
> helper reads each image's descriptors capped at the file's
> `feature_counts` and lets `clusters_to_pair_matches` compute the same L2
> distances in Rust in one pass. `_densify.py` needed no change: it matches
> image pairs live and never reads `.matches` pair keys. Consumers that
> require TVGs are unaffected (cluster files report
> `has_two_view_geometries: false` and those paths already degrade
> gracefully); `sfmr-colmap`'s Rust DB export still rejects grouped cluster
> input with its clear error, but the Python `to-colmap-db`/solve path goes
> through the helper and accepts cluster files._

**Still open — derived pairs.** `matches-format` does not depend on `sfmtool-core`, so the
expansion stays where it is (`sfmtool-core`, bound as
`_sfmtool.matching.clusters_to_pair_matches`). The single Python entry point
is a new helper in `sfmtool.feature_match`:

```python
def pairs_from_matches(data: dict, sift_paths: list[Path] | None = None) -> dict
```

returning the four pair arrays: stored pairs verbatim when present, else the
cluster expansion; `match_descriptor_distances` recomputed from the `.sift`
descriptors when `sift_paths` is given, else `NaN`s. Migration sites (all
currently index the pair keys of `read_matches` output directly):
`src/sfmtool/feature_match/_run.py`, `_db_populate.py`,
`src/sfmtool/analyze/summary.py`, `src/sfmtool/_densify.py`,
`src/sfmtool/colmap/db_setup.py`. These MUST be migrated in the same change
that makes `sfm match --cluster` write cluster-bearing files.

## 2. The Rust kernel: `sfmtool-core::patch::cluster_refine`

> _Status (2026-07-09): **Shipped** — `feat(patch): cluster-patch refinement
> kernel + binding + CLI`. As specified, with these deviations:_
>
> - _**Window sigma units.** The default below is written
>   `GaussianDisk { sigma: 15.0 / 4.0 }`, but `PatchWindow`'s sigma is in
>   normalized patch coordinates (the grid spans `[-1, 1]²`), where the
>   prototype's `radius / 2` keypoint-frame width is **0.5** — the `15/4`
>   figure is the same width expressed in grid px. The shipped default is
>   `GaussianDisk { sigma: 0.5 }`._
> - _**Reuse map.** `AffineCoreMap` was promoted (with a `from_coeffs`
>   constructor) and is the kernel's map type, but `sample_support_affine`
>   was **not** shared: its contract (border-gated maps, no validity
>   reporting, `u8` re-rounding for remap parity) does not fit a sampler
>   that must express out-of-frame and keep continuous values. The template
>   sampler and the fused member kernel live in `cluster_refine/kernels.rs`
>   with the same `bilinear_geometry` convention.
>   `score_raw_against_reference` was likewise not promoted — the fused
>   sample+reduce loop realizes the identical algebra (mean-removed dot over
>   the member's windowed norm, flat member channel → 0 contribution,
>   averaged over the reference's channel count) in one pass._
> - _**`NotEvaluated` trigger.** "Seed support out of frame" is decided at
>   the seed evaluation (`t = 0, D = 0`); once the cascade runs,
>   out-of-frame proposals score `+1.0` per the all-in-frame rule and the
>   member still vets on its final ZNCC/shift._
> - _**Level selection** is evaluated per objective evaluation (the map's
>   linear part changes as `D` evolves), not once per member._

New module `crates/sfmtool-core/src/patch/cluster_refine/` with `mod.rs`,
`params.rs`, `kernels.rs`, `tests.rs` (mirroring `keypoint_localize/`).

### API

```rust
#[repr(u8)]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
// Discriminants MUST match matches_format::ClusterMemberStatus (shipped):
// the binding passes the u8 array straight into the cluster_patches/ section.
pub enum MemberStatus {
    Reference = 0,
    Kept = 1,
    RejectedLowZncc = 2,
    RejectedShift = 3,
    DuplicateImage = 4,   // outscored by another kept member in the same image,
                          // or shares the reference's image
    NotEvaluated = 5,     // degenerate shape, template/seed support out of
                          // frame, or the cluster itself was unrefinable
    RejectedUnlocalizable = 6, // the member's own patch scored a keypoint
                          // position uncertainty above
                          // max_keypoint_uncertainty (excluded before
                          // reference selection and refinement; added
                          // 2026-07-10)
}

#[derive(Clone, Debug)]
pub struct ClusterRefineParams {
    pub radius: f64,          // template half-width, keypoint-frame units
    pub resolution: u32,      // support samples per axis
    pub window: PatchWindow,  // reuse normal_refine::params::PatchWindow
    pub min_zncc: f64,
    pub max_shift_px: f64,
    pub max_keypoint_uncertainty: f64, // localizability gate (σ_pos,
                              // template-grid px; 0 disables) — added
                              // 2026-07-10
    pub max_iters: u32,       // Nelder-Mead iterations per stage
    pub convergence: f64,     // simplex value-spread stop, affine stage
    pub intermediate_convergence: f64, // …the shift/sim stages (looser) —
                              // added 2026-07-11, see "Performance"
    pub stall_iters: u32,     // NM stall exit: iterations without progress
    pub stall_tol: f64,       // …and what counts as progress (ZNCC units)
}
impl Default for ClusterRefineParams {
    // radius 4.0, resolution 25,
    // window: PatchWindow::GaussianDisk { sigma: 15.0 / 4.0 }  (= resolution/4,
    //   the prototype's sigma = radius/2 in keypoint-frame units),
    // min_zncc 0.85, max_shift_px 3.0, max_keypoint_uncertainty 0.35,
    // max_iters 120, convergence 1e-5,
    // intermediate_convergence 1e-4, stall_iters 20, stall_tol 1e-4
}

/// One image's SIFT feature geometry (borrowed views of the `.sift` arrays).
pub struct FeatureGeometry<'a> {
    pub positions_xy: ArrayView2<'a, f32>,   // (N, 2)
    pub affine_shapes: ArrayView3<'a, f32>,  // (N, 2, 2)
}

pub struct ClusterRefineResult {
    pub reference_members: Vec<u32>,  // (C,) global member index or u32::MAX
    pub member_status: Vec<MemberStatus>, // (M,)
    pub member_affines: Array3<f64>,  // (M, 2, 3): (2x2 A | absolute position p)
    pub member_zncc: Vec<f32>,        // (M,) NaN if not evaluated
    pub member_shift_px: Vec<f32>,    // (M,)
}

pub fn refine_cluster_patches(
    pyramids: &[ImageU8Pyramid],
    features: &[FeatureGeometry<'_>],   // len == pyramids.len()
    cluster_starts: &[u32],
    member_images: &[u32],
    member_features: &[u32],
    params: &ClusterRefineParams,
    progress: Option<&AtomicUsize>,     // one tick per finished cluster
) -> ClusterRefineResult
```

Result arrays map 1:1 onto the `cluster_patches/` section. The kernel is
pure: no I/O, no `.sift` reads — the caller supplies decoded pyramids and
feature geometry (mirrors how `ProjectedImage` isolates the other kernels
from decoding, `normal_refine/params.rs:20`).

> _Addition (2026-07-10): after refinement, the sibling kernel
> `warp_consistency_residuals` (`cluster_refine/consistency.rs`) computes a
> per-member warp-consistency residual from a joint weak-perspective
> factorization of all cluster warps — a reconstruction-free contamination
> signal stored as `cluster_patches/member_consistency_residual` (the
> binding computes it inside the same `refine_cluster_patches` call; the
> result dict and the format gained the matching member-parallel array).
> Design, evidence, and references in
> [cluster-warp-consistency.md](cluster-warp-consistency.md)._

### Algorithm (per cluster)

Parallelism: `(0..cluster_count).into_par_iter()` over clusters, writing
disjoint member ranges — results are deterministic under any thread
schedule. Per-cluster scratch (sample buffers, simplex) lives inside the
closure, thread-local by construction (the `SearchScratch` convention,
`keypoint_localize/search.rs:39`).

1. **Validate members.** A member is usable when its feature index is in
   bounds and `|det A| ≥ 1e-9`. Additionally (added 2026-07-10), when
   `max_keypoint_uncertainty > 0`, each remaining member must pass the
   **localizability gate**: sample the member's own full `R×R` grid at its
   SIFT geometry (identity warp, mip-selected level, same bilinear
   convention as the template — but every grid pixel, since the scorer's
   gradients cover the full grid) and score it with
   [`patch_localizability`](patch-localizability.md) against the shared
   refinement window (`σ_noise = 3.0`, the global constant). A member whose
   `σ_pos` exceeds the threshold is `RejectedUnlocalizable` and excluded
   before reference selection — an unlocalizable patch (flat / straight
   edge) can neither anchor nor join the cluster. Samples outside the frame
   clamp to the nearest valid pixel (border replicate), so a border member
   is scored on its visible content rather than skipping the gate; only
   degenerate (non-finite) geometry skips it, and the template/seed frame
   gates downstream still apply.
   The threshold unit is **template-grid px** (default `0.35`, the same
   default value as `embed-patches`; the scored quantity differs slightly —
   the grid here is `resolution = 25` with the refinement window
   (`GaussianDisk σ = 0.5`) rather than the consensus 24 with the scorer's
   `σ = 0.6` window). Note the grid-px unit is **not**
   resolution-invariant in practice: on dino_dog_toy, moving the template
   from 15 to 31 samples per axis cut `RejectedUnlocalizable` from 1,913
   to 372 members at the same `0.35` threshold, so the gate weakens as
   resolution rises. A future update should re-express the threshold in a
   resolution-independent unit (keypoint-frame or source px) so the gate's
   physical meaning survives resolution changes. Fewer than 2 usable
   members → every member not already `RejectedUnlocalizable` is
   `NotEvaluated`, `reference_members[c] = u32::MAX`, done.
2. **Reference selection.** The usable member with the largest scale
   `√|det A|`; ties break to the lowest global member index (determinism).
   Reference policy is data, not format — smarter policies (template
   self-agreement, descriptor centrality) can replace this without format
   change (open question in the design spec).
3. **Template.** Build the canonical support grid: `resolution²` samples at
   keypoint-frame offsets `u_k ∈ [−radius, radius]` (pixel-center offsets,
   the `(k + 0.5)/n`-style grid of `warp_map_for` in the prototype). The
   reference's sample positions are `x = pos_ref + A_ref · u` — an affine
   support map, exactly the shape of `view_selection::affine_core_map`'s
   output. Sample every image channel with the bit-exact bilinear
   convention (`remap.rs:385` `bilinear_geometry`, x−0.5 pixel centers) into
   planar per-channel f32; **any sample outside the image → the reference
   candidate is unusable** (step 2 falls to the next-largest scale; all
   candidates unusable → cluster unrefinable). Z-normalize each channel with
   the sqrt-window fold (`normal_refine/znorm.rs`: `weighted_moments_pub` →
   mean / inv-norm → `znorm_write`), dropping flat channels
   (`FLAT_NORM_SQ_EPS`); all channels flat → unusable. The window weights
   come from `build_support(params.window, resolution)`
   (`normal_refine::support`).
4. **Pyramid level.** Per sampled image (template and each member), select
   the pyramid level `ℓ = clamp(floor(log2(s_min)), 0, L−1)` where `s_min`
   is the smaller singular value of the support map's linear part in that
   image (sample spacing in source pixels), and divide the map by `2^ℓ`
   before sampling — the standard mip rule, preventing the aliasing the
   level-0-only prototype tolerated. (Full anisotropic footprints,
   `remap_aniso_with_pyramid`, are a later refinement; bilinear-in-level
   matches the other kernels' `Sampler::Bilinear` default.)
5. **Per non-reference member** (in member order):
   - Same image as the reference → `DuplicateImage`, skip.
   - Seed `M₀ = A_mem · A_ref⁻¹` (f64), translation anchored at the
     detections: the warp is
     `W(x) = pos_mem + t + (I + D) · M₀ · (x − pos_ref)`.
   - **Objective** `f(t, D)`: sample the member image at `W` of the
     template's support positions (level-corrected, planar f32), compute the
     windowed ZNCC per surviving channel against the z-normalized template
     (weighted moments → normalize → dot, `score_raw_against_reference`
     flow, `view_selection.rs:538`), average channels, negate. Any support
     sample out of frame → score `+1.0` (worst) so the simplex retreats;
     the all-in-frame rule matches `normalized_stack`'s rejection semantics.
   - **Nelder-Mead cascade**, stages `shift` (θ = t, 2 params) →
     `similarity` (t, σ, φ; `D = e^σ R(φ) − I`, σ clamped to ±1.5) →
     `affine` (t, D, 6 params), each stage initialized from the previous
     (σ, φ promoted to `D` entries), simplex seeded at `θ₀ + scale_i·e_i`
     with scales 0.5 px for translations and 0.05 for σ/φ/D entries;
     standard coefficients (reflect 1, expand 2, contract 0.5, shrink 0.5),
     stop at `max_iters` or value spread < `convergence`. Fully
     deterministic. No perspective stage and no multi-view congealing pass
     (experimentally dead ends — see the design spec).
   - Record `member_zncc = −f(θ*)`, `member_shift_px = |t*|`, and the
     absolute affine `A = (I + D)M₀` with the member's refined absolute
     keypoint position `p = pos_mem + t` in the last column into
     `member_affines` (so `x_mem = A·(x − x_ref) + p` composes without the
     seed, and the translation stays recoverable as `t_abs = p − A·x_ref`
     with `x_ref` read from the reference row). Vet: `zncc < min_zncc` →
     `RejectedLowZncc`; else `shift > max_shift_px` → `RejectedShift`; else
     provisionally kept.
6. **One member per image.** Among provisionally-kept members sharing an
   image, keep the highest ZNCC (ties → lowest member index); the rest
   become `DuplicateImage`. Reference row gets `Reference` status, the
   identity affine with its own keypoint position in the last column
   (`A = I`, `p = x_ref` — identity | x_ref), `zncc = 1.0`, `shift = 0`.

### Reuse map

| need | existing symbol | action |
|---|---|---|
| pyramids | `ImageU8Pyramid::build/level` (`camera/remap.rs:282`) | use as-is |
| bilinear convention | `bilinear_geometry` (`remap.rs:385`), `sample_bilinear_u8` (`:338`) | use as-is |
| affine support sampling | `affine_core_map` / `sample_support_affine` (`patch/view_selection.rs:314/385`) | promote to `pub(in crate::patch)` (add a constructor from an explicit 2×3 instead of patch corners) and share |
| window weights | `build_support`, `PatchWindow::GaussianDisk` (`normal_refine/support.rs`, `params.rs:44`) | use as-is |
| z-normalization + wZNCC | `weighted_moments_pub`, `znorm_write`, flat-channel drop (`normal_refine/znorm.rs:218/295/103`) | use as-is (already `pub(in crate::patch)`) |
| score vs template | `score_raw_against_reference` (`view_selection.rs:538`) | promote to `pub(in crate::patch)` or extract into `znorm.rs` |
| result-struct shape | `KeypointLocalization` parallel arrays (`keypoint_localize/params.rs:128`) | mirror |
| rayon + progress | `localize_patch_cloud_keypoints` (`keypoint_localize.rs:874-912`) | mirror (par over clusters, `AtomicUsize` ticks) |
| planar centered f32 layout | `ContextTile` doc (`keypoint_localize.rs:76-100`) | follow for the template buffer |

Genuinely new code: the Nelder-Mead optimizer (small, self-contained — put
it in `cluster_refine/kernels.rs`; the existing subpixel solver is
translation-only ECC Gauss–Newton and does not generalize to the affine
stage without new Jacobian plumbing), the warp parameterization/cascade, and
the reference-selection / dedupe / status bookkeeping.

### AVX2

> _Status (2026-07-09): **Shipped.** The fused loop gathers from a
> per-(member, level) **centered planar-f32 tile** (`LevelTile`), converted
> once when the level is first touched and grown lazily to cover each
> evaluation's grid bounding box — the `ContextTile` render-once convention;
> centering keeps the f32 variance `S2 − S1²/W` cancellation-safe and the
> windowed ZNCC is shift-invariant, so nothing needs undoing. Because the
> tile is a subset of the image that always covers the evaluation's clipped
> footprint, the tile-bounds lane test **is** the in-frame test (out-of-tile
> tap ⇔ out-of-frame tap). Scalar path and AVX2 agree within 1e-4 (dual-path
> test); non-finite coordinates from degenerate warps convert to `i32::MIN`
> in the AVX2 mask and are guarded explicitly in the scalar path._

> _Revision (2026-07-11): **Fused-channel pair-load kernel** (bit-identical
> results, 1.93 → 1.15 µs per evaluation on dino_dog_toy — the channel
> fusion and the pair loads landed as one revision; see "Performance"). Two
> restructurings of the shipped kernel, both preserving each channel's
> accumulation order exactly:_
>
> - _**Channels fused into the lane loop.** The loop is now k-major with the
>   template channels unrolled inside (monomorphized over the active channel
>   count, ≤ 4), so the coordinate FMAs, in-frame mask, tap indices, and
>   fractional blend weights — all channel-invariant — are computed once per
>   8 support points instead of once per channel._
> - _**Pair loads instead of hardware gathers.** The two horizontal bilinear
>   taps of a lane are adjacent floats; one unaligned 64-bit scalar load
>   fetches both, and eight such loads + shuffle/deinterleave replace the
>   four `_mm256_i32gather_ps` per channel-group. Half the fetched elements,
>   and plain loads also sidestep the microcoded gather penalty on hybrid
>   (E-core) parts, where most rayon threads land._

Runtime dispatch per the house pattern:
`is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma")`,
`#[target_feature(enable = "avx2,fma")] unsafe fn`, scalar fallback written
with 4 independent accumulators so it autovectorizes (see
`weighted_moments_scalar`, `znorm.rs:238`).

- **Already vectorized, reused for free**: the weighted moments
  (`weighted_moments_avx2`, `znorm.rs:272`) and the z-normalize write
  (`znorm_write_avx2`, `:339`) — together they cover the normalize-and-dot
  half of every objective evaluation.
- **The one new hot loop**: sampling `resolution²` support points through an
  affine map (the cascade evaluates it ~500× per member: 4 bilinear taps +
  weight blend per point). Vectorize 8 points per iteration: the sample
  coordinates are affine in the grid index, so `x`/`y` vectors are computed
  by FMA from lane offsets; tap addresses via `_mm256_i32gather_ps` on the
  planar f32 level (precedent: `score_cell_one_channel_avx2`,
  `keypoint_localize/kernels.rs:374`); fractional blends via FMA. Fold the
  moments accumulation (`Σw·I`, `Σw·I²`, `Σtmpl·I`) into the same pass so
  each evaluation is one fused sample+reduce loop — this is the shape of
  `search_shift`'s three-map accumulation (`search.rs:289`), transplanted
  from integer shifts to gathered affine samples.
- Keep the in-frame test vectorized as a lane mask; any failing lane aborts
  the evaluation (all-in-frame rule), so the mask doubles as the early-out.
- u8→f32 channel conversion happens once per (member, level) when the level
  tile is first touched, not per evaluation, following the `ContextTile`
  render-once convention.

### Performance (2026-07-11 profiling pass)

`cluster_refine/prof.rs` carries the house opt-in phase timers
(`SFMTOOL_PROFILE=1`, the `keypoint_localize::prof` pattern): per-phase
thread-summed CPU time (gate sample/score, template build, member cascades,
tile builds, objective evaluations), plus event counters — members, gate
rejections, cascades, evaluations **per cascade stage**, tile (re)builds and
their pixel volume. Timers compile to one branch on a cached flag when the
variable is unset.

Baseline profile (dino_dog_toy: 85 images @ 2040×1536, 105,326 clusters,
373,194 members, i9-14900HX, 32 threads): kernel wall 6.07 s, 194 CPU-s.
`refine_member` was 96.5% of cluster CPU; inside it `eval_zncc` 70%
(70.4M calls, 1.93 µs — 265 per member: shift 36 / sim 82 / affine 147, the
affine stage crawling into the 120-iteration cap), the Nelder-Mead
transcription's per-iteration `Vec` churn ~27%, tile builds 2.3%. The
localizability gate and template builds are noise (≤3% combined).

Changes, in order, with their share of the win:

1. **Fused-channel pair-load AVX2 kernel** (above) — bit-identical,
   1.93 → 1.15 µs/evaluation.
2. **Allocation-free Nelder-Mead** — fixed `[f64; 6]` simplex buffers and a
   stable insertion sort replace the per-iteration `Vec` allocations and
   `sort_by`; bit-identical (same arithmetic, same tie order), −30 CPU-s.
3. **Cascade stopping tradeoffs** (the only result-changing change):
   `intermediate_convergence = 1e-4` for the shift/sim stages (they only
   seed the next stage), and a **stall exit** (`stall_iters = 20`,
   `stall_tol = 1e-4`): stop any stage when the best value has not improved
   by more than the tolerance for that many consecutive iterations. The
   reflect-heavy 6-dim affine crawl shrinks its value *spread* far more
   slowly than it stops making *progress*, so the spread test alone ran most
   members into the iteration cap long after the score stopped moving; the
   stall exit releases exactly those. Evaluations 265 → 213 per member.
   Sweep on dino_dog_toy: kept members +0.03%, mean kept ZNCC −0.0001,
   warp-consistency median 0.0677 → 0.0669 / p90 0.1993 → 0.1972 (slightly
   better), status flips 0.76% — and the synthetic warp-recovery suite stays
   green (a tighter `stall_tol = 2e-4` broke the scale-1.25/rot-20° case).

Result: kernel wall 6.07 → 3.23 s (1.9×), 194 → 103 CPU-s on the dataset
above. End-to-end `sfm cluster-patches` 10.8 → 7.8 s — the CLI now also
reads images and `.sift` files through a thread pool (the embed-patches
pattern) instead of serially.

Not pursued, recorded as candidates if this kernel needs another pass:
replacing the affine NM stage with a Gauss-Newton/ECC step on an analytic
windowed-ZNCC gradient (the cap-bound crawl is the remaining eval budget:
~60% of kernel CPU is still `eval_zncc`, mostly affine-stage), and
luminance-only refinement (3× fewer channel passes, but it changes matching
semantics — needs its own quality study).

## 3. PyO3 bindings

> _Status (2026-07-09): **Shipped** as specified: `refine_cluster_patches`
> in `matching/cluster.rs`, registered beside the two existing cluster
> functions; validation (parallel list lengths, per-image array shapes, CSR
> consistency, `member_images` range) raises `PyValueError` before the GIL
> is released; the kernel runs under `py.detach`. The pyramid build was
> factored as `py_patch_cloud::build_pyramids_from_image_list` (extraction +
> rayon build with a per-image `check` callback); the recon-coupled
> `build_pyramids_from_arrays` supplies the camera-dimension check and the
> cluster path passes a no-op. `window_sigma = None` resolves to the 0.5
> default (§2); the `convergence` knob is not exposed (fixed 1e-5), matching
> the signature below. Out-of-range `member_features` are left to the kernel
> (→ `not_evaluated`), matching §2 step 1._

In `crates/sfmtool-py/src/matching/cluster.rs`:

```rust
#[pyfunction]
#[pyo3(signature = (images, positions, affine_shapes,
                    cluster_starts, member_images, member_features, *,
                    radius = 4.0, resolution = 25,
                    window = "gaussian_disk", window_sigma = None,
                    min_zncc = 0.85, max_shift_px = 3.0,
                    max_keypoint_uncertainty = 0.35,
                    max_iters = 120, progress = None))]
fn refine_cluster_patches<'py>(
    py: Python<'py>,
    images: Vec<Bound<'py, PyAny>>,              // HxW / HxWxC uint8 arrays
    positions: Vec<PyReadonlyArray2<'py, f32>>,  // per image (N, 2)
    affine_shapes: Vec<PyReadonlyArray3<'py, f32>>, // per image (N, 2, 2)
    cluster_starts: PyReadonlyArray1<'py, u32>,
    member_images: PyReadonlyArray1<'py, u32>,
    member_features: PyReadonlyArray1<'py, u32>,
    // ... options ...
    progress: Option<ProgressCounter>,
) -> PyResult<Bound<'py, PyDict>>
```

- Image extraction + rayon pyramid build copies
  `build_pyramids_from_arrays` (`py_patch_cloud.rs:236`) — factor that
  helper so both call sites share it (it currently lives beside the
  recon-based `build_pyramids_and_poses`; the cluster path has no recon and
  no camera dimension check).
- Validate lengths (`images`, `positions`, `affine_shapes` equal; CSR arrays
  self-consistent) with `PyValueError`s before releasing the GIL, then run
  the kernel under `py.detach(...)` like `localize_keypoints`
  (`py_patch_cloud.rs:1265`).
- Return one `PyDict`: `reference_members` (C,) u32, `member_status` (M,)
  u8, `member_affines` (M,2,3) f64, `member_zncc` (M,) f32,
  `member_shift_px` (M,) f32 — all via `.into_pyarray(py)`.
- Register in `matching/mod.rs` beside the two existing cluster functions.

The `io/matches.rs` section threading is **shipped** with the format layer
(§1): `read_matches` / `write_matches` already round-trip the `clusters` /
`cluster_patches` dict keys the CLI command consumes and produces.

## 4. CLI command and orchestration

> _Status (2026-07-09): **Shipped** per the sketch below; documented in
> `specs/cli/cluster-patches-command.md`. Additions: the output path is
> refused when it already exists (write-once), the workspace resolves via
> the relative → absolute → ancestor-search chain `to-colmap-db` uses, and
> progress reporting reuses the `_poll_progress` ProgressCounter poller from
> `_embed_patches`._

`src/sfmtool/_commands/cluster_patches.py` — flat command, Image Feature
category, spec to live at `specs/cli/cluster-patches-command.md`:

```
sfm cluster-patches -i clusters.matches [-o out.matches]
    [--radius 4.0] [--resolution 25] [--min-zncc 0.85] [--max-shift 3.0]
    [--max-keypoint-uncertainty 0.35]
```

1. `read_matches(input)`; reject unless `has_clusters` (name the fix: run
   `sfm match --cluster`). Reject if `has_cluster_patches` already set
   (write-once: enrich the original instead).
2. Locate each image's `.sift` via the images section + workspace reference
   (`{workspace}/{image_parent}/{feature_prefix_dir}/{basename}.sift`),
   verify `sift_content_hashes`, and read `positions_xy` / `affine_shapes`
   (`read_sift`, capped at `feature_counts[i]`).
3. Load images with cv2 in the images-section order, call
   `refine_cluster_patches`, with a progress bar via the `ProgressCounter`
   pattern the other long kernels use.
4. Write a **new** `.matches` file: images + clusters sections copied
   verbatim, `cluster_patches` from the kernel output,
   `refine_options` = the CLI parameters, metadata updated
   (`has_cluster_patches: true`, fresh timestamp/content hash). Default
   output: input path with a `-patches` suffix before the extension.

## 5. Tests

> _Status (2026-07-09): **Shipped** with two deviations from the sketches
> below. (a) The `RejectedLowZncc` gate is pinned with a **flat** member
> image (every member channel windowed-flat → score 0): with an *unrelated
> smooth* texture the affine optimizer can chase a spurious ZNCC above the
> permissive 0.85 gate over the ~50-effective-sample window, tripping the
> shift gate instead. (b) `sfm match --cluster` does not yet write
> cluster-bearing files (the §1 derived-pairs migration), so
> `tests/test_cluster_patches.py` builds the CLI input programmatically —
> `cluster_match(...)` over the fixture's `.sift` files, written via
> `write_matches` — and keeps the spec's assertions (output verifies, > 50%
> of multi-member clusters keep ≥ 1 member, statuses within the enum with
> reference/kept present). The synthetic-recovery tolerance held: 0.3 px
> support-grid RMSE across the specified warp/noise ranges, unmodified._

- **`cluster_refine/tests.rs`**: (a) synthetic recovery — render a textured
  reference patch, warp it into a second image by a known affine (scale
  0.8–1.5×, rotation ≤ 20°, shear ≤ 0.15), perturb the seed by the
  experiment-observed noise (|Δlog s| 0.07, |Δrot| 4°, 1 px shift), assert
  the recovered affine maps the support grid within 0.3 px RMSE and
  `Kept`; (b) gate tests — flat texture → low ZNCC → `RejectedLowZncc`;
  seed shifted > `max_shift_px` from the optimum → `RejectedShift`;
  out-of-frame support → `NotEvaluated`; two members in one image →
  exactly one `Kept`; (c) determinism — two runs bit-identical; (d) AVX2
  vs scalar — force both paths on the same inputs, scores within 1e-4
  (the tolerance convention of the existing dual-path kernels).
- **`matches-format` tests** — **shipped** with §1: round-trips of
  clusters-only, clusters+cluster_patches, and pairwise+TVG (regression)
  files; verify-rejection coverage for both/neither backbones, bad CSR,
  out-of-range references, invalid statuses, and two kept members per
  image; plus the Python round-trip module `tests/test_matches_clusters.py`.
- **Python** `tests/rust_bindings/test_cluster_patches_rust_bindings.py`:
  tiny synthetic images end-to-end through the binding; dict schema and
  dtypes. `tests/test_cluster_patches.py`: CLI run over the
  `isolated_seoul_bull_17_images` fixture pipeline (extract → match
  --cluster → cluster-patches), asserting the output file verifies, > 50%
  of multi-member clusters keep ≥ 1 member, and statuses cover the enum.
- **Cross-check with the prototype** (manual, not CI): run the harness and
  the kernel on seoul_bull and compare kept-member sets and affines; the
  numbers should match `reports/2026-07-09-exp-pairwise-sift-warp.md`
  Part 2 within sampling noise.

## Out of scope (tracked in the design spec)

Smarter reference selection, fused reference templates, photometric
verification / surfel seeding / solver consumers, and any change to the
matcher itself.
