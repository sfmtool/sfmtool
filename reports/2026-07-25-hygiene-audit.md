# Hygiene audit — 2026-07-25

Read-only structural survey of the whole codebase (Python `src/sfmtool/` + `tests/`,
Rust `crates/`, top-level layout) for oversized multi-concern files, duplication,
misleading names, directory smells, and dead code. Produced by the `audit-hygiene`
skill; supersedes the 2026-07-07 snapshot (retired in the same commit — its 16
resolved findings are history, its 11 open items are all carried forward below,
most with better detail than the original).

**Headline:** the 2026-07-07 backlog was worked down hard — the three big splits
(`upload.rs` #233, `reconstruction/data.rs` #234, `point_track_detail.rs` #235) all
landed, along with the `_sfmtool` submodule migration, the keypoint kernel splits,
and a low-risk hygiene batch. But 53 commits landed in those 18 days, and the
geometry/matching wave (#199, #214–#228) created three of the four largest non-test
source files in the workspace — none of which the old report could see. The centre
of gravity has moved from *file size* to **duplication**: the single largest theme
in this snapshot is near-identical code families that must be fixed twice
(`bundle_adjust` finite/mixed, format `write`/`verify` invariants, `archive_io.rs`
×4, polar/sweep plain/geometric, six Python helper pairs). Several are already
drifting, and drift here is silent — no compile error, just two behaviours.

**Scale:** ~130k non-test source lines. `patch/` 21.2k, `features/` 18.6k,
`geometry/` 16.3k, `camera/` 12.2k, `sfm-explorer` 16.1k, `sfmtool-py` 16.7k,
`src/sfmtool/` 35.3k, `tests/` 32.0k.

---

## Rust — `sfmtool-core` (geometry & features)

**Mirrored finite/mixed bundle-adjust families — the finite half is a strict subset**
- Location: `crates/sfmtool-core/src/geometry/bundle_adjust.rs` (1529)
- Problem: Five function pairs are mirrored. Measured by diff, not eyeball:
  `solve_lm` (269–596, 328 lines) vs `solve_lm_mixed` (998–1402, 405) differ by
  **19 removed / 96 added** — ~309 of 328 lines appear verbatim in the mixed copy,
  including the whole Schur-complement accumulation, the 12-step damping ladder,
  the LU solve, and the scatter-back. Also `bundle_adjust_finite` (704–800) vs
  `bundle_adjust_mixed` (1408–1527), `residual_norms_depths` (110–135) vs `_mixed`
  (843–871), `robust_cost` (231–264) vs `_mixed` (950–989), `retriangulate`
  (141–184) vs `reestimate_points` (879–944). ~529 lines of "finite" shadowed by
  ~660 lines of "mixed". The mixed path reduces exactly to the finite path when
  `is_dir` is all-false, and the dispatcher (656–681) branches on nothing else.
  A bug fixed in the damping ladder must be fixed twice.
- Proposed fix: delete the finite family; have `bundle_adjust` always call the
  mixed family with an all-`false` mask. **Guard with a golden test asserting
  bit-identical `focal`/`residual_norms` on an all-false mask before removing
  anything.** Do not abstract the two behind a trait — the mask *is* the right
  abstraction and a trait would obscure the math.
- Effort: medium
- Risk: medium — the numerical core. A silent change to step scaling or
  convergence degrades reconstructions without failing to compile. The
  bit-identity test is mandatory, not optional.

**`grow_reconstruction` is a 602-line function with five separable phases**
- Location: `crates/sfmtool-core/src/geometry/reconstruction_growth.rs` (1126), fn at 522–1123
- Problem: Holds a nested `fn run_grow_ba` (626–716) and three closures, but the
  body still spans setup (533–611), the growth loop (788–1037, ~250 lines), and a
  focal-release finishing pass (1039–1123). The growth loop carries three
  independent sub-policies inline, including a **verified force-accept with full
  state save/rollback** (812–974, ~163 lines) that snapshots `quats`/`trans`/
  `points`/`ba_mask`/`since_ba` and restores at 960–972 — exactly the kind of code
  that must be readable in isolation, sitting 450 lines into a function. The file's
  other helpers (`resect_one`, `fill_new_points`, `ba_cluster_mask`, `build_result`)
  show the extraction pattern is established; it just stops at the main loop.
- Proposed fix: introduce an explicit `GrowState` struct (so the borrow friction
  that forced the closures disappears) and extract `try_force_accept` (837–974),
  `finish_with_focal_release` (1039–1123), `setup_growth_state` (533–611). Leaves
  ~150 lines of readable policy.
- Effort: medium
- Risk: medium — save/restore semantics are subtle (note `posed_order.pop()` is
  conditional). Covered by `reconstruction_growth/tests.rs` (860 lines).

**`covisibility.rs` serves three specs and hides 793 lines of inline tests**
- Location: `crates/sfmtool-core/src/features/cluster_match/covisibility.rs` (1744)
- Problem: Two violations. (1) `#[cfg(test)] mod tests { … }` at 952–1744 — **793
  inline lines**, by far the largest in the workspace, against a convention 46
  files in this subtree follow. The sibling `cluster_match/tests.rs` (283, declared
  `mod.rs:25`) covers the *track-cluster matcher*, so covisibility has no existing
  home. (2) The 951-line source half serves three different specs, cited in its own
  module doc at 15–19: `DisplacementNeighborhood` + `DisplacementTables` +
  `PairAccum` (152–479, `specs/core/pose-verification.md`); `ClusterCovisibility`
  construction and seed groups (480–718, 852–951,
  `specs/core/cluster-covisibility.md`); selection queries `sweep_order`/`thin*`/
  `reach` (719–850, `specs/core/covisibility-selection.md`). Three specs in one
  file is the clearest possible signal.
- Proposed fix: `cluster_match/covisibility/` with `mod.rs` (error enum,
  `SplitMix64`, `ClusterCovisibility` core, re-exports), `displacement.rs`
  (152–479), `selection.rs` (719–850), and `tests.rs` carrying the 793 lines.
- Effort: medium — `ClusterCovisibility`'s private fields are read by
  `sweep_order`/`thin_in_order`, so those move with accessors or stay in `mod.rs`.
- Risk: low — pure code motion, compiler-checked.

**`run_gpu_levels_prebuilt` is a 484-line function, two thirds of its file**
- Location: `crates/sfmtool-core/src/features/optical_flow/gpu/mod.rs` (740), fn at 253–736
- Problem: Phases are marked by the author's own banners and cleanly separable:
  pool sizing (268–289), per-level uniform buffers + bind groups (290–501, ~212),
  final-upsample resources (502–599, ~98), single-command-buffer encoding
  (600–701), submit/wait/readback (702–736). Everything is inlined because the
  resources must outlive the encode phase — solvable by returning owned structs.
  At this length the "single submit, single sync" design the function exists to
  express is buried.
- Proposed fix: extract `build_level_resources(...) -> Vec<LevelResources>` and
  `build_final_upsample_resources(...) -> Option<UpsampleResources>`, each owning
  its buffers/bind groups. Caller becomes ~120 lines.
- Effort: medium
- Risk: low-medium — wgpu resources must live until submit completes, so the
  extracted structs must bind to locals, not temporaries. `gpu/tests.rs` (625) runs
  on the `noop` backend and covers this.

**Parallel plain/`_geometric` matcher families across `polar.rs` and `sweep.rs`**
- Location: `crates/sfmtool-core/src/features/feature_match/polar.rs` (844), `sweep.rs` (459)
- Problem: Four pairs implement the same algorithm twice, differing only by whether
  a geometric filter is applied to the sliding window. Measured:
  `polar_mutual_best_match` (324–445) vs `_geometric` (705–842) — only 41 changed
  lines out of 122/138, so ~100 duplicated (epipole computation, polar transform,
  angle-offset alignment, dual sort, mutual-consistency filter all copied);
  `polar_match_one_way` (231–323) vs `_geometric` (564–704) ~85 shared;
  `extend_for_wraparound` (154–230) vs `_geometric` (491–563) ~45;
  `mutual_best_match_sweep` (128–216) vs `_geometric` (348–427) ~60. Roughly **290
  duplicated lines**. The wraparound index mapping
  `((ext_idx - n_prepended).rem_euclid(orig_len2))` appears verbatim in both polar
  one-way variants — a bug there needs two fixes. The call sites (`mod.rs:105–138`)
  are an `if use_geometric` branch passing near-identical argument lists.
- Proposed fix: make the filter an
  `Option<(&StereoPairGeometry, &GeometricFilterConfig)>` parameter on the plain
  functions; the `_geometric` variants become thin wrappers (the `pub` API must be
  preserved — `sfmtool-py/src/matching/sweep.rs` binds all four).
- Effort: medium
- Risk: low-medium — the geometric path's remapping through `valid_indices` is the
  delicate part. `polar/tests.rs` (433) and `sweep/tests.rs` (355) cover both.

**Numeric helpers copied verbatim across `geometry/` — a silent determinism hazard**
- Location: `crates/sfmtool-core/src/geometry/` (6 files) + 2 outside
- Problem: `splitmix64` is **byte-identical** in four files —
  `homography_estimation.rs:185`, `epipolar_estimation.rs:301`,
  `absolute_pose.rs:371`, `reconstruction_growth.rs:165` — plus a fifth copy in
  `patch/cluster_refine/consistency.rs:60` and a struct form in
  `covisibility.rs:128`. `median` is byte-identical in `pose_verification.rs:207`
  and `reconstruction_growth.rs:181`. `polar_rotation`/`rotation_angle` are
  byte-identical in `rotation_init.rs:152,163` and `pose_verification.rs:191,202`.
  `inlier_fraction_of` is functionally identical in `pose_verification.rs:227` and
  `reconstruction_growth.rs:229`. `cam_at` (`bundle_adjust.rs:98`) and
  `cam_with_focal` (`reconstruction_growth.rs:197`) have identical bodies and the
  latter's doc says so. Larger: `PairAccum` + `pair_correspondences` +
  `ImageClusters` are duplicated between `focal_vote.rs:142–205` and
  `rotation_init.rs:172–265` (~45 lines). These are RNG and threshold primitives —
  drift between copies changes results without any compile error.
- Proposed fix: `geometry/numeric.rs` (`pub(crate)`) holding the shared helpers.
  The precedent exists — `focal_vote::ortho_cost` is already `pub(crate)` so
  `rotation_init` can share it. Note `inlier_fraction_of` reads a per-module
  `INLIER_PX` const (both `3.0`): make the threshold a parameter rather than
  assuming they stay equal.
- Effort: low
- Risk: low

**`optical_flow/mod.rs` carries ~490 lines of container types before its driver**
- Location: `crates/sfmtool-core/src/features/optical_flow/mod.rs` (958)
- Problem: Declares the submodule tree (37–96) *and* defines the shared types every
  submodule consumes: `FlowFieldRef` (99–198), `FlowField` (199–342), `GrayImage`
  (343–449), `DisFlowParams` (450–564), `FlowTiming` (565–588) — before the driver
  functions begin at 589. `GrayImage` is imported far outside optical flow
  (`sift/scale_space.rs`'s module doc says "Conventions follow the optical-flow
  module: `GrayImage`"), so a foundational type lives inside a `mod.rs` that also
  orchestrates a pipeline. The weakest finding in this section — not incoherent,
  just carrying two altitudes.
- Proposed fix: `optical_flow/{image,flow_field,params}.rs`, re-exported from
  `mod.rs` so no external path changes. `mod.rs` drops to ~470 lines of driver.
- Effort: low
- Risk: low — pure motion behind unchanged re-exports.

---

## Rust — `sfmtool-core` (camera, patch, reconstruction)

**`camera/distortion.rs` holds three concerns, one with its own spec**
- Location: `crates/sfmtool-core/src/camera/distortion.rs` (1350)
- Problem: (1) 141–748 — the actual distortion/undistortion kernels, coherent and
  matching the filename. (2) 77–133 + 926–1157 — the **coarse ray-grid projection**
  accelerator (`GridProj`, `lerp`, `bilerp`, `grid_proj`, `ray_to_pixel_grid*`),
  ~290 lines of a caching/interpolation strategy with its own design document
  (`specs/core/ray-grid-projection.md`) and nothing to do with lens distortion.
  (3) 1189–1348 — `best_fit_inside_pinhole`/`best_fit_outside_pinhole`/
  `boundary_samples`, deriving an undistorted pinhole approximation. A reader
  looking for grid projection has no reason to open `distortion.rs`.
- Proposed fix: `camera/distortion/ray_grid.rs` and
  `camera/distortion/pinhole_fit.rs` — the `distortion/` dir already exists
  (`kernels.rs`, `tests.rs`), so this is purely additive.
- Effort: low
- Risk: low — splitting an inherent impl across files is legal; no re-export changes.

**395 lines of hand-mirrored `CameraModel` ↔ `SfmrCamera` mapping in `intrinsics.rs`**
- Location: `crates/sfmtool-core/src/camera/intrinsics.rs` (913)
- Problem: The first half (23–513) is a coherent 13-variant enum plus accessors.
  The second half is string-keyed serialization: `TryFrom<&SfmrCamera>` (534–660)
  and `From<&CameraIntrinsics>` (666–909, **243 lines**). Both directions
  hand-enumerate the same parameter names, so every literal appears an even number
  of times — `"principal_point_x"`/`"_y"` 24× each, `"radial_distortion_k1"` 18×,
  `"focal_length_x"`/`"_y"` 14× each. Adding a camera model means editing two
  100+ line matches; forgetting one direction yields a model that reads but cannot
  round-trip, with nothing structurally preventing it.
- Proposed fix: move both impls to `camera/intrinsics/sfmr_conv.rs`, then collapse
  the mirroring — one `fn params(&self) -> Vec<(&'static str, f64)>` +
  `fn from_params(name, &HashMap)` per variant so the name list is written once.
- Effort: medium
- Risk: medium — a transcription slip changes on-disk `.sfmr` camera parameters.
  Round-trip every variant through `intrinsics/tests.rs` (690) before and after.

**`camera/remap.rs` mixes image containers with resampling and inlines `prof`**
- Location: `crates/sfmtool-core/src/camera/remap.rs` (1265)
- Problem: (a) 28–122 is a 95-line inline `pub mod prof { … }` — but the repo has
  **five** sibling `prof.rs` modules under `patch/*/`, and `camera/remap/` already
  exists for `tests.rs`, so this is the only inline one. (b) 157–325 (`ImageU8`,
  `ImageU8Pyramid`) and 915–991 (`ImageF32WithGrad`) are general image containers,
  not resampling. They are among the crate's most-used types (30+ import sites
  across `spherical/`, `patch/`, `sfmtool-py`, `sfm-explorer`, benches) yet their
  canonical path is `camera::remap::ImageU8`, which reads as neither a camera nor a
  remap concept. (c) The remaining ~800 lines of samplers and `remap_*` entry
  points are genuinely coherent.
- Proposed fix: `camera/remap/prof.rs` (mechanical, five precedents). Move the
  three image types to `camera/image.rs` with a `pub use` shim in `remap` so the 30
  sites need not churn at once.
- Effort: low (prof) / medium (types)
- Risk: low — path change only, shimmed.

**Two files whose own doc comments announce a split that was never made**
- Location: `crates/sfmtool-core/src/patch/keypoint_subpixel/kernels.rs` (987);
  `crates/sfmtool-core/src/reconstruction/data.rs` (598)
- Problem: `kernels.rs`'s module doc (7–16) literally says "Two halves:
  **Rendering / render-once tile** … **Scoring kernels**." The boundary is exact:
  rendering 32–814, scoring 816–987. Every item is `pub(super)`, so the split is
  free. Separately, `data.rs` was split in `fdab9c7` and is otherwise coherent —
  except 285–402 (~118 lines): `observation_affine_shape` (299–369) back-projects a
  keypoint ray onto the patch plane and projects the half-axis tips, and
  `max_embedded_feature_size_per_point` (382–402) reduces over it. That is
  projection algebra with its own spec reference, sitting between two 3-line
  accessors. (This is the leftover the #234 status note flagged and deferred.)
- Proposed fix: `kernels/{render,score}.rs`; move the two `data.rs` methods to
  `data/affine_shape.rs`, finishing the split #234 started.
- Effort: low
- Risk: low — pure motion, items already `pub(super)`.

**Four remaining inline `#[cfg(test)] mod tests { … }` blocks worth moving**
- Location: `patch/cluster_refine/consistency.rs:371–589` (219 inline of 589);
  `patch/normal_refine/fronto_cache.rs:512–670` (159 of 670);
  `patch/normal_refine/view_subset.rs:118–259` (142 of 259);
  `features/sift/simd.rs:110–175` (66 of 175)
- Problem: The subtree has **29** sibling `tests.rs` files against these four, so
  the exception is genuinely rare. `view_subset.rs` is **55% test code**;
  `consistency.rs`'s 219 test lines sit right after a 192-line
  `warp_consistency_residuals`, so the file reads as algorithm plus appendix. All
  live in directories that already have a module-level `tests.rs`. `simd.rs`'s
  `#[cfg(all(test, target_arch = "x86_64"))]` gate is not a justification — the
  attribute sits on a `mod tests;` declaration just as well.
- **Deliberately not flagged:** `sfmr-format/src/verify.rs:547–573` (27 lines) and
  `sfmr-colmap/src/colmap_io/read.rs:511–528` (18 lines). Each is a single focused
  test on a single private helper directly above it; creating a directory for 18
  lines costs more navigation than it saves. The old report's blanket
  inline-tests finding did not make this distinction.
- Proposed fix: move the four to sibling `tests.rs` files.
- Effort: low
- Risk: low

---

## Rust — format crates

**`verify_matches` is a 719-line function — the largest in the workspace**
- Location: `crates/matches-format/src/verify.rs` (882), fn at 164–882
- Problem: One function interleaves (a) *section hashing* — read each ZIP entry in
  lexicographic order, feed an `Xxh3`, compare to the stored digest — with (b)
  *structural validation* on the raw bytes it happens to keep alive. The two
  alternate six times: metadata hash 207–216, images hash 218–279 (with a value
  check wedged in at 234–257), pairs hash 283–337 then validation 339–421, clusters
  423–483 then 484–569, cluster-patch 570–656 then 657–760, TVG 761–866, overall
  867–882. Because validation reads the *raw* buffers, the function holds
  `pairs_raw`, `match_counts_raw`, `match_fi_raw`, `feature_counts_raw` live across
  hundreds of lines and re-derives indices by hand (`pair_idxs[k * 2 + 1]`).
  Nothing can be unit-tested in isolation.
- Proposed fix: one `hash_<section>(...) -> (u128, RawBufs)` and one
  `check_<section>(raws, …, &mut errors)` per section; the entry point becomes an
  ~80-line sequence.
- Effort: medium
- Risk: medium — hashing depends on **entry read order**; reordering silently
  changes the computed content hash. Extract mechanically and keep
  `matches-format/src/tests.rs` (1468) green.

**Every format invariant is implemented twice, with different index math**
- Location: `crates/matches-format/src/{write,verify}.rs`, `crates/sfmr-format/src/{write,verify}.rs`
- Problem: The writer validates typed `Array2` data and fails fast; the verifier
  re-validates the same rules from raw `&[u8]` and accumulates errors. Near-verbatim
  logic: pair sorting + bounds + feature-index bounds at
  `matches-format/src/write.rs:834–882` vs `verify.rs:341–421`, character-identical
  error strings but `pairs.image_index_pairs[[k, 0]]` on one side and
  `pair_idxs[k * 2]` on the other. Measured: of long (>25 char) string literals,
  **45 are identical between matches-format's write.rs and verify.rs** (of 90/77),
  and **29 between sfmr-format's** (of 67/47). Those literals are both error
  messages *and* ZIP entry-name templates — and the entry-name templates appear a
  **third** time in `read.rs` (~23 sfmr templates, 3× each). A rename or a tightened
  rule must land in three places or the file silently fails verification.
- Proposed fix: per crate, add `entries.rs` with one fn per archive entry name and
  have write/read/verify call it — kills the triplication, nearly risk-free. Then
  hoist shared invariants into `check_*(…) -> Vec<String>` in `types.rs` over a view
  trait both paths implement.
- Effort: medium (entry names alone: low)
- Risk: low for entry names; medium for the invariant merge — the paths differ in
  `break` vs `break 'outer`, so verifier output ordering may shift.

**`archive_io.rs` is copy-pasted into all four format crates, and camrig diverged in the right direction**

> _Status (2026-07-31): Done — extracted to a new `sfmtool-archive-io` workspace
> crate. The four copies are deleted; camrig's non-cloning `write_binary_entry`
> signature is now the only one, removing all 55 clone sites (26 sfmr, 25
> matches, 4 sift — the finding's "51" is the sfmr+matches subtotal).
> `sfmr-format`/`matches-format` moved to a new `write_binary_entry_hashed`
> helper that writes the entry and folds the uncompressed bytes into the section
> hasher in one call. `raw_to_u32`/`raw_to_f64` moved there too, taking
> the unaligned-buffer regression test with them (it had only ever guarded the
> sfmr copy). `read_uint128_array` was **kept**, correcting this finding: it is
> dead only in `sift-format`, but live in `sfmr-format/src/read.rs:10` and
> `matches-format/src/read.rs:12`, so the shared copy simply drops sift's
> `#[allow(dead_code)]`._
>
> _Scope note: the write/hash pairing is enforced for **binary** entries only.
> Fourteen JSON entries across the two writers still pair `write_json_entry`
> with a manual `hasher.update(&bytes)`, and they feed the same section hashes —
> so the invariant is structural for binary entries and conventional for JSON.
> A `write_json_entry_hashed`, plus a `write_binary_entry_digested` for the
> seven remaining one-shot-digest sites in sift/camrig, would finish the job and
> let the unhashed `write_binary_entry` go private._
>
> _Evidence note: content-hash preservation was **not** established by the
> cross-verification run first cited here. Both verifiers recompute digests from
> the bytes present in the file, so any self-consistent archive passes both —
> that experiment shows the two trees produce mutually intelligible archives,
> not that a hash value is unchanged. The claim does hold, on stronger evidence
> gathered afterwards: a static comparison showing all 51 migrated call sites
> are byte-identical old-vs-new in (entry name, data expression, zstd level,
> hasher identity, order), and a direct old-vs-new comparison of stored section
> digests over a fixture exercising every optional section, which came out
> bit-identical (with `.sift` archives byte-identical end to end). Only the JSON
> sections carrying `HashMap` fields differed, for the unrelated reason recorded
> below._

- Location: `crates/{sfmr,sift,matches,camrig}-format/src/archive_io.rs` (163/164/164/142)
- Problem: Verified by diff. `sfmr` vs `sift`: one line. `sift` vs `matches`: a doc
  comment plus that line. So three copies are functionally identical. **`camrig` is
  the divergent one and it diverged forward**: (a) it deleted `read_uint128_array`,
  which `sift-format` still carries as dead code explicitly marked
  `#[allow(dead_code)]` at `archive_io.rs:95` with zero uses; (b) its
  `write_binary_entry` returns `Result<(), _>` with the comment "the caller already
  owns `data` and can hash it directly, so nothing is returned — this avoids
  cloning large binary buffers," while the other three still end in
  `Ok(data.to_vec())`. That copy is not free: `sfmr-format/src/write.rs` calls it
  **26 times** and `matches-format/src/write.rs` **25 times**, each cloning a whole
  uncompressed column — positions, tracks, 128×128 thumbnails, patch bitmaps —
  purely to feed `hasher.update(&bytes)`. The fix exists in one crate and cannot
  reach the other three. Same habit: `raw_to_u32` at `sfmr-format/src/verify.rs:23`
  and `matches-format/src/verify.rs:23` are byte-identical, but only the sfmr copy
  has the unaligned-buffer regression test.
- Proposed fix: new workspace crate `crates/archive-io` (the four format crates
  currently have **zero** inter-crate deps, so no cycle risk). Adopt camrig's
  `write_binary_entry` signature, drop `read_uint128_array`, move
  `raw_to_u32`/`raw_to_f64` there.
- Effort: medium
- Risk: low — 8 functions, covered directly by each crate's round-trip tests.
- Note: this supersedes the old report's `archive_io` finding, which was deferred
  twice (2026-06-23 #9) on the grounds that the copies were still in sync. They are
  no longer, and the drift now carries a measurable allocation cost.

**Archives are not byte-reproducible: `HashMap` fields serialize in randomized order**
- _Added 2026-07-31, found while validating the `archive_io` extraction — not part
  of the original snapshot._
> _Status (2026-07-31): Done — all five affected fields moved to `BTreeMap`.
> Verified: three separate processes writing the same reconstruction now emit
> byte-identical `.sfmr` archives (they differed before). Pre-change archives,
> whose JSON carries keys in the old arbitrary order, still read and verify —
> confirmed against the checked-in `kerry_park.camrig` and against `.sfmr`/
> `.matches` files written by the pre-change tree. Locked in by
> `sfmr-format/src/tests.rs::json_maps_serialize_in_sorted_key_order`; note an
> in-process double-write cannot catch a regression here (equal keys in
> equal-capacity maps iterate identically within one process), so the test
> asserts sorted key order instead._

- Location — **wider than first recorded**: this is five fields across three
  crates, not two in one. `crates/sfmr-format/src/types.rs:66`
  (`parameters: HashMap<String, f64>`), `:114`
  (`tool_options: HashMap<String, serde_json::Value>`);
  `crates/matches-format/src/types.rs:145` (`matching_options`), `:270`
  (`verification_options`); `crates/camrig-format/src/types.rs:125`
  (`parameters`). `.matches` was affected too — a varying
  `two_view_geometries_xxh128` is reproducible from the pre-change tree alone,
  via `TvgMetadata.verification_options`.
- Problem: Writing the *same* `SfmrData` twice in two processes produces archives
  that differ in bytes and in stored `metadata` hash — measured: two runs of one
  unchanged binary emitted `basic.sfmr` at 4,873 and 4,874 bytes. Rust's `HashMap`
  iteration order is randomized per process, and `serde_json` serializes in
  iteration order, so the camera-parameter JSON key order varies run to run.
  Verification is unaffected (the hash is computed over what was actually
  written), so this is not a correctness bug — but it rules out byte-comparing
  two archives as a regression test, which is exactly the check a future change to
  the write path most wants.
- Proposed fix: `BTreeMap<String, f64>` for `parameters`, or
  `#[serde(serialize_with = …)]` sorting keys. Either makes writes reproducible
  and changes no existing file's validity. Note it *does* change the metadata hash
  of newly written files relative to old ones, which is already true run-to-run.
- Effort: low
- Risk: low

**Four monolithic per-section entry points in the two big format crates**
- Location: `sfmr-format/src/write.rs:96–628` (`write_sfmr_with_options`, **533**);
  `sfmr-format/src/verify.rs:41–545` (`verify_sfmr`, 505);
  `sfmr-format/src/read.rs:43–515` (`read_sfmr`, 473);
  `matches-format/src/write.rs:29–452` (`write_matches`, 424)
- Problem: The old report flagged `write_sfmr_with_options` at ~537 lines; it is now
  533 — effectively untouched. All four share a shape: a linear run of
  `// === Section ===` blocks, each opening a hasher, emitting 4–12 entries in
  lexicographic order, digesting, storing. The boundaries are already marked by
  comments. Secondary: `matches-format/src/write.rs:536–812` `validate_dimensions`
  (277) and `sfmr-format/src/write.rs:768–1031` `validate_dimensions_with` (264,
  carrying `#[allow(clippy::too_many_arguments)]`).
- Proposed fix: one `write_<section>_section(...) -> Result<u128>` per section;
  entry points become ~60-line orchestrators. Split `validate_dimensions*` per
  section too, which also removes the `too_many_arguments` allow.
- Effort: medium
- Risk: medium — same content-hash ordering hazard. Do write and verify together so
  ordering stays in lockstep.

---

## Rust — `sfm-explorer`

**`image_browser.rs::show()` is a 534-line method holding four subsystems**
- Location: `crates/sfm-explorer/src/image_browser.rs` (735), `show()` at 141–674
- Problem: Nine params behind `#[allow(clippy::too_many_arguments)]`. It inlines an
  animation player state machine (keyboard 267–317, fps clock/loop/direction
  319–365), a pan/scroll controller (minibar drag-scrub 229–265, gesture+drag pan
  367–396, auto-scroll clamp 398–417), a virtualized thumbnail grid painter
  (426–540 with four layered highlight rules at 454–510), and a complete minibar
  widget (barcode blit 542–561, play/pause + fps 563–627, viewport indicator
  629–645, ticks 647–671). Genuinely four things.
- Proposed fix: `image_browser/minibar.rs` takes `NavigationMinibar` (72–96) +
  229–265 + 542–671; `image_browser/animation.rs` takes `AnimationState` (41–70) +
  267–365. `show()` drops to ~200 lines of layout+paint. Also delete
  `enum PlayDirection` (line 34, `#[allow(dead_code)]`) — `Backward` is never
  constructed.
- Effort: medium
- Risk: medium — minibar and grid share scroll-offset state; extraction must thread
  it explicitly or scrubbing/auto-scroll regress. The headless egui harness added in
  #235 (`point_track_detail/tests.rs`) is the model for covering this before the
  split — `ui_basic.rs` only runs on Windows/macOS.

**`draw_overlays` — five copy-pasted match arms, plus two disagreeing `error_color`s**
- Location: `crates/sfm-explorer/src/image_detail/overlay.rs` (568), fn at 19–387
- Problem: The `match overlay_mode` at 51–290 has five arms — `ReprojError`
  114–152, `TrackLength` 153–191, `MaxTrackAngle` 192–223, `DepthReliability`
  224–255, `ConditionNumber` 256–289 — that are the same body (compute range → loop
  features → cull → `circle_filled(center, 5.0, color)` → yellow stroke if selected
  → `draw_colorbar`), differing only in value extractor, colormap fn, and label.
  ~175 lines collapse to one call. Related: `colormap.rs:60–118` holds five
  `*_color` fns with byte-identical `t` normalization, and there are **two different
  `error_color` functions** — `colormap.rs:60–67` (`value, vmin, vmax`, samples
  `ERROR_COLORMAP`) vs `point_track_detail/metrics.rs:19–33` (hard-coded 0–2px
  green/yellow/red ramp) — so the two panels disagree on what a 1px error looks like.
- Proposed fix: one
  `draw_value_overlay(painter, features, extract, colormap, label)`. Reconcile the
  two `error_color` ramps and delete the loser.
- Effort: low
- Risk: low for the arm collapse; medium for the colormap reconciliation — it
  changes on-screen colours in one panel, which is user-visible.

**`app.rs` chrome inside the render pipeline; a latent thumbnail panic; two dead `clear()`s**
- Location: `crates/sfm-explorer/src/app.rs` (735), `image_browser.rs`, `image_detail/mod.rs` (651)
- Problem: `app.rs` is otherwise a coherent 4-phase loop matching its module doc,
  but `run_egui_pass` (449–660, 212 lines) inlines the File/View menu bar (495–588)
  and the "Load Demo Data" modal (590–627) — ~135 lines of chrome in a render
  module. Separately: **(a) latent panic.** `image_browser.rs:718–734` and
  `point_track_detail/table.rs:358–379` are near-identical `load_thumbnail` copies,
  but the browser one **hard-codes 128×128** (726, 731) while the table one reads
  `rgb_slice.shape()`. `THUMBNAIL_SIZE` is a const (`gpu_types.rs:263`), so if it
  ever changes, `ColorImage::from_rgba_unmultiplied` — which does
  `assert_eq!(size[0] * size[1] * 4, rgba.len())` — panics in the browser only.
  (Not silent corruption: it aborts.) The same RGB→RGBA expansion appears a third
  time at `image_detail/mod.rs:286–292`. **(b)** Two never-called `pub fn clear`
  under `#[allow(dead_code)]`: `point_track_detail/mod.rs:218`,
  `image_detail/mod.rs:531`. **(c)** `image_detail/mod.rs` builds the same 8-field
  `FeatureOverlayState` literal six times (320–329, 335–344, 372–381, 422–431,
  436–445, 518–527).
- Proposed fix: move `app.rs:495–627` to a new `menu.rs`; unify the two
  `load_thumbnail`s on the shape-reading version; delete both dead `clear()`s or
  wire them into the reconstruction-swap path; add `FeatureOverlayState::new(..)`.
- Effort: medium
- Risk: low — all mechanical; the `load_thumbnail` unification is a bug fix.

**`point_track_detail::metrics` is imported by a sibling UI panel**
- Location: `crates/sfm-explorer/src/point_track_detail/metrics.rs`, `image_detail/mod.rs:613,650`
- Problem: `compute_point_diagnostics` and `compute_max_pairwise_angle` are
  triangulation math, not panel code, but they live inside one panel and are
  imported by another. `point_track_detail/mod.rs:38–41` re-exports them with the
  comment "Re-exported at the old path" — introduced by #235 to keep the split pure,
  which documents the smell rather than fixing it. Flagged here so it isn't lost.
- Proposed fix: promote to a crate-level `src/metrics.rs` and drop the compat
  re-export.
- Effort: low
- Risk: low

---

## Rust — `sfmtool-py`

**`clone_with_changes` is a single 558-line function**
- Location: `crates/sfmtool-py/src/reconstruction/clone.rs` (731), fn at 74–632
- Problem: The *file* is coherent (its doc says it is the extracted body of
  `clone_with_changes`), but the function is the only one over 150 lines in the
  crate. It is a 23-arm kwargs `match` followed by five sequential post-passes:
  image-count application 445–473, `rebuild_observation_source` 474–484, histogram
  resize 485–491, deferred `patch_bitmaps` 492–525, track rebuild 526–618, derived
  recompute 619–631. The arms are near-identical boilerplate, which is why this file
  holds 40 `PyValueError::new_err` sites, the crate's highest.
- Proposed fix: split by field group into `apply_{point,pose,image,track}_fields`,
  each taking `&mut` builder state; keep the five post-passes in the outer function.
  The existing `extract_ndarray!`/`extract_array1!`/`extract_array2!` macros already
  carry the per-arm boilerplate, so the split is mechanical.
- Effort: medium
- Risk: medium — the arms have ordering dependencies (deferred `patch_bitmaps` at
  492 exists precisely because it must run after the point fields); a naive split
  reorders them.

---

## Python — `src/sfmtool/`

**Dead code: two sweep-matching wrappers reachable only from tests**
- Location: `src/sfmtool/feature_match/_polar_sweep.py` (171), `_rectified_sweep.py` (175) — 346 lines
- Problem: Verified by full-tree search — the only references anywhere outside
  their own files are `tests/test_densify.py:21` and `:26`. The production path is
  `feature_match/_core.py`, which imports the Rust matchers directly from
  `.._sfmtool.matching` (line 16) — the same entry points these modules wrap.
  `_densify.py:31` goes through `feature_match.match_image_pair`, not these.
  `feature_match/__init__.py` does not export them. Meanwhile `_core.py:7` and
  `__init__.py:7` still document "automatically selects between rectified and
  polar", which now happens inside Rust.
- Proposed fix: delete both; retarget the `test_densify.py` assertions at
  `.._sfmtool.matching` directly (they test Rust behaviour anyway).
- Effort: low
- Risk: low — one test file's imports break; the fix is mechanical.

**`xform/_arg_parser.py`: a 440-line argv loop plus four copy-pasted key=value parsers**
- Location: `src/sfmtool/xform/_arg_parser.py` (829)
- Problem: Two problems, neither of which is the "co-location" the old report named.
  (a) `parse_transform_args` (390–829, **440 lines**) is one `while` loop with 27
  `if/elif arg ==` branches. The guard
  `if i + 1 >= len(args): raise click.UsageError("--X requires an argument")` is
  written out **20 times**; the 12-line optional-value tokenization block is
  copy-pasted verbatim four times (480–491, 501–512, 517–528, 533–544), each with
  the comment "same tokenization as `--refine-normals`". (b)
  `parse_refine_normals_params` (125–193), `parse_refine_keypoints_params`
  (194–265), `parse_localize_keypoints_params` (266–326),
  `parse_to_embedded_patches_params` (327–371) are ~250 lines of the same 40-line
  body, differing only in the option name, the `_*_KEYS` caster table, and the
  transform class. They have already drifted — only `_REFINE_NORMALS_KEYS` handling
  has the `caster is str` short-circuit (160–161).
- Proposed fix: (b) first — collapse the four into
  `_parse_kv_params(option_name, keys, ctor, param)` with three-line public
  wrappers; that removes ~200 lines. Then (a) — replace the if/elif chain with a
  spec table `{flag: (arity, builder)}` plus shared token readers.
- Effort: medium
- Risk: medium — this is the `sfm xform` CLI surface and the optional-value
  tokenization deliberately mirrors Click (comment at 480–485). Error text is
  user-visible and must be preserved exactly; `tests/xform/` must be green either side.

**`sift/file.py` holds three concerns and its name describes one**
- Location: `src/sfmtool/sift/file.py` (877)
- Problem: File I/O proper — validation 205–244, `SiftReader` 245–299, write path
  300–454, path resolution 455–539 — is ~350 lines and matches the name. The file
  also carries the **extraction pipeline** `image_files_to_sift_files` (593–761) +
  `_opencv` (764–787), which imports and drives the sibling `extract_colmap.py`/
  `extract_opencv.py` from inside the function body (612–613) — the orchestration
  layer for the extract modules lives in the I/O module — and a **visualization**
  function `draw_sift_features` (794–877). Also stranded: xxh128 helpers (60–110)
  and pure feature geometry `compute_orientation`/`feature_size*` (117–167),
  neither of which touches a file.
- Proposed fix: `sift/extract.py` for 593–787 (becomes the peer of
  `extract_colmap.py`/`extract_opencv.py`/`extract_sfmtool.py`, and the deferred
  import at 612 becomes a normal one); `visualization/_sift_display.py` for
  794–877; optionally `sift/geometry.py` for 117–167. Re-export from
  `sift/__init__.py`.
- Effort: medium
- Risk: low — pure moves behind existing re-exports.

**Strip modules form one closed pipeline — the `strips/` subpackage still earns itself**
- Location: `src/sfmtool/_solve_strips.py` (486), `_compare_strips.py` (479), `_inspect_strips.py` (241), `_strip_montage.py` (210), `_patch_ncc.py` (178) — 1594 lines across 5 flat siblings
- Problem: Import graph verified. `_patch_ncc` ← only `_solve_strips.py:18`.
  `_solve_strips` ← only `_compare_strips.py:31`, `_inspect_strips.py:28`.
  `_strip_montage` ← only `_compare_strips.py:32`, `_inspect_strips.py:29`. Only two
  edges leave the cluster: `_compare.py:239` imports `render_comparison_strips`, and
  `_commands/inspect.py:177` imports `parse_point_specs`/`render_inspect_strips`.
  The modules cross-reference each other in their docstrings
  (`_strip_montage.py:9`, `_compare_strips.py:18–19`, `_solve_strips.py:10`,
  `_inspect_strips.py:12`) — a hand-written substitute for the package boundary
  that isn't there. Layering is clean: scoring → strip solving → pixel layout → two
  consumers.
- Proposed fix: `strips/` with `_ncc.py`, `_solve.py`, `_montage.py`, `_compare.py`,
  `_inspect.py`, and `__init__.py` exporting exactly the three names the outside
  needs.
- Effort: low
- Risk: low — five renames plus two import-site edits;
  `tests/test_cli_inspect_strips.py:11,158` imports by path and needs updating.

**`draw_epipolar_visualization`: 509 lines, two input modes × three render modes**
- Location: `src/sfmtool/visualization/_epipolar_display.py:111–619` (module 619)
- Problem: The largest function in the package, and a 2×3 matrix of mutually
  exclusive paths in one body: image resolution 162–221, SIFT loading 222–248, then
  **feature acquisition** splitting sweep-matching (249–343) vs track-based
  (344–385); then **rendering** splitting rectified (397–457) / undistorted
  (458–517) / original-curve (518–579); then output assembly 580–619 branching on
  `side_by_side` and `save_which`. The `sweep_max_features is not None` test is
  re-asked at 249, 390, 436 and 614 — the mode is threaded through the body rather
  than resolved once. 15 parameters (111–129) is the same symptom.
- Proposed fix: extract `_resolve_image_pair`, `_feature_pairs_from_sweep`,
  `_feature_pairs_from_tracks`, and three `_render_*` functions each returning
  `(img1, img2)`; driver becomes ~60 lines. Bundle the render knobs into a dataclass.
- Effort: medium
- Risk: medium — the `sfm epipolar` output path; branches share local state
  (`rectification`, `rectification_safe`, `colors`, `feature_pairs`). Only visual
  output is verifiable, so lean on `tests/test_epipolar.py`.

**Six duplicated helper pairs, one already drifting**
- Location: across `src/sfmtool/`
- Problem: All verified:
  - `_apply_range_filter` — `_commands/to_colmap_bin.py:89–119` vs
    `_commands/to_nerfstudio.py:136–166`: **31 lines, `diff` shows exactly one
    differing line** (`print(` vs `click.echo(`).
  - `_classify_ratio` — `motion/report.py:245–257` vs
    `visualization/_discontinuity_display.py:209–219`: same four thresholds, but one
    reads `_RATIO_LOWER`/`_RATIO_UPPER` and the other hardcodes `0.75`/`1.33`.
    **Live drift hazard** — retuning the constants silently desyncs the JSON report
    from the rendered display.
  - `_load_gray` — `feature_match/_flow_matching.py:154–159` vs
    `motion/flow_stats.py:12–17`: identical 6 lines including
    `IMREAD_IGNORE_ORIENTATION`.
  - Sequence-descriptor naming — `_sfmr_naming.py:58–77` vs
    `feature_match/_run.py:385–427`: identical descriptor block *and* identical
    date-prefix + max-counter-scan logic, differing only in `.sfmr` vs `.matches`.
  - `_camera_centers` — `_embed_patches.py:197–213` vs `rig/panorama.py:56–65`:
    same `C = -Rᵀt` loop, different signatures.
  - `_rotation_angle_deg` — `_compare_fragments.py:332–335` vs
    `motion/recon_discontinuity.py:23–25`: **same name, different semantics**
    (angle of one transform vs angle between two quaternions). Not a merge
    candidate — a name collision that will mislead anyone grepping.
- Proposed fix: `_apply_range_filter` → `_filenames.py` taking an `echo` callable;
  `_load_gray` → a shared image-IO helper; have `_discontinuity_display` import
  `_classify_ratio` from `motion/report.py`; factor the naming logic into
  `next_dated_filename(...)`; unify `_camera_centers`. Rename one
  `_rotation_angle_deg`.
- Effort: low
- Risk: low — the naming unification is the only one with user-visible output
  (filenames) and must keep the exact format string.

**`feature_match/_run.py` holds matching orchestration and `.matches` merging**
- Location: `src/sfmtool/feature_match/_run.py` (960)
- Problem: 28–604 is the "run a matching job" concern, sharing the `_db_populate`
  imports at 20–25. 607–960 is `_run_merge` (**354 lines**), a different concern:
  read N `.matches` files, unify the image list, validate content hashes, remap pair
  indexes, dedupe by feature-index pair keeping lowest descriptor distance, merge
  two-view geometries. Verified: `_run_merge` calls **nothing** from the first 600
  lines and shares none of the module's top-level imports — it does all its own
  imports locally (620–621, 645). It lives here only because both are dispatched
  from `_commands/match.py`.
- Proposed fix: move 607–960 to `feature_match/_merge.py`.
- Effort: low
- Risk: low — one import edit; no shared state.

**`_commands/cluster_patches.py` is the only command module carrying its algorithm**
- Location: `src/sfmtool/_commands/cluster_patches.py` (337)
- Problem: The convention is stated in `feature_match/_run.py:7–9` ("Extracted from
  `_commands/match.py` so the command module stays a thin Click wrapper") and holds
  for 27 of 29 command modules. Here `_run_cluster_patches` (144–337, **194 lines**)
  is the implementation, importing `cv2`, `numpy`, `ThreadPoolExecutor`,
  `read_matches`/`write_sift` and `refine_cluster_patches` inside the function body
  and doing SIFT lookup, hash verification, threaded refinement and `.matches`
  writing inline. Measured non-Click helper lines across `_commands/`:
  `cluster_patches.py` 218, `solve.py` 180 (thin dispatchers), `inspect.py` 65,
  everything else under 60. The pipeline can't be called except through Click.
- Proposed fix: move 118–337 to a top-level `_cluster_patches.py`, matching the
  existing `_embed_patches.py`/`_patch_compaction.py` siblings — better still,
  group the three as one patch-processing topic.
- Effort: low
- Risk: low

**Flat modules whose only consumer is a single subpackage or sibling**
- Location: `src/sfmtool/_rectification.py` (212); `_compare.py` (818) + `_compare_fragments.py` (411) + `_compare_strips.py` (479)
- Problem: Importer graph built for all 26 flat `_*.py` modules. `_rectification.py`
  has exactly one production importer, `visualization/_epipolar_display.py:14` — a
  `visualization/` implementation detail at package top level, whose natural
  neighbour `check_rectification_safe` already lives in
  `feature_match/_geometry.py:98` and is imported two lines away. Separately,
  `_compare_fragments.py` and `_compare_strips.py` are imported by **nothing but
  `_compare.py`**, and `_compare.py` by nothing but `_commands/compare.py` — 1708
  lines behind a single CLI entry point, flat, while every other multi-module CLI
  topic already has a subpackage. `compare_reconstructions` (`_compare.py:57–364`,
  308 lines) is a 7-phase driver printing `[1/6]`…`[7/7]` — note the stale `[N/6]`
  labels at 107/113/117/128 against `[6/7]`/`[7/7]` at 203/220, itself a symptom.
- Proposed fix: `_rectification.py` → `visualization/`. Create `compare/` holding
  `core.py` + `fragments.py`, with strips coming from the `strips/` package; while
  moving, split the driver's phases and fix the `[N/6]` labels.
- Effort: low (rectification) / medium (compare)
- Risk: low — `tests/test_epipolar.py:323,347` import `sfmtool._rectification` by
  path and need updating.

---

## Tests

**`tests/` needs subdirectories — the patch cluster alone is 15 modules, 4,894 lines**
- Location: `tests/` (58 flat `test_*.py`, 20,153 lines; `tests/xform/` 21 modules and `tests/rust_bindings/` 33 already show the pattern, with a per-subdir `tests/xform/conftest.py`)
- Problem: The old report's `tests/patch/` proposal is now **15 modules / 4,894
  lines** (24% of the top-level suite): `test_patch_keypoint_subpixel.py` 613,
  `test_embed_patches_compaction.py` 580, `test_patch_normal_refine.py` 525,
  `test_warp_map_pose.py` 507, `test_embed_patches_command.py` 416,
  `test_render_patches.py` 405, `test_patch_view_selection.py` 337,
  `test_patch_keypoint_localization.py` 331, `test_refine_normals_keypoints.py` 292,
  `test_photometric_ransac.py` 245, `test_cluster_patches.py` 214,
  `test_consensus_atlas.py` 172, `test_patch_cloud.py` 108,
  `test_oriented_patch.py` 79, `test_warp_map_from_numpy.py` 70. Four more clusters
  sit flat: **rig/spherical** (6 modules, 1,914), **matching** (6, 2,164), **sift**
  (3, 1,288), **camrig** (3 + helper, 1,045).
- Proposed fix: `tests/patch/` first (biggest, and it unlocks the shared-helper fix
  below via `tests/patch/conftest.py`), then `tests/rig/`, `tests/matching/`,
  `tests/sift/`, `tests/camrig/`, each with `__init__.py` mirroring `tests/xform/`.
- Effort: low
- Risk: low — pure `git mv`; `pixi run test` uses `-n auto` with no path pinning and
  neither `scripts/coverage.sh` nor `ci.yml` hard-codes test paths.

**The conftest solve-retry loop is still duplicated verbatim**
- Location: `tests/conftest.py` (571) — `build_cluster_reconstruction` 214–249, `kerry_park_camrig_workspace_once` 517–559
- Problem: Confirmed still present, unchanged since 2026-06-23 (#11). Identical
  algorithm: rmtree `colmap_dir`, glob-unlink stale `.sfmr`,
  `seed = 42 if attempt == 1 else None`, solve, load, keep best by point count,
  break on threshold, canonicalize from a `_best*` stash. The code admits it —
  513–514 reads "Retry with a fresh randomization (mirroring
  ``build_cluster_reconstruction``)". They differ only in the solve callable and a
  trailing `_drop_camera_coincident_points`. Two copies of a flaky-solve retry
  policy means a tuning fix lands in one and not the other.
- Proposed fix: extract `_solve_with_retries(solve_fn, ..., max_attempts,
  random_seed=42)` taking a callable; each fixture passes a closure. The camrig
  version's `except RuntimeError: continue` becomes a flag.
- Effort: low
- Risk: medium — these are session-scoped fixtures gating most integration tests; a
  behaviour change re-flakes the suite against CI's non-deterministic GLOMAP.

**Patch test helpers copy-pasted across six test modules and three scripts**
- Location: `tests/`, `scripts/`
- Problem: `_load_images(recon)` — byte-identical cv2 BGR→RGB loader, comment and
  all — at `test_patch_keypoint_subpixel.py:29`,
  `test_patch_keypoint_localization.py:28`, `test_patch_normal_refine.py:24`,
  `test_refine_normals_keypoints.py:24`, `test_embed_patches_compaction.py:33`,
  `test_patch_view_selection.py:27`, plus `scripts/viz_keypoint_localization.py:50`,
  `scripts/viz_keypoint_localization_strips.py:54`,
  `scripts/viz_view_selection_strips.py:44` — **9 copies**.
  `_rotation_matrices(recon)` — **6 copies**. `_sample_point_ids` is a 7th
  near-duplicate.
- Proposed fix: land with the `tests/patch/` move — both go in
  `tests/patch/conftest.py`, mirroring `tests/xform/conftest.py`. For the scripts, a
  `scripts/_viz_common.py` (they already share `_infinity_first_sample`, `_chip`,
  `_compose`, `_label_for` too).
- Effort: low
- Risk: low

**`test_densify.py` is misnamed — only 19% of it is about densify**
- Location: `tests/test_densify.py` (686, 35 tests)
- Problem: Its own docstring concedes it ("densify command, feature matching module,
  and supporting utilities"). Breakdown: epipolar primitives 32–127,
  `get_intrinsic_matrix` 128–151, `GeometricFilterConfig` 152–210, rectified-sweep
  211–274, polar-sweep 275–515, `match_image_pair` 516–552 — **484 lines (32–552) of
  `feature_match/` tests**. Only 553–686 (134 lines) tests densify.
- Proposed fix: fold 32–127 into `test_epipolar.py`, move 152–552 to
  `tests/matching/test_sweep.py`, move 128–151 next to the camera tests; leaves
  ~140 lines of genuine densify coverage.
- Effort: low
- Risk: low — class-level moves, no fixture rewiring.

**`test_embed_patches_compaction.py` mixes four topics**
- Location: `tests/test_embed_patches_compaction.py` (580)
- Problem: Confirmed as the old report said, with ranges: compaction round-trips
  61–353, multi-round `embed_patches` pipeline 354–485, grazing-observation drop
  486–543, and five tests of `sfmtool._progress` utilities at **544–580**
  (`_progress_poll_loop`, `_poll_progress`, `ProgressCounter`) that have nothing to
  do with patches.
- Proposed fix: move 544–580 to `tests/test_progress.py`; fold the rest into
  `tests/patch/`.
- Effort: low
- Risk: low

---

## Top-level layout, scripts, docs

**`scripts/`: 9 of 20 files (2,309 lines, 52%) have zero inbound references**
- Location: `scripts/`
- Problem: Grepped every filename across the repo excluding `.git`, `target`,
  `.pixi`, `pixi.lock`. **Referenced:** `coverage.sh` (`pixi.toml:87`,
  `AGENTS.md:22`, `ci.yml:132`, `codecov.yml:6`), `ci_mem_sample.sh`
  (`ci.yml:177,187`), `benchmark_sift.py` (`pixi.toml:90`),
  `init_dataset_seoul_bull.sh` (`conftest.py:260`), `init_dataset_kerry_park.sh`
  (`conftest.py:421`), `init_dataset_dino_dog_toy.sh`
  (`benchmark_flow_matching.py:11`), `init_dataset_seattle_backyard.sh` (via the
  `init_dataset_*.sh` glob in `AGENTS.md:75`), `patch_crossval.py` (7 spec refs),
  `kdforest_vs_flann.py`, `validate_refine_subset.py`, `bench_normal_refine.py`.
  **Zero references:** `viz_keypoint_localization_strips.py` (488),
  `viz_view_selection_strips.py` (469), `viz_keypoint_localization.py` (420),
  `solve_crossval.py` (274), `sift_crossval.py` (221, mentioned only in
  `solve_crossval.py:6`'s docstring), `benchmark_flow_matching.py` (172),
  `benchmark_optical_flow.py` (128), `exp_plus_descent_localize_compare.py` (75),
  `benchmark_advect.py` (62).
  **Correction to the superseded report:** its scripts finding claimed
  `sift_crossval.py` and `solve_crossval.py` are "cited by specs". They are not —
  grepping `specs/` and `docs/` for all three crossval scripts returns hits only for
  `patch_crossval.py`. That claim should not be carried forward.
- Proposed fix: **delete** the four unanchored one-offs —
  `benchmark_optical_flow.py`, `benchmark_advect.py`, `benchmark_flow_matching.py`,
  `exp_plus_descent_localize_compare.py` (437 lines). **Keep but index** the five
  that name a spec and a test in their docstrings (three `viz_*`, two `*_crossval`,
  1,872 lines) — they are real dev tools, just invisible. Add a `scripts/README.md`
  table (script → what it inspects → which spec/test) and back-references from the
  specs their docstrings already name.
- Effort: low
- Risk: low — nothing in `pixi.toml`, `ci.yml` or `conftest.py` invokes any deletion.
  **Needs the author's confirmation before deleting** (carried over unresolved from
  the previous two audits).

**Stale prose after the `_sfmtool` migration, and a wrong test count in AGENTS.md**
- Location: `AGENTS.md:65`, `crates/sfmtool-py/src/lib.rs:18–19`, `src/sfmtool/sift/extract_sfmtool.py:6,41`, `specs/core/epipolar-curves.md:5,257`
- Problem: The PyO3 migration (#217) is structurally clean — ten uniform
  `install_submodule` calls, no duplicate registrations, no migration TODOs. The
  prose lagged: `lib.rs:18–19` claims "the wildcard flat surface is gone" while
  `src/sfmtool/__init__.py:6–15` still does
  `from sfmtool._sfmtool.<sub> import *` for all ten submodules;
  `extract_sfmtool.py:6,41` docstrings say `sfmtool._sfmtool.extract_sift` when the
  real import at `:168` is `from sfmtool._sfmtool.sift import extract_sift`;
  `specs/core/epipolar-curves.md:5,257` names `py_epipolar.rs`, which no longer
  exists (now `analysis/epipolar.rs`). Independently, `AGENTS.md:65` says "`tests/`
  — pytest, ~43 modules" — the actual count is **112** (58 top-level + 21 `xform/` +
  33 `rust_bindings/`), 32,021 lines. Also `spatial/kdtree.rs:570–577` defines
  `to_cow_slice`, a hand-rolled duplicate of the crate-wide `to_contiguous!` macro
  (`lib.rs:37–44`, used 151 times).
- Proposed fix: five one-line text corrections, the AGENTS.md count, and delete
  `to_cow_slice`.
- Effort: low
- Risk: low

**Retire `reports/2026-07-07-next-steps.md`**
- Location: `reports/2026-07-07-next-steps.md`
- Problem: All five implementation tasks carry `Done` status lines (items 1–3
  2026-07-08, items 4–5 2026-07-18). What remains is only sections A/B/C — design
  topics, two already labelled "carried forward" from the previous retired report.
  That is the AGENTS.md criterion verbatim: "the substantive findings are resolved
  and only minor or discussion-grade items remain."
- Proposed fix: delete, carrying topics A (camera bookmarks), B (`xform --crop`) and
  C (pose-aware per-tile source stacks) into the next `suggest-next-steps` snapshot
  or into issues.
- Effort: low
- Risk: low — git preserves history.
- Note: `reports/2026-06-13-perf-patch-normal-refinement.md` was assessed and
  **should stay** — §5 items 2, 3, 4, 5 and 7 are open with concrete measured
  numbers; it is a live technical backlog, not a stale snapshot.

---

## Carried-forward items now resolved

Two open findings from the 2026-07-07 report are resolved and are **not** carried
forward: `specs/drafts/` is clean (contains only `.keep` — the zup-migration doc was
already retired), and `reports/2026-06-11-dataset-init-scripts.md` no longer exists.

## Explicitly not flagged

Verified long-but-coherent, listed so a future audit doesn't re-litigate them:
`sfmtool-py/src/reconstruction/sfmr_reconstruction.rs` (1015 — one `#[pyclass]`, no
method over 80 lines), `colmap/io.py` (873 — deliberate mirror-image converters),
`analyze/summary.py` (668 — a uniform per-file-type dispatch table),
`visualization/_flow_display.py` (710 — modes already extracted),
`motion/recon_discontinuity.py` (799), `_densify.py` (765 — matches its docstring's
5-step pipeline), `geometry/rotation_init.rs` (909) and `pose_verification.rs` (701)
(staged pipelines whose top functions sit just under 250 lines),
`spherical/tile_rig.rs` (1091), `patch/cloud.rs` (968), `patch/keypoint_localize.rs`
(1008), `sfm-explorer/viewer_3d/mod.rs` (642 — a genuine coordinator),
`platform/windows.rs` (665), `spatial/kdtree.rs` (583), `sfmr-colmap/` (SQLite and
binary paths share nothing meaningful). The three recent splits landed cleanly:
`scene_renderer/upload/mod.rs` is a 20-line declaration file,
`point_track_detail/mod.rs` (232) an 82-line dispatcher plus shared consts, and
`reconstruction/data.rs` (598) is coherent apart from the affine-shape leftover
noted above. The `foo.rs` + `foo/tests.rs` pairing throughout is the documented
convention and is applied consistently.

---

## Top 3

1. **`bundle_adjust.rs` finite/mixed duplication** — ~529 lines of the numerical
   core exist twice, with 309 of `solve_lm`'s 328 lines verbatim in
   `solve_lm_mixed`. Highest value because every future fix to the damping ladder,
   the Schur accumulation or the convergence test currently has to be made twice,
   and a miss is silent. Gate it on a bit-identity test.
2. **`archive_io.rs` unification into a shared crate** — _done 2026-07-31 as
   `sfmtool-archive-io`; see the status note on the finding above._ The only finding that is
   simultaneously a duplication fix (4 copies), a dead-code removal
   (`read_uint128_array`), and a measurable performance fix (51 whole-column clones
   in the two big writers, already avoided in `camrig-format` but unreachable by the
   others). Low risk, 8 functions, and the four format crates have zero inter-crate
   deps today so there is no cycle to manage.
3. **`tests/` subdirectories, starting with `tests/patch/`** — 15 modules and 4,894
   lines, pure `git mv`, no CI or tooling depends on the paths, and it immediately
   unlocks deleting 9 copies of `_load_images` and 6 of `_rotation_matrices` via a
   `tests/patch/conftest.py`. Best effort-to-value ratio in the report.

Runner-up worth calling out because it is a latent crash rather than a smell: the
hard-coded `128` in `image_browser.rs::load_thumbnail` (726, 731) will panic on the
`assert_eq!` inside `ColorImage::from_rgba_unmultiplied` the moment `THUMBNAIL_SIZE`
changes, while the sibling copy in `point_track_detail/table.rs` reads the array
shape and would survive. One-line fix.
