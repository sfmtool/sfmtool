# Hygiene audit — 2026-08-08

Read-only structural survey of the whole codebase (Python `src/sfmtool/` + `tests/`,
Rust `crates/`, top-level layout) for oversized multi-concern files, duplication,
misleading names, directory smells, and dead code. Produced by the `audit-hygiene`
skill; supersedes the 2026-07-25 snapshot (retired in the same commit — **9 of its
findings are fully resolved and 2 partially**, and every still-open item is carried
forward below, re-measured against HEAD `bc68f2d`).

Every line count, line range and diff figure in this report was re-derived at
`bc68f2d`. Nothing was copied from the previous snapshot; where the two disagree,
this one is the measurement (four places where the old report was wrong are called
out inline, marked **Correction**).

**Headline:** the 2026-07-25 backlog was worked down along exactly the axis it
asked for. `bundle_adjust.rs`'s mirrored finite/mixed families are gone (1519 →
963, one `solve_lm`), and `bc68f2d` landed a nine-way pure-motion batch that
finished `covisibility.rs`, `kernels.rs`, `optical_flow/mod.rs`,
`camera/distortion.rs`, `reconstruction/data.rs`, `camera/remap.rs`'s `prof`
module, and the last five inline `mod tests` blocks in `sfmtool-core`. What is
left is a **narrower and harder** backlog: the size findings that survive are the
ones with real state-threading friction (`grow_reconstruction`,
`run_gpu_levels_prebuilt`, `image_browser::show`, `clone_with_changes`), and the
duplication findings that survive are the ones whose copies are *not* textually
identical — polar/sweep, write/verify/read, `inlier_fraction_of`, the six
`median`s. Those are the ones where a mechanical merge is unsafe and the drift is
therefore already happening.

Two things are new and neither is a size smell. First, **the workspace has never
run `cargo doc`**: 145 rustdoc warnings, 71 of them broken intra-doc links, and
the command appears in no pixi task, no CI job and no line of `AGENTS.md`.
Second, three files the 2026-07-25 report explicitly declined to flag have since
grown past the point where that judgement holds — `patch/keypoint_localize.rs`
1008 → 1484, and `patch/member_coherence.rs` (1553) did not exist at all.

**Scale:** 105,043 non-test Rust lines + 35,169 Python (`src/sfmtool/`), against
52,110 Rust `tests.rs` lines + 35,786 `tests/`. By subtree: `core/patch` 27.1k,
`core/features` 18.7k, `core/geometry` 16.3k, `core/camera` 12.4k,
`core/analysis` 8.7k, `core/spherical` 7.1k; `sfm-explorer` 22.8k, `sfmtool-py`
18.7k, the five format/archive crates 12.6k.

---

## Rust — `sfmtool-core` (geometry & features)

**`grow_reconstruction` is a 602-line function with five separable phases**
- Location: `crates/sfmtool-core/src/geometry/reconstruction_growth.rs` (1091), fn at
  **487–1088**
- Problem: Carried forward unchanged in substance; the line numbers shifted by 35
  when `numeric.rs` was extracted, and the function is **exactly the same 602 lines**
  it was on 2026-07-25 (`git log` shows no commit has touched the body since). It
  holds a nested `fn run_grow_ba` (591–681) and three closures, but the body still
  spans setup (533–586), the growth loop (~750–1000), and a focal-release finishing
  pass (~1004–1088, the re-triangulation comment at 1064 marks its last phase). The
  growth loop carries three independent sub-policies inline, including a verified
  force-accept with full state save/rollback that snapshots
  `quats`/`trans`/`points`/`ba_mask`/`since_ba` and restores on rejection — exactly
  the kind of code that must be readable in isolation, sitting 300+ lines into a
  function. The file's other helpers (`resect_one` 235, `fill_new_points` 318,
  `build_covisibility` 364, `ba_cluster_mask` 383, `build_result` 424) show the
  extraction pattern is established; it just stops at the main loop.
- Proposed fix: introduce an explicit `GrowState` struct (so the borrow friction that
  forced the closures disappears) and extract `try_force_accept`,
  `finish_with_focal_release` (1004–1088), `setup_growth_state` (533–586). Leaves
  ~150 lines of readable policy.
- Effort: medium
- Risk: medium — save/restore semantics are subtle (`posed_order.pop()` is
  conditional). Covered by `reconstruction_growth/tests.rs` (860 lines).

**`run_gpu_levels_prebuilt` is a 484-line method, two thirds of its file**
- Location: `crates/sfmtool-core/src/features/optical_flow/gpu/mod.rs` (740), method
  at **253–736**
- Problem: Unchanged since the last snapshot — the file has not been touched since
  `e4e4af1`. The method is 484 of the file's 740 lines and the *only* thing in it
  over 70. Phases are marked by the author's own `// ---` banners and cleanly
  separable: pool sizing (266–289), per-level uniform buffers + bind groups
  (290–501, ~212 lines), final-upsample resources (502–599), single-command-buffer
  encoding (600–~700), submit/wait/readback (~700–736). Everything is inlined
  because the resources must outlive the encode phase — solvable by returning owned
  structs. At this length the "single submit, single sync" design the method exists
  to express is buried.
- Proposed fix: extract `build_level_resources(...) -> Vec<LevelResources>` and
  `build_final_upsample_resources(...) -> Option<UpsampleResources>`, each owning its
  buffers/bind groups. Caller becomes ~120 lines.
- Effort: medium
- Risk: low-medium — wgpu resources must live until submit completes, so the
  extracted structs must bind to locals, not temporaries. `gpu/tests.rs` runs on the
  `noop` backend and covers this.

**Parallel plain/`_geometric` matcher families across `polar.rs` and `sweep.rs`**
- Location: `crates/sfmtool-core/src/features/feature_match/polar.rs` (844),
  `sweep.rs` (451)
- Problem: Five pairs implement the same algorithm twice, differing only by whether a
  geometric filter is applied to the sliding window. **Correction:** the previous
  snapshot's line ranges were all wrong — each one ran to the start of the *next*
  function's doc comment rather than to the function's closing brace, which inflated
  every span by 20–50 lines. Re-measured brace-to-brace, with `diff` counts:

  | plain | geometric | changed lines |
  |---|---|---|
  | `polar_mutual_best_match` 324–438 (115) | 705–841 (137) | **32** |
  | `polar_match_one_way` 231–286 (56) | 564–675 (112) | 68 |
  | `extend_for_wraparound` 154–203 (50) | 491–555 (65) | 27 |
  | `mutual_best_match_sweep` 124–185 (62) | 340–416 (77) | 23 |
  | `match_one_way_sweep` 45–98 (54) | 213–312 (100) | 56 |

  Roughly **280 duplicated lines**, and the headline pair is worse than the old
  report claimed: `polar_mutual_best_match` differs from its geometric twin by only
  32 lines of 115/137, so ~99 lines — the epipole computation, the polar transform,
  the angle-offset alignment, the dual sort and the mutual-consistency filter — are
  verbatim in both. The wraparound index mapping
  `((ext_idx as isize - n_prepended as isize).rem_euclid(orig_len2 as isize)) as usize`
  appears **byte-identically at `polar.rs:280` and `polar.rs:669`** and nowhere else
  in the crate; a bug there needs two fixes. The call sites
  (`feature_match/mod.rs:94,106,164`) are an `if use_geometric` branch passing
  near-identical argument lists.
- Proposed fix: make the filter an
  `Option<(&StereoPairGeometry, &GeometricFilterConfig)>` parameter on the plain
  functions; the `_geometric` variants become thin wrappers (the `pub` API must be
  preserved — `sfmtool-py/src/matching/sweep.rs` binds all four public ones).
- Effort: medium
- Risk: low-medium — the geometric path's remapping through `valid_indices` is the
  delicate part, and it is the reason the one-way pairs diverge most (68 and 56
  changed lines). `polar/tests.rs` and `sweep/tests.rs` cover both.

**Numeric-helper duplication that `numeric.rs` did not reach — including six `median`s with three different NaN policies**
- Location: `crates/sfmtool-core/src/geometry/{focal_vote,rotation_init,pose_verification,reconstruction_growth}.rs`,
  `analysis/{observation_adjacency,cluster_census}.rs`, `patch/{keypoint_localize.rs,normal_refine/consensus.rs}`
- Problem: The 2026-08-01 `geometry/numeric.rs` extraction (74 lines: `splitmix64`,
  `median`, `polar_rotation`, `rotation_angle`, `cam_at`) held — verified,
  `splitmix64` now has exactly **one** definition workspace-wide, down from six.
  Three things it did not reach:

  (a) **`PairAccum` / `ImageClusters` / `pair_correspondences`**, deferred on
  2026-08-01 as "structural, not a scalar helper". Measured now:
  `rotation_init.rs:152–166` and `focal_vote.rs:259–273` are the `PairAccum` struct
  plus its `mean_disp` impl and are **byte-identical over all 15 lines** (`diff`
  empty). The `type ImageClusters = Vec<Vec<(u32, [f64; 2])>>` alias is declared in
  both. `pair_correspondences` (`rotation_init.rs:218–243` vs
  `focal_vote.rs:277–301`) differs by **11 lines of 26/25** — every one of them is
  the presence or absence of the cluster-id output vector; the merge-join loop is
  line-for-line identical.

  (b) **`inlier_fraction_of`** (`pose_verification.rs:194` vs
  `reconstruction_growth.rs:194` — the same line number in both files, which is how
  close they are). 18 lines each; `diff` shows **5 changed lines**, all inside the
  `.filter(…)` closure: `pose_verification` inlines the projection while
  `reconstruction_growth` routes through its local `residual_norm`, which adds a
  non-finite guard on the world point. Both read a per-module `const INLIER_PX: f64
  = 3.0` (`pose_verification.rs:64`, `reconstruction_growth.rs:86`), and
  `pose_verification.rs:63`'s doc comment says "matches the growth kernel's
  `INLIER_PX`" — a comment doing a compiler's job.

  (c) **`median`, six times in `sfmtool-core`, with three incompatible NaN
  policies.** `geometry/numeric.rs:29` (`&[f64] -> f64`, `total_cmp`, empty → `0.0`);
  `geometry/focal_vote.rs:242` (`-> Option<f64>`, `total_cmp`);
  `analysis/observation_adjacency.rs:160` `median_in_place` (`partial_cmp` +
  `unwrap_or(Equal)`, `debug_assert!` on empty — so an empty slice indexes out of
  bounds in release); `analysis/cluster_census.rs:396` `median_in_place`
  (`total_cmp`, empty → `NaN`); `patch/keypoint_localize.rs:377`
  (`unwrap_or(Equal)`, empty → `NaN`); `patch/normal_refine/consensus.rs:59`
  (`partial_cmp().unwrap()` — **panics** on a NaN input). All six compute the same
  number on clean data. On a NaN they variously sort it last, sort it arbitrarily,
  or abort. Three of them are reached from the same photometric pipeline.
- Proposed fix: (a) promote `PairAccum` + `ImageClusters` + the merge-join into
  `geometry/numeric.rs` (or a `geometry/pair_tables.rs`), with
  `pair_correspondences` returning cluster ids and the focal-vote caller dropping
  them. (b) One `inlier_fraction_of(cam, r, t, obs, pts, inlier_px)` in
  `numeric.rs` taking the threshold as a parameter, over whichever residual
  definition is chosen deliberately — this one is **not** compiler-checked, so
  decide the NaN behaviour explicitly and test it. (c) One `median_in_place(&mut
  [f64]) -> f64` in a crate-level home with a documented NaN rule; the seventh
  copy, `sfmr-format/src/depth_stats.rs:171` `median_sorted`, is in a different
  crate and can stay.
- Effort: low (a, c) / medium (b — needs a decision, not a move)
- Risk: low for (a) and (c); medium for (b), since it changes a threshold test in
  two solvers.

**`keypoint_localize.rs` grew 47% since it was declared coherent; its main function is 424 lines**
- _New — this file was in the 2026-07-25 report's "Explicitly not flagged" list at
  1008 lines. It is now 1484._
- Location: `crates/sfmtool-core/src/patch/keypoint_localize.rs` (1484), fn
  `localize_patch_keypoints_with_basis` at **529–952**
- Problem: Four commits since the last snapshot added the consensus-basis cap
  (`399568a`, `f10359b`), candidate-track spawning (`ebd0920`) and stored-keypoint
  anchoring (`e58816f`), all into the same function. It is now the second-longest
  function in `sfmtool-core` (after `grow_reconstruction`) and 3.5× the next-longest
  in its own file (`register_tail`, 136). Its phases are already commented and
  cleanly bounded: search-resolution + cache geometry setup 537–649 (113 lines),
  consensus-basis pick 650–678, render-once caches 679–705, **the congealing round
  loop 706–918 (213 lines)**, Phase B tail hand-off 919–952. The tail phase is
  already extracted (`register_tail` at 1069) — the pattern exists and stops at the
  loop, the same shape as `grow_reconstruction`. Note the file *does* already have a
  populated subdirectory (`basis.rs`, `kernels.rs`, `params.rs`, `search.rs`,
  `prof.rs`, `tests.rs`), so there is nowhere for a reader to be surprised: the
  splitting convention is established and this function was simply never included in
  it.
- Proposed fix: extract `run_congealing_rounds(&mut states, …) -> RoundOutcome`
  (706–918) into `keypoint_localize/rounds.rs` beside the existing `search.rs`, and
  `setup_search_geometry(params) -> SearchGeometry` (537–~600) into `params.rs`,
  which already owns the tunables. Leaves a ~180-line driver.
- Effort: medium — the round loop mutates `states`, `iacc`, `keep_mask` and the
  cache buffers in place, so the extraction needs an explicit state struct rather
  than a signature.
- Risk: medium — this is the congealing kernel and its correctness is positional
  (the "integer read stays exact" invariant the module doc spends 30 lines on).
  `keypoint_localize/tests.rs` is 2276 lines, the largest test module in the
  workspace, and includes a reference-implementation equivalence test; run it either
  side.

**`member_coherence.rs` is the largest non-test file in `sfmtool-core` and its module doc names the split**
- _New — this file did not exist at the 2026-07-25 snapshot (added by `6d12156`)._
- Location: `crates/sfmtool-core/src/patch/member_coherence.rs` (**1553**), with a
  sibling `member_coherence/tests.rs` (1453)
- Problem: Its own module doc (lines 4–37) states the file as two operations plus
  two entry points: "[`member_zncc_matrix`] renders each member's patch and returns
  the `k×k` matrix …; [`decide_member_coherence`] reads a verdict off that matrix".
  The code follows that boundary exactly. **Matrix construction** — `member_zncc_matrix`
  510, `coarse_factors_for` 582, `fill_member_zncc` 600, `fill_scale` 726,
  `box_downsample` 792, `member_has_texture` 852 — is ~370 lines of rendering and
  image plumbing (`box_downsample` is a pure image-reduction kernel with nothing to
  do with coherence). **The decision rule** — `max_support_block` 880,
  `quantile_sorted` 936, `core_coherence` 978, `self_normalized_thresholds`
  1009–1030, `decide_member_coherence` **1155–1399 (245 lines)**, `core_deficit`
  1409 — is ~560 lines of pure policy over an already-computed matrix, with no
  rendering in it at all. `decide_member_coherence` alone carries a 124-line doc
  comment (1031–1154) documenting two sub-policies ("The self-normalized admission
  bar" at 1043, "Multi-scale exoneration" at 1090) that are separately named and
  separately tunable. The two halves share only the types.
- Proposed fix: `member_coherence/{matrix.rs, decide.rs}`, leaving `member_coherence.rs`
  as the params/types/entry-point file (49–509 plus the four `validate_*` /
  `*_from_reconstruction` functions at 1441–1553). The `member_coherence/` directory
  already exists for `tests.rs`, so this is purely additive.
- Effort: low — pure code motion; `decide.rs` reads only `MemberMatrix`'s public
  surface plus the `SELF_BAR_*` / `EXONERATION_*` consts, which are already `pub`.
- Risk: low — compiler-checked.

---

## Rust — `sfmtool-core` (camera)

**372 lines of hand-mirrored `CameraModel` ↔ `SfmrCamera` mapping in `intrinsics.rs`**
- Location: `crates/sfmtool-core/src/camera/intrinsics.rs` (913)
- Problem: Unchanged since the last snapshot. The first half (23–513) is a coherent
  13-variant enum plus accessors. The second half is string-keyed serialization:
  `impl TryFrom<&SfmrCamera> for CameraIntrinsics` (**534–660**, 127 lines) and
  `impl From<&CameraIntrinsics> for SfmrCamera` (**666–910**, 245 lines). Both
  directions hand-enumerate the same parameter names, so every literal appears an
  even number of times — re-counted at HEAD: `"principal_point_x"` and
  `"principal_point_y"` **24× each**, `"radial_distortion_k1"` **18×**,
  `"focal_length_x"` and `"focal_length_y"` **14× each**. Adding a camera model means
  editing two 100+ line matches; forgetting one direction yields a model that reads
  but cannot round-trip, with nothing structurally preventing it.
- Proposed fix: move both impls to `camera/intrinsics/sfmr_conv.rs`, then collapse
  the mirroring — one `fn params(&self) -> Vec<(&'static str, f64)>` +
  `fn from_params(name, &HashMap)` per variant so the name list is written once.
- Effort: medium
- Risk: medium — a transcription slip changes on-disk `.sfmr` camera parameters.
  Round-trip every variant through `intrinsics/tests.rs` before and after.

**`camera/remap.rs` still hosts the crate's most-used image containers**
> _Half of this finding landed in `bc68f2d`: the 95-line inline `pub mod prof` is
> now `camera/remap/prof.rs` (114 lines), matching the five sibling `prof.rs`
> modules. `remap.rs` 1279 → 1167. What follows is the open half only._
- Location: `crates/sfmtool-core/src/camera/remap.rs` (1167)
- Problem: `ImageU8` (52–172), `ImageU8Pyramid` (173–~250) and `ImageF32WithGrad`
  (810–~890) are general image containers, not resampling. Re-measured: they are
  named from **25 files** across `patch/`, `spherical/`, `sfmtool-py` and
  `sfm-explorer` (26 files reference `ImageU8` at all), yet their canonical path is
  `camera::remap::ImageU8`, which reads as neither a camera nor a remap concept. The
  remaining ~800 lines of samplers and `remap_*` entry points are genuinely
  coherent and match the filename.
- Proposed fix: move the three types to `camera/image.rs` with a `pub use` shim in
  `remap` so the 25 sites need not churn at once.
- Effort: medium
- Risk: low — path change only, shimmed.

---

## Rust — format crates

**`verify_matches` is a 719-line function — still the largest in the workspace**
- Location: `crates/matches-format/src/verify.rs` (843), fn at **125–843**
- Problem: The file shrank 882 → 843 when `raw_to_u32`/`raw_to_f64` moved into
  `sfmtool-archive-io`, but the function did not: it is **719 lines**, exactly as
  measured on 2026-07-25, and one of only two functions in the file
  (`structure_errors` at 16 is the other). It interleaves (a) *section hashing* —
  read each ZIP entry in lexicographic order, feed an `Xxh3`, compare to the stored
  digest — with (b) *structural validation* on the raw bytes it happens to keep
  alive. The two alternate six times, once per section. Because validation reads the
  *raw* buffers, the function holds `pairs_raw`, `match_counts_raw`, `match_fi_raw`,
  `feature_counts_raw` live across hundreds of lines and re-derives indices by hand
  (`pair_idxs[k * 2 + 1]`). Nothing in it can be unit-tested in isolation.
- Proposed fix: one `hash_<section>(...) -> (u128, RawBufs)` and one
  `check_<section>(raws, …, &mut errors)` per section; the entry point becomes an
  ~80-line sequence.
- Effort: medium
- Risk: medium — hashing depends on **entry read order**; reordering silently changes
  the computed content hash. Extract mechanically and keep
  `matches-format/src/tests.rs` (1468) green.

**Every archive entry name is written three times, and every format invariant twice**

> _Status (2026-08-08): Layer (1) done — `entries.rs` added to both format crates,
> one function per archive entry; `read`/`write`/`verify` now carry no name
> literals at all. Layer (2), the invariant merge, is untouched and stands._
>
> _Two corrections to the measurement below, both found while doing the work._
> _First, the `sfmr-format` count of **27 is an undercount; it is 33**, on this
> report's own metric (templates whose literal text is identical in all three of
> read/write/verify). The extraction regex used here required `[a-z_]` before
> the `/`, which silently skipped the entire `points3d/` section — that prefix
> contains a digit. `matches-format`'s 32 is correct, giving **65 triplicated
> templates**, not 59. Two adjacent numbers, since this metric is easy to
> conflate: normalizing interpolation-variable names (so `{r}` and
> `{patch_bitmap_r}` count as one template) makes it 34 + 32; and the **union**
> of distinct templates across the three files — which is what `entries.rs` has
> to cover, and so what its function count tracks — is 40 + 32._
>
> _Second, and more important: the claim in Top 3 that "a mistake cannot be
> subtle — a wrong name fails every round-trip test immediately" **was true
> before this change and is false after it**, which inverts the risk rating. Once
> all three paths read the name from one function, a wrong name is wrong
> *consistently*: writer and reader agree, every round-trip passes, and the only
> symptom is an archive no other build can open. This was confirmed empirically —
> corrupting `tracks/keypoints_xy` in `entries.rs` left all 48 `sfmr-format`
> tests green. At the time, only 3 of 33 `sfmr` names and **0 of 32** `matches`
> names were pinned by a literal assertion. Both crates therefore gained a
> `tests::entry_names_are_pinned` golden test covering every name (and both
> spellings of the three version-renamed ones); re-running the same corruption
> against it fails loudly. Do not do the remaining format extractions without an
> equivalent guard._

- Location: `crates/{sfmr,matches}-format/src/{write,read,verify}.rs`
- Problem: Two layers, both re-measured and both worse than last recorded.

  (1) **Entry-name templates are triplicated.** Extracting every
  `"<section>/<name>.…"` string literal from each file and intersecting:
  `sfmr-format` has **27 templates that appear identically in all three** of
  `read.rs` (28 distinct), `write.rs` (27) and `verify.rs` (28) —
  `"images/thumbnails_y_x_rgb.{image_count}.128.128.3.uint8.zst"`,
  `"tracks/observation_confidence.{observation_count}.uint8.zst"`, and 25 more.
  `matches-format` has **32 identical across all three** (32/32/36 distinct). A
  renamed entry, or a new dimension token, must land in three files or the archive
  silently fails verification. The old report estimated "~23 sfmr templates"; it is
  27, and it never measured the matches side.

  (2) **Invariants are implemented twice with different index math.** The writer
  validates typed `Array2` data and fails fast; the verifier re-validates the same
  rules from raw `&[u8]` and accumulates errors — character-identical error strings
  but `pairs.image_index_pairs[[k, 0]]` on one side and `pair_idxs[k * 2]` on the
  other. Measured over string literals longer than 25 characters (which are these
  files' error messages and entry templates): **45 of `matches-format`'s 90
  write-side literals also appear in verify.rs** (of 77 there), and **31 of
  `sfmr-format`'s 72** (of 49) — up from 29/67/47 at the last snapshot, i.e. the
  overlap is still growing.
- Proposed fix: per crate, add `entries.rs` with one fn per archive entry name and
  have write/read/verify call it — this alone kills the triplication and is nearly
  risk-free, since a wrong name fails every round-trip test immediately. Then hoist
  shared invariants into `check_*(…) -> Vec<String>` in `types.rs` over a view trait
  both paths implement.
- Effort: medium (entry names alone: low)
- Risk: low for entry names; medium for the invariant merge — the paths differ in
  `break` vs `break 'outer`, so verifier output ordering may shift.

**Four monolithic per-section entry points, three of which grew**
- Location: `sfmr-format/src/write.rs:96–661` (`write_sfmr_with_options`, **566**);
  `sfmr-format/src/verify.rs:17–546` (`verify_sfmr`, **530**);
  `sfmr-format/src/read.rs:43–553` (`read_sfmr`, **511**);
  `matches-format/src/write.rs:29–452` (`write_matches`, **424**)
- Problem: Re-measured brace-to-brace. Three of the four are *longer* than the last
  snapshot recorded (`write_sfmr_with_options` 533 → 566, `verify_sfmr` 505 → 530,
  `read_sfmr` 473 → 511) — the `observation_confidence` section added in `6a8ba0b`
  went into all three unchanged in shape. All four share the same structure: a linear
  run of `// === Section ===` blocks, each opening a hasher, emitting 4–12 entries in
  lexicographic order, digesting, storing. The boundaries are already marked by
  comments. Secondary: `matches-format/src/write.rs:536–812` `validate_dimensions`
  (**277**) and `sfmr-format/src/write.rs:807–1092` `validate_dimensions_with`
  (**286**, carrying `#[allow(clippy::too_many_arguments)]`).
- Proposed fix: one `write_<section>_section(...) -> Result<u128>` per section; entry
  points become ~60-line orchestrators. Split `validate_dimensions*` per section too,
  which also removes the `too_many_arguments` allow.
- Effort: medium
- Risk: medium — same content-hash ordering hazard. Do write and verify together so
  ordering stays in lockstep.

**`sfmtool-archive-io` enforces write-then-hash for binary entries only**
- _Carried forward from the 2026-07-31 status note on the (now resolved)
  `archive_io.rs` finding, which flagged this as the unfinished remainder._
- Location: `crates/sfmtool-archive-io/src/lib.rs` (`write_json_entry` at 202,
  `write_binary_entry` at 224, `write_binary_entry_hashed` at 243);
  `crates/{sfmr,matches}-format/src/write.rs`
- Problem: The extraction made the write/hash pairing structural for binary entries —
  measured: `sfmr-format/src/write.rs` and `matches-format/src/write.rs` now call
  `write_binary_entry_hashed` **28 and 25 times** and the unhashed
  `write_binary_entry` **zero times**. JSON did not follow. Those same two writers
  make **19 `write_json_entry` calls** and carry **14 hand-written
  `<section>_hasher.update(&bytes)` lines** paired with them (e.g.
  `sfmr-format/src/write.rs:219/225`, `280/286`, `318/324`, `363/369`, `372/378`,
  `463/469`, `590/596`). Those JSON bytes feed the same section digests as the binary
  ones, so the invariant that made the binary path safe — you cannot write an entry
  without hashing it — holds for 53 call sites and is a convention for 14. Separately,
  `sift-format` (4) and `camrig-format` (3) still call the unhashed
  `write_binary_entry` because they take a one-shot digest rather than a running
  hasher.
- Proposed fix: add `write_json_entry_hashed` (write + fold, mirroring the binary
  one) and a `write_binary_entry_digested` for the seven one-shot sites, then make
  the unhashed `write_binary_entry` private to the crate so the escape hatch closes.
- Effort: low
- Risk: low — mechanical, and any mistake fails the round-trip tests at once.

---

## Rust — `sfm-explorer`

**`image_browser.rs::show()` is a 539-line method holding four subsystems**
- Location: `crates/sfm-explorer/src/image_browser.rs` (774), `show()` at **175–713**
- Problem: Grew from 534 to 539 lines since the last snapshot. Nine parameters behind
  `#[allow(clippy::too_many_arguments)]` (attribute at 174). It inlines an animation
  player state machine (keyboard handling around 310–330, fps clock / loop /
  direction around 360–395), a pan/scroll controller, a virtualized thumbnail grid
  painter with layered highlight rules, and a complete minibar widget (barcode blit,
  play/pause + fps, viewport indicator, ticks). Genuinely four things. `AnimationState`
  (declared at 47, with `new`/`reset` at 61/71) and the minibar's own state
  (`new`/`invalidate` at 89/97) live in the same file, so the split has natural seams.
- Proposed fix: `image_browser/minibar.rs` takes the minibar state plus its drag-scrub
  and paint code; `image_browser/animation.rs` takes `AnimationState` plus the
  keyboard/clock block. `show()` drops to ~200 lines of layout+paint. Also resolve
  `enum PlayDirection` (38–44, `#[allow(dead_code)]` at 40): re-verified at HEAD, it
  is **matched** at 317 and 364 but the only value ever constructed is
  `PlayDirection::Forward` (line 64), so `Backward` is unreachable and the two match
  arms handling it are dead. Either wire up a reverse-play control or delete the
  variant.
- Effort: medium
- Risk: medium — minibar and grid share scroll-offset state; extraction must thread it
  explicitly or scrubbing/auto-scroll regress. The headless egui harness in
  `point_track_detail/tests.rs` (794 lines) is the model for covering this before the
  split — `ui_basic.rs` only runs on Windows/macOS.

**`draw_overlays` — five copy-pasted match arms, five copy-pasted colormap wrappers, and two disagreeing `error_color`s**
- Location: `crates/sfm-explorer/src/image_detail/overlay.rs` (568), fn at 19–~390;
  `crates/sfm-explorer/src/colormap.rs` (206)
- Problem: The `match overlay_mode` has five value-driven arms — `ReprojError`
  114–152, `TrackLength` 153–191, `MaxTrackAngle` 192–223, `DepthReliability`
  224–255, `ConditionNumber` 256–~290 — that are the same body (compute range → loop
  features → cull → `circle_filled` → yellow stroke if selected → `draw_colorbar`),
  differing only in value extractor, colormap fn, and label. Measured: `diff` of the
  `ReprojError` arm (39 lines) against the `DepthReliability` arm (32) shows **21
  changed lines**, and every one is the extractor, the colormap name or the legend
  string. ~175 lines collapse to one call. Related: `colormap.rs:60–118` holds
  `error_color`, `track_length_color`, `max_track_angle_color`,
  `depth_reliability_color` and `condition_number_color` whose bodies are the
  **byte-identical** six-line `t` normalization, differing only in which of the two
  colormap constants they `.sample(t)`. And there are still **two different
  `error_color` functions** — `colormap.rs:60` (`value, vmin, vmax`, samples
  `ERROR_COLORMAP`) vs `point_track_detail/metrics.rs:19` (`error: f32`, hard-coded
  0–2px green/yellow/red ramp) — so the two panels disagree on what a 1px error looks
  like.
- Proposed fix: one `draw_value_overlay(painter, features, extract, colormap, label)`;
  one `fn ramp(value, vmin, vmax, map: &Colormap)` with five one-line wrappers (or
  none). Reconcile the two `error_color` ramps and delete the loser.
- Effort: low
- Risk: low for the arm and wrapper collapse; medium for the colormap reconciliation —
  it changes on-screen colours in one panel, which is user-visible, and
  `point_track_detail/tests.rs:781` pins the current ramp.

**A latent thumbnail panic, a third RGB→RGBA expansion, one dead `clear()`, and six repeated struct literals**
> _Status (2026-08-14): Parts (a) and the RGB→RGBA half done; (b) and (c) remain
> open. All three expansions now go through one `src/texture.rs`
> (`rgb_to_color_image` + `thumbnail_color_image`), which reads the extent off the
> data — so `image_browser` holds no `128` at all and the abort is gone. The
> **`THUMBNAIL_SIZE` widening in the proposed fix turned out to be the wrong
> move** and was not done: that const is the GPU atlas cell size, and coupling the
> browser's CPU decode to a `scene_renderer` internal would have re-declared a fact
> the reconstruction already carries. Reading the shape needs no constant, so
> nothing outside `scene_renderer` names it and it stays `pub(super)`.
> Two things the finding did not mention, both fixed here: `build_barcode`'s own
> `const THUMB_H = 128` (721) also **bounded the row scan with the height**, so a
> non-square thumbnail would have read out of bounds; its band edges are now
> proportional to the real height. The deeper duplication this exposes is **not**
> fixed and is worth its own finding: `.sfmr` pins 128×128 independently in
> `sfmr-format`'s `entries.rs:158`, `read.rs:200–203` and `write.rs:890`, and
> `scene_renderer/gpu_types.rs:286` declares it a fourth time — four
> declarations of one fact, with no compiler tie between them._
- Location: `crates/sfm-explorer/src/image_browser.rs:757–773`,
  `point_track_detail/table.rs:362–384`, `image_detail/mod.rs` (691)
- Problem: Re-verified at HEAD; the previous snapshot's `app.rs` half of this finding
  is now obsolete (see below) but these are all still live.
  **(a) The latent panic.** `image_browser.rs:757` and `point_track_detail/table.rs:362`
  are near-identical `load_thumbnail` copies, but the browser one **hard-codes 128**
  three times (766, 770) while the table one reads `rgb_slice.shape()` (371–372).
  `THUMBNAIL_SIZE` is a const — but at `scene_renderer/gpu_types.rs:286` it is
  `pub(super)`, so `image_browser` **cannot reference it even if it wanted to**. If it
  ever changes, `ColorImage::from_rgba_unmultiplied` — which does
  `assert_eq!(size[0] * size[1] * 4, rgba.len())` — panics in the browser only. Not
  silent corruption: it aborts. The same RGB→RGBA expansion appears a third time at
  `image_detail/mod.rs:328–330`.
  **(b)** One never-called `pub fn clear` remains under `#[allow(dead_code)]`:
  `image_detail/mod.rs:571–572`. (The `point_track_detail` one the old report paired
  it with is now live — called from `mod.rs:239` — so that half is resolved.)
  **(c)** `image_detail/mod.rs` builds the same `FeatureOverlayState` literal **six
  times**: 359, 374, 411, 462, 476, 558.
- Proposed fix: unify the two `load_thumbnail`s on the shape-reading version and widen
  `THUMBNAIL_SIZE` to `pub(crate)`; extract one `rgb_to_rgba(slice) -> ColorImage`;
  delete the dead `clear()` or wire it into the reconstruction-swap path; add
  `FeatureOverlayState::new(..)`.
- Effort: low
- Risk: low — all mechanical; the `load_thumbnail` unification is a bug fix.

**`point_track_detail::metrics` is triangulation math imported by a sibling UI panel**
- Location: `crates/sfm-explorer/src/point_track_detail/metrics.rs`,
  `point_track_detail/mod.rs:41`, `image_detail/mod.rs`
- Problem: `compute_point_diagnostics` and `compute_max_pairwise_angle` are
  triangulation math, not panel code, but they live inside one panel and are imported
  by another. `point_track_detail/mod.rs:39–41` re-exports them with the comment
  "Re-exported at the old path: `image_detail` imports both as
  `crate::point_track_detail::<name>`" — introduced to keep an earlier split pure,
  which documents the smell rather than fixing it. The same file also holds the
  second `error_color` flagged above, which is the other symptom of the same
  misplacement.
- Proposed fix: promote to a crate-level `src/metrics.rs` and drop the compat
  re-export; move `error_color` to `colormap.rs` while there.
- Effort: low
- Risk: low

**One inline `#[cfg(test)] mod tests` block left in the workspace**
> _Down from six. `bc68f2d` moved the five in `sfmtool-core` to sibling `tests.rs`
> files (`consistency.rs`, `fronto_cache.rs`, `view_subset.rs`, `simd.rs`,
> `keypoint_localize/basis.rs`), and `sfmr-format/src/verify.rs`'s 27-line block went
> with `raw_to_u32` into `sfmtool-archive-io`. **`sfmtool-core` now has none.**_
- Location: `crates/sfm-explorer/src/platform/windows.rs:747–832` (86 inline lines of
  832)
- Problem: Carried forward from the 2026-07-25 snapshot, which deferred it because
  the explorer was mid-refactor. That refactor has since landed (scene-graph phases
  1–5, `4ea71c3` through `2374262`), so the reason to defer is gone. `platform/` has
  only `mod.rs` and `windows.rs`, so this would create the directory.
- **Explicitly still not flagged:** `sfmr-colmap/src/colmap_io/read.rs:511–528` (18
  lines) — a single focused test on the single private helper directly above it
  (`capped_capacity`); creating a directory for 18 lines costs more navigation than it
  saves.
- Proposed fix: `platform/windows/tests.rs`. Note it is Windows-only code, so the move
  compiles unverified on Linux — do it on a Windows checkout or lean on CI's `test-os`
  job.
- Effort: low
- Risk: low

---

## Rust — `sfmtool-py`

**`clone_with_changes` is a single 596-line function, and growing**
- Location: `crates/sfmtool-py/src/reconstruction/clone.rs` (768), fn at **74–669**
- Problem: Grew from 558 to **596 lines** (file 731 → 768) since the last snapshot.
  The *file* is coherent — its doc says it is the extracted body of
  `clone_with_changes`, and the only other item is `rebuild_observation_source` at
  675 — but the function is the only one over 150 lines in the crate. It is a large
  kwargs `match` followed by five sequential post-passes (image-count application,
  `rebuild_observation_source`, histogram resize, deferred `patch_bitmaps`, track
  rebuild, derived recompute). The arms are near-identical boilerplate, which is why
  this file holds **39 `PyValueError::new_err` sites**, the crate's highest.
- Proposed fix: split by field group into `apply_{point,pose,image,track}_fields`,
  each taking `&mut` builder state; keep the post-passes in the outer function. The
  existing `extract_ndarray!`/`extract_array1!`/`extract_array2!` macros already carry
  the per-arm boilerplate, so the split is mechanical.
- Effort: medium
- Risk: medium — the arms have ordering dependencies (the deferred `patch_bitmaps`
  pass exists precisely because it must run after the point fields); a naive split
  reorders them.

---

## Python — `src/sfmtool/`

**`xform/_arg_parser.py`: a 440-line argv loop plus four copy-pasted key=value parsers**
- Location: `src/sfmtool/xform/_arg_parser.py` (830)
- Problem: (a) `parse_transform_args` (**391–830**, 440 lines) is one `while` loop
  with an if/elif chain. The guard
  `if i + 1 >= len(args): raise click.UsageError("--X requires an argument")` is
  written out **20 times**. The optional-value tokenization block is copy-pasted
  four times — once at `--refine-normals` (480–497, with an 8-line comment explaining
  that it mirrors Click's `is_flag=False, flag_value=""` behaviour) and three more
  at `--refine-keypoints` (499–512), `--localize-keypoints` (514–527) and
  `--to-embedded-patches` (529–542), each carrying the comment "Optional value, same
  tokenization as --refine-normals" (lines 500, 515, 530).
  (b) `parse_refine_normals_params` (125–177), `parse_refine_keypoints_params`
  (194–247), `parse_localize_keypoints_params` (267–317) and
  `parse_to_embedded_patches_params` (328–372) are **~203 lines of the same ~50-line
  body**. `diff` of the first two shows 43 changed lines of 53/54, and inspection
  shows **every one of them is the option name, the `_*_KEYS` table name, or the
  transform class** — the control flow (split on comma, reject missing `=`, reject
  empty key, reject unknown key, reject duplicate, `caster is str` short-circuit,
  wrap `ValueError`) is line-for-line identical.
  **Correction:** the previous snapshot claimed these "have already drifted — only
  `_REFINE_NORMALS_KEYS` handling has the `caster is str` short-circuit (160–161)".
  That was wrong when written: at the 2026-07-25 report commit (`2bb9218`) the tree
  already had four `caster is str` occurrences, and at HEAD they are at 159, 228,
  302 and 359. There is no drift here; the duplication is the whole finding.
- Proposed fix: (b) first — collapse the four into
  `_parse_kv_params(option_name, keys, ctor, param)` with three-line public wrappers;
  that removes ~150 lines. Then (a) — replace the if/elif chain with a spec table
  `{flag: (arity, builder)}` plus one shared optional-value token reader.
- Effort: medium
- Risk: medium — this is the `sfm xform` CLI surface and the optional-value
  tokenization deliberately mirrors Click. Error text is user-visible and must be
  preserved exactly (it is asserted on); `tests/xform/` (21 modules, 4,491 lines) must
  be green either side.

**`sift/file.py` holds three concerns and its name describes one**
- Location: `src/sfmtool/sift/file.py` (877)
- Problem: Unchanged. File I/O proper — validation 228–249, `SiftReader` 250–304,
  write path 305–459, path resolution 460–592 — is ~360 lines and matches the name.
  The file also carries the **extraction pipeline** `image_files_to_sift_files`
  (**593–763**) + `image_files_to_sift_files_opencv` (764–793), which imports and
  drives the sibling `extract_colmap.py`/`extract_opencv.py` from inside the function
  body — the orchestration layer for the extract modules lives in the I/O module — and
  a **visualization** function `draw_sift_features` (**794–877**). Also stranded:
  xxh128 helpers (60–116) and pure feature geometry
  `compute_orientation`/`feature_size*` (117–173), neither of which touches a file.
- Proposed fix: `sift/extract.py` for 593–793 (becomes the peer of
  `extract_colmap.py`/`extract_opencv.py`/`extract_sfmtool.py`, and the deferred
  import becomes a normal one); `visualization/_sift_display.py` for 794–877;
  optionally `sift/geometry.py` for 117–173. Re-export from `sift/__init__.py`.
- Effort: medium
- Risk: low — pure moves behind existing re-exports.

**Strip modules form one closed pipeline — the `strips/` subpackage still earns itself**
- Location: `src/sfmtool/_solve_strips.py` (486), `_compare_strips.py` (479),
  `_inspect_strips.py` (241), `_strip_montage.py` (210), `_patch_ncc.py` (178) — 1,594
  lines across 5 flat siblings
- Problem: Import graph re-derived from `import` statements only (the previous
  snapshot's version was right; this confirms it at HEAD). Internal edges:
  `_patch_ncc` ← `_solve_strips.py:18`; `_solve_strips` ← `_compare_strips.py:31`,
  `_inspect_strips.py:28`; `_strip_montage` ← `_compare_strips.py:32`,
  `_inspect_strips.py:29`. Exactly **two edges leave the cluster**:
  `_compare.py:239` imports `render_comparison_strips` and `_commands/inspect.py:177`
  imports `parse_point_specs`/`render_inspect_strips` — both deferred, function-body
  imports. The modules cross-reference each other in their docstrings — a
  hand-written substitute for the package boundary that isn't there. Layering is
  clean: scoring → strip solving → pixel layout → two consumers.
- Proposed fix: `strips/` with `_ncc.py`, `_solve.py`, `_montage.py`, `_compare.py`,
  `_inspect.py`, and `__init__.py` exporting exactly the three names the outside needs.
- Effort: low
- Risk: low — five renames plus two import-site edits;
  `tests/test_cli_inspect_strips.py` imports `_solve_strips` and `_inspect_strips` by
  path and needs updating.

**`draw_epipolar_visualization`: 509 lines, two input modes × three render modes**
- Location: `src/sfmtool/visualization/_epipolar_display.py:111–619` (module 619)
- Problem: Unchanged. The largest function in the package, and a 2×3 matrix of
  mutually exclusive paths in one body: image resolution and SIFT loading, then
  **feature acquisition** splitting sweep-matching vs track-based, then **rendering**
  splitting rectified / undistorted / original-curve, then output assembly branching
  on `side_by_side` and `save_which`. The `sweep_max_features is not None` test is
  re-asked at **249, 390, 436 and 614** — the mode is threaded through the body rather
  than resolved once. **17 parameters** (111–129) is the same symptom.
- Proposed fix: extract `_resolve_image_pair`, `_feature_pairs_from_sweep`,
  `_feature_pairs_from_tracks`, and three `_render_*` functions each returning
  `(img1, img2)`; driver becomes ~60 lines. Bundle the render knobs into a dataclass.
- Effort: medium
- Risk: medium — the `sfm epipolar` output path; branches share local state
  (`rectification`, `rectification_safe`, `colors`, `feature_pairs`). Only visual
  output is verifiable, so lean on `tests/test_epipolar.py` (611).

**Six duplicated helper pairs — two of them now measurably divergent, not merely at risk**
- Location: across `src/sfmtool/`
- Problem: All six re-verified at HEAD, with `diff` counts. Two have crossed from
  "drift hazard" to "already drifted", which is a change from the last snapshot:
  - `_apply_range_filter` — `_commands/to_colmap_bin.py:89–119` vs
    `_commands/to_nerfstudio.py:136–166`: 31 lines, `diff` shows **exactly one
    differing line** (`print(` vs `click.echo(`).
  - `_load_gray` — `feature_match/_flow_matching.py:154–159` vs
    `motion/flow_stats.py:12–17`: 6 lines, **one differing line**, and it is the
    docstring ("as grayscale" vs "as grayscale uint8"). The `cv2.imread` call with
    `IMREAD_IGNORE_ORIENTATION`, the `FileNotFoundError` and the `cvtColor` are
    identical.
  - `_classify_ratio` — `motion/report.py:245–258` vs
    `visualization/_discontinuity_display.py:209–220`: **already divergent.** The
    report reads `_RATIO_UPPER = 1.0 / _RATIO_LOWER` = 1.3333…; the display hardcodes
    `1.33`. A normalized ratio in [1.33, 1.3333] is "acceleration" on screen and
    unclassified in the JSON. They also disagree on the empty case (`""` vs `None`)
    and on `elif` vs `if`.
  - Sequence-descriptor naming — `_sfmr_naming.py:20–77` vs
    `feature_match/_run.py:385–429`: **already divergent.** Both compute the same
    date prefix, the same `%Y%m%d-NN` max-counter scan over `iterdir()`, and the same
    `summarize_paths_by_sequence` → `RangeExpr` → `prefix_range` descriptor. But
    `_sfmr_naming` has an `else` branch producing `{first_name}-total-{N}-images`
    when the paths are not one sequence, and `_run.py` has no such branch — it
    silently emits no descriptor. Same intent, two behaviours, and only one of them
    is documented.
  - `_camera_centers` — `_embed_patches.py:240` vs `rig/panorama.py:56`: same
    `C = −Rᵀt` computation, different signatures (recon vs raw arrays).
  - `_rotation_angle_deg` — `_compare_fragments.py:332` vs
    `motion/recon_discontinuity.py:23`: **same name, different semantics** (angle of
    one transform vs angle between two quaternions). Not a merge candidate — a name
    collision that will mislead anyone grepping.
- Proposed fix: `_apply_range_filter` → `_filenames.py` taking an `echo` callable;
  `_load_gray` → a shared image-IO helper; have `_discontinuity_display` import
  `_classify_ratio` from `motion/report.py` (which resolves the 1.33 divergence in the
  direction of the constants); factor the naming logic into
  `next_dated_filename(base_dir, suffix, operation, image_paths)` used by both, which
  resolves the missing-`else` divergence; unify `_camera_centers`. Rename one
  `_rotation_angle_deg`.
- Effort: low
- Risk: low — the naming unification is the only one with user-visible output
  (filenames) and must keep the exact format string; note that fixing it *changes*
  `.matches` names in the multi-sequence case, which is the point.

**`feature_match/_run.py` holds matching orchestration and `.matches` merging**
- Location: `src/sfmtool/feature_match/_run.py` (960)
- Problem: Unchanged. 28–604 is the "run a matching job" concern, sharing the
  `_db_populate` imports at 20–25. **607–960 is `_run_merge` (354 lines)**, a
  different concern: read N `.matches` files, unify the image list, validate content
  hashes, remap pair indexes, dedupe by feature-index pair keeping lowest descriptor
  distance, merge two-view geometries. Re-verified: `_run_merge` calls **nothing**
  from the first 600 lines and uses **none** of the module's top-level imports — it
  does all its own imports locally (`datetime` at 616, `read_matches`/`write_matches`
  at 618, `._pairs` at 641). It lives here only because both are dispatched from
  `_commands/match.py`.
- Proposed fix: move 607–960 to `feature_match/_merge.py`.
- Effort: low
- Risk: low — one import edit; no shared state.

**`_commands/cluster_patches.py` is the only command module carrying its algorithm**
- Location: `src/sfmtool/_commands/cluster_patches.py` (347)
- Problem: The convention is stated in `feature_match/_run.py:7–9` ("Extracted from
  `_commands/match.py` so the command module stays a thin Click wrapper") and holds
  for 27 of 29 command modules. Here `_run_cluster_patches` (**151–347, 197 lines**)
  plus `_resolve_workspace` (125–149) is 223 lines of implementation below an 85-line
  Click declaration, importing `cv2`, `numpy`, `ThreadPoolExecutor`,
  `read_matches`/`write_sift` and `refine_cluster_patches` inside the function body
  and doing SIFT lookup, hash verification, threaded refinement and `.matches` writing
  inline. The pipeline cannot be called except through Click.
- Proposed fix: move 125–347 to a top-level `_cluster_patches.py`, matching the
  existing `_embed_patches.py`/`_patch_compaction.py` siblings — better still, group
  the three as one patch-processing topic.
- Effort: low
- Risk: low

**Flat modules whose only consumer is a single subpackage or sibling**
- Location: `src/sfmtool/_rectification.py` (212); `_compare.py` (818) +
  `_compare_fragments.py` (411) + `_compare_strips.py` (479)
- Problem: Importer graph rebuilt for all 26 flat `_*.py` modules; every one now has
  at least one production importer (the two dead sweep wrappers were deleted on
  2026-08-01), so this is a placement finding, not a dead-code one.
  `_rectification.py` has exactly one production importer,
  `visualization/_epipolar_display.py:14` — a `visualization/` implementation detail
  at package top level, whose natural neighbour `check_rectification_safe` already
  lives in `feature_match/_geometry.py` and is imported two lines away. Separately,
  `_compare_fragments.py` and `_compare_strips.py` are imported by **nothing but
  `_compare.py`**, and `_compare.py` by nothing but `_commands/compare.py` — 1,708
  lines behind a single CLI entry point, flat, while every other multi-module CLI
  topic already has a subpackage. `compare_reconstructions` (`_compare.py:57–366`,
  310 lines) is a 7-phase driver printing `[N/M]` labels that **still disagree with
  themselves**: `[1/6]` at 107, `[2/6]` 113, `[3/6]` 117, `[4/6]` 128, `[5/6]` 171,
  then `[6/7]` 203 and `[7/7]` 220. Five of seven say the wrong total.
- Proposed fix: `_rectification.py` → `visualization/`. Create `compare/` holding
  `core.py` + `fragments.py`, with strips coming from the `strips/` package; while
  moving, split the driver's phases and fix the `[N/6]` labels.
- Effort: low (rectification) / medium (compare)
- Risk: low — `tests/test_epipolar.py` imports `sfmtool._rectification` by path and
  needs updating.

---

## Tests

**`tests/` subdirectories: four clusters still flat**
> _Status (2026-08-11): Done — all four clusters landed as `tests/rig/` (6),
> `tests/matching/` (6), `tests/sift/` (3) and `tests/camrig/` (3), leaving 25 flat
> modules. `tests/_camrig_helpers.py` became `tests/camrig/conftest.py`. Four path
> citations outside `tests/` were updated in the same commit
> (`spherical/tile_rig/tests.rs` ×2, `specs/core/cluster-patch-refinement.md`,
> `specs/core/track-cluster-matching.md`), which is the hazard the risk note below
> called out._
>
> _One hazard the risk note **did not** call out, and which "pure `git mv`" is
> precisely wrong about: two modules computed the test-data directory by walking
> up from `__file__`, so the extra directory level silently repointed them at
> `tests/test-data/`. `_camrig_helpers.py`'s copy was noticed while moving it;
> `test_camrig_cp.py:18` held a **third** copy of the same constant and was not,
> and it failed 6 tests on the first run after the move. Both now read
> `TEST_DATA_DIR` (the root `conftest.py`'s own `__file__` walk), and
> `test_camrig_cp.py` imports `_IMAGE_DATA` from the package conftest rather than
> redefining it. A future cluster move should grep the moved set for `__file__`
> and `parent.parent`, not just for inbound path citations — a depth-dependent
> path inside a moved file is invisible to the citation grep._
>
> _Partially done. `tests/patch/` landed on 2026-08-01 and has since grown to 16
> modules / 5,850 lines (including its `conftest.py`). The four remaining clusters
> are unchanged._
- Location: `tests/` — **43 flat `test_*.py`, 14,670 lines**; `tests/patch/` (16),
  `tests/xform/` (21, 4,491), `tests/rust_bindings/` (39, 10,165) show the pattern
- Problem: Four coherent clusters still sit flat among unrelated modules:
  **rig/spherical** — `test_fisheye_rig.py` 332, `test_pano2rig.py` 384,
  `test_panorama.py` 399, `test_spherical_tile_rig.py` 231,
  `test_per_spherical_tile_source_stack.py` 474, `test_sphere_points.py` 94 = **1,914
  lines**;
  **matching** — `test_match.py` 365, `test_densify.py` 702, `test_cluster_matching.py`
  319, `test_matches_clusters.py` 278, `test_pairs_from_matches.py` 211,
  `test_flow.py` 380 = **2,255**;
  **sift** — `test_sift_extract.py` 366, `test_sift_file.py` 711,
  `test_sift_workspace.py` 211 = **1,288**;
  **camrig** — `test_camrig.py` 445, `test_camrig_cp.py` 303,
  `test_camrig_resolve.py` 260, plus the loose helper module `_camrig_helpers.py` 37 =
  **1,045**. That last one is the clearest signal: a shared-helper module already
  exists at the top level and would simply become `tests/camrig/conftest.py`.
  Together, 6,502 lines (44% of the flat suite) belong in four directories.
- Proposed fix: `tests/rig/`, `tests/matching/`, `tests/sift/`, `tests/camrig/`, each
  with `__init__.py`, mirroring `tests/xform/` and `tests/patch/`.
- Effort: low
- Risk: low — pure `git mv`; `pixi run test` is `pytest -n auto` with no path pinning,
  and neither `scripts/coverage.sh` nor `ci.yml` hard-codes test paths. Watch for spec
  and `scripts/viz_*.py` docstrings citing old paths — that is what caught out the
  `tests/patch/` move.

**The conftest solve-retry loop is still duplicated verbatim**
- Location: `tests/conftest.py` (571) — `build_cluster_reconstruction` **154–255**,
  `kerry_park_camrig_workspace_once` **480–562**
- Problem: Confirmed still present and unchanged. Identical algorithm: rmtree
  `colmap_dir`, glob-unlink stale `.sfmr`, `seed = 42 if attempt == 1 else None`,
  solve, load, keep best by point count, break on threshold, restore from a `_best*`
  stash, glob-unlink again. The code admits it — `conftest.py:513–514` reads "Retry
  with a fresh randomization (mirroring ``build_cluster_reconstruction``)". They
  differ only in the solve callable, `max_attempts` (6 vs 10), the ranking key
  (`(image_count, point_count)` vs point count with a completeness gate), and a
  trailing `except RuntimeError: continue`. Two copies of a flaky-solve retry policy
  means a tuning fix lands in one and not the other.
- Proposed fix: extract `_solve_with_retries(solve_fn, *, max_attempts, rank,
  accept, random_seed=42)` taking a callable; each fixture passes a closure. The
  camrig version's `except RuntimeError: continue` becomes a flag.
- Effort: low
- Risk: medium — these are session-scoped fixtures gating most integration tests; a
  behaviour change re-flakes the suite against CI's non-deterministic GLOMAP.
- **Nondeterminism status, re-measured (this supersedes the 2026-08-01 evidence note
  on this finding).** The previous snapshot recorded three consecutive local runs of
  `tests/patch/` each failing a *different* test. **That does not reproduce at
  `bc68f2d`.** Nine runs of `pytest tests/patch` were executed for this audit: six
  serial and three under `-n auto` (the flag `pixi run test` actually uses).
  **Eight were 160/160 green.** The only non-pass in any of them was a
  session-fixture `ERROR` on `test_embed_patches_default_output_path` in the very
  first, cold run — which was also the only one run with `-x`, so it stopped after
  11 tests; that same test passes in isolation, and every one of the eight
  subsequent runs (five serial, three xdist) was fully green. So the retry loops are
  currently doing their job and the three named tests
  (`test_embed_patches_subpixel_lk_round_trips`,
  `test_stored_keypoints_at_reprojection_match_centered`,
  `test_select_views_infinity_admitted_are_in_front`) did not fail once. Two caveats
  before treating this as resolved: this is a single machine, and one cold-start
  fixture error in nine runs is not zero. The finding stands on the duplication
  alone; the priority bump the 2026-08-01 note asked for should be **withdrawn**.

**Patch test helpers: the `scripts/` half is still open**
> _The `tests/` half landed on 2026-08-01 — 6 copies of `_load_images`, 5 of
> `_sample_point_ids` and 3 of `_rotation_matrices` collapsed into
> `tests/patch/conftest.py`._
- Location: `scripts/viz_keypoint_localization.py:50`,
  `scripts/viz_keypoint_localization_strips.py:54`,
  `scripts/viz_view_selection_strips.py:44`
- Problem: The three `viz_*` scripts still each carry their own `_load_images` copy
  (the byte-identical cv2 BGR→RGB loader, comment and all), and they also share
  `_infinity_first_sample`, `_chip`, `_compose` and `_label_for`. They cannot import
  `tests/patch/conftest.py`, so the tests-side fix did not reach them.
- Proposed fix: `scripts/_viz_common.py`. Note this is entangled with the
  `scripts/` question below — if the three `viz_*` scripts are kept, give them a
  shared module; if they are deleted, this finding goes with them.
- Effort: low
- Risk: low

**`test_densify.py` is misnamed — only 12% of it is about densify**
- Location: `tests/test_densify.py` (702, 38 tests)
- Problem: Worse than the last snapshot's 19%, because the sweep-wrapper deletion
  moved ~123 lines of matcher scaffolding *into* this module. Its own docstring
  concedes the mixture. Re-derived breakdown: `TestEssentialMatrix` 33,
  `TestFundamentalMatrix` 68, `TestEpipole` 94, `TestRectificationSafe` 110 —
  epipolar primitives, 33–128; `TestGetIntrinsicMatrix` 129–152 — a camera test;
  `TestGeometricFilterConfig` 153–177, then the local matcher scaffolding
  (`_make_geometric_params` 178, `_geometric_args` 212, `_as_positions` 244,
  `_as_descriptors` 248, `mutual_best_match_sweep` 252, `polar_mutual_best_match`
  286) and `TestRectifiedSweepMatching` 335, `TestPolarSweepMatching` 379,
  `TestMatchImagePair` 532, `TestPruneImagePairs` 569 — **feature_match tests,
  153–614**. Only `TestDensifyCLI` 615 and `TestDensifyE2E` 657–702 — **88 lines** —
  test densify.
- Proposed fix: fold 33–128 into `test_epipolar.py`, move 129–152 next to the camera
  tests, move 153–614 to `tests/matching/test_sweep.py` (taking the scaffolding with
  it), leaving ~90 lines of genuine densify coverage.
- Effort: low
- Risk: low — class-level moves, no fixture rewiring.

**`test_embed_patches_compaction.py` mixes four topics**
- Location: `tests/patch/test_embed_patches_compaction.py` (674)
- Problem: Re-derived at HEAD (the previous snapshot's ranges were stale by the move
  and by subsequent growth). Compaction round-trips 50–401, multi-round
  `embed_patches` pipeline 402–579, grazing-observation drop 580–623, and **five
  tests of `sfmtool._progress` utilities at 624–674** (`_progress_poll_loop`,
  `_poll_progress`, `ProgressCounter`) that have nothing to do with patches — the
  module imports them at line 25, the only non-patch import in the file.
- Proposed fix: move 624–674 (plus the line-25 import) to `tests/test_progress.py`.
- Effort: low
- Risk: low

---

## Top-level layout, scripts, docs, tooling

**`cargo doc` is in no check, and the workspace has 145 rustdoc warnings**

> _Status (2026-08-08): Done — the gate is wired and the backlog is at zero._
> _Re-measured at HEAD: 140 real warnings + 5 per-crate summary lines = the 145
> counted here (71 unresolved intra-doc links, 54 private-item links, 6 redundant
> explicit targets, 4 unparseable code blocks, 3 unclosed HTML tags, 2
> function/module ambiguities). Both `cargo doc --workspace --no-deps` and
> `… --document-private-items` are now **0 warnings**._
>
> _Three decisions. **(1) The gate runs `--document-private-items`**: the plain
> build's warning set was verified to be a strict subset of the private build's
> (72 of 140 sites, no site unique to the plain build), the private build is the
> one anyone working on the crates actually reads, and it checks the links written
> inside private modules — which is most of this workspace's cross-referencing.
> **(2) Warnings become errors via `[workspace.lints.rustdoc]` in the root
> `Cargo.toml`**, with `[lints] workspace = true` added to all nine crates, rather
> than `RUSTDOCFLAGS="-D warnings"` in the CI step. The tradeoff: the manifest form
> also fails a bare `cargo doc` typed by hand or run by an IDE, not just the pixi
> task and CI, and it needs no env plumbing; the cost is that it must name lints
> explicitly, so a newly-added warn-by-default rustdoc lint will not be denied
> until someone adds it. The seven warn-by-default lints are listed;
> `rustdoc::all` was rejected because it also switches on allow-by-default lints
> (`missing_crate_level_docs`, `private_doc_tests`, `unescaped_backticks`) — a
> separate and much larger decision. `rustdoc::unportable_markdown` is omitted:
> rustc has removed it, and naming a removed lint is itself a warning.
> **(3) All 54 private-item links became plain code spans**; no item's visibility
> was changed. Every target is a deliberate implementation detail — a private
> sibling module (`normal_refine::{params, znorm, search, …}`, `optical_flow::gpu::
> {context, variational, …}`), a private constant (`COINCIDENT_CAMERA_FRACTION`,
> `DISTORTION_EPS`, `ANCHOR_CAP`), or a `pub(crate)` / `pub(in crate::patch)`
> helper (`build_level_context`, `Support`, `RefineTile`, `view_jacobian`).
> Re-exporting any of them would enlarge the public API to satisfy a doc link,
> which is a real semantic change; a code span renders identically in published
> docs and only costs the hyperlink in the internal build._
>
> _Of the 71 unresolved links, none were deleted: 16 in `camera/distortion.rs` plus
> ~20 elsewhere took a `Self::` prefix, sibling-module refs took `super::`,
> cross-crate refs took the full path (`sfmtool_core::…`, `crate::camera::…`), and
> the two `is both a function and a module` ambiguities took `()`. Six links named
> items that no longer exist and were repointed at the real successor:
> `KdTree2d::nearest_k_within_radius` → `PointCloud::…`, `ImagePyramid::level_in_full`
> → `Self::level`, `build_pyramids_from_arrays` → `build_pyramids_from_cameras`,
> `WarpMap::from_cameras` fully-qualified, and `gpu::variational`'s "standalone
> `refine`" → `GpuFlowContext::run_dis_and_variational`. Unescaped generics
> (`Vec<f32>`, `Vec<f64>`, `Py<PyAny>`) and Python-docstring brackets
> (`list[int]`, `list[bytes]`, `image_starts[i]:image_starts[i+1]`) are now code
> spans._
>
> _One thing was deliberately not "fixed": the 4 unparseable code blocks are all
> PyO3 docstrings whose indented `Args:` / `Returns:` continuation paragraphs are
> Markdown indented code blocks by accident of Python docstring convention, which
> rustdoc then tries to compile as Rust. De-indenting them would damage the
> `help()` output these docstrings exist for, so each of the four items carries a
> commented `#[allow(rustdoc::invalid_rust_codeblocks)]` instead — visible and
> per-item, not a crate-wide blanket, so the next one is a conscious choice._
>
> _Wiring: `doc` task in `pixi.toml`'s `[tasks]`, a "Cargo doc" step in `ci.yml`'s
> `lint` job (verified `pixi run -e lint doc` builds — the `lint` env has the
> Python interpreter `sfmtool-py`'s PyO3 build script needs), and `AGENTS.md`'s
> "Task completion checks" under "Rust changes" plus a "Things that can surprise
> you" entry. The gate was confirmed to fail: a deliberate `Self::compute_svd_typo`
> in `warp_map.rs` made `pixi run doc` exit 101 with
> `error: unresolved link`, then was reverted._

- _New. Surfaced as a question by the previous round; measured here for the first
  time._
- Location: `AGENTS.md:31–39` ("Task completion checks"), `.github/workflows/ci.yml`
  (the `lint` job at 90–100), `pixi.toml:45–92` (`[tasks]`)
- Problem: Grepping `cargo doc`, `rustdoc` and `broken_intra_doc` across
  `.github/`, `pixi.toml`, `AGENTS.md`, `Cargo.toml`, all nine `crates/*/Cargo.toml`
  and `scripts/` returns **zero hits**. The lint job runs `ruff format --check`,
  `ruff check`, `cargo fmt --check` and `cargo clippy --workspace --all-targets -D
  warnings`; clippy does not check doc links. So nothing in the repo has ever built
  the docs. Running it (`pixi run -e test cargo doc --workspace --no-deps
  --document-private-items`) **exits 0** and emits **145 warnings**:
  - **71 unresolved intra-doc links** — `sfmtool-core` 44, `sfmtool-py` 19,
    `sfm-explorer` 5, `sfmr-colmap` 3. The worst single file is
    `camera/distortion.rs` with 16, all of them method references written
    `[`distort`]` / `[`distort_ray`]` / `[`ray_to_pixel`]` inside an inherent-impl
    doc where rustdoc needs `Self::`. Also 4 in `sfmtool-py/src/patches/args.rs`, 3
    each in `normal_refine/znorm.rs`, `normal_refine/params.rs`, `camera/warp_map.rs`,
    `camera/distortion/ray_grid.rs`.
  - **54 "public documentation links to private item"** — these render as plain text
    for anyone reading published docs.
  - **6 redundant explicit link targets, 4 unparseable Rust code blocks, 3 unclosed
    HTML tags** (`Vec<f32>` / `Vec<f64>` / `Vec<PyAny>` written unescaped in doc
    comments, at `optical_flow/gpu/context.rs:211`,
    `sfmtool-py/src/geometry/rot_quaternion.rs:10`, `sfmtool-py/src/flow/warp.rs:388`
    — these silently swallow the rest of the line in rendered output), and **2 "is
    both a function and a module"** ambiguities
    (`sfmtool_core::geometry::bundle_adjust` at
    `sfmtool-py/src/geometry/bundle_adjust.rs:5`, and `resect_translation`).

  This matters more here than in most repos: this codebase's doc comments are load-
  bearing — modules routinely spend 30–120 lines explaining an invariant and cross-
  linking the spec — and every one of those 71 links is a navigation aid that already
  does not work.
- Proposed fix: add `doc = "cargo doc --workspace --no-deps"` to `pixi.toml`'s
  `[tasks]`, add `#![warn(rustdoc::broken_intra_doc_links)]` (or
  `-D warnings` via `RUSTDOCFLAGS`) to the CI lint job, and add the command to
  `AGENTS.md`'s task-completion checks under "Rust changes". Fix the existing 145
  before turning the gate on, or the first commit after it lands is red. The 16 in
  `distortion.rs` are one mechanical `Self::` prefix.
- Effort: low for the tooling; medium for the 145-warning backlog (mostly mechanical)
- Risk: low — documentation only. The one judgement call is the 54 private-item links:
  some should become plain code spans, some argue their target should be `pub(crate)`
  and re-exported.

**`scripts/`: 9 of 20 files (2,309 lines, 52%) have zero inbound references**
- Location: `scripts/`
- Problem: Re-grepped every filename across the repo excluding `.git`, `target`,
  `.pixi`, `pixi.lock`, `reports/` and `scripts/` itself (the last two matter — the
  previous audit's own report cites all 20 filenames, which makes a naive grep say
  everything is referenced).
  **Referenced from outside `scripts/`:** `coverage.sh` (`pixi.toml`, `AGENTS.md`,
  `ci.yml`, `codecov.yml`), `ci_mem_sample.sh` (`ci.yml`), `benchmark_sift.py`
  (`pixi.toml`), `init_dataset_seoul_bull.sh` and `init_dataset_kerry_park.sh`
  (`tests/conftest.py`), `patch_crossval.py` (3 specs), `kdforest_vs_flann.py`,
  `validate_refine_subset.py`, `bench_normal_refine.py` (1 spec each).
  Referenced only from within `scripts/`: `init_dataset_dino_dog_toy.sh` (from
  `benchmark_flow_matching.py:11`) and `init_dataset_seattle_backyard.sh` (covered by
  the `init_dataset_*.sh` glob in `AGENTS.md:82`) — both fine.
  **Zero references anywhere:** `viz_keypoint_localization_strips.py` (488),
  `viz_view_selection_strips.py` (469), `viz_keypoint_localization.py` (420),
  `solve_crossval.py` (274), `sift_crossval.py` (221, mentioned only in
  `solve_crossval.py:6`'s docstring), `benchmark_flow_matching.py` (172),
  `benchmark_optical_flow.py` (128), `exp_plus_descent_localize_compare.py` (75),
  `benchmark_advect.py` (62) — **2,309 lines exactly**, unchanged from the last
  snapshot.
- Proposed fix: **delete** the four unanchored one-offs —
  `benchmark_optical_flow.py`, `benchmark_advect.py`, `benchmark_flow_matching.py`,
  `exp_plus_descent_localize_compare.py` (437 lines). **Keep but index** the five that
  name a spec and a test in their docstrings (three `viz_*`, two `*_crossval`, 1,872
  lines) — they are real dev tools, just invisible. Add a `scripts/README.md` table
  (script → what it inspects → which spec/test) and back-references from the specs
  their docstrings already name.
- Effort: low
- Risk: low — nothing in `pixi.toml`, `ci.yml` or `conftest.py` invokes any deletion.
  **Needs the author's confirmation before deleting** (carried over unresolved from
  the previous three audits — this is its fourth appearance, which is itself the
  signal: either decide, or move the "keep" set into `scripts/README.md` and stop
  re-reporting it).

**`AGENTS.md`'s Python module count is off by 59%**
- Location: `AGENTS.md:43` and `:69`
- Problem: `AGENTS.md:43` says "`src/sfmtool/` — Python package (**~93 modules**)".
  `git ls-files 'src/sfmtool/*.py'` returns **148** (134 excluding `__init__.py`).
  `AGENTS.md:69` says "`tests/` — pytest, **~114 modules**"; the actual count is
  **119** — that one was corrected to 113 on 2026-08-01 and has drifted by six since,
  which is tolerable, but the `src` figure was never corrected and predates several
  subpackages. This is the same class of finding the 2026-07-25 report closed for the
  test count; the sibling number was missed. Also: `AGENTS.md`'s "Structure at a
  glance" does not mention the top-level `skills/` directory (four skills, symlinked
  into `.claude/skills/` — the symlinks are deliberate and correct, but a reader
  looking at the tree has nothing telling them so).
- Proposed fix: two number edits plus one bullet. Better: state the counts as
  "~150 modules" style round numbers, or drop them — a figure that must be re-derived
  every audit is a maintenance liability, and this is the second audit in a row to
  correct one.
- Effort: low
- Risk: low

**Retire `reports/2026-07-07-next-steps.md`**
- Location: `reports/2026-07-07-next-steps.md`
- Problem: Carried forward unactioned. All five implementation tasks carry `Done`
  status lines (items 1–3 2026-07-08, items 4–5 2026-07-18). What remains is only
  sections A/B/C — design topics, two of them already labelled "carried forward" from
  the previous retired report. That is the `AGENTS.md` retirement criterion verbatim:
  "the substantive findings are resolved and only minor or discussion-grade items
  remain".
- Proposed fix: delete, carrying topics A (camera bookmarks), B (`xform --crop`) and
  C (pose-aware per-tile source stacks) into the next `suggest-next-steps` snapshot or
  into issues.
- Effort: low
- Risk: low — git preserves history.
- Note: `reports/2026-06-13-perf-patch-normal-refinement.md` was re-assessed and
  **should stay** — §5 items 2, 3, 4, 5 and 7 remain open with concrete measured
  numbers; it is a live technical backlog, not a stale snapshot.

---

## Carried-forward items now resolved

Verified fixed at `bc68f2d` and **not** carried forward:

1. **`bundle_adjust.rs` finite/mixed mirroring** (`1fa50dd`). The file is 963 lines
   and contains exactly one of each: `residual_norms_depths` (274), `reestimate_points`
   (312), `robust_cost` (384), `solve_lm` (432), `bundle_adjust_staged` (843). No
   `_mixed` or `_finite` symbol survives anywhere in the crate.
2. **`covisibility.rs` three specs + 793 inline test lines** (`bc68f2d`). Now
   `covisibility.rs` 501 + `covisibility/displacement.rs` 337 + `selection.rs` 145 +
   `tests.rs` 788.
3. **`optical_flow/mod.rs` container types** (`bc68f2d`). 958 → 485, with
   `flow_field.rs` 248, `image.rs` 119, `params.rs` 141.
4. **`camera/distortion.rs` three concerns** (`bc68f2d`). 1350 → 890, with
   `distortion/ray_grid.rs` 309 and `distortion/pinhole_fit.rs` 187.
5. **`keypoint_subpixel/kernels.rs` announced-but-unmade split** (`bc68f2d`). Now a
   39-line declaration/re-export file over `kernels/render.rs` 801 and
   `kernels/score.rs` 200.
6. **`reconstruction/data.rs` affine-shape leftover** (`bc68f2d`). 630 → 509, with
   `data/affine_shape.rs` 139.
7. **Inline `#[cfg(test)] mod tests` blocks.** Six at the last snapshot, one now —
   `sfmtool-core` has none, and `sfmr-format/src/verify.rs`'s block left with
   `raw_to_u32`. Only `sfm-explorer/src/platform/windows.rs:747` remains (carried
   forward above as its own small finding).
8. **`splitmix64` / `polar_rotation` / `rotation_angle` / `cam_at` duplication**
   (`37a591e`). `geometry/numeric.rs` is 74 lines and `splitmix64` has exactly one
   definition workspace-wide. (The parts `numeric.rs` did not reach are carried
   forward above as their own finding.)
9. **Dead sweep-matching wrappers** (`82e89fc`) and the `archive_io.rs` ×4 duplication
   (`e4e4af1`) and the `HashMap` reproducibility bug — all confirmed gone. Re-checked
   the whole flat-module import graph: **every** `src/sfmtool/_*.py` module now has at
   least one production importer.

Partially resolved, with only the named remainder carried forward:
`camera/remap.rs` (the `prof` half landed; the image containers did not),
`tests/` subdirectories (`tests/patch/` landed; four clusters did not), and the patch
test helpers (`tests/` landed; `scripts/` did not).

---

## Explicitly not flagged

Verified long-but-coherent at HEAD, listed so a future audit does not re-litigate
them. Sizes are current, since several have moved:

- `crates/sfmtool-core/src/geometry/bundle_adjust.rs` (963) — post-merge, one
  numerical kernel. `solve_lm` is 410 lines and stays that way: it is a single
  Levenberg–Marquardt iteration (Schur accumulation, 12-step damping ladder, LU
  solve, scatter-back) whose phases are not independently meaningful.
- `crates/sfmtool-core/src/patch/view_selection.rs` (1018), `cluster_refine/mod.rs`
  (876), `cluster_refine/kernels.rs` (799), `keypoint_subpixel.rs` (861),
  `keypoint_subpixel/kernels/render.rs` (801), `spherical/photometric_ransac.rs`
  (839), `spherical/per_tile_source_stack.rs` (866), `spherical/tile_rig.rs` (1091),
  `patch/cloud.rs` (968), `features/sift/scale_space.rs` (881),
  `geometry/rotation_init.rs` (889), `geometry/pose_verification.rs` (668) — all
  checked function-by-function; the longest function in any of them is 218 lines
  (`cluster_refine/mod.rs:545`) and most are under 120.
- `crates/sfm-explorer/src/scene_graph/mod.rs` (868) — new since the last snapshot
  and already well decomposed: 20 items, largest function ~90 lines, with the tree
  rendering split into `show_node` / `show_node_header` / `node_context_menu` /
  `show_tint_menu` / `show_align_menu` / `show_cameras_group` / `show_camera_rows` /
  `show_points_group`.
- `crates/sfm-explorer/src/state.rs` (678), `viewer_3d/mod.rs` (747),
  `platform/windows.rs` (832 — largest fn 93) — coordinators, not grab-bags.
- `crates/sfm-explorer/src/app.rs` (785). **This is a change from the last
  snapshot**, which flagged `run_egui_pass` as "212 lines inlining ~135 lines of
  chrome". The viewport-HUD work (`2856e36`) deleted the View menu, and
  `run_egui_pass` is now **180 lines** (525–704) of which the File menu is ~60 and
  the demo modal ~35. A `menu.rs` extraction is still defensible but no longer
  earns a finding on its own.
- `crates/sfmtool-py/src/reconstruction/sfmr_reconstruction.rs` (1050) — one
  `#[pyclass]`; longest method 63 lines (`triangulation_diagnostics`).
- `crates/sfmtool-core/src/patch/keypoint_localize.rs`'s `#[cfg(test)] fn znorm_core`
  (324) and `fn template_zncc` (352) **look** like duplicates of
  `keypoint_subpixel/kernels/score.rs:19` and `ecc_score`, and are not: both are
  gated `#[cfg(test)]` and their doc comments say they are the reference
  implementations the production accumulator is scored against. Deliberate, and the
  cross-references are written down. Same for
  `patch/normal_refine/znorm.rs::znormalize_into`.
- `src/sfmtool/_embed_patches.py` (826) — `embed_patches` is 374–826, but **155 of
  those lines are the docstring** (through ~530) and the body is a linear staged
  pipeline whose stages are already extracted (`_refine_subpixel`,
  `_drop_grazing_observations`, `_cull_by_localizability`, `_localizations_from_recon`,
  `compact_to_embedded_patches`). Round 1 (543–728) and rounds 2..N (729–~820) look
  parallel but do genuinely different work (round 1 alone does `select_views` +
  `localize_keypoints`); they are not a merge candidate.
- `src/sfmtool/colmap/io.py` (873 — deliberate mirror-image converters),
  `analyze/summary.py` (668 — a uniform per-file-type dispatch table),
  `visualization/_flow_display.py` (710 — modes already extracted),
  `motion/recon_discontinuity.py` (799), `_densify.py` (765 — matches its docstring's
  5-step pipeline), `_undistort_images.py` (596).
- `crates/sfmr-colmap/` — the SQLite and binary paths share nothing meaningful; the
  18-line inline test block in `colmap_io/read.rs` stays (see above).
- `crates/sfmtool-archive-io/` — the extraction landed cleanly: 263 lines of library
  over 315 of tests, and all four format crates consume it.
- `specs/drafts/` is empty except `.keep`; `skills/` ↔ `.claude/skills/` symlinks are
  intentional and git-tracked as symlinks; the `foo.rs` + `foo/tests.rs` pairing is
  applied consistently across all nine crates.

---

## Top 3

1. **Wire `cargo doc` into the checks, then clear the 145 warnings.**
   Best effort-to-value ratio in the report by a wide margin. The tooling change is
   three lines (a `pixi.toml` task, a CI lint step, an `AGENTS.md` bullet); the
   backlog behind it is 145 warnings that are mostly one mechanical edit each (16 of
   them are a missing `Self::` in one file). The value is not the warnings — it is
   that this codebase's doc comments carry design rationale that nothing else
   records, 71 of their cross-links are already broken, and there is currently **no
   mechanism at all** by which anyone would find out. Every other finding in this
   report is about code that at least a compiler is watching. Zero runtime risk.

2. **Entry-name templates → `entries.rs`, per format crate.**
   > _Status (2026-08-08): Done. See the corrections on the finding itself — the
   > sfmr count was 33 rather than 27, and the "a mistake cannot be subtle"
   > justification below is **wrong**: centralizing the names removes the
   > round-trip's ability to catch a typo, because both sides then share it. A
   > golden `entry_names_are_pinned` test in each crate supplies the guard that
   > claim assumed. The knock-on argument still holds — the two larger format
   > findings are now cheaper and safer, since the section splits can no longer
   > misspell or reorder an entry name._
   27 identical string templates in `sfmr-format` and 32 in `matches-format`, each
   written three times across `read.rs`/`write.rs`/`verify.rs`. This is the cheapest
   real duplication fix left: one function per entry name, no logic moves, and a
   mistake cannot be subtle — a wrong name fails every round-trip test in
   `tests.rs` (1,757 and 1,468 lines respectively) immediately. It also de-risks the
   two larger format findings behind it (`verify_matches`'s 719 lines and the four
   monolithic section entry points), because once the names are centralized, the
   section splits stop being able to silently reorder or misspell an entry — which is
   the exact hazard that makes those two "medium risk" today. Do it first and the
   others get cheaper.

3. **`tests/rig/`, `tests/matching/`, `tests/sift/`, `tests/camrig/`.**
   > _Status (2026-08-11): Done. All four directories created; 18 modules plus
   > `_camrig_helpers.py` moved. The `testpaths = ["tests"]` / no-path-pinning
   > assumption held — `pyproject.toml`, `ci.yml` and `scripts/coverage.sh` needed
   > no edits. The predicted citation problem was real but smaller than the
   > `tests/patch/` move: two spec files and one Rust doc comment, no
   > `scripts/viz_*.py` hits. "Pure `git mv`" was the one wrong call — two moved
   > modules derived the test-data path from `__file__`, which the extra level
   > breaks; see the finding above. The `test_densify.py` split this unblocks is
   > **not** done and remains open as its own finding._

   6,502 lines and 18 modules out of the 43 still flat, pure `git mv`, no CI or
   tooling depends on the paths, and the pattern is already established three times
   over (`xform/`, `patch/`, `rust_bindings/`). `tests/camrig/` is nearly free and
   immediately removes a wart: `tests/_camrig_helpers.py` is a shared-helper module
   sitting loose at the top level that simply becomes `tests/camrig/conftest.py`.
   Doing `tests/matching/` also unblocks the `test_densify.py` split, which is the
   single most misnamed file in the repository — 88 of its 702 lines test densify.
   The one thing to watch, learned the hard way on the `tests/patch/` move: specs and
   `scripts/viz_*.py` docstrings cite test paths, so grep for them in the same commit.

Runner-up, called out because it is a latent crash rather than a smell: the
hard-coded `128` in `image_browser.rs::load_thumbnail` (766, 770) will panic on the
`assert_eq!` inside `ColorImage::from_rgba_unmultiplied` the moment `THUMBNAIL_SIZE`
changes, while the sibling copy in `point_track_detail/table.rs` reads the array
shape and would survive. Two-line fix, plus widening `THUMBNAIL_SIZE` from
`pub(super)` to `pub(crate)` so the browser can actually name it.

> _Status (2026-08-14): Done, without the widening — see the finding itself. The
> viewer now derives the extent from the loaded reconstruction, so no second
> declaration of 128 was needed; `THUMBNAIL_SIZE` stays `pub(super)`. Scope grew by
> one real bug the runner-up had not spotted: `build_barcode` used the same
> constant as **both** the band divisor and the row-scan width bound._

---

## Appendix — design topics carried forward from the retired next-steps snapshot

`reports/2026-07-07-next-steps.md` was retired in the commit that created this
report. All five of its implementation tasks carried `Done` annotations (items
1–3 on 2026-07-08, items 4–5 on 2026-07-18), leaving only the design topics
below — the AGENTS.md retirement criterion verbatim ("the substantive findings
are resolved and only minor or discussion-grade items remain").

These are **not hygiene findings** — they are unspecced feature proposals, kept
here only because AGENTS.md says to fold unfinished items into the next
regenerated snapshot rather than let them evaporate. They are reproduced
unchanged from the retired report; each premise was re-verified against the
tree on 2026-08-08:

- **A** — `specs/gui/gui-viewport-navigation.md:666` still reads
  `- [ ] Save/restore camera positions`, and no bookmark state exists in
  `sfm-explorer`. Still open. (Note the explorer has moved a long way since
  the topic was written — scene-graph phases 1–5 landed between 2026-07-25 and
  2026-08-08 — so the sketch's file references are worth re-checking before
  acting on it.)
- **B** — no crop transform in `src/sfmtool/xform/`, no `--crop` in
  `_commands/xform.py` or `specs/cli/xform-command.md`. Still open.
- **C** — `PerSphericalTileSourceStack` still exposes only
  `build_rotation_only`; `specs/core/per-spherical-tile-source-stack.md:6–7`
  still says "The pose-aware variant described under 'Pose-aware variant' is
  still future work". Still open. Its sketch depends on
  `WarpMap::build_with_pose_impl`, which **does** exist
  (`camera/warp_map.rs:276`) — that half is the "DONE" sub-item at spec line
  666, so only the per-tile consumer is missing.

### A. Camera bookmarks (save/restore named viewpoints) in SfM Explorer — carried forward

- **Motivation:** `specs/gui/gui-viewport-navigation.md` § "Future
  Enhancements" still lists `- [ ] Save/restore camera positions` (re-verified
  unimplemented in this round's spec audit). For inspection and before/after
  comparison it's invaluable to jump back to a saved vantage point —
  generalizing Camera View Mode's per-image snapshots to user-named arbitrary
  viewpoints.
- **Sketch:** A bank of bookmark slots storing full orbit-camera state
  (`position`, `orientation`, `target_distance`, `fov`, `world_up`). Number
  keys recall, modifier+number stores; optional dock panel listing named
  bookmarks. Persist per-reconstruction in a sidecar JSON keyed by `.sfmr`
  path. Recall reuses the existing animated-transition path (slerp + ease)
  built for Alt+click target moves and `switch_camera_view`.
- **Where it would live:** New `specs/gui/gui-camera-bookmarks.md`; state in
  `viewer_3d/camera.rs` + `state.rs`; persistence beside the file-load path.
- **Open questions:** Sidecar vs. `.sfmr` metadata vs. global app state? Named
  vs. numbered slots? Capture overlay/selection state or pose only?

### B. `sfm xform --crop` (3D bounding-volume crop) — carried forward

- **Motivation:** Spatial filtering in `xform` is still by image or by point
  statistic (re-verified: no crop transform exists). There's no "keep only the
  points inside this region" — the most natural operation for isolating one
  object or carving background clutter before a Nerfstudio export.
- **Sketch:** A `CropTransform` taking an axis-aligned box
  (`--crop xmin,ymin,zmin,xmax,ymax,zmax`) or centre+radius sphere, dropping
  outside points and remapping observations through the existing point-removal
  path. Cameras stay, or optionally drop when they observe nothing. Natural
  synergy with topic A: the GUI could emit the crop numbers from an
  interactive selection.
- **Where it would live:** New filter in `src/sfmtool/xform/`, `--crop` in
  `_commands/xform.py`, new subsection in `specs/cli/xform-command.md`.
- **Open questions:** Coordinate frame (raw vs. physically-scaled units)?
  Auto-drop empty cameras? Box vs. sphere vs. oriented box, and how does a
  user author the numbers without a viewer?

### C. Pose-aware per-tile source stacks (parallax-correct panoramas)

- **Motivation:** `specs/core/per-spherical-tile-source-stack.md` explicitly
  defers a pose-aware `build_with_pose` — today's `build_rotation_only`
  assumes scene-at-infinity, so `sfm panorama` is only correct for
  near-concentric captures. Real handheld rigs translate; nearby geometry
  ghosts across tile seams and the photometric RANSAC has to spend its
  clusters absorbing parallax instead of exposure/occlusion differences. This
  is the biggest quality ceiling on the panorama pipeline the last month of
  work built up.
- **Sketch:** Add a depth proxy per tile — the reconstruction already carries
  triangulated points, so a per-tile median depth (or small plane fit) from
  points inside the tile's frustum gives each tile a projection surface;
  `build_with_pose` warps each source through the existing pose-aware
  `WarpMap::from_cameras_with_pose` path onto that surface instead of the
  infinity rotation-only warp. Tiles with too few points fall back to
  rotation-only. The RANSAC/consensus and batched-atlas stages are unchanged —
  they just receive better-registered stacks.
- **Where it would live:** Extend `specs/core/per-spherical-tile-source-stack.md`
  (the "Pose-aware variant" stub) + a `--parallax` flag in
  `specs/cli/panorama-command.md`; implementation in
  `crates/sfmtool-core/src/spherical/` reusing `camera/warp_map.rs`.
- **Open questions:** Per-tile plane vs. per-tile depth constant vs. coarse
  mesh? What point density is enough, and is the fallback per-tile or global?
  Does the consensus scoring need a depth-aware validity mask where the proxy
  surface is wrong? How much of the byte-identical batching contract
  (`tile_index_base` reseeding) survives a per-tile geometry input?

