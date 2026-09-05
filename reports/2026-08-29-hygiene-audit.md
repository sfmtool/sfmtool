# Hygiene audit — 2026-08-29

Read-only structural survey of the whole codebase (Python `src/sfmtool/` + `tests/`,
Rust `crates/`, top-level layout) for oversized multi-concern files, duplication,
misleading names, directory smells, and dead code. Produced by the `audit-hygiene`
skill against HEAD `94d3739`. **Supersedes `reports/2026-08-08-hygiene-audit.md`,
retired in the same commit** — of its 38 findings, 15 are fully resolved and 3
partially; every still-open item is carried forward below, re-measured here rather
than copied.

Every line count, line range and diff figure was re-derived at `94d3739`. Where this
report and the retired one disagree, this one is the measurement.

**Scale:** 118,972 non-test Rust lines + 68,148 Rust test lines; 35,617 Python
(`src/sfmtool/`) + 38,677 Python test lines. Non-test Rust is up 13% in the 21 days
since the last snapshot (49 commits, 526 files, +51,564/−12,306).

**Headline: the last snapshot's backlog was worked, but its _acquittals_ went stale
faster than its findings did.** The 2026-08-08 report closed a lot — the rustdoc gate
and its 145 warnings, the `camera_models!` registry, `member_coherence`, the
polar/sweep merge, the four `tests/` subpackages, the last five inline `mod tests`
blocks in `sfmtool-core`, and the fifteen-way 128×128 thumbnail edge (now one
`sfmr_format::THUMBNAIL_SIZE` with a compile-time assert tying it to `sift_format`).
What it got wrong was the "Explicitly not flagged" list. Five files it checked
function-by-function and cleared have since grown 29–91%:
`camera/distortion/kernels.rs` 793 → **1515**, `camera/distortion.rs` 890 → **1233**,
`sfm-explorer/state.rs` 678 → **997**, `scene_graph/mod.rs` 868 → **1218**,
`geometry/bundle_adjust.rs` 963 → **1375** (with `solve_lm`, declared "stays that
way" at 410 lines, now 514). An acquittal in a report is a measurement with a
three-week half-life, and this codebase moves faster than that.

The second theme is **regression against fixes that landed inside the audit
window.** `#297` ("one median, one NaN rule, replacing seven copies") landed on
2026-08-14 and wrote its NaN and even-count contract into `numeric.rs`'s module
docs. `#318` landed `geometry/resect_images.rs` on 2026-08-27 carrying a **new
private `median`** that disagrees with that contract on even counts. `sfmtool-py`'s
`patches/args.rs` exists to hold the bindings' shared string→enum parsers; **five**
of its eight intended callers hand-inline the parse instead, byte-identically, error
string and all. Neither is a slow drift — both are new code written beside the
shared thing without using it, which is what happens when the shared thing has no
enforcement behind it.

---

## Rust — `sfmtool-core` (geometry)

**`resect_images.rs` reintroduced three primitives the crate already owns, and all
three disagree with their counterpart**
> _Status (2026-08-29): **Done.** All three local copies deleted; `resect_images.rs`
> 1227 → 1209. `median` → `crate::numeric::median_in_place`; `orthonormalized` moved
> to `geometry/numeric.rs` beside `polar_rotation`, with the sign convention that
> separates the two written down on both; `angle_between` → a new `pub(crate)`
> radian primitive in `camera::report`, with the existing `angle_between_deg`
> reduced to `angle_between(a, b).to_degrees()` so there is one body. The lower-middle
> rule was **not** preserved: `numeric.rs`'s averaging rule is the crate contract and
> the spec (`specs/gui/resect-image.md:201`) only says "median", so nothing had to be
> reconciled._
>
> _**The finding understated the angle case, and the new test says by how much.**
> It called `acos`-of-dot merely less accurate. At ε = 1e-8 rad the dot product
> rounds to `1.0` in `f64` and `acos` returns **exactly 0** — the measurement is not
> degraded, it is gone. `camera/report/tests.rs::angle_between_is_accurate_far_below_a_pixel`
> pins the accurate form to 1e-9 relative across ε = 1e-3 … 1e-11 **and** asserts the
> naive form collapses, so a future one-liner cannot quietly come back._
>
> _Two claims in the finding were checked and are worth correcting. (1) The
> unguarded-empty `median` was called a latent panic; it is not reachable and never
> was — both call sites filter empties — so the only live defect was the even-count
> rule. (2) "Needs the resect tests run either side and any moved threshold
> understood" anticipated moved numbers. **Nothing moved**: the whole Rust workspace
> is green at 2,025 tests (`sfmtool-core` lib 1,439, the 16 resect tests among them),
> the Python suite at 2,215 passed / 1 skipped after `maturin develop --release`
> (including the 10 `test_resect_images_rust_bindings.py` cases, of which
> `test_same_input_gives_the_same_answer` is a bit-identical determinism check), and
> `cargo fmt`, `clippy --workspace --all-targets` and the `pixi run doc` gate are all
> clean. That is not evidence the change is inert — it is evidence the suite never
> covered the divergence, which is why both corrected behaviours now have tests that fail under
> the old rule (`scene_scale_averages_the_two_middles_of_an_even_population` is built
> with an even population at both levels of the reduction precisely so the two rules
> cannot agree)._
>
> _**Not done: the mechanical guard.** The Top 3 write-up asked for a clippy
> `disallowed_methods` entry or a grep-based test making `numeric.rs` the only median
> structurally. Neither fits: clippy cannot see a private local `fn`, and no
> source-scanning test exists anywhere in this workspace to follow. Left open rather
> than inventing a mechanism for one finding — but this is the second median
> regression in six weeks, so it wants a decision rather than a third._
- Location: `crates/sfmtool-core/src/geometry/resect_images.rs` (1227, new since the
  last snapshot) — `median` **1188**, `angle_between` **1142**, `orthonormalized`
  **1148**
- Problem: The file landed on 2026-08-27 (#318). Each of its three tail helpers has
  an existing owner in this crate, and each differs from it in a way that is a
  behaviour change rather than a style choice:
  - **`median` (1188) vs `crate::numeric::median_in_place` (`numeric.rs:78`).**
    `numeric.rs` was created by `234546b` (#297) on 2026-08-14 specifically to end
    seven copies of this, and its module docs state the contract: `total_cmp`
    ordering, NaN when empty, and **even counts average the two middle values**.
    The new copy is `values.sort_by(f64::total_cmp); values[(values.len()-1)/2]` —
    the **lower** of the two middles, documented in its own doc comment as a
    deliberate choice ("so the answer is a member of the sample rather than a mean
    of two") with no reference to the module that says otherwise. It also has no
    empty guard and would panic; that is not reachable today, because both call
    sites (`scene_scale`, 1180 and 1182) filter empties first. The even-count
    divergence *is* live: `scene_scale` is the reconstruction's length unit, and it
    feeds the resection's inlier scale.
  - **`angle_between` (1142) vs `camera::report::angle_between_deg`
    (`camera/report.rs:521`).** Same quantity, different units — and the newer of
    the two uses the formula the other one's doc comment exists to warn against.
    `report.rs` computes `atan2(|a×b|, a·b)` and says why: "rather than `acos` of a
    normalized dot product: it needs neither input to be unit-length and stays
    accurate at both ends of the range, where `acos` loses most of its significant
    digits." `resect_images` computes `a.dot(b).clamp(-1,1).acos()`. This is not
    cosmetic: its main consumer is `bearing_span` (1124), whose result is tested as
    `bearing_span(&world) <= pixel_angle` (1074) — a degeneracy gate against a
    **one-pixel** angle, i.e. exactly the near-zero regime where `acos` is worst.
  - **`orthonormalized` (1148) vs `geometry::numeric::polar_rotation`
    (`geometry/numeric.rs:34`).** Both are SVD polar projections to SO(3). They
    handle sign differently and both behaviours are wanted somewhere — `polar_rotation`
    maps `M ≈ −R` back to `R` (documented, for the conjugate-homography callers);
    `orthonormalized` takes the proper projection via `diag(1,1,det)` (undocumented
    as a choice). But `geometry/numeric.rs`'s own module doc says it holds primitives
    that "previously existed as two or more byte-identical private copies in sibling
    modules … a copy that drifts changes reconstruction results without failing to
    compile". This is the second variant, in a sibling module, undocumented.
- Proposed fix: delete `median` and call `numeric::median_in_place` (or, if the
  lower-middle rule is genuinely wanted, add it to `numeric.rs` as a named second
  function so the choice is visible beside the default). Move `angle_between` to
  `camera::report` beside `angle_between_deg` as its radian sibling, sharing the
  `atan2` body. Move `orthonormalized` into `geometry/numeric.rs` beside
  `polar_rotation`, with one doc paragraph saying which sign rule each caller needs.
- Effort: low
- Risk: medium — the median swap **changes numbers** on even populations, and
  `scene_scale` gates resection thresholds; the `angle_between` swap changes them
  near zero. Both are improvements, but `geometry/resect_images/tests.rs` (961) and
  `tests/rust_bindings/test_resect_images_rust_bindings.py` (412) must be run either
  side and any moved threshold understood, not just re-pinned.

**`bundle_adjust.rs` grew 43% in three weeks; `solve_lm` is 514 lines and now
generic over a const it also reads as a boolean**
- Location: `crates/sfmtool-core/src/geometry/bundle_adjust.rs` (963 → **1375**);
  `solve_lm` **636–1150** (514, was 410); `bundle_adjust_staged` **1155–1373** (218)
- Problem: The last snapshot cleared this file explicitly ("post-merge, one numerical
  kernel. `solve_lm` is 410 lines and stays that way"). The b-spline release work has
  since added `BSPLINE_CAM_COLS`, `bspline_columns` (260–303) and
  `bspline_step_admissible` (304–347), and threaded the spline through `solve_lm` via
  a pattern worth naming: the function is `fn solve_lm<const CAM_COLS: usize>`, and
  its first statement is `let opt_bspline = CAM_COLS == BSPLINE_CAM_COLS;` (658) —
  a const generic immediately re-derived as a runtime `bool` and branched on at four
  points inside the body (733, 835, 1042 in absolute terms), plus a `debug_assert!`
  that `opt_k1` and `opt_bspline` are never both set. So the two instantiations
  (`solve_lm::<BSPLINE_CAM_COLS>` at 1304, `solve_lm::<BASE_CAM_COLS>` at 1325)
  monomorphize into two copies of 514 lines that each carry the other's dead branches.
  The acquittal's reasoning — that an LM iteration's phases are not independently
  meaningful — still holds for the *original* 410 lines; it does not extend to the
  camera-column parameterization layered on top, which is exactly the separable part.
- Proposed fix: lift the per-model camera-column work behind a small trait or enum
  (`CameraColumns`: `n_cols()`, `fill_columns(&mut row, …)`, `step_admissible(&…)`)
  with `Base` / `K1` / `Bspline` implementations, so `solve_lm` stops re-deriving
  `opt_bspline` and the three step-admissibility rules stop being if/else in the
  damping ladder. `solve_lm` keeps its numerical body; the model dispatch leaves it.
- Effort: medium
- Risk: high — this is the optimizer. Every reconstruction number in the test suite
  is downstream of it. `bundle_adjust/tests.rs` (3,637) and
  `tests/rust_bindings/test_bundle_adjust_rust_bindings.py` (1,058) are the
  guardrail; the byte-for-byte-identical-results claim in the doc at 627–628 must
  survive.

**`grow_reconstruction` is a 606-line function with five separable phases**
> _Carried forward unchanged from 2026-08-08. Re-measured: the function is
> **488–1093** (606) and the file **1096** — up 5 lines, so nothing has moved._
- Location: `crates/sfmtool-core/src/geometry/reconstruction_growth.rs:488`
- Problem: The largest function in `sfmtool-core` and the second largest in the
  workspace. Its phases (candidate selection, resection, triangulation, filtering,
  local BA) are sequential and each has a natural signature; they are currently
  distinguished only by comment banners.
- Proposed fix: one function per phase, threading an explicit `GrowthState`.
- Effort: high
- Risk: high — reconstruction results change if any phase's state threading is
  altered.

**`focal_vote.rs` grew 46% and its options entry point is 396 lines**
- Location: `crates/sfmtool-core/src/geometry/focal_vote.rs` (736 → **1073**);
  `focal_vote_with_options` **594–989** (396); sibling `focal_vote/column_scan.rs`
  (1170)
- Problem: New since the last snapshot's measurement. The module now carries **24
  tuning constants** (255–286) spanning three unrelated policies — epipolar pair
  selection (`MIN_SHARED_STRICT` … `MAX_EPIPOLAR_PAIRS`), rotation self-calibration
  (`ROTATION_MAX_IMAGES` … `ROTATION_MIN_INLIERS`), and the ortho-cost grid
  (`ORTHO_GRID_N` … `ORTHO_COST_FLOOR`) — and `focal_vote_with_options` runs all
  three families plus consensus and diagnostics in one body. The three families
  already have their own vote types (`EpipolarVote` 82, `RotationVote` 104) and their
  own consensus reducer (`family_consensus` 343), so the seams are drawn; only the
  driver is monolithic. Separately, `log_iqr` (391) and `log_median` (410) are local
  statistics that belong beside `crate::numeric`'s median rather than in a
  focal-length module.
- Proposed fix: `focal_vote/{epipolar.rs, rotation.rs, ortho.rs}` taking the
  constants that only they use; the entry point becomes a ~90-line driver over the
  three vote producers and `family_consensus`. Move `log_median`/`log_iqr` to
  `numeric.rs`.
- Effort: medium
- Risk: medium — the constants are the tuning surface; moving one into the wrong
  module makes it invisible to the next person tuning it. `focal_vote/tests.rs`
  (1,023) covers the outcomes but not the pair-selection intermediate.

**`run_gpu_levels_prebuilt` is a 484-line method, two thirds of its file**
> _Carried forward unchanged from 2026-08-08. Re-measured at `254–737` (484) of
> `optical_flow/gpu/mod.rs` (741). Untouched in the window._
- Location: `crates/sfmtool-core/src/features/optical_flow/gpu/mod.rs:254`
- Problem: One method holds buffer allocation, bind-group construction, the
  per-level dispatch loop and readback. The wgpu resource setup is the separable
  two thirds.
- Proposed fix: a `GpuLevelResources` struct built once, leaving the dispatch loop.
- Effort: medium
- Risk: medium — GPU code, exercised only on hardware CI does not have.

---

## Rust — `sfmtool-core` (camera)

**Both halves of the `distortion.rs` split nearly doubled, and one of them is no
longer distortion**
> _Status (2026-09-04): **Partially done.** The `kernels.rs` half landed as
> `585b258`. `kernels/` is now seven family modules — `brown` (45),
> `equidistant` (514: the three polynomial fisheyes, the ray-direction helpers
> and the distortion-free `θ = r/f` block, which are one model family split
> across three banners), `thin_prism` (257), `rad_tan` (334),
> `sfmtool_fisheye` (259), `sfmtool_pinhole` (151) and `blend` (59) — with
> `kernels/mod.rs` (45) re-exporting all of them, so `distortion.rs`'s
> `use kernels::*` is unchanged and nothing outside the module moved. Every
> function body is byte-identical; the only textual edits are the visibility
> spelling (`pub(super)`, which after the move would mean "visible in
> `kernels`", became `pub(in crate::camera::distortion)` — the form
> `patch/keypoint_subpixel/kernels/` already uses — which pushed twelve
> signatures past 100 columns for `cargo fmt` to rewrap), the per-family `use`
> lines, and nine cross-family doc links rewritten as `super::` paths.
> `newton_thin_prism` and `newton_rad_tan_thin_prism` are now one file apart
> (`thin_prism.rs`, `rad_tan.rs`), still unmerged. Two counts in the finding
> are off: the file holds 43 *items* — 40 free functions, two consts and a type
> alias — and the families are ten blocks, not eight.
>
> **Still open: the `camera/projection.rs` half.** The `impl CameraIntrinsics`
> block calls `radial_fisheye_ray_jacobian` and `sfmtool_fisheye_ray_jacobian`
> directly, and both are private to `camera::distortion`. Moving the block to a
> sibling module means widening those two through `kernels/mod.rs` and
> `distortion.rs` both — trading this finding's own "nothing outside the module
> moves" property for the file split — so it is left for a change that decides
> that deliberately rather than as a side effect._
- Location: `crates/sfmtool-core/src/camera/distortion/kernels.rs` (793 → **1515**);
  `crates/sfmtool-core/src/camera/distortion.rs` (890 → **1233**)
- Problem: Two findings that share a cause.
  - **`kernels.rs` is 43 free functions in one flat file, spanning eight model
    families.** Its own module doc explains the split's premise: "Kept separate so
    `distortion.rs` holds only the two public `impl` blocks and the model dispatch."
    The premise held; the file did not. The families are already contiguous and could
    be cut today with no reordering: Brown/OpenCV **16–51**, equidistant fisheye
    **52–193**, simple-radial fisheye **194–220**, radial fisheye **221–266**, thin
    prism **267–506**, rad-tan thin prism **507–828**, equidistant ray helpers
    **829–1036**, sfmtool fisheye (B-spline) **1100–1343**, sfmtool pinhole
    **1344–1473**, shared blend tail **1474–1515**. The two Newton solvers
    (`newton_thin_prism` 380, `newton_rad_tan_thin_prism` 666) are 89 and 122 lines
    of the same iteration shape in adjacent families — the merge candidate a
    per-family split would put in front of whoever does it next.
  - **`distortion.rs`'s second impl block is projection, not distortion.**
    `impl CameraModel` (**113–810**, 698 lines) is genuinely distortion: `distort`,
    `undistort`, `distort_jacobian`, `distort_ray`, `undistort_to_ray`. But
    `impl CameraIntrinsics` (**811–1227**, 417 lines) is the pixel↔ray API —
    `project`, `unproject`, `pixel_to_ray`, `ray_to_pixel`,
    `ray_to_pixel_with_jacobian`, `min_pixel_scale`, `pixel_radius_to_world`,
    `pixel_radius_to_angle`, and their four batch forms. Nothing about that is named
    by the file it lives in, and `camera/mod.rs` is a 22-line re-export list with no
    `projection` member, so the canonical path for `CameraIntrinsics::project` is
    `camera::distortion`.
- Proposed fix: `kernels/{brown.rs, equidistant.rs, thin_prism.rs, rad_tan.rs,
  bspline.rs, blend.rs}` with `kernels/mod.rs` re-exporting — all callers are
  `pub(super)` inside `distortion.rs`, so nothing outside the module moves. Separately,
  `camera/projection.rs` for `impl CameraIntrinsics`, added to `camera/mod.rs`.
- Effort: low (both are pure moves along existing boundaries)
- Risk: low — no logic changes; `camera/distortion/tests.rs` (3,993) is the check.

**`camera/remap.rs` still hosts the crate's most-used image containers**
> _Carried forward from 2026-08-08 (the `prof` half landed in `bc68f2d`). Re-measured:
> `remap.rs` is **1167**, unchanged in the window._
- Location: `crates/sfmtool-core/src/camera/remap.rs`
- Problem: `ImageU8` (52–172), `ImageU8Pyramid` (173–~250) and `ImageF32WithGrad`
  (810–~890) are general image containers, not resampling. They are named from 25
  files across `patch/`, `spherical/`, `sfmtool-py` and `sfm-explorer`, yet their
  canonical path is `camera::remap::ImageU8`, which reads as neither a camera nor a
  remap concept.
- Proposed fix: move the three types to `camera/image.rs` with a `pub use` shim in
  `remap`.
- Effort: medium
- Risk: low — path change only, shimmed.

**`keypoint_localize.rs`: 1,483 lines, main function 424**
> _Carried forward unchanged from 2026-08-08 (1484 → 1483). No movement in the window._
- Location: `crates/sfmtool-core/src/patch/keypoint_localize.rs`;
  `localize_patch_keypoints_with_basis` **520–943** (424)
- Problem: The module already has four children (`basis`, `kernels`, `params`,
  `search`), so the pattern is established; the driver is what has not been cut. It
  runs seed selection, per-view state construction, tail registration and
  finalization in one body, with `TailGeometry` (947), `basis_template` (964),
  `register_tail` (1060) and `finalize` (1261) as its private helpers below it.
- Proposed fix: `keypoint_localize/tail.rs` for `TailGeometry` + `basis_template` +
  `within_max_shift` + `register_tail` (947–1199, 253 lines), leaving the driver at
  ~420 in the parent.
- Effort: medium
- Risk: medium — patch keypoint results are pinned by
  `keypoint_localize/tests.rs` (2,379) and `tests/patch/` (four modules, 2,205 lines).

**Numeric-helper duplication: `numeric.rs` reached seven copies, and three new ones
appeared behind it**
> _Carried forward from 2026-08-08. Its part (c) (six `median`s → one) landed as
> `234546b`/#297 on 2026-08-14. Parts (a) and (b) remain, and the count has since
> gone back **up**._
>
> _Status (2026-08-29): **Partially done.** The `resect_images.rs:1188` median is
> gone (see the resect finding above), taking the count from three back to two.
> Still open: `sfmtool-py`'s `patches/args.rs:15::np_median`, which uses
> `partial_cmp().unwrap()` and so **panics on any NaN in the population** — the exact
> policy `numeric.rs`'s docs were written to end. It is a different crate, so the fix
> is not a one-line import: `numeric.rs` is `pub(crate)` in `sfmtool-core` and would
> have to be exported (or the binding's median re-expressed through an existing
> public entry point). Also still open in full: the **two `pub(crate) mod numeric`**
> half of this finding — `crate::numeric` and `crate::geometry::numeric` still
> coexist, and the latter still holds the three `cam_*` camera constructors that are
> not numeric primitives by any reading. `orthonormalized` was added to it today,
> which makes it one item longer but no better named._
- Location: `crates/sfmtool-core/src/numeric.rs` (121) and
  `crates/sfmtool-core/src/geometry/numeric.rs` (112); new copies at
  `geometry/resect_images.rs:1188`, `sfmtool-py/src/patches/args.rs:15`,
  `sfmr-format/src/depth_stats.rs:171`
- Problem: Two distinct issues.
  - **Three medians again, with three NaN policies.** `numeric::median_in_place`
    (`total_cmp`, NaN when empty, averages the two middles) is the declared contract.
    `resect_images.rs:1188` sorts with `total_cmp` but takes the lower middle and
    panics when empty (see the resect finding above). `sfmtool-py`'s
    `patches/args.rs:15::np_median` uses `partial_cmp(…).unwrap()` — which
    **panics on any NaN in the population**, the exact failure `numeric.rs`'s docs
    were written to close — and its doc comment justifies itself as matching
    `numpy.median`, a fourth semantics. (`depth_stats.rs:171::median_sorted` and
    `kdforest/build.rs:234::median_value` are legitimately different: one takes a
    pre-sorted slice, the other is a generic quickselect pivot.)
  - **Two `pub(crate) mod numeric` in one crate, and one of them is not numeric.**
    `lib.rs:28` declares `crate::numeric` (medians); `geometry/mod.rs:14` declares
    `crate::geometry::numeric` (SplitMix64, `polar_rotation`, `rotation_angle`, and
    then `cam_at`, `cam_with`, `cam_with_bspline` — three *camera constructors*,
    which are not numeric primitives by any reading). A reader seeing
    `use crate::numeric::…` at one import site and `use super::numeric::…` at
    another has to open both to know which is which.
- Proposed fix: point `resect_images` and `np_median` at `numeric.rs`, adding a named
  lower-middle variant there if that rule is genuinely wanted. Rename
  `geometry/numeric.rs` to `geometry/kernels_common.rs` (or fold its RNG/rotation
  half into `crate::numeric` and move the three `cam_*` builders to
  `camera/intrinsics.rs`, where they belong).
- Effort: low
- Risk: medium for the medians (they change results — see the resect finding); low
  for the rename.

---

## Rust — format crates

**Four monolithic per-section entry points**
> _Carried forward from 2026-08-08. Layer 1 (`entries.rs` per crate) landed; the
> section splits it was meant to de-risk did not. Re-measured brace-to-brace at
> `94d3739`; all four are within ±6 lines of the last snapshot._
- Location: `sfmr-format/src/write.rs:97` (`write_sfmr_with_options`, **567**);
  `sfmr-format/src/read.rs:44` (`read_sfmr`, **517**); `sfmr-format/src/verify.rs:18`
  (`verify_sfmr`, **514**); `matches-format/src/write.rs:30` (`write_matches`, **424**)
- Problem: All four are a linear run of `// === Section ===` blocks, each opening a
  hasher, emitting 4–12 entries in lexicographic order, digesting, storing. The
  boundaries are already marked by comments. Secondary:
  `sfmr-format/src/write.rs:809` `validate_dimensions_with` (**286**, carrying
  `#[allow(clippy::too_many_arguments)]`) and `matches-format/src/write.rs:537`
  `validate_dimensions` (**277**).
- Proposed fix: one `write_<section>_section(…) -> Result<u128>` per section; entry
  points become ~60-line orchestrators. Split `validate_dimensions*` per section too,
  which also removes the `too_many_arguments` allow.
- Effort: medium
- Risk: medium — content-hash ordering hazard. Do write and verify together so
  ordering stays in lockstep. `entries.rs` having landed is what makes this safer
  than it was: the section splits can no longer misspell or reorder an entry name.

**`verify_matches` is a 710-line function — still the largest in the workspace**
> _Carried forward from 2026-08-08 (719 → 710)._
- Location: `crates/matches-format/src/verify.rs:126` (file 835)
- Problem: One function holding every invariant `.matches` has. Same section
  structure as the three above, and the same fix applies; it is listed separately
  because it is the single largest body in the repository and would be the first one
  to try the pattern on.
- Effort: medium
- Risk: medium

---

## Rust — `sfm-explorer`

**The G-buffer contract is declared six times, once per pipeline plus once where the
textures are actually allocated**
> _Status (2026-08-30): **Done.** The formats, the reversed-Z depth state and the
> slot-0 quad layout now live in `scene_renderer/gpu_types.rs` beside `THUMBNAIL_SIZE`,
> as `GBUFFER_{COLOR,LINEAR_DEPTH,PICK}_FORMAT`, `HW_DEPTH_FORMAT`,
> `GBUFFER_DEPTH_STATE`, `QUAD_VERTEX_LAYOUT` and a `gbuffer_targets(color_blend)`
> helper. `sizing.rs` allocates from them and all five pass-1 pipelines declare from
> them; the quad layout also absorbed the sixth and seventh copies in `target.rs` and
> `track_ray.rs`, which the finding did not count._
>
> _**Two things the finding did not separate.** The colour attachment is the one
> thing the five pipelines legitimately disagree about — points, frustums and patches
> composite with premultiplied alpha, image quads and distorted quads overwrite — so
> blending is `gbuffer_targets`' parameter rather than part of the constant. And the
> `@location(1)` comments the five carried were not restatements of the format: each
> said what that pipeline writes there (real view-space depth for splats, `0.0` for
> the three that opt out of EDL). Those are kept at the call sites; only the format
> triple was shared._
>
> _The pass-2 colour target was folded in as the same hazard: `edl.rs`, `target.rs`,
> `track_ray.rs` and `bg_distorted.rs` each spelled the EDL output format out, four
> more independent declarations of a format `sizing.rs` also owns. One
> `EDL_OUTPUT_FORMAT`, defined as `GBUFFER_COLOR_FORMAT` because the EDL resolve
> samples the first and writes the second and a mismatched pair shifts the whole
> viewport's gamma without failing validation._
>
> _**The structural tie the finding asked for is a test, not the constants.** Sharing
> a constant makes the six declarations one; it does not make the producers provably
> match the consumer, because nothing forces a new pipeline to use it.
> `the_gbuffer_pipelines_match_the_textures_sizing_allocates` builds the four
> textures from the constants and binds all five pass-1 pipelines inside a pass
> assembled the way `ensure_size` + `render` assemble the real one. Verified to fail
> both ways it can: a format changed only in `gbuffer_targets` fails at
> `create_render_pipeline` (shader/target mismatch), and a texture that disagrees
> with the pipelines fails at `set_pipeline` (`IncompatibleColorAttachment`). It runs
> on the `noop` backend, so it fails on any machine._
>
> _163 net lines out of the pipeline descriptors (-191/+28). `pixi run ui-test` (11 windowed
> tests on a real GPU) and the 386 headless lib tests pass._
- Location: `crates/sfm-explorer/src/scene_renderer/pipelines/{points,frustum,
  image_quad,patch,distorted_quad}.rs` and `scene_renderer/sizing.rs`
- Problem: The render target is a three-attachment G-buffer — `Rgba8UnormSrgb`
  colour, `R32Float` depth-readback, `R32Uint` picking — under a reversed-Z depth
  buffer. That contract is spelled out **verbatim in five pipeline modules**
  (`points.rs:106,131`, `frustum.rs:114,140`, `image_quad.rs:139,164`,
  `patch.rs:137,162`, `distorted_quad.rs:61,83`) and a sixth time in `sizing.rs`
  (90, 107), which allocates the textures. A clone scan finds the 14-line
  `depth_stencil` + `multisample` + `fragment.targets` run byte-identical across all
  five pipelines, and the `QuadVertex` slot-0 vertex layout byte-identical across
  **six**. Nothing structurally ties the producers to the consumer: changing the
  picking format in `sizing.rs` and missing one pipeline is a `create_render_pipeline`
  panic at startup if you are lucky, and a draw into the wrong attachment if you are
  not. This is the same shape as the 128×128 thumbnail edge the last snapshot closed
  (fifteen spellings → one `THUMBNAIL_SIZE` with a compile-time assert), and it has
  the same remedy.
- Proposed fix: in `scene_renderer/gpu_types.rs` (which already owns `THUMBNAIL_SIZE`
  and `MAX_ATLAS_COLS`), add `pub(super) const GBUFFER_TARGETS: [Option<ColorTargetState>; 3]`,
  `pub(super) const DEPTH_STATE: DepthStencilState` and
  `pub(super) const QUAD_VERTEX_LAYOUT: VertexBufferLayout`, and have `sizing.rs`
  build its textures from the same constants. ~110 lines deleted and one place to
  change the buffer layout.
- Effort: low
- Risk: low — `wgpu` validates the pipeline against the pass at creation, and
  `pipelines/tests.rs` (111) plus `scene_renderer/upload/tests.rs` (1,527) run
  headless on the `noop` backend, so a mismatch fails locally rather than only on a
  GPU.

**The intrinsics work landed the same displacement-field summary twice, and its
second copy says so**
- Location: `crates/sfm-explorer/src/intrinsics_detail/derived.rs:175`
  (`distortion_extent`) + `:203` (`grid_rows`);
  `crates/sfm-explorer/src/image_detail/intrinsics/mod.rs:161`
  (`CameraLayer::compute`) + `:233` (`grid_rows`)
- Problem: The Intrinsics **panel** and the intrinsics **overlay layer** both
  summarize `sfmtool_core::camera::report::distortion_field`, and both do it
  themselves. Each computes grid rows to keep cells square, calls `distortion_field`,
  tests every sample's `theta_deg` against `report::trustworthy_max_theta_deg`,
  reduces the trusted samples to a maximum `hypot` displacement, and counts the
  untrusted ones against the total. The panel's version returns
  `DistortionExtent { max_px, limit_deg, excluded }`; the layer's writes the same
  four numbers into `CameraLayer { max_px, extrapolated, limit_deg, … }`. The two
  `grid_rows` bodies are identical modulo `FIELD_COLS` being a const in one and a
  parameter in the other — and the layer's doc comment names the duplication out
  loud: "the same rule the Intrinsics panel's own field uses, so the panel's number
  and this layer's legend describe one field at the default density." That comment
  is a hand-written substitute for a shared function, and it is only true while
  nobody edits one side.
- Proposed fix: `camera::report::distortion_extent(cam, cols) -> DistortionExtent`
  in `sfmtool-core` beside `distortion_field`, returning grid, max, limit and
  excluded counts; both panels consume it. That also puts the number under
  `camera/report/tests.rs` (1,048) instead of under two GUI test modules.
- Effort: low
- Risk: low — the two are supposed to agree, so any test that breaks is reporting a
  real disagreement.

**`dock.rs::ui` is a 377-line `TabViewer` method with six inline tab bodies**
> _Status (2026-09-04): **Done** as `aa00ae0`. `show_viewer_3d`,
> `show_image_browser`, `show_image_detail`, `show_point_track_detail` and
> `show_intrinsics_detail` now sit on `impl TabContext<'_>` beside
> `apply_scene_graph_response`, and `ui` is a 20-line dispatch. Pure extraction:
> each method body is its arm's body, comments included; the only differences
> are lines `cargo fmt` rejoined once they were two indent levels shallower.
> Re-measured before the change: `dock.rs` was **845** and `fn ui` **85–515**
> (431 lines) over **seven** arms — the finding says six because it does not
> count the one-line `ActionLog` arm._
- Location: `crates/sfm-explorer/src/dock.rs` (560 → **720**); `fn ui` **78–455**
- Problem: `TabViewer::ui` is a single `match tab` with six arms, each of which
  builds that panel's arguments, calls its `show`, and then handles its response
  inline: `SceneGraph` 80–84, `Viewer3D` 84–150, `ImageBrowser` 150–231,
  `ImageDetail` 231–334, `PointTrackDetail` 334–422, `IntrinsicsDetail` 422–455.
  The arms share nothing but `self`, and the file already demonstrates the extracted
  form — `apply_scene_graph_response` (464) was pulled out of the first arm, which is
  why that arm is five lines and the rest are eighty to a hundred. The remaining five
  never got the same treatment, and the file grew 29% in three weeks as each new
  panel added one more.
- Proposed fix: `fn show_viewer_3d`, `show_image_browser`, `show_image_detail`,
  `show_point_track_detail`, `show_intrinsics_detail` on `TabContext`, following
  `apply_scene_graph_response`. `ui` becomes a ~15-line dispatch.
- Effort: low
- Risk: low — pure extraction; `TabContext`'s fields are all `&mut` already.

**`AppState` is 36 fields and a 543-line impl mixing selection state with
reconstruction operations**
- Location: `crates/sfm-explorer/src/state.rs` (678 → **997**); `struct AppState`
  **160–338** (36 fields); `impl AppState` **348–890**
- Problem: The last snapshot cleared this file as "a coordinator, not a grab-bag" at
  678 lines. It has since gained the resect feature and the intrinsics selection, and
  the impl now holds two different kinds of method. Most are small selection/lookup
  accessors (`select_image`, `select_camera`, `selected_point_in`, `camera_of`,
  `node`, `window_title` — 15 of them, all under 30 lines). Four are **reconstruction
  operations**: `align_node` (657–700, 44), `resect_image` (**701–809, 109**),
  `load_resect_matches` (810–838, 29) and `reload_node` (457–496, 40) — these load
  `.matches` files off disk, call into `sfmtool_core::geometry`, build new
  `SceneNode`s and format user-facing error strings. That is a command layer living
  inside the UI's selection struct, and it is where the file's 47% growth went.
- Proposed fix: `state/ops.rs` holding `align_node`, `resect_image`,
  `load_resect_matches` and `reload_node` as an `impl AppState` block, leaving
  `state.rs` as the struct, its defaults and the accessors. No signature changes.
- Effort: low
- Risk: low — a file split within one impl.

**`scene_graph/mod.rs` grew 40% and is the only module in its directory**
- Location: `crates/sfm-explorer/src/scene_graph/` — `mod.rs` (868 → **1218**),
  `tests.rs` (2,759)
- Problem: Cleared at 868 by the last snapshot as "already well decomposed", which is
  still true of its *functions* — 30+ items, largest is `show_node_header` at 137
  lines. What has changed is that four distinct groups have formed and the directory
  has nowhere to put them: the tree walk (`show_node`, `show_node_header`,
  `TreeOutput`, `NodeContext`, 261–491), the menus (`node_context_menu`,
  `show_tint_menu`, `show_align_menu`, `image_context_menu`, 492–632 + 976–1021), the
  camera/image rows (`show_camera_intrinsics_group` through `show_camera_image_rows`
  plus `ResectAvailability` and four hint constants, 633–975), and the leaf widgets
  and formatters (`eye_toggle`, `glyph_toggle`, `counts_text`, `compact_count`,
  `with_thousands`, 1112–1218). The camera-rows group alone is 343 lines and is what
  the intrinsics and resect work added.
- Proposed fix: `scene_graph/{menus.rs, cameras.rs, widgets.rs}`, leaving the tree
  walk and panel entry point in `mod.rs` at ~500. `tests.rs` (2,759) should follow
  the same cut.
- Effort: low
- Risk: low — all items are private to the module; the split is mechanical.

**`image_browser.rs::show()` is a 556-line method holding four subsystems**
> _Carried forward from 2026-08-08 (539 → **556**)._
- Location: `crates/sfm-explorer/src/image_browser.rs:177` (file 789)
- Problem: Grid layout, thumbnail loading and caching, selection/hover hit-testing
  and the filter bar in one body.
- Proposed fix: one method per subsystem on `ImageBrowser`.
- Effort: medium
- Risk: low

**`draw_overlays` — five copy-pasted match arms, five copy-pasted colormap wrappers,
and two disagreeing `error_color`s**
> _Carried forward unchanged from 2026-08-08. Re-verified at HEAD: `overlay.rs` is
> **635**, `draw_overlays` **19–386** (368), and `colormap.rs` (206) still holds the
> five six-line wrappers with byte-identical bodies._
- Location: `crates/sfm-explorer/src/image_detail/overlay.rs`;
  `crates/sfm-explorer/src/colormap.rs:60–118`;
  `crates/sfm-explorer/src/point_track_detail/metrics.rs:19`
- Problem: `match overlay_mode` has five value-driven arms (`ReprojError` 115,
  `TrackLength` 154, `MaxTrackAngle` 193, `DepthReliability` 225,
  `ConditionNumber` 257) that are the same body — compute range, loop features, cull,
  `circle_filled`, yellow stroke if selected, `draw_colorbar` — differing only in
  value extractor, colormap fn and label. `colormap.rs`'s `error_color`,
  `track_length_color`, `max_track_angle_color`, `depth_reliability_color` and
  `condition_number_color` are byte-identical six-line `t` normalizations differing
  only in which of two colormap constants they sample. And there are still **two**
  `error_color`s: `colormap.rs:60` (`value, vmin, vmax`, samples `ERROR_COLORMAP`)
  vs `point_track_detail/metrics.rs:19` (`error: f32`, hard-coded 0–2px ramp) — the
  two panels disagree on what a 1px error looks like.
- Proposed fix: one `draw_value_overlay(painter, features, extract, colormap, label)`;
  one `fn ramp(value, vmin, vmax, map: &Colormap)`. Reconcile the two `error_color`
  ramps and delete the loser.
- Effort: low
- Risk: low

**`point_track_detail::metrics` is triangulation math imported by a sibling UI panel**
> _Carried forward unchanged from 2026-08-08. Re-verified: no crate-level
> `src/metrics.rs` exists; `point_track_detail/mod.rs` still re-exports at the old
> path._
- Location: `crates/sfm-explorer/src/point_track_detail/metrics.rs`
- Problem: `compute_point_diagnostics` and `compute_max_pairwise_angle` are
  triangulation math living inside one panel and imported by another, behind a
  compat re-export whose comment documents the smell rather than fixing it.
- Proposed fix: promote to a crate-level `src/metrics.rs`; move `error_color` to
  `colormap.rs` while there (see above).
- Effort: low
- Risk: low

**One inline `#[cfg(test)] mod tests` block left in the workspace**
> _Carried forward from 2026-08-08. Re-verified: `platform/windows.rs:746` is the
> only remaining inline block across all nine crates; every other `#[cfg(test)]` is
> a `mod tests;` declaration. Two new modules landed in the window
> (`point_track_detail`, `image_detail/intrinsics`) and both used the sibling-file
> form, so the convention is holding — this is the last exception._
- Location: `crates/sfm-explorer/src/platform/windows.rs:746–832` (86 of 832)
- Proposed fix: move to `platform/windows/tests.rs`.
- Effort: low
- Risk: low

---

## Rust — `sfmtool-py`

**`patches/args.rs` exists to hold the shared string→enum parsers; five of eight
callers hand-inline them instead**
> _Status (2026-09-04): **Partially done** as `f51a109`. `parse_sampler` joins
> `parse_patch_window` in `args.rs`, and all five inline matches in
> `localize_keypoints`, `refine_keypoints`, `refine_normals`, `member_coherence`
> and `select_views` now call the two shared parsers. Both error strings are
> byte-identical to the ones they replaced, so `unknown window` goes from six
> spellings in the crate to one and `unknown sampler` from five to one —
> `tests/patch/test_patch_normal_refine.py` and
> `tests/rust_bindings/test_localizability_rust_bindings.py` assert on them, and
> both still pass. Neither `matching/cluster.rs` nor `patches/localizability.rs`
> had a sampler match to share: the five were the whole set. **Still open: the
> 29-line prologue** the three biggest bindings open with, which this finding
> proposes lifting into `views.rs` as `resolve_patch_scene`._
- Location: `crates/sfmtool-py/src/patches/args.rs` (88) and
  `patches/{localize_keypoints,refine_keypoints,refine_normals,member_coherence,
  select_views}.rs`
- Problem: `args.rs`'s module doc reads "Shared parameter-string parsers and small
  numeric helpers for the patch-kernel bindings", and it exports
  `parse_patch_window`, `parse_reduce`, `parse_normal` and `parse_extent`.
  `parse_patch_window` has **three** callers (`matching/cluster.rs:403`,
  `patches/localizability.rs:103` and `:225`). The other **five** binding modules
  write the match out by hand — `localize_keypoints.rs:198`, `refine_keypoints.rs:208`,
  `refine_normals.rs:201`, `member_coherence.rs:218`, `select_views.rs:142` — and a
  clone scan confirms all five are **byte-identical**, error string
  (`"unknown window: {other:?} (expected uniform|gaussian|gaussian_disk)"`) included.
  Immediately below each one sits a **second** identical match, on `sampler`
  (`"unknown sampler: {other:?} (expected bilinear|bilinear_mip|anisotropic)"`, five
  copies) — for which `args.rs` has no helper at all, so nobody could have shared it.
  Counting error strings across the crate: `window` ×6, `sampler` ×5, `reduce` ×2,
  `normal policy` ×2, `extent policy` ×2. About 90 lines, and the failure mode is
  quiet: adding a window kernel updates `args.rs` and three call sites, and the other
  five keep rejecting the new name with a message listing the old three.
  Separately, the three biggest bindings (`refine_normals` 302, `refine_keypoints`
  305, `localize_keypoints` 295) open with a **29-line identical prologue** —
  `resolve_scene`, `recon_opt`, the point-index range check against the cloud, and
  the "view lists required when the scene is a bare `CameraViews`" guard — before
  diverging at line 30.
- Proposed fix: add `parse_sampler` to `args.rs`; route all five inline matches
  through `parse_patch_window`/`parse_sampler`. Then lift the shared prologue into
  `views.rs` (which already owns `resolve_scene`/`resolve_pyramids`) as
  `resolve_patch_scene(recon, cloud, view_sets_present) -> PyResult<…>`.
- Effort: low
- Risk: low — the error strings are identical today, so nothing user-visible changes;
  `tests/rust_bindings/` and `tests/patch/` cover the accept paths.

**`clone_with_changes` is a single 599-line function**
> _Carried forward from 2026-08-08 (596 → **599**). Untouched in the window._
- Location: `crates/sfmtool-py/src/reconstruction/clone.rs:74` (file 771)
- Problem: One function taking every mutable field of a reconstruction as an
  `Option`, validating each, then rebuilding. Each field's validate-and-apply block
  is independent.
- Proposed fix: one `apply_<field>` per block, or a `CloneEdits` builder.
- Effort: medium
- Risk: medium — it is the write path for `sfm xform`; `tests/xform/` (21 modules)
  is the guard.

---

## Python — `src/sfmtool/`

**A render-farm SDK is a hard dependency for two path-formatting functions, imported
twelve ways**
- Location: `pyproject.toml:17` (`"deadline"`); 12 import sites across
  `src/sfmtool/`
- Problem: `deadline` — the AWS Deadline Cloud client — is one of six runtime
  dependencies, and the entire use of it is two pure functions from one submodule:
  `deadline.job_attachments.api.summarize_path_list` and `summarize_paths_by_sequence`.
  Twelve modules import them, and the import style is split with no rule: **six at
  module top level** (`sift/file.py:17`, `_global_sfm.py:10`, `_sfmr_naming.py:12`,
  `_commands/solve.py:11`, `_commands/sift.py:11`, `_incremental_sfm.py:11`) and
  **six deferred into function bodies** (`_compare.py:373`, `feature_match/_run.py:390`,
  `camrig/cp.py:48`, `motion/recon_discontinuity.py:584`, `analyze/summary.py:429`,
  `_commands/motion.py:93`). Six of those are in `_commands/` or on the CLI's import
  path, so the deferral is not buying startup time in any consistent way. There is
  no local seam: replacing, vendoring or stubbing these two functions means editing
  twelve files, and `_sfmr_naming.py` — which imports *both* and is the module whose
  whole job is filename construction — is the obvious place that seam should have
  been. (Import cost is unmeasured: `deadline` is not installed in this container and
  no pixi environment is provisioned here, so this finding rests on the coupling, not
  on a benchmark.)
- Proposed fix: re-export both through `_sfmr_naming.py` (or a two-line
  `_path_summary.py`) and have the other eleven sites import from there; make the
  import style uniform at that one seam. Then the question "do we still need
  `deadline`?" becomes answerable by reading one file.
- Effort: low
- Risk: low — pure re-export; no behaviour change.

**`_commands/solve.py` is now the only command module carrying its algorithm**
- Location: `src/sfmtool/_commands/solve.py` (508) — `_run_sequential_overlap_sfm`
  **327–447** (121), `_run_sfm` **450–508** (59)
- Problem: The convention is written down in `feature_match/_run.py:7–9` ("Extracted
  from `_commands/match.py` so the command module stays a thin Click wrapper") and,
  since `_commands/cluster_patches.py` was fixed on 2026-08-21, holds for **28 of 29**
  command modules. `solve.py` is the exception: 180 lines of pipeline below a
  203-line Click declaration, and the natural home already exists —
  `src/sfmtool/_incremental_sfm.py` (292) and `_global_sfm.py` (127) are siblings at
  the package top level, doing exactly this kind of work. As it stands the
  sequential-overlap solve cannot be called except through Click.
- Proposed fix: move both functions to `_sequential_overlap_sfm.py` beside the two
  existing solvers (or into `_incremental_sfm.py` if the overlap variant is properly
  a mode of it), and defer-import from the callback the way
  `_commands/embed_patches.py:283` does.
- Effort: low
- Risk: low — one import edit; `tests/test_solve.py` (524) covers the CLI surface.

**`xform/_arg_parser.py`: a 440-line argv loop plus four copy-pasted key=value
parsers**
> _Carried forward unchanged from 2026-08-08. Re-verified at HEAD: file **830**,
> `parse_transform_args` **391–830** (440), the `"--X requires an argument"` guard
> written out **20 times**, and all four `parse_*_params` functions still present at
> 125, 194, 267 and 328._
- Location: `src/sfmtool/xform/_arg_parser.py`
- Problem: (a) One `while` loop with an if/elif chain; the optional-value
  tokenization block is copy-pasted four times, each carrying a comment saying it
  mirrors the first. (b) `parse_refine_normals_params`, `parse_refine_keypoints_params`,
  `parse_localize_keypoints_params` and `parse_to_embedded_patches_params` are ~203
  lines of the same ~50-line body; every difference is the option name, the `_*_KEYS`
  table or the transform class.
- Proposed fix: (b) first — `_parse_kv_params(option_name, keys, ctor, param)` with
  three-line public wrappers, removing ~150 lines. Then (a) — a spec table
  `{flag: (arity, builder)}` plus one shared optional-value token reader.
- Effort: medium
- Risk: medium — error text is user-visible and asserted on; `tests/xform/` must be
  green either side.

**`sift/file.py` holds three concerns and its name describes one**
> _Carried forward unchanged from 2026-08-08 (877 → **882**)._
- Location: `src/sfmtool/sift/file.py`
- Problem: File I/O proper (validation 229, `SiftReader` 251, write path 309–464,
  path resolution 465–597) is ~370 lines and matches the name. The file also carries
  the **extraction pipeline** `image_files_to_sift_files` (**598–767**) +
  `image_files_to_sift_files_opencv` (**769–798**), which imports and drives the
  sibling `extract_colmap.py`/`extract_opencv.py` from inside the function body, and
  a **visualization** function `draw_sift_features` (**799–882**). Also stranded:
  xxh128 helpers (61–117) and pure feature geometry `compute_orientation` /
  `feature_size*` (118–174), neither of which touches a file.
- Proposed fix: `sift/extract.py` for 598–798 (becomes the peer of the three
  `extract_*.py` modules, and the deferred import becomes a normal one);
  `visualization/_sift_display.py` for 799–882; optionally `sift/geometry.py` for
  118–174. Re-export from `sift/__init__.py`.
- Effort: medium
- Risk: low — pure moves behind existing re-exports.

**Strip modules form one closed pipeline — the `strips/` subpackage still earns
itself**
> _Carried forward unchanged from 2026-08-08. Re-verified: all five modules still
> flat at the package top level._
- Location: `src/sfmtool/_solve_strips.py` (486), `_compare_strips.py` (479),
  `_inspect_strips.py` (241), `_strip_montage.py` (210), `_patch_ncc.py` (178) —
  1,594 lines across 5 flat siblings
- Problem: Exactly two edges leave the cluster (`_compare.py:239` and
  `_commands/inspect.py:177`), both deferred function-body imports. The modules
  cross-reference each other in their docstrings — a hand-written substitute for the
  package boundary that is not there. Layering is clean: scoring → strip solving →
  pixel layout → two consumers.
- Proposed fix: `strips/` with `_ncc.py`, `_solve.py`, `_montage.py`, `_compare.py`,
  `_inspect.py`, and an `__init__.py` exporting exactly the three names the outside
  needs.
- Effort: low
- Risk: low — five renames plus two import-site edits;
  `tests/test_cli_inspect_strips.py` imports two of them by path.

**`draw_epipolar_visualization`: 509 lines, two input modes × three render modes**
> _Carried forward unchanged from 2026-08-08. Re-verified: `_epipolar_display.py`
> **619**, the function **111–619** (509) with **16 positional parameters**._
- Location: `src/sfmtool/visualization/_epipolar_display.py:111`
- Problem: A 2×3 matrix of mutually exclusive paths in one body — feature acquisition
  splits sweep-matching vs track-based, rendering splits rectified / undistorted /
  original-curve, output assembly branches on `side_by_side` and `save_which`. The
  `sweep_max_features is not None` test is re-asked four times; the mode is threaded
  through the body rather than resolved once.
- Proposed fix: extract `_resolve_image_pair`, `_feature_pairs_from_sweep`,
  `_feature_pairs_from_tracks` and three `_render_*` functions each returning
  `(img1, img2)`; the driver becomes ~60 lines. Bundle the render knobs into a
  dataclass.
- Effort: medium
- Risk: medium — only visual output is verifiable, so lean on `tests/test_epipolar.py`
  (611).

**Six duplicated helper pairs — two of them measurably divergent**
> _Carried forward from 2026-08-08. **All six re-verified present and unchanged at
> HEAD**, including both divergences._
- Location: across `src/sfmtool/`
- Problem:
  - `_apply_range_filter` — `_commands/to_colmap_bin.py:89` vs
    `_commands/to_nerfstudio.py:136`: 31 lines, exactly one differing line
    (`print(` vs `click.echo(`).
  - `_load_gray` — `feature_match/_flow_matching.py:154` vs `motion/flow_stats.py:12`:
    6 lines, one differing line, and it is the docstring.
  - `_classify_ratio` — `motion/report.py:245` vs
    `visualization/_discontinuity_display.py:209`: **divergent.** The report reads
    `_RATIO_UPPER = 1.0 / _RATIO_LOWER` = 1.3333…; the display hardcodes `1.33`
    (`_discontinuity_display.py:217`). A normalized ratio in [1.33, 1.3333] is
    "acceleration" on screen and unclassified in the JSON. They also disagree on the
    empty case (`""` vs `None`).
  - Sequence-descriptor naming — `_sfmr_naming.py:20–77` vs
    `feature_match/_run.py:385–429`: **divergent.** `_sfmr_naming` has an `else`
    branch producing `{first_name}-total-{N}-images` (line 76) when the paths are not
    one sequence; `_run.py` has no such branch and silently emits no descriptor.
  - `_camera_centers` — `_embed_patches.py:240` vs `rig/panorama.py:56`: same
    `C = −Rᵀt`, different signatures.
  - `_rotation_angle_deg` — `_compare_fragments.py:332` vs
    `motion/recon_discontinuity.py:23`: **same name, different semantics** (angle of
    one transform vs angle between two quaternions). Not a merge candidate — a name
    collision that misleads anyone grepping.
- Proposed fix: `_apply_range_filter` → `_filenames.py` taking an `echo` callable;
  `_load_gray` → a shared image-IO helper; have `_discontinuity_display` import
  `_classify_ratio` from `motion/report.py` (resolving 1.33 in the direction of the
  constants); factor the naming logic into
  `next_dated_filename(base_dir, suffix, operation, image_paths)` used by both;
  unify `_camera_centers`. Rename one `_rotation_angle_deg`.
- Effort: low
- Risk: low — the naming unification is the only one with user-visible output, and
  fixing it *changes* `.matches` names in the multi-sequence case, which is the point.

**Flat modules whose only consumer is a single subpackage or sibling**
> _Carried forward from 2026-08-08. Re-verified: `_rectification.py` still has
> exactly one production importer; the `_compare*` trio is still flat; the `[N/6]`
> labels still disagree with themselves._
- Location: `src/sfmtool/_rectification.py` (212); `_compare.py` (818) +
  `_compare_fragments.py` (411) + `_compare_strips.py` (479)
- Problem: `_rectification.py`'s only production importer is
  `visualization/_epipolar_display.py:14` — a `visualization/` implementation detail
  at package top level, whose natural neighbour `check_rectification_safe` already
  lives in `feature_match/_geometry.py` and is imported two lines away (271).
  Separately, `_compare_fragments.py` and `_compare_strips.py` are imported by
  nothing but `_compare.py`, and `_compare.py` by nothing but `_commands/compare.py`
  — 1,708 lines behind a single CLI entry point, flat, while every other multi-module
  CLI topic already has a subpackage. `compare_reconstructions`
  (`_compare.py:57–364`, 308 lines) is a 7-phase driver whose `[N/M]` labels still
  disagree: `[1/6]` 107, `[2/6]` 113, `[3/6]` 117, `[4/6]` 128, `[5/6]` 171, then
  `[6/7]` 203 and `[7/7]` 220. Five of seven state the wrong total.
- Proposed fix: `_rectification.py` → `visualization/`. Create `compare/` holding
  `core.py` + `fragments.py`, with strips coming from the `strips/` package; while
  moving, split the driver's phases and fix the labels.
- Effort: low (rectification) / medium (compare)
- Risk: low — `tests/test_epipolar.py:323,347` imports `sfmtool._rectification` by
  path and needs updating.

---

## Tests

**The conftest solve-retry loop is still duplicated verbatim**
> _Carried forward unchanged from 2026-08-08. Re-verified: `conftest.py` is **571**,
> `build_cluster_reconstruction` still at **154**, `kerry_park_camrig_workspace_once`
> still at **480**._
- Location: `tests/conftest.py`
- Problem: Identical algorithm in both: rmtree `colmap_dir`, glob-unlink stale
  `.sfmr`, `seed = 42 if attempt == 1 else None`, solve, load, keep best by point
  count, break on threshold, restore from a `_best*` stash, glob-unlink again. The
  code admits it — line 513 reads "Retry with a fresh randomization (mirroring
  ``build_cluster_reconstruction``)". They differ only in the solve callable,
  `max_attempts` (6 vs 10), the ranking key and a trailing
  `except RuntimeError: continue`. Two copies of a flaky-solve retry policy means a
  tuning fix lands in one and not the other.
- Proposed fix: `_solve_with_retries(solve_fn, *, max_attempts, rank, accept,
  random_seed=42)` taking a callable; each fixture passes a closure.
- Effort: low
- Risk: medium — session-scoped fixtures gating most integration tests; a behaviour
  change re-flakes the suite against CI's non-deterministic GLOMAP.

**`test_densify.py` is misnamed — 12% of it is about densify**
> _Carried forward unchanged from 2026-08-08, now at its post-move path. Re-verified:
> **702** lines, 12 classes._
- Location: `tests/matching/test_densify.py`
- Problem: Densify is `TestDensifyCLI` (615) and `TestDensifyE2E` (657) — ~88 of 702
  lines. The other ten classes are epipolar geometry (`TestEssentialMatrix` 33,
  `TestFundamentalMatrix` 68, `TestEpipole` 94, `TestRectificationSafe` 110),
  intrinsics (`TestGetIntrinsicMatrix` 129), filter config
  (`TestGeometricFilterConfig` 153) and sweep matching
  (`TestRectifiedSweepMatching` 335, `TestPolarSweepMatching` 379,
  `TestMatchImagePair` 532, `TestPruneImagePairs` 569). Its own docstring concedes
  the mixture.
- Proposed fix: `tests/matching/test_epipolar_geometry.py` (33–152),
  `test_sweep_matching.py` (335–614), leaving `test_densify.py` at ~150.
- Effort: low
- Risk: low — pure test moves; the `tests/matching/` package already exists.

**`test_embed_patches_compaction.py` mixes four topics**
> _Carried forward unchanged from 2026-08-08. Re-verified: **674** lines, 16 tests._
- Location: `tests/patch/test_embed_patches_compaction.py`
- Problem: Four unrelated groups. Halfvec array round-trip and image-hash shape
  (50–74); compaction proper (75–411, seven tests — the name); the `embed_patches`
  CLI's round behaviour (412–579, three tests); `_drop_grazing_observations`
  (580–637); and **progress-polling instrumentation** (638–674, five tests:
  `test_progress_poll_loop_*`, `test_poll_progress_*`, `test_progress_counter_*`),
  which has nothing to do with patches at all and belongs beside `_progress.py`.
- Proposed fix: split out `tests/test_progress.py` (the five progress tests) and
  `tests/patch/test_embed_patches_rounds.py` (412–579); the remainder matches the
  name.
- Effort: low
- Risk: low

**Patch visualization helpers: the `scripts/` half is still open, in reduced form**
> _Carried forward from 2026-08-08, **partially resolved**: the `_load_images` copies
> the finding led with are gone from all three scripts. What it also listed remains._
- Location: `scripts/viz_keypoint_localization.py` (—),
  `scripts/viz_keypoint_localization_strips.py` (488),
  `scripts/viz_view_selection_strips.py` (469)
- Problem: `_infinity_first_sample` is still written three times (164, 196, 149),
  `_compose` three times (323, 342, 340), `_label_for` three times (378, 425, 405)
  and `_chip` twice (143, 129, differing only in a default `scale=0.34` vs `0.3`).
  They cannot import `tests/patch/conftest.py`, so the tests-side consolidation did
  not reach them.
- Proposed fix: `scripts/_viz_common.py`.
- Effort: low
- Risk: low

---

## Top-level layout, docs, tooling

**`AGENTS.md`'s module counts are now off by 61% and 21%, and this is the third audit
to correct them**
> _Carried forward from 2026-08-08, where the fix was proposed and not applied. Both
> numbers have drifted **further** since._
- Location: `AGENTS.md:56` and `:82`
- Problem: `AGENTS.md:56` says "`src/sfmtool/` — Python package (**~93 modules**)";
  `git ls-files` returns **150**. `AGENTS.md:82` says "`tests/` — pytest, **~114
  modules**"; the count is **138**. The last snapshot measured 148 and 119 and
  proposed the fix; three weeks later they are 150 and 138 and the text is unchanged.
  Also still true: "Structure at a glance" does not mention the top-level `skills/`
  directory (four skills, symlinked into `.claude/skills/` — deliberate and correct,
  but a reader looking at the tree has nothing telling them so). `AGENTS.md` has
  otherwise been kept current — the `test-tasks` feature split that fixed the
  `pixi run test` ambiguity noted in the last report has landed
  (`pixi.toml:105`, `:148`), and the specs reorganization (#319, #320) is reflected.
- Proposed fix: this is the third correction of the same two numbers. Delete them, or
  state them as "~150" / "~140" round figures. A figure that must be re-derived every
  audit is a maintenance liability, not documentation.
- Effort: low
- Risk: low

---

## Carried-forward items now resolved

Verified closed at `94d3739`; recorded so a future audit does not re-open them.

- **`cargo doc` is in no check, and the workspace has 145 rustdoc warnings**
  (2026-08-08) — gate wired, backlog at zero. `AGENTS.md:180–189` documents the
  private-item and `Self::`/`super::` rules it enforces.
- **The 128×128 thumbnail edge was declared in fifteen places across two languages**
  (2026-08-08, parts (a)+(b)+(c)) — **fully closed.** One
  `sfmr_format::THUMBNAIL_SIZE` (`types.rs:30`) with a compile-time
  `assert!(sfmr_format::THUMBNAIL_SIZE == sift_format::THUMBNAIL_SIZE)`
  (`sfmtool-core/src/lib.rs:58`), re-exported through core and the PyO3 bindings.
  All four Python extractors and `_undistort_images.py:352` import it; the explorer
  reads `sfmtool_core::THUMBNAIL_SIZE` via `gpu_types.rs:296`. No bare `128` remains
  on any thumbnail path in either language.
- **372 lines of hand-mirrored `CameraModel` ↔ `SfmrCamera` mapping** (2026-08-08) —
  closed by the `camera_models!` registry (`camera/intrinsics/registry.rs`, 538).
- **`member_coherence.rs` is the largest non-test file in `sfmtool-core`** — split
  into `member_coherence/{matrix.rs, decide.rs}`.
- **Parallel plain/`_geometric` matcher families across `polar.rs` and `sweep.rs`** —
  both halves merged.
- **`sfmtool-archive-io` enforces write-then-hash for binary entries only** — closed.
- **`feature_match/_run.py` holds matching orchestration and `.matches` merging** —
  `_merge.py` (371) split out; `_run.py` is 605.
- **`_commands/cluster_patches.py` is the only command module carrying its
  algorithm** — moved to `_cluster_patches.py` (238); the command module is a
  127-line Click wrapper. (Note: `_commands/solve.py` is now the last holdout — see
  the finding above.)
- **`tests/` subdirectories: four clusters still flat** — `tests/rig/`,
  `tests/matching/`, `tests/sift/`, `tests/camrig/` all created.
- **Five inline `#[cfg(test)] mod tests` blocks in `sfmtool-core`** — all moved to
  sibling `tests.rs`; one remains workspace-wide (`platform/windows.rs`).
- **Entry-name templates → `entries.rs`, per format crate** — layer 1 landed in both
  crates with a golden pin test. (Layers 2+, the section splits, remain open above.)
- **Retire `reports/2026-07-07-next-steps.md`** — done.
- **`pixi run test` / `test-rust` / `maturin develop` are ambiguous** (noted as
  tooling drift on the intrinsics finding) — fixed by moving the tasks into a
  `test-tasks` feature that only the `test` environment pulls in (`pixi.toml:105`).
- **`scripts/`: 9 of 20 files have zero inbound references** — declined on
  2026-08-20 and explicitly marked not to be re-reported. Not re-reported.

---

## Explicitly not flagged

Verified long-but-coherent at `94d3739`. Sizes are current. **Read the headline
before trusting this list for longer than a month** — five of the last snapshot's
acquittals are findings in this one.

- `crates/sfmtool-core/src/patch/view_selection.rs` (1051), `cluster_refine/mod.rs`
  (899), `cluster_refine/kernels.rs` (799), `keypoint_subpixel.rs` (861),
  `keypoint_subpixel/kernels/render.rs` (826), `spherical/photometric_ransac.rs`
  (839), `spherical/per_tile_source_stack.rs` (866), `spherical/tile_rig.rs` (1091),
  `patch/cloud.rs` (1023), `features/sift/scale_space.rs` (881),
  `geometry/rotation_init.rs` (891) — longest function in any of them is 223 lines
  (`cluster_refine/mod.rs:559`); most are under 120.
- `crates/sfmtool-core/src/geometry/focal_vote/column_scan.rs` (1170) — one scan
  algorithm, already decomposed: its longest function is 75 lines
  (`epipolar_vote`, 1052). It is large because the scan is, not because it holds
  two things. (Its parent `focal_vote.rs` *is* flagged, above.)
- `crates/sfm-explorer/src/intrinsics_detail/` (7 modules, 1,872 non-test lines) and
  `image_detail/intrinsics/` (5 modules, 1,691 non-test) — the newest code in the
  repo and the best-decomposed. Both were built as packages from the first commit,
  every module has a purpose paragraph, and the largest function across the twelve is
  `projection_plot.rs:508::draw_curves` at 102 lines. The one duplication between them
  is flagged above; the structure is not the problem.
- `crates/sfm-explorer/src/app.rs` (879). `run_egui_pass` is back to **245** lines
  (552–796) from the 180 the last snapshot measured, and a `menu.rs` extraction is
  again defensible — but the growth is the File menu and the demo modal, not a new
  concern, and `dock.rs::ui` (above) is the same shape with three times the payoff.
  Do that one first; re-measure this after.
- `crates/sfm-explorer/src/viewer_3d/mod.rs` (750), `platform/windows.rs` (832 —
  largest fn 93), `image_detail/mod.rs` (775 — `show` is 200 lines with `input`,
  `overlay` and `intrinsics` already split out) — coordinators, not grab-bags.
- `crates/sfmtool-py/src/reconstruction/sfmr_reconstruction.rs` (1051) — one
  `#[pyclass]`; longest method 63 lines.
- `crates/sfmtool-core/src/geometry/resect_images.rs`'s **size** (1227) — the file is
  a coherent single algorithm with `matches_join` already split out and its longest
  function at 308. Only its three tail helpers are flagged, above.
- `src/sfmtool/_embed_patches.py` (826) — `embed_patches` is 374–826, but ~155 of
  those lines are the docstring and the body is a linear staged pipeline whose stages
  are already extracted.
- `src/sfmtool/colmap/io.py` (876 — deliberate mirror-image converters),
  `analyze/summary.py` (668 — a uniform per-file-type dispatch table),
  `visualization/_flow_display.py` (710 — modes already extracted),
  `motion/recon_discontinuity.py` (799), `_densify.py` (765),
  `_undistort_images.py` (599).
- **No dead Python modules.** Every one of the 150 modules under `src/sfmtool/` has
  at least one production importer (import graph rebuilt for this audit); no
  `_old`/`_v2`/`.bak` leftovers anywhere in the tree.
- **Test-side helper duplication is clean.** A 12-line clone scan across `tests/` and
  `src/` finds only three cross-file pairs, all trivial. The 2026-08-01 and
  2026-08-11 consolidations held.
- `specs/` — 127 files, now organized into `cli/`, `core/`, `formats/`, `gui/`,
  `workspace/` with a `README.md` index in each area (#319, #320). Spec/code
  agreement is `audit-specs`' remit, not this one, but the *structure* is in good
  shape.

---

## Top 3

1. **`resect_images.rs`'s three re-introduced primitives.**
   The highest-value item in the report, and not primarily for tidiness. Two of the
   three are behaviour: a median that disagrees with the crate's declared contract on
   even counts, feeding `scene_scale`, which sets the resection's inlier scale; and
   an `acos`-of-dot angle used as a **one-pixel** degeneracy gate, in a crate whose
   other implementation carries a doc comment explaining why that formula loses most
   of its significant digits in exactly that regime. The fix is three deletions and
   three imports. The reason to do it first is what it says about the last three
   weeks: `#297` centralized seven medians on 2026-08-14 and wrote the contract into
   the module docs, and `#318` landed a new copy thirteen days later. A contract
   that lives only in a doc comment gets re-broken by the next person who needs a
   median at 11pm. Whatever the fix, it should end with `numeric.rs` being the only
   place in the crate that spells one — and, ideally, with a clippy `disallowed_methods`
   entry or a grep-based test that says so mechanically.

2. **The `sfm-explorer` G-buffer contract → `gpu_types.rs`.**
   Best pure effort-to-value ratio. The three-attachment format triple and the
   reversed-Z depth state are written out verbatim in five pipeline modules and a
   sixth time in `sizing.rs`, which allocates the textures they must match — six
   independent spellings of one contract with nothing tying them together, ~110 lines.
   The constants go in the file that already owns `THUMBNAIL_SIZE` for exactly this
   reason, `wgpu` validates the result at pipeline creation, and
   `pipelines/tests.rs` + `scene_renderer/upload/tests.rs` run headless on the `noop`
   backend so a mismatch fails locally rather than on hardware nobody's CI has. This
   is the same fix the last cycle applied to the thumbnail edge, which is now the
   cleanest thing in the crate.

3. **`kernels.rs` → per-family, and `dock.rs::ui` → per-tab.**
   Two unrelated splits paired because they are the same trade: both are pure moves
   along boundaries the code has already drawn, both target files that grew 29–91% in
   three weeks, and neither changes a line of logic. `kernels.rs` (1515, 43 functions,
   nine contiguous model families, all callers `pub(super)` within one module) is the
   larger win and the cheaper one — and it puts the two 89/122-line Newton solvers
   next to each other, which is the merge nobody will find while they are 280 lines
   apart. `dock.rs::ui` (377 lines, six arms) already has its extracted form
   demonstrated in the same file by `apply_scene_graph_response`; the other five arms
   just never got it, and every new panel adds one more.

   > _Status (2026-09-04): Both splits landed — `kernels.rs` as `585b258`,
   > `dock.rs::ui` as `aa00ae0`. The second half of the `kernels.rs` finding
   > (`camera/projection.rs`) did not; see that finding for why._

Runner-up, called out because it is the cheapest correctness-shaped fix in the
report: `sfmtool-py`'s five hand-inlined `parse_patch_window` matches. `args.rs`
exists to hold them, three callers use it, five do not, and the failure mode is that
adding a window kernel silently leaves five bindings rejecting the new name with a
message listing the old three. One `parse_sampler` function and five call-site edits.

> _Status (2026-09-04): **Done** as `f51a109` — `parse_sampler` added, and all five
> window matches plus all five sampler matches routed through `args.rs`, with the
> error strings unchanged. The prologue half of that finding is untouched._

---

## Appendix — design topics carried forward

Not hygiene findings — unspecced feature proposals inherited from
`reports/2026-07-07-next-steps.md` via the 2026-08-08 snapshot, kept because
`AGENTS.md` says to fold unfinished items into the next regenerated report rather
than let them evaporate. Each premise re-verified at `94d3739`:

- **A — Camera bookmarks (save/restore named viewpoints) in SfM Explorer.**
  `specs/gui/viewport-navigation.md` still carries the unticked
  `- [ ] Save/restore camera positions`, and no bookmark state exists in
  `sfm-explorer`. Still open. The explorer has moved a very long way since the topic
  was written (scene-graph phases, resect, and the six-phase camera-intrinsics work
  all landed after it), so re-read the current `state.rs` / `viewer_3d/` before
  acting on any sketch of it.
- **B — `sfm xform --crop` (3D bounding-volume crop).** No crop transform in
  `src/sfmtool/xform/`, no `--crop` in `_commands/xform.py` or
  `specs/cli/reconstruction/xform/xform-command.md`. Still open.
- **C — Pose-aware per-tile source stacks (parallax-correct panoramas).**
  `PerSphericalTileSourceStack` still exposes only `build_rotation_only`; the spec
  still calls the pose-aware variant future work. Its dependency
  `WarpMap::build_with_pose_impl` does exist, so only the per-tile consumer is
  missing. Still open.
