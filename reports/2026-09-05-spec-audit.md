# Spec audit — 2026-09-05

Spec/code consistency snapshot at `077a83d` (main, after #367), produced by the
`audit-specs` skill. Read-only analysis; every spec was checked mechanically and a
sample was read against its code. The two prior spec audits are both retired
(2026-07-07 in #368, 2026-08-30 in #361), so this report and its successors are the
only record of coverage; their `### <path>` headings are what the sampler greps.

**Sample:** 17 of 130 specs read against their code (130 = every `.md` under `specs/`
and `docs/` minus the `README.md` indexes and `TEMPLATE.md`). Seed `20260905`, N = 10;
pool was the 117 specs never deep-read by a prior audit (the 13 read on 2026-08-30 were
recovered from git, `4483f05:reports/2026-08-30-spec-audit.md`, and excluded). Random
draw: `optical-flow`, `gpu-optical-flow`, `sift`, `estimate-intrinsics`,
`from-colmap-bin-command`, `candidate-track-spawning`, `affine-factorization`,
`match-command`, `rotation-locked-resection`, `motion-command`. Included regardless
of the draw: `track-cluster-matching` (27 prose lines shared with its Rust module, the
largest spec↔code overlap in the corpus, and 18 documented defaults),
`point-track-detail` (14 shared lines), `camera-intrinsics` (9 shared lines with
`state.rs`, 11 defaults), `epipolar-curves` (10 shared lines),
`sfmtool-pinhole-kernels` (22 lines shared with its fisheye sibling; kernels split per
family on 2026-09-04), and the two GUI specs whose code moved most since 2026-08-30 —
`mcp-server` (`mcp/` is 7 of the 10 most-changed files in the repo) and `panel-layout`
(`layout.rs` +1,297 lines). Corpus-wide mechanical checks below cover all 130.

Budget note: N = 10 is the skill default; the seven extra reads are evidence-driven
and are listed so the next run can see what it need not repeat.

---

## Mechanical findings (all 130 specs)

### 1. Documented defaults vs code

364 rows scraped from parameter tables (`Default` / `Value` / `Current` columns) and
from fenced code (struct-literal and keyword defaults) across 70 specs; 286 verified
against code by two agents (the 17 sampled specs' 78 rows were verified inside their
sections instead), keyed on (spec → owning command or function → parameter).

| slice | rows | match | mismatch | untied | scraper false positives |
|---|--:|--:|--:|--:|--:|
| `specs/cli/` + `docs/` (25 specs) | 119 | 116 | 0 | 0 | 3 |
| `specs/core/`, `gui/`, `formats/`, `workspace/`, `drafts/` (28 specs) | 167 | 152 | **1** | 1 | 13 |
| the 17 sampled specs (verified in their sections) | 78 | 77 | **1** | 0 | — |

**Mismatches (behavioural; code is right in both):**

| spec:line | owner | parameter | spec says | code says |
|---|---|---|---|---|
| `core/patch/sift-to-patch-reconstruction.md:192` | `sfm embed-patches --patch-size` | `patch_size` | `5.0` | `11.0` (`_commands/embed_patches.py:65`). Commit `d161988` (#243) raised it and updated `embed-patches-command.md:52` and `cluster-patches.md:215`, but missed this core spec's table. |
| `core/geometry/rotation-locked-resection.md:16-18` | `resect_translation` (core) | `max_error_px`, `min_inliers` | core defaults `8.0`, `10` | the core function has no defaults; both are required (`resect_translation.rs:150-157`). The values exist only in the binding signature. See that spec's section. |

Untied: `core/analysis/cluster-census.md:304` `flag_threshold` documents a knob of the
not-yet-built `census_echo` caller (no such symbol anywhere), and the spec itself says
the value is "not yet data-derived": forward-looking, correctly labelled.

Adjacent completeness gaps, not wrong defaults: the fenced `bundle_adjust(...)`
signature at `core/geometry/bundle-adjustment.md:245-267` omits three parameters the
binding has between `obs_point` and `opt_f` (`point_at_infinity=None`,
`protected=None`, `protected_loss_scale=3.0`, `sfmtool-py/src/geometry/bundle_adjust.rs:96-98`);
and `core/features/sift.md:662` documents `extract_sift(image, params=None)` where the
binding also takes `max_described=None` (`sfmtool-py/src/sift/extract.rs:268`).

Three defaults are documented as values the code reaches indirectly and are fine:
`ws init --max-features 8192` (Click `None`, resolved in `sift/extract_sfmtool.py:63`),
`epipolar --sweep-window-size 30` (set at `epipolar.py:226`), `pano2rig --face-size`
= width/4 (`rig/pano2rig.py:122-128`).

Scraper tuning for the next run (16 false positives): measured-result tables headed
`metric | value` (`ray-grid-projection.md:122-124`), `Typical values` range columns
(`spherical-tiles-rig.md:98-99`), cost-comparison tables (`camera-views.md:980-982`,
`flow-based-matching.md:156`), loop-counter inits inside pseudocode
(`randomized-kdtree-forest.md:120`), literal `output_path=` in usage blocks (three
format-spec rows), a type-column bleed (`panorama-command.md:38`), and example
values (`scale-by-measurements-command.md:47`, `getting-started.md:35`).

### 2. Duplicate prose between specs and code

Normalized prose lines (lowercased, markup stripped, ≥ 55 chars) bucketed across all
130 specs and every `//` / `///` / `//!` comment and Python docstring under `src/` and
`crates/`. Pairs sharing ≥ 4 lines:

| shared lines | spec | code / other spec |
|--:|---|---|
| 27 | `core/features/track-cluster-matching.md` | `sfmtool-core/src/features/cluster_match/mod.rs` |
| 22 | `core/camera/sfmtool-fisheye-kernels.md` | `core/camera/sfmtool-pinhole-kernels.md` (spec↔spec) |
| 14 | `gui/point-track-detail.md` | `sfm-explorer/src/point_track_detail/mod.rs` |
| 10 | `core/camera/epipolar-curves.md` | `sfmtool-core/src/camera/epipolar.rs` |
| 9 | `core/geometry/absolute-pose.md` | `core/geometry/epipolar-estimation.md` (spec↔spec) |
| 9 | `gui/camera-intrinsics.md` | `sfm-explorer/src/state.rs` |
| 8 | `cli/reconstruction/xform/localize-keypoints-command.md` | `cli/reconstruction/xform/refine-keypoints-command.md` (spec↔spec) |
| 8 | `core/patch/patch-cloud.md` | `sfmtool-core/src/patch/cloud.rs` |
| 8 | `core/reconstruction/batch-triangulation-api.md` | `sfmtool-core/src/reconstruction/triangulation.rs` |
| 7 | `core/features/track-cluster-matching.md` | `src/sfmtool/feature_match/_cluster_matching.py` |
| 7 | `core/patch/patch-normal-refinement.md` | `sfmtool-core/src/patch/normal_refine/params.rs` |
| 6 | `core/spherical/spherical-tiles-rig.md` | `sfmtool-core/src/spherical/tile_rig.rs` |
| 6 | `gui/action-log.md` | `sfm-explorer/src/action_log/mod.rs` |
| 5 | `core/features/track-cluster-matching.md` | `sfmtool-py/src/matching/cluster.rs` |
| 5 | `gui/scene-graph.md` | `sfm-explorer/src/scene.rs` |
| 5 | `gui/panel-layout.md` | `sfm-explorer/src/window.rs` |
| 4 | `core/spherical/tile-batched-consensus-atlas.md` | `sfmtool-core/src/spherical/photometric_ransac.rs` |
| 4 | `formats/matches-file-format.md` | `formats/sfmr-file-format.md` (spec↔spec) |
| 4 | `gui/camera-intrinsics.md` | `sfmtool-core/src/camera/report.rs` |
| 4 | `gui/multi-panel-image-browser.md` | `sfm-explorer/src/state.rs` |
| 4 | `gui/panel-layout.md` | `sfm-explorer/src/layout.rs` |

The top four spec↔code pairs were all pulled into the sample; their per-spec sections
say which copy should shrink. `patch-cloud`, `batch-triangulation-api`,
`patch-normal-refinement`, `spherical-tiles-rig` and `action-log` were **not** read
this run; their 6–8 shared lines are the leads for the next one. The three spec↔spec
pairs are parallel structure between sibling specs (two kernel families, two pose
solvers, two keypoint ops) and are assessed only where sampled.

### 3. Spec shape (61 `specs/core/` specs)

| measure | count |
|---|--:|
| `specs/core/` specs with no ` ```rust ` block anywhere | 34 of 61 |
| …of which convey the interface by another means (entry table, flag list, prose signature) — **not verified this run except where sampled** | — |
| specs with a `Non-goals` section (whole corpus) | 26 |
| specs with deferral language (`not yet implemented`, `future work`, `deferred`, `natural v2`, `later phase`) | 21 |
| standing specs carrying a `**Status:**` line | 0 |
| amendment drafts in `specs/drafts/` linked from their standing spec | 4 of 4 |
| headings matching work-order patterns (`Step N`, `Phase N`, `Consumers`, `migration`) | 18 across 10 specs |

The 34 no-`rust`-block specs are a shortlist, not a verdict. Of the 18 work-order-pattern
headings, the `Step N` / `Phase N` ones in `motion-command`, `select-by-distribution`,
`epipolar-curves` and `photometric-subsets-ransac` are algorithm stages (checked where
sampled: see `motion-command` and `epipolar-curves` below); the three format specs'
`Versioning and migration` sections are standing content. The remaining `Consumers`
headings (`cluster-patches` §"Consumers (future work, out of scope here)",
`batch-triangulation-api`, `point-correspondence`, `per-spherical-tile-source-stack`)
were not read this run.

### 4. Opening paragraphs

Mechanical pre-pass over all 130 first sentences: 40 carry a backticked identifier,
6 start with a symbol or link, 2 contain a link. Most backticks are file extensions in
CLI specs (`.sfmr`, `.matches`) and are fine. The bulk read is in the next block.

Bulk read of all 130 first paragraphs (one agent pass over the extracted list).
**34 of 130 fail the cold-reader test; 8 of those are the quiet failure**: true,
precise, and never says what the thing is for. The 15 worst, with a proposed first
sentence each. The fix is one sentence per spec and compounds for every future reader:

| spec | failure | first sentence today | proposed first sentence |
|---|---|---|---|
| `core/spherical/photometric-subsets-ransac.md` | not prose: opens on two Markdown link definitions | "[`PerSphericalTileSourceStack`]: per-spherical-tile-source-stack.md" | When several photographs all see the same direction on the sphere, this picks the largest group whose pixels actually agree, so a panorama tile is coloured from a consensus rather than an average and the views that disagree are labelled as occlusion or parallax. |
| `core/features/sift.md` | change proposal in conditional voice | "sfmtool relies on COLMAP and OpenCV for many of its algorithms…" | This is sfmtool's own SIFT feature extractor in Rust: it detects scale-space keypoints and computes their descriptors, so the pipeline and the GUI can extract features without shelling out to an external binary. |
| `core/camera/image-warping.md` | opens on an absence | "The Rust codebase has complete implementations of `distort()` and `undistort()` for all 11…" | Image warping applies a camera model's distortion, or its inverse, to a whole image rather than one pixel at a time, so undistortion, re-distortion and camera-model conversion happen in sfmtool with control over the output camera and interpolation. |
| `core/camera/epipolar-curves.md` | formula first; never says what the module does | "The epipolar constraint `p2ᵀ F p1 = 0` only holds when…" | For a feature in one image, the places it can appear in another form a straight line only for pinhole cameras; for fisheye and distorted lenses that locus bends, and this spec says how sfmtool traces those curves so match checks and visualizations work for any camera model. |
| `core/patch/patch-normal-refinement.md` | symbol soup before words | "A reconstructed 3D point `X` is seen by cameras `{(Kᵢ, Tᵢ)}`." | Every reconstructed point sits on a small piece of surface whose facing direction starts as a guess; this refinement rotates it to the orientation that makes the point look most alike across all the images that see it. |
| `core/analysis/source-clusters.md` | repo vocabulary as common usage | "A reconstruction member is drawn from a cluster selection and holds a subset of its clusters." | A reconstruction is usually built from a few large feature groups, leaving most of the groups its images share unused; this finds those left-behind groups and hands them back a feature-size band at a time. |
| `core/analysis/cluster-census.md` | never names or defines the thing | "A reconstruction can be internally consistent and wrong." | A cluster census scores a finished reconstruction against feature correspondences the solve never used, so a group of cameras attached at the wrong pose, or geometry bent to absorb a wrong focal, becomes a measurable fraction of violated evidence. |
| `core/features/track-cluster-matching.md` | describes the status quo it replaces | "Traditional SfM feature matching is pair-centric." | Track-cluster matching finds correspondences across a whole image set at once: it indexes every image's SIFT descriptors together and reads off groups of mutually-near descriptors, so a point's observations in many images emerge as one group instead of being stitched from pairwise matches. |
| `core/patch/cluster-patches.md` | opens with a link and summarizes a different spec | "[Track-cluster matching](…) materializes candidate track clusters…" | A patch cluster is a group of matched features fitted to the actual image content: one member is the reference, and every other member carries a photometrically refined affine warp mapping the reference's patch into its image, all computable before any camera pose exists. |
| `cli/reconstruction/embed-patches-command.md` | bare mode identifiers; no purpose | "Convert a `sift_files` reconstruction into an `embedded_patches` `.sfmr` — a wholesale switch of `feature_source`." | Rewrites a reconstruction so it no longer depends on the external feature files it was solved from: each observation's pointer into a .sift file becomes an image patch and keypoint stored inline, producing a self-contained .sfmr. |
| `core/spherical/per-spherical-tile-source-stack.md` | bare symbol plus index notation | "Many algorithms operating on a `SphericalTileRig` need the same input: for each tile `t`…" | Panorama work divides the sphere into small tiles and asks, per tile, what each source photograph saw in that direction; this gathers exactly that, each source warped into the tile's frame as an image pyramid so coarse-to-fine algorithms share one warp. |
| `core/geometry/estimate-intrinsics.md` | bare function name; needs the link to parse | "`estimate_intrinsics` is the high-level face of the structure-free focal vote…" | Estimates a camera model family and shared focal length for a set of images from feature correspondences alone, before any reconstruction exists, returning one verdict for callers that want a camera rather than a diagnostic table. |
| `core/features/optical-flow.md` | quiet | "A pure-Rust DIS (Dense Inverse Search) optical flow implementation in sfmtool-core…" | Dense optical flow estimates where every pixel of one image moved to in the next, the correspondence signal video-based SfM uses to generate candidate tracks; this is sfmtool's own DIS implementation, so flow is available without an OpenCV round-trip. |
| `core/camera/camera-model-registry.md` | quiet, meta | "Two types describe a camera in this workspace, and there is exactly one place where they meet." | A camera is represented two ways in sfmtool, a loosely-typed record mirroring the on-disk reconstruction and a closed enum the algorithms compute with; this spec says why the split is kept and how the conversion stays exhaustive and generated in one place. |
| `core/features/covisibility-selection.md` | quiet | "Three queries over a set of images' shared-cluster counts…" | Three questions a caller asks before deciding which images to reconstruct with, all answered from how many feature groups each pair of images shares: how different two overlapping views look, which images are redundant, and how much of the capture a subset still reaches. |

Also failing, below the cap: `flow-based-matching`, `bundle-adjustment` (opens on two
script paths), `affine-factorization`, `cluster-selection`, `matches-file-format`,
`gui/action-log`, `gui/camera-intrinsics`, `gui/mcp-server`,
`member-coherence-validation`, `blender-viewport-navigation-implementation-overview`,
`gui/patch-rendering`, `randomized-kdtree-forest`, `cluster-patches-command`,
`localize-keypoints-command`, `motion-command`, and all four `specs/drafts/*-amendment.md`
(each opens "Amends [link]…" and never says what the proposed change is).

Exemplary, for the next writer: `core/patch/cluster-patch-refinement.md` (describes
the input in plain terms before naming any type), `formats/sift-file-format.md` (what
the file holds, then why the design), `gui/cross-panel-hover.md` (the behaviour as the
user sees it, then why it sits beside click-to-select).

### 5. Coverage both ways

| measure | count |
|---|--:|
| distinct spec paths cited from code (`specs/…md` in a `.rs`/`.py`/`.wgsl`) | 76 |
| code files citing a spec | 147 Rust, 28 Python |
| specs never cited from code | 54 of 130 |
| …of which CLI command specs | 24 of 30 (only 8 of 29 `_commands/` modules cite any spec) |
| …of which `specs/gui/` | 10 of 20 |
| cited spec paths that do not exist | 0 |
| spec→code relative links that do not resolve | 0 |

The CLI number is a convention gap, not drift: `_commands/` modules mostly carry a
docstring and no spec link, so a reader at the command has no pointer to the design.
Cheap fix, one line per module. The GUI half is the same pattern in `sfm-explorer`.
Full surface table under **Code without specs**.

---

## Sampled specs (17 read against code)

### specs/core/features/optical-flow.md
**Summary:** The pure-Rust DIS (Dense Inverse Search) dense optical flow in `sfmtool-core`: the algorithm and update rule, the parameter table and three presets, module layout, key types, the Rust and Python entry points, initial-flow chaining, four optimization layers, and benchmark/validation tables.
**Implementing code:** `crates/sfmtool-core/src/features/optical_flow/` (`mod.rs`: `compute_optical_flow`, `_timed`, `_with_init`, `compose_flow`, `resize_flow_to`; `params.rs`: `DisFlowParams::{fast,default_quality,high_quality}`; `dis.rs`, `variational.rs`, `pyramid.rs`, `interp.rs::densify_flow`, `flow_field.rs`, `image.rs`); bindings `crates/sfmtool-py/src/flow/optical.rs`.
**Inconsistencies:** (code is right in each)
  - Spec 268–278 "Cross-Validation Against OpenCV": a table of five Python tests with thresholds and "worst actual" numbers describes tests that do not exist. `grep -rn "DISOpticalFlow\|calcOpticalFlow" tests src crates` is empty; `tests/matching/test_flow.py` uses OpenCV only to synthesize images.
  - Spec 262–266 benchmark groups are prefixed `features/`; `benches/optical_flow.rs:48-150` registers `optical_flow/end_to_end` etc. `cargo bench -- features/optical_flow/pyramid` matches nothing.
  - Spec 280–283 "Dependencies: No new crate dependencies." `Cargo.toml:13-14`: `default = ["gpu"]`, `gpu = ["dep:wgpu", "dep:pollster", "dep:bytemuck"]`. Neither flow spec mentions the `gpu` cargo feature, nor that `--no-default-features` leaves an unconstructable `GpuFlowContext` stub (`mod.rs:49-53`).
  - Module tree at 118–141 lists five files; `mod.rs:38-46` declares nine (`flow_field.rs`, `image.rs`, `params.rs`, every `tests.rs`).
  - Binding list at 192–199 omits `compute_optical_flow_timed` (`optical.rs:365`, asserted by `test_flow_registration.py:11`). The preset table (111) names the middle preset `default_quality`; the bindings accept `"fast" | "default" | "high_quality"` (`optical.rs:351-354`), so `preset="default_quality"` raises `ValueError`.
  - "Public API" (176–188) gives no Rust signatures. `compute_optical_flow_with_init(img_a, img_b, params, initial_flow, gpu)` (`mod.rs:400`) puts `params` before the initial flow; the Python order is `(img_a, img_b, initial_flow_u, initial_flow_v, preset, use_gpu)`.
  - Defaults correct: θ_ps 8 / θ_ov 0.4 / θ_it 12, θ_sf = coarsest−2, the θ_ss formula, δ 5 / γ 10 / α 10 / θ_vi 7, θ_vo = base×(s+1), ε 0.001, all three preset rows, `gpu_min_pixels` 50 000.
**Non-goals / deferrals checked:** none in this spec.
**Third copies:** `variational.rs:37-45` (9 lines) and `params.rs:24-29` (6 lines) each re-derive the Jacobi-vs-SOR rationale (spec 91–94, 236–238), and disagree: "~1.3–2× the equivalent SOR count" vs "roughly 4/3×". Shrink `variational.rs` to a contract line + pointer so the number lives once.
**Shape:** (3) The Rust interface is prose only, no signature, no example; `mod.rs:8-36` has two runnable examples the spec could adopt. (5) "No new crate dependencies" is change-relative; "Motivation" (8–14) argues for building the thing; the "Worst actual" column is a PR result table.
**Recommendation:** update spec: delete the OpenCV cross-validation section, fix the bench names and dependency claim, refresh the module tree, add Rust signatures.
**Unclear / incorrect / suspicious:** **The densification weight is dead.** `interp.rs:113` computes `weight = 1.0 / diff.max(1.0)` where `diff = |tgt − ref|` on images normalized to [0, 1] (`image.rs:50`), so `diff ≤ 1` always and the weight is exactly 1.0 for every pixel. The "photometric-error-weighted averaging (Eq. 3)" of spec 62, 166–167 is a uniform average; the DIS paper works on 0–255 intensities where the clamp bites, and the GPU shader replicates the formula. Either the weight should be `1/max(1, 255·diff)` or the spec should stop claiming error weighting. Also `pyramid.rs:20` documents a "6-tap binomial `[1, 5, 10, 10, 5, 1] / 32`" kernel while `GAUSS6_1D` (`:90`) is a σ = 1.0 Gaussian.

### specs/core/features/gpu-optical-flow.md
**Summary:** The wgpu compute-shader implementation of the same DIS pipeline: measured speedups, the CPU/GPU hybrid and its per-level `gpu_min_pixels` routing, transfer-minimizing decisions, buffer pools, the five shaders, buffer layout, GPU-vs-CPU agreement, a WGSL Jacobi listing, and timing profiles.
**Implementing code:** `crates/sfmtool-core/src/features/optical_flow/gpu/` (`mod.rs`: `GpuFlowContext::{new, run_dis_and_variational, build_gpu_pyramid, run_gpu_levels_prebuilt}`; `context.rs`, `dis_pipeline.rs`, `variational.rs`, `pyramid_pipeline.rs`, `shaders/*.wgsl`); routing at `../dis.rs:69`, `../mod.rs:177-185`.
**Inconsistencies:** (code is right in each)
  - The WGSL "Jacobi Kernel Reference" (213–283) does not compile against the real bind-group layout and contradicts the spec's own §4: it shows eleven `@group(0)` storage bindings with separate `a11 … b2`; `jacobi_step.wgsl:17-25` has eight, with `coefficients: array<vec4<f32>>` at binding 6, and the shader header says the packing exists "to stay within the 8-storage-buffer-per-stage default limit". Entry point is `main`, not `jacobi_step`.
  - "Persistent Buffer Pools" (104–109): "`Mutex<Option<Pool>>` … after the first frame pair, no further GPU allocations occur". No `Mutex` exists under `optical_flow/`; `mod.rs:134-135, 288-289` call `create_pool(device, …)` on every invocation.
  - §1 Gaussian Pyramid (115–116): "Workgroups use shared memory to load a tile + 3-pixel halo." No `var<workgroup>` or `workgroupBarrier` in any of the nine shaders; `blur_downsample.wgsl:35-60` reads straight from storage.
  - Buffer-layout table (185–195) lists `patch_results: storage<vec3<f32>>`; `inverse_search.wgsl:28-29` writes two `array<f32>` buffers, `patch_flow_u`/`patch_flow_v`. The same table lists `a11, a12, a22, b1, b2` as five buffers, contradicting the packed `vec4` at line 152.
  - Spec 125 "sequential loop over the 64 pixels (8×8 patch)": the shader loops `ps × ps` from `params.patch_size`, and every GPU timing table in the spec is `high_quality` where `patch_size = 12` (144 pixels).
  - Spec 56–61 places the routing decision in `refine_flow_at_level`; that is one of two. `mod.rs:177-185` picks a `gpu_start_scale` for the batched path, gated on `params.variational_refinement`, so the `fast` preset never takes the single-submission path the diagram (42–54) shows.
  - Correct: `gpu_min_pixels` 50 000; merged DIS+variational submission; GPU pyramid + one seed readback; `resize_flow_to` identity return; the 6-tap kernel constants; workgroup sizes 64 / 16×16.
**Non-goals / deferrals checked:** 1, still true ("Future Work: workgroup shared memory for Jacobi"). Per AGENTS.md that belongs as a present-tense sentence plus a `specs/drafts/` amendment, not a "Future Work" heading.
**Third copies:** `gpu/mod.rs:103-110` (8 lines) re-derives the transfer-saving rationale of spec 79–82 and is the more precise copy; the spec bullet is the redundant one.
**Shape:** (1) Opens "This document describes the wgpu compute shader implementation…", meta-framing. Propose: "The DIS optical flow pipeline runs its five per-level stages as wgpu compute shaders, so pyramid levels above `gpu_min_pixels` are refined on the GPU while coarse levels stay on the CPU." (4) The 66-line WGSL listing is self-described as a "Transliteration of `jacobi_pixel_scalar_to_row`" and has already drifted; replace with the binding contract plus a link to the shader.
**Recommendation:** update spec: four structural claims (persistent pools, shared-memory pyramid, buffer layout, WGSL listing) describe code that is not there.
**Unclear / incorrect / suspicious:** `inverse_search.wgsl:1` says "one thread per patch (Option B from spec)", an option list in neither spec. `blur_downsample.wgsl:9` says the pass is "selected by `params.pass`" but `Params` has no `pass` field; selection is by entry point. `gpu/mod.rs:4` still titles the module "GPU-accelerated variational refinement" though it owns DIS, pyramid and upsample too. Neither flow spec is cited from any source file.

### specs/core/features/sift.md
**Summary:** sfmtool's pure-Rust SIFT detector/descriptor: the five-stage algorithm and its parameter table, SIMD/rayon parallelism, the tiled DoG/detect fusion (Tier 1/1.5, Tier 2 rejected), the cap-aware coarse-to-fine octave walk, the Python extraction-orchestration pipelining, the detect/describe split API, and the PyO3 bindings.
**Implementing code:** `crates/sfmtool-core/src/features/sift/mod.rs` (`SiftParams`, `detect_keypoints`, `compute_descriptors`, `compute_descriptor`, `extract_sift`, `extract_sift_partial`), `sift/{scale_space,detect,orientation,descriptor,gray,simd}.rs`, `crates/sfmtool-py/src/sift/extract.rs`, `src/sfmtool/sift/extract_sfmtool.py`.
**Inconsistencies:**
  - **Documented signature would not compile.** Spec 535–536 shows `compute_descriptors(scale_space, keypoints)` and `compute_descriptor(scale_space, keypoint)`; `mod.rs:476-495` both take two more args, `magnification: f32, clamp: f32`. Code is right.
  - Spec 685–686 cites `crates/sfmtool-core/benches/sift.rs`; `benches/` holds only `kdtree_forest.rs`, `optical_flow.rs`, `patch_render.rs`. The real benchmark is `scripts/benchmark_sift.py` (`pixi.toml:118`, `bench-sift`), which spec line 443 already names.
  - Spec 664–665 names a backend file `extract_rust.py`; the file is `src/sfmtool/sift/extract_sfmtool.py` (spec line 445 has it right).
  - Module tree at 646–653 lists 5 files; the directory also has `gray.rs` and `simd.rs` (both relied on elsewhere in the spec) plus per-module `tests.rs`.
  - Spec 662 binding signature `extract_sift(image, params=None)`; `extract.rs:268` is `(image, params=None, max_described=None)` — the third arg is the binding for the spec's own `extract_sift_partial` story and is undiscoverable from the spec.
  - Parameter table 232–246 lists 13 of 15 `SiftParams` fields: `blur_radius_factor` (default 2.25) and `image_to_gray` are prose-only. All 13 tabled defaults verified against `impl Default` (`mod.rs:127-146`).
**Non-goals / deferrals checked:** 6. Five hold (CPU-only; no `--detect`/`--describe`/`--top-k` on `sfm sift`; `.sift` v1 single-descriptor-array; no Tier 2 blur-chain fusion; `gray_formula` still a plain string). One partly overtaken: 634–636 "pyramid can be rebuilt per-octave on demand … to settle with benchmarks" — `ScaleSpace::build_chain` + `extend_octave` (`mod.rs:283, 350`) already build each octave's last two levels lazily. Both amendment drafts link back (`sift-gpu-amendment.md:5`, `sift-incremental-extraction-amendment.md:6`).
**Third copies:** (a) `extract_sfmtool.py:190-250` (~50 lines) re-derives the whole "K scales with cores, memory-bounded, GIL/rayon never oversubscribe" argument of spec 417–445 — shrink the docstring to contract + link (spec 443–445 already points here). (b) `mod.rs:295-329` (~35 lines) re-derives the cap-aware-walk correctness proof of spec 210–228 and is longer than the spec's version — keep the two guards as an in-code invariant, cut the measured-percentage narrative. (c) `mod.rs:74-93` (~20 lines) repeats the `blur_radius_factor` mass argument and the `contrast_threshold` OpenCV/COLMAP comparison of spec 96–100, 154–159.
**Shape:** (1) Opening describes a proposed change in conditional voice for shipped code: "sfmtool relies on COLMAP and OpenCV … Adding a Rust implementation directly in sfmtool-core would give us more room for flexibility". Propose: "sfmtool-core implements SIFT feature detection and description in pure Rust, so the GUI, the matcher and the CLI can extract features without a COLMAP or OpenCV round-trip; this spec defines those library functions and their COLMAP-convention interface." (5) Residue: line 519 "**Yes — split keypoint finding from descriptor creation.**" answers a decision-memo question; 664 "a new `extract_rust.py` backend slots in" is future tense for done work.
**Recommendation:** update spec — six concrete drifts; the `compute_descriptors` signature is the one a reader copies and fails on.
**Unclear / incorrect / suspicious:** Spec 529 annotates `scale_space` with "(DoG can be freed)"; `mod.rs:256` says the DoG is never stored, which the spec's own Tier 1 section establishes — the API comment predates Tier 1. The ~110-line "Extraction-orchestration pipelining" section (396–506) is Python CLI content living in a `core/` algorithm spec; a candidate to move to `sift-command.md`, worth discussing rather than silently moving.

### specs/cli/image-feature/match-command.md
**Summary:** `sfm match`: five matching methods plus `--merge`, their options and mutual-exclusion rules, the camera-config interaction, and (since 239ee24) the split where `--cluster` writes only a clusters-bearing `.matches` and `--derive-pairs` owns the COLMAP two-view-geometry boundary artifact.
**Implementing code:** `src/sfmtool/_commands/match.py` (`match`, `_MODE_OPTIONS`, `_reject_stray_mode_options`, `_run_derive_pairs_mode`), `src/sfmtool/feature_match/_run.py` (`_run_matching`, `_write_clusters_matches`), `feature_match/_derive_pairs.py`, `feature_match/_pairs.py::pairs_from_matches`.
**Inconsistencies:**
  - None in this spec. Checked: all six defaults (`--sequential-overlap` 10, `--flow-preset` default, `--flow-skip` 5, `--cluster-alpha` 0.8, `--cluster-d` 10, `--cluster-preset` accurate → `match.py:102-152`); the 11 `--camera-model` names; the rejection rules at 65–69 (`match.py:34-57, 270-278, 334-345`); "`--cluster` opens no database" (`_run.py:80-124`, early return before any `pycolmap` import); default output path and `-clusters` suffix; `tvg-matches/` naming and provenance fields (`_derive_pairs.py:107-128`); metadata keys (`_run.py:335-361`).
  - **`specs/formats/matches-file-format.md:419-421` is stale against 239ee24** (reported here per the brief): "the geometric-verification step materializes the expansion by writing a new pairwise `.matches` … (the write-once workflow, unchanged)". Verification is no longer a step of the match run; it is on-demand `sfm match --derive-pairs`. Lines 22–24 and 411–418 of the same spec are already right. Code is right; that paragraph should name `--derive-pairs` and link the command spec.
**Non-goals / deferrals checked:** 2, both hold: rig same-frame pair exclusion is not applied by `--derive-pairs` (`exclude_index_pairs` exists only on the in-solve path, `_run.py:557`); nothing on the `--cluster` path consults `pycolmap`.
**Third copies:** the "two-view geometries exist for COLMAP's mapper alone, so verification is a boundary concern" rationale is stated four times: spec 130–134, `_derive_pairs.py:4-11`, `_run.py:44-52`, and the `--derive-pairs` help at `match.py:159-162`. Shrink the `_run.py` docstring — it argues the boundary in a function that no longer performs the derivation.
**Shape:** no shape findings — purpose-first opening, options table as interface, a runnable example per mode (200–221).
**Recommendation:** update spec — one paragraph in `matches-file-format.md:419-421`; this spec is in sync.
**Unclear / incorrect / suspicious:** `_run_matching` takes `cluster_min_size` (default 2, `_run.py:41`) and records it as `matcher_options["min_size"]` (`:94`), but no CLI flag exposes it, so a `--cluster` file's recorded options include a knob the Options table never mentions and a user cannot change. Document as fixed-at-2 or expose it.

### specs/core/geometry/estimate-intrinsics.md
**Summary:** `estimate_intrinsics`, the typed answer over the structure-free focal vote: model verdict, fisheye confirmation via certified rotation mass, consensus focal, verdict-only scan votes, plus `ColumnPolicy::Auto`'s four-reason weak-vote escalation that screens on a pinhole-only vote before paying for the camera-model columns.
**Implementing code:** `crates/sfmtool-core/src/geometry/estimate_intrinsics.rs` (`ColumnPolicy`, `EscalationReason`, `escalation_reasons`, `IntrinsicsOptions`, `IntrinsicsEstimate`, `estimate_intrinsics`); `crates/sfmtool-py/src/geometry/estimate_intrinsics.rs`; `src/sfmtool/_commands/estimate_intrinsics.py`; constants from `geometry/focal_vote.rs:244-279`.
**Inconsistencies:**
  - Fleet population contradicts the code doc written with it: spec 175 "42 captures, 6 of them fisheye"; `estimate_intrinsics.rs:103-105` "over 40 captures it fires on all 4 fisheye ones and on 9 of the 36 rectilinear ones". Both landed in 9148c33; one is a typo, and spec 170–171 ("three of them pool exactly 9") depends on it. Not verifiable from the repo; reconcile at the source of the measurement.
  - Spec 162 arithmetic: with step 1.0566 (161), a ratio of 1.153 is 2.58 grid steps, not "2.2".
  - Spec 158 "the consensus came from the rotation family" is loose: `escalation_reasons` gates on `vote.family == Some(Rotation)` (`:132`), and `focal_vote.rs:349-362` sets `family` to the *majority* family while `focal_px` is the pooled median unless bimodal. Say "the rotation family was the majority".
  - The implementation note (206–211) omits the binding's `columns=None` default, which is not `"auto"`: `:97,116-119` make `None` mean `Fixed` with both columns.
  - Defaults correct: `min_rotation_mass` 1, `THIN_POOL` 9, `MIN_POOL` 2, `MAX_EPIPOLAR_PAIRS` 18, `ORTHO_GRID_{N,LO,HI}` 48/0.3/4.0, `FAMILY_DISAGREEMENT_BAND` 0.25, `IntrinsicsOptions::default` = `Fixed` + both columns, binding `seed=0, epipolar_min_disp_frac=0.02, min_rotation_mass=1`. The Rust example call (109–111) compiles as written; the four `EscalationReason` names match `as_str` (`:88-94`).
**Non-goals / deferrals checked:** none in this spec ("Any future change to arbitration lives in the vote" at 197 is a placement rule).
**Third copies:** the fleet-measurement rationale is stated three times: spec 145–191, module doc `:4-31` (28 lines), and `escalation_reasons` doc `:98-118` (21 lines); plus `min_rotation_mass` doc `:162-173` (12 lines) restating spec 126–133. This triplication is exactly what let the 42/6-vs-40/4 drift happen. `escalation_reasons`'s doc should shrink to the four cut points + link, dropping the fleet counts.
**Shape:** (1) Opening is a bare backticked symbol defined by reference to another spec; a reader learns neither what a "focal vote" nor a "camera-model column" is. Proposed: "Estimating a camera's focal length and model from nothing but 2D cluster tracks, with no poses and no structure, is what the focal vote does, and it answers with a diagnostic table; this module turns that table into one camera: which model won, its focal, and whether a fisheye verdict is corroborated." No other shape findings.
**Recommendation:** update spec: fleet counts (with the code doc), the 2.2/2.58 figure, the majority-family wording, the binding's `columns=None` default.
**Unclear / incorrect / suspicious:** the CLI's `--json` payload (`estimate_intrinsics.py:472-490`) never surfaces `screening_vote`, and `_report_lines` does not print it, so the pinhole numbers spec 182–187 says a caller must read off `screening_vote` are unreachable from `sfm estimate-intrinsics --model auto` when escalation fires. Core spec and core code agree; the gap belongs to `estimate-intrinsics-command.md` (whose one defaults row, `--seed` 0, matches `:412`).

### specs/core/geometry/affine-factorization.md
**Summary:** Alternating-least-squares Tomasi–Kanade factorization with missing data and residual trimming (affine camera per image, 3D point per cluster, per-observation keep mask), plus the metric upgrade solving the symmetric `Q = A·Aᵀ` and returning both reflection hypotheses with per-image rotations and scales.
**Implementing code:** `crates/sfmtool-core/src/geometry/affine_factorization.rs` (`MAX_DENSE_ENTRIES`, `AffineFactorizationParams`, `AffineFactorization`, `MetricHypothesis`, `quantile_linear`, `lstsq`, `factorize_affine`, `metric_upgrade`); `crates/sfmtool-py/src/geometry/affine_factorization.rs`.
**Inconsistencies:**
  - Spec 197–199 names a consumer that no longer exists: "First consumer: `exp_pinhole_bootstrap.py` swaps its `als_factorize()` … for the bindings". That script was deleted in `ecfa671` (2026-09-02, "the pre-fisheye bootstrap retires"), which updated `cluster-census.md` and `cluster-covisibility.md` but not this spec. There is now **no** caller of `factorize_affine`/`metric_upgrade` outside `tests/rust_bindings/test_affine_factorization_rust_bindings.py`.
  - Spec 210 "parity with the first consumer constrains the results, not the method": the premise of open question 3 is void for the same reason.
  - Spec 91 says the metric upgrade fails only on "degenerate" systems; `metric_upgrade` also returns `None` when `λ_max` is NaN or ≤ 0 (`:466-468`), i.e. `Q` not positive-definite. The doc comment at `:414-415` has the same gap.
  - Defaults and signatures correct: `rounds` 25, `trim_fraction` 0.05 (`:45-52`, binding `:176`); `MAX_DENSE_ENTRIES = 4_194_304` = 64 MB; sweep minimums, trimming from `rounds/2`, strict `<` against the quantile, used-image ≥ 4, numpy `t >= 0.5` lerp branch: all match. `factorize_affine(oc, oi, oxy, n, c, rounds=25, trim_fraction=0.05)` and `hyps[0].gauge` run as written; all six getter shapes match.
**Non-goals / deferrals checked:** none; three Open questions checked instead: (1) monotone descent, still open; (2) rank-3 init alternatives, still open; (3) normal equations vs orthogonal decomposition, effectively settled (`lstsq` uses SVD, `:169-172`) and its parity premise is gone.
**Third copies:** none serious; the numpy-quantile "contractual" clause appears in three places as one clause each.
**Shape:** (5) Spec 197–199 is a migration plan for a deleted script. Otherwise a model spec: purpose-first opening, Rust API with rationale ("Core stays I/O-free"), worked binding example, invariant-level algorithm notes.
**Recommendation:** update spec: drop the retired-consumer bullet (say the bindings have no in-repo consumer beyond tests), revise open question 3, add the non-positive-`Q` failure to the contract.
**Unclear / incorrect / suspicious:** with the bootstrap script gone the module is dead weight outside its tests; worth a deliberate keep-or-retire decision rather than silent drift. Spec 56 says "fixed count, default 25" but `rounds == 0` is legal and yields zero cameras and raw residuals (`:260-264`); unstated.

### specs/core/geometry/rotation-locked-resection.md
**Summary:** The linear translation-only resection used when a camera's world-to-camera rotation is already known: cross-product rows `[r_k]ₓ·t = −[r_k]ₓ·R·X_k` in ray space, three rounds of trimmed IRLS against a pixel-residual gate, and a model-dependent cheirality test (half-space for the perspective family, positive range along the ray for `needs_ray_path` models).
**Implementing code:** `crates/sfmtool-core/src/geometry/resect_translation.rs` (`INVALID_RESIDUAL`, `TRIM_ROUNDS`, `RIDGE`, `TranslationResection`, `resect_translation`); `crates/sfmtool-py/src/geometry/resect_translation.rs`. In-repo caller: `geometry/rotation_init.rs:36,719` with its own `RESECT_MAX_ERROR_PX` / `RESECT_MIN_INLIERS`.
**Inconsistencies:** (code is right in each)
  - Spec 57–58: output is "the **survivors'** pixel residual norms". `residual_norms` is per **input** observation, length `n`, with `INVALID_RESIDUAL` (1e6) for non-survivors and invalid rays (`:217-225, 56-58`). The spec's own binding block (71) already says `(n,)`; a reader zipping against the inlier subset would index wrongly.
  - Spec 16–18 attributes defaults to the **core** inputs ("`max_error_px` … default `8.0`", "`min_inliers` (default `10`)"). The core function has no defaults; both are required (`:150-157`). 8.0/10 exist only in the binding signature (`py resect_translation.rs:43-44`). The one Rust caller passes its own constants.
  - Two contract behaviours unstated: `None` up front when `n < min_inliers.max(1)` (`:159`); observations whose `pixel_to_ray` is non-finite or near-zero are permanently excluded (`:163-186`).
  - Correct: `TRIM_ROUNDS = 3`, strict `<` gate, `c.z < 0.0` / `ray.dot(&c) > 0.0` tests (`:125-129`), `RIDGE = 1e-12`; the documented Python call runs as written and the dict keys/shapes match.
**Non-goals / deferrals checked:** 2, both hold (`refine_absolute_pose` exists for joint updates and this module touches no rotation; no RANSAC loop, only `TRIM_ROUNDS`).
**Third copies:** the sign-blindness/chirality argument is written three times at near-identical length: spec 41–55 (15 lines), module doc `:21-30` (10), `residual_norm` doc `:99-114` (16), plus a fourth short restatement in the binding. `residual_norm`'s doc should shrink to "the in-front test carries the chirality and is model-dependent; see the spec", keeping only that the perspective expression is unchanged, hence bit-identical.
**Shape:** not one of the five, but worth naming: a `core/` spec with no Rust signature anywhere. `TranslationResection`'s fields and `resect_translation`'s argument order are reachable only in prose (16–18, 57–58), and that prose is where both concrete divergences live. A Rust interface block would have prevented them.
**Recommendation:** update spec: per-observation residuals with `INVALID_RESIDUAL`, move the 8.0/10 defaults to the Binding section, add a Rust signature block.
**Unclear / incorrect / suspicious:** Spec 11–12 offers "a rig calibration, an external attitude" as motivating callers; the only in-repo caller is `rotation_init.rs`'s far-field skeleton, and the other two read as if they existed.

### specs/cli/colmap-interop/from-colmap-bin-command.md
**Summary:** `sfm from-colmap-bin`: importing a COLMAP binary reconstruction (`cameras.bin`, `images.bin`, `points3D.bin`) into a `.sfmr`, the COLMAP→canonical convention conversion applied at that boundary, and four options.
**Implementing code:** `src/sfmtool/_commands/from_colmap_bin.py:13-148`; `src/sfmtool/colmap/io.py` (`colmap_binary_to_rust_sfmr` 366–436, `_colmap_poses_points_to_canonical` 263–292, `_detect_infinity_points` 350–363); `colmap/convention.py`; `crates/sfmr-colmap/src/colmap_io/read.rs::read_colmap_binary` 43–170.
**Inconsistencies:**
  - Spec 5–6 and the Input Directory listing (41–46) name only the three `.bin` files. `read.rs:127-170` also reads `rigs.bin` and `frames.bin` when present, and `io.py:402` conjugates the rig sensor poses and carries rig/frame data into the `.sfmr`. A reader importing a rig solve would not learn that rig structure survives.
  - The spec never says `--image-dir` must sit inside an initialized workspace; `from_colmap_bin.py:93-98` hard-fails otherwise ("No workspace found at or above … Initialize one with 'sfm ws init'").
  - Spec 15 says `W` is applied to "points, infinity directions" on import; COLMAP stores all points finite, and `_detect_infinity_points` runs after conversion. Wording inherited from the export direction.
  - Defaults correct: `--tool-name` `unknown` (`:35`), `--detect-infinity` default on (`:38-44`); both usage examples run as written.
**Non-goals / deferrals checked:** none in this spec.
**Third copies:** none; `colmap/convention.py:4-22` and the `io.py` docstrings state contract + link without re-deriving `S`/`W`.
**Shape:** no shape findings.
**Recommendation:** update spec: add `rigs.bin`/`frames.bin`, the workspace requirement, and fix the infinity-directions clause.
**Unclear / incorrect / suspicious:** `from_colmap_bin.py:84` rejects a non-`.sfmr` output extension with a `UsageError` the spec does not mention; every failure is re-wrapped as `ClickException` (`:147`).

### specs/cli/reconstruction/motion-command.md
**Summary:** `sfm motion` in both modes: optical-flow adaptive-stride analysis of raw image sequences, and four-signal discontinuity detection (pose extrapolation, step-size ratio, covisibility drop, obs-count outlier) over a `.sfmr`, plus the v1 `--json` schema.
**Implementing code:** `src/sfmtool/_commands/motion.py`; `motion/image_sequence.py::analyze_image_sequence`; `motion/recon_discontinuity.py` (`analyze_reconstruction` 565, `_compute_extrapolation_errors` 95, `_compute_step_ratios` 287, `_compute_overlap_drops` 322, `_compute_obs_z_scores` 386, `_flag_frame` 420); `motion/constants.py`; `motion/report.py`; `visualization/_discontinuity_display.py`.
**Inconsistencies:** (code is right in each)
  - Spec 332: each flagged discontinuity reports "Whether the images are connected in the covisibility graph (graph distance)". No graph distance is computed or printed anywhere, and the JSON `discontinuities` table (433–448) has no such field.
  - Spec 283–285: step-ratio windows are "`STEP_RATIO_WINDOW` edges on each side (default 8)". `_compute_step_ratios:310-311` slices `[i-window+1:i]` and `[i+1:i+window]`: **7** edges each side, and needs ≥ 2 per side or the edge is `None`.
  - The covisibility section (290–299) omits the guard at `_compute_overlap_drops:344`: `if n < 3 * window: return [None] * n_edges`. On any sequence shorter than 48 frames the `Cov` signal never fires; for the checked-in datasets (17/24/26 frames) it is dead. It also needs ≥ 3 non-`None` neighbour edges (`:374`).
  - Spec 255–258 describes left extrapolation as "frames i-3, i-2, i-1 … Fit a quadratic (or cubic)". Code fits a 2-point linear and a 3-point quadratic and reports the **min** of the two errors (`:121-169`, mirrored right at `:175-223`); cubic is never fitted. The min-of-two is deliberate (suppresses far-end contamination) and belongs in the spec.
  - The Tests section (576–610) describes a 10-frame fixture with a jump between 4 and 5; `tests/test_motion_report.py:78-115` uses the 17-frame seoul_bull workspace with a break at 10→11 and asserts `len(discontinuities) >= 1`. The listed "NaN round-trips as `null`" test does not exist (nearest: `test_recon_json_has_no_nan_or_infinity`, `:161`).
  - Defaults correct: `--initial-stride` 1, `--min-stride` 1, `--max-stride` 32, `--no-adaptive` off (`motion.py:27-53`); thresholds step 1.5 / overlap 1.8 / obs_z 2.5 / pose factor 3.0 / rot 15.0 (`constants.py:21-33`, `report.py:234-240`); ratio bands 0.75, 1/0.75, 0.85, 1/0.85 (`image_sequence.py:207-212`). Undocumented: `--max-stride` is `IntRange(min=2)`.
**Non-goals / deferrals checked:** 6, five hold (no image-sequence `segments`; no `pose_trans_factor`/`pose_rot_deg` flags; no histogram comparison metric; no non-sequential mode; no image-sequence segmentation rule). The "extrapolation order" open question (629–631) is settled by the shipped linear+quadratic-min policy and should be closed.
**Third copies:** `recon_discontinuity.py:694-706` (13 lines) re-derives why `POSE_TRANS_FACTOR` is 3× and why rotation is fixed (spec 394–395, `constants.py:30-31`); `constants.py:4-19` (16 lines) restates the three secondary signals' rationale (spec 274–309). Keep `constants.py` as the in-code home; shrink the inline block to invariant + link.
**Shape:** (5) The "Code and patterns to build on" table (614–626) is a build plan, and two rows are counterfactual: `analyze/images.py::_compute_camera_centers()` / `_compute_rotation_angle()` exist but the motion code uses `RotQuaternion.camera_center` and its own `_rotation_angle_deg`. The Tests section is a test order. The "Step 1/2/3" headings are genuine pipeline stages (mirrored by `# Step N:` comments at `:663, 681`): keep.
**Recommendation:** update spec: graph distance, window size, the Cov short-sequence guard, the min-of-two extrapolation; retire the build-plan and test-plan sections.
**Unclear / incorrect / suspicious:** Spec 320–323 cites "the KerryPark 831→832 case"; no such frame numbers exist in `test-data/images/kerry_park` (24 rig frames). Two divergent copies of the classification bands: `_discontinuity_display.py:209-219` uses the literal `1.33`, `report.py:245-257` uses `1/0.75`, so a ratio in (1.33, 1.3333] prints as "acceleration" but is not flagged. `shared_points` comes through `build_covisibility_pairs`' 90° angle filter (`_image_pair_graph.py:45-55`), so `0` can mean "angle-filtered", which the JSON field description (445) does not convey.

### specs/core/patch/candidate-track-spawning.md
**Summary:** The `spawn_candidate_tracks` primitive: place a synthetic patch at an in-plane offset from a parent patch, localize/refine/triangulate it through the existing batch kernels, and report per-candidate status through three ordered gates. Also specifies exposing `starting_keypoints` on the `localize_keypoints` binding.
**Implementing code:** `crates/sfmtool-core/src/patch/spawn.rs` (`SpawnParams` 40–74, `SpawnStatus` 80–91, `SpawnedTracks` 96–120, `spawn_candidate_tracks` 170–384); `crates/sfmtool-py/src/patches/spawn.rs:65-208`; `patches/localize_keypoints.rs:131-138, 320-358`; tests in `patch/spawn/tests.rs` and `tests/rust_bindings/test_spawn_candidate_tracks_rust_bindings.py`.
**Inconsistencies:**
  - Spec 9–13 states as present fact "Two callers: surfel-normal expansion … densification …". Neither exists: the only references to `spawn_candidate_tracks` outside the primitive are the PyO3 wrapper (`sfmtool-py/src/patches/mod.rs:49`) and the two test modules. Say they are the intended callers, or that the primitive currently has none.
  - Spec 130–132: `starting_keypoints` is "same shape as `refine_keypoints`'s existing parameter". Not the same type: `refine_keypoints` takes `HashMap<u32, Vec<[f64;2]>>` (`refine_keypoints.rs:156`); `localize_keypoints` takes `HashMap<u32, Vec<Option<[f64;2]>>>` (`localize_keypoints.rs:156`), and the per-view `None` escape hatch is the point. The code's own doc (`:64-74`) says "same shape … with one addition".
  - Defaults and signature correct: `SpawnParams::default` (`spawn.rs:63-74`) = `resolution 24, search 6.0, max_shift_px 8.0, subpixel_sweeps 1, min_views 3, max_reproj_rms_px 2.0` matches spec 83–88; the binding's keyword-only tail matches; the call at spec 117–121 runs as written; status discriminants 0–3 match; all four documented `ValueError` cases are implemented (`py spawn.rs:98-144`).
**Non-goals / deferrals checked:** 3, all hold (no offset selection or track assembly in `spawn.rs`; no rayon of its own; infinity parent rejected at `py spawn.rs:124-133`). The two-callers claim is the reverse case: a claim of code that does not exist.
**Third copies:** `spawn.rs:4-26` (23 lines) restates the spec's Overview clause-for-clause, and `py spawn.rs:17-28` (12 lines) a third time. The module doc is defensible as the crate entry point; the binding docstring should shrink to Args/Returns + its existing spec link (`:64`). The two-callers sentence is currently wrong in all three copies.
**Shape:** (5) The "`localize_keypoints` gains `starting_keypoints`" section (126–139) is a change order ("the Python binding predates the parameter", "as part of this change") for a shipped parameter; re-tense into the API section. The Testing section (141–159) is a test order, though every listed case exists.
**Recommendation:** update spec: fix the two-callers claim, re-tense the `starting_keypoints` section, correct "same shape".
**Unclear / incorrect / suspicious:** the parent-at-infinity `ValueError` (122–124) is implemented but untested (`TestSpawnValidation` covers the other three). Spec 69–70 defines `high_reproj` as RMS over the gate; the code also forces `sum_sq = INFINITY` when the triangulated position fails to project into a surviving view (`spawn.rs:361-369`), so a projection failure surfaces as `high_reproj` rather than `bad_triangulation`.

### specs/core/features/track-cluster-matching.md
**Summary:** The background-floor track-cluster matcher: one k-NN query over a kd-forest built on the whole descriptor corpus, a per-descriptor radius `alpha · dist[i, d]`, density-ordered seeding into a hard cluster partition, and a derived per-image-pair match view. Carries the empirical tuning (α = 0.8, the `d` sweep), a cost section, and a three-layer "Production Implementation" section (Rust core, PyO3, Python/CLI).
**Implementing code:** `crates/sfmtool-core/src/features/cluster_match/mod.rs` (`BackgroundFloorParams`, `Clusters`, `PairMatches`, `background_floor_clusters`, `clusters_to_pair_matches`); `crates/sfmtool-py/src/matching/cluster.rs:84`; `src/sfmtool/feature_match/_cluster_matching.py:47`; `feature_match/_run.py:80-118, 265, 507, 546`; `_commands/match.py:126-153, 275-281`.
**Inconsistencies:** (code is right in every case)
  - Spec 842–881, 173–175, 250–253, 491–493, 958 say `sfm match --cluster` runs geometric verification and writes a TVG-bearing file under `workspace/tvg-matches/`. `_run.py:80-118` writes a clusters-only backbone into `workspace/matches/` with `-clusters` and opens no database; verification moved to `--derive-pairs` (239ee24). Spec 426–433, 915–933, 941–946 already say so, so the spec contradicts itself.
  - Spec 901–908 "`--camera-model` therefore stays available with `--cluster`"; `match.py:275-281` raises `UsageError`. `match-command.md:122` agrees with the code.
  - Spec 174, 854 name `pycolmap.geometric_verification(db_path)`; the codebase only calls `pycolmap.verify_matches(db, pairs_path, options=…)` (`_run.py:504,630`, `_derive_pairs.py:84`). The documented call would fail.
  - Spec 785–787 "Unlike `KdForest`, which `__init__.py` re-exports … they are not lifted to the package top level": both halves wrong. `sfmtool.background_floor_clusters` and `sfmtool.clusters_to_pair_matches` exist via `from sfmtool._sfmtool.matching import *` (`__init__.py:11`), and `KdForest` arrives the same way (`:14`).
  - Spec 352 calls the radius multiplier `bg_alpha`; it is `alpha` in every layer and in the spec's own table at 954.
  - Spec 955 lists `min_size` at layer "core/py/cli"; there is no `--cluster-min-size` flag, so `_run_matching(cluster_min_size=2)` is unreachable from the CLI.
  - Spec 474–480 presents the `d = 28` prototype's counts (seoul 1,550 / seattle 4,980 / kerry 3,153 / dino 29,644) as what production reproduces; §Choosing `d` (381–384) says `d = 10` gives seoul −20%, kerry −15%, dino 32,181. No sentence states 28 as shipped, but the 120–125 and 141–146 tables are unlabelled.
  - Spec 309–337 attributes ~50 s of pycolmap verification and a COLMAP-DB write to `sfm match --cluster`; that command no longer does either.
  - Spec 640–644 shows `search_batch_with_distances(…)`; code calls `search_batch_with_distances_ordered(…, forest.locality_order())` (`mod.rs:200`). Spec 578 `#[derive(thiserror::Error)]`; code hand-rolls `Display` (`mod.rs:106-132`). Both hedged, but under an "Algorithm (exact)" heading they read as prescriptive.
  - Defaults: all 18 rows verified correct (`d=10`, `alpha=0.8`, `min_size=2` at `mod.rs:62-69`, `cluster.rs:84-86`, `_cluster_matching.py:51-55`, `match.py:137-152`). Rows 359–363 (`threshold=cliff`, `cliff_pct=50`, `t_scale=1.0`, `refine=0`, `prefilter=off`) exist nowhere in `src/` or `crates/`: prototype-only, and the table is not marked as such.
**Non-goals / deferrals checked:** 8. Overtaken: (1) "Persisting clusters to disk is out of scope here" (870): `_write_clusters_matches` (`_run.py:265-376`) persists the CSR backbone; (2) "The CLI consumes only the pair output today; the clusters are returned so a future cluster artefact can be added" (867–869): inverted, `_run.py:108` keeps `clusters` and discards `_pairs`; (3) "Add a short `## Cluster matching` section to `match-command.md`" (910–913) is done (`match-command.md:92`). Five still hold: no `sfm solve --cluster`; no native TVG verifier; no mean-shift `refine`; no `cliff`/`otsu`/`gmm` threshold path; no isolated-point prefilter.
**Third copies:** the direction is spec→code. `mod.rs:4-22`, `cluster.rs:4-6` and `_cluster_matching.py:4-15` are contract-plus-link and should stay. The spec's `#### Public types` / `#### Public functions` (516–619, 104 lines) reproduce `mod.rs:45-157, 326-334` doc comments near-verbatim; 741–774 (34 lines) reproduces `cluster.rs:62-104, 144-162`; 809–831 (23 lines) reproduces `_cluster_matching.py:47-67`; add §Algorithm (exact) 621–685, §Parallelism, §Determinism and three §Tests blocks. **The spec is the copy that should shrink**: roughly 300 lines of Layer 1/2/3 down to a linked interface summary plus the invariants the code cannot carry (L2 vs squared, the density-order tie-break contract, the `k`-wide candidate stride at `alpha ≥ 1`).
**Shape:** (4) 621–702 is a step-by-step restatement of `background_floor_clusters`; the shipped switch to `_ordered` search was exactly the behaviour-preserving refactor that forces spec edits, and did not get them. (5) Residue throughout the Production section: "New `src/sfmtool/feature_match/_cluster_matching.py`, mirroring `_flow_matching.py`" (806), "Add a fourth matching method" (885), "Add an orchestrator in `_run.py`, e.g. `_run_cluster_matching(…)`" (875, with a signature that is not the shipped one), "> Factor the verify-and-write back half…" (872–873), "> Use whichever error idiom the crate already uses… `thiserror` above is illustrative" (589–591).
**Recommendation:** update spec: rewrite the Production Implementation section from a build brief into a present-tense contract-plus-link, and fix the seven factual drifts above.
**Unclear / incorrect / suspicious:** `matching_mode="cluster"` (`db_setup.py:130-146` → `_run_cluster_matching`) is live code no CLI reaches: `solve.py:303` only produces `"flow"`/`"exhaustive"`. The spec mentions the in-solve mode at 466–468 without saying it is unreachable, while asserting at 421 that no `sfm solve --cluster` exists. `cluster_match/covisibility.rs` + `covisibility/` (501 lines) live inside the module this spec owns but are specced in `cluster-covisibility.md`; the Layer 1 "Location" section (509–512) does not mention the sibling.

### specs/gui/mcp-server.md
**Summary:** The opt-in MCP control surface the viewer hosts with `--mcp`: the 23-tool catalog with every argument and reply shape, the wire-vocabulary rule, the single-threaded drain/defer architecture, the HTTP/loopback transport and its security posture, and the Rust seam (`Command`, `Outcome`, `apply_with_window`, `serve`).
**Implementing code:** `crates/sfm-explorer/src/mcp/tools.rs` (`catalog`, `parse`, `parse_set_view`, `parse_placement`), `mcp/mod.rs` (`Command`, `apply_with_window`, `screenshot_caption`), `mcp/server.rs` (`serve`, `APPLY_TIMEOUT`), `mcp/{read,render,display,layout,window,frame,view,write}.rs`; `src/cli.rs::DEFAULT_MCP_PORT`; `src/state.rs:155-208, 258-334`; `Cargo.toml:44-50, 82, 90`.
**Inconsistencies:**
  - **Spec 1585, `serve`'s documented signature does not compile.** Spec: `serve(port, tx: UnboundedSender<Request>, proxy: EventLoopProxy<UserEvent>)`; `server.rs:97-101`: `serve(port, tx, wake: impl Fn() + Send + Sync + 'static)`. The spec contradicts itself: line 1899 says `serve` "takes a wake closure rather than an `EventLoopProxy`, so nothing in `server` depends on winit".
  - **Spec 1651, the Cargo feature list would fail to build.** It lists `dep:serde_json`; `Cargo.toml:44-50` has `mcp = ["dep:rmcp", "dep:tokio", "dep:axum", "dep:serde", "dep:base64"]`, and `serde_json` is a plain non-optional dependency (`:82`). Cargo rejects `dep:` on a non-optional dep.
  - Spec 51–57, the `cli.rs` flag description: omits `-h` (`cli.rs:69`), and "treats everything else as a path" is false; `cli.rs:86-90` refuses any `-`-prefixed argument with `sfm-explorer has no option "…"`, and `--mcp=<non-number>` is a hard error.
  - Spec 790, 1327–1336 vs `tools.rs:1211-1216`: the exact form's companions are required, not preserved. With `orientation_wxyz` present, `parse_placement` calls `required_vec3("position")` and `required_f64("target_distance")`, so `{ "orientation_wxyz": […] }` alone is refused with `set_view needs position.` Code is right (the exact form is a verbatim restore).
  - Spec 924–926, the screenshot text block: `server.rs:264-266` emits `"{w}×{h} px. {caption}"` and the caption (`mod.rs:589-611`) is itself `The window, {w}×{h}. …`, so the size appears twice.
  - Spec 2041: `screenshot` `max_dimension` has an undocumented floor, `"minimum": 16` (`tools.rs:557`).
  - Defaults verified exact: `--mcp` 8787; `list_camera_images` 50/500; `get_action_log` 200/1000; apply timeout 10 s; `set_view` FOV 5–160; `SCALE_LADDER`, `GRID_LADDER`; every `get_image_detail_display` default; `TINT_PALETTE` 7; 23 tools in the spec's order with `screenshot` annotated read-only (asserted in `mcp/tests.rs:3051-3072`); every Action Log text verbatim; `ttl_ms(0)`, `CacheScope::Private`, loopback bind, three-origin allowlist, `rmcp = "~3.2"`. Reply shapes for all 23 tools match field for field.
**Non-goals / deferrals checked:** 19, all still absent (the five reserved loose-image tool names, `set_camera_intrinsics`, `run_align`, MCP resources, subscriptions, `kinds` filter, `near`, panel position, monitor-by-name, exclusive fullscreen, remote/auth/TLS, headless, persistence, and the five open questions). None overtaken.
**Third copies:** `window.rs:26-32` (7 lines) re-derives the minimized-and-maximized precedence rationale the spec carries at length. `tools.rs:7-17` restates the wire-vocabulary rule but links the spec: leave it.
**Shape:** (5) Spec 686–693 describes the prior state ("Before this pair existed the Image Detail toolbar … wrote straight into the two settings structs and logged nothing"). Rewrite present-tense. Otherwise a model of the form: cold-readable opening, catalog table + `Command` block as interface, rationale and worked JSON throughout.
**Recommendation:** update spec: three copyable-and-wrong artefacts (the `serve` signature, the feature list, the CLI flag description) plus the `set_view` exact-form requirement and the prior-state paragraph.
**Unclear / incorrect / suspicious:** the doubled size in the screenshot text block can disagree for a panel (the caption's is the last laid-out size, the prefix's the actual encoded size), which is exactly the frame the spec says the caption is unreliable in. Drop one; the spec should say which.

### specs/gui/panel-layout.md
**Summary:** The lifecycle of the dock arrangement and the window's placement as one document: closing/re-opening panels, the Panels menu, home positions, the versioned `sfm_explorer_layout` JSON file with its full validation vocabulary, monitor fitting, the startup default-layout file, and the `layout.rs` / `window.rs` Rust API.
**Implementing code:** `crates/sfm-explorer/src/layout.rs` (`LAYOUT_VERSION`, `Layout`, `LayoutNode`, `LayoutWindow`, `LayoutError`, `Home`, `Tab::{ALL, wire_name, home}`, `WindowLayout`, `default_layout_path`, the eleven `AppState` methods, `panels_menu`), `layout/tests.rs::DEFAULT_JSON`, `src/window.rs` (`WindowState`, `MonitorInfo`, `NormalRect`, `fit_to_monitor`, `WindowHost`), `src/dock.rs`, `src/cli.rs`, `src/sfmtool/_commands/explorer.py`.
**Inconsistencies:**
  - **`layout.rs:127-128` documents the wrong error text; the spec is right.** The `LayoutError` doc says `Display` produces `main.second.first: unknown key "fracton"`; `WindowLayout::from_value` passes `"layout"` as the root path (`:629`), so the real message is `layout.main.second.first: …`, which spec 430–435 and the tests assert. **Update code**, one doc line.
  - Spec 105, 163, 615: three future-tense references to MCP that has shipped ("and, later, an MCP `show_panel`"; "these two functions are also what the MCP tools **will** spell panels with"). `show_panel`/`hide_panel`/`set_window_layout`/`get_window_layout` exist (`mcp/tools.rs:507-523`), and the spec's own Non-goals already point at them as built.
  - Otherwise none. Checked: `LAYOUT_VERSION = 2`; the home table (`left 0.18` / `below 0.20` / `right 0.33`, viewer at root) against `Tab::home` (`:226-241`); the seven wire names; `DEFAULT_LAYOUT_FILE_NAME`; the default-layout JSON is byte-identical to `DEFAULT_JSON`; every validation message against `layout.rs:476-634` and `window.rs:226-288, 447`; version-check ordering; `WindowHost::apply`'s documented order against `window.rs:568-604`; `--no-default-layout` in `cli.rs:70` and `explorer.py:33, 66-67`.
**Non-goals / deferrals checked:** 9, all still true (no last-state restore, no collapse/scroll serialization, no panel duplication, no drag logging; `is_closeable` not overridden; no `.sfmtool/layout.json`; no `--layout PATH`; per-axis fitting; off-screen rectangle left to the window manager).
**Third copies:** `window.rs:14-20` (7 lines) reproduces § "Why a host trait…" nearly sentence for sentence; `window.rs:383-392` (10 lines) reproduces the opening of § "Fitting a rectangle to the desktop". Both link the spec and are short; if either shrinks, the `fit_to_monitor` one, to its return contract plus link.
**Shape:** no shape findings.
**Recommendation:** update code (one `LayoutError` doc line) and update spec (three "later"/"will" residues); everything else matches.
**Unclear / incorrect / suspicious:** the brief's premise that aed8048 split `dock.rs` into a `dock/` directory is wrong: it split `TabViewer::ui`'s 431-line match into per-tab methods inside `dock.rs`, which is still one file with `dock/tests.rs` beside it. Separately, `sfm_explorer_layout` present but not a number yields `Not a layout file` (`layout.rs:584`), which § "Validation" does not list among the version rules.

### specs/gui/point-track-detail.md
**Summary:** The Point Track Detail dock tab: the point-summary header (Point ID, xyz, error, track length, max pair angle, depth-z, cond), the stored-patch tile, the per-observation table (thumbnail, patch tile, image, name, feat #, size, error, angle, xy), row interactions, cross-panel selection effects, and the panel's state and response types.
**Implementing code:** `crates/sfm-explorer/src/point_track_detail/{mod,prepare,header,table,patch}.rs` (`PointTrackDetail`, `TrackObservationData`, `THUMB_SIZE`/`PATCH_TILE` = 48, `STORED_PATCH_SIZE` = 64, `ERROR_RAMP_MAX_PX` = 2.0, `format_feature_size`, `build_patch_frame`); numerics in `crates/sfm-explorer/src/metrics.rs`; plumbing `dock.rs:412-470`; cache `state.rs:531-535, 1156`.
**Inconsistencies:** (code is right in each)
  - Spec 248 says Size is "one averaged number when near-circular and `<larger>x<smaller>` when oval"; `format_feature_size` (`table.rs:417-423`) always prints `{major:.1}x{minor:.1}`, which is what spec 182–187 says. 248 is stale.
  - Spec 398–400 "the SIFT cache is pre-populated … by `app.rs`"; it is `dock.rs::show_point_track_detail` (`dock.rs:415-434`), which also pre-caches full-res images (`:437-455`).
  - The `PointTrackDetail` block (323–347) is stale four ways vs `mod.rs:66-97`: `prepared_point` is `Option<PointRef>` not `Option<usize>` (deliberately, so a new recon reusing an index re-prepares); the texture maps are keyed by `ImageRef`, not `usize`; `inverse_depth_z`, `condition_number`, `scroll_offset_y` are missing though 238–242 documents the first two as displayed. `TrackObservationData` (350–363) lacks `feature_extents` and `image_full_name` (`mod.rs:53-62`), the two fields the Size column and the name tooltip need.
  - Spec 205–211: `full_res_cache` is "keyed by image index" and "cleared when the reconstruction changes"; it is keyed by `ImageRef` and dropped per-recon by `AppState::forget_recon` (`state.rs:777`).
  - Spec 77's header example prints `max∠: 12.3°`; `header.rs:76` prints `max pair angle: 12.3°` (as spec 237 says); the example omits `depth z`/`cond` and puts the colour swatch last where the code draws it first.
  - Spec 150–174: the "thumbnail dot's colour" paragraph (162–172) is pasted inside the Columns table, so the `Angle` and `Feature (x, y)` rows (173–174) render as literal `|`-text.
  - Correct: the 0–2 px error ramp, grey NaN dot, 48 px tiles / 64 px render, patch column offset, 128×128 source thumbnails, sort by image index, `Feat #` semantics, response fields.
**Non-goals / deferrals checked:** 5. Still absent: stored-patch alpha tile, per-ray hover, reprojected second dot. Partly overtaken: "3D uncertainty visualization" (424–428): `metrics::compute_point_diagnostics` (`metrics.rs:79-121`) already computes depth uncertainty and the header shows `depth z`/`cond`. The `TODO (unbounded growth)` LRU (213–219) is genuinely unimplemented.
**Third copies:** `table.rs:28-44` (17 lines) re-derives the fixed-ramp/grey-NaN rationale of spec 162–172; `table.rs:405-416` and `prepare.rs:337-344` are a second and third copy of the half-vector-doubling argument of spec 176–187; `mod.rs:83-92` is byte-identical to spec 334–344. Shrink the spec's state-block comments (pure transcription) and collapse `prepare.rs:337-344` to a pointer; keep the constant-side rationale in `table.rs`, since it lives on the constant it justifies.
**Shape:** (4) The Panel State block transcribes the struct with its doc comments, so every field rename breaks it, and already has. (5) The `> **TODO (unbounded growth)**` block is work-order residue in a standing spec. Also: **the spec carries no relative link to its implementing code**; nothing points at `point_track_detail/` or `metrics.rs`. 5ff41ed's move left nothing stale only because there was nothing to be stale.
**Recommendation:** update spec: every divergence is the spec trailing the code; add the code links and re-fix the broken column table.
**Unclear / incorrect / suspicious:** `mod.rs:11` says "the work lives in five children" and lists four; the fifth was `metrics`, moved out by 5ff41ed.

### specs/gui/camera-intrinsics.md
**Summary:** Three coupled pieces: the Camera Intrinsics scene-graph group (and the Cameras→Camera Images rename), the independently toggled intrinsics overlay on Image Detail (principal point, angular axes, iso-rings, distortion field, hover readout), and the Camera Intrinsics dock panel (parameters, derived rows, projection plot, extrinsics/rig block), plus the `camera::report` core API and the "trustworthy domain" theory both rest on.
**Implementing code:** `crates/sfmtool-core/src/camera/report.rs` (all nine documented functions; signatures match); `crates/sfm-explorer/src/scene.rs:114-125, 207`; `state.rs:135-175, 429, 775-795, 854-930`; `scene_graph/cameras.rs`; `image_detail/intrinsics/{mod,controls,axes,field,hover}.rs`; `intrinsics_detail/{mod,header,parameters,derived,extrinsics,format,projection_plot}.rs`.
**Inconsistencies:**
  - Spec 171 names `AppState::retain_recon`; no such symbol exists. The filtering is `forget_recon` (`state.rs:775-779`) and `select_recon` (`:791-795`).
  - **The worked example (872–898) is internally impossible.** Headed `kerry_park · Camera #0 · OPENCV_FISHEYE`, it lists the four parameter rows of `SIMPLE_RADIAL_FISHEYE`; `OpenCVFisheye` (`intrinsics.rs:68-77`) has eight (`focal_length_{x,y}`, `principal_point_{x,y}`, `k1..k4`).
  - Spec 917 shows `35 mm equivalent | 19.1 mm` in the fisheye derived table; `equiv_focal_length_35mm` returns `None` for every `needs_ray_path()` model (`report.rs:438-440`), and spec 1249 asserts exactly that in its own test list.
  - Spec 426/445 give kerry_park camera 0 as `f 240.1`; the fixture (`test-data/images/kerry_park/rig_config.json`) is `fx=129.150, fy=129.257, cx=cy=240.0`, so the row would read `129.1/129.3` under `focal_text`'s `fx≠fy` rule (`cameras.rs:132-149`). Spec 1156 states the true 129.15, contradicting 426/445, and the same `f 240.1` was copied into the code's doc examples (`cameras.rs:73, 132`). The derived example's `176.4°/197.2°/98.6°` likewise contradicts 1156–1170's `212.9°/150.5°/84.1°` for the same camera.
  - **Code doc wrong:** `DistortionSample` (`report.rs:137-139`) says "An overlay draws the arrow from `reference` to `pixel`", the exact convention spec ~700–712 and `field.rs:10-21, 155-168` rule out (`draw_arrow` tails at `arrow.pixel`).
  - The hover readout (~805–830) documents three states of the `distortion` line; `hover.rs:89` has a fourth, `outside the model's domain`, when `displacement_at` returns `None`.
  - Defaults correct: `enabled/axes/rings/distortion/grid_cols` = `true/true/false/true/16`; `SCALE_LADDER`; `MIN_ARROW_PX` 8.0, `HEAD_FLOOR` 3.0, `ARROW_FLOOR` 0.75; `FIELD_COLS` 16, `PROFILE_AZIMUTHS` 32, `BAND_VISIBLE_PX` 0.05; auto-expand at ≤ 4 cameras; tab title. The 11 scraped "defaults" at 895–917 are fixture example values, and wrong ones (above).
**Non-goals / deferrals checked:** 7, all still true, including the load-bearing one: the Python `_CAMERA_PARAM_NAMES` table (`camera/cameras.py:21`) is still hand-written and `parameter_names` has no PyO3 binding, so 206–209 and 1386–1392 remain accurate.
**Third copies:** `state.rs:135-153` reproduces spec 521–541's field docs verbatim (9 lines): shrink the **spec** to a defaults table + link. `report.rs:70-91` re-derives "Field of view is swept, not subtracted": genuine theory with no core spec of its own; keep it and consider promoting a `specs/core/camera/camera-report.md`. The largest: `image_detail/intrinsics/field.rs:4-70`, a 67-line module doc re-deriving the whole "trustworthy half of the grid" argument; shrink to invariant + link (it already drifted, citing "the spec asks for" three times at `:107-114`).
**Shape:** (1) The opener describes a change: "This spec adds it in three coupled pieces" (10), a `Before | After` rename table (57–63), "the rename lands in code and those three docs together, or not at all" (86). Propose: "The viewer's Camera Intrinsics surfaces answer what the camera is: a Camera Intrinsics group in the Scene Graph, an intrinsics overlay layer on the Image Detail panel, and a Camera Intrinsics dock panel, all reading one derived report from `camera::report`." (5) Residue throughout: "the rename is a bug fix" (47), "renamed from `CAMERA_LIST_HEIGHT` in this phase" (465), "The constraint phase 3 hit" (879), "phase 3 renames it" (1361), three "the first draft of this line…" asides (462, 1011, 1016), a `## Testing` written as a to-build list, and `## Open questions: None outstanding` (1432).
**Recommendation:** update spec (the kerry_park example numbers and parameter list are what a reader would copy; replace `retain_recon`; strip the phase/draft residue) and one code fix (`DistortionSample`'s arrow-direction doc).
**Unclear / incorrect / suspicious:** `derived.rs:18` says FIELD_COLS is "the same density the Image Detail overlay layer **will** default to", future tense for a shipped default. `cameras.rs:141` renders `1 image` singular, which the camera-row description (447–450) does not mention.

### specs/core/camera/epipolar-curves.md
**Summary:** Distortion-aware epipolar "lines": instead of `F p1`, back-project through camera 1's full model, bracket the depth interval whose reprojection stays inside camera 2's image (Phase 1, log-depth), then adaptively subdivide in `t = 1/λ` (Phase 2, worst-first) into a polyline. Covers the Rust API, the PyO3 binding, degeneracies, and the anchor-depth seeding in `sfm epipolar`.
**Implementing code:** `crates/sfmtool-core/src/camera/epipolar.rs` (`EpipolarCurveOptions`, `plot_epipolar_curve`, `plot_epipolar_curves_batch`, `find_inimage_seed`, `bisect_boundary`, `subdivide_worst_first`); `crates/sfmtool-py/src/analysis/epipolar.rs`; `src/sfmtool/visualization/_epipolar_display.py` (`_curve_anchor_depths:55`, `_draw_polyline:93`).
**Inconsistencies:** (code is right in each)
  - Spec 82–83 / 267–268: the return is an empty polyline or a sampled curve. The code can return a **1-vertex** polyline (`epipolar.rs:213` when the bracket collapses below `BRACKET_LOG_TOL`, `:217` when `p_out` fails to project). `_draw_polyline` guards `len < 2`, so harmless, but the documented shape is wrong.
  - The constants table (314–318) omits `MIN_BASELINE = 1e-9` (`:111`, the value behind the "near-zero baseline" degeneracy at 224) and `MIN_ANCHOR = 1e-12` (`:113`); `anchor_depth` is `.abs()`-ed before `ln` (`:197`), unstated.
  - Phase-1 step 3 (155–157): "accept `log_λ_out = log_anchor + BRACKET_MAX_STEPS · LOG_STEP`". Code returns the farthest in-image probe measured from `log_seed`, not `log_anchor` (`bisect_boundary:274, 284`); these differ whenever the seed search walked away from the anchor.
  - Spec 243–244 cross-references "Phase 2 step 2" for the failed-midpoint truncation; the behaviour is in step 3.
  - Defaults all correct: `curvature_tolerance` 0.5, `max_vertices` 256, `LOG_STEP = LN_2`, `BRACKET_MAX_STEPS` 24, `BRACKET_LOG_TOL` 1e-3. The Python signature at 261–268 matches the `#[pyo3(signature)]` byte for byte, keyword-only `*` included.
**Non-goals / deferrals checked:** 4, all hold (`compute_fundamental_matrix`/`compute_epipole*` still present and used by rectification; the in-frame-epipole half-line survives only on the `--undistort` branch; `--curve-tolerance` not exposed; polar/sweep/rectification still pinhole-assuming).
**Third copies:** `epipolar.rs:123-131` duplicates spec 60–68 verbatim; `:145-160` duplicates 79–92; `:220-225` re-derives the `t = 1/λ` argument (172–180); `:330-338` re-derives worst-first (181–187). **The spec should shrink**: its fenced Rust block (57–115) transcribes the doc comments; keep the signatures and the why, drop the copied prose. The code doc already links the spec (`:154`).
**Shape:** (1) First heading is `## The Problem` and the first sentence is the formula; it never says what the module produces. Propose: "Epipolar curve sampling produces, for a pixel in one image, the polyline in a second image along which its match must lie, traced through each camera's full projection model, so it is correct for the fisheye and wide-FOV cameras whose match locus is not a straight line." (5) Residue at four sites: 31 "New API goes in `…/epipolar.rs`"; 44–46 "the special case in the current display code disappears"; 214 "Why this is better everywhere"; 290–294 "This replaces the previous scene-median computation (`_median_scene_depth`)", a function that no longer exists anywhere. "Phase 1 / Phase 2" are genuine algorithm phases: the code uses the same names (`:199, :220`).
**Recommendation:** update spec: every divergence is the spec lagging correct code, plus the residue cleanup and the transcription trim.
**Unclear / incorrect / suspicious:** Spec 307 gives `anchor_depth`'s default as "observed track depth … else baseline length"; `_curve_anchor_depths:74-75` has a third fallback, substituting `1.0` for a degenerate baseline, relying on Rust returning empty anyway.

### specs/core/camera/sfmtool-pinhole-kernels.md
**Summary:** The computation behind `SFMTOOL_PINHOLE` (pinhole base + monotone cubic B-spline radial correction): forward projection and the fold gate, inverse radius recovery, the analytic ray Jacobian via the perspective family's `g(r²)` composition, monotonicity enforcement, the zero-spline bit-identity short-circuit, and classification flags. Defers the parameter list to `formats/sfmtool-camera-models.md` and the shared basis/monotonicity machinery to the fisheye sibling.
**Implementing code:** `crates/sfmtool-core/src/camera/distortion/kernels/sfmtool_pinhole.rs` (`PINHOLE_AXIS_EPS`, `distort_sfmtool_pinhole`, `undistort_sfmtool_pinhole`, `sfmtool_pinhole_radial_factor`, `sfmtool_pinhole_unfolded`); `kernels/sfmtool_fisheye.rs:83::recover_radial_bspline`; `distortion/bspline.rs`; `distortion.rs` dispatch and the `ray_to_pixel_with_jacobian` fast path (`:988-1010`); `intrinsics.rs` classification; `intrinsics/registry.rs:242`; `geometry/bundle_adjust.rs:306`.
**Inconsistencies:**
  - None found on any formula, constant, or code pointer after the aed8048 split. Verified: the forward map, `PINHOLE_AXIS_EPS = 1e-15`, `g = 1 + δ/ρ` and `dg/d(r²) = (ρδ' − δ)/(2ρ³)` (`:124`), the 2×2 Jacobian (`distortion.rs:298-300`), the on-axis `diag(fx/rz, −fy/rz)` form (`:1005-1008`); every inverse branch (`fisheye.rs:88-120`); every classification flag; all five rows of the zero-spline table including the fast-path predicate (`distortion.rs:995-999`); `bspline_step_admissible`.
  - Two omissions: (a) spec 175 defines `bspline_is_inactive` as identity coefficients or a non-finite `ρ_max`; `bspline_is_identity` (`bspline.rs:45`) also returns true for `len() < MIN_BSPLINE_COEFFS = 2`, which `intrinsics.rs:326` spells out. (b) The Inverse section never mentions the kernel's own `r_d < PINHOLE_AXIS_EPS` identity short-circuit (`:72-74`), which makes `recover_radial_bspline`'s `r_d ≤ 0` branch at spec 75 unreachable from this kernel.
**Non-goals / deferrals checked:** 3 (no Non-goals section; the "does not exist / not done here" claims): the slope half is not repeated in the fold gate (`:150`); folded splines are unreachable through a solve (`bundle_adjust.rs:1035`); the fisheye carries its own Jacobian kernel. All hold.
**Third copies:** `sfmtool_pinhole.rs:83-110` (28 lines) re-derives the on-axis-limit argument (the gauge, the `a/(2ρ)` divergence, the `O(ρ²)` companion factors) that spec 131–143 carries in full. `sfmtool_fisheye.rs:65-82` (18 lines) re-derives `recover_radial_bspline`'s branch structure, present in **both** sibling specs. The code docs should shrink to contract + the load-bearing warning ("the second return is not a bounded radial derivative on its own") + link.
**Shape:** The ~30 lines shared with the fisheye sibling are mostly legitimate parallel structure (code-pointer paragraphs, classification bullets, testing bullets). The exception is 85–95, the closed-form-tail + bracket-safeguarded-Newton derivation (≈ 11 lines matching fisheye 71–81): one shared function derived twice. It should live once in the fisheye spec and be linked, the pattern this spec already uses for basis evaluation (13–16) and monotonicity (153–158). Consistent with `camera-model-registry.md`: `SFMTOOL_PINHOLE` is registered in the `custom` block exactly as that spec describes. No residue, purpose-first opening.
**Recommendation:** update spec: fold 85–95 into a link to the fisheye sibling's inverse section, and shrink the two code doc comments; no code change.
**Unclear / incorrect / suspicious:** Spec 82 says the no-invertible-radius identity policy "is the policy `sfmtool_fisheye_to_ray` applies"; the fisheye applies its base model's inverse (`equidistant_to_ray`, `sfmtool_fisheye.rs:188`), not the identity. Same principle, different map.

---

## Code without specs

Inventory of every significant surface and the spec that covers it (one agent pass over
the command list, the crate list, every `sfmtool-core` module directory, every
`sfm-explorer` top-level module, every `src/sfmtool/` subpackage and flat module, and the
formats). **160 surfaces: 118 specced, 27 covered only indirectly, 15 unspecced.**

| surface | user-facing? | spec | verdict |
|---|---|---|---|
| 28 CLI commands (`align` … `xform`; `ws` → `ws-init-command.md`) | yes | `specs/cli/<category>/<name>-command.md` | specced |
| `sfm explorer` | yes | `specs/gui/*` spec the viewer, not the launcher | indirect, acceptable |
| 24 `xform` sub-operations | yes | own file (6) or the `xform-command.md` op table (18) | specced |
| `xform --filter-by-keypoint-uncertainty` | yes | named in passing in `patch-localizability.md` / `cluster-patches-command.md`; **absent from the op table** | indirect |
| `sift-format`, `matches-format`, `sfmr-format`, `camrig-format`, `sfmtool-archive-io` | yes | `specs/formats/*` (1:1) | specced |
| `sfmtool-core`, `sfm-explorer` | — | `specs/core/**`, `specs/gui/**` | specced |
| `sfmr-colmap` | via CLI | named in two specs; behaviour in the three colmap-interop CLI specs | indirect, acceptable |
| `sfmtool-py` | yes (public binding) | no crate-level spec; 24 specs link individual binding files | indirect |
| `sfmtool-core` modules: 7 `analysis/`, 7 `camera/`, 4 `features/`, 21 `geometry/`, 9 `patch/`, 4 `reconstruction/`, `spatial/keypoint_reach`, 5 `spherical/` | — | 1:1 with `specs/core/<group>/` | specced |
| `analysis/point_inspect` | yes (`sfm inspect`) | — | **unspecced** |
| `camera/viewport` | — | — | **unspecced** |
| `camera/rectification` | — | mentioned in `epipolar-curves.md` only | indirect |
| `features/feature_match` (descriptor distance, best-match, ratio test) | — | named in 4 specs, specified by none | indirect |
| `geometry/{convention, numeric, rigid_transform, rot_quaternion, se3_transform, transform, rotation}`, crate-level `numeric` | — | — | unspecced, acceptable (value types) except `convention` |
| `reconstruction/{edit, filter}` | — | behaviourally in `xform-command.md` | indirect, acceptable |
| `sfm-explorer`: 18 top-level modules + `align.rs`, `cli.rs` | yes | `specs/gui/*` | specced |
| `sfm-explorer/metrics/` | yes (panel numbers) | `batch-triangulation-api.md` names the numerics; no GUI spec owns the display | indirect |
| `sfm-explorer/{colormap,texture,test_support,shaders/}` | — | — | acceptable as unspecced |
| `src/sfmtool/{align,merge,feature_match,rig,sift,xform,camera}/` | yes | core/format/workspace specs | specced |
| `src/sfmtool/{analyze,motion,camrig,colmap,visualization}/` | yes | only via their CLI specs | indirect |
| `_global_sfm.py`, `_incremental_sfm.py` | yes | `solve-command.md` describes `-g`/`-i`, does not name the modules | indirect |
| `_densify.py` + `_patch_ncc.py` | yes | `densify-command.md` does not mention NCC/photoconsistency | indirect / **unspecced** |
| `_rectification.py`, `_undistort_images.py` | yes | `ray-grid-projection.md` names them; no spec owns the pipeline | indirect |
| `_compare_strips`, `_inspect_strips`, `_strip_montage`, `_compare_fragments`, `_feature_source`, and the 7 plumbing modules | — | `compare-`/`inspect-command.md` or nothing | specced / acceptable |
| `.sfmr`, `.sift`, `.matches`, `.camrig`, container, cluster-selection, camera-model catalog; `camera_config.json`, `rig_config.json`, workspace layout | yes | `specs/formats/*`, `specs/workspace/*` (1:1) | specced |

Entries worth arguing about:

### `crates/sfmtool-core/src/features/feature_match/`
**What it does:** Descriptor distance, best-match selection and the ratio test: the core of every non-cluster `sfm match` mode.
**Why it matters:** load-bearing, with real tunables (ratio, cross-check) that no spec states.
**Recommendation:** write `specs/core/features/descriptor-matching.md`; link it from `match-command.md`, which today documents the modes but not the matcher.

### `src/sfmtool/motion/` (detection rule)
**What it does:** Pose/flow discontinuity detection over sequences and reconstructions, with thresholds in `constants.py`.
**Why it matters:** user-facing; `motion-command.md` documents flags but its detection-rule prose is the spec section this run found most wrong (dead `Cov` signal, wrong window, min-of-two extrapolation).
**Recommendation:** either move the detection rule into a `specs/core/analysis/motion-discontinuity.md` written from the code, or rewrite the algorithm section of `motion-command.md`; not both.

### `src/sfmtool/_densify.py` + `_patch_ncc.py`
**What it does:** Densification with additional matches, scored by patch-strip weighted-NCC photoconsistency.
**Why it matters:** user-facing and tunable; `densify-command.md` carries the spec's own "experimental" caveat but never mentions the scoring.
**Recommendation:** write `specs/core/patch/weighted-ncc-photoconsistency.md` and link it from the command spec, or fold a scoring section into the command spec if the algorithm is to stay experimental.

### `xform --filter-by-keypoint-uncertainty`
**What it does:** Drops points whose patch localizability score exceeds a threshold.
**Why it matters:** a tunable filter a user can invoke that the `xform` op table does not list.
**Recommendation:** add a row plus threshold semantics to `xform-command.md`, cross-linking `patch-localizability.md`.

### `crates/sfmtool-core/src/analysis/point_inspect/`
**What it does:** Re-derives a point's rays from workspace `.sift` files and reports per-observation residual/angle; drives `sfm inspect`'s point view.
**Recommendation:** write `specs/core/analysis/point-inspect.md`, short; or a section in `inspect-command.md`.

### `crates/sfmtool-py`
**What it does:** The whole PyO3 public surface, registered by submodule.
**Recommendation:** a short `specs/core/python-bindings.md` stating module registration and naming rules; per-kernel signatures stay in their own specs. Previous audits called a binding spec unnecessary; the case for a *registration* note is that two of this run's findings (`extract_sift`'s hidden `max_described`, the `background_floor_clusters` re-export claim) are about how bindings surface, not what they compute.

### `crates/sfmtool-core/src/geometry/convention/`
**What it does:** COLMAP/OpenCV ↔ canonical `.sfmr` convention conversion: a correctness-critical sign convention.
**Recommendation:** add a note to `sfmr-file-format.md` § "Conversions happen at the I/O boundary" naming this module as the single conversion site.

Smaller notes, one line each: `camera/viewport` should be named in `viewport-navigation.md` as the state it manipulates; `sfm-explorer/metrics/` should be named in `point-track-detail.md` as the numbers' source (that spec links no code at all); `_global_sfm`/`_incremental_sfm` should be named in `solve-command.md` with the parameters passed through; `feature_match/_geometric_filter.py`'s model and thresholds belong in `match-command.md`.

Specs whose implementing code could not be found: `specs/gui/blender-viewport-navigation-implementation-overview.md` is a study of Blender's gesture handling, not a spec of any sfmtool surface; it reads as background research and belongs in `specs/drafts/` or `docs/`. Two standing specs carry no repo links at all and were matched by name only: `core/reconstruction/point-estimation.md` (→ `reconstruction/point_estimation/`) and `formats/cluster-selection.md` (→ `matches-format/src/select.rs`); both should gain interface-section links.

---

## Top priorities

Ranked by what a reader would do differently believing the spec. Across the 17 specs
read, the code was right in every divergence but three (two stale doc comments and the
one below), so all but one of these are spec fixes.

1. **One code defect, found through the spec: the optical-flow densification weight is
   dead.** `optical-flow.md` promises photometric-error-weighted averaging; `interp.rs:113`
   computes `1.0 / diff.max(1.0)` on images normalized to [0, 1], so the weight is
   exactly 1.0 for every pixel and the GPU shader matches. Either the weight is
   `1/max(1, 255·diff)` as in the DIS paper's intensity scale, or the spec stops claiming
   weighting. Decide with a flow-quality measurement, not by reading; the `bench-flow`
   task and `tests/matching/test_flow.py` are the harness. Same PR: `pyramid.rs:20`'s
   binomial-kernel comment describes a kernel the code does not use.

2. **Six documented calls that fail if copied.** `mcp-server.md:1585` `serve` takes an
   `EventLoopProxy` the code replaced with a wake closure, and `:1651`'s Cargo feature
   list would not build; `sift.md:535-536` `compute_descriptors` omits two arguments and
   `:662` `extract_sift` omits `max_described`; `track-cluster-matching.md:174, 854` names
   `pycolmap.geometric_verification`, which the codebase never calls;
   `bundle-adjustment.md:245-267` omits three binding parameters. One doc-only PR,
   verified by pasting each call into a test.

3. **`track-cluster-matching.md` contradicts itself about what `--cluster` writes, and
   300 of its 962 lines transcribe the code.** Five passages still describe the
   pre-239ee24 verify-and-write flow while three others describe the shipped
   clusters-only file; `--camera-model` is documented as allowed and is rejected; the
   `d = 28` prototype tables are unlabelled. Rewrite the Production Implementation
   section as a present-tense contract-plus-link and fix
   `matches-file-format.md:419-421` in the same PR. This is the largest spec↔code
   overlap in the corpus and the clearest case where the *spec* is the copy to shrink.

4. **Four GUI/flow specs whose structural claims describe code that is not there.**
   `gpu-optical-flow.md`: persistent `Mutex` buffer pools, shared-memory pyramid tiles, a
   five-buffer coefficient layout and a 66-line WGSL listing, none of which the shaders
   contain. `camera-intrinsics.md:872-917`: a worked kerry_park example with the wrong
   camera model's parameter list, a 35 mm row the code hides for fisheye, and an
   `f 240.1` that the fixture (129.15) and the spec's own line 1156 contradict, copied
   into `cameras.rs:73, 132`. `point-track-detail.md:323-363`: a state block four fields
   out of date, and no link to its code. `motion-command.md`: a `Cov` signal the spec
   presents as live that never fires under 48 frames, a window one edge wider than the
   code, and a "graph distance" field nothing computes. Plus `sift-to-patch-
   reconstruction.md:192`'s `patch_size` 5.0 (ships 11.0), the run's one wrong default.

5. **Thirty-four opening paragraphs fail the cold-reader test.** The table under
   Mechanical findings §4 proposes a first sentence for the 15 worst. It is the cheapest
   fix in the report and the one every future reader hits first; do it as one PR with
   no other content so it reviews in minutes.

Two patterns worth naming for the next writer rather than the next fixer. First, third
copies drift in both directions: where the code doc re-derives the spec (`extract_sfmtool.py`
50 lines, `intrinsics/field.rs` 67 lines, `estimate_intrinsics.rs`'s fleet counts
triplicated and already disagreeing 42/6 vs 40/4, `resect_translation.rs`'s chirality
argument four times, `sfmtool_pinhole.rs` 28 lines) the doc should shrink; where the spec
transcribes the code (`track-cluster-matching`, `epipolar-curves`'s fenced block,
`point-track-detail`'s state block) the spec should. The test is the same either way:
which copy would need editing after a behaviour-preserving refactor. Second, the
template's Rust-interface section earned its place this run: the two specs with no
signature anywhere (`rotation-locked-resection`, `optical-flow`) are where the
argument-order and defaults-attribution errors live, and `mcp-server`'s catalog table
and `panel-layout`'s two API blocks, the run's cleanest specs, show that a table or a
fenced block both satisfy it. No departure from `specs/TEMPLATE.md` that works better
than the template was found; no amendment proposed.
