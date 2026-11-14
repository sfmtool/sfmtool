# Spec audit — 2026-11-14

**Sample:** 20 of 228 specs read against their code (budget N = 10 drawn, plus
10 read regardless of the draw). Seed `27228`; the pool was the never-audited
specs (168 candidates). Every finding was checked against `main` at `7a7ee72`
(#1245). Corpus-wide mechanical checks below cover all 228 (201 standing specs
outside `drafts/` and `research/`).

Drawn at random (10): `core/bench/nearby-sources.md`,
`core/bench/track-at-pixel.md`, `core/bench/pairwise-flow-normal.md`,
`core/bench/pairwise-flow-normal-measurements.md`,
`core/camera/photograph-cache.md`, `cli/reconstruction/solve-command.md`,
`STANDARDS.md`, `core/camera/reading-and-writing-images.md`,
`core/camera/hdr-to-srgb.md`, `core/camera/image-warping.md`.

Included regardless of the draw:

- **Every format spec with a check-6 hit** (failure 6 only, plus three `.sfmr`
  field spot-checks against `sfmtool-sfmr-format`): `archive-io-crate`,
  `camrig-file-format`, `colmap-interop`, `kdf-file-format`,
  `matches-file-format`, `sfmr-file-format`, `sfmtool-camera-models`,
  `sift-file-format`, `sift-file-format-measurements`. These are reported in the
  format-independence table rather than as per-spec sections, because the deep
  read covered failure 6 only.
- **The draft paired with a sampled spec:** `drafts/track-at-pixel.md`, read
  against `core/bench/track-at-pixel.md`.

Not added on the code-changed rule: 141 commits landed since the last spec
audit, spread across `patch/`, `bench/`, `camera/`, `geometry/` and the
viewer's `mcp/`. Four of the ten drawn specs (`nearby-sources`,
`track-at-pixel`, `pairwise-flow-normal`, `image-warping`) cover code changed in
that window, so the draw already weighted toward it. `gui/mcp-server.md`
(7,000 lines, much changed code) is the clearest omission; give it a dedicated
run.

Non-goals and deferral entries checked in the sampled specs: **51**, 1 stale
(`image-warping.md` L950–952 calls a built feature "proposed"), 1 that describes
nothing (`image-warping.md`, "pose derivation stays in the Python wrapper").
Corpus-wide there are 84 standing specs with a `Non-goals` heading and 16 with
deferral language ("not yet implemented", "future work", "deferred", …).

`reports/2026-10-09-spec-audit.md` stays: it holds the only readings of the
specs this run skipped. Its open format findings are re-measured below.

## Mechanical findings

### 1. Documented defaults vs actual defaults

Script keyed on spec → owning command or struct → parameter, reading tables,
prose ("defaults to …") and fenced code; Click options followed through the
helpers a command imports (`_range_options.py`, `_workspace_init_options.py`,
`xform/_arg_parser.py`); Rust `impl Default`, `const DEFAULT: Self`, and
`#[pyo3(signature)]`.

| | count |
|---|---|
| specs scanned / with ≥1 extracted default | 218 / 92 |
| (spec, param) pairs extracted | 777 (147 descriptive, skipped) |
| matched to a code default | 572 |
| agree | 531 |
| no code found (mostly GUI/MCP prose rows, kd-forest symbols) | 89 |
| candidate mismatches | 41 |
| **verified mismatches** | **1** |

| spec:line | owner | param | spec says | code says | right side |
|---|---|---|---|---|---|
| `specs/cli/reconstruction/xform/xform-command.md:352` | `sfm xform --camera-model` (`xform/_arg_parser.py:438` → `RefitTarget`) | `coeffs` | `8` | `None` → keeps the camera's own count when it already has that spline model, else `DEFAULT_COEFF_COUNT = 8` (`crates/sfmtool-core/src/camera/refit_intrinsics.rs:48,88–92`) | **Code.** The same spec's prose (375–377) and `refit-camera-intrinsics.md:492` already say so; only the table cell is wrong. Behavioural: a reader expecting 8 coefficients on a 12-coefficient spline gets 12. |

The sampled specs agreed with the code on every default the deep readers
checked: 35 in `track-at-pixel`, 24 in `nearby-sources`, 24 in
`pairwise-flow-normal`, the photograph-cache budget, the HDR constants, and
every `sfm solve` option default. Discarded false-positive classes: wrong owner
from a row's own text; wrong owner from the workspace-wide fallback; options
defaulting to `None` that the command fills in; shorthand vs full Rust
expressions; rounding/formatting; rules rather than numbers; colliding constant
names (`IRLS_ROUNDS` is 5 in `translation_averaging.rs` and 3 in
`cell_registration.rs`). Not checked: the 147 descriptive cells and 89 pairs
with no code found.

### 2. Duplicate prose between spec and code

Long normalised lines (≥50 chars) shared between a standing spec and a code
comment: **80 file pairs, 115 lines in all**. This is low. The largest pairs:

| lines | spec | code |
|---:|---|---|
| 7 | `core/bench/editable-track.md` | `crates/sfmtool-core/src/bench/steps.rs` |
| 3 | `core/reconstruction/drop-repeat-observations.md` | `reconstruction/repeat_observations.rs` |
| 3 | `core/reconstruction/prune-covered-observations.md` | `reconstruction/prune_covered.rs` |
| 3 | `core/reconstruction/switch-camera-model.md` | `reconstruction/switch_camera_model.rs` |
| 3 | `core/geometry/translation-averaging.md` | `geometry/translation_averaging.rs` |
| 3 | `gui/architecture.md` | `sfm-explorer/tests/ui_basic.rs` |

The line-identity scan misses paraphrase. The deep reads found paraphrased
third copies that it did not: `sampler.rs`/`remap.rs` (image-warping, ~60
lines), `image.rs` `decode_jpeg` and `crates/sfmtool-core/Cargo.toml:22–41`
(image I/O), `hdr_to_srgb.rs:7–29`, `track_at_pixel.rs:186–193,396–402`,
`pairwise_flow/flow.rs:8–14`. Check 2 is a lower bound.

### 3. Spec shape (`specs/core/`, 101 specs)

- **No ` ```rust ` block: 23.** Eight are measurements files, which need none.
  The other 15 are a shortlist for failure 2, not findings:
  `analysis/{cluster-census, image-pair-graph, keypoint-reach, reconstruction-alignment, source-clusters}`,
  `camera/sfmtool-fisheye-kernels`, `features/{cluster-selection, flow-based-matching, gpu-optical-flow}`,
  `geometry/{reconstruction-growth, translation-averaging}`,
  `patch/{cluster-patches, cluster-warp-consistency, keypoint-localization-search-cache}`,
  `spherical/photometric-subsets-ransac`.
- **First ` ```rust ` block past halfway: 8** — `adjacency-surfel-normals`
  (71%), `observation-adjacency-graph` (69%), `sift` (54%),
  `affine-factorization` (58%), `relative-pose` (54%), `rotation-init` (80%),
  `rotation-locked-resection` (59%), `member-coherence-validation` (70%).
- **Work-order residue headings:** none in standing specs. The *Step N* / *Phase
  N* headings in `motion-command.md`, `select-by-distribution-command.md`,
  `photometric-subsets-ransac.md` and `epipolar-curves.md` are algorithm
  stages; the *Migration* headings in `workspace.md` and the three format specs
  are format-version migration. All acquitted. The imperative-language grep
  ("new module", "not yet implemented", "TODO") found nothing in standing specs.
  Residue does appear in prose; the deep reads found it in `image-warping.md`
  (eight places), `nearby-sources.md`, `track-at-pixel.md` and
  `reading-and-writing-images.md`.

### 4. Opening paragraphs

All 228 openings were extracted and read as a list (Overview/Purpose headings
skipped through). 93 are flagged mechanically, nearly all for a backticked
command or file extension (`` `sfm solve` ``, `` `.sfmr` ``), which is fine.
The ones that fail on reading:

| spec | opening (start) | failure |
|---|---|---|
| `core/bench/refine-normals.md` | "`bench::refine_normals` does to every finite point … what the bench does to one point when …" | bare symbol first; defined by reference to "the bench" |
| `core/bench/refit-points.md` | "`bench::refit_points` does to every point … what the bench does to one point when a person puts it on the bench, presses *Fit* …" | same |
| `core/features/gpu-optical-flow.md` | "The DIS optical flow in `sfmtool-core` ([optical-flow.md](…)) can run its stages …" | an unexpanded acronym and a link needed to parse it |
| `core/geometry/translation-averaging.md` | "With every frame's rotation known and the direction of the baseline … read from correspondences ([baseline-direction…" | starts with preconditions and a link before saying what it computes |
| `cli/image-processing/flow-command.md` | "The flow command is a diagnostic for [flow-based matching](…):" | the link carries the meaning |
| `core/geometry/reconstruction-growth.md` | "Register the un-posed images of a cluster-track set against a seeded reconstruction." | repo term ("cluster-track set") as common usage |
| `core/analysis/covered-by-finer.md` | "A feature detector answers at a scale, and a wide answer averages over whatever detail lies under it." | figurative (the detector "answers"); never says what the analysis is for |
| `core/bench/bench.md` | "Working out whether a photograph really shows the same speck of surface as another one is not a single decisive act." | says what the bench is not, not what it is for |
| `core/bench/*-measurements.md` (3) | "This file records the measurements behind [x.md], which specifies `fn_name`: …" | bare symbol and link in the first clause |

Land each replacement opening in its own PR (see failure 1).

### 5. Coverage both ways

- **Specs never cited from code** (by file name in any `.rs`/`.py`): **44 of
  201** standing specs. 24 are CLI command specs, which is expected (commands
  cite nothing). The rest:
  `core/analysis/{image-pair-graph, reconstruction-alignment}`,
  `core/camera/{projection-jacobian, reading-and-writing-images}`,
  `core/features/{descriptor-matching, flow-based-matching}`,
  `core/geometry/relative-pose`, `core/reconstruction/point-correspondence`,
  `formats/colmap-interop`, `gui/{cross-panel-hover, image-animation, user-experience, viewer-3d-bench-layer, windows-precision-touchpad}`,
  and six measurements files. `reading-and-writing-images.md` is the sharpest:
  `image.rs` carries 40 lines of its rationale and never links it.
- **Code surfaces citing no spec:** see **Code without specs** below.

### 6. Format-spec independence (all `specs/formats/`, every run)

| file | outbound links | identifier hits | findings | verdict |
|---|---:|---:|---:|---|
| archive-container.md | 0 | 0 | 0 | clean |
| archive-io-crate.md | 11 | 96 | 0 | crate spec; code names are its subject; no format rule missing from archive-container.md. Acquit |
| colmap-interop.md | 11 | 1 | 0 | crate spec; COLMAP defines the format (README says so). Acquit |
| kdf-file-format.md | 2 | 1 | 1 | links background; one implementation name |
| matches-file-format.md | 5 | 1 | 1 | links background; enumerations have legends; every version 1→10 has a migration statement |
| camrig-file-format.md | 2 | 1 | 4 | all in § "Rig type and attributes", introduced by the 2026-10-31 camrig fix |
| sfmr-file-format.md | 25 | 13 | 16 | mostly lists of this library's writers and commands inside field definitions |
| sfmtool-camera-models.md | 2 | 0 | 0 | background link; invariant defined at :110 |
| sift-file-format.md | 1 | 0 | 0 | background link |
| sift-file-format-measurements.md | 0 | 1 | 0 | measurements companion |

**Findings (format proper):**

- **One workspace rule, three statements with names** — `kdf-file-format.md:153`,
  `matches-file-format.md:1092–1095`, `sfmr-file-format.md:367` each name
  `sfmtool_archive_io::resolve_workspace_dir` (c). Fix: make sfmr §
  "Workspace Concept and Path Resolution" the one normative statement of the
  rule, link it from the other two, and move the function names to
  Implementations. For `.matches`: "a reader that finds no workspace still reads
  the file; a consumer that needs one reports its absence".
- **camrig § "Rig type and attributes":** `atlas_cols` and "atlas packing"
  (245, 253) are never defined in the format (a); 253–271 name
  `evenly_distributed_sphere_points`, `SphericalTileRig::from_camrig`,
  `measured_max_nn_angle()`, `resample_atlas` and "`sfmtool` estimates it from
  50 000 probe directions" (b+c) → "a reader that rebuilds a tile rig requires
  `centre`, both angles (finite) and `atlas_cols`"; "a writer may estimate the
  coverage angle by sampling, so it may fall short of the exact maximum"; drop
  "half-FOV is this angle times `overlap_factor`" (`overlap_factor` is not
  stored). 128–131 calls `rig_attributes` "Not load-bearing", yet a
  `spherical_tiles` reader requires four of its keys (b) — state those keys as a
  validity rule. 243–244 "(e.g. `sfm insv2rig`)" → name the camera, not the
  command (minor).
- **sfmr — operations where meaning belongs (b):** "The bundle adjustment solves
  them under a prior … and records a solved parameter in its camera's
  `per_image_adjustments`" (~858–862) → "a camera lists a parameter when its
  images carry their own values of it"; "sfmtool refuses a reconstruction with
  non-zero rates on any other model" (~918–922) → "a writer stores non-zero
  rates only on `SFMTOOL_PINHOLE`; a reader refuses …"; "solves it as
  `ln(1 + s)`" (~990), drop or move to the Why column.
- **sfmr — implementation names in field definitions (c):** the per-image
  adjustments consumer passage (~1019–1050: `read_sfmr`,
  `write_sfmr_with_options`, `image_intrinsics(i)`, `drop_image_adjustments`,
  the COLMAP/nerfstudio/undistort passages) → keep two consumer rules (apply the
  formulas; a converter to a format without per-image intrinsics writes one
  camera per adjusted image or refuses); the writer catalogue (~1755–1790);
  "names this column goes by outside the file" (~1812–1827); command lists
  (~2500–2540, still open from 2026-10-09); the content-hash writer list
  (~2674–2735). Move all to Implementations, keeping the writer-role rules.
- **sfmr — deferred meaning (a):** "the reference-view rule" (~2224, 2481,
  2526, 2638) defined only in `core/patch/reference-view.md` → "a writer picks
  one observation of the point by a rule of its choosing and records it in
  `reference_observations`"; **"the sampler rule's `a` … from which a reader
  works out each render's sampler"** (~2300–2303, 2467–2468, 2562) — a reader
  is told to derive a value from a rule defined only in
  `core/camera/image-warping.md`; restate the rule (zoom anisotropy against `a`
  selects the sampler) in the format. Open since 2026-10-09. Blur-matched ZNCC
  (~2374, 2587), borderline: one sentence on the blur kernel and its σ.
- **sfmr — versioning gaps (e):** the reprojection-error column changed from
  mean to RMS with no version (~1799–1806), and `observation_confidence`'s
  meaning "changed without a version bump, twice" (~2380–2398), so a reader
  cannot tell which value a file holds — tie each to a version range or bump.
  `normal_confidence` is "Optional (version 5+)" (1864–1883, 1690) but neither
  the v4→5 migration nor the optional-column list mentions it. ~2483 "required
  in a version 12 file" → "version 12 or later" (the reader gates on
  `version < 12`, `read.rs:990–1003`).
- **sfmr field spot-checks (3, all match the code):** per-image adjustments
  (names, order, finiteness, focal > −1, shapes; `types.rs:623–760`);
  `point_constraints` legend — matches, except the spec says a reader accepts a
  legend "naming any subset" and `parse_point_constraint_names` rejects an empty
  list (`types.rs:364`): say "any non-empty subset"; `reference_observations`
  (`types.rs:581–604`, `read.rs:990–1003`).

**2026-10-09 format findings:** fixed — sfmr v12 verifier step, `lineage`
wording, `without_any`/`stored_reference`; matches version-9 slip, verifier,
step 6, `piecewise`; camrig's three bullets (the angle fix introduced the new
names above). Still open — sfmr API names in § `reference_observations`, the
command lists, "the sampler rule"; mannered prose at sfmr :98, :1263, :1927,
:3526, :3575 and matches :18, :24, :535, :954, :1634, :1662, :1772.

## Sampled specs

### specs/core/bench/nearby-sources.md
**Summary:** The four sources a pixel query draws candidate tracks from — reconstructed points near the ray, feature-cluster tracks, epipolar-guided descriptor matches, and constellation seeds from the `.kdf` index — plus the ray triangulation they share; [nearby-tracks.md](../../../specs/core/bench/nearby-tracks.md) combines them. Close match: all 24 parameter defaults agree across the spec table, the Rust `Default` impls, the binding docstrings and the harness `DEFAULTS`; signatures, error variants, member policy, ratio/loose-match logic and the drop-the-worst loop match.
**Implementing code:** `crates/sfmtool-core/src/bench/nearby.rs` and `nearby/{candidate,points,clusters,guided,constellation,triangulate}.rs`; `track_at_pixel/neighbourhood.rs:30–139` (`ViewCamera`); `crates/sfmtool-py/src/bench/nearby.rs`.
**Inconsistencies:**
  - Binding docstring `crates/sfmtool-py/src/bench/nearby.rs:379` says "A point is kept when it is finite, …", wrong since #1245 (spec 195–200, `points.rs:507–509` keep bearings). Code (doc) wrong: delete "is finite,".
  - Spec 287–290 says a wide-angle lens "sees past 90 degrees" and an unseeable point has infinite error — true only in the points source (`points.rs:551–553`); triangulation returns `None` at `depth <= 0` regardless of lens (`triangulate.rs:683–686`), as spec 277 says. Spec wrong: scope the note.
  - Spec 89 lists `NearbySource` as Points | Clusters | Guided | Constellation; code also has `FarField` (`candidate.rs:108–112`, never on a candidate). Spec incomplete.
  - Spec 276–277 omits that `triangulate_sightings` also returns `None` when a sighting's image is not in `views` (`triangulate.rs:631,651`).
  - Comment `points.rs:562` says "the observation the index found" goes first; the code moves the first observation in the image (`:563`) — equal only because `ObservationIndex` indexes the first per image (`neighbourhood.rs:213–216`). Reword.
**Third copies:** `constellation.rs:311–313` (2 lines, grazing-surface rationale of spec 263–265) and `guided.rs:88–95` (`KeypointRays`, 8 lines, pose-independence rationale of spec 164–167) — shrink the code copies to contract + link. `candidate.rs:435–441` (`SAME_SIGHTING_PX`) carries threshold rationale found only in code — move it into nearby-tracks.md.
**Opening:** "A person looking at one photograph of a reconstruction can point at a pixel and ask what the other photographs agree is there, or near there." Passes. Paragraph 2 (15–20, "the first step of the anchor finder… and the third part of that finder moved into core") is muddled; proposed: "The sources are the first step of the track-at-pixel harness's anchor finder, where they were designed and measured; core combines them in nearby-tracks.md." One spec per PR.
**Shape / prose:** Interface easy to find, with rationale and a real call (117–184). Residue (failure 5) at 308–314: parity history ("The reference used to sort them with NumPy's default `argsort`… it now sorts stably… changed Kerry Park's by one query") → "Both implementations keep tied keypoints in row order (the reference sorts stably)." Failure 7 at 127: "so the first sighting is where both look" → "so both read the first sighting".
**Non-goals/deferrals checked:** 1 (no Non-goals section; "is a later step", line 23, is pipeline order and the step exists in `layers.rs`).
**Recommendation:** update code — fix the stale binding docstring; and under STANDARDS § "Caller input never panics", validate the forest's image count in `constellation_seeds` (below). Small spec edits for the three spec-side items.
**Unclear / incorrect / suspicious:** *Probable panic on caller input (unconfirmed by running):* `constellation_seeds` (`constellation.rs:327–333`) checks only `index.keypoints.len()`; the image ids `search_descriptors` reports come from the forest's origin table (`kdforest/constellation.rs:205`), so a forest built over more images than `views` indexes `cameras[i]` out of bounds; the binding does not check either. `constellation.rs:367,370` silently skip `seed_cluster_with`/`search_descriptors` errors, including real `KdfError` read failures. `points.rs:541` counts `observations().len()` against `min_views` while sightings drop keypoint-less observations. Core has no direct `nearby_points` test with a `w = 0` point (only via `find_tests.rs:430`).

### specs/core/bench/track-at-pixel.md
**Summary:** The operation behind the viewer's *Create Track Here* and the MCP `create_track_at_pixel`: from a pixel in one photograph, build a candidate track (members found by geometry, descriptors and constellation search, then fitted, cleaned and gated) ready to be judged on the bench. Accurate: all 35 Parameters-table defaults match the `Default` impls, all 15 `StageRecord` variants and the error variants match, and the finish order, gate order, anchoring rules and member constants match.
**Implementing code:** `crates/sfmtool-core/src/bench/track_at_pixel.rs` (+ `track_at_pixel/{members,finish,neighbourhood,tests}.rs`), `crates/sfmtool-py/src/bench/track_at_pixel.rs`, `crates/sfm-explorer/src/bench/track_at_pixel.rs`.
**Inconsistencies:**
  - Spec 84–91 (`TrackAtPixelOptions` block) omits the `pub thresholds: Thresholds` field (`track_at_pixel.rs:403`), though the Parameters table lists it. Code right.
  - Spec 454–459 says the Python binding overrides options "by name … the field as the Rust struct names it"; the binding (`sfmtool-py/src/bench/track_at_pixel.rs:139–178`) has no `transfer.`/`sweep.`/`constellation.` `min_radius_px`/`max_radius_px` keys and no `thresholds.*` keys — those raise `ValueError`. Discuss: add the keys (the harness purpose argues for it) or say the binding takes a subset.
  - Spec 139–140 ("the other variants refuse a query that names no place before any member runs") vs `track_at_pixel.rs:726–732`: `Cancelled` can occur between members and `InputMismatch` is a count mismatch. Reword.
  - Spec 217 ("a member whose shape is degenerate … carries nothing") vs `members.rs:192–199`: only the queried image's member is tested, and a degenerate one drops the whole cluster. Say "the queried member".
  - Minor omissions (code right): normal-prior step skipped for a bearing (`finish.rs:430`); cleaning stops below 2 `in` views (`finish.rs:397`); a member with no ZNCC counts as 1 (`members.rs:225`).
  - **The draft `specs/drafts/track-at-pixel.md` is stale against both:** L52 and open question L316 (query keypoint not held / how far it may move) are settled by anchoring with `max_query_offset_px = 2.0`; L287 gives the median gate as 0.8, code uses 0.7 on the blur-matched score; L219 "nine candidates", the harness has 11; L28 says `KdForest` or `LazyKdForest`, code takes `LazyKdForestU8` only; L60–62 refusal stages differ from `RefusalStage`; L292 puts the viewer gesture and MCP tool in the future tense; L129 links the deleted `patch-normal-refinement.md`.
**Third copies:** `FinishOptions::min_zncc_median` doc (`track_at_pixel.rs:186–193`) repeats the sweep result ("22 more of Kerry Park's correct tracks"); `TrackAtPixelOptions::thresholds` doc (`:396–402`) and the TODO at `:422` repeat the spec's reasoning for the two bars. Shrink the Rust docs to contract + link.
**Opening:** "A person looking at one photograph of a reconstruction can point at a spot on a surface and ask for the point there: …" Passes — reads cold and says what it is for. Only repo term is "bench"; optional gloss "…ready to be judged and committed as a new point".
**Shape / prose:** Interface first, with "Why these inputs"/"Why the report is…" rationale and a real call — good. Residue: L37 "the Rust operation is a port of it"; L364–366 dated history ("since 2026-10-10; from 2026-10-09 it read the plain score…") — move plain-score history to the harness README. Failure 7 at L235, "where the ray grazes it … runs as far as the centroid" → "where the ray meets the plane at a cosine under 0.15, the hypothesis is the point on the ray at the centroid's distance from the camera centre."
**Non-goals/deferrals checked:** 9, none stale (7 Non-goals bullets; the draft link — live, but the draft is stale; the TODO at `:422` is an undocumented deferral).
**Recommendation:** update code — STANDARDS § "Long-running functions take a Progress": `build_track_at_pixel` takes `progress` but calls its searches, `fit` and `evaluate` with `&Progress::none()` (`finish.rs:164,177,276,463`; `members.rs:851,881,933`), so cancellation is checked only between members; pass the progress (or a phase of it) down, and give the Python binding a way to pass one. Update spec for the `thresholds` field and the binding claim. Convert or retire the draft (keep the live open questions; move harness/Evaluation text to `scripts/track_at_pixel/README.md`).
**Unclear / incorrect / suspicious:** `ConstellationParams` (`features/kdforest/constellation.rs:73`) is a settings struct named `*Params`, against the `*Options` rule (outside this spec). A short `views` gives `InputMismatch` before `NoSuchImage` — fine, unspecified.

### specs/core/bench/pairwise-flow-normal.md
**Summary:** A bench step that estimates which way a point's patch faces by rendering its views on candidate planes and fitting the plane to the optical flow between pairs of views, over several rounds of shrinking radius. Every one of the 24 option defaults matches `Default` (`pairwise_flow.rs:166–196`); fields, outcomes, decision rules, progress phases, cancellation, pair shuffle, noise simulation and Display strings match.
**Implementing code:** `crates/sfmtool-core/src/bench/normal/pairwise_flow.rs` + `pairwise_flow/{crops,model,flow,texture,fit,support,rounds,report}.rs`; `bench/tests/pairwise_flow_normal.rs`; `examples/pairwise_flow_measure.rs`; `sfmtool-py/src/bench/normal.rs:679–735`.
**Inconsistencies:**
  - Spec 203–216 (`check` list) omits the `tukey_c` refusal (`pairwise_flow.rs:289`) and the separate `max_grid_tilt_deg` name (`:271`); the read-back fit refusal is named `"resolution"` (`:313–314`), not an options field. Code right.
  - Spec 130–132: `samples_per_patch`/`crop_half_width` values for "no view saw the plane" unstated; code gives NaN and 0 (`report.rs:46–51`).
  - Spec 87, 186 "exactly two observations"; code counts `in` observations, first per image (`rounds.rs:64–66`), as spec 738 says. Make 87/186 (and `report.rs:91`) say "two observations the step renders".
  - Spec 816 "a fixed SplitMix64 sequence": `texture.rs:350–358` writes its own `splitmix` instead of `numeric::splitmix64` (`numeric.rs:233`), which the pair shuffle uses. Code duplication.
  - A two-view `Undetermined` prints "no radius with two determined pairs" (`report.rs:268–271`), not spec 191's reason (no determined fit at `single_pair_radius`). Fix the code's sentence.
  - Minor: `PairwiseFlowRounds::last`, `PairwiseFlowRound::{radius,rendered}` (`report.rs:212–232`) unmentioned.
**Third copies:** `flow.rs:8–14` repeats the 0.093 vs 0.040 px measurement, and its "on crops this small" is stale (crops reach 512 samples) — shrink to contract + link. Inside the spec, Theory repeats measurement figures in nine places (352–361, 371–375, 500–503, 573–581, 653–658, 735–736, 741–743, 760–764, 823–828): a second copy of the measurements file; keep one clause + § link each. Defaults appear four times (struct comments, Parameters table, field docs, `Default`), all agree today.
**Opening:** "A patch in a patch-based reconstruction is a small oriented square standing in the world at a 3D point, and its normal says which way the square faces." Passes.
**Shape / prose:** API first with rationale and example. Failure 4: "The pieces" (786–791) transcribes the module list already in the module doc — delete; Testing (855–913) transcribes test names — cut to what each establishes. "Cost" note belongs in the measurements file. Failure 7 at 271 "the rounds are where the step moves" → "the plane changes from round to round, so a person checking a point needs…".
**Non-goals/deferrals checked:** 7 (3 Non-goals, 4 deferrals), none stale. `DisFlowParams` "as backlog" (259–261) is still `Params`.
**Recommendation:** update code — `texture.rs` to call `numeric::splitmix64`; fix the two-view Undetermined sentence; STANDARDS § "A settings struct is named *Options" (and § "A convention has no documented exceptions"): rename `DisFlowParams` → `DisFlowOptions`. Then update spec (check list, NaN/0, "observations the step renders", drop duplicated numbers and "The pieces").
**Unclear / incorrect / suspicious:** Parameters 850/852 say `max_sigma_deg` and `max_mean_chi2` were "the best of the settings tried", but the measurements show only σ ≤ 10° run; cite runs or say "not measured against other values". The claim (54–55, 281) that the glossary keeps this step's outcome names apart is not yet true (GLOSSARY.md:419–420 covers only the other steps).

### specs/core/bench/pairwise-flow-normal-measurements.md
**Summary:** The measurements behind the design spec: question, setup and result per section. Its figures agree with every citation in the design spec, the set counts add up, and the defaults it argues for (`flow_floor_px` 0.03, `max_mean_chi2` 2, `rounds` 3, radii, 48 px, 512 samples, 10 pairs, 0.999 quantile) match the code.
**Implementing code:** `examples/pairwise_flow_measure.rs` (`--chi2` → `max_mean_chi2`); the other runs used prototypes not checked in.
**Inconsistencies:**
  - 110–112: automatic coarsest scale "0 or 1" on these crops vs design spec 497–498 "is 1" on 96-sample crops. Make them agree.
  - No support for the design spec's "best of the settings tried" for `max_sigma_deg` — only σ ≤ 10° appears (207, 222, 235, 271, 304).
**Third copies:** none in code beyond `flow.rs:8–14`; the design spec carries ~nine of its results.
**Opening:** "This file records the measurements behind pairwise-flow-normal.md, which specifies `pairwise_flow_normal`: the bench step that …" Borderline: bare symbol and repo words. Proposed: "These are the measurements behind the pairwise-flow normal, a step that estimates which way the small square patch at a reconstructed 3D point faces from the optical flow between pairs of its photographs; they justify that step's design and default settings." One spec per PR.
**Shape / prose:** Fits its kind. § "The Rust step on real points (2026-11-13)" is dated, § "Real points with the revised design" is not — date it.
**Non-goals/deferrals checked:** 2 ("not yet isolated", 259; "do not yet tell these points from good ones", 282), both live.
**Recommendation:** update spec (date the section; record the σ bars tried or drop the claim; reconcile the coarsest-scale claim).
**Unclear / incorrect / suspicious:** seoul_bull is left out of the Rust run with no reason. Line 162 compares support rules under a σ with no texture term, so the χ² ≤ 2 bar "tuned on these references" (174) was tuned under a different σ from the step's; say the bar was re-checked under the revised σ only at 1 and 2.

### specs/core/camera/photograph-cache.md
**Summary:** The process-wide, byte-budgeted LRU cache of decoded photograph pyramids that the viewer's panels and background jobs (and `sfm web-export`, at budget 0) read photographs through. The API block, types, fields, defaults (¼ of physical memory clamped to 1–16 GiB, 4 GiB fallback, the budget environment variable in 10⁶-byte MB), eviction, staleness stamps, counters and concurrency all match the code.
**Implementing code:** `crates/sfmtool-core/src/camera/photograph_cache.rs` (+ `photograph_cache/tests.rs`); `machine.rs:183–209`; `patch/display_bitmaps.rs:29` (`DISPLAY_PYRAMID_LEVELS`); `patch/view_selection.rs:648,669`; `web_export.rs:289`; viewer `state.rs`, `state/photograph_requests.rs`, `state/edits.rs:715`. No Python binding.
**Inconsistencies:**
  - Spec 219–247 ("In the viewer") lists Image Detail, Track View, background steps and open as the readers; it omits Track Table (Crop column via `display_photograph`/`peek_state`, `track_table_panel.rs:2016,2034,2064`; reference-pick job calls `get`, `track_table_panel/picks.rs:195`) and the image menu's `peek` (`image_menu.rs:132,174`). Code right; add them.
  - Spec 82–83, 135–138 vs `photograph_cache.rs:329–331`: at budget 0, `insert` is a no-op, though the `insert` rationale implies a following `get` returns the inserted pixels. Code right; add half a sentence.
  - Spec 198–200 restates which OS calls read physical memory; the code reaches them through `machine::physical_memory_bytes`. True, but should point at `machine.rs` rather than restate it. Minor.
**Third copies:** none material. Module/type docs (`:4–19`, `:146–159`) are contract + link. The `decode` doc (`:388–394`) repeats the rayon-deadlock reasoning of spec 267–281, but that is an invariant its maintainer needs — acceptable.
**Opening:** "Many of the viewer's operations read the reconstruction's photographs as pixels: the Image Detail panel draws one, Track View cuts patch tiles out of several, and the bench's searches and fits measure how well a patch matches in each image." Weak: repo vocabulary ("the bench", "Track View"), the cache itself is not named until sentence 4, and the core-library and web-export use is absent. Proposed: "The photograph cache holds decoded photographs as image pyramids, keyed by file path, shared across threads and bounded by a byte budget, so that the viewer's panels and background jobs decode each photograph once rather than once per use." Land in its own PR, checked by someone who knows the cache.
**Shape / prose:** Works well — interface with rationale and a real call first; "Why it exists" with measured numbers is a good departure. Mild failure 4 at 273–280: jpeg-decoder per-component threads and the exr private pool are dependency internals that change on upgrade; keep only the invariant (nothing the decode starts takes work from the global rayon pool). No residue, no mannered prose. Both open questions (337–339) are still open.
**Non-goals/deferrals checked:** 4 of 4 hold (cluster patches reads its own photographs; thumbnails/background from elsewhere; no on-disk cache; budget only from the env var).
**Recommendation:** update code — STANDARDS § "Caller input never panics": `PhotographCache::new` asserts `levels >= 1` (`photograph_cache.rs:193`, documented `# Panics`, spec l.56) on public `sfmtool-core` API; clamp or return an error, then make the spec follow. Separately update spec for the two omissions and trim 273–280.
**Unclear / incorrect / suspicious:** `insert` says the pyramid "should have `levels` levels" (`:326–327`) but never checks; a wrong depth would be served silently (test-only use today). `camera/mod.rs` is still a `mod.rs` (module-file backlog).

### specs/cli/reconstruction/solve-command.md
**Summary:** `sfm solve` runs COLMAP's incremental or GLOMAP's global mapper (via pycolmap) on images or a `.matches` file and writes `.sfmr` models. Option names, short flags (`-i/-g/-s/-o/-r`) and the stated defaults (`--refine-rig`/`--detect-infinity` on, `--flow-preset default`, `--flow-skip 5`) match the Click decorators; the `-i`/`-g` exclusivity, `--seq-overlap` exclusions, clusters-file refusal, save order and the "No 3D points" error all match. Drift is in what the spec omits or states loosely. (`--help` not run; decorators read.)
**Implementing code:** `src/sfmtool/_commands/solve.py`; `_solve_driver.py`; `_incremental_sfm.py` (`_save_reconstructions`, `_resolve_output_path`); `_global_sfm.py`; `_solve_setup.py`.
**Inconsistencies:**
  - Spec 64, 100–101 vs `solve.py:225–226`: `--flow-match` with a `.matches` input is a UsageError; the spec never mentions the refusal. Code right; add it beside the `--seq-overlap` restrictions (line 171).
  - Spec 60 (`--colmap-dir` default "auto") vs `solve.py:268,356`, `_solve_driver.py:101`: without the option the database goes in a temporary directory deleted at exit; with `--seq-overlap` each window uses `solve_NNN/`. Code right; say both.
  - Spec 61 (`--sfmr-dir` default `sfmr/`) vs `_incremental_sfm.py:261`, help at `solve.py:68`: the default is `<workspace>/sfmr`, not `sfmr/` under the current directory. Fix spec.
  - Spec 59 ("timestamped in `sfmr/`") vs spec 193 and `_get_next_sfm_filename`: the name is `<YYYYMMDD>-<NN>-solve-<descriptor>`, a date and counter, not a timestamp.
  - Spec 192 (`{stem}-{N}{suffix}`) vs `_incremental_sfm.py:259–260`: N is the size rank starting at 1 for the second-largest model, and `.sfmr` is appended when `--output` has no suffix; neither is stated.
  - Spec 185–186 vs `_global_sfm.py:68`: GLOMAP returning no models raises "Global mapping failed or produced no reconstructions.", not the quoted error. Minor.
**Third copies:** `solve.py:228–231` re-derives spec 142–148 (mapper reads two-view geometries from the DB) — shrink to one line + link. `_save_reconstructions` docstring (`_incremental_sfm.py:116–123`) restates the rationale of spec 182–198 — shrink to contract (what is skipped, save order, return) + link. The command docstring's examples duplicating spec 335–358 are ordinary help text: acceptable.
**Opening:** "Runs Structure from Motion to produce a 3D reconstruction (`.sfmr` file) from images or pre-computed matches." Passes. Optional: "`sfm solve` runs Structure from Motion on images or a `.matches` file and writes the reconstruction as `.sfmr` files." (one spec per PR if changed).
**Shape / prose:** Failure 7 at 184, "a single junk fragment must never discard the good models that share the output loop" → "a model with no points does not stop the other models from being saved." Failure 7 at 187, "surfaced loudly rather than swallowed silently" → "the command prints a warning for each skipped model." Mild failure 4 at 307 (names `_setup_db_with_camrigs`). Line 197 is unwrapped.
**Non-goals/deferrals checked:** 2 (no Non-goals section), both still true: windows are not aligned or merged (155); `--flow-match` is not applied to windows (174; `_solve_driver.py:100–129`).
**Recommendation:** update spec — the code is right in all six cases; plus shrink the two comments to contract-plus-link.
**Unclear / incorrect / suspicious:** `--flow-match` with `--seq-overlap` is silently ignored while a `.matches` input is refused — inconsistent; `--flow-preset`/`--flow-skip` are silently ignored without `--flow-match` (both candidates for a UsageError, STANDARDS-style "refuse rather than ignore"; discuss). `-h` alias undocumented.

### specs/STANDARDS.md
**Summary:** The project's quality rules (interface contracts, naming and module layout, build gates), each as rule / reason / what to look for / how to confirm / out of scope, which `audit-hygiene` and this skill read. Audited as a set of factual claims about the tree. Every `STANDARDS.md §`/`#anchor` reference in the repo resolves (AGENTS.md, the four skills, GLOSSARY.md:399, ten core/cli specs, the PR template, `sfm-explorer/src/lib.rs:15`, `tests/test_module_layout.py:13`), and every named example exists.
**Implementing code:** `crates/sfmtool-core/src/numeric.rs` (median/quantile family, `:107–221`) and `numeric/tests.rs` (`the_workspace_*` tests and allowlists); `crates/sfmtool-core/tests/license_headers.rs`; `tests/test_module_layout.py`; `tests/test_lazy_loading.py`; `#![warn(unreachable_pub)]` in `sfm-explorer/src/lib.rs:20`; `ci.yml` clippy step and `msrv` job; `[workspace.lints.rustdoc]`; `--durations=15` in `pyproject.toml`.
**Inconsistencies:**
  - L459: `import sfmtool` "binds the extension's names and nothing else"; `src/sfmtool/__init__.py:10–13,53` binds only `THUMBNAIL_SIZE`, `ProgressCounter` and `build_profile`, and binding modules load lazily through `_LAZY_SUBPACKAGES` (as L427–429 itself says). Reword L459.
  - L414: "Each part of the package has one of these shapes" — `src/sfmtool/web_export/` is plain-named and defines its code in `__init__.py`, fitting none of the four. **Needs decision:** change `web_export/` or add a shape.
  - L368–369: `lib.rs` re-exports the catalog "from mcp/call_check.rs"; the path is `call_check` → `mcp/mod.rs:74` → `lib.rs:79`. Trivial.
  - L657–658: coverage "measured at nearly three times the time of a plain run" holds for pytest (851.5 s vs 302.7 s); for sfmtool-core it is ~8× (76.7 s vs 9.6 s, `scripts/coverage.sh:63–66`). Qualify.
  - L505: `git log --diff-filter=A -- '*mod.rs'` is wrong in a shallow clone (the grafted root shows as adding all 44 `mod.rs`). Say "in a full clone".
**Third copies:** AGENTS.md:84 "Every `.rs` and `.py` file" is broader than STANDARDS L537 (and `license_headers.rs:95–98`); the test's own failure message (`:128`) repeats the broad wording. AGENTS.md:81–82 "never `foo/mod.rs`" drops L487–489's convert-when-restructuring clause. `skills/audit-hygiene/SKILL.md:567–573` asks for three tallies "with the commands STANDARDS.md gives"; STANDARDS gives two. AGENTS.md is the copy to shrink — to one-liners linking the headings.
**Opening:** "The rules every library function, binding, Python module, source file, test and spec in sfmtool is held to, whatever module it is in: …" Passes, but has no main verb. Proposed: "This file states the quality rules that every library function, binding, Python module, source file, test and spec in sfmtool must meet, whatever module it is in." Move the how-audits-use-it text (8–14) to a second paragraph.
**Shape / prose:** Present tense; the rule/reason/look-for/confirm/out-of-scope format works and is applied evenly — a candidate amendment to `specs/TEMPLATE.md` for any rules-type document. Borderline residue at 208–217 ("Six median copies used three NaN policies…") and 310–317 — shrink each to one sentence of reason. Failure 7 at L29, "the next audit acquits the code against it" → "the next audit treats the code as matching its spec".
**Compliance spot-checks:** `*_with_progress`/`_silent` fns 0; `Option<&Progress>` 0; `select_nth_unstable` outside numeric.rs 9 (allowlisted); `mod.rs` 44, `foo.rs`+`foo/` 249; `*Params`/`*Config` settings structs 27 excl. `CameraParams` (matches GLOSSARY.md:351); bare `pub` fields in sfm-explorer 0; `# Panics` sections in library crates 83 (backlog per L133–136). **`.kdf` breaks § one I/O shape per file format (L343–347):** `write_kdf`/`read_kdf`/`verify_kdf`/`kdf_file_summary` are in `sfmtool.spatial` (`crates/sfmtool-py/src/spatial/kdf.rs:1003–1006`), not `sfmtool.fileio`; `kdf_file_summary` stands where `read_kdf_metadata` belongs; `verify_kdf` calls `sfmtool_core::features::kdforest::verify_kdf` (`kdf.rs:951`) rather than the format crate. The other four formats have the full set in `fileio`.
**Recommendation:** update spec for the four wording slips and the shallow-clone command; update code for `.kdf` (move to `sfmtool.fileio` with the standard four names, verified in `sfmtool-kdf-format`); discuss `web_export/`. Opening rewrite in its own PR.
**Unclear / incorrect / suspicious:** 17 "the exception" phrases remain in specs/ and AGENTS.md outside STANDARDS and drafts — not triaged against § "A convention has no documented exceptions".

### specs/core/camera/reading-and-writing-images.md
**Summary:** The one image reader and one writer shared by Rust and Python (RGB/RGBA layout, alpha, ignored EXIF orientation, 8-bit writes, format by extension, JPEG quality 95 at 4:2:0 with Huffman optimization off, PNG settings, atomic temp-file write). Signatures, behaviour and caller lists match.
**Implementing code:** `crates/sfmtool-core/src/camera/image.rs:16–676` (`read_rgb`/`read_rgba`, `decode`, `write_*`, `write_via_temporary`, `image_has_alpha`, `image_dimensions`, `decode_jpeg`), `camera/image/tests.rs`; bindings `crates/sfmtool-py/src/fileio/image.rs`; Python callers `_workspace_image.py`, `_image_load.py`, `_undistort_images.py`, `_to_nerfstudio.py`; viewer `upload/bg_image.rs:125`, `cluster_patches.rs:761`.
**Inconsistencies:**
  - Spec 81–87 calls a JPEG side over 65535 "the encoder's refusal" (reads as `ImageError::Encoding`); code returns `Parameter` (`image.rs:186–194`, and its own doc at 130–133). Code right.
  - Spec 84–87 lists four writer error variants; the binding also maps `Limits` to `ValueError` (`py image.rs:307`).
  - Spec 67–69 pairs `image_dimensions` with `image_has_alpha`, but the binding maps every `image_dimensions` error to a bare `OSError` without the path, and a missing file is not `FileNotFoundError` (`py image.rs:29–31`), whereas `image_has_alpha` uses `read_error`. Code should change.
  - Spec 122 "the workspace `image` dependency" — it is a per-crate `image = "0.25"` in four manifests. Nit.
  - Spec 286: web-export atlas pages written "at `--jpeg-quality`"; `atlas.rs:138` silently clamps to 1–100. Minor.
  - Spec 15–17, 289–290 (OpenCV never reads or writes an image file; tests use OpenCV only for images no sfmtool writer produces): ~13 test reads use `cv2.imread` (e.g. `tests/matching/test_flow.py:219`) and `tests/test_pick_images_command.py:66`, `tests/patch/test_embed_patches_command.py:425` write ordinary 8-bit files with `cv2.imwrite`. Narrow the spec to library code or move those tests to the bindings.
**Third copies:** (a) `decode_jpeg` doc (`image.rs:413–449`) and `decode` doc (82–93) re-derive spec 121–172 (zune-jpeg misreads, libjpeg grey fill, re-decode of cut files) and never link the spec. (b) `crates/sfmtool-core/Cargo.toml:22–41` comments carry rationale and measurements the spec lacks (image crate's encoder 4:4:4-only, 14% larger, ¼ OpenCV's speed; rayon feature 3% faster; why no rayon). Shrink both to contract + link; move (b)'s measurements into the spec. `specs/python-bindings.md:297` omits `write_image_from_hdr` and the three HDR constants from the `fileio/image.rs` row.
**Opening:** "sfmtool decodes an image file into pixels with one reader and encodes pixels into an image file with one writer, in Rust and in Python alike, so a value computed from the pixels is the same whichever side computed it, and no image array is in BGR order on its way to or from a file." Passes.
**Shape / prose:** Interface with reasons and real calls — good. "Formats, reading" (121–172, ~50 lines) drifts into code detail (`EndTrackingReader` re-decode logic); give it its own subsection. Residue at 152–153 "as every JPEG was before" → present tense. No mannered prose.
**Non-goals/deferrals checked:** 9; all hold (orientation ignored, 8-bit writes, 12-bit/lossless JPEG refused, 1 GiB file / 512 MiB decode limits, RGBA→JPEG refused, Huffman off). Live deferral: switch back to zune-jpeg once it passes `read_decodes_a_jpeg_with_one_scan_per_component` (`tests.rs:537`). No test for refusing EXR/HDR writes or a 16384-px WebP read.
**Recommendation:** update code — `image_dimensions` binding should use `read_error`; STANDARDS § "Caller input never panics": `ImageU8::new` (`image.rs:39`, `# Panics` on data-length mismatch) and `get_pixel` are public and panic (backlog; add a `Result` constructor). Update spec for the error mapping, `Limits`, dependency wording. Shrink the two third copies.
**Unclear / incorrect / suspicious:** The EXR/HDR `ValueError` claim (178–180) relies on the image crate rejecting Rgb8 for those encoders, untested. "One writer" has two Rust exceptions (web-export atlas, MCP screenshots), admitted only at 283–287.

### specs/core/camera/hdr-to-srgb.md
**Summary:** The fixed recipe (versioned) that converts a phone's HLG or PQ BT.2020 10-bit frame to an sRGB image for extracted video frames. Constants, the eight recipe steps, the row-sum-preserving matrix, the EETF, threshold-search rounding, errors, parallel rows and the binding's timing dict all match.
**Implementing code:** `crates/sfmtool-core/src/camera/image/hdr_to_srgb.rs:37–384`, tests `hdr_to_srgb/tests.rs:52–250` (all eight areas the spec lists); binding `crates/sfmtool-py/src/fileio/image.rs:193–258`; caller `src/sfmtool/_video_frames.py:990–1040`.
**Inconsistencies:**
  - Spec 36–40, 57–60 describe a fixed "JPEG at quality 95"; the binding (`py image.rs:194`) takes keyword-only `jpeg_quality=95` and writes whatever format the path names. Code right; add it to the interface.
  - Spec 82 says every step is "in `f32`"; the PQ/EETF constants are computed once in `f64` and stored as `f32` (`hdr_to_srgb.rs:203–215`). Add a clause.
**Third copies:** module doc (`hdr_to_srgb.rs:7–29`) summarises the recipe and re-derives "Why single precision" (spec 177–188), and already disagrees ("0 to 10,000 cd/m²" vs the spec's "times at most 2^16"). Shrink to contract + link. `specs/workspace/video-frames.md:56` restates recipe v1 — acceptable for a record spec, needs updating at v2.
**Opening:** "Phones record HDR video: 10-bit pixels whose values are not the brightness an ordinary (SDR) image stores, but a signal coded with an HDR transfer function, …" Passes.
**Shape / prose:** Interface first with reasons and calls; then recipe, "Why this recipe", "Determinism and precision", Testing. Two practices worth adding to `specs/TEMPLATE.md`: reference values given with their pre-rounding values, and a recipe version that every output-changing edit must bump. No residue, no mannered prose.
**Non-goals/deferrals checked:** 3, all hold (no adaptive analysis; Dolby Vision RPU ignored; profile 5 not converted correctly).
**Recommendation:** update spec (`jpeg_quality`/output format; f64-computed constants); shrink the module doc. "Caller input never panics": compliant.
**Unclear / incorrect / suspicious:** a Dolby Vision profile 5 file is silently mis-converted — `_video_frames.py` should refuse or warn (extract-frames-command.md). Transfer names are case-sensitive ("HLG"/"PQ"), implied but unstated.

### specs/core/camera/image-warping.md
**Summary:** Whole-image resampling from one camera to another through a precomputed per-pixel `WarpMap`, with bilinear, mip and anisotropic samplers and the per-view rule that picks a sampler for patch tiles. Core claims hold: `WarpMap` layout and NaN convention, SVD `u` direction, equirect `x_period` wrap, per-pass parallel thresholds, the sampler rule (√2 and `a = 1.5`), `render_tile` dispatch, AVX2 bit-identical to scalar. Drift is in pyramid details, error cases, Python examples, the test list and one stale deferral.
**Implementing code:** `crates/sfmtool-core/src/camera/{warp_map.rs, remap.rs, remap/aniso_avx2.rs, sampler.rs, image.rs (pyramid 679–760), distortion/projection.rs, distortion/ray_grid.rs}`; `crates/sfmtool-py/src/flow/warp.rs`.
**Inconsistencies:**
  - L629: pyramid has `floor(log2(min(w,h)))` levels; `ImageU8Pyramid::full_levels` (`image.rs:691–694`) gives one more, minimum 2. Code right.
  - L587, L626: "Gaussian pyramid … box-filter or Gaussian"; the code is a 2×2 box filter only (`image.rs:300–304`), though the doc at `image.rs:678` and the bindings also say "Gaussian". Fix spec and docs.
  - L179: `remap_aniso` "returns an error (or panics)" — it panics (`remap.rs:446`; test `test_remap_aniso_panics_without_svd`).
  - L366–368: perspective `ray_to_pixel` is `None` only for θ≥π/2; code also rejects folded SimpleRadial/Radial (`distortion/tests.rs:1281,1331`) and has a rolling-shutter branch (`projection.rs:139`). Spec stale.
  - L272 `project(rx/−rz, −ry/−rz)` is right; the code doc at `projection.rs:131` writes `ry/(−rz)`, wrong sign under S=diag(1,−1,−1). Fix code doc.
  - `compute_jacobians` doc (`warp_map.rs:599–600`) says boundary pixels get the identity; `jacobian_at` (`:662–690`) uses one-sided differences, as spec L130 says. Fix code doc.
  - Python section: L964 `camera.to_pinhole()` does not exist; L1185 "keyword-only" but `#[pyo3(signature=(src, dst))]` is positional-or-keyword; L1186–1188 lists a Python `compute_svd()` that is not bound (`remap_aniso` computes the SVD itself, `warp.rs:292`); L983–984 says output has the input's shape, but H×W×1 returns H×W (`warp.rs:391`). Spec wrong on all four.
  - L1197 "dst camera bounds are validated" — the check is against the **src** image bounds (`warp_map.rs:349`).
  - L1017–1019 claims `from_cameras(fisheye, equirect)` has no NaN pixels — false (the fisheye limits the domain); the test (`warp_map/tests.rs:156`) checks only the centre.
  - **Phantom tests:** L1039–1057 (checkerboard-aniso, anisotropy-cap, aniso level-selection) and L1063–1066 (`cv2.remap`, `pycolmap.undistort_image` comparisons) name tests that exist nowhere in `remap/tests.rs`, `warp_map/tests.rs` or `tests/`.
  - L68–80 struct omits the `jacobians` field and `compute_jacobians`/`get_jacobian`/`from_patch`/`from_numpy`/`ImagePyramid`.
**Third copies:** `sampler.rs:64–85` (`Sampler` variant docs) repeats the cost ratios of spec L897–903 plus an unsourced "within ~1° of anisotropic"; `sampler.rs:148–161` (`minor_axis_loss`) restates L766–774 almost verbatim; `sampler.rs:322–331` restates L711–723; `sampler.rs:184–190` repeats the rule bullets of L781–785; `remap.rs:291–310` (`mip_level_for_sigma`) carries the GL-LOD/√2/f32-rounding rationale of L607–612; `remap.rs:334–337` repeats "1.8–4×". Shrink the code docs to contract + link. `warp_map.rs:66–113` (`*_PAR_MIN_PIXELS` timings, tabulated at L237–241): keep one copy, the code is a fair home.
**Opening:** "Image warping resamples a whole image from one camera into another through a precomputed per-pixel map: for each output pixel, the map holds the position in the input image to sample." Passes. The "two building blocks" paragraph (L14–21) omits the sampler rule (~40% of the spec) and calls the anisotropic sampler "area-weighted"; add "…and a per-view rule that picks between single-tap mip and anisotropic sampling for patch tiles." One spec per PR.
**Shape / prose:** Work-order residue: L372 "A new `Equirectangular` variant"; L956 "Expose … through sfmtool-py"; L499 "should:"; L1070 "*(Folded in from … draft)*"; L1093, L1188, L1203 "new constructor(s)"; L255 "The codebase already solved". L919 table row "photometric normal refinement (since deleted)" is stale. The `ray_to_pixel` domain rule is stated three times (L279–283, 353–359, 361–368); L1143–1148 transcribes `build_with_pose_impl`; L321–328 (`ViewCamera::project_homogeneous`) belongs in the bench spec. No mannered prose of note.
**Non-goals/deferrals checked:** 4, 1 stale: L950–952 calls storing `a` beside per-observation readings "proposed" in a draft — built (`observation_reading_options.anisotropic_threshold`, `readings.rs:113`, sfmr-file-format.md:2300). "Pose derivation stays in the Python wrapper" describes nothing (no caller in `src/`). Per-pixel depth map and GPU still absent (hold).
**Recommendation:** update code — STANDARDS § "Caller input never panics": `max_anisotropy = 0` reaches `.clamp(1, 0)` (`remap.rs:526`), which panics on any compressive pixel, and Python `WarpMap.remap_aniso(img, max_anisotropy=0)` (`warp.rs:280–291`) passes it straight through, so it raises `PanicException`; `remap_aniso`, `remap_bilinear_mip` and `WarpMap::new` also assert on caller input. Fix the two code-doc errors; `PyImagePyramid::new` (`warp.rs:323–324`) re-implements `full_levels` — call it. Then update the spec for everything above (delete or write the phantom tests).
**Unclear / incorrect / suspicious:** "~1° of anisotropic" (`sampler.rs:66`) has no source in any spec. The phantom test list gives false confidence in coverage. Settings structs `ViewSelectParams`, `KeypointLocalizeParams`, `KeypointSubpixelParams`, `MemberCoherenceParams` hold caller choices — `*Options` rule backlog.


## Code without specs

| surface | user-facing? | spec |
|---|---|---|
| CLI: 33 top-level commands in `cli.COMMANDS` plus `version` | yes | all 34 have a `specs/cli/**/…-command.md` (`ws` via `ws-init` and `ws-upgrade`) |
| `sfmtool-progress` | library | `gui/operation-progress.md`, STANDARDS |
| `sfmtool-archive-io` | library | `formats/archive-io-crate.md`, `formats/archive-container.md` |
| `sfmtool-{sift,matches,sfmr,camrig,kdf}-format` | file formats | `formats/*-file-format.md` |
| `sfmtool-colmap` | interop | `formats/colmap-interop.md` |
| `sfmtool-core` modules `analysis bench camera features geometry patch reconstruction spatial spherical` | library | `specs/core/<module>/` (2–19 specs each) |
| `sfmtool-core::numeric` (median/quantile/splitmix, ~700 lines with tests) | library | STANDARDS § on medians and quantiles; no design spec |
| `sfmtool-core::machine` | library | `cli/other/version-command.md` |
| `sfmtool-core::web_export` (atlas) | via `sfm web-export` | `cli/visualization/web-export-command.md` |
| `sfmtool-core::workspace` (feature options, named image lists; ~4,000 lines) | via workspace files | `workspace/workspace.md`, `workspace/named-image-lists.md` |
| `sfm-explorer` | yes | `specs/gui/` (30 specs), `gui/mcp-server.md` |
| `sfmtool-py` | Python API | `python-bindings.md` |
| Python subpackages `_commands align analyze camera camrig colmap compare feature_match merge motion rig sift strips visualization web_export xform` | yes | each referenced by at least one CLI or core spec; `compare/` only through `compare-command.md` |

### `crates/sfmtool-core/src/numeric.rs`
**What it does:** The one home of medians, quantiles and order statistics (NaN policy, in-place and sorted variants) and the `splitmix64` generator, enforced by allowlist tests in `numeric/tests.rs`.
**Why it matters:** internal-but-load-bearing. STANDARDS requires every median to come from here, and this audit found code that should use it but does not (`pairwise_flow/texture.rs:350–358` rolls its own SplitMix64).
**Recommendation:** add a short `specs/core/numeric.md` (or a section of `specs/core/README.md`) giving the NaN policy and which function to call. STANDARDS states the rule but not the interface.

### `src/sfmtool/web_export/`
**What it does:** Holds the static web viewer (`index.html`, `web-export.js`) and its Python code in `__init__.py`.
**Why it matters:** user-facing through `sfm web-export`. It is specified by `web-export-command.md`, but its package shape fits none of the four STANDARDS.md shapes (see the STANDARDS section).
**Recommendation:** acceptable as specced; settle the shape question.

## Top priorities

1. **Panics on caller input in public API, reachable from Python** (STANDARDS §
   "Caller input never panics"; update code). `WarpMap.remap_aniso(img,
   max_anisotropy=0)` reaches `.clamp(1, 0)` at
   `crates/sfmtool-core/src/camera/remap.rs:526` and raises `PanicException`.
   `constellation_seeds` probably indexes `cameras[i]` out of bounds when the
   `.kdf` forest covers more images than `views`
   (`bench/nearby/constellation.rs:327–333`; not run). This one is unconfirmed
   and should get a test first. Also `PhotographCache::new` (`levels == 0`) and
   `ImageU8::new`/`get_pixel`.
2. **`.sfmr` format spec still leans on code and has unversioned meaning
   changes** (failure 6). A reader cannot tell whether a file's
   reprojection-error column holds the mean or the RMS, or which meaning of
   `observation_confidence` it holds. The sampler rule and the reference-view
   rule are still defined only in core specs (open since 2026-10-09).
   `normal_confidence` is missing from the v4→5 migration. Fix the versioning
   gaps first, since they decide whether old files can be read correctly. Then
   move the writer and command lists into Implementations.
3. **`image-warping.md` describes tests and Python API that do not exist.** The
   checkerboard-aniso and anisotropy-cap tests, the `cv2.remap` and
   `pycolmap.undistort_image` comparisons, `camera.to_pinhole()` and the Python
   `compute_svd()` are all named but absent. Its pyramid is a box filter with
   one more level than stated. About 60 lines of sampler rationale are copied
   into `sampler.rs`/`remap.rs`. Rewrite the Python and Testing sections against
   the code, and cut the code docs back to contract plus link.
4. **Standards breaches the specs record as backlog** (update code):
   `build_track_at_pixel` passes `&Progress::none()` to its searches and fits
   (`track_at_pixel/finish.rs:164,177,276,463`, `members.rs:851,881,933`), so
   it can only be cancelled between members. The `.kdf` I/O lives in
   `sfmtool.spatial` with non-standard names rather than in `sfmtool.fileio`.
   `DisFlowParams`, `ConstellationParams` and the four patch `*Params` structs
   are settings structs, which the standard names `*Options`.
5. **Small behavioural and doc drift with a clear right side:**
   - `xform-command.md:352` says `coeffs` defaults to `8`. The code keeps the
     camera's own count when the camera already has that spline model.
   - `sfm solve`'s `--sfmr-dir` default is `<workspace>/sfmr`, not `sfmr/`.
     The spec also omits the `--flow-match` + `.matches` refusal.
   - The `nearby_points` binding docstring still says "finite" since #1245.
   - `drafts/track-at-pixel.md` is stale on six points and should be converted
     or retired.
