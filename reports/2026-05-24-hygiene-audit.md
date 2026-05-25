# Codebase Hygiene Audit — 2026-05-24 (namespace focus)

> _Status annotations added 2026-05-24 (after commits cb3f185 and 52518a2) appear inline per recommendation. The body below is the original read-only snapshot and is unchanged._

Read-only structural survey of the sfmtool repository, scoped to **flat
namespacing** in both the Python package (`src/sfmtool/`) and the Rust
workspace (`crates/`). The question driving this snapshot: where does the code
lean on flat module collections, and how would more nested namespaces help?

Method: enumerated every top-level module / `pub mod` declaration, clustered by
shared name prefix and by which CLI command consumes it, and checked the public
re-export surface (`__init__.py`, the PyO3 `#[pymodule]`) to size the blast
radius of any regrouping. No code was modified.

## The core smell, in one sentence

Three of the largest namespaces are **flat collections in which the file/module
name prefix already spells out the package that doesn't exist** — `_camrig_*`,
`_discontinuity_*`, `_analyze_*` (Python) and `py_*_io` (Rust). The prefix is a
hand-rolled namespace; a directory or submodule would make it real and
enforced.

| Namespace | Members | Nesting today |
|---|---|---|
| `src/sfmtool/*.py` (top level) | **53** flat `_*.py` modules | 4 subpackages |
| `crates/sfmtool-core/src` | **29** flat `pub mod` in `lib.rs` | 3 already nested (`alignment/`, `feature_match/`, `optical_flow/`) |
| `crates/sfmtool-py/src` | **30** flat `py_*.rs`; **68** classes/functions on one flat module | 0 submodules |

The top-level Python count has grown from 46 → 53 since the last audit while the
subpackage count held at 4, so the flat surface is widening, not shrinking. The
recent `_discontinuity.py` split (#24) is illustrative: a 1217-line file became
four flat siblings (`_discontinuity_reconstruction`, `_discontinuity_image_sequence`,
`_discontinuity_json`, `_discontinuity_constants`) rather than a `discontinuity/`
package — the right call on file size, but it added four entries to the flat list
instead of one package.

---

## Recommendations

### 1. `_camrig_*` → `camrig/` package

> _Status (2026-05-24): Done — `camrig/` package created in commit cb3f185._

- Location: `src/sfmtool/_camrig_cp.py` (477), `_camrig_create.py` (376), `_camrig_resolver.py` (314), `_camrig_pattern.py` (78) — ~1245 lines across 4 flat modules
- Problem: The newest cluster, fully flat. The `_camrig_` prefix is doing a directory's job; the four modules back the `sfm camrig` command group (`_commands/camrig.py`) plus rig resolution used by `inspect`/`solve`. Internal cross-imports already use the prefix as a faux package (`from .._camrig_cp`, `from ._camrig_pattern`).
- Proposed fix: regroup under `camrig/` → `camrig/cp.py`, `camrig/create.py`, `camrig/resolver.py`, `camrig/pattern.py`. Imports become `from ..camrig import resolver`.
- Effort: low · Risk: low — none of the four are re-exported from `__init__.py`; all callers are internal.

### 2. `_analyze_*` / `_inspect_summary` → `analyze/` package (and fix the split prefix)

> _Status (2026-05-24): Done — `analyze/` package created in commit 52518a2; `_inspect_summary` is now `analyze/summary.py`, resolving the split prefix._

- Location: `src/sfmtool/_analyze_images.py` (501), `_analyze_metrics.py` (238), `_analyze_graphs.py` (159), `_analyze_depth.py` (87), `_inspect_summary.py` (475)
- Problem: Two issues at once. (a) Five sibling modules back the `sfm analyze` / `sfm inspect` commands — a flat cluster. (b) **Misleading-name smell**: four use the `_analyze_` prefix but the fifth is `_inspect_summary`, and `_inspect_summary` is imported by *both* `_commands/analyze.py` and `_commands/inspect.py`. The prefix no longer reliably signals the cluster after the inspect/analyze reshuffle (#20).
- Proposed fix: `analyze/` package → `analyze/images.py`, `analyze/metrics.py`, `analyze/graphs.py`, `analyze/depth.py`, `analyze/summary.py`. The package name resolves the prefix ambiguity for free.
- Effort: low · Risk: low — internal callers only.

### 3. `_discontinuity_*` → `discontinuity/` package

> _Status (2026-05-24): Done — landed as the `motion/` package (the command was renamed discontinuity→motion) in commit cb3f185._

- Location: `_discontinuity_reconstruction.py` (954), `_discontinuity_image_sequence.py` (253), `_discontinuity_json.py` (328), `_discontinuity_constants.py`
- Problem: Four flat siblings, the product of the recent file-size split (#24). The split was correct; leaving them as flat top-level neighbors was the missed half. `_discontinuity_constants.py` existing as its own top-level module is a particularly clear "this wants to be a package-private `constants.py`" signal.
- Proposed fix: `discontinuity/` package → `reconstruction.py`, `image_sequence.py`, `json.py`, `constants.py`.
- Effort: low · Risk: low — backs only `_commands/discontinuity.py`.

### 4. `_align_*` and `_merge_*` → `align/` and `merge/` packages

> _Status (2026-05-24): Done — `align/` and `merge/` packages created in commit 52518a2. The `merge/correspondences.py` vs top-level `_point_correspondence.py` overlap noted here still stands._

- Location: `_align.py`, `_align_by_cameras.py`, `_align_by_points.py`, `_multi_align.py` (4) · `_merge.py`, `_merge_correspondences.py`, `_merge_pose_refinement.py` (3)
- Problem: Two more prefix-as-namespace clusters, each backing a single command (`align`, `merge`). `_merge_correspondences` also overlaps the correspondence concern with `_point_correspondence.py`.
- Proposed fix: `align/` and `merge/` packages, mirroring the existing `xform/` subpackage style.
- Effort: low · Risk: low.

### 5. Camera / rig / panorama cluster → `camera/` + `rig/`

> _Status (2026-05-24): Not done — the camera/rig/panorama modules are all still flat at the package root._

- Location: `_cameras.py`, `_camera_config.py`, `_camera_setup.py`, `_rig_config.py`, `_rig_frames.py`, `_insv2rig.py`, `_pano2rig.py`, `_spherical_tile_rig.py`, `_panorama.py` (9 modules)
- Problem: The single largest conceptual cluster on the flat surface, mixing intrinsics, per-directory config resolution, rig ingestion (insta360/pano → rig), and equirect rendering. With `_camrig_*` (rec #1) adjacent, there are effectively ~13 camera/rig modules sitting flat.
- Proposed fix: split into `camera/` (intrinsics, config, setup) and `rig/` (frames, config, `insv2rig`, `pano2rig`, `spherical_tile_rig`, `panorama`).
- Effort: medium · Risk: medium — `_camera_config.py` is referenced in CLAUDE.md and several commands; `_spherical_tile_rig.py` is re-exported from `__init__.py`, so its public path moves (keep a shim or update `__init__`).

### 6. COLMAP and SIFT I/O clusters → `colmap/` + `sift/`

> _Status (2026-05-24): Partially done — `sift/` package created in commit 52518a2 (read/write landed as `sift/file.py`, not `format.py`; the extraction entry points still live there). `colmap/` not done._

- Location: `_colmap_db.py` (860), `_colmap_io.py` (674), `_to_colmap_db.py`, `_extract_sift_colmap.py` · `_sift_file.py` (764), `_extract_sift_opencv.py`, `_extract_sift_colmap.py`
- Problem: Two I/O families; `_extract_sift_colmap.py` straddles both. `_colmap_db.py` and `_colmap_io.py` are also two of the largest files in the package.
- Proposed fix: `colmap/` and `sift/` packages.
- Effort: medium · Risk: medium — `_sift_file.py`, `_extract_sift_colmap.py`, `_extract_sift_opencv.py` are all re-exported from `__init__.py`; the public surface moves and must be re-exported deliberately.

### 7. `sfmtool-core`: `geometry/`, `camera/`, `spherical/` modules

> _Status (2026-05-24): Not done — `sfmtool-core/src/lib.rs` still declares the modules flat._

- Location: `crates/sfmtool-core/src/lib.rs` — 29 flat `pub mod` declarations
- Problem: Three subsystems are already nested (`alignment/`, `feature_match/`, `optical_flow/`), proving the pattern fits; the rest haven't followed. Clear flat groups: transforms (`rigid_transform`, `rot_quaternion`, `rotation`, `se3_transform`, `transform`, `viewing_angle` — 6), camera/imaging (`camera`, `camera_intrinsics`, `distortion`, `frustum`, `rectification`, `remap`, `warp_map` — 7), spherical/rig (`sphere_points`, `spherical_tile_rig`, `per_spherical_tile_source_stack`, `consensus_atlas` — 4), and the new infinity pair (`find_infinity`, `infinity`).
- Proposed fix: `geometry/`, `camera/`, `spherical/` modules; pair the infinity modules. `lib.rs` shrinks from a 29-line wall of `pub mod` to ~8 grouped declarations, and `use` paths gain a meaningful segment (`camera::Distortion` vs `distortion::Distortion`).
- Effort: low (mechanical) · Risk: medium — public paths change for external `use sfmtool_core::se3_transform::…`, and `sfmtool-py` imports many of these. Re-export from `lib.rs` to soften.

### 8. `sfmtool-py`: introduce PyO3 submodules (the only flatness that reaches the public API)

> _Status (2026-05-24): Not done — `sfmtool-py` still registers all classes/functions on one flat module._

- Location: `crates/sfmtool-py/src/` — 30 flat `py_*.rs` files; `lib.rs` registers **68** `add_class`/`add_function` calls onto a single module `m`, with **zero** `add_submodule` calls
- Problem: The `py_` prefix is the exact Rust analog of Python's `_` prefix — a flat namespace by naming convention. Everything lands flat in `sfmtool._sfmtool`, and `__init__.py` does `from sfmtool._sfmtool import *`, dumping ~68 names into `sfmtool`'s top level with no structure. Clear groups: I/O (`py_sfmr_io`, `py_sift_io`, `py_matches_io`, `py_camrig_io`, `py_colmap_binary`, `py_colmap_db`), matching (`py_descriptor_match`, `py_image_match`, `py_sweep_match`), geometry (`py_rigid_transform`, `py_rot_quaternion`, `py_se3_transform`, `py_camera_intrinsics`, `py_sphere_points`), flow (`py_optical_flow`, `py_warp_map`).
- Proposed fix: expose PyO3 child modules via `add_submodule` — `_sfmtool.io`, `_sfmtool.match`, `_sfmtool.geometry`, `_sfmtool.flow` — and group the `py_*.rs` files into matching subdirectories. `__init__.py` can then re-export deliberately (`from ._sfmtool.io import read_sfmr`) instead of `import *`.
- Effort: medium-high · Risk: medium — the `import *` consumers and the binding call sites all move; PyO3 submodule registration has `sys.modules` sharp edges.

---

## What's already good (don't touch)

- `_commands/` — one file per CLI subcommand; the commands are peers, so this flatness is *correct*.
- `feature_match/`, `xform/`, `visualization/` (Python) and `sfm-explorer` (`scene_renderer/pipelines/`, `viewer_3d/`, `platform/`) — the models the rest should follow.
- The small format crates (`sift-format`, `matches-format`, `camrig-format`, ~6 files each) are coherent and need no grouping.

---

## Top 3 (best effort-to-value)

> _Status (2026-05-24): All three completed — camrig and discontinuity→motion in cb3f185, analyze in 52518a2._

1. **`camrig/` package (rec #1)** — 4 modules, ~1245 lines, zero public-surface impact, all internal callers. The cheapest high-value regrouping and the newest/worst-growing cluster.
2. **`discontinuity/` package (rec #3)** — finishes the job the recent file split (#24) started; a top-level `_discontinuity_constants.py` is an unambiguous "make me package-private" signal. Internal callers only.
3. **`analyze/` package (rec #2)** — same low-risk move, and it simultaneously resolves the `_analyze_` vs `_inspect_summary` prefix inconsistency left by the inspect/analyze reshuffle.

All three are low-effort/low-risk Python regroupings with no public-API churn — ideal for proving the convention before tackling the higher-risk `sfmtool-core` re-export work (rec #7) and the `sfmtool-py` submodule restructure (rec #8), which is the only flatness that actually leaks into the public Python API.
