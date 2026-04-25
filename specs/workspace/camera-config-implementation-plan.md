# Camera Config Implementation Plan

Implementation plan for the features specified in
[camera-config.md](camera-config.md). Phases are ordered so each is
independently shippable: stopping after Phase 3 still gives a usable feature;
Phases 4-6 add polish and convenience.

## Phase 1 — File parsing and resolution

**New module: `src/sfmtool/_camera_config.py`** (parallel to `_rig_config.py`).

- `load_camera_config(path: Path) -> dict | None` — reads JSON, validates
  `version == 1`, returns the `camera_intrinsics` dict (or `None` if missing).
- `find_camera_config_for_directory(image_dir: Path, workspace_dir: Path) -> tuple[Path, dict] | None`
  — closest-ancestor walk from `image_dir` up to (and including)
  `workspace_dir`, returning the resolved file path and parsed dict. Stop
  cleanly if `image_dir` isn't inside `workspace_dir`. The directory (not the
  image) is the resolution key — every image in the same directory shares
  the same answer.
- A `CameraConfigResolver` class (or module-level memoized function) holds a
  `dict[Path, tuple[Path, dict] | None]` keyed by image directory. First
  image in a directory walks the parent chain; every subsequent image in the
  same directory is an `O(1)` hit. Build a fresh resolver per CLI invocation
  — not a process-lifetime cache, since `camera_config.json` files can
  change between runs.
- Schema validation: reject unknown top-level keys, validate `model` against
  the keys of `_CAMERA_PARAM_NAMES`, validate that any focal/principal value
  in `parameters` is paired with `width` *and* `height`.

**Tests: `tests/test_camera_config.py`** — file-not-found returns `None`;
closest-ancestor across depth 0/1/2; nested override; outside-workspace
rejection; schema errors (bad version, missing width/height when focal
supplied, unknown model).

## Phase 2 — Intrinsics construction with resolution scaling

The internal currency is `CameraIntrinsics` (the Rust-bound type from
`sfmtool._sfmtool`), not `pycolmap.Camera`. pycolmap stays at the edge — only
`_colmap_db.py` converts via the existing `colmap_camera_from_intrinsics` when
writing to the database.

**New helper in `_camera_setup.py`:**

```python
def build_intrinsics_from_camera_config(
    camera_config: dict | None,        # parsed camera_intrinsics block, or None
    image_path: Path,                  # for EXIF fallback + actual size
    camera_model_override: str | None, # only used when camera_config is None
) -> tuple[CameraIntrinsics, bool]:    # bool = treat as prior (full block given)
```

Logic:

1. If `camera_config is None` → call existing `_infer_camera(image_path,
   camera_model_override)` (returns `pycolmap.Camera`), convert via
   `pycolmap_camera_to_intrinsics`, return `(intrinsics, False)`.
2. If `camera_config` has `model` only → same as #1 but pass
   `camera_config["model"]` as the override; `(intrinsics, False)`.
3. If `camera_config` has distortion-only `parameters` (no focal/principal)
   → infer focal+principal from EXIF, overlay the distortion coefficients
   onto the resulting `CameraIntrinsics`, return `(intrinsics, False)`.
4. If `camera_config` has full `parameters` + `width`/`height`:
   - Read actual image dimensions.
   - If `(actual_w, actual_h) == (calib_w, calib_h)` → build
     `CameraIntrinsics` from the dict as-is.
   - Else if `|actual_w/actual_h - calib_w/calib_h| < tol` (e.g. `1e-3`) →
     scale focal+principal by `s = actual_w / calib_w`, distortion unchanged,
     then build.
   - Else → raise `CameraConfigError` with both sizes in the message.
   - Return `(intrinsics, True)` so the caller knows to flag
     `has_prior_focal_length=True` at the pycolmap boundary.

`_infer_camera` itself stays in pycolmap terms — it's the wrapper around
`pycolmap.infer_camera_from_image()` and lives at the EXIF/pycolmap boundary
by design. Just don't propagate its return type outward.

The rig path (`_camera_from_rig_intrinsics`) is left alone for now per the
earlier decision; revisit when we unify with `rig_config.json`.

**Tests: extend `tests/test_camera_config.py`** — model-only delegates to
`_infer_camera`; distortion-only overlays correctly; full-block at matching
resolution; full-block at uniform downscale (assert exact fx/fy/cx/cy values,
distortion unchanged); aspect mismatch raises; the prior-flag bool returns
correctly across the four cases.

## Phase 3 — Integration into camera setup

**Update `_camera_setup.py`** to expose a single entry point that callers
thread a `CameraConfigResolver` into:

```python
def intrinsics_for_image(
    image_path: Path,
    resolver: CameraConfigResolver | None,  # None = no workspace, EXIF-only
    camera_model_override: str | None,
) -> tuple[CameraIntrinsics, bool]:         # bool = prior flag
```

This is what every non-rig camera-creation site should call. Internally it
asks the resolver for the closest `camera_config.json` (cached by image
directory) and dispatches to `build_intrinsics_from_camera_config`.

Each command builds one resolver up front from the workspace dir and
threads it through; this guarantees a single solve walks the parent chain
once per unique image directory, not once per image.

**Caller updates in `_colmap_db.py`** (the pycolmap boundary):

- `_setup_for_sfm` (single-camera-per-folder),
  `_setup_for_sfm_from_matches`, and the non-rig branches of the
  matches-flow path — replace direct `_infer_camera` calls. Get a
  `CameraIntrinsics` from `intrinsics_for_image`, convert to
  `pycolmap.Camera` via `colmap_camera_from_intrinsics`, set
  `has_prior_focal_length` from the returned bool, then write to the DB.
- `_setup_db_with_rigs` — only the unmatched-image fallback path needs
  updating; rig sensors keep their current behavior.

The workspace dir is already discoverable in every command (it's resolved
during input expansion). Plumb it through whatever signature reaches these
helpers.

**Tests: `tests/test_cli_solve.py`, `test_cli_match.py`, etc.** — add a
fixture variant of `isolated_seoul_bull_17_images` that ships a
`camera_config.json` at the workspace root, and assert the resulting
`.sfmr` has the configured intrinsics (or scaled values, in a downscale
variant).

## Phase 4 — CLI conflict detection

Add `_check_camera_model_conflict(image_paths, resolver,
camera_model_override)` in `_camera_setup.py`. For each image, ask the
shared `CameraConfigResolver` (built once at command entry) whether its
directory resolves a `camera_config.json`; if any does *and*
`camera_model_override is not None`, raise `click.UsageError` listing one
example image and its resolved file.

Wire this into the command entry points that accept `--camera-model`:

- `_commands/solve.py`
- `_commands/match.py`
- `_commands/to_colmap_db.py`
- `_commands/densify.py` (if it accepts `--camera-model`; check while wiring)

Run the check after image expansion and before any expensive work.

**Tests** — single test per command verifying the error fires and exits
non-zero before any real work happens.

## Phase 5 — Doc and spec cross-references

Small touch-ups, no code:

- `specs/workspace/workspace.md` — add a one-paragraph "Camera intrinsics"
  subsection pointing at `camera-config.md`.
- `specs/cli/{init,solve,match,to-colmap-db,densify}-command.md` — note in
  each that `--camera-model` is rejected when a `camera_config.json`
  resolves.
- `CLAUDE.md` — mention `_camera_config.py` in the Python utility modules
  list.

## Phase 6 (deferred) — `sfm cam cp`

Introduce a new top-level command **group** `sfm cam` for camera-related
operations, with `cp` (copy) as its first subcommand:

```
sfm cam cp <reconstruction.sfmr> <camera_config.json>
sfm cam cp <reconstruction.sfmr> --index N <camera_config.json>
```

Behavior:

- Reads the reconstruction, extracts the chosen camera's `CameraIntrinsics`,
  serializes to the `camera_config.json` schema (with `version: 1`), writes
  the file.
- If the reconstruction has exactly one camera, `--index` may be omitted.
  If it has multiple and `--index` is not supplied, exit with an error
  listing the count and suggesting `--index`.
- The `cp` naming sets up `mv`, `ls`, `print`, etc. as future subcommands
  if they earn their keep.

This is the first command group in the CLI (everything else under `sfm`
is currently flat). Implementation details:

- New file `_commands/cam.py` defining a Click group `cam` with one
  subcommand `cp`. Register the group on `cli` in `cli.py` the same way
  individual commands are registered today.
- Spec at `specs/cli/cam-command.md` (parallel to other CLI specs).
- `pycolmap_camera_to_intrinsics` in `_cameras.py` already produces the
  right shape — the command is mostly file plumbing.

This makes Workflow 1 ("Bootstrap from an Unknown Camera") first-class;
ships after Phases 1-3 prove out.

## Decisions to lock before Phase 2 starts

1. **Aspect-ratio tolerance.** Default to `1e-3` (relative) and revisit if
   it bites. Strict equality fails on real numbers; too loose silently
   mis-scales.
2. **Distortion-only overlay precedence.** When `parameters` supplies only
   some distortion coefficients (e.g. `k1` but not `k2`), unspecified ones
   zero out rather than fall back to EXIF — "specify what you mean."
3. **Resolution caching.** Resolve and cache *per image directory*, not per
   image. A typical workspace has thousands of images across a handful of
   directories, so the parent-chain walk runs once per unique directory —
   `O(unique_dirs × depth)` — and every other image in that directory is an
   `O(1)` cache hit. The cache lives on a `CameraConfigResolver` instance
   built fresh per CLI invocation; both Phase 3's `intrinsics_for_image`
   and Phase 4's conflict check share the same resolver so neither pays
   the walk cost twice.
