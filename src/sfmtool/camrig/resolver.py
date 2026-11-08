# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Discover and resolve `.camrig` files for `sfm solve`.

A `.camrig` dropped in a workspace describes the camera(s) for the images its
stored patterns match. `sfm solve` auto-discovers these:

- a **single-sensor** `.camrig` (such as one written by `sfm camrig create`)
  supplies one camera used as the intrinsics prior;
- a **multi-sensor** `.camrig` (such as one written by `sfm insv2rig`)
  describes a rig — its sensors, their shared/per-sensor cameras, and their
  `sensor_from_rig` poses — and drives rig-aware SfM.

When every image being solved is covered by one `.camrig`, that file is used,
taking precedence over `camera_config.json` and `rig_config.json`. With
`--camera-model`, each of its cameras the solve uses is refit to that model
and the refit camera is the starting intrinsics; the file is not changed.
`sfm pick-images` finds its rig with the same rule, through
`find_covering_camrig`, so the list it writes is one `sfm solve` takes.
"""

from dataclasses import dataclass
from pathlib import Path

from .pattern import match_pattern, match_pattern_with_frames

# Allowed relative difference between an image's aspect ratio and the
# `.camrig` camera's aspect ratio before the rig is rejected.
_ASPECT_TOLERANCE = 1e-3


class CamrigSolveError(Exception):
    """Raised when discovered `.camrig` files cannot be used for a solve."""


@dataclass
class CamrigRig:
    """A multi-sensor `.camrig` resolved for a solve.

    `data` is the `read_camrig` dict (metadata, cameras, sensor patterns,
    camera indexes, and `sensor_from_rig` pose arrays). `assignments` maps each
    resolved image path being solved to its `(sensor_index, frame_index)` in
    the rig — the sensor it belongs to and the frame it was captured in.
    """

    camrig_file: Path
    data: dict
    assignments: dict[Path, tuple[int, int]]


@dataclass
class CamrigSolveResult:
    """The `.camrig` covering a solve.

    Exactly one of `camera` / `rig` is set: `camera` for a single-sensor
    `.camrig` (one camera dict, `{model, width, height, parameters}`), `rig`
    for a multi-sensor `.camrig`.
    """

    camrig_file: Path
    camera: dict | None = None
    rig: CamrigRig | None = None

    @property
    def is_multi_sensor(self) -> bool:
        return self.rig is not None


def _covered_paths(camrig_dir: Path, patterns: list[str]) -> set[Path]:
    """Resolved absolute paths the `.camrig`'s patterns match from its directory."""
    covered: set[Path] = set()
    for pattern in patterns:
        covered.update(match_pattern(camrig_dir, pattern))
    return covered


def _check_image_resolution(
    image_paths, camera: dict, camrig_file: Path, label: str = ""
) -> None:
    """Verify a set of images suits one `.camrig` camera.

    `sfm solve` uses one camera per sensor, so every image of that sensor must
    share a resolution, and that resolution must match the camera's aspect
    ratio (focal length and principal point are scaled uniformly to the actual
    image size). A mixed-resolution set would otherwise be silently mis-scaled
    off the first image alone.
    """
    from ..camera.setup import _read_image_size

    where = f"{camrig_file}{label}"
    by_size: dict[tuple[int, int], list] = {}
    for image_path in image_paths:
        by_size.setdefault(_read_image_size(Path(image_path)), []).append(image_path)

    if len(by_size) > 1:
        lines = []
        for (w, h), paths in sorted(by_size.items()):
            example = sorted(str(p) for p in paths)[0]
            lines.append(f"  {w}x{h}  ({len(paths)} image(s))  e.g. {example}")
        raise CamrigSolveError(
            f"the images covered by {where} have mixed resolutions; a "
            ".camrig camera describes one resolution, so every image it "
            "covers must share one:\n" + "\n".join(lines)
        )

    (image_w, image_h) = next(iter(by_size))
    camera_w, camera_h = camera["width"], camera["height"]
    if abs(image_w / image_h - camera_w / camera_h) > _ASPECT_TOLERANCE:
        raise CamrigSolveError(
            f"image resolution {image_w}x{image_h} does not match the aspect "
            f"ratio of the camera in {where} ({camera_w}x{camera_h}); "
            "the .camrig was calibrated for a different image shape."
        )


def _camera_config_note(camrig_file, image_paths, camera_config_resolver) -> None:
    """Print a note when a `.camrig` overrides a `camera_config.json`."""
    if camera_config_resolver is None:
        return
    for image_path in image_paths:
        if camera_config_resolver.resolve_for_image(Path(image_path)) is not None:
            print(
                f"Note: {camrig_file} takes precedence over "
                "camera_config.json for these images."
            )
            break


def _build_rig_assignments(
    camrig_file: Path, data: dict
) -> dict[Path, tuple[int, int]]:
    """Match every sensor's pattern, returning `path -> (sensor, frame)`.

    Raises `CamrigSolveError` if two sensors' patterns match the same file
    (each file must belong to exactly one sensor) or if two files of one
    sensor capture the same frame index (each sensor contributes at most one
    image per rig frame — otherwise a COLMAP frame would carry two images for
    one sensor). The latter can happen with variable-width frame fields, e.g.
    `frame_%d.jpg` matching both `frame_1.jpg` and `frame_001.jpg`.
    """
    camrig_dir = camrig_file.parent
    patterns = data["sensor_image_patterns"]
    assignments: dict[Path, tuple[int, int]] = {}
    for sensor_idx, pattern in enumerate(patterns):
        frame_to_path: dict[int, Path] = {}
        for path, frame_idx in match_pattern_with_frames(camrig_dir, pattern):
            existing = assignments.get(path)
            if existing is not None:
                raise CamrigSolveError(
                    f"{camrig_file}: image {path} is matched by sensor "
                    f"{existing[0]} and sensor {sensor_idx}; each image must "
                    "belong to exactly one sensor."
                )
            clash = frame_to_path.get(frame_idx)
            if clash is not None:
                raise CamrigSolveError(
                    f"{camrig_file}: sensor {sensor_idx} pattern {pattern!r} "
                    f"matches {clash} and {path} with the same frame index "
                    f"{frame_idx}; each sensor contributes at most one image "
                    "per rig frame."
                )
            frame_to_path[frame_idx] = path
            assignments[path] = (sensor_idx, frame_idx)
    return assignments


def _refit_camera(
    camera: dict, camera_model: str, camera_index: int, camrig_file: Path
) -> dict:
    """The `.camrig` camera dict `camera` refit to `camera_model`, as a dict."""
    from .._sfmtool.geometry import CameraIntrinsics
    from ..camera.setup import refit_intrinsics_to_model

    try:
        refit = refit_intrinsics_to_model(
            CameraIntrinsics.from_dict(camera),
            camera_model,
            f"camera {camera_index} of {camrig_file}",
        )
    except ValueError as e:
        raise CamrigSolveError(str(e)) from None
    return refit.to_dict()


def _resolve_single_sensor(
    camrig_file: Path,
    data: dict,
    image_paths,
    camera_model,
    camera_config_resolver,
) -> CamrigSolveResult:
    """Validate and resolve a single-sensor `.camrig` covering a solve."""
    camera_index = data["camera_indexes"][0]
    camera = data["cameras"][camera_index]
    _check_image_resolution(image_paths, camera, camrig_file)
    _camera_config_note(camrig_file, image_paths, camera_config_resolver)
    print(f"Camera intrinsics: using {camrig_file}")
    if camera_model is not None:
        camera = _refit_camera(camera, camera_model, camera_index, camrig_file)
    return CamrigSolveResult(camrig_file=camrig_file, camera=camera)


def _resolve_multi_sensor(
    camrig_file: Path,
    data: dict,
    solve_set: set[Path],
    camera_model,
    camera_config_resolver,
) -> CamrigSolveResult:
    """Validate and resolve a multi-sensor `.camrig` covering a solve."""
    assignments = _build_rig_assignments(camrig_file, data)

    # Per-sensor resolution check: each sensor's solve images must suit that
    # sensor's pool camera.
    solve_assignments = {p: assignments[p] for p in solve_set}
    by_sensor: dict[int, list[Path]] = {}
    for path, (sensor_idx, _frame) in solve_assignments.items():
        by_sensor.setdefault(sensor_idx, []).append(path)
    for sensor_idx, paths in sorted(by_sensor.items()):
        camera = data["cameras"][data["camera_indexes"][sensor_idx]]
        _check_image_resolution(
            paths, camera, camrig_file, label=f" (sensor {sensor_idx})"
        )

    _camera_config_note(camrig_file, solve_set, camera_config_resolver)
    sensor_count = data["metadata"]["sensor_count"]
    print(
        f"Camera rig: using {camrig_file} "
        f"({sensor_count} sensors, {len(by_sensor)} present in this solve)"
    )
    if camera_model is not None:
        # Refit each pool camera a present sensor uses, once, and hand the
        # solve a copy of the rig data holding the refit cameras.
        cameras = list(data["cameras"])
        used = sorted({data["camera_indexes"][s] for s in by_sensor})
        for camera_index in used:
            cameras[camera_index] = _refit_camera(
                cameras[camera_index], camera_model, camera_index, camrig_file
            )
        data = {**data, "cameras": cameras}
    return CamrigSolveResult(
        camrig_file=camrig_file,
        rig=CamrigRig(
            camrig_file=camrig_file, data=data, assignments=solve_assignments
        ),
    )


def find_covering_camrig(
    image_paths,
    workspace_dir,
    images: str = "images being solved",
    act: str = "solve",
) -> tuple[Path, dict] | None:
    """Find the `.camrig` that covers every one of `image_paths`.

    Scans `workspace_dir` for `.camrig` files and returns the path and the
    `read_camrig` dict of the one whose patterns match all of `image_paths`,
    or `None` when no `.camrig` matches any of them. `images` names the
    images in a message ("images being solved") and `act` is the verb that
    advises what to do instead ("solve").

    Raises `CamrigSolveError` when the images span more than one `.camrig`,
    or when a `.camrig` covers only some of them: `sfm solve` takes one
    `.camrig` for all of its images or none.
    """
    if workspace_dir is None:
        return None
    workspace_dir = Path(workspace_dir).resolve()
    camrig_files = sorted(workspace_dir.rglob("*.camrig"))
    if not camrig_files:
        return None

    from .._sfmtool.fileio import read_camrig

    given = {Path(p).resolve() for p in image_paths}

    # Each .camrig that covers at least one of the images, with its data and
    # the images it covers.
    hits: list[tuple[Path, dict, set[Path]]] = []
    for camrig_file in camrig_files:
        try:
            data = read_camrig(str(camrig_file))
        except Exception as e:
            raise CamrigSolveError(f"could not read {camrig_file}: {e}") from None
        patterns = data["sensor_image_patterns"]
        if not patterns:
            continue  # geometry-only rig, not backed by workspace images
        covered = _covered_paths(camrig_file.parent, patterns) & given
        if covered:
            hits.append((camrig_file, data, covered))

    if not hits:
        return None
    if len(hits) > 1:
        names = ", ".join(str(c) for c, _, _ in sorted(hits, key=lambda h: h[0]))
        raise CamrigSolveError(
            f"the {images} are covered by multiple .camrig files "
            f"({names}); {act} each .camrig's images separately."
        )

    camrig_file, data, covered = hits[0]
    uncovered = given - covered
    if uncovered:
        example = sorted(uncovered)[0]
        by = (
            "its pattern"
            if data["metadata"]["sensor_count"] == 1
            else "any sensor pattern"
        )
        raise CamrigSolveError(
            f"{camrig_file} matches only {len(covered)} of {len(given)} "
            f"{images}; e.g. {example} is not matched by {by}. "
            f"{act.capitalize()} only the images the .camrig covers, or remove "
            "the .camrig."
        )
    return camrig_file, data


def resolve_camrig_for_solve(
    image_paths,
    workspace_dir,
    camera_model,
    camera_config_resolver=None,
) -> CamrigSolveResult | None:
    """Find the `.camrig` covering every image being solved.

    Scans `workspace_dir` for `.camrig` files and returns a `CamrigSolveResult`
    for the one that covers all of `image_paths`, or `None` when no `.camrig`
    applies. A single-sensor `.camrig` yields a `.camera`; a multi-sensor
    `.camrig` yields a `.rig`.

    With `camera_model` (the solve's `--camera-model`), each `.camrig` camera
    the solve uses is refit to that model, and the result carries the refit
    cameras in place of the file's.

    Raises `CamrigSolveError` when the discovered `.camrig` files cannot be
    used: the images span multiple `.camrig` files, a `.camrig` covers only
    some of them, or a camera's refit to `camera_model` is refused.
    """
    found = find_covering_camrig(image_paths, workspace_dir)
    if found is None:
        return None
    camrig_file, data = found
    if data["metadata"]["sensor_count"] == 1:
        return _resolve_single_sensor(
            camrig_file, data, image_paths, camera_model, camera_config_resolver
        )
    solve_set = {Path(p).resolve() for p in image_paths}
    return _resolve_multi_sensor(
        camrig_file, data, solve_set, camera_model, camera_config_resolver
    )
