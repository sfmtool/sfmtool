# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Discover and resolve `.camrig` files for `sfm solve`.

A `.camrig` dropped in a workspace describes the camera(s) for the images its
stored patterns match. `sfm solve` auto-discovers these:

- a **single-sensor** `.camrig` (such as one written by `sfm camrig create`)
  supplies one camera, used as the intrinsics prior for the images it covers;
- a **multi-sensor** `.camrig` (such as one written by `sfm insv2rig`)
  describes a rig — its sensors, their shared/per-sensor cameras, and their
  `sensor_from_rig` poses — and drives rig-aware SfM.

The images being solved are covered either by one multi-sensor `.camrig` or
by one or more single-sensor `.camrig` files, each image by exactly one file,
and each single-sensor file is then one camera of the solve. The `.camrig`
files take precedence over `camera_config.json` and `rig_config.json`. With
`--camera-model`, each of their cameras the solve uses is refit to that model
and the refit camera is the starting intrinsics; the files are not changed.
`sfm pick-images` applies the same rule, through `find_covering_camrigs`, so
the list it writes is one `sfm solve` takes.
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
class CamrigCamera:
    """A single-sensor `.camrig` resolved for a solve.

    `camera` is its camera dict (`{model, width, height, parameters}`), refit
    to `--camera-model` when one is given, and `images` holds the resolved
    paths of the images being solved that the file covers.
    """

    camrig_file: Path
    camera: dict
    images: set[Path]


@dataclass
class CamrigSolveResult:
    """The `.camrig` files covering a solve.

    Exactly one of `cameras` / `rig` is set: `cameras` holds one
    `CamrigCamera` per single-sensor `.camrig` covering some of the images,
    in path order, and `rig` a multi-sensor `.camrig` covering all of them.
    """

    cameras: list[CamrigCamera] | None = None
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
    from ..camera.setup import IntrinsicsRefitError, refit_intrinsics_to_model

    try:
        refit = refit_intrinsics_to_model(
            CameraIntrinsics.from_dict(camera),
            camera_model,
            f"camera {camera_index} of {camrig_file}",
        )
    except IntrinsicsRefitError as e:
        raise CamrigSolveError(str(e)) from None
    return refit.to_dict()


def _resolve_single_sensor(
    camrig_file: Path,
    data: dict,
    covered: set[Path],
    camera_model,
    camera_config_resolver,
) -> CamrigCamera:
    """Validate and resolve a single-sensor `.camrig` covering `covered`."""
    camera_index = data["camera_indexes"][0]
    camera = data["cameras"][camera_index]
    _check_image_resolution(sorted(covered), camera, camrig_file)
    _camera_config_note(camrig_file, sorted(covered), camera_config_resolver)
    print(f"Camera intrinsics: using {camrig_file}")
    if camera_model is not None:
        camera = _refit_camera(camera, camera_model, camera_index, camrig_file)
    return CamrigCamera(camrig_file=camrig_file, camera=camera, images=covered)


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
        rig=CamrigRig(
            camrig_file=camrig_file, data=data, assignments=solve_assignments
        ),
    )


def find_covering_camrigs(
    image_paths,
    workspace_dir,
    images: str = "images being solved",
    act: str = "solve",
) -> list[tuple[Path, dict, set[Path]]]:
    """Find the `.camrig` files that together cover every one of `image_paths`.

    Scans `workspace_dir` for `.camrig` files and returns, in path order, the
    path, the `read_camrig` dict and the covered images (resolved paths) of
    each one whose patterns match any of `image_paths`; the list is empty when
    no `.camrig` matches any of them. `images` names the images in a message
    ("images being solved") and `act` is the verb that advises what to do
    instead ("solve").

    Raises `CamrigSolveError` when an image is matched by more than one
    `.camrig`, when a multi-sensor `.camrig` covers some of the images and
    another `.camrig` covers others, or when the `.camrig` files cover only
    some of the images: `sfm solve` takes the cameras of all of its images
    from `.camrig` files, or of none.
    """
    if workspace_dir is None:
        return []
    workspace_dir = Path(workspace_dir).resolve()
    camrig_files = sorted(workspace_dir.rglob("*.camrig"))
    if not camrig_files:
        return []

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
        return []

    # Each image takes its camera from one .camrig.
    covered_by: dict[Path, list[Path]] = {}
    for camrig_file, _data, covered in hits:
        for path in covered:
            covered_by.setdefault(path, []).append(camrig_file)
    doubly = sorted(p for p, files in covered_by.items() if len(files) > 1)
    if doubly:
        example = doubly[0]
        names = " and ".join(str(f) for f in covered_by[example])
        raise CamrigSolveError(
            f"{len(doubly)} of the {len(given)} {images} are matched by more "
            f"than one .camrig; e.g. {example} is matched by {names}. Each "
            "image must be matched by one .camrig only."
        )

    rigs = [c for c, data, _ in hits if data["metadata"]["sensor_count"] > 1]
    if rigs and len(hits) > 1:
        names = ", ".join(str(c) for c, _, _ in hits)
        raise CamrigSolveError(
            f"the {images} are covered by multiple .camrig files ({names}), "
            f"and {rigs[0]} is a multi-sensor rig, which must cover all of "
            f"them alone; {act} the rig's images separately."
        )

    uncovered = given - covered_by.keys()
    if uncovered:
        example = sorted(uncovered)[0]
        if len(hits) == 1:
            camrig_file, data, covered = hits[0]
            by = "any sensor pattern" if rigs else "its pattern"
            raise CamrigSolveError(
                f"{camrig_file} matches only {len(covered)} of {len(given)} "
                f"{images}; e.g. {example} is not matched by {by}. "
                f"{act.capitalize()} only the images the .camrig covers, or "
                "remove the .camrig."
            )
        names = ", ".join(str(c) for c, _, _ in hits)
        raise CamrigSolveError(
            f"the .camrig files {names} match only {len(covered_by)} of "
            f"{len(given)} {images}; e.g. {example} is not matched by any of "
            f"their patterns. {act.capitalize()} only the images the .camrig "
            "files cover, or add a .camrig for the others."
        )
    return hits


def resolve_camrig_for_solve(
    image_paths,
    workspace_dir,
    camera_model,
    camera_config_resolver=None,
) -> CamrigSolveResult | None:
    """Find the `.camrig` files covering every image being solved.

    Scans `workspace_dir` for `.camrig` files and returns a `CamrigSolveResult`
    for the ones that together cover all of `image_paths`
    (`find_covering_camrigs`), or `None` when no `.camrig` applies.
    Single-sensor `.camrig` files yield `.cameras`, one per file; a
    multi-sensor `.camrig` yields a `.rig`.

    With `camera_model` (the solve's `--camera-model`), each `.camrig` camera
    the solve uses is refit to that model, and the result carries the refit
    cameras in place of the files'.

    Raises `CamrigSolveError` when the discovered `.camrig` files cannot be
    used: an image is matched by more than one `.camrig`, a multi-sensor
    `.camrig` comes with another `.camrig`, the `.camrig` files cover only
    some of the images, or a camera's refit to `camera_model` is refused.
    """
    hits = find_covering_camrigs(image_paths, workspace_dir)
    if not hits:
        return None
    camrig_file, data, covered = hits[0]
    if data["metadata"]["sensor_count"] > 1:
        return _resolve_multi_sensor(
            camrig_file, data, covered, camera_model, camera_config_resolver
        )
    return CamrigSolveResult(
        cameras=[
            _resolve_single_sensor(
                camrig_file, data, covered, camera_model, camera_config_resolver
            )
            for camrig_file, data, covered in hits
        ]
    )
