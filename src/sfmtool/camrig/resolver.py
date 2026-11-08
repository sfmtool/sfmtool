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

Each image being solved takes its camera from the `.camrig` that covers it:
each single-sensor file is one camera of the solve, and each multi-sensor file
one rig, with its sensors, cameras and rig frames. The images no `.camrig`
covers take their cameras from `camera_config.json` or EXIF inference, as they
would with no `.camrig`, and are single-image frames beside the rig frames. An
image matched by more than one `.camrig` is refused. The `.camrig` files take
precedence over `camera_config.json` for the images they cover. With
`--camera-model`, each of their cameras the solve uses is refit to that model
and the refit camera is the starting intrinsics; the files are not changed.
`sfm match` and `sfm to-colmap-db` set up the cameras of their databases
through `resolve_camrig_for_solve` too, and `sfm pick-images` applies the same
rule through `find_covering_camrigs`, so the list it writes is one `sfm solve`
takes.
"""

from dataclasses import dataclass, field
from pathlib import Path

from .pattern import match_pattern_and_padding_misses, match_pattern_with_frames

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

    `cameras` holds one `CamrigCamera` per single-sensor `.camrig` covering
    some of the images and `rigs` one `CamrigRig` per multi-sensor `.camrig`
    covering some of them, each in path order; either list may be empty. The
    images neither holds take their cameras from `camera_config.json` or EXIF
    inference.
    """

    cameras: list[CamrigCamera] = field(default_factory=list)
    rigs: list[CamrigRig] = field(default_factory=list)

    @property
    def is_multi_sensor(self) -> bool:
        """Whether a multi-sensor `.camrig` covers any of the images."""
        return bool(self.rigs)


def _covered_paths(
    camrig_dir: Path, patterns: list[str]
) -> tuple[set[Path], dict[Path, str]]:
    """Resolved absolute paths the `.camrig`'s patterns match from its
    directory, and the paths a pattern misses only because their number is
    padded otherwise than its frame field writes it, each with that pattern."""
    covered: set[Path] = set()
    missed: dict[Path, str] = {}
    for pattern in patterns:
        matches, misses = match_pattern_and_padding_misses(camrig_dir, pattern)
        covered.update(matches)
        for path in misses:
            missed.setdefault(path, pattern)
    return covered, missed


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
    one sensor). The latter can happen with a wildcard beside the frame
    field, e.g. `*_%d.jpg` matching both `a_1.jpg` and `b_1.jpg`.
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
) -> CamrigRig:
    """Validate and resolve a multi-sensor `.camrig` covering `solve_set`."""
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
    return CamrigRig(camrig_file=camrig_file, data=data, assignments=solve_assignments)


def find_covering_camrigs(
    image_paths,
    workspace_dir,
    images: str = "images being solved",
) -> list[tuple[Path, dict, set[Path]]]:
    """Find the `.camrig` files that cover any of `image_paths`.

    Scans `workspace_dir` for `.camrig` files and returns, in path order, the
    path, the `read_camrig` dict and the covered images (resolved paths) of
    each one whose patterns match any of `image_paths`; the list is empty when
    no `.camrig` matches any of them. The files, single-sensor or
    multi-sensor, may leave some of the images uncovered; `sfm solve` takes
    those images' cameras from `camera_config.json` or EXIF inference.
    `images` names the images in a message ("images being solved").

    Raises `CamrigSolveError` when a `.camrig` cannot be read, an image is
    matched by more than one `.camrig`, or an image no `.camrig` matches is
    matched by a `.camrig`'s pattern apart from the zero-padding of its
    number, since its frame field matches a number only as printf writes it.
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
    # The images a .camrig's pattern misses only by the padding of their
    # number, each with the first such .camrig and pattern.
    padding_missed: dict[Path, tuple[Path, str]] = {}
    for camrig_file in camrig_files:
        try:
            data = read_camrig(str(camrig_file))
        except Exception as e:
            raise CamrigSolveError(f"could not read {camrig_file}: {e}") from None
        patterns = data["sensor_image_patterns"]
        if not patterns:
            continue  # geometry-only rig, not backed by workspace images
        covered, missed = _covered_paths(camrig_file.parent, patterns)
        covered &= given
        for path in missed.keys() & given:
            padding_missed.setdefault(path, (camrig_file, missed[path]))
        if covered:
            hits.append((camrig_file, data, covered))

    covered_any = set().union(*(covered for _f, _d, covered in hits))
    lost = sorted(padding_missed.keys() - covered_any)
    if lost:
        camrig_file, pattern = padding_missed[lost[0]]
        raise CamrigSolveError(
            f"{len(lost)} of the {len(given)} {images} are matched by no "
            f".camrig only because of the zero-padding of their number; e.g. "
            f"{lost[0]} is matched by the pattern {pattern!r} of {camrig_file} "
            "apart from its padding. A frame field matches a number only as "
            "printf writes it (%04d matches 0007 and 12345, not 007), so "
            "change the pattern or rename the files."
        )
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
    return hits


def _uncovered_note(
    uncovered: set[Path], total: int, camera_config_resolver, images: str
):
    """Print how many of the `images` no `.camrig` covers, with an example,
    and the source of their cameras."""
    from_config = 0
    if camera_config_resolver is not None:
        from_config = sum(
            camera_config_resolver.resolve_for_image(path) is not None
            for path in uncovered
        )
    from_exif = len(uncovered) - from_config
    if from_exif == 0:
        sources = "camera_config.json"
    elif from_config == 0:
        sources = "EXIF inference"
    else:
        sources = f"camera_config.json ({from_config}) and EXIF inference ({from_exif})"
    print(
        f"Camera intrinsics: {len(uncovered)} of {total} {images} "
        f"are covered by no .camrig, e.g. {sorted(uncovered)[0]}; they use "
        f"{sources}."
    )


def resolve_camrig_for_solve(
    image_paths,
    workspace_dir,
    camera_model,
    camera_config_resolver=None,
    images: str = "images being solved",
) -> CamrigSolveResult | None:
    """Find the `.camrig` files covering the images being solved.

    Scans `workspace_dir` for `.camrig` files and returns a `CamrigSolveResult`
    for the ones that cover any of `image_paths` (`find_covering_camrigs`), or
    `None` when no `.camrig` applies. Each single-sensor `.camrig` yields one
    of `.cameras` and each multi-sensor `.camrig` one of `.rigs`. When the
    files leave some images uncovered, a line names how many, with an example,
    and whether their cameras come from `camera_config.json` or EXIF
    inference; `images` names the images in that line and in the messages
    ("images being solved").

    With `camera_model` (the solve's `--camera-model`), each `.camrig` camera
    the solve uses is refit to that model, and the result carries the refit
    cameras in place of the files'.

    Raises `CamrigSolveError` when the discovered `.camrig` files cannot be
    used: an image is matched by more than one `.camrig` or by two sensors of
    one, two images of one sensor share a frame index, the images of a camera
    do not suit its resolution, or a camera's refit to `camera_model` is
    refused.
    """
    hits = find_covering_camrigs(image_paths, workspace_dir, images)
    if not hits:
        return None
    result = CamrigSolveResult()
    for camrig_file, data, covered in hits:
        if data["metadata"]["sensor_count"] > 1:
            result.rigs.append(
                _resolve_multi_sensor(
                    camrig_file, data, covered, camera_model, camera_config_resolver
                )
            )
        else:
            result.cameras.append(
                _resolve_single_sensor(
                    camrig_file, data, covered, camera_model, camera_config_resolver
                )
            )
    given = {Path(p).resolve() for p in image_paths}
    uncovered = given.difference(
        *(c.images for c in result.cameras), *(r.assignments for r in result.rigs)
    )
    if uncovered:
        _uncovered_note(uncovered, len(given), camera_config_resolver, images)
    return result
