# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Build a `.camrig` file from one rig of a workspace's `rig_config.json`.

Backs `sfm camrig cp RIG_CONFIG.json OUTPUT.camrig [--rig N]`, the one
reader of `rig_config.json`. The `.camrig` gives each sensor the images whose
paths start with its `image_prefix`, groups them into the frames the file's
frame keys define, and holds the cameras and the `sensor_from_rig` poses the
file describes.

- **Images.** A sensor's images are the image files under the folder of the
  `rig_config.json` whose workspace-relative path starts with its
  `image_prefix`, an image taking the first sensor in the file whose prefix it
  starts with.
- **Patterns.** Each sensor's images must form one sequence of
  `sfm pick-images` (`collect_sequences`): one folder, names that differ only
  in their file number. The sequence's pattern, relative to the output's
  folder, is the sensor's image pattern, with a frame field for the number.
- **Frames.** `rig_config.json` puts two images in one frame when the paths
  after their prefixes are equal; the `.camrig` when their frame fields
  capture the same number. The derived patterns are matched against the files
  on disk, as `sfm solve` matches them, and the conversion is refused unless
  they give every sensor exactly its images and group them into exactly the
  same frames.
- **Sensors.** The reference sensor is sensor 0, the others follow in the
  file's order.
- **Poses.** `rig_config.json` holds COLMAP-convention `cam_from_rig` poses,
  so each is conjugated by the camera-frame flip `S` into the canonical
  `sensor_from_rig` the `.camrig` holds, its quaternion made unit; the
  reference sensor is the identity.
- **Cameras.** A sensor's camera is built on the sensor's first image
  (`_camera_from_sensor_entry`): its `camera_params`, or EXIF inference with
  its `camera_model_name` as a hint when it has none. Equal cameras share one
  entry of the pool.
"""

import json
import os
from pathlib import Path

import numpy as np

from .._filenames import has_extension
from .cp import CamrigCpError, _write

# The image files `sfm solve` and `sfm pick-images` take from a directory.
_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")

# How many file names a message lists before it gives the rest as a count.
_NAMES_SHOWN = 10


def _names(names: list[str]) -> str:
    shown = ", ".join(names[:_NAMES_SHOWN])
    if len(names) > _NAMES_SHOWN:
        shown += f" and {len(names) - _NAMES_SHOWN} more"
    return shown


def _sensor_images(root: Path, rig_configs: list[dict]) -> dict[tuple, list[str]]:
    """The workspace-relative paths of the image files of each `(rig, sensor)`
    of `rig_configs`, sorted, assigned by `_match_image_to_sensor`."""
    from ..rig.config import _match_image_to_sensor

    candidates: set[str] = set()
    for rig in rig_configs:
        for sensor in rig["cameras"]:
            folder = root / sensor["image_prefix"].rpartition("/")[0]
            if not folder.is_dir():
                continue
            for path in folder.rglob("*"):
                if path.is_file() and has_extension(path, _IMAGE_EXTENSIONS):
                    candidates.add(path.relative_to(root).as_posix())
    images: dict[tuple, list[str]] = {}
    for rel in sorted(candidates):
        match = _match_image_to_sensor(rel, rig_configs)
        if match is not None:
            images.setdefault(match, []).append(rel)
    return images


def _camera_from_sensor_entry(sensor: dict, image_path: Path):
    """The `pycolmap.Camera` of a validated `rig_config.json` sensor entry,
    at the size of `image_path`.

    With `camera_params`, the camera is the parameters for its
    `camera_model_name`, flagged with a prior focal length; without them, it
    is inferred from the image's EXIF with `camera_model_name`, if any, as the
    model (`_infer_camera`, which seeds a fisheye focal length). The file
    carries no image size, so the size is always the image's.
    """
    from ..camera.setup import _infer_camera

    model_name = sensor.get("camera_model_name")
    camera = _infer_camera(str(image_path), model_name)
    camera_params = sensor.get("camera_params")
    if camera_params is not None:
        camera.params = [float(p) for p in camera_params]
        camera.has_prior_focal_length = True
    return camera


def _sensor_pattern(rel_images: list[str], root: Path, out_dir: Path, prefix: str):
    """The pattern, relative to `out_dir`, of the one sequence the images of
    the sensor with `prefix` form."""
    from .._pick_images import PickImagesInputError, collect_sequences
    from .._sfmtool.fileio import validate_camrig_pattern

    paths = [root / rel for rel in rel_images]
    try:
        sequences = collect_sequences(paths, out_dir)
    except PickImagesInputError as e:
        raise CamrigCpError(
            f"the images under {prefix!r} do not form one numbered sequence:\n{e}"
        ) from None
    if len(sequences) != 1:
        listing = "; ".join(
            f"{s.name} ({len(s.numbers)} images)" for s in sequences[:_NAMES_SHOWN]
        )
        raise CamrigCpError(
            f"the images under {prefix!r} form {len(sequences)} sequences, and a "
            f".camrig sensor has one image pattern: {listing}"
        )
    pattern = sequences[0].name
    try:
        validate_camrig_pattern(pattern)
    except ValueError as e:
        raise CamrigCpError(
            f"the images under {prefix!r} give the pattern {pattern!r}, which a "
            f".camrig cannot hold: {e}"
        ) from None
    return pattern


def _sensor_from_rig(sensor: dict) -> tuple[np.ndarray, np.ndarray]:
    """The canonical `sensor_from_rig` of a `rig_config.json` sensor entry as
    a WXYZ quaternion and a translation. The file's quaternion need not be
    unit; the conjugation returns a unit one."""
    from ..colmap.convention import relative_pose_conjugate_s

    if sensor.get("ref_sensor", False):
        return np.array([1.0, 0.0, 0.0, 0.0]), np.zeros(3)
    q = np.asarray(sensor["cam_from_rig_rotation"], dtype=np.float64)
    t = np.asarray(sensor.get("cam_from_rig_translation", [0.0, 0.0, 0.0]), float)
    q_canonical, t_canonical = relative_pose_conjugate_s(q, t)
    return np.asarray(q_canonical, np.float64), np.asarray(t_canonical, np.float64)


def _check_frames(
    output: Path,
    patterns: list[str],
    sensor_images: list[list[Path]],
    frame_keys: dict[Path, str],
) -> dict[Path, tuple[int, int]]:
    """Match `patterns` from `output`'s folder as `sfm solve` does and refuse
    unless they give sensor `i` exactly `sensor_images[i]` and put two images
    in one frame exactly when their `frame_keys` are equal. Returns the
    `path -> (sensor, frame)` assignments."""
    from .resolver import CamrigSolveError, _build_rig_assignments

    try:
        assignments = _build_rig_assignments(
            output, {"sensor_image_patterns": patterns}
        )
    except (CamrigSolveError, ValueError) as e:
        raise CamrigCpError(f"the derived image patterns do not work: {e}") from None

    out_dir = output.parent

    def rel(path: Path) -> str:
        return Path(os.path.relpath(path, out_dir)).as_posix()

    problems = []
    for sensor, images in enumerate(sensor_images):
        expected = set(images)
        matched = {p for p, (s, _f) in assignments.items() if s == sensor}
        extra = sorted(rel(p) for p in matched - expected)
        missing = sorted(rel(p) for p in expected - matched)
        if extra:
            problems.append(
                f"the pattern {patterns[sensor]!r} also matches {len(extra)} "
                f"file(s) that rig_config.json does not give this sensor: "
                f"{_names(extra)}"
            )
        if missing:
            problems.append(
                f"the pattern {patterns[sensor]!r} does not match {len(missing)} "
                f"image(s) of its sensor: {_names(missing)}"
            )
    if problems:
        raise CamrigCpError("\n".join(problems))

    frames_of_key: dict[str, set[int]] = {}
    keys_of_frame: dict[int, set[str]] = {}
    for path, key in frame_keys.items():
        frame = assignments[path][1]
        frames_of_key.setdefault(key, set()).add(frame)
        keys_of_frame.setdefault(frame, set()).add(key)
    split = [
        f"{key!r} (frames {sorted(frames)})"
        for key, frames in sorted(frames_of_key.items())
        if len(frames) > 1
    ]
    joined = [
        f"frame {frame} ({', '.join(repr(k) for k in sorted(keys))})"
        for frame, keys in sorted(keys_of_frame.items())
        if len(keys) > 1
    ]
    if split:
        problems.append(
            f"{len(split)} frame key(s) of rig_config.json get more than one "
            f"frame number from the patterns: {_names(split)}"
        )
    if joined:
        problems.append(
            f"{len(joined)} frame number(s) of the patterns join images that "
            f"rig_config.json puts in different frames: {_names(joined)}"
        )
    if problems:
        raise CamrigCpError("\n".join(problems))
    return assignments


def _covering_camrigs(output: Path, root: Path, images: set[Path]) -> list[Path]:
    """The `.camrig` files under `root`, other than `output`, whose patterns
    match any of `images`.

    A file whose patterns miss an image only by the zero-padding of its number
    does not cover it, so it is not named: `sfm solve` refuses an image two
    `.camrig` files cover, and reports a padding miss only for an image no
    `.camrig` covers, which `output` now does."""
    from .._sfmtool.fileio import read_camrig
    from .resolver import _covered_paths

    found = []
    for camrig_file in sorted(root.rglob("*.camrig")):
        if camrig_file.resolve() == output.resolve():
            continue
        try:
            patterns = read_camrig(str(camrig_file))["sensor_image_patterns"]
        except Exception:
            continue
        covered, _padding_missed = _covered_paths(camrig_file.parent, patterns)
        if covered & images:
            found.append(camrig_file)
    return found


def copy_from_rig_config(
    source: Path,
    output: Path,
    *,
    rig_index: int | None,
    name: str | None,
    zstd_level: int = 3,
) -> dict:
    """Build a `.camrig` from rig `rig_index` of the `rig_config.json` at
    `source`, the default being its only rig, and write it to `output`.

    The prefixes of `source` are relative to its folder, so `output` must be
    in that folder or under it, and the folder of `output`, the rig root, must
    hold every image of the rig. Raises `CamrigCpError` when the file cannot
    be read, a sensor has no images, a sensor's images do not form one
    numbered sequence or have mixed resolutions, or the derived patterns do
    not group the images into the same sensors and frames as the file does.

    Returns a summary dict: the counts, and per sensor (in output order) its
    `image_prefix`, pattern, image count and camera, whether that camera was
    inferred from EXIF (no `camera_params`), and the other `.camrig` files that
    already cover some of the rig's images.
    """
    from ..camera.cameras import pycolmap_camera_to_intrinsics
    from ..rig.config import RigConfigError, _infer_frame_key, validate_rig_config
    from .resolver import CamrigSolveError, _check_image_resolution

    source = Path(source).resolve()
    output = Path(os.path.abspath(output))
    root = source.parent
    try:
        data = json.loads(source.read_text())
        rig_configs = validate_rig_config(data, str(source))
    except json.JSONDecodeError as e:
        raise CamrigCpError(f"Invalid JSON in {source}: {e}") from None
    except OSError as e:
        raise CamrigCpError(str(e)) from None
    except RigConfigError as e:
        raise CamrigCpError(
            f"a .json source is read as a rig_config.json, and this one is not one: {e}"
        ) from None

    if rig_index is None:
        if len(rig_configs) != 1:
            raise CamrigCpError(
                f"{source} has {len(rig_configs)} rigs; pass --rig N "
                f"(0..{len(rig_configs) - 1}) to choose one."
            )
        rig_index = 0
    if rig_index < 0 or rig_index >= len(rig_configs):
        raise CamrigCpError(
            f"--rig {rig_index} is out of range; {source} has {len(rig_configs)} "
            f"rig(s) (0..{len(rig_configs) - 1})."
        )

    out_dir = output.parent.resolve()
    if out_dir != root and root not in out_dir.parents:
        raise CamrigCpError(
            f"{output} is outside {root}, the folder the image prefixes of "
            f"{source.name} are relative to; write the .camrig in that folder "
            "or under it, so that `sfm solve` finds it."
        )
    output = out_dir / output.name

    sensors = rig_configs[rig_index]["cameras"]
    # The reference sensor first, then the others in the file's order.
    order = sorted(
        range(len(sensors)), key=lambda s: (not sensors[s].get("ref_sensor"), s)
    )
    images_of = _sensor_images(root, rig_configs)

    empty = [
        sensors[s]["image_prefix"] for s in order if (rig_index, s) not in images_of
    ]
    if empty:
        raise CamrigCpError(
            f"no image files under {root} start with the prefix(es) "
            f"{', '.join(repr(p) for p in empty)} of rig {rig_index}, so there "
            "is nothing to derive their image patterns and cameras from."
        )
    outside = [
        rel
        for s in order
        for rel in images_of[(rig_index, s)]
        if out_dir not in (root / rel).resolve().parents
    ]
    if outside:
        raise CamrigCpError(
            f"{len(outside)} image(s) of rig {rig_index} are not under "
            f"{out_dir}, the folder of the .camrig and the root its patterns are "
            f"relative to: {_names(outside)}"
        )

    patterns: list[str] = []
    sensor_images: list[list[Path]] = []
    frame_keys: dict[Path, str] = {}
    cameras: list[dict] = []
    inferred: list[int] = []
    quats = np.zeros((len(order), 4))
    trans = np.zeros((len(order), 3))
    for out_index, s in enumerate(order):
        sensor = sensors[s]
        prefix = sensor["image_prefix"]
        rel_images = images_of[(rig_index, s)]
        patterns.append(_sensor_pattern(rel_images, root, out_dir, prefix))
        paths = [(root / rel).resolve() for rel in rel_images]
        sensor_images.append(paths)
        for rel, path in zip(rel_images, paths):
            frame_keys[path] = _infer_frame_key(rel, prefix)

        try:
            camera = _camera_from_sensor_entry(sensor, paths[0])
        except Exception as e:
            raise CamrigCpError(
                f"could not build the camera of the sensor {prefix!r} from "
                f"{rel_images[0]}: {e}"
            ) from None
        camera_dict = pycolmap_camera_to_intrinsics(camera).to_dict()
        try:
            _check_image_resolution(paths, camera_dict, source, f" (sensor {prefix!r})")
        except CamrigSolveError as e:
            raise CamrigCpError(str(e)) from None
        cameras.append(camera_dict)
        if sensor.get("camera_params") is None:
            inferred.append(out_index)
        quats[out_index], trans[out_index] = _sensor_from_rig(sensor)

    assignments = _check_frames(output, patterns, sensor_images, frame_keys)

    # Equal cameras share one pool entry, in first-seen order.
    pool: list[dict] = []
    camera_indexes: list[int] = []
    for camera in cameras:
        if camera not in pool:
            pool.append(camera)
        camera_indexes.append(pool.index(camera))

    _write(
        output,
        name=name or output.stem,
        rig_type="generic",
        cameras=pool,
        patterns=patterns,
        camera_indexes=camera_indexes,
        quats=quats,
        trans=trans,
        zstd_level=zstd_level,
    )
    all_images = {p for paths in sensor_images for p in paths}
    return {
        "output_file": output,
        "kind": "rig",
        "rig_index": rig_index,
        "rig_count": len(rig_configs),
        "sensor_count": len(order),
        "camera_count": len(pool),
        "frame_count": len({frame for _s, frame in assignments.values()}),
        "image_backed": True,
        "rig_type": "generic",
        "sensors": [
            {
                "image_prefix": sensors[s]["image_prefix"],
                "pattern": patterns[i],
                "image_count": len(sensor_images[i]),
                "camera": cameras[i],
                "camera_inferred_from": (
                    Path(os.path.relpath(sensor_images[i][0], root)).as_posix()
                    if i in inferred
                    else None
                ),
            }
            for i, s in enumerate(order)
        ],
        "covering_camrigs": _covering_camrigs(output, root, all_images),
    }
