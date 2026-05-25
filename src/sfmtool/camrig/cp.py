# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Build a `.camrig` file by copying from an existing `.sfmr` or `.camrig`.

Backs `sfm camrig cp`. Where `sfm camrig create` builds a rig from a directory
of images on disk, `cp` builds one from a file that already carries cameras
(and, for a rig, sensor poses): a solved `.sfmr` reconstruction or another
`.camrig` file.

A *selector* chooses which slice of the source becomes the output rig:

- ``--rig`` (`.sfmr`) copies a whole rig — its sensors, cameras, and
  ``sensor_from_rig`` poses;
- ``--camera`` (`.sfmr` or `.camrig`) copies one camera as a single-sensor
  ``generic`` rig at the identity pose;
- ``--sensors`` (`.camrig`) copies a subset of sensors as a smaller rig.

The output is always a `.camrig`. A `.camrig` is located in a workspace by its
per-sensor image patterns, so `cp` fills them in: a `.camrig` source's patterns
carry over verbatim, and for a `.sfmr` source each sensor's pattern is inferred
from its images' file names. A multi-sensor rig needs a frame field in every
pattern; when inference cannot supply one for every sensor the rig is written
geometry-only (no patterns).
"""

from pathlib import Path

import numpy as np

# A lone sensor sits at the identity `sensor_from_rig` pose.
_IDENTITY_QUATS = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float64)
_IDENTITY_TRANS = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)


class CamrigCpError(Exception):
    """Raised when a `.camrig` cannot be copied — bad input or bad selection."""


def _infer_pattern(image_names: list[str], *, require_frame_field: bool) -> str | None:
    """Infer one `.camrig` image pattern from a list of image names.

    Uses the same path-sequence summariser the rest of the CLI relies on.
    Returns the pattern, or `None` when the names do not collapse to a single
    pattern — or, with `require_frame_field`, a single *numbered* sequence, as
    a multi-sensor rig needs a frame field in every pattern.
    """
    from deadline.job_attachments.api import summarize_paths_by_sequence

    from .._sfmtool import validate_camrig_pattern

    if not image_names:
        return None
    summaries = summarize_paths_by_sequence(list(image_names))
    if len(summaries) != 1:
        return None
    summary = summaries[0]
    if require_frame_field and not summary.index_set:
        return None
    # summarize_paths_by_sequence may emit OS separators; .camrig patterns are
    # always forward-slash relative paths.
    pattern = summary.path.replace("\\", "/")
    try:
        validate_camrig_pattern(pattern)
    except ValueError:
        return None
    return pattern


def _normalize_pattern(pattern: str) -> str:
    """Validate an explicit `--pattern` and return it in forward-slash form."""
    from .._sfmtool import validate_camrig_pattern

    normalized = pattern.replace("\\", "/").strip()
    try:
        validate_camrig_pattern(normalized)
    except ValueError as e:
        raise CamrigCpError(f"invalid --pattern: {e}") from None
    return normalized


def _compact_pool(
    all_cameras: list[dict], used_indexes: list[int]
) -> tuple[list[dict], list[int]]:
    """Reduce a camera pool to the cameras actually used.

    Returns `(pool, remapped)`: `pool` holds the distinct cameras referenced by
    `used_indexes` in first-seen order, and `remapped[i]` is the new pool index
    for the sensor whose source camera index was `used_indexes[i]`.
    """
    pool: list[dict] = []
    remap: dict[int, int] = {}
    remapped: list[int] = []
    for ci in used_indexes:
        if ci not in remap:
            remap[ci] = len(pool)
            pool.append(all_cameras[ci])
        remapped.append(remap[ci])
    return pool, remapped


def _write(
    output: Path,
    *,
    name: str,
    rig_type: str,
    cameras: list[dict],
    patterns: list[str],
    camera_indexes: list[int],
    quats: np.ndarray,
    trans: np.ndarray,
    rig_attributes: dict | None = None,
    zstd_level: int = 3,
) -> None:
    """Write a `.camrig` file from prepared columnar rig data."""
    from .._sfmtool import write_camrig

    write_camrig(
        path=str(output),
        name=name,
        rig_type=rig_type,
        cameras=cameras,
        sensor_image_patterns=patterns,
        camera_indexes=[int(c) for c in camera_indexes],
        quaternions_wxyz=np.ascontiguousarray(quats, dtype=np.float64),
        translations_xyz=np.ascontiguousarray(trans, dtype=np.float64),
        rig_attributes=rig_attributes,
        zstd_level=zstd_level,
    )


def _parse_sensor_range(expr: str, sensor_count: int, source: Path) -> list[int]:
    """Parse a `--sensors` range expression into sorted, validated indices."""
    from .._sfmtool import RangeExpr

    try:
        selected = sorted(set(RangeExpr(expr)))
    except Exception as e:
        raise CamrigCpError(f"invalid --sensors range {expr!r}: {e}") from None
    if not selected:
        raise CamrigCpError(f"--sensors {expr!r} selects no sensors.")
    bad = [s for s in selected if s < 0 or s >= sensor_count]
    if bad:
        raise CamrigCpError(
            f"--sensors {expr!r} includes sensor(s) {bad} outside the valid "
            f"range 0..{sensor_count - 1} ({source} has {sensor_count} sensors)."
        )
    return selected


def _single_camera_camrig(
    output: Path,
    *,
    cameras: list[dict],
    camera_index: int,
    image_names: list[str],
    owner_indexes,
    pattern: str | None,
    name: str | None,
    zstd_level: int,
    source_label: str,
) -> dict:
    """Write a single-sensor `.camrig` for one camera of a `.sfmr` source."""
    if camera_index < 0 or camera_index >= len(cameras):
        raise CamrigCpError(
            f"--camera {camera_index} is out of range; {source_label} has "
            f"{len(cameras)} camera(s) (0..{len(cameras) - 1})."
        )
    camera = cameras[camera_index]
    if pattern is None:
        owned = [
            image_names[i]
            for i in range(len(image_names))
            if int(owner_indexes[i]) == camera_index
        ]
        inferred = _infer_pattern(owned, require_frame_field=False)
        if inferred is None:
            raise CamrigCpError(
                f"could not infer an image pattern for camera {camera_index} "
                f"from {source_label}; pass --pattern to set one explicitly."
            )
        patterns = [inferred]
    else:
        patterns = [_normalize_pattern(pattern)]

    _write(
        output,
        name=name or output.stem,
        rig_type="generic",
        cameras=[camera],
        patterns=patterns,
        camera_indexes=[0],
        quats=_IDENTITY_QUATS,
        trans=_IDENTITY_TRANS,
        zstd_level=zstd_level,
    )
    return {
        "output_file": output,
        "kind": "camera",
        "sensor_count": 1,
        "camera_count": 1,
        "image_backed": True,
        "rig_type": "generic",
    }


def _rig_camrig_from_sfmr(
    output: Path,
    recon,
    cameras: list[dict],
    rfd: dict,
    rig_index: int,
    name: str | None,
    zstd_level: int,
) -> dict:
    """Write a multi-sensor `.camrig` for rig `rig_index` of a `.sfmr` source."""
    rig = rfd["rigs_metadata"]["rigs"][rig_index]
    offset = int(rig["sensor_offset"])
    scount = int(rig["sensor_count"])
    global_sensors = list(range(offset, offset + scount))

    sensor_camera = np.asarray(rfd["sensor_camera_indexes"])
    sensor_q = np.asarray(rfd["sensor_quaternions_wxyz"], dtype=np.float64)
    sensor_t = np.asarray(rfd["sensor_translations_xyz"], dtype=np.float64)
    image_sensor = np.asarray(rfd["image_sensor_indexes"])
    image_names = list(recon.image_names)

    used = [int(sensor_camera[s]) for s in global_sensors]
    pool, camera_indexes = _compact_pool(cameras, used)

    quats = sensor_q[global_sensors]
    trans = sensor_t[global_sensors]

    # A multi-sensor rig pairs frames across sensors by frame index, so every
    # pattern must carry a frame field; a one-sensor rig does not need one.
    require_frame_field = scount > 1
    patterns: list[str | None] = []
    inferable = True
    for s in global_sensors:
        owned = [
            image_names[i] for i in range(len(image_names)) if int(image_sensor[i]) == s
        ]
        p = _infer_pattern(owned, require_frame_field=require_frame_field)
        if p is None:
            inferable = False
        patterns.append(p)

    if inferable:
        sensor_patterns: list[str] = [p for p in patterns if p is not None]
        image_backed = True
    else:
        sensor_patterns = []
        image_backed = False

    rig_name = name or (rig["name"] or output.stem)
    _write(
        output,
        name=rig_name,
        rig_type="generic",
        cameras=pool,
        patterns=sensor_patterns,
        camera_indexes=camera_indexes,
        quats=quats,
        trans=trans,
        zstd_level=zstd_level,
    )
    return {
        "output_file": output,
        "kind": "rig",
        "sensor_count": scount,
        "camera_count": len(pool),
        "image_backed": image_backed,
        "rig_type": "generic",
    }


def copy_from_sfmr(
    source: Path,
    output: Path,
    *,
    rig_index: int | None,
    camera_index: int | None,
    pattern: str | None,
    name: str | None,
    zstd_level: int = 3,
) -> dict:
    """Build a `.camrig` from a `.sfmr` reconstruction.

    `camera_index` copies one pool camera as a single-sensor rig; `rig_index`
    copies a whole rig. With neither set the selector defaults to the lone rig,
    or the lone camera of a rig-less reconstruction. Returns a summary dict.
    """
    from .._sfmtool import SfmrReconstruction

    source = Path(source)
    output = Path(output)
    try:
        recon = SfmrReconstruction.load(source)
    except Exception as e:
        raise CamrigCpError(f"could not load {source}: {e}") from None

    cameras = [c.to_dict() for c in recon.cameras]
    if not cameras:
        raise CamrigCpError(f"{source} contains no cameras — nothing to copy.")
    rfd = recon.rig_frame_data

    if camera_index is not None:
        return _single_camera_camrig(
            output,
            cameras=cameras,
            camera_index=camera_index,
            image_names=list(recon.image_names),
            owner_indexes=recon.camera_indexes,
            pattern=pattern,
            name=name,
            zstd_level=zstd_level,
            source_label=str(source),
        )

    if rfd is None:
        if rig_index not in (None, 0):
            raise CamrigCpError(
                f"--rig {rig_index} is out of range; {source} has no rig data."
            )
        if rig_index == 0:
            raise CamrigCpError(
                f"{source} has no rig data; --rig is not applicable. "
                "Use --camera to copy a single camera."
            )
        if len(cameras) == 1:
            return _single_camera_camrig(
                output,
                cameras=cameras,
                camera_index=0,
                image_names=list(recon.image_names),
                owner_indexes=recon.camera_indexes,
                pattern=pattern,
                name=name,
                zstd_level=zstd_level,
                source_label=str(source),
            )
        raise CamrigCpError(
            f"{source} has no rig data and {len(cameras)} cameras; pass "
            f"--camera N (0..{len(cameras) - 1}) to choose one."
        )

    rigs = rfd["rigs_metadata"]["rigs"]
    if rig_index is None:
        if len(rigs) == 1:
            rig_index = 0
        else:
            raise CamrigCpError(
                f"{source} has {len(rigs)} rigs; pass --rig N "
                f"(0..{len(rigs) - 1}) to choose one."
            )
    if rig_index < 0 or rig_index >= len(rigs):
        raise CamrigCpError(
            f"--rig {rig_index} is out of range; {source} has {len(rigs)} "
            f"rig(s) (0..{len(rigs) - 1})."
        )
    return _rig_camrig_from_sfmr(
        output, recon, cameras, rfd, rig_index, name, zstd_level
    )


def copy_from_camrig(
    source: Path,
    output: Path,
    *,
    sensors_expr: str | None,
    camera_index: int | None,
    pattern: str | None,
    name: str | None,
    zstd_level: int = 3,
) -> dict:
    """Build a `.camrig` from another `.camrig` file.

    `camera_index` copies one pool camera as a single-sensor rig; `sensors_expr`
    copies a subset of sensors. With neither set the whole rig is copied,
    preserving its `rig_type` and `rig_attributes`. Returns a summary dict.
    """
    from .._sfmtool import read_camrig

    source = Path(source)
    output = Path(output)
    try:
        data = read_camrig(str(source))
    except Exception as e:
        raise CamrigCpError(f"could not read {source}: {e}") from None

    cameras = data["cameras"]
    cam_indexes = [int(c) for c in data["camera_indexes"]]
    quats = np.asarray(data["quaternions_wxyz"], dtype=np.float64)
    trans = np.asarray(data["translations_xyz"], dtype=np.float64)
    src_patterns = list(data["sensor_image_patterns"])
    meta = data["metadata"]
    sensor_count = int(meta["sensor_count"])

    if camera_index is not None:
        if camera_index < 0 or camera_index >= len(cameras):
            raise CamrigCpError(
                f"--camera {camera_index} is out of range; {source} has "
                f"{len(cameras)} camera(s) (0..{len(cameras) - 1})."
            )
        if pattern is not None:
            patterns = [_normalize_pattern(pattern)]
        else:
            owners = [i for i in range(sensor_count) if cam_indexes[i] == camera_index]
            # Carry a pattern across only when one sensor unambiguously owns
            # this camera; otherwise the lone sensor is geometry-only.
            if src_patterns and len(owners) == 1:
                patterns = [src_patterns[owners[0]]]
            else:
                patterns = []
        _write(
            output,
            name=name or output.stem,
            rig_type="generic",
            cameras=[cameras[camera_index]],
            patterns=patterns,
            camera_indexes=[0],
            quats=_IDENTITY_QUATS,
            trans=_IDENTITY_TRANS,
            zstd_level=zstd_level,
        )
        return {
            "output_file": output,
            "kind": "camera",
            "sensor_count": 1,
            "camera_count": 1,
            "image_backed": bool(patterns),
            "rig_type": "generic",
        }

    if sensors_expr is None:
        selected = list(range(sensor_count))
    else:
        selected = _parse_sensor_range(sensors_expr, sensor_count, source)

    used = [cam_indexes[i] for i in selected]
    pool, camera_indexes = _compact_pool(cameras, used)
    sel_quats = quats[selected]
    sel_trans = trans[selected]
    sensor_patterns = [src_patterns[i] for i in selected] if src_patterns else []

    # A subset of a typed rig is no longer that type; only a whole-rig copy
    # keeps the source's rig_type and rig_attributes.
    whole_rig = selected == list(range(sensor_count))
    if whole_rig:
        rig_type = meta["rig_type"]
        rig_attributes = meta["rig_attributes"]
        rig_name = name or (meta["name"] or output.stem)
    else:
        rig_type = "generic"
        rig_attributes = None
        rig_name = name or output.stem

    _write(
        output,
        name=rig_name,
        rig_type=rig_type,
        cameras=pool,
        patterns=sensor_patterns,
        camera_indexes=camera_indexes,
        quats=sel_quats,
        trans=sel_trans,
        rig_attributes=rig_attributes,
        zstd_level=zstd_level,
    )
    return {
        "output_file": output,
        "kind": "rig",
        "sensor_count": len(selected),
        "camera_count": len(pool),
        "image_backed": bool(sensor_patterns),
        "rig_type": rig_type,
    }
