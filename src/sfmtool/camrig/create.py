# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Build a single-camera `.camrig` file from a directory of images.

Backs `sfm camrig create`. Matches the images a `.camrig` image pattern
describes, derives one heuristic camera (pycolmap EXIF inference, optionally
overridden), and writes a one-sensor `.camrig` whose stored image pattern is
that pattern verbatim.

The rig is rejected — with a message explaining why — when the matched images
are not consistent with a single camera: mixed resolutions, mixed
pycolmap-inferred models, or focal lengths that vary too much. The caller can
then narrow the pattern and build separate `.camrig` files per subset.
"""

from collections import Counter
from pathlib import Path

from ..camera.cameras import _CAMERA_PARAM_NAMES
from .pattern import match_pattern

# Image file extensions a pattern is allowed to match.
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp")

# EXIF-inferred focal lengths may differ by at most this fraction across the
# matched images before the rig is rejected as inconsistent.
_FOCAL_TOLERANCE = 0.01


class CamrigCreateError(Exception):
    """Raised when a `.camrig` cannot be built — bad input or inconsistent images."""


def parse_resolution(text: str) -> tuple[int, int]:
    """Parse a ``WIDTHxHEIGHT`` resolution string into ``(width, height)``."""
    cleaned = text.lower().replace(" ", "")
    width_str, sep, height_str = cleaned.partition("x")
    if not sep:
        raise CamrigCreateError(
            f"--resolution must be WIDTHxHEIGHT, e.g. 4000x3000; got {text!r}"
        )
    try:
        width, height = int(width_str), int(height_str)
    except ValueError:
        raise CamrigCreateError(
            f"--resolution must be WIDTHxHEIGHT integers; got {text!r}"
        ) from None
    if width <= 0 or height <= 0:
        raise CamrigCreateError(
            f"--resolution dimensions must be positive; got {text!r}"
        )
    return width, height


def normalize_pattern(pattern: str) -> str:
    """Validate the `.camrig` image pattern and return it in forward-slash form.

    Backslashes are accepted on input and normalized to `/`. The rest of the
    grammar — a relative path with no leading `/` and no `..` component, at
    most one frame field (`%d`, `%0Nd`), and every `**` a whole path segment —
    is checked by the `camrig-format` crate via `validate_camrig_pattern`, the
    same rule the format's `validate()` enforces (see
    `specs/formats/camrig-file-format.md`).
    """
    from .._sfmtool.io import validate_camrig_pattern

    normalized = pattern.replace("\\", "/").strip()
    try:
        validate_camrig_pattern(normalized)
    except ValueError as e:
        raise CamrigCreateError(str(e)) from None
    return normalized


def find_images(rig_root: Path, pattern: str) -> list[str]:
    """Match `pattern` under `rig_root`, returning sorted POSIX-relative paths.

    `pattern` is interpreted with the shared `.camrig` pattern semantics, so a
    frame field matches digits only and `*` / `**` respect path-segment
    boundaries — the same matching `sfm solve` later applies to the stored
    pattern. Raises `CamrigCreateError` if nothing matches, or if a match is
    not an image file — the stored pattern must identify exactly the sensor's
    images.
    """
    rig_root = Path(rig_root).resolve()
    matches = match_pattern(rig_root, pattern)
    if not matches:
        raise CamrigCreateError(f"no files match pattern {pattern!r} under {rig_root}")
    non_images = [p for p in matches if p.suffix.lower() not in IMAGE_EXTENSIONS]
    if non_images:
        listing = "\n".join(f"  {p}" for p in non_images[:10])
        more = (
            "" if len(non_images) <= 10 else f"\n  ... and {len(non_images) - 10} more"
        )
        raise CamrigCreateError(
            f"pattern {pattern!r} matched {len(non_images)} non-image file(s):\n"
            f"{listing}{more}\n"
            "Narrow the pattern (e.g. add an extension like '*.jpg') so it "
            "matches only this sensor's images."
        )
    return [p.relative_to(rig_root).as_posix() for p in matches]


def _check_resolution_consistency(
    sizes: list[tuple[int, int]],
    rel_paths: list[str],
    target: tuple[int, int] | None,
) -> tuple[int, int]:
    """Return the `(width, height)` every image shares, or raise."""
    if target is not None:
        mismatched = [
            (rel_paths[i], sizes[i]) for i in range(len(sizes)) if sizes[i] != target
        ]
        if mismatched:
            listing = "\n".join(
                f"  {rel}  is {w}x{h}" for rel, (w, h) in mismatched[:10]
            )
            more = (
                ""
                if len(mismatched) <= 10
                else f"\n  ... and {len(mismatched) - 10} more"
            )
            raise CamrigCreateError(
                f"{len(mismatched)} image(s) do not match --resolution "
                f"{target[0]}x{target[1]}:\n{listing}{more}"
            )
        return target

    distinct = Counter(sizes)
    if len(distinct) > 1:
        lines = []
        for (w, h), count in distinct.most_common():
            example = next(
                rel_paths[i] for i in range(len(sizes)) if sizes[i] == (w, h)
            )
            lines.append(f"  {w}x{h}  ({count} image(s))  e.g. {example}")
        raise CamrigCreateError(
            "images matched by the pattern have inconsistent resolutions; a "
            ".camrig file describes one camera, so every image must share one "
            "resolution:\n" + "\n".join(lines) + "\n"
            "Create separate .camrig files per resolution, or narrow the "
            "pattern."
        )
    return sizes[0]


def _read_image_sizes(abs_paths: list[Path]) -> list[tuple[int, int]]:
    """Return `(width, height)` for each image file."""
    from ..camera.setup import _read_image_size

    return [_read_image_size(p) for p in abs_paths]


def _camera_from_explicit_params(
    camera_model: str,
    params_text: str,
    target_resolution: tuple[int, int] | None,
    abs_paths: list[Path],
    rel_paths: list[str],
) -> dict:
    """Build a camera dict from `--camera-model` plus a COLMAP-ordered list."""
    model = camera_model.upper()
    names = _CAMERA_PARAM_NAMES[model]
    try:
        values = [
            float(token) for token in params_text.split(",") if token.strip() != ""
        ]
    except ValueError:
        raise CamrigCreateError(
            f"--params must be a comma-separated list of numbers; got {params_text!r}"
        ) from None
    if len(values) != len(names):
        raise CamrigCreateError(
            f"--params for model {model} needs {len(names)} values "
            f"({', '.join(names)}); got {len(values)}."
        )
    sizes = _read_image_sizes(abs_paths)
    width, height = _check_resolution_consistency(sizes, rel_paths, target_resolution)
    return _validated_camera(model, width, height, dict(zip(names, values)))


def _camera_from_inference(
    abs_paths: list[Path],
    rel_paths: list[str],
    camera_model: str | None,
    target_resolution: tuple[int, int] | None,
    focal_length: float | None,
    focal_length_x: float | None,
    focal_length_y: float | None,
    principal_point_x: float | None,
    principal_point_y: float | None,
) -> dict:
    """Build a camera dict from pycolmap EXIF inference, applying overrides."""
    from ..camera.setup import _infer_camera
    from ..camera.cameras import pycolmap_camera_to_intrinsics

    inferred = [
        pycolmap_camera_to_intrinsics(_infer_camera(p, camera_model)) for p in abs_paths
    ]

    sizes = [(c.width, c.height) for c in inferred]
    width, height = _check_resolution_consistency(sizes, rel_paths, target_resolution)

    models = {c.model for c in inferred}
    if len(models) > 1:
        lines = []
        for m in sorted(models):
            example = next(
                rel_paths[i] for i in range(len(inferred)) if inferred[i].model == m
            )
            lines.append(f"  {m}  e.g. {example}")
        raise CamrigCreateError(
            "pycolmap inferred different camera models across the images:\n"
            + "\n".join(lines)
            + "\nPass --camera-model to force a single model for every image."
        )
    model = inferred[0].model

    focals = [c.focal_lengths[0] for c in inferred]
    f_min, f_max = min(focals), max(focals)
    if f_min > 0 and (f_max - f_min) / f_min > _FOCAL_TOLERANCE:
        i_min, i_max = focals.index(f_min), focals.index(f_max)
        raise CamrigCreateError(
            "EXIF-inferred focal lengths vary by more than "
            f"{_FOCAL_TOLERANCE * 100:.0f}% across the images — they look "
            "like different cameras or zoom settings:\n"
            f"  {rel_paths[i_min]}  f~{f_min:.1f}px\n"
            f"  {rel_paths[i_max]}  f~{f_max:.1f}px\n"
            "Use --camera-model with an explicit --focal-length / --params, "
            "or split the images into separate rigs."
        )

    parameters = dict(inferred[0].to_dict()["parameters"])
    param_names = _CAMERA_PARAM_NAMES[model]

    def override(name: str, value: float | None) -> None:
        if value is None:
            return
        if name not in param_names:
            flag = "--" + name.replace("_", "-")
            raise CamrigCreateError(
                f"{flag} is not a parameter of camera model {model}"
            )
        parameters[name] = value

    if focal_length is not None:
        if "focal_length" in param_names:
            parameters["focal_length"] = focal_length
        else:
            parameters["focal_length_x"] = focal_length
            parameters["focal_length_y"] = focal_length
    override("focal_length_x", focal_length_x)
    override("focal_length_y", focal_length_y)
    override("principal_point_x", principal_point_x)
    override("principal_point_y", principal_point_y)

    return _validated_camera(model, width, height, parameters)


def _validated_camera(model: str, width: int, height: int, parameters: dict) -> dict:
    """Build a `{model, width, height, parameters}` dict, validating it."""
    from .._sfmtool import CameraIntrinsics

    try:
        intrinsics = CameraIntrinsics.from_dict(
            {
                "model": model,
                "width": int(width),
                "height": int(height),
                "parameters": {k: float(v) for k, v in parameters.items()},
            }
        )
    except Exception as e:
        raise CamrigCreateError(f"invalid camera parameters: {e}") from None
    return intrinsics.to_dict()


def build_camrig_from_images(
    output_file: Path,
    image_pattern: str,
    *,
    camera_model: str | None = None,
    resolution: str | None = None,
    focal_length: float | None = None,
    focal_length_x: float | None = None,
    focal_length_y: float | None = None,
    principal_point_x: float | None = None,
    principal_point_y: float | None = None,
    params: str | None = None,
    name: str | None = None,
    zstd_level: int = 3,
) -> dict:
    """Build and write a one-sensor `.camrig` for a directory of images.

    `image_pattern` is a `.camrig` image pattern (globs and/or `%d`-style
    frame fields) resolved relative to `output_file`'s directory (the rig
    root) and is stored verbatim as the sensor's image pattern. Returns a
    summary dict with `output_file`, `image_count`, `pattern`, `name`, and
    `camera`. Raises `CamrigCreateError` on any failure.
    """
    import numpy as np

    from .._sfmtool.io import write_camrig

    output_file = Path(output_file)
    rig_root = output_file.parent
    pattern = normalize_pattern(image_pattern)

    named_overrides = {
        "--focal-length": focal_length,
        "--focal-length-x": focal_length_x,
        "--focal-length-y": focal_length_y,
        "--principal-point-x": principal_point_x,
        "--principal-point-y": principal_point_y,
    }
    given_named = [flag for flag, value in named_overrides.items() if value is not None]
    if params is not None and given_named:
        raise CamrigCreateError(
            f"--params cannot be combined with {', '.join(given_named)}; "
            "--params already specifies the full parameter list."
        )
    if focal_length is not None and (
        focal_length_x is not None or focal_length_y is not None
    ):
        raise CamrigCreateError(
            "--focal-length sets both axes; do not also pass "
            "--focal-length-x / --focal-length-y."
        )
    if params is not None and camera_model is None:
        raise CamrigCreateError(
            "--params requires --camera-model (to know the parameter order)."
        )

    target_resolution = parse_resolution(resolution) if resolution else None

    rel_paths = find_images(rig_root, pattern)
    abs_paths = [rig_root / rel for rel in rel_paths]

    if params is not None:
        camera = _camera_from_explicit_params(
            camera_model, params, target_resolution, abs_paths, rel_paths
        )
    else:
        camera = _camera_from_inference(
            abs_paths,
            rel_paths,
            camera_model,
            target_resolution,
            focal_length,
            focal_length_x,
            focal_length_y,
            principal_point_x,
            principal_point_y,
        )

    rig_name = name if name else output_file.stem
    write_camrig(
        path=str(output_file),
        name=rig_name,
        rig_type="generic",
        cameras=[camera],
        sensor_image_patterns=[pattern],
        camera_indexes=[0],
        quaternions_wxyz=np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float64),
        translations_xyz=np.array([[0.0, 0.0, 0.0]], dtype=np.float64),
        zstd_level=zstd_level,
    )

    return {
        "output_file": output_file,
        "image_count": len(rel_paths),
        "pattern": pattern,
        "name": rig_name,
        "camera": camera,
    }
