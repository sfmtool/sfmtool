# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Discover and resolve `.camrig` files for `sfm solve`.

A single-sensor `.camrig` dropped in a workspace supplies a camera
(intrinsics) for the images its stored pattern matches. `sfm solve`
auto-discovers these: when every image being solved is covered by one
single-sensor `.camrig`, that camera is used as the prior, taking precedence
over `camera_config.json` and `rig_config.json`.

Multi-sensor `.camrig` rigs are not yet consumed by `solve`; encountering one
that covers the images is an error rather than a silent fallback.
"""

from pathlib import Path

from ._camrig_pattern import match_pattern

# Allowed relative difference between an image's aspect ratio and the
# `.camrig` camera's aspect ratio before the rig is rejected.
_ASPECT_TOLERANCE = 1e-3


class CamrigSolveError(Exception):
    """Raised when discovered `.camrig` files cannot be used for a solve."""


def _covered_paths(camrig_dir: Path, patterns: list[str]) -> set[Path]:
    """Resolved absolute paths the `.camrig`'s patterns match from its directory."""
    covered: set[Path] = set()
    for pattern in patterns:
        covered.update(match_pattern(camrig_dir, pattern))
    return covered


def _check_image_resolution(image_paths, camera: dict, camrig_file: Path) -> None:
    """Verify the solved images suit the `.camrig`'s single camera.

    `sfm solve` uses one camera for every image on the `.camrig` path, so the
    images must all share one resolution, and that resolution must match the
    `.camrig` camera's aspect ratio (focal length and principal point are
    scaled uniformly to the actual image size). A mixed-resolution image set
    would otherwise be silently mis-scaled off the first image alone.
    """
    from ._camera_setup import _read_image_size

    by_size: dict[tuple[int, int], list] = {}
    for image_path in image_paths:
        by_size.setdefault(_read_image_size(Path(image_path)), []).append(image_path)

    if len(by_size) > 1:
        lines = []
        for (w, h), paths in sorted(by_size.items()):
            example = sorted(str(p) for p in paths)[0]
            lines.append(f"  {w}x{h}  ({len(paths)} image(s))  e.g. {example}")
        raise CamrigSolveError(
            f"the images covered by {camrig_file} have mixed resolutions; a "
            ".camrig describes one camera, so every image it covers must "
            "share one resolution:\n" + "\n".join(lines)
        )

    (image_w, image_h) = next(iter(by_size))
    camera_w, camera_h = camera["width"], camera["height"]
    if abs(image_w / image_h - camera_w / camera_h) > _ASPECT_TOLERANCE:
        raise CamrigSolveError(
            f"image resolution {image_w}x{image_h} does not match the aspect "
            f"ratio of the camera in {camrig_file} ({camera_w}x{camera_h}); "
            "the .camrig was calibrated for a different image shape."
        )


def resolve_camrig_for_solve(
    image_paths,
    workspace_dir,
    camera_model,
    camera_config_resolver=None,
):
    """Find the `.camrig` camera covering every image being solved.

    Scans `workspace_dir` for `.camrig` files and returns the camera dict
    (`{model, width, height, parameters}`) of the single-sensor `.camrig` that
    matches all of `image_paths`, or `None` when no `.camrig` applies.

    Raises `CamrigSolveError` when the discovered `.camrig` files cannot be
    used: the images span multiple `.camrig` files, a `.camrig` covers only
    some of them, a matching `.camrig` is multi-sensor, or `--camera-model`
    was given alongside a matching `.camrig`.
    """
    if workspace_dir is None:
        return None
    workspace_dir = Path(workspace_dir).resolve()
    camrig_files = sorted(workspace_dir.rglob("*.camrig"))
    if not camrig_files:
        return None

    from ._sfmtool import read_camrig

    solve_set = {Path(p).resolve() for p in image_paths}

    single_sensor_hits: dict[Path, tuple[set[Path], dict]] = {}
    multi_sensor_hits: list[Path] = []

    for camrig_file in camrig_files:
        try:
            data = read_camrig(str(camrig_file))
        except Exception as e:
            raise CamrigSolveError(f"could not read {camrig_file}: {e}") from None
        patterns = data["sensor_image_patterns"]
        if not patterns:
            continue  # geometry-only rig, not backed by workspace images
        covered = _covered_paths(camrig_file.parent, patterns)
        if not covered & solve_set:
            continue
        if data["metadata"]["sensor_count"] != 1:
            multi_sensor_hits.append(camrig_file)
        else:
            single_sensor_hits[camrig_file] = (covered, data)

    if multi_sensor_hits:
        raise CamrigSolveError(
            f"{multi_sensor_hits[0]} is a multi-sensor rig; sfm solve does "
            "not yet support multi-sensor .camrig files."
        )
    if not single_sensor_hits:
        return None
    if len(single_sensor_hits) > 1:
        names = ", ".join(str(c) for c in sorted(single_sensor_hits))
        raise CamrigSolveError(
            "the images being solved are covered by multiple .camrig files "
            f"({names}); solve each .camrig's images separately."
        )

    camrig_file, (covered, data) = next(iter(single_sensor_hits.items()))
    uncovered = solve_set - covered
    if uncovered:
        example = sorted(uncovered)[0]
        raise CamrigSolveError(
            f"{camrig_file} matches only {len(solve_set & covered)} of "
            f"{len(solve_set)} images being solved; e.g. {example} is not "
            "matched by its pattern. Solve only the images the .camrig "
            "covers, or remove the .camrig."
        )
    if camera_model is not None:
        raise CamrigSolveError(
            "--camera-model cannot be used together with a .camrig; "
            f"{camrig_file} already provides the camera. Remove --camera-model."
        )

    camera = data["cameras"][data["camera_indexes"][0]]
    _check_image_resolution(image_paths, camera, camrig_file)

    if camera_config_resolver is not None:
        for image_path in image_paths:
            if camera_config_resolver.resolve_for_image(Path(image_path)) is not None:
                print(
                    f"Note: {camrig_file} takes precedence over "
                    "camera_config.json for these images."
                )
                break

    print(f"Camera intrinsics: using {camrig_file}")
    return camera
