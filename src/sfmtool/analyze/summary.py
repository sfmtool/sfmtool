# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Summary printers for `sfm inspect`.

One ``print_*_summary`` function per inspectable file type. Each prints a
compact label/value block by default and a fuller report with ``verbose=True``.
"""

import textwrap
from pathlib import Path

import click
import numpy as np

from .._sfmtool import KdTree3d, SfmrReconstruction
from .._histogram_utils import print_histogram

# Matches sfmtool_core::infinity::DEFAULT_INVERSE_DEPTH_Z_CUTOFF: below this a
# point's depth is statistically indistinguishable from infinity.
DEPTH_RELIABILITY_Z_CUTOFF = 4.0


class InspectError(Exception):
    """Raised when a file fails its integrity check during inspection."""


def _print_block(rows: list[tuple[str, str]]) -> None:
    """Print aligned ``label: value`` rows."""
    width = max(len(label) for label, _ in rows)
    for label, value in rows:
        click.echo(f"{(label + ':'):<{width + 1}} {value}")


def _human_size(num_bytes: int) -> str:
    """Format a byte count as a human-readable string."""
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024.0 or unit == "GB":
            if unit == "B":
                return f"{int(size)} B"
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} GB"


def _check_integrity(valid: bool, errors: list[str]) -> None:
    """Print integrity errors and raise if a file failed verification."""
    if valid:
        return
    for err in errors:
        click.echo(f"  {err}")
    raise InspectError("integrity verification failed")


# ---------------------------------------------------------------------------
# .sfmr reconstructions
# ---------------------------------------------------------------------------


def print_sfmr_summary(path: Path, verbose: bool = False) -> None:
    """Inspect a `.sfmr` reconstruction file."""
    from .._sfmtool import read_sfmr_metadata, verify_sfmr

    valid, errors = verify_sfmr(str(path))

    if verbose:
        recon = SfmrReconstruction.load(path)
        print_reconstruction_summary(recon, recon_name=path.name)
        click.echo(f"Integrity: {'OK' if valid else 'FAILED'}")
        _check_integrity(valid, errors)
        return

    meta = read_sfmr_metadata(str(path))
    point_count = meta["point_count"]
    infinity = meta.get("infinity_point_count", 0)
    points_value = f"{point_count:,}"
    if infinity:
        points_value += f"  ({infinity:,} at infinity)"

    rows = [
        ("File", path.name),
        ("Format", f".sfmr version {meta['version']}"),
        ("Operation", f"{meta['operation']}  ({meta['tool']} {meta['tool_version']})"),
        ("Images", f"{meta['image_count']:,}"),
        ("Cameras", f"{meta['camera_count']:,}"),
        ("3D points", points_value),
        ("Observations", f"{meta['observation_count']:,}"),
    ]
    if meta.get("rig_count"):
        rows.append(
            (
                "Rigs",
                f"{meta['rig_count']} rigs, {meta.get('sensor_count', 0)} sensors, "
                f"{meta.get('frame_count', 0)} frames",
            )
        )
    rows.append(("Integrity", "OK" if valid else "FAILED"))
    _print_block(rows)
    _check_integrity(valid, errors)


# ---------------------------------------------------------------------------
# .sift feature files
# ---------------------------------------------------------------------------


def print_sift_summary(path: Path, verbose: bool = False) -> None:
    """Inspect a `.sift` feature file."""
    from .._sfmtool import read_sift_metadata, verify_sift
    from ..sift.file import SiftReader, feature_size_x, feature_size_y

    valid, errors = verify_sift(str(path))
    info = read_sift_metadata(str(path))
    meta = info["metadata"]
    tool_meta = info["feature_tool_metadata"]
    hashes = info["content_hash"]

    rows = [
        ("File", path.name),
        ("Format", ".sift"),
        (
            "Image",
            f"{meta['image_name']}  ({meta['image_width']}x{meta['image_height']})",
        ),
        ("Features", f"{meta['feature_count']:,}"),
        ("Feature tool", tool_meta["feature_tool"]),
        ("Integrity", "OK" if valid else "FAILED"),
    ]
    _print_block(rows)

    if verbose:
        click.echo("")
        _print_block(
            [
                ("Image file size", _human_size(meta["image_file_size"])),
                ("Image file hash", meta["image_file_xxh128"]),
                ("Feature tool hash", hashes["feature_tool_xxh128"]),
                ("Content hash", hashes["content_xxh128"]),
                ("Feature options", str(tool_meta["feature_options"])),
            ]
        )
        click.echo("\nTop 5 features (by size):")
        with SiftReader(path) as reader:
            positions = reader.read_positions(count=5)
            affine_shapes = reader.read_affine_shapes(count=5)
        sizes_x = feature_size_x(affine_shapes)
        sizes_y = feature_size_y(affine_shapes)
        for i, (pos, sx, sy) in enumerate(zip(positions, sizes_x, sizes_y), 1):
            click.echo(
                f"  {i}. pos=({pos[0]:.2f}, {pos[1]:.2f}), size=({sx:.2f}, {sy:.2f})"
            )

    _check_integrity(valid, errors)


# ---------------------------------------------------------------------------
# .matches files
# ---------------------------------------------------------------------------


def print_matches_summary(path: Path, verbose: bool = False) -> None:
    """Inspect a `.matches` file."""
    from .._sfmtool import read_matches_metadata, verify_matches

    valid, errors = verify_matches(str(path))
    meta = read_matches_metadata(str(path))

    tool_label = f"{meta['matching_tool']} {meta['matching_tool_version']}".strip()
    rows = [
        ("File", path.name),
        ("Format", f".matches version {meta['version']}"),
        ("Matching", f"{meta['matching_method']}  ({tool_label})"),
        ("Images", f"{meta['image_count']:,}"),
        ("Image pairs", f"{meta['image_pair_count']:,}"),
        ("Matches", f"{meta['match_count']:,}"),
        ("Two-view geom", "yes" if meta["has_two_view_geometries"] else "no"),
        ("Integrity", "OK" if valid else "FAILED"),
    ]
    _print_block(rows)

    if verbose:
        from .._sfmtool import read_matches

        click.echo("")
        ws = meta.get("workspace", {})
        _print_block(
            [
                ("Timestamp", meta.get("timestamp", "unknown")),
                ("Workspace", ws.get("absolute_path", "unknown")),
                ("Matching options", str(meta.get("matching_options", {}))),
            ]
        )

        data = read_matches(str(path))
        match_counts = np.asarray(data["match_counts"])
        if match_counts.size > 0:
            click.echo(
                f"\nMatches per pair: min {int(match_counts.min())}, "
                f"max {int(match_counts.max())}, mean {match_counts.mean():.1f}"
            )
            print_histogram(
                match_counts.astype(np.float64),
                "Matches per pair",
                show_stats=False,
            )

        distances = np.asarray(data["match_descriptor_distances"])
        if distances.size > 0:
            click.echo(
                f"\nDescriptor distances: min {distances.min():.1f}, "
                f"max {distances.max():.1f}, mean {distances.mean():.1f}"
            )
            print_histogram(
                distances.astype(np.float64),
                "Descriptor distance",
                show_stats=False,
            )

        if meta["has_two_view_geometries"]:
            tvg = data["tvg_metadata"]
            click.echo(
                f"\nTwo-view geometries: {tvg['inlier_count']:,} total inliers "
                f"({tvg.get('verification_tool', 'unknown')})"
            )
            inlier_counts = np.asarray(data["inlier_counts"])
            if inlier_counts.size > 0:
                print_histogram(
                    inlier_counts.astype(np.float64),
                    "Inliers per pair",
                    show_stats=False,
                )

    _check_integrity(valid, errors)


# ---------------------------------------------------------------------------
# .camrig camera rigs
# ---------------------------------------------------------------------------


def print_camrig_summary(path: Path, verbose: bool = False) -> None:
    """Inspect a `.camrig` camera rig file."""
    from .._sfmtool import read_camrig_metadata, verify_camrig

    valid, errors = verify_camrig(str(path))
    info = read_camrig_metadata(str(path))
    meta = info["metadata"]

    rows = [
        ("File", path.name),
        ("Format", f".camrig version {meta['version']}"),
        ("Name", meta["name"]),
        ("Rig type", meta["rig_type"]),
        ("Sensors", f"{meta['sensor_count']:,}"),
        ("Cameras", f"{meta['camera_count']:,}"),
        ("Integrity", "OK" if valid else "FAILED"),
    ]
    _print_block(rows)

    if verbose:
        from .._sfmtool import read_camrig

        attrs = meta.get("rig_attributes")
        if isinstance(attrs, dict) and attrs:
            click.echo("\nRig attributes:")
            _print_block([(f"  {k}", str(v)) for k, v in attrs.items()])

        click.echo(f"\nContent hash: {info['content_hash']['content_xxh128']}")

        data = read_camrig(str(path))
        click.echo("\nCameras:")
        for idx, cam in enumerate(data["cameras"]):
            click.echo(f"  Camera {idx}: {cam['model']} {cam['width']}x{cam['height']}")

    _check_integrity(valid, errors)


# ---------------------------------------------------------------------------
# Image files
# ---------------------------------------------------------------------------

_IMAGE_FORMAT_NAMES = {
    ".png": "PNG image",
    ".jpg": "JPEG image",
    ".jpeg": "JPEG image",
}


def print_image_summary(path: Path, verbose: bool = False) -> None:
    """Inspect an image file."""
    from ..camera.setup import _read_image_size

    width, height = _read_image_size(path)
    rows = [
        ("File", path.name),
        ("Format", _IMAGE_FORMAT_NAMES.get(path.suffix.lower(), "image")),
        ("Dimensions", f"{width}x{height}"),
        ("File size", _human_size(path.stat().st_size)),
    ]
    _print_block(rows)

    if verbose:
        from ..camera.setup import _infer_camera

        click.echo("")
        try:
            cam = _infer_camera(path)
            click.echo("Inferred camera (EXIF):")
            _print_block(
                [
                    ("  Model", cam.model.name),
                    ("  Dimensions", f"{cam.width}x{cam.height}"),
                    ("  Focal length", f"{cam.params[0]:.1f} px"),
                ]
            )
        except Exception as e:
            click.echo(f"Inferred camera (EXIF): unavailable ({e})")


# ---------------------------------------------------------------------------
# .sfmr verbose report
# ---------------------------------------------------------------------------


def _print_tool_options(tool_options) -> None:
    """Print the operation's recorded tool parameters under the Metadata block.

    ``tool_options`` is the free-form dict the producing command stored — the
    ordered ``transforms`` list for ``xform``, solver flags for ``solve``, and
    so on. The ``transforms`` list is printed first (one entry per line);
    remaining keys follow in sorted order. Nothing is printed when it is absent
    or empty.
    """
    if not tool_options:
        return
    click.echo("  Tool options:")
    transforms = tool_options.get("transforms")
    if isinstance(transforms, list):
        click.echo("    transforms:")
        for transform in transforms:
            click.echo(f"      - {transform}")
    for key in sorted(k for k in tool_options if k != "transforms"):
        click.echo(f"    {key}: {tool_options[key]}")


def print_reconstruction_summary(
    recon: SfmrReconstruction, recon_name: str | None = None
):
    """Print a detailed summary of a reconstruction file."""
    metadata = recon.source_metadata

    if recon_name is None:
        recon_name = metadata.get("source_path", "reconstruction")
        if "/" in recon_name or "\\" in recon_name:
            recon_name = Path(recon_name).name

    click.echo(f"\nReconstruction file: {recon_name}")
    click.echo("=" * 70)

    # Metadata
    click.echo("\nMetadata:")
    click.echo(f"  Operation: {metadata.get('operation', 'unknown')}")
    click.echo(f"  Tool: {metadata.get('tool', 'unknown')}")
    click.echo(f"  Tool version: {metadata.get('tool_version', 'unknown')}")
    click.echo(f"  Timestamp: {metadata.get('timestamp', 'unknown')}")
    _print_tool_options(metadata.get("tool_options"))

    # Workspace
    workspace_info = metadata.get("workspace", {})
    click.echo("\nWorkspace:")
    click.echo(f"  Absolute path: {workspace_info.get('absolute_path', 'unknown')}")
    relative_path = workspace_info.get("relative_path", "unknown")
    click.echo(f"  Relative path: {relative_path}")
    click.echo(f"  Resolved workspace: {recon.workspace_dir}")
    click.echo(f"  Feature tool: {workspace_info.get('feature_tool', 'unknown')}")

    # Counts
    click.echo("\nReconstruction summary:")
    click.echo(f"  Images: {recon.image_count}")

    # Image path summarization (optional dependency)
    try:
        from deadline.job_attachments.api import summarize_path_list

        click.echo("  Image paths:")
        click.echo(
            textwrap.indent(summarize_path_list(recon.image_names).rstrip(), "    ")
        )
    except ImportError:
        pass

    click.echo(f"  Cameras: {recon.camera_count}")
    points_line = f"  3D points: {recon.point_count}"
    if recon.infinity_point_count:
        points_line += f"  ({recon.infinity_point_count} at infinity)"
    click.echo(points_line)
    click.echo(f"  Observations: {recon.observation_count}")

    if recon.point_count > 0:
        avg_obs = recon.observation_count / recon.point_count
        click.echo(f"  Avg observations per point: {avg_obs:.2f}")

    # Camera information
    from ..camera.cameras import _CAMERA_PARAM_NAMES

    click.echo("\nCameras:")
    for idx, cam in enumerate(recon.cameras):
        click.echo(f"  Camera {idx}: {cam.model} {cam.width}x{cam.height}")
        params = cam.parameters
        if params:
            canonical_order = _CAMERA_PARAM_NAMES.get(cam.model)
            if canonical_order is not None:
                keys = list(canonical_order)
                extra = sorted(set(params.keys()) - set(keys))
                keys.extend(extra)
            else:
                keys = list(params.keys())

            name_width = max(len(k) for k in keys)
            click.echo(f"    {'Parameter':<{name_width}}  {'Value':>14}")
            click.echo(f"    {'-' * name_width}  {'-' * 14}")
            for key in keys:
                val = params.get(key)
                if val is not None:
                    click.echo(f"    {key:<{name_width}}  {val:>14.6f}")

    # Rig information
    rfd = recon.rig_frame_data
    if rfd is not None:
        rigs_meta = rfd["rigs_metadata"]
        click.echo("\nRig configuration:")
        click.echo(f"  Rigs: {rigs_meta['rig_count']}")
        click.echo(f"  Total sensors: {rigs_meta['sensor_count']}")
        click.echo(f"  Frames: {rfd['frames_metadata']['frame_count']}")
        for rig_def in rigs_meta.get("rigs", []):
            click.echo(f"  Rig '{rig_def['name']}':")
            click.echo(f"    Sensors: {rig_def['sensor_count']}")
            click.echo(f"    Reference sensor: {rig_def['ref_sensor_name']}")
            sensor_names = rig_def.get("sensor_names", [])
            offset = rig_def.get("sensor_offset", 0)
            for i, sensor_name in enumerate(sensor_names):
                cam_idx = int(rfd["sensor_camera_indexes"][offset + i])
                is_ref = sensor_name == rig_def["ref_sensor_name"]
                ref_marker = " (ref)" if is_ref else ""
                if not is_ref:
                    t = rfd["sensor_translations_xyz"][offset + i]
                    click.echo(
                        f"    {sensor_name}{ref_marker}: camera {cam_idx}, "
                        f"translation=({t[0]:.4f}, {t[1]:.4f}, {t[2]:.4f})"
                    )
                else:
                    click.echo(f"    {sensor_name}{ref_marker}: camera {cam_idx}")
    else:
        click.echo("\nRig configuration: none")

    # 3D point statistics
    if recon.point_count > 0:
        positions = recon.positions
        errors = recon.errors

        click.echo("\n3D Point statistics:")
        click.echo("  Position range:")
        click.echo(f"    X: [{positions[:, 0].min():.3f}, {positions[:, 0].max():.3f}]")
        print_histogram(positions[:, 0], "X distribution", show_stats=False)
        click.echo(f"    Y: [{positions[:, 1].min():.3f}, {positions[:, 1].max():.3f}]")
        print_histogram(positions[:, 1], "Y distribution", show_stats=False)
        click.echo(f"    Z: [{positions[:, 2].min():.3f}, {positions[:, 2].max():.3f}]")
        print_histogram(positions[:, 2], "Z distribution", show_stats=False)

        click.echo("  Reprojection error:")
        click.echo(f"    Mean: {errors.mean():.3f} pixels")
        click.echo(f"    Median: {np.median(errors):.3f} pixels")
        click.echo(f"    Min: {errors.min():.3f} pixels")
        click.echo(f"    Max: {errors.max():.3f} pixels")
        print_histogram(errors, "Error distribution", show_stats=False)

        _print_depth_reliability(recon)

    # Observation statistics
    if recon.point_count > 0:
        observation_counts = recon.observation_counts

        click.echo("\nObservation statistics:")
        click.echo(f"  Min observations per point: {observation_counts.min()}")
        click.echo(f"  Max observations per point: {observation_counts.max()}")
        click.echo(
            f"  Median observations per point: {int(np.median(observation_counts))}"
        )
        print_histogram(
            observation_counts, "Track length distribution", show_stats=False
        )

    # Nearest neighbor distances
    if recon.point_count > 1:
        positions = recon.positions
        nn_distances = KdTree3d(positions).nearest_neighbor_distances()

        click.echo("\nNearest neighbor distances:")
        click.echo(f"  Min: {nn_distances.min():.6f}")
        click.echo(f"  Max: {nn_distances.max():.6f}")
        click.echo(f"  Mean: {nn_distances.mean():.6f}")
        click.echo(f"  Median: {np.median(nn_distances):.6f}")
        print_histogram(nn_distances, "NN distance distribution", show_stats=False)

    click.echo("")


def _print_depth_reliability(recon: SfmrReconstruction) -> None:
    """Print per-point triangulation conditioning.

    The inverse-depth z-score (depth / σ_depth) is the scale-free reliability
    signal: low values mean the depth is statistically indistinguishable from
    infinity. The normal-matrix condition number is the cheap geometric proxy.
    Points at infinity and sub-2-view points have no finite depth and are
    excluded.
    """
    diag = recon.triangulation_diagnostics(noise_px=1.0)
    z = np.asarray(diag["inverse_depth_z"])
    cond = np.asarray(diag["condition_number"])

    finite = np.isfinite(z)
    n = int(finite.sum())
    if n == 0:
        return
    zf = z[finite]

    click.echo("  Depth reliability (finite, >=2-view points):")
    click.echo(f"    Diagnosed points: {n:,}")
    click.echo("    Inverse-depth z (depth/sigma; low => near-infinity):")
    click.echo(f"      Median: {np.median(zf):.2f}")
    click.echo(f"      Min: {zf.min():.2f}   Max: {zf.max():.2f}")
    below = int((zf < DEPTH_RELIABILITY_Z_CUTOFF).sum())
    click.echo(
        f"      Below z={DEPTH_RELIABILITY_Z_CUTOFF:g} (near-infinity): "
        f"{below:,} ({100.0 * below / n:.1f}%)"
    )
    # Clip the long tail so the histogram bins the bulk of the points usefully.
    hi = max(float(np.percentile(zf, 99)), DEPTH_RELIABILITY_Z_CUTOFF)
    print_histogram(zf, "Inverse-depth z", min_val=0.0, max_val=hi, show_stats=False)

    cond_f = cond[np.isfinite(cond)]
    if cond_f.size:
        click.echo(f"    Condition number (median): {np.median(cond_f):.1f}")


# ---------------------------------------------------------------------------
# 3D point IDs (pt3d_<hash>_<index>)
# ---------------------------------------------------------------------------


def print_point_summary(
    recon: SfmrReconstruction,
    point_index: int,
    point_id: str,
    sfmr_path: Path,
    verbose: bool = False,
) -> None:
    """Inspect a single 3D point referenced by its ``pt3d_<hash>_<index>`` ID.

    The compact summary uses only the stored reconstruction. ``--verbose`` adds
    the full triangulation analysis, which re-derives the point's observation
    rays from the workspace ``.sift`` files (they must be present).
    """
    xyzw = np.asarray(recon.positions_xyzw)[point_index]
    color = np.asarray(recon.colors)[point_index]
    error = float(np.asarray(recon.errors)[point_index])
    at_infinity = xyzw[3] == 0.0

    point_ids = np.asarray(recon.track_point_ids)
    obs_images = np.asarray(recon.track_image_indexes)[point_ids == point_index]

    coord_label = "Direction" if at_infinity else "Position"
    rows = [
        ("Point", point_id),
        ("File", sfmr_path.name),
        ("Type", "at infinity (w=0)" if at_infinity else "finite (w=1)"),
        (coord_label, f"({xyzw[0]:.3f}, {xyzw[1]:.3f}, {xyzw[2]:.3f})"),
        ("Color", f"rgb({int(color[0])}, {int(color[1])}, {int(color[2])})"),
        ("Reprojection error", f"{error:.3f} px"),
        ("Observations", f"{len(obs_images)} images"),
    ]
    _print_block(rows)

    if not verbose:
        return

    diag = recon.inspect_point(point_index)
    tp = diag["triangulated_point"]
    ev = diag["eigenvalues"]
    resolvable = diag["resolvable_distance"]
    fh = diag["finite_horizon"]
    sufficient = "sufficient" if resolvable >= fh else "insufficient"

    click.echo("\nTriangulation analysis:")
    click.echo(f"  Classification (re-derived): {diag['classification']}")
    click.echo(f"  Triangulated point: ({tp[0]:.3f}, {tp[1]:.3f}, {tp[2]:.3f})")
    click.echo(f"  Depth along mean view direction: {diag['depth']:.2f}")
    click.echo(f"  In front of all cameras: {'yes' if diag['in_front'] else 'no'}")
    click.echo(
        f"  Condition number: {diag['condition_number']:.1f}   "
        f"eigenvalues: [{ev[0]:.3g}, {ev[1]:.3g}, {ev[2]:.3g}]"
    )
    click.echo(
        f"  Inverse-depth z: {diag['inverse_depth_z']:.2f} "
        f"(sigma {diag['sigma']:.3g}; cutoff {DEPTH_RELIABILITY_Z_CUTOFF:g})"
    )
    click.echo(
        f"  Resolvable distance: {resolvable:.1f}  vs finite_horizon "
        f"(camera extents) {fh:.1f}  -> {sufficient} baseline"
    )
    click.echo(f"  Observing-camera baseline span: {diag['baseline_span']:.2f}")
    click.echo(f"  Ray spread (max angle to mean): {diag['max_ray_angle_deg']:.3f} deg")

    click.echo("\n  Observations (incidence = ray angle off optical axis):")
    for o in sorted(
        diag["observations"], key=lambda d: d["incidence_deg"], reverse=True
    ):
        edge = "  <- near fisheye edge" if o["incidence_deg"] >= 80.0 else ""
        click.echo(
            f"    {o['image_name']}  feat {o['feature_index']}  "
            f"incidence {o['incidence_deg']:.1f} deg{edge}"
        )
