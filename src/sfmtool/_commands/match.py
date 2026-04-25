# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Feature matching command — produces .matches files."""

import os
import re
import tempfile
from datetime import datetime
from pathlib import Path

import click
import numpy as np

from .._cli_utils import timed_command
from .._filenames import expand_paths


@click.command("match")
@timed_command
@click.help_option("--help", "-h")
@click.argument("paths", nargs=-1, type=click.Path(exists=True))
@click.option(
    "--exhaustive",
    "-e",
    "exhaustive",
    is_flag=True,
    help="Run exhaustive pairwise matching.",
)
@click.option(
    "--max-features",
    "max_feature_count",
    type=click.IntRange(min=1),
    help="Maximum number of features to use from each image.",
)
@click.option(
    "--output",
    "-o",
    "output_path",
    type=click.Path(),
    help="Output .matches file path. If not specified, generates a timestamped filename.",
)
@click.option(
    "--range",
    "-r",
    "range_expr",
    help="A range expression of file numbers to use from the input directories.",
)
@click.option(
    "--sequential",
    "-s",
    "sequential",
    is_flag=True,
    help="Run sequential matching (pairs nearby images in sequence order). "
    "Best for ordered image collections with known capture order.",
)
@click.option(
    "--sequential-overlap",
    "sequential_overlap",
    type=click.IntRange(min=1),
    default=10,
    help="Number of overlapping image pairs for --sequential. Default: 10.",
)
@click.option(
    "--flow",
    "flow_match",
    is_flag=True,
    help="Use optical flow-based matching instead of exhaustive descriptor matching. "
    "Best for sequential video frames with small inter-frame motion.",
)
@click.option(
    "--flow-preset",
    "flow_preset",
    type=click.Choice(["fast", "default", "high_quality"]),
    default="default",
    help="Optical flow quality preset for --flow. Default: default.",
)
@click.option(
    "--flow-skip",
    "flow_wide_baseline_skip",
    type=click.IntRange(min=1),
    default=5,
    help="Sliding window size for --flow. 1 = adjacent pairs only. Default: 5.",
)
@click.option(
    "--camera-model",
    "camera_model",
    type=click.Choice(
        [
            "SIMPLE_PINHOLE",
            "PINHOLE",
            "SIMPLE_RADIAL",
            "RADIAL",
            "OPENCV",
            "OPENCV_FISHEYE",
            "SIMPLE_RADIAL_FISHEYE",
            "RADIAL_FISHEYE",
            "THIN_PRISM_FISHEYE",
            "RAD_TAN_THIN_PRISM_FISHEYE",
        ],
        case_sensitive=False,
    ),
    default=None,
    help="Camera model to use (overrides auto-detection).",
)
@click.option(
    "--merge",
    "merge",
    is_flag=True,
    help="Merge multiple .matches files into one. "
    "PATHS should be .matches files instead of image directories.",
)
def match(
    paths,
    exhaustive,
    sequential,
    sequential_overlap,
    max_feature_count,
    output_path,
    range_expr,
    flow_match,
    flow_preset,
    flow_wide_baseline_skip,
    camera_model,
    merge,
):
    """Match features between image pairs and write a .matches file.

    Requires a workspace initialized with 'sfm init' and SIFT features
    extracted with 'sfm sift --extract'.

    Examples:
        # Exhaustive matching
        sfm match --exhaustive images/

        # Sequential matching for ordered collections
        sfm match --sequential images/

        # Flow-based matching for sequential video
        sfm match --flow images/

        # With feature count limit
        sfm match --exhaustive --max-features 4096 images/

        # Merge matches from different methods
        sfm match --merge seq.matches exhaustive.matches -o combined.matches
    """
    if merge:
        try:
            _run_merge(paths, output_path)
        except Exception as e:
            raise click.ClickException(str(e))
        return

    from ..cli import deduce_workspace

    if not paths:
        raise click.UsageError("Must provide image paths.")

    method_count = sum([exhaustive, sequential, flow_match])
    if method_count > 1:
        raise click.UsageError(
            "Cannot specify more than one matching method. "
            "Choose one of: --exhaustive (-e), --sequential (-s), or --flow"
        )
    if method_count == 0:
        raise click.UsageError(
            "Must specify a matching method: "
            "--exhaustive (-e), --sequential (-s), or --flow"
        )

    numbers = None
    if range_expr:
        from openjd.model import IntRangeExpr

        numbers = IntRangeExpr.from_str(range_expr)

    paths = [Path(p) for p in paths]
    filenames = expand_paths(
        paths, extensions=(".png", ".jpg", ".jpeg"), numbers=numbers
    )
    if not filenames:
        raise click.UsageError("No image files found in the provided paths.")

    absolute_paths = [Path(os.path.normpath(os.path.abspath(p))) for p in filenames]
    workspace_dir = deduce_workspace({p.parent for p in absolute_paths})

    from .._camera_config import CameraConfigResolver
    from .._camera_setup import _check_camera_model_conflict

    camera_config_resolver = CameraConfigResolver(workspace_dir)
    _check_camera_model_conflict(absolute_paths, camera_config_resolver, camera_model)

    try:
        if flow_match:
            matching_method = "flow"
        elif sequential:
            matching_method = "sequential"
        else:
            matching_method = "exhaustive"
        _run_matching(
            absolute_paths,
            workspace_dir,
            matching_method=matching_method,
            max_feature_count=max_feature_count,
            output_path=output_path,
            camera_model=camera_model,
            flow_preset=flow_preset,
            flow_wide_baseline_skip=flow_wide_baseline_skip,
            sequential_overlap=sequential_overlap,
        )
    except Exception as e:
        raise click.ClickException(str(e))


def _run_matching(
    image_paths: list[Path],
    workspace_dir: Path,
    matching_method: str,
    max_feature_count: int | None,
    output_path: str | None,
    camera_model: str | None,
    flow_preset: str = "default",
    flow_wide_baseline_skip: int = 5,
    sequential_overlap: int = 10,
):
    """Run matching and produce a .matches file."""
    import pycolmap

    from .._workspace import load_workspace_config
    from .._sift_file import image_files_to_sift_files

    ws_config = load_workspace_config(workspace_dir)
    feature_tool = ws_config.get("feature_tool", "colmap")
    feature_options = ws_config.get("feature_options")
    feature_prefix_dir = ws_config.get("feature_prefix_dir")

    # Ensure SIFT features exist
    click.echo("Checking SIFT features...")
    sift_paths = image_files_to_sift_files(
        image_paths,
        feature_tool=feature_tool,
        feature_options=feature_options,
        feature_prefix_dir=feature_prefix_dir,
    )

    image_count = len(image_paths)
    click.echo(f"Found {image_count} images with SIFT features")

    # Build workspace-relative image names
    image_names = []
    for p in image_paths:
        rel = os.path.relpath(p, workspace_dir).replace("\\", "/")
        image_names.append(rel)

    # Create a temporary COLMAP database, populate features, run matching
    with tempfile.TemporaryDirectory(prefix="sfm_match_") as tmpdir:
        db_path = Path(tmpdir) / "database.db"

        click.echo("Populating COLMAP database with features...")
        _populate_db_features(
            db_path,
            image_paths,
            sift_paths,
            image_names,
            workspace_dir,
            max_feature_count,
            camera_model,
        )

        # Run matching
        click.echo(f"Running {matching_method} matching...")
        if matching_method == "exhaustive":
            pycolmap.match_exhaustive(db_path)
        elif matching_method == "sequential":
            pairing_options = pycolmap.SequentialPairingOptions(
                overlap=sequential_overlap,
                quadratic_overlap=True,
            )
            pycolmap.match_sequential(db_path, pairing_options=pairing_options)
        elif matching_method == "flow":
            _run_flow_matching(
                image_paths,
                sift_paths,
                workspace_dir,
                db_path,
                Path(tmpdir),
                max_feature_count=max_feature_count,
                flow_preset=flow_preset,
                flow_wide_baseline_skip=flow_wide_baseline_skip,
            )
        else:
            raise ValueError(f"Unsupported matching method: {matching_method}")

        # Read matches + TVGs back from the DB
        click.echo("Reading matches from database...")
        from .._sfmtool import read_colmap_db_matches

        matches_data = read_colmap_db_matches(str(db_path), include_tvg=True)

    # The Rust reader sorts images lexicographically from the DB.
    # Re-derive image_names, sift_paths, and image_paths in that order.
    rust_image_names = matches_data["image_names"]
    name_to_sift = {name: sp for name, sp in zip(image_names, sift_paths)}
    name_to_path = {name: ip for name, ip in zip(image_names, image_paths)}
    image_names = list(rust_image_names)
    sift_paths = [name_to_sift[n] for n in image_names]
    image_paths = [name_to_path[n] for n in image_names]

    # Compute descriptor distances from .sift files
    click.echo("Computing descriptor distances...")
    _compute_descriptor_distances(matches_data, sift_paths, max_feature_count)

    # Fill in metadata
    matches_data["metadata"]["matching_method"] = matching_method
    if matching_method == "flow":
        matches_data["metadata"]["matching_tool"] = "sfmtool-flow"
        matches_data["metadata"]["matching_tool_version"] = ""
    else:
        matches_data["metadata"]["matching_tool"] = "colmap"
        matches_data["metadata"]["matching_tool_version"] = pycolmap.__version__
    matches_data["metadata"]["matching_options"] = {}
    if max_feature_count:
        matches_data["metadata"]["matching_options"]["max_feature_count"] = (
            max_feature_count
        )
    if matching_method == "flow":
        matches_data["metadata"]["matching_options"]["flow_preset"] = flow_preset
        matches_data["metadata"]["matching_options"]["flow_skip"] = (
            flow_wide_baseline_skip
        )
    if matching_method == "sequential":
        matches_data["metadata"]["matching_options"]["sequential_overlap"] = (
            sequential_overlap
        )
    matches_data["metadata"]["version"] = 1
    matches_data["metadata"]["workspace"] = {
        "absolute_path": str(workspace_dir),
        "relative_path": "",
        "contents": {
            "feature_tool": feature_tool,
            "feature_type": ws_config.get("feature_type", "sift"),
            "feature_options": feature_options or {},
            "feature_prefix_dir": feature_prefix_dir or "",
        },
    }
    matches_data["metadata"]["timestamp"] = datetime.now().astimezone().isoformat()

    # Fill in feature tool hashes and sift content hashes
    _fill_sift_hashes(matches_data, sift_paths, image_names, image_paths)

    # Determine output path
    if output_path:
        out = Path(output_path)
    else:
        has_tvg = matches_data["has_two_view_geometries"]
        if has_tvg:
            out_dir = workspace_dir / "tvg-matches"
        else:
            out_dir = workspace_dir / "matches"
        out = _generate_output_path(out_dir, image_paths, matching_method)

    # Set relative_path from output location to workspace
    out_abs = Path(os.path.abspath(out))
    matches_data["metadata"]["workspace"]["relative_path"] = os.path.relpath(
        workspace_dir, out_abs.parent
    ).replace("\\", "/")

    # Write the .matches file
    from .._sfmtool import write_matches

    click.echo(f"Writing {out}...")
    write_matches(out, matches_data)

    pair_count = matches_data["metadata"]["image_pair_count"]
    match_count = matches_data["metadata"]["match_count"]
    click.echo(f"Done: {pair_count} pairs, {match_count} matches")
    if matches_data["has_two_view_geometries"]:
        inlier_count = matches_data["tvg_metadata"]["inlier_count"]
        click.echo(f"  Two-view geometries: {inlier_count} total inliers")


def _populate_db_features(
    db_path: Path,
    image_paths: list[Path],
    sift_paths: list[Path],
    image_names: list[str],
    workspace_dir: Path,
    max_feature_count: int | None,
    camera_model: str | None,
    camera_config_resolver=None,
):
    """Create a COLMAP DB and populate it with cameras, images, keypoints, descriptors."""
    from .._camera_config import CameraConfigResolver
    from .._colmap_db import _setup_db_single_camera, _setup_db_with_rigs
    from .._rig_config import _load_rig_config

    if camera_config_resolver is None:
        camera_config_resolver = CameraConfigResolver(workspace_dir)

    rig_config = _load_rig_config(workspace_dir)

    if rig_config is not None:
        _setup_db_with_rigs(
            image_paths,
            sift_paths,
            workspace_dir,
            db_path,
            max_feature_count,
            rig_configs=rig_config,
            camera_model=camera_model,
            camera_config_resolver=camera_config_resolver,
        )
    else:
        _setup_db_single_camera(
            image_paths,
            sift_paths,
            workspace_dir,
            db_path,
            max_feature_count,
            camera_model=camera_model,
            camera_config_resolver=camera_config_resolver,
        )


def _compute_descriptor_distances(matches_data, sift_paths, max_feature_count):
    """Compute L2 descriptor distances for all matches from .sift files."""
    from .._sift_file import SiftReader

    pair_count = matches_data["metadata"]["image_pair_count"]
    if pair_count == 0:
        return

    desc_cache = {}

    def get_descriptors(img_idx):
        if img_idx not in desc_cache:
            with SiftReader(sift_paths[img_idx]) as reader:
                desc = reader.read_descriptors(count=max_feature_count)
            desc_cache[img_idx] = desc.astype(np.float32)
        return desc_cache[img_idx]

    image_index_pairs = matches_data["image_index_pairs"]
    match_counts = matches_data["match_counts"]
    match_feature_indexes = matches_data["match_feature_indexes"]
    distances = matches_data["match_descriptor_distances"]

    offset = 0
    for k in range(pair_count):
        idx_i = int(image_index_pairs[k, 0])
        idx_j = int(image_index_pairs[k, 1])
        count = int(match_counts[k])

        desc_i = get_descriptors(idx_i)
        desc_j = get_descriptors(idx_j)

        for m in range(offset, offset + count):
            fi = int(match_feature_indexes[m, 0])
            fj = int(match_feature_indexes[m, 1])
            diff = desc_i[fi].astype(np.float32) - desc_j[fj].astype(np.float32)
            distances[m] = float(np.sqrt(np.dot(diff, diff)))

        offset += count


def _fill_sift_hashes(matches_data, sift_paths, image_names, image_paths):
    """Fill feature_tool_hashes and sift_content_hashes from .sift files."""
    from .._sfmtool import read_sift_metadata

    image_count = len(image_names)
    feature_tool_hashes = np.zeros((image_count, 16), dtype=np.uint8)
    sift_content_hashes = np.zeros((image_count, 16), dtype=np.uint8)

    for i, sift_path in enumerate(sift_paths):
        result = read_sift_metadata(str(sift_path))
        content_hash = result["content_hash"]
        ft_hash = bytes.fromhex(content_hash["feature_tool_xxh128"])
        ct_hash = bytes.fromhex(content_hash["content_xxh128"])
        feature_tool_hashes[i] = np.frombuffer(ft_hash, dtype=np.uint8)
        sift_content_hashes[i] = np.frombuffer(ct_hash, dtype=np.uint8)

    matches_data["feature_tool_hashes"] = feature_tool_hashes
    matches_data["sift_content_hashes"] = sift_content_hashes


def _generate_output_path(
    base_dir: Path, image_paths: list[Path], matching_method: str
) -> Path:
    """Generate a timestamped output path for a .matches file."""
    from deadline.job_attachments.api import summarize_paths_by_sequence
    from openjd.model import IntRangeExpr

    base_dir.mkdir(parents=True, exist_ok=True)

    now = datetime.now().astimezone()
    date_prefix = now.strftime("%Y%m%d")

    # Generate image descriptor
    filenames = [p.name for p in image_paths]
    summaries = summarize_paths_by_sequence(filenames)
    descriptor = ""
    if len(summaries) == 1 and summaries[0].index_set:
        summary = summaries[0]
        prefix = summary.path.split("%")[0].rstrip("_-")
        range_str = str(IntRangeExpr.from_list(sorted(summary.index_set)))
        range_str = range_str.replace(":", "x")
        descriptor = f"{prefix}_{range_str}"

    # Find max counter for this date
    pattern = re.compile(rf"^{re.escape(date_prefix)}-(\d{{2,}})(?:-.*)?\.matches$")
    max_counter = -1
    if base_dir.exists():
        for f in base_dir.iterdir():
            if f.is_file():
                m = pattern.match(f.name)
                if m:
                    max_counter = max(max_counter, int(m.group(1)))

    next_counter = max_counter + 1
    counter_str = f"{next_counter:02d}" if next_counter < 100 else str(next_counter)

    parts = [date_prefix, counter_str, matching_method]
    if descriptor:
        parts.append(descriptor)
    filename = "-".join(parts) + ".matches"

    return base_dir / filename


def _run_flow_matching(
    image_paths: list[Path],
    sift_paths: list[Path],
    workspace_dir: Path,
    db_path: Path,
    colmap_dir: Path,
    max_feature_count: int | None = None,
    flow_preset: str = "default",
    flow_wide_baseline_skip: int = 5,
) -> None:
    """Run flow-based matching and write results to COLMAP database.

    Computes optical flow between sequential image pairs, finds feature
    correspondences via advection + descriptor filtering, writes matches
    to the database, and runs geometric verification via pycolmap.
    """
    import pycolmap

    from ..feature_match._flow_matching import flow_match_sequential

    # Build image_id mapping from the database
    image_id_map = {}  # image index -> database image_id
    with pycolmap.Database.open(db_path) as db:
        images = db.read_all_images()
        rel_to_id = {}
        for img in images:
            rel_to_id[img.name] = img.image_id

        for idx, image_path in enumerate(image_paths):
            rel_path = os.path.relpath(image_path, workspace_dir).replace("\\", "/")
            if rel_path in rel_to_id:
                image_id_map[idx] = rel_to_id[rel_path]

    # Run the flow matching pipeline
    all_matches = flow_match_sequential(
        image_paths=[Path(p) for p in image_paths],
        sift_paths=sift_paths,
        preset=flow_preset,
        window_size=flow_wide_baseline_skip,
        max_feature_count=max_feature_count,
    )

    if not all_matches:
        click.echo("Warning: Flow matching produced no matches")
        return

    # Write matches to database and build pairs file for geometric verification
    pairs_path = colmap_dir / "flow_pairs.txt"
    with (
        pycolmap.Database.open(db_path) as db,
        open(pairs_path, "w") as pairs_file,
    ):
        for (idx_i, idx_j), matches in all_matches.items():
            if idx_i not in image_id_map or idx_j not in image_id_map:
                continue
            img_id_i = image_id_map[idx_i]
            img_id_j = image_id_map[idx_j]

            db.write_matches(img_id_i, img_id_j, matches)

            rel_i = os.path.relpath(image_paths[idx_i], workspace_dir).replace(
                "\\", "/"
            )
            rel_j = os.path.relpath(image_paths[idx_j], workspace_dir).replace(
                "\\", "/"
            )
            pairs_file.write(f"{rel_i} {rel_j}\n")

    tvg_options = pycolmap.TwoViewGeometryOptions()

    # Run geometric verification on matched pairs
    click.echo("Running geometric verification...")
    pycolmap.verify_matches(str(db_path), str(pairs_path), options=tvg_options)


def _run_merge(paths, output_path):
    """Merge multiple .matches files into a single file.

    Builds a unified image list, remaps pair indexes, concatenates matches
    for each pair, and deduplicates by feature index pair (keeping the match
    with the lowest descriptor distance). Two-view geometry data is dropped
    since it is invalidated by the merge.
    """
    from datetime import datetime

    from .._sfmtool import read_matches, write_matches

    if not paths:
        raise click.UsageError("Must provide .matches file paths.")
    if len(paths) < 2:
        raise click.UsageError("Need at least 2 .matches files to merge.")
    if not output_path:
        raise click.UsageError("--output / -o is required for --merge.")

    input_paths = []
    for p in paths:
        path = Path(p)
        if path.suffix.lower() != ".matches":
            raise click.UsageError(f"Expected a .matches file: {p}")
        input_paths.append(path)

    out = Path(output_path)
    if out.suffix.lower() != ".matches":
        raise click.UsageError(f"Output path must be a .matches file: {output_path}")

    # Read all input files
    click.echo(f"Merging {len(input_paths)} .matches files...")
    all_data = []
    for path in input_paths:
        click.echo(f"  Reading {path.name}...")
        data = read_matches(str(path))
        pairs = data["metadata"]["image_pair_count"]
        matches = data["metadata"]["match_count"]
        click.echo(
            f"    {data['metadata']['image_count']} images, "
            f"{pairs} pairs, {matches} matches"
        )
        all_data.append(data)

    # Build unified image list and validate consistency
    image_info = {}  # name -> {feature_count, tool_hash, content_hash}
    for data in all_data:
        names = list(data["image_names"])
        for i, name in enumerate(names):
            tool_hash = bytes(data["feature_tool_hashes"][i])
            content_hash = bytes(data["sift_content_hashes"][i])
            feature_count = int(data["feature_counts"][i])
            if name in image_info:
                existing = image_info[name]
                if existing["content_hash"] != content_hash:
                    raise click.ClickException(
                        f"Image '{name}' has different SIFT content hashes "
                        f"across input files — cannot merge matches from "
                        f"different feature extractions."
                    )
            else:
                image_info[name] = {
                    "feature_count": feature_count,
                    "tool_hash": tool_hash,
                    "content_hash": content_hash,
                }

    unified_names = sorted(image_info.keys())
    name_to_idx = {name: i for i, name in enumerate(unified_names)}
    image_count = len(unified_names)
    click.echo(f"  Unified image count: {image_count}")

    # Collect all matches per pair, remapping indexes
    # pair_key = (idx_i, idx_j) with idx_i < idx_j in the unified list
    pair_matches = {}  # (idx_i, idx_j) -> {(feat_i, feat_j): distance}
    pair_tvg = {}  # (idx_i, idx_j) -> tvg_info dict (best TVG for this pair)

    for data in all_data:
        names = list(data["image_names"])
        local_to_unified = [name_to_idx[n] for n in names]
        has_tvg = data.get("has_two_view_geometries", False)

        pairs = data["image_index_pairs"]
        counts = data["match_counts"]
        feat_idxs = data["match_feature_indexes"]
        distances = data["match_descriptor_distances"]

        if has_tvg:
            tvg_config_types = data["config_types"]
            tvg_config_indexes = data["config_indexes"]
            tvg_inlier_counts = data["inlier_counts"]
            tvg_inlier_feat_idxs = data["inlier_feature_indexes"]
            tvg_f_matrices = data["f_matrices"]
            tvg_e_matrices = data["e_matrices"]
            tvg_h_matrices = data["h_matrices"]
            tvg_quaternions = data["quaternions_wxyz"]
            tvg_translations = data["translations_xyz"]

        offset = 0
        inlier_offset = 0
        for k in range(len(counts)):
            local_i = int(pairs[k, 0])
            local_j = int(pairs[k, 1])
            uni_i = local_to_unified[local_i]
            uni_j = local_to_unified[local_j]
            swap = uni_i > uni_j
            if swap:
                uni_i, uni_j = uni_j, uni_i

            count = int(counts[k])
            key = (uni_i, uni_j)
            if key not in pair_matches:
                pair_matches[key] = {}

            for m in range(offset, offset + count):
                fi = int(feat_idxs[m, 0])
                fj = int(feat_idxs[m, 1])
                dist = float(distances[m])
                if swap:
                    fi, fj = fj, fi
                feat_key = (fi, fj)
                # Keep the match with the lowest descriptor distance
                if (
                    feat_key not in pair_matches[key]
                    or dist < pair_matches[key][feat_key]
                ):
                    pair_matches[key][feat_key] = dist

            offset += count

            # Collect TVG for this pair (keep the one with the most inliers)
            if has_tvg:
                ic = int(tvg_inlier_counts[k])
                existing = pair_tvg.get(key)
                if existing is None or ic > existing["inlier_count"]:
                    config_str = tvg_config_types[int(tvg_config_indexes[k])]
                    inlier_slice = tvg_inlier_feat_idxs[
                        inlier_offset : inlier_offset + ic
                    ].copy()
                    f_mat = tvg_f_matrices[k].copy()
                    e_mat = tvg_e_matrices[k].copy()
                    h_mat = tvg_h_matrices[k].copy()
                    quat = tvg_quaternions[k].copy()
                    trans = tvg_translations[k].copy()

                    if swap:
                        inlier_slice = inlier_slice[:, ::-1].copy()
                        f_mat = f_mat.T.copy()
                        e_mat = e_mat.T.copy()
                        if np.any(h_mat != 0):
                            try:
                                h_mat = np.linalg.inv(h_mat)
                            except np.linalg.LinAlgError:
                                h_mat = np.zeros((3, 3), dtype=np.float64)
                        # Invert pose: q_inv = (w, -x, -y, -z), t_inv = -R^T @ t
                        w, x, y, z = quat
                        R = np.array(
                            [
                                [
                                    1 - 2 * (y * y + z * z),
                                    2 * (x * y - z * w),
                                    2 * (x * z + y * w),
                                ],
                                [
                                    2 * (x * y + z * w),
                                    1 - 2 * (x * x + z * z),
                                    2 * (y * z - x * w),
                                ],
                                [
                                    2 * (x * z - y * w),
                                    2 * (y * z + x * w),
                                    1 - 2 * (x * x + y * y),
                                ],
                            ]
                        )
                        quat = np.array([w, -x, -y, -z])
                        trans = -R.T @ trans

                    pair_tvg[key] = {
                        "config_type": config_str,
                        "inlier_count": ic,
                        "inlier_feature_indexes": inlier_slice,
                        "f_matrix": f_mat,
                        "e_matrix": e_mat,
                        "h_matrix": h_mat,
                        "quaternion_wxyz": quat,
                        "translation_xyz": trans,
                    }
                inlier_offset += ic

    # Build output arrays
    sorted_pairs = sorted(pair_matches.keys())
    total_matches = sum(len(m) for m in pair_matches.values())
    click.echo(f"  Merged: {len(sorted_pairs)} pairs, {total_matches} matches")

    out_image_index_pairs = np.zeros((len(sorted_pairs), 2), dtype=np.uint32)
    out_match_counts = np.zeros(len(sorted_pairs), dtype=np.uint32)
    out_match_feature_indexes = np.zeros((total_matches, 2), dtype=np.uint32)
    out_match_descriptor_distances = np.zeros(total_matches, dtype=np.float32)

    offset = 0
    for k, (idx_i, idx_j) in enumerate(sorted_pairs):
        out_image_index_pairs[k] = [idx_i, idx_j]
        matches = pair_matches[(idx_i, idx_j)]
        out_match_counts[k] = len(matches)
        for (fi, fj), dist in matches.items():
            out_match_feature_indexes[offset] = [fi, fj]
            out_match_descriptor_distances[offset] = dist
            offset += 1

    # Build TVG output arrays if any input had TVGs
    has_output_tvg = len(pair_tvg) > 0
    if has_output_tvg:
        config_type_set = {"undefined"}
        for tvg in pair_tvg.values():
            config_type_set.add(tvg["config_type"])
        config_type_list = sorted(config_type_set)
        config_type_to_idx = {ct: i for i, ct in enumerate(config_type_list)}
        undefined_idx = config_type_to_idx["undefined"]

        out_config_indexes = np.full(len(sorted_pairs), undefined_idx, dtype=np.uint8)
        out_inlier_counts = np.zeros(len(sorted_pairs), dtype=np.uint32)
        out_f_matrices = np.zeros((len(sorted_pairs), 3, 3), dtype=np.float64)
        out_e_matrices = np.zeros((len(sorted_pairs), 3, 3), dtype=np.float64)
        out_h_matrices = np.zeros((len(sorted_pairs), 3, 3), dtype=np.float64)
        out_quaternions = np.zeros((len(sorted_pairs), 4), dtype=np.float64)
        out_quaternions[:, 0] = 1.0  # identity quaternion default
        out_translations = np.zeros((len(sorted_pairs), 3), dtype=np.float64)

        all_inlier_idxs = []
        total_inliers = 0
        tvg_pair_count = 0

        for k, (idx_i, idx_j) in enumerate(sorted_pairs):
            tvg = pair_tvg.get((idx_i, idx_j))
            if tvg is not None:
                out_config_indexes[k] = config_type_to_idx[tvg["config_type"]]
                out_inlier_counts[k] = tvg["inlier_count"]
                out_f_matrices[k] = tvg["f_matrix"]
                out_e_matrices[k] = tvg["e_matrix"]
                out_h_matrices[k] = tvg["h_matrix"]
                out_quaternions[k] = tvg["quaternion_wxyz"]
                out_translations[k] = tvg["translation_xyz"]
                if tvg["inlier_count"] > 0:
                    all_inlier_idxs.append(tvg["inlier_feature_indexes"])
                total_inliers += tvg["inlier_count"]
                tvg_pair_count += 1

        if all_inlier_idxs:
            out_inlier_feature_indexes = np.concatenate(all_inlier_idxs, axis=0)
        else:
            out_inlier_feature_indexes = np.zeros((0, 2), dtype=np.uint32)

        click.echo(
            f"  TVGs: {tvg_pair_count}/{len(sorted_pairs)} pairs, "
            f"{total_inliers} total inliers"
        )

    # Build unified image arrays
    feature_counts = np.array(
        [image_info[n]["feature_count"] for n in unified_names], dtype=np.uint32
    )
    feature_tool_hashes = np.array(
        [list(image_info[n]["tool_hash"]) for n in unified_names], dtype=np.uint8
    )
    sift_content_hashes = np.array(
        [list(image_info[n]["content_hash"]) for n in unified_names], dtype=np.uint8
    )

    # Build metadata from the first input file as a base
    base_meta = all_data[0]["metadata"]
    source_methods = []
    for data in all_data:
        method = data["metadata"].get("matching_method", "unknown")
        source_methods.append(method)

    merged_data = {
        "metadata": {
            "version": 1,
            "matching_method": "merged",
            "matching_tool": "sfmtool",
            "matching_tool_version": "",
            "matching_options": {
                "source_files": [p.name for p in input_paths],
                "source_methods": source_methods,
            },
            "workspace": base_meta["workspace"],
            "timestamp": datetime.now().astimezone().isoformat(),
            "image_count": image_count,
            "image_pair_count": len(sorted_pairs),
            "match_count": total_matches,
            "has_two_view_geometries": has_output_tvg,
        },
        "content_hash": {},
        "image_names": unified_names,
        "feature_tool_hashes": feature_tool_hashes,
        "sift_content_hashes": sift_content_hashes,
        "feature_counts": feature_counts,
        "image_index_pairs": out_image_index_pairs,
        "match_counts": out_match_counts,
        "match_feature_indexes": out_match_feature_indexes,
        "match_descriptor_distances": out_match_descriptor_distances,
        "has_two_view_geometries": has_output_tvg,
    }

    if has_output_tvg:
        merged_data["config_types"] = config_type_list
        merged_data["config_indexes"] = out_config_indexes
        merged_data["inlier_counts"] = out_inlier_counts
        merged_data["inlier_feature_indexes"] = out_inlier_feature_indexes
        merged_data["f_matrices"] = out_f_matrices
        merged_data["e_matrices"] = out_e_matrices
        merged_data["h_matrices"] = out_h_matrices
        merged_data["quaternions_wxyz"] = out_quaternions
        merged_data["translations_xyz"] = out_translations
        merged_data["tvg_metadata"] = {
            "image_pair_count": len(sorted_pairs),
            "inlier_count": total_inliers,
            "verification_tool": "merged",
            "verification_options": {
                "source_files": [p.name for p in input_paths],
            },
        }

    # Update workspace relative path for the output location
    out_abs = Path(os.path.abspath(out))
    ws_abs = base_meta["workspace"]["absolute_path"]
    merged_data["metadata"]["workspace"]["relative_path"] = os.path.relpath(
        ws_abs, out_abs.parent
    ).replace("\\", "/")

    out.parent.mkdir(parents=True, exist_ok=True)
    click.echo(f"Writing {out}...")
    write_matches(out, merged_data)
    click.echo(f"Done: {len(sorted_pairs)} pairs, {total_matches} matches")
