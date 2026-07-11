# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Matching orchestration for `sfm match`.

Drives descriptor and flow-based matching to a `.matches` file, and merges
several `.matches` files into one. Extracted from `_commands/match.py` so the
command module stays a thin Click wrapper; the database/descriptor bookkeeping
these routines depend on lives in `_db_populate.py`.
"""

import os
import re
import tempfile
from datetime import datetime
from pathlib import Path

import click
import numpy as np

from ._db_populate import (
    _compute_descriptor_distances,
    _fill_sift_hashes,
    _populate_db_features,
)


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
    cluster_d: int = 10,
    cluster_alpha: float = 0.8,
    cluster_min_size: int = 2,
    cluster_preset: str = "accurate",
    clusters_output: str | None = None,
):
    """Run matching and produce a .matches file.

    For ``--cluster``, two files are written: the clusters-bearing `.matches`
    (the matcher's durable primary artifact, written before geometric
    verification) and the verified pairwise+TVG `.matches` the solver
    consumes. ``output_path`` keeps its meaning as the verified output;
    ``clusters_output`` overrides the cluster file's default location
    (``matches/<verified stem>-clusters.matches`` under the workspace).
    """
    import pycolmap

    from .._workspace import load_workspace_config
    from ..sift.file import image_files_to_sift_files

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

    matcher_options = None
    verified_out = None
    clusters_out = None
    if matching_method == "cluster":
        # Sort images lexicographically by workspace-relative name so the
        # cluster corpus order matches the DB reader's order — the cluster
        # file and the verified pairwise file then share image indices.
        order = sorted(range(len(image_names)), key=lambda i: image_names[i])
        image_names = [image_names[i] for i in order]
        image_paths = [image_paths[i] for i in order]
        sift_paths = [sift_paths[i] for i in order]

        matcher_options = {
            "mode": "background-floor",
            "d": cluster_d,
            "alpha": cluster_alpha,
            "min_size": cluster_min_size,
            "preset": cluster_preset,
        }
        # The verified output always carries TVGs, so its path is known up
        # front; the cluster file's default name derives from it.
        if output_path:
            verified_out = Path(output_path)
        else:
            verified_out = _generate_output_path(
                workspace_dir / "tvg-matches", image_paths, matching_method
            )
        if clusters_output:
            clusters_out = Path(clusters_output)
        else:
            clusters_out = (
                workspace_dir / "matches" / f"{verified_out.stem}-clusters.matches"
            )

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
            # The cluster and flow matchers match outside the database and use
            # it only for pycolmap geometric verification, which never reads
            # descriptors — skip writing them (the largest rows by far). Fail
            # safe for any future method: only the known DB-external matchers
            # opt out.
            include_descriptors=matching_method not in ("cluster", "flow"),
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
        elif matching_method == "cluster":

            def _persist_clusters(clusters):
                _write_clusters_matches(
                    clusters,
                    clusters_out,
                    image_paths=image_paths,
                    sift_paths=sift_paths,
                    image_names=image_names,
                    workspace_dir=workspace_dir,
                    ws_config=ws_config,
                    matcher_options=matcher_options,
                    max_feature_count=max_feature_count,
                )

            _run_cluster_matching(
                image_paths,
                sift_paths,
                workspace_dir,
                db_path,
                Path(tmpdir),
                max_feature_count=max_feature_count,
                d=cluster_d,
                alpha=cluster_alpha,
                min_size=cluster_min_size,
                preset=cluster_preset,
                on_clusters=_persist_clusters,
            )
        else:
            raise ValueError(f"Unsupported matching method: {matching_method}")

        # Read matches + TVGs back from the DB
        click.echo("Reading matches from database...")
        from .._sfmtool.io import read_colmap_db_matches

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
    elif matching_method == "cluster":
        from importlib.metadata import version as get_version

        matches_data["metadata"]["matching_tool"] = "sfmtool"
        matches_data["metadata"]["matching_tool_version"] = get_version("sfmtool")
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
    if matching_method == "cluster":
        matches_data["metadata"]["matching_options"].update(
            {
                "mode": "background-floor",
                "d": cluster_d,
                "alpha": cluster_alpha,
                "min_size": cluster_min_size,
                "preset": cluster_preset,
            }
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
    if verified_out is not None:
        out = verified_out
    elif output_path:
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
    from .._sfmtool.io import write_matches

    click.echo(f"Writing {out}...")
    write_matches(out, matches_data)

    pair_count = matches_data["metadata"]["image_pair_count"]
    match_count = matches_data["metadata"]["match_count"]
    click.echo(f"Done: {pair_count} pairs, {match_count} matches")
    if matches_data["has_two_view_geometries"]:
        inlier_count = matches_data["tvg_metadata"]["inlier_count"]
        click.echo(f"  Two-view geometries: {inlier_count} total inliers")
    if clusters_out is not None:
        click.echo(f"  Clusters artifact: {clusters_out}")
        click.echo(f"  Verified matches: {out}")


def _write_clusters_matches(
    clusters,
    out_path: Path,
    *,
    image_paths: list[Path],
    sift_paths: list[Path],
    image_names: list[str],
    workspace_dir: Path,
    ws_config: dict,
    matcher_options: dict,
    max_feature_count: int | None,
) -> None:
    """Write the cluster matcher's primary artifact: a clusters-bearing
    `.matches` file (clusters backbone, no pairs, no TVGs).

    Called from the matching flow after cluster materialization and before
    geometric verification, so the durable artifact exists even if
    verification fails. `image_names` must be in corpus order (the order
    `member_images` indexes).
    """
    from importlib.metadata import version as get_version

    from .._sfmtool.io import read_sift_metadata, write_matches

    cluster_count = len(clusters.cluster_starts) - 1
    member_count = len(clusters.member_images)

    # Per-image feature counts as used to build the corpus: the .sift file's
    # count capped at max_feature_count, so member_features indices line up.
    feature_counts = np.zeros(len(sift_paths), dtype=np.uint32)
    for i, sift_path in enumerate(sift_paths):
        n = int(read_sift_metadata(str(sift_path))["metadata"]["feature_count"])
        if max_feature_count:
            n = min(n, max_feature_count)
        feature_counts[i] = n

    matching_options = dict(matcher_options)
    if max_feature_count:
        matching_options["max_feature_count"] = max_feature_count

    out_abs = Path(os.path.abspath(out_path))
    data = {
        "metadata": {
            "version": 3,
            "matching_method": "cluster",
            "matching_tool": "sfmtool",
            "matching_tool_version": get_version("sfmtool"),
            "matching_options": matching_options,
            "workspace": {
                "absolute_path": str(workspace_dir),
                "relative_path": os.path.relpath(workspace_dir, out_abs.parent).replace(
                    "\\", "/"
                ),
                "contents": {
                    "feature_tool": ws_config.get("feature_tool", "colmap"),
                    "feature_type": ws_config.get("feature_type", "sift"),
                    "feature_options": ws_config.get("feature_options") or {},
                    "feature_prefix_dir": ws_config.get("feature_prefix_dir") or "",
                },
            },
            "timestamp": datetime.now().astimezone().isoformat(),
            "image_count": len(image_names),
            "cluster_count": cluster_count,
            "cluster_member_count": member_count,
            "has_two_view_geometries": False,
            "has_clusters": True,
            "has_cluster_patches": False,
        },
        "image_names": image_names,
        "feature_counts": feature_counts,
        "has_clusters": True,
        "cluster_starts": clusters.cluster_starts,
        "member_images": clusters.member_images,
        "member_features": clusters.member_features,
        "matcher_options": matching_options,
        "has_cluster_patches": False,
        "has_two_view_geometries": False,
    }
    _fill_sift_hashes(data, sift_paths, image_names, image_paths)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    click.echo(f"Writing {out_path}...")
    write_matches(out_path, data)
    click.echo(
        f"  Clusters: {cluster_count} clusters, {member_count} members "
        "(primary artifact, written before verification)"
    )


def _generate_output_path(
    base_dir: Path, image_paths: list[Path], matching_method: str
) -> Path:
    """Generate a timestamped output path for a .matches file."""
    from deadline.job_attachments.api import summarize_paths_by_sequence

    from .._sfmtool import RangeExpr

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
        range_str = str(RangeExpr.from_list(sorted(summary.index_set)))
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

    from ._flow_matching import flow_match_sequential

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


def _run_cluster_matching(
    image_paths: list[Path],
    sift_paths: list[Path],
    workspace_dir: Path,
    db_path: Path,
    colmap_dir: Path,
    max_feature_count: int | None = None,
    d: int = 10,
    alpha: float = 0.8,
    min_size: int = 2,
    preset: str = "accurate",
    exclude_index_pairs: set[tuple[int, int]] | None = None,
    on_clusters=None,
) -> None:
    """Run background-floor track-cluster matching and write results to the DB.

    Builds one descriptor corpus from every image's SIFT features, materializes
    track clusters with the per-point background floor, expands them into
    per-image-pair matches, writes those to the database, and runs geometric
    verification via pycolmap. The clusters themselves are the matcher's
    primary artefact; ``on_clusters``, when given, is called with the
    materialized ``ClusterSet`` before any pair expansion is written or
    verified — `sfm match --cluster` uses it to persist the clusters-bearing
    `.matches` file.

    ``exclude_index_pairs`` is a set of normalized ``(i, j)`` image-index pairs
    (indices into ``image_paths``) to drop from the output — used for
    multi-sensor rigs to suppress the spurious same-frame matches that
    back-to-back sensors with no shared view produce, which the clustering
    cannot know to avoid on descriptors alone. The exclusion applies to the
    pair expansion only, never to the clusters handed to ``on_clusters`` —
    the stored clusters are the raw matcher output.
    """
    import pycolmap

    from ._cluster_matching import cluster_match

    clusters, pairs = cluster_match(
        image_paths,
        sift_paths,
        d=d,
        alpha=alpha,
        min_size=min_size,
        preset=preset,
        max_feature_count=max_feature_count,
    )
    cluster_count = len(clusters.cluster_starts) - 1
    pair_count = len(pairs.image_index_pairs)
    click.echo(
        f"Materialized {cluster_count} track clusters: "
        f"{len(pairs.match_feature_indexes)} candidate matches "
        f"across {pair_count} image pairs"
    )
    if on_clusters is not None:
        on_clusters(clusters)
    if pair_count == 0:
        click.echo("Warning: Cluster matching produced no matches")
        return

    # Map image index (corpus order) -> database image_id
    with pycolmap.Database.open(db_path) as db:
        rel_to_id = {img.name: img.image_id for img in db.read_all_images()}
    rel_names = [
        os.path.relpath(p, workspace_dir).replace("\\", "/") for p in image_paths
    ]

    # Write matches to database and build pairs file for geometric verification
    pairs_path = colmap_dir / "cluster_pairs.txt"
    match_offset = 0
    excluded = 0
    with (
        pycolmap.Database.open(db_path) as db,
        open(pairs_path, "w") as pairs_file,
    ):
        for k in range(pair_count):
            idx_i = int(pairs.image_index_pairs[k, 0])
            idx_j = int(pairs.image_index_pairs[k, 1])
            count = int(pairs.match_counts[k])
            matches_slice = pairs.match_feature_indexes[
                match_offset : match_offset + count
            ]
            match_offset += count

            if exclude_index_pairs and (idx_i, idx_j) in exclude_index_pairs:
                excluded += 1
                continue

            rel_i, rel_j = rel_names[idx_i], rel_names[idx_j]
            if rel_i not in rel_to_id or rel_j not in rel_to_id:
                continue
            db.write_matches(rel_to_id[rel_i], rel_to_id[rel_j], matches_slice)
            pairs_file.write(f"{rel_i} {rel_j}\n")

    if excluded:
        click.echo(f"Excluded {excluded} same-frame rig image pairs from matching")

    # Run geometric verification on matched pairs
    click.echo("Running geometric verification...")
    tvg_options = pycolmap.TwoViewGeometryOptions()
    pycolmap.verify_matches(str(db_path), str(pairs_path), options=tvg_options)


def _run_merge(paths, output_path):
    """Merge multiple .matches files into a single file.

    Builds a unified image list, remaps pair indexes, concatenates matches
    for each pair, and deduplicates by feature index pair (keeping the match
    with the lowest descriptor distance). Two-view geometry is preserved: for
    each pair the best available TVG is carried into the output, and the
    output's has_two_view_geometries reflects whether any survived.
    """
    from datetime import datetime

    from .._sfmtool.io import read_matches, write_matches

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

    # Read all input files. Pairwise arrays come through the derived-pairs
    # helper, so cluster-bearing inputs merge like any other (their expanded
    # matches carry NaN descriptor distances).
    from ._pairs import pairs_from_matches

    click.echo(f"Merging {len(input_paths)} .matches files...")
    all_data = []
    for path in input_paths:
        click.echo(f"  Reading {path.name}...")
        data = read_matches(str(path))
        pairs_view = pairs_from_matches(data)
        click.echo(
            f"    {data['metadata']['image_count']} images, "
            f"{len(pairs_view['image_index_pairs'])} pairs, "
            f"{len(pairs_view['match_feature_indexes'])} matches"
        )
        all_data.append((data, pairs_view))

    # Build unified image list and validate consistency
    image_info = {}  # name -> {feature_count, tool_hash, content_hash}
    for data, _pairs_view in all_data:
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

    for data, pairs_view in all_data:
        names = list(data["image_names"])
        local_to_unified = [name_to_idx[n] for n in names]
        has_tvg = data.get("has_two_view_geometries", False)

        pairs = pairs_view["image_index_pairs"]
        counts = pairs_view["match_counts"]
        feat_idxs = pairs_view["match_feature_indexes"]
        distances = pairs_view["match_descriptor_distances"]

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
                # Keep the match with the lowest descriptor distance; a known
                # distance beats NaN (cluster-derived matches carry NaN).
                prev = pair_matches[key].get(feat_key)
                if prev is None or (
                    not np.isnan(dist) and (np.isnan(prev) or dist < prev)
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
    base_meta = all_data[0][0]["metadata"]
    source_methods = []
    for data, _pairs_view in all_data:
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
