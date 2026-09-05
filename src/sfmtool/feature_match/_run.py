# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Matching orchestration for `sfm match`.

Drives descriptor and flow-based matching to a `.matches` file. Extracted from
`_commands/match.py` so the command module stays a thin Click wrapper; the
database/descriptor bookkeeping these routines depend on lives in
`_db_populate.py`, and merging several `.matches` files into one lives in
`_merge.py`.
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
):
    """Run matching and produce a .matches file.

    Every method writes exactly one file. ``--exhaustive``, ``--sequential``
    and ``--flow`` go through a COLMAP database and emit a verified
    pairwise+TVG `.matches`; ``--cluster`` emits the clusters-bearing
    `.matches` and touches no database at all, which is what makes it
    deterministic. The verified pairwise+TVG derivative of a cluster file is a
    COLMAP-boundary artifact produced on demand by
    :func:`._derive_pairs._run_derive_pairs`.
    """
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

    if matching_method == "cluster":
        # Sort images lexicographically by workspace-relative name so the
        # cluster corpus order matches the order every `.matches` reader uses,
        # and image indices stay comparable with any pairwise file derived
        # from this one.
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
        if output_path:
            out = Path(output_path)
        else:
            out = _generate_output_path(
                workspace_dir / "matches",
                image_paths,
                matching_method,
                stem_suffix="-clusters",
            )

        click.echo(f"Running {matching_method} matching...")
        clusters, _pairs = _materialize_clusters(
            image_paths,
            sift_paths,
            max_feature_count=max_feature_count,
            d=cluster_d,
            alpha=cluster_alpha,
            min_size=cluster_min_size,
            preset=cluster_preset,
        )
        _write_clusters_matches(
            clusters,
            out,
            image_paths=image_paths,
            sift_paths=sift_paths,
            image_names=image_names,
            workspace_dir=workspace_dir,
            ws_config=ws_config,
            matcher_options=matcher_options,
            max_feature_count=max_feature_count,
        )
        return

    import pycolmap

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
            # The flow matcher matches outside the database and uses it only
            # for pycolmap geometric verification, which never reads
            # descriptors — skip writing them (the largest rows by far). Fail
            # safe for any future method: only the known DB-external matchers
            # opt out.
            include_descriptors=matching_method != "flow",
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
    from .._sfmtool.io import write_matches

    click.echo(f"Writing {out}...")
    write_matches(out, matches_data)

    pair_count = matches_data["metadata"]["image_pair_count"]
    match_count = matches_data["metadata"]["match_count"]
    click.echo(f"Done: {pair_count} pairs, {match_count} matches")
    if matches_data["has_two_view_geometries"]:
        inlier_count = matches_data["tvg_metadata"]["inlier_count"]
        click.echo(f"  Two-view geometries: {inlier_count} total inliers")


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
    """Write the cluster matcher's artifact: a clusters-bearing `.matches`
    file (clusters backbone, no pairs, no TVGs).

    This is the whole output of `sfm match --cluster`. `image_names` must be
    in corpus order (the order `member_images` indexes).
    """
    from concurrent.futures import ThreadPoolExecutor
    from importlib.metadata import version as get_version

    from .._sfmtool.io import read_sift_metadata, read_sift_partial, write_matches

    cluster_count = len(clusters.cluster_starts) - 1
    member_count = len(clusters.member_images)
    member_images = np.asarray(clusters.member_images)
    member_features = np.asarray(clusters.member_features)

    def _read_one(i: int, sift_path: Path):
        """That image's feature count as used to build the corpus, and its
        members' detected geometry gathered from the `.sift` rows their
        feature indexes name."""
        n = int(read_sift_metadata(str(sift_path))["metadata"]["feature_count"])
        if max_feature_count:
            n = min(n, max_feature_count)
        on_image = member_images == i
        feats = member_features[on_image]
        if len(feats) == 0:
            return n, on_image, None, None
        sift = read_sift_partial(str(sift_path), int(feats.max()) + 1)
        # Gathered, never converted: the arrays are already float32 and the
        # backbone stores the same bits the .sift holds.
        return n, on_image, sift["positions_xy"][feats], sift["affine_shapes"][feats]

    # Decode through a thread pool (the .sift reader releases the GIL).
    with ThreadPoolExecutor() as pool:
        per_image = list(pool.map(_read_one, range(len(sift_paths)), sift_paths))

    # Per-image feature counts as used to build the corpus: the .sift file's
    # count capped at max_feature_count, so member_features indices line up.
    feature_counts = np.array([n for n, _, _, _ in per_image], dtype=np.uint32)
    # The backbone's stage geometry. A matcher output's stage is detection,
    # so these are the .sift values the feature indexes stand for; a consumer
    # reads a member's position and extent without opening a .sift file.
    member_positions = np.zeros((member_count, 2), dtype=np.float32)
    member_affine_shapes = np.zeros((member_count, 2, 2), dtype=np.float32)
    for _, on_image, positions, affine_shapes in per_image:
        if positions is None:
            continue
        member_positions[on_image] = positions
        member_affine_shapes[on_image] = affine_shapes

    matching_options = dict(matcher_options)
    if max_feature_count:
        matching_options["max_feature_count"] = max_feature_count

    out_abs = Path(os.path.abspath(out_path))
    data = {
        "metadata": {
            "version": 6,
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
        "member_positions": member_positions,
        "member_affine_shapes": member_affine_shapes,
        "matcher_options": matching_options,
        "has_cluster_patches": False,
        "has_two_view_geometries": False,
    }
    _fill_sift_hashes(data, sift_paths, image_names, image_paths)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    click.echo(f"Writing {out_path}...")
    write_matches(out_path, data)
    click.echo(f"Done: {cluster_count} clusters, {member_count} members")


def _generate_output_path(
    base_dir: Path,
    image_paths: list[Path],
    matching_method: str,
    stem_suffix: str = "",
) -> Path:
    """Generate a timestamped output path for a .matches file.

    ``stem_suffix`` is appended to the generated stem before the extension,
    so a method whose artifact is not the pairwise one can label it (the
    cluster matcher passes ``"-clusters"``).
    """
    from deadline.job_attachments.api import summarize_paths_by_sequence

    from .._sfmtool.reconstruction import RangeExpr

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
    filename = "-".join(parts) + stem_suffix + ".matches"

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


def _materialize_clusters(
    image_paths: list[Path],
    sift_paths: list[Path],
    *,
    max_feature_count: int | None = None,
    d: int = 10,
    alpha: float = 0.8,
    min_size: int = 2,
    preset: str = "accurate",
):
    """Run the background-floor track-cluster matcher and report its size.

    Builds one descriptor corpus from every image's SIFT features, materializes
    track clusters with the per-point background floor, and returns the
    ``(ClusterSet, PairMatches)`` pair the matcher produces — the clusters are
    the matcher's artifact, the pair expansion its canonical pairwise view.
    Both `sfm match --cluster` and the in-solve matching mode go through here,
    so they report identically and cluster identically.
    """
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
    click.echo(
        f"Materialized {cluster_count} track clusters: "
        f"{len(pairs.match_feature_indexes)} candidate matches "
        f"across {len(pairs.image_index_pairs)} image pairs"
    )
    return clusters, pairs


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
) -> None:
    """Run background-floor track-cluster matching and write results to the DB.

    Materializes track clusters, expands them into per-image-pair matches,
    writes those to the database, and runs geometric verification via
    pycolmap. This is the in-solve matching mode (`sfm solve` from images with
    `matching_mode="cluster"`), which needs the matches and two-view
    geometries in the solve's own database; `sfm match --cluster` writes the
    clusters to a `.matches` file instead and never opens a database.

    ``exclude_index_pairs`` is a set of normalized ``(i, j)`` image-index pairs
    (indices into ``image_paths``) to drop from the output — used for
    multi-sensor rigs to suppress the spurious same-frame matches that
    back-to-back sensors with no shared view produce, which the clustering
    cannot know to avoid on descriptors alone.
    """
    import pycolmap

    _clusters, pairs = _materialize_clusters(
        image_paths,
        sift_paths,
        max_feature_count=max_feature_count,
        d=d,
        alpha=alpha,
        min_size=min_size,
        preset=preset,
    )
    pair_count = len(pairs.image_index_pairs)
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
