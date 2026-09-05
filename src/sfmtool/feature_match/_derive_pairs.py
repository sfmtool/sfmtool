# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""The COLMAP boundary artifact: `sfm match --derive-pairs`.

Turns a clusters-bearing `.matches` file into the verified pairwise +
two-view-geometry `.matches` file COLMAP's mapper needs. Two-view geometries
exist for that mapper — it reads its correspondence graph from the database's
two-view geometry table — and nothing in sfmtool's own pipeline consumes them,
so verification is a boundary concern rather than a matcher concern, and it
runs when a caller asks for it instead of on every match.
"""

import os
import tempfile
from datetime import datetime
from pathlib import Path

import click

from ._db_populate import _compute_descriptor_distances, _fill_sift_hashes


def _run_derive_pairs(
    matches_file: Path,
    output_path: str | None = None,
    camera_model: str | None = None,
) -> None:
    """Derive verified pairwise matches + TVGs from a clusters `.matches` file.

    The pairs come from the canonical cluster expansion, are written into a
    throwaway COLMAP database along with the workspace's features and cameras,
    verified by `pycolmap.verify_matches`, and read back with their two-view
    geometries. The output records the source file's matcher options plus a
    `derived_pairs` provenance record naming the clusters file it came from.
    """
    from importlib.metadata import version as get_version

    import pycolmap

    from .._sfmtool.io import read_colmap_db_matches, read_matches, write_matches
    from ..colmap.db_setup import (
        _setup_for_sfm_from_matches,
        resolve_image_and_sift_paths,
    )
    from ._pairs import pairs_from_matches

    matches_file = Path(matches_file)
    click.echo(f"Loading clusters from: {matches_file}")
    source = read_matches(matches_file)
    source_meta = source["metadata"]
    source_hash = source["content_hash"]["content_xxh128"]

    pairs_data = pairs_from_matches(source)
    source_names = list(source["image_names"])
    derived_pair_count = len(pairs_data["image_index_pairs"])
    click.echo(
        f"Derived {derived_pair_count} image pairs "
        f"({len(pairs_data['match_feature_indexes'])} matches) from "
        f"{source_meta.get('cluster_count', 0)} clusters"
    )
    if derived_pair_count == 0:
        raise RuntimeError(f"{matches_file} expands to no image pairs")

    with tempfile.TemporaryDirectory(prefix="sfm_derive_pairs_") as tmpdir:
        colmap_dir = Path(tmpdir)
        db_path, workspace_dir, _image_paths, _rig_used = _setup_for_sfm_from_matches(
            matches_file,
            colmap_dir,
            camera_model=camera_model,
            matches_data=source,
        )

        # Verification is driven by an explicit pair list, so only the pairs
        # the clusters actually produced are estimated.
        pairs_path = colmap_dir / "derived_pairs.txt"
        with open(pairs_path, "w") as pairs_file:
            for idx_i, idx_j in pairs_data["image_index_pairs"]:
                pairs_file.write(
                    f"{source_names[int(idx_i)]} {source_names[int(idx_j)]}\n"
                )

        click.echo("Running geometric verification...")
        pycolmap.verify_matches(
            str(db_path), str(pairs_path), options=pycolmap.TwoViewGeometryOptions()
        )

        click.echo("Reading matches from database...")
        matches_data = read_colmap_db_matches(str(db_path), include_tvg=True)

    # The Rust reader sorts images lexicographically from the DB; resolve the
    # per-image files in that order so every array stays parallel.
    image_names = list(matches_data["image_names"])
    ws_contents = dict(source_meta["workspace"].get("contents", {}))
    image_paths, sift_paths = resolve_image_and_sift_paths(
        workspace_dir, image_names, ws_contents.get("feature_prefix_dir", "") or ""
    )

    source_options = dict(source_meta.get("matching_options") or {})
    max_feature_count = source_options.get("max_feature_count")

    click.echo("Computing descriptor distances...")
    _compute_descriptor_distances(matches_data, sift_paths, max_feature_count)

    if output_path:
        out = Path(output_path)
    else:
        stem = matches_file.stem
        suffix = "-clusters"
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
        out = workspace_dir / "tvg-matches" / f"{stem}.matches"
    out_abs = Path(os.path.abspath(out))
    out_abs.parent.mkdir(parents=True, exist_ok=True)

    source_options["derived_pairs"] = {
        "source_path": os.path.relpath(
            os.path.abspath(matches_file), out_abs.parent
        ).replace("\\", "/"),
        "source_content_xxh128": source_hash,
    }

    metadata = matches_data["metadata"]
    metadata["version"] = 1
    metadata["matching_method"] = source_meta.get("matching_method", "cluster")
    metadata["matching_tool"] = "sfmtool"
    metadata["matching_tool_version"] = get_version("sfmtool")
    metadata["matching_options"] = source_options
    metadata["workspace"] = {
        "absolute_path": str(workspace_dir),
        "relative_path": os.path.relpath(workspace_dir, out_abs.parent).replace(
            "\\", "/"
        ),
        "contents": {
            "feature_tool": ws_contents.get("feature_tool", "colmap"),
            "feature_type": ws_contents.get("feature_type", "sift"),
            "feature_options": ws_contents.get("feature_options") or {},
            "feature_prefix_dir": ws_contents.get("feature_prefix_dir") or "",
        },
    }
    metadata["timestamp"] = datetime.now().astimezone().isoformat()

    _fill_sift_hashes(matches_data, sift_paths, image_names, image_paths)

    click.echo(f"Writing {out}...")
    write_matches(out, matches_data)

    pair_count = metadata["image_pair_count"]
    match_count = metadata["match_count"]
    click.echo(f"Done: {pair_count} pairs, {match_count} matches")
    if matches_data["has_two_view_geometries"]:
        inlier_count = matches_data["tvg_metadata"]["inlier_count"]
        click.echo(f"  Two-view geometries: {inlier_count} total inliers")
