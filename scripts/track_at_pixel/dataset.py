# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Prepare a ground-truth reconstruction for the track-at-pixel harness.

The harness needs four things the checked-in ground truth does not carry: a
workspace it may write into, a ``.sift`` file per image (the constellation
query is over detected keypoints), a ``.kdf`` descriptor index over those files
in the reconstruction's own image order, so a match names a reconstruction image
directly, and a cluster-patches ``.matches`` file built from that same index.
``prepare`` builds all four in a cache directory, never beside the source
data, and reuses them on later runs. The source workspace is the nearest
directory at or above the ``.sfmr`` that holds a ``.sfm-workspace.json``; the
``.sfmr`` is copied to the same path relative to the cache workspace, so that
the copy resolves its workspace to the cache, and ``prepare`` stops if it does
not. A ``.matches`` file supplied by the caller
is used as given instead of the built one.
"""

from __future__ import annotations

import json
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]

# Name -> the ground-truth .sfmr. It must lie in a workspace: a directory at or
# above it holding the images and the .sfm-workspace.json marker that says how
# their features are extracted.
DATASETS = {
    "seoul_bull": REPO
    / "test-data/images/seoul_bull_sculpture/seoul_bull_sculpture_ground_truth_v01.sfmr",
}


@dataclass
class PreparedDataset:
    name: str
    ground_truth: Path  # the source .sfmr the workspace was copied from
    workspace: Path
    source_workspace: Path  # the source's workspace, read only
    sfmr: Path
    kdf: Path
    matches: Path  # clusters + cluster-patches, over the same images


def _copy_if_stale(source: Path, dest: Path) -> None:
    if not dest.exists() or dest.stat().st_mtime < source.stat().st_mtime:
        shutil.copy2(source, dest)


def default_cache_dir() -> Path:
    return Path(tempfile.gettempdir()) / "sfmtool-track-at-pixel"


def prepare(
    name_or_path: str,
    cache_dir: Path | None = None,
    *,
    matches: Path | None = None,
    quiet: bool = False,
) -> PreparedDataset:
    """Copy the ground truth into a writable workspace, extract SIFT, build the index."""
    from sfmtool.reconstruction import SfmrReconstruction

    gt = Path(DATASETS.get(name_or_path, name_or_path)).resolve()
    if not gt.is_file():
        raise SystemExit(f"no ground truth at {gt} (known datasets: {list(DATASETS)})")
    name = name_or_path if name_or_path in DATASETS else gt.stem
    source_ws = next(
        (
            d
            for d in [gt.parent, *gt.parent.parents]
            if (d / ".sfm-workspace.json").is_file()
        ),
        None,
    )
    if source_ws is None:
        raise SystemExit(f"{gt} has no .sfm-workspace.json at or above it")
    marker = source_ws / ".sfm-workspace.json"

    workspace = (cache_dir or default_cache_dir()) / name
    workspace.mkdir(parents=True, exist_ok=True)
    # The same path relative to the workspace, so a stored relative workspace
    # path (``..`` for a file in a subdirectory) lands on the cache.
    sfmr = workspace / gt.relative_to(source_ws)
    sfmr.parent.mkdir(parents=True, exist_ok=True)
    # Copy only what is missing or stale, so runs started in parallel against
    # one cache never write a file another is reading.
    _copy_if_stale(gt, sfmr)
    _copy_if_stale(marker, workspace / marker.name)
    for extra in ("camera_config.json", "rig_config.json"):
        if (source_ws / extra).is_file():
            _copy_if_stale(source_ws / extra, workspace / extra)

    recon = SfmrReconstruction.load(sfmr)
    # The copy must resolve its workspace to the cache. One that fell back to
    # the source's stored absolute path would read the source's workspace,
    # and anything written beside its images would land there.
    if Path(recon.workspace_dir).resolve() != workspace.resolve():
        raise SystemExit(
            f"the copy at {sfmr} resolves its workspace to {recon.workspace_dir}, "
            f"not the cache workspace {workspace}; stopping"
        )
    names = list(recon.image_names)
    image_paths = []
    for image_name in names:
        dest = workspace / image_name
        if not dest.exists():
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_ws / image_name, dest)
        image_paths.append(dest)

    config = json.loads(marker.read_text())
    _extract_sift(image_paths, config, quiet=quiet)

    kdf = workspace / "track_at_pixel_index.kdf"
    if not kdf.exists():
        _build_kdf(workspace, names, config, kdf)

    if matches is None:
        matches = workspace / "track_at_pixel_clusters-patches.matches"
        if not matches.exists():
            _build_cluster_patches(workspace, image_paths, names, config, kdf, matches)
    elif not Path(matches).is_file():
        raise SystemExit(f"no .matches file at {matches}")
    return PreparedDataset(
        name=name,
        ground_truth=gt,
        workspace=workspace,
        source_workspace=source_ws,
        sfmr=sfmr,
        kdf=kdf,
        matches=Path(matches),
    )


def _extract_sift(image_paths: list[Path], config: dict, *, quiet: bool) -> None:
    from sfmtool.sift.file import get_sift_path_for_image
    from sfmtool.sift.extract import image_files_to_sift_files

    missing = [p for p in image_paths if not get_sift_path_for_image(p).exists()]
    if not missing:
        return
    if not quiet:
        print(f"extracting SIFT for {len(missing)} image(s)")
    image_files_to_sift_files(
        [str(p) for p in missing],
        feature_tool=config["feature_tool"],
        feature_options=config["feature_options"],
        feature_prefix_dir=config["feature_prefix_dir"],
    )


def read_keypoints(workspace: Path, image_name: str):
    """``(positions (N,2) f32, affine_shapes (N,2,2) f32, sift_path)`` for one image."""
    from sfmtool.sift.file import get_sift_path_for_image
    from sfmtool.fileio import SiftFile

    path = get_sift_path_for_image(workspace / image_name)
    reader = SiftFile(path)
    xy, affine = reader.read_positions_and_shapes()
    reader.close()
    return (
        np.ascontiguousarray(xy, dtype=np.float32),
        np.ascontiguousarray(affine, dtype=np.float32),
        path,
    )


def _build_kdf(workspace: Path, names: list[str], config: dict, out: Path) -> None:
    """One corpus image per reconstruction image, in the reconstruction's order."""
    from sfmtool.spatial import KdForest, write_kdf
    from sfmtool.sift.file import get_sift_path_for_image
    from sfmtool.fileio import SiftFile

    descriptors, positions, shapes, image_of, feature_of = [], [], [], [], []
    for index, image_name in enumerate(names):
        reader = SiftFile(get_sift_path_for_image(workspace / image_name))
        rows = np.asarray(reader.read_descriptors())
        xy, affine = reader.read_positions_and_shapes()
        reader.close()
        descriptors.append(rows)
        positions.append(np.asarray(xy, dtype=np.float32))
        shapes.append(np.asarray(affine, dtype=np.float32))
        image_of.append(np.full(len(rows), index, dtype=np.uint32))
        feature_of.append(np.arange(len(rows), dtype=np.uint32))

    sources = {
        "workspace": {
            "absolute_path": str(workspace),
            "relative_path": ".",
            "contents": {
                "feature_tool": config["feature_tool"],
                "feature_type": config["feature_type"],
                "feature_options": json.dumps(config["feature_options"]),
                "feature_prefix_dir": config["feature_prefix_dir"],
            },
        },
        "image_names": names,
        "feature_tool_hashes": [bytes(16)] * len(names),
        "sift_content_hashes": [bytes(16)] * len(names),
        "image_indexes": np.concatenate(image_of).tolist(),
        "image_feature_indexes": np.concatenate(feature_of).tolist(),
        "positions": np.vstack(positions),
        "affine_shapes": np.vstack(shapes),
    }
    forest = KdForest(np.vstack(descriptors), num_trees=4, leaf_size=16, seed=5)
    write_kdf(forest, str(out), sources=sources)


def _build_cluster_patches(
    workspace: Path,
    image_paths: list[Path],
    names: list[str],
    config: dict,
    kdf: Path,
    out: Path,
) -> None:
    """Cluster from the ``.kdf`` itself, then refine the clusters into patches.

    The clustering is ``sfm match --cluster``'s background-floor clustering run
    over the index rather than over a fresh forest. The file is written the way
    that command writes its own, by ``sfmtool.fileio.write_clusters_matches``
    with each member's detected position and affine shape read from its
    ``.sift`` row, and refined by running ``sfm cluster-patches`` on it with
    its default options. The index's feature ids run image by image in
    ``names`` order, each image's features in ``.sift`` row order, which is
    what makes a cluster member's feature index a ``.sift`` row.
    """
    import os
    import subprocess
    from datetime import datetime
    from importlib.metadata import version

    from sfmtool.fileio import (
        read_sift_metadata,
        read_sift_partial,
        write_clusters_matches,
    )
    from sfmtool.matching import background_floor_clusters_kdf
    from sfmtool.sift.file import get_sift_path_for_image

    sift_paths = [get_sift_path_for_image(p) for p in image_paths]
    sift_metadata = [read_sift_metadata(str(p)) for p in sift_paths]
    counts = [int(m["metadata"]["feature_count"]) for m in sift_metadata]
    image_starts = np.concatenate([[0], np.cumsum(counts)]).astype(np.uint32)
    options = {"d": 10, "alpha": 0.8, "min_size": 2}
    starts, images, features = background_floor_clusters_kdf(
        str(kdf), image_starts, **options
    )
    images, features = np.asarray(images), np.asarray(features)
    matcher_options = {**options, "index": kdf.name}

    # Each member's detected geometry, gathered from its image's .sift rows.
    member_positions = np.zeros((len(images), 2), dtype=np.float32)
    member_affine_shapes = np.zeros((len(images), 2, 2), dtype=np.float32)
    for i, sift_path in enumerate(sift_paths):
        on_image = images == i
        if not on_image.any():
            continue
        rows = features[on_image]
        sift = read_sift_partial(str(sift_path), int(rows.max()) + 1)
        member_positions[on_image] = sift["positions_xy"][rows]
        member_affine_shapes[on_image] = sift["affine_shapes"][rows]

    def hashes(key):
        return [bytes.fromhex(m["content_hash"][key]) for m in sift_metadata]

    clusters_path = out.with_name("track_at_pixel_clusters.matches")
    if clusters_path.exists():
        clusters_path.unlink()
    write_clusters_matches(
        clusters_path,
        matching_method="cluster",
        matching_tool="sfmtool",
        matching_tool_version=version("sfmtool"),
        matching_options=matcher_options,
        workspace={
            "absolute_path": str(workspace),
            "relative_path": Path(
                os.path.relpath(workspace, clusters_path.parent)
            ).as_posix(),
            "contents": {
                "feature_tool": config.get("feature_tool", "colmap"),
                "feature_type": config.get("feature_type", "sift"),
                "feature_options": config.get("feature_options") or {},
                "feature_prefix_dir": config.get("feature_prefix_dir") or "",
            },
        },
        timestamp=datetime.now().astimezone().isoformat(),
        image_names=names,
        feature_tool_hashes=hashes("feature_tool_xxh128"),
        sift_content_hashes=hashes("content_xxh128"),
        feature_counts=np.array(counts, dtype=np.uint32),
        image_dims=np.array(
            [
                (m["metadata"]["image_width"], m["metadata"]["image_height"])
                for m in sift_metadata
            ],
            dtype=np.uint32,
        ),
        cluster_starts=starts,
        member_images=images,
        member_features=features,
        member_positions=member_positions,
        member_affine_shapes=member_affine_shapes,
        matcher_options=matcher_options,
    )
    subprocess.run(
        ["sfm", "cluster-patches", "-i", str(clusters_path), "-o", str(out)],
        check=True,
    )
