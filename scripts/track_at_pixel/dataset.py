# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Prepare a ground-truth reconstruction for the track-at-pixel harness.

The harness needs four things the checked-in ground truth does not carry: a
workspace it may write into, a ``.sift`` file per image (the constellation
query is over detected keypoints), a ``.kdf`` descriptor index over those files
in the reconstruction's own image order, so a match names a reconstruction image
directly, and a cluster-patches ``.matches`` file built from that same index.
``prepare`` builds all four in a cache directory, never beside the checked-in
data, and reuses them on later runs. A ``.matches`` file supplied by the caller
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

# Name -> the ground-truth .sfmr. Its directory must hold the images and the
# .sfm-workspace.json marker that says how their features are extracted.
DATASETS = {
    "seoul_bull": REPO
    / "test-data/images/seoul_bull_sculpture/seoul_bull_sculpture_ground_truth.sfmr",
}


@dataclass
class PreparedDataset:
    name: str
    ground_truth: Path  # the checked-in file the workspace was copied from
    workspace: Path
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
    from sfmtool._sfmtool.reconstruction import SfmrReconstruction

    gt = Path(DATASETS.get(name_or_path, name_or_path)).resolve()
    if not gt.is_file():
        raise SystemExit(f"no ground truth at {gt} (known datasets: {list(DATASETS)})")
    name = name_or_path if name_or_path in DATASETS else gt.stem
    source_ws = gt.parent
    marker = source_ws / ".sfm-workspace.json"
    if not marker.is_file():
        raise SystemExit(f"{gt} has no .sfm-workspace.json beside it")

    workspace = (cache_dir or default_cache_dir()) / name
    workspace.mkdir(parents=True, exist_ok=True)
    sfmr = workspace / gt.name
    # Copy only what is missing or stale, so runs started in parallel against
    # one cache never write a file another is reading.
    _copy_if_stale(gt, sfmr)
    _copy_if_stale(marker, workspace / marker.name)
    for extra in ("camera_config.json", "rig_config.json"):
        if (source_ws / extra).is_file():
            _copy_if_stale(source_ws / extra, workspace / extra)

    recon = SfmrReconstruction.load(sfmr)
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
        sfmr=sfmr,
        kdf=kdf,
        matches=Path(matches),
    )


def _extract_sift(image_paths: list[Path], config: dict, *, quiet: bool) -> None:
    from sfmtool.sift.file import get_sift_path_for_image, image_files_to_sift_files

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
    from sfmtool.sift.file import SiftReader, get_sift_path_for_image

    path = get_sift_path_for_image(workspace / image_name)
    reader = SiftReader(path)
    xy, affine = reader.read_positions_and_shapes()
    reader.close()
    return (
        np.ascontiguousarray(xy, dtype=np.float32),
        np.ascontiguousarray(affine, dtype=np.float32),
        path,
    )


def _build_kdf(workspace: Path, names: list[str], config: dict, out: Path) -> None:
    """One corpus image per reconstruction image, in the reconstruction's order."""
    from sfmtool._sfmtool.spatial import KdForest, write_kdf
    from sfmtool.sift.file import SiftReader, get_sift_path_for_image

    descriptors, positions, shapes, image_of, feature_of = [], [], [], [], []
    for index, image_name in enumerate(names):
        reader = SiftReader(get_sift_path_for_image(workspace / image_name))
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


# `sfm cluster-patches`' own defaults.
CLUSTER_PATCH_DEFAULTS = {
    "patch_size": 12.0,
    "resolution": 25,
    "min_zncc": 0.85,
    "max_shift": 3.0,
    "max_keypoint_uncertainty": 0.35,
}


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
    over the index rather than over a fresh forest; the file is written by that
    command's own writer and refined by ``sfm cluster-patches``' own step. The
    index's feature ids run image by image in ``names`` order, each image's
    features in ``.sift`` row order, which is what makes a cluster member's
    feature index a ``.sift`` row.
    """
    from types import SimpleNamespace

    from sfmtool._cluster_patches import _run_cluster_patches
    from sfmtool._sfmtool.io import read_sift_metadata
    from sfmtool._sfmtool.matching import background_floor_clusters_kdf
    from sfmtool.feature_match._run import _write_clusters_matches
    from sfmtool.sift.file import get_sift_path_for_image

    sift_paths = [get_sift_path_for_image(p) for p in image_paths]
    counts = [
        int(read_sift_metadata(str(p))["metadata"]["feature_count"]) for p in sift_paths
    ]
    image_starts = np.concatenate([[0], np.cumsum(counts)]).astype(np.uint32)
    options = {"d": 10, "alpha": 0.8, "min_size": 2}
    starts, images, features = background_floor_clusters_kdf(
        str(kdf), image_starts, **options
    )
    clusters_path = out.with_name("track_at_pixel_clusters.matches")
    if clusters_path.exists():
        clusters_path.unlink()
    _write_clusters_matches(
        SimpleNamespace(
            cluster_starts=starts, member_images=images, member_features=features
        ),
        clusters_path,
        image_paths=image_paths,
        sift_paths=sift_paths,
        image_names=names,
        workspace_dir=workspace,
        ws_config=config,
        matcher_options={**options, "index": kdf.name},
        max_feature_count=None,
    )
    _run_cluster_patches(clusters_path, str(out), **CLUSTER_PATCH_DEFAULTS)
