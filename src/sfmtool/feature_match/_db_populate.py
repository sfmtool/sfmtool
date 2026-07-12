# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""COLMAP database population and descriptor bookkeeping for `sfm match`.

Helpers that build a COLMAP database from extracted SIFT features and fill in
the descriptor distances and content hashes that the `.matches` format records.
Extracted from `_commands/match.py` so the command module stays a thin Click
wrapper.
"""

from pathlib import Path

import numpy as np


def _populate_db_features(
    db_path: Path,
    image_paths: list[Path],
    sift_paths: list[Path],
    image_names: list[str],
    workspace_dir: Path,
    max_feature_count: int | None,
    camera_model: str | None,
    camera_config_resolver=None,
    include_descriptors: bool = True,
):
    """Create a COLMAP DB and populate it with cameras, images, keypoints, and
    (unless ``include_descriptors=False``) descriptors — matchers that only use
    the database for geometric verification never read the descriptors back,
    and they are by far the largest rows."""
    from ..camera.config import CameraConfigResolver
    from ..colmap.db_builders import _setup_db_single_camera, _setup_db_with_rigs
    from ..rig.config import _load_rig_config

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
            include_descriptors=include_descriptors,
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
            include_descriptors=include_descriptors,
        )


def _compute_descriptor_distances(matches_data, sift_paths, max_feature_count):
    """Compute L2 descriptor distances for all matches from .sift files.

    Vectorized per pair: one fancy-indexed gather of each side's matched
    descriptor rows and a batched norm, instead of a Python loop per match.
    Exact regardless of summation order — squared u8 differences sum to at
    most 128·255² < 2²⁴, so every f32 partial sum is an exactly-representable
    integer. The `.sift` reads are decoded through a thread pool (the reader
    releases the GIL), keyed off the unique image indices that actually
    appear in pairs.
    """
    from concurrent.futures import ThreadPoolExecutor

    from ..sift.file import SiftReader

    pair_count = matches_data["metadata"]["image_pair_count"]
    if pair_count == 0:
        return

    image_index_pairs = matches_data["image_index_pairs"]
    match_counts = matches_data["match_counts"]
    match_feature_indexes = matches_data["match_feature_indexes"]
    distances = matches_data["match_descriptor_distances"]

    def read_one(img_idx):
        with SiftReader(sift_paths[img_idx]) as reader:
            # Cached as u8 (the f32 upcast happens per gathered batch below):
            # a whole-corpus f32 cache would be 4x the descriptor bytes.
            return img_idx, reader.read_descriptors(count=max_feature_count)

    used = np.unique(image_index_pairs)
    with ThreadPoolExecutor() as pool:
        desc_cache = dict(pool.map(read_one, (int(i) for i in used)))

    offsets = np.zeros(pair_count + 1, dtype=np.int64)
    np.cumsum(match_counts, out=offsets[1:])
    for k in range(pair_count):
        lo, hi = offsets[k], offsets[k + 1]
        if lo == hi:
            continue
        desc_i = desc_cache[int(image_index_pairs[k, 0])]
        desc_j = desc_cache[int(image_index_pairs[k, 1])]
        diff = desc_i[match_feature_indexes[lo:hi, 0]].astype(np.float32)
        diff -= desc_j[match_feature_indexes[lo:hi, 1]]
        np.sqrt(np.einsum("ij,ij->i", diff, diff), out=distances[lo:hi])


def _fill_sift_hashes(matches_data, sift_paths, image_names, image_paths):
    """Fill feature_tool_hashes, sift_content_hashes, and image_dims from
    .sift files (their metadata records the source image dimensions)."""
    from .._sfmtool.io import read_sift_metadata

    image_count = len(image_names)
    feature_tool_hashes = np.zeros((image_count, 16), dtype=np.uint8)
    sift_content_hashes = np.zeros((image_count, 16), dtype=np.uint8)
    image_dims = np.zeros((image_count, 2), dtype=np.uint32)

    for i, sift_path in enumerate(sift_paths):
        result = read_sift_metadata(str(sift_path))
        content_hash = result["content_hash"]
        ft_hash = bytes.fromhex(content_hash["feature_tool_xxh128"])
        ct_hash = bytes.fromhex(content_hash["content_xxh128"])
        feature_tool_hashes[i] = np.frombuffer(ft_hash, dtype=np.uint8)
        sift_content_hashes[i] = np.frombuffer(ct_hash, dtype=np.uint8)
        meta = result["metadata"]
        image_dims[i] = (meta["image_width"], meta["image_height"])

    matches_data["feature_tool_hashes"] = feature_tool_hashes
    matches_data["sift_content_hashes"] = sift_content_hashes
    matches_data["image_dims"] = image_dims
