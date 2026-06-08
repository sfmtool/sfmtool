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
):
    """Create a COLMAP DB and populate it with cameras, images, keypoints, descriptors."""
    from .._camera_config import CameraConfigResolver
    from ..colmap.db_builders import _setup_db_single_camera, _setup_db_with_rigs
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
    from ..sift.file import SiftReader

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
