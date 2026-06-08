# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Rig frame pair exclusion for feature matching."""

import os
from pathlib import Path


def _build_cross_frame_pairs(db_path: Path) -> list[tuple[str, str]]:
    """Generate all image pairs excluding same-frame pairs.

    Images sharing the same frame_id are from the same rig capture instant and
    should not be matched against each other.
    """
    import pycolmap

    with pycolmap.Database.open(db_path) as db:
        images = db.read_all_images()

    INVALID_FRAME_ID = 0xFFFFFFFF
    frame_to_names: dict[int, list[str]] = {}
    next_synthetic_id = -1
    for img in images:
        fid = img.frame_id
        if fid == INVALID_FRAME_ID:
            frame_to_names[next_synthetic_id] = [img.name]
            next_synthetic_id -= 1
        else:
            if fid not in frame_to_names:
                frame_to_names[fid] = []
            frame_to_names[fid].append(img.name)

    same_frame_pairs: set[tuple[str, str]] = set()
    for names in frame_to_names.values():
        if len(names) <= 1:
            continue
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                pair = tuple(sorted([names[i], names[j]]))
                same_frame_pairs.add(pair)

    all_names = sorted(img.name for img in images)
    pairs = []
    for i in range(len(all_names)):
        for j in range(i + 1, len(all_names)):
            pair = (all_names[i], all_names[j])
            if pair not in same_frame_pairs:
                pairs.append(pair)

    excluded = len(same_frame_pairs)
    if excluded > 0:
        n_frames = len(frame_to_names)
        print(
            f"Rig matching: {len(all_names)} images in {n_frames} frames, "
            f"excluded {excluded} same-frame pairs, "
            f"matching {len(pairs)} cross-frame pairs"
        )

    return pairs


def _build_same_frame_index_pairs(
    db_path: Path, image_paths: list, image_dir: Path
) -> set[tuple[int, int]]:
    """Build set of same-frame image index pairs for flow matching filtering."""
    import pycolmap

    with pycolmap.Database.open(db_path) as db:
        images = db.read_all_images()

    INVALID_FRAME_ID = 0xFFFFFFFF
    name_to_frame: dict[str, int] = {}
    for img in images:
        if img.frame_id != INVALID_FRAME_ID:
            name_to_frame[img.name] = img.frame_id

    index_to_frame: dict[int, int] = {}
    for idx, image_path in enumerate(image_paths):
        rel_path = os.path.relpath(image_path, image_dir).replace("\\", "/")
        if rel_path in name_to_frame:
            index_to_frame[idx] = name_to_frame[rel_path]

    frame_to_indices: dict[int, list[int]] = {}
    for idx, fid in index_to_frame.items():
        if fid not in frame_to_indices:
            frame_to_indices[fid] = []
        frame_to_indices[fid].append(idx)

    same_frame: set[tuple[int, int]] = set()
    for indices in frame_to_indices.values():
        if len(indices) <= 1:
            continue
        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                same_frame.add(
                    (min(indices[i], indices[j]), max(indices[i], indices[j]))
                )

    return same_frame
