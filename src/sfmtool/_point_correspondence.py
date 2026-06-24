# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared utilities for finding corresponding 3D points between SfM reconstructions."""

from collections import defaultdict

import numpy as np

from ._sfmtool import SfmrReconstruction
from ._sfmtool.analysis import find_point_correspondences_py
from ._sfmtool.io import read_sift_partial
from .sift.file import get_sift_path_from_recon


def find_point_correspondences(
    source_recon: SfmrReconstruction,
    target_recon: SfmrReconstruction,
    shared_images: list[tuple[int, int]],
) -> tuple[dict[int, int], np.ndarray, np.ndarray]:
    """Find corresponding 3D points between two reconstructions.

    Uses shared images to find features that appear in both reconstructions
    (at the same feature index), then maps those features to their respective
    3D points to establish point-to-point correspondences.

    Args:
        source_recon: Source SfmrReconstruction object
        target_recon: Target SfmrReconstruction object
        shared_images: List of (source_img_idx, target_img_idx) pairs

    Returns:
        Tuple of:
        - correspondences: Dict mapping source_point_id -> target_point_id
        - source_positions: (N, 3) array of source point positions
        - target_positions: (N, 3) array of target point positions

    Raises:
        ValueError: If no point correspondences are found
    """
    shared_src = np.array([s for s, _ in shared_images], dtype=np.uint32)
    shared_tgt = np.array([t for _, t in shared_images], dtype=np.uint32)

    source_ids, target_ids = find_point_correspondences_py(
        source_recon.track_image_indexes.astype(np.uint32),
        source_recon.track_feature_indexes.astype(np.uint32),
        source_recon.track_point_ids.astype(np.uint32),
        target_recon.track_image_indexes.astype(np.uint32),
        target_recon.track_feature_indexes.astype(np.uint32),
        target_recon.track_point_ids.astype(np.uint32),
        shared_src,
        shared_tgt,
    )

    if len(source_ids) == 0:
        raise ValueError(
            "No point correspondences found. "
            "This may indicate no shared features or incompatible reconstructions."
        )

    correspondences = dict(zip(source_ids.tolist(), target_ids.tolist()))
    source_positions = source_recon.positions[source_ids]
    target_positions = target_recon.positions[target_ids]
    return correspondences, source_positions, target_positions


def _observation_xy_by_image(
    recon: SfmrReconstruction,
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Group a reconstruction's observations by image, with their 2D keypoint
    coordinates.

    Returns a mapping ``image_index -> (xy, point_ids)`` where ``xy`` is an
    ``(K, 2)`` array of pixel positions (read from each image's ``.sift`` file)
    and ``point_ids`` is the parallel ``(K,)`` array of 3D point ids. Only the
    observations actually used by the reconstruction are included.
    """
    track_image_indexes = np.asarray(recon.track_image_indexes)
    track_feature_indexes = np.asarray(recon.track_feature_indexes)
    track_point_ids = np.asarray(recon.track_point_ids)

    image_names = recon.image_names

    out: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for img_idx in np.unique(track_image_indexes):
        img_idx = int(img_idx)
        mask = track_image_indexes == img_idx
        feat_idxs = track_feature_indexes[mask]
        point_ids = track_point_ids[mask]

        image_rel = image_names[img_idx]
        sift_path = get_sift_path_from_recon(recon, image_rel)
        if not sift_path.exists():
            raise FileNotFoundError(
                f"Missing SIFT file for '{image_rel}' at {sift_path}. "
                "Coordinate-based comparison reads keypoint positions from each "
                "reconstruction's workspace .sift files; run feature extraction in "
                "that workspace, or use --by-feature-index if the inputs share "
                "identical .sift files."
            )

        sift_data = read_sift_partial(str(sift_path), int(feat_idxs.max()) + 1)
        positions = np.asarray(sift_data["positions_xy"], dtype=np.float64)
        if int(feat_idxs.max()) >= len(positions):
            raise ValueError(
                f"Reconstruction references feature index {int(feat_idxs.max())} "
                f"for '{image_rel}' but its .sift file at {sift_path} has only "
                f"{len(positions)} feature(s). The reconstruction and its .sift "
                "files have drifted; re-extract features in that workspace, or "
                "compare with --by-feature-index."
            )
        out[img_idx] = (positions[feat_idxs], point_ids)
    return out


def _vote_point_correspondences(
    source_by_image: dict[int, tuple[np.ndarray, np.ndarray]],
    target_by_image: dict[int, tuple[np.ndarray, np.ndarray]],
    shared_images: list[tuple[int, int]],
    pixel_threshold: float,
    min_votes: int,
) -> dict[int, int]:
    """Vote source->target 3D point correspondences from per-image observations.

    For each shared image, observations are matched by *mutual* nearest 2D
    keypoint within ``pixel_threshold`` pixels. Each such match casts one vote
    for the ``(source_point, target_point)`` pair. The votes are then resolved
    greedily into a one-to-one mapping (strongest-supported pairs first), keeping
    only pairs with at least ``min_votes`` supporting images.

    This is the pure, sift-free core of
    :func:`find_point_correspondences_by_coordinate` and is unit-testable with
    synthetic observation dicts.
    """
    pair_votes: dict[tuple[int, int], int] = defaultdict(int)

    for src_img, tgt_img in shared_images:
        if src_img not in source_by_image or tgt_img not in target_by_image:
            continue
        src_xy, src_pids = source_by_image[src_img]
        tgt_xy, tgt_pids = target_by_image[tgt_img]
        if len(src_xy) == 0 or len(tgt_xy) == 0:
            continue

        # Pairwise squared distances (observation counts per image are small).
        d2 = np.sum((src_xy[:, None, :] - tgt_xy[None, :, :]) ** 2, axis=2)  # (M, N)
        src_to_tgt = np.argmin(d2, axis=1)
        tgt_to_src = np.argmin(d2, axis=0)
        thresh2 = pixel_threshold * pixel_threshold

        for si, tj in enumerate(src_to_tgt):
            # Mutual nearest neighbor, within the pixel threshold.
            if tgt_to_src[tj] == si and d2[si, tj] <= thresh2:
                pair_votes[(int(src_pids[si]), int(tgt_pids[tj]))] += 1

    # Resolve to a one-to-one mapping: assign strongest-supported pairs first.
    ordered = sorted(pair_votes.items(), key=lambda kv: kv[1], reverse=True)
    correspondences: dict[int, int] = {}
    used_targets: set[int] = set()
    for (src_pid, tgt_pid), votes in ordered:
        if votes < min_votes:
            break
        if src_pid in correspondences or tgt_pid in used_targets:
            continue
        correspondences[src_pid] = tgt_pid
        used_targets.add(tgt_pid)
    return correspondences


def find_point_correspondences_by_coordinate(
    source_recon: SfmrReconstruction,
    target_recon: SfmrReconstruction,
    shared_images: list[tuple[int, int]],
    pixel_threshold: float = 2.0,
    min_votes: int = 2,
) -> tuple[dict[int, int], np.ndarray, np.ndarray]:
    """Find corresponding 3D points by matching observation keypoint coordinates.

    Unlike :func:`find_point_correspondences`, which keys observations on their
    feature *index* and therefore requires both reconstructions to reference the
    same ``.sift`` files, this matches observations in shared images by 2D pixel
    proximity. That makes it robust to different feature backends (e.g. COLMAP
    SIFT vs the sfmtool extractor), where the same scene keypoint lands at the
    same pixel but under a different feature index.

    For each shared image the two reconstructions' observations are matched by
    mutual nearest 2D keypoint within ``pixel_threshold`` pixels; each match
    votes for the ``(source_point, target_point)`` pair it links, and the votes
    are resolved into a one-to-one point mapping requiring at least ``min_votes``
    supporting images.

    Both reconstructions must have their workspace ``.sift`` files available
    (resolved via ``recon.workspace_dir``), as with the SIFT-reading ``xform``
    filters.

    Args:
        source_recon: Source SfmrReconstruction object
        target_recon: Target SfmrReconstruction object
        shared_images: List of (source_img_idx, target_img_idx) pairs
        pixel_threshold: Max 2D keypoint distance (pixels) to call a match
        min_votes: Minimum number of shared images supporting a point pair

    Returns:
        Tuple of:
        - correspondences: Dict mapping source_point_id -> target_point_id
        - source_positions: (N, 3) array of source point positions
        - target_positions: (N, 3) array of target point positions

    Raises:
        ValueError: If no point correspondences are found
    """
    source_by_image = _observation_xy_by_image(source_recon)
    target_by_image = _observation_xy_by_image(target_recon)

    correspondences = _vote_point_correspondences(
        source_by_image,
        target_by_image,
        shared_images,
        pixel_threshold=pixel_threshold,
        min_votes=min_votes,
    )

    if not correspondences:
        raise ValueError(
            "No point correspondences found by coordinate matching. "
            "Try a larger pixel_threshold or check that the reconstructions "
            "cover the same images."
        )

    source_ids = np.array(list(correspondences.keys()), dtype=np.int64)
    target_ids = np.array(list(correspondences.values()), dtype=np.int64)
    source_positions = source_recon.positions[source_ids]
    target_positions = target_recon.positions[target_ids]
    return correspondences, source_positions, target_positions
