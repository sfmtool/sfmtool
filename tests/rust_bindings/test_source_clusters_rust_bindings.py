# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the source-cluster join bindings
(``sfmtool._sfmtool.analysis.source_clusters``, ``assign_bands``).

The join names the clusters of a selection that a member's admission never held
and that at least two of the member's placed frames still see, reads each one's
feature radius off the stored affines, and assigns it to a band of radius in
units of the member's own admission floor.
"""

import numpy as np
import numpy.testing as npt
import pytest

from sfmtool._sfmtool.analysis import assign_bands, source_clusters

#: Bands one octave apart, anchored at the floor, with the top band open.
EDGES = np.array([np.inf, 1.0, 0.5, 0.25])
REFINE_RADIUS = 8.0


def _selection(clusters):
    """``(starts, images, features, affines)`` from ``(image, feature, scale)``."""
    starts, images, features, affines = [0], [], [], []
    for rows in clusters:
        for img, feat, scale in rows:
            images.append(img)
            features.append(feat)
            affines.append([[scale, 0.0, 100.0 + feat], [0.0, scale, 200.0 + feat]])
        starts.append(len(images))
    return (
        np.array(starts, np.uint32),
        np.array(images, np.uint32),
        np.array(features, np.uint32),
        np.array(affines, float),
    )


def _join(clusters, obs, frames, n_images=4, edges=EDGES):
    starts, images, features, affines = _selection(clusters)
    return source_clusters(
        starts,
        images,
        features,
        affines,
        REFINE_RADIUS,
        n_images,
        np.array([o[0] for o in obs], np.uint32),
        np.array([o[1] for o in obs], np.uint32),
        np.array(frames, np.uint32),
        edges,
    )


#: Cluster 0 is the member's own; 1 and 2 are candidates; 3 is seen on one
#: placed frame only.
CLUSTERS = [
    [(0, 10, 4.0), (1, 11, 4.0)],
    [(0, 20, 2.0), (1, 21, 2.0), (2, 22, 2.0)],
    [(0, 30, 0.6), (1, 31, 0.6)],
    [(0, 40, 1.0), (3, 41, 1.0)],
]
ADMISSION = [(0, 10), (1, 11)]
PLACED = [0, 1, 2]


def test_the_candidates_are_what_the_admission_never_held():
    out = _join(CLUSTERS, ADMISSION, PLACED)
    assert out["n_file_clusters"] == 4
    assert out["n_admitted"] == 1
    assert out["n_rows_matched"] == 2
    npt.assert_array_equal(out["candidates"], [1, 2])
    # Cluster 3 lives on image 3, which is not placed, so only one placed frame
    # sees it.
    npt.assert_array_equal(out["admission_radius"], [REFINE_RADIUS * 4.0])
    assert out["admission_floor_px"] == REFINE_RADIUS * 4.0


def test_a_clusters_radius_is_its_widest_rows():
    clusters = [
        [(0, 10, 4.0), (1, 11, 4.0)],
        [(0, 20, 0.5), (1, 21, 3.0)],
    ]
    out = _join(clusters, ADMISSION[:1], PLACED)
    npt.assert_allclose(out["candidate_radius"], [REFINE_RADIUS * 3.0])


def test_the_selected_rows_carry_the_pixel_and_the_shape():
    out = _join(CLUSTERS, ADMISSION, PLACED)
    npt.assert_array_equal(out["obs_cluster"], [1, 1, 1, 2, 2])
    npt.assert_array_equal(out["obs_image"], [0, 1, 2, 0, 1])
    npt.assert_array_equal(out["obs_feature"], [20, 21, 22, 30, 31])
    npt.assert_array_equal(out["obs_uv"][0], [120.0, 220.0])
    npt.assert_array_equal(out["obs_shape"][0], [[2.0, 0.0], [0.0, 2.0]])
    assert out["obs_uv"].shape == (5, 2)
    assert out["obs_shape"].shape == (5, 2, 2)
    assert out["n_selected"] == 5


def test_a_row_takes_the_first_selection_row_carrying_its_key():
    # (0, 10) appears in cluster 0 and again in cluster 1; the member's row
    # admits the first, so cluster 1 stays a candidate.
    clusters = [
        [(0, 10, 4.0), (1, 11, 4.0)],
        [(0, 10, 1.0), (1, 12, 1.0)],
    ]
    out = _join(clusters, [(0, 10)], PLACED)
    assert out["n_rows_matched"] == 1
    npt.assert_array_equal(out["candidates"], [1])


def test_the_bands_are_half_open_against_the_floor():
    radius = np.array([1.0, 0.999, 4.0, 0.5, 0.4999, 0.2])
    npt.assert_array_equal(assign_bands(radius, 1.0, EDGES), [0, 1, 0, 1, 2, -1])
    # The same values against a floor of two land in the same bands scaled.
    npt.assert_array_equal(assign_bands(2.0 * radius, 2.0, EDGES), [0, 1, 0, 1, 2, -1])


def test_the_join_bands_the_candidates_it_found():
    out = _join(CLUSTERS, ADMISSION, PLACED)
    floor = out["admission_floor_px"]
    npt.assert_array_equal(
        out["candidate_band"], assign_bands(out["candidate_radius"], floor, EDGES)
    )


def test_an_empty_admission_has_no_floor():
    out = _join(CLUSTERS, [], PLACED)
    assert np.isnan(out["admission_floor_px"])
    assert out["n_admitted"] == 0
    assert out["n_rows_matched"] == 0
    # Nothing can be banded against a floor that does not exist.
    npt.assert_array_equal(out["candidate_band"], np.full(3, -1))


def test_the_join_is_checked():
    starts, images, features, affines = _selection(CLUSTERS)
    good = dict(
        cluster_starts=starts,
        member_images=images,
        member_features=features,
        member_affines=affines,
        refine_radius=REFINE_RADIUS,
        n_images=4,
        obs_image=np.zeros(1, np.uint32),
        obs_feature=np.zeros(1, np.uint32),
        frames=np.array(PLACED, np.uint32),
        band_edges=EDGES,
    )

    def call(**over):
        return source_clusters(**{**good, **over})

    with pytest.raises(ValueError, match="same length"):
        call(member_features=features[:-1])
    with pytest.raises(ValueError, match="member_affines must have shape"):
        call(member_affines=affines[:, :, :2])
    with pytest.raises(ValueError, match="obs_image and obs_feature"):
        call(obs_feature=np.zeros(2, np.uint32))
    with pytest.raises(ValueError, match="at least one boundary"):
        call(cluster_starts=np.zeros(0, np.uint32))
    with pytest.raises(ValueError, match="non-decreasing"):
        call(cluster_starts=np.array([0, 4, 2, 6, 8], np.uint32))
    with pytest.raises(ValueError, match="exceeds the"):
        call(cluster_starts=np.array([0, 2, 5, 7, 999], np.uint32))
