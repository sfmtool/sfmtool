# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""The source clusters the admission never held, and the extended member."""

import types

import numpy as np

from seed_relax.fill import extend_member, source_clusters

NAMES = ["cam/000.jpg", "cam/001.jpg", "cam/002.jpg", "cam/003.jpg"]
REFINE_RADIUS = 8.0

#: Six source clusters, as ``(image, feature, shape scale)`` per member.
#: 0 and 1 are the member's own admission; 2, 3 and 4 are candidates; 5 is
#: seen on one placed frame only.
SOURCE = [
    [(0, 10, 4.0), (1, 11, 3.0), (2, 12, 3.0)],
    [(0, 20, 2.0), (1, 21, 2.0)],
    [(0, 30, 1.5), (1, 31, 1.0), (2, 32, 1.0)],
    [(1, 40, 0.8), (2, 41, 0.8)],
    [(0, 50, 0.3), (2, 51, 0.3), (3, 52, 0.3)],
    [(2, 60, 1.0), (3, 61, 1.0)],
]
#: The frames the relaxation placed: image 3 is not among them.
PLACED = [0, 1, 2]


def _handle(names=NAMES):
    starts, images, features, affines = [0], [], [], []
    for members in SOURCE:
        for img, feat, scale in members:
            images.append(img)
            features.append(feat)
            affines.append(
                np.array([[scale, 0.0, 100.0 + feat], [0.0, scale, 200.0 + feat]])
            )
        starts.append(len(images))
    return types.SimpleNamespace(
        image_names=list(names),
        refine_radius=REFINE_RADIUS,
        cluster_starts=np.array(starts, np.int64),
        member_images=np.array(images, np.int64),
        member_features=np.array(features, np.int64),
        member_affines=np.stack(affines),
    )


def _member():
    import seed_candidate_eval as EV
    from sfmtool._sfmtool.geometry import CameraIntrinsics

    cam = CameraIntrinsics.from_dict(
        {
            "model": "SIMPLE_PINHOLE",
            "width": 640,
            "height": 480,
            "parameters": {
                "focal_length": 500.0,
                "principal_point_x": 320.0,
                "principal_point_y": 240.0,
            },
        }
    )
    obs_c, obs_i, obs_f, uv = [], [], [], []
    for c in (0, 1):
        for img, feat, _s in SOURCE[c]:
            obs_c.append(c)
            obs_i.append(img)
            obs_f.append(feat)
            uv.append([100.0 + feat, 200.0 + feat])
    posed = np.zeros(len(NAMES), bool)
    posed[PLACED] = True
    keep = np.ones(len(obs_c), bool)
    # One row the rotation model refused: outside the membership, inside the
    # admission.
    keep[-1] = False
    return EV.Member(
        0,
        "rotation_only",
        NAMES,
        cam,
        500.0,
        np.zeros((len(NAMES), 3)),
        np.zeros((len(NAMES), 3)),
        posed,
        np.tile(np.array([0.0, 0.0, -1.0]), (2, 1)),
        (
            np.array(obs_c, np.int64),
            np.array(obs_i, np.int64),
            np.array(uv, float),
            np.array(obs_f, np.int64),
        ),
        keep=keep,
    )


def test_the_join_finds_the_clusters_the_member_already_holds():
    src = source_clusters(_handle(), _member(), frames=PLACED)
    assert "refused" not in src
    assert src["n_file_clusters"] == len(SOURCE)
    assert src["n_admitted"] == 2
    assert src["n_rows_matched"] == src["n_rows_member"] == 5


def test_a_candidate_needs_two_placed_frames_and_no_admission():
    src = source_clusters(_handle(), _member(), frames=PLACED)
    # 5 is seen once among the placed frames (image 3 is not placed), 0 and 1
    # are the admission itself.
    assert src["cand"].tolist() == [2, 3, 4]
    # Only the candidates' rows on placed frames come back.
    assert set(src["obs_img"].tolist()) <= set(PLACED)
    assert sorted(set(src["obs_cl"].tolist())) == [2, 3, 4]


def test_a_cluster_takes_its_widest_members_radius():
    src = source_clusters(_handle(), _member(), frames=PLACED)
    # Cluster 2's widest member has scale 1.5, and the reading is the refine
    # radius times the mean of the affine's two column norms.
    assert abs(float(src["cand_radius"][0]) - REFINE_RADIUS * 1.5) < 1e-12
    # The admission floor is cluster 1's, the narrower of the two admitted.
    assert abs(float(src["adm_radius"].min()) - REFINE_RADIUS * 2.0) < 1e-12


def test_a_different_image_table_is_refused():
    other = _handle(names=NAMES[:3] + ["cam/999.jpg"])
    assert source_clusters(other, _member(), frames=PLACED)["refused"]


def test_the_extension_leaves_the_membership_alone():
    m = _member()
    src = source_clusters(_handle(), m, frames=PLACED)
    add = np.array([2, 4], np.int64)
    mx, slot = extend_member(m, src, add)
    # The new ids continue after the member's own clusters.
    assert slot == {2: m.n_cl, 4: m.n_cl + 1}
    assert mx.n_cl == m.n_cl + 2
    # The model's own inlier set is untouched; the admission grew.
    assert mx.rows.tolist() == m.rows.tolist()
    assert len(mx.rows_all) > len(m.rows_all)
    added = np.setdiff1d(mx.rows_all, m.rows_all)
    assert set(mx.obs_c[added].tolist()) == {m.n_cl, m.n_cl + 1}
    # Cluster 4's row on the unplaced image 3 is carried but not admitted.
    assert 3 not in mx.obs_i[mx.rows_all].tolist()
    # Their placeholder positions are unit bearings.
    assert np.allclose(np.linalg.norm(mx.pts[m.n_cl :], axis=1), 1.0)
