# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""The source-cluster join against the numpy version it replaced.

The reference below is `seed_relax.fill.source_clusters` and
`seed_relax.rings.assign_rings` as they stood before the kernel took over,
carried verbatim.  They are kept HERE and nowhere else, so the two can be run
on the same handle and compared.

Unlike the point estimation, this one IS bit-identical: nothing here is a
linear solve.  The radius is the same arithmetic in the same association, the
cluster radius is a maximum, the join is a sorted-key search with the same
tie rule, and the band comparison is the same pair of inequalities.  The test
asserts equality, not closeness.
"""

import types

import numpy as np
import pytest

from seed_relax import rings
from seed_relax.fill import source_clusters
from seed_relax.fleet_constants import RING_RATIO_P1


def fake_source(names, starts, images, features, affines, refine_radius):
    """A stand-in for the selection handle, carrying only what the join reads."""
    return types.SimpleNamespace(
        image_names=list(names),
        refine_radius=float(refine_radius),
        cluster_starts=np.asarray(starts, np.int64),
        member_images=np.asarray(images, np.int64),
        member_features=np.asarray(features, np.int64),
        member_affines=np.asarray(affines, float),
    )


class Member:
    """The two identity arrays, the image table and the frames the join reads."""

    def __init__(self, names, obs_i, obs_f, frames):
        self.names = [str(n) for n in names]
        self.obs_i = np.asarray(obs_i, np.int64)
        self.obs_f = np.asarray(obs_f, np.int64)
        self.frames = np.asarray(frames, np.int64)


#: Six selection clusters as ``(image, feature, shape scale)`` per row, the
#: same shape as the fill-in's own fixture: 0 and 1 are the member's admission,
#: 2, 3 and 4 are candidates, and 5 is seen on one placed frame only.
SELECTION = [
    [(0, 10, 4.0), (1, 11, 3.0), (2, 12, 3.0)],
    [(0, 20, 2.0), (1, 21, 2.0)],
    [(0, 30, 1.5), (1, 31, 1.0), (2, 32, 1.0)],
    [(1, 40, 0.8), (2, 41, 0.8)],
    [(0, 50, 0.3), (2, 51, 0.3), (3, 52, 0.3)],
    [(2, 60, 1.0), (3, 61, 1.0)],
]
NAMES = ["cam/000.jpg", "cam/001.jpg", "cam/002.jpg", "cam/003.jpg"]
PLACED = [0, 1, 2]


@pytest.fixture(name="source_and_member")
def _source_and_member():
    starts, images, features, affines = [0], [], [], []
    for members in SELECTION:
        for img, feat, scale in members:
            images.append(img)
            features.append(feat)
            affines.append(
                np.array([[scale, 0.0, 100.0 + feat], [0.0, scale, 200.0 + feat]])
            )
        starts.append(len(images))
    source = fake_source(NAMES, starts, images, features, affines, 8.0)
    obs_i, obs_f = [], []
    for c in (0, 1):
        for img, feat, _s in SELECTION[c]:
            obs_i.append(img)
            obs_f.append(feat)
    return source, Member(NAMES, obs_i, obs_f, PLACED)


# ── The numpy version, as it stood ────────────────────────────────────────


def reference(source, m, frames=None):
    """The join the kernel replaced.  Returns the same dict, minus the bands."""
    names_f = [str(n).replace("\\", "/") for n in source.image_names]
    if names_f != list(m.names):
        return {"refused": "image table differs from the member's"}
    starts = np.asarray(source.cluster_starts, np.int64)
    n_cl = len(starts) - 1
    aff = np.asarray(source.member_affines)
    img = np.asarray(source.member_images, np.int64)
    feat = np.asarray(source.member_features, np.int64)
    cl = np.repeat(np.arange(n_cl, dtype=np.int64), np.diff(starts))
    rad = (
        0.5
        * float(source.refine_radius)
        * (np.linalg.norm(aff[:, :, 0], axis=1) + np.linalg.norm(aff[:, :, 1], axis=1))
    )
    cl_rad = np.zeros(n_cl)
    np.maximum.at(cl_rad, cl, rad)

    key_f = img * (1 << 32) + feat
    order = np.argsort(key_f, kind="stable")
    kf = key_f[order]
    key_m = m.obs_i.astype(np.int64) * (1 << 32) + m.obs_f.astype(np.int64)
    pos = np.searchsorted(kf, key_m)
    hit = (pos < len(kf)) & (kf[np.minimum(pos, len(kf) - 1)] == key_m)
    adm = np.zeros(n_cl, bool)
    adm[cl[order[np.minimum(pos, len(kf) - 1)][hit]]] = True

    frames = np.asarray(m.frames if frames is None else frames, np.int64)
    on_frame = np.zeros(len(names_f), bool)
    on_frame[frames] = True
    keep_row = on_frame[img]
    cnt = np.bincount(cl[keep_row], minlength=n_cl)
    cand = np.nonzero((cnt >= 2) & (~adm))[0]

    take = keep_row & np.isin(cl, cand)
    return {
        "n_file_clusters": int(n_cl),
        "n_admitted": int(adm.sum()),
        "n_rows_matched": int(hit.sum()),
        "n_rows_member": int(len(key_m)),
        "adm_radius": cl_rad[adm],
        "cand": cand,
        "cand_radius": cl_rad[cand],
        "obs_cl": cl[take],
        "obs_img": img[take],
        "obs_feat": feat[take],
        "obs_uv": np.ascontiguousarray(aff[take][:, :, 2], float),
        "obs_shape": np.ascontiguousarray(aff[take][:, :, :2], float),
    }


def reference_rings(cand_radius, floor, edges):
    """The ring assignment the kernel replaced."""
    x = np.asarray(cand_radius, float) / float(floor)
    ring = np.full(len(x), -1, np.int64)
    for k in range(len(edges) - 1):
        hi, lo = float(edges[k]), float(edges[k + 1])
        ring[(x < hi) & (x >= lo)] = k
    return ring


# ── The comparison ────────────────────────────────────────────────────────


def compare(source, m, frames=None):
    """Run both and assert every key matches to the bit."""
    want = reference(source, m, frames)
    got = source_clusters(source, m, frames)
    assert "refused" not in got, got.get("refused")
    for key in ("n_file_clusters", "n_admitted", "n_rows_matched", "n_rows_member"):
        assert got[key] == want[key], key
    for key in (
        "adm_radius",
        "cand",
        "cand_radius",
        "obs_cl",
        "obs_img",
        "obs_feat",
        "obs_uv",
        "obs_shape",
    ):
        np.testing.assert_array_equal(got[key], want[key], err_msg=key)
    if len(want["adm_radius"]):
        floor = float(want["adm_radius"].min())
        assert got["adm_floor_px"] == floor
        edges = rings.octave_edges(RING_RATIO_P1)
        np.testing.assert_array_equal(
            got["ring"], reference_rings(want["cand_radius"], floor, edges)
        )
        np.testing.assert_array_equal(
            rings.assign_rings(want["cand_radius"], floor, edges),
            reference_rings(want["cand_radius"], floor, edges),
        )
    return got


def test_the_two_agree_on_the_fixture_handle(source_and_member):
    source, m = source_and_member
    got = compare(source, m)
    assert got["n_admitted"] > 0
    assert len(got["cand"]) > 0


def test_the_two_agree_when_only_some_frames_are_placed(source_and_member):
    source, m = source_and_member
    frames = np.asarray(m.frames, np.int64)[: max(2, len(m.frames) // 2)]
    compare(source, m, frames)


def test_the_bands_are_read_the_same_way_on_a_synthetic_spread():
    edges = rings.octave_edges(RING_RATIO_P1)
    rng = np.random.default_rng(20260829)
    radius = np.concatenate(
        [
            rng.uniform(0.01, 40.0, 2000),
            # Values sitting exactly on the edges, which is where a half-open
            # rule can be got wrong.
            np.array([float(e) for e in edges[1:]]) * 3.0,
        ]
    )
    for floor in (3.0, 0.5, 12.5):
        np.testing.assert_array_equal(
            rings.assign_rings(radius, floor, edges),
            reference_rings(radius, floor, edges),
        )


def test_a_handle_whose_image_table_differs_is_refused(source_and_member):
    source, m = source_and_member
    other = fake_source(
        list(source.image_names) + ["extra.jpg"],
        np.asarray(source.cluster_starts),
        np.asarray(source.member_images),
        np.asarray(source.member_features),
        np.asarray(source.member_affines),
        float(source.refine_radius),
    )
    out = source_clusters(other, m)
    assert out["refused"]
    assert reference(other, m)["refused"] == out["refused"]


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_the_two_agree_on_random_selections(seed):
    """A selection drawn at random, with duplicate keys and unplaced frames."""
    rng = np.random.default_rng(seed)
    n_images = 8
    names = [f"img{k:03d}.jpg" for k in range(n_images)]
    n_cl = 60
    sizes = rng.integers(1, 6, n_cl)
    starts = np.concatenate(([0], np.cumsum(sizes))).astype(np.int64)
    n_mem = int(starts[-1])
    img = rng.integers(0, n_images, n_mem).astype(np.int64)
    # A small feature vocabulary, so the same (image, feature) key repeats.
    feat = rng.integers(0, 40, n_mem).astype(np.int64)
    aff = np.zeros((n_mem, 2, 3))
    scale = rng.uniform(0.05, 3.0, n_mem)
    aff[:, 0, 0] = scale
    aff[:, 1, 1] = scale * rng.uniform(0.7, 1.4, n_mem)
    aff[:, 0, 1] = rng.normal(0.0, 0.1, n_mem)
    aff[:, 1, 0] = rng.normal(0.0, 0.1, n_mem)
    aff[:, 0, 2] = rng.uniform(0.0, 640.0, n_mem)
    aff[:, 1, 2] = rng.uniform(0.0, 480.0, n_mem)
    source = fake_source(names, starts, img, feat, aff, 6.0)

    take = rng.choice(n_mem, size=n_mem // 4, replace=False)
    member = Member(names, img[take], feat[take], np.arange(n_images))
    compare(source, member)
    compare(source, member, np.array([0, 1, 2, 5]))
