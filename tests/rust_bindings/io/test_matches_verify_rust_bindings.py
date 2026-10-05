# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests that the `verify_matches` binding reports a malformed pairwise
`.matches` file rather than raising out of Rust.

`verify_matches` is a validator, so every input has to come back as a
`(is_valid, errors)` tuple. The pairwise backbone's checks walk raw CSR arrays
whose lengths the metadata declares, and each case below is a shape
`write_matches` cannot produce -- only a truncated download, a bad disk or a
hand-edited archive can. Each one used to index past the end of an array and
surface as `pyo3_runtime.PanicException`.

The archives store each `.zst` payload with ZIP_STORED, so `zipfile` moves the
compressed bytes between files verbatim: an array of the wrong length is
grafted in by writing a smaller file's entry under the larger file's entry
name, with no zstd codec needed here.
"""

import zipfile

import numpy as np

from sfmtool._sfmtool.io import verify_matches, write_matches


def _pairwise_data(
    image_count: int,
    image_index_pairs: list[list[int]],
    match_counts: list[int],
) -> dict:
    """A minimal valid pairwise `.matches` dict.

    Feature indexes are all 0, which is in range for every image (the feature
    counts below are all >= 100), so nothing but the damage under test can
    fail verification.
    """
    match_count = sum(match_counts)
    return {
        "metadata": {
            "version": 4,
            "matching_method": "sequential",
            "matching_tool": "sfmtool",
            "matching_tool_version": "0.2",
            "matching_options": {"overlap": 10},
            "workspace": {
                "absolute_path": "/tmp/workspace",
                "relative_path": "..",
                "contents": {
                    "feature_tool": "sfmtool",
                    "feature_type": "sift",
                    "feature_options": {},
                    "feature_prefix_dir": "features/sift-sfmtool-abc123",
                },
            },
            "timestamp": "2026-07-09T10:00:00Z",
            "image_count": image_count,
            "image_pair_count": len(image_index_pairs),
            "match_count": match_count,
            "has_two_view_geometries": False,
            "has_clusters": False,
            "has_cluster_patches": False,
        },
        "image_names": [f"frames/frame_{k:03d}.jpg" for k in range(image_count)],
        "feature_tool_hashes": [b"\x00" * 16] * image_count,
        "sift_content_hashes": [b"\x01" * 16] * image_count,
        "feature_counts": np.full(image_count, 100, dtype=np.uint32),
        "image_dims": np.full((image_count, 2), [640, 480], dtype=np.uint32),
        "has_clusters": False,
        "image_index_pairs": np.array(image_index_pairs, dtype=np.uint32),
        "match_counts": np.array(match_counts, dtype=np.uint32),
        "match_feature_indexes": np.zeros((match_count, 2), dtype=np.uint32),
        "match_descriptor_distances": np.full(match_count, 100.0, dtype=np.float32),
        "has_cluster_patches": False,
        "has_two_view_geometries": False,
    }


def _write(path, **kwargs):
    write_matches(path, _pairwise_data(**kwargs))
    return path


def _graft(dst, base, donor, donor_entry: str, as_entry: str):
    """Copy `base` to `dst`, replacing `as_entry` with `donor`'s `donor_entry`.

    Naming the donor entry differently from the one it replaces is what makes
    a wrong-length array: a file written with fewer pairs, matches or images
    carries a shorter array under a name that spells out the smaller count.
    """
    with zipfile.ZipFile(donor) as zf:
        payload = zf.read(donor_entry)
    with zipfile.ZipFile(base) as src, zipfile.ZipFile(dst, "w") as out:
        for info in src.infolist():
            data = payload if info.filename == as_entry else src.read(info.filename)
            out.writestr(info, data)
    return dst


def _assert_reports(path, fragment: str):
    """`verify_matches` returns a failing report naming `fragment`.

    Calling it at all is half the assertion: before the pairwise length gate
    existed, each of these inputs raised `PanicException` out of Rust instead
    of returning.
    """
    valid, errors = verify_matches(path)
    assert valid is False
    assert any(fragment in e for e in errors), errors


# The reference file every case below damages: 3 images, 2 pairs, 5 matches.
_FULL = dict(image_count=3, image_index_pairs=[[0, 1], [0, 2]], match_counts=[3, 2])


def test_verify_reports_short_image_index_pairs(tmp_path):
    base = _write(tmp_path / "base.matches", **_FULL)
    donor = _write(
        tmp_path / "one_pair.matches",
        image_count=3,
        image_index_pairs=[[0, 1]],
        match_counts=[3],
    )
    _assert_reports(
        _graft(
            tmp_path / "damaged.matches",
            base,
            donor,
            "image_pairs/image_index_pairs.1.2.uint32.zst",
            "image_pairs/image_index_pairs.2.2.uint32.zst",
        ),
        "image_index_pairs byte length 8 != expected 16 (2 uint32 pairs)",
    )


def test_verify_reports_short_match_feature_indexes(tmp_path):
    base = _write(tmp_path / "base.matches", **_FULL)
    donor = _write(
        tmp_path / "four_matches.matches",
        image_count=3,
        image_index_pairs=[[0, 1], [0, 2]],
        match_counts=[2, 2],
    )
    _assert_reports(
        _graft(
            tmp_path / "damaged.matches",
            base,
            donor,
            "image_pairs/match_feature_indexes.4.2.uint32.zst",
            "image_pairs/match_feature_indexes.5.2.uint32.zst",
        ),
        "match_feature_indexes byte length 32 != expected 40 (5 uint32 pairs)",
    )


def test_verify_reports_short_feature_counts(tmp_path):
    base = _write(tmp_path / "base.matches", **_FULL)
    donor = _write(
        tmp_path / "two_images.matches",
        image_count=2,
        image_index_pairs=[[0, 1]],
        match_counts=[3],
    )
    _assert_reports(
        _graft(
            tmp_path / "damaged.matches",
            base,
            donor,
            "images/feature_counts.2.uint32.zst",
            "images/feature_counts.3.uint32.zst",
        ),
        "feature_counts byte length 8 != expected 12 (3 uint32 values)",
    )


def test_verify_reports_match_counts_overrunning_match_count(tmp_path):
    # Every array is the right length here; it is the CSR run lengths that
    # overrun, which would walk the bounds check off the end of
    # match_feature_indexes. The donor's match_counts entry is sized by the
    # pair count, so it grafts in under its own name.
    base = _write(tmp_path / "base.matches", **_FULL)
    donor = _write(
        tmp_path / "six_matches.matches",
        image_count=3,
        image_index_pairs=[[0, 1], [0, 2]],
        match_counts=[4, 2],
    )
    entry = "image_pairs/match_counts.2.uint32.zst"
    _assert_reports(
        _graft(tmp_path / "damaged.matches", base, donor, entry, entry),
        "Sum of match_counts (6) != match_count (5)",
    )


def test_verify_accepts_the_undamaged_file(tmp_path):
    valid, errors = verify_matches(_write(tmp_path / "base.matches", **_FULL))
    assert valid is True
    assert errors == []
