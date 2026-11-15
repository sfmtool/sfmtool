# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the image-pairing bindings: ``sfmtool.analysis.pair_images`` and
``SfmrReconstruction.find_image_file_hashes``. See
specs/core/analysis/reconstruction-alignment.md, "Pairing the images"."""

from pathlib import Path

import numpy as np
import pytest

from sfmtool import ProgressCounter
from sfmtool.analysis import pair_images
from sfmtool.reconstruction import SfmrReconstruction

from tests.conftest import SEOUL_BULL_GROUND_TRUTH


def _hash(n: int) -> bytes:
    return bytes([n]) * 16


def test_pair_images_pairs_by_file_then_by_name():
    pairing = pair_images(
        ["a.jpg", "b.jpg", "c.jpg"],
        [_hash(1), _hash(2), None],
        ["renamed.jpg", "b.jpg", "c.jpg"],
        [_hash(1), _hash(9), None],
    )
    assert pairing["pairs"].dtype == np.uint32
    assert pairing["pairs"].tolist() == [[0, 0], [1, 1], [2, 2]]
    assert pairing["by_file_hash"] == 1
    assert pairing["by_name"] == 2
    assert pairing["by_name_different_files"].dtype == np.uint32
    assert pairing["by_name_different_files"].tolist() == [[1, 1]]
    assert pairing["summary"] == (
        "images paired: 1 by file hash, 2 by name, 1 of them different files"
    )


def test_pair_images_with_nothing_shared_returns_empty_arrays():
    pairing = pair_images(["a.jpg"], [None], ["b.jpg"], [None])
    assert pairing["pairs"].shape == (0, 2)
    assert pairing["by_name_different_files"].shape == (0, 2)


def test_pair_images_refuses_a_hash_per_image_mismatch():
    with pytest.raises(ValueError, match="source_names has 2 entries"):
        pair_images(["a.jpg", "b.jpg"], [None], [], [])


def test_pair_images_refuses_a_hash_that_is_not_16_bytes():
    with pytest.raises(ValueError, match=r"target_file_hashes\[0\] has 15 bytes"):
        pair_images([], [], ["a.jpg"], [b"x" * 15])


def test_an_embedded_reconstruction_gives_its_stored_hashes():
    truth = SfmrReconstruction.load(SEOUL_BULL_GROUND_TRUTH)
    hashes = truth.find_image_file_hashes()
    assert len(hashes) == truth.image_count
    assert all(isinstance(h, bytes) and len(h) == 16 for h in hashes)
    assert len(set(hashes)) == truth.image_count


def test_a_sift_files_reconstruction_reads_its_sift_files(seoul_bull_workspace_once):
    # The workspace holds copies of the ground truth's photographs under other
    # names, so its .sift files record the hashes the ground truth stores.
    recon = SfmrReconstruction.load(seoul_bull_workspace_once)
    assert recon.feature_source == "sift_files"
    counter = ProgressCounter()
    hashes = recon.find_image_file_hashes(progress=counter)
    assert counter.value == recon.image_count

    truth = SfmrReconstruction.load(SEOUL_BULL_GROUND_TRUTH)
    stored = dict(zip(truth.image_names, truth.find_image_file_hashes()))
    for name, found in zip(recon.image_names, hashes):
        assert found == stored[Path(name).name], name


def test_a_sift_files_reconstruction_without_its_sift_files_knows_no_hash(
    seoul_bull_sfmr_only,
):
    recon = SfmrReconstruction.load(seoul_bull_sfmr_only)
    assert recon.find_image_file_hashes() == [None] * recon.image_count
