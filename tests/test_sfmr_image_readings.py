# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""The image readings of ``.sfmr`` version 15: each image's sharpness and
gradient structure tensor, where every writer gets them, and how an older file
is saved. From version 16 a file stores the sharpness as ``float16``. See
``specs/formats/sfmr-file-format.md`` § "Image readings"."""

from pathlib import Path

import numpy as np
import pytest
from click.testing import CliRunner

from sfmtool.cli import main
from sfmtool.fileio import SiftFile, read_image_rgb, read_sfmr, write_sfmr
from sfmtool.reconstruction import SfmrReconstruction
from sfmtool.sift import ImageToGray, image_readings_from_rgb
from sfmtool.sift.file import get_sift_path_from_recon


def _as_stored(sharpness: np.ndarray) -> np.ndarray:
    """``sharpness`` as a version 16 file stores it: each value rounded to the
    nearest float16, ties to even, and widened back to float32."""
    return sharpness.astype(np.float16).astype(np.float32)


def _sift_readings(recon: SfmrReconstruction) -> tuple[dict, np.ndarray, np.ndarray]:
    """The ``image_to_gray``, sharpness and tensors each image's ``.sift``
    records, in the reconstruction's image order, with the sharpness as a
    ``.sfmr`` file stores it."""
    conversions, sharpness, tensors = [], [], []
    for name in recon.image_names:
        with SiftFile(get_sift_path_from_recon(recon, name)) as reader:
            conversions.append(
                reader.feature_tool_metadata["feature_options"]["image_to_gray"]
            )
            sharpness.append(reader.metadata["sharpness"])
            tensors.append(reader.metadata["gradient_structure_tensor_xx_yy_xy"])
    assert all(c == conversions[0] for c in conversions)
    return (
        conversions[0],
        _as_stored(np.asarray(sharpness, dtype=np.float32)),
        np.asarray(tensors, dtype=np.float32),
    )


def _image_readings(recon: SfmrReconstruction, image_to_gray: dict):
    """Each image's readings computed from its image file, with the sharpness
    as a ``.sfmr`` file stores it."""
    conversion = ImageToGray.from_json(image_to_gray)
    rows = [
        image_readings_from_rgb(
            read_image_rgb(Path(recon.workspace_dir) / name), conversion
        )
        for name in recon.image_names
    ]
    return (
        _as_stored(np.asarray([r.sharpness for r in rows], dtype=np.float32)),
        np.asarray(
            [list(r.gradient_structure_tensor_xx_yy_xy) for r in rows],
            dtype=np.float32,
        ),
    )


def _assert_bits_equal(a: np.ndarray, b: np.ndarray):
    assert a.dtype == b.dtype == np.float32
    assert a.shape == b.shape
    np.testing.assert_array_equal(a.view(np.uint32), b.view(np.uint32))


class TestWriters:
    def test_a_workspace_reconstruction_carries_its_sift_readings(
        self, seoul_bull_workspace
    ):
        # Built through the Python path that copies the thumbnails from the
        # .sift files, which copies the readings beside them.
        recon = SfmrReconstruction.load(seoul_bull_workspace)
        image_to_gray, sharpness, tensors = _sift_readings(recon)
        assert recon.image_to_gray == image_to_gray
        _assert_bits_equal(recon.sharpness, sharpness)
        _assert_bits_equal(recon.gradient_structure_tensors_xx_yy_xy, tensors)

    def test_from_colmap_bin_copies_the_sift_readings(self, seoul_bull_workspace):
        workspace = seoul_bull_workspace.parent
        colmap_dir = workspace / "colmap_model"
        runner = CliRunner()
        result = runner.invoke(
            main, ["to-colmap-bin", str(seoul_bull_workspace), str(colmap_dir)]
        )
        assert result.exit_code == 0, result.output
        out = workspace / "imported.sfmr"
        result = runner.invoke(
            main,
            [
                "from-colmap-bin",
                str(colmap_dir),
                "--image-dir",
                str(workspace / "test_17_image"),
                "-o",
                str(out),
            ],
        )
        assert result.exit_code == 0, result.output
        imported = SfmrReconstruction.load(out)
        image_to_gray, sharpness, tensors = _sift_readings(imported)
        assert imported.image_to_gray == image_to_gray
        _assert_bits_equal(imported.sharpness, sharpness)
        _assert_bits_equal(imported.gradient_structure_tensors_xx_yy_xy, tensors)

    def test_a_merge_moves_each_row_with_its_image(
        self, seoul_bull_workspace, tmp_path
    ):
        from sfmtool.merge.reconstructions import merge_reconstructions
        from sfmtool.reconstruction import RangeExpr
        from sfmtool.xform import IncludeRangeFilter, apply_transforms

        original = SfmrReconstruction.load(seoul_bull_workspace)
        subsets = [
            apply_transforms(original, [IncludeRangeFilter(RangeExpr(spec))])
            for spec in ("1-10", "6-17")
        ]
        merged_path = tmp_path / "merged.sfmr"
        merge_reconstructions(reconstructions=subsets).save(
            merged_path, operation="merge_test"
        )
        merged = SfmrReconstruction.load(merged_path)
        by_name = dict(zip(original.image_names, original.sharpness))
        assert merged.image_to_gray == original.image_to_gray
        np.testing.assert_array_equal(
            merged.sharpness,
            np.asarray([by_name[n] for n in merged.image_names], dtype=np.float32),
        )


class TestOlderFiles:
    def test_the_ground_truth_reads_without_readings(
        self, seoul_bull_ground_truth_sfmr
    ):
        data = read_sfmr(seoul_bull_ground_truth_sfmr)
        assert data["metadata"]["version"] < 15
        assert data["sharpness"] is None
        assert data["gradient_structure_tensors_xx_yy_xy"] is None
        assert data["image_to_gray"] is None
        recon = SfmrReconstruction.load(seoul_bull_ground_truth_sfmr)
        assert recon.sharpness is None
        assert recon.gradient_structure_tensors_xx_yy_xy is None
        assert recon.image_to_gray is None

    def test_an_embedded_patches_file_is_saved_with_readings_from_its_images(
        self, seoul_bull_ground_truth_sfmr
    ):
        recon = SfmrReconstruction.load(seoul_bull_ground_truth_sfmr)
        from sfmtool import ProgressCounter

        out = seoul_bull_ground_truth_sfmr.parent / "resaved.sfmr"
        counter = ProgressCounter()
        recon.save(out, progress=counter)
        # The gather counts every image it read.
        assert counter.value == recon.image_count
        # The value itself is not changed by the save.
        assert recon.sharpness is None

        saved = SfmrReconstruction.load(out)
        assert read_sfmr(out)["metadata"]["version"] == 16
        sharpness, tensors = _image_readings(saved, saved.image_to_gray)
        _assert_bits_equal(saved.sharpness, sharpness)
        _assert_bits_equal(saved.gradient_structure_tensors_xx_yy_xy, tensors)

    def test_an_older_file_without_its_images_is_not_saved(
        self, seoul_bull_ground_truth_sfmr
    ):
        recon = SfmrReconstruction.load(seoul_bull_ground_truth_sfmr)
        workspace = seoul_bull_ground_truth_sfmr.parent
        missing = recon.image_names[3]
        (workspace / missing).unlink()
        changed = recon.image_names[5]
        (workspace / changed).write_bytes(b"not the photograph")
        out = workspace / "resaved.sfmr"
        with pytest.raises(ValueError) as error:
            recon.save(out)
        message = str(error.value)
        assert "2 of 17 images" in message
        assert missing in message and changed in message
        assert "do not hash" in message and "no such file" in message
        assert not out.exists()

    def test_write_sfmr_refuses_a_dict_without_readings(
        self, seoul_bull_ground_truth_sfmr
    ):
        data = read_sfmr(seoul_bull_ground_truth_sfmr)
        out = seoul_bull_ground_truth_sfmr.parent / "rewritten.sfmr"
        with pytest.raises(ValueError, match="has none"):
            write_sfmr(out, data)
        assert not out.exists()


class TestDicts:
    def test_the_readings_round_trip_through_the_dicts(self, seoul_bull_sfmr_only):
        data = read_sfmr(seoul_bull_sfmr_only)
        assert data["sharpness"].dtype == np.float32
        assert data["gradient_structure_tensors_xx_yy_xy"].shape == (17, 3)
        out = seoul_bull_sfmr_only.parent / "rewritten.sfmr"
        write_sfmr(out, data)
        back = read_sfmr(out)
        assert back["image_to_gray"] == data["image_to_gray"]
        _assert_bits_equal(back["sharpness"], data["sharpness"])

    @pytest.mark.parametrize(
        ("change", "message"),
        [
            ({"sharpness": None}, "together or not at all"),
            ({"image_to_gray": None}, "together or not at all"),
            ({"image_to_gray": {"formula": "G", "bits": 8}}, "unknown key"),
        ],
    )
    def test_a_dict_with_readings_the_format_refuses(
        self, seoul_bull_sfmr_only, change, message
    ):
        data = read_sfmr(seoul_bull_sfmr_only)
        data.update(change)
        with pytest.raises(ValueError, match=message):
            write_sfmr(seoul_bull_sfmr_only.parent / "bad.sfmr", data)

    def test_a_negative_reading_is_refused(self, seoul_bull_sfmr_only):
        data = read_sfmr(seoul_bull_sfmr_only)
        data["sharpness"] = data["sharpness"].copy()
        data["sharpness"][2] = -1.0
        with pytest.raises(ValueError, match="images/sharpness row 2"):
            write_sfmr(seoul_bull_sfmr_only.parent / "bad.sfmr", data)


class TestGettersAndClone:
    def test_the_getters_are_read_only_and_gathered_once(self, seoul_bull_sfmr_only):
        recon = SfmrReconstruction.load(seoul_bull_sfmr_only)
        for name in ("sharpness", "gradient_structure_tensors_xx_yy_xy"):
            first = getattr(recon, name)
            assert getattr(recon, name) is first
            assert not first.flags.writeable
            with pytest.raises(ValueError, match="read-only"):
                first[0] = first[0]
        assert recon.image_to_gray is not recon.image_to_gray
        assert recon.image_to_gray == read_sfmr(seoul_bull_sfmr_only)["image_to_gray"]

    def test_a_subset_and_a_reorder_keep_each_row_with_its_image(
        self, seoul_bull_sfmr_only
    ):
        recon = SfmrReconstruction.load(seoul_bull_sfmr_only)
        by_name = dict(zip(recon.image_names, recon.sharpness))
        subset = recon.subset_by_image_indices(np.asarray([9, 2, 4], dtype=np.uint32))
        np.testing.assert_array_equal(
            subset.sharpness, [by_name[n] for n in subset.image_names]
        )
        # Renaming the images through clone_with_changes: the rows follow the
        # names, and an image this reconstruction does not hold has none.
        names = list(recon.image_names)
        names[0], names[1] = names[1], names[0]
        names[2] = "not_in_the_reconstruction.jpg"
        renamed = recon.clone_with_changes(image_names=names)
        assert renamed.sharpness is None
        swapped = recon.clone_with_changes(
            image_names=names[:2] + list(recon.image_names[2:])
        )
        np.testing.assert_array_equal(
            swapped.sharpness, [by_name[n] for n in swapped.image_names]
        )

    def test_clone_with_changes_replaces_or_drops_the_readings(
        self, seoul_bull_sfmr_only
    ):
        recon = SfmrReconstruction.load(seoul_bull_sfmr_only)
        n = recon.image_count
        sharpness = np.linspace(0.0, 1.0, n, dtype=np.float32)
        tensors = np.tile(np.asarray([0.5, 0.25, -0.125], dtype=np.float32), (n, 1))
        image_to_gray = {"formula": "G", "quantize_bits": 8}
        replaced = recon.clone_with_changes(
            image_to_gray=image_to_gray,
            sharpness=sharpness,
            gradient_structure_tensors_xx_yy_xy=tensors,
        )
        assert replaced.image_to_gray == image_to_gray
        _assert_bits_equal(replaced.sharpness, sharpness)
        dropped = recon.clone_with_changes(
            image_to_gray=None, sharpness=None, gradient_structure_tensors_xx_yy_xy=None
        )
        assert dropped.sharpness is None and dropped.image_to_gray is None
        with pytest.raises(ValueError, match="passed together"):
            recon.clone_with_changes(sharpness=sharpness)
        with pytest.raises(ValueError, match="must be finite and not negative"):
            bad = sharpness.copy()
            bad[0] = np.nan
            recon.clone_with_changes(
                image_to_gray=image_to_gray,
                sharpness=bad,
                gradient_structure_tensors_xx_yy_xy=tensors,
            )
        with pytest.raises(ValueError, match="image_to_gray"):
            recon.clone_with_changes(
                image_to_gray={"formula": "R +"},
                sharpness=sharpness,
                gradient_structure_tensors_xx_yy_xy=tensors,
            )

    def test_a_dropped_value_is_saved_with_readings_from_its_sift_files(
        self, seoul_bull_workspace
    ):
        recon = SfmrReconstruction.load(seoul_bull_workspace)
        dropped = recon.clone_with_changes(
            image_to_gray=None, sharpness=None, gradient_structure_tensors_xx_yy_xy=None
        )
        out = seoul_bull_workspace.parent / "resaved.sfmr"
        dropped.save(out)
        saved = SfmrReconstruction.load(out)
        _assert_bits_equal(saved.sharpness, recon.sharpness)
        assert saved.image_to_gray == recon.image_to_gray
