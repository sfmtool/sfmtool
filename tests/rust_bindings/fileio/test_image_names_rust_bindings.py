# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""An image name that is not a path relative to the workspace directory raises.

The `.sfmr` and `.matches` format crates hold every entry of
`images/names.json.zst` to the image-name rule
(`sfmtool_archive_io::validate_image_name`); its cases are tested there. These
tests check that the refusal reaches Python as the exception the format-error
table gives it: a `ValueError` from a writer and an `OSError` from a reader.
"""

import json
import re
import shutil
import zipfile
from compression import zstd

import numpy as np
import pytest

from sfmtool.fileio import read_sfmr, verify_sfmr, write_matches, write_sfmr

from tests.conftest import SEOUL_BULL_GROUND_TRUTH
from tests.rust_bindings.fileio.test_matches_file_rust_bindings import _pairwise_dict


def _names_error(name: str) -> str:
    return re.escape(f"image name 1 {json.dumps(name)}")


def test_write_sfmr_refuses_a_name_outside_the_workspace(tmp_path):
    data = read_sfmr(str(SEOUL_BULL_GROUND_TRUTH))
    n = len(data["image_names"])
    # Readings any write requires; their values do not matter here.
    data["image_to_gray"] = {"formula": "0.2126*R + 0.7152*G + 0.0722*B"}
    data["sharpness"] = np.ones(n, dtype=np.float32)
    data["gradient_structure_tensors_xx_yy_xy"] = np.ones((n, 3), dtype=np.float32)
    data["image_names"][1] = "../outside.jpg"
    path = tmp_path / "bad.sfmr"
    with pytest.raises(ValueError, match=_names_error("../outside.jpg")):
        write_sfmr(str(path), data)
    assert not path.exists()


def test_read_sfmr_refuses_a_name_with_a_backslash(tmp_path):
    path = tmp_path / "bad.sfmr"
    with (
        zipfile.ZipFile(SEOUL_BULL_GROUND_TRUTH) as src,
        zipfile.ZipFile(path, "w") as dst,
    ):
        for info in src.infolist():
            data = src.read(info)
            if info.filename == "images/names.json.zst":
                names = json.loads(zstd.decompress(data))
                names[1] = "images\\b.jpg"
                data = zstd.compress(json.dumps(names).encode())
            dst.writestr(info, data)
    with pytest.raises(OSError, match=_names_error("images\\b.jpg")):
        read_sfmr(str(path))
    valid, errors = verify_sfmr(str(path))
    assert not valid
    assert any(re.search(_names_error("images\\b.jpg"), e) for e in errors), errors

    # The unchanged file reads.
    shutil.copy(SEOUL_BULL_GROUND_TRUTH, path)
    assert read_sfmr(str(path))["image_names"][1] == "seoul_bull_sculpture_02.jpg"


def test_write_matches_refuses_an_absolute_name(tmp_path):
    data = _pairwise_dict()
    data["image_names"][1] = "/abs/b.jpg"
    path = tmp_path / "bad.matches"
    with pytest.raises(ValueError, match=_names_error("/abs/b.jpg")):
        write_matches(str(path), data)
    assert not path.exists()
