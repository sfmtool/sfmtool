# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the `.sfmr` path check the commands share."""

from pathlib import Path

import click
import pytest
from click.testing import CliRunner

from sfmtool._commands._sfmr_path import check_sfmr_path
from sfmtool.cli import main


def test_accepts_sfmr_in_any_case():
    assert check_sfmr_path("a/recon.sfmr", "Input path") == Path("a/recon.sfmr")
    assert check_sfmr_path(Path("recon.SFMR"), "Input path") == Path("recon.SFMR")


def test_rejects_other_extension_with_label_and_path():
    with pytest.raises(click.UsageError) as excinfo:
        check_sfmr_path("recon.txt", "Reconstruction path")
    assert excinfo.value.message == (
        f"Reconstruction path must be a .sfmr file, got: {Path('recon.txt')}"
    )


@pytest.mark.parametrize(
    "args, label",
    [
        (["merge", "a.txt", "b.sfmr", "-o", "out.sfmr"], "Reconstruction path"),
        (["merge", "a.sfmr", "b.sfmr", "-o", "out.txt"], "Output path"),
        (["to-colmap-bin", "in.txt", "out"], "Input path"),
    ],
)
def test_commands_report_shared_wording(tmp_path, monkeypatch, args, label):
    monkeypatch.chdir(tmp_path)
    for name in ("a.txt", "a.sfmr", "b.sfmr", "in.txt"):
        (tmp_path / name).touch()
    result = CliRunner().invoke(main, args)
    assert result.exit_code == 2
    assert f"{label} must be a .sfmr file, got:" in result.output
