# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""`sfm version`: the line it prints, the two `git describe` strings it
reads and the report of this machine after it, checked without depending on
the state of this repository or on this machine's hardware."""

import importlib
import json
import re
import shutil
import subprocess
import sys
import types
from pathlib import Path

import pytest
from click.testing import CliRunner

from sfmtool._machine_report import (
    REPORT_KEYS,
    ffmpeg_facts,
    pycolmap_facts,
    report_text,
)
from sfmtool._sfmtool import _build_git_describe, _machine_facts
from sfmtool._version import checkout_describe, version_text
from sfmtool.cli import main

DESCRIBE = re.compile(r"(.+-\d+-g)?[0-9a-f]{7,}(-dirty)?")


def test_release_prints_the_version_alone():
    release = "v0.3-0-g276e8d986"
    assert version_text("0.3", None, None) == "sfmtool 0.3"
    assert version_text("0.3", release, release) == "sfmtool 0.3"
    assert version_text("0.3", None, release) == "sfmtool 0.3"
    assert version_text("0.3", release, None) == "sfmtool 0.3"
    assert version_text("0.3", "0.3-0-g276e8d986", None) == "sfmtool 0.3"


def test_past_the_tag_prints_the_describe():
    past = "v0.3-385-g276e8d986"
    assert version_text("0.3", past, past) == f"sfmtool 0.3 ({past})"
    assert version_text("0.3", past, None) == f"sfmtool 0.3 ({past})"
    assert version_text("0.3", None, past) == f"sfmtool 0.3 ({past})"
    # A shallow clone with no tag: `--always` gives the hash alone.
    assert version_text("0.3", "276e8d986", None) == "sfmtool 0.3 (276e8d986)"


def test_a_tag_other_than_the_version_is_printed():
    other = "v0.4-0-g276e8d986"
    assert version_text("0.3", other, other) == f"sfmtool 0.3 ({other})"
    assert version_text("0.3", "v0.30-0-g276e8d986", None) == (
        "sfmtool 0.3 (v0.30-0-g276e8d986)"
    )


def test_dirty_is_printed_even_at_the_tag():
    dirty = "v0.3-0-g276e8d986-dirty"
    assert version_text("0.3", dirty, dirty) == f"sfmtool 0.3 ({dirty})"
    past_dirty = "v0.3-385-g276e8d986-dirty"
    assert version_text("0.3", past_dirty, past_dirty) == (
        f"sfmtool 0.3 ({past_dirty})"
    )


def test_extension_differing_from_the_checkout_prints_both():
    assert version_text("0.3", "v0.3-386-g1a2b3c4d5", "v0.3-385-g276e8d986") == (
        "sfmtool 0.3 (checkout v0.3-386-g1a2b3c4d5, "
        "extension built from v0.3-385-g276e8d986)"
    )
    # Both labelled even when one of them is the release.
    assert version_text("0.3", "v0.3-0-g276e8d986-dirty", "v0.3-0-g276e8d986") == (
        "sfmtool 0.3 (checkout v0.3-0-g276e8d986-dirty, "
        "extension built from v0.3-0-g276e8d986)"
    )


def test_checkout_describe_skips_a_directory_outside_a_checkout(tmp_path, monkeypatch):
    """No `.git` two levels above the package: git is not run at all."""
    package_dir = tmp_path / "src" / "sfmtool"
    package_dir.mkdir(parents=True)

    def no_subprocess(*args, **kwargs):
        raise AssertionError("git was run outside a checkout")

    monkeypatch.setattr(subprocess, "run", no_subprocess)
    assert checkout_describe(package_dir) is None


def test_checkout_describe_skips_a_failing_git(tmp_path):
    """A `.git` that is not a repository: git exits with an error."""
    package_dir = tmp_path / "src" / "sfmtool"
    package_dir.mkdir(parents=True)
    (tmp_path / ".git").mkdir()
    assert checkout_describe(package_dir) is None


def test_checkout_describe_skips_a_missing_git(tmp_path, monkeypatch):
    """No `git` on PATH: starting it fails, and that is not an error."""
    package_dir = tmp_path / "src" / "sfmtool"
    package_dir.mkdir(parents=True)
    (tmp_path / ".git").mkdir()
    empty = tmp_path / "empty"
    empty.mkdir()
    monkeypatch.setenv("PATH", str(empty))
    assert checkout_describe(package_dir) is None


@pytest.mark.skipif(shutil.which("git") is None, reason="needs git")
def test_checkout_describe_reads_a_real_checkout(tmp_path):
    """In a repository laid out as `<root>/src/sfmtool`, at a tag and then
    with an uncommitted change."""
    package_dir = tmp_path / "src" / "sfmtool"
    package_dir.mkdir(parents=True)
    module = package_dir / "__init__.py"
    module.write_text("x = 1\n", encoding="utf-8")

    def git(*args):
        subprocess.run(
            [
                "git",
                "-C",
                str(tmp_path),
                "-c",
                "user.name=Test",
                "-c",
                "user.email=test@example.com",
                "-c",
                "commit.gpgsign=false",
                *args,
            ],
            check=True,
            capture_output=True,
            timeout=30,
        )

    git("init", "-q")
    git("add", ".")
    git("commit", "-q", "-m", "first")
    git("tag", "v0.3")
    describe = checkout_describe(package_dir)
    assert re.fullmatch(r"v0\.3-0-g[0-9a-f]+", describe), describe
    assert version_text("0.3", describe, None) == "sfmtool 0.3"

    module.write_text("x = 2\n", encoding="utf-8")
    dirty = checkout_describe(package_dir)
    assert dirty == f"{describe}-dirty"
    assert version_text("0.3", dirty, None) == f"sfmtool 0.3 ({dirty})"

    # A linked worktree, whose `.git` is a file rather than a directory.
    worktree = tmp_path / "worktree"
    git("worktree", "add", "-q", "--detach", str(worktree), "v0.3")
    assert (worktree / ".git").is_file()
    assert checkout_describe(worktree / "src" / "sfmtool") == describe


def test_extension_records_its_build_describe():
    """The extension under test was built from this repository's git
    checkout, so `build.rs` recorded a describe."""
    describe = _build_git_describe()
    assert isinstance(describe, str)
    assert DESCRIBE.fullmatch(describe), describe


def _expected_bare_line():
    """The line `sfm version` printed before it printed a report, computed
    from its parts as that command did."""
    from importlib.metadata import version

    import sfmtool

    package_dir = Path(sfmtool.__file__).resolve().parent
    return version_text(
        version("sfmtool"), checkout_describe(package_dir), _build_git_describe()
    )


def test_bare_prints_the_version_line_and_imports_nothing_heavy():
    """`--bare` prints the line `sfm version` printed before the report was
    added, and reads no fact that needs pycolmap, numpy, OpenCV or the GPU.
    Run in a fresh interpreter, since this one has imported them already."""
    script = (
        "import sys\n"
        "from sfmtool.cli import main\n"
        "try:\n"
        "    main(['version', '--bare'])\n"
        "except SystemExit:\n"
        "    pass\n"
        "heavy = sorted(n for n in ('pycolmap', 'numpy', 'cv2', "
        "'sfmtool._machine_report') if n in sys.modules)\n"
        "print('loaded:', heavy)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=120
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == [_expected_bare_line(), "loaded: []"]


def test_bare_json_prints_the_sfmtool_object():
    result = CliRunner().invoke(main, ["version", "--bare", "--json"])
    assert result.exit_code == 0, result.output
    report = json.loads(result.output)
    assert list(report) == ["sfmtool"]
    sfmtool = report["sfmtool"]
    assert list(sfmtool) == [
        "version",
        "line",
        "checkout_describe",
        "extension_describe",
        "build_profile",
    ]
    assert sfmtool["line"] == _expected_bare_line()
    assert sfmtool["extension_describe"] == _build_git_describe()
    assert sfmtool["build_profile"] in ("debug", "release")


def _report(**changes):
    """A report as `collect_report` returns it, with made-up facts."""
    report = {
        "sfmtool": {
            "version": "0.3",
            "line": "sfmtool 0.3 (v0.3-385-g276e8d986)",
            "checkout_describe": "v0.3-385-g276e8d986",
            "extension_describe": "v0.3-385-g276e8d986",
            "build_profile": "release",
        },
        "install": {"editable": True, "location": "/src/sfmtool/src/sfmtool"},
        "python": {"version": "3.14.9", "executable": "/env/bin/python"},
        "os": {"platform": "Linux-6.8-x86_64-with-glibc2.39", "machine": "x86_64"},
        "cpu": {"model": "Example CPU 9000", "logical_cores": 16},
        "threads": {"rayon": 16, "rayon_num_threads_env": None},
        "simd": "AVX2+FMA",
        "memory_bytes": 64 * 2**30,
        "gpu": {
            "status": "found",
            "name": "Example GPU",
            "backend": "vulkan",
            "device_type": "discrete",
        },
        "pycolmap": {
            "version": "4.2.1",
            "colmap_version": "4.2.1",
            "cuda_built": False,
            "cuda_devices": None,
        },
        "numpy": {"version": "2.5.4"},
        "opencv": {"version": "5.0.0"},
        "ffmpeg": {"path": "/usr/bin/ffmpeg", "version": "7.1"},
    }
    report.update(changes)
    return report


def test_report_text_prints_one_aligned_line_per_fact():
    assert report_text(_report()) == "\n".join(
        [
            "sfmtool 0.3 (v0.3-385-g276e8d986)",
            "Build:    release, v0.3-385-g276e8d986",
            "Install:  editable, /src/sfmtool/src/sfmtool",
            "Python:   3.14.9, /env/bin/python",
            "OS:       Linux-6.8-x86_64-with-glibc2.39, x86_64",
            "CPU:      Example CPU 9000, 16 logical cores",
            "Threads:  16",
            "SIMD:     AVX2+FMA",
            "Memory:   64.0 GiB",
            "GPU:      Example GPU (vulkan, discrete)",
            "pycolmap: 4.2.1, COLMAP 4.2.1, CPU only",
            "numpy:    2.5.4",
            "OpenCV:   5.0.0",
            "ffmpeg:   7.1, /usr/bin/ffmpeg",
        ]
    )


def _line(report, label):
    (line,) = [
        line for line in report_text(report).splitlines() if line.startswith(label)
    ]
    return line


@pytest.mark.parametrize(
    ("extension", "checkout", "text"),
    [
        (
            "v0.3-385-g276e8d986",
            "v0.3-386-g1a2b3c4d5",
            "extension built from v0.3-385-g276e8d986, checkout v0.3-386-g1a2b3c4d5",
        ),
        (
            "v0.3-385-g276e8d986",
            None,
            "extension built from v0.3-385-g276e8d986, checkout unknown",
        ),
        (
            None,
            "v0.3-386-g1a2b3c4d5",
            "extension built from unknown, checkout v0.3-386-g1a2b3c4d5",
        ),
        (None, None, "commit unknown"),
    ],
)
def test_build_line_labels_the_commits_unless_they_agree(extension, checkout, text):
    sfmtool = {
        **_report()["sfmtool"],
        "extension_describe": extension,
        "checkout_describe": checkout,
        "build_profile": "debug",
    }
    assert _line(_report(sfmtool=sfmtool), "Build:") == f"Build:    debug, {text}"


def test_report_text_states_what_is_missing():
    report = _report(
        install={"editable": False, "location": "/site-packages/sfmtool"},
        cpu={"model": None, "logical_cores": None},
        threads={"rayon": 4, "rayon_num_threads_env": "4"},
        memory_bytes=None,
        gpu={"status": "no_adapter", "reason": "No suitable adapter found"},
        pycolmap={"version": None, "error": "not installed"},
        opencv={"version": None, "error": "not installed"},
        ffmpeg={"path": None, "version": None},
    )
    assert _line(report, "Install:") == "Install:  installed, /site-packages/sfmtool"
    assert _line(report, "CPU:") == "CPU:      unknown model, cores unknown"
    assert _line(report, "Threads:") == "Threads:  4 (RAYON_NUM_THREADS=4)"
    assert _line(report, "Memory:") == "Memory:   unknown"
    assert _line(report, "GPU:") == (
        "GPU:      no GPU adapter (No suitable adapter found)"
    )
    assert _line(report, "pycolmap:") == "pycolmap: not installed"
    assert _line(report, "OpenCV:") == "OpenCV:   not installed"
    assert _line(report, "ffmpeg:") == "ffmpeg:   not found on PATH"


@pytest.mark.parametrize(
    ("gpu", "text"),
    [
        (
            {
                "status": "found",
                "name": "llvmpipe",
                "backend": "vulkan",
                "device_type": "CPU (software)",
            },
            "llvmpipe (vulkan, CPU (software))",
        ),
        ({"status": "timed_out"}, "no answer from the GPU driver within 10 s"),
        ({"status": "failed"}, "the GPU adapter request failed"),
        ({"status": "not_built"}, "not built with GPU support"),
    ],
)
def test_report_text_gpu_states(gpu, text):
    assert _line(_report(gpu=gpu), "GPU:") == f"GPU:      {text}"


@pytest.mark.parametrize(
    ("cuda", "text"),
    [
        ({"cuda_built": True, "cuda_devices": 1}, "CUDA, 1 device"),
        ({"cuda_built": True, "cuda_devices": 0}, "CUDA, 0 devices"),
        (
            {"cuda_built": True, "cuda_devices": None, "cuda_error": "boom"},
            "CUDA build (boom)",
        ),
    ],
)
def test_report_text_cuda_states(cuda, text):
    pycolmap = {"version": "4.0.4", "colmap_version": "4.0.4", **cuda}
    assert _line(_report(pycolmap=pycolmap), "pycolmap:") == (
        f"pycolmap: 4.0.4, COLMAP 4.0.4, {text}"
    )


def _fake_pycolmap(monkeypatch, **attributes):
    module = types.ModuleType("pycolmap")
    module.__version__ = "4.0.4"
    module.COLMAP_version = "COLMAP 4.0.4"
    for name, value in attributes.items():
        setattr(module, name, value)
    monkeypatch.setitem(sys.modules, "pycolmap", module)


def test_pycolmap_facts_reads_cuda_support(monkeypatch):
    _fake_pycolmap(monkeypatch, has_cuda=False)
    assert pycolmap_facts() == {
        "version": "4.0.4",
        "colmap_version": "4.0.4",
        "cuda_built": False,
        "cuda_devices": None,
    }
    _fake_pycolmap(monkeypatch, has_cuda=True, get_num_cuda_devices=lambda: 2)
    assert pycolmap_facts()["cuda_devices"] == 2

    def no_driver():
        raise RuntimeError("no CUDA driver")

    _fake_pycolmap(monkeypatch, has_cuda=True, get_num_cuda_devices=no_driver)
    facts = pycolmap_facts()
    assert facts["cuda_devices"] is None
    assert facts["cuda_error"] == "RuntimeError: no CUDA driver"
    _fake_pycolmap(monkeypatch, has_cuda=True)
    assert pycolmap_facts()["cuda_error"] == "this pycolmap cannot count CUDA devices"


def test_pycolmap_facts_reports_a_failed_import(monkeypatch):
    monkeypatch.setitem(sys.modules, "pycolmap", None)
    assert pycolmap_facts() == {"version": None, "error": "not installed"}

    real_import = importlib.import_module

    def broken(name, *args):
        if name == "pycolmap":
            raise OSError("DLL load failed")
        return real_import(name, *args)

    monkeypatch.delitem(sys.modules, "pycolmap")
    monkeypatch.setattr(importlib, "import_module", broken)
    assert pycolmap_facts() == {"version": None, "error": "OSError: DLL load failed"}


def test_ffmpeg_facts(monkeypatch, tmp_path):
    monkeypatch.setenv("PATH", str(tmp_path))
    assert ffmpeg_facts() == {"path": None, "version": None}

    found = str(tmp_path / "ffmpeg")
    monkeypatch.setattr(shutil, "which", lambda name: found)
    # The file does not exist, so starting it fails.
    facts = ffmpeg_facts()
    assert facts["path"] == found and facts["version"] is None
    assert facts["error"].startswith("FileNotFoundError")

    def ran(args, **kwargs):
        return subprocess.CompletedProcess(
            args, 0, "ffmpeg version 7.1-full_build Copyright (c) 2000-2024\n", ""
        )

    monkeypatch.setattr(subprocess, "run", ran)
    assert ffmpeg_facts() == {"path": found, "version": "7.1-full_build"}


def test_machine_facts_refuses_a_bad_timeout():
    for bad in (-1.0, float("nan"), float("inf"), 1e300):
        with pytest.raises(ValueError, match="gpu_timeout_s"):
            _machine_facts(bad)


def test_json_report_has_the_documented_keys():
    """One real run on this machine: the keys and types of each fact, not
    their values, which depend on the hardware."""
    result = CliRunner().invoke(main, ["version", "--json"])
    assert result.exit_code == 0, result.output
    report = json.loads(result.output)
    assert tuple(report) == REPORT_KEYS
    assert report["sfmtool"]["line"] == _expected_bare_line()
    assert report["simd"] in ("AVX2+FMA", "AVX2", "SSE2", "none (scalar)")
    assert report["threads"]["rayon"] >= 1
    assert report["gpu"]["status"] in (
        "found",
        "no_adapter",
        "timed_out",
        "failed",
        "not_built",
    )
    assert isinstance(report["install"]["editable"], bool)
    assert report["python"]["executable"] == sys.executable
    for library in ("pycolmap", "numpy", "opencv"):
        assert "version" in report[library], library
    assert set(report["ffmpeg"]) >= {"path", "version"}


def test_version_command_prints_the_report(monkeypatch):
    report = _report()
    monkeypatch.setattr(
        "sfmtool._machine_report.collect_report", lambda sfmtool: report
    )
    result = CliRunner().invoke(main, ["version"])
    assert result.exit_code == 0, result.output
    assert result.output == report_text(report) + "\n"
