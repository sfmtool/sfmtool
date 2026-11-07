# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""The report `sfm version` prints after its first line: the facts that decide
how sfmtool runs on this machine, for bug reports and for agents.

specs/cli/other/version-command.md gives each fact's meaning and source, and
the keys of the JSON form. The facts read in Rust (SIMD level, CPU, thread
pool, memory, GPU adapter) come from the extension's `_machine_facts`; this
module adds the ones about the Python side and other programs. Each probe
that can fail reports the failure as text in its fact and never raises.
"""

import importlib
import json
import os
import platform
import shutil
import subprocess
import sys
import threading
from pathlib import Path

# How long the GPU adapter request may take before it is reported as timed
# out. It takes well under a second on a working driver; the limit is there so
# a driver that hangs cannot hang `sfm version`.
GPU_TIMEOUT_S = 10.0

# How long `ffmpeg -version` may take.
FFMPEG_TIMEOUT_S = 5.0

# The width the labels of the text report are padded to.
_LABEL_WIDTH = 10


def collect_report(sfmtool: dict) -> dict:
    """Every fact of the report, as the JSON object `sfm version --json`
    prints. `sfmtool` is the object `sfm version --bare --json` prints, with
    the version and the two git describes."""
    from ._sfmtool import _machine_facts

    # The GPU adapter request is the slowest probe after the pycolmap import,
    # and the binding releases the GIL while it waits, so the two overlap.
    machine: dict = {}
    thread = threading.Thread(
        target=lambda: machine.update(_machine_facts(GPU_TIMEOUT_S)),
        name="machine-facts",
        daemon=True,
    )
    thread.start()
    report = {
        "sfmtool": sfmtool,
        "install": install_facts(),
        "python": {"version": platform.python_version(), "executable": sys.executable},
        "os": {"platform": platform.platform(), "machine": platform.machine()},
        "pycolmap": pycolmap_facts(),
        "numpy": module_facts("numpy"),
        "opencv": module_facts("cv2"),
        "ffmpeg": ffmpeg_facts(),
    }
    thread.join()
    report["cpu"] = {
        "model": machine["cpu_model"],
        "logical_cores": machine["logical_cores"],
    }
    report["threads"] = {
        "rayon": machine["rayon_threads"],
        "rayon_num_threads_env": os.environ.get("RAYON_NUM_THREADS"),
    }
    report["simd"] = machine["simd"]
    report["memory_bytes"] = machine["memory_bytes"]
    report["gpu"] = machine["gpu"]
    return {key: report[key] for key in REPORT_KEYS}


# The top-level keys of the JSON report, in the order they are printed.
REPORT_KEYS = (
    "sfmtool",
    "install",
    "python",
    "os",
    "cpu",
    "threads",
    "simd",
    "memory_bytes",
    "gpu",
    "pycolmap",
    "numpy",
    "opencv",
    "ffmpeg",
)


def install_facts() -> dict:
    """Whether the `sfmtool` package is an editable install of a checkout,
    from its `direct_url.json` (PEP 610), and the directory it is run from."""
    from importlib.metadata import PackageNotFoundError, distribution

    location = str(Path(__file__).resolve().parent)
    try:
        direct_url = distribution("sfmtool").read_text("direct_url.json")
    except PackageNotFoundError:
        return {"editable": None, "location": location}
    editable = False
    if direct_url:
        try:
            editable = bool(json.loads(direct_url).get("dir_info", {}).get("editable"))
        except ValueError, AttributeError:
            editable = False
    return {"editable": editable, "location": location}


def _error_text(error: BaseException) -> str:
    return f"{type(error).__name__}: {error}"


def module_facts(name: str) -> dict:
    """The version of the Python module `name`, imported as sfmtool imports
    it: `{"version": ...}`, or `{"version": None, "error": ...}` with
    `"not installed"` when no such module exists, or the import's error."""
    try:
        module = importlib.import_module(name)
    except ModuleNotFoundError as error:
        if error.name == name:
            return {"version": None, "error": "not installed"}
        return {"version": None, "error": _error_text(error)}
    except Exception as error:  # A broken install can raise anything.
        return {"version": None, "error": _error_text(error)}
    return {"version": str(getattr(module, "__version__", "unknown"))}


def pycolmap_facts() -> dict:
    """pycolmap's version, the COLMAP version it wraps, whether it was built
    with CUDA (`has_cuda`) and, for a CUDA build, how many CUDA devices it can
    use now (`get_num_cuda_devices`, which only CUDA builds have)."""
    facts = module_facts("pycolmap")
    if facts["version"] is None:
        return facts
    pycolmap = sys.modules["pycolmap"]
    colmap = getattr(pycolmap, "COLMAP_version", None)
    if isinstance(colmap, str):
        colmap = colmap.removeprefix("COLMAP ").strip() or None
    else:
        colmap = None
    facts["colmap_version"] = colmap
    facts["cuda_built"] = bool(getattr(pycolmap, "has_cuda", False))
    facts["cuda_devices"] = None
    if facts["cuda_built"]:
        count = getattr(pycolmap, "get_num_cuda_devices", None)
        if count is None:
            facts["cuda_error"] = "this pycolmap cannot count CUDA devices"
        else:
            try:
                facts["cuda_devices"] = int(count())
            except Exception as error:
                facts["cuda_error"] = _error_text(error)
    return facts


def ffmpeg_facts() -> dict:
    """Where `ffmpeg` is on PATH, which `sfm insv2rig` runs to read video, and
    the first line of `ffmpeg -version`: `{"path": None}` when it is not
    found, and an `"error"` when it is found but does not run."""
    path = shutil.which("ffmpeg")
    if path is not None:
        path = os.path.normpath(path)
    if path is None:
        return {"path": None, "version": None}
    try:
        result = subprocess.run(
            [path, "-version"],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=FFMPEG_TIMEOUT_S,
        )
    except (OSError, subprocess.SubprocessError) as error:
        return {"path": path, "version": None, "error": _error_text(error)}
    first = result.stdout.splitlines()[0] if result.stdout else ""
    if result.returncode != 0 or not first:
        return {
            "path": path,
            "version": None,
            "error": f"`ffmpeg -version` exited with status {result.returncode}",
        }
    # "ffmpeg version 7.1-full_build Copyright (c) 2000-2024 ..." -> "7.1-full_build"
    words = first.split()
    version = (
        words[2] if len(words) > 2 and words[:2] == ["ffmpeg", "version"] else first
    )
    return {"path": path, "version": version}


def _gpu_text(gpu: dict) -> str:
    status = gpu["status"]
    if status == "found":
        return f"{gpu['name']} ({gpu['backend']}, {gpu['device_type']})"
    if status == "no_adapter":
        return f"no GPU adapter ({gpu['reason']})"
    if status == "timed_out":
        return f"no answer from the GPU driver within {GPU_TIMEOUT_S:g} s"
    if status == "failed":
        return "the GPU adapter request failed"
    if status == "not_built":
        return "not built with GPU support"
    return status


def _library_text(facts: dict) -> str:
    if facts["version"] is None:
        return facts["error"]
    return facts["version"]


def _pycolmap_text(facts: dict) -> str:
    if facts["version"] is None:
        return facts["error"]
    parts = [facts["version"]]
    if facts["colmap_version"] is not None:
        parts.append(f"COLMAP {facts['colmap_version']}")
    if not facts["cuda_built"]:
        parts.append("CPU only")
    elif facts["cuda_devices"] is not None:
        count = facts["cuda_devices"]
        parts.append(f"CUDA, {count} device{'' if count == 1 else 's'}")
    else:
        parts.append(f"CUDA build ({facts['cuda_error']})")
    return ", ".join(parts)


def _ffmpeg_text(facts: dict) -> str:
    if facts["path"] is None:
        return "not found on PATH"
    if facts["version"] is None:
        return f"{facts['path']} ({facts['error']})"
    return f"{facts['version']} ({facts['path']})"


def _memory_text(memory_bytes: int | None) -> str:
    if memory_bytes is None:
        return "unknown"
    return f"{memory_bytes / 2**30:.1f} GiB"


def report_text(report: dict) -> str:
    """The text `sfm version` prints for `report`, as `collect_report` returns
    it: the `--bare` line, then one line per fact with its label padded."""
    sfmtool = report["sfmtool"]
    install = report["install"]
    if install["editable"] is None:
        install_kind = "no package metadata"
    else:
        install_kind = "editable" if install["editable"] else "installed"
    cpu = report["cpu"]
    cores = cpu["logical_cores"]
    cpu_parts = [cpu["model"] or "unknown model"]
    cpu_parts.append(f"{cores} logical cores" if cores is not None else "cores unknown")
    threads = report["threads"]
    threads_text = str(threads["rayon"])
    if threads["rayon_num_threads_env"] is not None:
        threads_text += f" (RAYON_NUM_THREADS={threads['rayon_num_threads_env']})"
    describes = (
        f"extension built from {sfmtool['extension_describe'] or 'unknown'}, "
        f"checkout {sfmtool['checkout_describe'] or 'unknown'}"
    )
    lines = [
        ("Build", f"{sfmtool['build_profile']}, {describes}"),
        ("Install", f"{install_kind}, {install['location']}"),
        ("Python", f"{report['python']['version']}, {report['python']['executable']}"),
        ("OS", f"{report['os']['platform']}, {report['os']['machine']}"),
        ("CPU", ", ".join(cpu_parts)),
        ("Threads", threads_text),
        ("SIMD", report["simd"]),
        ("Memory", _memory_text(report["memory_bytes"])),
        ("GPU", _gpu_text(report["gpu"])),
        ("pycolmap", _pycolmap_text(report["pycolmap"])),
        ("numpy", _library_text(report["numpy"])),
        ("OpenCV", _library_text(report["opencv"])),
        ("ffmpeg", _ffmpeg_text(report["ffmpeg"])),
    ]
    body = [f"{label + ':':<{_LABEL_WIDTH}}{value}" for label, value in lines]
    return "\n".join([sfmtool["line"], *body])
