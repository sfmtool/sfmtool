# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0
"""Launch, inspect and stop an SfM Explorer for a bug bash.

    python viewer.py launch --copy C:/DataSets/Foo/sfmr/a.sfmr --copy ...
    python viewer.py status --session <dir>
    python viewer.py stop --session <dir>

`launch` picks a random free port, copies each `--copy` source into a
`bugbash-<date>-<port>/` folder beside it (inside its own workspace, so its
images still resolve), copies the viewer binary into a session directory so a
rebuild or a second run is not blocked by a locked executable, starts the
viewer detached with `--mcp <port> --no-default-layout` on the copies, waits
for the endpoint to answer, and prints the session as JSON. Every later call
names the port; nothing is kept in shell state.

The session's `catalog` says whether `sfm explorer-call` can check calls to
this viewer: "matches" when the extension's tool catalog is the viewer's, or
the refusal that names the difference, in which case rebuild the extension
with `pixi run maturin develop --release` or pass `--validate-tool-names-only`
to each call.
"""

import argparse
import datetime
import json
import os
import random
import shutil
import signal
import socket
import subprocess
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
EXE_NAME = "sfm-explorer.exe" if os.name == "nt" else "sfm-explorer"
PORT_RANGE = (20000, 60000)


def free_port(rng: random.Random) -> int:
    """A random port in PORT_RANGE that nothing is listening on right now."""
    for _ in range(200):
        port = rng.randrange(*PORT_RANGE)
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", port))
            except OSError:
                continue
            return port
    raise SystemExit("no free port found in %d-%d" % PORT_RANGE)


def copy_sources(sources: list[str], port: int) -> list[dict]:
    """Copy each source into bugbash-<date>-<port>/ beside it.

    A source may be written SRC=NAME to choose the copy's stem, which is also
    the label the viewer gives the node.
    """
    tag = f"bugbash-{datetime.date.today().isoformat()}-{port}"
    copies = []
    for spec in sources:
        src, _, name = spec.partition("=")
        src_path = Path(src).resolve()
        if not src_path.is_file():
            raise SystemExit(f"not a file: {src_path}")
        dest_dir = src_path.parent / tag
        dest_dir.mkdir(exist_ok=True)
        dest = dest_dir / f"{name or src_path.stem}.sfmr"
        if dest.exists():
            raise SystemExit(f"refusing to overwrite {dest}")
        shutil.copy2(src_path, dest)
        copies.append({"source": str(src_path), "copy": dest.as_posix()})
    return copies


def start(exe: Path, port: int, files: list[str], log: Path) -> int:
    args = [str(exe), "--mcp", str(port), "--no-default-layout", *files]
    out = open(log, "wb")
    if os.name == "nt":
        flags = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
        proc = subprocess.Popen(
            args, stdout=out, stderr=subprocess.STDOUT, creationflags=flags
        )
    else:
        proc = subprocess.Popen(
            args, stdout=out, stderr=subprocess.STDOUT, start_new_session=True
        )
    return proc.pid


def explorer_call(port: int, tool: str, *options: str) -> subprocess.CompletedProcess:
    """Run `sfm explorer-call` on the viewer at `port`."""
    sfm = shutil.which("sfm")
    if sfm is None:
        raise SystemExit("`sfm` is not on PATH: run this with `pixi run python`")
    return subprocess.run(
        [sfm, "explorer-call", "--mcp", str(port), tool, *options],
        capture_output=True,
        text=True,
        encoding="utf-8",
    )


def get_scene(port: int) -> dict:
    """The viewer's get_scene reply. Only the tool name is checked, so this
    works whether or not the extension's catalog is the viewer's."""
    done = explorer_call(port, "get_scene", "--validate-tool-names-only")
    if done.returncode != 0:
        raise OSError(done.stderr.strip() or f"status {done.returncode}")
    return json.loads(done.stdout)


def catalog_state(port: int) -> str:
    """ "matches", or the refusal naming how the viewer's catalog differs."""
    done = explorer_call(port, "get_scene")
    return "matches" if done.returncode == 0 else done.stderr.strip()


def wait_ready(port: int, pid: int, timeout_s: float) -> dict:
    """Wait until the endpoint answers get_scene from the viewer we started."""
    deadline = time.time() + timeout_s
    last = None
    while time.time() < deadline:
        if not alive(pid):
            raise SystemExit(f"viewer {pid} exited before its endpoint answered")
        try:
            scene = get_scene(port)
            if f":{port}]" in scene.get("window_title", ""):
                return scene
        except OSError as e:
            last = e
        time.sleep(0.5)
    raise SystemExit(f"endpoint on {port} did not answer in {timeout_s}s: {last}")


def alive(pid: int) -> bool:
    if os.name == "nt":
        out = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}", "/NH"],
            capture_output=True,
            text=True,
        ).stdout
        return str(pid) in out
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def launch(a: argparse.Namespace) -> None:
    built = REPO / "target" / "release" / EXE_NAME
    if not built.is_file():
        raise SystemExit(
            f"{built} is missing: run `pixi run cargo build --release -p sfm-explorer`"
        )
    rng = random.Random(a.seed)
    port = a.port or free_port(rng)
    session = Path(a.session or tempfile.mkdtemp(prefix=f"gui-bug-bash-{port}-"))
    session.mkdir(parents=True, exist_ok=True)
    exe = built
    if not a.no_copy_exe:
        exe = session / EXE_NAME
        shutil.copy2(built, exe)
    copies = copy_sources(a.copy, port)
    files = [c["copy"] for c in copies] + list(a.open)
    log = session / "viewer.log"
    pid = start(exe, port, files, log)
    scene = wait_ready(port, pid, a.timeout)
    info = {
        "port": port,
        "pid": pid,
        "session": session.as_posix(),
        "exe": exe.as_posix(),
        "log": log.as_posix(),
        "copies": copies,
        "opened": files,
        "labels": [n["label"] for n in scene.get("scene", [])],
        "catalog": catalog_state(port),
        "started": datetime.datetime.now().isoformat(timespec="seconds"),
    }
    (session / "session.json").write_text(json.dumps(info, indent=2))
    print(json.dumps(info, indent=2))


def load(session: str) -> dict:
    return json.loads((Path(session) / "session.json").read_text())


def status(a: argparse.Namespace) -> None:
    info = load(a.session)
    info["alive"] = alive(info["pid"])
    if info["alive"]:
        try:
            scene = get_scene(info["port"])
            info["labels"] = [n["label"] for n in scene.get("scene", [])]
            info["catalog"] = catalog_state(info["port"])
        except OSError as e:
            info["endpoint_error"] = str(e)
    print(json.dumps(info, indent=2))


def stop(a: argparse.Namespace) -> None:
    info = load(a.session)
    pid = info["pid"]
    if not alive(pid):
        print(f"viewer {pid} is not running")
        return
    if os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(pid), "/F"], capture_output=True)
    else:
        os.kill(pid, signal.SIGTERM)
    print(f"stopped viewer {pid} on port {info['port']}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="command", required=True)

    lp = sub.add_parser("launch", help="copy sources and start a viewer")
    lp.add_argument(
        "--copy",
        action="append",
        default=[],
        metavar="SRC[=NAME]",
        help="a .sfmr to copy into bugbash-<date>-<port>/ beside it and open",
    )
    lp.add_argument(
        "--open",
        action="append",
        default=[],
        metavar="PATH",
        help="a file to open as it is, without copying (corrupt-file probes)",
    )
    lp.add_argument("--port", type=int, help="use this port instead of a random one")
    lp.add_argument("--seed", type=int, help="seed for the random port")
    lp.add_argument("--session", help="session directory (default: a new temp dir)")
    lp.add_argument(
        "--no-copy-exe",
        action="store_true",
        help="run target/release directly (blocks rebuilds on Windows)",
    )
    lp.add_argument("--timeout", type=float, default=120.0)
    lp.set_defaults(func=launch)

    for name, func in (("status", status), ("stop", stop)):
        sp = sub.add_parser(name)
        sp.add_argument("--session", required=True)
        sp.set_defaults(func=func)

    a = p.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
