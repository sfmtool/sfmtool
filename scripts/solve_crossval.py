#!/usr/bin/env python3
# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0
"""End-to-end SfM cross-validation: sfmtool SIFT vs a reference backend.

Unlike `sift_crossval.py`, which compares *detector output* (keypoint positions
and descriptors), this script runs full reconstructions and compares the
resulting models. For each dataset it builds two workspaces that differ only in
the SIFT backend (`--feature-tool sfmtool` vs `colmap`); everything downstream —
the COLMAP exhaustive matcher, the solver, the seed and the feature budget — is
held constant, so differences in the table are attributable to the features.

For each reconstruction it reports the largest model's posed-image count, number
of 3D points, mean/median reprojection error (px) and mean track length.

Solver: flat single-camera datasets use incremental SfM; the kerry_park fisheye
rig uses global SfM (GLOMAP) by default, because incremental initial-pair
bootstrap is fragile on that rig for *either* backend (override with --solver).

Run inside the project environment, e.g.:

    pixi run -e test python scripts/solve_crossval.py
    pixi run -e test python scripts/solve_crossval.py --datasets seoul dino
    pixi run -e test python scripts/solve_crossval.py --backends sfmtool colmap \
        --solver incremental --max-features 4096 --keep
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA = REPO_ROOT / "test-data" / "images"

# Per-dataset workspace layout. "flat" datasets are a single camera in images/;
# "rig" datasets place each sensor in its own directory plus a rig_config.json.
DATASETS: dict[str, dict] = {
    "seoul": {
        "label": "seoul_bull (17)",
        "src": "seoul_bull_sculpture",
        "kind": "flat",
        "glob": "seoul_bull_sculpture_*.jpg",
        "total": 17,
    },
    "dino": {
        "label": "dino_dog_toy (85)",
        "src": "dino_dog_toy",
        "kind": "flat",
        "glob": "dino_dog_toy_*.jpg",
        "total": 85,
    },
    "seattle": {
        "label": "seattle_backyard (26)",
        "src": "seattle_backyard",
        "kind": "flat",
        "glob": "seattle_backyard_*.jpg",
        "total": 26,
    },
    "kerry": {
        "label": "kerry_park rig (48)",
        "src": "kerry_park",
        "kind": "rig",
        "sensors": ["fisheye_left", "fisheye_right"],
        "total": 48,
        "default_solver": "global",
    },
}


def _die(msg: str) -> None:
    print(f"error: {msg}", file=sys.stderr)
    sys.exit(1)


def setup_workspace(ds: dict, backend: str, root: Path) -> tuple[Path, list[str]]:
    """Create a fresh workspace for `ds` with the given backend; return (ws, inputs).

    `inputs` are the solve input arguments relative to the workspace.
    """
    ws = root / f"{ds['src']}_{backend}"
    if ws.exists():
        shutil.rmtree(ws)
    if ds["kind"] == "flat":
        (ws / "images").mkdir(parents=True)
        srcs = sorted((DATA / ds["src"]).glob(ds["glob"]))
        if not srcs:
            _die(f"no images for {ds['src']} (looked for {ds['glob']})")
        for p in srcs:
            shutil.copy(p, ws / "images")
        inputs = ["images"]
    else:
        for sensor in ds["sensors"]:
            (ws / sensor).mkdir(parents=True)
            for p in sorted((DATA / ds["src"] / sensor).glob("frame_*.jpg")):
                shutil.copy(p, ws / sensor)
        shutil.copy(DATA / ds["src"] / "rig_config.json", ws / "rig_config.json")
        inputs = list(ds["sensors"])
    return ws, inputs


def run_solve(
    sfm: str,
    ws: Path,
    inputs: list[str],
    backend: str,
    solver: str,
    seed: int,
    maxf: int,
) -> bool:
    """Init the workspace and run an end-to-end solve. Returns True on success."""
    init_cmd = [sfm, "ws", "init", "--feature-tool", backend]
    if backend == "colmap":
        # COLMAP SIFT defaults to GPU; force CPU so this runs headless.
        init_cmd.append("--no-gpu")
    init_cmd.append(str(ws))
    init = subprocess.run(init_cmd, capture_output=True, text=True)
    if init.returncode != 0:
        print(f"    ws init failed:\n{init.stderr.strip()}")
        return False

    flag = "-g" if solver == "global" else "-i"
    solve_cmd = [
        sfm,
        "solve",
        flag,
        "--seed",
        str(seed),
        "--max-features",
        str(maxf),
        "-o",
        "recon.sfmr",
        *inputs,
    ]
    with open(ws / "solve.log", "w") as log:
        rc = subprocess.run(
            solve_cmd, cwd=ws, stdout=log, stderr=subprocess.STDOUT
        ).returncode
    return rc == 0 and any(ws.glob("recon*.sfmr"))


def best_model(ws: Path):
    """Load every recon*.sfmr in `ws`; return the model with the most posed images."""
    from sfmtool._sfmtool import SfmrReconstruction

    best, best_n = None, -1
    for f in sorted(ws.glob("recon*.sfmr")):
        try:
            r = SfmrReconstruction.load(str(f))
        except Exception as e:  # noqa: BLE001
            print(f"    (failed to load {f.name}: {e})")
            continue
        if r.image_count > best_n:
            best, best_n = r, r.image_count
    return best


def metrics(r) -> dict | None:
    """Summary metrics for one reconstruction, or None if it is empty."""
    if r is None:
        return None
    err = np.asarray(r.errors, dtype=np.float64)
    err = err[np.isfinite(err)]
    obs = np.asarray(r.observation_counts, dtype=np.float64)
    return {
        "imgs": r.image_count,
        "pts": r.point_count,
        "mean_err": float(err.mean()) if err.size else float("nan"),
        "med_err": float(np.median(err)) if err.size else float("nan"),
        "track": float(obs.mean()) if obs.size else float("nan"),
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--datasets",
        nargs="+",
        choices=list(DATASETS),
        default=list(DATASETS),
        help="datasets to compare (default: all 4)",
    )
    ap.add_argument(
        "--backends",
        nargs="+",
        choices=["sfmtool", "colmap"],
        default=["sfmtool", "colmap"],
        help="SIFT backends to compare (default: both)",
    )
    ap.add_argument(
        "--solver",
        choices=["auto", "incremental", "global"],
        default="auto",
        help="SfM solver; 'auto' uses each dataset's recommended solver (default)",
    )
    ap.add_argument(
        "--max-features", type=int, default=4096, help="feature budget per image"
    )
    ap.add_argument("--seed", type=int, default=42, help="solver random seed")
    ap.add_argument(
        "--sfm-bin", default="sfm", help="sfm CLI entry point (default: sfm)"
    )
    ap.add_argument("--workdir", help="where to build workspaces (default: a temp dir)")
    ap.add_argument(
        "--keep", action="store_true", help="keep workspaces instead of deleting"
    )
    args = ap.parse_args()

    sys.path.insert(0, str(REPO_ROOT / "src"))
    root = (
        Path(args.workdir)
        if args.workdir
        else Path(tempfile.mkdtemp(prefix="solve_crossval_"))
    )
    root.mkdir(parents=True, exist_ok=True)
    print(
        f"workdir: {root}   solver: {args.solver}   max-features: {args.max_features}\n"
    )

    hdr = (
        f"{'dataset':<22} {'backend':<8} {'solver':<11} {'posed':>7} {'points':>8} "
        f"{'mean_err':>9} {'med_err':>8} {'track':>6}"
    )
    print(hdr)
    print("-" * len(hdr))

    try:
        for key in args.datasets:
            ds = DATASETS[key]
            solver = (
                args.solver
                if args.solver != "auto"
                else ds.get("default_solver", "incremental")
            )
            for backend in args.backends:
                ws, inputs = setup_workspace(ds, backend, root)
                ok = run_solve(
                    args.sfm_bin,
                    ws,
                    inputs,
                    backend,
                    solver,
                    args.seed,
                    args.max_features,
                )
                m = metrics(best_model(ws)) if ok else None
                if m is None:
                    print(
                        f"{ds['label']:<22} {backend:<8} {solver:<11}   (no reconstruction; see {ws}/solve.log)"
                    )
                    continue
                print(
                    f"{ds['label']:<22} {backend:<8} {solver:<11} "
                    f"{m['imgs']:>3}/{ds['total']:<3} {m['pts']:>8} "
                    f"{m['mean_err']:>9.3f} {m['med_err']:>8.3f} {m['track']:>6.2f}"
                )
            print()
    finally:
        if not args.keep:
            shutil.rmtree(root, ignore_errors=True)
        else:
            print(f"workspaces kept under {root}")


if __name__ == "__main__":
    main()
