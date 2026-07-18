#!/usr/bin/env python
"""Benchmark photometric patch-normal refinement: wall time vs quality.

Loads a reconstruction, samples a reproducible subset of points, runs
``PatchCloud.refine_normals`` under a series of parameter configurations
(a base config plus one-at-a-time ``--vary`` sweeps), and writes one JSON
line per run with wall time and quality metrics:

- ``mean_phi`` / ``median_phi`` / ``p10_phi``: achieved consensus
  photoconsistency over the scored subset. Directly comparable across runs
  only while the scoring config (resolution / window / sampler) is fixed.
- ``mean_dphi``: mean improvement over the init normal under the run's own
  frozen support.
- ``med_angle_ref_deg`` / ``frac_ref_2deg``: angular agreement with a
  high-fidelity reference run (``--reference`` overrides on top of the
  base config) — the cross-config accuracy proxy.

The refinement itself is deterministic, so repeats only de-noise the wall
time (the minimum is reported as ``wall_s``).

Example:
  pixi run -e test python scripts/bench_normal_refine.py \\
      /tmp/pxv/seoul/sfmr/seoul.sfmr --n-points 300 --seed 0 \\
      --vary resolution=8,12,16,24,32 --vary sampler=bilinear,anisotropic \\
      --out /tmp/seoul_bench.jsonl

Set ``--profile`` to print the Rust phase-timing summary (SFMTOOL_PROFILE=1)
for every run.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

INT_KEYS = {"resolution", "init_steps", "refine_levels", "robust_iters", "min_views"}
FLOAT_KEYS = {"angular_range_deg", "window_sigma", "min_valid_fraction"}
STR_KEYS = {"objective", "sampler", "window"}

DEFAULT_BASE = {
    "resolution": 16,
    "angular_range_deg": 25.0,
    "init_steps": 7,
    "refine_levels": 3,
    "objective": "robust",
    "robust_iters": 3,
    "sampler": "anisotropic",
    "window": "gaussian_disk",
    # The bench reports per-patch confidence metrics, so request them (confidence
    # is off by default in production for the extra runtime it costs).
    "compute_confidence": True,
}

DEFAULT_REFERENCE = {
    "resolution": 32,
    "init_steps": 9,
    "refine_levels": 4,
    "sampler": "anisotropic",
    "objective": "robust",
    "robust_iters": 3,
}


def parse_kv(item: str) -> tuple[str, object]:
    key, _, value = item.partition("=")
    key = key.strip()
    value = value.strip()
    if key in INT_KEYS:
        return key, int(value)
    if key in FLOAT_KEYS:
        return key, float(value)
    if key in STR_KEYS:
        return key, value
    raise SystemExit(f"unknown refine_normals parameter: {key!r}")


def parse_config(spec: str) -> dict:
    return dict(parse_kv(item) for item in spec.split(",") if item.strip())


def parse_vary(spec: str) -> tuple[str, list[object]]:
    key, _, values = spec.partition("=")
    key = key.strip()
    out = [parse_kv(f"{key}={v}")[1] for v in values.split(",") if v.strip()]
    if not out:
        raise SystemExit(f"--vary {spec!r} lists no values")
    return key, out


def angle_deg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Per-row angle between unit-ish vectors, in degrees."""
    an = a / np.linalg.norm(a, axis=1, keepdims=True)
    bn = b / np.linalg.norm(b, axis=1, keepdims=True)
    dot = np.clip((an * bn).sum(axis=1), -1.0, 1.0)
    return np.degrees(np.arccos(dot))


class Workload:
    """One reconstruction + images + a fixed sampled point subset."""

    def __init__(self, sfmr_path: str, n_points: int, seed: int):
        import cv2

        from sfmtool._sfmtool import patches, reconstruction

        self._patches = patches
        self.recon = reconstruction.SfmrReconstruction.load(sfmr_path)
        ws = self.recon.workspace_dir
        t0 = time.perf_counter()
        self.images = [
            np.ascontiguousarray(
                cv2.imread(os.path.join(ws, name), cv2.IMREAD_COLOR)
            )
            for name in self.recon.image_names
        ]
        self.load_s = time.perf_counter() - t0

        cloud = self.fresh_cloud()
        all_ids = np.asarray(cloud.point_indexes, dtype=np.uint32)
        rng = np.random.default_rng(seed)
        if n_points and n_points < len(all_ids):
            self.ids = np.sort(rng.choice(all_ids, size=n_points, replace=False))
        else:
            self.ids = np.sort(all_ids)
        # Cloud indices of the sampled ids (point_ids are unique per patch).
        index_of = {pid: i for i, pid in enumerate(all_ids.tolist())}
        self.rows = np.asarray([index_of[pid] for pid in self.ids.tolist()])
        self.init_normals = np.asarray(
            [cloud[int(i)].normal for i in self.rows], dtype=np.float64
        )

    def fresh_cloud(self):
        # Benchmark normal refinement, a finite-point operation; opt out of the
        # default that includes (refinement-skipped) infinity points so timings
        # and counts stay comparable to historical runs.
        return self._patches.PatchCloud.from_reconstruction(
            self.recon,
            normal="mean_viewing",
            extent_value=5.0,
            exclude_points_at_infinity=True,
        )

    def overhead_seconds(self) -> float:
        """Fixed per-call cost (image conversion + pyramid build), via an
        empty-subset call that refines nothing. Warmed up once: the very
        first call pays cold caches / page faults and is not representative."""
        cloud = self.fresh_cloud()
        cloud.refine_normals(self.recon, self.images, point_indexes=[])
        t0 = time.perf_counter()
        cloud.refine_normals(self.recon, self.images, point_indexes=[])
        return time.perf_counter() - t0

    def run(self, params: dict, repeats: int) -> dict:
        walls = []
        res = None
        for _ in range(max(1, repeats)):
            cloud = self.fresh_cloud()
            t0 = time.perf_counter()
            res = cloud.refine_normals(
                self.recon,
                self.images,
                point_indexes=self.ids.tolist(),
                **params,
            )
            walls.append(time.perf_counter() - t0)
        out = {k: np.asarray(v)[self.rows] for k, v in res.items()}
        out["walls"] = walls
        return out


def summarize(run: dict, init_normals: np.ndarray, ref_normals: np.ndarray | None,
              overhead_s: float) -> dict:
    phi = run["photoconsistency"]
    scored = np.isfinite(phi)
    n = int(scored.sum())
    both_phi = scored & np.isfinite(run["init_photoconsistency"])
    dphi = phi[both_phi] - run["init_photoconsistency"][both_phi]
    ang_init = angle_deg(run["normal"][scored], init_normals[scored])
    out = {
        "wall_s": round(min(run["walls"]), 4),
        "walls": [round(w, 4) for w in run["walls"]],
        "overhead_s": round(overhead_s, 4),
        "net_s": round(min(run["walls"]) - overhead_s, 4),
        "n_scored": n,
        "ms_per_patch": round(1e3 * (min(run["walls"]) - overhead_s) / max(n, 1), 3),
        "mean_phi": round(float(phi[scored].mean()), 5) if n else None,
        "median_phi": round(float(np.median(phi[scored])), 5) if n else None,
        "p10_phi": round(float(np.percentile(phi[scored], 10)), 5) if n else None,
        "mean_dphi": round(float(dphi.mean()), 5) if dphi.size else None,
        "mean_conf": round(float(run["confidence"][scored].mean()), 4) if n else None,
        "mean_views": round(float(run["valid_view_count"][scored].mean()), 2) if n else None,
        "med_angle_init_deg": round(float(np.median(ang_init)), 3) if n else None,
    }
    if ref_normals is not None and n:
        both = scored & np.isfinite(ref_normals).all(axis=1)
        ang = angle_deg(run["normal"][both], ref_normals[both])
        out["med_angle_ref_deg"] = round(float(np.median(ang)), 3)
        out["p90_angle_ref_deg"] = round(float(np.percentile(ang, 90)), 3)
        out["frac_ref_2deg"] = round(float((ang <= 2.0).mean()), 3)
        out["frac_ref_5deg"] = round(float((ang <= 5.0).mean()), 3)
        # Confidence-gated agreement: weakly-constrained optima scatter under
        # any config, so the gated median isolates the well-posed patches.
        hi = run["confidence"][both] >= 0.25
        if hi.sum() >= 10:
            out["med_angle_ref_hiconf_deg"] = round(float(np.median(ang[hi])), 3)
            out["n_hiconf"] = int(hi.sum())
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sfmr", help="path to the .sfmr reconstruction")
    ap.add_argument("--n-points", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--base", type=parse_config, default={},
                    help="comma-separated k=v overrides of the base config")
    ap.add_argument("--vary", action="append", type=parse_vary, default=[],
                    metavar="KEY=V1,V2,...",
                    help="sweep KEY over the listed values (base for the rest); repeatable")
    ap.add_argument("--reference", type=parse_config, default={},
                    help="k=v overrides (on top of base) for the high-fidelity reference run; "
                         "pass 'none' via --no-reference to skip")
    ap.add_argument("--no-reference", action="store_true")
    ap.add_argument("--profile", action="store_true",
                    help="set SFMTOOL_PROFILE=1 (Rust phase-timing summaries on stderr)")
    ap.add_argument("--out", default=None, help="append JSON lines here (default stdout only)")
    args = ap.parse_args()

    if args.profile:
        os.environ["SFMTOOL_PROFILE"] = "1"

    base = {**DEFAULT_BASE, **args.base}
    dataset = os.path.splitext(os.path.basename(args.sfmr))[0]

    wl = Workload(args.sfmr, args.n_points, args.seed)
    overhead = wl.overhead_seconds()
    print(
        f"# {dataset}: {len(wl.recon.image_names)} images (loaded in {wl.load_s:.1f}s), "
        f"{len(wl.ids)} sampled points, per-call overhead {overhead:.2f}s",
        file=sys.stderr,
    )

    ref_normals = None
    records = []
    if not args.no_reference:
        ref_params = {**base, **DEFAULT_REFERENCE, **args.reference}
        run = wl.run(ref_params, repeats=1)
        full = np.full((len(wl.ids), 3), np.nan)
        scored = np.isfinite(run["photoconsistency"])
        full[scored] = run["normal"][scored]
        ref_normals = full
        rec = {"dataset": dataset, "config": "reference", "params": ref_params,
               **summarize(run, wl.init_normals, None, overhead)}
        records.append(rec)
        print(json.dumps(rec), flush=True)

    configs: list[tuple[str, dict]] = [("base", dict(base))]
    for key, values in args.vary:
        for v in values:
            if v == base.get(key):
                continue  # the base run already covers it
            configs.append((f"{key}={v}", {**base, key: v}))

    for label, params in configs:
        run = wl.run(params, repeats=args.repeats)
        rec = {"dataset": dataset, "config": label, "params": params,
               **summarize(run, wl.init_normals, ref_normals, overhead)}
        records.append(rec)
        print(json.dumps(rec), flush=True)

    if args.out:
        with open(args.out, "a", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    main()
