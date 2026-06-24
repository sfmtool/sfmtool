#!/usr/bin/env python3
# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0
"""Cross-validate the sfmtool Rust SIFT against a reference detector (COLMAP or OpenCV).

The sfmtool detector is the Rust `sfmtool._sfmtool.extract_sift` binding; the
reference is one of the project's existing extraction backends
(`sfmtool.sift.extract_colmap` / `extract_opencv`). Both report keypoints in the
same conventions — COLMAP pixel-center coordinates and a 2x2 affine-shape matrix
— so positions and scales are directly comparable.

This measures *agreement with a reference*, not ground-truth correctness. The two
detectors differ in defaults (e.g. contrast threshold) and in grayscale
conversion (the Rust default matches COLMAP's BT.709; OpenCV uses BT.601), so do
not expect identical output — the point is to quantify how close we are and to
surface gross discrepancies (e.g. an incompatible descriptor layout).

Run inside the project environment, e.g.:

    pixi run -e test python scripts/sift_crossval.py
    pixi run -e test python scripts/sift_crossval.py --reference opencv img1.jpg img2.jpg
"""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent


def _die(msg: str) -> None:
    print(f"error: {msg}", file=sys.stderr)
    sys.exit(1)


def feature_size(affine_shapes: np.ndarray) -> np.ndarray:
    """Average of the two affine-shape column norms (matches sfmtool.sift.file)."""
    if len(affine_shapes) == 0:
        return np.zeros((0,), dtype=np.float64)
    col0 = np.linalg.norm(affine_shapes[:, :, 0], axis=1)
    col1 = np.linalg.norm(affine_shapes[:, :, 1], axis=1)
    return 0.5 * (col0 + col1)


def normalize_rows(desc: np.ndarray) -> np.ndarray:
    d = desc.astype(np.float32)
    n = np.linalg.norm(d, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return d / n


def mutual_matches(query: np.ndarray, train: np.ndarray):
    """Cross-checked nearest-neighbour matches (L2). Returns list of (qi, ti, dist)."""
    import cv2

    if len(query) == 0 or len(train) == 0:
        return []
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(np.ascontiguousarray(query), np.ascontiguousarray(train))
    return [(m.queryIdx, m.trainIdx, m.distance) for m in matches]


def extract_rust(image_path: Path):
    """Run the sfmtool Rust SIFT on an RGB image; returns (positions, affine, desc)."""
    import cv2

    from sfmtool._sfmtool.sift import extract_sift as rust_extract_sift

    bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if bgr is None:
        _die(f"could not read image {image_path}")
    rgb = np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    positions, affine_shapes, descriptors = rust_extract_sift(rgb)
    return positions, affine_shapes, descriptors


def extract_reference(image_path: Path, reference: str):
    """Run the reference backend; returns (positions, affine, desc)."""
    if reference == "opencv":
        from sfmtool.sift.extract_opencv import (
            extract_sift_with_opencv,
            get_default_opencv_feature_options,
        )

        opts = get_default_opencv_feature_options()
        results = extract_sift_with_opencv([image_path], opts)
    elif reference == "colmap":
        from sfmtool.sift.extract_colmap import (
            extract_sift_with_colmap,
            get_colmap_feature_options,
        )

        # CPU SIFT so it runs without a GPU; similarity (non-affine) shapes to
        # match the Rust detector's representation.
        opts = get_colmap_feature_options(use_gpu=False, estimate_affine_shape=False)
        results = extract_sift_with_colmap([image_path], opts)
    else:
        _die(f"unknown reference {reference!r}")

    # Each result is (feature_tool_metadata, metadata, positions, affine_shapes,
    # descriptors, thumbnail).
    _, _, positions, affine_shapes, descriptors, _ = results[0]
    return positions, affine_shapes, descriptors


def compare(pos_a, aff_a, desc_a, pos_b, aff_b, desc_b, loc_tol: float):
    """Compare detector A (ours) vs detector B (reference)."""
    na, nb = len(pos_a), len(pos_b)
    out = {"n_ours": na, "n_ref": nb}
    if na == 0 or nb == 0:
        return out

    size_a = feature_size(aff_a)
    size_b = feature_size(aff_b)

    # --- Location repeatability: cross-checked spatial NN within loc_tol ---
    pa = np.ascontiguousarray(pos_a, dtype=np.float32)
    pb = np.ascontiguousarray(pos_b, dtype=np.float32)
    spatial = [(qi, ti, d) for (qi, ti, d) in mutual_matches(pa, pb) if d <= loc_tol]
    matched = len(spatial)
    out["loc_matched"] = matched
    out["recall"] = matched / nb  # fraction of reference keypoints we recover
    out["precision"] = matched / na  # fraction of ours that hit a reference kp
    out["median_loc_err"] = float(np.median([d for _, _, d in spatial])) if spatial else float("nan")

    # Scale-convention factor on matched pairs (ours / reference).
    if spatial:
        ratios = np.array([size_a[qi] / size_b[ti] for qi, ti, _ in spatial if size_b[ti] > 0])
        out["scale_ratio_median"] = float(np.median(ratios)) if len(ratios) else float("nan")
    else:
        out["scale_ratio_median"] = float("nan")

    # --- Descriptor agreement on spatially-matched pairs (cosine) ---
    nd_a = normalize_rows(desc_a)
    nd_b = normalize_rows(desc_b)
    if spatial:
        cos = np.array([float(nd_a[qi] @ nd_b[ti]) for qi, ti, _ in spatial])
        out["desc_cos_mean"] = float(cos.mean())
        out["desc_cos_median"] = float(np.median(cos))
        # Random-pair baseline for context.
        rng = np.random.default_rng(0)
        ri = rng.integers(0, na, size=min(2000, na))
        rj = rng.integers(0, nb, size=min(2000, nb))
        k = min(len(ri), len(rj))
        out["desc_cos_random"] = float(np.mean(np.sum(nd_a[ri[:k]] * nd_b[rj[:k]], axis=1)))

    # --- Descriptor matching: mutual NN in descriptor space, spatial consistency ---
    dmatches = mutual_matches(nd_a, nd_b)
    out["desc_matches"] = len(dmatches)
    if dmatches:
        consistent = sum(
            1
            for qi, ti, _ in dmatches
            if np.hypot(*(pos_a[qi] - pos_b[ti])) <= loc_tol
        )
        out["desc_match_spatial_consistency"] = consistent / len(dmatches)
    return out


def fmt(v):
    if isinstance(v, float):
        return "nan" if v != v else f"{v:.3f}"
    return str(v)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("images", nargs="*", help="image files (default: a few seoul_bull samples)")
    ap.add_argument("--reference", choices=["colmap", "opencv"], default="colmap")
    ap.add_argument("--loc-tol", type=float, default=2.0, help="location match tolerance, px")
    args = ap.parse_args()

    images = args.images
    if not images:
        images = sorted(glob.glob(str(REPO_ROOT / "test-data/images/seoul_bull_sculpture/*.jpg")))[:3]
    if not images:
        _die("no images given and no default samples found")

    # Allow `import sfmtool...` from a source checkout if not pip-installed.
    sys.path.insert(0, str(REPO_ROOT / "src"))

    try:
        import cv2  # noqa: F401
    except ImportError:
        _die("opencv (cv2) is required; run via `pixi run -e test`")

    print(f"reference: {args.reference}   loc-tol: {args.loc_tol}px")
    cols = [
        "n_ours", "n_ref", "recall", "precision", "median_loc_err",
        "scale_ratio_median", "desc_cos_median", "desc_cos_random",
        "desc_match_spatial_consistency",
    ]
    print("image".ljust(28) + "  " + "  ".join(c[:14].rjust(14) for c in cols))

    agg: dict[str, list[float]] = {c: [] for c in cols}
    for img in images:
        path = Path(img)
        pa, aa, da = extract_rust(path)
        pb, ab, db = extract_reference(path, args.reference)
        res = compare(pa, aa, da, pb, ab, db, args.loc_tol)
        row = "  ".join(fmt(res.get(c, float("nan")))[:14].rjust(14) for c in cols)
        print(path.name[:28].ljust(28) + "  " + row)
        for c in cols:
            v = res.get(c)
            if isinstance(v, (int, float)) and v == v:
                agg[c].append(float(v))

    print("-" * (28 + 2 + len(cols) * 16))
    means = "  ".join(
        (f"{np.mean(agg[c]):.3f}" if agg[c] else "nan")[:14].rjust(14) for c in cols
    )
    print("mean".ljust(28) + "  " + means)


if __name__ == "__main__":
    main()
