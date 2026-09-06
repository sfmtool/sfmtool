#!/usr/bin/env python
# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Measure optical-flow endpoint error against exact analytic ground truth.

`benchmark_optical_flow.py` times the flow; this one scores it. Scenes are affine
warps of real test-data images, so the ground-truth flow is known in closed form
rather than approximated by inverting a sampled field:

    img_b(y) = img_a(A y)   =>   img_a(x) = img_b(A^-1 x)
    ground-truth flow F(x)  =   A^-1 x - x

in the sfmtool pixel-center convention (col + 0.5, row + 0.5). Pixels whose
correspondence leaves the frame are excluded, as is a border margin.

The two-layer scene adds a motion boundary — a rectangle translated over a static
background — which is the regime the DIS densification weight exists to handle,
since overlapping patches straddling the boundary disagree. Metrics there are
reported both over the whole frame and over the moving region alone, with the
occlusion band around the boundary excluded as ambiguous.

Usage:
    pixi run python scripts/benchmark_flow_accuracy.py [IMAGE ...]
        [--preset fast|default|high_quality] [--gpu] [--noise SIGMA]

Examples:
    pixi run python scripts/benchmark_flow_accuracy.py
    pixi run python scripts/benchmark_flow_accuracy.py --preset high_quality
    pixi run python scripts/benchmark_flow_accuracy.py --noise 5 --gpu
"""

import argparse
import os

import cv2
import numpy as np

from sfmtool._sfmtool.flow import compute_optical_flow

# Border excluded from every metric: patches that run off the image have no
# meaningful flow, and including them measures the boundary policy, not the flow.
MARGIN = 12

DEFAULT_IMAGES = [
    "test-data/images/dino_dog_toy/dino_dog_toy_01.jpg",
    "test-data/images/seattle_backyard/seattle_backyard_01.jpg",
    "test-data/images/seoul_bull_sculpture/seoul_bull_sculpture_01.jpg",
]


def load_gray(path, max_width=640):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise SystemExit(f"could not read {path}")
    if max_width and img.shape[1] > max_width:
        scale = max_width / img.shape[1]
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(img)


def affine_scene(img, A):
    """Warp img by A, returning (img_b, gt_u, gt_v, valid) with exact ground truth."""
    h, w = img.shape
    cols, rows = np.meshgrid(
        np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64)
    )
    px = cols + 0.5
    py = rows + 0.5

    # img_b(y) = img_a(A y): for each destination pixel y, sample the source at A y.
    sx = A[0, 0] * px + A[0, 1] * py + A[0, 2]
    sy = A[1, 0] * px + A[1, 1] * py + A[1, 2]
    img_b = cv2.remap(
        img,
        (sx - 0.5).astype(np.float32),
        (sy - 0.5).astype(np.float32),
        cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REFLECT101,
    )

    inv = np.linalg.inv(np.vstack([A, [0.0, 0.0, 1.0]]))[:2]
    qx = inv[0, 0] * px + inv[0, 1] * py + inv[0, 2]
    qy = inv[1, 0] * px + inv[1, 1] * py + inv[1, 2]
    gt_u = (qx - px).astype(np.float32)
    gt_v = (qy - py).astype(np.float32)

    valid = (qx >= MARGIN) & (qx < w - MARGIN) & (qy >= MARGIN) & (qy < h - MARGIN)
    valid[:MARGIN, :] = False
    valid[-MARGIN:, :] = False
    valid[:, :MARGIN] = False
    valid[:, -MARGIN:] = False
    return np.ascontiguousarray(img_b), gt_u, gt_v, valid


def rot_zoom(w, h, degrees, scale):
    """Affine matrix rotating by `degrees` and scaling by `scale` about the center."""
    c = np.cos(np.deg2rad(degrees)) * scale
    s = np.sin(np.deg2rad(degrees)) * scale
    cx, cy = w / 2.0, h / 2.0
    A = np.array([[c, -s, 0.0], [s, c, 0.0]], dtype=np.float64)
    A[0, 2] = cx - (A[0, 0] * cx + A[0, 1] * cy)
    A[1, 2] = cy - (A[1, 0] * cx + A[1, 1] * cy)
    return A


def two_layer_scene(img, fg_shift):
    """Static background plus a translated rectangle: one hard motion boundary.

    Returns (img_b, gt_u, gt_v, valid, moving) where `moving` is the subset of
    `valid` inside the translated rectangle.
    """
    h, w = img.shape
    identity = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    shifted = np.array([[1.0, 0.0, -fg_shift[0]], [0.0, 1.0, -fg_shift[1]]])
    bg_b, bg_u, bg_v, bg_valid = affine_scene(img, identity)
    fg_b, fg_u, fg_v, fg_valid = affine_scene(img, shifted)

    y0, y1 = h // 4, 3 * h // 4
    x0, x1 = w // 4, 3 * w // 4
    mask = np.zeros((h, w), dtype=bool)
    mask[y0:y1, x0:x1] = True

    img_b = np.where(mask, fg_b, bg_b).astype(np.uint8)
    gt_u = np.where(mask, fg_u, bg_u)
    gt_v = np.where(mask, fg_v, bg_v)
    valid = np.where(mask, fg_valid, bg_valid)

    # Occlusion/disocclusion band along the boundary has no well-defined
    # correspondence; scoring it would measure the fill policy, not the flow.
    band = int(np.ceil(max(abs(fg_shift[0]), abs(fg_shift[1])) + 4))
    edge = np.zeros((h, w), dtype=bool)
    edge[y0 - band : y0 + band, x0 - band : x1 + band] = True
    edge[y1 - band : y1 + band, x0 - band : x1 + band] = True
    edge[y0 - band : y1 + band, x0 - band : x0 + band] = True
    edge[y0 - band : y1 + band, x1 - band : x1 + band] = True

    keep = valid & ~edge
    return np.ascontiguousarray(img_b), gt_u, gt_v, keep, mask & keep


def build_scenes(img):
    """The affine scenes, as (label, (img_b, gt_u, gt_v, valid)) pairs."""
    h, w = img.shape
    translate = np.array([[1.0, 0.0, -3.7], [0.0, 1.0, 2.3]])
    shear = np.array([[1.0, 0.02, 0.0], [0.01, 1.0, 0.0]])
    return [
        ("translate 3.7,-2.3", affine_scene(img, translate)),
        ("rotate 2deg", affine_scene(img, rot_zoom(w, h, 2.0, 1.0))),
        ("zoom 1.03", affine_scene(img, rot_zoom(w, h, 0.0, 1.03))),
        ("rot 1.5deg + zoom 1.02", affine_scene(img, rot_zoom(w, h, 1.5, 1.02))),
        ("shear 0.02", affine_scene(img, shear)),
    ]


def epe_stats(flow_u, flow_v, gt_u, gt_v, valid):
    """Mean EPE, median EPE, and the percentage of pixels past 1px and 3px."""
    err = np.sqrt((flow_u - gt_u) ** 2 + (flow_v - gt_v) ** 2)[valid]
    if err.size == 0:
        return (float("nan"),) * 4
    return (
        float(err.mean()),
        float(np.median(err)),
        float((err > 1.0).mean() * 100.0),
        float((err > 3.0).mean() * 100.0),
    )


def main():
    parser = argparse.ArgumentParser(
        description="Score optical flow against exact analytic ground truth"
    )
    parser.add_argument("images", nargs="*", default=None, help="images to warp")
    parser.add_argument(
        "--preset", default="default", choices=["fast", "default", "high_quality"]
    )
    parser.add_argument("--gpu", action="store_true", help="use the GPU flow path")
    parser.add_argument(
        "--noise",
        type=float,
        default=0.0,
        help="Gaussian sensor noise sigma in 8-bit levels, applied to both frames",
    )
    parser.add_argument("--max-width", type=int, default=640)
    args = parser.parse_args()

    rng = np.random.default_rng(7)

    def perturb(img):
        if args.noise <= 0.0:
            return img
        noisy = img.astype(np.float64) + rng.normal(0.0, args.noise, img.shape)
        return np.ascontiguousarray(np.clip(noisy, 0, 255).astype(np.uint8))

    def run(img_a, img_b):
        return compute_optical_flow(
            perturb(img_a), perturb(img_b), preset=args.preset, use_gpu=args.gpu
        )

    rows = []
    for path in args.images or DEFAULT_IMAGES:
        img = load_gray(path, args.max_width)
        name = os.path.basename(path).rsplit("_", 1)[0]

        for label, (img_b, gt_u, gt_v, valid) in build_scenes(img):
            flow_u, flow_v = run(img, img_b)
            rows.append((name, label, *epe_stats(flow_u, flow_v, gt_u, gt_v, valid)))

        img_b, gt_u, gt_v, valid, moving = two_layer_scene(img, (6.0, 0.0))
        flow_u, flow_v = run(img, img_b)
        rows.append(
            (name, "two-layer (all)", *epe_stats(flow_u, flow_v, gt_u, gt_v, valid))
        )
        rows.append(
            (name, "two-layer (moving)", *epe_stats(flow_u, flow_v, gt_u, gt_v, moving))
        )

    print(f"# preset={args.preset} gpu={int(args.gpu)} noise={args.noise}")
    header = (
        f"{'image':<22}{'scene':<24}{'meanEPE':>9}{'medEPE':>9}{'>1px%':>8}{'>3px%':>8}"
    )
    print(header)
    for name, label, mean, med, p1, p3 in rows:
        print(f"{name:<22}{label:<24}{mean:>9.4f}{med:>9.4f}{p1:>8.2f}{p3:>8.2f}")
    agg = np.array([r[2:] for r in rows])
    print(
        f"{'ALL':<22}{'mean over scenes':<24}"
        f"{agg[:, 0].mean():>9.4f}{agg[:, 1].mean():>9.4f}"
        f"{agg[:, 2].mean():>8.2f}{agg[:, 3].mean():>8.2f}"
    )


if __name__ == "__main__":
    main()
