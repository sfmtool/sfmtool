# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Per-reconstruction surfel engine for the ``compare --strips`` montage.

``_SolveStrips`` wraps one reconstruction and renders any of its points as an
oriented-surfel patch strip (one tile per observing view), and exposes the
per-point quantities the montage ranks on (triangulation angle, image radius,
keypoint feature size, world footprint). The ranking, row selection, and montage
assembly live in ``_compare_strips``.
"""

from __future__ import annotations

import numpy as np

from ._patch_ncc import gauss_window, render_track_strip
from ._sfmtool import (
    OrientedPatch,
    PatchCloud,
    RigidTransform,
    SfmrReconstruction,
)
from ._sfmtool.flow import WarpMap
from ._workspace_image import read_workspace_image

# One rendered strip: (strip image, mean pairwise NCC, number of views shown).
Strip = tuple[np.ndarray, float, int]


class _SolveStrips:
    """Renders one reconstruction's points as oriented-surfel patch strips."""

    def __init__(
        self,
        recon: SfmrReconstruction,
        workspace,
        *,
        patch: int,
        extent_factor: float,
        k_neighbors: int = 12,
    ) -> None:
        self.recon = recon
        self.workspace = workspace
        self.patch = patch
        self.names = list(recon.image_names)
        self.positions = np.asarray(recon.positions, np.float64)
        cameras = list(recon.cameras)
        cam_idx = np.asarray(recon.camera_indexes)
        quats = np.asarray(recon.quaternions_wxyz, np.float64)
        trans = np.asarray(recon.translations, np.float64)
        self.cam_of = [cameras[int(cam_idx[i])] for i in range(len(self.names))]
        self.pose_of = [
            RigidTransform.from_wxyz_translation(quats[i].tolist(), trans[i].tolist())
            for i in range(len(self.names))
        ]
        self.rot_of = [
            np.asarray(self.pose_of[i].to_rotation_matrix(), np.float64)
            for i in range(len(self.names))
        ]
        # World-space camera centers (center = -R^T t for x_cam = R x_world + t).
        self.centers = [-self.rot_of[i].T @ trans[i] for i in range(len(self.names))]
        self._images: dict[int, np.ndarray] = {}
        self._feat_sizes: dict[int, np.ndarray] = {}

        # Finite points only for this comparison montage (opt out of the default,
        # which includes points at infinity).
        self.cloud = PatchCloud.from_reconstruction(
            recon,
            normal="stored",
            k_neighbors=k_neighbors,
            extent_value=extent_factor,
            exclude_points_at_infinity=True,
        )
        self._cloud_index = {int(p): i for i, p in enumerate(self.cloud.point_ids)}

        self.obs: dict[int, list[int]] = {}
        # Per-observation (image_index, feature_index) for feature-size lookups.
        self.feat_obs: dict[int, list[tuple[int, int]]] = {}
        for pid, im, feat in zip(
            np.asarray(recon.track_point_ids).tolist(),
            np.asarray(recon.track_image_indexes).tolist(),
            np.asarray(recon.track_feature_indexes).tolist(),
        ):
            lst = self.obs.setdefault(int(pid), [])
            if im not in lst:
                lst.append(int(im))
            self.feat_obs.setdefault(int(pid), []).append((int(im), int(feat)))

    def image(self, i: int) -> np.ndarray:
        if i not in self._images:
            try:
                self._images[i] = read_workspace_image(self.workspace, self.names[i])
            except FileNotFoundError as e:
                raise FileNotFoundError(
                    f"{e}. --strips renders patches from each reconstruction's "
                    "workspace images; ensure that workspace is present alongside "
                    "the .sfmr."
                ) from e
        return self._images[i]

    def refine(self, point_ids, *, angular_range_deg: float, init_steps: int) -> float:
        """Refine the listed points' normals in place; return mean ΔΦ over them."""
        res = self.cloud.refine_normals(
            self.recon,
            [self.image(i) for i in range(len(self.names))],
            resolution=self.patch,
            angular_range_deg=angular_range_deg,
            init_steps=init_steps,
            point_ids=[int(p) for p in point_ids],
        )
        phi = np.asarray(res["photoconsistency"])
        init = np.asarray(res["init_photoconsistency"])
        deltas = [
            phi[self._cloud_index[int(p)]] - init[self._cloud_index[int(p)]]
            for p in point_ids
            if int(p) in self._cloud_index
        ]
        return float(np.mean(deltas)) if deltas else float("nan")

    def _normal(self, pid: int) -> np.ndarray:
        i = self._cloud_index.get(int(pid))
        if i is None:
            return np.array([0.0, 0.0, 1.0])
        n = np.asarray(self.cloud[i].normal, np.float64)
        return n if np.linalg.norm(n) > 0.5 else np.array([0.0, 0.0, 1.0])

    def _half(self, pid: int) -> float:
        i = self._cloud_index.get(int(pid))
        return float(self.cloud[i].half_extent[0]) if i is not None else 1e-2

    def tri_angle(self, pid: int) -> float:
        """Max angle (degrees) between viewing rays to this point — its
        triangulation angle. Small angles mean the point's depth is weakly
        constrained (little parallax)."""
        center = self.positions[int(pid)]
        rays = []
        for i in set(self.obs.get(int(pid), [])):
            v = center - self.centers[i]
            n = np.linalg.norm(v)
            if n > 1e-9:
                rays.append(v / n)
        if len(rays) < 2:
            return 0.0
        rays = np.array(rays)
        cosines = np.clip(rays @ rays.T, -1.0, 1.0)
        return float(np.degrees(np.arccos(cosines.min())))

    def image_radius(self, pid: int) -> float:
        """Median image-plane radius of the point's projections, normalized to
        the image half-diagonal (0 = principal point, ~1 = corner). Large values
        are peripheral points, where lens distortion matters most."""
        center = self.positions[int(pid)]
        radii = []
        for i in set(self.obs.get(int(pid), [])):
            pc = self.rot_of[i] @ (center - self.centers[i])
            if pc[2] <= 1e-9:
                continue
            cam = self.cam_of[i]
            u, v = cam.project(pc[0] / pc[2], pc[1] / pc[2])
            cx, cy = cam.principal_point
            half_diag = 0.5 * float(np.hypot(cam.width, cam.height))
            radii.append(float(np.hypot(u - cx, v - cy)) / half_diag)
        return float(np.median(radii)) if radii else 0.0

    def _image_feature_sizes(self, i: int) -> np.ndarray:
        """Per-keypoint image-plane feature size (pixels) for image ``i``,
        loaded from its .sift file and cached."""
        if i not in self._feat_sizes:
            from sfmtool.sift.file import (
                SiftReader,
                feature_size,
                get_sift_path_from_recon,
            )

            path = get_sift_path_from_recon(self.recon, self.names[i])
            try:
                shapes = np.asarray(SiftReader(path).read_affine_shapes())
            except (FileNotFoundError, OSError) as e:
                raise FileNotFoundError(
                    f"Cannot read SIFT features for '{self.names[i]}' at {path}. "
                    "The feature-size / world-size strip rankings need each "
                    "reconstruction's .sift files in its workspace."
                ) from e
            self._feat_sizes[i] = feature_size(shapes)
        return self._feat_sizes[i]

    def feature_size_px(self, pid: int) -> float:
        """Median image-plane feature size (keypoint diameter proxy, in pixels)
        over the point's observations. Large features are coarse/blurry blobs;
        small features are fine, sharply localized corners."""
        sizes = []
        for im, feat in self.feat_obs.get(int(pid), []):
            fs = self._image_feature_sizes(im)
            if 0 <= feat < len(fs):
                sizes.append(float(fs[feat]))
        return float(np.median(sizes)) if sizes else 0.0

    def world_feature_size(self, pid: int) -> float:
        """Median metric footprint of the point's features: image feature size
        scaled by depth / focal length, i.e. the world-space extent each
        keypoint covers on the surface (gauge-dependent, but consistent within
        this reconstruction)."""
        center = self.positions[int(pid)]
        vals = []
        for im, feat in self.feat_obs.get(int(pid), []):
            fs = self._image_feature_sizes(im)
            if not (0 <= feat < len(fs)):
                continue
            pc = self.rot_of[im] @ (center - self.centers[im])
            focal = float(np.mean(self.cam_of[im].focal_lengths))
            if pc[2] <= 1e-9 or focal <= 0:
                continue
            vals.append(float(fs[feat]) * float(pc[2]) / focal)
        return float(np.median(vals)) if vals else 0.0

    def strip(
        self,
        pid: int,
        *,
        tile: int,
        max_views: int | None = None,
        context: int | None = None,
    ) -> Strip | None:
        """Render point ``pid`` as a patch strip; ``(strip, mean_ncc, n)`` or None.

        With ``context`` (in pixels, ``> patch``) each tile renders a wider
        ``context``x``context`` window around the point at the same sampling
        density, with a 1px box marking the validated extent; NCC is still scored
        on that inner region.
        """
        obs_imgs = sorted(self.obs.get(int(pid), []))
        if not obs_imgs:
            return None
        if max_views and len(obs_imgs) > max_views:
            # Evenly-spaced representative subset of the sorted views so the
            # columns stay compact (already deduplicated when self.obs is built).
            sel = np.linspace(0, len(obs_imgs) - 1, max_views).round().astype(int)
            obs_imgs = [obs_imgs[i] for i in dict.fromkeys(sel.tolist())]
        center = self.positions[int(pid)]
        up = self.rot_of[obs_imgs[0]].T @ np.array([0.0, -1.0, 0.0])

        if context and context > self.patch:
            render_res = context
            ext = self._half(pid) * (context / self.patch)
            inner = ((context - self.patch) // 2, self.patch)
        else:
            render_res = self.patch
            ext = self._half(pid)
            inner = None
        patch = OrientedPatch.from_center_normal(
            center.tolist(), self._normal(pid).tolist(), up.tolist(), [ext, ext]
        )

        def patch_of(i: int) -> np.ndarray:
            wm = WarpMap.from_patch(patch, self.cam_of[i], self.pose_of[i], render_res)
            return np.asarray(wm.remap_bilinear(self.image(i)), np.float32)

        w = gauss_window(self.patch)
        return render_track_strip(obs_imgs, patch_of, w, tile=tile, inner=inner)
