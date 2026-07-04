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

import cv2
import numpy as np

from ._patch_ncc import gauss_window, render_track_strip
from ._sfmtool import (
    OrientedPatch,
    PatchCloud,
    SfmrReconstruction,
)
from ._sfmtool.geometry import RigidTransform
from ._sfmtool.flow import WarpMap
from ._workspace_image import read_workspace_image

# One rendered strip: (strip image, mean pairwise NCC, number of views shown).
Strip = tuple[np.ndarray, float, int]


def _corner_label(tile: np.ndarray, text: str) -> None:
    """Draw a small yellow label in the bottom-left of ``tile`` (in place)."""
    cv2.putText(
        tile,
        text,
        (3, tile.shape[0] - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.32,
        (0, 255, 255),
        1,
        cv2.LINE_AA,
    )


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
        exclude_points_at_infinity: bool = True,
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
        # Stored per-point RGBA patch bitmaps (embedded_patches with bitmaps),
        # indexed by 3D-point id; None when the recon carries no bitmaps. Used as
        # the reference patch when present (see `reference_patch`).
        bitmaps = recon.patch_bitmaps
        self._bitmaps = np.asarray(bitmaps) if bitmaps is not None else None

        # An embedded_patches recon already carries a per-point patch frame (with
        # its sizing baked in); read it back rather than re-deriving extents from
        # keypoint scales (which embedded keypoints don't carry). A sift_files
        # recon has no stored frame, so build one — sizing the extent from the
        # .sift feature sizes. `compare --strips` excludes points at infinity (it
        # cannot render a cross-recon surfel for them); `inspect --strips` keeps
        # them and renders each as a tangent-sphere infinity patch.
        stored = recon.patches
        if stored is not None:
            self.cloud = stored
        else:
            self.cloud = PatchCloud.from_reconstruction(
                recon,
                normal="stored",
                k_neighbors=k_neighbors,
                extent_value=extent_factor,
                exclude_points_at_infinity=exclude_points_at_infinity,
            )
        self._cloud_index = {int(p): i for i, p in enumerate(self.cloud.point_indexes)}

        self.obs: dict[int, list[int]] = {}
        # Per-observation (image_index, feature_index) for feature-size lookups.
        # `track_feature_indexes` is sift_files-only (None on embedded_patches),
        # and only the feature-size/world-size rankings need it, so build feat_obs
        # only when it's present.
        self.feat_obs: dict[int, list[tuple[int, int]]] = {}
        track_feats = recon.track_feature_indexes
        feats = np.asarray(track_feats).tolist() if track_feats is not None else None
        # Per-observation 2D keypoint, keyed by (point, image), for per-observation
        # reprojection error. `keypoints_xy` is embedded_patches-only (None on
        # sift_files); when absent, reprojection error is unavailable.
        kxy = recon.keypoints_xy
        kxy = np.asarray(kxy, np.float64) if kxy is not None else None
        self.kpt_obs: dict[int, dict[int, tuple[float, float]]] = {}
        for k, (pid, im) in enumerate(
            zip(
                np.asarray(recon.track_point_indexes).tolist(),
                np.asarray(recon.track_image_indexes).tolist(),
            )
        ):
            lst = self.obs.setdefault(int(pid), [])
            if im not in lst:
                lst.append(int(im))
            if feats is not None:
                self.feat_obs.setdefault(int(pid), []).append((int(im), int(feats[k])))
            if kxy is not None:
                self.kpt_obs.setdefault(int(pid), {})[int(im)] = (
                    float(kxy[k, 0]),
                    float(kxy[k, 1]),
                )

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

    def prime_images(self, images: list[np.ndarray]) -> None:
        """Seed the image cache with already-loaded frames (parallel to
        ``recon.image_names``), so callers that loaded the images for refinement
        avoid reading them a second time."""
        self._images = {i: images[i] for i in range(len(images))}

    def refine(self, point_ids, *, angular_range_deg: float, init_steps: int) -> float:
        """Refine the listed points' normals in place; return mean ΔΦ over them."""
        res = self.cloud.refine_normals(
            self.recon,
            [self.image(i) for i in range(len(self.names))],
            resolution=self.patch,
            angular_range_deg=angular_range_deg,
            init_steps=init_steps,
            point_indexes=[int(p) for p in point_ids],
            # `compare --strips` is the strip-render reference: anchor every
            # view at the reprojected center so the comparison metric is the
            # same regardless of whether the recon happens to carry inline
            # keypoints. Without this lock the auto-default would silently
            # switch to stored keypoints on embedded_patches recons and
            # change strip output / ΔΦ values.
            use_stored_keypoints=False,
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

    def _w(self, pid: int) -> float:
        """Homogeneous weight of the point's patch: ``1.0`` finite, ``0.0`` at
        infinity (a direction rather than a position)."""
        i = self._cloud_index.get(int(pid))
        return float(self.cloud[i].w) if i is not None else 1.0

    def _oriented_patch(self, pid: int, *, up: np.ndarray, ext: float) -> OrientedPatch:
        """Build the surfel for ``pid``: a finite oriented patch, or a
        tangent-sphere patch when the point is at infinity (``w == 0``)."""
        center = self.positions[int(pid)]
        if self._w(pid) == 0.0:
            return OrientedPatch.from_infinity_direction(
                center.tolist(), up.tolist(), [ext, ext]
            )
        return OrientedPatch.from_center_normal(
            center.tolist(), self._normal(pid).tolist(), up.tolist(), [ext, ext]
        )

    def _render_view(self, patch: OrientedPatch, i: int, res: int) -> np.ndarray:
        """Render image ``i``'s ``res``x``res`` tile of ``patch`` as float32."""
        wm = WarpMap.from_patch(patch, self.cam_of[i], self.pose_of[i], res)
        return np.asarray(wm.remap_bilinear(self.image(i)), np.float32)

    def _stored_bitmap(self, pid: int) -> np.ndarray | None:
        """The raw stored patch bitmap for ``pid`` (RGBA, colour channels RGB),
        or ``None`` if the recon carries no (non-empty) bitmap for it."""
        if self._bitmaps is None or not (0 <= int(pid) < len(self._bitmaps)):
            return None
        bmp = self._bitmaps[int(pid)]
        return bmp if bmp.any() else None

    def _bitmap_ref_tile(self, bmp: np.ndarray, tile: int) -> np.ndarray:
        """Reference tile for a stored bitmap: the RGB patch rendered *without*
        alpha blending, and — when the bitmap carries an alpha channel — its alpha
        shown as a grayscale tile beside it.

        The stored colour channels are RGB; this tile feeds the BGR montage
        (written via cv2), so convert RGB→BGR here."""
        if bmp.ndim == 3:
            color = cv2.cvtColor(np.ascontiguousarray(bmp[..., :3]), cv2.COLOR_RGB2BGR)
        else:
            color = cv2.cvtColor(bmp, cv2.COLOR_GRAY2BGR)
        rgb_t = cv2.resize(color, (tile, tile), interpolation=cv2.INTER_NEAREST)
        _corner_label(rgb_t, "rgb")
        if not (bmp.ndim == 3 and bmp.shape[-1] == 4):
            return rgb_t
        alpha = cv2.cvtColor(np.ascontiguousarray(bmp[..., 3]), cv2.COLOR_GRAY2BGR)
        a_t = cv2.resize(alpha, (tile, tile), interpolation=cv2.INTER_NEAREST)
        _corner_label(a_t, "A")
        sep = np.full((tile, 2, 3), 60, np.uint8)
        return np.hstack([rgb_t, sep, a_t])

    def reference_patch(
        self, pid: int, *, tile: int, max_views: int | None = None
    ) -> np.ndarray | None:
        """Render the point's reference patch as a BGR tile (``tile`` tall).

        For a stored bitmap, the un-blended RGB patch plus its alpha as a
        grayscale tile beside it (so the tile is wider than it is tall).
        Otherwise the cross-view consensus: the mean of the point's per-view core
        patches rendered at the surfel orientation, a single ``tile``x``tile``
        square. ``None`` if the point has no observations and no stored bitmap.
        """
        pid = int(pid)
        bmp = self._stored_bitmap(pid)
        if bmp is not None:
            return self._bitmap_ref_tile(bmp, tile)
        obs_imgs = sorted(self.obs.get(pid, []))
        if not obs_imgs:
            return None
        if max_views and len(obs_imgs) > max_views:
            sel = np.linspace(0, len(obs_imgs) - 1, max_views).round().astype(int)
            obs_imgs = [obs_imgs[i] for i in dict.fromkeys(sel.tolist())]
        up = self.rot_of[obs_imgs[0]].T @ np.array([0.0, 1.0, 0.0])
        patch = self._oriented_patch(pid, up=up, ext=self._half(pid))
        cores = [self._render_view(patch, i, self.patch) for i in obs_imgs]
        mean = np.stack([np.asarray(c, np.float64) for c in cores]).mean(0)
        p8 = np.clip(mean, 0, 255).astype(np.uint8)
        # `cores` are RGB (rendered from RGB source images); convert to BGR for
        # the cv2-written montage.
        bgr = (
            cv2.cvtColor(p8, cv2.COLOR_RGB2BGR)
            if p8.ndim == 3
            else cv2.cvtColor(p8, cv2.COLOR_GRAY2BGR)
        )
        return cv2.resize(bgr, (tile, tile), interpolation=cv2.INTER_NEAREST)

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
            # Canonical cameras look down -Z, so in-front depth is -z.
            depth = -pc[2]
            if depth <= 1e-9:
                continue
            cam = self.cam_of[i]
            u, v = cam.project(pc[0] / depth, pc[1] / depth)
            cx, cy = cam.principal_point
            half_diag = 0.5 * float(np.hypot(cam.width, cam.height))
            radii.append(float(np.hypot(u - cx, v - cy)) / half_diag)
        return float(np.median(radii)) if radii else 0.0

    def reproj_error(self, pid: int, i: int) -> float:
        """Reprojection error (px) of point ``pid`` in image ``i``: the distance
        between the point's projection and its stored 2D keypoint. ``nan`` when
        the recon carries no keypoint for the observation or the point is behind
        the camera. For a point at infinity (``w == 0``) the camera-frame point is
        ``R · direction`` (no translation)."""
        kpt = self.kpt_obs.get(int(pid), {}).get(int(i))
        if kpt is None:
            return float("nan")
        center = self.positions[int(pid)]
        if self._w(pid) == 0.0:
            pc = self.rot_of[i] @ center
        else:
            pc = self.rot_of[i] @ (center - self.centers[i])
        # Canonical cameras look down -Z, so in-front depth is -z.
        depth = -pc[2]
        if depth <= 1e-9:
            return float("nan")
        u, v = self.cam_of[i].project(pc[0] / depth, pc[1] / depth)
        return float(np.hypot(u - kpt[0], v - kpt[1]))

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
            # Canonical cameras look down -Z, so in-front depth is -z.
            depth = -float(pc[2])
            if depth <= 1e-9 or focal <= 0:
                continue
            vals.append(float(fs[feat]) * depth / focal)
        return float(np.median(vals)) if vals else 0.0

    def strip(
        self,
        pid: int,
        *,
        tile: int,
        max_views: int | None = None,
        context: int | None = None,
        annotate: bool = False,
        normal_dot: bool = False,
    ) -> Strip | None:
        """Render point ``pid`` as a patch strip; ``(strip, mean_ncc, n)`` or None.

        With ``context`` (in pixels, ``> patch``) each tile renders a wider
        ``context``x``context`` window around the point at the same sampling
        density, with a 1px box marking the validated extent; NCC is still scored
        on that inner region.

        With ``annotate`` each tile is also labeled with that observation's NCC
        against the other views and its reprojection error (px).

        With ``normal_dot`` each tile gets an obliquity marker (see
        ``render_track_strip``): the unit vector from the patch centre toward that
        view's camera, projected onto the patch tangent plane, drawn as a dot on
        the patch-extent box — centre = fronto-parallel, edge = grazing view.
        Points at infinity (``w == 0``) have no finite camera-to-surface vector,
        so they get no marker.
        """
        obs_imgs = sorted(self.obs.get(int(pid), []))
        if not obs_imgs:
            return None
        if max_views and len(obs_imgs) > max_views:
            # Evenly-spaced representative subset of the sorted views so the
            # columns stay compact (already deduplicated when self.obs is built).
            sel = np.linspace(0, len(obs_imgs) - 1, max_views).round().astype(int)
            obs_imgs = [obs_imgs[i] for i in dict.fromkeys(sel.tolist())]
        up = self.rot_of[obs_imgs[0]].T @ np.array([0.0, 1.0, 0.0])

        if context and context > self.patch:
            render_res = context
            ext = self._half(pid) * (context / self.patch)
            inner = ((context - self.patch) // 2, self.patch)
        else:
            render_res = self.patch
            ext = self._half(pid)
            inner = None
        patch = self._oriented_patch(pid, up=up, ext=ext)

        def patch_of(i: int) -> np.ndarray:
            return self._render_view(patch, i, render_res)

        reproj = [self.reproj_error(pid, i) for i in obs_imgs] if annotate else None
        normal_offsets = (
            self._normal_offsets(pid, patch, obs_imgs) if normal_dot else None
        )
        w = gauss_window(self.patch)
        return render_track_strip(
            obs_imgs,
            patch_of,
            w,
            tile=tile,
            inner=inner,
            reproj_errs=reproj,
            per_view_scores=annotate,
            normal_offsets=normal_offsets,
        )

    def _normal_offsets(
        self, pid: int, patch: OrientedPatch, obs_imgs: list[int]
    ) -> list[tuple[float, float] | None]:
        """Per-view obliquity offsets ``(s, t)`` in the *displayed tile* frame:
        the unit vector from the patch centre toward each view's camera, projected
        onto ``(u_axis, -v_axis)``. The tile's ``+x`` is ``+u_axis`` and its
        ``+y`` (downward) is ``-v_axis`` -- the raster reverses ``v`` -- so ``t``
        projects onto ``-v_axis`` to land the dot on the same side of the tile as
        the camera. ``(0, 0)`` is a fronto-parallel view (camera on the surface
        normal); ``|(s, t)| -> 1`` as the view grazes the surface. The half-edge
        scaling cancels, so this is just the tangential part of the unit view
        direction. ``None`` for a point at infinity (no finite camera-to-surface
        vector)."""
        if self._w(pid) == 0.0:
            return [None] * len(obs_imgs)
        u = np.asarray(patch.u_axis, np.float64)
        v = np.asarray(patch.v_axis, np.float64)
        center = self.positions[int(pid)]
        offsets: list[tuple[float, float] | None] = []
        for i in obs_imgs:
            d = self.centers[i] - center
            norm = float(np.linalg.norm(d))
            if norm < 1e-9:
                offsets.append(None)
                continue
            d = d / norm
            # Tile down (+y) maps to -v_axis (raster reverses v), so t projects
            # onto -v_axis to place the dot on the camera's side of the tile.
            s, t = float(d @ u), float(-(d @ v))
            r = float(np.hypot(s, t))
            if r > 1.0:  # numerical guard; |(s,t)| = sin(theta) <= 1 in exact math
                s, t = s / r, t / r
            offsets.append((s, t))
        return offsets
