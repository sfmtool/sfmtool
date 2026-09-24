# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""What a track-at-pixel candidate is allowed to read.

:class:`DatasetContext` is built once per dataset: the reconstruction, the
decoded photographs as a shared pyramid set, the lazy descriptor index, every
image's ``.sift`` keypoints, the per-point patch frames laid out as arrays
with 2D (per image) and 3D spatial indexes over them, and the clusters of a
cluster-patches ``.matches`` file with a 2D index over their members.

:class:`HoldoutContext` is that with one point removed, and is the only thing a
candidate is handed. The point is deleted from ``edited`` (index-stable, so
every other point keeps its index) and filtered out of every neighbourhood
query, so a candidate cannot find the answer by looking it up.

A holdout can also be **empty**: every point is gone, not just the one under
test. ``edited`` then holds the cameras and no points, and the neighbourhood
queries over reconstructed points (:meth:`HoldoutContext.observations_near`,
:meth:`HoldoutContext.points_near`, :meth:`HoldoutContext.scene_depths`) return
nothing. The photographs, the descriptor index, the ``.sift`` keypoints and the
cluster-patches clusters are unchanged, since none of them comes from the
reconstruction's points. This is the state early in building a
reconstruction, when a track has to be built with no reconstructed
neighbours to take its normal, depth or size from.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from dataset import PreparedDataset, read_keypoints


def rotation_from_wxyz(q) -> np.ndarray:
    w, x, y, z = (float(v) for v in q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


@dataclass
class Camera:
    """One image's lens and ``cam_from_world`` pose."""

    intrinsics: object
    R: np.ndarray
    t: np.ndarray

    @property
    def center(self) -> np.ndarray:
        return -self.R.T @ self.t

    @property
    def focal(self) -> float:
        return float(np.mean(self.intrinsics.focal_lengths))

    def to_camera(self, xyz: np.ndarray, w: float = 1.0) -> np.ndarray:
        """World -> camera frame; a ``w = 0`` direction is rotated only."""
        return self.R @ np.asarray(xyz, float) + (self.t if w else 0.0)

    def depth(self, xyz: np.ndarray, w: float = 1.0) -> float:
        """Distance in front of the camera along its axis (cameras look down -Z)."""
        return -float(self.to_camera(xyz, w)[2])

    def project(self, xyz: np.ndarray, w: float = 1.0) -> np.ndarray | None:
        """Pixel of a world point (or bearing at ``w = 0``); ``None`` behind the camera."""
        p = self.to_camera(xyz, w)
        if -p[2] <= 1e-12:
            return None
        ray = p / np.linalg.norm(p)
        return np.asarray(self.intrinsics.ray_to_pixel(ray.tolist()), float)

    def ray(self, pixel) -> np.ndarray:
        """World-frame unit ray through a pixel."""
        d = np.asarray(self.intrinsics.pixel_to_ray(float(pixel[0]), float(pixel[1])))
        return self.R.T @ d


# The bitmap resolution the bench's reading and fit render at: the localizer's
# default `resolution`, which `open_localizer` keeps.
BENCH_RESOLUTION = 24


def texel_scale(
    camera: Camera,
    center,
    u_halfvec,
    v_halfvec,
    w: float = 1.0,
    resolution: int = BENCH_RESOLUTION,
) -> dict | None:
    """How many image pixels one patch-bitmap texel covers in this view.

    The patch's ``resolution`` texels span ``2 |u_halfvec|`` along ``u`` (and
    the same along ``v``), so the Jacobian from texel coordinates to pixels is
    the projection's derivative along each half-vector scaled by
    ``2 / resolution``. It is taken at the patch centre by central differences.
    ``scale`` is ``sqrt(|det J|)``: 1 means the bitmap samples this view at
    about one texel per pixel, above 1 means the bitmap is coarser than the
    image, below 1 means it is finer. ``anisotropy`` is the ratio of ``J``'s
    singular values, 1 for a view facing the patch. ``None`` when the patch
    does not project into the view.
    """
    c = np.asarray(center, float)
    u = np.asarray(u_halfvec, float)
    v = np.asarray(v_halfvec, float)
    step = 1e-3
    cols = []
    for axis in (u, v):
        plus = camera.project(c + step * axis, w)
        minus = camera.project(c - step * axis, w)
        if plus is None or minus is None:
            return None
        # d(pixel)/d(half-vector units), then per texel: 2 / resolution.
        cols.append((plus - minus) / (2 * step) * (2.0 / resolution))
    jac = np.column_stack(cols)
    sv = np.linalg.svd(jac, compute_uv=False)
    if sv[1] <= 0:
        return None
    return {
        "scale": float(np.sqrt(abs(np.linalg.det(jac)))),
        "anisotropy": float(sv[0] / sv[1]),
        "jacobian": jac,
    }


class DatasetContext:
    """Everything loaded once per dataset."""

    def __init__(self, prepared: PreparedDataset):
        from sfmtool._sfmtool.patches import ImagePyramidSet
        from sfmtool._sfmtool.reconstruction import (
            EditedReconstruction,
            SfmrReconstruction,
        )
        from sfmtool._sfmtool.spatial import KdTree2d, KdTree3d, LazyKdForest
        from sfmtool._workspace_image import read_workspace_image

        self.prepared = prepared
        self.recon = SfmrReconstruction.load(prepared.sfmr)
        recon = self.recon
        self.image_names = list(recon.image_names)
        self.image_stems = [Path(n).stem for n in self.image_names]
        self.images = [
            read_workspace_image(recon.workspace_dir, n) for n in self.image_names
        ]
        self.pyramids = ImagePyramidSet(recon, self.images)
        self.forest = LazyKdForest(str(prepared.kdf))
        self.keypoints = [
            read_keypoints(prepared.workspace, n)[:2] for n in self.image_names
        ]

        cams = recon.cameras
        cam_idx = np.asarray(recon.camera_indexes)
        quats = np.asarray(recon.quaternions_wxyz)
        trans = np.asarray(recon.translations)
        self.cameras = [
            Camera(cams[int(cam_idx[i])], rotation_from_wxyz(quats[i]), trans[i])
            for i in range(recon.image_count)
        ]

        # Per-point frames, read through the edited view so the arrays are in
        # exactly the form a bench record carries.
        full = EditedReconstruction(recon)
        n = recon.point_count
        self.point_xyz = np.zeros((n, 3))
        self.point_w = np.zeros(n)
        self.point_u = np.zeros((n, 3))
        self.point_v = np.zeros((n, 3))
        self.point_images: list[np.ndarray] = []
        self.point_keypoints: list[np.ndarray] = []
        for i in range(n):
            rec = full.point(i)
            self.point_xyz[i] = rec["position"]
            self.point_w[i] = rec["w"]
            self.point_u[i] = rec.get("patch_u_halfvec", np.zeros(3))
            self.point_v[i] = rec.get("patch_v_halfvec", np.zeros(3))
            self.point_images.append(np.asarray(rec["image_indexes"], dtype=np.int64))
            self.point_keypoints.append(np.asarray(rec["keypoints_xy"], dtype=float))
        normal = np.cross(self.point_u, self.point_v)
        norm = np.linalg.norm(normal, axis=1, keepdims=True)
        self.point_normal = np.where(norm > 0, normal / np.maximum(norm, 1e-300), 0.0)
        self.point_half = np.linalg.norm(self.point_u, axis=1)

        # 2D: per image, every observation's keypoint -> (point, row).
        self._obs_point: list[np.ndarray] = []
        self._obs_xy: list[np.ndarray] = []
        self._obs_tree = []
        for image in range(recon.image_count):
            pts, xy = [], []
            for p in range(n):
                hit = np.flatnonzero(self.point_images[p] == image)
                if hit.size:
                    pts.append(p)
                    xy.append(self.point_keypoints[p][hit[0]])
            pts_arr = np.asarray(pts, dtype=np.int64)
            xy_arr = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
            self._obs_point.append(pts_arr)
            self._obs_xy.append(xy_arr)
            self._obs_tree.append(KdTree2d(xy_arr) if len(xy_arr) else None)

        # 3D: finite points only.
        self._finite = np.flatnonzero(self.point_w != 0)
        self._tree3d = KdTree3d(self.point_xyz[self._finite])

        self._load_clusters(prepared.matches)

    def _load_clusters(self, path: Path) -> None:
        """The cluster-patches ``.matches`` file, re-indexed onto this reconstruction.

        Member image indexes are mapped by name, so a file whose image table is
        ordered or subset differently still lands on the right images; a member
        in an image the reconstruction does not hold is kept with image ``-1``
        and never indexed. Each image gets a 2D index over its members'
        positions, which are the refined ones for every status the refinement
        measured and the detections for the rest.
        """
        from sfmtool._sfmtool.analysis import cluster_radii
        from sfmtool._sfmtool.io import MatchesFile
        from sfmtool._sfmtool.spatial import KdTree2d

        mf = MatchesFile(str(path))
        if not mf.has_clusters:
            raise SystemExit(f"{path} has no clusters section")
        self.matches = mf
        name_to_image = {n: i for i, n in enumerate(self.image_names)}
        to_recon = np.asarray(
            [name_to_image.get(n, -1) for n in mf.image_names], dtype=np.int64
        )
        self.cluster_starts = np.asarray(mf.cluster_starts, dtype=np.int64)
        self.member_images = to_recon[np.asarray(mf.member_images, dtype=np.int64)]
        self.member_features = np.asarray(mf.member_features, dtype=np.int64)
        self.member_positions = np.asarray(mf.member_positions(), dtype=np.float64)
        self.member_shapes = np.asarray(mf.member_affine_shapes(), dtype=np.float64)
        k = len(self.member_images)
        self.member_cluster = np.repeat(
            np.arange(len(self.cluster_starts) - 1), np.diff(self.cluster_starts)
        )
        self.has_cluster_patches = bool(mf.has_cluster_patches)
        if self.has_cluster_patches:
            self.member_status = np.asarray(mf.member_status, dtype=np.int64)
            self.member_zncc = np.asarray(mf.member_zncc, dtype=np.float64)
            self.member_shift_px = np.asarray(mf.member_shift_px, dtype=np.float64)
            self.reference_members = np.asarray(mf.reference_members, dtype=np.int64)
            self.cluster_radius = np.asarray(cluster_radii(mf), dtype=np.float64)
        else:
            self.member_status = np.full(k, 5)  # not_evaluated
            self.member_zncc = np.full(k, np.nan)
            self.member_shift_px = np.full(k, np.nan)
            self.reference_members = np.full(len(self.cluster_starts) - 1, 0xFFFFFFFF)
            self.cluster_radius = np.full(len(self.cluster_starts) - 1, np.nan)
        self._member_rows = []
        self._member_tree = []
        for image in range(len(self.image_names)):
            rows = np.flatnonzero(self.member_images == image)
            self._member_rows.append(rows)
            self._member_tree.append(
                KdTree2d(self.member_positions[rows]) if len(rows) else None
            )

    def holdout(self, point: int, *, empty: bool = False) -> "HoldoutContext":
        return HoldoutContext(self, point, empty=empty)

    @property
    def empty_recon(self):
        """The reconstruction's cameras with none of its points, built once."""
        if getattr(self, "_empty_recon", None) is None:
            self._empty_recon = self.recon.filter_points_by_mask(
                np.zeros(self.recon.point_count, dtype=bool)
            )
        return self._empty_recon


class HoldoutContext:
    """The dataset with one point removed, or with every point removed when
    ``empty``: what a candidate is handed."""

    def __init__(self, dataset: DatasetContext, point: int | None, *, empty=False):
        from sfmtool._sfmtool.reconstruction import EditedReconstruction

        self.dataset = dataset
        self._excluded = point
        self.empty = empty
        if empty:
            self.edited = EditedReconstruction(dataset.empty_recon)
        else:
            self.edited = EditedReconstruction(dataset.recon)
            if point is not None:
                self.edited.delete_point(int(point))

    # -- pass-throughs a candidate needs -----------------------------------
    @property
    def recon(self):
        """The base reconstruction. Its point columns still hold the held-out
        point; read points through :meth:`observations_near` /
        :meth:`points_near` / ``edited`` instead."""
        return self.dataset.recon

    @property
    def pyramids(self):
        return self.dataset.pyramids

    @property
    def forest(self):
        return self.dataset.forest

    def keypoints(self, image: int):
        return self.dataset.keypoints[image]

    def camera(self, image: int) -> Camera:
        return self.dataset.cameras[image]

    def image_stem(self, image: int) -> str:
        return self.dataset.image_stems[image]

    def image_size(self, image: int) -> tuple[int, int]:
        c = self.dataset.cameras[image].intrinsics
        return int(c.width), int(c.height)

    def texel_scales(self, track, *, verdict: str | None = "in") -> dict[int, dict]:
        """:func:`texel_scale` for each observation of a track-stage ``track``.

        Keyed by observation index, over the observations with ``verdict``
        (every observation when ``None``). Empty when the track has no patch.
        """
        placement = track.placement
        if placement is None:
            return {}
        out = {}
        for i, o in enumerate(track.observations):
            if verdict is not None and o["verdict"] != verdict:
                continue
            s = texel_scale(
                self.camera(int(o["image"])),
                placement["center"],
                placement["u_halfvec"],
                placement["v_halfvec"],
                float(placement["w"]),
            )
            if s is not None:
                out[i] = s
        return out

    # -- neighbourhood queries (the held-out point is never returned) ------
    def observations_near(self, image: int, pixel, radius_px: float) -> list[dict]:
        """Reconstruction observations in ``image`` within ``radius_px`` of ``pixel``.

        Each carries the point, its keypoint, the pixel distance and offset, the
        point's depth along the query camera's axis (``None`` for a bearing), its
        outward normal, its world half-extent and that half-extent's apparent
        size in ``image``'s pixels.
        """
        ds = self.dataset
        tree = ds._obs_tree[image]
        if tree is None or self.empty:
            return []
        q = np.asarray([pixel], dtype=np.float64)
        offsets, idx = tree.within_radius(q, float(radius_px))
        rows = np.asarray(idx[offsets[0] : offsets[1]], dtype=np.int64)
        cam = ds.cameras[image]
        out = []
        for r in rows:
            p = int(ds._obs_point[image][r])
            if p == self._excluded:
                continue
            xy = ds._obs_xy[image][r]
            finite = ds.point_w[p] != 0
            depth = cam.depth(ds.point_xyz[p]) if finite else None
            half_px = (
                ds.point_half[p] * cam.focal / depth
                if finite and depth and depth > 0
                else float("nan")
            )
            out.append(
                {
                    "point": p,
                    "keypoint": xy.copy(),
                    "offset_px": xy - np.asarray(pixel, float),
                    "distance_px": float(np.linalg.norm(xy - np.asarray(pixel))),
                    "depth": depth,
                    "normal": ds.point_normal[p].copy(),
                    "half_extent": float(ds.point_half[p]),
                    "half_px": float(half_px),
                    "at_infinity": not finite,
                    "track_length": int(len(ds.point_images[p])),
                }
            )
        out.sort(key=lambda o: o["distance_px"])
        return out

    def clusters_near(self, image: int, pixel, radius_px: float) -> list[dict]:
        """Clusters with a member in ``image`` within ``radius_px`` of ``pixel``.

        Clusters come from the cluster-patches ``.matches`` file, which is built
        from the detected SIFT keypoints and the descriptor index alone, so it
        holds nothing of the held-out point. One entry per cluster, nearest
        first, carrying the member in ``image`` that is nearest the pixel
        (``member``), and every member of the cluster (``members``) with its
        reconstruction image, ``.sift`` row, position, affine shape, status (the
        ``member_status`` legend: ``reference``, ``kept``, ``rejected_low_zncc``,
        ``rejected_shift``, ``duplicate_image``, ``not_evaluated``,
        ``rejected_unlocalizable``), ZNCC against the reference and shift from
        its seed. ``radius`` is the cluster's patch radius in pixels of its
        widest member; the patch cluster is the ``reference`` plus the ``kept``.
        """
        ds = self.dataset
        tree = ds._member_tree[image]
        if tree is None:
            return []
        q = np.asarray([pixel], dtype=np.float64)
        offsets, idx = tree.within_radius(q, float(radius_px))
        rows = ds._member_rows[image][np.asarray(idx[offsets[0] : offsets[1]])]
        nearest: dict[int, tuple[float, int]] = {}
        for r in rows:
            d = float(np.linalg.norm(ds.member_positions[r] - q[0]))
            c = int(ds.member_cluster[r])
            if c not in nearest or d < nearest[c][0]:
                nearest[c] = (d, int(r))
        out = []
        for c, (d, r) in sorted(nearest.items(), key=lambda kv: kv[1][0]):
            start, end = ds.cluster_starts[c], ds.cluster_starts[c + 1]
            reference = int(ds.reference_members[c])
            out.append(
                {
                    "cluster": c,
                    "distance_px": d,
                    "member": _member(ds, r),
                    "members": [_member(ds, m) for m in range(start, end)],
                    "reference": None if reference == 0xFFFFFFFF else reference,
                    "radius": float(ds.cluster_radius[c]),
                }
            )
        return out

    def points_near(
        self, xyz, *, k: int = 12, radius: float | None = None
    ) -> list[dict]:
        """Finite points nearest ``xyz`` in 3D, with their frames."""
        if self.empty:
            return []
        ds = self.dataset
        q = np.asarray([xyz], dtype=np.float64)
        if radius is None:
            idx = np.asarray(ds._tree3d.nearest_k(q, k + 1)[0])
        else:
            idx = np.asarray(ds._tree3d.nearest_k_within_radius(q, k + 1, radius)[0])
        out = []
        for j in idx:
            if j >= len(ds._finite):
                continue
            p = int(ds._finite[j])
            if p == self._excluded:
                continue
            out.append(
                {
                    "point": p,
                    "position": ds.point_xyz[p].copy(),
                    "distance": float(np.linalg.norm(ds.point_xyz[p] - xyz)),
                    "normal": ds.point_normal[p].copy(),
                    "half_extent": float(ds.point_half[p]),
                    "images": ds.point_images[p].copy(),
                }
            )
        return out[:k]

    def scene_depths(self, image: int) -> np.ndarray:
        """Depths of the finite points in front of ``image``'s camera and inside its frame.

        The held-out point is left out, and an empty holdout has none.
        """
        if self.empty:
            return np.zeros(0)
        ds = self.dataset
        cam = self.camera(image)
        w, h = self.image_size(image)
        out = []
        for p in ds._finite:
            if p == self._excluded:
                continue
            px = cam.project(ds.point_xyz[p])
            if px is not None and 0 <= px[0] < w and 0 <= px[1] < h:
                out.append(cam.depth(ds.point_xyz[p]))
        d = np.asarray(out)
        return d[d > 0]


MEMBER_STATUS = (
    "reference",
    "kept",
    "rejected_low_zncc",
    "rejected_shift",
    "duplicate_image",
    "not_evaluated",
    "rejected_unlocalizable",
)


def _member(ds: DatasetContext, m: int) -> dict:
    status = int(ds.member_status[m])
    return {
        "index": int(m),
        "image": int(ds.member_images[m]),
        "feature": int(ds.member_features[m]),
        "position": ds.member_positions[m].copy(),
        "shape": ds.member_shapes[m].copy(),
        "status": MEMBER_STATUS[status] if status < len(MEMBER_STATUS) else status,
        "zncc": float(ds.member_zncc[m]),
        "shift_px": float(ds.member_shift_px[m]),
    }
