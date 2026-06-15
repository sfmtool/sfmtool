# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Photometric patch-normal refinement as an ``sfm xform`` operation.

Unlike the geometric transforms, ``RefineNormalsTransform`` is a *modifier*: it
removes no points and moves nothing — it only rewrites each finite point's
``estimated_normal`` to the normal that maximizes cross-view photometric
consensus. Because the refinement is photometric it reads the workspace's source
images (``workspace_dir / image_name``), the same way the SIFT-reading filters
reach back for the ``.sift`` files; a missing image is a hard error. The default
``extent=feature_size`` patch sizing additionally reads the ``.sift`` files.

See ``specs/cli/xform-refine-normals-command.md`` and
``specs/core/patch-normal-refinement.md``.
"""

import numpy as np

from .._sfmtool import PatchCloud, SfmrReconstruction

# Confidence (the peakedness of Φ at the optimum) is normalized to roughly
# [0, 1] by the core routine; below this it is reported — but not acted on — as
# "low confidence". The threshold is a reporting heuristic only (v1 never gates
# on confidence; see the spec's "Confidence is report-only" section).
_LOW_CONFIDENCE_THRESHOLD = 0.1

_OBJECTIVES = ("robust", "mean")
_WINDOWS = ("gaussian_disk", "gaussian", "uniform")
_SAMPLERS = ("bilinear", "anisotropic")
_INITIAL_NORMALS = ("stored", "mean_viewing", "geometric")
_EXTENTS = ("feature_size", "fixed", "relative_spacing", "pixel_radius")
_CACHES = ("off", "fronto")
_QUALITIES = ("none", "coarse", "fine")

# Quality presets: convenience that sets (cache, cache_supersample) together.
# ``coarse`` trades a little tail accuracy for ~2× speed via the fronto-parallel
# cache; ``fine`` is the exact source-rendering path. ``none`` defers to the
# explicit ``cache`` / ``cache_supersample`` knobs. See
# ``specs/core/fronto-parallel-patch-cache.md``.
_QUALITY_PRESETS = {
    "coarse": ("fronto", 2.0),
    "fine": ("off", 1.0),
}


class RefineNormalsTransform:
    """Refine per-point surface normals by photometric cross-view consensus.

    All knobs default to the ``PatchCloud.refine_normals`` /
    ``PatchCloud.from_reconstruction`` binding defaults, so the two layers
    cannot drift. The point count, positions, poses, and cameras are unchanged;
    only ``estimated_normals`` is rewritten (finite points only — points at
    infinity have no surface element and pass through).
    """

    def __init__(
        self,
        *,
        # forwarded to PatchCloud.refine_normals
        angular_range_deg: float = 25.0,
        init_steps: int = 7,
        refine_levels: int = 3,
        resolution: int = 24,
        objective: str = "robust",
        robust_iters: int = 3,
        window: str = "gaussian_disk",
        window_sigma: float = 0.6,
        sampler: str = "bilinear",
        min_valid_fraction: float = 0.6,
        min_views: int = 3,
        cache: str = "fronto",
        cache_supersample: float = 2.0,
        quality: str = "none",
        # forwarded to PatchCloud.from_reconstruction
        initial_normals: str = "stored",
        extent: str = "feature_size",
        extent_value: float = 5.0,
    ):
        if angular_range_deg <= 0:
            raise ValueError(
                f"angular_range_deg must be positive, got {angular_range_deg}"
            )
        if init_steps < 2:
            raise ValueError(f"init_steps must be >= 2, got {init_steps}")
        if refine_levels < 1:
            raise ValueError(f"refine_levels must be >= 1, got {refine_levels}")
        if resolution < 2:
            raise ValueError(f"resolution must be >= 2, got {resolution}")
        if objective not in _OBJECTIVES:
            raise ValueError(
                f"objective must be one of {_OBJECTIVES}, got {objective!r}"
            )
        if robust_iters < 1:
            raise ValueError(f"robust_iters must be >= 1, got {robust_iters}")
        if window not in _WINDOWS:
            raise ValueError(f"window must be one of {_WINDOWS}, got {window!r}")
        if window_sigma <= 0:
            raise ValueError(f"window_sigma must be positive, got {window_sigma}")
        if sampler not in _SAMPLERS:
            raise ValueError(f"sampler must be one of {_SAMPLERS}, got {sampler!r}")
        if not 0.0 <= min_valid_fraction <= 1.0:
            raise ValueError(
                f"min_valid_fraction must be in [0, 1], got {min_valid_fraction}"
            )
        if min_views < 2:
            raise ValueError(f"min_views must be >= 2, got {min_views}")
        if cache not in _CACHES:
            raise ValueError(f"cache must be one of {_CACHES}, got {cache!r}")
        if cache_supersample < 1.0:
            raise ValueError(
                f"cache_supersample must be >= 1.0, got {cache_supersample}"
            )
        if quality not in _QUALITIES:
            raise ValueError(f"quality must be one of {_QUALITIES}, got {quality!r}")
        if initial_normals not in _INITIAL_NORMALS:
            raise ValueError(
                f"initial_normals must be one of {_INITIAL_NORMALS}, got {initial_normals!r}"
            )
        if extent not in _EXTENTS:
            raise ValueError(f"extent must be one of {_EXTENTS}, got {extent!r}")
        if extent_value <= 0:
            raise ValueError(f"extent_value must be positive, got {extent_value}")

        self.angular_range_deg = angular_range_deg
        self.init_steps = init_steps
        self.refine_levels = refine_levels
        self.resolution = resolution
        self.objective = objective
        self.robust_iters = robust_iters
        self.window = window
        self.window_sigma = window_sigma
        self.sampler = sampler
        self.min_valid_fraction = min_valid_fraction
        self.min_views = min_views
        # A quality preset (other than ``none``) wins over the explicit cache
        # knobs, so the two never disagree.
        if quality != "none":
            self.cache, self.cache_supersample = _QUALITY_PRESETS[quality]
        else:
            self.cache = cache
            self.cache_supersample = cache_supersample
        self.quality = quality
        self.initial_normals = initial_normals
        self.extent = extent
        self.extent_value = extent_value

    def _load_images(self, recon: SfmrReconstruction) -> list[np.ndarray]:
        """Load every source image, parallel to the reconstruction's images.

        Resolves ``workspace_dir / image_name`` exactly as the SIFT-reading
        filters resolve their ``.sift`` paths. A missing image is a hard error.
        ``refine_normals`` requires one full-resolution image per reconstruction
        image (matching its camera resolution), so all are loaded up front.
        """
        from sfmtool._workspace_image import read_workspace_image

        return [
            read_workspace_image(recon.workspace_dir, name)
            for name in recon.image_names
        ]

    def apply(self, recon: SfmrReconstruction) -> SfmrReconstruction:
        images = self._load_images(recon)

        cloud = PatchCloud.from_reconstruction(
            recon,
            normal=self.initial_normals,
            extent=self.extent,
            extent_value=self.extent_value,
        )
        point_ids = np.asarray(cloud.point_ids)
        print(
            f"  Refining normals for {len(point_ids)} finite points "
            f"across {recon.image_count} images..."
        )

        result = cloud.refine_normals(
            recon,
            images,
            resolution=self.resolution,
            angular_range_deg=self.angular_range_deg,
            init_steps=self.init_steps,
            refine_levels=self.refine_levels,
            objective=self.objective,
            robust_iters=self.robust_iters,
            window=self.window,
            window_sigma=self.window_sigma,
            min_valid_fraction=self.min_valid_fraction,
            min_views=self.min_views,
            sampler=self.sampler,
            cache=self.cache,
            cache_supersample=self.cache_supersample,
        )

        refined = np.asarray(result["normal"], dtype=np.float32)
        photo = np.asarray(result["photoconsistency"], dtype=np.float64)
        init = np.asarray(result["init_photoconsistency"], dtype=np.float64)
        conf = np.asarray(result["confidence"], dtype=np.float64)

        # Copy-and-scatter keeps the normals of excluded (infinity) points
        # intact while overwriting every finite point. point_ids are 3D-point
        # indices, so normals[pid] indexes the right row directly.
        normals = np.asarray(recon.estimated_normals, dtype=np.float32).copy()
        normals[point_ids] = refined

        self._print_summary(photo, init, conf)

        return recon.clone_with_changes(estimated_normals=normals)

    def _print_summary(
        self, photo: np.ndarray, init: np.ndarray, conf: np.ndarray
    ) -> None:
        """One-line ``xform``-style summary over the patches actually scored.

        Patches without enough valid views return NaN scores (kept their seed),
        so the statistics are over the finitely-scored subset.
        """
        scored = np.isfinite(photo) & np.isfinite(init)
        n = int(scored.sum())
        if n == 0:
            print("  Refined 0 normals (no patches had enough valid views)")
            return

        mean_init = float(init[scored].mean())
        mean_photo = float(photo[scored].mean())
        improved = int(np.count_nonzero(photo[scored] > init[scored] + 1e-6))
        low_conf = int(np.count_nonzero(conf[scored] < _LOW_CONFIDENCE_THRESHOLD))
        print(
            f"  Refined {n} normals (mean Φ {mean_init:.2f} → {mean_photo:.2f}, "
            f"{mean_photo - mean_init:+.2f}; {improved} improved, {low_conf} low-confidence)"
        )

    def description(self) -> str:
        return (
            f"Refine normals (initial={self.initial_normals}, extent={self.extent}, "
            f"range={self.angular_range_deg}°, objective={self.objective}, "
            f"sampler={self.sampler})"
        )
