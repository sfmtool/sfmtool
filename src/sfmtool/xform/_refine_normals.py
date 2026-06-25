# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Photometric patch-normal refinement as an ``sfm xform`` operation.

Unlike the geometric transforms, ``RefineNormalsTransform`` is a *modifier*: it
removes no points and moves nothing — it only rewrites each finite point's
``normal`` to the one that maximizes cross-view photometric consensus. Because
the refinement is photometric it reads the workspace's source images
(``workspace_dir / image_name``), the same way the SIFT-reading filters reach
back for the ``.sift`` files; a missing image is a hard error.

The operation requires an ``embedded_patches`` reconstruction (enforced by
``apply_transforms``): it reads the point's stored patch frame back
(``recon.patches``) and refines the normal over the stored per-observation
keypoints, anchoring each view on its real detected feature. The refined patch
cloud is always re-persisted as the per-point patch frame in the ``.sfmr``
``points3d/`` section (format version 3+) so the frame stays consistent with the
rewritten normal; with ``bitmaps`` the per-point RGBA patch textures are rendered
and persisted too. Frame *sizing* and *seeding* (``extent`` / ``extent_value`` /
``normal``) belong to ``--to-embedded-patches``, the step that builds the frame;
refine-normals reuses that frame and takes none of those knobs.

See ``specs/cli/xform-refine-normals-command.md`` and
``specs/core/patch-normal-refinement.md``.
"""

import numpy as np

from .._sfmtool import SfmrReconstruction

# Confidence (the peakedness of Φ at the optimum) is normalized to roughly
# [0, 1] by the core routine; below this it is reported — but not acted on — as
# "low confidence". The threshold is a reporting heuristic only (v1 never gates
# on confidence; see the spec's "Confidence is report-only" section).
_LOW_CONFIDENCE_THRESHOLD = 0.1

_OBJECTIVES = ("robust", "mean")
_WINDOWS = ("gaussian_disk", "gaussian", "uniform")
_SAMPLERS = ("bilinear", "anisotropic")
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

    All knobs default to the ``PatchCloud.refine_normals`` binding defaults, so
    the two layers cannot drift. The point count, positions, poses, and cameras
    are unchanged; only ``normals`` is rewritten (finite points only — points at
    infinity have no surface element and pass through). The refined patch
    geometry is always re-persisted alongside the normals (the stored frame must
    stay consistent with them).

    Requires an ``embedded_patches`` reconstruction (enforced by
    ``apply_transforms``): it refines over the stored per-observation keypoints
    and reuses the stored per-point patch frame, both of which only that source
    carries. Convert first with ``--to-embedded-patches`` (the step that builds
    the frame — its ``extent`` / ``extent_value`` / ``normal`` knobs size and
    seed it; refine-normals takes none of those).
    """

    # Precondition checked per-step by `apply_transforms` (see `_apply.py`).
    required_feature_source = "embedded_patches"

    def __init__(
        self,
        *,
        # whether to render and persist the per-point RGBA patch bitmaps
        bitmaps: bool = False,
        # forwarded to PatchCloud.refine_normals
        angular_range_deg: float = 25.0,
        init_steps: int = 7,
        refine_levels: int = 3,
        resolution: int = 24,
        objective: str = "robust",
        robust_iters: int = 3,
        search_robust_iters: int | None = None,
        window: str = "gaussian_disk",
        window_sigma: float = 0.6,
        sampler: str = "bilinear",
        min_valid_fraction: float = 0.6,
        min_views: int = 3,
        cache: str = "fronto",
        cache_supersample: float = 2.0,
        quality: str = "none",
        confidence: bool = False,
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
        if search_robust_iters is not None and search_robust_iters < 0:
            raise ValueError(
                f"search_robust_iters must be >= 0 or None, got {search_robust_iters}"
            )
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
        if not isinstance(confidence, bool):
            raise ValueError(f"confidence must be a bool, got {confidence!r}")
        if not isinstance(bitmaps, bool):
            raise ValueError(f"bitmaps must be a bool, got {bitmaps!r}")

        self.bitmaps = bitmaps
        self.angular_range_deg = angular_range_deg
        self.init_steps = init_steps
        self.refine_levels = refine_levels
        self.resolution = resolution
        self.objective = objective
        self.robust_iters = robust_iters
        self.search_robust_iters = search_robust_iters
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
        self.confidence = confidence

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

        # The reconstruction is embedded_patches (enforced by apply_transforms), so
        # the patch frame is already stored — read it back as the cloud rather than
        # rebuilding it (frame sizing/seeding happened in --to-embedded-patches). The
        # normal is refined over the stored per-observation keypoints
        # (use_stored_keypoints), anchoring each view on its real detected feature.
        cloud = recon.patches
        if cloud is None:
            raise ValueError(
                "reconstruction has no patch frames to refine; expected "
                "embedded_patches (run `sfm xform --to-embedded-patches` first)"
            )
        point_ids = np.asarray(cloud.point_ids)
        # A point at infinity has a fixed normal (normalize(-d)) the refiner skips;
        # keep its stored (0, 0, 0) normal untouched in the write-back below.
        finite = ~np.asarray(recon.point_is_at_infinity)[point_ids]
        print(
            f"  Refining normals for {int(finite.sum())} finite points "
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
            search_robust_iters=self.search_robust_iters,
            window=self.window,
            window_sigma=self.window_sigma,
            min_valid_fraction=self.min_valid_fraction,
            min_views=self.min_views,
            sampler=self.sampler,
            cache=self.cache,
            cache_supersample=self.cache_supersample,
            compute_confidence=self.confidence,
            use_stored_keypoints=True,
            render_bitmaps=self.bitmaps,
        )

        refined = np.asarray(result["normal"], dtype=np.float32)
        photo = np.asarray(result["photoconsistency"], dtype=np.float64)
        init = np.asarray(result["init_photoconsistency"], dtype=np.float64)
        conf = np.asarray(result["confidence"], dtype=np.float64)

        # Copy-and-scatter the refined normals to finite points only; infinity
        # points keep their stored normal. point_ids are 3D-point indices, so
        # normals[pid] indexes the right row directly.
        normals = np.asarray(recon.normals, dtype=np.float32).copy()
        finite_pids = point_ids[finite]
        normals[finite_pids] = refined[finite]

        self._print_summary(photo, init, conf)

        # The input is already embedded_patches, so always persist the refined
        # patch cloud (its u/v frame now matches the refined normal) — otherwise the
        # stored frame would disagree with the rewritten normals. With `bitmaps`,
        # also persist the fused per-point RGBA patch textures (scattered to
        # per-point rows by the binding).
        if self.bitmaps:
            n_filled = int(np.count_nonzero(result["bitmaps"].any(axis=(1, 2, 3))))
            print(
                f"  Saving {len(point_ids)} patches and {n_filled} bitmaps "
                f"to the reconstruction"
            )
            return recon.clone_with_changes(
                normals=normals, patches=cloud, patch_bitmaps=result["bitmaps"]
            )
        return recon.clone_with_changes(normals=normals, patches=cloud)

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
        # Confidence is computed (and reported) only when explicitly requested.
        conf_note = ""
        if self.confidence:
            low_conf = int(np.count_nonzero(conf[scored] < _LOW_CONFIDENCE_THRESHOLD))
            conf_note = f", {low_conf} low-confidence"
        print(
            f"  Refined {n} normals (mean Φ {mean_init:.2f} → {mean_photo:.2f}, "
            f"{mean_photo - mean_init:+.2f}; {improved} improved{conf_note})"
        )

    def description(self) -> str:
        # This string is also reused as the operation name in the precondition
        # error. `bitmaps` gates whether the RGBA textures are rendered.
        bitmaps = ", bitmaps" if self.bitmaps else ""
        return (
            f"Refine normals (range={self.angular_range_deg}°, "
            f"objective={self.objective}, sampler={self.sampler}{bitmaps})"
        )
