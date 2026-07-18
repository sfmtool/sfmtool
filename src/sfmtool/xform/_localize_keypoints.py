# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Cross-view keypoint localization (search) as an ``sfm xform`` operation.

Unlike ``RefineKeypointsTransform`` (a pure in-place modifier), this op is
**structural**: ``PatchCloud.localize_keypoints`` congeals each point's
per-view keypoints by a discrete cross-view search and **drops views that
won't co-register** (drift too far, leave the frame, graze the patch plane, or
stop agreeing with the leave-one-out consensus). After a ``min_views`` cull the
survivors are renumbered and the reconstruction is rebuilt — ``keypoints_xy``
and all three track arrays — via :func:`compact_to_embedded_patches`, the same
helper the ``embed-patches`` pipeline uses. The output therefore has fewer
observations (and possibly fewer points) than the input.

The localizer renders no bitmaps, and any stored ones would be stale after the
keypoints move and views drop, so the output carries patch *frames* but **no
bitmaps** — re-run ``--refine-keypoints bitmaps=true`` (or
``--refine-normals bitmaps=true``) to regenerate them.

Because the search is photometric it reads the workspace's source images
(``workspace_dir / image_name``), the same way ``--refine-keypoints`` does; a
missing image is a hard error.

See ``specs/cli/xform-localize-keypoints-command.md`` and
``specs/core/patch-keypoint-localization.md``.
"""

from .._sfmtool.reconstruction import SfmrReconstruction
from ._images import load_workspace_images

_WINDOWS = ("gaussian_disk", "gaussian", "uniform")
_SAMPLERS = ("bilinear", "bilinear_mip", "anisotropic")
_SEARCH_STRATEGIES = ("exhaustive", "plus_descent")


class LocalizeKeypointsTransform:
    """Localize per-observation 2D keypoints by cross-view search (structural).

    All search knobs default to the ``PatchCloud.localize_keypoints`` binding
    defaults, so the two layers cannot drift; ``min_views`` (default 2) is the
    compaction cull threshold. Localization runs over each point's full track
    (``view_sets=None``) and the write-back rebuilds the track arrays from the
    kept views via ``compact_to_embedded_patches`` — views and points can be
    dropped, and the survivors are renumbered densely.

    Requires an ``embedded_patches`` reconstruction (enforced by
    ``apply_transforms``): the localizer searches over the stored per-point
    patch frame (``recon.patches``), which only that source carries, seeding
    each view at the point's own projection. Convert first with
    ``--to-embedded-patches``.
    """

    # Precondition checked per-step by `apply_transforms` (see `_apply.py`).
    required_feature_source = "embedded_patches"

    def __init__(
        self,
        *,
        # compaction cull: drop a point whose kept-view count is below this
        min_views: int = 2,
        # forwarded to PatchCloud.localize_keypoints
        max_iters: int = 5,
        search: float = 6.0,
        max_shift_px: float = 3.0,
        min_relative_zncc: float = 0.7,
        min_grazing_cos: float = 0.1,
        resolution: int = 24,
        window: str = "gaussian_disk",
        window_sigma: float = 0.6,
        sampler: str = "bilinear",
        robust_iters: int = 3,
        convergence_px: float = 0.05,
        search_resolution_multiplier: float = 1.0,
        search_strategy: str = "plus_descent",
    ):
        if min_views < 1:
            raise ValueError(f"min_views must be >= 1, got {min_views}")
        if max_iters < 1:
            raise ValueError(f"max_iters must be >= 1, got {max_iters}")
        if search <= 0:
            raise ValueError(f"search must be positive, got {search}")
        if max_shift_px <= 0:
            raise ValueError(f"max_shift_px must be positive, got {max_shift_px}")
        if not 0 <= min_relative_zncc <= 1:
            raise ValueError(
                f"min_relative_zncc must be in [0, 1], got {min_relative_zncc}"
            )
        if not 0 <= min_grazing_cos <= 1:
            raise ValueError(
                f"min_grazing_cos must be in [0, 1], got {min_grazing_cos}"
            )
        if resolution < 2:
            raise ValueError(f"resolution must be >= 2, got {resolution}")
        if window not in _WINDOWS:
            raise ValueError(f"window must be one of {_WINDOWS}, got {window!r}")
        if window_sigma <= 0:
            raise ValueError(f"window_sigma must be positive, got {window_sigma}")
        if sampler not in _SAMPLERS:
            raise ValueError(f"sampler must be one of {_SAMPLERS}, got {sampler!r}")
        if robust_iters < 1:
            raise ValueError(f"robust_iters must be >= 1, got {robust_iters}")
        if convergence_px <= 0:
            raise ValueError(f"convergence_px must be positive, got {convergence_px}")
        if search_resolution_multiplier <= 0:
            raise ValueError(
                f"search_resolution_multiplier must be positive, "
                f"got {search_resolution_multiplier}"
            )
        if search_strategy not in _SEARCH_STRATEGIES:
            raise ValueError(
                f"search_strategy must be one of {_SEARCH_STRATEGIES}, "
                f"got {search_strategy!r}"
            )

        self.min_views = min_views
        self.max_iters = max_iters
        self.search = search
        self.max_shift_px = max_shift_px
        self.min_relative_zncc = min_relative_zncc
        self.min_grazing_cos = min_grazing_cos
        self.resolution = resolution
        self.window = window
        self.window_sigma = window_sigma
        self.sampler = sampler
        self.robust_iters = robust_iters
        self.convergence_px = convergence_px
        self.search_resolution_multiplier = search_resolution_multiplier
        self.search_strategy = search_strategy

    def apply(self, recon: SfmrReconstruction) -> SfmrReconstruction:
        from .._patch_compaction import (
            compact_to_embedded_patches,
            image_file_hashes_from_images,
        )

        images = load_workspace_images(recon)

        # The reconstruction is embedded_patches (enforced by apply_transforms),
        # so the patch frame is already stored — read it back as the cloud rather
        # than rebuilding it. With view_sets=None the localizer runs over each
        # point's full track, seeding each view at the point's own projection,
        # and drops the views that won't co-register in-loop.
        cloud = recon.patches
        if cloud is None:
            raise ValueError(
                "reconstruction has no patch frames to localize keypoints over; "
                "expected embedded_patches (run `sfm xform --to-embedded-patches` "
                "first)"
            )
        print(
            f"  Localizing keypoints for {recon.point_count} points "
            f"across {recon.image_count} images..."
        )

        localizations = cloud.localize_keypoints(
            recon,
            images,
            view_sets=None,
            max_iters=self.max_iters,
            search=self.search,
            max_shift_px=self.max_shift_px,
            min_relative_zncc=self.min_relative_zncc,
            min_grazing_cos=self.min_grazing_cos,
            resolution=self.resolution,
            window=self.window,
            window_sigma=self.window_sigma,
            sampler=self.sampler,
            robust_iters=self.robust_iters,
            convergence_px=self.convergence_px,
            search_resolution_multiplier=self.search_resolution_multiplier,
            search_strategy=self.search_strategy,
        )

        # An embedded_patches recon already stores its per-image hashes; the
        # recompute-from-images fallback is defensive only.
        hashes = recon.image_file_hashes
        if hashes is None:
            hashes = image_file_hashes_from_images(recon)

        # The structural write-back: renumber the surviving points, rebuild
        # keypoints_xy + all three track arrays + the culled patch frames, and
        # carry over positions/colors/errors/normals. Bitmaps are dropped
        # (patch_bitmaps=None): the localizer renders none, and any stored ones
        # are stale after the keypoints move and views drop.
        out = compact_to_embedded_patches(
            recon,
            cloud,
            localizations,
            hashes,
            patch_bitmaps=None,
            min_views=self.min_views,
        )

        self._print_summary(recon, out)
        return out

    def _print_summary(
        self, before: SfmrReconstruction, after: SfmrReconstruction
    ) -> None:
        """Two-line ``xform``-style summary of the structural effect."""
        obs_before = len(before.track_point_indexes)
        obs_after = len(after.track_point_indexes)
        print(
            f"  Localized keypoints: {before.point_count} -> {after.point_count} "
            f"points, {obs_before} -> {obs_after} observations"
        )
        if after.point_count:
            mean_views = obs_after / after.point_count
            print(f"  Kept {mean_views:.1f} views per surviving point (mean)")

    def description(self) -> str:
        # This string is also reused as the operation name in the precondition
        # error.
        return (
            f"Localize keypoints (search={self.search}, "
            f"strategy={self.search_strategy}, min_views={self.min_views})"
        )
