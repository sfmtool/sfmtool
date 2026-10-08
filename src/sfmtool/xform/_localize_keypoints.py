# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Cross-view keypoint localization (search) as an ``sfm xform`` operation.

Unlike ``RefineKeypointsTransform`` (a pure in-place modifier), this op is
**structural**: ``PatchCloud.localize_keypoints`` congeals each point's
per-view keypoints by a discrete cross-view search and **drops views that
won't co-register** (drift too far, leave the frame, graze the patch plane, pin
no 2D position of their own, or stop agreeing with the leave-one-out
consensus). After a ``min_views`` cull the
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

See ``specs/cli/reconstruction/xform/localize-keypoints-command.md`` and
``specs/core/patch/patch-keypoint-localization.md``.
"""

from .._sfmtool.reconstruction import SfmrReconstruction
from ._images import load_workspace_images
from ._patch_params import _SAMPLERS, _WINDOWS

_SEARCH_STRATEGIES = ("exhaustive", "plus_descent")


def _check(ok, message):
    """A validator raising ``ValueError(message.format(v))`` unless ``ok(v)``."""

    def validate(value):
        if not ok(value):
            raise ValueError(message.format(value))

    return validate


# The PatchCloud.localize_keypoints keyword arguments this op accepts, each with
# the range check run on a value the caller gives. A key the caller does not give
# is not passed on, so the binding's own default applies.
_OPTION_VALIDATORS = {
    "max_iters": _check(lambda v: v >= 1, "max_iters must be >= 1, got {}"),
    "search": _check(lambda v: v > 0, "search must be positive, got {}"),
    "max_shift_px": _check(lambda v: v > 0, "max_shift_px must be positive, got {}"),
    "min_relative_zncc": _check(
        lambda v: 0 <= v <= 1, "min_relative_zncc must be in [0, 1], got {}"
    ),
    "min_absolute_zncc": _check(
        lambda v: 0 <= v <= 1, "min_absolute_zncc must be in [0, 1], got {}"
    ),
    "max_member_zncc_self_similarity_radius": _check(
        lambda v: v >= 0, "max_member_zncc_self_similarity_radius must be >= 0, got {}"
    ),
    "min_grazing_cos": _check(
        lambda v: 0 <= v <= 1, "min_grazing_cos must be in [0, 1], got {}"
    ),
    "resolution": _check(lambda v: v >= 2, "resolution must be >= 2, got {}"),
    "window": _check(
        lambda v: v in _WINDOWS, f"window must be one of {_WINDOWS}, got {{!r}}"
    ),
    "window_sigma": _check(lambda v: v > 0, "window_sigma must be positive, got {}"),
    "sampler": _check(
        lambda v: v in _SAMPLERS, f"sampler must be one of {_SAMPLERS}, got {{!r}}"
    ),
    "robust_iters": _check(lambda v: v >= 1, "robust_iters must be >= 1, got {}"),
    "convergence_px": _check(
        lambda v: v > 0, "convergence_px must be positive, got {}"
    ),
    "search_resolution_multiplier": _check(
        lambda v: v > 0, "search_resolution_multiplier must be positive, got {}"
    ),
    "search_strategy": _check(
        lambda v: v in _SEARCH_STRATEGIES,
        f"search_strategy must be one of {_SEARCH_STRATEGIES}, got {{!r}}",
    ),
    "basis_max_views": _check(lambda v: v >= 0, "basis_max_views must be >= 0, got {}"),
}


class LocalizeKeypointsTransform:
    """Localize per-observation 2D keypoints by cross-view search (structural).

    ``options`` are ``PatchCloud.localize_keypoints`` keyword arguments; only
    the ones given are forwarded, so every other search setting is the binding's
    own default. ``min_views`` (default 2) is the compaction cull threshold.
    Localization runs over each point's full track (``view_sets=None``) and the
    write-back rebuilds the track arrays from the kept views via
    ``compact_to_embedded_patches`` — views and points can be dropped, and the
    survivors are renumbered densely.

    Requires an ``embedded_patches`` reconstruction (enforced by
    ``apply_transforms``): the localizer searches over the stored per-point
    patch frame (``recon.patches``), which only that source carries, seeding
    each view at the point's own projection. Convert first with
    ``--to-embedded-patches``.

    See ``specs/cli/reconstruction/xform/localize-keypoints-command.md`` for the
    keys and their defaults.
    """

    # Precondition checked per-step by `apply_transforms` (see `_apply.py`).
    required_feature_source = "embedded_patches"

    def __init__(self, *, min_views: int = 2, **options):
        if min_views < 1:
            raise ValueError(f"min_views must be >= 1, got {min_views}")
        for key, value in options.items():
            validate = _OPTION_VALIDATORS.get(key)
            if validate is None:
                raise TypeError(
                    f"LocalizeKeypointsTransform got an unexpected option {key!r}"
                )
            validate(value)

        self.min_views = min_views
        # Only the options the caller gave; the binding supplies the rest.
        self.options = dict(options)

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
            recon, images, view_sets=None, **self.options
        )

        # An embedded_patches recon already stores its per-image hashes; the
        # recompute-from-images fallback is defensive only.
        hashes = recon.image_file_hashes
        if hashes is None:
            hashes = image_file_hashes_from_images(recon)

        # The structural write-back: renumber the surviving points, rebuild
        # keypoints_xy + all three track arrays + the culled patch frames, carry
        # over positions/colors/errors, and re-derive normals from the frames.
        # Bitmaps are dropped (patch_bitmaps=None): the localizer renders none,
        # and any stored ones are stale after the keypoints move and views drop.
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
        # error. It names min_views and the options given, not the binding
        # defaults behind the rest.
        settings = ", ".join(
            f"{key}={value}"
            for key, value in {"min_views": self.min_views, **self.options}.items()
        )
        return f"Localize keypoints ({settings})"
