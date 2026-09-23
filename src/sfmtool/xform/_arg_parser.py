# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Order-preserving argument parsing for the ``sfm xform`` pipeline.

``xform`` is an ordered pipeline of repeatable, interleaved heterogeneous
options (``--rotate … --scale … --rotate …``). Click's ``kwargs`` collapses
each option into a per-option tuple that loses the cross-option ordering the
pipeline depends on, so the command walks ``sys.argv`` by hand here to build
the ordered list of transforms. The Click ``@option`` decorators on the
command itself still provide ``--help``, completion, and unknown-option
rejection; this module is the complementary ordered parser.
"""

import re
from collections.abc import Callable
from pathlib import Path

import click
import numpy as np

from .._sfmtool.reconstruction import RangeExpr
from . import (
    AddPatchBitmapsTransform,
    AddThumbnailsTransform,
    AlignToInputTransform,
    AlignToTransform,
    BundleAdjustTransform,
    ClassifyPointsAtInfinityTransform,
    DropPatchBitmapsTransform,
    DropThumbnailsTransform,
    ExcludeGlobFilter,
    ExcludeRangeFilter,
    FilterByLocalizabilityTransform,
    FilterByPatchSizeTransform,
    FilterByReprojectionErrorTransform,
    FindPointsAtInfinityTransform,
    IncludeGlobFilter,
    IncludeRangeFilter,
    LocalizeKeypointsTransform,
    MinimalTransform,
    RefineKeypointsTransform,
    RefineNormalsTransform,
    RemoveIsolatedPointsFilter,
    RemoveLargeFeaturesFilter,
    RemoveNarrowTracksFilter,
    RemoveShortTracksFilter,
    RotateTransform,
    ScaleByMeasurementsTransform,
    ScaleTransform,
    SelectByDistributionFilter,
    SwitchCameraModelTransform,
    ToEmbeddedPatchesTransform,
    TranslateTransform,
)


def parse_angle(angle_str: str) -> float:
    """Parse angle string with unit suffix.

    Args:
        angle_str: Angle string like "90deg" or "1.57rad"

    Returns:
        Angle in radians
    """
    match = re.match(r"^([+-]?[\d.]+)(deg|degrees|rad|radians)$", angle_str.strip())
    if not match:
        raise ValueError(
            f"Invalid angle format: '{angle_str}'. Expected format: <number><unit> "
            f"where unit is 'deg', 'degrees', 'rad', or 'radians'"
        )

    value_str, unit = match.groups()
    value = float(value_str)

    if unit in ("deg", "degrees"):
        return np.radians(value)
    elif unit in ("rad", "radians"):
        return value
    else:
        raise ValueError(f"Unrecognized angle unit: {unit}")


def _parse_bool(value: str) -> bool:
    """Parse a ``key=value`` boolean flag (``true``/``false``/``1``/``0``/…)."""
    v = value.strip().lower()
    if v in ("true", "1", "yes", "on"):
        return True
    if v in ("false", "0", "no", "off"):
        return False
    raise ValueError(value)


def _parse_opt_int(value: str) -> int | None:
    """Parse an optional integer (``none``/``off`` → ``None``, else ``int``)."""
    v = value.strip().lower()
    if v in ("none", "off"):
        return None
    return int(value)


def _parse_kv_params(
    param: str, option_name: str, keys: dict[str, Callable[[str], object]]
) -> dict:
    """Parse one optional comma-separated ``key=value`` option's overrides."""
    kwargs: dict = {}
    for token in param.split(","):
        token = token.strip()
        if not token:
            # Tolerate empty segments, including a bare option or trailing comma.
            continue
        if "=" not in token:
            raise click.UsageError(
                f"Invalid {option_name} token '{token}': expected key=value"
            )
        key, value = token.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            raise click.UsageError(f"Invalid {option_name} token '{token}': empty key")
        if key not in keys:
            raise click.UsageError(
                f"Unknown {option_name} key '{key}' "
                f"(expected one of: {', '.join(sorted(keys))})"
            )
        if key in kwargs:
            raise click.UsageError(f"Duplicate {option_name} key '{key}'")
        caster = keys[key]
        if caster is str:
            kwargs[key] = value
        else:
            try:
                kwargs[key] = caster(value)
            except ValueError:
                raise click.UsageError(
                    f"Invalid value for {option_name} key '{key}': "
                    f"'{value}' is not a valid {caster.__name__}"
                )
    return kwargs


def _take_arg(args: list[str], i: int, option: str) -> tuple[str, int]:
    """Return a required value and its index, preserving the missing-value error."""
    if i + 1 >= len(args):
        raise click.UsageError(f"{option} requires an argument")
    i += 1
    return args[i], i


# Each --refine-normals key maps to a caster for its value; the
# RefineNormalsTransform constructor owns range/enum validation. Keys mirror the
# PatchCloud.refine_normals binding parameters. (Frame-sizing / cloud-building
# knobs — extent, extent_value, initial_normals — and the save_patches opt-in
# live on `--to-embedded-patches`, the step that actually builds the patch frame;
# refine-normals reuses the stored frame, so it doesn't take them.)
_REFINE_NORMALS_KEYS: dict[str, Callable[[str], object]] = {
    "bitmaps": _parse_bool,
    "angular_range_deg": float,
    "init_steps": int,
    "refine_levels": int,
    "resolution": int,
    "objective": str,
    "robust_iters": int,
    "search_robust_iters": _parse_opt_int,
    "window": str,
    "window_sigma": float,
    "sampler": str,
    "min_valid_fraction": float,
    "min_views": int,
    "cache": str,
    "cache_supersample": float,
    "quality": str,
    "confidence": _parse_bool,
}


def parse_refine_normals_params(param: str) -> RefineNormalsTransform:
    """Parse a ``--refine-normals`` comma-separated ``key=value`` string.

    An empty string runs the v1 defaults. Unknown keys, malformed tokens (no
    ``=`` or an empty key), and unparseable values raise ``click.UsageError``;
    range/enum validation is the transform constructor's job (its ``ValueError``
    is re-raised as ``UsageError`` by the caller).
    """
    return RefineNormalsTransform(
        **_parse_kv_params(param, "--refine-normals", _REFINE_NORMALS_KEYS)
    )


# Each --refine-keypoints key maps to a caster for its value; the
# RefineKeypointsTransform constructor owns range/enum validation. Keys mirror
# the PatchCloud.refine_keypoints binding parameters. (Frame-sizing knobs live on
# `--to-embedded-patches`, the step that builds the patch frame; refine-keypoints
# reuses the stored frame and the stored per-observation seeds.)
_REFINE_KEYPOINTS_KEYS: dict[str, Callable[[str], object]] = {
    "bitmaps": _parse_bool,
    "resolution": int,
    "window": str,
    "window_sigma": float,
    "sampler": str,
    "robust_iters": int,
    "max_outer_sweeps": int,
    "outer_convergence_px": float,
    "max_gn_steps": int,
    "convergence_px": float,
    "max_offset_px": float,
    "consensus_refresh": str,
}


def parse_refine_keypoints_params(param: str) -> RefineKeypointsTransform:
    """Parse a ``--refine-keypoints`` comma-separated ``key=value`` string.

    An empty string runs the binding defaults. Unknown keys, malformed tokens
    (no ``=`` or an empty key), and unparseable values raise
    ``click.UsageError``; range/enum validation is the transform constructor's
    job (its ``ValueError`` is re-raised as ``UsageError`` by the caller).
    """
    return RefineKeypointsTransform(
        **_parse_kv_params(param, "--refine-keypoints", _REFINE_KEYPOINTS_KEYS)
    )


# The two --add-patch-bitmaps keys. The other sub-pixel parameters tune a solve
# this step does not run, so they are not offered.
_ADD_PATCH_BITMAPS_KEYS: dict[str, Callable[[str], object]] = {
    "resolution": int,
    "sampler": str,
}


def parse_add_patch_bitmaps_params(param: str) -> AddPatchBitmapsTransform:
    """Parse an ``--add-patch-bitmaps`` comma-separated ``key=value`` string.

    An empty string renders at the defaults. Unknown keys, malformed tokens and
    unparseable values raise ``click.UsageError``; range and enum validation is
    the transform constructor's.
    """
    return AddPatchBitmapsTransform(
        **_parse_kv_params(param, "--add-patch-bitmaps", _ADD_PATCH_BITMAPS_KEYS)
    )


# The one --minimal key. ``wspath`` is the command-line spelling of the stated
# workspace path; Rust, Python and the wire all call it ``workspace_path``, and
# the CLI abbreviates it because it sits inside a comma-separated value.
_MINIMAL_KEYS: dict[str, Callable[[str], object]] = {
    "wspath": str,
}


def parse_minimal_params(param: str) -> MinimalTransform:
    """Parse a ``--minimal`` comma-separated ``key=value`` string.

    An empty string leaves the save measuring the workspace path, as every other
    save does. Unknown keys, malformed tokens and empty keys raise
    ``click.UsageError``.
    """
    kwargs = _parse_kv_params(param, "--minimal", _MINIMAL_KEYS)
    return MinimalTransform(workspace_path=kwargs.get("wspath"))


# Each --localize-keypoints key maps to a caster for its value; the
# LocalizeKeypointsTransform constructor owns range/enum validation. Keys mirror
# the PatchCloud.localize_keypoints binding parameters, plus the compaction cull
# `min_views`. (There is no `bitmaps` key: the localizer renders none, and the
# structural rebuild drops any stored ones as stale — run `--refine-keypoints`
# afterward to regenerate them, since that op renders bitmaps by default.)
_LOCALIZE_KEYPOINTS_KEYS: dict[str, Callable[[str], object]] = {
    "min_views": int,
    "max_iters": int,
    "search": float,
    "max_shift_px": float,
    "min_relative_zncc": float,
    "min_absolute_zncc": float,
    "max_member_keypoint_uncertainty": float,
    "min_grazing_cos": float,
    "resolution": int,
    "window": str,
    "window_sigma": float,
    "sampler": str,
    "robust_iters": int,
    "convergence_px": float,
    "search_resolution_multiplier": float,
    "search_strategy": str,
    "basis_max_views": int,
}


def parse_localize_keypoints_params(param: str) -> LocalizeKeypointsTransform:
    """Parse a ``--localize-keypoints`` comma-separated ``key=value`` string.

    An empty string runs the binding defaults (plus ``min_views=2``). Unknown
    keys, malformed tokens (no ``=`` or an empty key), and unparseable values
    raise ``click.UsageError``; range/enum validation is the transform
    constructor's job (its ``ValueError`` is re-raised as ``UsageError`` by the
    caller).
    """
    return LocalizeKeypointsTransform(
        **_parse_kv_params(param, "--localize-keypoints", _LOCALIZE_KEYPOINTS_KEYS)
    )


# Each --to-embedded-patches key maps to a caster; the transform constructor owns
# range/enum validation. Keys mirror the ToEmbeddedPatchesTransform parameters.
_TO_EMBEDDED_PATCHES_KEYS: dict[str, Callable[[str], object]] = {
    "normal": str,
    "k_neighbors": int,
    "extent": str,
    "extent_value": float,
    "feature_reduce": str,
    "pixel_reduce": str,
}


def parse_to_embedded_patches_params(param: str) -> ToEmbeddedPatchesTransform:
    """Parse a ``--to-embedded-patches`` comma-separated ``key=value`` string.

    An empty string runs the defaults. Unknown keys, malformed tokens, and
    unparseable values raise ``click.UsageError``; range/enum validation is the
    transform constructor's job.
    """
    return ToEmbeddedPatchesTransform(
        **_parse_kv_params(param, "--to-embedded-patches", _TO_EMBEDDED_PATCHES_KEYS)
    )


def auto_output_path(input_path: Path, suffix: str = "transformed") -> Path:
    """Generate an output path of the form {stem}-{suffix}[-N].sfmr next to the input.

    Picks ``{stem}-{suffix}.sfmr`` if available, otherwise the smallest counter
    starting at 2: ``{stem}-{suffix}-2.sfmr``, ``-3.sfmr``, ... ``suffix`` defaults
    to ``transformed`` (``sfm xform``); ``sfm embed-patches`` passes ``embedded``.
    """
    base = input_path.with_name(f"{input_path.stem}-{suffix}.sfmr")
    if not base.exists():
        return base
    counter = 2
    while True:
        candidate = input_path.with_name(f"{input_path.stem}-{suffix}-{counter}.sfmr")
        if not candidate.exists():
            return candidate
        counter += 1


def parse_transform_args(args: list[str], max_features: int | None = None) -> list:
    """Parse command-line arguments to extract transforms in order.

    ``max_features`` is a global value option (not an ordered transform); it is
    obtained reliably from the Click ``kwargs`` and shared by every
    ``--find-points-at-infinity`` operation in the chain.
    """
    transforms = []
    i = 0

    while i < len(args):
        arg = args[i]

        if arg == "--rotate":
            param, i = _take_arg(args, i, arg)

            parts = param.split(",")
            if len(parts) != 4:
                raise click.UsageError(
                    f"--rotate expects 4 comma-separated values (axisX,axisY,axisZ,angle), got: {param}"
                )

            try:
                axis_x = float(parts[0])
                axis_y = float(parts[1])
                axis_z = float(parts[2])
                angle_rad = parse_angle(parts[3])
            except ValueError as e:
                raise click.UsageError(f"Invalid --rotate parameter '{param}': {e}")

            axis = np.array([axis_x, axis_y, axis_z])
            transforms.append(RotateTransform(axis, angle_rad))

        elif arg == "--translate":
            param, i = _take_arg(args, i, arg)

            parts = param.split(",")
            if len(parts) != 3:
                raise click.UsageError(
                    f"--translate expects 3 comma-separated values (X,Y,Z), got: {param}"
                )

            try:
                x = float(parts[0])
                y = float(parts[1])
                z = float(parts[2])
            except ValueError as e:
                raise click.UsageError(f"Invalid --translate parameter '{param}': {e}")

            translation = np.array([x, y, z])
            transforms.append(TranslateTransform(translation))

        elif arg == "--scale":
            param, i = _take_arg(args, i, arg)

            try:
                scale_factor = float(param)
            except ValueError as e:
                raise click.UsageError(f"Invalid --scale parameter '{param}': {e}")

            transforms.append(ScaleTransform(scale_factor))

        elif arg == "--remove-short-tracks":
            param, i = _take_arg(args, i, arg)

            try:
                max_size = int(param)
            except ValueError as e:
                raise click.UsageError(
                    f"Invalid --remove-short-tracks parameter '{param}': {e}"
                )

            transforms.append(RemoveShortTracksFilter(max_size))

        elif arg == "--bundle-adjust":
            transforms.append(BundleAdjustTransform())

        elif arg == "--drop-thumbnails":
            transforms.append(DropThumbnailsTransform())

        elif arg == "--drop-patch-bitmaps":
            transforms.append(DropPatchBitmapsTransform())

        elif arg == "--add-thumbnails":
            transforms.append(AddThumbnailsTransform())

        elif arg == "--minimal" or arg.startswith("--minimal="):
            # Optional value, same tokenization as --add-patch-bitmaps.
            if arg.startswith("--minimal="):
                param = arg[len("--minimal=") :]
            elif i + 1 < len(args) and not args[i + 1].startswith("-"):
                i += 1
                param = args[i]
            else:
                param = ""

            try:
                transforms.append(parse_minimal_params(param))
            except ValueError as e:
                raise click.UsageError(f"Invalid --minimal parameter: {e}")

        elif arg == "--add-patch-bitmaps" or arg.startswith("--add-patch-bitmaps="):
            # Optional value, same tokenization as --refine-normals.
            if arg.startswith("--add-patch-bitmaps="):
                param = arg[len("--add-patch-bitmaps=") :]
            elif i + 1 < len(args) and not args[i + 1].startswith("-"):
                i += 1
                param = args[i]
            else:
                param = ""

            try:
                transforms.append(parse_add_patch_bitmaps_params(param))
            except ValueError as e:
                raise click.UsageError(f"Invalid --add-patch-bitmaps parameter: {e}")

        elif arg == "--refine-normals" or arg.startswith("--refine-normals="):
            # Optional value. Mirror Click's optional-value tokenization (the
            # command declares it is_flag=False, flag_value=""): a value joined
            # with ``=`` is taken verbatim; otherwise the next token is the value
            # iff it isn't another option (so ``--refine-normals --bundle-adjust``
            # and a trailing ``--refine-normals`` both run the defaults).
            if arg.startswith("--refine-normals="):
                param = arg[len("--refine-normals=") :]
            elif i + 1 < len(args) and not args[i + 1].startswith("-"):
                i += 1
                param = args[i]
            else:
                param = ""

            try:
                transforms.append(parse_refine_normals_params(param))
            except ValueError as e:
                raise click.UsageError(f"Invalid --refine-normals parameter: {e}")

        elif arg == "--refine-keypoints" or arg.startswith("--refine-keypoints="):
            # Optional value, same tokenization as --refine-normals.
            if arg.startswith("--refine-keypoints="):
                param = arg[len("--refine-keypoints=") :]
            elif i + 1 < len(args) and not args[i + 1].startswith("-"):
                i += 1
                param = args[i]
            else:
                param = ""

            try:
                transforms.append(parse_refine_keypoints_params(param))
            except ValueError as e:
                raise click.UsageError(f"Invalid --refine-keypoints parameter: {e}")

        elif arg == "--localize-keypoints" or arg.startswith("--localize-keypoints="):
            # Optional value, same tokenization as --refine-normals.
            if arg.startswith("--localize-keypoints="):
                param = arg[len("--localize-keypoints=") :]
            elif i + 1 < len(args) and not args[i + 1].startswith("-"):
                i += 1
                param = args[i]
            else:
                param = ""

            try:
                transforms.append(parse_localize_keypoints_params(param))
            except ValueError as e:
                raise click.UsageError(f"Invalid --localize-keypoints parameter: {e}")

        elif arg == "--to-embedded-patches" or arg.startswith("--to-embedded-patches="):
            # Optional value, same tokenization as --refine-normals.
            if arg.startswith("--to-embedded-patches="):
                param = arg[len("--to-embedded-patches=") :]
            elif i + 1 < len(args) and not args[i + 1].startswith("-"):
                i += 1
                param = args[i]
            else:
                param = ""

            try:
                transforms.append(parse_to_embedded_patches_params(param))
            except ValueError as e:
                raise click.UsageError(f"Invalid --to-embedded-patches parameter: {e}")

        elif arg == "--remove-narrow-tracks":
            param, i = _take_arg(args, i, arg)

            try:
                min_angle_rad = parse_angle(param)
            except ValueError as e:
                raise click.UsageError(
                    f"Invalid --remove-narrow-tracks parameter '{param}': {e}"
                )

            transforms.append(RemoveNarrowTracksFilter(min_angle_rad))

        elif arg == "--remove-isolated":
            param, i = _take_arg(args, i, arg)

            parts = param.split(",")
            if len(parts) != 2:
                raise click.UsageError(
                    f"--remove-isolated expects 2 comma-separated values (factor,value_spec), got: {param}"
                )

            try:
                factor = float(parts[0])
            except ValueError as e:
                raise click.UsageError(
                    f"Invalid factor in --remove-isolated '{param}': {e}"
                )

            value_spec = parts[1]
            transforms.append(RemoveIsolatedPointsFilter(factor, value_spec))

        elif arg == "--align-to":
            param, i = _take_arg(args, i, arg)

            reference_path = Path(param)
            transforms.append(AlignToTransform(reference_path))

        elif arg == "--align-to-input":
            transforms.append(AlignToInputTransform())

        elif arg == "--remove-large-features":
            param, i = _take_arg(args, i, arg)

            try:
                max_size = float(param)
            except ValueError as e:
                raise click.UsageError(
                    f"Invalid --remove-large-features parameter '{param}': {e}"
                )

            transforms.append(RemoveLargeFeaturesFilter(max_size))

        elif arg == "--filter-by-reprojection-error":
            param, i = _take_arg(args, i, arg)

            try:
                threshold = float(param)
            except ValueError as e:
                raise click.UsageError(
                    f"Invalid --filter-by-reprojection-error parameter '{param}': {e}"
                )

            transforms.append(FilterByReprojectionErrorTransform(threshold))

        elif arg == "--filter-by-keypoint-uncertainty":
            param, i = _take_arg(args, i, arg)

            try:
                threshold = float(param)
            except ValueError as e:
                raise click.UsageError(
                    f"Invalid --filter-by-keypoint-uncertainty parameter '{param}': {e}"
                )

            try:
                transforms.append(FilterByLocalizabilityTransform(threshold))
            except ValueError as e:
                raise click.UsageError(
                    f"Invalid --filter-by-keypoint-uncertainty parameter '{param}': {e}"
                )

        elif arg == "--filter-by-patch-size":
            param, i = _take_arg(args, i, arg)

            try:
                multiplier = float(param)
            except ValueError as e:
                raise click.UsageError(
                    f"Invalid --filter-by-patch-size parameter '{param}': {e}"
                )

            try:
                transforms.append(FilterByPatchSizeTransform(multiplier))
            except ValueError as e:
                raise click.UsageError(
                    f"Invalid --filter-by-patch-size parameter '{param}': {e}"
                )

        elif arg == "--include-range":
            param, i = _take_arg(args, i, arg)

            try:
                range_expr = RangeExpr(param)
            except ValueError as e:
                raise click.UsageError(
                    f"Invalid --include-range parameter '{param}': {e}"
                )

            transforms.append(IncludeRangeFilter(range_expr))

        elif arg == "--exclude-range":
            param, i = _take_arg(args, i, arg)

            try:
                range_expr = RangeExpr(param)
            except ValueError as e:
                raise click.UsageError(
                    f"Invalid --exclude-range parameter '{param}': {e}"
                )

            transforms.append(ExcludeRangeFilter(range_expr))

        elif arg == "--scale-by-measurements":
            param, i = _take_arg(args, i, arg)

            measurements_path = Path(param)
            if not measurements_path.exists():
                raise click.UsageError(
                    f"Measurements file not found: {measurements_path}"
                )

            transforms.append(ScaleByMeasurementsTransform(measurements_path))

        elif arg == "--include-glob":
            param, i = _take_arg(args, i, arg)
            transforms.append(IncludeGlobFilter(param))

        elif arg == "--exclude-glob":
            param, i = _take_arg(args, i, arg)
            transforms.append(ExcludeGlobFilter(param))

        elif arg == "--include-by-distribution":
            param, i = _take_arg(args, i, arg)

            parts = param.split(",")
            try:
                count = int(parts[0])
            except ValueError as e:
                raise click.UsageError(
                    f"Invalid --include-by-distribution parameter '{param}': {e}"
                )
            if count < 2:
                raise click.UsageError(
                    f"--include-by-distribution COUNT must be >= 2, got {count}"
                )
            verbose = False
            for modifier in parts[1:]:
                if modifier.strip() == "verbose":
                    verbose = True
                else:
                    raise click.UsageError(
                        f"Unknown --include-by-distribution modifier '{modifier}' "
                        "(expected 'verbose')"
                    )

            transforms.append(SelectByDistributionFilter(count, verbose=verbose))

        elif arg == "--camera-model":
            param, i = _take_arg(args, i, arg)

            try:
                transforms.append(SwitchCameraModelTransform(param))
            except ValueError as e:
                raise click.UsageError(
                    f"Invalid --camera-model parameter '{param}': {e}"
                )

        elif arg == "--find-points-at-infinity":
            param, i = _take_arg(args, i, arg)

            parts = param.split(",")
            if not 1 <= len(parts) <= 4:
                raise click.UsageError(
                    "--find-points-at-infinity expects "
                    "eps_deg[,desc_thresh[,min_views[,noise_floor_px]]], "
                    f"got: {param}"
                )

            try:
                eps_deg = float(parts[0])
                desc_thresh = float(parts[1]) if len(parts) > 1 else 200.0
                min_views = int(parts[2]) if len(parts) > 2 else 2
                noise_floor_px = float(parts[3]) if len(parts) > 3 else 1.0
            except ValueError as e:
                raise click.UsageError(
                    f"Invalid --find-points-at-infinity parameter '{param}': {e}"
                )

            try:
                transforms.append(
                    FindPointsAtInfinityTransform(
                        eps_deg,
                        desc_thresh,
                        min_views,
                        max_features=max_features,
                        noise_floor_px=noise_floor_px,
                    )
                )
            except ValueError as e:
                raise click.UsageError(
                    f"Invalid --find-points-at-infinity parameter '{param}': {e}"
                )

        elif arg == "--classify-points-at-infinity":
            param, i = _take_arg(args, i, arg)

            try:
                noise_floor_px = float(param)
            except ValueError as e:
                raise click.UsageError(
                    f"Invalid --classify-points-at-infinity parameter '{param}': {e}"
                )

            transforms.append(ClassifyPointsAtInfinityTransform(noise_floor_px))

        elif arg == "--max-features":
            # A global value option, not an ordered transform: its value is
            # obtained reliably via Click kwargs, so just step over its token
            # here to keep it out of the ordered transform list.
            if i + 1 < len(args):
                i += 1

        i += 1

    # An --add-* step after --minimal restores part of what the shorthand
    # dropped; it says so when it runs, so the combination reads as intended.
    seen_minimal = False
    for transform in transforms:
        if isinstance(transform, MinimalTransform):
            seen_minimal = True
        elif seen_minimal and isinstance(
            transform, (AddThumbnailsTransform, AddPatchBitmapsTransform)
        ):
            transform.restores_minimal = True

    return transforms
