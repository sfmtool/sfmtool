# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Order-preserving argument parsing for the ``sfm xform`` pipeline.

``xform`` is an ordered pipeline of repeatable, interleaved heterogeneous
options (``--rotate … --scale … --rotate …``). Click's ``kwargs`` collapses
each option into a per-option tuple that loses the cross-option ordering the
pipeline depends on, so the command walks its own argument list by hand here to
build the ordered list of transforms. The Click ``@option`` decorators on the
command itself still provide ``--help``, completion, and unknown-option
rejection; this module is the complementary ordered parser.

The two readings must agree. The walk follows Click's token rules (``--opt
value`` and ``--opt=value`` for a value option, the next token as a required
value even when it begins with ``-``, ``--`` ending the options), refuses any
option it does not know, and ``check_against_click`` compares what it read with
what Click collected, so an option Click accepted is never dropped silently.
"""

import re
from collections.abc import Callable
from dataclasses import dataclass, field
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
    FilterByPatchSizeTransform,
    FilterByReprojectionErrorTransform,
    FilterByZnccSelfSimilarityRadiusTransform,
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
    "max_member_zncc_self_similarity_radius": float,
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


def _parse_camera_list(value: str) -> list[int]:
    """Parse ``0+1+3`` into camera indexes."""
    return [int(v) for v in value.split("+") if v.strip()]


# Each --bundle-adjust key maps to a caster; the transform and the adjustment
# own the range checks.
_BUNDLE_ADJUST_KEYS: dict[str, Callable[[str], object]] = {
    "cameras": _parse_camera_list,
}


def parse_bundle_adjust_params(param: str) -> BundleAdjustTransform:
    """Parse a ``--bundle-adjust`` comma-separated ``key=value`` string.

    An empty string is the bare option. ``cameras=0+1`` releases the lens of
    those cameras only and holds the rest.
    """
    kwargs = _parse_kv_params(param, "--bundle-adjust", _BUNDLE_ADJUST_KEYS)
    return BundleAdjustTransform(cameras=kwargs.get("cameras"))


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


_CAMERA_MODEL_KEYS: dict[str, Callable[[str], object]] = {
    "coeffs": int,
    "fit_to": float,
    "spline_domain": float,
    "cameras": _parse_camera_list,
}


def parse_camera_model_params(param: str) -> SwitchCameraModelTransform:
    """Parse ``MODEL[,coeffs=N,fit_to=DEG,spline_domain=DEG,cameras=0+1]``."""
    camera_model, _, rest = param.partition(",")
    if not camera_model.strip():
        raise click.UsageError("--camera-model needs a model name first")
    kwargs = _parse_kv_params(rest, "--camera-model", _CAMERA_MODEL_KEYS)
    try:
        return SwitchCameraModelTransform(
            camera_model,
            coeff_count=kwargs.get("coeffs"),
            theta_fit_deg=kwargs.get("fit_to"),
            spline_domain_deg=kwargs.get("spline_domain"),
            cameras=kwargs.get("cameras"),
        )
    except ValueError as e:
        raise click.UsageError(f"Invalid --camera-model parameter '{param}': {e}")


def _parse_scalar(
    param: str,
    option: str,
    caster: Callable,
    constructor: Callable,
    *,
    catch_constructor: bool = False,
):
    """Parse a single required value, retaining the option's diagnostic."""
    try:
        value = caster(param)
    except ValueError as e:
        raise click.UsageError(f"Invalid {option} parameter '{param}': {e}")
    if catch_constructor:
        try:
            return constructor(value)
        except ValueError as e:
            raise click.UsageError(f"Invalid {option} parameter '{param}': {e}")
    return constructor(value)


def _parse_optional(param: str, option: str, parser: Callable):
    try:
        return parser(param)
    except ValueError as e:
        raise click.UsageError(f"Invalid {option} parameter: {e}")


def _parse_rotate(param: str, _max_features: int | None):
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
    return RotateTransform(np.array([axis_x, axis_y, axis_z]), angle_rad)


def _parse_translate(param: str, _max_features: int | None):
    parts = param.split(",")
    if len(parts) != 3:
        raise click.UsageError(
            f"--translate expects 3 comma-separated values (X,Y,Z), got: {param}"
        )
    try:
        x, y, z = (float(part) for part in parts)
    except ValueError as e:
        raise click.UsageError(f"Invalid --translate parameter '{param}': {e}")
    return TranslateTransform(np.array([x, y, z]))


def _parse_remove_isolated(param: str, _max_features: int | None):
    parts = param.split(",")
    if len(parts) != 2:
        raise click.UsageError(
            f"--remove-isolated expects 2 comma-separated values (factor,value_spec), got: {param}"
        )
    try:
        factor = float(parts[0])
    except ValueError as e:
        raise click.UsageError(f"Invalid factor in --remove-isolated '{param}': {e}")
    return RemoveIsolatedPointsFilter(factor, parts[1])


def _parse_scale_by_measurements(param: str, _max_features: int | None):
    measurements_path = Path(param)
    if not measurements_path.exists():
        raise click.UsageError(f"Measurements file not found: {measurements_path}")
    return ScaleByMeasurementsTransform(measurements_path)


def _parse_include_by_distribution(param: str, _max_features: int | None):
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
    return SelectByDistributionFilter(count, verbose=verbose)


def _parse_find_points_at_infinity(param: str, max_features: int | None):
    parts = param.split(",")
    if not 1 <= len(parts) <= 4:
        raise click.UsageError(
            "--find-points-at-infinity expects "
            "eps_deg[,desc_thresh[,min_views[,sigma_px]]], "
            f"got: {param}"
        )
    try:
        eps_deg = float(parts[0])
        desc_thresh = float(parts[1]) if len(parts) > 1 else 200.0
        min_views = int(parts[2]) if len(parts) > 2 else 2
        sigma_px = float(parts[3]) if len(parts) > 3 else None
    except ValueError as e:
        raise click.UsageError(
            f"Invalid --find-points-at-infinity parameter '{param}': {e}"
        )
    try:
        return FindPointsAtInfinityTransform(
            eps_deg,
            desc_thresh,
            min_views,
            max_features=max_features,
            sigma_px=sigma_px,
        )
    except ValueError as e:
        raise click.UsageError(
            f"Invalid --find-points-at-infinity parameter '{param}': {e}"
        )


def _parse_classify_points_at_infinity(param: str, _max_features: int | None):
    """An optional ``sigma_px`` override; bare means the measured noise."""
    if not param.strip():
        return ClassifyPointsAtInfinityTransform()
    return _parse_scalar(
        param,
        "--classify-points-at-infinity",
        float,
        ClassifyPointsAtInfinityTransform,
        catch_constructor=True,
    )


# The rule says whether a value is absent ("none"), required ("required": joined
# with "=" or the next token, whatever it starts with), or optional ("optional":
# joined with "=" or the next token when that token is not an option).
# Builders keep each option's validation and error text near its registration.
_TRANSFORM_OPTIONS: dict[str, tuple[str, Callable[[str, int | None], object]]] = {
    "--rotate": ("required", _parse_rotate),
    "--translate": ("required", _parse_translate),
    "--scale": (
        "required",
        lambda p, _: _parse_scalar(p, "--scale", float, ScaleTransform),
    ),
    "--remove-short-tracks": (
        "required",
        lambda p, _: _parse_scalar(
            p, "--remove-short-tracks", int, RemoveShortTracksFilter
        ),
    ),
    "--bundle-adjust": (
        "optional",
        lambda p, _: _parse_optional(p, "--bundle-adjust", parse_bundle_adjust_params),
    ),
    "--drop-thumbnails": ("none", lambda _p, _: DropThumbnailsTransform()),
    "--drop-patch-bitmaps": ("none", lambda _p, _: DropPatchBitmapsTransform()),
    "--add-thumbnails": ("none", lambda _p, _: AddThumbnailsTransform()),
    "--minimal": (
        "optional",
        lambda p, _: _parse_optional(p, "--minimal", parse_minimal_params),
    ),
    "--add-patch-bitmaps": (
        "optional",
        lambda p, _: _parse_optional(
            p, "--add-patch-bitmaps", parse_add_patch_bitmaps_params
        ),
    ),
    "--refine-normals": (
        "optional",
        lambda p, _: _parse_optional(
            p, "--refine-normals", parse_refine_normals_params
        ),
    ),
    "--refine-keypoints": (
        "optional",
        lambda p, _: _parse_optional(
            p, "--refine-keypoints", parse_refine_keypoints_params
        ),
    ),
    "--localize-keypoints": (
        "optional",
        lambda p, _: _parse_optional(
            p, "--localize-keypoints", parse_localize_keypoints_params
        ),
    ),
    "--to-embedded-patches": (
        "optional",
        lambda p, _: _parse_optional(
            p, "--to-embedded-patches", parse_to_embedded_patches_params
        ),
    ),
    "--remove-narrow-tracks": (
        "required",
        lambda p, _: _parse_scalar(
            p, "--remove-narrow-tracks", parse_angle, RemoveNarrowTracksFilter
        ),
    ),
    "--remove-isolated": ("required", _parse_remove_isolated),
    "--align-to": ("required", lambda p, _: AlignToTransform(Path(p))),
    "--align-to-input": ("none", lambda _p, _: AlignToInputTransform()),
    "--remove-large-features": (
        "required",
        lambda p, _: _parse_scalar(
            p, "--remove-large-features", float, RemoveLargeFeaturesFilter
        ),
    ),
    "--filter-by-reprojection-error": (
        "required",
        lambda p, _: _parse_scalar(
            p,
            "--filter-by-reprojection-error",
            float,
            FilterByReprojectionErrorTransform,
        ),
    ),
    "--filter-by-zncc-self-similarity-radius": (
        "required",
        lambda p, _: _parse_scalar(
            p,
            "--filter-by-zncc-self-similarity-radius",
            float,
            FilterByZnccSelfSimilarityRadiusTransform,
            catch_constructor=True,
        ),
    ),
    "--filter-by-patch-size": (
        "required",
        lambda p, _: _parse_scalar(
            p,
            "--filter-by-patch-size",
            float,
            FilterByPatchSizeTransform,
            catch_constructor=True,
        ),
    ),
    "--include-range": (
        "required",
        lambda p, _: _parse_scalar(p, "--include-range", RangeExpr, IncludeRangeFilter),
    ),
    "--exclude-range": (
        "required",
        lambda p, _: _parse_scalar(p, "--exclude-range", RangeExpr, ExcludeRangeFilter),
    ),
    "--scale-by-measurements": ("required", _parse_scale_by_measurements),
    "--include-glob": ("required", lambda p, _: IncludeGlobFilter(p)),
    "--exclude-glob": ("required", lambda p, _: ExcludeGlobFilter(p)),
    "--include-by-distribution": ("required", _parse_include_by_distribution),
    "--camera-model": ("required", lambda p, _: parse_camera_model_params(p)),
    "--find-points-at-infinity": ("required", _parse_find_points_at_infinity),
    "--classify-points-at-infinity": ("optional", _parse_classify_points_at_infinity),
}


# Options of the command that are not ordered transforms, with their value rule.
# Click supplies their values through ``kwargs``; the walk only steps over them.
# ``--help``/``-h`` are eager and end the command before the walk runs, but are
# listed so the table covers every option the command declares.
_GLOBAL_OPTIONS: dict[str, str] = {
    "--max-features": "required",
    "--help": "none",
    "-h": "none",
}

# The key under which ``OrderedArgsCommand`` keeps the command's argument list.
_ARGS_META_KEY = "sfmtool.xform.args"


class OrderedArgsCommand(click.Command):
    """A Click command that keeps its own argument list for the ordered walk.

    Reading the list Click was handed, rather than ``sys.argv``, means the walk
    sees exactly the tokens Click parsed, wherever the positional arguments sit
    among the options and however the command was invoked.
    """

    def parse_args(self, ctx: click.Context, args: list[str]) -> list[str]:
        ctx.meta[_ARGS_META_KEY] = list(args)
        return super().parse_args(ctx, args)


def command_args(ctx: click.Context) -> list[str]:
    """Return the argument list an ``OrderedArgsCommand`` was handed."""
    return ctx.meta[_ARGS_META_KEY]


@dataclass
class ParsedXformArgs:
    """The result of one ordered walk over the ``xform`` arguments."""

    transforms: list = field(default_factory=list)
    # Tokens that are neither an option nor an option's value, in order.
    positionals: list[str] = field(default_factory=list)
    # Each transform option's raw values in command-line order ("" for an
    # optional-value option given bare, and for a value-free option).
    values: dict[str, list[str]] = field(default_factory=dict)


def _is_option_token(token: str) -> bool:
    """Whether Click reads ``token`` as an option rather than as a value."""
    return token.startswith("-") and len(token) > 1


def parse_xform_args(
    args: list[str], max_features: int | None = None
) -> ParsedXformArgs:
    """Walk the ``xform`` argument list, building the transforms in order.

    A required value is written ``--opt value`` or ``--opt=value``; the
    separated form takes the next token whatever it starts with, so
    ``--translate -1,2,3`` reads as Click reads it. An optional value is joined,
    or is the next token when that token is not an option. A value-free option
    given ``=value``, and any option this command does not declare, raise
    ``click.UsageError``. After ``--`` every token is positional.

    ``max_features`` is a global value option (not an ordered transform); it is
    obtained from the Click ``kwargs`` and shared by every
    ``--find-points-at-infinity`` operation in the chain.
    """
    parsed = ParsedXformArgs()
    options_ended = False
    i = 0
    while i < len(args):
        arg = args[i]
        if not options_ended and arg == "--":
            options_ended = True
            i += 1
            continue
        if options_ended or not _is_option_token(arg):
            parsed.positionals.append(arg)
            i += 1
            continue

        option, separator, joined_value = arg.partition("=")
        spec = _TRANSFORM_OPTIONS.get(option)
        if spec is not None:
            value_rule, build = spec
        elif option in _GLOBAL_OPTIONS:
            value_rule, build = _GLOBAL_OPTIONS[option], None
        else:
            raise click.UsageError(f"No such option: {option}")

        if value_rule == "none":
            if separator:
                raise click.UsageError(f"{option} does not take a value")
            param = ""
        elif separator:
            param = joined_value
        elif value_rule == "required":
            param, i = _take_arg(args, i, option)
        elif i + 1 < len(args) and not _is_option_token(args[i + 1]):
            i += 1
            param = args[i]
        else:
            param = ""

        if build is not None:
            parsed.values.setdefault(option, []).append(param)
            parsed.transforms.append(build(param, max_features))
        i += 1

    _mark_minimal_restores(parsed.transforms)
    return parsed


def parse_transform_args(args: list[str], max_features: int | None = None) -> list:
    """Parse a list of transform options into the ordered transforms.

    The list holds options only; a positional token raises ``click.UsageError``
    rather than being skipped.
    """
    parsed = parse_xform_args(args, max_features=max_features)
    if parsed.positionals:
        raise click.UsageError(f"Unexpected argument: {parsed.positionals[0]}")
    return parsed.transforms


def check_against_click(
    parsed: ParsedXformArgs, kwargs: dict, paths: list[str | None]
) -> None:
    """Raise ``click.UsageError`` where the ordered walk and Click disagree.

    ``kwargs`` is the command's Click keyword arguments and ``paths`` its
    positional arguments as Click bound them (``None`` for an omitted one).
    Each transform option's values must match Click's tuple one for one, and
    the positional tokens must be the paths, so every option Click accepted is
    in the ordered chain.
    """
    given_paths = [p for p in paths if p is not None]
    if parsed.positionals != given_paths:
        raise click.UsageError(
            f"The ordered option reading found the arguments {parsed.positionals}, "
            f"but the command received {given_paths}"
        )
    for option, (value_rule, _build) in _TRANSFORM_OPTIONS.items():
        read = parsed.values.get(option, [])
        expected = [True] * len(read) if value_rule == "none" else read
        received = list(kwargs.get(option[2:].replace("-", "_"), ()))
        if received != expected:
            raise click.UsageError(
                f"The ordered option reading found {option} values {read}, "
                f"but the command received {received}"
            )


def _mark_minimal_restores(transforms: list) -> None:
    """Flag each ``--add-*`` step that follows a ``--minimal``."""
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
