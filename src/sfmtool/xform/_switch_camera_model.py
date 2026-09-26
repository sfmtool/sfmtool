# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Switch cameras of a reconstruction to another camera model, as a fit.

The new model is fitted to the old one over the angles where the old one is
trusted, by ``SfmrReconstruction.switch_camera_model`` in the Rust core (see
``specs/core/reconstruction/switch-camera-model.md``). Poses, points and
keypoints are unchanged; the report says how the reprojection errors moved.
"""

import math

from ..camera.cameras import _CAMERA_PARAM_NAMES, EQUIDISTANT_FISHEYE
from .._sfmtool.reconstruction import SfmrReconstruction

# The spline models are user-chosen targets here, and only here: they are still
# never a `solve` or `match` camera model, which is why `_CAMERA_PARAM_NAMES`
# leaves them out.
SPLINE_TARGETS = ("SFMTOOL_FISHEYE", "SFMTOOL_PINHOLE")
SWITCH_TARGETS = frozenset(_CAMERA_PARAM_NAMES) | {EQUIDISTANT_FISHEYE, *SPLINE_TARGETS}


class SwitchCameraModelTransform:
    """Fit a camera of another model to each chosen camera and replace it.

    Args:
        target_model: The model name, case-insensitive: a COLMAP lens model,
            ``EQUIDISTANT_FISHEYE``, ``SFMTOOL_FISHEYE`` or ``SFMTOOL_PINHOLE``.
        coeff_count: Spline coefficients for a spline target (default: the
            camera's own count when it already has that spline model,
            otherwise 8).
        theta_fit_deg: The largest incidence angle the fit samples (default:
            the camera's trusted bound, or its observations' extent).
        spline_domain_deg: Where a spline target's domain ends (default: the
            far image corner).
        cameras: Camera-table indexes to switch (default: every camera).

    A spline camera switched to its own spline model, with no
    ``theta_fit_deg``, is refitted over its whole spline domain: that is how a
    spline's coefficient count and domain change. Its domain end is kept
    exactly unless ``spline_domain_deg`` is given; its count is
    ``coeff_count``, which defaults to 8 here as for any spline target.
    """

    def __init__(
        self,
        target_model: str,
        coeff_count: int | None = None,
        theta_fit_deg: float | None = None,
        spline_domain_deg: float | None = None,
        cameras: list[int] | None = None,
    ):
        target_upper = target_model.strip().upper()
        if target_upper not in SWITCH_TARGETS:
            supported = ", ".join(sorted(SWITCH_TARGETS))
            raise ValueError(
                f"Unknown camera model '{target_model}'. Supported: {supported}"
            )
        if coeff_count is not None and target_upper not in SPLINE_TARGETS:
            raise ValueError(
                f"coeffs= applies only to {' and '.join(SPLINE_TARGETS)}, "
                f"not to {target_upper}"
            )
        self.target_model = target_upper
        self.coeff_count = coeff_count
        self.theta_fit_deg = theta_fit_deg
        self.spline_domain_deg = spline_domain_deg
        self.cameras = cameras

    def apply(self, recon: SfmrReconstruction) -> SfmrReconstruction:
        switched, report = recon.switch_camera_model(
            self.target_model,
            cameras=self.cameras,
            coeff_count=self.coeff_count,
            theta_fit_deg=self.theta_fit_deg,
            spline_domain_deg=self.spline_domain_deg,
        )
        for entry in report["cameras"]:
            for line in format_camera_report(entry):
                print(f"  {line}")
        return switched

    def description(self) -> str:
        parts = [self.target_model]
        if self.coeff_count is not None:
            parts.append(f"coeffs={self.coeff_count}")
        if self.theta_fit_deg is not None:
            parts.append(f"fit_to={self.theta_fit_deg:g}")
        if self.spline_domain_deg is not None:
            parts.append(f"spline_domain={self.spline_domain_deg:g}")
        if self.cameras is not None:
            parts.append("cameras=" + "+".join(str(c) for c in self.cameras))
        return f"Switch camera model to {','.join(parts)}"


def _deg(value: float | None) -> str:
    return "none" if value is None else f"{value:.1f}°"


def _summary(s: dict) -> str:
    if math.isnan(s["median_px"]):
        return "none"
    return (
        f"median {s['median_px']:.3f}, p90 {s['p90_px']:.3f}, max {s['max_px']:.3f} px"
    )


def format_camera_report(entry: dict) -> list[str]:
    """The lines ``sfm xform --camera-model`` prints for one switched camera."""
    fit = entry["fit"]
    source = entry["source"]
    target = entry["target"]
    obs = entry["observations"]
    extent = fit["extent"]
    lines = [
        f"Camera {entry['camera']} ({entry['images']} images): "
        f"{source.model} -> {target.model}",
    ]
    params = ", ".join(
        f"{k}={v:.6g}" for k, v in target.to_dict()["parameters"].items()
    )
    lines.append(f"  parameters: {params}")
    if fit["theta_fit_source"] == "spline_domain":
        # A spline refitted as its own model: the fit covers its whole domain.
        over = f"the whole spline domain θ ≤ {fit['theta_fit_deg']:.1f}°"
    else:
        over = f"θ ≤ {fit['theta_fit_deg']:.1f}° ({fit['theta_fit_source']})"
    fit_line = (
        f"  fit over {over}: "
        f"rms {fit['rms_px']:.3f} px, radial rms {fit['radial_rms_px']:.3f} px, "
        f"max {fit['max_px']:.3f} px"
    )
    if fit["spline_domain_deg"] is not None:
        fit_line += f"; spline domain {fit['spline_domain_deg']:.1f}°"
    lines.append(fit_line)
    held = format_monotone_constraint(fit["monotone_constraint"])
    if held:
        lines.append(f"  {held}")
    for dropped in fit["dropped"]:
        lines.append(f"  {dropped}")
    lines.append(
        f"  extent: edge {_deg(extent['edge_deg'])}, corner {_deg(extent['corner_deg'])}; "
        f"source trusted to {_deg(extent['source_trusted_deg'])}, "
        f"folds at {_deg(extent['source_fold_deg'])}"
    )
    lines.append(
        f"  observations: {obs['observations']} measured"
        + (f", {obs['unmeasured']} without a pixel" if obs["unmeasured"] else "")
        + f", up to {_deg(obs['max_theta_deg'])}"
    )
    lines.append(f"    before: {_summary(obs['before'])}")
    lines.append(f"    after:  {_summary(obs['after'])}")
    lines.append(f"    changed by more than 1 px: {obs['changed_over_1px']}")
    if obs["trusted_deg"] is not None:
        lines.append(
            f"    past the source's trusted bound ({obs['trusted_deg']:.1f}°): "
            f"{obs['past_trusted']}"
        )
        if obs["past_trusted"]:
            lines.append(f"      before: {_summary(obs['past_trusted_before'])}")
            lines.append(f"      after:  {_summary(obs['past_trusted_after'])}")
    outermost = format_outermost_keypoint(entry["outermost"])
    if outermost:
        lines.append(f"  {outermost}")
    return lines


def format_monotone_constraint(constraint: dict) -> str | None:
    """``monotone constraint bound at 23 angles, 95.2°-118.7°, where the fit
    departs from the source`` when a spline fit's monotonicity constraint bound, as
    ``CameraIntrinsics.refit`` reports it; ``None`` when it did not."""
    if not constraint["active"]:
        return None
    count = constraint["active_angles"]
    low, high = constraint["range_deg"]
    where = (
        f"{low:.1f}°" if f"{low:.1f}" == f"{high:.1f}" else f"{low:.1f}°-{high:.1f}°"
    )
    return (
        f"monotone constraint bound at {count} angle{'' if count == 1 else 's'}, "
        f"{where}, where the fit departs from the source"
    )


def format_outermost_keypoint(outermost: dict) -> str | None:
    """``outermost keypoint: 229.7 px, 101.2° observed; 244.1 px, 107.9°
    detected (24 .sift files)`` for one camera's outermost keypoints, as
    ``SfmrReconstruction.outermost_keypoints`` reports them; ``None`` when there
    is neither."""
    parts = []
    for source in ("observed", "detected"):
        reach = outermost[source]
        if reach is not None:
            parts.append(
                f"{reach['radius_px']:.1f} px, {reach['theta_deg']:.1f}° {source}"
            )
    if not parts:
        return None
    files = outermost["detected_images"]
    read = f" ({files} .sift files)" if outermost["detected"] is not None else ""
    return f"outermost keypoint: {'; '.join(parts)}{read}"
