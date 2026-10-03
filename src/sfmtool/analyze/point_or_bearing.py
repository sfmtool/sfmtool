# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Report the point-or-bearing likelihood-ratio test beside the stored representation.

`sfm analyze --depth-reliability` and `sfm inspect --verbose` print, for each
stored point, what the test in
``SfmrReconstruction.point_or_bearing_scores`` would call it (a finite point or
a bearing) and compare that with how the point is stored. Nothing here changes
a reconstruction. ``SfmrReconstruction.classify_points_at_infinity`` (``sfm
xform --classify-points-at-infinity``) is what stores the verdicts, so the
disagreements listed here are the points it would change, less the ones it
declines to (a bearing behind a camera, a fit with no usable point). The
``inverse_depth_z`` printed beside the test is a diagnostic: reclassification,
discovery and the bench all decide on the test.
"""

import re
import textwrap
from dataclasses import dataclass

import click
import numpy as np

from .._histogram_utils import print_histogram
from .._sfmtool.analysis import DEFAULT_DEPTH_LIKELIHOOD_RATIO_THRESHOLD
from .._sfmtool.reconstruction import SfmrReconstruction

# How many disagreeing points `analyze --depth-reliability` lists per direction.
MAX_LISTED_DISAGREEMENTS = 20

# How many promoted point indexes `inspect --verbose` names on its one line.
MAX_NAMED_IN_INSPECT = 5


@dataclass
class PointOrBearingReport:
    """The test's verdicts over every point of a reconstruction.

    ``noise`` is ``reprojection_noise()``'s dict, or ``None`` when it could not
    be measured (``noise_error`` says why). ``scores`` is
    ``point_or_bearing_scores()``'s dict over every point, or ``None`` when no
    noise level was available (``scores_error`` says why).
    """

    threshold: float
    sigma_given: bool
    noise: dict | None
    noise_error: str | None
    scores: dict | None
    scores_error: str | None
    at_infinity: np.ndarray

    @property
    def sigma_px(self) -> float:
        return float(self.scores["sigma_px"])

    @property
    def scored(self) -> np.ndarray:
        return np.asarray(self.scores["scored"], dtype=bool)

    @property
    def is_finite(self) -> np.ndarray:
        return np.asarray(self.scores["is_finite"], dtype=bool)

    @property
    def promotions(self) -> np.ndarray:
        """Indexes of points at infinity the test calls finite."""
        return np.flatnonzero(self.at_infinity & self.scored & self.is_finite)

    @property
    def demotions(self) -> np.ndarray:
        """Indexes of finite points the test calls bearings."""
        return np.flatnonzero(~self.at_infinity & self.scored & ~self.is_finite)


def _io_reason(error: OSError) -> str:
    """A short reason for a failed pixel read, naming a missing .sift file once."""
    match = re.search(r"failed to read SIFT file '([^']+)'", str(error))
    if match:
        return f"cannot read {match.group(1)}"
    return str(error)


def no_noise_reason(recon: SfmrReconstruction) -> str:
    """Why ``reprojection_noise()`` measured nothing: no observation of a finite
    point at all, or observations whose pixels gave no usable residual."""
    at_infinity = np.asarray(recon.point_is_at_infinity)
    observed = np.asarray(recon.track_point_indexes)
    if not (~at_infinity[observed]).any():
        return (
            "the reconstruction has no observation of a finite point, "
            "so it has no measured noise level"
        )
    return (
        "no observation of a finite point gives a usable residual (no finite "
        "pixel, or none the camera model can image), so there is no measured "
        "noise level"
    )


def point_or_bearing_report(
    recon: SfmrReconstruction,
    sigma_px: float | None = None,
    threshold: float | None = None,
) -> PointOrBearingReport:
    """Measure the noise level and score every point.

    ``sigma_px`` overrides the measured noise level; the measured value is still
    read so the report can print both. Failures to measure or score (no finite
    point, a missing ``.sift`` file) are recorded in the report rather than
    raised, so a printer can say what is unavailable.
    """
    if threshold is None:
        threshold = DEFAULT_DEPTH_LIKELIHOOD_RATIO_THRESHOLD
    at_infinity = np.asarray(recon.positions_xyzw)[:, 3] == 0.0

    noise = None
    noise_error = None
    try:
        noise = recon.reprojection_noise()
        if noise["sigma_px"] is None:
            noise_error = no_noise_reason(recon)
    except OSError as e:
        noise_error = f"the reprojection noise could not be measured: {_io_reason(e)}"

    scores = None
    scores_error = None
    sigma = sigma_px
    if sigma is None and noise is not None:
        sigma = noise["sigma_px"]
    if sigma is None:
        scores_error = noise_error
    else:
        try:
            scores = recon.point_or_bearing_scores(sigma_px=sigma, threshold=threshold)
        except OSError as e:
            scores_error = f"the points could not be scored: {_io_reason(e)}"
        except ValueError as e:
            scores_error = str(e)

    return PointOrBearingReport(
        threshold=float(threshold),
        sigma_given=sigma_px is not None,
        noise=noise,
        noise_error=noise_error,
        scores=scores,
        scores_error=scores_error,
        at_infinity=at_infinity,
    )


def _px(value: float) -> str:
    """A pixel noise level to four significant figures, however small."""
    return f"{value:.4g} px"


def _measured_over(noise: dict) -> str:
    """How many observations a measured noise level is over, and how many were
    left out of it as outliers."""
    return (
        f"over {noise['observation_count']:,} observations of finite points, "
        f"{noise['outlier_count']:,} excluded as outliers"
    )


def _noise_line(report: PointOrBearingReport) -> str:
    """The noise level used and where it came from, on one line."""
    noise = report.noise
    measured = None
    if noise is not None and noise["sigma_px"] is not None:
        measured = f"{_px(noise['sigma_px'])} {_measured_over(noise)}"
    if report.sigma_given:
        tail = f"measured {measured}" if measured else "none measured"
        return f"{_px(report.sigma_px)} given ({tail})"
    return f"{_px(report.sigma_px)}, measured {_measured_over(noise)}"


def _point_id(recon: SfmrReconstruction, index: int) -> str:
    return f"pt3d_{recon.content_xxh128[:8].lower()}_{index}"


def _format_float(value: float, width: int, precision: int = 1) -> str:
    if np.isnan(value):
        return f"{'n/a':>{width}}"
    if np.isinf(value):
        return f"{'inf':>{width}}"
    text = f"{value:,.{precision}f}"
    if len(text) > width:
        # Scores reach 1e12 and more at a tiny --sigma-px; keep the column.
        text = f"{value:.{max(width - 7, 1)}e}"
    return f"{text:>{width}}"


def _print_group(
    title: str, mask: np.ndarray, report: PointOrBearingReport, stored_finite: bool
) -> None:
    """Score statistics and verdict counts over one stored representation."""
    scored = mask & report.scored
    n = int(scored.sum())
    click.echo(f"\n  {title}: {n:,} scored")
    if n == 0:
        return
    s = report.scores
    depth_score = np.asarray(s["depth_score"])[scored]
    bearing_cost = np.asarray(s["bearing_cost"])[scored]
    finite = report.is_finite[scored]
    t = report.threshold

    click.echo(
        f"    Depth score: median {np.median(depth_score):,.2f}, "
        f"min {depth_score.min():,.2f}, max {depth_score.max():,.2f}"
    )
    n_finite = int(finite.sum())
    n_bearing = n - n_finite
    by_bound = int((finite & (depth_score < t)).sum())
    by_cost = int((~finite & (bearing_cost < t)).sum())
    finite_line = f"    Finite verdict:  {n_finite:,} ({100.0 * n_finite / n:.1f}%)"
    if by_bound:
        finite_line += f", {by_bound:,} on the midpoint bound alone"
    bearing_line = f"    Bearing verdict: {n_bearing:,} ({100.0 * n_bearing / n:.1f}%)"
    if by_cost:
        bearing_line += f", {by_cost:,} with the bearing cost under the threshold"
    # The verdict that agrees with the stored representation goes first.
    for line in (
        (finite_line, bearing_line)
        if stored_finite
        else (
            bearing_line,
            finite_line,
        )
    ):
        click.echo(line)

    # Log scale: the score spans orders of magnitude, and a wide-angle track
    # scores near 0. Scores below 1 are clipped to 1 so they land in the first
    # bucket rather than stretching the axis.
    log_score = np.log10(np.clip(depth_score, 1.0, None))
    hi = max(float(log_score.max()), float(np.log10(max(t, 1.0))))
    print_histogram(
        log_score,
        f"log10(depth score), clipped at 1; threshold at {np.log10(max(t, 1.0)):.2f}",
        min_val=0.0,
        max_val=hi if hi > 0.0 else 1.0,
        show_stats=False,
    )


def _print_disagreements(
    recon: SfmrReconstruction,
    report: PointOrBearingReport,
    title: str,
    indexes: np.ndarray,
    descending: bool,
    stored_z: np.ndarray,
    noise_px: float,
) -> None:
    """List the disagreeing points, fitting only the ones listed."""
    count = indexes.size
    click.echo(f"    {title}: {count:,}")
    if count == 0:
        return
    s = report.scores
    order = np.argsort(np.asarray(s["depth_score"])[indexes], kind="stable")
    if descending:
        order = order[::-1]
    listed = indexes[order][:MAX_LISTED_DISAGREEMENTS]

    fitted = recon.point_or_bearing_scores(
        point_indexes=listed,
        sigma_px=report.sigma_px,
        threshold=report.threshold,
        fit=True,
    )
    fit = fitted["fit"]
    likelihood_ratio = np.asarray(fit["depth_likelihood_ratio"])
    inverse_depth = np.asarray(fit["inverse_depth"])
    with np.errstate(divide="ignore", invalid="ignore"):
        distance = np.where(fit["fitted"], 1.0 / inverse_depth, np.nan)

    # The inverse-depth z: at the stored point for a finite point, and at the
    # fitted point for a point at infinity, which has no stored depth. The
    # fitted points go into a copy, whose diagnostics are read for them alone.
    # Its per-point noise is max(error, noise_px), and a point at
    # infinity's stored error is its bearing's residual, so the copy's errors
    # are recomputed at the fitted points first.
    z = np.full(listed.size, np.nan)
    on_infinity = report.at_infinity[listed]
    z[~on_infinity] = stored_z[listed[~on_infinity]]
    # A fit that leaves the point at infinity (inverse depth 0, which a low
    # threshold reaches) places nothing, so its z stays n/a.
    placed = on_infinity & np.asarray(fit["fitted"]) & (inverse_depth > 0.0)
    if placed.any():
        xyzw = np.asarray(recon.positions_xyzw, dtype=np.float64).copy()
        xyzw[listed[placed], :3] = np.asarray(fit["point"])[placed]
        xyzw[listed[placed], 3] = 1.0
        placed_recon = recon.clone_with_changes(positions=xyzw)
        try:
            placed_recon.recompute_point_errors()
        except OSError:
            # No pixels to measure the fitted points against: leave z unknown
            # rather than print it at the bearing's error.
            placed[:] = False
        if placed.any():
            diag = placed_recon.triangulation_diagnostics(noise_px=noise_px)
            z[placed] = np.asarray(diag["inverse_depth_z"])[listed[placed]]

    views = np.asarray(fitted["num_views"])
    score = np.asarray(fitted["depth_score"])
    bound = np.asarray(fitted["midpoint_bound"])
    id_width = max(len(_point_id(recon, int(i))) for i in listed)
    click.echo(
        f"      {'Point':<{id_width}}  {'Views':>5}  {'Score':>11}  {'Bound':>11}  "
        f"{'Lambda':>11}  {'z':>6}  {'Distance':>10}"
    )
    for k, index in enumerate(listed):
        click.echo(
            f"      {_point_id(recon, int(index)):<{id_width}}  {int(views[k]):>5}  "
            f"{_format_float(score[k], 11)}  {_format_float(bound[k], 11)}  "
            f"{_format_float(likelihood_ratio[k], 11)}  "
            f"{_format_float(z[k], 6, 2)}  {_format_float(distance[k], 10)}"
        )
    if count > listed.size:
        click.echo(f"      ... and {count - listed.size:,} more")


def print_point_or_bearing(
    recon: SfmrReconstruction,
    report: PointOrBearingReport,
    stored_z: np.ndarray,
    noise_px: float = 1.0,
) -> None:
    """The detailed report `sfm analyze --depth-reliability` prints.

    ``stored_z`` is ``triangulation_diagnostics()["inverse_depth_z"]``, the
    inverse-depth z at each stored finite point, at the noise floor noise_px.
    """
    click.echo("\nPoint or bearing (likelihood-ratio test on the depth):")
    if report.scores is None:
        click.echo(f"  Unavailable: {report.scores_error}")
        return

    click.echo(f"  Noise level: {_noise_line(report)}")
    noise = report.noise
    if noise is not None and len(noise["per_camera_sigma_px"]) > 1:
        for cam, (sigma, count) in enumerate(
            zip(noise["per_camera_sigma_px"], noise["per_camera_observation_count"])
        ):
            sigma_text = "none" if np.isnan(sigma) else _px(sigma)
            click.echo(
                f"    Camera {cam}: {sigma_text} over {int(count):,} observations"
            )
    click.echo(f"  Threshold: {report.threshold:g}")
    unscored = int((~report.scored).sum())
    if unscored:
        click.echo(f"  Not scored (fewer than two usable rays): {unscored:,}")

    _print_group("Finite points", ~report.at_infinity, report, stored_finite=True)
    _print_group("Points at infinity", report.at_infinity, report, stored_finite=False)

    click.echo("\n  Disagreements with the stored representation:")
    _print_disagreements(
        recon,
        report,
        "Finite points the test calls bearings",
        report.demotions,
        descending=False,
        stored_z=stored_z,
        noise_px=noise_px,
    )
    _print_disagreements(
        recon,
        report,
        "Points at infinity the test calls finite",
        report.promotions,
        descending=True,
        stored_z=stored_z,
        noise_px=noise_px,
    )
    _print_reclassification(recon, report)
    if report.demotions.size or report.promotions.size:
        legend = (
            "Score: depth score. Bound: midpoint bound. Lambda: likelihood "
            "ratio of the plain least-squares fit. z: the "
            f"inverse-depth z diagnostic, with per-point noise max(error, {noise_px:g} px), at "
            "the stored point and its stored error, or for a point at infinity at "
            "the fitted point and its error there. Distance: the fitted point's "
            "distance from its observing cameras' centroid."
        )
        click.echo(
            textwrap.fill(
                legend,
                width=78,
                initial_indent="    ",
                subsequent_indent="    ",
                break_on_hyphens=False,
            )
        )


def _print_reclassification(
    recon: SfmrReconstruction, report: PointOrBearingReport
) -> None:
    """What ``classify_points_at_infinity`` would store, at the report's noise level.

    The pass decides at the default threshold, so when the report's threshold
    differs its counts differ from the disagreements above.
    """
    try:
        _, summary = recon.classify_points_at_infinity(sigma_px=report.sigma_px)
    except (OSError, ValueError) as e:
        click.echo(f"    Reclassification: unavailable ({e})")
        return
    line = (
        f"    Reclassification would promote {summary['promoted']:,} and demote "
        f"{summary['demoted']:,}"
    )
    if summary["refitted"]:
        line += (
            f", and move {summary['refitted']:,} off a position behind or on "
            "top of a camera"
        )
    declined = []
    if summary["bearing_behind_camera"]:
        declined.append(
            f"{summary['bearing_behind_camera']:,} left finite with the bearing "
            "behind a camera"
        )
    if summary["no_usable_point"]:
        declined.append(
            f"{summary['no_usable_point']:,} left at infinity without a usable "
            "fitted point"
        )
    if summary["left_unusable"]:
        declined.append(
            f"{summary['left_unusable']:,} left behind or on top of a camera, "
            "with no usable point and the bearing behind a camera"
        )
    if declined:
        line += "; " + ", ".join(declined)
    click.echo(line)
    if report.threshold != DEFAULT_DEPTH_LIKELIHOOD_RATIO_THRESHOLD:
        click.echo(
            f"    (it decides at the threshold "
            f"{DEFAULT_DEPTH_LIKELIHOOD_RATIO_THRESHOLD:g}, not {report.threshold:g})"
        )


def print_point_or_bearing_brief(recon: SfmrReconstruction) -> None:
    """The short form `sfm inspect --verbose` prints under its 3D point statistics."""
    report = point_or_bearing_report(recon)
    if report.scores is None:
        click.echo(f"  Point or bearing: unavailable ({report.scores_error})")
        return
    click.echo(
        f"  Point or bearing (likelihood-ratio test, threshold {report.threshold:g}):"
    )
    click.echo(f"    Noise level: {_noise_line(report)}")
    n_finite = int((~report.at_infinity & report.scored).sum())
    n_infinity = int((report.at_infinity & report.scored).sum())
    click.echo(
        f"    Finite points called bearings: {report.demotions.size:,} of {n_finite:,} scored"
    )
    promotions = report.promotions
    line = f"    Points at infinity called finite: {promotions.size:,} of {n_infinity:,} scored"
    if promotions.size:
        score = np.asarray(report.scores["depth_score"])[promotions]
        named = promotions[np.argsort(score, kind="stable")[::-1]]
        shown = ", ".join(str(int(i)) for i in named[:MAX_NAMED_IN_INSPECT])
        more = promotions.size - min(promotions.size, MAX_NAMED_IN_INSPECT)
        line += f" (points {shown}" + (f" and {more:,} more)" if more else ")")
    click.echo(line)
