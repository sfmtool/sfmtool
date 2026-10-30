# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Print per-image metrics analysis with reprojection error breakdown."""

from pathlib import Path

import click
import numpy as np

from .._sfmtool.reconstruction import SfmrReconstruction
from .._filenames import number_from_filename
from .._histogram_utils import create_histogram_string


def _compute_per_image_metrics(
    rust_recon: SfmrReconstruction,
    range_numbers: set[int] | None = None,
) -> list[dict]:
    """Compute per-image metrics using true per-observation reprojection errors."""
    num_images = rust_recon.image_count
    track_image_indexes = rust_recon.track_image_indexes
    track_point_indexes = rust_recon.track_point_indexes
    observation_counts = rust_recon.observation_counts
    image_names = rust_recon.image_names

    results = []
    for img_idx in range(num_images):
        if range_numbers is not None:
            file_number = number_from_filename(image_names[img_idx])
            if file_number is None or file_number not in range_numbers:
                continue
        obs_mask = track_image_indexes == img_idx
        obs_point_ids = track_point_indexes[obs_mask]
        obs_count = len(obs_point_ids)

        if obs_count == 0:
            results.append(
                {
                    "image_index": img_idx,
                    "image_name": image_names[img_idx],
                    "observation_count": 0,
                    "mean_error": float("nan"),
                    "median_error": float("nan"),
                    "max_error": float("nan"),
                    "mean_track_length": float("nan"),
                    "errors": np.array([]),
                }
            )
            continue

        obs_data = rust_recon.compute_observation_reprojection_errors(img_idx)
        obs_errors = obs_data[:, 1]
        valid_errors = obs_errors[~np.isnan(obs_errors)]

        point_track_lengths = observation_counts[obs_point_ids]

        if len(valid_errors) == 0:
            results.append(
                {
                    "image_index": img_idx,
                    "image_name": image_names[img_idx],
                    "observation_count": obs_count,
                    "mean_error": float("nan"),
                    "median_error": float("nan"),
                    "max_error": float("nan"),
                    "mean_track_length": float(np.mean(point_track_lengths)),
                    "errors": np.array([]),
                }
            )
            continue

        results.append(
            {
                "image_index": img_idx,
                "image_name": image_names[img_idx],
                "observation_count": obs_count,
                "mean_error": float(np.mean(valid_errors)),
                "median_error": float(np.median(valid_errors)),
                "max_error": float(np.max(valid_errors)),
                "mean_track_length": float(np.mean(point_track_lengths)),
                "errors": valid_errors,
            }
        )

    return results


# The legend line of each per-image correction column.
_CORRECTION_LEGENDS = {
    "Radial": "Radial = the image's radial correction at its camera's spline domain "
    "end, px (f·e_1)",
    "RShut": "RShut = the rolling-shutter displacement at the edge centre read "
    "farthest from the principal point's scanline, px",
}


def _correction_columns(recon: SfmrReconstruction) -> list[tuple[str, np.ndarray]]:
    """The per-image correction columns the table shows, as ``(title,
    values)`` with one value per image of ``recon``: ``Radial`` when a camera
    varies the radial correction per image and ``RShut`` when one varies the
    rolling-shutter rates, in pixels as
    :func:`sfmtool.analyze._correction_sizes.image_correction_sizes` measures
    them (NaN for an image of another camera); none for a reconstruction with
    neither."""
    varied = {name for names in recon.per_image_adjustments for name in names}
    if not varied & {"radial_bspline", "rolling_shutter"}:
        return []
    from ._correction_sizes import image_correction_sizes

    sizes = image_correction_sizes(recon)
    return [
        (title, sizes[key])
        for title, key in (("Radial", "radial_bspline"), ("RShut", "rolling_shutter"))
        if key in varied
    ]


def print_metrics_analysis(
    sfmr_path: Path,
    recon_name: str | None = None,
    range_expr: str | None = None,
) -> None:
    """Print per-image metrics analysis sorted by mean reprojection error."""
    from .._sfmtool.reconstruction import RangeExpr

    if recon_name is None:
        recon_name = sfmr_path.name

    range_numbers = None
    if range_expr is not None:
        range_numbers = set(RangeExpr(range_expr))

    rust_recon = SfmrReconstruction.load(sfmr_path)

    if rust_recon.point_count == 0:
        click.echo(f"\nPer-image metrics analysis for: {recon_name}")
        click.echo("=" * 70)
        click.echo("No 3D points in reconstruction \u2014 nothing to analyze.")
        click.echo("")
        return

    per_image = _compute_per_image_metrics(rust_recon, range_numbers=range_numbers)

    errors = rust_recon.rms_reprojection_errors
    recon_median_error = float(np.median(errors))
    flag_threshold_warn = recon_median_error * 1.5
    flag_threshold_alert = recon_median_error * 2.0

    per_image_sorted = sorted(
        per_image,
        key=lambda d: d["mean_error"] if not np.isnan(d["mean_error"]) else -1.0,
        reverse=True,
    )

    num_alert = sum(
        1
        for d in per_image_sorted
        if not np.isnan(d["mean_error"]) and d["mean_error"] > flag_threshold_alert
    )
    num_warn = sum(
        1
        for d in per_image_sorted
        if not np.isnan(d["mean_error"])
        and d["mean_error"] > flag_threshold_warn
        and d["mean_error"] <= flag_threshold_alert
    )
    num_no_obs = sum(1 for d in per_image_sorted if d["observation_count"] == 0)

    # The per-image correction columns, each shown when some camera varies it.
    correction_columns = _correction_columns(rust_recon)
    correction_header = "".join(f"  {title:>7}" for title, _ in correction_columns)

    click.echo(f"\nPer-image metrics analysis for: {recon_name}")
    click.echo("=" * 100)
    click.echo(
        f"Reconstruction: {rust_recon.image_count} images, "
        f"{rust_recon.point_count:,} points, "
        f"{rust_recon.observation_count:,} observations"
    )
    if range_expr is not None:
        click.echo(
            f"Range filter: {range_expr} ({len(per_image)} of {rust_recon.image_count} images)"
        )
    click.echo(
        f"Mean reprojection error: {errors.mean():.3f} px "
        f"(median: {recon_median_error:.3f} px)"
    )

    if num_alert > 0 or num_warn > 0 or num_no_obs > 0:
        parts = []
        if num_alert > 0:
            parts.append(f"{num_alert} high-error")
        if num_warn > 0:
            parts.append(f"{num_warn} elevated-error")
        if num_no_obs > 0:
            parts.append(f"{num_no_obs} zero-observation")
        click.echo(f"Flagged images: {', '.join(parts)}")

    click.echo("")

    all_errors = [e for d in per_image_sorted for e in d["errors"]]
    if all_errors:
        hist_min = 0.0
        hist_max = float(np.max(all_errors))
    else:
        hist_min, hist_max = 0.0, 1.0

    name_width = 50
    hist_width = 50
    hist_indent = " " * 6
    click.echo(
        f"{'Image (by mean error desc)':<{name_width}} {'Obs':>5}  {'MeanErr':>8}  {'MedErr':>8}  "
        f"{'MaxErr':>8}  {'MeanTL':>6}{correction_header}  {'Flag':<4}"
    )
    click.echo("-" * (100 + len(correction_header)))

    for entry in per_image_sorted:
        name = entry["image_name"]
        if len(name) > name_width:
            name = "..." + name[-(name_width - 3) :]

        obs = entry["observation_count"]
        corrections = "".join(
            f"  {values[entry['image_index']]:>7.2f}"
            for _, values in correction_columns
        )

        if obs == 0:
            click.echo(
                f"{name:<{name_width}} {obs:>5}  {'N/A':>8}  {'N/A':>8}  "
                f"{'N/A':>8}  {'N/A':>6}{corrections}  {'--':<4}"
            )
            continue

        mean_err = entry["mean_error"]
        med_err = entry["median_error"]
        max_err = entry["max_error"]
        mean_tl = entry["mean_track_length"]

        if mean_err > flag_threshold_alert:
            flag = "!!"
        elif mean_err > flag_threshold_warn:
            flag = "!"
        else:
            flag = ""

        click.echo(
            f"{name:<{name_width}} {obs:>5}  {mean_err:>8.3f}  {med_err:>8.3f}  "
            f"{max_err:>8.3f}  {mean_tl:>6.1f}{corrections}  {flag:<4}"
        )

        if len(entry["errors"]) > 0:
            hist_str, _, _ = create_histogram_string(
                entry["errors"],
                num_buckets=hist_width,
                min_val=hist_min,
                max_val=hist_max,
            )
            click.echo(f"{hist_indent}{hist_str}")

    click.echo("-" * (100 + len(correction_header)))
    click.echo(
        f"{hist_indent}{'0':<{hist_width // 2}}{hist_max:>{hist_width - hist_width // 2}.1f} px"
    )
    click.echo(
        f"!!  mean error > {flag_threshold_alert:.3f} px "
        f"(2x reconstruction median of {recon_median_error:.3f} px)"
    )
    click.echo(
        f"!   mean error > {flag_threshold_warn:.3f} px "
        f"(1.5x reconstruction median of {recon_median_error:.3f} px)"
    )
    click.echo("--  no observations (image registered but no points)")
    click.echo("MeanTL = mean track length (avg number of images observing each point)")
    for title, _ in correction_columns:
        click.echo(_CORRECTION_LEGENDS[title])
    click.echo("")
