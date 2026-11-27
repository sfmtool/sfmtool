# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Scale a reconstruction to the median of its stored distance measurements."""

import click

from .._sfmtool.fileio import LENGTH_UNIT_NAMES
from .._sfmtool.geometry import Se3Transform
from .._sfmtool.reconstruction import SfmrReconstruction

OPTION = "--scale-to-median-measurement"


def _title(entry: dict) -> str:
    """``Distance #2 "paw width"``: a measurement's kind and row, and its label."""
    label = entry["label"]
    return (
        f'Distance #{entry["index"]} "{label}"'
        if label
        else f"Distance #{entry['index']}"
    )


class ScaleToMedianMeasurementTransform:
    """Scale to ``unit`` by the median factor of the stored distance measurements.

    The factor and its statistics are core's ``distance_measurement_scale``,
    read through ``SfmrReconstruction.distance_measurement_scale`` on the
    reconstruction as it stands at this step; the scale is ``Se3Transform``,
    as for ``--scale``, and the step then declares ``unit`` as the
    reconstruction's ``world_space_unit``.
    """

    def __init__(self, unit: str):
        if unit not in LENGTH_UNIT_NAMES:
            *others, last = LENGTH_UNIT_NAMES
            raise ValueError(f"expected a unit, one of {', '.join(others)} or {last}")
        self.unit = unit
        self._factor: float | None = None

    def apply(self, recon: SfmrReconstruction) -> SfmrReconstruction:
        try:
            scale = recon.distance_measurement_scale(self.unit)
        except ValueError as e:
            raise ValueError(f"{OPTION} {self.unit}: {e}") from e
        _print_scale(scale)
        self._factor = scale["factor"]
        recon = Se3Transform(scale=self._factor) @ recon
        return recon.clone_with_changes(world_space_unit=self.unit)

    def description(self) -> str:
        if self._factor is not None:
            return f"Scale to median measurement: {self._factor:.4g}x to {self.unit}"
        return f"Scale to median measurement ({self.unit})"


def _print_scale(scale: dict) -> None:
    """Print the measurements, the left-out ones, the statistics and the factor."""
    unit = scale["unit"]
    s = scale["statistics"]
    factors = scale["factors"]
    excluded = scale["excluded"]
    total = len(factors) + len(excluded)

    click.echo(f"\nScale to median measurement (unit: {unit}):")
    click.echo(f"  Used {s['count']} of {total} distance measurements:")
    for f in factors:
        click.echo(
            f"    {_title(f)}: stated {f['stated']:.4g} {unit}, "
            f"scene {f['scene_distance']:.4g} su, factor {f['factor']:.4g}, "
            f"deviation {f['deviation_percent']:+.2f}%"
        )
    if excluded:
        click.echo("  Left out:")
        for e in excluded:
            click.echo(f"    {_title(e)}: {e['reason']}")

    if s["sample_std"] is None:
        spread = "std: none for a single measurement"
    else:
        spread = (
            f"std: {s['sample_std']:.4g} ({s['sample_std_percent']:.2f}% of the median)"
        )
    mad_percent = 100.0 * s["median_absolute_deviation"] / s["median"]
    click.echo(
        f"  median: {s['median']:.4g}, mean: {s['mean']:.4g}, {spread}, "
        f"median absolute deviation: {mad_percent:.2f}% of the median"
    )
    click.echo(
        f"  min: {s['min']:.4g}, max: {s['max']:.4g}, "
        f"largest deviation: {s['max_abs_deviation_percent']:.2f}%"
    )

    if s["disagrees"]:
        bar = s["disagreement_percent"]
        past = [f for f in factors if abs(f["deviation_percent"]) > bar]
        click.echo(
            f"\n  WARNING: {len(past)} of {s['count']} measurements deviate more "
            f"than {bar:g}% from the median ("
            + ", ".join(_title(f) for f in past)
            + "). Check their endpoints, or the reconstruction for distortion, "
            "before trusting the scale."
        )

    click.echo(f"\nScaling reconstruction by {scale['factor']:.4g}")
