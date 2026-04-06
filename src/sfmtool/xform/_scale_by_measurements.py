# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Scale reconstruction to physical units using known point-pair distances."""

from __future__ import annotations

import re
import statistics
from pathlib import Path

import click
import numpy as np
import yaml

from .._sfmtool import Se3Transform, SfmrReconstruction

# Conversion factors: unit -> meters
_UNIT_TO_METERS = {
    "mm": 0.001,
    "cm": 0.01,
    "m": 1.0,
    "in": 0.0254,
    "ft": 0.3048,
}

_VALID_UNITS = set(_UNIT_TO_METERS.keys())

# Pattern for Point IDs: pt3d_{8 hex}_{index}
_POINT_ID_RE = re.compile(r"^pt3d_([0-9a-f]{8})_(\d+)$")

# Pattern for distance values with optional unit suffix
_DISTANCE_RE = re.compile(r"^([+-]?(?:\d+\.?\d*|\.\d+))([a-z]*)$")


def _parse_point_id(point_id: str) -> tuple[str, int]:
    m = _POINT_ID_RE.match(point_id)
    if not m:
        raise ValueError(
            f"Invalid Point ID format: '{point_id}'. "
            f"Expected: pt3d_{{8 hex chars}}_{{index}}"
        )
    return m.group(1), int(m.group(2))


def _parse_distance(value, target_unit: str) -> float:
    if isinstance(value, (int, float)):
        if value <= 0:
            raise ValueError(f"Distance must be positive, got {value}")
        return float(value)

    value_str = str(value).strip()
    m = _DISTANCE_RE.match(value_str)
    if not m:
        raise ValueError(
            f"Invalid distance format: '{value_str}'. "
            f"Expected a number with optional unit suffix (e.g., '304.8', '12in', '0.3048m')"
        )

    num = float(m.group(1))
    suffix = m.group(2)

    if num <= 0:
        raise ValueError(f"Distance must be positive, got {num}")

    if not suffix:
        return num

    if suffix not in _UNIT_TO_METERS:
        raise ValueError(
            f"Unknown unit suffix '{suffix}'. Supported: {', '.join(sorted(_VALID_UNITS))}"
        )

    meters = num * _UNIT_TO_METERS[suffix]
    return meters / _UNIT_TO_METERS[target_unit]


def _build_observation_index(
    recon: SfmrReconstruction,
) -> tuple[dict[str, int], dict[tuple[int, int], int]]:
    name_to_idx = {name: i for i, name in enumerate(recon.image_names)}

    obs_index: dict[tuple[int, int], int] = {}
    for i in range(len(recon.track_image_indexes)):
        key = (int(recon.track_image_indexes[i]), int(recon.track_feature_indexes[i]))
        obs_index[key] = int(recon.track_point_ids[i])

    return name_to_idx, obs_index


def _resolve_point_cross_recon(
    point_id: str,
    point_index: int,
    source_recon: SfmrReconstruction,
    input_name_to_idx: dict[str, int],
    input_obs_index: dict[tuple[int, int], int],
) -> tuple[int, str]:
    obs_mask = source_recon.track_point_ids == point_index
    src_image_idxs = source_recon.track_image_indexes[obs_mask]
    src_feat_idxs = source_recon.track_feature_indexes[obs_mask]

    resolved_points: dict[int, str] = {}
    for src_img_idx, src_feat_idx in zip(src_image_idxs, src_feat_idxs):
        src_img_name = source_recon.image_names[int(src_img_idx)]
        input_img_idx = input_name_to_idx.get(src_img_name)
        if input_img_idx is None:
            continue
        key = (input_img_idx, int(src_feat_idx))
        if key in input_obs_index:
            input_pt_idx = input_obs_index[key]
            img_basename = Path(src_img_name).name
            desc = f"{img_basename} feat #{src_feat_idx}"
            resolved_points[input_pt_idx] = desc

    if not resolved_points:
        obs_images = [
            Path(source_recon.image_names[int(i)]).name for i in src_image_idxs
        ]
        raise ValueError(
            f"Could not resolve {point_id} in input reconstruction. "
            f"Source point observed in: {', '.join(obs_images)}. "
            f"None of these (image, feature) pairs appear in the input reconstruction's tracks."
        )

    if len(set(resolved_points.keys())) > 1:
        click.echo(
            f"  Warning: {point_id} resolved to multiple points in input: "
            f"{list(resolved_points.keys())}. Using most common."
        )

    pt_idx = next(iter(resolved_points))
    return pt_idx, resolved_points[pt_idx]


def _print_histogram(scale_factors: list[float], target_unit: str) -> None:
    n = len(scale_factors)
    if n < 2:
        return

    min_s = min(scale_factors)
    max_s = max(scale_factors)

    if min_s == max_s:
        click.echo(f"\nAll {n} measurements agree: scale={min_s:.4g}")
        return

    num_bins = 5
    bin_width = (max_s - min_s) / num_bins

    counts = [0] * num_bins
    for s in scale_factors:
        bin_idx = int((s - min_s) / bin_width)
        if bin_idx >= num_bins:
            bin_idx = num_bins - 1
        counts[bin_idx] += 1

    if bin_width >= 1.0:
        fmt = ".1f"
    elif bin_width >= 0.1:
        fmt = ".2f"
    elif bin_width >= 0.01:
        fmt = ".3f"
    else:
        fmt = ".4g"

    labels = [format(min_s + i * bin_width, fmt) for i in range(num_bins)]
    max_label_w = max(len(lbl) for lbl in labels)

    click.echo(f"\nScale factor distribution ({n} measurements):")
    for i in range(num_bins):
        label = labels[i].rjust(max_label_w)
        bar = "##" * counts[i]
        count_str = f"  {counts[i]}" if counts[i] > 0 else ""
        click.echo(f"  {label} |{bar}{count_str}")

    median_s = statistics.median(scale_factors)
    mean_s = statistics.mean(scale_factors)
    std_s = statistics.stdev(scale_factors) if n > 1 else 0.0
    pct = (std_s / mean_s * 100) if mean_s != 0 else 0.0
    click.echo(
        f"  median: {median_s:.4g}, mean: {mean_s:.4g}, std: {std_s:.4g} ({pct:.2f}%)"
    )


class ScaleByMeasurementsTransform:
    """Scale reconstruction to physical units using known point-pair distances."""

    def __init__(self, yaml_path: Path):
        self.yaml_path = Path(yaml_path)
        self._scale_factor: float | None = None
        self._target_unit: str | None = None

    def apply(self, recon: SfmrReconstruction) -> SfmrReconstruction:
        config = self._load_and_validate()
        target_unit = config["unit"]
        self._target_unit = target_unit
        measurements = config["measurements"]
        sfmr_path = config.get("sfmr")

        parsed_measurements = []
        for i, m in enumerate(measurements):
            label = m.get("label", f"measurement {i + 1}")
            hash_a, idx_a = _parse_point_id(m["point_a"])
            hash_b, idx_b = _parse_point_id(m["point_b"])
            if hash_a != hash_b:
                raise ValueError(
                    f"Measurement '{label}': point_a and point_b have different hash prefixes "
                    f"({hash_a} vs {hash_b}). Both points must come from the same reconstruction."
                )
            real_dist = _parse_distance(m["distance"], target_unit)
            parsed_measurements.append(
                {
                    "label": label,
                    "hash_prefix": hash_a,
                    "idx_a": idx_a,
                    "idx_b": idx_b,
                    "point_id_a": m["point_a"],
                    "point_id_b": m["point_b"],
                    "real_distance": real_dist,
                }
            )

        input_hash_prefix = (recon.content_xxh128 or "")[:8]
        input_name_to_idx, input_obs_index = _build_observation_index(recon)

        by_prefix: dict[str, list[dict]] = {}
        for pm in parsed_measurements:
            by_prefix.setdefault(pm["hash_prefix"], []).append(pm)

        source_sfmr_path = None
        if sfmr_path:
            source_sfmr_path = (self.yaml_path.parent / sfmr_path).resolve()

        resolved: dict[str, int] = {}

        for prefix, group in by_prefix.items():
            if prefix == input_hash_prefix:
                for pm in group:
                    for pt_id, idx in [
                        (pm["point_id_a"], pm["idx_a"]),
                        (pm["point_id_b"], pm["idx_b"]),
                    ]:
                        if pt_id in resolved:
                            continue
                        if idx >= recon.point_count:
                            raise ValueError(
                                f"Point index {idx} out of range "
                                f"(reconstruction has {recon.point_count} points)"
                            )
                        resolved[pt_id] = idx
            else:
                source = self._load_source(prefix, source_sfmr_path)

                click.echo(
                    f"Resolving Point IDs from source ({prefix}...) -> input ({input_hash_prefix}...):"
                )

                for pm in group:
                    for pt_id, idx in [
                        (pm["point_id_a"], pm["idx_a"]),
                        (pm["point_id_b"], pm["idx_b"]),
                    ]:
                        if pt_id in resolved:
                            continue
                        if idx >= source.point_count:
                            raise ValueError(
                                f"Point index {idx} out of range in source reconstruction "
                                f"(has {source.point_count} points)"
                            )
                        input_idx, via = _resolve_point_cross_recon(
                            pt_id, idx, source, input_name_to_idx, input_obs_index
                        )
                        resolved[pt_id] = input_idx
                        click.echo(f"  {pt_id} -> point {input_idx} (via {via})")

                del source

        click.echo(f"\nScale by measurements (target unit: {target_unit}):")
        scale_factors = []
        for i, pm in enumerate(parsed_measurements):
            idx_a = resolved[pm["point_id_a"]]
            idx_b = resolved[pm["point_id_b"]]
            pos_a = recon.positions[idx_a]
            pos_b = recon.positions[idx_b]
            recon_dist = float(np.linalg.norm(pos_a - pos_b))
            if recon_dist < 1e-12:
                raise ValueError(
                    f"Measurement '{pm['label']}': points are coincident in reconstruction "
                    f"(distance={recon_dist:.2e})"
                )
            scale = pm["real_distance"] / recon_dist
            scale_factors.append(scale)
            click.echo(
                f'  {i + 1}. "{pm["label"]}": '
                f"recon={recon_dist:.4g}, real={pm['real_distance']:.4g}{target_unit} "
                f"-> scale={scale:.4g}"
            )

        median_scale = statistics.median(scale_factors)
        self._scale_factor = median_scale

        _print_histogram(scale_factors, target_unit)

        if len(scale_factors) > 1:
            max_dev = max(abs(s - median_scale) for s in scale_factors)
            pct_dev = max_dev / median_scale * 100 if median_scale != 0 else 0
            if pct_dev > 5.0:
                click.echo(
                    f"\n  WARNING: Max deviation from median is {pct_dev:.1f}% -- "
                    f"measurements disagree significantly. Check for misidentified points "
                    f"or reconstruction distortion."
                )

        click.echo(f"\nScaling reconstruction by {median_scale:.4g}")

        transform = Se3Transform(scale=median_scale)
        recon = transform @ recon
        recon = recon.clone_with_changes(world_space_unit=target_unit)
        return recon

    def _load_and_validate(self) -> dict:
        with open(self.yaml_path) as f:
            config = yaml.safe_load(f)

        if not isinstance(config, dict):
            raise ValueError(
                f"Measurements file must be a YAML mapping, got {type(config)}"
            )

        if "unit" not in config:
            raise ValueError("Measurements file must have a 'unit' field")
        unit = config["unit"]
        if unit not in _VALID_UNITS:
            raise ValueError(
                f"Unknown unit '{unit}'. Supported: {', '.join(sorted(_VALID_UNITS))}"
            )

        if "measurements" not in config:
            raise ValueError("Measurements file must have a 'measurements' field")
        measurements = config["measurements"]
        if not isinstance(measurements, list) or len(measurements) == 0:
            raise ValueError("'measurements' must be a non-empty list")

        for i, m in enumerate(measurements):
            if not isinstance(m, dict):
                raise ValueError(f"Measurement {i + 1} must be a mapping")
            for field in ("point_a", "point_b", "distance"):
                if field not in m:
                    raise ValueError(
                        f"Measurement {i + 1} missing required field '{field}'"
                    )

        return config

    def _load_source(
        self, hash_prefix: str, sfmr_path: Path | None
    ) -> SfmrReconstruction:
        if sfmr_path is not None:
            if not sfmr_path.exists():
                raise FileNotFoundError(f"Source .sfmr file not found: {sfmr_path}")
            source = SfmrReconstruction.load(sfmr_path)
            source_prefix = (source.content_xxh128 or "")[:8]
            if source_prefix != hash_prefix:
                raise ValueError(
                    f"Source .sfmr file hash prefix '{source_prefix}' does not match "
                    f"Point ID hash prefix '{hash_prefix}'. "
                    f"Check the 'sfmr' path in {self.yaml_path.name}."
                )
            return source

        raise ValueError(
            f"Point IDs reference reconstruction with hash prefix '{hash_prefix}' "
            f"which doesn't match the input reconstruction. "
            f"Add an 'sfmr' field to {self.yaml_path.name} with the path to the "
            f"source .sfmr file."
        )

    def description(self) -> str:
        if self._scale_factor is not None and self._target_unit is not None:
            return (
                f"Scale by measurements ({self.yaml_path.name}): "
                f"{self._scale_factor:.4g}x to {self._target_unit}"
            )
        return f"Scale by measurements ({self.yaml_path.name})"
