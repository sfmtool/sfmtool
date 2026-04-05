# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Utilities for creating and displaying histograms in terminal output."""

import numpy as np


def estimate_z_from_histogram(
    hist_counts: np.ndarray,
    min_z: float,
    max_z: float,
    percentile: float,
) -> float:
    """Estimate Z value at given percentile from histogram.

    Uses cumulative distribution to find the bucket containing the percentile,
    then linearly interpolates within that bucket for better accuracy.
    """
    if percentile < 0 or percentile > 100:
        raise ValueError(f"Percentile must be between 0 and 100, got {percentile}")

    total = hist_counts.sum()
    if total == 0:
        return (min_z + max_z) / 2

    cumsum = np.cumsum(hist_counts)
    target_count = (percentile / 100.0) * total

    bucket_idx = np.searchsorted(cumsum, target_count)
    bucket_idx = min(bucket_idx, len(hist_counts) - 1)

    num_buckets = len(hist_counts)
    bucket_width = (max_z - min_z) / num_buckets
    bucket_left = min_z + bucket_idx * bucket_width

    count_before = cumsum[bucket_idx - 1] if bucket_idx > 0 else 0
    bucket_count = hist_counts[bucket_idx]

    if bucket_count > 0:
        fraction_in_bucket = (target_count - count_before) / bucket_count
        z_value = bucket_left + fraction_in_bucket * bucket_width
    else:
        z_value = bucket_left

    return z_value


def render_histogram_string(counts: np.ndarray) -> str:
    """Render histogram counts as a Unicode block string."""
    if len(counts) == 0 or counts.max() == 0:
        return " " * len(counts)

    max_count = counts.max()
    hist_chars = []
    blocks = " \u2581\u2582\u2583\u2584\u2585\u2586\u2587\u2588"

    for count in counts:
        if count == 0:
            hist_chars.append(" ")
        else:
            level = int((count / max_count) * 8)
            level = max(1, min(8, level))
            hist_chars.append(blocks[level])

    return "".join(hist_chars)


def create_histogram(
    values: np.ndarray,
    num_buckets: int = 64,
    min_val: float | None = None,
    max_val: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Create histogram counts from values."""
    if len(values) == 0:
        return np.array([]), np.array([])

    if min_val is None:
        min_val = np.min(values)
    if max_val is None:
        max_val = np.max(values)

    if min_val == max_val:
        counts = np.zeros(num_buckets)
        counts[num_buckets // 2] = len(values)
        bin_edges = np.linspace(min_val - 0.5, min_val + 0.5, num_buckets + 1)
    else:
        counts, bin_edges = np.histogram(
            values, bins=num_buckets, range=(min_val, max_val)
        )

    return counts, bin_edges


def create_histogram_string(
    values: np.ndarray,
    num_buckets: int = 64,
    min_val: float | None = None,
    max_val: float | None = None,
) -> tuple[str, np.ndarray, np.ndarray]:
    """Create a Unicode block histogram string from values."""
    counts, bin_edges = create_histogram(values, num_buckets, min_val, max_val)
    hist_str = render_histogram_string(counts)
    return hist_str, bin_edges, counts


def print_histogram(
    values: np.ndarray,
    title: str,
    num_buckets: int = 64,
    min_val: float | None = None,
    max_val: float | None = None,
    show_stats: bool = True,
) -> None:
    """Print a histogram with optional statistics."""
    hist_str, bin_edges, counts = create_histogram_string(
        values, num_buckets, min_val, max_val
    )

    print(f"    {title}:")
    print(f"      {hist_str}")

    if len(bin_edges) > 0:
        actual_min = bin_edges[0]
        actual_max = bin_edges[-1]
        print(f"      {actual_min:<8.3f}{'':>{num_buckets - 16}}{actual_max:>8.3f}")

    if show_stats and len(values) > 0:
        print(f"      Mean: {np.mean(values):.4f}, Median: {np.median(values):.4f}")
