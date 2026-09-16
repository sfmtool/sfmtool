# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared validation for photometric patch sampling parameters."""

_WINDOWS = ("gaussian_disk", "gaussian", "uniform")
_SAMPLERS = ("bilinear", "bilinear_mip", "anisotropic")


def validate_patch_params(*, window: str, window_sigma: float, sampler: str) -> None:
    """Validate the window and sampler parameters shared by patch operations."""
    if window not in _WINDOWS:
        raise ValueError(f"window must be one of {_WINDOWS}, got {window!r}")
    if window_sigma <= 0:
        raise ValueError(f"window_sigma must be positive, got {window_sigma}")
    if sampler not in _SAMPLERS:
        raise ValueError(f"sampler must be one of {_SAMPLERS}, got {sampler!r}")
