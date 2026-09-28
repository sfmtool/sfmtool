// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The building blocks of finding the tracks near a pixel: reading the pixel's
//! patch along its ray, the far-field sweep built on that read, and the
//! distance range a set of sightings allows along the pixel's ray.
//!
//! `specs/drafts/nearby-tracks.md` is the plan these belong to, and
//! `specs/core/bench/far-field-sweep.md` and
//! `specs/core/bench/distance-range.md` the design of the pieces here. Like
//! the rest of [`super`], nothing here writes the reconstruction: the sweep
//! reads the photographs and fits bench tracks, and returns what it read.
//!
//! The photographs are read in grey, blurred, through a [`GreyImages`] cache
//! the caller builds once per set of views and keeps, since the same images
//! are read by every query.

mod far_field;
mod grey;
mod patch_read;
mod range;

#[cfg(test)]
mod tests;

pub use far_field::{
    far_field_sweep, FarFieldError, FarFieldGrouping, FarFieldMetrics, FarFieldOptions,
    FarFieldReading, FarFieldSweep, Refit, WideAmong,
};
pub use grey::{blurred_grey, sample_grey, GreyImage, GreyImages, GREY_BLUR_SIGMA};
pub use patch_read::{read_patch_along_ray, PatchRead, PatchSamples, RayPatch, PATCH_GRID};
pub use range::{
    camera_spread, classify_range, distance_range, DistanceRangeError, RangeClass, RangeOptions,
};
