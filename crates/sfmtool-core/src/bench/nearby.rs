// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The building blocks of finding the tracks near a pixel: reading the pixel's
//! patch along its ray, the far-field sweep built on that read, the distance
//! range a set of sightings allows along the pixel's ray, and the matching
//! sources that find candidate tracks near the pixel.
//!
//! `specs/drafts/nearby-tracks.md` is the plan these belong to, and
//! `specs/core/bench/far-field-sweep.md`,
//! `specs/core/bench/distance-range.md` and
//! `specs/core/bench/nearby-sources.md` the design of the pieces here. Like
//! the rest of [`super`], nothing here writes the reconstruction: the sweep
//! reads the photographs and fits bench tracks, the sources read the
//! reconstruction and its index files, and each returns what it found.
//!
//! The photographs are read in grey, blurred, through a [`GreyImages`] cache
//! the caller builds once per set of views and keeps, since the same images
//! are read by every query.

mod candidate;
mod clusters;
mod far_field;
mod grey;
mod guided;
mod patch_read;
mod points;
mod range;
mod triangulate;

#[cfg(test)]
mod source_tests;
#[cfg(test)]
mod tests;

pub use candidate::{NearbyCandidate, NearbySource, NearbySourceError};
pub use clusters::{nearby_cluster_tracks, ClusterMembers, ClusterTracksOptions};
pub use far_field::{
    far_field_sweep, FarFieldError, FarFieldGrouping, FarFieldMetrics, FarFieldOptions,
    FarFieldReading, FarFieldSweep, Refit, WideAmong,
};
pub use grey::{blurred_grey, sample_grey, GreyImage, GreyImages, GREY_BLUR_SIGMA};
pub use guided::{guided_matches, GuidedOptions, GuidedSource, ImageDescriptors, KeypointRays};
pub use patch_read::{read_patch_along_ray, PatchRead, PatchSamples, RayPatch, PATCH_GRID};
pub use points::{nearby_points, PointsOptions};
pub use range::{
    camera_spread, classify_range, distance_range, DistanceRangeError, RangeClass, RangeOptions,
};
pub use triangulate::{triangulate_sightings, RayMeeting};
