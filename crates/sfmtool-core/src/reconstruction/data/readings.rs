// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The in-memory form of each observation's stored readings
//! ([`ObservationReadings`]), the rows of the `.sfmr` columns flagged by
//! `has_observation_readings`.

pub use sfmtool_sfmr_format::{ObservationReading, ObservationReadingOptions};

use crate::camera::sampler::SamplerChoice;
use crate::patch::self_similarity::{SelfSimilarityParams, FLAT_FLOOR};

/// One row per observation, parallel to
/// [`PointSet::tracks`](super::PointSet::tracks), and the options the readings
/// were taken with ([`crate::patch::observation_reading`]).
///
/// A row is a record of the render it names: a writer that renders the
/// observation's tile reads it again, a writer without the photographs
/// carries it through, moving it with its observation, and a writer that
/// replaces an observation's photograph clears it to
/// [`ObservationReading::NOT_MEASURED`].
#[derive(Debug, Clone, PartialEq)]
pub struct ObservationReadings {
    /// Per observation, parallel to `tracks`.
    pub rows: Vec<ObservationReading>,
    /// What the readings were taken with.
    pub options: ObservationReadingOptions,
}

impl ObservationReadings {
    /// `count` rows with nothing measured, under `options`.
    pub fn not_measured(count: usize, options: ObservationReadingOptions) -> Self {
        Self {
            rows: vec![ObservationReading::NOT_MEASURED; count],
            options,
        }
    }

    /// The rows `idx` of `self`, in that order: what a pass that drops or
    /// reorders observations keeps.
    pub fn select(&self, idx: &[usize]) -> Self {
        Self {
            rows: idx.iter().map(|&i| self.rows[i]).collect(),
            options: self.options,
        }
    }
}

/// The reading options of renders made with `sampler`: the default
/// self-similarity reading ([`SelfSimilarityParams::default`], [`FLAT_FLOOR`]),
/// which is the one every reading in this crate takes, and the sampler rule's
/// threshold where `sampler` is the rule.
pub fn observation_reading_options(sampler: SamplerChoice) -> ObservationReadingOptions {
    let params = SelfSimilarityParams::default();
    ObservationReadingOptions {
        max_radius: params.max_radius,
        flat_floor: FLAT_FLOOR,
        noise: params.noise,
        relative_tolerance: params.relative_tolerance,
        anisotropic_threshold: sampler.anisotropic_threshold(),
    }
}
