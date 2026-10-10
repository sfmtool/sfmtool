// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The in-memory form of each observation's stored readings
//! ([`ObservationReadings`]), the rows of the `.sfmr` columns flagged by
//! `has_observation_readings`.

pub use sfmtool_sfmr_format::{
    ObservationReading, ObservationReadingOptions, ReadingSampler, ReadingWindow,
};

use crate::camera::sampler::{Sampler, SamplerChoice};
use crate::patch::normal_refine::PatchWindow;
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

    /// The rows a writer that read some observations again writes, under
    /// `options`: each row it read (`Some`), and for each it did not read
    /// (`None`, as where a point has no patch or a photograph is not to hand)
    /// the row `stored` holds where `stored` stands under the same options,
    /// and a row with nothing measured where it does not or there is none, so
    /// no column mixes readings taken under different options.
    ///
    /// # Panics
    ///
    /// Panics if `stored` is not parallel to `read`.
    pub fn merge_read(
        stored: Option<&ObservationReadings>,
        read: Vec<Option<ObservationReading>>,
        options: ObservationReadingOptions,
    ) -> Self {
        let carried = stored.filter(|s| s.options == options);
        if let Some(s) = carried {
            assert_eq!(s.rows.len(), read.len(), "stored rows must be parallel");
        }
        let rows = read
            .into_iter()
            .enumerate()
            .map(|(j, row)| {
                row.or_else(|| carried.map(|s| s.rows[j]))
                    .unwrap_or(ObservationReading::NOT_MEASURED)
            })
            .collect();
        Self { rows, options }
    }
}

/// The reading options of renders at `resolution` made with `sampler`, their
/// scores read over `window`: the default self-similarity reading
/// ([`SelfSimilarityParams::default`], [`FLAT_FLOOR`]), which is the one every
/// reading in this crate takes, and the sampler rule's threshold where
/// `sampler` is the rule.
pub fn observation_reading_options(
    sampler: SamplerChoice,
    resolution: usize,
    window: PatchWindow,
) -> ObservationReadingOptions {
    let params = SelfSimilarityParams::default();
    let reading_sampler = match sampler {
        SamplerChoice::PerView { .. } => ReadingSampler::PerView,
        SamplerChoice::Fixed(Sampler::Bilinear) => ReadingSampler::Bilinear,
        SamplerChoice::Fixed(Sampler::BilinearMip) => ReadingSampler::BilinearMip,
        SamplerChoice::Fixed(Sampler::Anisotropic) => ReadingSampler::Anisotropic,
    };
    let (score_window, score_window_sigma) = match window {
        PatchWindow::Uniform => (ReadingWindow::Uniform, None),
        PatchWindow::Gaussian { sigma } => (ReadingWindow::Gaussian, Some(sigma)),
        PatchWindow::GaussianDisk { sigma } => (ReadingWindow::GaussianDisk, Some(sigma)),
    };
    ObservationReadingOptions {
        resolution: resolution as u32,
        sampler: reading_sampler,
        score_window,
        score_window_sigma,
        max_radius: params.max_radius,
        flat_floor: FLAT_FLOOR,
        noise: params.noise,
        relative_tolerance: params.relative_tolerance,
        anisotropic_threshold: sampler.anisotropic_threshold(),
    }
}
