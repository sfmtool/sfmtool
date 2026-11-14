// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! What [`pairwise_flow_normal`](super::pairwise_flow_normal) did: every
//! round, with its crops, its pairs' fits at each radius, the combination at
//! each radius and its answer, and the outcome of the last round.
//!
//! The angles and σ are `f64` degrees and the radii `f64` patch half-extents,
//! as the fits that produce them run in `f64`; the report holds a few hundred
//! numbers per round, so their size does not matter.

/// What [`pairwise_flow_normal`](super::pairwise_flow_normal) did, carried by
/// [`NormalEstimate::PairwiseFlow`](super::super::NormalEstimate::PairwiseFlow).
///
/// The outcome, chosen radius, normal and σ are those of the last round run.
/// A last round with no answer leaves the given normal unchanged, even where
/// an earlier round had one.
#[derive(Debug, Clone, PartialEq)]
pub struct PairwiseFlowRounds {
    /// Every round, in the order run.
    pub rounds: Vec<PairwiseFlowRound>,
    /// The last round's outcome.
    pub outcome: PairwiseFlowOutcome,
    /// The last round's chosen radius, in patch half-extents, where the
    /// patch was turned; `None` under [`PairwiseFlowOutcome::Undetermined`]
    /// and [`PairwiseFlowOutcome::Unstable`].
    pub chosen_radius: Option<f64>,
    /// The normal the patch was turned to, facing as the given normal faces;
    /// `None` where it was not turned.
    pub normal: Option<[f64; 3]>,
    /// That normal's σ, degrees, largest axis.
    pub sigma_deg: Option<f64>,
    /// Degrees from the given normal to [`Self::normal`].
    pub turn_deg: Option<f64>,
}

/// One round: the crops rendered on one plane through the point, the flow
/// and fits of its pairs, and its answer.
#[derive(Debug, Clone, PartialEq)]
pub struct PairwiseFlowRound {
    /// The unit normal of the plane the crops were rendered on, facing the
    /// point's views.
    pub plane_normal: [f64; 3],
    /// World units per grid sample; `NaN` where no view saw the plane.
    pub sample_spacing: f64,
    /// Grid samples along one side of the patch: by the rule, or fewer where
    /// the crop cap coarsened the grid; `NaN` where no view saw the plane.
    pub samples_per_patch: f64,
    /// The rendered crops' half-width, grid samples; 0 where no view saw the
    /// plane.
    pub crop_half_width: u32,
    /// One per observation of the track, in its order.
    pub crops: Vec<PairwiseFlowCrop>,
    /// The pairs of rendered crops read: every pair, or
    /// [`PairwiseFlowNormalOptions::max_pairs`](super::PairwiseFlowNormalOptions::max_pairs)
    /// of them chosen as the options say; in observation order.
    pub pairs: Vec<PairwiseFlowPair>,
    /// One per entry of
    /// [`PairwiseFlowNormalOptions::radii`](super::PairwiseFlowNormalOptions::radii).
    pub radii: Vec<PairwiseFlowRadius>,
    /// The round's outcome, by the rule, on this round alone.
    pub outcome: PairwiseFlowOutcome,
    /// The radius the round read its answer at, patch half-extents: the
    /// support rule's choice, or a single pair's radius. Set also where that
    /// answer is then refused as too far from the mean viewing direction.
    pub chosen_radius: Option<f64>,
    /// The round's answer, facing the point's views; `None`: no answer by
    /// the rule.
    pub normal: Option<[f64; 3]>,
    /// The answer's σ, degrees, largest axis.
    pub sigma_deg: Option<f64>,
    /// Degrees between the normal read at [`Self::chosen_radius`] and the
    /// mean viewing direction of the point's views.
    pub from_mean_view_deg: Option<f64>,
    /// The next round's plane: the answer, else the fallback; `None` where
    /// no round follows, because the step stopped here or this was the last.
    pub next_plane: Option<[f64; 3]>,
}

/// How a round ended, and how the step ended.
///
/// Only [`Self::Turned`] and [`Self::SinglePair`] turn the patch; the other
/// two read back the track as given, and neither is a refusal.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PairwiseFlowOutcome {
    /// The patch was turned to the combined normal at the chosen radius: the
    /// largest radius with at least two determined pairs whose mean χ² is at
    /// most `max_mean_chi2`, its normal within
    /// `max_normal_from_mean_view_deg` of the mean viewing direction.
    Turned,
    /// A point with exactly two observations, one pair, read in one round at
    /// `single_pair_radius` with no agreement to test: its σ is at most
    /// `max_sigma_deg` and its normal within `max_normal_from_mean_view_deg`
    /// of the mean viewing direction, and the patch was turned to it.
    SinglePair,
    /// Too little evidence: no radius has two determined pairs (each with σ
    /// at most `max_sigma_deg`), so no agreement can be tested; or a point
    /// with exactly two observations has no determined fit at
    /// `single_pair_radius`. The track as given. A point with more
    /// observations follows this rule in every round, even a round whose
    /// too-oblique views leave it one rendered pair.
    Undetermined,
    /// The evidence does not settle on a plane the views can see: some
    /// radius has two or more determined pairs but at every such radius they
    /// disagree (mean χ² over `max_mean_chi2`), or the answer is more than
    /// `max_normal_from_mean_view_deg` from the mean viewing direction. The
    /// track as given.
    Unstable,
}

/// One observation's crop in a round.
#[derive(Debug, Clone, PartialEq)]
pub enum PairwiseFlowCrop {
    /// Rendered on the round's plane.
    Rendered {
        /// The observation's image.
        image: u32,
        /// Degrees between the view's line of sight and the round's plane
        /// normal, at the point.
        angle_deg: f64,
        /// How many grid samples one of the view's own pixels spans at the
        /// point, along its coarser axis on the plane: over 1 the view was
        /// upsampled, under 1 it was read through the photograph's pyramid.
        samples_per_pixel: f64,
        /// The share of the crop's samples that fell on the photograph.
        coverage: f64,
    },
    /// More than `max_view_angle_deg` from the round's plane normal, and so
    /// left out of the round.
    TooOblique {
        /// The observation's image.
        image: u32,
        /// Degrees between the view's line of sight and the round's plane
        /// normal, at the point.
        angle_deg: f64,
    },
    /// Not `in`, a second row in one image, a camera centre at the point, or
    /// a point the camera's model does not project.
    NotAttempted,
}

/// One pair of rendered crops in a round.
#[derive(Debug, Clone, PartialEq)]
pub struct PairwiseFlowPair {
    /// The two observations, the flow read from the first's crop to the
    /// second's.
    pub observations: [usize; 2],
    /// Their images.
    pub images: [u32; 2],
    /// Degrees between the two views, from the point.
    pub angle_deg: f64,
    /// The level of the crops' pyramids the pair is read at.
    pub level: u32,
    /// The half-width of the square read, samples at that level.
    pub half_width: u32,
    /// One per radius; `None`: past the square read, no flow (a square
    /// narrower than the flow's `patch_size`), no sample carrying
    /// information, or a tilt the disc does not fix.
    pub fits: Vec<Option<PairwiseFlowFit>>,
}

/// One pair's plane at one radius.
#[derive(Debug, Clone, PartialEq)]
pub struct PairwiseFlowFit {
    /// The unit normal, facing the pair's cameras.
    pub normal: [f64; 3],
    /// Degrees, largest axis: the larger of the two below.
    pub sigma_deg: f64,
    /// From the texture-weighted Hessian, degrees, largest axis.
    pub sigma_texture_deg: f64,
    /// From the fit's residuals, degrees, largest axis.
    pub sigma_residual_deg: f64,
    /// The pair's free shift, grid samples.
    pub offset: [f64; 2],
    /// The Tukey-weighted root mean square residual over the samples that
    /// carry information, grid samples.
    pub rms_residual: f64,
    /// The variance of the photometric residual over the disc, the crop the
    /// flow ends in read at the end of each sample's flow less the crop it
    /// starts in, in the flow's grey units squared (black 0, white 1),
    /// floored at 8-bit quantization.
    pub residual_variance: f64,
    /// [`Self::sigma_deg`] is at most `max_sigma_deg`, so the fit enters the
    /// combination and χ².
    pub determined: bool,
    /// Within the agreement window of the radius's medoid, so the fit is in
    /// the combined normal.
    pub combined: bool,
}

/// The pairs combined at one radius.
#[derive(Debug, Clone, PartialEq)]
pub struct PairwiseFlowRadius {
    /// Patch half-extents.
    pub radius: f64,
    /// The combined normal, facing the point's views; `None`: no determined
    /// pair.
    pub normal: Option<[f64; 3]>,
    /// The combined normal's σ, degrees; `NaN` where there is none.
    pub sigma_deg: f64,
    /// The mean χ² of the determined pairs' normals against the combined
    /// one; `NaN` with fewer than two.
    pub mean_chi2: f64,
    /// Pairs with a fit at this radius.
    pub pairs_fitted: usize,
    /// Pairs whose fit is determined.
    pub pairs_determined: usize,
    /// Pairs in the combined normal.
    pub pairs_combined: usize,
}

impl PairwiseFlowRounds {
    /// The last round run.
    pub fn last(&self) -> Option<&PairwiseFlowRound> {
        self.rounds.last()
    }
}

impl PairwiseFlowRound {
    /// The radius row at `radius`, patch half-extents.
    pub fn radius(&self, radius: f64) -> Option<&PairwiseFlowRadius> {
        self.radii.iter().find(|r| r.radius == radius)
    }

    /// How many crops were rendered.
    pub fn rendered(&self) -> usize {
        self.crops
            .iter()
            .filter(|c| matches!(c, PairwiseFlowCrop::Rendered { .. }))
            .count()
    }
}

impl std::fmt::Display for PairwiseFlowRounds {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let n = self.rounds.len();
        write!(f, "{n} round{}", if n == 1 { "" } else { "s" })?;
        let Some(last) = self.last() else {
            return write!(f, ", normal left unchanged: undetermined");
        };
        let pairs = last.pairs.len();
        let chosen = last
            .chosen_radius
            .and_then(|r| last.radius(r).map(|row| (r, row)));
        match self.outcome {
            PairwiseFlowOutcome::Turned => {
                if let Some((r, row)) = chosen {
                    write!(
                        f,
                        "; {} of {pairs} pairs agree at {r} half-extents (\u{3c7}\u{b2} {:.1})",
                        row.pairs_combined, row.mean_chi2
                    )?;
                }
                if let Some(sigma) = self.sigma_deg {
                    write!(f, ", \u{3c3} {sigma:.2}\u{b0}")?;
                }
                Ok(())
            }
            PairwiseFlowOutcome::SinglePair => {
                if let Some(r) = self.chosen_radius {
                    write!(f, "; one pair at {r} half-extents")?;
                }
                if let Some(sigma) = self.sigma_deg {
                    write!(f, ", \u{3c3} {sigma:.2}\u{b0}")?;
                }
                write!(f, ", single pair")
            }
            PairwiseFlowOutcome::Undetermined => {
                write!(f, "; {pairs} pair{}", if pairs == 1 { "" } else { "s" })?;
                write!(f, ", no radius with two determined pairs")?;
                write!(f, ", normal left unchanged: undetermined")
            }
            PairwiseFlowOutcome::Unstable => {
                write!(f, "; {pairs} pair{}", if pairs == 1 { "" } else { "s" })?;
                match (last.chosen_radius, last.from_mean_view_deg) {
                    (Some(r), Some(away)) => write!(
                        f,
                        ", the answer at {r} half-extents is {away:.0}\u{b0} from the mean \
                         viewing direction"
                    )?,
                    _ => {
                        let least = last
                            .radii
                            .iter()
                            .filter(|row| row.pairs_determined >= 2 && !row.mean_chi2.is_nan())
                            .min_by(|a, b| a.mean_chi2.total_cmp(&b.mean_chi2));
                        match least {
                            Some(row) => write!(
                                f,
                                ", no radius where they agree (least \u{3c7}\u{b2} {:.1} at {} \
                                 half-extents)",
                                row.mean_chi2, row.radius
                            )?,
                            None => write!(f, ", no radius where they agree")?,
                        }
                    }
                }
                write!(f, ", normal left unchanged: unstable")
            }
        }
    }
}
