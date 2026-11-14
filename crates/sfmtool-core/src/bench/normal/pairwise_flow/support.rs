// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The pairs' normals combined at each radius, the support chosen from the
//! combinations, and a round's answer.
//!
//! - [`combine`]: at one radius, the medoid of the determined pairs' normals,
//!   the σ-weighted mean of those within the agreement window of it, and the
//!   mean χ² of the determined pairs against that mean.
//! - [`choose_support`]: the largest radius with at least two determined
//!   pairs whose mean χ² is at most the bar, else why there is none.
//! - [`decide`]: a round's outcome, answer and next plane from its
//!   combinations, with the bar on the mean viewing direction and the
//!   fallback.
//!
//! Everything is `f64`: a handful of pairs' normals per radius, from fits
//! that run in `f64`, with no cost to save.

use nalgebra::Vector3;

use super::report::PairwiseFlowOutcome;
use super::PairwiseFlowNormalOptions;

/// The angle between two unit vectors, degrees, accurate at small angles.
pub(crate) fn angle_deg(a: Vector3<f64>, b: Vector3<f64>) -> f64 {
    a.cross(&b).norm().atan2(a.dot(&b)).to_degrees()
}

/// One pair's normal at one radius, facing the point's views, and its σ in
/// degrees.
pub(crate) type PairNormal = (Vector3<f64>, f64);

/// The pairs combined at one radius.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Combination {
    /// The σ-weighted mean of the pairs within the agreement window of the
    /// medoid; `None` where no pair is determined.
    pub(crate) normal: Option<Vector3<f64>>,
    /// Its σ, the inverse square root of the summed weights, degrees; `NaN`
    /// where there is none.
    pub(crate) sigma_deg: f64,
    /// The mean over the determined pairs of their angle from the combined
    /// normal over their σ, squared; `NaN` with fewer than two.
    pub(crate) mean_chi2: f64,
    /// Pairs with a fit.
    pub(crate) fitted: usize,
    /// Pairs whose σ is at most the bar.
    pub(crate) determined: Vec<bool>,
    /// Pairs in the combined normal.
    pub(crate) combined: Vec<bool>,
}

impl Combination {
    /// How many pairs are determined.
    pub(crate) fn determined_count(&self) -> usize {
        self.determined.iter().filter(|&&d| d).count()
    }

    /// How many pairs are in the combined normal.
    pub(crate) fn combined_count(&self) -> usize {
        self.combined.iter().filter(|&&c| c).count()
    }
}

/// Combine `fits`, one per pair (`None` where the pair has no fit at this
/// radius), counting a fit determined where its σ is at most `max_sigma_deg`.
///
/// The medoid is the determined normal with the least sum of angles to the
/// other determined normals, the first in pair order on a tie. A determined
/// pair joins the combination where its normal is within
/// `agreement_sigmas · σ + agreement_margin_deg` of the medoid, `σ` its own;
/// the combined normal is the mean of those normals weighted by `1/σ²`,
/// normalised.
pub(crate) fn combine(
    fits: &[Option<PairNormal>],
    max_sigma_deg: f64,
    options: &PairwiseFlowNormalOptions,
) -> Combination {
    let fitted = fits.iter().filter(|f| f.is_some()).count();
    let determined: Vec<bool> = fits
        .iter()
        .map(|f| f.is_some_and(|(_, s)| s.is_finite() && s <= max_sigma_deg))
        .collect();
    let chosen: Vec<(usize, PairNormal)> = fits
        .iter()
        .enumerate()
        .filter(|&(k, _)| determined[k])
        .filter_map(|(k, f)| f.map(|f| (k, f)))
        .collect();
    let mut combined = vec![false; fits.len()];
    let none = |combined| Combination {
        normal: None,
        sigma_deg: f64::NAN,
        mean_chi2: f64::NAN,
        fitted,
        determined: determined.clone(),
        combined,
    };
    let Some(medoid) = chosen
        .iter()
        .map(|&(_, (n, _))| {
            let spread: f64 = chosen.iter().map(|&(_, (m, _))| angle_deg(n, m)).sum();
            (spread, n)
        })
        .fold(
            None,
            |best: Option<(f64, Vector3<f64>)>, (spread, n)| match best {
                Some((b, _)) if b <= spread => best,
                _ => Some((spread, n)),
            },
        )
        .map(|(_, n)| n)
    else {
        return none(combined);
    };
    let mut sum = Vector3::zeros();
    let mut weight = 0.0;
    for &(k, (n, sigma)) in &chosen {
        let window = options.agreement_sigmas * sigma + options.agreement_margin_deg;
        if angle_deg(n, medoid) <= window {
            let w = 1.0 / (sigma * sigma);
            sum += n * w;
            weight += w;
            combined[k] = true;
        }
    }
    // The medoid is always within its own window, so the sum holds at least
    // one normal; it is zero only for normals that cancel, which no pairs
    // facing the same views give.
    if !(weight > 0.0 && sum.norm() > 0.0) {
        return none(combined);
    }
    let normal = sum.normalize();
    let mean_chi2 = if chosen.len() >= 2 {
        chosen
            .iter()
            .map(|&(_, (n, sigma))| (angle_deg(n, normal) / sigma).powi(2))
            .sum::<f64>()
            / chosen.len() as f64
    } else {
        f64::NAN
    };
    Combination {
        normal: Some(normal),
        sigma_deg: (1.0 / weight).sqrt(),
        mean_chi2,
        fitted,
        determined,
        combined,
    }
}

/// What the support rule found over a round's radii.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum Support {
    /// The index of the largest radius with at least two determined pairs
    /// whose mean χ² is at most the bar.
    Chosen(usize),
    /// No radius has two determined pairs.
    Undetermined,
    /// Some radius has two determined pairs, and at every one of them the
    /// mean χ² is over the bar.
    Unstable,
}

/// The support rule over `combinations`, one per radius in ascending order.
///
/// A radius with one determined pair is not used: its χ² over one pair would
/// say nothing about agreement.
pub(crate) fn choose_support(combinations: &[Combination], max_mean_chi2: f64) -> Support {
    let mut any = false;
    let mut chosen = None;
    for (k, c) in combinations.iter().enumerate() {
        if c.determined_count() < 2 || c.normal.is_none() {
            continue;
        }
        any = true;
        if c.mean_chi2 <= max_mean_chi2 {
            chosen = Some(k);
        }
    }
    match (chosen, any) {
        (Some(k), _) => Support::Chosen(k),
        (None, true) => Support::Unstable,
        (None, false) => Support::Undetermined,
    }
}

/// A round's answer and what follows it.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Decision {
    pub(crate) outcome: PairwiseFlowOutcome,
    /// The index of the radius the answer was read at.
    pub(crate) chosen: Option<usize>,
    /// The answer, facing the views.
    pub(crate) normal: Option<Vector3<f64>>,
    pub(crate) sigma_deg: Option<f64>,
    /// Degrees from the normal read at the chosen radius to the mean viewing
    /// direction.
    pub(crate) from_mean_view_deg: Option<f64>,
    /// The next round's plane; `None` where the step stops after this round.
    pub(crate) next_plane: Option<Vector3<f64>>,
}

/// How a round is read.
pub(crate) enum RoundKind<'a> {
    /// A point with exactly two observations: its one pair's fit at each
    /// radius, read at the radius of index `radius` with no agreement test.
    SinglePair {
        fits: &'a [Option<PairNormal>],
        radius: usize,
    },
    /// A point with more: the support rule, and the fallback, the
    /// combination at the fallback radius over every pair with a finite σ,
    /// when there is no answer.
    Several { fallback: Option<Vector3<f64>> },
}

/// Decide a round from its `combinations`, one per radius, the views' unit
/// `mean_view`, and whether a round may follow (`more`).
pub(crate) fn decide(
    combinations: &[Combination],
    kind: RoundKind<'_>,
    mean_view: Vector3<f64>,
    more: bool,
    options: &PairwiseFlowNormalOptions,
) -> Decision {
    let bar = options.max_normal_from_mean_view_deg;
    let nothing = |outcome, next_plane| Decision {
        outcome,
        chosen: None,
        normal: None,
        sigma_deg: None,
        from_mean_view_deg: None,
        next_plane,
    };
    // A candidate answer at radius `k`: turned, or refused as too far from
    // the mean viewing direction.
    let answer = |k: usize, normal: Vector3<f64>, sigma: f64, outcome| {
        let away = angle_deg(normal, mean_view);
        if away > bar {
            Decision {
                outcome: PairwiseFlowOutcome::Unstable,
                chosen: Some(k),
                normal: None,
                sigma_deg: None,
                from_mean_view_deg: Some(away),
                next_plane: None,
            }
        } else {
            Decision {
                outcome,
                chosen: Some(k),
                normal: Some(normal),
                sigma_deg: Some(sigma),
                from_mean_view_deg: Some(away),
                next_plane: more.then_some(normal),
            }
        }
    };
    match kind {
        RoundKind::SinglePair { fits, radius } => match fits.get(radius).copied().flatten() {
            Some((normal, sigma)) if sigma.is_finite() && sigma <= options.max_sigma_deg => {
                answer(radius, normal, sigma, PairwiseFlowOutcome::SinglePair)
            }
            _ => nothing(PairwiseFlowOutcome::Undetermined, None),
        },
        RoundKind::Several { fallback } => {
            let outcome = match choose_support(combinations, options.max_mean_chi2) {
                Support::Chosen(k) => {
                    let c = &combinations[k];
                    let normal = c.normal.expect("a chosen radius has a combined normal");
                    return answer(k, normal, c.sigma_deg, PairwiseFlowOutcome::Turned);
                }
                Support::Undetermined => PairwiseFlowOutcome::Undetermined,
                Support::Unstable => PairwiseFlowOutcome::Unstable,
            };
            let next = fallback
                .filter(|_| more)
                .filter(|&n| angle_deg(n, mean_view) <= bar);
            nothing(outcome, next)
        }
    }
}

#[cfg(test)]
mod tests;
