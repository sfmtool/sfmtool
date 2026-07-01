// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! View-obliquity priors for normal refinement — two independent uses of the same
//! per-view quantity `cos θᵢ = v̂ᵢ·n` (the cosine between a view's
//! surface→camera direction `v̂ᵢ` and the candidate normal `n`):
//!
//! - **(A) consensus view-weight** ([`fill_kept_obliquity_priors`]): a multiplicative
//!   prior `|cos θ|^power` folded into the robust IRLS view weights, so an oblique
//!   view contributes less to the consensus template and score. A soft, continuous
//!   version of a hard grazing-view cut. On a point whose views span a range of
//!   obliquities it down-weights the grazing ones; on a low-parallax point (all
//!   views near-collinear, hence near-equal obliquity) it renormalizes away — see
//!   (B) for that case.
//! - **(B) fronto-parallel prior** ([`fronto_prior`]): an additive reward
//!   `weight·mean_v cos²θ` on the candidate normal itself. Its maximizer is the
//!   normal facing the observing cameras (fronto-parallel), so it supplies the
//!   constraint the data can't when `Φ` is flat — the narrow-baseline degeneracy
//!   where tilting the plane shifts every view's patch identically. It only tips
//!   near-ties: wherever real parallax curves `Φ`, the photoconsistency term
//!   dominates the small prior.
//!
//! Both are opt-in (weight/power `0` ⇒ no effect, and (A) then passes `None` so the
//! consensus runs byte-for-byte as before).

use nalgebra::Vector3;

/// Floor on a per-view obliquity prior, so an exactly edge-on view (`cos θ = 0`)
/// keeps a vanishing but nonzero weight rather than zeroing a row and risking an
/// all-zero normalization.
pub(super) const OBLIQUITY_PRIOR_FLOOR: f64 = 1e-6;

/// Fill `buf` with the multiplicative obliquity prior `|v̂·n|^power` per **kept**
/// view (in `kept` order), for the consensus view-weight (A); returns whether the
/// prior is active (`power != 0`). When inactive `buf` is left cleared and the
/// caller passes `None` to the consensus (which then runs exactly as before,
/// uniform init + pure IRLS). Filling a caller-owned buffer keeps a candidate
/// evaluation allocation-free once the buffer has warmed up, matching the
/// [`ConsensusScratch`](super::consensus::ConsensusScratch) discipline.
///
/// `view_dirs` holds the unit surface→camera direction per view in the full
/// `views` order; `kept` indexes into it (matching the `xs` view order the
/// consensus reads). Each prior is floored at [`OBLIQUITY_PRIOR_FLOOR`].
pub(super) fn fill_kept_obliquity_priors(
    buf: &mut Vec<f64>,
    view_dirs: &[Vector3<f64>],
    kept: &[usize],
    n: &Vector3<f64>,
    power: f64,
) -> bool {
    buf.clear();
    if power == 0.0 {
        return false;
    }
    buf.extend(kept.iter().map(|&vi| {
        view_dirs[vi]
            .dot(n)
            .abs()
            .powf(power)
            .max(OBLIQUITY_PRIOR_FLOOR)
    }));
    true
}

/// The additive fronto-parallel prior `weight · mean_v (v̂·n)²` on a candidate
/// normal (B). `0` when `weight == 0` (or no views). Squared, so it is sign-
/// agnostic (a back-facing candidate is penalized identically to its front-facing
/// mirror); its maximum is the normal aligned with the dominant viewing direction.
pub(super) fn fronto_prior(view_dirs: &[Vector3<f64>], n: &Vector3<f64>, weight: f64) -> f64 {
    if weight == 0.0 || view_dirs.is_empty() {
        return 0.0;
    }
    let s: f64 = view_dirs
        .iter()
        .map(|d| {
            let c = d.dot(n);
            c * c
        })
        .sum();
    weight * s / view_dirs.len() as f64
}
