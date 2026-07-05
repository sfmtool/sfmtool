// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! D-optimal view-subset selection for the refinement basis (see
//! `specs/core/patch-normal-refine-view-subset.md`).
//!
//! [`refine_patch_normal`](super::refine_patch_normal) estimates a **2-DOF**
//! surface normal, which a handful of well-chosen views over-determine; on an
//! expanded (`select_views`) set the surplus views mostly inflate the
//! per-candidate render cost. The only term of the plane-induced homography
//! `H = R − t·nᵀ/d` carrying the normal is the rank-1 `t·nᵀ/d`, so a view's
//! sensitivity to `n` scales with how *obliquely* it sees the surfel — a
//! near-frontal view (`v̂·n ≈ 1`) is nearly stationary in `n`. Two DOF need that
//! obliquity spread across **azimuth** around the normal, or one tilt direction
//! stays loose. [`select_refine_subset`] therefore treats the pick as a
//! D-optimal experimental design over each view's tangent-plane information
//! vector: anchor on the least-oblique (sharpest-appearance) view, then greedily
//! add the view maximising the information-matrix determinant. Pure geometry —
//! no rendering — so its cost is negligible against the renders it saves.

use nalgebra::{Matrix2, Vector2, Vector3};

use crate::patch::cloud::OrientedPatch;

use super::parameterization::tangent_basis;

/// A perfectly frontal view (`sinθ` below this) carries no tangent direction;
/// its information vector is zero.
const MIN_TANGENT_NORM: f64 = 1e-6;

/// Select the (at most) `k` most normal-informative views of `patch` — the
/// D-optimal refinement basis of `specs/core/patch-normal-refine-view-subset.md`.
///
/// `view_dirs` holds the unit surface→camera direction per view (the caller's
/// full `views` order); the returned indices index into it, ascending. Returns
/// **all** indices when the cap is a no-op (`k == 0`, `m ≤ k`, or the point is
/// at infinity — its normal is fixed, refinement skips it) or when no
/// front-facing view exists to anchor on. Deterministic: greedy ties break on
/// the lowest index.
pub(super) fn select_refine_subset(
    patch: &OrientedPatch,
    view_dirs: &[Vector3<f64>],
    k: u32,
) -> Vec<usize> {
    let m = view_dirs.len();
    let all = || (0..m).collect();
    if k == 0 || m <= k as usize || patch.w == 0.0 {
        return all();
    }
    let n = patch.normal();
    let (t1, t2) = tangent_basis(&n);

    // Per-view tangent geometry: cosθ (obliquity) and the 2-D information
    // vector `wᵢ = sinθᵢ·ûᵢ` in the (t1, t2) tangent basis of the normal; its
    // outer product is the view's contribution to the 2×2 information matrix.
    // A back-facing view (cosθ ≤ 0; shouldn't occur in a vetted set) is
    // excluded from selection and carries no information.
    let mut cos = vec![f64::NEG_INFINITY; m];
    let mut w = vec![Vector2::zeros(); m];
    for (i, d) in view_dirs.iter().enumerate() {
        let c = d.dot(&n).clamp(-1.0, 1.0);
        if c <= 0.0 {
            continue;
        }
        cos[i] = c;
        let g = d - n * c; // tangent projection; ‖g‖ = sinθ
        if g.norm() > MIN_TANGENT_NORM {
            w[i] = Vector2::new(g.dot(&t1), g.dot(&t2));
        }
    }

    // Anchor: the least-oblique view — a clean, low-foreshortening appearance
    // anchor so the consensus reference the subset fuses stays sharp.
    let Some(anchor) = (0..m)
        .filter(|&i| cos[i].is_finite())
        .max_by(|&a, &b| cos[a].total_cmp(&cos[b]))
    else {
        return all(); // every view back-facing: nothing selectable, keep all
    };
    let mut selected = vec![false; m];
    selected[anchor] = true;
    let mut m_sel: Matrix2<f64> = w[anchor] * w[anchor].transpose();

    // Greedy D-optimal fill: add the view that most enlarges the information
    // volume det(M + wᵢwᵢᵀ) — naturally favouring oblique views azimuthally
    // complementary to those already chosen.
    for _ in 1..k as usize {
        let mut best: Option<(usize, f64)> = None;
        for i in 0..m {
            if selected[i] || !cos[i].is_finite() {
                continue;
            }
            let det = (m_sel + w[i] * w[i].transpose()).determinant();
            if best.is_none_or(|(_, d)| det > d) {
                best = Some((i, det));
            }
        }
        let Some((i, _)) = best else {
            break; // fewer than k front-facing views: keep what we have
        };
        selected[i] = true;
        m_sel += w[i] * w[i].transpose();
    }

    // No conditioning fallback to "all views": conditioning is a property of the
    // view-direction *geometry*, not the count. The greedy above already returns
    // the best-conditioned K available, and if that still leaves one tilt DOF
    // loose (a degenerate single-azimuth-arc point), the full view set is no
    // better conditioned — inflating to all views would only add render cost
    // without constraining the loose DOF, which the fronto-parallel prior
    // resolves at refine time exactly as for any low-parallax point. (Photometric
    // robustness — the one thing more views genuinely buy — is an orthogonal axis
    // addressed by ZNCC-weighting the pick, not by view count; see the spec's
    // deferred follow-up.)
    (0..m).filter(|&i| selected[i]).collect()
}

#[cfg(test)]
mod tests {
    use nalgebra::Point3;

    use super::*;

    /// Finite test patch at the origin with normal +z.
    fn patch() -> OrientedPatch {
        OrientedPatch::from_center_normal(Point3::origin(), Vector3::z(), Vector3::y(), [0.5, 0.5])
    }

    /// Unit view direction tilted `theta` off +z toward azimuth `phi` (both
    /// radians) — obliquity `theta`, tangent azimuth `phi`.
    fn dir(theta: f64, phi: f64) -> Vector3<f64> {
        Vector3::new(
            theta.sin() * phi.cos(),
            theta.sin() * phi.sin(),
            theta.cos(),
        )
    }

    #[test]
    fn noop_cases_return_all_views() {
        let dirs: Vec<_> = (0..4)
            .map(|i| dir(0.4, i as f64 * std::f64::consts::FRAC_PI_2))
            .collect();
        let all: Vec<usize> = (0..dirs.len()).collect();
        // k == 0 (disabled) and m <= k both keep every view.
        assert_eq!(select_refine_subset(&patch(), &dirs, 0), all);
        assert_eq!(select_refine_subset(&patch(), &dirs, 4), all);
        assert_eq!(select_refine_subset(&patch(), &dirs, 7), all);
        // A point at infinity has a fixed normal — nothing to subset.
        let inf = OrientedPatch::from_infinity_direction(
            Point3::new(0.0, 0.0, 1.0),
            Vector3::y(),
            [0.02, 0.02],
        );
        assert_eq!(select_refine_subset(&inf, &dirs, 2), all);
    }

    #[test]
    fn picks_azimuth_spread_oblique_views_over_frontal_cluster() {
        // Ten near-frontal views clustered at one azimuth (nearly no tangent
        // information) plus three oblique views spread 120° apart in azimuth.
        // The greedy pick must take the azimuthally-complementary oblique views,
        // not the highest-cosθ cluster.
        let mut dirs: Vec<_> = (0..10).map(|i| dir(0.01 + 0.001 * i as f64, 0.0)).collect();
        let oblique: Vec<usize> = (0..3)
            .map(|j| {
                dirs.push(dir(0.7, j as f64 * 2.0 * std::f64::consts::FRAC_PI_3));
                dirs.len() - 1
            })
            .collect();
        let sel = select_refine_subset(&patch(), &dirs, 4);
        assert_eq!(sel.len(), 4);
        for &i in &oblique {
            assert!(sel.contains(&i), "oblique view {i} missing from {sel:?}");
        }
        // The fourth slot is the anchor — the least-oblique cluster view.
        assert!(
            sel.contains(&0),
            "anchor (least-oblique) missing from {sel:?}"
        );
    }

    #[test]
    fn anchor_is_the_least_oblique_view() {
        // Two strongly oblique views 90° apart carry nearly all the information
        // (so the k = 3 subset keeps its conditioning); view 2 is the most
        // frontal and must be selected as the appearance anchor regardless of
        // its (tiny) information contribution.
        let dirs = vec![
            dir(0.8, 0.0),
            dir(0.1, 2.0),
            dir(0.05, 4.0), // least oblique — the anchor
            dir(0.8, std::f64::consts::FRAC_PI_2),
            dir(0.1, 3.0),
            dir(0.1, 5.0),
        ];
        let sel = select_refine_subset(&patch(), &dirs, 3);
        assert_eq!(
            sel,
            vec![0, 2, 3],
            "anchor (view 2) + the two oblique views"
        );
    }

    #[test]
    fn large_view_set_is_capped_to_k() {
        // A large view set is always capped to the best K — there is no
        // fall-back-to-all keyed on view count (the original bug) or on subset
        // conditioning: the greedy returns the best K available regardless.
        let dirs: Vec<_> = (0..24)
            .map(|i| dir(0.4, i as f64 * std::f64::consts::TAU / 24.0))
            .collect();
        let sel = select_refine_subset(&patch(), &dirs, 5);
        assert_eq!(sel.len(), 5, "large set must be capped to K, got {sel:?}");
    }

    #[test]
    fn degenerate_arc_returns_best_k_without_falling_back() {
        // All views crowded into a narrow azimuth arc: the tilt DOF orthogonal to
        // the arc is under-constrained no matter how many views are used, so there
        // is nothing to gain by inflating to all — the selection keeps the best K
        // (the fronto-parallel prior resolves the loose DOF at refine time).
        let dirs: Vec<_> = (0..12).map(|i| dir(0.5, 0.02 * i as f64)).collect();
        let sel = select_refine_subset(&patch(), &dirs, 5);
        assert_eq!(
            sel.len(),
            5,
            "degenerate geometry still returns best K, got {sel:?}"
        );
    }

    #[test]
    fn back_facing_views_are_never_selected() {
        let dirs = vec![
            dir(0.1, 0.0),
            -dir(0.5, 1.0), // back-facing
            dir(0.6, 0.5),
            dir(0.6, 0.5 + std::f64::consts::FRAC_PI_2),
            -Vector3::z(), // back-facing
        ];
        let sel = select_refine_subset(&patch(), &dirs, 3);
        assert_eq!(sel, vec![0, 2, 3]);
    }

    #[test]
    fn selection_is_deterministic() {
        let dirs: Vec<_> = (0..12)
            .map(|i| dir(0.1 + 0.05 * (i % 5) as f64, i as f64 * 0.7))
            .collect();
        let first = select_refine_subset(&patch(), &dirs, 5);
        for _ in 0..10 {
            assert_eq!(select_refine_subset(&patch(), &dirs, 5), first);
        }
        // Sorted ascending (a stable gather order for the caller).
        let mut sorted = first.clone();
        sorted.sort_unstable();
        assert_eq!(first, sorted);
    }
}
