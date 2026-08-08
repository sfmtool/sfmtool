// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

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
