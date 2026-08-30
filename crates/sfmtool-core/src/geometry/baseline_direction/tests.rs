// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

use super::*;

fn trim(tol_deg: f64) -> BaselineTrim {
    BaselineTrim {
        tol_rad: tol_deg.to_radians(),
        rounds: 5,
        keep_fraction: 0.6,
    }
}

/// Two centres and a cloud in front of both: the rays of one edge.
fn edge(c_i: [f64; 3], c_j: [f64; 3], points: &[[f64; 3]]) -> (Vec<f64>, Vec<f64>) {
    let mut ui = Vec::new();
    let mut uj = Vec::new();
    for p in points {
        let a = Vector3::new(p[0] - c_i[0], p[1] - c_i[1], p[2] - c_i[2]).normalize();
        let b = Vector3::new(p[0] - c_j[0], p[1] - c_j[1], p[2] - c_j[2]).normalize();
        ui.extend_from_slice(&[a.x, a.y, a.z]);
        uj.extend_from_slice(&[b.x, b.y, b.z]);
    }
    (ui, uj)
}

/// A spread of points in front of a pair on the x axis.
///
/// Wide in y on purpose: the coplanarity normals of a pair separated along x
/// are perpendicular to x, and a cloud thin in y puts them all near one
/// direction, which leaves the null space nearly two-dimensional and the
/// baseline barely determined.
fn cloud(n: usize) -> Vec<[f64; 3]> {
    (0..n)
        .map(|k| {
            let t = k as f64 / n as f64;
            [
                -2.0 + 4.0 * t,
                -6.0 + 12.0 * ((7 * k) % 11) as f64 / 11.0,
                -4.0 - 4.0 * ((5 * k) % 13) as f64 / 13.0,
            ]
        })
        .collect()
}

#[test]
fn the_direction_is_the_baseline_the_centres_actually_have() {
    let pts = cloud(40);
    let (ui, uj) = edge([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], &pts);
    let out = baseline_directions(&ui, &uj, &[0, pts.len()], trim(0.05));
    let b = out[0].expect("an edge this wide states a direction");
    assert!(
        (b.direction[0] - 1.0).abs() < 1e-9,
        "direction {:?}",
        b.direction
    );
    assert!(b.direction[1].abs() < 1e-9 && b.direction[2].abs() < 1e-9);
    assert_eq!(b.n_rows, pts.len());
    assert_eq!(b.n_used, pts.len());
    assert!(b.residual_median_rad < 1e-12);
    assert_eq!(b.cheiral_fraction, 1.0);
}

#[test]
fn the_sign_is_the_cheiral_one() {
    let pts = cloud(40);
    // Swapping the two centres has to swap the direction, not keep it.
    let (ui, uj) = edge([1.0, 0.0, 0.0], [0.0, 0.0, 0.0], &pts);
    let out = baseline_directions(&ui, &uj, &[0, pts.len()], trim(0.05));
    let b = out[0].expect("stated");
    assert!(
        (b.direction[0] + 1.0).abs() < 1e-9,
        "direction {:?}",
        b.direction
    );
    assert_eq!(b.cheiral_fraction, 1.0);
}

#[test]
fn an_edge_with_no_parallax_states_nothing() {
    // Both centres in the same place: every cross product is noise, and no row
    // clears the bound.
    let pts = cloud(40);
    let (ui, uj) = edge([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], &pts);
    let out = baseline_directions(&ui, &uj, &[0, pts.len()], trim(0.05));
    assert!(out[0].is_none());
}

#[test]
fn an_edge_with_fewer_than_three_rows_past_the_bound_states_nothing() {
    let pts = cloud(6);
    let (ui, uj) = edge([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], &pts);
    // A bound above every row's parallax leaves nothing.
    let out = baseline_directions(&ui, &uj, &[0, pts.len()], trim(80.0));
    assert!(out[0].is_none());
}

#[test]
fn rows_inside_the_bound_are_dropped_not_weighted() {
    let mut pts = cloud(40);
    // Ten points at a great distance carry almost no parallax.
    for k in 0..10 {
        pts.push([k as f64 * 0.01, 0.0, -1.0e7]);
    }
    let n = pts.len();
    let (ui, uj) = edge([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], &pts);
    let out = baseline_directions(&ui, &uj, &[0, n], trim(0.05));
    let b = out[0].expect("stated");
    assert_eq!(b.n_rows, n);
    assert!(b.n_used < n, "the distant rows have to be dropped");
    assert!((b.direction[0] - 1.0).abs() < 1e-9);
    // The widest parallax is read over every row, the dropped ones included.
    assert!(b.parallax_max_deg > b.parallax_median_deg);
}

#[test]
fn the_trim_throws_out_the_rows_that_contradict_the_rest() {
    let pts = cloud(40);
    let (ui, mut uj) = edge([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], &pts);
    let clean = baseline_directions(&ui, &uj, &[0, pts.len()], trim(0.05))[0].unwrap();
    assert!(clean.residual_median_rad < 1e-12);
    // Six rows mismatched: their second ray points at a different point of the
    // cloud, which is the outlier the trim exists for.
    for r in [3usize, 9, 14, 22, 30, 37] {
        let other = (r + 17) % pts.len();
        for c in 0..3 {
            uj[3 * r + c] = uj[3 * other + c];
        }
    }
    let out = baseline_directions(&ui, &uj, &[0, pts.len()], trim(0.05));
    let b = out[0].expect("stated");
    assert!(
        (b.direction[0] - 1.0).abs() < 1e-9,
        "the trim has to survive the contradiction, got {:?}",
        b.direction
    );
    // The trimmed rows still count toward the reported residual, so the median
    // over every used row rises even though the direction did not move.
    assert!(b.residual_median_rad < 1e-12);
    assert_eq!(b.n_used, pts.len());
}

#[test]
fn a_whole_graph_is_one_call() {
    let pts = cloud(30);
    let (a_i, a_j) = edge([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], &pts);
    let (b_i, b_j) = edge([0.0, 0.0, 0.0], [0.0, 1.0, 0.0], &pts);
    // A third edge with no parallax at all.
    let (c_i, c_j) = edge([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], &pts);
    let mut ui = a_i;
    ui.extend_from_slice(&b_i);
    ui.extend_from_slice(&c_i);
    let mut uj = a_j;
    uj.extend_from_slice(&b_j);
    uj.extend_from_slice(&c_j);
    let n = pts.len();
    let out = baseline_directions(&ui, &uj, &[0, n, 2 * n, 3 * n], trim(0.05));
    assert_eq!(out.len(), 3);
    assert!((out[0].unwrap().direction[0] - 1.0).abs() < 1e-9);
    assert!((out[1].unwrap().direction[1] - 1.0).abs() < 1e-9);
    assert!(out[2].is_none());
}

#[test]
fn the_batch_repeats_itself_bit_for_bit() {
    let pts = cloud(64);
    let (ui, uj) = edge([0.0, 0.0, 0.0], [1.0, 0.3, -0.2], &pts);
    let a = baseline_directions(&ui, &uj, &[0, pts.len()], trim(0.05));
    let b = baseline_directions(&ui, &uj, &[0, pts.len()], trim(0.05));
    assert_eq!(a, b);
}

#[test]
fn the_trim_count_rounds_half_to_even() {
    // Python's `round`, which the trim count is taken with.
    assert_eq!(round_half_even(0.5), 0.0);
    assert_eq!(round_half_even(1.5), 2.0);
    assert_eq!(round_half_even(2.5), 2.0);
    assert_eq!(round_half_even(3.5), 4.0);
    assert_eq!(round_half_even(2.4), 2.0);
    assert_eq!(round_half_even(2.6), 3.0);
}
