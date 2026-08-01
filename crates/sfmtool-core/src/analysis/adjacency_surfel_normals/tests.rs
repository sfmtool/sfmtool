// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

use super::*;

/// A cloud whose point 0 is the hub being fitted and whose points `1..=k` are
/// its neighbours. Only the hub carries a CSR row and only the hub is selected,
/// so anything that shows up on another point's output came from a bug.
struct Hub {
    positions: Vec<[f64; 3]>,
    offsets: Vec<u32>,
    neighbours: Vec<u32>,
    view_dirs: Vec<[f64; 3]>,
    selected: Vec<bool>,
    extra_rows: Vec<Vec<[f64; 3]>>,
}

impl Hub {
    fn new(neighbours: &[[f64; 3]]) -> Self {
        let k = neighbours.len();
        let mut positions = vec![[0.0, 0.0, 0.0]];
        positions.extend_from_slice(neighbours);
        let n = positions.len();
        let mut offsets = vec![k as u32; n + 1];
        offsets[0] = 0;
        let mut selected = vec![false; n];
        selected[0] = true;
        Self {
            positions,
            offsets,
            neighbours: (1..=k as u32).collect(),
            view_dirs: vec![[0.0, 0.0, 1.0]; n],
            selected,
            extra_rows: vec![Vec::new(); n],
        }
    }

    /// The hub's reference viewing direction.
    fn view(mut self, view: [f64; 3]) -> Self {
        self.view_dirs[0] = view;
        self
    }

    fn select(mut self, p: usize, on: bool) -> Self {
        self.selected[p] = on;
        self
    }

    fn extras(mut self, p: usize, rows: &[[f64; 3]]) -> Self {
        self.extra_rows[p] = rows.to_vec();
        self
    }

    fn fit(&self, params: &AdjacencySurfelParams) -> AdjacencySurfelNormals {
        let mut extras = ExtraNeighbours::none();
        if self.extra_rows.iter().any(|r| !r.is_empty()) {
            extras.offsets.push(0);
            for row in &self.extra_rows {
                extras.positions.extend_from_slice(row);
                extras.offsets.push(extras.positions.len() as u32);
            }
        }
        estimate_adjacency_surfel_normals(
            &self.positions,
            &self.offsets,
            &self.neighbours,
            &self.view_dirs,
            &self.selected,
            &extras,
            params,
        )
    }

    fn fit_default(&self) -> AdjacencySurfelNormals {
        self.fit(&AdjacencySurfelParams::default())
    }
}

/// Unit-radius positions at the given in-plane angles, on `z = 0`.
fn ring(angles_deg: &[f64]) -> Vec<[f64; 3]> {
    angles_deg
        .iter()
        .map(|a| {
            let r = a.to_radians();
            [r.cos(), r.sin(), 0.0]
        })
        .collect()
}

/// Eight evenly spread directions, each in the middle of its own default
/// sector — a maximally spread, isotropic neighbourhood, placed where no
/// libm's last-ulp disagreement can push a row across a sector boundary.
fn full_ring() -> Vec<[f64; 3]> {
    ring(&[22.5, 67.5, 112.5, 157.5, 202.5, 247.5, 292.5, 337.5])
}

/// Five directions inside a single 80-degree wedge: real support, no coverage.
fn wedge() -> Vec<[f64; 3]> {
    ring(&[5.0, 25.0, 50.0, 70.0, 85.0])
}

fn angle_deg(a: [f64; 3], b: [f64; 3]) -> f64 {
    let (a, b) = (Vector3::from(a), Vector3::from(b));
    (a.dot(&b) / (a.norm() * b.norm()))
        .clamp(-1.0, 1.0)
        .acos()
        .to_degrees()
}

/// Every output field is `NaN` / `false` at `p`.
fn assert_unfitted(out: &AdjacencySurfelNormals, p: usize) {
    assert!(out.normals[p].iter().all(|v| v.is_nan()), "normal at {p}");
    for (name, value) in [
        ("n_eff", out.n_eff[p]),
        ("anisotropy", out.anisotropy[p]),
        ("sectors", out.sectors[p]),
        ("sigma_deg", out.sigma_deg[p]),
        ("resid_deg", out.resid_deg[p]),
        ("n_support", out.n_support[p]),
    ] {
        assert!(value.is_nan(), "{name} at {p} is {value}, expected NaN");
    }
    assert!(!out.determined[p], "determined at {p}");
}

// ── The fit ───────────────────────────────────────────────────────────────

#[test]
fn exact_plane_recovers_its_normal_and_takes_the_view_side() {
    let neighbours = full_ring();
    let out = Hub::new(&neighbours).fit_default();

    assert!(angle_deg(out.normals[0], [0.0, 0.0, 1.0]) < 1e-9);
    assert_eq!(out.n_support[0], 8.0);
    assert!((out.n_eff[0] - 8.0).abs() < 1e-9);
    assert!((out.anisotropy[0] - 1.0).abs() < 1e-9);
    assert_eq!(out.sectors[0], 8.0);
    assert!(out.resid_deg[0] < 1e-9);
    assert!(out.determined[0]);

    // The same plane viewed from the other side comes back with the opposite
    // sign — the geometry cannot distinguish them, the caller's view can.
    let flipped = Hub::new(&neighbours).view([0.0, 0.0, -1.0]).fit_default();
    assert!(angle_deg(flipped.normals[0], [0.0, 0.0, -1.0]) < 1e-9);
}

#[test]
fn the_sigma_floor_engages_on_a_perfect_plane() {
    // Every residual is zero, so the raw robust scale is zero; without the
    // floor the next pass would redescend the whole neighbourhood away.
    let out = Hub::new(&full_ring()).fit_default();
    assert!((out.sigma_deg[0] - 2.0).abs() < 1e-9);

    let params = AdjacencySurfelParams {
        sigma_floor_deg: 5.0,
        ..Default::default()
    };
    let out = Hub::new(&full_ring()).fit(&params);
    assert!((out.sigma_deg[0] - 5.0).abs() < 1e-9);
}

#[test]
fn gross_off_surface_neighbours_are_redescended_away() {
    // Eight neighbours on the surface plus two that sit almost straight along
    // the viewing direction — points on another surface, seen next to this one.
    let mut neighbours = full_ring();
    neighbours.push([0.15, 0.0, 1.0]);
    neighbours.push([0.10, 0.12, 1.0]);
    let hub = Hub::new(&neighbours);

    // Without the robust loop the two of them tilt the plane visibly.
    let raw = hub.fit(&AdjacencySurfelParams {
        irls_iters: 0,
        ..Default::default()
    });
    assert!(angle_deg(raw.normals[0], [0.0, 0.0, 1.0]) > 5.0);
    assert!(
        raw.sigma_deg[0].is_nan(),
        "no pass ran, so there is no scale"
    );

    // With it, their weights reach zero and the clean plane comes back exactly.
    let out = hub.fit_default();
    assert!(angle_deg(out.normals[0], [0.0, 0.0, 1.0]) < 1e-9);
    assert_eq!(out.n_support[0], 10.0);
    assert!(
        (out.n_eff[0] - 8.0).abs() < 1e-9,
        "n_eff {} should count the eight survivors",
        out.n_eff[0]
    );
    assert!(out.resid_deg[0] < 1e-9);
    assert!(out.determined[0]);
}

#[test]
fn a_point_whose_weights_all_redescend_keeps_the_weights_it_had() {
    // A symmetric cone: every neighbour sits 30 degrees off the fitted plane,
    // so no row is an inlier relative to the others. A tight Tukey constant
    // then sends every weight to zero at once, and the point stalls rather than
    // solving an all-zero scatter matrix (whose eigenvectors are arbitrary).
    let (c, s) = (30.0f64.to_radians().cos(), 30.0f64.to_radians().sin());
    let cone: Vec<[f64; 3]> = [0.0, 90.0, 180.0, 270.0]
        .iter()
        .map(|a| {
            let r = f64::to_radians(*a);
            [c * r.cos(), c * r.sin(), s]
        })
        .collect();
    let params = AdjacencySurfelParams {
        tukey_c: 0.1,
        ..Default::default()
    };
    let out = Hub::new(&cone).fit(&params);

    // The initial unit weights survive, so the normal is the plain fit.
    assert!(angle_deg(out.normals[0], [0.0, 0.0, 1.0]) < 1e-9);
    assert!(out.normals[0].iter().all(|v| v.is_finite()));
    assert!((out.n_eff[0] - 4.0).abs() < 1e-9);
    assert_eq!(out.n_support[0], 4.0);
    assert_eq!(out.sectors[0], 4.0);
    // The recorded scale is the last one computed while the point was active:
    // the very first pass, on residuals of sin(30 degrees).
    let expected = (1.4826 * s).asin().to_degrees();
    assert!(
        (out.sigma_deg[0] - expected).abs() < 1e-9,
        "sigma_deg {} != {expected}",
        out.sigma_deg[0]
    );
    assert!(out.determined[0]);
}

// ── Support, coverage and anisotropy ──────────────────────────────────────

#[test]
fn fewer_than_two_neighbours_yields_no_normal() {
    assert_unfitted(&Hub::new(&[]).fit_default(), 0);
    assert_unfitted(&Hub::new(&[[1.0, 0.0, 0.0]]).fit_default(), 0);

    // A neighbour that coincides with the hub carries no direction and is
    // dropped before it can count as support.
    let out = Hub::new(&[[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]).fit_default();
    assert_unfitted(&out, 0);
}

#[test]
fn a_collinear_neighbour_line_fails_the_anisotropy_gate() {
    let line: Vec<[f64; 3]> = [-3.0, -2.0, -1.0, 1.0, 2.0, 3.0]
        .iter()
        .map(|x| [*x, 0.0, 0.0])
        .collect();
    let out = Hub::new(&line).fit_default();

    assert_eq!(out.n_support[0], 6.0);
    assert!((out.n_eff[0] - 6.0).abs() < 1e-9);
    assert!(
        out.anisotropy[0] < 1e-12,
        "a rank-1 spread has no second in-plane axis, got {}",
        out.anisotropy[0]
    );
    assert!(!out.determined[0]);

    // The isotropic neighbourhood, by contrast, pins the plane.
    assert!(Hub::new(&full_ring()).fit_default().determined[0]);
}

#[test]
fn a_one_sided_neighbourhood_fails_the_sector_gate() {
    let out = Hub::new(&wedge()).fit_default();

    assert_eq!(out.sectors[0], 2.0);
    assert!(
        (out.n_eff[0] - 5.0).abs() < 1e-9,
        "support is not the problem"
    );
    assert!(
        out.anisotropy[0] >= AdjacencySurfelParams::default().det_aniso,
        "anisotropy is not the problem, got {}",
        out.anisotropy[0]
    );
    assert!(!out.determined[0]);

    // The same number of rows spread over both sides occupies twice the
    // sectors, and that alone flips the verdict.
    let mut spread = wedge();
    spread.extend(ring(&[185.0, 205.0, 230.0, 250.0, 265.0]));
    let out = Hub::new(&spread).fit_default();
    assert_eq!(out.sectors[0], 4.0);
    assert!(out.determined[0]);
}

// ── Extras ────────────────────────────────────────────────────────────────

#[test]
fn extras_fill_the_empty_sectors_of_an_under_determined_point() {
    assert!(!Hub::new(&wedge()).fit_default().determined[0]);

    let helpers = ring(&[140.0, 230.0, 320.0]);
    let out = Hub::new(&wedge()).extras(0, &helpers).fit_default();

    assert_eq!(out.n_support[0], 8.0, "extras count as support");
    assert!((out.n_eff[0] - 8.0).abs() < 1e-9);
    assert_eq!(out.sectors[0], 5.0);
    assert!(angle_deg(out.normals[0], [0.0, 0.0, 1.0]) < 1e-9);
    assert!(out.determined[0]);
}

#[test]
fn extras_are_the_only_support_a_point_needs() {
    // A point with no graph neighbours at all is fittable from extras alone.
    let out = Hub::new(&[])
        .extras(0, &ring(&[0.0, 120.0, 240.0]))
        .fit_default();
    assert_eq!(out.n_support[0], 3.0);
    assert!(angle_deg(out.normals[0], [0.0, 0.0, 1.0]) < 1e-9);
    assert_eq!(out.sectors[0], 3.0);
}

#[test]
fn extras_for_unselected_points_are_ignored() {
    let hub = Hub::new(&full_ring())
        .extras(3, &[[0.0, 0.0, 5.0], [0.0, 5.0, 0.0]])
        .fit_default();
    let plain = Hub::new(&full_ring()).fit_default();

    assert_eq!(hub.normals[0], plain.normals[0]);
    assert_eq!(hub.n_support[0], plain.n_support[0]);
    assert_unfitted(&hub, 3);
}

// ── Selection ─────────────────────────────────────────────────────────────

#[test]
fn unselected_points_stay_nan_end_to_end() {
    let out = Hub::new(&full_ring()).fit_default();
    for p in 1..out.normals.len() {
        assert_unfitted(&out, p);
    }

    // Deselecting the only fittable point leaves nothing behind.
    let none = Hub::new(&full_ring()).select(0, false).fit_default();
    for p in 0..none.normals.len() {
        assert_unfitted(&none, p);
    }
}

#[test]
fn an_empty_cloud_returns_empty_arrays() {
    let out = estimate_adjacency_surfel_normals(
        &[],
        &[],
        &[],
        &[],
        &[],
        &ExtraNeighbours::none(),
        &AdjacencySurfelParams::default(),
    );
    assert!(out.normals.is_empty());
    assert!(out.determined.is_empty());
}

#[test]
fn every_point_can_be_fitted_at_once() {
    // Two independent hubs in one call: the parallel scatter must land each
    // point's fit on its own row.
    let mut positions = vec![[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]];
    let mut neighbours: Vec<u32> = Vec::new();
    let mut offsets = vec![0u32, 0, 0];
    for (hub, plane_z) in [(0usize, true), (1usize, false)] {
        for d in full_ring() {
            let base = positions[hub];
            positions.push(if plane_z {
                [base[0] + d[0], base[1] + d[1], base[2]]
            } else {
                // The second hub's neighbours span the y = 0 plane instead.
                [base[0] + d[0], base[1], base[2] + d[1]]
            });
            neighbours.push((positions.len() - 1) as u32);
        }
        offsets[hub + 1] = neighbours.len() as u32;
    }
    let n = positions.len();
    offsets.resize(n + 1, neighbours.len() as u32);
    let mut selected = vec![false; n];
    selected[0] = true;
    selected[1] = true;
    let mut view_dirs = vec![[0.0, 0.0, 1.0]; n];
    view_dirs[1] = [0.0, 1.0, 0.0];

    let out = estimate_adjacency_surfel_normals(
        &positions,
        &offsets,
        &neighbours,
        &view_dirs,
        &selected,
        &ExtraNeighbours::none(),
        &AdjacencySurfelParams::default(),
    );

    assert!(angle_deg(out.normals[0], [0.0, 0.0, 1.0]) < 1e-9);
    assert!(angle_deg(out.normals[1], [0.0, 1.0, 0.0]) < 1e-9);
    assert!(out.determined[0] && out.determined[1]);
    for p in 2..n {
        assert_unfitted(&out, p);
    }
}
