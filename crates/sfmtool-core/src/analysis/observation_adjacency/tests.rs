// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

use super::*;

/// A synthetic scene: every point sits at the same 3D position (so the range
/// vet passes by default) and every camera at the origin, until a test moves
/// them.
struct Scene {
    keypoints: Vec<[f64; 2]>,
    point_idx: Vec<u32>,
    image_idx: Vec<u32>,
    radii: Vec<f32>,
    infinity: Vec<bool>,
    positions: Vec<[f64; 3]>,
    centers: Vec<[f64; 3]>,
}

impl Scene {
    fn new(n_points: usize, n_images: usize) -> Self {
        Self {
            keypoints: Vec::new(),
            point_idx: Vec::new(),
            image_idx: Vec::new(),
            radii: vec![1.0; n_points],
            infinity: vec![false; n_points],
            positions: vec![[0.0, 0.0, 10.0]; n_points],
            centers: vec![[0.0, 0.0, 0.0]; n_images],
        }
    }

    fn obs(&mut self, point: u32, image: u32, x: f64, y: f64) -> &mut Self {
        self.keypoints.push([x, y]);
        self.point_idx.push(point);
        self.image_idx.push(image);
        self
    }

    fn build(&self, params: &ObservationAdjacencyParams) -> ObservationAdjacency {
        build_observation_adjacency(
            &self.keypoints,
            &self.point_idx,
            &self.image_idx,
            &self.radii,
            &self.infinity,
            &self.positions,
            &self.centers,
            params,
        )
    }
}

fn neighbours_of(adj: &ObservationAdjacency, p: usize) -> Vec<u32> {
    adj.neighbours[adj.offsets[p] as usize..adj.offsets[p + 1] as usize].to_vec()
}

fn separations_of(adj: &ObservationAdjacency, p: usize) -> Vec<f32> {
    adj.separation_med[adj.offsets[p] as usize..adj.offsets[p + 1] as usize].to_vec()
}

/// Two points seen in two images at a fixed keypoint separation.
fn pair_at(separation: f64) -> Scene {
    let mut scene = Scene::new(2, 2);
    for image in 0..2 {
        scene.obs(0, image, 0.0, 0.0);
        scene.obs(1, image, separation, 0.0);
    }
    scene
}

fn has_edge(adj: &ObservationAdjacency, p: usize, q: u32) -> bool {
    neighbours_of(adj, p).contains(&q)
}

/// `q ∈ N(p) ⇔ p ∈ N(q)`, with matching statistics on both directions.
fn assert_symmetric(adj: &ObservationAdjacency) {
    for p in 0..adj.point_count() {
        for slot in adj.offsets[p] as usize..adj.offsets[p + 1] as usize {
            let q = adj.neighbours[slot] as usize;
            let back = (adj.offsets[q] as usize..adj.offsets[q + 1] as usize)
                .find(|&s| adj.neighbours[s] == p as u32)
                .expect("edge is present in both directions");
            assert_eq!(adj.separation_med[slot], adj.separation_med[back]);
            assert_eq!(adj.separation_min[slot], adj.separation_min[back]);
            assert_eq!(adj.separation_max[slot], adj.separation_max[back]);
            assert_eq!(adj.shared_images[slot], adj.shared_images[back]);
            assert_eq!(adj.annulus_hits[slot], adj.annulus_hits[back]);
            assert_eq!(adj.range_mismatch[slot], adj.range_mismatch[back]);
        }
    }
}

// ── Annulus ───────────────────────────────────────────────────────────────

#[test]
fn annulus_includes_both_edges() {
    let params = ObservationAdjacencyParams::default();
    for separation in [params.a_lo, params.b_max] {
        let adj = pair_at(separation).build(&params);
        assert_eq!(adj.directed_edge_count(), 2, "separation {separation}");
        assert_eq!(neighbours_of(&adj, 0), vec![1]);
        assert_eq!(separations_of(&adj, 0), vec![separation as f32]);
    }
}

#[test]
fn annulus_excludes_outside_either_edge() {
    let params = ObservationAdjacencyParams::default();
    for separation in [params.a_lo - 0.5, params.b_max + 0.5] {
        let adj = pair_at(separation).build(&params);
        assert_eq!(adj.directed_edge_count(), 0, "separation {separation}");
    }
}

#[test]
fn pair_radius_is_the_smaller_of_the_two() {
    // r_pair = min(1, 4) = 1, so a separation of 4 px is 4 pair radii and a
    // b_max of 3 rejects it while a b_max of 5 keeps it.
    let mut scene = pair_at(4.0);
    scene.radii[1] = 4.0;
    let tight = ObservationAdjacencyParams {
        b_max: 3.0,
        ..Default::default()
    };
    assert_eq!(scene.build(&tight).directed_edge_count(), 0);

    let loose = ObservationAdjacencyParams {
        b_max: 5.0,
        ..Default::default()
    };
    let adj = scene.build(&loose);
    assert_eq!(adj.directed_edge_count(), 2);
    assert_eq!(separations_of(&adj, 0), vec![4.0]);
}

#[test]
fn a_lo_zero_admits_coincident_observations() {
    let scene = pair_at(0.0);
    assert_eq!(
        scene
            .build(&ObservationAdjacencyParams::default())
            .directed_edge_count(),
        0
    );

    let collapse = ObservationAdjacencyParams {
        a_lo: 0.0,
        ..Default::default()
    };
    let adj = scene.build(&collapse);
    assert_eq!(adj.directed_edge_count(), 2);
    assert_eq!(separations_of(&adj, 0), vec![0.0]);
    assert_eq!(adj.annulus_hits[0], 2);
}

// ── Majority and support ──────────────────────────────────────────────────

#[test]
fn majority_vote_needs_more_than_one_hit_in_three() {
    // Image 0 puts the pair in the annulus; images 1 and 2 do not.
    let mut scene = Scene::new(2, 3);
    scene.obs(0, 0, 0.0, 0.0).obs(1, 0, 2.0, 0.0);
    for image in 1..3 {
        scene.obs(0, image, 0.0, 0.0);
        scene.obs(1, image, 40.0, 0.0);
    }
    let params = ObservationAdjacencyParams::default();
    assert_eq!(scene.build(&params).directed_edge_count(), 0);

    // Two of three passes.
    let mut scene = Scene::new(2, 3);
    for image in 0..2 {
        scene.obs(0, image, 0.0, 0.0);
        scene.obs(1, image, 2.0, 0.0);
    }
    scene.obs(0, 2, 0.0, 0.0).obs(1, 2, 40.0, 0.0);
    let adj = scene.build(&params);
    assert_eq!(adj.directed_edge_count(), 2);
    assert_eq!(adj.shared_images[0], 3);
    assert_eq!(adj.annulus_hits[0], 2);
}

#[test]
fn min_shared_images_floor() {
    let mut scene = Scene::new(2, 2);
    scene.obs(0, 0, 0.0, 0.0).obs(1, 0, 2.0, 0.0);
    assert_eq!(
        scene
            .build(&ObservationAdjacencyParams::default())
            .directed_edge_count(),
        0
    );

    let lenient = ObservationAdjacencyParams {
        min_shared_images: 1,
        ..Default::default()
    };
    let adj = scene.build(&lenient);
    assert_eq!(adj.directed_edge_count(), 2);
    assert_eq!(adj.shared_images[0], 1);
}

// ── Range vet ─────────────────────────────────────────────────────────────

#[test]
fn range_vet_separates_surface_pair_from_stacked_pair() {
    // Four points, two pairs, both at 2 px separation in both images: 0/1 sit
    // side by side on a surface, 2/3 sit one behind the other.
    let mut scene = Scene::new(4, 2);
    scene.positions[0] = [0.0, 0.0, 10.0];
    scene.positions[1] = [0.2, 0.0, 10.0];
    scene.positions[2] = [0.0, 1.0, 10.0];
    scene.positions[3] = [0.0, 1.0, 15.0];
    scene.centers[1] = [1.0, 0.0, 0.0];
    for image in 0..2 {
        scene.obs(0, image, 0.0, 0.0);
        scene.obs(1, image, 2.0, 0.0);
        // Far enough away that no cross edges form.
        scene.obs(2, image, 0.0, 100.0);
        scene.obs(3, image, 2.0, 100.0);
    }

    let params = ObservationAdjacencyParams::default();
    let adj = scene.build(&params);
    assert_eq!(neighbours_of(&adj, 0), vec![1]);
    assert!(adj.range_mismatch[0] < 0.05);
    assert!(neighbours_of(&adj, 2).is_empty());
    assert!(neighbours_of(&adj, 3).is_empty());

    // Infinite tolerance disables the vet, so the stacked pair comes back.
    let disabled = ObservationAdjacencyParams {
        range_tol: f64::INFINITY,
        ..params
    };
    let adj = scene.build(&disabled);
    assert_eq!(neighbours_of(&adj, 2), vec![3]);
    assert!(adj.range_mismatch[adj.offsets[2] as usize] > 0.3);
}

// ── Exclusions ────────────────────────────────────────────────────────────

#[test]
fn infinity_and_non_positive_radius_exclude_points() {
    let params = ObservationAdjacencyParams::default();

    let mut scene = pair_at(2.0);
    scene.infinity[1] = true;
    assert_eq!(scene.build(&params).directed_edge_count(), 0);

    let mut scene = pair_at(2.0);
    scene.radii[1] = 0.0;
    assert_eq!(scene.build(&params).directed_edge_count(), 0);

    let mut scene = pair_at(2.0);
    scene.radii[0] = -1.0;
    assert_eq!(scene.build(&params).directed_edge_count(), 0);
}

// ── CSR layout ────────────────────────────────────────────────────────────

#[test]
fn csr_is_symmetric_and_ordered_by_separation_then_index() {
    // A hub at the origin with three neighbours at 4, 2 and 2 pair radii.
    let mut scene = Scene::new(4, 2);
    let layout = [
        (0u32, 0.0, 0.0),
        (1, 4.0, 0.0),
        (2, 0.0, 2.0),
        (3, -2.0, 0.0),
    ];
    for image in 0..2 {
        for &(point, x, y) in &layout {
            scene.obs(point, image, x, y);
        }
    }

    let adj = scene.build(&ObservationAdjacencyParams::default());
    assert_symmetric(&adj);
    assert_eq!(adj.point_count(), 4);
    assert_eq!(adj.offsets[0], 0);
    assert_eq!(*adj.offsets.last().unwrap() as usize, adj.neighbours.len());

    // Ties on separation break on the neighbour index.
    assert_eq!(neighbours_of(&adj, 0), vec![2, 3, 1]);
    assert_eq!(separations_of(&adj, 0), vec![2.0, 2.0, 4.0]);
    assert!(has_edge(&adj, 1, 0));

    for p in 0..adj.point_count() {
        let seps = separations_of(&adj, p);
        assert!(seps.windows(2).all(|w| w[0] <= w[1]), "row {p} is ordered");
    }
}

#[test]
fn statistics_span_the_annulus_hits() {
    // Separation 2 in one image and 4 in the other, plus a miss in the third.
    let mut scene = Scene::new(2, 3);
    scene.obs(0, 0, 0.0, 0.0).obs(1, 0, 2.0, 0.0);
    scene.obs(0, 1, 0.0, 0.0).obs(1, 1, 4.0, 0.0);
    scene.obs(0, 2, 0.0, 0.0).obs(1, 2, 40.0, 0.0);

    let adj = scene.build(&ObservationAdjacencyParams::default());
    assert_eq!(adj.directed_edge_count(), 2);
    assert_eq!(adj.separation_min[0], 2.0);
    assert_eq!(adj.separation_max[0], 4.0);
    assert_eq!(adj.separation_med[0], 3.0); // even count → mean of the middles
    assert_eq!(adj.shared_images[0], 3);
    assert_eq!(adj.annulus_hits[0], 2);
    assert_eq!(adj.range_mismatch[0], 0.0);
}

// ── Degenerate inputs ─────────────────────────────────────────────────────

#[test]
fn empty_inputs_produce_an_empty_graph() {
    let params = ObservationAdjacencyParams::default();

    // No points at all.
    let adj = build_observation_adjacency(&[], &[], &[], &[], &[], &[], &[], &params);
    assert_eq!(adj.offsets, vec![0]);
    assert_eq!(adj.point_count(), 0);
    assert_eq!(adj.directed_edge_count(), 0);

    // Points but no observations.
    let scene = Scene::new(3, 2);
    let adj = scene.build(&params);
    assert_eq!(adj.offsets, vec![0; 4]);

    // One observation per point, each in its own image: nothing is shared.
    let mut scene = Scene::new(2, 2);
    scene.obs(0, 0, 0.0, 0.0).obs(1, 1, 2.0, 0.0);
    assert_eq!(scene.build(&params).directed_edge_count(), 0);

    // A single observation of a single point.
    let mut scene = Scene::new(1, 1);
    scene.obs(0, 0, 0.0, 0.0);
    let adj = scene.build(&params);
    assert_eq!(adj.offsets, vec![0, 0]);
}

#[test]
fn out_of_range_indexes_are_ignored() {
    let mut scene = pair_at(2.0);
    // An observation naming a point and an image that do not exist.
    scene.obs(9, 0, 1.0, 1.0).obs(0, 9, 1.0, 1.0);
    let adj = scene.build(&ObservationAdjacencyParams::default());
    assert_eq!(adj.point_count(), 2);
    assert_eq!(neighbours_of(&adj, 0), vec![1]);
}

#[test]
fn repeated_observation_in_one_image_counts_once() {
    let mut scene = Scene::new(2, 2);
    for image in 0..2 {
        scene.obs(0, image, 0.0, 0.0);
        scene.obs(0, image, 0.5, 0.0);
        scene.obs(1, image, 2.0, 0.0);
    }
    let adj = scene.build(&ObservationAdjacencyParams::default());
    assert_eq!(adj.directed_edge_count(), 2);
    assert_eq!(adj.shared_images[0], 2);
}
