// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

use super::*;

/// A synthetic set of observations over one or more images.
///
/// Radii and keypoints are chosen off cell-center distance boundaries
/// throughout, so a 1-ULP difference can never flip a cell in or out.
struct Scene {
    sizes: Vec<[u32; 2]>,
    image_idx: Vec<u32>,
    keypoints: Vec<[f64; 2]>,
    radii: Vec<f32>,
}

impl Scene {
    fn new(sizes: &[[u32; 2]]) -> Self {
        Self {
            sizes: sizes.to_vec(),
            image_idx: Vec::new(),
            keypoints: Vec::new(),
            radii: Vec::new(),
        }
    }

    fn obs(&mut self, image: u32, x: f64, y: f64, radius: f32) -> &mut Self {
        self.image_idx.push(image);
        self.keypoints.push([x, y]);
        self.radii.push(radius);
        self
    }

    fn build(&self, cell_px: u32) -> ObservationCoverage {
        ObservationCoverage::build(
            &self.sizes,
            &self.image_idx,
            &self.keypoints,
            &self.radii,
            cell_px,
        )
    }
}

/// The cells with a non-zero count in one image, as `(cx, cy)` pairs.
fn covered_cells(coverage: &ObservationCoverage, image: usize) -> Vec<(u32, u32)> {
    let (cells, w, _) = coverage.grid(image).expect("image exists");
    cells
        .iter()
        .enumerate()
        .filter(|(_, &c)| c > 0)
        .map(|(i, _)| (i as u32 % w, i as u32 / w))
        .collect()
}

/// A one-cell-per-observation stamp: with `cell_px = 1` a radius of 0.4 at a
/// cell center reaches no other center (the nearest is 1.0 away).
fn stamp(scene: &mut Scene, image: u32, cx: u32, cy: u32) {
    scene.obs(image, cx as f64 + 0.5, cy as f64 + 0.5, 0.4);
}

// ── Rasterization ─────────────────────────────────────────────────────────

#[test]
fn disk_covers_the_cells_whose_centers_it_contains() {
    // 40x40 px at 4 px cells: 10x10 cells with centers at 2, 6, 10, ...
    // The keypoint sits on the center of cell (2, 2); the four edge-adjacent
    // centers are 4 px away and the four diagonal ones 5.657 px away.
    let mut scene = Scene::new(&[[40, 40]]);
    scene.obs(0, 10.0, 10.0, 5.1);
    let coverage = scene.build(4);
    assert_eq!(
        covered_cells(&coverage, 0),
        vec![(2, 1), (1, 2), (2, 2), (3, 2), (2, 3)]
    );

    // Widen past the diagonals but not to the next ring at 8 px.
    let mut scene = Scene::new(&[[40, 40]]);
    scene.obs(0, 10.0, 10.0, 6.0);
    let coverage = scene.build(4);
    assert_eq!(covered_cells(&coverage, 0).len(), 9);
}

#[test]
fn grid_dimensions_round_up_and_cells_may_pass_the_image_edge() {
    // 10 px at 4 px cells: 3 cells, the last covering pixels [8, 12) with its
    // center at 10 — outside the image, but a real cell all the same.
    let coverage = Scene::new(&[[10, 6]]).build(4);
    let (cells, w, h) = coverage.grid(0).expect("image exists");
    assert_eq!((w, h), (3, 2));
    assert_eq!(cells.len(), 6);
}

#[test]
fn overlapping_footprints_accumulate_and_saturate() {
    let mut scene = Scene::new(&[[40, 40]]);
    // Both reach only the center of cell (2, 2): the nearest other center is
    // 4 px away.
    scene.obs(0, 10.0, 10.0, 1.0);
    scene.obs(0, 10.0, 10.0, 1.0);
    let coverage = scene.build(4);
    let (cells, w, _) = coverage.grid(0).expect("image exists");
    assert_eq!(cells[2 * w as usize + 2], 2);

    let mut scene = Scene::new(&[[40, 40]]);
    for _ in 0..300 {
        scene.obs(0, 10.0, 10.0, 1.0);
    }
    let coverage = scene.build(4);
    let (cells, w, _) = coverage.grid(0).expect("image exists");
    assert_eq!(cells[2 * w as usize + 2], 255, "counts saturate at 255");
}

#[test]
fn footprints_are_clipped_to_the_grid() {
    // At the top-left corner only the center of cell (0, 0) — 2.83 px away —
    // is in reach; the next centers are 6.32 px away.
    let mut scene = Scene::new(&[[40, 40]]);
    scene.obs(0, 0.0, 0.0, 5.0);
    let coverage = scene.build(4);
    assert_eq!(covered_cells(&coverage, 0), vec![(0, 0)]);

    // Far outside the image: nothing at all, on either side.
    let mut scene = Scene::new(&[[40, 40]]);
    scene.obs(0, -100.0, -100.0, 5.0);
    scene.obs(0, 500.0, 500.0, 5.0);
    let coverage = scene.build(4);
    assert!(covered_cells(&coverage, 0).is_empty());
}

#[test]
fn degenerate_radii_and_keypoints_contribute_nothing() {
    let mut scene = Scene::new(&[[40, 40]]);
    scene.obs(0, 10.0, 10.0, 0.0);
    scene.obs(0, 10.0, 10.0, -3.0);
    scene.obs(0, 10.0, 10.0, f32::NAN);
    scene.obs(0, 10.0, 10.0, f32::INFINITY);
    scene.obs(0, f64::NAN, 10.0, 5.0);
    scene.obs(0, 10.0, f64::INFINITY, 5.0);
    let coverage = scene.build(4);
    assert!(covered_cells(&coverage, 0).is_empty());
}

#[test]
fn observations_of_a_missing_image_are_ignored() {
    let mut scene = Scene::new(&[[40, 40]]);
    scene.obs(7, 10.0, 10.0, 5.1);
    let coverage = scene.build(4);
    assert_eq!(coverage.image_count(), 1);
    assert!(covered_cells(&coverage, 0).is_empty());
}

#[test]
fn images_are_independent() {
    let mut scene = Scene::new(&[[40, 40], [40, 40], [0, 0]]);
    scene.obs(0, 10.0, 10.0, 1.0);
    let coverage = scene.build(4);
    assert_eq!(coverage.image_count(), 3);
    assert_eq!(covered_cells(&coverage, 0), vec![(2, 2)]);
    assert!(covered_cells(&coverage, 1).is_empty());
    let (cells, w, h) = coverage.grid(2).expect("image exists");
    assert!(cells.is_empty());
    assert_eq!((w, h), (0, 0));
    assert!(coverage.grid(3).is_none());
}

#[test]
fn build_keeps_the_cell_size() {
    let coverage = Scene::new(&[[40, 40]]).build(8);
    assert_eq!(coverage.cell_px(), 8);
    let (_, w, h) = coverage.grid(0).expect("image exists");
    assert_eq!((w, h), (5, 5));
}

#[test]
#[should_panic(expected = "cell_px must be positive")]
fn zero_cell_size_panics() {
    Scene::new(&[[40, 40]]).build(0);
}

#[test]
#[should_panic(expected = "radii_px must have one entry per observation")]
fn mismatched_observation_lengths_panic() {
    ObservationCoverage::build(&[[40, 40]], &[0], &[[1.0, 1.0]], &[], DEFAULT_CELL_PX);
}

// ── counts_at ─────────────────────────────────────────────────────────────

#[test]
fn counts_at_reads_the_containing_cell() {
    let mut scene = Scene::new(&[[40, 40]]);
    scene.obs(0, 10.0, 10.0, 1.0);
    scene.obs(0, 10.0, 10.0, 1.0);
    let coverage = scene.build(4);

    // Cell (2, 2) spans pixels [8, 12) on both axes.
    let counts = coverage.counts_at(
        &[0, 0, 0, 0],
        &[[8.0, 8.0], [11.9, 11.9], [7.9, 10.0], [10.0, 12.1]],
    );
    assert_eq!(counts, vec![2, 2, 0, 0]);
}

#[test]
fn counts_at_returns_zero_outside_the_grid() {
    let mut scene = Scene::new(&[[40, 40]]);
    scene.obs(0, 10.0, 10.0, 1.0);
    let coverage = scene.build(4);

    let counts = coverage.counts_at(
        &[0, 0, 0, 0, 0, 3],
        &[
            [-0.5, 10.0],
            [10.0, -0.5],
            [40.5, 10.0],
            [10.0, 40.5],
            [f64::NAN, 10.0],
            [10.0, 10.0],
        ],
    );
    assert_eq!(counts, vec![0; 6]);
}

// ── covered_fraction ──────────────────────────────────────────────────────

/// A 5x5-cell image at 1 px cells, queried from the center of cell (2, 2)
/// with a radius that reaches the four edge-adjacent centers (1.0 away) but
/// not the diagonals (1.414 away): a five-cell neighbourhood.
const PLUS_RADIUS: f32 = 1.1;

#[test]
fn covered_fraction_counts_the_neighbourhood() {
    let mut scene = Scene::new(&[[5, 5]]);
    let coverage = scene.build(1);
    let empty = coverage.covered_fraction(&[0], &[[2.5, 2.5]], &[PLUS_RADIUS]);
    assert_eq!(empty, vec![0.0], "nothing claimed yet");

    stamp(&mut scene, 0, 2, 2);
    stamp(&mut scene, 0, 1, 2);
    let coverage = scene.build(1);
    let partial = coverage.covered_fraction(&[0], &[[2.5, 2.5]], &[PLUS_RADIUS]);
    assert!((partial[0] - 0.4).abs() < 1e-6, "2 of 5 cells: {partial:?}");

    for (cx, cy) in [(3, 2), (2, 1), (2, 3)] {
        stamp(&mut scene, 0, cx, cy);
    }
    let coverage = scene.build(1);
    let full = coverage.covered_fraction(&[0], &[[2.5, 2.5]], &[PLUS_RADIUS]);
    assert_eq!(full, vec![1.0], "all 5 cells claimed");
}

#[test]
fn covered_fraction_is_zero_when_the_disk_holds_no_cell_center() {
    let mut scene = Scene::new(&[[5, 5]]);
    stamp(&mut scene, 0, 2, 2);
    let coverage = scene.build(1);

    // Radii and points that catch no center: a degenerate radius, a radius
    // smaller than the 0.707 from a cell corner to the nearest center, a
    // query outside the grid, a non-finite radius, and a missing image.
    let fractions = coverage.covered_fraction(
        &[0, 0, 0, 0, 9],
        &[
            [2.5, 2.5],
            [2.0, 2.0],
            [-20.0, -20.0],
            [2.5, 2.5],
            [2.5, 2.5],
        ],
        &[0.0, 0.4, 2.0, f32::NAN, 2.0],
    );
    assert_eq!(fractions, vec![0.0; 5]);
}

// ── uncovered_sectors ─────────────────────────────────────────────────────

/// A query point on a 21x21-cell image at 1 px cells, offset from every cell
/// center so no displacement is axis-aligned: with four sectors the sector
/// boundaries are the axes, so no cell can sit on one.
const SECTOR_QUERY: [f64; 2] = [10.7, 10.9];
/// Reaches out three cells; the nearest distances either side are 3.26 and
/// 3.41, so no center sits on the disk boundary.
const SECTOR_RADIUS: f32 = 3.3;

/// The cell indexes the [`SECTOR_RADIUS`] disk around [`SECTOR_QUERY`] can
/// touch, on either axis (inclusive).
const SPAN_LO: u32 = 7;
const SPAN_HI: u32 = 13;

#[test]
fn uncovered_sectors_reports_every_direction_of_an_empty_image() {
    let coverage = Scene::new(&[[21, 21]]).build(1);
    let masks = coverage.uncovered_sectors(&[0], &[SECTOR_QUERY], &[SECTOR_RADIUS], 4);
    assert_eq!(masks, vec![0b1111]);

    let masks = coverage.uncovered_sectors(&[0], &[SECTOR_QUERY], &[SECTOR_RADIUS], 8);
    assert_eq!(masks, vec![0b1111_1111]);
}

#[test]
fn uncovered_sectors_reports_nothing_when_the_disk_is_covered() {
    let mut scene = Scene::new(&[[21, 21]]);
    // One footprint that reaches well past the query disk. 5.15 px cannot land
    // on a center distance: those squares carry at most two decimals.
    scene.obs(0, SECTOR_QUERY[0], SECTOR_QUERY[1], 5.15);
    let coverage = scene.build(1);
    let masks = coverage.uncovered_sectors(&[0], &[SECTOR_QUERY], &[SECTOR_RADIUS], 4);
    assert_eq!(masks, vec![0]);
}

#[test]
fn uncovered_sectors_reports_exactly_the_uncovered_half() {
    // Sectors 0 and 1 are the +y half (dy > 0), sectors 2 and 3 the -y half.
    // Cell centers sit at cy + 0.5, so dy > 0 is exactly cy >= 11.
    let mut scene = Scene::new(&[[21, 21]]);
    for cy in 11..=SPAN_HI {
        for cx in SPAN_LO..=SPAN_HI {
            stamp(&mut scene, 0, cx, cy);
        }
    }
    let coverage = scene.build(1);
    let masks = coverage.uncovered_sectors(&[0], &[SECTOR_QUERY], &[SECTOR_RADIUS], 4);
    assert_eq!(masks, vec![0b1100], "only the -y sectors are still open");
}

#[test]
fn uncovered_sectors_ignores_directions_outside_the_grid() {
    // A 3x3-cell image queried near the top-left corner: every in-grid center
    // lies in sector 0 (dx > 0, dy > 0), so no other bit can be set.
    let coverage = Scene::new(&[[3, 3]]).build(1);
    let masks = coverage.uncovered_sectors(&[0], &[[0.3, 0.4]], &[1.5], 4);
    assert_eq!(masks, vec![0b0001]);
}

#[test]
fn uncovered_sectors_skips_the_cell_at_the_query_point() {
    // A radius that reaches only the cell the query sits on top of: that cell
    // has no direction, so nothing is reported even though it is uncovered.
    let coverage = Scene::new(&[[21, 21]]).build(1);
    let masks = coverage.uncovered_sectors(&[0], &[[10.5, 10.5]], &[0.9], 4);
    assert_eq!(masks, vec![0]);
}

#[test]
fn uncovered_sectors_bins_the_four_quadrants() {
    // Leave exactly one cell of the disk unclaimed and read back its sector.
    // Displacements of the hole's center from SECTOR_QUERY = (10.7, 10.9):
    //   (11, 11) -> ( 0.8,  0.6)  sector 0
    //   ( 9, 11) -> (-1.2,  0.6)  sector 1
    //   (10, 10) -> (-0.2, -0.4)  sector 2
    //   (11,  9) -> ( 0.8, -1.4)  sector 3
    for (hole, expected) in [
        ((11, 11), 0b0001),
        ((9, 11), 0b0010),
        ((10, 10), 0b0100),
        ((11, 9), 0b1000),
    ] {
        let mut scene = Scene::new(&[[21, 21]]);
        for cy in SPAN_LO..=SPAN_HI {
            for cx in SPAN_LO..=SPAN_HI {
                if (cx, cy) != hole {
                    stamp(&mut scene, 0, cx, cy);
                }
            }
        }
        let coverage = scene.build(1);
        let masks = coverage.uncovered_sectors(&[0], &[SECTOR_QUERY], &[SECTOR_RADIUS], 4);
        assert_eq!(masks, vec![expected], "hole at {hole:?}");
    }
}

#[test]
fn uncovered_sectors_rejects_unrepresentable_sector_counts() {
    let coverage = Scene::new(&[[21, 21]]).build(1);
    for n in [0, MAX_SECTORS + 1] {
        let masks = coverage.uncovered_sectors(&[0], &[SECTOR_QUERY], &[SECTOR_RADIUS], n);
        assert_eq!(masks, vec![0], "n_sectors = {n}");
    }
    // The widest representable fan still works.
    let masks = coverage.uncovered_sectors(&[0], &[SECTOR_QUERY], &[SECTOR_RADIUS], MAX_SECTORS);
    assert!(masks[0] != 0);
}

#[test]
fn sector_of_matches_the_half_open_convention() {
    // Sector k spans [k, k + 1) * 2*PI / n: +x is the half-open start of
    // sector 0 (atan2 is exactly +0 there) and anything just short of a full
    // turn lands in the last sector. The rest sit mid-sector by 45 degrees.
    assert_eq!(sector_of(1.0, 0.0, 4), 0);
    assert_eq!(sector_of(1.0, 1.0, 4), 0);
    assert_eq!(sector_of(-1.0, 1.0, 4), 1);
    assert_eq!(sector_of(-1.0, -1.0, 4), 2);
    assert_eq!(sector_of(1.0, -1.0, 4), 3);
    assert_eq!(sector_of(1.0, -0.001, 4), 3);
    assert_eq!(sector_of(1.0, 0.0, 1), 0);
}

// ── image_covered_fraction ────────────────────────────────────────────────

#[test]
fn image_covered_fraction_counts_the_whole_grid() {
    let mut scene = Scene::new(&[[5, 5], [5, 5], [0, 4]]);
    assert_eq!(scene.build(1).image_covered_fraction(0), 0.0);

    for (cx, cy) in [(2, 2), (1, 2), (3, 2), (2, 1), (2, 3)] {
        stamp(&mut scene, 0, cx, cy);
    }
    let coverage = scene.build(1);
    let fraction = coverage.image_covered_fraction(0);
    assert!((fraction - 0.2).abs() < 1e-6, "5 of 25 cells: {fraction}");
    assert_eq!(coverage.image_covered_fraction(1), 0.0, "untouched image");
    assert_eq!(coverage.image_covered_fraction(2), 0.0, "no cells");
    assert_eq!(coverage.image_covered_fraction(9), 0.0, "missing image");
}

// ── Empty inputs ──────────────────────────────────────────────────────────

#[test]
fn empty_inputs_are_well_defined() {
    let coverage = ObservationCoverage::build(&[], &[], &[], &[], DEFAULT_CELL_PX);
    assert_eq!(coverage.image_count(), 0);
    assert!(coverage.grid(0).is_none());
    assert_eq!(coverage.counts_at(&[], &[]), Vec::<u8>::new());
    assert_eq!(coverage.covered_fraction(&[], &[], &[]), Vec::<f32>::new());
    assert_eq!(
        coverage.uncovered_sectors(&[], &[], &[], 4),
        Vec::<u32>::new()
    );

    // Images but no observations.
    let coverage = Scene::new(&[[40, 40]]).build(4);
    assert_eq!(coverage.counts_at(&[0], &[[10.0, 10.0]]), vec![0]);
    assert_eq!(coverage.image_covered_fraction(0), 0.0);
}

#[test]
#[should_panic(expected = "image_indexes must have one entry per query")]
fn mismatched_query_lengths_panic() {
    let coverage = Scene::new(&[[40, 40]]).build(4);
    coverage.counts_at(&[0, 0], &[[1.0, 1.0]]);
}
