// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

use ndarray::Array2;

use super::BulkPointOrder;
use crate::camera::sampler::{RegionPool, RegionPoolStats};
use crate::reconstruction::{ObservationSource, PointSet, SfmrReconstruction, TrackObservation};

/// One point: its observations as `(image, x, y)`, and its stored reference.
type Point = (&'static [(u32, f32, f32)], i32);

/// The points of the order's test, with the place each one's reference sorts
/// at in a comment.
const POINTS: [Point; 8] = [
    // image 2, row 5
    (&[(2, 10.0, 5.7)], 0),
    // image 1, row 1, x 1: its reference is its second observation
    (&[(0, 50.0, 3.2), (1, 1.0, 1.0)], 1),
    // image 1, row 1, x 30
    (&[(1, 30.0, 1.9)], 0),
    // image 1, row 1, x 30, a tie with point 2 broken by point index, though
    // its y is less than point 2's
    (&[(1, 30.0, 1.2)], 0),
    // image 1, row 0: before point 1, though its x is greater
    (&[(1, 5.0, 0.5)], 0),
    // no reference
    (&[(0, 1.0, 1.0)], -1),
    // a reference past its track, which is none
    (&[(0, 2.0, 2.0)], 3),
    // image 0
    (&[(0, 900.0, 900.0)], 0),
];

/// A point set holding [`POINTS`], with its keypoints inline.
fn point_set() -> PointSet {
    let mut set = SfmrReconstruction::demo(POINTS.len()).point_set;
    let (mut tracks, mut counts, mut offsets, mut xy) =
        (Vec::new(), Vec::new(), vec![0usize], Vec::new());
    for (p, (observations, _)) in POINTS.iter().enumerate() {
        for &(image, x, y) in *observations {
            tracks.push(TrackObservation {
                image_index: image,
                point_index: p as u32,
            });
            xy.extend([x, y]);
        }
        counts.push(observations.len() as u32);
        offsets.push(tracks.len());
    }
    set.observations = ObservationSource::EmbeddedPatches {
        keypoints_xy: Array2::from_shape_vec((tracks.len(), 2), xy).unwrap(),
        image_file_hashes: vec![[0; 16]; 3],
    };
    set.tracks = tracks;
    set.observation_counts = counts;
    set.observation_offsets = offsets;
    set.reference_observations = Some(POINTS.iter().map(|&(_, r)| r).collect());
    set
}

#[test]
fn the_points_go_by_reference_image_then_row_then_x_with_the_unreferenced_last() {
    let set = point_set();
    assert_eq!(set.points.len(), POINTS.len());
    assert_eq!(
        BulkPointOrder::of_all_points(&set).positions(),
        [7, 4, 1, 2, 3, 0, 5, 6]
    );
}

#[test]
fn without_keypoints_the_points_of_one_reference_image_go_by_point_index() {
    let mut set = point_set();
    set.observations = ObservationSource::SiftFiles {
        feature_indexes: vec![0; set.tracks.len()],
        keypoints_xy: None,
        feature_tool_hashes: vec![[0; 16]; 3],
        sift_content_hashes: vec![[0; 16]; 3],
    };
    assert_eq!(
        BulkPointOrder::of_all_points(&set).positions(),
        [7, 1, 2, 3, 4, 0, 5, 6]
    );
}

#[test]
fn without_references_the_points_go_by_point_index() {
    let mut set = point_set();
    set.reference_observations = None;
    assert_eq!(
        BulkPointOrder::of_all_points(&set).positions(),
        [0, 1, 2, 3, 4, 5, 6, 7]
    );
}

/// A kernel over some of the points gets positions within its own list; a
/// point index out of range sorts as one with no reference, and a point named
/// twice keeps its two entries in their order.
#[test]
fn the_order_of_a_list_of_points_is_of_positions_within_it() {
    let set = point_set();
    let order = BulkPointOrder::of_points(&set, &[5, 0, 99, 7, 0]);
    assert_eq!(order.positions(), [3, 1, 4, 0, 2]);
}

/// Each result goes back to the position it was computed for.
#[test]
fn restore_puts_each_result_at_its_position() {
    let order = BulkPointOrder::of_all_points(&point_set());
    let in_order: Vec<usize> = order.positions().iter().map(|&p| 10 * p).collect();
    assert_eq!(order.restore(in_order), [0, 10, 20, 30, 40, 50, 60, 70]);
}

/// Assert that a bulk kernel's answer does not depend on its point order and
/// that every read of every point goes through the one pool its caller made.
///
/// `call` runs the kernel over `order`, reading through the pool it is given,
/// and returns its answer as text. It runs twice:
///
/// - over the points by index, as the kernel took them before it took the
///   bulk kernels' order, on one thread, so one point is in progress at a
///   time, through a pool that holds any read;
/// - over `order`, which must differ from the index order, on four threads,
///   through a pool of half the most bytes the first run held at once, so the
///   points in progress together want more than it holds.
///
/// The two answers are the same. The small pool holds no more than its
/// capacity and has every byte back after the call; it counts as many reads
/// as the first run's pool did, so a read that does not fit is taken through
/// the tiles rather than dropped or read some other way; and some of its reads
/// copy their regions and others take the tile path, so the pool was the
/// limit.
pub(crate) fn assert_the_point_order_changes_nothing(
    order: &BulkPointOrder,
    call: impl Fn(&BulkPointOrder, &RegionPool) -> String + Sync,
) {
    let by_index = BulkPointOrder::identity(order.positions().len());
    assert_ne!(
        order.positions(),
        by_index.positions(),
        "the point order differs from the index order, so the test can see it"
    );
    let threads = |n| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(n)
            .build()
            .expect("a rayon pool")
    };
    let any = RegionPool::with_capacity(u64::MAX);
    let a = threads(1).install(|| call(&by_index, &any));
    let a_stats = any.stats();

    let small = RegionPool::with_capacity(a_stats.peak_bytes / 2);
    let b = threads(4).install(|| call(order, &small));
    let b_stats = small.stats();
    assert!(a == b, "the answers differ:\nby index: {a}\nin order: {b}");
    assert!(
        b_stats.peak_bytes <= small.capacity(),
        "{b_stats:?} over {}",
        small.capacity()
    );
    assert_eq!(small.held_bytes(), 0, "{b_stats:?}");
    let reads = |s: RegionPoolStats| s.copied_reads + s.reused_reads + s.tiled_reads;
    assert_eq!(
        reads(b_stats),
        reads(a_stats),
        "the small pool counts every read the first did: {a_stats:?} then {b_stats:?}"
    );
    assert!(
        b_stats.copied_reads > 0 && b_stats.tiled_reads > 0,
        "a pool of {} bytes copies some reads and takes others through the tiles: {b_stats:?}",
        small.capacity()
    );
}
