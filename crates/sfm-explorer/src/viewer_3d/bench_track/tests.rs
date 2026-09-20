// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! What the figure is made of, asserted without a window and without a GPU.
//!
//! The fixture is the bench's own ([`crate::bench::tests::state`]): the demo
//! rewritten as `embedded_patches` with every keypoint its point's exact
//! projection, with one of its points put on the bench. Exact projections are
//! what makes the round trip below meaningful -- each mark's `q_i` has to come
//! back onto the very keypoint it was unprojected from.

use nalgebra::Point3;
use sfmtool_core::bench::{EditableTrack, Stage, Verdict};

use super::*;
use crate::bench::geometry;
use crate::state::AppState;

/// Where the eye stands for these figures. Off to one side of the demo, so the
/// arrowhead's barbs have a direction to be squared to.
const EYE: Point3<f64> = Point3::new(10.0, -10.0, 10.0);

/// A state holding the demo node with a point on its bench, and that item's
/// label.
///
/// Crate-visible: the pass's own tests upload what this builds, so what reaches
/// the GPU is the figure this module asserts rather than a second one written
/// to be uploaded.
pub(crate) fn staged() -> (AppState, crate::scene::ReconId, String) {
    let (mut state, id) = crate::bench::tests::state();
    let label = crate::bench::tests::put_on_bench(&mut state, id);
    (state, id, label)
}

/// The figure `track` draws on `id`'s node, from [`EYE`].
pub(crate) fn figure_of(
    state: &AppState,
    id: crate::scene::ReconId,
    track: &EditableTrack,
) -> Option<Figure> {
    figure(&bench_of(state, id, track), EYE)
}

/// The value the dock hands the viewport for `track` on `id`'s node, with no
/// row selected and nothing holding the node.
pub(crate) fn bench_of<'a>(
    state: &'a AppState,
    id: crate::scene::ReconId,
    track: &'a EditableTrack,
) -> BenchTrack<'a> {
    let node = state.node(id).expect("a loaded node");
    BenchTrack {
        node: id,
        track,
        edited: node.edited(),
        transform: &node.transform,
        selected: None,
        busy: false,
    }
}

/// The staged track, as a value to edit before asking for its figure.
fn staged_track() -> (AppState, crate::scene::ReconId, EditableTrack) {
    let (state, id, label) = staged();
    let track = (**state.bench_track(id, &label).expect("the staged track")).clone();
    (state, id, track)
}

/// The track's frame, which every test here has one of.
fn frame_of(track: &EditableTrack) -> &sfmtool_core::patch::cloud::OrientedPatch {
    match &track.stage {
        Stage::Track(payload) => payload.frame.as_ref().expect("a point carries its surfel"),
        Stage::Cluster(_) => panic!("the staged track is at the track stage"),
    }
}

#[test]
fn the_figure_holds_the_frame_the_disc_the_arrow_and_one_mark_per_observation() {
    let (state, id, track) = staged_track();
    let with_keypoints = track
        .observations
        .iter()
        .filter(|observation| observation.site().is_some())
        .count();
    assert!(
        with_keypoints >= 2,
        "the fixture should place its sightings"
    );

    let figure = figure_of(&state, id, &track).expect("a track-stage track draws a figure");

    assert_eq!(figure.frame.len(), 4, "the square is four edges");
    assert_eq!(
        figure.disc.len(),
        CIRCLE_SEGMENTS * 3,
        "the disc is one triangle per segment, fanned from the centre"
    );
    let normal = figure.normal.expect("a finite frame carries its normal");
    assert_eq!(normal.len(), 3, "the segment and the arrowhead's two barbs");
    assert_eq!(figure.marks.len(), with_keypoints);
    for mark in &figure.marks {
        assert_eq!(mark.circle.len(), CIRCLE_SEGMENTS);
    }
    // The arrow stands off the centre along the outward normal, one side
    // length long, and the barbs come back off its tip.
    let centre = figure.frame[0].a;
    assert_ne!(normal[0].b, normal[0].a, "the normal has length");
    assert_eq!(normal[1].a, normal[0].b, "a barb starts at the tip");
    assert_eq!(normal[2].a, normal[0].b);
    assert!(centre.iter().all(|c| c.is_finite()));
}

#[test]
fn every_endpoint_of_a_finite_figure_is_a_place() {
    let (state, id, track) = staged_track();
    let figure = figure_of(&state, id, &track).expect("a figure");
    for stroke in figure.strokes() {
        assert_eq!((stroke.a[3], stroke.b[3]), (1.0, 1.0));
    }
    for vertex in &figure.disc {
        assert_eq!(vertex.at[3], 1.0);
    }
}

#[test]
fn a_track_at_infinity_draws_directions_and_no_normal() {
    let (state, id, mut track) = staged_track();
    // The same surfel taken to the sky. Its bearing is the one the first
    // observation looks along rather than the one the world origin does, so
    // that observation's ray meets the tangent plane in front of its camera and
    // the marks are drawn: a bearing every sighting points away from would
    // leave the `w = 0` mark path untouched.
    let seen_from = state
        .node(id)
        .expect("a loaded node")
        .edited()
        .base
        .image_table
        .images[track.observations[0].image as usize]
        .camera_center();
    let Stage::Track(payload) = &mut track.stage else {
        unreachable!("the staged track is at the track stage");
    };
    payload.at_infinity = true;
    let frame = payload.frame.as_mut().expect("a surfel");
    frame.center = Point3::from((frame.center - seen_from).normalize());
    frame.w = 0.0;

    let figure = figure_of(&state, id, &track).expect("a figure");

    assert!(
        figure.normal.is_none(),
        "a direction patch's normal is fixed by its bearing, so there is          nothing to say and nothing to grab"
    );
    assert!(!figure.marks.is_empty(), "the sightings still mark the sky");
    for stroke in figure.strokes() {
        assert_eq!(
            (stroke.a[3], stroke.b[3]),
            (0.0, 0.0),
            "every endpoint of a direction patch is a direction"
        );
    }
    for vertex in &figure.disc {
        assert_eq!(vertex.at[3], 0.0);
    }
}

#[test]
fn a_cluster_stage_item_draws_nothing() {
    let (state, id, _) = staged();
    let cluster = EditableTrack::empty_cluster();
    assert!(
        figure_of(&state, id, &cluster).is_none(),
        "a cluster has no geometry behind it, so there is nothing to draw"
    );
}

#[test]
fn a_track_with_no_frame_draws_nothing() {
    let (state, id, mut track) = staged_track();
    let Stage::Track(payload) = &mut track.stage else {
        unreachable!("the staged track is at the track stage");
    };
    payload.frame = None;

    assert!(figure_of(&state, id, &track).is_none());
}

#[test]
fn each_mark_reprojects_onto_the_keypoint_it_came_from() {
    let (state, id, track) = staged_track();
    let figure = figure_of(&state, id, &track).expect("a figure");
    let node = state.node(id).expect("a loaded node");
    let image_table = &node.edited().base.image_table;

    let placed: Vec<_> = track
        .observations
        .iter()
        .filter(|observation| observation.site().is_some())
        .collect();
    assert_eq!(placed.len(), figure.marks.len());

    // What is drawn stands `PLANE_LIFT` off the plane, and the plane is where
    // the keypoint's ray was met, so the lift comes back off first.
    let Stage::Track(payload) = &track.stage else {
        panic!("a track-stage track");
    };
    let frame = payload.frame.as_ref().expect("a frame");
    let lift = frame.normal() * (super::PLANE_LIFT * frame.half_extent[0]);

    for (observation, mark) in placed.iter().zip(&figure.marks) {
        let (camera, pose) =
            geometry::view_of(image_table, observation.image as usize).expect("the view");
        let q = nalgebra::Vector3::new(
            f64::from(mark.segment.b[0]),
            f64::from(mark.segment.b[1]),
            f64::from(mark.segment.b[2]),
        ) - lift;
        let back = geometry::project(&camera, &pose, q, 1.0).expect("q_i projects");
        let site = observation.site().expect("a placed sighting");
        let offset = (back[0] - site[0]).hypot(back[1] - site[1]);
        // A thousandth of a pixel: the figure carries its endpoints as `f32`,
        // which is what the pass takes, so the round trip is the unprojection
        // exactly and the storage to within its own last bits.
        assert!(
            offset < 1e-3,
            "q_i should reproject onto its own keypoint, not {offset} px away"
        );
    }
}

#[test]
fn the_marks_carry_their_verdicts_while_the_frame_and_the_normal_stay_in() {
    let (state, id, track) = staged_track();
    let verdicts = [Verdict::In, Verdict::Candidate, Verdict::Out];
    assert!(track.observations.len() >= verdicts.len());

    let mut judged = track.clone();
    for (observation, verdict) in judged.observations.iter_mut().zip(verdicts) {
        observation.verdict = verdict;
    }
    let figure = figure_of(&state, id, &judged).expect("a figure");

    let in_color = rgba(crate::bench::IN_COLOR);
    for stroke in figure.frame.iter().chain(figure.normal.iter().flatten()) {
        assert_eq!(
            stroke.color, in_color,
            "the frame and the normal are the patch, not any one sighting"
        );
    }
    for vertex in &figure.disc {
        assert_eq!(vertex.color, in_color);
    }
    for (mark, verdict) in figure.marks.iter().zip(verdicts) {
        let expected = rgba(crate::bench::verdict_color(verdict));
        assert_eq!(mark.segment.color, expected);
        for stroke in &mark.circle {
            assert_eq!(stroke.color, expected);
        }
    }
}

#[test]
fn the_fog_distance_is_four_half_lengths_of_the_frame() {
    let (state, id, track) = staged_track();
    let half = frame_of(&track).half_extent[0];
    let figure = figure_of(&state, id, &track).expect("a figure");

    // The demo node is unaligned, so the world half-length is the stored one.
    assert!((f64::from(figure.fog_distance) - 4.0 * half).abs() < 1e-6 * half.max(1.0));
}

/// The order the pointer takes the handles in, where their reaches overlap.
///
/// Built from the projected geometry directly rather than driven through a
/// frame, because what is claimed is the **order** and a view that puts three
/// handles within eight pixels of each other is a view no test could read
/// anything else off. The normal's segment leaves the frame's centre, where the
/// dot and every mark of a well-placed track already sit, so it has to come
/// last or it would take presses meant for them.
#[test]
fn the_normal_segment_is_the_last_handle_the_pointer_can_take() {
    let centre = Pos2::new(100.0, 100.0);
    let handles = Handles {
        dot: Some(centre),
        // Far enough away to keep out of this: the priority under test is the
        // one among the three that share the centre.
        corners: [None; 4],
        circles: vec![(2, Pos2::new(160.0, 100.0))],
        // Out along the panel's `+x`, through both of them.
        normal: Some((centre, Pos2::new(300.0, 100.0))),
        // No arrowhead in this one: what is claimed is the order among the
        // three handles that share the centre.
        tilt: None,
    };

    assert_eq!(
        handles.hit(Pos2::new(104.0, 100.0)),
        Some(Handle::Dot),
        "the dot loses its own reach to the segment leaving it",
    );
    assert_eq!(
        handles.hit(Pos2::new(162.0, 100.0)),
        Some(Handle::Circle { observation: 2 }),
        "a mark's circle loses its reach to the segment crossing it",
    );
    // And where the segment alone is in reach, it is what the pointer has.
    assert_eq!(handles.hit(Pos2::new(230.0, 104.0)), Some(Handle::Normal));
    assert_eq!(handles.hit(Pos2::new(230.0, 120.0)), None);
}

/// The other end of that order: the arrowhead is the **first** handle the
/// pointer can take.
///
/// It has to be. It is a point at the far end of the very segment
/// [`Handle::Normal`] is, so a segment tested first would take every press
/// meant for it; and a view that carries the head over a corner or a mark is an
/// ordinary view of a patch seen at a slant. Where there is no arrowhead -- a
/// track at infinity draws no normal at all -- the same press falls through to
/// whatever else is under it.
#[test]
fn the_arrowhead_is_the_first_handle_the_pointer_can_take() {
    let centre = Pos2::new(100.0, 100.0);
    let head = Pos2::new(300.0, 100.0);
    let crowded = |tilt: Option<Tilt>| Handles {
        dot: Some(centre),
        // A corner and a mark carried onto the head by the view.
        corners: [Some(head), None, None, None],
        circles: vec![(2, head)],
        normal: Some((centre, head)),
        tilt,
    };

    let handles = crowded(Some(Tilt::Aim));
    assert_eq!(handles.hit(head), Some(Handle::Arrowhead(Tilt::Aim)));
    assert_eq!(
        handles.hit(head + egui::vec2(0.0, 4.0)),
        Some(Handle::Arrowhead(Tilt::Aim)),
        "the head keeps its whole reach against the corner and the mark on it",
    );
    // Off its reach, the segment it caps is what is left.
    assert_eq!(handles.hit(Pos2::new(230.0, 104.0)), Some(Handle::Normal));

    // With no arrowhead the corner under it takes the press, which is the same
    // order read with the first entry removed.
    assert_eq!(crowded(None).hit(head), Some(Handle::Corner(0)));
}
