// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Which measurements are drawn, where their endpoints are placed, and the
//! screen geometry both views share: the hit test, the indicator and the
//! segment test. On the fixture in [`crate::measurements::fixture`], with no
//! window.

use super::*;
use crate::measurements::fixture::{measured_state, POINTS};
use crate::scene::ImageRef;

fn ids(layer: &MeasurementLayer) -> Vec<(DrawnId, Role)> {
    layer.drawn.iter().map(|d| (d.id.clone(), d.role)).collect()
}

/// With the option off only the selected measurement is drawn; with it on
/// every one of the reconstruction, the selected one last, wider and the
/// others dimmer.
#[test]
fn the_option_draws_the_selected_measurement_or_all_of_them() {
    let (mut state, id) = measured_state(Some("m"));
    assert!(state.measurement_layer(false).drawn.is_empty());
    assert_eq!(
        state.measurement_layer(true).drawn.len(),
        5,
        "with the option on, every stored measurement"
    );
    let selected = MeasurementRef::distance(id, 2);
    state.select_measurement(selected);
    assert_eq!(
        ids(&state.measurement_layer(false)),
        vec![(DrawnId::Stored(selected), Role::Selected)]
    );
    let all = state.measurement_layer(true);
    assert_eq!(all.drawn.len(), 5);
    let last = all.drawn.last().unwrap();
    assert_eq!(
        last.id,
        DrawnId::Stored(selected),
        "the selected one on top"
    );
    assert_eq!(last.width, EMPHASISED_LINE_WIDTH);
    for other in &all.drawn[..4] {
        assert_eq!(other.role, Role::Other);
        assert_eq!(other.width, LINE_WIDTH);
        assert!(other.color.a() < last.color.a(), "the others are dimmer");
    }
    assert_eq!(state.measurement_layer(false).drawn[0].width, LINE_WIDTH);
}

/// A hidden reconstruction draws none of its stored measurements, with the
/// option off or on; the focused item is the bench's and is drawn whatever
/// the eye says.
#[test]
fn a_hidden_reconstruction_draws_no_stored_measurement() {
    let (mut state, id) = measured_state(Some("m"));
    state.select_measurement(MeasurementRef::distance(id, 0));
    state.scene[0].visible = false;
    assert!(state.measurement_layer(false).drawn.is_empty());
    assert!(state.measurement_layer(true).drawn.is_empty());
    let label = state
        .start_measurement(ImageRef::new(id, 0), [100.0, 50.0])
        .unwrap();
    assert_eq!(
        ids(&state.measurement_layer(true)),
        vec![(DrawnId::Item { node: id, label }, Role::Item)]
    );
}

/// Each endpoint's place: finite through the node's display transform, on a
/// point at infinity, or refused; and its pixel, whatever its place.
#[test]
fn an_endpoint_is_placed_finite_at_infinity_or_refused() {
    let (mut state, _) = measured_state(Some("m"));
    state.scene[0].history.transform_mut().scale = 2.0;
    state.scene[0].history.transform_mut().translation = nalgebra::Vector3::new(1.0, 2.0, 3.0);
    let layer = state.measurement_layer(true);
    let finite =
        |index: usize, side: EndpointSide| match layer.drawn[index].endpoints[side.index()].place {
            Place::Finite(p) => p,
            other => panic!("{index} {side:?} is {other:?}"),
        };
    let world =
        |p: [f64; 3]| nalgebra::Point3::new(2.0 * p[0] + 1.0, 2.0 * p[1] + 2.0, 2.0 * p[2] + 3.0);
    for (got, want) in [
        (finite(0, EndpointSide::A), world(POINTS[0])),
        (finite(0, EndpointSide::B), world(POINTS[1])),
    ] {
        assert!((got - want).norm() < 1e-6, "{got} != {want}");
    }
    assert_eq!(layer.drawn[2].endpoints[1].place, Place::Refused);
    assert_eq!(layer.drawn[3].endpoints[1].place, Place::AtInfinity);
    let bare = crate::measurements::fixture::BARE_PIXEL;
    assert_eq!(
        layer.drawn[2].endpoints[1].pixel,
        [f64::from(bare[0]), f64::from(bare[1])]
    );
    assert_eq!(layer.drawn[0].world_points().len(), 2);
    assert_eq!(layer.drawn[2].world_points().len(), 1);
    assert_eq!(layer.drawn[3].world_points().len(), 1);
}

/// The focused item is drawn in the bench's colour with its one placed
/// endpoint, its marks handles; on a view-only bench they are not handles.
#[test]
fn the_focused_item_is_drawn_as_the_bench_layer() {
    let (mut state, id) = measured_state(Some("m"));
    let label = state
        .start_measurement(ImageRef::new(id, 0), [100.0, 50.0])
        .unwrap();
    state.selected_endpoint = Some(EndpointSide::A);
    let layer = state.measurement_layer(false);
    let item = layer.drawn.last().unwrap();
    assert_eq!(item.id, DrawnId::Item { node: id, label });
    assert_eq!(item.endpoints.len(), 1, "an item with one endpoint");
    assert_eq!(item.endpoints[0].side, EndpointSide::A);
    assert_eq!(
        item.color,
        crate::bench::verdict_color(sfmtool_core::bench::Verdict::In)
    );
    assert!(item.handles);
    assert_eq!(item.selected_endpoint, Some(EndpointSide::A));
    assert_eq!(
        mark_radius(item, EndpointSide::A),
        MARK_RADIUS * SELECTED_MARK_SCALE
    );
    assert_eq!(mark_radius(item, EndpointSide::B), MARK_RADIUS);
    assert!(
        item.hover.contains("endpoint B not placed"),
        "{}",
        item.hover
    );
}

/// A tinted node's measurements take its tint.
#[test]
fn a_tinted_node_s_measurements_take_its_tint() {
    let (mut state, id) = measured_state(Some("m"));
    state.select_measurement(MeasurementRef::distance(id, 0));
    let plain = state.measurement_layer(false).drawn[0].color;
    assert_eq!(
        plain,
        Color32::from_rgb(STORED_RGB[0], STORED_RGB[1], STORED_RGB[2])
    );
    let tint = &crate::scene::TINT_PALETTE[1];
    state.scene[0].tint = crate::scene::NodeTint::Tint(tint);
    let tinted = state.measurement_layer(false).drawn[0].color;
    assert_eq!(
        tinted,
        Color32::from_rgb(tint.rgb[0], tint.rgb[1], tint.rgb[2])
    );
}

/// A segment meets a rectangle when either end is inside or it crosses it,
/// and not when it passes by.
#[test]
fn a_segment_meets_a_rect_where_any_part_is_inside() {
    let rect = Rect::from_min_max(Pos2::new(0.0, 0.0), Pos2::new(100.0, 100.0));
    assert!(segment_meets_rect(
        Pos2::new(50.0, 50.0),
        Pos2::new(500.0, 50.0),
        rect
    ));
    assert!(segment_meets_rect(
        Pos2::new(-50.0, 50.0),
        Pos2::new(150.0, 50.0),
        rect
    ));
    assert!(!segment_meets_rect(
        Pos2::new(-50.0, -10.0),
        Pos2::new(150.0, -10.0),
        rect
    ));
    assert!(!segment_meets_rect(
        Pos2::new(150.0, 0.0),
        Pos2::new(250.0, 100.0),
        rect
    ));
    assert!(segment_meets_rect(
        Pos2::new(-10.0, 50.0),
        Pos2::new(50.0, -10.0),
        rect
    ));
}

/// The indicator's tip is on the inset border along the direction it was
/// asked for, it points that way, and a press on it or its label lands.
#[test]
fn the_indicator_sits_on_the_border_pointing_the_way_asked() {
    let rect = Rect::from_min_max(Pos2::new(0.0, 0.0), Pos2::new(800.0, 600.0));
    let right = Indicator::new(
        rect,
        Vec2::new(5.0, 0.0),
        "Distance #2",
        Vec2::new(70.0, 14.0),
    )
    .unwrap();
    assert_eq!(right.direction, Vec2::new(1.0, 0.0));
    assert!((right.tip.x - (800.0 - INDICATOR_INSET)).abs() < 1e-3);
    assert!((right.tip.y - 300.0).abs() < 1e-3);
    assert!(
        right.label_rect.max.x < right.tip.x,
        "the label is behind the arrow"
    );
    assert!(right.contains(right.tip));
    assert!(right.contains(right.label_rect.center()));
    assert!(!right.contains(rect.center()));
    let up_left = Indicator::new(rect, Vec2::new(-1.0, -1.0), "x", Vec2::splat(10.0)).unwrap();
    assert!(
        (up_left.tip.y - INDICATOR_INSET).abs() < 1e-3,
        "{:?}",
        up_left.tip
    );
    assert!(up_left.tip.x < 400.0);
    assert!(Indicator::new(rect, Vec2::ZERO, "x", Vec2::splat(10.0)).is_none());
}

/// A press lands on a mark before a line, the topmost measurement's first,
/// and on the indicator before anything.
#[test]
fn the_hit_test_takes_marks_before_lines() {
    let (mut state, id) = measured_state(Some("m"));
    state.select_measurement(MeasurementRef::distance(id, 1));
    let layer = state.measurement_layer(true);
    let (a, b) = (Pos2::new(100.0, 100.0), Pos2::new(300.0, 100.0));
    let screen = Screen {
        measurements: vec![
            ScreenMeasurement {
                index: 0,
                marks: vec![ScreenMark {
                    side: EndpointSide::B,
                    pos: Pos2::new(200.0, 104.0),
                    warning: false,
                }],
                line: None,
            },
            ScreenMeasurement {
                index: 4,
                marks: vec![
                    ScreenMark {
                        side: EndpointSide::A,
                        pos: a,
                        warning: false,
                    },
                    ScreenMark {
                        side: EndpointSide::B,
                        pos: b,
                        warning: false,
                    },
                ],
                line: Some([a, b]),
            },
        ],
        indicator: None,
    };
    let top = layer.drawn[4].id.clone();
    assert_eq!(top, DrawnId::Stored(MeasurementRef::distance(id, 1)));
    assert_eq!(
        screen.hit(&layer, Pos2::new(103.0, 101.0)),
        Some(Target::Mark(top.clone(), EndpointSide::A))
    );
    // On the top line, but within a mark's reach of the lower one's B.
    assert_eq!(
        screen.hit(&layer, Pos2::new(200.0, 100.0)),
        Some(Target::Mark(layer.drawn[0].id.clone(), EndpointSide::B))
    );
    assert_eq!(
        screen.hit(&layer, Pos2::new(250.0, 103.0)),
        Some(Target::Line(top))
    );
    assert_eq!(screen.hit(&layer, Pos2::new(250.0, 120.0)), None);
}

/// A click on a mark selects its endpoint and a double-click looks through
/// it; a click on a line selects the measurement; the indicator asks for no
/// gesture of the dock's.
#[test]
fn a_click_asks_for_what_the_target_stands_for() {
    let id = DrawnId::Stored(MeasurementRef::distance(
        crate::scene::ReconId::from_raw(1),
        0,
    ));
    assert_eq!(
        click_gesture(&Target::Mark(id.clone(), EndpointSide::B), false),
        Some(MeasurementGesture::SelectEndpoint(
            id.clone(),
            EndpointSide::B
        ))
    );
    assert_eq!(
        click_gesture(&Target::Mark(id.clone(), EndpointSide::B), true),
        Some(MeasurementGesture::LookThrough(id.clone(), EndpointSide::B))
    );
    assert_eq!(
        click_gesture(&Target::Line(id.clone()), false),
        Some(MeasurementGesture::Select(id))
    );
    assert_eq!(click_gesture(&Target::Indicator, false), None);
}

/// Where one headless frame painted `text` as a measurement's hover text with
/// the pointer at `pointer` in `bounds`.
fn hover_text_rect(pointer: Pos2, bounds: Rect) -> Rect {
    let ctx = egui::Context::default();
    let input = egui::RawInput {
        screen_rect: Some(Rect::from_min_size(Pos2::ZERO, Vec2::new(800.0, 600.0))),
        ..Default::default()
    };
    let painted = crate::test_support::painted_text_rects(&ctx, input, |ui| {
        draw_hover_text(ui.painter(), pointer, "distance #1  1.2300 m", bounds);
    });
    assert_eq!(painted.len(), 1, "{painted:?}");
    painted[0].rect
}

#[test]
fn the_hover_text_sits_right_of_the_pointer_and_moves_left_at_the_right_edge() {
    let bounds = Rect::from_min_size(Pos2::new(0.0, 0.0), Vec2::new(600.0, 400.0));

    // In the open, the text starts 14 points right of and below the pointer.
    let pointer = Pos2::new(200.0, 150.0);
    let text = hover_text_rect(pointer, bounds);
    assert_eq!(text.min, pointer + Vec2::new(14.0, 14.0));

    // At the right edge it is left of the pointer, at the same height, and
    // its ground is the same 10 points away.
    let pointer = Pos2::new(595.0, 150.0);
    let text = hover_text_rect(pointer, bounds);
    assert_eq!(text.right() + 4.0, pointer.x - 10.0, "{text:?}");
    assert_eq!(text.top(), pointer.y + 14.0, "{text:?}");
    assert!(bounds.contains_rect(text.expand(4.0)), "{text:?}");
}

/// A frame whose key is the last one's builds no layer: the cache keeps the
/// same `Arc`, over a thousand frames, with every measurement drawn and the
/// focused item among them.
#[test]
fn an_unchanged_key_builds_the_layer_once() {
    let (mut state, id) = measured_state(Some("m"));
    state
        .put_measurement_on_bench(MeasurementRef::distance(id, 0))
        .unwrap();
    let mut cache = MeasurementLayerCache::default();
    cache.update(&state, true);
    let built = Arc::clone(cache.layer());
    assert_eq!(built.drawn.len(), 6, "five stored and the item");
    for _ in 0..1000 {
        cache.update(&state, true);
    }
    assert_eq!(cache.builds, 1);
    assert!(Arc::ptr_eq(&built, cache.layer()), "the same layer");
}

/// What the changes in [`each_input_of_the_key_builds_the_layer_again`]
/// carry from one to the next.
struct Changing {
    id: ReconId,
    show_all: bool,
    /// The held task's gate, which its job waits on until it is dropped.
    gate: Option<std::sync::mpsc::Sender<()>>,
    task: Option<u64>,
    other: Option<ReconId>,
}

/// One change in [`each_input_of_the_key_builds_the_layer_again`].
type Change = fn(&mut AppState, &mut Changing);

/// Each thing the layer is built from builds it again when it changes, once,
/// and the layer then is the one `measurement_layer` builds: a table over the
/// key's fields, each change made on top of the ones before it.
#[test]
fn each_input_of_the_key_builds_the_layer_again() {
    use crate::background::{Finished, Operation};

    let (mut state, id) = measured_state(Some("m"));
    let mut at = Changing {
        id,
        show_all: true,
        gate: None,
        task: None,
        other: None,
    };
    let mut cache = MeasurementLayerCache::default();
    cache.update(&state, at.show_all);
    assert_eq!(cache.builds, 1);
    let changes: [(&str, Change); 15] = [
        ("show all", |_, at| at.show_all = false),
        ("the selected measurement", |state, at| {
            state.select_measurement(MeasurementRef::distance(at.id, 2))
        }),
        ("the selected endpoint", |state, at| {
            state.select_endpoint(MeasurementRef::distance(at.id, 2), EndpointSide::B)
        }),
        ("the focused item", |state, at| {
            state
                .put_measurement_on_bench(MeasurementRef::distance(at.id, 0))
                .unwrap();
        }),
        ("a step on the item", |state, at| {
            let label = state.focused_measurement_label(at.id).unwrap().to_string();
            state
                .set_bench_measurement_label(at.id, &label, "door")
                .unwrap();
        }),
        ("the item's label on the bench", |state, at| {
            let label = state.focused_measurement_label(at.id).unwrap().to_string();
            state.rename_bench_item(at.id, &label, "renamed").unwrap();
        }),
        ("a task starting on the node", |state, at| {
            let (open, wait) = std::sync::mpsc::channel::<()>();
            let job: crate::background::Job = Box::new(move |_| {
                let _ = wait.recv();
                Finished::Failed("nothing".to_string())
            });
            at.task = Some(
                state
                    .start_background_task(Operation::BENCH_FIT, at.id, job)
                    .unwrap(),
            );
            at.gate = Some(open);
        }),
        ("the task finishing", |state, at| {
            drop(at.gate.take());
            state.finish_background_task(at.task.take().unwrap());
        }),
        ("the document", |state, at| {
            state
                .delete_measurement(MeasurementRef::distance(at.id, 4))
                .unwrap();
        }),
        ("the display transform", |state, _| {
            state.scene[0].history.transform_mut().scale = 2.0
        }),
        ("the tint", |state, _| {
            state.scene[0].tint = crate::scene::NodeTint::Tint(&crate::scene::TINT_PALETTE[1]);
        }),
        ("a node added", |state, at| {
            at.other = Some(state.append_node(crate::scene::SceneNode::demo(
                sfmtool_core::SfmrReconstruction::demo(4),
            )));
        }),
        ("solo on the other node", |state, at| {
            state.set_solo(at.other)
        }),
        ("solo off", |state, _| state.set_solo(None)),
        ("the node's eye", |state, _| state.scene[0].visible = false),
    ];
    for (name, change) in changes {
        let builds = cache.builds;
        change(&mut state, &mut at);
        cache.update(&state, at.show_all);
        cache.update(&state, at.show_all);
        assert_eq!(cache.builds, builds + 1, "{name} builds the layer once");
        assert_eq!(
            **cache.layer(),
            state.measurement_layer(at.show_all),
            "after {name}"
        );
        // What the changes drew differently, where only the cache shows it.
        let item = cache.layer().drawn.iter().find(|d| d.role == Role::Item);
        match name {
            "the item's label on the bench" => assert_eq!(item.unwrap().title, "renamed"),
            "a task starting on the node" => assert!(!item.unwrap().handles),
            "the task finishing" => assert!(item.unwrap().handles),
            _ => {}
        }
    }
}

/// The 3D view and Image Detail keep a layer each, built with their own
/// *Show all measurements*: frames that bring both up to date, in the order
/// the dock does, build each once.
#[test]
fn the_two_views_keep_a_layer_each() {
    let (mut state, id) = measured_state(Some("m"));
    state.select_measurement(MeasurementRef::distance(id, 2));
    let mut viewer = crate::viewer_3d::Viewer3D::new();
    let mut detail = crate::image_detail::ImageDetail::new();
    for _ in 0..3 {
        viewer.measurements.update(&state, false);
        detail.measurements.update(&state, true);
    }
    assert_eq!(viewer.measurements.builds, 1);
    assert_eq!(detail.measurements.builds, 1);
    assert_eq!(viewer.measurements.layer().drawn.len(), 1);
    assert_eq!(detail.measurements.layer().drawn.len(), 5);
}
