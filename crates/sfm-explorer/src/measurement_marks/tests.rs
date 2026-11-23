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
