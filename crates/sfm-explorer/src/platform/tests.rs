// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The gesture translation every backend feeds into, driven headlessly:
//! what a `GestureEvent` becomes as an egui event, and where it lands.

use super::*;
use crate::test_support::run_frame_headless;

fn wheel_deltas(events: &[egui::Event]) -> Vec<egui::Vec2> {
    events
        .iter()
        .map(|event| match event {
            egui::Event::MouseWheel { unit, delta, .. } => {
                assert!(
                    matches!(unit, egui::MouseWheelUnit::Point),
                    "a touchpad pan is measured in points, not {unit:?}"
                );
                *delta
            }
            other => panic!("unexpected event {other:?}"),
        })
        .collect()
}

/// The bug this exists for: DirectManipulation takes the touchpad contacts
/// for the whole window, so no wheel event ever reaches egui and every
/// `ScrollArea` in the app — the scene graph, the camera intrinsics
/// panel, the point track table — sat still under a two-finger scroll.
#[test]
fn a_pan_gesture_becomes_a_trackpad_scroll_event() {
    let events = gesture_scroll_events(
        &[GestureEvent::Pan { dx: 3.0, dy: -12.0 }],
        egui::Modifiers::default(),
    );
    // X is flipped, Y is not: see the conversion.
    assert_eq!(wheel_deltas(&events), vec![egui::vec2(-3.0, -12.0)]);
}

/// Every pan of the frame is forwarded: the handler coalesces its own
/// events already, and dropping any of the rest would lose scroll distance.
#[test]
fn every_pan_of_the_frame_is_forwarded() {
    let events = gesture_scroll_events(
        &[
            GestureEvent::Pan { dx: 0.0, dy: 4.0 },
            GestureEvent::Zoom { scale: 1.1 },
            GestureEvent::Pan { dx: 1.0, dy: 2.0 },
        ],
        egui::Modifiers::default(),
    );
    assert_eq!(
        wheel_deltas(&events),
        vec![egui::vec2(0.0, 4.0), egui::vec2(-1.0, 2.0)]
    );
}

/// A pinch belongs to the panel that zooms its own content; egui's scroll
/// areas have nothing to do with it.
#[test]
fn a_pinch_produces_no_scroll() {
    assert!(gesture_scroll_events(
        &[GestureEvent::Zoom { scale: 1.2 }],
        egui::Modifiers::default()
    )
    .is_empty());
}

/// Ctrl+pan is already a zoom gesture for the panels that read DM events,
/// and egui turns a Ctrl+wheel into a zoom of the whole UI — forwarding it
/// would zoom twice over.
#[test]
fn a_pan_under_the_zoom_modifier_is_dropped() {
    for modifiers in [
        egui::Modifiers {
            ctrl: true,
            ..Default::default()
        },
        egui::Modifiers {
            command: true,
            ..Default::default()
        },
    ] {
        assert!(
            gesture_scroll_events(&[GestureEvent::Pan { dx: 0.0, dy: 9.0 }], modifiers).is_empty(),
            "{modifiers:?} should suppress the scroll"
        );
    }
}

/// The panels that handle DM themselves read their wheel input through
/// [`ScrollInput`], and it has to keep ignoring the events this module now
/// injects — otherwise the 3D viewport would pan twice per gesture.
#[test]
fn the_injected_events_are_not_read_a_second_time_as_trackpad_scroll() {
    let ctx = egui::Context::default();
    let input = egui::RawInput {
        events: gesture_scroll_events(
            &[GestureEvent::Pan { dx: 0.0, dy: 20.0 }],
            egui::Modifiers::default(),
        ),
        ..Default::default()
    };
    run_frame_headless(&ctx, input, |ui| {
        let scroll = ScrollInput::from_ctx(ui.ctx(), true);
        assert_eq!(scroll.delta, egui::vec2(0.0, 20.0), "the event did arrive");
        assert!(!scroll.has_trackpad_scroll(), "DM was active this frame");
        assert!(!scroll.has_mouse_wheel(), "a pan is not a wheel notch");
    });
}

/// Where the pointer is put in these tests: near the top-left corner of the
/// content, so it counts as hovering either scroll area: the horizontal one
/// shrinks to a single row of labels, and anything further down would miss it.
const TEST_POINTER: egui::Pos2 = egui::pos2(50.0, 12.0);

/// Drive one frame's worth of events each through real egui and report how far
/// a scroll area moved along `axis` (0 = X, 1 = Y) by the last of them.
///
/// The scroll area sits inside a panel, as every one of the app's does:
/// egui resolves "is the pointer over this rect" against the layers a panel
/// or area registers, and a scroll area built straight on the root `Ui` is
/// on no layer at all and never reads the wheel.
fn scroll_offset_over_frames(axis: usize, frames: Vec<Vec<egui::Event>>) -> f32 {
    let ctx = egui::Context::default();
    let mut offset = 0.0;
    let mut frame = |time: f64, events: Vec<egui::Event>| {
        let input = egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(
                egui::Pos2::ZERO,
                egui::vec2(200.0, 200.0),
            )),
            time: Some(time),
            events,
            ..Default::default()
        };
        run_frame_headless(&ctx, input, |ui| {
            offset = egui::CentralPanel::default()
                .show(ui, |ui| {
                    egui::ScrollArea::new(egui::Vec2b::new(axis == 0, axis == 1))
                        .id_salt("dm_scroll_test")
                        .max_height(100.0)
                        .max_width(100.0)
                        .show(ui, |ui| {
                            // The content has to overflow along `axis` for
                            // egui to have anywhere to scroll to.
                            let rows = |ui: &mut egui::Ui| {
                                for row in 0..200 {
                                    ui.label(format!("row {row}"));
                                }
                            };
                            if axis == 0 {
                                ui.horizontal(rows);
                            } else {
                                rows(ui);
                            }
                        })
                })
                .inner
                .state
                .offset[axis];
        });
    };
    for (step, events) in frames.into_iter().enumerate() {
        frame(step as f64 / 60.0, events);
    }
    offset
}

/// The frames a gesture needs after the one that carries it: egui spreads a
/// wheel delta over the frames that follow, so these are what let it land.
fn settling_frames() -> Vec<Vec<egui::Event>> {
    (0..38).map(|_| Vec::new()).collect()
}

/// Drive a pan through real egui frames with the pointer already over the
/// scroll area, and report how far the area moved along `axis`.
fn scroll_offset_after_pan(pan: GestureEvent, axis: usize) -> f32 {
    // The first frame places the pointer and builds the area; the second
    // carries the gesture, which egui applies to the area it now knows is
    // hovered.
    let mut frames = vec![
        vec![egui::Event::PointerMoved(TEST_POINTER)],
        gesture_scroll_events(&[pan], egui::Modifiers::default()),
    ];
    frames.extend(settling_frames());
    scroll_offset_over_frames(axis, frames)
}

/// End to end through real egui frames: a scroll area under the pointer
/// moves, by the distance the fingers travelled, which is the whole point
/// of the conversion.
#[test]
fn a_vertical_pan_scrolls_a_scroll_area_under_the_pointer() {
    let offset = scroll_offset_after_pan(GestureEvent::Pan { dx: 0.0, dy: -60.0 }, 1);
    // A pan whose content moves *up* reveals rows further down, so the
    // offset grows — by the 60 points the fingers covered, since a
    // `Point`-unit delta is applied unscaled.
    assert!(
        (offset - 60.0).abs() < 1.0,
        "the scroll area moved {offset}, not the 60 points of the gesture"
    );
}

/// The bug this exists for: X was carried across with DM's own sign, and
/// every scroll area in the app then scrolled horizontally the opposite way
/// to the image strip — the one panel that had always panned under DM.
#[test]
fn a_horizontal_pan_scrolls_the_way_the_image_strip_does() {
    let offset = scroll_offset_after_pan(GestureEvent::Pan { dx: 60.0, dy: 0.0 }, 0);
    // `image_browser` moves its own scroll offset by `+= dx`; a scroll area
    // has to land in the same place for the same gesture.
    assert!(
        (offset - 60.0).abs() < 1.0,
        "the scroll area moved {offset}, not the +60 the image strip would"
    );
}

/// Drive a click and then a pan through real egui frames, and report how far
/// the scroll area moved. `restore_pointer` says whether the release is
/// followed by the `CursorMoved`
/// [`windows::restore_pointer_after_click`](super::windows::restore_pointer_after_click)
/// produces; without it the frames are what a Windows click really delivers.
fn scroll_offset_after_click_and_pan(restore_pointer: bool) -> f32 {
    let button = |pressed| egui::Event::PointerButton {
        pos: TEST_POINTER,
        button: egui::PointerButton::Primary,
        pressed,
        modifiers: egui::Modifiers::default(),
    };
    // `egui-winit` ends a contact by forgetting where the pointer is, and on
    // Windows every click arrives as one.
    let mut click = vec![button(true), button(false), egui::Event::PointerGone];
    if restore_pointer {
        click.push(egui::Event::PointerMoved(TEST_POINTER));
    }
    let mut frames = vec![
        vec![egui::Event::PointerMoved(TEST_POINTER)],
        click,
        // The pan carries no pointer event of its own: DirectManipulation has
        // the contacts, so nothing moves the cursor during a two-finger scroll.
        gesture_scroll_events(
            &[GestureEvent::Pan { dx: 0.0, dy: -60.0 }],
            egui::Modifiers::default(),
        ),
    ];
    frames.extend(settling_frames());
    scroll_offset_over_frames(1, frames)
}

/// The bug this exists for: clicking a row in Track View's list, or a camera
/// in the Scene tree, stopped the touchpad from scrolling that list until the
/// mouse was moved. A Windows click reaches egui as a contact, `egui-winit`
/// ends it with `PointerGone`, and from the next frame egui has no
/// `interact_pos`, which is the first thing `ScrollArea` asks for before it
/// will take a scroll, so the pan this module injects landed nowhere.
#[test]
fn a_pan_after_a_click_scrolls_because_the_pointer_was_put_back() {
    assert!(
        scroll_offset_after_click_and_pan(false) < 1.0,
        "the bug: with the pointer gone the pan reaches no scroll area"
    );
    let offset = scroll_offset_after_click_and_pan(true);
    assert!(
        (offset - 60.0).abs() < 1.0,
        "the pan moved the area {offset}, not the 60 points of the gesture"
    );
}
