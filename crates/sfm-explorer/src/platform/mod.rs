// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Platform-specific gesture handling.
//!
//! Provides cross-platform abstractions for precision touchpad gestures.

#[cfg(target_os = "windows")]
pub mod windows;

/// Cross-platform gesture event.
///
/// These events represent high-level gestures detected from precision touchpad
/// input, providing pixel-level deltas for smooth viewport navigation.
#[derive(Debug, Clone, Copy)]
#[allow(dead_code)]
pub enum GestureEvent {
    /// Pan/scroll gesture with pixel deltas.
    ///
    /// Positive dx means panning right, positive dy means panning up.
    Pan { dx: f64, dy: f64 },

    /// Pinch zoom gesture.
    ///
    /// Scale > 1.0 means zoom in (fingers spreading apart),
    /// scale < 1.0 means zoom out (fingers pinching together).
    Zoom { scale: f64 },
}

/// Turn this frame's DirectManipulation pan gestures into egui scroll events.
///
/// DM claims precision-touchpad contacts for the whole window (`SetContact` on
/// `DM_POINTERHITTEST`, in the `windows` submodule), so Windows never synthesises a
/// `WM_MOUSEWHEEL` for a two-finger scroll and egui's own `egui::ScrollArea`s
/// — the scene graph tree and its inner lists, the intrinsics panel, the point
/// track table — receive nothing at all. Feeding the pan back in as a
/// `Point`-unit `MouseWheel` event is what makes them scroll. The panels that
/// read [`GestureEvent`]s themselves do not double-handle it: they take their
/// wheel input through [`ScrollInput`], which suppresses `Point` scroll for
/// exactly the frames DM was active.
///
/// A pan with Ctrl (or Cmd) held is dropped. Those frames are already claimed
/// as a zoom gesture by the panels that handle DM, and egui reads a Ctrl+wheel
/// as a request to zoom the whole UI — the two together would zoom twice over.
pub fn gesture_scroll_events(
    gestures: &[GestureEvent],
    modifiers: egui::Modifiers,
) -> Vec<egui::Event> {
    if modifiers.ctrl || modifiers.command {
        return Vec::new();
    }
    gestures
        .iter()
        .filter_map(|event| match *event {
            // `dy` already carries egui's own sign (positive = the content
            // moves down), which `Event::MouseWheel` documents for `delta`, so
            // it crosses unscaled and unflipped. `dx` does not: DM reports a
            // horizontal pan opposite to the way egui reads a wheel's X, which
            // is why the image strip adds `dx` to its scroll offset where it
            // subtracts a wheel's `delta.x` from it (`image_browser.rs`).
            // Negating here is what puts the two on the same convention.
            GestureEvent::Pan { dx, dy } => Some(egui::Event::MouseWheel {
                unit: egui::MouseWheelUnit::Point,
                delta: egui::vec2(-dx as f32, dy as f32),
                // `Move` is what egui asks for when the phase is unknown. DM
                // reports no touch down/up around the gesture, and claiming
                // `Start`/`End` here would reset egui's scroll smoothing
                // partway through one.
                phase: egui::TouchPhase::Move,
                modifiers,
            }),
            // Pinch stays with the panels that zoom their own content; no
            // scroll area has anything to do with it.
            GestureEvent::Zoom { .. } => None,
        })
        .collect()
}

/// Accumulated scroll-wheel input for one frame.
///
/// Built once per frame in the main UI loop and shared across all panels,
/// so that each panel applies the same DirectManipulation-aware suppression
/// without duplicating event-reading logic.
pub struct ScrollInput {
    /// Total scroll delta accumulated from all `MouseWheel` events this frame.
    pub delta: egui::Vec2,
    /// Unit type of the scroll events.
    /// `Point` = trackpad two-finger scroll, `Line` = discrete mouse wheel.
    pub unit: egui::MouseWheelUnit,
    /// Modifiers held during the scroll events.
    pub modifiers: egui::Modifiers,
    /// Whether DirectManipulation is actively providing gesture data this frame.
    /// When true, trackpad-style scroll events are suppressed to avoid
    /// double-handling (DM gesture events provide higher-quality input).
    dm_active: bool,
}

impl Default for ScrollInput {
    fn default() -> Self {
        Self {
            delta: egui::Vec2::ZERO,
            unit: egui::MouseWheelUnit::Line,
            modifiers: egui::Modifiers::default(),
            dm_active: false,
        }
    }
}

impl ScrollInput {
    /// Accumulate scroll events from the egui context for this frame.
    ///
    /// `dm_active` should be true when the DirectManipulation gesture handler
    /// is operational and produced gesture events this frame.
    pub fn from_ctx(ctx: &egui::Context, dm_active: bool) -> Self {
        let mut delta = egui::Vec2::ZERO;
        let mut unit = egui::MouseWheelUnit::Line;
        let mut modifiers = egui::Modifiers::default();

        ctx.input(|i| {
            for event in &i.events {
                if let egui::Event::MouseWheel {
                    unit: u,
                    delta: d,
                    modifiers: m,
                    ..
                } = event
                {
                    delta += *d;
                    unit = *u;
                    modifiers = *m;
                }
            }
        });

        Self {
            delta,
            unit,
            modifiers,
            dm_active,
        }
    }

    /// Whether trackpad-style scroll navigation should be used.
    ///
    /// Returns true when scroll events came from a trackpad (`Point` units)
    /// and DirectManipulation is NOT actively handling gestures. When DM is
    /// active, its gesture events provide higher-quality input and trackpad
    /// scroll would be double-handling.
    pub fn has_trackpad_scroll(&self) -> bool {
        matches!(self.unit, egui::MouseWheelUnit::Point)
            && !self.dm_active
            && self.delta != egui::Vec2::ZERO
    }

    /// Whether discrete mouse-wheel scroll should be used.
    pub fn has_mouse_wheel(&self) -> bool {
        !matches!(self.unit, egui::MouseWheelUnit::Point) && self.delta != egui::Vec2::ZERO
    }
}

/// Check if the platform's tracked pointer position is within the given egui rect.
///
/// This uses the pointer position tracked directly from WM_POINTER messages,
/// which remains valid even when egui's hover state goes stale (e.g. after a
/// double-click with no subsequent mouse movement). Falls back to egui's
/// `latest_pos` on non-Windows platforms.
pub fn pointer_in_rect(ctx: &egui::Context, rect: egui::Rect) -> bool {
    #[cfg(test)]
    if let Some(pos) = TEST_POINTER_POS.with(|p| p.get()) {
        return rect.contains(pos);
    }
    #[cfg(target_os = "windows")]
    {
        let (px, py) = windows::pointer_client_pos();
        let ppp = ctx.pixels_per_point();
        let logical_pos = egui::pos2(px as f32 / ppp, py as f32 / ppp);
        rect.contains(logical_pos)
    }
    #[cfg(not(target_os = "windows"))]
    {
        ctx.input(|i| i.pointer.latest_pos())
            .is_some_and(|p| rect.contains(p))
    }
}

// Test-only override of the platform pointer position, in logical points.
// On Windows the real source is a pair of statics fed by the window's pointer
// messages, which no amount of synthesized egui input will move; making the
// override thread-local instead keeps parallel tests from clobbering each
// other's pointer, which shared statics would not.
#[cfg(test)]
thread_local! {
    static TEST_POINTER_POS: std::cell::Cell<Option<egui::Pos2>> =
        const { std::cell::Cell::new(None) };
}

/// Test-only: place the platform's tracked pointer at `pos` (logical points),
/// or clear the override with `None`, for the current thread.
#[cfg(test)]
pub fn set_test_pointer_pos(pos: Option<egui::Pos2>) {
    TEST_POINTER_POS.with(|p| p.set(pos));
}

/// Trait for platform-specific gesture handlers.
///
/// Implementations should process raw pointer/touch input and produce
/// high-level gesture events that can be polled each frame.
///
/// Note: This trait does not require `Send` because gesture handlers typically
/// wrap platform-specific COM objects that must be accessed from the UI thread.
#[allow(dead_code)]
pub trait GestureHandler {
    /// Poll all pending gesture events.
    ///
    /// Returns a vector of events that have occurred since the last poll.
    /// Events are returned in chronological order.
    fn poll_events(&self) -> Vec<GestureEvent>;
}

#[cfg(test)]
mod tests {
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
    /// `ScrollArea` in the app — the scene graph, the intrinsics panel, the
    /// point track table — sat still under a two-finger scroll.
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
                gesture_scroll_events(&[GestureEvent::Pan { dx: 0.0, dy: 9.0 }], modifiers)
                    .is_empty(),
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
    /// Drive a pan through real egui frames and report how far a scroll area
    /// under the pointer moved along `axis` (0 = X, 1 = Y).
    ///
    /// The scroll area sits inside a panel, as every one of the app's does:
    /// egui resolves "is the pointer over this rect" against the layers a panel
    /// or area registers, and a scroll area built straight on the root `Ui` is
    /// on no layer at all and never reads the wheel.
    fn scroll_offset_after_pan(pan: GestureEvent, axis: usize) -> f32 {
        let ctx = egui::Context::default();
        // Near the top-left corner of the content, so the pointer counts as
        // hovering either area: the horizontal one shrinks to a single row of
        // labels, and anything further down would miss it.
        let pointer = egui::pos2(50.0, 12.0);
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
        let with_pointer = |extra: Vec<egui::Event>| {
            let mut events = vec![egui::Event::PointerMoved(pointer)];
            events.extend(extra);
            events
        };

        // The first frame places the pointer and builds the area; the second
        // carries the gesture, which egui applies to the area it now knows is
        // hovered. egui spreads a wheel delta over the frames that follow, so
        // the rest are what let it land.
        frame(0.0, with_pointer(Vec::new()));
        frame(
            1.0 / 60.0,
            with_pointer(gesture_scroll_events(&[pan], egui::Modifiers::default())),
        );
        for step in 2..40 {
            frame(step as f64 / 60.0, with_pointer(Vec::new()));
        }
        offset
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
}
