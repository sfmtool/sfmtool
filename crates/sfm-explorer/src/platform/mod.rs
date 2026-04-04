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