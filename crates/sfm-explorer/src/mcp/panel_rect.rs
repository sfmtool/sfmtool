// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Panel body geometry for MCP screenshots and size reporting.

// ── Where a panel is, in the picture of the window ───────────────────────
//
// A panel screenshot is a crop of the one presented frame, and the rectangle to
// crop to is the dock's own: `LeafNode::viewport` is the tab *body*, below the
// tab bar, in logical points. The functions here are pure over
// `(&DockState<Tab>, f32)`, which is what puts the arithmetic under headless
// test rather than only on a machine with a window.

/// The body rectangle of `panel`, in logical points, or `None` when the panel
/// is not docked or has not been laid out.
///
/// `Rect::NOTHING` is what a leaf carries until the dock has drawn it once, and
/// it is not a rectangle to crop to — a picture of it would be empty.
pub(crate) fn panel_body_points(
    dock: &egui_dock::DockState<crate::dock::Tab>,
    panel: crate::dock::Tab,
) -> Option<egui::Rect> {
    let path = dock.find_tab(&panel)?;
    let leaf = dock.leaf(path.node_path()).ok()?;
    let rect = leaf.viewport;
    (rect.is_finite() && rect.is_positive()).then_some(rect)
}

/// The size a panel's body comes back at, in physical pixels.
pub(super) fn panel_body_size(
    dock: &egui_dock::DockState<crate::dock::Tab>,
    panel: crate::dock::Tab,
    pixels_per_point: f32,
) -> Option<[u32; 2]> {
    let [_, _, width, height] = panel_body_pixels(dock, panel, pixels_per_point)?;
    Some([width, height])
}

/// The panel's body as a whole-pixel rectangle `[x, y, width, height]`, from
/// the dock's points and the frame's points-to-pixels scale.
///
/// Rounded rather than truncated, and the far edge rounded before the size is
/// taken from it, so two panels sharing a divider do not both lose the pixel
/// under it.
fn panel_body_pixels(
    dock: &egui_dock::DockState<crate::dock::Tab>,
    panel: crate::dock::Tab,
    pixels_per_point: f32,
) -> Option<[u32; 4]> {
    let rect = panel_body_points(dock, panel)?;
    let scale = if pixels_per_point > 0.0 {
        pixels_per_point
    } else {
        1.0
    };
    let left = (rect.min.x * scale).round().max(0.0) as u32;
    let top = (rect.min.y * scale).round().max(0.0) as u32;
    let right = (rect.max.x * scale).round().max(0.0) as u32;
    let bottom = (rect.max.y * scale).round().max(0.0) as u32;
    Some([
        left,
        top,
        right.saturating_sub(left),
        bottom.saturating_sub(top),
    ])
}

/// The same rectangle clipped to a surface of `surface` pixels, or `None` when
/// nothing of the panel lies inside it.
pub(super) fn panel_crop(
    dock: &egui_dock::DockState<crate::dock::Tab>,
    panel: crate::dock::Tab,
    pixels_per_point: f32,
    surface: [u32; 2],
) -> Option<[u32; 4]> {
    let [x, y, width, height] = panel_body_pixels(dock, panel, pixels_per_point)?;
    let x = x.min(surface[0]);
    let y = y.min(surface[1]);
    let width = width.min(surface[0] - x);
    let height = height.min(surface[1] - y);
    (width > 0 && height > 0).then_some([x, y, width, height])
}
