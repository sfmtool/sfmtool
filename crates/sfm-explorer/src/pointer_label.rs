// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Where a label painted beside the pointer goes, so that it stays inside its
//! panel without covering the pointer.
//!
//! A label that follows the pointer, such as a measurement's hover text or
//! Image Detail's hover tooltip, sits to the right of the pointer by default.
//! Clamping that place into the panel would slide the label under the pointer
//! near the right edge and hide what the user is pointing at, so
//! [`place_beside_pointer`] moves it to another side of the pointer instead,
//! the same gap away. `specs/gui/interaction-rules.md` § G6 states the rule.

use egui::{Pos2, Rect, Vec2};

/// The rectangle a label of `size` takes beside `pointer` inside `panel`.
///
/// `offset` is the label's default place, from the pointer to the label's
/// top-left corner; its `x` is the gap between the pointer and the label's
/// near edge, and must be positive, so that the default place is to the
/// right of the pointer. The sides are tried in this order, and the first
/// that fits inside `panel` is taken:
///
/// 1. **Right**, the default: the label's left edge `offset.x` right of the
///    pointer, at the height `offset.y` puts it, slid up or down into the
///    panel when that height leaves it.
/// 2. **Left**, the default mirrored: its right edge `offset.x` left of the
///    pointer, at the same height. It is tried first among the others
///    because it keeps the label on the pointer's row, so a label read at
///    the right edge of a panel is read at the same height as anywhere else.
/// 3. **Below** and **above**, its near edge `offset.x` below or above the
///    pointer, centred on the pointer across and slid sideways into the
///    panel. They are needed only when the label is too wide for either
///    side, and the one the default place leans towards comes first: below
///    for a label whose default top is at or below the pointer, above for
///    one whose top is above it.
///
/// A label that slides along a side never covers the pointer, since the gap
/// keeps it off the pointer's column or row. When no side fits, as in a
/// panel smaller than the label, the default place is clamped into the
/// panel, its top-left corner winning where the label is the larger.
pub(crate) fn place_beside_pointer(pointer: Pos2, size: Vec2, panel: Rect, offset: Vec2) -> Rect {
    let gap = offset.x;
    let fits_across = size.x <= panel.width();
    let fits_down = size.y <= panel.height();
    // The default place's top, slid into the panel: a side label's height.
    let side_top = slide(pointer.y + offset.y, size.y, panel.top(), panel.bottom());
    // Centred across the pointer and slid into the panel: a label's left
    // edge below or above it.
    let across_left = slide(
        pointer.x - size.x / 2.0,
        size.x,
        panel.left(),
        panel.right(),
    );

    let right = Rect::from_min_size(Pos2::new(pointer.x + gap, side_top), size);
    let left = Rect::from_min_size(Pos2::new(pointer.x - gap - size.x, side_top), size);
    let below = Rect::from_min_size(Pos2::new(across_left, pointer.y + gap), size);
    let above = Rect::from_min_size(Pos2::new(across_left, pointer.y - gap - size.y), size);
    let (first_vertical, second_vertical) = if offset.y >= 0.0 {
        (below, above)
    } else {
        (above, below)
    };

    let sides = [
        (right, fits_down),
        (left, fits_down),
        (first_vertical, fits_across),
        (second_vertical, fits_across),
    ];
    for (rect, slides_into_panel) in sides {
        if slides_into_panel && panel.contains_rect(rect) {
            return rect;
        }
    }
    let default = pointer + offset;
    Rect::from_min_size(
        Pos2::new(
            slide(default.x, size.x, panel.left(), panel.right()),
            slide(default.y, size.y, panel.top(), panel.bottom()),
        ),
        size,
    )
}

/// `start` moved so that `start..start + length` lies in `low..high`, or so
/// that it starts at `low` when it is longer than the range.
fn slide(start: f32, length: f32, low: f32, high: f32) -> f32 {
    start.min(high - length).max(low)
}

#[cfg(test)]
mod tests;
