// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

use super::place_beside_pointer;
use egui::{pos2, vec2, Pos2, Rect, Vec2};

/// A panel the size of a small Image Detail.
fn panel() -> Rect {
    Rect::from_min_size(pos2(100.0, 50.0), vec2(400.0, 300.0))
}

/// A one-line label.
const SIZE: Vec2 = vec2(120.0, 20.0);

/// The measurement hover text's offset, right of and below the pointer.
const BELOW_RIGHT: Vec2 = vec2(10.0, 10.0);

/// The Image Detail tooltip's offset, right of the pointer and above it.
const ABOVE_RIGHT: Vec2 = vec2(9.0, -23.0);

/// The label is inside the panel, leaves the pointer uncovered, and keeps at
/// least `gap` from it on one side.
fn assert_beside(label: Rect, pointer: Pos2, gap: f32) {
    assert!(panel().contains_rect(label), "{label:?} leaves the panel");
    let off_column = label.left() >= pointer.x + gap || label.right() <= pointer.x - gap;
    let off_row = label.top() >= pointer.y + gap || label.bottom() <= pointer.y - gap;
    assert!(
        off_column || off_row,
        "{label:?} is not {gap} clear of the pointer at {pointer:?}"
    );
}

#[test]
fn in_the_open_a_label_keeps_its_default_place() {
    let pointer = pos2(250.0, 200.0);
    for offset in [BELOW_RIGHT, ABOVE_RIGHT] {
        assert_eq!(
            place_beside_pointer(pointer, SIZE, panel(), offset),
            Rect::from_min_size(pointer + offset, SIZE),
        );
    }
}

#[test]
fn at_the_right_edge_a_label_moves_to_the_left_at_the_same_height() {
    let pointer = pos2(panel().right() - 5.0, 200.0);
    for offset in [BELOW_RIGHT, ABOVE_RIGHT] {
        let label = place_beside_pointer(pointer, SIZE, panel(), offset);
        assert_eq!(label.right(), pointer.x - offset.x, "{label:?}");
        assert_eq!(label.top(), pointer.y + offset.y, "{label:?}");
        assert_beside(label, pointer, offset.x);
    }
}

#[test]
fn at_the_bottom_edge_a_label_stays_right_and_slides_up() {
    let pointer = pos2(250.0, panel().bottom() - 5.0);
    let label = place_beside_pointer(pointer, SIZE, panel(), BELOW_RIGHT);
    assert_eq!(label.left(), pointer.x + BELOW_RIGHT.x, "{label:?}");
    assert_eq!(label.bottom(), panel().bottom(), "{label:?}");
    assert_beside(label, pointer, BELOW_RIGHT.x);
}

#[test]
fn at_the_top_edge_a_label_stays_right_and_slides_down() {
    let pointer = pos2(250.0, panel().top() + 5.0);
    let label = place_beside_pointer(pointer, SIZE, panel(), ABOVE_RIGHT);
    assert_eq!(label.left(), pointer.x + ABOVE_RIGHT.x, "{label:?}");
    assert_eq!(label.top(), panel().top(), "{label:?}");
    assert_beside(label, pointer, ABOVE_RIGHT.x);
}

#[test]
fn at_the_left_edge_a_label_keeps_its_default_place() {
    let pointer = pos2(panel().left() + 2.0, 200.0);
    let label = place_beside_pointer(pointer, SIZE, panel(), BELOW_RIGHT);
    assert_eq!(label, Rect::from_min_size(pointer + BELOW_RIGHT, SIZE));
}

#[test]
fn in_every_corner_a_label_is_inside_and_off_the_pointer() {
    let p = panel();
    let corners = [
        pos2(p.left() + 3.0, p.top() + 3.0),
        pos2(p.right() - 3.0, p.top() + 3.0),
        pos2(p.left() + 3.0, p.bottom() - 3.0),
        pos2(p.right() - 3.0, p.bottom() - 3.0),
    ];
    for pointer in corners {
        for offset in [BELOW_RIGHT, ABOVE_RIGHT] {
            let label = place_beside_pointer(pointer, SIZE, p, offset);
            assert_beside(label, pointer, offset.x);
        }
    }
}

#[test]
fn a_label_too_wide_for_either_side_goes_below_or_above_as_its_default_leans() {
    // Wider than the space on either side of a pointer in the middle.
    let wide = vec2(300.0, 20.0);
    let pointer = panel().center();

    let label = place_beside_pointer(pointer, wide, panel(), BELOW_RIGHT);
    assert_eq!(label.top(), pointer.y + BELOW_RIGHT.x, "{label:?}");
    assert_eq!(label.center().x, pointer.x, "centred across the pointer");
    assert_beside(label, pointer, BELOW_RIGHT.x);

    let label = place_beside_pointer(pointer, wide, panel(), ABOVE_RIGHT);
    assert_eq!(label.bottom(), pointer.y - ABOVE_RIGHT.x, "{label:?}");
    assert_beside(label, pointer, ABOVE_RIGHT.x);

    // Near the bottom, below does not fit and it goes above, slid sideways
    // into the panel by the right edge.
    let pointer = pos2(panel().right() - 100.0, panel().bottom() - 5.0);
    let label = place_beside_pointer(pointer, wide, panel(), BELOW_RIGHT);
    assert_eq!(label.bottom(), pointer.y - BELOW_RIGHT.x, "{label:?}");
    assert_eq!(label.right(), panel().right(), "{label:?}");
    assert_beside(label, pointer, BELOW_RIGHT.x);
}

#[test]
fn in_a_panel_too_small_for_any_side_the_default_place_is_clamped() {
    let tiny = Rect::from_min_size(pos2(0.0, 0.0), vec2(100.0, 30.0));
    let pointer = pos2(90.0, 20.0);
    let label = place_beside_pointer(pointer, SIZE, tiny, BELOW_RIGHT);
    // Wider than the panel: its left edge wins. Its height fits, so it is
    // slid up to the panel's bottom.
    assert_eq!(label, Rect::from_min_size(pos2(0.0, 10.0), SIZE));
}
