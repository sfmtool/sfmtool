// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The sort rule the viewer's tables share: which way a click on a heading
//! orders the rows, and the triangle that marks the sorted column.
//!
//! Track View's observation table (`specs/gui/track-view.md`) and the Track
//! Table panel (`specs/gui/track-table.md`) both follow it. A first click on a
//! column puts its bad end first where it has one and is increasing
//! otherwise; a second click on the same column reverses the order. Each table
//! keeps its own column type and says, through [`SortRule`], which of its
//! columns start decreasing. Where the rows with no value go is the table's
//! own business: both put them last in either order.

/// A table's column type, and which of its columns a first click orders
/// decreasing.
pub(crate) trait SortRule: Copy + Eq {
    /// Whether a click that makes this the column the rows are ordered by
    /// orders them decreasing: true for a column whose large values are the
    /// bad end.
    fn worst_first_descending(self) -> bool;
}

/// The order a table's rows are drawn in: by which column, and which way.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct TableSort<C> {
    /// The column the rows are ordered by.
    pub(crate) column: C,
    /// Whether the order is decreasing rather than increasing.
    pub(crate) descending: bool,
}

impl<C: SortRule> TableSort<C> {
    /// The order a first click on `column`'s heading leaves: that column in
    /// the direction [`SortRule::worst_first_descending`] gives.
    pub(crate) fn first_click(column: C) -> Self {
        Self {
            column,
            descending: column.worst_first_descending(),
        }
    }

    /// The order a click on `column`'s heading leaves: the same column the
    /// other way when the rows are already ordered by it, and otherwise
    /// [`Self::first_click`].
    pub(crate) fn clicked(self, column: C) -> Self {
        if self.column == column {
            Self {
                column,
                descending: !self.descending,
            }
        } else {
            Self::first_click(column)
        }
    }
}

/// Half the width of the triangle beside the heading the rows are ordered by.
pub(crate) const SORT_TRIANGLE_HALF_WIDTH: f32 = 4.0;

/// The small triangle beside the heading the rows are ordered by, centred at
/// `c`: pointing up for increasing order and down for decreasing.
pub(crate) fn paint_sort_triangle(
    painter: &egui::Painter,
    c: egui::Pos2,
    descending: bool,
    color: egui::Color32,
) {
    let (half, rise) = (SORT_TRIANGLE_HALF_WIDTH, 3.0);
    // Clockwise on the screen, as egui's tessellator wants a filled shape.
    let points = if descending {
        vec![
            egui::pos2(c.x - half, c.y - rise),
            egui::pos2(c.x + half, c.y - rise),
            egui::pos2(c.x, c.y + rise),
        ]
    } else {
        vec![
            egui::pos2(c.x - half, c.y + rise),
            egui::pos2(c.x, c.y - rise),
            egui::pos2(c.x + half, c.y + rise),
        ]
    };
    painter.add(egui::Shape::convex_polygon(
        points,
        color,
        egui::Stroke::NONE,
    ));
}
