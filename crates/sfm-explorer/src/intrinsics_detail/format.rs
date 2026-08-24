// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! How the panel spells a number, a vector and a matrix.
//!
//! One module because the panel's whole job is to be read against another copy
//! of the same numbers — `sfm inspect`'s output, or whatever the user pastes
//! them into — and three sections spelling six decimals three different ways
//! would defeat that before any of them was wrong.
//!
//! Everything is monospaced and right-aligned by *padding*, not by layout:
//! in a monospaced font a `{:>W.6}` field is already a column, and a padded
//! string survives being copied to the clipboard, which a layout does not.

/// Decimals every number in the panel is shown to — the same six
/// `sfm inspect` prints, so the two can be diffed line by line.
pub(super) const DECIMALS: usize = 6;

/// Column width of a value cell: sign, up to six integer digits, the point and
/// [`DECIMALS`] after it. Wide enough for a focal length in pixels and for the
/// large translations a metric reconstruction can carry.
const VALUE_WIDTH: usize = 14;

/// One number, right-aligned in a [`VALUE_WIDTH`] monospaced column.
pub(super) fn value(v: f64) -> String {
    format!(
        "{v:>width$.decimals$}",
        width = VALUE_WIDTH,
        decimals = DECIMALS
    )
}

/// A row of numbers as one monospaced string, two spaces between columns.
pub(super) fn row(values: impl IntoIterator<Item = f64>) -> String {
    values.into_iter().map(value).collect::<Vec<_>>().join("  ")
}

/// A matrix as one monospaced block, one line per row — what the `Copy ▾`
/// entries put on the clipboard.
pub(super) fn matrix_text(rows: &[Vec<f64>]) -> String {
    rows.iter()
        .map(|r| row(r.iter().copied()))
        .collect::<Vec<_>>()
        .join("\n")
}

/// Draw a matrix as a grid of right-aligned monospaced cells.
///
/// `id_salt` scopes the grid's own id: two matrices in one panel are two grids,
/// and egui needs to tell them apart across frames.
pub(super) fn matrix_grid(ui: &mut egui::Ui, id_salt: &str, rows: &[Vec<f64>]) {
    egui::Grid::new(id_salt)
        .num_columns(rows.first().map_or(1, Vec::len))
        .spacing([12.0, 2.0])
        .show(ui, |ui| {
            for r in rows {
                for v in r {
                    ui.monospace(value(*v));
                }
                ui.end_row();
            }
        });
}

/// A labelled row of numbers: the label, then the numbers as one monospaced
/// run. Used for the vectors — `t`, the quaternion, the camera centre, the
/// world axes — which are one line each and would be three grids otherwise.
pub(super) fn labelled_row(ui: &mut egui::Ui, label: &str, values: impl IntoIterator<Item = f64>) {
    ui.horizontal(|ui| {
        ui.label(label);
        ui.monospace(row(values));
    });
}

/// A three-vector out of nalgebra, as three `f64`s.
pub(super) fn xyz(v: &nalgebra::Vector3<f64>) -> [f64; 3] {
    [v.x, v.y, v.z]
}
