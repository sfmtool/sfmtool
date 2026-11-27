// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The dialog that `Scale to Median Measurement...` on a reconstruction's
//! Scene Graph context menu opens, the gate the menu entry reads, and the
//! words both print.
//!
//! See `specs/gui/edits/scale-to-median-measurement.md`. The dialog asks one
//! thing, the physical unit to scale to, and shows what the node's stored
//! distance measurements make of it: the median factor
//! (`sfmtool_core::reconstruction::measurement_scale`), a histogram of each
//! measurement's deviation from it, how well they agree, a row per
//! measurement, and the ones that give no factor and why. It reads the node's
//! cached readings afresh every frame, so it stays true of the version on
//! screen through an undo or an edit made while it is up; the edit itself
//! (`AppState::scale_node_to_median_measurement`) reads them again when
//! `Apply` is pressed, so nothing the dialog computed is carried into it.
//!
//! The gate is here rather than in the menu because the wire reads it too, so
//! the entry and the tool cannot disagree about when there is a scale.

use std::fmt::Display;

use eframe::egui;
use sfmtool_core::reconstruction::measurement_scale::{
    distance_measurement_scale, DistanceMeasurementScale, DistanceMeasurementScaleError,
    SCALE_DISAGREEMENT_PERCENT,
};
use sfmtool_core::LengthUnit;

use crate::intrinsics_detail::histogram::{self, Histogram};
use crate::measurements::{self, DistanceRow, MeasurementRef};
use crate::scene::{ReconId, SceneNode};

#[cfg(test)]
mod tests;

/// What the histogram rounds each deviation to, in percent: ten times finer
/// than the two decimals the table prints, and far coarser than the `float32`
/// a stated distance is stored in ([`deviation_histogram`]).
const DEVIATION_RESOLUTION_PERCENT: f64 = 0.001;

/// The narrowest the dialog is drawn: wide enough for the per-measurement
/// table, which would otherwise be clipped by a window as narrow as its
/// wrapping sentences.
const DIALOG_MIN_WIDTH: f32 = 520.0;

/// Width of the dialog's histogram.
const HISTOGRAM_WIDTH: f32 = 360.0;

/// The most rows the per-measurement table shows before it scrolls.
const TABLE_MAX_HEIGHT: f32 = 220.0;

/// The scale `node`'s stored distance measurements give it in `unit`, read
/// from the measurements of the version on screen and the node's cached
/// readings of them ([`SceneNode::measurement_readings`]).
pub(crate) fn node_scale(
    node: &SceneNode,
    unit: LengthUnit,
) -> Result<DistanceMeasurementScale, DistanceMeasurementScaleError> {
    let readings = node.measurement_readings();
    distance_measurement_scale(node.edited().measurements(), &readings.distance, unit)
}

/// `text` as a sentence: its first letter capitalised and a full stop at the
/// end, for core's error and exclusion texts, which are written to follow a
/// colon.
pub(crate) fn sentence(text: impl Display) -> String {
    let text = text.to_string();
    let mut chars = text.chars();
    let first: String = chars
        .next()
        .into_iter()
        .flat_map(char::to_uppercase)
        .collect();
    let rest = chars.as_str();
    if rest.ends_with('.') {
        format!("{first}{rest}")
    } else {
        format!("{first}{rest}.")
    }
}

/// Why `Scale to Median Measurement...` is greyed on `node` past a busy node
/// and an open proposal, or `None` when it can run.
///
/// Greyed where the version on screen stores no distance measurement, and
/// where no measurement gives a factor in any of the five units: the core
/// function's `NoMeasurements` and `NoUsableMeasurement`, in its own words.
/// A measurement's usability hardly depends on the unit -- only a factor that
/// overflows or underflows in one unit and not another does -- so the entry
/// is live when any unit gives a scale, and the dialog says which do not.
pub(crate) fn refusal(node: &SceneNode) -> Option<String> {
    let mut first = None;
    for unit in LengthUnit::ALL {
        match node_scale(node, unit) {
            Ok(_) => return None,
            Err(e) => {
                first.get_or_insert(e);
            }
        }
    }
    first.map(sentence)
}

/// The unit the dialog opens on for `node`: the file's `world_space_unit`
/// when it declares one, and otherwise the unit most of the usable
/// measurements state their distance in, a tie going to the first of
/// [`LengthUnit::ALL`] (mm, cm, m, in, ft).
pub(crate) fn default_unit(node: &SceneNode) -> LengthUnit {
    if let Some(unit) = node
        .recon()
        .metadata
        .world_space_unit
        .as_deref()
        .and_then(LengthUnit::from_name)
    {
        return unit;
    }
    let usable: Vec<usize> = LengthUnit::ALL
        .into_iter()
        .find_map(|unit| node_scale(node, unit).ok())
        .map(|scale| scale.factors.iter().map(|f| f.index).collect())
        .unwrap_or_default();
    let rows = measurements::distance_rows(node.edited().measurements());
    let count = |unit: LengthUnit| {
        usable
            .iter()
            .filter(|&&i| rows.get(i).is_some_and(|row| row.unit == unit))
            .count()
    };
    // `max_by_key` keeps the last of equal keys, so the units are walked in
    // reverse to keep the first.
    LengthUnit::ALL
        .into_iter()
        .rev()
        .max_by_key(|&unit| count(unit))
        .unwrap_or(LengthUnit::Metre)
}

/// A factor or a statistic as the dialog, the Action Log and the wire's texts
/// print it: four significant digits in plain notation, and exponent notation
/// where plain notation would run to many zeros (`2.716`, `1020`, `1.000e-6`).
pub(crate) fn number_text(value: f64) -> String {
    // Rounded to four significant digits first, so a value just under a power
    // of ten, such as the `0.99999998` a scale that agrees with its unit
    // leaves, prints as `1.000` and not with a fifth digit.
    let rounded = format!("{value:.3e}").parse::<f64>().unwrap_or(value);
    let magnitude = rounded.abs();
    if rounded == 0.0 || !rounded.is_finite() || (1e-4..1e7).contains(&magnitude) {
        measurements::significant(rounded, 4)
    } else {
        format!("{value:.3e}")
    }
}

/// A percentage as the statistics print it, `0.42`: two decimals, so a value
/// that is float noise, such as the median absolute deviation of measurements
/// that agree to the `float32` they are stored in, reads `0.00`.
pub(crate) fn percent_text(percent: f64) -> String {
    format!("{:.2}", if percent.abs() < 0.005 { 0.0 } else { percent })
}

/// A signed percentage, `+0.42` or `−0.20`, with `+0.00` for one that rounds to
/// zero, as the table prints deviations.
pub(crate) fn signed_percent_text(percent: f64) -> String {
    measurements::signed(percent, 2)
}

/// `1 scene unit = 2.716 mm`: what the factor says, in words.
pub(crate) fn scale_text(unit: LengthUnit, factor: f64) -> String {
    format!("1 scene unit = {} {}", number_text(factor), unit.name())
}

/// The factor relative to the unit the file declares, `file_unit`: the number
/// a coordinate already in that unit is multiplied by, which is `1` for
/// measurements that agree with the file exactly.
pub(crate) fn correction(file_unit: LengthUnit, unit: LengthUnit, factor: f64) -> f64 {
    factor * unit.in_metres() / file_unit.in_metres()
}

/// `The file declares m. By these measurements its scale is off by −0.20%
/// (×0.998).`: the measurements compared with the unit the file declares, in
/// words that cannot be taken for the multiplier the scale applies, which is a
/// line of its own. `None` for a file that declares no unit, or one this
/// format does not know.
pub(crate) fn correction_text(
    file_unit: Option<&str>,
    unit: LengthUnit,
    factor: f64,
) -> Option<String> {
    let file_unit = file_unit.and_then(LengthUnit::from_name)?;
    let correction = correction(file_unit, unit, factor);
    Some(format!(
        "The file declares {}. By these measurements its scale is off by {}% (\u{d7}{}).",
        file_unit.name(),
        signed_percent_text(100.0 * (correction - 1.0)),
        number_text(correction)
    ))
}

/// What the dialog shows for one node in one unit, read afresh every frame.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct ScaleView {
    /// The node's label, which the title names.
    pub(crate) label: String,
    /// The `world_space_unit` the version on screen declares.
    pub(crate) file_unit: Option<String>,
    /// The unit chosen.
    pub(crate) unit: LengthUnit,
    /// Every stored distance measurement, for its label and stated distance.
    pub(crate) rows: Vec<DistanceRow>,
    /// The scale, or why there is none.
    pub(crate) scale: Result<DistanceMeasurementScale, DistanceMeasurementScaleError>,
}

impl ScaleView {
    /// The view of `node` in `unit`.
    pub(crate) fn of(node: &SceneNode, unit: LengthUnit) -> Self {
        Self {
            label: node.label.clone(),
            file_unit: node.recon().metadata.world_space_unit.clone(),
            unit,
            rows: measurements::distance_rows(node.edited().measurements()),
            scale: node_scale(node, unit),
        }
    }

    /// `Distance #4  paw width`: a measurement's title and its label, the
    /// label left out where it is empty.
    pub(crate) fn title(&self, recon: ReconId, index: usize) -> String {
        let title = MeasurementRef::distance(recon, index).title();
        match self.rows.get(index) {
            Some(row) if !row.label.is_empty() => format!("{title}  {}", row.label),
            _ => title,
        }
    }

    /// The statistics lines, in the order the dialog draws them.
    pub(crate) fn statistics_lines(&self, recon: ReconId) -> Vec<String> {
        let Ok(scale) = &self.scale else {
            return Vec::new();
        };
        let s = &scale.statistics;
        let total = s.count + scale.excluded.len();
        let n = number_text;
        let mut lines = vec![
            format!("Used {} of {total} distance measurements", s.count),
            format!("Median {}, mean {}", n(s.median), n(s.mean)),
            match (s.sample_std, s.sample_std_percent) {
                (Some(std), Some(percent)) => format!(
                    "Standard deviation {} ({}% of the median)",
                    n(std),
                    measurements::significant(percent, 3)
                ),
                _ => "Standard deviation: none for a single measurement".to_string(),
            },
            format!(
                "Median absolute deviation {}% of the median, min {}, max {}",
                percent_text(100.0 * s.median_absolute_deviation / s.median),
                n(s.min),
                n(s.max)
            ),
        ];
        if let Some(worst) = scale.factors.iter().max_by(|a, b| {
            a.deviation_percent
                .abs()
                .total_cmp(&b.deviation_percent.abs())
        }) {
            lines.push(format!(
                "Largest deviation {}% ({})",
                measurements::signed(worst.deviation_percent, 2),
                self.title(recon, worst.index)
            ));
        }
        lines
    }

    /// The warning the dialog shows when the measurements disagree, or `None`.
    pub(crate) fn disagreement(&self) -> Option<String> {
        let scale = self.scale.as_ref().ok()?;
        if !scale.statistics.disagrees {
            return None;
        }
        let off = scale
            .factors
            .iter()
            .filter(|f| f.deviation_percent.abs() > SCALE_DISAGREEMENT_PERCENT)
            .count();
        let them = if off == 1 {
            "1 measurement is".to_string()
        } else {
            format!("{off} measurements are")
        };
        Some(format!(
            "{} The measurements disagree: {them} more than {SCALE_DISAGREEMENT_PERCENT}% from \
             the median. Check their endpoints before applying.",
            measurements::WARNING
        ))
    }

    /// Why `Apply` is greyed for the scale itself, or `None`: the scale is an
    /// error.
    pub(crate) fn scale_refusal(&self) -> Option<String> {
        self.scale.as_ref().err().map(sentence)
    }
}

/// What the user asked for, once they pressed `Apply`: the node and the unit.
/// The factor is not carried, since the edit reads the readings again.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct MeasurementScaleAnswer {
    pub(crate) recon: ReconId,
    pub(crate) unit: LengthUnit,
}

/// The dialog, and what it remembers while it is up.
#[derive(Default)]
pub(crate) struct MeasurementScalePrompt {
    pending: Option<MeasurementScaleAnswer>,
}

impl MeasurementScalePrompt {
    /// Ask about `recon`, opening on `unit`.
    ///
    /// Idempotent while the dialog is already up, so a menu item racing itself
    /// cannot stack two of them, and the question already on screen is the one
    /// that gets answered.
    pub(crate) fn ask(&mut self, recon: ReconId, unit: LengthUnit) {
        if self.pending.is_none() {
            self.pending = Some(MeasurementScaleAnswer { recon, unit });
        }
    }

    /// The node the dialog is up for and the unit chosen in it, or `None`.
    pub(crate) fn pending(&self) -> Option<MeasurementScaleAnswer> {
        self.pending
    }

    /// Draw one frame of the dialog over `node`, the node it is up for as the
    /// scene holds it now, returning the answer on the frame `Apply` is
    /// pressed. `busy` is why an edit of the node is refused right now.
    ///
    /// A node that is no longer loaded closes the dialog. Enter applies and
    /// Escape cancels, as in the Bundle Adjust dialog; Enter does nothing while
    /// `Apply` is greyed.
    pub(crate) fn show(
        &mut self,
        ctx: &egui::Context,
        node: Option<&SceneNode>,
        busy: Option<String>,
    ) -> Option<MeasurementScaleAnswer> {
        let pending = self.pending.as_mut()?;
        let Some(node) = node.filter(|node| node.id == pending.recon) else {
            self.pending = None;
            return None;
        };
        let mut apply = false;
        let mut cancel = false;
        let mut still_open = true;
        let mut refusal = None;

        egui::Window::new(format!("Scale {} to Median Measurement", node.label))
            .id(egui::Id::new("measurement_scale_prompt"))
            // Wide enough for the table's five columns; the sentences around it wrap.
            .min_width(DIALOG_MIN_WIDTH)
            .open(&mut still_open)
            .collapsible(false)
            .resizable(false)
            .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
            .show(ctx, |ui| {
                unit_row(ui, &mut pending.unit);
                // After the unit row, so a change of unit is shown on the frame
                // it is made.
                let view = ScaleView::of(node, pending.unit);
                ui.add_space(6.0);
                body(ui, node.id, &view);
                ui.add_space(8.0);
                refusal = busy.clone().or_else(|| view.scale_refusal());
                ui.horizontal(|ui| {
                    let button = ui
                        .add_enabled(refusal.is_none(), egui::Button::new("Apply"))
                        .on_disabled_hover_text(refusal.clone().unwrap_or_default())
                        .on_hover_text(
                            "Multiply every coordinate of this reconstruction by the factor \
                             and declare the unit, as one version. Undo (Ctrl+Z) puts both back.",
                        );
                    if button.clicked() {
                        apply = true;
                    }
                    if ui.button("Cancel").clicked() {
                        cancel = true;
                    }
                });
                apply |= ui.input(|i| i.key_pressed(egui::Key::Enter));
                cancel |= ui.input(|i| i.key_pressed(egui::Key::Escape));
            });

        let answer = (apply && refusal.is_none()).then_some(*pending);
        if answer.is_some() || cancel || !still_open {
            self.pending = None;
        }
        answer
    }
}

/// The unit picker: `Unit` and a drop-down of the five units.
fn unit_row(ui: &mut egui::Ui, unit: &mut LengthUnit) {
    ui.horizontal(|ui| {
        // `labelled_by` the label, which names the drop-down for a screen
        // reader and in `get_widgets`.
        let label = ui.label("Unit");
        egui::ComboBox::from_id_salt("measurement_scale_unit")
            .selected_text(unit.name())
            .show_ui(ui, |ui| {
                for choice in LengthUnit::ALL {
                    ui.selectable_value(unit, choice, choice.name());
                }
            })
            .response
            .labelled_by(label.id)
            .on_hover_text(
                "The physical unit to scale the reconstruction to. Each measurement's stated \
                 distance is converted to it, so the factor is the same scale in any unit.",
            );
    });
}

/// Everything between the unit row and the buttons.
fn body(ui: &mut egui::Ui, recon: ReconId, view: &ScaleView) {
    let scale = match &view.scale {
        Ok(scale) => scale,
        Err(e) => {
            ui.label(sentence(e));
            if let DistanceMeasurementScaleError::NoUsableMeasurement { excluded } = e {
                excluded_list(ui, recon, view, excluded);
            }
            return;
        }
    };
    let factor = scale.factor();
    ui.label(egui::RichText::new(scale_text(view.unit, factor)).strong());
    ui.label(format!(
        "Multiplies every coordinate by {}",
        number_text(factor)
    ));
    match correction_text(view.file_unit.as_deref(), view.unit, factor) {
        Some(text) => ui.label(text),
        None => ui.label("The file declares no unit: its coordinates are in scene units."),
    };
    ui.add_space(6.0);

    let shown = deviation_histogram(scale);
    let value = |v: f64| format!("{}%", measurements::signed(v, 2));
    let name = |i: usize| format!("#{i}");
    let labels = histogram::Labels {
        component: "deviation from the median",
        unit: "%",
        items: ["measurement", "measurements"],
        per: "",
        selects: false,
        value: &value,
        name: &name,
    };
    ui.label(egui::RichText::new("Deviation from the median, %").weak());
    let width = HISTOGRAM_WIDTH.min(ui.available_width());
    histogram::show_histogram(
        ui,
        egui::Id::new("measurement_scale_histogram"),
        &shown,
        &labels,
        None,
        width,
    );
    ui.add_space(6.0);

    for line in view.statistics_lines(recon) {
        ui.label(line);
    }
    if let Some(warning) = view.disagreement() {
        ui.label(egui::RichText::new(warning).color(ui.visuals().warn_fg_color));
    }
    ui.add_space(6.0);
    factor_table(ui, recon, view, scale);
    if !scale.excluded.is_empty() {
        ui.add_space(6.0);
        excluded_list(ui, recon, view, &scale.excluded);
    }
}

/// The histogram the dialog draws: each usable measurement's signed deviation
/// from the median, in percent, drawn with the Camera Intrinsics panel's
/// painter, with the median as the solid line at `0` and the disagreement bar,
/// [`SCALE_DISAGREEMENT_PERCENT`] either side, as the dashed ones.
///
/// Deviations rather than the factors themselves, so the axis reads the same
/// in every unit and the bar is the same two lines whatever the scale.
///
/// **The bins are centred on the median.** Their width is the one
/// [`Histogram::new`] chooses, the step of the 1/2/5 ladder nearest the
/// values' spread over about `⌈√n⌉` bins, at least 2 and at most 20
/// ([`histogram::nearest_step`]), or one percent where every value is the
/// same. But the edges are at `±w/2, ±3w/2, …` rather than at multiples of
/// `w`, so a bin runs from `−w/2` to `+w/2` and a deviation of exactly `0` is
/// in a bar symmetric about the median line; with edges at multiples of `w`,
/// measurements that agree exactly fill a bar from `0` to `w` and read as a
/// spread of up to `w`. The axis is symmetric too: it reaches the largest
/// deviation on either side, and the disagreement bar where that lies within a
/// spread of the values, as `Histogram::new` reaches its bounds, so the median
/// line is in the middle even when every deviation is on one side of it.
///
/// Each deviation is rounded to [`DEVIATION_RESOLUTION_PERCENT`] first. A
/// stated distance is stored as a `float32`, so measurements that agree
/// exactly still deviate by about `1e-6` percent, which would otherwise make a
/// spread a millionth of a percent wide and bins to match.
pub(crate) fn deviation_histogram(scale: &DistanceMeasurementScale) -> Histogram {
    let round = |percent: f64| {
        (percent / DEVIATION_RESOLUTION_PERCENT).round() * DEVIATION_RESOLUTION_PERCENT
    };
    let mut values: Vec<(usize, f64)> = scale
        .factors
        .iter()
        .map(|f| (f.index, round(f.deviation_percent)))
        .filter(|(_, v)| v.is_finite())
        .collect();
    values.sort_by(|a, b| a.1.total_cmp(&b.1).then(a.0.cmp(&b.0)));
    let bound = SCALE_DISAGREEMENT_PERCENT;
    let lo = values.first().map_or(0.0, |v| v.1);
    let hi = values.last().map_or(0.0, |v| v.1);
    let span = hi - lo;
    let width = if span > 0.0 {
        let target = (values.len() as f64)
            .sqrt()
            .ceil()
            .clamp(histogram::MIN_BINS as f64, histogram::MAX_BINS as f64);
        histogram::nearest_step(span / target)
    } else {
        1.0
    };
    // How far the axis reaches either side of the median: the largest
    // deviation, and the bar where it lies within a spread of the values.
    let reach = lo.abs().max(hi.abs());
    let reach = reach.max(bound.min(reach + span));
    // Bins either side of the centre one: enough that the outermost edge,
    // `(k + 1/2)·w`, is past `reach`.
    let k = (reach / width - 0.5 - 1e-9).ceil().max(0.0) as i64;
    let edges: Vec<f64> = (-k..=k + 1).map(|i| (i as f64 - 0.5) * width).collect();
    let mut bins = vec![Vec::new(); edges.len() - 1];
    for (index, value) in values {
        let bin = ((value - edges[0]) / width + 1e-9).floor();
        let bin = bin.clamp(0.0, (bins.len() - 1) as f64) as usize;
        bins[bin].push((index, value));
    }
    Histogram {
        edges,
        bins,
        mean: 0.0,
        two_sd: [-bound, bound],
    }
}

/// One row per usable measurement: its index and label, its stated distance in
/// its own unit, its distance in scene units, its factor and its signed
/// deviation from the median, a row more than
/// [`SCALE_DISAGREEMENT_PERCENT`] off drawn in the warning colour.
fn factor_table(
    ui: &mut egui::Ui,
    recon: ReconId,
    view: &ScaleView,
    scale: &DistanceMeasurementScale,
) {
    egui::ScrollArea::vertical()
        .id_salt("measurement_scale_table")
        .max_height(TABLE_MAX_HEIGHT)
        .show(ui, |ui| {
            egui::Grid::new("measurement_scale_factors")
                .num_columns(5)
                .spacing([12.0, 2.0])
                .striped(true)
                .show(ui, |ui| {
                    for heading in [
                        "Measurement",
                        "Stated",
                        "Scene distance",
                        "Factor",
                        "Deviation",
                    ] {
                        ui.label(egui::RichText::new(heading).strong());
                    }
                    ui.end_row();
                    let warn = ui.visuals().warn_fg_color;
                    for f in &scale.factors {
                        let off = f.deviation_percent.abs() > SCALE_DISAGREEMENT_PERCENT;
                        let cell = |text: String| {
                            let text = egui::RichText::new(text);
                            if off {
                                text.color(warn)
                            } else {
                                text
                            }
                        };
                        let stated = view
                            .rows
                            .get(f.index)
                            .map_or_else(String::new, DistanceRow::stated_text);
                        ui.label(cell(view.title(recon, f.index)));
                        ui.label(cell(stated));
                        ui.label(cell(format!(
                            "{} su",
                            measurements::significant(f.scene_distance, 4)
                        )));
                        ui.label(cell(number_text(f.factor)));
                        ui.label(cell(format!(
                            "{}%",
                            measurements::signed(f.deviation_percent, 2)
                        )));
                        ui.end_row();
                    }
                });
        });
}

/// The measurements that give no factor, each with its reason.
fn excluded_list(
    ui: &mut egui::Ui,
    recon: ReconId,
    view: &ScaleView,
    excluded: &[sfmtool_core::reconstruction::measurement_scale::ExcludedMeasurement],
) {
    ui.label(egui::RichText::new("Left out").strong());
    for e in excluded {
        ui.label(format!(
            "{}: {}",
            view.title(recon, e.index),
            sentence(&e.reason)
        ));
    }
}
