// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The small histogram each `per image` row of the parameter table draws: how
//! a camera's images' values of one component are spread.
//!
//! [`Histogram`] is the arithmetic, shared with MCP `get_camera_intrinsics`,
//! which reports the same bins; [`show_histogram`] paints one and turns a
//! pointer over a bin into a tooltip, a hovered image and a selection. Like
//! the projection plot it is hand-painted ([`super::plot`]): the crate has no
//! plotting dependency.
//!
//! # The bins
//!
//! A histogram of `n` values aims for `⌈√n⌉` bins, at least
//! [`MIN_BINS`] and at most [`MAX_BINS`], and rounds the width that gives,
//! the spread of the values over that count, to the nearest step of the 1/2/5
//! ladder the panel's axes walk ([`nearest_step`]), so every edge is a
//! multiple of a round step in the unit the row shows. Nearest rather than up,
//! since rounding a width of 2.1 up to 5 would leave 17 images in four bins.
//! The bins then run from the step below the smaller of the
//! smallest value and the lower two-standard-deviation bound to the step past
//! the larger of the largest value and the upper bound, so both bounds and
//! every value are on the axis. A bin holds the values from its lower edge up
//! to, and not including, its upper edge. Values that are all equal, a single
//! image's among them, make one bin centred on the value, as wide as the
//! smallest ladder step of at least a fifth of it, or one unit wide for a
//! value of zero.

use egui::{FontId, Pos2, Rect, Sense, Shape, Stroke, Vec2};
use sfmtool_core::reconstruction::image_adjustment_statistics::ComponentStatistics;

use super::plot::{ladder_step, ticks, Palette, Scale, DASH, SMALL_FONT};

/// The fewest bins a histogram of distinct values aims for.
pub(crate) const MIN_BINS: usize = 2;
/// The most bins a histogram aims for, however many images it is over.
pub(crate) const MAX_BINS: usize = 20;

/// Width of a histogram in the panel, or the cell's width where that is
/// narrower.
pub(super) const WIDTH: f32 = 180.0;
/// Height of the bars' area.
const BARS_HEIGHT: f32 = 28.0;
/// Height of the axis's tick labels under the bars.
const AXIS_HEIGHT: f32 = 12.0;
/// The whole histogram's height.
pub(super) const HEIGHT: f32 = BARS_HEIGHT + AXIS_HEIGHT;
/// The most images a bin's tooltip lists before it says how many more there
/// are.
const LISTED: usize = 12;

/// One component's values over a camera's images, binned, in the unit the
/// panel shows them in.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Histogram {
    /// The bin edges, ascending, one more than there are bins.
    pub(crate) edges: Vec<f64>,
    /// Each bin's images, as an image index and its value, ordered by value
    /// and then by index.
    pub(crate) bins: Vec<Vec<(usize, f64)>>,
    /// The mean the row states.
    pub(crate) mean: f64,
    /// The values two standard deviations either side of the mean, past which
    /// the row counts an image.
    pub(crate) two_sd: [f64; 2],
}

impl Histogram {
    /// The histogram of `values`, each an image index and its value in the
    /// unit the file stores, shown `scale` times larger, with the mean and
    /// the two-standard-deviation bounds of `statistics`.
    ///
    /// `log_space` is the focal's case: `statistics` were taken over
    /// `ln(1 + s)`, so the mean is already the geometric one and the bounds
    /// are `exp(mean ln(1 + s) ± 2 sd) − 1`, which mark the same images as
    /// past two standard deviations as the row's count does.
    pub(crate) fn of_component(
        values: impl IntoIterator<Item = (usize, f64)>,
        statistics: &ComponentStatistics,
        scale: f64,
        log_space: bool,
    ) -> Self {
        let mean = scale * statistics.mean;
        let two_sd = if log_space {
            let centre = statistics.mean.ln_1p();
            [-1.0, 1.0].map(|side| scale * (centre + side * 2.0 * statistics.std).exp_m1())
        } else {
            [-1.0, 1.0].map(|side| mean + side * 2.0 * scale * statistics.std)
        };
        let values: Vec<(usize, f64)> = values
            .into_iter()
            .map(|(image, value)| (image, scale * value))
            .collect();
        Self::new(values, mean, two_sd)
    }

    /// The histogram of `values` in the unit shown, with its marks.
    ///
    /// A value that is not finite is in no bin.
    pub(crate) fn new(mut values: Vec<(usize, f64)>, mean: f64, two_sd: [f64; 2]) -> Self {
        values.retain(|(_, v)| v.is_finite());
        values.sort_by(|a, b| a.1.total_cmp(&b.1).then(a.0.cmp(&b.0)));
        let plain: Vec<f64> = values.iter().map(|(_, v)| *v).collect();
        let edges = bin_edges(&plain, two_sd);
        let mut bins = vec![Vec::new(); edges.len().saturating_sub(1)];
        for (image, value) in values {
            if let Some(k) = bin_of(&edges, value) {
                bins[k].push((image, value));
            }
        }
        Self {
            edges,
            bins,
            mean,
            two_sd,
        }
    }

    /// How many images each bin holds.
    pub(crate) fn counts(&self) -> Vec<usize> {
        self.bins.iter().map(Vec::len).collect()
    }

    /// The image a click on bin `k` selects: the one after `selected` in the
    /// bin's order when `selected` is in it, wrapping round, and the bin's
    /// first otherwise, so repeated clicks step through the bin's images.
    pub(crate) fn next_in_bin(&self, k: usize, selected: Option<usize>) -> Option<usize> {
        let bin = self.bins.get(k)?;
        let first = bin.first()?.0;
        let Some(at) = selected.and_then(|s| bin.iter().position(|(i, _)| *i == s)) else {
            return Some(first);
        };
        Some(bin[(at + 1) % bin.len()].0)
    }

    /// The bin holding image `image`, if any.
    fn bin_holding(&self, image: usize) -> Option<usize> {
        self.bins
            .iter()
            .position(|bin| bin.iter().any(|(i, _)| *i == image))
    }

    /// The width of one bin.
    fn step(&self) -> f64 {
        match self.edges.as_slice() {
            [a, b, ..] => b - a,
            _ => 1.0,
        }
    }
}

/// The bin edges for `values` (sorted or not, all finite) whose two
/// standard-deviation bounds are `two_sd`, by the rule in the module's
/// documentation; empty for no values.
pub(crate) fn bin_edges(values: &[f64], two_sd: [f64; 2]) -> Vec<f64> {
    let Some(lo) = values.iter().copied().reduce(f64::min) else {
        return Vec::new();
    };
    let hi = values.iter().copied().fold(lo, f64::max);
    let span = hi - lo;
    if span <= 1e-9 * lo.abs().max(hi.abs()) {
        let width = ladder_step(lo.abs() / 5.0).unwrap_or(1.0);
        return vec![lo - width / 2.0, lo + width / 2.0];
    }
    let target = (values.len() as f64)
        .sqrt()
        .ceil()
        .clamp(MIN_BINS as f64, MAX_BINS as f64);
    let step = nearest_step(span / target);
    // The bounds of the values' own statistics lie within a spread of them;
    // held there, bounds handed in from elsewhere cannot make a histogram of
    // a thousand empty bins.
    let marks = two_sd
        .into_iter()
        .filter(|v| v.is_finite())
        .map(|v| v.clamp(lo - span, hi + span));
    let lo = marks.clone().fold(lo, f64::min);
    let hi = marks.fold(hi, f64::max);
    let first = (lo / step + 1e-9).floor() as i64;
    let last = (hi / step + 1e-9).floor() as i64 + 1;
    (first..=last).map(|k| k as f64 * step).collect()
}

/// The step of the 1/2/5 ladder nearest `rough` on a log scale, for a
/// positive finite `rough`.
pub(crate) fn nearest_step(rough: f64) -> f64 {
    let decade = 10.0_f64.powf(rough.log10().floor());
    [1.0, 2.0, 5.0, 10.0]
        .into_iter()
        .map(|m| m * decade)
        .min_by(|a, b| (a / rough).ln().abs().total_cmp(&(b / rough).ln().abs()))
        .unwrap_or(decade)
}

/// The bin of `edges` that holds `value`, one of the values the edges were
/// made for: from its lower edge up to, and not including, its upper one.
///
/// The edges cover every such value, so a value that rounding puts a hair
/// past the first or last edge, as it can where the values' spread is a
/// hundred-millionth of their size, is held in the end bin rather than
/// dropped.
fn bin_of(edges: &[f64], value: f64) -> Option<usize> {
    let bins = edges.len().checked_sub(1).filter(|&n| n > 0)?;
    let step = (edges[bins] - edges[0]) / bins as f64;
    let k = ((value - edges[0]) / step + 1e-9).floor();
    k.is_finite()
        .then(|| k.clamp(0.0, (bins - 1) as f64) as usize)
}

/// Decimals that tell two edges `step` apart, and no more.
fn decimals_for(step: f64) -> usize {
    if step.is_finite() && step > 0.0 {
        (-step.log10() - 1e-9).ceil().clamp(0.0, 9.0) as usize
    } else {
        3
    }
}

/// An edge or tick: `0` for a value that rounds to zero, and a negative one
/// with the minus the panel's other numbers print (U+2212).
fn edge_text(value: f64, decimals: usize) -> String {
    let text = format!("{:.decimals$}", value.abs());
    if text.chars().all(|c| c == '0' || c == '.') {
        "0".to_string()
    } else if value < 0.0 {
        format!("\u{2212}{text}")
    } else {
        text
    }
}

/// What one row's histogram is about, for its tooltips and its bins' names.
pub(super) struct Labels<'a> {
    /// The component, `focal`, `Δcx`, `ωz`, as the row's label names it.
    pub(super) component: &'a str,
    /// The unit its values are in, `%`, `px`, `mrad per short side`, or empty
    /// for the radial coefficients.
    pub(super) unit: &'a str,
    /// A value as the tooltip lists it, in the row's spelling.
    pub(super) value: &'a dyn Fn(f64) -> String,
    /// An image's name.
    pub(super) name: &'a dyn Fn(usize) -> String,
}

/// What the pointer did on a histogram this frame.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(super) struct HistogramResponse {
    /// The image a click on a bin selected.
    pub(super) select: Option<usize>,
    /// The image the next click on the bin under the pointer selects.
    pub(super) hovered: Option<usize>,
}

/// Paint `histogram` `width` wide and handle the pointer over its bins.
///
/// The bars are the bins' counts, scaled to the fullest bin. The mean is a
/// solid line and the two-standard-deviation bounds dashed ones, in the
/// colour of the projection plot's ideal map, and zero a grey line when it is
/// on the axis. The bin holding `selected`, the selected image, is outlined.
/// Hovering a bin highlights it and lists its images in a tooltip, and a
/// click selects the next of them ([`Histogram::next_in_bin`]).
pub(super) fn show_histogram(
    ui: &mut egui::Ui,
    id: egui::Id,
    histogram: &Histogram,
    labels: &Labels<'_>,
    selected: Option<usize>,
    width: f32,
) -> HistogramResponse {
    let mut out = HistogramResponse::default();
    let (rect, _) = ui.allocate_exact_size(Vec2::new(width, HEIGHT), Sense::hover());
    let (Some(&lo), Some(&hi)) = (histogram.edges.first(), histogram.edges.last()) else {
        return out;
    };
    let palette = Palette::resolve(ui.visuals());
    let bars = Rect::from_min_size(rect.min, Vec2::new(rect.width(), BARS_HEIGHT));
    let x = Scale {
        lo,
        hi,
        screen_lo: bars.left() + 1.0,
        screen_hi: bars.right() - 1.0,
    };
    let decimals = decimals_for(histogram.step());
    let fullest = histogram
        .bins
        .iter()
        .map(Vec::len)
        .max()
        .unwrap_or(0)
        .max(1);
    let total: usize = histogram.bins.iter().map(Vec::len).sum();
    let selected_bin = selected.and_then(|s| histogram.bin_holding(s));
    let unit = if labels.unit.is_empty() {
        String::new()
    } else {
        format!(" {}", labels.unit)
    };

    let painter = ui.painter_at(rect.expand(1.0));
    let mut shapes = Vec::new();
    for (k, bin) in histogram.bins.iter().enumerate() {
        if bin.is_empty() {
            continue;
        }
        let column = Rect::from_x_y_ranges(
            x.at(histogram.edges[k])..=x.at(histogram.edges[k + 1]),
            bars.y_range(),
        );
        let from = edge_text(histogram.edges[k], decimals);
        let to = edge_text(histogram.edges[k + 1], decimals);
        let count = bin.len();
        let images = if count == 1 { "image" } else { "images" };
        let response = ui.interact(column, id.with(k), Sense::click());
        let name = format!(
            "per image {} {from} to {to}{unit}: {count} {images}",
            labels.component
        );
        response.widget_info(|| egui::WidgetInfo::labeled(egui::WidgetType::Button, true, &name));
        let next = histogram.next_in_bin(k, selected);
        let hovered = response.hovered();
        if hovered {
            out.hovered = next;
        }
        if response.clicked() {
            out.select = next;
        }
        response.on_hover_ui(|ui| {
            ui.label(format!(
                "{} from {from} to {to}{unit}: {count} of {total} images",
                labels.component
            ));
            for &(image, value) in bin.iter().take(LISTED) {
                ui.monospace(format!(
                    "{}  {}",
                    (labels.name)(image),
                    (labels.value)(value)
                ));
            }
            if count > LISTED {
                ui.label(format!("… and {} more", count - LISTED));
            }
            if let Some(next) = next {
                let line = if Some(next) == selected {
                    format!("{} is selected.", (labels.name)(next))
                } else {
                    format!("Click to select {}.", (labels.name)(next))
                };
                ui.label(egui::RichText::new(line).weak());
            }
        });

        let height = (BARS_HEIGHT - 2.0) * count as f32 / fullest as f32;
        let bar = Rect::from_x_y_ranges(
            (column.left() + 0.5)..=(column.right() - 0.5),
            (bars.bottom() - height)..=bars.bottom(),
        );
        let fill = if hovered {
            palette.model
        } else {
            palette.model.gamma_multiply(0.6)
        };
        shapes.push(Shape::rect_filled(bar, 0.0, fill));
        if hovered || selected_bin == Some(k) {
            shapes.push(Shape::rect_stroke(
                bar,
                0.0,
                Stroke::new(1.0, ui.visuals().strong_text_color()),
                egui::StrokeKind::Outside,
            ));
        }
    }

    // Zero first, so the mean drawn over it stays visible where the two meet,
    // as they do right after a solve.
    let rule = |value: f64| {
        let px = x.at(value);
        [Pos2::new(px, bars.top()), Pos2::new(px, bars.bottom())]
    };
    if lo <= 0.0 && 0.0 <= hi {
        painter.line_segment(rule(0.0), Stroke::new(1.0, palette.axis));
    }
    painter.extend(shapes);
    painter.line_segment(
        [bars.left_bottom(), bars.right_bottom()],
        Stroke::new(1.0, palette.axis),
    );
    for bound in histogram.two_sd {
        if bound.is_finite() && bound != histogram.mean && (lo..=hi).contains(&bound) {
            painter.extend(Shape::dashed_line(
                &rule(bound),
                Stroke::new(1.0, palette.ideal),
                DASH.0 / 2.0,
                DASH.1 / 2.0,
            ));
        }
    }
    if (lo..=hi).contains(&histogram.mean) {
        painter.line_segment(rule(histogram.mean), Stroke::new(1.5, palette.ideal));
    }

    // Tick labels on the ladder, each centred under its tick, moved inward
    // where that would run past the histogram's ends, and left off where it
    // would run into the one before it.
    let font = FontId::proportional(SMALL_FONT);
    let count = ((width / 45.0) as usize).max(2);
    let mut right_of_last = f32::NEG_INFINITY;
    let marks = ticks(lo, hi, count);
    let tick_decimals = match marks.as_slice() {
        [a, b, ..] => decimals_for(b - a),
        _ => decimals,
    };
    for value in marks {
        let px = x.at(value);
        let label =
            painter.layout_no_wrap(edge_text(value, tick_decimals), font.clone(), palette.text);
        let left = (px - label.size().x / 2.0)
            .min(rect.right() - label.size().x)
            .max(rect.left());
        let at = Pos2::new(left, bars.bottom() + 1.0);
        if at.x < right_of_last + 4.0 || at.x + label.size().x > rect.right() + 0.5 {
            continue;
        }
        painter.line_segment(
            [
                Pos2::new(px, bars.bottom()),
                Pos2::new(px, bars.bottom() + 2.0),
            ],
            Stroke::new(1.0, palette.axis),
        );
        right_of_last = at.x + label.size().x;
        painter.galley(at, label, palette.text);
    }
    out
}
