// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! What the panel's two hand-painted plots share: the projection plot
//! ([`super::projection_plot`]) and the per-image histograms
//! ([`super::histogram`]).
//!
//! Both are drawn with [`egui::Painter`], since the crate has no plotting
//! dependency, so they share the palette, the linear map from data to screen,
//! the 1/2/5 ladder their axes walk and the small font their tick labels are
//! set in. One copy of each keeps the two reading as one panel.

use egui::Color32;

/// Dash and gap of a dashed stroke: the projection plot's ideal map and the
/// histograms' two-standard-deviation bounds.
pub(super) const DASH: (f32, f32) = (5.0, 4.0);

/// Font size of the rules' labels and the axes' tick labels.
pub(super) const SMALL_FONT: f32 = 9.0;

/// The palette, resolved per frame against the viewer's theme.
pub(super) struct Palette {
    pub(super) axis: Color32,
    pub(super) grid: Color32,
    pub(super) text: Color32,
    /// The camera's curve, and while a proposal is open the proposed camera's;
    /// a histogram's bars.
    pub(super) model: Color32,
    /// The ideal map; a histogram's mean and its two-standard-deviation
    /// bounds.
    pub(super) ideal: Color32,
    pub(super) band: Color32,
    pub(super) extrapolated_wash: Color32,
    pub(super) extrapolated_rule: Color32,
    /// The current camera's curve while a proposal is open.
    pub(super) current: Color32,
    pub(super) current_band: Color32,
    pub(super) current_wash: Color32,
    /// The proposed camera's extrapolated region.
    pub(super) proposed_wash: Color32,
    /// Where the monotonicity constraint bound.
    pub(super) monotone: Color32,
    /// The plate under each label: the panel's own colour, nearly opaque.
    pub(super) plate: Color32,
}

impl Palette {
    pub(super) fn resolve(visuals: &egui::Visuals) -> Self {
        if visuals.dark_mode {
            Self {
                axis: Color32::from_gray(110),
                grid: Color32::from_gray(58),
                text: visuals.text_color(),
                model: Color32::from_rgb(0x6c, 0xb6, 0xff),
                ideal: Color32::from_rgb(0xe0, 0xa8, 0x4c),
                band: Color32::from_rgba_unmultiplied(0x6c, 0xb6, 0xff, 44),
                extrapolated_wash: Color32::from_rgba_unmultiplied(0xff, 0x8a, 0x5c, 26),
                extrapolated_rule: Color32::from_rgb(0xff, 0x9d, 0x76),
                current: Color32::from_gray(205),
                current_band: Color32::from_rgba_unmultiplied(205, 205, 205, 40),
                current_wash: Color32::from_rgba_unmultiplied(205, 205, 205, 30),
                proposed_wash: Color32::from_rgba_unmultiplied(0x6c, 0xb6, 0xff, 22),
                monotone: Color32::from_rgb(0xd0, 0x8c, 0xff),
                plate: plate(visuals),
            }
        } else {
            Self {
                axis: Color32::from_gray(120),
                grid: Color32::from_gray(206),
                text: visuals.text_color(),
                model: Color32::from_rgb(0x14, 0x5c, 0xb0),
                ideal: Color32::from_rgb(0x9a, 0x62, 0x00),
                band: Color32::from_rgba_unmultiplied(0x14, 0x5c, 0xb0, 40),
                extrapolated_wash: Color32::from_rgba_unmultiplied(0xc0, 0x4a, 0x10, 26),
                extrapolated_rule: Color32::from_rgb(0xa8, 0x40, 0x0c),
                current: Color32::from_gray(70),
                current_band: Color32::from_rgba_unmultiplied(70, 70, 70, 36),
                current_wash: Color32::from_rgba_unmultiplied(70, 70, 70, 26),
                proposed_wash: Color32::from_rgba_unmultiplied(0x14, 0x5c, 0xb0, 20),
                monotone: Color32::from_rgb(0x7a, 0x2c, 0xb0),
                plate: plate(visuals),
            }
        }
    }
}

/// The plate under the plots' labels: the panel's fill, nearly opaque.
fn plate(visuals: &egui::Visuals) -> Color32 {
    let fill = visuals.panel_fill;
    Color32::from_rgba_unmultiplied(fill.r(), fill.g(), fill.b(), 215)
}

/// One linear map from a data interval to a screen interval.
#[derive(Clone, Copy)]
pub(super) struct Scale {
    pub(super) lo: f64,
    pub(super) hi: f64,
    pub(super) screen_lo: f32,
    pub(super) screen_hi: f32,
}

impl Scale {
    pub(super) fn at(&self, value: f64) -> f32 {
        let t = if self.hi > self.lo {
            (value - self.lo) / (self.hi - self.lo)
        } else {
            0.5
        };
        self.screen_lo + (self.screen_hi - self.screen_lo) * t as f32
    }
}

/// The smallest 1, 2 or 5 times a power of ten that is at least `rough`, or
/// `None` for a `rough` that is not a positive finite number.
///
/// The 1/2/5 ladder rather than an even division so a step reads as a number
/// a person would have picked — 20°, not 18.8°.
pub(super) fn ladder_step(rough: f64) -> Option<f64> {
    if !(rough.is_finite() && rough > 0.0) {
        return None;
    }
    let decade = 10.0_f64.powf(rough.log10().floor());
    Some(
        [1.0, 2.0, 5.0, 10.0]
            .into_iter()
            .map(|m| m * decade)
            .find(|s| *s >= rough * (1.0 - 1e-12))
            .unwrap_or(decade * 10.0),
    )
}

/// Round `span / count` up to the 1/2/5 ladder ([`ladder_step`]), and return
/// the multiples of that step that fall inside `[lo, hi]`.
pub(super) fn ticks(lo: f64, hi: f64, count: usize) -> Vec<f64> {
    let span = hi - lo;
    if span <= 0.0 || span.is_nan() || count == 0 {
        return Vec::new();
    }
    let Some(step) = ladder_step(span / count as f64) else {
        return Vec::new();
    };
    let first = (lo / step).ceil() * step;
    let mut out = Vec::new();
    let mut value = first;
    while value <= hi + step * 1e-9 && out.len() <= count * 3 {
        out.push(value);
        value += step;
    }
    out
}
