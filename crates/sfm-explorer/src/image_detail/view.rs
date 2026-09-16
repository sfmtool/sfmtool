// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Where the Image Detail panel is looking, and the one function that moves it
//! there.
//!
//! See `specs/gui/multi-panel-image-browser.md` § "Image Detail: 2D pan and
//! zoom navigation" and § "Revealing a feature named by another panel". The
//! view itself is two numbers on [`super::ImageDetail`] -- a `pan` in panel
//! points and a `zoom` relative to fit -- and everything that *asks* for a view
//! says so as a [`Look`], which [`look_at`] turns into that pair.
//!
//! Three things about the shape here:
//!
//! - **The arithmetic is a pure function over a [`ViewGeometry`]**, so the
//!   panel's own reveal and the wire's `set_image_detail_view` are one
//!   computation rather than two that could come to disagree about where a
//!   pixel lands. It also puts the whole of it under headless test: a
//!   `ViewGeometry` is five numbers and needs no window.
//! - **A [`Look`] is an intent, not a `pan`.** "Centre this pixel" and "fit
//!   this rectangle" want different arithmetic and the caller has neither the
//!   panel's size nor the image's, which is the reason the request travels as
//!   the question rather than as an answer.
//! - **The geometry is published rather than asked for.** The panel is the only
//!   thing that knows how big its body is, and it knows that only while it is
//!   drawing, so it writes the frame's geometry into
//!   [`AppState::image_detail_view`](crate::state::AppState::image_detail_view)
//!   and the wire reads it back. A view tool arriving before the panel has ever
//!   drawn therefore has nothing to measure against, and says so.

use crate::scene::ImageRef;

use super::{MAX_ZOOM, PAN_MARGIN, REVEAL_MARGIN};

/// What the panel was looking at on the frame it last drew an image: the view
/// it held, and the frame it held it in.
///
/// Both halves, because neither is meaningful alone. `pan` is in panel points,
/// so it frames a different part of an image of another resolution, and `zoom`
/// is relative to a fit that only the panel's size and the image's size settle.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct ViewGeometry {
    /// The image on screen.
    pub(crate) image: ImageRef,
    /// Its size in its own pixels.
    pub(crate) image_size: [f32; 2],
    /// The panel body it is drawn in, in points.
    pub(crate) panel_size: [f32; 2],
    /// Offset of the image centre from the panel centre, in panel points.
    pub(crate) pan: [f32; 2],
    /// Magnification, where 1.0 fits the image to the panel.
    pub(crate) zoom: f32,
}

impl ViewGeometry {
    /// Panel points one source pixel spans at zoom 1: the fit.
    ///
    /// The smaller of the two ratios, which is what fitting an image inside a
    /// panel of another aspect ratio means -- the other axis is letterboxed.
    pub(crate) fn fit_scale(&self) -> f32 {
        (self.panel_size[0] / self.image_size[0]).min(self.panel_size[1] / self.image_size[1])
    }

    /// Panel points one source pixel spans at the standing zoom.
    pub(crate) fn scale(&self) -> f32 {
        self.fit_scale() * self.zoom
    }

    /// The displayed extent of the whole image, in panel points.
    pub(crate) fn display_size(&self) -> [f32; 2] {
        let scale = self.scale();
        [self.image_size[0] * scale, self.image_size[1] * scale]
    }

    /// The rectangle of the image the panel shows, `[x0, y0, x1, y1]` in image
    /// pixels.
    ///
    /// **Not clipped to the image.** At fit zoom the letterboxed axis runs past
    /// both edges, and a pan that has pushed the image partly off the panel
    /// runs past one; clipping would hide exactly the thing a caller checking
    /// where it landed wants to see. What is always true is that the centre of
    /// this rectangle is the image pixel at the centre of the panel, which is
    /// what a [`Look`] aims.
    pub(crate) fn visible_rect(&self) -> [f32; 4] {
        let scale = self.scale();
        let display = self.display_size();
        let left = (display[0] / 2.0 - self.panel_size[0] / 2.0 - self.pan[0]) / scale;
        let top = (display[1] / 2.0 - self.panel_size[1] / 2.0 - self.pan[1]) / scale;
        [
            left,
            top,
            left + self.panel_size[0] / scale,
            top + self.panel_size[1] / scale,
        ]
    }

    /// The same view with `pan` and `zoom` replaced, the frame left alone.
    fn with(self, pan: [f32; 2], zoom: f32) -> Self {
        Self { pan, zoom, ..self }
    }
}

/// Where a caller is asking the panel to look.
///
/// One enum rather than a bag of optional fields for the reason
/// [`crate::mcp::ViewCommand`] is one: these are different questions, and a
/// request carrying two of them would have no answer.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum Look {
    /// The row-click reveal: bring `pixel` into view, and only if it is not
    /// already comfortably in it. Never touches the zoom.
    Reveal {
        /// Where, in the image's own pixels.
        pixel: [f32; 2],
    },
    /// Put `pixel` at the centre of the panel, at `zoom` where one is named.
    Pixel {
        /// Where, in the image's own pixels.
        pixel: [f32; 2],
        /// The magnification to look at it at, or `None` to keep the standing
        /// one.
        zoom: Option<f32>,
    },
    /// Fit `[x0, y0, x1, y1]`, in image pixels, to the panel.
    Rect([f32; 4]),
    /// The whole image: zoom 1, centred.
    Fit,
}

/// The view `look` settles on, starting from `view`.
///
/// Pure, and the single definition of what each request means, so the panel's
/// own reveal and a tool call that asks for the same thing land in the same
/// place. The frame -- the image, its size, the panel's size -- comes back
/// unchanged; only `pan` and `zoom` move.
///
/// Every outcome obeys the two limits the hand obeys: the zoom is clamped to
/// `[1, MAX_ZOOM]`, and the pan to the rule that keeps
/// [`PAN_MARGIN`] points of the image on the panel. Centring a pixel of the
/// image asks for a pan of at most half the displayed extent, which the pan
/// limit allows for any panel wider than twice the margin, so the clamp bites
/// only in a very small panel.
pub(crate) fn look_at(view: ViewGeometry, look: &Look) -> ViewGeometry {
    match *look {
        Look::Reveal { pixel } => reveal(view, pixel),
        Look::Pixel { pixel, zoom } => {
            let zoomed = view.with(view.pan, clamp_zoom(zoom.unwrap_or(view.zoom)));
            centred_on(zoomed, pixel)
        }
        Look::Rect(rect) => {
            let [x0, y0, x1, y1] = ordered(rect);
            let (width, height) = (
                (x1 - x0).max(f32::MIN_POSITIVE),
                (y1 - y0).max(f32::MIN_POSITIVE),
            );
            // The zoom that makes the rectangle exactly fill the panel on its
            // tighter axis, measured against the fit: `fit_scale` is the panel
            // points the whole image spans per pixel, so the ratio of the two
            // is the magnification asked for.
            let fit = view.fit_scale();
            let zoom =
                ((view.panel_size[0] / width) / fit).min((view.panel_size[1] / height) / fit);
            let zoomed = view.with(view.pan, clamp_zoom(zoom));
            centred_on(zoomed, [(x0 + x1) / 2.0, (y0 + y1) / 2.0])
        }
        Look::Fit => view.with([0.0, 0.0], 1.0),
    }
}

/// The reveal's own rule: nothing at fit zoom, nothing for a pixel already
/// comfortably on screen, and otherwise the pixel at the panel centre.
///
/// The two refusals are what keep walking down a track's rows from jerking the
/// image about, and what keeps a fitted image from sliding off centre for a
/// feature near its edge. "Comfortably" is the middle
/// `1 - 2 * REVEAL_MARGIN` of the panel per axis: a pixel a few points inside
/// the edge is on screen but not visible in any useful sense, half of its
/// neighbourhood cut off.
fn reveal(view: ViewGeometry, pixel: [f32; 2]) -> ViewGeometry {
    if view.zoom <= 1.0 {
        return view;
    }
    let scale = view.scale();
    let display = view.display_size();
    // Where the pixel sits relative to the panel centre, in panel points.
    let offset = [
        view.pan[0] - display[0] / 2.0 + pixel[0] * scale,
        view.pan[1] - display[1] / 2.0 + pixel[1] * scale,
    ];
    let inside = [
        view.panel_size[0] * (0.5 - REVEAL_MARGIN),
        view.panel_size[1] * (0.5 - REVEAL_MARGIN),
    ];
    if offset[0].abs() <= inside[0] && offset[1].abs() <= inside[1] {
        return view;
    }
    centred_on(view, pixel)
}

/// `view` panned so that `pixel` sits at the centre of the panel, clamped.
fn centred_on(view: ViewGeometry, pixel: [f32; 2]) -> ViewGeometry {
    let scale = view.scale();
    let display = view.display_size();
    let pan = [
        display[0] / 2.0 - pixel[0] * scale,
        display[1] / 2.0 - pixel[1] * scale,
    ];
    view.with(clamp_pan(pan, display, view.panel_size), view.zoom)
}

/// A zoom inside the panel's own range, and never a NaN: an unusable number
/// lands on the fit rather than on a view nothing can be seen in.
fn clamp_zoom(zoom: f32) -> f32 {
    if !zoom.is_finite() {
        return 1.0;
    }
    zoom.clamp(1.0, MAX_ZOOM)
}

/// A pan that keeps at least [`PAN_MARGIN`] points of the image on the panel.
fn clamp_pan(pan: [f32; 2], display: [f32; 2], panel: [f32; 2]) -> [f32; 2] {
    let limit = [
        (display[0] + panel[0]) / 2.0 - PAN_MARGIN,
        (display[1] + panel[1]) / 2.0 - PAN_MARGIN,
    ];
    [
        pan[0].clamp(-limit[0], limit[0]),
        pan[1].clamp(-limit[1], limit[1]),
    ]
}

/// A rectangle with its corners the way round the arithmetic wants them, so a
/// caller that named the far corner first still gets the rectangle it drew.
fn ordered([x0, y0, x1, y1]: [f32; 4]) -> [f32; 4] {
    [x0.min(x1), y0.min(y1), x0.max(x1), y0.max(y1)]
}
