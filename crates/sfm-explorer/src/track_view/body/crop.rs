// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The crop each row draws beside its tile: the photograph itself, cut around
//! the patch's outline as that outline lands in it.
//!
//! The tile is the patch warped square, so it hides how the lens and the view
//! bend the patch; the crop shows the bending. It is the Image Detail bench
//! layer's outline and the pixels under it, without the layer's dot, its
//! normal or the SIFT overlays: at the track stage the patch re-anchored where
//! the observation sits ([`crate::bench::geometry::anchored_frame`]), its
//! boundary projected through the camera's own model
//! ([`crate::bench::geometry::project_outline`]), and at the cluster stage the
//! observation's own parallelogram ([`crate::bench::geometry::parallelogram`]).
//! The crop is the bounding box of that outline widened by [`CROP_MARGIN_PX`]
//! on every side and rounded out to whole pixels, so the whole outline is in
//! it and a fisheye's curved edges show as curves, then widened on its shorter
//! side until it is square, with the outline in the middle ([`region()`]).
//!
//! Hovering the crop shows it in context ([`context()`]): the same crop
//! centred in [`CONTEXT_FACTOR`] times its width and height of the
//! photograph, at the crop's own sampling, so the crop is its middle third
//! texel for texel, as a tile is the middle third of its hover view. Over it
//! go the tile's hover view's marks ([`super::tile::paint_marks`]): a dot where
//! the observation sits and, at the track stage, a ring where the track's
//! point projects, joined by a dashed line. Under it a caption says what the
//! marks are and gives the patch's two axes in the photograph's pixels.

use egui::{Color32, ColorImage, Pos2, Rect, Stroke};
use sfmtool_core::bench::{EditableTrack, Stage, Verdict};
use sfmtool_core::camera::remap::{sample_bilinear_u8, ImageU8Pyramid};
use sfmtool_core::SfmrReconstruction;

use super::tile::{ProjectionOf, CONTEXT_FACTOR, CONTEXT_HOVER_SIDE};
use crate::bench::geometry;

/// How far past the outline's bounding box the crop reaches on every side, in
/// the photograph's pixels: enough that the stroke of the outline never sits
/// on the crop's own edge.
pub(super) const CROP_MARGIN_PX: f64 = 1.0;

/// The most texels the longer side of a row's crop is sampled at. A crop wider
/// than this many photograph pixels is read from the pyramid level whose
/// pixels are nearest the step, rather than uploaded whole, so a patch that
/// spans half a photograph costs a small texture like any other. For display
/// only: the crop's caption reads the patch's axes from the geometry in the
/// photograph's own pixels, not from these texels.
const CROP_MAX_TEXELS: f64 = 128.0;

/// Where a patch lands in one photograph: its outline, how long its two axes
/// are there, where the observation sits and where the track's point
/// projects.
#[derive(Debug, Clone)]
pub(super) struct Outline {
    /// The boundary in photograph pixels, in the order it is walked, `None`
    /// for a sample that did not project. Closed: the last sample joins the
    /// first.
    pub(super) samples: Vec<Option<[f64; 2]>>,
    /// The lengths of the patch's two axes in photograph pixels, the axis
    /// running across the row's tile first and the one running up and down it
    /// second, each measured along the axis as it lands, through the lens.
    /// `None` for an axis part of which does not project.
    pub(super) axes_px: [Option<f64>; 2],
    /// Where the observation sits, in photograph pixels
    /// ([`crate::bench::observation_site`]).
    pub(super) keypoint: [f64; 2],
    /// Where the track's point projects into the photograph, or before the
    /// track is triangulated its patch's centre: the other end of the distance
    /// the row's *Proj. err* cell reports. `None` at the cluster stage, which
    /// has no point, and where the point does not project.
    pub(super) projection: Option<[f64; 2]>,
    /// What was projected, or `None` at the cluster stage.
    pub(super) projection_of: Option<ProjectionOf>,
}

/// The part of a photograph a crop covers, in whole pixels.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct Region {
    /// The top-left pixel.
    pub(super) min: [i64; 2],
    /// Width and height, in pixels, each at least one.
    pub(super) size: [i64; 2],
}

/// A crop or its hover view, before it is uploaded.
#[derive(Debug, Clone)]
pub(super) struct CropPicture {
    /// The pixels. Texels outside the photograph are transparent.
    pub(super) image: ColorImage,
    /// The part of the photograph the row's crop covers, which is the whole
    /// picture for the crop itself and its middle third for the hover view.
    pub(super) crop: Region,
    /// The outline, in the picture's texels, `(0, 0)` its top-left corner.
    pub(super) outline: Vec<Option<Pos2>>,
    /// [`Outline::axes_px`].
    pub(super) axes_px: [Option<f64>; 2],
    /// [`Outline::keypoint`], in the picture's texels.
    pub(super) keypoint: Pos2,
    /// [`Outline::projection`], in the picture's texels. It can lie outside
    /// the picture.
    pub(super) projection: Option<Pos2>,
    /// How far the observation sits from the projection, in the photograph's
    /// pixels, or `None` where nothing projected.
    pub(super) projection_px: Option<f64>,
    /// [`Outline::projection_of`].
    pub(super) projection_of: Option<ProjectionOf>,
}

/// Where the patch of `track` lands in the photograph of `observation`, or
/// `None` when there is nothing to outline: no patch yet at the track stage,
/// nothing saying where the observation sits, or no shape at the cluster
/// stage.
pub(super) fn outline(
    recon: &SfmrReconstruction,
    track: &EditableTrack,
    observation: usize,
) -> Option<Outline> {
    let row = track.observations.get(observation)?;
    let keypoint = crate::bench::observation_site(row)?.pixel;
    match &track.stage {
        Stage::Track(payload) => {
            let patch = payload.placement.as_ref()?;
            let (camera, pose) = geometry::view_of(&recon.image_table, row.image as usize)?;
            let frame = geometry::anchored_frame(patch, &camera, &pose, row);
            let (samples, per_edge) = geometry::project_outline(&frame, &camera, &pose);
            // Each axis sampled as densely as the edges beside it, so a
            // fisheye's bent axis is measured along its bend.
            let axis = |along: fn(f64) -> (f64, f64)| {
                let n = 2 * per_edge;
                let mut points = Vec::with_capacity(n + 1);
                for i in 0..=n {
                    let (s, t) = along(2.0 * i as f64 / n as f64 - 1.0);
                    let (xyz, w) = frame.corner_homogeneous(s, t);
                    points.push(camera.project_homogeneous(&pose, xyz, w)?);
                }
                Some(polyline_length(&points))
            };
            // What the row's *Proj. err* cell measures to, as the tile's hover
            // view projects it.
            let target = payload.position.unwrap_or(patch.center);
            Some(Outline {
                samples,
                axes_px: [axis(|a| (a, 0.0)), axis(|a| (0.0, a))],
                keypoint,
                projection: camera.project_homogeneous(&pose, target.coords, patch.w),
                projection_of: Some(match payload.position {
                    Some(_) => ProjectionOf::Point,
                    None => ProjectionOf::PatchCentre,
                }),
            })
        }
        Stage::Cluster(payload) => {
            let site = crate::bench::observation_site(row)?;
            let shape = site.shape?;
            let corners = geometry::parallelogram(site.pixel, shape, payload.radius);
            let axis = |c: usize| 2.0 * payload.radius * shape[0][c].hypot(shape[1][c]);
            Some(Outline {
                samples: corners.into_iter().map(Some).collect(),
                axes_px: [Some(axis(0)), Some(axis(1))],
                keypoint,
                projection: None,
                projection_of: None,
            })
        }
    }
}

/// The square of whole pixels a row's crop covers, or `None` when no sample of
/// `outline` landed.
///
/// The box the outline lies in, widened by [`CROP_MARGIN_PX`] and rounded out
/// to whole pixels, then widened on its shorter side, evenly on either side,
/// until it is square. The photograph is not stretched: the extra pixels are
/// more of the photograph around the outline, which stays in the middle, and
/// the crop fills the square cell it is drawn in.
pub(super) fn region(outline: &Outline) -> Option<Region> {
    let tight = tight_region(outline)?;
    let side = tight.size[0].max(tight.size[1]);
    Some(Region {
        min: [0, 1].map(|a| tight.min[a] - (side - tight.size[a]) / 2),
        size: [side, side],
    })
}

/// The whole pixels `outline` lies in, widened by [`CROP_MARGIN_PX`].
fn tight_region(outline: &Outline) -> Option<Region> {
    let mut landed = outline.samples.iter().flatten();
    let first = landed.next()?;
    let (mut lo, mut hi) = (*first, *first);
    for p in landed {
        lo = [lo[0].min(p[0]), lo[1].min(p[1])];
        hi = [hi[0].max(p[0]), hi[1].max(p[1])];
    }
    let min = [
        (lo[0] - CROP_MARGIN_PX).floor(),
        (lo[1] - CROP_MARGIN_PX).floor(),
    ];
    let max = [
        (hi[0] + CROP_MARGIN_PX).ceil(),
        (hi[1] + CROP_MARGIN_PX).ceil(),
    ];
    if !(min.iter().chain(&max)).all(|v| v.is_finite()) {
        return None;
    }
    Some(Region {
        min: [min[0] as i64, min[1] as i64],
        size: [
            ((max[0] - min[0]) as i64).max(1),
            ((max[1] - min[1]) as i64).max(1),
        ],
    })
}

/// How many texels a crop of `region` is sampled at on each side: one per
/// photograph pixel, unless the longer side is over [`CROP_MAX_TEXELS`], when
/// both sides are scaled down together.
fn texels(region: &Region) -> [usize; 2] {
    let longest = region.size[0].max(region.size[1]) as f64;
    let step = (longest / CROP_MAX_TEXELS).max(1.0);
    region
        .size
        .map(|side| ((side as f64 / step).round() as usize).max(1))
}

/// The crop one row draws, or `None` when [`outline()`] has nothing to
/// outline or no sample of it landed.
pub(super) fn image(
    recon: &SfmrReconstruction,
    track: &EditableTrack,
    observation: usize,
    src: &ImageU8Pyramid,
) -> Option<CropPicture> {
    let outline = outline(recon, track, observation)?;
    let region = region(&outline)?;
    let texels = texels(&region);
    Some(picture(src, &outline, region, texels))
}

/// A crop in context: the same crop centred in [`CONTEXT_FACTOR`] times its
/// width and height of the photograph, sampled at the crop's own step so that
/// the crop is the middle third of it texel for texel.
pub(super) fn context(
    recon: &SfmrReconstruction,
    track: &EditableTrack,
    observation: usize,
    src: &ImageU8Pyramid,
) -> Option<CropPicture> {
    let outline = outline(recon, track, observation)?;
    let crop = region(&outline)?;
    let texels = texels(&crop);
    let k = CONTEXT_FACTOR as usize;
    // Whole crops of photograph on each side of the crop: one at a factor of 3.
    let each_side = k / 2;
    let wide = Region {
        min: [
            crop.min[0] - crop.size[0] * each_side as i64,
            crop.min[1] - crop.size[1] * each_side as i64,
        ],
        size: crop.size.map(|side| side * k as i64),
    };
    let mut picture = picture(src, &outline, wide, texels.map(|t| t * k));
    picture.crop = crop;
    Some(picture)
}

/// `region` of the photograph sampled at `texels`, with `outline` carried
/// into the picture's texels. The crop is the whole picture; a caller that
/// wants it elsewhere moves it.
///
/// At one texel per photograph pixel each texel is that pixel's own value.
/// Coarser, each texel is read bilinearly from the pyramid level whose pixel
/// is the largest not wider than the step, so the picture averages what it
/// skips instead of aliasing it. Texels whose centre falls off the photograph
/// are transparent.
fn picture(
    src: &ImageU8Pyramid,
    outline: &Outline,
    region: Region,
    texels: [usize; 2],
) -> CropPicture {
    let step = [
        region.size[0] as f64 / texels[0] as f64,
        region.size[1] as f64 / texels[1] as f64,
    ];
    let level = (step[0].min(step[1]).log2().floor().max(0.0) as usize)
        .min(src.num_levels().saturating_sub(1));
    let scale = (1u64 << level) as f64;
    let img = src.level(level);
    let full = src.level(0);
    let (width, height) = (f64::from(full.width()), f64::from(full.height()));
    let channels = img.channels().max(1);
    let mut rgba = Vec::with_capacity(texels[0] * texels[1] * 4);
    for j in 0..texels[1] {
        let y = region.min[1] as f64 + (j as f64 + 0.5) * step[1];
        for i in 0..texels[0] {
            let x = region.min[0] as f64 + (i as f64 + 0.5) * step[0];
            if x < 0.0 || y < 0.0 || x >= width || y >= height {
                rgba.extend_from_slice(&[0, 0, 0, 0]);
                continue;
            }
            let (lx, ly) = ((x / scale) as f32, (y / scale) as f32);
            let value = |c: u32| {
                (sample_bilinear_u8(img, lx, ly, c.min(channels - 1)) + 0.5).clamp(0.0, 255.0) as u8
            };
            rgba.extend_from_slice(&[value(0), value(1), value(2), 255]);
        }
    }
    let to_texel = |p: [f64; 2]| {
        Pos2::new(
            ((p[0] - region.min[0] as f64) / step[0]) as f32,
            ((p[1] - region.min[1] as f64) / step[1]) as f32,
        )
    };
    let keypoint = outline.keypoint;
    CropPicture {
        image: ColorImage::from_rgba_unmultiplied(texels, &rgba),
        crop: region,
        outline: outline.samples.iter().map(|p| p.map(to_texel)).collect(),
        axes_px: outline.axes_px,
        keypoint: to_texel(keypoint),
        projection: outline.projection.map(to_texel),
        projection_px: outline
            .projection
            .map(|p| (p[0] - keypoint[0]).hypot(p[1] - keypoint[1]))
            .filter(|d| d.is_finite()),
        projection_of: outline.projection_of,
    }
}

/// The summed length of the segments joining `points`.
fn polyline_length(points: &[[f64; 2]]) -> f64 {
    points
        .windows(2)
        .map(|w| (w[1][0] - w[0][0]).hypot(w[1][1] - w[0][1]))
        .sum()
}

/// A [`CropPicture`] uploaded, with the colour its outline is stroked in.
pub(super) struct DrawnCrop {
    texture: egui::TextureHandle,
    /// The picture's size in texels, which the places below are in.
    size: egui::Vec2,
    outline: Vec<Option<Pos2>>,
    keypoint: Pos2,
    projection: Option<Pos2>,
    color: Color32,
    caption: String,
}

impl DrawnCrop {
    /// Upload `picture` under `name`, its outline to be stroked in the colour
    /// of `verdict`, as Image Detail strokes it.
    ///
    /// `options` is the filtering: a row's crop magnifies each photograph pixel
    /// to a block, as the tile does, so the person sees the photograph's own
    /// resolution; the hover view is drawn at a scale that is not a whole
    /// number of screen points per texel and is filtered linearly.
    pub(super) fn new(
        ctx: &egui::Context,
        picture: CropPicture,
        verdict: Verdict,
        name: String,
        options: egui::TextureOptions,
    ) -> Self {
        let caption = context_caption(&picture);
        let size = egui::vec2(picture.image.size[0] as f32, picture.image.size[1] as f32);
        Self {
            texture: ctx.load_texture(name, picture.image, options),
            size,
            outline: picture.outline,
            keypoint: picture.keypoint,
            projection: picture.projection,
            color: crate::bench::verdict_color(verdict),
            caption,
        }
    }

    /// The largest rect of the picture's shape that fits in `bounds`, centred
    /// in it.
    fn fitted(&self, bounds: Rect) -> Rect {
        let scale =
            (bounds.width() / self.size.x.max(1.0)).min(bounds.height() / self.size.y.max(1.0));
        Rect::from_center_size(bounds.center(), self.size * scale)
    }

    /// Draw the row's crop inside `cell`: the picture fitted to it and the
    /// outline over it in a thin stroke.
    pub(super) fn paint_cell(&self, painter: &egui::Painter, cell: Rect) {
        let rect = self.fitted(cell);
        self.paint_picture(painter, rect);
        self.paint_outline(painter, rect, &[Stroke::new(1.0, self.color)]);
    }

    /// Draw the hover view: the picture fitted to a [`CONTEXT_HOVER_SIDE`]
    /// square, the outline, the tile's hover view's marks, then the caption.
    ///
    /// The outline is a stroke over a wider dark one, as the marks are, so it
    /// reads over a bright photograph and a dark one alike. The marks are
    /// clipped to the picture, so a projection outside it shows as the line
    /// running off the edge towards it.
    pub(super) fn show(&self, ui: &mut egui::Ui) {
        let bounds = Rect::from_min_size(
            Pos2::ZERO,
            egui::vec2(CONTEXT_HOVER_SIDE, CONTEXT_HOVER_SIDE),
        );
        let (rect, _) = ui.allocate_exact_size(self.fitted(bounds).size(), egui::Sense::hover());
        let painter = ui.painter_at(rect);
        self.paint_picture(&painter, rect);
        let halo = Stroke::new(3.0, Color32::from_black_alpha(170));
        self.paint_outline(&painter, rect, &[halo, Stroke::new(1.5, self.color)]);
        let to_screen = self.to_screen(rect);
        super::tile::paint_marks(
            &painter,
            Some(to_screen(self.keypoint)),
            self.projection.map(to_screen),
        );
        ui.set_max_width(CONTEXT_HOVER_SIDE);
        ui.label(&self.caption);
    }

    /// The map from the picture's texels to the screen, for the picture drawn
    /// over `rect`.
    fn to_screen(&self, rect: Rect) -> impl Fn(Pos2) -> Pos2 {
        let scale = egui::vec2(rect.width() / self.size.x, rect.height() / self.size.y);
        move |at: Pos2| rect.min + at.to_vec2() * scale
    }

    fn paint_picture(&self, painter: &egui::Painter, rect: Rect) {
        painter.image(
            self.texture.id(),
            rect,
            Rect::from_min_max(Pos2::ZERO, Pos2::new(1.0, 1.0)),
            Color32::WHITE,
        );
    }

    /// Stroke the outline over the picture drawn at `rect`, once per stroke in
    /// `strokes`, in order. A sample that did not project breaks the curve
    /// rather than being bridged, as in Image Detail.
    fn paint_outline(&self, painter: &egui::Painter, rect: Rect, strokes: &[Stroke]) {
        let to_screen = self.to_screen(rect);
        let n = self.outline.len();
        let segments: Vec<[Pos2; 2]> = (0..n)
            .filter_map(|i| {
                let a = self.outline[i]?;
                let b = self.outline[(i + 1) % n]?;
                Some([to_screen(a), to_screen(b)])
            })
            .collect();
        let painter = painter.with_clip_rect(rect);
        for &stroke in strokes {
            for segment in &segments {
                painter.line_segment(*segment, stroke);
            }
        }
    }
}

/// The bullets under a crop's hover view: what the picture is, what its marks
/// are, and the patch's two axes in the photograph's pixels.
pub(super) fn context_caption(picture: &CropPicture) -> String {
    let [w, h] = picture.crop.size;
    let axis = |axis: Option<f64>| match axis {
        Some(px) => format!("{px:.1} px"),
        None => "not all in this photograph".to_string(),
    };
    let view = Rect::from_min_size(
        Pos2::ZERO,
        egui::vec2(picture.image.size[0] as f32, picture.image.size[1] as f32),
    );
    format!(
        "\u{2022} Outline: the patch as it lands in the photograph\n\
         \u{2022} View: {CONTEXT_FACTOR}x the crop, which is {w} x {h} px of the photograph\n\
         \u{2022} Dot: where the observation sits\n\
         {}\n\
         \u{2022} Axes: {} across the tile, {} up it",
        super::tile::projection_bullet(
            picture.projection_of,
            picture.projection.map(|at| !view.contains(at)),
            picture.projection_px,
        ),
        axis(picture.axes_px[0]),
        axis(picture.axes_px[1]),
    )
}
