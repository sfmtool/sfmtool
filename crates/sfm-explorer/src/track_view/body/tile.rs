// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The per-observation tile the table draws: what one sighting of the track
//! actually looks like.
//!
//! The tile is the column the numbers beside it are about. A ZNCC of 0.42 is a
//! number; the tile that produced it is the thing a person can judge, and the
//! whole reason the bench exists is that a person judges.
//!
//! Which picture it is follows the stage, because the two stages register
//! different things:
//!
//! - At the **track stage** it is the patch re-rendered from this
//!   observation's own view, re-anchored where the observation sits -- through
//!   [`super::patch::patch_color_image`], the one warp the body draws every
//!   track-stage tile with, whichever mode it is in
//! - At the **cluster stage** there is no surface, so it is the observation's
//!   own grid: the `R x R` samples the refinement kernel reads at that place
//!   and its shape, through the kernel's own sampler
//!   (`sfmtool_core::patch::cluster_refine::sample_member_grid`), which is what
//!   makes the picture the thing the ZNCC beside it was computed over rather
//!   than a second opinion about it.
//!
//! **Where the observation sits is [`crate::bench::observation_site`]'s**, at
//! either stage: the keypoint a reading wrote, else the refined cluster
//! position, else the seed the step that proposed it left. A candidate a
//! descriptor search has just added has only that seed, and the tile is the
//! whole of what says whether the search found the right surface -- so it is
//! cut around the seed rather than left blank, or, at the track stage, cut
//! around wherever the bare projection of the point happens to land.
//!
//! **Hovering a tile shows it in context** ([`context()`]): the same picture
//! over [`CONTEXT_FACTOR`] times the patch's width, at the tile's own sampling,
//! so the tile is the middle of it texel for texel. With it come the places the
//! hover view draws over the picture: the patch's box, the keypoint, and, at
//! the track stage, where the track's point projects, which is the other end of
//! the row's reprojection error.
//!
//! **The *Zoom* column reads the patch's warp without drawing it**
//! ([`patch_jacobian`]): the Jacobian at the centre of the same warp, from the
//! same patch re-anchored on the same place, camera and pose, read by core's
//! `patch_grid_jacobian` with no photograph in it, on the patch grid at the
//! reconstruction's patch resolution `R` ([`crate::bench::patch_resolution`])
//! rather than at the tile's display resolution. So the column and
//! `get_bench_track` give the same numbers whether or not the photograph has
//! been decoded, and whether or not the tile's middle is on it, in the grid
//! px the shift and the self-similarity radius are stated in.
//!
//! The tile and its hover view are rendered at display resolutions
//! ([`super::patch::PATCH_RES`], and [`CONTEXT_FACTOR`] times it), which only
//! decide how many texels are drawn: no number in a cell, a caption or a
//! report is read from them. **Which sampler draws them** is decided at `R`,
//! not at the display resolution ([`tile_sampler`]): the sampler rule applied
//! to the same Jacobian the *Zoom* column reads, so a view is drawn with the
//! sampler the bench's kernels render it with.

use sfmtool_core::bench::{EditableTrack, Stage};
use sfmtool_core::camera::image::ImageU8Pyramid;
use sfmtool_core::camera::sampler::Sampler;
use sfmtool_core::camera::warp_map::patch_grid_jacobian;
use sfmtool_core::camera::CameraIntrinsics;
use sfmtool_core::geometry::RigidTransform;
use sfmtool_core::patch::cloud::OrientedPatch;
use sfmtool_core::patch::cluster_refine::{sample_member_grid, ClusterRefineParams};
use sfmtool_core::SfmrReconstruction;

use super::patch::PatchJacobian;

/// The tile for one observation, uploaded, or `None` when [`image()`] had
/// nothing to render.
pub(super) fn render(
    ctx: &egui::Context,
    recon: &SfmrReconstruction,
    track: &EditableTrack,
    observation: usize,
    src: &ImageU8Pyramid,
    name: String,
) -> Option<egui::TextureHandle> {
    let tile = image(recon, track, observation, src)?;
    Some(ctx.load_texture(name, tile, egui::TextureOptions::NEAREST))
}

/// The Jacobian at the centre of one observation's track-stage tile, in
/// photograph pixels per patch-grid px, without the photograph: core's
/// `patch_grid_jacobian` of the patch re-anchored where the observation sits
/// ([`super::patch::render_frame`]), the frame [`image()`] renders the tile
/// through, on a grid of `recon`'s patch resolution `R`
/// ([`crate::bench::patch_resolution`]). What the *Zoom* cell prints the zoom
/// of and the *Zoom* column orders by, and what `get_bench_track` reports for
/// the row as `patch_jacobian`. The tile's display resolution
/// [`super::patch::PATCH_RES`] does not enter it.
///
/// Geometry alone: a tile whose middle is off the photograph still has one.
/// `None` in each case where there is no warp: a row at the cluster stage,
/// whose tile is the refinement kernel's grid rather than a warp; a
/// track-stage track with no patch yet; an observation with nothing saying
/// where it sits; and a patch whose centre does not project, behind the
/// camera or outside the camera model's domain.
pub(crate) fn patch_jacobian(
    recon: &SfmrReconstruction,
    track: &EditableTrack,
    observation: usize,
) -> Option<PatchJacobian> {
    let (placement, camera, pose, keypoint) = track_tile_geometry(recon, track, observation)?;
    let frame = super::patch::render_frame(placement, camera, &pose, Some(keypoint));
    let resolution = crate::bench::patch_resolution(recon) as usize;
    patch_grid_jacobian(&frame, camera, &pose, resolution).map(PatchJacobian)
}

/// The sampler one observation's track-stage tile and its hover view are
/// rendered with: the sampler rule applied to [`patch_jacobian`], the choice
/// the bench's kernels make for the same view, or
/// [`super::patch::FALLBACK_SAMPLER`] where there is no Jacobian.
pub(crate) fn tile_sampler(
    recon: &SfmrReconstruction,
    track: &EditableTrack,
    observation: usize,
) -> Sampler {
    patch_jacobian(recon, track, observation).map_or(super::patch::FALLBACK_SAMPLER, |jacobian| {
        jacobian.sampler()
    })
}

/// What a track-stage row's tile is warped through: the track's patch, the
/// observation's camera and pose, and where the observation sits. `None` at
/// the cluster stage and wherever one of them is missing.
fn track_tile_geometry<'a>(
    recon: &'a SfmrReconstruction,
    track: &'a EditableTrack,
    observation: usize,
) -> Option<(
    &'a OrientedPatch,
    &'a CameraIntrinsics,
    RigidTransform,
    [f64; 2],
)> {
    let Stage::Track(payload) = &track.stage else {
        return None;
    };
    let row = track.observations.get(observation)?;
    let site = crate::bench::observation_site(row)?;
    let placement = payload.placement.as_ref()?;
    let image = recon.image_table.images.get(row.image as usize)?;
    let camera = recon.image_table.cameras.get(image.camera_index as usize)?;
    Some((
        placement,
        camera,
        crate::scene::cam_from_world(image),
        site.pixel,
    ))
}

/// The picture one row draws, or `None` when there is nothing to render: no
/// patch yet at the track stage, nothing saying where the observation sits, or
/// a degenerate shape at the cluster stage.
///
/// Pure, so a headless test can ask what a row shows rather than only whether
/// it showed something. `src` is the observation's own photograph as the viewer's
/// photograph cache holds it: the pyramid built at the decode, whose level
/// 0 is the photograph and whose lower levels are what the cluster sampler
/// mip-selects over.
pub(super) fn image(
    recon: &SfmrReconstruction,
    track: &EditableTrack,
    observation: usize,
    src: &ImageU8Pyramid,
) -> Option<egui::ColorImage> {
    let row = track.observations.get(observation)?;
    let site = crate::bench::observation_site(row)?;
    match &track.stage {
        Stage::Track(_) => {
            let (frame, camera, pose, keypoint) = track_tile_geometry(recon, track, observation)?;
            Some(super::patch::patch_color_image(
                frame,
                camera,
                &pose,
                Some(keypoint),
                src,
                tile_sampler(recon, track, observation),
            ))
        }
        Stage::Cluster(payload) => {
            // The cluster's own radius, always: it is what every one of its
            // shapes is written against, so the tile is the square the person
            // asked for before an evaluation and the square the ZNCC was
            // measured over after one. The grid is the template's when one has
            // been cut, and the kernel's default -- what the next evaluation
            // will use -- before that.
            let mut params = ClusterRefineParams {
                radius: payload.radius,
                ..ClusterRefineParams::default()
            };
            if let Some(template) = payload.template.as_ref() {
                params.resolution = template.samples.shape()[0] as u32;
            }
            let grid = sample_member_grid(src, site.pixel, site.shape?, &params)?;
            let resolution = params.resolution.max(2) as usize;
            let channels = grid.len() / (resolution * resolution);
            Some(color_image(&grid, resolution, channels))
        }
    }
}

/// How many times the patch's width the hover view of a tile shows. For
/// display only: the hover view is rendered at this many times
/// [`super::patch::PATCH_RES`], and no number is read from it.
///
/// Odd, so that the patch sits in the middle with whole patch widths of the
/// photograph on every side and its box falls on texel boundaries whatever the
/// tile's resolution: at `3` the patch is the middle third of the picture, with
/// one patch width of the photograph on every side.
pub(super) const CONTEXT_FACTOR: u32 = 3;

/// A tile in context: the picture the hover view draws, and the places it
/// marks on it, each in the picture's own texels, `(0, 0)` its top-left corner
/// and `(side, side)` its bottom-right.
#[derive(Debug, Clone)]
pub(super) struct TileContext {
    /// The picture: the tile's own frame widened [`CONTEXT_FACTOR`] times, at
    /// the tile's sampling.
    pub(super) image: egui::ColorImage,
    /// The texels the row's tile covers, which is the patch.
    pub(super) patch_box: egui::Rect,
    /// Where the observation sits: the patch's centre, which is the
    /// picture's. `None` only at the track stage, where the keypoint's ray
    /// does not meet the patch's plane and the tile falls back to the patch as
    /// it stands.
    pub(super) keypoint: Option<egui::Pos2>,
    /// Where the track's point projects, read onto the patch's plane as the
    /// picture shows it. It can lie outside the picture. `None` at the cluster
    /// stage, which has no point, and where the point does not project or its
    /// ray misses the plane.
    pub(super) projection: Option<egui::Pos2>,
    /// How far, in the photograph's own pixels, the observation sits from that
    /// projection: the row's reprojection error, or before the track is
    /// triangulated its distance to the patch centre's projection. `None`
    /// where there is nothing to project.
    pub(super) projection_px: Option<f64>,
    /// What was projected, or `None` at the cluster stage, which projects
    /// nothing.
    pub(super) projection_of: Option<ProjectionOf>,
}

/// The hover view of one observation's tile, or `None` wherever [`image()`]
/// would render no tile.
///
/// At the track stage the frame the tile is rendered through, re-anchored
/// where the observation sits, is widened by [`CONTEXT_FACTOR`] about its
/// centre and rendered at that many times the tile's resolution. At the cluster
/// stage the member grid is sampled over that many times the cluster's radius
/// at that many times its resolution. Either way a texel of the picture is a
/// texel of the tile, and the tile is the middle of the picture.
///
/// What is projected is what the row's *Proj. err* column measures to: the
/// track's triangulated point, or before there is one the patch's own centre.
/// The projected pixel is read onto the widened frame's plane through
/// `OrientedPatch::keypoint_plane_offset`, the reading the tile's own anchoring
/// uses, so the marker sits on the part of the photograph the picture shows at
/// that pixel.
pub(super) fn context(
    recon: &SfmrReconstruction,
    track: &EditableTrack,
    observation: usize,
    src: &ImageU8Pyramid,
) -> Option<TileContext> {
    let row = track.observations.get(observation)?;
    let site = crate::bench::observation_site(row)?;
    let img_idx = row.image as usize;
    let k = CONTEXT_FACTOR;
    match &track.stage {
        Stage::Track(payload) => {
            let frame = payload.placement.as_ref()?;
            let image = recon.image_table.images.get(img_idx)?;
            let camera = recon.image_table.cameras.get(image.camera_index as usize)?;
            let pose = crate::scene::cam_from_world(image);
            let anchored = frame.anchored_at_keypoint(camera, &pose, site.pixel);
            let mut wide = anchored.clone().unwrap_or_else(|| frame.clone());
            wide.half_extent = wide.half_extent.map(|h| h * f64::from(k));
            let side = super::patch::PATCH_RES * k;
            let picture = super::patch::frame_color_image(
                &wide,
                camera,
                &pose,
                src,
                side,
                tile_sampler(recon, track, observation),
            );
            let side = side as f32;
            // The pixel of the photograph at `pixel`, as a place in the
            // picture: its ray met with the widened frame's plane, read on the
            // frame's axes, `s` along `u` to the right and `t` along `v`
            // upward, each `-1 ..= 1` across the picture.
            let to_picture = |pixel: [f64; 2]| {
                let offset = wide.keypoint_plane_offset(camera, &pose, pixel)?;
                let s = offset.dot(&wide.u_axis) / wide.half_extent[0];
                let t = offset.dot(&wide.v_axis) / wide.half_extent[1];
                let at = egui::pos2(
                    ((s + 1.0) * 0.5) as f32 * side,
                    ((1.0 - t) * 0.5) as f32 * side,
                );
                (at.x.is_finite() && at.y.is_finite()).then_some(at)
            };
            let keypoint = match anchored {
                Some(_) => Some(egui::pos2(side / 2.0, side / 2.0)),
                None => to_picture(site.pixel),
            };
            let target = payload.position.unwrap_or(frame.center);
            let projected = camera.project_homogeneous(&pose, target.coords, frame.w);
            Some(TileContext {
                image: picture,
                patch_box: middle_box(side),
                keypoint,
                projection: projected.and_then(to_picture),
                projection_px: projected
                    .map(|p| (p[0] - site.pixel[0]).hypot(p[1] - site.pixel[1]))
                    .filter(|d| d.is_finite()),
                projection_of: Some(match payload.position {
                    Some(_) => ProjectionOf::Point,
                    None => ProjectionOf::PatchCentre,
                }),
            })
        }
        Stage::Cluster(payload) => {
            // The tile's own parameters with the radius and the resolution
            // both widened, which keeps the step between samples, and the mip
            // level it selects, the tile's.
            let mut params = ClusterRefineParams {
                radius: payload.radius,
                ..ClusterRefineParams::default()
            };
            if let Some(template) = payload.template.as_ref() {
                params.resolution = template.samples.shape()[0] as u32;
            }
            params.resolution = params.resolution.max(2) * k;
            params.radius *= f64::from(k);
            let grid = sample_member_grid(src, site.pixel, site.shape?, &params)?;
            let resolution = params.resolution as usize;
            let channels = grid.len() / (resolution * resolution);
            let side = resolution as f32;
            Some(TileContext {
                image: color_image(&grid, resolution, channels),
                patch_box: middle_box(side),
                keypoint: Some(egui::pos2(side / 2.0, side / 2.0)),
                projection: None,
                projection_px: None,
                projection_of: None,
            })
        }
    }
}

/// The middle [`CONTEXT_FACTOR`]th of a picture `side` texels across: where the
/// patch is in its hover view.
fn middle_box(side: f32) -> egui::Rect {
    let width = side / CONTEXT_FACTOR as f32;
    egui::Rect::from_center_size(egui::pos2(side / 2.0, side / 2.0), egui::vec2(width, width))
}

/// The sampler's interleaved `R x R x C` samples as an RGBA image.
///
/// The samples are the source's own 0..255 range in `f32`, so they are rounded
/// and clamped rather than rescaled: a tile is a picture of the photograph, and
/// stretching its levels would make two tiles of one surface look different.
/// One channel is repeated across RGB, which is what a grey photograph is.
pub(super) fn color_image(grid: &[f32], resolution: usize, channels: usize) -> egui::ColorImage {
    let mut rgba = Vec::with_capacity(resolution * resolution * 4);
    for texel in grid.chunks_exact(channels.max(1)) {
        let level = |c: usize| texel.get(c).copied().unwrap_or(texel[0]).clamp(0.0, 255.0) as u8;
        match channels {
            0 => rgba.extend_from_slice(&[0, 0, 0, 255]),
            1 => {
                let grey = level(0);
                rgba.extend_from_slice(&[grey, grey, grey, 255]);
            }
            _ => rgba.extend_from_slice(&[level(0), level(1), level(2), 255]),
        }
    }
    egui::ColorImage::from_rgba_unmultiplied([resolution, resolution], &rgba)
}

/// Side of the hover view's picture on screen, in points: large enough to
/// judge a patch width of photograph on each side of the patch, small enough
/// for a tooltip beside a row.
pub(super) const CONTEXT_HOVER_SIDE: f32 = 288.0;

/// What the projection in a [`TileContext`] is the projection of, which is
/// what the row's *Proj. err* column measures to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum ProjectionOf {
    /// The track's triangulated point.
    Point,
    /// The patch's centre, before the track is triangulated.
    PatchCentre,
}

/// A [`TileContext`] uploaded: what the table keeps per row once the hover
/// view has been asked for, and draws from on every frame it is shown.
pub(super) struct DrawnContext {
    texture: egui::TextureHandle,
    /// The picture's side in texels, which the places below are in.
    side: f32,
    patch_box: egui::Rect,
    keypoint: Option<egui::Pos2>,
    projection: Option<egui::Pos2>,
    caption: String,
}

impl DrawnContext {
    /// Upload `context` under `name`.
    ///
    /// Filtered linearly rather than nearest, unlike the tile: the picture is
    /// drawn at a scale that is not a whole number of screen points per texel,
    /// where nearest filtering would draw some texels wider than others.
    pub(super) fn new(ctx: &egui::Context, context: TileContext, name: String) -> Self {
        let caption = context_caption(&context);
        let side = context.image.size[0] as f32;
        Self {
            texture: ctx.load_texture(name, context.image, egui::TextureOptions::LINEAR),
            side,
            patch_box: context.patch_box,
            keypoint: context.keypoint,
            projection: context.projection,
            caption,
        }
    }

    /// Draw the hover view: the picture at [`CONTEXT_HOVER_SIDE`], the patch's
    /// box, the keypoint, the projection and the dashed line between them,
    /// then the caption.
    ///
    /// Each mark is a light stroke over a wider dark one, so it reads over a
    /// bright photograph and a dark one alike. The marks are clipped to the
    /// picture, so a projection outside it shows as the line running off the
    /// edge towards it.
    pub(super) fn show(&self, ui: &mut egui::Ui) {
        let (rect, _) = ui.allocate_exact_size(
            egui::vec2(CONTEXT_HOVER_SIDE, CONTEXT_HOVER_SIDE),
            egui::Sense::hover(),
        );
        let painter = ui.painter_at(rect);
        painter.image(
            self.texture.id(),
            rect,
            egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)),
            egui::Color32::WHITE,
        );
        let scale = CONTEXT_HOVER_SIDE / self.side.max(1.0);
        let to_screen = |at: egui::Pos2| rect.min + at.to_vec2() * scale;
        let halo = egui::Stroke::new(3.0, egui::Color32::from_black_alpha(170));
        let light = egui::Stroke::new(1.5, egui::Color32::from_gray(240));

        let patch =
            egui::Rect::from_min_max(to_screen(self.patch_box.min), to_screen(self.patch_box.max));
        for stroke in [halo, light] {
            painter.rect_stroke(patch, 0.0, stroke, egui::StrokeKind::Middle);
        }
        paint_marks(
            &painter,
            self.keypoint.map(to_screen),
            self.projection.map(to_screen),
        );
        ui.set_max_width(CONTEXT_HOVER_SIDE);
        ui.label(&self.caption);
    }
}

/// Draw the marks a hover view puts over its picture, in screen points: a dot
/// at `keypoint`, where the observation sits, a ring at `projection`, where
/// the track's point or its patch's centre projects, and a dashed line from
/// the one to the other.
///
/// Each mark is a light stroke over a wider dark one, so it reads over a
/// bright photograph and a dark one alike, the ring in amber. The tile's hover
/// view and the crop's both draw through this, so the two mark one place the
/// same way.
pub(super) fn paint_marks(
    painter: &egui::Painter,
    keypoint: Option<egui::Pos2>,
    projection: Option<egui::Pos2>,
) {
    let halo = egui::Stroke::new(3.0, egui::Color32::from_black_alpha(170));
    let light = egui::Stroke::new(1.5, egui::Color32::from_gray(240));
    let projection_color = egui::Color32::from_rgb(255, 196, 64);
    if let (Some(keypoint), Some(projection)) = (keypoint, projection) {
        let line = [keypoint, projection];
        for stroke in [halo, light] {
            painter.extend(egui::Shape::dashed_line(&line, stroke, 5.0, 4.0));
        }
    }
    if let Some(at) = projection {
        painter.circle_stroke(at, 5.0, egui::Stroke::new(4.0, halo.color));
        painter.circle_stroke(at, 5.0, egui::Stroke::new(2.0, projection_color));
    }
    if let Some(at) = keypoint {
        painter.circle(at, 3.0, light.color, egui::Stroke::new(1.5, halo.color));
    }
}

/// The bullets under a tile's hover view: what the picture is, and what each
/// mark on it is.
pub(super) fn context_caption(context: &TileContext) -> String {
    let side = context.image.size[0] as f32;
    let view = egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(side, side));
    format!(
        "\u{2022} Box: the patch, in {CONTEXT_FACTOR}x its width of the photograph\n\
         \u{2022} Dot: where the observation sits\n\
         {}",
        projection_bullet(
            context.projection_of,
            context.projection.map(|at| !view.contains(at)),
            context.projection_px,
        )
    )
}

/// The bullet a hover view's caption gives its ring: what was projected, and
/// how far from the observation it landed.
///
/// `outside` is whether the ring lies outside the picture, `None` where there
/// is no ring to draw; `projection_px` the distance in the photograph's
/// pixels, `None` where nothing projected. Shared by the tile's hover view and
/// the crop's, so the two say one thing about one mark.
pub(super) fn projection_bullet(
    projection_of: Option<ProjectionOf>,
    outside: Option<bool>,
    projection_px: Option<f64>,
) -> String {
    let what = match projection_of {
        None => return "\u{2022} No ring: a cluster has no point to project".to_string(),
        Some(ProjectionOf::Point) => "the track's point",
        Some(ProjectionOf::PatchCentre) => "the patch's centre",
    };
    match (outside, projection_px) {
        (Some(outside), Some(px)) => format!(
            "\u{2022} Ring: where {what} projects, {px:.2} px away along the dashed line{}",
            if outside { ", outside this view" } else { "" }
        ),
        (None, Some(px)) => format!(
            "\u{2022} No ring: {what} projects {px:.2} px away, off the patch's plane as this \
             view sees it",
        ),
        _ => format!("\u{2022} No ring: {what} does not project into this photograph"),
    }
}
