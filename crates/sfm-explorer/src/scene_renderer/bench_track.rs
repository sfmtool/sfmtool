// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The bench track pass: the upload, the uniform write and the draw of the
//! overlay ribbon store's figures.
//!
//! The figure itself is [`crate::viewer_3d::bench_track`]'s -- world points,
//! world directions and colours, with no GPU in it. What this module adds is
//! the three things the frame needs: the figure's segments, written into its
//! region of the [overlay ribbon store](super::overlay_ribbons), and its centre
//! disc, in a vertex buffer of its own; the camera block the shader projects
//! through; and the pass that draws them onto the EDL output.
//!
//! The measurements the 3D view shows are drawn in the same call, with the
//! same shader and bind group: their rings, dots and lines are
//! [`OverlayInstance`]s that `viewer_3d::measurements::instances` builds when
//! what they are drawn from changes, written into a region of their own in the
//! same store. They differ from the figure's pieces in what the instances
//! carry -- their shapes and widths, a part behind the scene that holds at
//! `BEHIND_OPACITY` rather than fading, and the pull toward the eye -- and in
//! one thing the pass does: they are a render pass of their own into a layer
//! texture of their own (`MEASUREMENT_TARGET_FORMAT`), which egui paints over
//! the scene texture, so a translucent colour composites on encoded values,
//! as egui painted them.
//!
//! The selected point's track rays are drawn by the same pass too, out of the
//! store's first region on the scene target, which `upload/track_rays.rs`
//! writes: segments in the rays' orange that hold at `BEHIND_OPACITY` behind
//! the scene and are not pulled toward the eye. They are drawn first, under
//! the figure.
//!
//! It runs **after** the EDL resolve and the target indicator, so the figure
//! sits on top of everything the scene texture holds -- it is the thing being
//! worked on; only the measurements' layer is painted over it. Like the
//! target indicator it samples the shared
//! hardware depth buffer rather than testing against it, which is what lets a
//! fragment behind the scene still be drawn, dimmed, instead of discarded
//! (`specs/gui/point-cloud-rendering.md` § "Depth-Aware Transparency"). Each
//! pass has one colour target and no depth attachment, so it writes no pick id
//! and no depth: what a click picks is the scene under it.
//!
//! The figure and the measurements are written only when they change: nothing
//! in their instances depends on the camera, and the camera block this pass
//! writes each frame carries the frame's pixels per point and pull toward the
//! eye ([`bench_track_uniforms`]). The figure's arrowhead turns its barbs to
//! face the eye in the shader ([`OverlayInstance::barb`]), so the figure is its
//! own key: an upload of a figure equal to the one last written writes nothing,
//! neither its region of the store nor its disc.

use wgpu::util::DeviceExt;

use super::gpu_types::{
    mat4_to_cols, pack_rgba8, BenchDiscVertex, BenchTrackUniforms, OverlayInstance,
    BENCH_LINE_HALF_WIDTH,
};
use super::overlay_ribbons::{OverlayTarget, ProducerId};
use super::SceneRenderer;
use crate::viewer_3d::bench_track::{Barb, Figure, Stroke};
use crate::viewer_3d::ViewportCamera;

/// What the overlay pieces are drawn with on one frame that their instances do
/// not hold: the panel's scale and the pull toward the eye. The 3D view sets it
/// each frame it is shown.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct OverlayView {
    /// Physical pixels to a panel point.
    pub(crate) pixels_per_point: f32,
    /// What fraction of its distance to the eye a piece flagged
    /// [`OverlayInstance::PULL_TOWARD_EYE`] is drawn toward it
    /// (`viewer_3d::measurements::toward_eye_fraction`).
    pub(crate) pull_fraction: f32,
}

impl Default for OverlayView {
    fn default() -> Self {
        Self {
            pixels_per_point: 1.0,
            pull_fraction: 0.0,
        }
    }
}

/// The segments and barbs `figure` draws: in their colours at the bench's
/// width in points, fading with how far behind the scene they are, and not
/// pulled toward the eye: its strokes, in [`Figure::strokes`]' order, and then
/// the arrowhead's two barbs.
pub(super) fn figure_instances(figure: &Figure) -> Vec<OverlayInstance> {
    let segment = |stroke: &Stroke| {
        OverlayInstance::segment(
            stroke.a,
            stroke.b,
            stroke.color,
            BENCH_LINE_HALF_WIDTH,
            OverlayInstance::FADE,
        )
    };
    let barb = |barb: &Barb| {
        OverlayInstance::barb(
            barb.tip,
            barb.foot,
            barb.tan_angle,
            barb.color,
            BENCH_LINE_HALF_WIDTH,
            OverlayInstance::FADE,
        )
    };
    figure
        .strokes()
        .map(segment)
        .chain(
            figure
                .normal
                .iter()
                .flat_map(|normal| normal.barbs.iter().map(barb)),
        )
        .collect()
}

/// The bench track pass's camera block for `camera` in a panel `size` physical
/// pixels across, with the figure's `fog_distance` and the frame's `view`.
pub(super) fn bench_track_uniforms(
    camera: &ViewportCamera,
    size: (u32, u32),
    fog_distance: f32,
    view: OverlayView,
) -> BenchTrackUniforms {
    let (w, h) = size;
    let aspect = w as f64 / h as f64;
    let view_matrix = camera.view_matrix();
    let proj = camera.projection_matrix(aspect);
    BenchTrackUniforms {
        view_proj: mat4_to_cols(&(proj * view_matrix)),
        view: mat4_to_cols(&view_matrix),
        proj: mat4_to_cols(&proj),
        screen_size: [w as f32, h as f32],
        near: camera.near as f32,
        fog_distance,
        pixels_per_point: view.pixels_per_point,
        pull_fraction: view.pull_fraction,
        _pad: [0.0; 2],
    }
}

/// How the measurements' pass opens their layer: cleared to transparent, so the
/// layer holds this frame's measurements and nothing else, premultiplied.
pub(super) const MEASUREMENT_LAYER_LOAD: wgpu::LoadOp<wgpu::Color> =
    wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT);

impl SceneRenderer {
    /// Write the rings, dots and lines of the measurements the 3D view draws
    /// into their region of the overlay ribbon store, or empty it when there
    /// are none. Called only when they have changed.
    pub(crate) fn upload_measurements(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        instances: &[OverlayInstance],
    ) {
        self.overlay_ribbons.write(
            device,
            queue,
            ProducerId::Measurements,
            OverlayTarget::MeasurementLayer,
            instances,
        );
    }

    /// How many instances and vertices the measurement and figure passes
    /// draw this frame.
    fn bench_pass_count(&self) -> u32 {
        self.overlay_ribbons.drawn_count(OverlayTarget::Scene)
            + self.bench_track_disc_count
            + self
                .overlay_ribbons
                .drawn_count(OverlayTarget::MeasurementLayer)
    }

    /// Write the figure the focused item draws into its region of the overlay
    /// ribbon store, and upload its centre disc; or clear both. Called every
    /// frame, it writes nothing when `figure` equals the figure it last wrote,
    /// which a frame that only moves the camera leaves it.
    ///
    /// `None` -- a cluster-stage item, a track with no frame, or no focused
    /// item at all -- draws nothing.
    pub(crate) fn upload_bench_track(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        figure: Option<&Figure>,
    ) {
        if figure == self.bench_figure_written.as_ref() {
            return;
        }
        self.bench_figure_written = figure.cloned();
        let Some(figure) = figure else {
            self.clear_bench_track(device, queue);
            return;
        };

        let edges = figure_instances(figure);
        let disc: Vec<BenchDiscVertex> = figure
            .disc
            .iter()
            .map(|vertex| BenchDiscVertex {
                position: vertex.at,
                color: pack_rgba8(vertex.color),
            })
            .collect();
        if edges.is_empty() && disc.is_empty() {
            self.clear_bench_track(device, queue);
            return;
        }

        self.overlay_ribbons.write(
            device,
            queue,
            ProducerId::BenchFigure,
            OverlayTarget::Scene,
            &edges,
        );
        self.bench_track_disc_count = disc.len() as u32;
        self.bench_track_disc_buffer = (!disc.is_empty()).then(|| {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("bench track disc"),
                contents: bytemuck::cast_slice(&disc),
                usage: wgpu::BufferUsages::VERTEX,
            })
        });
        self.bench_track_fog_distance = figure.fog_distance;
    }

    /// Drop the figure's geometry: nothing on the bench to draw. Its region of
    /// the store is emptied and keeps its capacity.
    fn clear_bench_track(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) {
        self.overlay_ribbons.write(
            device,
            queue,
            ProducerId::BenchFigure,
            OverlayTarget::Scene,
            &[],
        );
        self.bench_track_disc_buffer = None;
        self.bench_track_disc_count = 0;
        self.bench_track_fog_distance = 0.0;
    }

    /// Update the bench track pass's camera block, with the frame's pixels per
    /// point and pull toward the eye.
    pub(crate) fn update_bench_track_uniforms(
        &self,
        queue: &wgpu::Queue,
        camera: &ViewportCamera,
        view: OverlayView,
    ) {
        let (w, h) = self.current_size;
        if w == 0 || h == 0 || self.bench_pass_count() == 0 {
            return;
        }
        let Some(buf) = &self.bench_track_uniform_buffer else {
            return;
        };
        let uniforms = bench_track_uniforms(camera, (w, h), self.bench_track_fog_distance, view);
        queue.write_buffer(buf, 0, bytemuck::bytes_of(&uniforms));
    }

    /// Draw the measurements into their layer, and the track rays and then
    /// the figure onto the EDL output texture. `show_track_rays` false leaves
    /// the rays out of this frame without emptying their region
    /// (`AppState::preview_hides_track_rays`).
    ///
    /// The measurements' layer is cleared to transparent every frame, whether
    /// or not there is a measurement to draw, and their pieces are drawn into
    /// it with straight-alpha blending, which over a transparent layer keeps
    /// the layer premultiplied. egui paints it over the scene texture as it
    /// paints its own textures, so a translucent mark or line composites on
    /// encoded values as egui painted it before. The figure is drawn onto the
    /// scene texture over the rays: the disc first and the segments over it,
    /// as the target indicator draws its star under its wireframe, since the
    /// outline is what says where the frame's boundary is, so nothing fills
    /// over it.
    pub(crate) fn render_bench_track(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        show_track_rays: bool,
    ) {
        let Some(bind_group) = &self.bench_track_bind_group else {
            return;
        };
        fn begin<'e>(
            encoder: &'e mut wgpu::CommandEncoder,
            label: &str,
            view: &wgpu::TextureView,
            load: wgpu::LoadOp<wgpu::Color>,
        ) -> wgpu::RenderPass<'e> {
            encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some(label),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load,
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: None,
                ..Default::default()
            })
        }

        if let Some(view) = &self.measurement_layer_view {
            let mut pass = begin(encoder, "measurement pass", view, MEASUREMENT_LAYER_LOAD);
            if let (Some(pipeline), Some(quad_vb)) =
                (&self.measurement_pipeline, &self.quad_vertex_buffer)
            {
                pass.set_pipeline(pipeline);
                pass.set_bind_group(0, bind_group, &[]);
                pass.set_vertex_buffer(0, quad_vb.slice(..));
                self.overlay_ribbons
                    .draw(&mut pass, OverlayTarget::MeasurementLayer);
            }
        }

        let drawn = self
            .overlay_ribbons
            .draws(OverlayTarget::Scene)
            .filter(|(producer, _)| show_track_rays || *producer != ProducerId::TrackRays)
            .count();
        if drawn == 0 && self.bench_track_disc_count == 0 {
            return;
        }
        let Some(edl_output_view) = &self.edl_output_view else {
            return;
        };
        // Preserve the EDL output and the passes over it.
        let mut pass = begin(
            encoder,
            "bench track pass",
            edl_output_view,
            wgpu::LoadOp::Load,
        );

        let edges = |pass: &mut wgpu::RenderPass<'_>, producer: ProducerId| {
            if let (Some(pipeline), Some(quad_vb)) =
                (&self.bench_track_edge_pipeline, &self.quad_vertex_buffer)
            {
                pass.set_pipeline(pipeline);
                pass.set_bind_group(0, bind_group, &[]);
                pass.set_vertex_buffer(0, quad_vb.slice(..));
                self.overlay_ribbons.draw_producer(pass, producer);
            }
        };

        if show_track_rays {
            edges(&mut pass, ProducerId::TrackRays);
        }

        if let (Some(pipeline), Some(disc_buf)) = (
            &self.bench_track_disc_pipeline,
            &self.bench_track_disc_buffer,
        ) {
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, bind_group, &[]);
            pass.set_vertex_buffer(0, disc_buf.slice(..));
            pass.draw(0..self.bench_track_disc_count, 0..1);
        }

        edges(&mut pass, ProducerId::BenchFigure);
    }
}

#[cfg(test)]
mod tests;
