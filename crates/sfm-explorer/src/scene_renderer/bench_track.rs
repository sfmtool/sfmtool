// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The bench track pass: the upload, the uniform write and the draw.
//!
//! The figure itself is [`crate::viewer_3d::bench_track`]'s -- world points,
//! world directions and colours, with no GPU in it. What this module adds is
//! the three things the frame needs: the two vertex buffers, the camera block
//! the shader projects through, and the pass that draws them onto the EDL
//! output.
//!
//! The measurements the 3D view shows are drawn in the same call, with
//! the same shader and bind group: their marks and lines are ribbons
//! ([`crate::viewer_3d::measurements::Ribbon`]) built on the CPU each frame
//! and uploaded as edges of the figure's own kind. They differ from the
//! figure's in two things the instance carries, their width and that the
//! part behind the scene holds at `BEHIND_OPACITY` rather than fading, and
//! in one the pass does not: they are a render pass of their own into a layer
//! texture of their own (`MEASUREMENT_TARGET_FORMAT`), which egui paints over
//! the scene texture, so a translucent colour composites on encoded values,
//! as egui painted them.
//!
//! It runs **after** the EDL resolve, and after the target indicator and the
//! track rays, so the figure sits on top of everything the frame has drawn --
//! it is the thing being worked on. Like both of those it samples the shared
//! hardware depth buffer rather than testing against it, which is what lets a
//! fragment behind the scene still be drawn, dimmed, instead of discarded
//! (`specs/gui/point-cloud-rendering.md` § "Depth-Aware Transparency"). Each
//! pass has one colour target and no depth attachment, so it writes no pick id
//! and no depth: what a click picks is the scene under it.
//!
//! The figure is rebuilt and re-uploaded every frame it is drawn, because the
//! arrowhead's barbs turn to face the eye: there is no state it could be keyed
//! on that a camera move does not invalidate. It is a few hundred instances.

use wgpu::util::DeviceExt;

use super::gpu_types::{
    BenchDiscVertex, BenchEdgeInstance, BenchTrackUniforms, BENCH_LINE_HALF_WIDTH,
};
use super::SceneRenderer;
use crate::viewer_3d::bench_track::Figure;
use crate::viewer_3d::measurements::Ribbon;
use crate::viewer_3d::ViewportCamera;

/// The edges `figure` draws: in its strokes' colours at the bench's width,
/// fading with how far behind the scene they are.
pub(super) fn figure_instances(figure: &Figure) -> Vec<BenchEdgeInstance> {
    figure
        .strokes()
        .map(|stroke| BenchEdgeInstance {
            endpoint_a: stroke.a,
            endpoint_b: stroke.b,
            color: stroke.color,
            half_width: BENCH_LINE_HALF_WIDTH,
            fade: 1.0,
        })
        .collect()
}

/// The edges the measurements' `ribbons` draw: each at its own width, and
/// held at `BEHIND_OPACITY` wherever it is behind the scene.
pub(super) fn measurement_instances(ribbons: &[Ribbon]) -> Vec<BenchEdgeInstance> {
    ribbons
        .iter()
        .map(|ribbon| BenchEdgeInstance {
            endpoint_a: ribbon.a,
            endpoint_b: ribbon.b,
            color: ribbon.color,
            half_width: ribbon.half_width,
            fade: 0.0,
        })
        .collect()
}

/// How the measurements' pass opens their layer: cleared to transparent, so the
/// layer holds this frame's measurements and nothing else, premultiplied.
pub(super) const MEASUREMENT_LAYER_LOAD: wgpu::LoadOp<wgpu::Color> =
    wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT);

impl SceneRenderer {
    /// Upload the marks and lines of the measurements the 3D view draws, or
    /// clear them when there are none.
    pub(crate) fn upload_measurements(&mut self, device: &wgpu::Device, ribbons: &[Ribbon]) {
        let edges = measurement_instances(ribbons);
        self.measurement_edge_count = edges.len() as u32;
        self.measurement_edge_buffer = (!edges.is_empty()).then(|| {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("measurement edges"),
                contents: bytemuck::cast_slice(&edges),
                usage: wgpu::BufferUsages::VERTEX,
            })
        });
    }

    /// How many instances and vertices the measurement and figure passes
    /// draw this frame.
    fn bench_pass_count(&self) -> u32 {
        self.bench_track_edge_count + self.bench_track_disc_count + self.measurement_edge_count
    }

    /// Upload the figure the focused item draws, or clear it.
    ///
    /// `None` -- a cluster-stage item, a track with no frame, or no focused
    /// item at all -- uploads nothing and draws nothing.
    pub(crate) fn upload_bench_track(&mut self, device: &wgpu::Device, figure: Option<&Figure>) {
        let Some(figure) = figure else {
            self.clear_bench_track();
            return;
        };

        let edges = figure_instances(figure);
        let disc: Vec<BenchDiscVertex> = figure
            .disc
            .iter()
            .map(|vertex| BenchDiscVertex {
                position: vertex.at,
                color: vertex.color,
            })
            .collect();
        if edges.is_empty() && disc.is_empty() {
            self.clear_bench_track();
            return;
        }

        self.bench_track_edge_count = edges.len() as u32;
        self.bench_track_edge_buffer = (!edges.is_empty()).then(|| {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("bench track edges"),
                contents: bytemuck::cast_slice(&edges),
                usage: wgpu::BufferUsages::VERTEX,
            })
        });
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

    /// Drop the figure's geometry: nothing on the bench to draw.
    pub(crate) fn clear_bench_track(&mut self) {
        self.bench_track_edge_buffer = None;
        self.bench_track_edge_count = 0;
        self.bench_track_disc_buffer = None;
        self.bench_track_disc_count = 0;
        self.bench_track_fog_distance = 0.0;
    }

    /// Update the bench track pass's camera block.
    pub(crate) fn update_bench_track_uniforms(&self, queue: &wgpu::Queue, camera: &ViewportCamera) {
        let (w, h) = self.current_size;
        if w == 0 || h == 0 || self.bench_pass_count() == 0 {
            return;
        }
        let Some(buf) = &self.bench_track_uniform_buffer else {
            return;
        };

        let aspect = w as f64 / h as f64;
        let view = camera.view_matrix();
        let view_proj = camera.projection_matrix(aspect) * view;

        let uniforms = BenchTrackUniforms {
            view_proj: super::gpu_types::mat4_to_cols(&view_proj),
            view: super::gpu_types::mat4_to_cols(&view),
            screen_size: [w as f32, h as f32],
            near: camera.near as f32,
            fog_distance: self.bench_track_fog_distance,
        };
        queue.write_buffer(buf, 0, bytemuck::bytes_of(&uniforms));
    }

    /// Draw the measurements into their layer, and the figure onto the EDL
    /// output texture.
    ///
    /// The measurements' layer is cleared to transparent every frame, whether
    /// or not there is a measurement to draw, and the ribbons are drawn into it
    /// with straight-alpha blending, which over a transparent layer keeps the
    /// layer premultiplied. egui paints it over the scene texture as it paints
    /// its own textures, so a translucent mark or line composites on encoded
    /// values as egui painted it before. The figure is drawn onto the scene
    /// texture: the disc first and the edges over it, as the target indicator
    /// draws its star under its wireframe, since the outline is what says
    /// where the frame's boundary is, so nothing fills over it.
    pub(crate) fn render_bench_track(&self, encoder: &mut wgpu::CommandEncoder) {
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
            if let (Some(pipeline), Some(quad_vb), Some(edge_buf)) = (
                &self.measurement_pipeline,
                &self.quad_vertex_buffer,
                &self.measurement_edge_buffer,
            ) {
                pass.set_pipeline(pipeline);
                pass.set_bind_group(0, bind_group, &[]);
                pass.set_vertex_buffer(0, quad_vb.slice(..));
                pass.set_vertex_buffer(1, edge_buf.slice(..));
                pass.draw(0..4, 0..self.measurement_edge_count);
            }
        }

        if self.bench_track_edge_count + self.bench_track_disc_count == 0 {
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

        if let (Some(pipeline), Some(disc_buf)) = (
            &self.bench_track_disc_pipeline,
            &self.bench_track_disc_buffer,
        ) {
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, bind_group, &[]);
            pass.set_vertex_buffer(0, disc_buf.slice(..));
            pass.draw(0..self.bench_track_disc_count, 0..1);
        }

        if let (Some(pipeline), Some(quad_vb), Some(edge_buf)) = (
            &self.bench_track_edge_pipeline,
            &self.quad_vertex_buffer,
            &self.bench_track_edge_buffer,
        ) {
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, bind_group, &[]);
            pass.set_vertex_buffer(0, quad_vb.slice(..));
            pass.set_vertex_buffer(1, edge_buf.slice(..));
            pass.draw(0..4, 0..self.bench_track_edge_count);
        }
    }
}

#[cfg(test)]
mod tests;
