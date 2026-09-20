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
//! It runs **after** the EDL resolve, and after the target indicator and the
//! track rays, so the figure sits on top of everything the frame has drawn --
//! it is the thing being worked on. Like both of those it samples the shared
//! hardware depth buffer rather than testing against it, which is what lets a
//! fragment behind the scene still be drawn, dimmed, instead of discarded
//! (`specs/gui/point-cloud-rendering.md` § "Depth-Aware Transparency").
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
use crate::viewer_3d::ViewportCamera;

impl SceneRenderer {
    /// Upload the figure the bench's active track draws, or clear it.
    ///
    /// `None` -- a cluster-stage item, a track with no frame, or no active item
    /// at all -- uploads nothing and draws nothing.
    pub fn upload_bench_track(&mut self, device: &wgpu::Device, figure: Option<&Figure>) {
        let Some(figure) = figure else {
            self.clear_bench_track();
            return;
        };

        let edges: Vec<BenchEdgeInstance> = figure
            .strokes()
            .map(|stroke| BenchEdgeInstance {
                endpoint_a: stroke.a,
                endpoint_b: stroke.b,
                color: stroke.color,
            })
            .collect();
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
    pub fn clear_bench_track(&mut self) {
        self.bench_track_edge_buffer = None;
        self.bench_track_edge_count = 0;
        self.bench_track_disc_buffer = None;
        self.bench_track_disc_count = 0;
        self.bench_track_fog_distance = 0.0;
    }

    /// Update the bench track pass's camera block.
    pub fn update_bench_track_uniforms(&self, queue: &wgpu::Queue, camera: &ViewportCamera) {
        let (w, h) = self.current_size;
        if w == 0 || h == 0 || self.bench_track_edge_count + self.bench_track_disc_count == 0 {
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
            line_half_width: BENCH_LINE_HALF_WIDTH,
            near: camera.near as f32,
            fog_distance: self.bench_track_fog_distance,
            _pad: [0.0; 3],
        };
        queue.write_buffer(buf, 0, bytemuck::bytes_of(&uniforms));
    }

    /// Draw the figure onto the EDL output texture.
    ///
    /// The disc first and the edges over it, as the target indicator draws its
    /// star under its wireframe: the outline is what says where the frame's
    /// boundary is, so nothing fills over it.
    pub fn render_bench_track(&self, encoder: &mut wgpu::CommandEncoder) {
        if self.bench_track_edge_count + self.bench_track_disc_count == 0 {
            return;
        }
        let Some(edl_output_view) = &self.edl_output_view else {
            return;
        };
        let Some(bind_group) = &self.bench_track_bind_group else {
            return;
        };

        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("bench track pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: edl_output_view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load, // preserve EDL + the two passes over it
                    store: wgpu::StoreOp::Store,
                },
                depth_slice: None,
            })],
            depth_stencil_attachment: None,
            ..Default::default()
        });

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
