// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

use super::SceneRenderer;

impl SceneRenderer {
    /// Render the target indicator onto the EDL output texture.
    ///
    /// This is Pass 2.5: renders after the EDL pass using additive blending,
    /// with depth-aware transparency from sampling the linear depth texture.
    pub fn render_target_indicator(&self, encoder: &mut wgpu::CommandEncoder) {
        let Some(edl_output_view) = &self.edl_output_view else {
            return;
        };

        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("target indicator pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: edl_output_view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load, // preserve EDL output
                    store: wgpu::StoreOp::Store,
                },
                depth_slice: None,
            })],
            depth_stencil_attachment: None,
            ..Default::default()
        });

        let Some(bind_group) = &self.target_bind_group else {
            return;
        };

        // Draw filled star polygon first (beneath the wireframe)
        if let (Some(star_pipeline), Some(star_buf)) =
            (&self.target_star_pipeline, &self.target_star_buffer)
        {
            pass.set_pipeline(star_pipeline);
            pass.set_bind_group(0, bind_group, &[]);
            pass.set_vertex_buffer(0, star_buf.slice(..));
            pass.draw(0..self.target_star_vertex_count, 0..1);
        }

        // Draw wireframe edges (vertical axis + circular ring) on top
        if let (Some(edge_pipeline), Some(quad_vb), Some(edge_buf)) = (
            &self.target_edge_pipeline,
            &self.quad_vertex_buffer,
            &self.target_edge_buffer,
        ) {
            pass.set_pipeline(edge_pipeline);
            pass.set_bind_group(0, bind_group, &[]);
            pass.set_vertex_buffer(0, quad_vb.slice(..));
            pass.set_vertex_buffer(1, edge_buf.slice(..));
            pass.draw(0..4, 0..self.target_edge_count);
        }
    }

    /// Render track rays onto the EDL output texture.
    ///
    /// Pass 2.75: renders after the target indicator using alpha blending,
    /// with depth-aware occlusion from sampling the hardware depth buffer.
    pub fn render_track_rays(&self, encoder: &mut wgpu::CommandEncoder) {
        if self.track_ray_count == 0 {
            return;
        }

        let Some(edl_output_view) = &self.edl_output_view else {
            return;
        };

        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("track ray pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: edl_output_view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load, // preserve EDL + target indicator output
                    store: wgpu::StoreOp::Store,
                },
                depth_slice: None,
            })],
            depth_stencil_attachment: None,
            ..Default::default()
        });

        if let (Some(pipeline), Some(bind_group), Some(quad_vb), Some(edge_buf)) = (
            &self.track_ray_pipeline,
            &self.track_ray_bind_group,
            &self.quad_vertex_buffer,
            &self.track_ray_edge_buffer,
        ) {
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, bind_group, &[]);
            pass.set_vertex_buffer(0, quad_vb.slice(..));
            pass.set_vertex_buffer(1, edge_buf.slice(..));
            pass.draw(0..4, 0..self.track_ray_count);
        }
    }

    /// Render the scene.
    ///
    /// Pass 0:  Background image (camera view only) → edl_output
    /// Pass 1:  Point splats + frustums → color + linear depth textures
    /// Pass 2:  EDL post-process → edl_output (Load or Clear depending on Pass 0)
    pub fn render(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        show_points: bool,
        show_camera_images: bool,
        in_camera_view: bool,
    ) {
        let Some(edl_output_view) = &self.edl_output_view else {
            return;
        };

        // BG_COLOR as wgpu::Color (linear sRGB ≈ (30, 30, 35))
        let bg_color = wgpu::Color {
            r: 0.013,
            g: 0.013,
            b: 0.017,
            a: 1.0,
        };

        // ── Pass 0: Background image (camera view only) ──
        // Renders the full-res image into edl_output, cleared to BG_COLOR
        // so letterbox/pillarbox bars match the normal background.
        let has_bg_image = in_camera_view && self.bg_image_bind_group.is_some();
        if has_bg_image {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("bg image pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: edl_output_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(bg_color),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: None,
                ..Default::default()
            });

            if let (Some(bind_group), Some(pipeline), Some(vb), Some(ib)) = (
                &self.bg_image_bind_group,
                &self.bg_image_distorted_pipeline,
                &self.bg_image_distorted_vertex_buffer,
                &self.bg_image_distorted_index_buffer,
            ) {
                pass.set_pipeline(pipeline);
                pass.set_bind_group(0, bind_group, &[]);
                pass.set_vertex_buffer(0, vb.slice(..));
                pass.set_index_buffer(ib.slice(..), wgpu::IndexFormat::Uint32);
                pass.draw_indexed(0..self.bg_image_distorted_index_count, 0, 0..1);
            }
        }

        // ── Pass 1: Point splat + frustum rendering ──
        {
            let Some(color_view) = &self.splat_color_view else {
                return;
            };
            let Some(depth_view) = &self.linear_depth_view else {
                return;
            };
            let Some(hw_depth_view) = &self.hw_depth_view else {
                return;
            };
            let Some(pick_view) = &self.pick_texture_view else {
                return;
            };

            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("point splat pass"),
                color_attachments: &[
                    // @location(0): color
                    // Cleared to transparent so premultiplied alpha blending
                    // doesn't create dark halos at splat edges.
                    Some(wgpu::RenderPassColorAttachment {
                        view: color_view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                            store: wgpu::StoreOp::Store,
                        },
                        depth_slice: None,
                    }),
                    // @location(1): linear depth
                    Some(wgpu::RenderPassColorAttachment {
                        view: depth_view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color {
                                r: 0.0,
                                g: 0.0,
                                b: 0.0,
                                a: 0.0,
                            }),
                            store: wgpu::StoreOp::Store,
                        },
                        depth_slice: None,
                    }),
                    // @location(2): pick ID (cleared to 0 = PICK_TAG_NONE)
                    Some(wgpu::RenderPassColorAttachment {
                        view: pick_view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color {
                                r: 0.0,
                                g: 0.0,
                                b: 0.0,
                                a: 0.0,
                            }),
                            store: wgpu::StoreOp::Store,
                        },
                        depth_slice: None,
                    }),
                ],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: hw_depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(0.0), // reversed-Z: 0 = far
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                ..Default::default()
            });

            if show_points {
                if let (Some(pipeline), Some(bind_group), Some(quad_vb), Some(instance_buf)) = (
                    &self.point_pipeline,
                    &self.point_bind_group,
                    &self.quad_vertex_buffer,
                    &self.instance_buffer,
                ) {
                    if self.point_count > 0 {
                        pass.set_pipeline(pipeline);
                        pass.set_bind_group(0, bind_group, &[]);
                        pass.set_vertex_buffer(0, quad_vb.slice(..));
                        pass.set_vertex_buffer(1, instance_buf.slice(..));
                        pass.draw(0..4, 0..self.point_count);
                    }
                }
            }

            // ── Pass 1b: Frustum wireframes ──
            // Uses same color + linear depth + hw depth attachments.
            // The frustum pipeline has an empty write mask for linear depth,
            // so point cloud depth is preserved for EDL and depth readback.
            if show_camera_images {
                if let (Some(pipeline), Some(bind_group), Some(quad_vb), Some(edge_buf)) = (
                    &self.frustum_pipeline,
                    &self.frustum_bind_group,
                    &self.quad_vertex_buffer,
                    &self.frustum_edge_buffer,
                ) {
                    if self.frustum_edge_count > 0 {
                        pass.set_pipeline(pipeline);
                        pass.set_bind_group(0, bind_group, &[]);
                        pass.set_vertex_buffer(0, quad_vb.slice(..));
                        pass.set_vertex_buffer(1, edge_buf.slice(..));
                        pass.draw(0..4, 0..self.frustum_edge_count);
                    }
                }

                // ── Pass 1c: Image quads on frustum far planes ──
                // Pinhole cameras: instanced flat quads
                if let (Some(pipeline), Some(bind_group), Some(quad_vb), Some(instance_buf)) = (
                    &self.image_quad_pipeline,
                    &self.image_quad_bind_group,
                    &self.quad_vertex_buffer,
                    &self.image_quad_instance_buffer,
                ) {
                    if self.image_quad_count > 0 {
                        pass.set_pipeline(pipeline);
                        pass.set_bind_group(0, bind_group, &[]);
                        pass.set_vertex_buffer(0, quad_vb.slice(..));
                        pass.set_vertex_buffer(1, instance_buf.slice(..));
                        pass.draw(0..4, 0..self.image_quad_count);
                    }
                }

                // Distorted cameras: tessellated indexed quads
                if let (Some(pipeline), Some(bind_group), Some(vbuf), Some(ibuf)) = (
                    &self.distorted_quad_pipeline,
                    &self.image_quad_bind_group,
                    &self.distorted_quad_vertex_buffer,
                    &self.distorted_quad_index_buffer,
                ) {
                    if self.distorted_quad_index_count > 0 {
                        pass.set_pipeline(pipeline);
                        pass.set_bind_group(0, bind_group, &[]);
                        pass.set_vertex_buffer(0, vbuf.slice(..));
                        pass.set_index_buffer(ibuf.slice(..), wgpu::IndexFormat::Uint32);
                        pass.draw_indexed(0..self.distorted_quad_index_count, 0, 0..1);
                    }
                }
            }
        }

        // ── Pass 2: EDL post-process ──
        // When Pass 0 ran, use LoadOp::Load to preserve the background image.
        // Otherwise, clear to BG_COLOR.
        {
            let edl_load_op = if has_bg_image {
                wgpu::LoadOp::Load
            } else {
                wgpu::LoadOp::Clear(bg_color)
            };

            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("edl pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: edl_output_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: edl_load_op,
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: None,
                ..Default::default()
            });

            if let (Some(pipeline), Some(bind_group)) = (&self.edl_pipeline, &self.edl_bind_group) {
                pass.set_pipeline(pipeline);
                pass.set_bind_group(0, bind_group, &[]);
                pass.draw(0..3, 0..1); // fullscreen triangle
            }
        }
    }
}
