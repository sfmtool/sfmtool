// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

use std::ops::Range;

use super::recon::ReconResources;
use super::SceneRenderer;

/// The vertices of the first draw of the point and patch passes, of the
/// splat corner buffer (`gpu_types::SPLAT_CORNERS`): the unit quad's four
/// corners, for every point the Track Table's filter keeps and every point of
/// a node it is not on.
pub(super) const FIRST_DRAW: Range<u32> = 0..4;
/// The vertices of the dimmed draw: the same four corners again, marked as
/// the dimmed draw's, for the points the filter dims and nothing else.
pub(super) const DIMMED_DRAW: Range<u32> = 4..8;

impl SceneRenderer {
    /// Draw every drawn node's point splats as the draw of `vertices`: the
    /// first draw of every node, or, with `dimmed_only`, the dimmed draw of
    /// the nodes the Track Table's filter dims.
    fn draw_points<'p>(
        &self,
        pass: &mut wgpu::RenderPass<'p>,
        vertices: Range<u32>,
        dimmed_only: bool,
    ) {
        if let (Some(pipeline), Some(preview_pipeline), Some(corners)) = (
            &self.point_pipeline,
            &self.point_preview_pipeline,
            &self.splat_corner_buffer,
        ) {
            pass.set_vertex_buffer(0, corners.slice(..));
            for bundle in self
                .drawn(|b| b.display.show_points)
                .filter(|b| !dimmed_only || b.draws_dimmed())
            {
                // The node a camera model proposal's preview is on is
                // drawn through the variant that reads its preview
                // buffer in slot 3; every other node as before.
                let preview = bundle.preview.as_ref();
                pass.set_pipeline(if preview.is_some() {
                    preview_pipeline
                } else {
                    pipeline
                });
                if let (Some(instance_buf), Some(alive_buf)) =
                    (&bundle.point_instance_buffer, &bundle.point_alive_buffer)
                {
                    if bundle.point_count > 0 {
                        pass.set_bind_group(0, &bundle.point_bind_group, &[]);
                        pass.set_vertex_buffer(1, instance_buf.slice(..));
                        pass.set_vertex_buffer(2, alive_buf.slice(..));
                        if let Some(preview) = preview {
                            pass.set_vertex_buffer(3, preview.points.slice(..));
                        }
                        pass.draw(vertices.clone(), 0..bundle.point_count);
                    }
                }
                // The overlay's additions, after the base's and with
                // the same global uniforms; their own bind group
                // carries only the shifted pick base.
                if let Some(additions) = &bundle.additions {
                    if additions.point_count > 0 {
                        pass.set_bind_group(0, &additions.bind_group, &[]);
                        pass.set_vertex_buffer(1, additions.point_instance_buffer.slice(..));
                        pass.set_vertex_buffer(2, additions.point_alive_buffer.slice(..));
                        if let Some(preview) = preview {
                            let Some(rows) = preview.additions.as_ref() else {
                                continue;
                            };
                            pass.set_vertex_buffer(3, rows.slice(..));
                        }
                        pass.draw(vertices.clone(), 0..additions.point_count);
                    }
                }
            }
        }
    }

    /// Draw every drawn node's patch surfels as the draw of `vertices`, by
    /// the rule of [`Self::draw_points`].
    fn draw_patches<'p>(
        &self,
        pass: &mut wgpu::RenderPass<'p>,
        vertices: Range<u32>,
        dimmed_only: bool,
    ) {
        if let (Some(pipeline), Some(preview_pipeline), Some(corners)) = (
            &self.patch_pipeline,
            &self.patch_preview_pipeline,
            &self.splat_corner_buffer,
        ) {
            pass.set_vertex_buffer(0, corners.slice(..));
            for bundle in self
                .drawn(|b| b.display.show_patches)
                .filter(|b| !dimmed_only || b.draws_dimmed())
            {
                // The base's atlas, then the additions' own: two draws
                // in one pass, and nothing in the shader tells them
                // apart. Each atlas draws from its own uniform block,
                // and under a camera model proposal's preview reads its
                // own surfel preview buffer through the variant
                // pipeline.
                let preview = bundle.preview.as_ref();
                let atlases = [
                    (bundle.patch.as_ref(), preview.map(|p| p.patches.as_ref())),
                    (
                        bundle.additions.as_ref().and_then(|a| a.patch.as_ref()),
                        preview.map(|p| p.addition_patches.as_ref()),
                    ),
                ];
                for (patch, rows) in atlases {
                    let Some(patch) = patch.filter(|patch| patch.count > 0) else {
                        continue;
                    };
                    match rows {
                        None => pass.set_pipeline(pipeline),
                        Some(Some(rows)) => {
                            pass.set_pipeline(preview_pipeline);
                            pass.set_vertex_buffer(3, rows.slice(..));
                        }
                        // Unreachable while the rows are built from
                        // the same bundle; drawn as nothing rather
                        // than read past a buffer.
                        Some(None) => continue,
                    }
                    pass.set_bind_group(0, &patch.bind_group, &[]);
                    pass.set_vertex_buffer(1, patch.instance_buffer.slice(..));
                    pass.set_vertex_buffer(2, patch.alive_buffer.slice(..));
                    pass.draw(vertices.clone(), 0..patch.count);
                }
            }
        }
    }
}

impl SceneRenderer {
    /// Render the target indicator onto the EDL output texture.
    ///
    /// This is Pass 2.5: renders after the EDL pass using additive blending,
    /// with depth-aware transparency from sampling the linear depth texture.
    pub(crate) fn render_target_indicator(&self, encoder: &mut wgpu::CommandEncoder) {
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
    pub(crate) fn render_track_rays(&self, encoder: &mut wgpu::CommandEncoder) {
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
    pub(crate) fn render(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        show_points: bool,
        show_camera_images: bool,
        show_patches: bool,
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

            // Every scene pass below is a loop over the loaded reconstructions:
            // bind that node's bind group (which carries its ReconUniforms
            // slice, thumbnail atlas and frustum colors), then the existing
            // instanced draw. Nodes share the one depth buffer, so mutual
            // occlusion between reconstructions is automatic.
            //
            // A layer's effective visibility is the AND of three switches: the
            // global HUD toggle (the `show_*` arguments, master switches across
            // all nodes), the node's effective whole-node visibility (eye AND
            // solo), and the node's own group eye — the last two both resolved
            // by `SceneRenderer::drawn`.
            let bundles = |layer: fn(&ReconResources) -> bool| self.drawn(layer);

            if show_points {
                self.draw_points(&mut pass, FIRST_DRAW, false);
            }

            // ── Pass 1b: Frustum wireframes ──
            // Uses same color + linear depth + hw depth attachments.
            // The frustum pipeline has an empty write mask for linear depth,
            // so point cloud depth is preserved for EDL and depth readback.
            if show_camera_images {
                if let (Some(pipeline), Some(quad_vb)) =
                    (&self.frustum_pipeline, &self.quad_vertex_buffer)
                {
                    pass.set_pipeline(pipeline);
                    pass.set_vertex_buffer(0, quad_vb.slice(..));
                    for bundle in bundles(|b| b.display.show_camera_images) {
                        if let (Some(bind_group), Some(edge_buf)) =
                            (&bundle.frustum_bind_group, &bundle.frustum_edge_buffer)
                        {
                            if bundle.frustum_edge_count > 0 {
                                pass.set_bind_group(0, bind_group, &[]);
                                pass.set_vertex_buffer(1, edge_buf.slice(..));
                                pass.draw(0..4, 0..bundle.frustum_edge_count);
                            }
                        }
                    }
                }

                // ── Pass 1c: Image quads on frustum far planes ──
                // Pinhole cameras: instanced flat quads
                if let (Some(pipeline), Some(quad_vb)) =
                    (&self.image_quad_pipeline, &self.quad_vertex_buffer)
                {
                    pass.set_pipeline(pipeline);
                    pass.set_vertex_buffer(0, quad_vb.slice(..));
                    for bundle in bundles(|b| b.display.show_camera_images) {
                        if let (Some(bind_group), Some(instance_buf)) = (
                            &bundle.image_quad_bind_group,
                            &bundle.image_quad_instance_buffer,
                        ) {
                            if bundle.image_quad_count > 0 {
                                pass.set_bind_group(0, bind_group, &[]);
                                pass.set_vertex_buffer(1, instance_buf.slice(..));
                                pass.draw(0..4, 0..bundle.image_quad_count);
                            }
                        }
                    }
                }

                // Distorted cameras: tessellated indexed quads
                if let Some(pipeline) = &self.distorted_quad_pipeline {
                    pass.set_pipeline(pipeline);
                    for bundle in bundles(|b| b.display.show_camera_images) {
                        if let (Some(bind_group), Some(vbuf), Some(ibuf)) = (
                            &bundle.image_quad_bind_group,
                            &bundle.distorted_quad_vertex_buffer,
                            &bundle.distorted_quad_index_buffer,
                        ) {
                            if bundle.distorted_quad_index_count > 0 {
                                pass.set_bind_group(0, bind_group, &[]);
                                pass.set_vertex_buffer(0, vbuf.slice(..));
                                pass.set_index_buffer(ibuf.slice(..), wgpu::IndexFormat::Uint32);
                                pass.draw_indexed(0..bundle.distorted_quad_index_count, 0, 0..1);
                            }
                        }
                    }
                }
            }

            // ── Pass 1d: Patch surfels (textured oriented quads) ──
            // Writes real hardware depth (occludes/occluded by points and
            // image quads) but 0.0 linear depth (EDL passthrough), and
            // PICK_TAG_POINT | point pick index so clicks select the patch's
            // point.
            if show_patches {
                self.draw_patches(&mut pass, FIRST_DRAW, false);
            }

            // ── Pass 1e: The Track Table filter's dimmed draw ──
            // The points and surfels the filter dims, at its dim alpha, after
            // every node's first draw, so a point the filter keeps is never
            // hidden behind a dimmed one drawn before it: a kept point behind
            // a dimmed one shows through it, and a dimmed one behind a kept
            // one fails the depth test. Both draws write depth and pick ids,
            // so a dimmed point is still picked where it is in front.
            if show_points {
                self.draw_points(&mut pass, DIMMED_DRAW, true);
            }
            if show_patches {
                self.draw_patches(&mut pass, DIMMED_DRAW, true);
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
