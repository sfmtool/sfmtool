// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Camera frustum wireframe + image quad upload, and the per-image color
//! storage buffer they share.

use super::super::gpu_types::{
    DistortedQuadVertex, FrustumEdge, ImageQuadInstance, DISTORTION_SUBDIVISIONS,
    FISHEYE_SUBDIVISIONS,
};
use super::super::SceneRenderer;
use crate::scene::ReconId;
use sfmtool_core::camera::frustum::{compute_distorted_frustum_grid, compute_frustum_corners};
use sfmtool_core::SfmrReconstruction;
use wgpu::util::DeviceExt;

/// Alpha of an ordinary frustum's color: semi-transparent white.
///
/// **Load-bearing for the node tint.** `frustum.wgsl` tints a frustum only when
/// its alpha is below full, which is how it tints a node's own frustums without
/// touching the highlight colors below — so this must stay strictly less than
/// [`FRUSTUM_ALPHA_HIGHLIGHT`], and any future *non*-highlight color has to use
/// it too. [`frustum_colors`] is asserted against that in the upload tests.
pub(super) const FRUSTUM_ALPHA_DEFAULT: u32 = 180;

/// Alpha every highlight color is written at. Full opacity is both how a
/// highlight stands out against the default frustums and how the shader tells
/// it apart from them.
pub(super) const FRUSTUM_ALPHA_HIGHLIGHT: u32 = 255;

/// The default (un-highlighted) frustum color: semi-transparent white.
pub(super) const FRUSTUM_COLOR_DEFAULT: u32 =
    255 | (255 << 8) | (255 << 16) | (FRUSTUM_ALPHA_DEFAULT << 24);

/// The selected camera: opaque cyan.
const FRUSTUM_COLOR_SELECTED: u32 = (255 << 8) | (255 << 16) | (FRUSTUM_ALPHA_HIGHLIGHT << 24);

/// A camera observing the selected point: opaque orange.
const FRUSTUM_COLOR_TRACK: u32 = 255 | (165 << 8) | (FRUSTUM_ALPHA_HIGHLIGHT << 24);

/// Alpha 0 → the shader discards the fragment entirely.
const FRUSTUM_COLOR_HIDDEN: u32 = 0;

/// The per-image color array for one node: default white, with the selected
/// camera, the selected point's track cameras, and the camera being viewed
/// through overriding it in that order of precedence.
///
/// Pulled out of [`SceneRenderer::update_frustum_colors`] so the alpha
/// invariant the tint depends on is assertable without a GPU.
pub(super) fn frustum_colors(
    image_count: usize,
    selected_image: Option<usize>,
    hidden_image: Option<usize>,
    track_images: &[usize],
) -> Vec<u32> {
    let mut colors: Vec<u32> = vec![FRUSTUM_COLOR_DEFAULT; image_count];
    if let Some(idx) = selected_image {
        if idx < image_count {
            colors[idx] = FRUSTUM_COLOR_SELECTED;
        }
    }
    for &idx in track_images {
        if idx < image_count && selected_image != Some(idx) {
            colors[idx] = FRUSTUM_COLOR_TRACK;
        }
    }
    // Hidden must be applied last so it wins over selected/track
    if let Some(idx) = hidden_image {
        if idx < image_count {
            colors[idx] = FRUSTUM_COLOR_HIDDEN;
        }
    }
    colors
}

impl SceneRenderer {
    /// Upload one reconstruction's camera frustum edge geometry to the GPU.
    ///
    /// Builds 8 edges per camera (4 side edges from apex to far corners + 4
    /// base edges around the far face). The stub depth is `length_scale *
    /// frustum_size_multiplier`.
    ///
    /// Colors are stored in a separate per-image storage buffer that can be
    /// updated cheaply via [`Self::update_frustum_colors`] without recomputing
    /// geometry.
    /// Hidden cameras are handled by setting alpha=0 in the color buffer (the
    /// shader discards those fragments), so geometry includes all cameras.
    pub fn upload_frustums(
        &mut self,
        device: &wgpu::Device,
        id: ReconId,
        recon: &SfmrReconstruction,
        length_scale: f32,
        frustum_size_multiplier: f32,
    ) {
        let far_z = (length_scale * frustum_size_multiplier) as f64;

        let mut edges: Vec<FrustumEdge> = Vec::with_capacity(recon.images.len() * 8);

        // Pinhole (instanced) image quads
        let mut pinhole_quads: Vec<ImageQuadInstance> = Vec::new();
        // Distorted (tessellated) image quads
        let mut distorted_vertices: Vec<DistortedQuadVertex> = Vec::new();
        let mut distorted_indices: Vec<u32> = Vec::new();

        self.ensure_recon(device, id);
        let has_thumbnails = self.recons[&id].thumbnail_texture.is_some();

        for (image_idx, image) in recon.images.iter().enumerate() {
            let camera = &recon.cameras[image.camera_index as usize];
            let center = image.camera_center();
            let r = image.camera_to_world_rotation_flat();
            let center_arr = [center.x, center.y, center.z];
            let apex = [center.x as f32, center.y as f32, center.z as f32];

            if camera.has_distortion() || camera.model.is_fisheye() {
                // ── Distorted or fisheye camera: tessellated grid ──
                let subdivisions = if camera.model.is_fisheye() {
                    FISHEYE_SUBDIVISIONS
                } else {
                    DISTORTION_SUBDIVISIONS
                };
                let n = subdivisions + 1;
                let grid =
                    compute_distorted_frustum_grid(&center_arr, &r, camera, far_z, subdivisions);

                // Helper to get grid position as [f32; 3]
                let pos = |i: usize, j: usize| -> [f32; 3] {
                    let idx = (j * n + i) * 3;
                    [
                        grid.positions[idx] as f32,
                        grid.positions[idx + 1] as f32,
                        grid.positions[idx + 2] as f32,
                    ]
                };

                // 4 corner indices: TL=(0,0), TR=(n-1,0), BR=(n-1,n-1), BL=(0,n-1)
                let far_corners = [pos(0, 0), pos(n - 1, 0), pos(n - 1, n - 1), pos(0, n - 1)];

                // 4 side edges: apex to each corner
                for fc in &far_corners {
                    edges.push(FrustumEdge {
                        endpoint_a: apex,
                        _pad0: 0,
                        endpoint_b: *fc,
                        frustum_index: image_idx as u32,
                    });
                }

                // Tessellated base edges: walk the grid boundary
                // Top edge: (0,0)→(1,0)→...→(n-1,0)
                for i in 0..n - 1 {
                    edges.push(FrustumEdge {
                        endpoint_a: pos(i, 0),
                        _pad0: 0,
                        endpoint_b: pos(i + 1, 0),
                        frustum_index: image_idx as u32,
                    });
                }
                // Right edge: (n-1,0)→(n-1,1)→...→(n-1,n-1)
                for j in 0..n - 1 {
                    edges.push(FrustumEdge {
                        endpoint_a: pos(n - 1, j),
                        _pad0: 0,
                        endpoint_b: pos(n - 1, j + 1),
                        frustum_index: image_idx as u32,
                    });
                }
                // Bottom edge: (n-1,n-1)→(n-2,n-1)→...→(0,n-1)
                for i in (0..n - 1).rev() {
                    edges.push(FrustumEdge {
                        endpoint_a: pos(i + 1, n - 1),
                        _pad0: 0,
                        endpoint_b: pos(i, n - 1),
                        frustum_index: image_idx as u32,
                    });
                }
                // Left edge: (0,n-1)→(0,n-2)→...→(0,0)
                for j in (0..n - 1).rev() {
                    edges.push(FrustumEdge {
                        endpoint_a: pos(0, j + 1),
                        _pad0: 0,
                        endpoint_b: pos(0, j),
                        frustum_index: image_idx as u32,
                    });
                }

                // Build tessellated image quad mesh
                if has_thumbnails {
                    let base_vertex = distorted_vertices.len() as u32;

                    // Emit N*N vertices
                    for j in 0..n {
                        for i in 0..n {
                            distorted_vertices.push(DistortedQuadVertex {
                                position: pos(i, j),
                                frustum_index: image_idx as u32,
                                uv: [i as f32 / (n - 1) as f32, j as f32 / (n - 1) as f32],
                                _pad: [0.0; 2],
                            });
                        }
                    }

                    // Emit (N-1)*(N-1)*2 triangles (6 indices per cell)
                    for j in 0..n - 1 {
                        for i in 0..n - 1 {
                            let tl = base_vertex + (j * n + i) as u32;
                            let tr = tl + 1;
                            let bl = base_vertex + ((j + 1) * n + i) as u32;
                            let br = bl + 1;
                            // Two triangles: TL-BL-TR, TR-BL-BR
                            distorted_indices.extend_from_slice(&[tl, bl, tr, tr, bl, br]);
                        }
                    }
                }
            } else {
                // ── Pinhole camera: existing flat quad path ──
                let (fx, fy) = camera.focal_lengths();
                let (cx, cy) = camera.principal_point();

                let corners = compute_frustum_corners(
                    &center_arr,
                    &r,
                    fx,
                    fy,
                    cx,
                    cy,
                    camera.width,
                    camera.height,
                    0.0,
                    far_z,
                );

                let far = [
                    [corners[12] as f32, corners[13] as f32, corners[14] as f32], // far TL
                    [corners[15] as f32, corners[16] as f32, corners[17] as f32], // far TR
                    [corners[18] as f32, corners[19] as f32, corners[20] as f32], // far BR
                    [corners[21] as f32, corners[22] as f32, corners[23] as f32], // far BL
                ];

                // 4 side edges: apex to each far corner
                for fc in &far {
                    edges.push(FrustumEdge {
                        endpoint_a: apex,
                        _pad0: 0,
                        endpoint_b: *fc,
                        frustum_index: image_idx as u32,
                    });
                }

                // 4 base edges around the far face
                for i in 0..4 {
                    let j = (i + 1) % 4;
                    edges.push(FrustumEdge {
                        endpoint_a: far[i],
                        _pad0: 0,
                        endpoint_b: far[j],
                        frustum_index: image_idx as u32,
                    });
                }

                // Pinhole image quad (instanced)
                if has_thumbnails {
                    pinhole_quads.push(ImageQuadInstance {
                        corner_tl: far[0],
                        frustum_index: image_idx as u32,
                        corner_tr: far[1],
                        _pad0: 0,
                        corner_bl: far[3],
                        _pad1: 0,
                        corner_br: far[2],
                        _pad2: 0,
                    });
                }
            }
        }

        // Upload frustum edges
        let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("frustum edges"),
            contents: bytemuck::cast_slice(&edges),
            usage: wgpu::BufferUsages::VERTEX,
        });

        // Create per-image color storage buffer (initialized to default white/alpha)
        let colors: Vec<u32> = vec![FRUSTUM_COLOR_DEFAULT; recon.images.len()];
        let color_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("frustum colors"),
            contents: bytemuck::cast_slice(&colors),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        let bundle = self.recons.get_mut(&id).expect("just ensured");
        bundle.frustum_edge_buffer = Some(buffer);
        bundle.frustum_edge_count = edges.len() as u32;
        bundle.frustum_image_count = recon.images.len() as u32;
        bundle.frustum_color_buffer = Some(color_buffer);

        // Upload pinhole image quads (instanced)
        if !pinhole_quads.is_empty() {
            let buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("image quad instances"),
                contents: bytemuck::cast_slice(&pinhole_quads),
                usage: wgpu::BufferUsages::VERTEX,
            });
            bundle.image_quad_instance_buffer = Some(buf);
            bundle.image_quad_count = pinhole_quads.len() as u32;
        } else {
            bundle.image_quad_instance_buffer = None;
            bundle.image_quad_count = 0;
        }

        // Upload distorted image quads (indexed)
        if !distorted_indices.is_empty() {
            let vbuf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("distorted quad vertices"),
                contents: bytemuck::cast_slice(&distorted_vertices),
                usage: wgpu::BufferUsages::VERTEX,
            });
            let ibuf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("distorted quad indices"),
                contents: bytemuck::cast_slice(&distorted_indices),
                usage: wgpu::BufferUsages::INDEX,
            });
            bundle.distorted_quad_vertex_buffer = Some(vbuf);
            bundle.distorted_quad_index_buffer = Some(ibuf);
            bundle.distorted_quad_index_count = distorted_indices.len() as u32;
        } else {
            bundle.distorted_quad_vertex_buffer = None;
            bundle.distorted_quad_index_buffer = None;
            bundle.distorted_quad_index_count = 0;
        }

        // Rebuild the bind groups that reference the new color buffer.
        self.rebuild_frustum_bind_group(device, id);
        // The image count moved, so the global pick index space is re-cut.
        self.assign_pick_bases();
    }

    /// Update per-image frustum colors without recomputing geometry.
    ///
    /// Writes a new color array to the existing storage buffer via `queue.write_buffer`.
    /// This is much cheaper than `upload_frustums` — just 4 bytes × image_count.
    ///
    /// Hidden images (e.g. the camera being viewed through) get alpha=0, which
    /// the shader discards so they don't render or participate in picking.
    pub fn update_frustum_colors(
        &self,
        queue: &wgpu::Queue,
        id: ReconId,
        image_count: usize,
        selected_image: Option<usize>,
        hidden_image: Option<usize>,
        track_images: &[usize],
    ) {
        // Indices stay local: they address the owning node's color buffer, not
        // the global pick space.
        let Some(color_buffer) = self
            .recons
            .get(&id)
            .and_then(|bundle| bundle.frustum_color_buffer.as_ref())
        else {
            return;
        };

        let colors = frustum_colors(image_count, selected_image, hidden_image, track_images);
        queue.write_buffer(color_buffer, 0, bytemuck::cast_slice(&colors));
    }

    /// Rebuild one node's bind groups that depend on its frustum color buffer.
    ///
    /// Called after the color buffer is created or replaced. Rebuilds:
    /// - Frustum wireframe bind group (global uniform + colors + recon uniforms)
    /// - Image quad bind group (per-recon atlas uniform + thumbnail texture +
    ///   shared sampler + colors + recon uniforms)
    pub(in crate::scene_renderer) fn rebuild_frustum_bind_group(
        &mut self,
        device: &wgpu::Device,
        id: ReconId,
    ) {
        // Shared pipeline resources and the node's own bundle are separate
        // fields, so both can be borrowed at once.
        let frustum_layout = self.frustum_bind_group_layout.as_ref();
        let frustum_uniforms = self.frustum_uniform_buffer.as_ref();
        let image_quad_layout = self.image_quad_bind_group_layout.as_ref();
        let sampler = self.image_quad_sampler.as_ref();
        let Some(bundle) = self.recons.get_mut(&id) else {
            return;
        };
        let Some(color_buf) = bundle.frustum_color_buffer.as_ref() else {
            return;
        };

        // Frustum wireframe bind group
        if let (Some(layout), Some(uniform_buf)) = (frustum_layout, frustum_uniforms) {
            bundle.frustum_bind_group =
                Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("frustum bind group"),
                    layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: uniform_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: color_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: bundle.uniform_buffer.as_entire_binding(),
                        },
                    ],
                }));
        }

        // Image quad bind group (shared by pinhole + distorted quad pipelines)
        if let (Some(layout), Some(uniform_buf), Some(tex_view), Some(sampler)) = (
            image_quad_layout,
            bundle.image_quad_uniform_buffer.as_ref(),
            bundle.thumbnail_view.as_ref(),
            sampler,
        ) {
            bundle.image_quad_bind_group =
                Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("image quad bind group"),
                    layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: uniform_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::TextureView(tex_view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: wgpu::BindingResource::Sampler(sampler),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: color_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 4,
                            resource: bundle.uniform_buffer.as_entire_binding(),
                        },
                    ],
                }));
        }
    }
}
