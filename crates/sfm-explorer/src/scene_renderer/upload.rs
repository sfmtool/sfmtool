// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Data upload logic — transfers point cloud, frustum, thumbnail, and background
//! image data to the GPU.

use super::auto_point_size::{
    compute_auto_point_size, compute_camera_nn_scale, compute_scene_bounds,
};
use super::distorted_mesh::generate_bg_distorted_mesh;
use super::gpu_types::*;
use super::SceneRenderer;
use sfmtool_core::camera::frustum::{compute_distorted_frustum_grid, compute_frustum_corners};
use sfmtool_core::SfmrReconstruction;
use wgpu::util::DeviceExt;

/// Length of a point-at-infinity track ray, as a multiple of the camera-cloud
/// extent — long enough to clearly head out past the scene toward infinity.
const INFINITY_RAY_SCENE_MULTIPLE: f64 = 2.0;

/// Bounding-box diagonal of the reconstruction's camera centers — a
/// characteristic scene scale, used to size rays toward points at infinity.
fn camera_cloud_extent(recon: &SfmrReconstruction) -> f64 {
    let mut iter = recon.images.iter().map(|im| im.camera_center().coords);
    let Some(first) = iter.next() else {
        return 0.0;
    };
    let (mut lo, mut hi) = (first, first);
    for c in iter {
        lo = lo.inf(&c);
        hi = hi.sup(&c);
    }
    (hi - lo).norm()
}

impl SceneRenderer {
    /// Upload point cloud data to the GPU.
    ///
    /// Converts positions from f64 to f32, packs colors into u32, and
    /// computes the auto point size from nearest-neighbor distances.
    pub fn upload_points(&mut self, device: &wgpu::Device, recon: &SfmrReconstruction) {
        let instances: Vec<PointInstance> = recon
            .points
            .iter()
            .map(|p| {
                // For an infinity point `position` holds a unit direction; the
                // shader detects it via alpha = 0 and transforms it with w = 0.
                let alpha: u32 = if p.is_at_infinity() { 0 } else { 255 };
                PointInstance {
                    position: [
                        p.position.x as f32,
                        p.position.y as f32,
                        p.position.z as f32,
                    ],
                    color: (p.color[0] as u32)
                        | ((p.color[1] as u32) << 8)
                        | ((p.color[2] as u32) << 16)
                        | (alpha << 24),
                }
            })
            .collect();

        // Compute auto point size from nearest-neighbor distances
        self.auto_point_size = compute_auto_point_size(&recon.points);

        // Compute characteristic inter-camera distance
        self.camera_nn_scale = compute_camera_nn_scale(&recon.images);

        // Compute scene bounding sphere for adaptive clip planes
        let (center, radius) = compute_scene_bounds(&recon.points);
        self.scene_center = center;
        self.scene_radius = radius;

        let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("point instances"),
            contents: bytemuck::cast_slice(&instances),
            usage: wgpu::BufferUsages::VERTEX,
        });

        self.instance_buffer = Some(buffer);
        self.point_count = instances.len() as u32;

        log::info!(
            "Uploaded {} points to GPU (auto point size: {:.4})",
            self.point_count,
            self.auto_point_size
        );
    }

    /// Upload camera frustum edge geometry to the GPU.
    ///
    /// Builds 8 edges per camera (4 side edges from apex to far corners + 4
    /// base edges around the far face). The stub depth is `length_scale *
    /// frustum_size_multiplier`.
    ///
    /// Colors are stored in a separate per-image storage buffer that can be
    /// updated cheaply via [`update_frustum_colors`] without recomputing geometry.
    /// Hidden cameras are handled by setting alpha=0 in the color buffer (the
    /// shader discards those fragments), so geometry includes all cameras.
    pub fn upload_frustums(
        &mut self,
        device: &wgpu::Device,
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

        let has_thumbnails = self.thumbnail_texture.is_some();

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
        self.frustum_edge_buffer = Some(buffer);
        self.frustum_edge_count = edges.len() as u32;
        self.frustum_image_count = recon.images.len() as u32;

        // Create per-image color storage buffer (initialized to default white/alpha)
        let color_default: u32 = 255 | (255 << 8) | (255 << 16) | (180 << 24);
        let colors: Vec<u32> = vec![color_default; recon.images.len()];
        let color_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("frustum colors"),
            contents: bytemuck::cast_slice(&colors),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        self.frustum_color_buffer = Some(color_buffer);

        // Rebuild bind group with uniform + color storage buffer
        self.rebuild_frustum_bind_group(device);

        // Upload pinhole image quads (instanced)
        if !pinhole_quads.is_empty() {
            let buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("image quad instances"),
                contents: bytemuck::cast_slice(&pinhole_quads),
                usage: wgpu::BufferUsages::VERTEX,
            });
            self.image_quad_instance_buffer = Some(buf);
            self.image_quad_count = pinhole_quads.len() as u32;
        } else {
            self.image_quad_instance_buffer = None;
            self.image_quad_count = 0;
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
            self.distorted_quad_vertex_buffer = Some(vbuf);
            self.distorted_quad_index_buffer = Some(ibuf);
            self.distorted_quad_index_count = distorted_indices.len() as u32;
        } else {
            self.distorted_quad_vertex_buffer = None;
            self.distorted_quad_index_buffer = None;
            self.distorted_quad_index_count = 0;
        }
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
        image_count: usize,
        selected_image: Option<usize>,
        hidden_image: Option<usize>,
        track_images: &[usize],
    ) {
        let Some(ref color_buffer) = self.frustum_color_buffer else {
            return;
        };

        let color_default: u32 = 255 | (255 << 8) | (255 << 16) | (180 << 24);
        let color_selected: u32 = (255 << 8) | (255 << 16) | (255 << 24);
        let color_track: u32 = 255 | (165 << 8) | (255 << 24); // orange
        let color_hidden: u32 = 0; // alpha=0 → shader discards

        let mut colors: Vec<u32> = vec![color_default; image_count];
        if let Some(idx) = selected_image {
            if idx < image_count {
                colors[idx] = color_selected;
            }
        }
        for &idx in track_images {
            if idx < image_count && selected_image != Some(idx) {
                colors[idx] = color_track;
            }
        }
        // Hidden must be applied last so it wins over selected/track
        if let Some(idx) = hidden_image {
            if idx < image_count {
                colors[idx] = color_hidden;
            }
        }
        queue.write_buffer(color_buffer, 0, bytemuck::cast_slice(&colors));
    }

    /// Rebuild bind groups that depend on the frustum color buffer.
    ///
    /// Called after the color buffer is created or replaced. Rebuilds:
    /// - Frustum wireframe bind group (uniform + color storage)
    /// - Image quad bind group (uniform + thumbnail texture + sampler + color storage)
    fn rebuild_frustum_bind_group(&mut self, device: &wgpu::Device) {
        let Some(color_buf) = &self.frustum_color_buffer else {
            return;
        };

        // Frustum wireframe bind group
        if let (Some(layout), Some(uniform_buf)) = (
            &self.frustum_bind_group_layout,
            &self.frustum_uniform_buffer,
        ) {
            self.frustum_bind_group = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
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
                ],
            }));
        }

        // Image quad bind group (shared by pinhole + distorted quad pipelines)
        if let (Some(layout), Some(uniform_buf), Some(tex_view), Some(sampler)) = (
            &self.image_quad_bind_group_layout,
            &self.image_quad_uniform_buffer,
            &self.image_quad_thumbnail_view,
            &self.image_quad_sampler,
        ) {
            self.image_quad_bind_group =
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
                    ],
                }));
        }
    }

    /// Upload embedded camera thumbnails into a GPU 2D texture atlas.
    ///
    /// Packs all 128×128 RGB thumbnails into a single large 2D texture arranged
    /// as a grid, avoiding the 256-layer limit of texture arrays. Also creates
    /// the image quad uniform buffer and bind group.
    pub fn upload_thumbnails(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        recon: &SfmrReconstruction,
    ) {
        let image_count = recon.images.len() as u32;
        if image_count == 0 {
            return;
        }

        // Compute atlas grid dimensions, respecting GPU texture size limits.
        // Images are packed into a 2D texture array: each layer ("page") holds a
        // cols×rows grid of thumbnails, and we add as many layers as needed.
        let max_texture_dim = device.limits().max_texture_dimension_2d;
        let max_array_layers = device.limits().max_texture_array_layers;
        let max_cells_per_axis = max_texture_dim / THUMBNAIL_SIZE;
        let cols = ((image_count as f32).sqrt().ceil() as u32)
            .min(MAX_ATLAS_COLS)
            .min(max_cells_per_axis);
        let rows_per_page = max_cells_per_axis;
        let images_per_page = cols * rows_per_page;
        let num_pages = image_count.div_ceil(images_per_page).min(max_array_layers);
        let max_images = images_per_page * num_pages;
        let image_count_clamped = image_count.min(max_images);
        if image_count_clamped < image_count {
            log::warn!(
                "GPU limits can only fit {image_count_clamped} of {image_count} thumbnails \
                 in {num_pages} atlas pages; extra thumbnails will not be displayed",
            );
        }
        // Shrink the last page's row count so the texture isn't larger than needed
        let total_rows = image_count_clamped.div_ceil(cols);
        let actual_rows_per_page = total_rows.min(rows_per_page);
        let atlas_width = cols * THUMBNAIL_SIZE;
        let atlas_height = actual_rows_per_page * THUMBNAIL_SIZE;

        self.atlas_cols = cols;
        self.atlas_rows = actual_rows_per_page;
        self.images_per_page = images_per_page;

        // Create 2D texture array atlas
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("thumbnail atlas"),
            size: wgpu::Extent3d {
                width: atlas_width,
                height: atlas_height,
                depth_or_array_layers: num_pages,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        // Upload each embedded thumbnail to its grid cell (RGB → RGBA)
        for i in 0..image_count_clamped as usize {
            let rgb_slice = recon.thumbnails_y_x_rgb.index_axis(ndarray::Axis(0), i);
            let mut rgba_data = Vec::with_capacity((THUMBNAIL_SIZE * THUMBNAIL_SIZE * 4) as usize);
            for pixel in rgb_slice.as_slice().unwrap().chunks_exact(3) {
                rgba_data.extend_from_slice(&[pixel[0], pixel[1], pixel[2], 255]);
            }

            let page = i as u32 / images_per_page;
            let idx_in_page = i as u32 % images_per_page;
            let col = idx_in_page % cols;
            let row = idx_in_page / cols;

            queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d {
                        x: col * THUMBNAIL_SIZE,
                        y: row * THUMBNAIL_SIZE,
                        z: page,
                    },
                    aspect: wgpu::TextureAspect::All,
                },
                &rgba_data,
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(THUMBNAIL_SIZE * 4),
                    rows_per_image: Some(THUMBNAIL_SIZE),
                },
                wgpu::Extent3d {
                    width: THUMBNAIL_SIZE,
                    height: THUMBNAIL_SIZE,
                    depth_or_array_layers: 1,
                },
            );
        }

        let texture_view = texture.create_view(&wgpu::TextureViewDescriptor {
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            ..Default::default()
        });

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("thumbnail sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        // Create dedicated uniform buffer for image quad atlas parameters
        let uniform_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("image quad uniforms"),
            contents: bytemuck::bytes_of(&ImageQuadUniforms {
                view_proj: [[0.0; 4]; 4],
                atlas_cols: cols,
                atlas_rows: actual_rows_per_page,
                images_per_page,
                _pad: 0,
            }),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // Store texture and sampler; bind group is created by
        // rebuild_frustum_bind_group after the color buffer exists.
        self.image_quad_thumbnail_view = Some(texture_view);
        self.image_quad_sampler = Some(sampler);

        self.image_quad_uniform_buffer = Some(uniform_buf);
        self.thumbnail_texture = Some(texture);
        log::info!(
            "Uploaded {} thumbnails as {}×{} × {} page(s) atlas ({}×{} grid per page)",
            image_count_clamped,
            atlas_width,
            atlas_height,
            num_pages,
            cols,
            actual_rows_per_page,
        );
    }

    /// Upload embedded patch surfels into a GPU instance buffer + texture atlas.
    ///
    /// Walks the per-point patch frame arrays, skipping points without a patch
    /// (all-zero `u` row), and packs each point's `(R, R, 4)` RGBA bitmap into a
    /// 2D texture array atlas with page-grid packing (mirroring the thumbnail
    /// atlas), so the patch count can exceed the GPU array-layer limit.
    ///
    /// v1 renders textured patches only: a reconstruction that carries patch
    /// frames but no bitmaps uploads nothing (flat-shaded fallback is deferred).
    pub fn upload_patches(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        recon: &SfmrReconstruction,
    ) {
        // The bind group below needs the patch pipeline's layout + uniform
        // buffer, which may not exist yet if no frame has been rendered.
        self.ensure_pipelines(device);

        // Reset so reloading a reconstruction without patches clears the old ones.
        self.patch_instance_buffer = None;
        self.patch_atlas_texture = None;
        self.patch_bind_group = None;
        self.patch_count = 0;

        let (Some(u_halfvecs), Some(v_halfvecs)) =
            (&recon.patch_u_halfvec_xyz, &recon.patch_v_halfvec_xyz)
        else {
            return;
        };
        let Some(bitmaps) = &recon.patch_bitmaps_y_x_rgba else {
            return;
        };
        // Tiles must be square and fit the GPU's 2D texture limit; on-disk files
        // are shape-verified, but an in-memory recon (e.g. built in Python) may
        // not be, so guard rather than trip a wgpu validation error.
        let resolution = bitmaps.shape()[1] as u32;
        let tile_cols = bitmaps.shape()[2] as u32;
        if resolution == 0 {
            return;
        }
        if tile_cols != resolution {
            log::warn!("patch bitmaps are non-square ({resolution}×{tile_cols}); skipping patches");
            return;
        }
        let max_texture_dim = device.limits().max_texture_dimension_2d;
        let max_array_layers = device.limits().max_texture_array_layers;
        if resolution > max_texture_dim {
            log::warn!(
                "patch bitmap resolution {resolution} exceeds the GPU texture limit \
                 {max_texture_dim}; skipping patches",
            );
            return;
        }

        // Collect the points that carry a patch: a point with no patch is an
        // all-zero `u` row. Bound the scan by every parallel array's length so a
        // short frame/bitmap array can't index out of range. The instance/atlas
        // buffers are compacted, so an instance's atlas slot is not its point
        // index.
        let n_rows = recon
            .points
            .len()
            .min(bitmaps.shape()[0])
            .min(u_halfvecs.nrows())
            .min(v_halfvecs.nrows());
        let point_indices: Vec<usize> = (0..n_rows)
            .filter(|&i| (0..3).any(|k| u_halfvecs[[i, k]] != 0.0))
            .collect();
        let patch_count = point_indices.len() as u32;
        if patch_count == 0 {
            return;
        }

        // Atlas grid dimensions: each layer ("page") holds a cols×rows grid of
        // patch tiles, respecting GPU texture size limits.
        let max_cells_per_axis = (max_texture_dim / resolution).max(1);
        let cols = ((patch_count as f32).sqrt().ceil() as u32).clamp(1, max_cells_per_axis);
        let rows_per_page = max_cells_per_axis;
        let patches_per_page = cols * rows_per_page;
        let num_pages = patch_count.div_ceil(patches_per_page).min(max_array_layers);
        let max_patches = patches_per_page * num_pages;
        let patch_count_clamped = patch_count.min(max_patches);
        if patch_count_clamped < patch_count {
            log::warn!(
                "GPU limits can only fit {patch_count_clamped} of {patch_count} patches \
                 in {num_pages} atlas pages; extra patches will not be displayed",
            );
        }
        // Shrink the last page's row count so the texture isn't larger than needed
        let total_rows = patch_count_clamped.div_ceil(cols);
        let actual_rows_per_page = total_rows.min(rows_per_page);
        let atlas_width = cols * resolution;
        let atlas_height = actual_rows_per_page * resolution;

        self.patch_atlas_cols = cols;
        self.patch_atlas_rows = actual_rows_per_page;
        self.patches_per_page = patches_per_page;

        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("patch atlas"),
            size: wgpu::Extent3d {
                width: atlas_width,
                height: atlas_height,
                depth_or_array_layers: num_pages,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        // Write each patch's RGBA tile into its atlas cell and build the
        // corresponding instance.
        let mut instances: Vec<PatchInstance> = Vec::with_capacity(patch_count_clamped as usize);
        for (slot, &i) in point_indices
            .iter()
            .enumerate()
            .take(patch_count_clamped as usize)
        {
            let tile = bitmaps.index_axis(ndarray::Axis(0), i);
            let page = slot as u32 / patches_per_page;
            let idx_in_page = slot as u32 % patches_per_page;
            let col = idx_in_page % cols;
            let row = idx_in_page / cols;

            queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d {
                        x: col * resolution,
                        y: row * resolution,
                        z: page,
                    },
                    aspect: wgpu::TextureAspect::All,
                },
                tile.as_slice().unwrap(),
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(resolution * 4),
                    rows_per_image: Some(resolution),
                },
                wgpu::Extent3d {
                    width: resolution,
                    height: resolution,
                    depth_or_array_layers: 1,
                },
            );

            let p = &recon.points[i];
            instances.push(PatchInstance {
                center: [
                    p.position.x as f32,
                    p.position.y as f32,
                    p.position.z as f32,
                ],
                w: p.w as f32,
                u_halfvec: [u_halfvecs[[i, 0]], u_halfvecs[[i, 1]], u_halfvecs[[i, 2]]],
                _pad0: 0.0,
                v_halfvec: [v_halfvecs[[i, 0]], v_halfvecs[[i, 1]], v_halfvecs[[i, 2]]],
                atlas_layer: slot as u32,
                point_index: i as u32,
            });
        }

        let instance_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("patch instances"),
            contents: bytemuck::cast_slice(&instances),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let texture_view = texture.create_view(&wgpu::TextureViewDescriptor {
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            ..Default::default()
        });
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("patch sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        if let (Some(layout), Some(uniform_buf)) =
            (&self.patch_bind_group_layout, &self.patch_uniform_buffer)
        {
            self.patch_bind_group = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("patch bind group"),
                layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: uniform_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&texture_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::Sampler(&sampler),
                    },
                ],
            }));
        }

        self.patch_instance_buffer = Some(instance_buffer);
        self.patch_atlas_texture = Some(texture);
        self.patch_count = patch_count_clamped;

        let atlas_bytes = atlas_width as u64 * atlas_height as u64 * 4 * num_pages as u64;
        log::info!(
            "Uploaded {} patches ({}×{} px) as {}×{} × {} page(s) atlas ({:.1} MiB)",
            patch_count_clamped,
            resolution,
            resolution,
            atlas_width,
            atlas_height,
            num_pages,
            atlas_bytes as f64 / (1024.0 * 1024.0),
        );
    }

    /// Load a full-resolution camera image for the background in camera view mode.
    ///
    /// Creates a single 2D texture at the image's native resolution and rebuilds
    /// the background image bind group. Skips reloading if the same image index
    /// is already loaded.
    pub fn upload_bg_image(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        recon: &SfmrReconstruction,
        image_index: usize,
    ) {
        if self.bg_image_loaded_index == Some(image_index) {
            return; // already loaded
        }

        let Some(img) = recon.images.get(image_index) else {
            return;
        };
        let image_path = recon.workspace_dir.join(&img.name);
        let dyn_image = match image::open(&image_path) {
            Ok(img) => img,
            Err(e) => {
                log::warn!("Failed to load bg image {}: {}", image_path.display(), e);
                return;
            }
        };

        let rgba = dyn_image.to_rgba8();
        let (w, h) = (rgba.width(), rgba.height());

        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("bg image"),
            size: wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &rgba,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(w * 4),
                rows_per_image: Some(h),
            },
            wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
        );

        let texture_view = texture.create_view(&Default::default());

        // Rebuild bind group with the new texture
        if let (Some(layout), Some(uniform_buf), Some(sampler)) = (
            &self.bg_image_bind_group_layout,
            &self.bg_image_uniform_buffer,
            &self.bg_image_sampler,
        ) {
            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("bg image bind group"),
                layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: uniform_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&texture_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::Sampler(sampler),
                    },
                ],
            });
            self.bg_image_bind_group = Some(bind_group);
        }

        self.bg_image_texture = Some(texture);
        self.bg_image_loaded_index = Some(image_index);
        log::info!("Loaded bg image {} ({}×{})", image_path.display(), w, h);

        // Generate tessellated mesh with world-space ray directions.
        // Uses the same camera-to-world rotation as frustum wireframes, so the
        // BG shader can use the standard view_proj = projection * view pipeline.
        let camera = &recon.cameras[img.camera_index as usize];
        let r = img.camera_to_world_rotation_flat();
        let subdivisions = if camera.model.is_fisheye() {
            BG_FISHEYE_SUBDIVISIONS
        } else if camera.has_distortion() {
            BG_DISTORTION_SUBDIVISIONS
        } else {
            BG_PINHOLE_SUBDIVISIONS
        };
        let (vertices, indices) = generate_bg_distorted_mesh(camera, &r, subdivisions);
        let vbuf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("bg distorted vertices"),
            contents: bytemuck::cast_slice(&vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let ibuf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("bg distorted indices"),
            contents: bytemuck::cast_slice(&indices),
            usage: wgpu::BufferUsages::INDEX,
        });
        self.bg_image_distorted_vertex_buffer = Some(vbuf);
        self.bg_image_distorted_index_buffer = Some(ibuf);
        self.bg_image_distorted_index_count = indices.len() as u32;
    }

    /// Upload track ray edge geometry for the selected point's observations.
    ///
    /// Each ray goes from the camera center along the true observation direction
    /// (unprojected from the SIFT feature position through camera intrinsics) to
    /// the nearest point on the ray to the 3D point. The gap between the ray
    /// endpoint and the 3D point visualizes reprojection error in 3D space.
    ///
    /// `sift_cache` is the shared SIFT position cache from `AppState`. Feature
    /// positions are looked up from this cache (the caller must ensure that
    /// relevant images have been cached via `ensure_sift_cached` before calling).
    pub fn upload_track_rays(
        &mut self,
        device: &wgpu::Device,
        recon: &SfmrReconstruction,
        point_idx: usize,
        sift_cache: &std::collections::HashMap<usize, crate::state::CachedSiftFeatures>,
    ) {
        let point = &recon.points[point_idx];
        let point_pos = point.position;
        let at_infinity = point.is_at_infinity();

        // A point at infinity has no finite location — its stored position is a
        // unit direction at the origin, which would project onto every forward
        // ray at t < 0 and collapse to a zero-length (invisible) ray. Instead,
        // shoot each ray outward along its own bearing to a fixed, scene-scaled
        // length (a multiple of the camera-cloud extent) so the bundle is
        // visible heading off toward infinity.
        let infinity_ray_length = if at_infinity {
            INFINITY_RAY_SCENE_MULTIPLE * camera_cloud_extent(recon)
        } else {
            0.0
        };

        // The observation keypoint lives in one of two places: SIFT feature
        // positions read from `.sift` companions (`sift_files`, indexed through
        // `feature_indexes` into the shared cache) or keypoints stored inline in
        // the reconstruction (`embedded_patches`). Both are photometrically
        // placed and need not point exactly at the 3D point, so the ray is
        // unprojected from whichever the reconstruction carries.
        let feature_indexes = recon.feature_indexes();
        let keypoints_xy = recon.keypoints_xy();
        let obs_start = recon.observation_offsets[point_idx];
        let observations = recon.observations_for_point(point_idx);
        let edges: Vec<EdgeInstance> = observations
            .iter()
            .enumerate()
            .filter_map(|(k, obs)| {
                let image = &recon.images[obs.image_index as usize];
                let camera = &recon.cameras[image.camera_index as usize];
                let center = image.camera_center();
                let endpoint_a = [center.x as f32, center.y as f32, center.z as f32];

                // The observed keypoint pixel for this observation, from the
                // SIFT feature or the inline embedded keypoint. Skip the
                // observation when neither source yields one (e.g. a missing or
                // truncated `.sift` file) rather than drawing a misleading ray.
                let obs_pixel: [f64; 2] = if let Some(fis) = feature_indexes {
                    let fi = fis[obs_start + k] as usize;
                    let cached = sift_cache.get(&(obs.image_index as usize))?;
                    let xy = cached.positions_xy.get(fi)?;
                    [xy[0] as f64, xy[1] as f64]
                } else {
                    let kxy = keypoints_xy?;
                    let row = obs_start + k;
                    [kxy[[row, 0]] as f64, kxy[[row, 1]] as f64]
                };

                // Unproject the keypoint to a camera-local unit ray, then rotate
                // to world space: d_world = R^T * d_cam.
                let d_cam = camera.pixel_to_ray(obs_pixel[0], obs_pixel[1]);
                let r_flat = image.camera_to_world_rotation_flat();
                let d_world = [
                    r_flat[0] * d_cam[0] + r_flat[1] * d_cam[1] + r_flat[2] * d_cam[2],
                    r_flat[3] * d_cam[0] + r_flat[4] * d_cam[1] + r_flat[5] * d_cam[2],
                    r_flat[6] * d_cam[0] + r_flat[7] * d_cam[1] + r_flat[8] * d_cam[2],
                ];

                let endpoint_b = if at_infinity {
                    // Point at infinity: shoot the ray outward along the
                    // observed bearing (a point at infinity has no parallax).
                    [
                        (center.x + infinity_ray_length * d_world[0]) as f32,
                        (center.y + infinity_ray_length * d_world[1]) as f32,
                        (center.z + infinity_ray_length * d_world[2]) as f32,
                    ]
                } else {
                    // Finite point: terminate at the nearest point on the
                    // observed ray (so reprojection error shows),
                    // t = dot(P - C, d_world) clamped to the forward direction.
                    let cp = [
                        point_pos.x - center.x,
                        point_pos.y - center.y,
                        point_pos.z - center.z,
                    ];
                    let t = (cp[0] * d_world[0] + cp[1] * d_world[1] + cp[2] * d_world[2]).max(0.0);
                    [
                        (center.x + t * d_world[0]) as f32,
                        (center.y + t * d_world[1]) as f32,
                        (center.z + t * d_world[2]) as f32,
                    ]
                };

                Some(EdgeInstance {
                    endpoint_a,
                    endpoint_b,
                })
            })
            .collect();

        if edges.is_empty() {
            self.track_ray_edge_buffer = None;
            self.track_ray_count = 0;
            return;
        }

        let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("track ray edges"),
            contents: bytemuck::cast_slice(&edges),
            usage: wgpu::BufferUsages::VERTEX,
        });
        self.track_ray_edge_buffer = Some(buffer);
        self.track_ray_count = edges.len() as u32;
    }

    /// Clear track ray geometry (no point selected).
    pub fn clear_track_rays(&mut self) {
        self.track_ray_edge_buffer = None;
        self.track_ray_count = 0;
    }

    /// Clear the background image when leaving camera view mode.
    pub fn clear_bg_image(&mut self) {
        self.bg_image_bind_group = None;
        self.bg_image_texture = None;
        self.bg_image_loaded_index = None;
        self.bg_image_distorted_vertex_buffer = None;
        self.bg_image_distorted_index_buffer = None;
        self.bg_image_distorted_index_count = 0;
    }
}
