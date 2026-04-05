// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

use super::SceneRenderer;

impl SceneRenderer {
    /// Ensure the offscreen textures match the requested size.
    ///
    /// Also creates the render pipelines on first call.
    pub fn ensure_size(
        &mut self,
        device: &wgpu::Device,
        egui_renderer: &mut eframe::egui_wgpu::Renderer,
        width: u32,
        height: u32,
    ) {
        self.ensure_pipelines(device);

        let width = width.max(1);
        let height = height.max(1);

        if self.current_size == (width, height) && self.splat_color_view.is_some() {
            return;
        }

        // ── Pass 1 textures ──

        // Splat color (intermediate — read by EDL pass)
        let splat_color = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("splat color"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let splat_color_view = splat_color.create_view(&Default::default());

        // Linear depth (intermediate — read by EDL pass, also readable for HUD)
        let linear_depth = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("linear depth"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let linear_depth_view = linear_depth.create_view(&Default::default());

        // Staging buffers for 5x5 readback region (shared by hover + click).
        // 5 rows at 256-byte alignment per row.
        self.depth_staging = Some(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("depth staging 5x5"),
            size: 256 * 5,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        }));
        self.pick_staging = Some(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("pick staging 5x5"),
            size: 256 * 5,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        }));

        // Hardware depth (depth testing only)
        let hw_depth = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("hw depth"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let hw_depth_view = hw_depth.create_view(&Default::default());

        // Pick buffer (R32Uint — entity tag + index per pixel)
        let pick_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("pick buffer"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R32Uint,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let pick_texture_view = pick_texture.create_view(&Default::default());

        // ── Pass 2 textures ──

        // EDL output (final — registered with egui)
        //
        // The texture uses Rgba8UnormSrgb so the scene renderer's passes get
        // correct hardware sRGB conversion. A separate Rgba8Unorm view is
        // created for egui registration, because egui_wgpu expects non-sRGB
        // textures (it treats sampled values as gamma-space and does its own
        // color management in the shader).
        let edl_output = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("edl output"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[wgpu::TextureFormat::Rgba8Unorm],
        });
        let edl_output_view = edl_output.create_view(&Default::default());

        // Create a non-sRGB view of the same texture for egui. egui_wgpu
        // creates its own managed textures as Rgba8Unorm and its shader
        // assumes texture samples are already in gamma space (no hardware
        // sRGB decoding). Using Rgba8Unorm here prevents a double
        // linearization that would darken the image.
        let edl_output_egui_view = edl_output.create_view(&wgpu::TextureViewDescriptor {
            format: Some(wgpu::TextureFormat::Rgba8Unorm),
            ..Default::default()
        });

        // Register or update the EDL output texture with egui
        match self.egui_texture_id {
            Some(id) => {
                egui_renderer.update_egui_texture_from_wgpu_texture(
                    device,
                    &edl_output_egui_view,
                    wgpu::FilterMode::Linear,
                    id,
                );
            }
            None => {
                let id = egui_renderer.register_native_texture(
                    device,
                    &edl_output_egui_view,
                    wgpu::FilterMode::Linear,
                );
                self.egui_texture_id = Some(id);
            }
        }

        // Recreate EDL bind group (references the new textures)
        if let (Some(layout), Some(sampler), Some(uniform_buf)) = (
            &self.edl_bind_group_layout,
            &self.edl_sampler,
            &self.edl_uniform_buffer,
        ) {
            let edl_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("edl bind group"),
                layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&splat_color_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&linear_depth_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::Sampler(sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: uniform_buf.as_entire_binding(),
                    },
                ],
            });
            self.edl_bind_group = Some(edl_bind_group);
        }

        // Recreate target indicator bind group (references hw depth texture)
        if let (Some(layout), Some(uniform_buf)) =
            (&self.target_bind_group_layout, &self.target_uniform_buffer)
        {
            let target_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("target indicator bind group"),
                layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: uniform_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&hw_depth_view),
                    },
                ],
            });
            self.target_bind_group = Some(target_bind_group);
        }

        // Recreate track ray bind group (references hw depth texture)
        if let (Some(layout), Some(uniform_buf)) = (
            &self.track_ray_bind_group_layout,
            &self.track_ray_uniform_buffer,
        ) {
            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("track ray bind group"),
                layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: uniform_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&hw_depth_view),
                    },
                ],
            });
            self.track_ray_bind_group = Some(bind_group);
        }

        self.splat_color_view = Some(splat_color_view);
        self.linear_depth_view = Some(linear_depth_view);
        self.linear_depth_texture = Some(linear_depth);
        self.hw_depth_view = Some(hw_depth_view);
        self.pick_texture = Some(pick_texture);
        self.pick_texture_view = Some(pick_texture_view);
        self.edl_output_view = Some(edl_output_view);
        self.current_size = (width, height);

        log::debug!("Scene textures resized to {}x{}", width, height);
    }
}
