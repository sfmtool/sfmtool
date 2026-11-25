// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Pipeline-creation tests against a headless `wgpu` device.
//!
//! Every pipeline in this module compiles a WGSL shader. wgpu-core runs naga's
//! full validation on the noop backend, so building them here is a real check
//! that the shaders parse and type-check — the one thing an edit to a `.wgsl`
//! file cannot otherwise fail on until the GUI is launched on a real GPU.

use super::super::gpu_types::{
    PointUniforms, ReconUniforms, GBUFFER_COLOR_FORMAT, GBUFFER_LINEAR_DEPTH_FORMAT,
    GBUFFER_PICK_FORMAT, HW_DEPTH_FORMAT,
};

fn device() -> (wgpu::Device, wgpu::Queue) {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::NOOP,
        backend_options: wgpu::BackendOptions {
            noop: wgpu::NoopBackendOptions::enabled(),
            ..Default::default()
        },
        ..wgpu::InstanceDescriptor::new_without_display_handle()
    });
    let adapter =
        pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
            .expect("noop adapter");
    pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default()))
        .expect("noop device")
}

#[test]
fn every_shader_compiles() {
    let (device, _queue) = device();
    super::points::create(&device);
    super::edl::create(&device);
    super::frustum::create(&device);
    let image_quad = super::image_quad::create(&device);
    super::patch::create(&device);
    super::target::create(&device);
    super::bench_track::create(&device);
    let bg_image = super::bg_image::create(&device);
    // These two reuse the bind-group layout of the pipeline they extend.
    super::bg_distorted::create(&device, &bg_image.bind_group_layout);
    super::distorted_quad::create(&device, &image_quad.bind_group_layout);
    device.poll(wgpu::PollType::Poll).expect("device poll");
}

#[test]
fn the_point_uniforms_struct_fits_the_buffer_the_pipeline_allocates() {
    // The buffer is sized from `size_of::<PointUniforms>()`, so a write of the
    // whole struct must land exactly. wgpu validates the bounds, which is what
    // catches a field added on the Rust side without the padding a uniform
    // buffer's 16-byte alignment demands.
    let (device, queue) = device();
    let resources = super::points::create(&device);
    queue.write_buffer(
        &resources.uniform_buffer,
        0,
        bytemuck::bytes_of(&PointUniforms {
            view_proj: [[0.0; 4]; 4],
            view: [[0.0; 4]; 4],
            camera_right: [1.0, 0.0, 0.0],
            _pad0: 0.0,
            camera_up: [0.0, 1.0, 0.0],
            selected_point_index: u32::MAX,
            hovered_point_index: u32::MAX,
            screen_width: 800.0,
            screen_height: 600.0,
            infinity_point_px: 3.0,
            _pad: [0.0; 4],
        }),
    );
    assert_eq!(
        std::mem::size_of::<PointUniforms>() % 16,
        0,
        "a WGSL uniform struct is rounded up to a 16-byte multiple; the Rust \
         side has to match or the tail of the buffer is garbage"
    );
    device.poll(wgpu::PollType::Poll).expect("device poll");
}

#[test]
fn the_recon_uniforms_struct_matches_its_wgsl_layout() {
    // Five shaders declare this block; all five must agree with the Rust
    // definition, and wgpu validates the write against the buffer size.
    let (device, queue) = device();
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("recon uniforms"),
        size: std::mem::size_of::<ReconUniforms>() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(
        &buffer,
        0,
        bytemuck::bytes_of(&ReconUniforms {
            model: [[0.0; 4]; 4],
            point_size: 1.0,
            point_pick_base: 0,
            image_pick_base: 0,
            pickable: 1,
            tint_color: [0.0; 4],
            show_infinity: 1.0,
            preview_s: 0.0,
            filter_mode: 0,
            points_as_patches: 0,
        }),
    );
    assert_eq!(std::mem::size_of::<ReconUniforms>() % 16, 0);
    // mat4 (64) + four scalars (16) + vec4 (16) + four scalars (16). The four
    // scalars before the vec4 are sized to satisfy *its* alignment; the four
    // after it fill the struct's 16-byte alignment exactly.
    assert_eq!(std::mem::size_of::<ReconUniforms>(), 112);
    device.poll(wgpu::PollType::Poll).expect("device poll");
}

#[test]
fn the_gbuffer_pipelines_match_the_textures_sizing_allocates() {
    // The structural tie between `gpu_types`' render-target constants, the
    // textures `scene_renderer::sizing` builds from them, and the pipelines
    // that declare their targets from them. wgpu checks a pipeline's target
    // formats and depth state against the pass it is set on, so binding all
    // five pass-1 pipelines inside a pass assembled the way `ensure_size` +
    // `render` assemble the real one is the whole contract, checked. Without
    // it, a format changed in one place and missed in another is a validation
    // error at the first frame on a real GPU — which no test here has.
    let (device, queue) = device();
    let errors = device.push_error_scope(wgpu::ErrorFilter::Validation);

    let attachment = |label, format, usage| {
        device
            .create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d {
                    width: 64,
                    height: 64,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage,
                view_formats: &[],
            })
            .create_view(&Default::default())
    };
    let target = wgpu::TextureUsages::RENDER_ATTACHMENT;
    let color = attachment("splat color", GBUFFER_COLOR_FORMAT, target);
    let linear_depth = attachment("linear depth", GBUFFER_LINEAR_DEPTH_FORMAT, target);
    let pick = attachment("pick buffer", GBUFFER_PICK_FORMAT, target);
    let hw_depth = attachment("hw depth", HW_DEPTH_FORMAT, target);

    let points = super::points::create(&device);
    let frustum = super::frustum::create(&device);
    let image_quad = super::image_quad::create(&device);
    let patch = super::patch::create(&device);
    let distorted_quad = super::distorted_quad::create(&device, &image_quad.bind_group_layout);

    let attach = |view| {
        Some(wgpu::RenderPassColorAttachment {
            view,
            resolve_target: None,
            ops: wgpu::Operations {
                load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                store: wgpu::StoreOp::Store,
            },
            depth_slice: None,
        })
    };
    let mut encoder = device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("gbuffer contract"),
            color_attachments: &[attach(&color), attach(&linear_depth), attach(&pick)],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &hw_depth,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(0.0), // reversed-Z: 0 = far
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            ..Default::default()
        });
        pass.set_pipeline(&points.pipeline);
        pass.set_pipeline(&points.preview_pipeline);
        pass.set_pipeline(&frustum.pipeline);
        pass.set_pipeline(&image_quad.pipeline);
        pass.set_pipeline(&patch.pipeline);
        pass.set_pipeline(&patch.preview_pipeline);
        pass.set_pipeline(&distorted_quad);
    }
    queue.submit([encoder.finish()]);

    let error = pollster::block_on(errors.pop());
    assert!(
        error.is_none(),
        "a pass-1 pipeline disagrees with the G-buffer `sizing` allocates: {error:?}"
    );
}

/// The value of `const <name>: u32 = <hex>u;` in a WGSL source.
fn wgsl_u32_const(source: &str, name: &str) -> u32 {
    let prefix = format!("const {name}: u32 = ");
    let line = source
        .lines()
        .find_map(|line| line.trim().strip_prefix(prefix.as_str()))
        .unwrap_or_else(|| panic!("common.wgsl declares no `{name}: u32`"));
    let literal = line
        .strip_suffix(';')
        .and_then(|l| l.strip_suffix('u'))
        .unwrap_or_else(|| panic!("`{name}` is not written as a `u` literal: {line}"));
    let hex = literal
        .strip_prefix("0x")
        .unwrap_or_else(|| panic!("`{name}` is not written in hex: {literal}"));
    u32::from_str_radix(hex, 16).expect("hex literal")
}

#[test]
fn the_wgsl_pick_tags_match_the_rust_ones() {
    // The scene shaders write these tags into the pick buffer and `picking.rs`
    // decodes them, so the two sides must agree bit for bit. They are written
    // once on each side, in `common.wgsl` and `picking.rs`.
    use super::super::picking::{PICK_TAG_FRUSTUM, PICK_TAG_NONE, PICK_TAG_POINT};
    let common = include_str!("../../shaders/common.wgsl");
    for (name, rust_value) in [
        ("PICK_TAG_NONE", PICK_TAG_NONE),
        ("PICK_TAG_FRUSTUM", PICK_TAG_FRUSTUM),
        ("PICK_TAG_POINT", PICK_TAG_POINT),
    ] {
        assert_eq!(
            wgsl_u32_const(common, name),
            rust_value,
            "{name} differs between common.wgsl and picking.rs"
        );
    }
}

#[test]
fn the_wgsl_mask_words_match_the_rust_ones() {
    // The mask writer puts these words in the per-point buffers and the point
    // and patch shaders branch on them, so the two sides must agree. They are
    // written once on each side, in `common.wgsl` and `upload/overlay.rs`.
    use super::super::upload::{
        MASK_ALIVE, MASK_DELETED, MASK_HAS_SURFEL, MASK_HIGHLIGHTED, MASK_PAINTED, MASK_REJECTED,
        MASK_UNDER_BRUSH,
    };
    let common = include_str!("../../shaders/common.wgsl");
    for (name, rust_value) in [
        ("MASK_DELETED", MASK_DELETED),
        ("MASK_ALIVE", MASK_ALIVE),
        ("MASK_HIGHLIGHTED", MASK_HIGHLIGHTED),
        ("MASK_PAINTED", MASK_PAINTED),
        ("MASK_UNDER_BRUSH", MASK_UNDER_BRUSH),
        ("MASK_HAS_SURFEL", MASK_HAS_SURFEL),
    ] {
        assert_eq!(
            wgsl_u32_const(common, name),
            rust_value,
            "{name} differs between common.wgsl and upload/overlay.rs"
        );
    }
    // The surfel bit sits above every word and apart from the filter's bit,
    // so a point with a surfel keeps its word and its filter state.
    assert_eq!(
        wgsl_u32_const(common, "MASK_WORD") & MASK_HAS_SURFEL,
        0,
        "MASK_WORD keeps the surfel bit"
    );
    assert_eq!(MASK_REJECTED & MASK_HAS_SURFEL, 0, "the two bits overlap");
}

#[test]
fn the_wgsl_filter_values_match_the_rust_ones() {
    // The Track Table's filter crosses the boundary as a bit of the mask word,
    // a mode in the node's uniforms, the vertex range of the dimmed draw and
    // the alpha the shaders draw it at; each is written once on each side.
    use super::super::gpu_types::{FILTER_DIM, FILTER_HIDE};
    use super::super::render::{DIMMED_DRAW, FIRST_DRAW};
    use super::super::upload::MASK_REJECTED;
    let common = include_str!("../../shaders/common.wgsl");
    for (name, rust_value) in [
        ("MASK_REJECTED", MASK_REJECTED),
        ("FILTER_DIM", FILTER_DIM),
        ("FILTER_HIDE", FILTER_HIDE),
    ] {
        assert_eq!(
            wgsl_u32_const(common, name),
            rust_value,
            "{name} differs between common.wgsl and the renderer"
        );
    }
    // The bit sits above every word, so a rejected point keeps its word.
    assert_eq!(
        wgsl_u32_const(common, "MASK_WORD") & MASK_REJECTED,
        0,
        "MASK_WORD keeps the filter bit"
    );
    // Both draws are the unit quad's four corners, each marked as its own
    // draw's, which the two shaders read at the same location.
    use super::super::gpu_types::{DIMMED_DRAW_LOCATION, SPLAT_CORNERS};
    for (first, dimmed) in FIRST_DRAW.zip(DIMMED_DRAW.clone()) {
        let (first, dimmed) = (
            SPLAT_CORNERS[first as usize],
            SPLAT_CORNERS[dimmed as usize],
        );
        assert_eq!(first.position, dimmed.position);
        assert_eq!((first.dimmed, dimmed.dimmed), (0, 1));
    }
    assert_eq!(DIMMED_DRAW.end as usize, SPLAT_CORNERS.len());
    let input = format!("@location({DIMMED_DRAW_LOCATION}) dimmed_draw: u32,");
    for (name, source) in [
        ("points.wgsl", include_str!("../../shaders/points.wgsl")),
        ("patch.wgsl", include_str!("../../shaders/patch.wgsl")),
    ] {
        assert!(
            source.lines().any(|l| l.trim() == input),
            "{name} does not declare `{input}`"
        );
    }
    let line = format!("const DIM_ALPHA: f32 = {};", crate::filter_3d::DIM_ALPHA);
    assert!(
        common.lines().any(|l| l.trim() == line),
        "common.wgsl does not declare `{line}`"
    );
}

#[test]
fn the_infinity_toggle_reaches_both_the_splats_and_the_surfels() {
    // "Points at ∞" is one per-recon uniform, `show_infinity`, and both passes
    // that draw a point at infinity have to read it: the splat in points.wgsl
    // and the surfel in patch.wgsl. A shader that skips it keeps drawing, and
    // answering picks for, points the toggle hides.
    for (name, source) in [
        ("points.wgsl", include_str!("../../shaders/points.wgsl")),
        ("patch.wgsl", include_str!("../../shaders/patch.wgsl")),
    ] {
        assert!(
            source
                .lines()
                .any(|l| l.trim() == "if recon.show_infinity == 0.0 || clip_c.w <= 0.0 {"),
            "{name} does not clip a point at infinity when `show_infinity` is 0"
        );
    }
}

#[test]
fn the_wgsl_preview_flags_and_failure_colour_match_the_rust_ones() {
    // The proposal job writes the flags into the preview buffers and the
    // failure colour is named once beside the tint palette; the point and
    // patch shaders branch on the one and draw the other, so both sides must
    // agree. Written once on each side, in `common.wgsl` and in
    // `camera_model_proposal/preview.rs` and `scene.rs`.
    use crate::camera_model_proposal::preview::{FLAG_LOSES, FLAG_MOVES, FLAG_STILL};
    let common = include_str!("../../shaders/common.wgsl");
    for (name, rust_value) in [
        ("PREVIEW_STILL", FLAG_STILL),
        ("PREVIEW_MOVES", FLAG_MOVES),
        ("PREVIEW_LOSES", FLAG_LOSES),
    ] {
        let line = format!("const {name}: f32 = {rust_value:.1};");
        assert!(
            common.lines().any(|l| l.trim() == line),
            "common.wgsl does not declare `{line}`"
        );
    }
    // The constant is the linear value of the sRGB colour, since the colour
    // target encodes what the shader writes.
    let line = common
        .lines()
        .find_map(|l| {
            l.trim()
                .strip_prefix("const FAILURE_COLOR: vec3<f32> = vec3<f32>(")
        })
        .expect("common.wgsl declares FAILURE_COLOR");
    let written: Vec<f64> = line
        .trim_end_matches(");")
        .split(',')
        .map(|v| v.trim().parse().expect("a number"))
        .collect();
    let linear = |c: u8| {
        let c = c as f64 / 255.0;
        if c <= 0.04045 {
            c / 12.92
        } else {
            ((c + 0.055) / 1.055).powf(2.4)
        }
    };
    let expected: Vec<f64> = crate::scene::FAILURE_RGB
        .iter()
        .map(|&c| linear(c))
        .collect();
    assert_eq!(written.len(), 3);
    for (w, e) in written.iter().zip(&expected) {
        assert!(
            (w - e).abs() < 1e-5,
            "FAILURE_COLOR is {written:?}, not {expected:?}"
        );
    }
}

#[test]
fn a_highlighted_point_with_a_surfel_shows_as_its_outline_and_not_its_dot() {
    // Under the Patches style a point with a surfel is drawn once, as the
    // surfel, whether or not it is selected, hovered or called out: the point
    // shader leaves its splat out with no exception for those states, and the
    // surfel draws them as an outline instead.
    let points = include_str!("../../shaders/points.wgsl");
    let line =
        "let as_surfel = recon.points_as_patches != 0u && (in.alive & MASK_HAS_SURFEL) != 0u;";
    assert!(
        points.lines().any(|l| l.trim() == line),
        "points.wgsl does not declare `{line}`"
    );
    // The outlined surfel is not culled as a back face but marked as one, and
    // from behind it keeps its outline band and discards the texels inside it.
    // The band is exempt from the edge cutoff's discard, so it traces the
    // whole square. Only a finite patch is ever back-facing.
    let patch = include_str!("../../shaders/patch.wgsl");
    for line in [
        "if in.w != 0.0 {",
        "if !is_outlined(in) {",
        "out.back_facing = 1u;",
        "let outline = (selected || hovered) && edge_px < outline_width_px();",
        "if !outline && (texel.a < uniforms.alpha_cutoff || in.back_facing != 0u) {",
        "let opacity = select(uniforms.patch_opacity * in.alpha, 1.0, outline);",
    ] {
        assert!(
            patch.lines().any(|l| l.trim() == line),
            "patch.wgsl does not declare `{line}`"
        );
    }
}

#[test]
fn the_hidden_outline_is_the_surfels_band_at_the_bench_figures_behind_opacity() {
    let patch = include_str!("../../shaders/patch.wgsl");
    for line in [
        // The surfel pipeline's own placement, for the outlined surfels alone,
        // through the same invariant position, so a fragment lands on the
        // depth its surfel wrote and the strict depth test drops it there.
        "@builtin(position) @invariant clip_pos: vec4<f32>,",
        "return place(in);",
        "return place_preview(in, preview, size_factor);",
        // The same band, in the same colours, at the opacity the bench figure
        // is drawn through the scene at.
        "if edge_distance_px(in.uv) >= outline_width_px() {",
        "return vec4<f32>(outline_color(in.mark, selected), BEHIND_OPACITY);",
        "rgb = outline_color(in.mark, selected);",
    ] {
        assert!(
            patch.lines().any(|l| l.trim() == line),
            "patch.wgsl does not declare `{line}`"
        );
    }
    // The opacity is written once, in common.wgsl, for both of its readers.
    let common = include_str!("../../shaders/common.wgsl");
    let line = "const BEHIND_OPACITY: f32 = 0.45;";
    assert!(
        common.lines().any(|l| l.trim() == line),
        "common.wgsl does not declare `{line}`"
    );
    let bench = include_str!("../../shaders/bench_track.wgsl");
    assert!(
        bench.contains("BEHIND_OPACITY *") && !bench.contains("const BEHIND_OPACITY"),
        "bench_track.wgsl does not read the shared BEHIND_OPACITY"
    );
}

/// An overlay segment's inset ends: an end clipped at the near plane is not
/// inset, a segment no longer on screen than its insets draws nothing, the cut
/// ends lie on the segment at the places that project there, and a cut end's
/// coverage falls off over a point as the sides' does while an end that is
/// not cut stays square.
#[test]
fn an_overlay_segment_is_inset_on_screen_except_at_a_clipped_end() {
    let bench = include_str!("../../shaders/bench_track.wgsl");
    for line in [
        "@location(8) inset: vec2<f32>,",
        "inset_a = 0.0;",
        "inset_b = 0.0;",
        "if edge_len <= inset_a + inset_b {",
        "let start = at_screen_fraction(clip_a, clip_b, inset_a / edge_len);",
        "let end = at_screen_fraction(clip_a, clip_b, 1.0 - inset_b / edge_len);",
        "let t = fraction * clip_a.w / ((1.0 - fraction) * clip_b.w + fraction * clip_a.w);",
        "let reach = select(select(0.0, aa_px, cut_a), select(0.0, aa_px, cut_b), is_b);",
        "select(NOT_CUT, -along, cut_a),",
        "select(NOT_CUT, along - drawn_len, cut_b),",
        "outside = max(abs(in.local.y) - half_px, max(in.past_ends.x, in.past_ends.y));",
    ] {
        assert!(
            bench.lines().any(|l| l.trim() == line),
            "bench_track.wgsl does not declare `{line}`"
        );
    }
    // Each end's inset is dropped in the branch that clips that end.
    let clip_a = bench
        .find("va = mix(va, vb, t);")
        .expect("end a is clipped");
    let clip_b = bench
        .find("vb = mix(va, vb, t);")
        .expect("end b is clipped");
    assert!(bench[clip_a..clip_b].contains("inset_a = 0.0;"));
    assert!(!bench[clip_a..clip_b].contains("inset_b = 0.0;"));
    assert!(bench[clip_b..].contains("inset_b = 0.0;"));
}

#[test]
fn the_hidden_outline_pipelines_draw_onto_the_edl_output_over_the_scenes_depth_read_only() {
    use super::super::gpu_types::{
        EDL_OUTPUT_FORMAT, GBUFFER_DEPTH_STATE, HIDDEN_OUTLINE_DEPTH_STATE,
    };
    // Under reversed-Z the scene keeps the greater depth, so the strict
    // reverse passes only where something nearer was drawn, and the pass
    // writes no depth of its own.
    assert_eq!(
        GBUFFER_DEPTH_STATE.depth_compare,
        Some(wgpu::CompareFunction::Greater)
    );
    assert_eq!(
        HIDDEN_OUTLINE_DEPTH_STATE.depth_compare,
        Some(wgpu::CompareFunction::Less)
    );
    assert_eq!(HIDDEN_OUTLINE_DEPTH_STATE.depth_write_enabled, Some(false));

    // Both pipelines set inside a pass assembled as `render_hidden_outlines`
    // assembles it: the EDL output as the one colour target, and the scene's
    // depth attached read-only. A pipeline that writes the G-buffer's other
    // targets, or depth, is a validation error here.
    let (device, queue) = device();
    let errors = device.push_error_scope(wgpu::ErrorFilter::Validation);
    let attachment = |label, format| {
        device
            .create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d {
                    width: 64,
                    height: 64,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                view_formats: &[],
            })
            .create_view(&Default::default())
    };
    let edl_output = attachment("edl output", EDL_OUTPUT_FORMAT);
    let hw_depth = attachment("hw depth", HW_DEPTH_FORMAT);
    let patch = super::patch::create(&device);
    let mut encoder = device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("hidden outline contract"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &edl_output,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                    store: wgpu::StoreOp::Store,
                },
                depth_slice: None,
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &hw_depth,
                depth_ops: None,
                stencil_ops: None,
            }),
            ..Default::default()
        });
        pass.set_pipeline(&patch.hidden_outline_pipeline);
        pass.set_pipeline(&patch.hidden_outline_preview_pipeline);
    }
    queue.submit([encoder.finish()]);
    let error = pollster::block_on(errors.pop());
    assert!(
        error.is_none(),
        "a hidden-outline pipeline disagrees with the pass it is drawn in: {error:?}"
    );
}

#[test]
fn the_splat_and_the_outline_share_the_highlight_colours_and_the_bench_stroke() {
    // The selection's yellow and the hover's cyan are written once, in
    // common.wgsl, and both shaders draw them by name.
    let common = include_str!("../../shaders/common.wgsl");
    for line in [
        "const SELECTED_COLOR: vec3<f32> = vec3<f32>(1.0, 1.0, 0.0);",
        "const HOVERED_COLOR: vec3<f32> = vec3<f32>(0.0, 1.0, 1.0);",
    ] {
        assert!(
            common.lines().any(|l| l.trim() == line),
            "common.wgsl does not declare `{line}`"
        );
    }
    for (name, source) in [
        ("points.wgsl", include_str!("../../shaders/points.wgsl")),
        ("patch.wgsl", include_str!("../../shaders/patch.wgsl")),
    ] {
        for colour in ["SELECTED_COLOR", "HOVERED_COLOR"] {
            assert!(source.contains(colour), "{name} does not draw {colour}");
        }
    }
    // The outline is as wide as the stroke the bench draws the patch being
    // edited with, so the two read as the same kind of mark: both in panel
    // points, the outline turned into the pixels its edge distance is measured
    // in by the frame's pixels per point, which the patch uniform block
    // carries as the bench pass's does.
    let patch = include_str!("../../shaders/patch.wgsl");
    for line in [
        format!(
            "const OUTLINE_WIDTH_POINTS: f32 = {};",
            2.0 * super::super::gpu_types::BENCH_LINE_HALF_WIDTH
        ),
        "return OUTLINE_WIDTH_POINTS * uniforms.pixels_per_point;".to_string(),
        "pixels_per_point: f32, // physical pixels to a panel point".to_string(),
    ] {
        assert!(
            patch.lines().any(|l| l.trim() == line),
            "patch.wgsl does not declare `{line}`"
        );
    }
    let bench = include_str!("../../shaders/bench_track.wgsl");
    assert!(bench.contains("let half_px = in.half_width * px;"));
    // The patch block's Rust side places `pixels_per_point` straight after
    // the camera position, where the WGSL declares it.
    assert_eq!(
        std::mem::offset_of!(super::super::gpu_types::PatchUniforms, pixels_per_point),
        std::mem::offset_of!(super::super::gpu_types::PatchUniforms, camera_pos) + 12
    );
}

/// The sRGB transfer function, `IEC 61966-2-1`, in `f64`: the reference the
/// shaders' decode is checked against.
fn srgb_to_linear_reference(encoded: f64) -> f64 {
    if encoded <= 0.04045 {
        encoded / 12.92
    } else {
        ((encoded + 0.055) / 1.055).powf(2.4)
    }
}

/// The scene shaders decode a packed sRGB byte into linear light with the
/// exact piecewise transfer function, so the `Rgba8UnormSrgb` target encodes
/// every byte back to itself: a stored colour is drawn as it is stored.
///
/// The noop device runs no shader, so the WGSL is pinned line by line and
/// mirrored here in `f32`, the precision the shader computes in.
#[test]
fn the_shaders_decode_a_packed_srgb_byte_to_the_light_the_target_encodes_back() {
    let common = include_str!("../../shaders/common.wgsl");
    for line in [
        "fn srgb_to_linear(encoded: vec3<f32>) -> vec3<f32> {",
        "let low = encoded / 12.92;",
        "let high = pow((encoded + 0.055) / 1.055, vec3<f32>(2.4));",
        "return select(high, low, encoded <= vec3<f32>(0.04045));",
        "fn unpack_srgb_rgba(packed: u32) -> vec4<f32> {",
        "let encoded = unpack4x8unorm(packed);",
        "return vec4<f32>(srgb_to_linear(encoded.rgb), encoded.a);",
    ] {
        assert!(
            common.lines().any(|l| l.trim() == line),
            "common.wgsl does not declare `{line}`"
        );
    }
    // The WGSL, in f32.
    let decode = |encoded: f32| -> f32 {
        let low = encoded / 12.92;
        let high = ((encoded + 0.055) / 1.055).powf(2.4);
        if encoded <= 0.04045 {
            low
        } else {
            high
        }
    };
    // Known values of the transfer function.
    for (encoded, linear) in [
        (0.0_f32, 0.0_f64),
        (0.04045, 0.0031308),
        (0.5, 0.2140411),
        (1.0, 1.0),
    ] {
        assert!(
            (f64::from(decode(encoded)) - linear).abs() < 1e-6,
            "{encoded} decodes to {}, not {linear}",
            decode(encoded)
        );
    }
    for byte in 0..=255u8 {
        let linear = decode(f32::from(byte) / 255.0);
        let reference = srgb_to_linear_reference(f64::from(byte) / 255.0);
        assert!(
            (f64::from(linear) - reference).abs() <= 1e-6 * reference.max(1e-3),
            "byte {byte}: {linear} against {reference}"
        );
        // What the sRGB target writes for it: the byte it came from.
        assert_eq!(
            egui::ecolor::gamma_u8_from_linear_f32(linear),
            byte,
            "byte {byte} does not survive the decode and the target's encode"
        );
    }
}

/// Every scene shader that unpacks a packed colour decodes it there: the
/// points, whose bytes are the stored colours, and the frustums, whose bytes
/// are the frustum and highlight colours. Neither divides a byte by 255
/// itself, which is what drew them encoded a second time, lighter and paler
/// than stored.
#[test]
fn the_point_and_frustum_shaders_decode_their_packed_colours() {
    for (name, source, line) in [
        (
            "points.wgsl",
            include_str!("../../shaders/points.wgsl"),
            "out.color = tinted(unpack_srgb_rgba(in.color_packed).rgb);",
        ),
        (
            "frustum.wgsl",
            include_str!("../../shaders/frustum.wgsl"),
            "let unpacked = unpack_srgb_rgba(frustum_colors[in.frustum_index]);",
        ),
    ] {
        assert!(
            source.lines().any(|l| l.trim() == line),
            "{name} does not declare `{line}`"
        );
        assert!(!source.contains("/ 255.0"), "{name} divides a byte by 255");
    }
}

/// The overlay pass decodes the pieces it draws onto the scene texture and
/// leaves the measurements' layer encoded: `vs_overlay` and `vs_disc` take
/// each colour through `unpack_srgb_rgba`, and `vs_measurement`, the entry the
/// measurements' pipeline is built on, draws egui's bytes as they are, so the
/// layer composites as egui paints.
#[test]
fn the_overlay_pass_decodes_on_the_scene_and_not_into_the_measurements_layer() {
    use super::bench_track::{MEASUREMENT_VERTEX_ENTRY, SCENE_VERTEX_ENTRY};
    let bench = include_str!("../../shaders/bench_track.wgsl");
    // The WGSL function `name`, up to the next one.
    let body = |name: &str| -> &str {
        let start = bench
            .find(&format!("fn {name}("))
            .unwrap_or_else(|| panic!("bench_track.wgsl has no fn {name}"));
        let rest = &bench[start..];
        let end = rest[1..].find("\nfn ").map_or(rest.len(), |e| e + 1);
        &rest[..end]
    };
    assert_eq!(SCENE_VERTEX_ENTRY, "vs_overlay");
    assert_eq!(MEASUREMENT_VERTEX_ENTRY, "vs_measurement");
    let scene_color = body("scene_color");
    assert!(scene_color.contains("let color = unpack_srgb_rgba(packed);"));
    assert!(scene_color.contains("return vec4<f32>(color.rgb * color.a, color.a);"));
    assert!(body(SCENE_VERTEX_ENTRY).contains("return overlay(in, scene_color(in.color));"));
    assert!(body("vs_disc").contains("let color = scene_color(in.color);"));
    let measurement = body(MEASUREMENT_VERTEX_ENTRY);
    assert!(measurement.contains("return overlay(in, unpack4x8unorm(in.color));"));
    assert!(
        !measurement.contains("srgb") && !measurement.contains("scene_color"),
        "the measurements' entry decodes: {measurement}"
    );
    // Premultiplied out of every entry, so the one blend serves both targets.
    assert_eq!(
        super::bench_track::BLEND,
        wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING
    );
    for line in [
        "return in.color * opacity;",
        "return in.color * (opacity * covered);",
        "@location(3) color: u32,",
        "@location(1) color: u32,",
    ] {
        assert!(
            bench.lines().any(|l| l.trim() == line),
            "bench_track.wgsl does not declare `{line}`"
        );
    }
}

/// An overlay instance packs its colour as a point's is packed, R in the low
/// byte and A in the high one, the order `unpack4x8unorm` reads, and is 60
/// bytes: two homogeneous places, the packed colour and five words.
#[test]
fn an_overlay_instance_packs_its_colour_r_first_into_one_word() {
    use super::super::gpu_types::{pack_rgba8, OverlayInstance};
    let rgba = [255, 92, 246, 153];
    let place = [0.0, 0.0, 0.0, 1.0];
    for piece in [
        OverlayInstance::segment(place, place, rgba, 1.0, OverlayInstance::HOLD),
        OverlayInstance::barb(place, place, 0.5, rgba, 1.0, OverlayInstance::FADE),
        OverlayInstance::ring(place, rgba, 2.0, 1.0, OverlayInstance::HOLD),
        OverlayInstance::disc(place, rgba, 2.0, OverlayInstance::HOLD),
    ] {
        assert_eq!(piece.color, 0x99F6_5CFF);
        assert_eq!(piece.color, pack_rgba8(rgba));
        // The bytes in the buffer, in the order the attribute reads them.
        let bytes = bytemuck::bytes_of(&piece);
        let at = std::mem::offset_of!(OverlayInstance, color);
        assert_eq!(bytes[at..at + 4], rgba);
    }
    assert_eq!(std::mem::size_of::<OverlayInstance>(), 60);
}

/// Each colour constant a shader writes for a named sRGB colour is that
/// colour's linear value, so the sRGB target shows the colour it names: the
/// line before it says `// sRGB (r, g, b)`, and the constant holds the decoded
/// triple.
#[test]
fn each_named_srgb_colour_constant_holds_its_linear_value() {
    let mut checked = 0;
    for (name, source) in [
        ("common.wgsl", include_str!("../../shaders/common.wgsl")),
        (
            "target_indicator.wgsl",
            include_str!("../../shaders/target_indicator.wgsl"),
        ),
    ] {
        let lines: Vec<&str> = source.lines().map(str::trim).collect();
        for (k, line) in lines.iter().enumerate() {
            let Some(named) = line
                .strip_prefix("// sRGB (")
                .and_then(|l| l.strip_suffix(')'))
            else {
                continue;
            };
            let bytes: Vec<u8> = named
                .split(',')
                .map(|v| v.trim().parse().expect("a byte"))
                .collect();
            let constant = lines[k + 1];
            let values: Vec<f64> = constant
                .split_once("= vec3<f32>(")
                .unwrap_or_else(|| panic!("{name}: no vec3 after `{line}`: {constant}"))
                .1
                .trim_end_matches(");")
                .split(',')
                .map(|v| v.trim().parse().expect("a number"))
                .collect();
            assert_eq!((bytes.len(), values.len()), (3, 3), "{name}: {line}");
            for (value, byte) in values.iter().zip(&bytes) {
                let expected = srgb_to_linear_reference(f64::from(*byte) / 255.0);
                assert!(
                    (value - expected).abs() < 1e-6,
                    "{name}: {constant} is not the linear value of {line}"
                );
            }
            checked += 1;
        }
    }
    // The brush's two reds, the failure colour, and the indicator's two.
    assert_eq!(checked, 5);
}
