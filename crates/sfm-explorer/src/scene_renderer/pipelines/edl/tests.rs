// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The EDL pass's shading of a point pixel, mirrored on the CPU.
//!
//! The noop device runs no shader, so `edl.wgsl`'s shading lines are pinned
//! one by one and mirrored here in `f32`, the precision the shader computes in,
//! and the mirror is checked at known colours.

use super::super::super::gpu_types::{
    edl_rim_light, EdlUniforms, BG_COLOR, EDL_OPACITY, EDL_STRENGTH,
};
use super::super::tests::srgb_to_linear_reference;

const SHADER: &str = include_str!("../../../shaders/edl.wgsl");

// The WGSL constants, as the pinned lines declare them.
const BACKGROUND_COLOR: [f32; 3] = [0.012983, 0.012983, 0.016807];
const LUMINANCE_WEIGHTS: [f32; 3] = [0.2126, 0.7152, 0.0722];
const RIM_LIGHT: [f32; 3] = [0.1, 0.1, 0.1];
const RIM_LIGHT_FROM: f32 = 1.0;
const RIM_LIGHT_TO: f32 = 4.0;

fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn mix(a: f32, b: f32, t: f32) -> f32 {
    a * (1.0 - t) + b * t
}

fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// `rim_light_weight` in `edl.wgsl`.
fn rim_light_weight(rgb: [f32; 3]) -> f32 {
    let background = dot(BACKGROUND_COLOR, LUMINANCE_WEIGHTS);
    let luminance = dot(rgb, LUMINANCE_WEIGHTS);
    1.0 - smoothstep(
        RIM_LIGHT_FROM * background,
        RIM_LIGHT_TO * background,
        luminance,
    )
}

/// `edl_shade` in `edl.wgsl` over the plain background, with the strength
/// and opacity the viewer sets. `response` is (depth, empty).
fn edl_shade(premultiplied: [f32; 3], alpha: f32, response: [f32; 2]) -> [f32; 3] {
    edl_shade_at(premultiplied, alpha, response, edl_rim_light(false))
}

/// `edl_shade` in `edl.wgsl` at a `rim_light` uniform.
fn edl_shade_at(
    premultiplied: [f32; 3],
    alpha: f32,
    response: [f32; 2],
    rim_light: f32,
) -> [f32; 3] {
    let shade = mix(
        1.0,
        (-(response[0] + response[1]) * EDL_STRENGTH).exp(),
        EDL_OPACITY,
    );
    let silhouette = 1.0 - mix(1.0, (-response[1] * EDL_STRENGTH).exp(), EDL_OPACITY);
    let rim = rim_light * rim_light_weight(premultiplied.map(|c| c / alpha)) * silhouette;
    [0, 1, 2].map(|k| premultiplied[k] * shade + RIM_LIGHT[k] * (rim * alpha))
}

/// The shading every point had before the rim, at the viewer's strength and
/// opacity: the colour times the shade of the whole response.
fn darkened(premultiplied: [f32; 3], response: f32) -> [f32; 3] {
    let shade = mix(1.0, (-response * EDL_STRENGTH).exp(), EDL_OPACITY);
    premultiplied.map(|c| c * shade)
}

/// An sRGB colour in linear light.
fn linear(srgb: [u8; 3]) -> [f32; 3] {
    srgb.map(|b| srgb_to_linear_reference(f64::from(b) / 255.0) as f32)
}

/// A linear colour as the sRGB target writes it.
fn encoded(rgb: [f32; 3]) -> [u8; 3] {
    rgb.map(egui::ecolor::gamma_u8_from_linear_f32)
}

fn luminance(rgb: [f32; 3]) -> f32 {
    dot(rgb, LUMINANCE_WEIGHTS)
}

/// The shader's rim lines are the ones mirrored here, its background is the
/// colour the scene is cleared to, and the strength and opacity it reads are
/// the ones the viewer writes.
#[test]
fn the_edl_shader_declares_the_rim_this_module_mirrors() {
    let background = format!(
        "const BACKGROUND_COLOR: vec3<f32> = vec3<f32>({}, {}, {});",
        BG_COLOR.r, BG_COLOR.g, BG_COLOR.b
    );
    for line in [
        background.as_str(),
        "const BACKGROUND_COLOR: vec3<f32> = vec3<f32>(0.012983, 0.012983, 0.016807);",
        "const LUMINANCE_WEIGHTS: vec3<f32> = vec3<f32>(0.2126, 0.7152, 0.0722);",
        "const RIM_LIGHT: vec3<f32> = vec3<f32>(0.1, 0.1, 0.1);",
        "const RIM_LIGHT_FROM: f32 = 1.0;",
        "const RIM_LIGHT_TO: f32 = 4.0;",
        "return vec2<f32>(0.0, 1.0);",
        "return vec2<f32>(max(0.0, (center_depth - neighbor_depth) / edl.point_size), 0.0);",
        "let background = dot(BACKGROUND_COLOR, LUMINANCE_WEIGHTS);",
        "let luminance = dot(rgb, LUMINANCE_WEIGHTS);",
        "return 1.0 - smoothstep(RIM_LIGHT_FROM * background, RIM_LIGHT_TO * background, luminance);",
        "let shade = mix(1.0, exp(-(response.x + response.y) * edl.strength), edl.opacity);",
        "let silhouette = 1.0 - mix(1.0, exp(-response.y * edl.strength), edl.opacity);",
        "let rim = edl.rim_light * rim_light_weight(premultiplied / alpha) * silhouette;",
        "return premultiplied * shade + RIM_LIGHT * (rim * alpha);",
        "response += edl_response(center_depth, inner_depth);",
        "response += edl_response(center_depth, outer_depth) * 0.5;",
        "let shaded_rgb = edl_shade(color.rgb, alpha, response);",
        "let lit_rgb = shaded_rgb + vec3<f32>(supernova) * alpha;",
    ] {
        assert!(
            SHADER.lines().any(|l| l.trim() == line),
            "edl.wgsl does not declare `{line}`"
        );
    }
    let uniforms = include_str!("../../uniforms.rs");
    for line in ["strength: EDL_STRENGTH,", "opacity: EDL_OPACITY,"] {
        assert!(
            uniforms.lines().any(|l| l.trim() == line),
            "uniforms.rs does not write `{line}`"
        );
    }
    // The background the sRGB target shows.
    assert_eq!(encoded(BACKGROUND_COLOR), [30, 30, 35]);
}

/// A point brighter than the band, the selection's yellow, the hover's cyan,
/// the brush's red and the bull's greys and greens among them, is shaded
/// exactly as every point was before the rim: its colour times the shade, at
/// every mix of depth and empty response.
#[test]
fn a_point_brighter_than_the_band_is_darkened_exactly_as_before() {
    let colours = [
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 1.0],
        [1.0, 0.1, 0.1],
        linear([72, 74, 76]),
        linear([110, 150, 60]),
        linear([255, 255, 255]),
    ];
    for colour in colours {
        assert_eq!(rim_light_weight(colour), 0.0, "{colour:?}");
        for alpha in [1.0, 0.25] {
            let premultiplied = colour.map(|c| c * alpha);
            for depth in [0.0, 0.3, 1.0, 4.0] {
                for empty in [0.0, 0.25, 0.5, 1.0] {
                    assert_eq!(
                        edl_shade(premultiplied, alpha, [depth, empty]),
                        darkened(premultiplied, depth + empty),
                        "{colour:?} at alpha {alpha}, response ({depth}, {empty})"
                    );
                }
            }
        }
    }
}

/// A near-black point, drawn (13, 12, 9) inside, takes a light rim where its
/// neighbours are empty: its silhouette is lighter than the background, so the
/// dot stands out from it, but stays well below a halo's brightness. Inside,
/// with no response, it keeps its own colour.
#[test]
fn a_near_black_points_silhouette_lightens_into_a_faint_rim() {
    let colour = linear([13, 12, 9]);
    assert!(luminance(colour) < luminance(BACKGROUND_COLOR));
    assert_eq!(rim_light_weight(colour), 1.0);
    assert_eq!(edl_shade(colour, 1.0, [0.0, 0.0]), colour);
    // A pixel at the edge of a lone dot, with every neighbour empty, and one
    // further in, with half of them empty.
    let edge = encoded(edl_shade(colour, 1.0, [0.0, 1.0]));
    let inner = encoded(edl_shade(colour, 1.0, [0.0, 0.5]));
    for (name, rim) in [("edge", edge), ("inner", inner)] {
        assert!(
            rim.iter().all(|&c| (45..=80).contains(&c)),
            "the {name} rim is {rim:?}, against a background of (30, 30, 35)"
        );
    }
    assert!(edge[0] > inner[0], "{edge:?} against {inner:?}");
}

/// A dark point's rim comes from its empty neighbours alone: where a nearer
/// point is beside it, it is darkened as any point is, so a dark point beside
/// a nearer one is not given a light band along that one's edge.
#[test]
fn a_dark_point_beside_a_nearer_one_is_darkened_not_lightened() {
    let colour = linear([13, 12, 9]);
    for depth in [0.25, 1.0, 4.0] {
        assert_eq!(
            edl_shade(colour, 1.0, [depth, 0.0]),
            darkened(colour, depth),
            "response ({depth}, 0)"
        );
    }
}

/// The rim fades out across the band from the background's luminance to four
/// times it: a smoothstep, so the shaded edge changes by a small step for a
/// small change of colour and points of similar colour get similar rims,
/// where a hard threshold would make the edge jump by most of the rim's light.
#[test]
fn the_rim_fades_smoothly_across_the_band() {
    let background = luminance(BACKGROUND_COLOR);
    let grey = |y: f32| [y, y, y];
    assert_eq!(rim_light_weight(grey(background)), 1.0);
    assert_eq!(rim_light_weight(grey(4.0 * background)), 0.0);
    let middle = rim_light_weight(grey(2.5 * background));
    assert!((middle - 0.5).abs() < 1e-3, "{middle}");
    // The silhouette of a lone dot, across greys from black to well past the
    // band, in steps of 1/50 of the band's width.
    let step = 3.0 * background / 50.0;
    let mut previous = edl_shade(grey(0.0), 1.0, [0.0, 1.0])[0];
    for k in 1..=200 {
        let shaded = edl_shade(grey(k as f32 * step), 1.0, [0.0, 1.0])[0];
        assert!(
            (shaded - previous).abs() < 0.004,
            "the edge jumps from {previous} to {shaded} at grey {}",
            k as f32 * step
        );
        previous = shaded;
    }
}

/// A point the Track Table's filter dims, drawn at `DIM_ALPHA` and so
/// premultiplied by it, takes its rim at that alpha too: its whole shaded
/// colour is the full point's times the alpha, so the blend lets the same
/// share of the background through the rim as through the rest of the dot.
#[test]
fn a_dimmed_dark_points_rim_takes_its_alpha() {
    let alpha = crate::filter_3d::DIM_ALPHA;
    let colour = linear([13, 12, 9]);
    let full = edl_shade(colour, 1.0, [0.0, 1.0]);
    let dimmed = edl_shade(colour.map(|c| c * alpha), alpha, [0.0, 1.0]);
    for k in 0..3 {
        assert!(
            (dimmed[k] - full[k] * alpha).abs() < 1e-7,
            "{dimmed:?} against {full:?} at alpha {alpha}"
        );
    }
}

/// While the camera's photograph is behind the scene, the rim is off: the
/// pass cannot read the photograph, and a dark point over it is shaded exactly
/// as any point is, a crisp dark dot. Over the plain background it is on. The
/// uniform is written from the same test that decides whether Pass 0 draws the
/// photograph.
#[test]
fn the_rim_is_drawn_only_over_the_plain_background() {
    assert_eq!(edl_rim_light(false), 1.0);
    assert_eq!(edl_rim_light(true), 0.0);
    let colour = linear([13, 12, 9]);
    for alpha in [1.0, crate::filter_3d::DIM_ALPHA] {
        let premultiplied = colour.map(|c| c * alpha);
        for (depth, empty) in [(0.0, 1.0), (0.0, 0.5), (1.0, 0.5)] {
            assert_eq!(
                edl_shade_at(premultiplied, alpha, [depth, empty], edl_rim_light(true)),
                darkened(premultiplied, depth + empty),
                "alpha {alpha}, response ({depth}, {empty})"
            );
        }
    }
    let uniforms = include_str!("../../uniforms.rs");
    let line = "rim_light: edl_rim_light(self.draws_background_photograph(in_camera_view)),";
    assert!(
        uniforms.lines().any(|l| l.trim() == line),
        "uniforms.rs does not write `{line}`"
    );
    let render = include_str!("../../render.rs");
    let line = "let has_bg_image = self.draws_background_photograph(in_camera_view);";
    assert!(
        render.lines().any(|l| l.trim() == line),
        "render.rs does not decide Pass 0 by `{line}`"
    );
}

/// `rim_light` follows `time` in both declarations of the EDL uniforms, so the
/// shader reads it from the bytes the viewer writes it to, and the block stays
/// the 64 bytes the buffer is made with.
#[test]
fn the_edl_uniforms_carry_the_rim_light_where_the_shader_reads_it() {
    let fields: Vec<&str> = SHADER
        .lines()
        .skip_while(|l| l.trim() != "struct EdlUniforms {")
        .skip(1)
        .take_while(|l| l.trim() != "}")
        .map(|l| l.trim().split(':').next().unwrap_or(""))
        .collect();
    assert_eq!(fields.last(), Some(&"rim_light"), "{fields:?}");
    assert_eq!(fields[fields.len() - 2], "time", "{fields:?}");
    // screen_size, four scalars, target_view_pos, five scalars and time end
    // at byte 56, where rim_light sits; the block is padded to 64.
    assert_eq!(std::mem::offset_of!(EdlUniforms, time), 52);
    assert_eq!(std::mem::offset_of!(EdlUniforms, rim_light), 56);
    assert_eq!(std::mem::size_of::<EdlUniforms>(), 64);
}
