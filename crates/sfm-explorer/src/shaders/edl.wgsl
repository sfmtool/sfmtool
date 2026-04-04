// Eye-Dome Lighting (EDL) post-process shader.
//
// Samples linear depth at 8 neighbors to detect depth discontinuities,
// then darkens edges to create a sense of surface and depth.

struct EdlUniforms {
    screen_size: vec2<f32>,
    radius: f32,
    strength: f32,
    opacity: f32,
    point_size: f32,
    target_view_pos: vec2<f32>,  // xy of target in view space
    target_view_z: f32,          // z of target in view space (positive = in front)
    target_active: f32,
    tan_half_fov: f32,
    aspect: f32,
    target_radius: f32,
    time: f32,
}

@group(0) @binding(0) var color_tex: texture_2d<f32>;
@group(0) @binding(1) var depth_tex: texture_2d<f32>;
@group(0) @binding(2) var tex_sampler: sampler;
@group(0) @binding(3) var<uniform> edl: EdlUniforms;

const NUM_NEIGHBORS: u32 = 8u;
const NEIGHBOR_DIRS: array<vec2<f32>, 8> = array<vec2<f32>, 8>(
    vec2<f32>( 1.0,  0.0),
    vec2<f32>( 0.0,  1.0),
    vec2<f32>(-1.0,  0.0),
    vec2<f32>( 0.0, -1.0),
    vec2<f32>( 0.707,  0.707),
    vec2<f32>(-0.707,  0.707),
    vec2<f32>(-0.707, -0.707),
    vec2<f32>( 0.707, -0.707),
);

struct FullscreenVertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn vs_fullscreen(@builtin(vertex_index) idx: u32) -> FullscreenVertexOutput {
    // Full-screen triangle covering the viewport (3 vertices, no buffer)
    let uv = vec2<f32>(f32((idx << 1u) & 2u), f32(idx & 2u));
    var out: FullscreenVertexOutput;
    out.position = vec4<f32>(uv * 2.0 - 1.0, 0.0, 1.0);
    out.uv = vec2<f32>(uv.x, 1.0 - uv.y); // flip Y for texture coords
    return out;
}

fn edl_response(center_depth: f32, neighbor_depth: f32) -> f32 {
    if neighbor_depth <= 0.0 {
        // No point at neighbor — treat as maximum edge (infinite distance).
        return 1.0;
    }
    // Linear depth difference normalized by point size, following the original
    // CloudCompare/Boucheny formulation: depth_diff / scale. Using point_size
    // as the scale makes the response dimensionless and scale-invariant.
    return max(0.0, (center_depth - neighbor_depth) / edl.point_size);
}

fn supernova_lighting(uv: vec2<f32>, frag_depth: f32) -> f32 {
    if edl.target_active <= 0.0 {
        return 0.0;
    }

    // Reconstruct fragment view-space position from UV + depth.
    // UV (0,0) is top-left, (1,1) is bottom-right.
    // NDC: x goes -1..1 left to right, y goes 1..-1 top to bottom.
    let ndc_x = uv.x * 2.0 - 1.0;
    let ndc_y = 1.0 - uv.y * 2.0;
    let frag_view = vec3<f32>(
        ndc_x * frag_depth * edl.aspect * edl.tan_half_fov,
        ndc_y * frag_depth * edl.tan_half_fov,
        -frag_depth,
    );

    // Target view-space position
    let target_view = vec3<f32>(
        edl.target_view_pos.x,
        edl.target_view_pos.y,
        -edl.target_view_z,
    );

    // 3D distance from fragment to target
    let diff = frag_view - target_view;
    let dist_3d = length(diff);

    // Inverse-square falloff envelope, with the target indicator radius as the
    // reference distance where intensity is 1.0.
    let r = edl.target_radius;
    let envelope = (r * r) / (dist_3d * dist_3d + r * r);
    if envelope < 0.005 {
        return 0.0;
    }

    // Radiating wave pulses: sin wave in distance-time space, sharpened
    // with pow() to create narrow spikes with wide gaps between them.
    let wave_speed = r * 3.0;       // waves travel 3× radius per second
    let wavelength = r * 2.0;       // one full wave cycle = 2× radius
    let phase = 6.283185 * (dist_3d / wavelength - edl.time * wave_speed / wavelength);
    let wave = pow(max(sin(phase), 0.0), 4.0);

    // Combine: envelope shapes the overall reach, wave adds the pulse structure
    let intensity = envelope * (0.3 + 0.7 * wave);

    return intensity * edl.target_active;
}

@fragment
fn fs_edl(in: FullscreenVertexOutput) -> @location(0) vec4<f32> {
    let color = textureSample(color_tex, tex_sampler, in.uv);
    let center_depth = textureSample(depth_tex, tex_sampler, in.uv).r;
    let alpha = color.a;

    // Nothing rendered here — discard to preserve whatever is in the render
    // target (background image in camera view, or BG_COLOR from clear).
    if alpha <= 0.0 {
        discard;
    }

    // Something was rendered (points, frustums, or both).
    // If we have linear depth, apply EDL shading. If not (frustum-only
    // pixels that don't write linear depth), show the color without EDL.
    //
    // Output premultiplied color + alpha — the pipeline's blend state
    // composites over the render target contents (background image or
    // BG_COLOR clear).
    if center_depth <= 0.0 {
        // Frustum or other geometry without linear depth — pass through
        // premultiplied color from Pass 1 without EDL shading.
        return vec4<f32>(color.rgb, alpha);
    }

    // Accumulate EDL response from neighbors at two radii for a thicker,
    // anti-aliased edge. The inner ring contributes full weight, the outer
    // ring contributes half weight for a smooth falloff.
    let pixel_size = vec2<f32>(1.0 / edl.screen_size.x, 1.0 / edl.screen_size.y);
    var response = 0.0;
    let inner_radius = edl.radius;
    let outer_radius = edl.radius + 1.5;

    for (var i = 0u; i < NUM_NEIGHBORS; i = i + 1u) {
        let inner_offset = NEIGHBOR_DIRS[i] * inner_radius * pixel_size;
        let inner_depth = textureSample(depth_tex, tex_sampler, in.uv + inner_offset).r;
        response += edl_response(center_depth, inner_depth);

        let outer_offset = NEIGHBOR_DIRS[i] * outer_radius * pixel_size;
        let outer_depth = textureSample(depth_tex, tex_sampler, in.uv + outer_offset).r;
        response += edl_response(center_depth, outer_depth) * 0.5;
    }

    response /= f32(NUM_NEIGHBORS) * 1.5; // normalize by total weight

    // Convert response to shade factor. The response is already in units of
    // point_size, so strength is the direct exponential decay rate.
    let shade = exp(-response * edl.strength);

    // Apply EDL shading, output premultiplied color + alpha.
    let shade_factor = mix(1.0, shade, edl.opacity);
    let supernova = supernova_lighting(in.uv, center_depth);

    // Supernova brightens all point pixels near the target, blowing out to white.
    let shaded_rgb = color.rgb * shade_factor;
    let lit_rgb = shaded_rgb + vec3<f32>(supernova);
    return vec4<f32>(lit_rgb * alpha, alpha);
}
