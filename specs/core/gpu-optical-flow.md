# GPU Compute Shaders for DIS Optical Flow

This document describes the wgpu compute shader implementation of the DIS optical
flow pipeline in `sfmtool-core`. The CPU implementation is documented in
[optical-flow.md](optical-flow.md); this spec covers the GPU-specific architecture.

## Performance

Measured on Laptop GPU, release build:

| Dataset | Preset | GPU | CPU | Speedup |
|---------|--------|-----|-----|---------|
| Seoul Bull 270×480 | default | 11ms | 10ms | ~1.0x |
| Seoul Bull 270×480 | high_quality | 15ms | 17ms | ~1.1x |
| Seattle Backyard 360×640 | default | 11ms | 10ms | ~1.0x |
| Seattle Backyard 360×640 | high_quality | 15ms | 24ms | **1.7x** |
| Dino Dog Toy 2040×1536 | default | 74ms | 72ms | ~1.0x |
| Dino Dog Toy 2040×1536 | high_quality | 59ms | 293ms | **5.0x** |
| Fisheye 3840×3840 | high_quality | 164ms | 1300ms | **7.9x** |

Default preset levels are all below the GPU threshold (~50K pixels), so GPU mode
effectively runs the same CPU code. Speedup scales with image size on the
high_quality preset, where finer pyramid levels have enough pixels to saturate
GPU parallelism.

## Pipeline Architecture

The DIS algorithm has 5 computational stages, running coarse-to-fine across
pyramid levels:

1. **Gaussian Pyramid** — separable 6-tap blur + 2x downsample
2. **Inverse Search** (DIS core) — per-patch gradient descent with precomputed Hessian
3. **Densification** — aggregate sparse patch flows to a dense field via photometric-error weighting
4. **Variational Refinement** — Jacobi iterations on intensity + gradient + smoothness energy
5. **Flow Upsampling** — bilinear 2x upsample between pyramid levels

The full GPU pipeline:

```
[CPU → GPU] Upload two full-resolution images (once)
[GPU] Build Gaussian pyramid (all levels)
[GPU → CPU] Read back one pyramid level (seed for CPU pyramid)
[CPU] Build CPU pyramid from seed (coarse levels only)
[CPU] Process coarse CPU levels (below gpu_min_pixels threshold)
[CPU → GPU] Upload transition flow (once, at CPU→GPU boundary)
[GPU] For each GPU-eligible level coarse-to-fine:
        Copy images from pyramid buffers → DIS buffers
        DIS inverse search → Densify → Variational refine → Upsample
[GPU] Final upsample(s) to full resolution (if finest_scale > 0)
[GPU → CPU] Read back final flow field at full resolution (once)
```

### Hybrid CPU/GPU Routing

The `gpu_min_pixels` field on `DisFlowParams` (default: 50,000, roughly 224×224)
controls per-level routing. Below the threshold, DIS and variational run on CPU;
above it, they run on GPU. The decision is per pyramid level in
`refine_flow_at_level` in `dis.rs`.

Per-level routing for fisheye 3840×3840 high_quality (7 pyramid levels):

| Level | Resolution | Pixels | Path | DIS | VAR |
|-------|-----------|--------|------|-----|-----|
| L7 | 30×30 | 900 | CPU | 0.6ms | 11.3ms |
| L6 | 60×60 | 3,600 | CPU | 1.2ms | 6.6ms |
| L5 | 120×120 | 14,400 | CPU | 2.7ms | 5.9ms |
| L4 | 240×240 | 57,600 | GPU | 4.7ms | 4.0ms |
| L3 | 480×480 | 230,400 | GPU | 5.2ms | 4.8ms |
| L2 | 960×960 | 921,600 | GPU | 11.3ms | 8.9ms |
| L1 | 1920×1920 | 3,686,400 | GPU | 38.7ms | 27.6ms |

### Minimizing CPU↔GPU Transfers

Several architectural decisions minimize data movement:

- **Merged DIS+variational submission.** Within a level, DIS and variational share
  a single GPU submission. Flow moves between them via `copy_buffer_to_buffer` (no
  CPU round-trip).

- **GPU inter-level upsampling.** Between consecutive GPU levels, `upsample_flow.wgsl`
  keeps flow on GPU. Only the transition from the last CPU level to the first GPU
  level uses a CPU upload.

- **GPU pyramid construction.** `blur_downsample.wgsl` builds the pyramid on GPU.
  Images are copied to DIS buffers via `copy_buffer_to_buffer` (no per-level CPU
  upload). A single pyramid level is read back to seed the CPU pyramid for coarse
  levels.

- **GPU final upsample.** After the last variational refinement, the flow stays on
  GPU for the final upsample(s) to full resolution. Readback happens at the final
  size.

- **Single-submission encoding.** All GPU levels (plus final upsample steps) are
  encoded into a single `CommandEncoder` with one `queue.submit()` + `device.poll(Wait)`.
  Per-level uniform buffers and bind groups are allocated upfront (`LevelBindGroups`).

- **Zero-copy resize.** `resize_flow_to` takes `FlowField` by value and returns it
  directly when dimensions already match (the common case after GPU final upsample).

### Persistent Buffer Pools

`GpuVariationalRefiner`, `GpuDisPipeline`, and `GpuPyramidPipeline` each maintain
persistent buffer pools using `Mutex<Option<Pool>>`. Buffers and bind groups are
allocated lazily on first use and grown as needed (never shrunk). After the first
frame pair, no further GPU allocations occur. Uniform parameter buffers use
`COPY_DST` usage for in-place updates via `queue.write_buffer()`.

## Compute Shaders

### 1. Gaussian Pyramid (`blur_downsample.wgsl`)

One compute dispatch per pass (horizontal, then vertical). Workgroups use shared
memory to load a tile + 3-pixel halo, then each thread writes one output pixel.
The 2x downsample is free — stride the output coordinates by 2. Two entry points
(`horiz` and `vert`) share the same shader module.

Kernel: 6-tap separable Gaussian `[0.017560, 0.129748, 0.352692, 0.352692, 0.129748, 0.017560]`
with between-pixel centering at offsets -2.5, -1.5, -0.5, +0.5, +1.5, +2.5.

### 2. Inverse Search (`inverse_search.wgsl`, `compute_gradients.wgsl`)

One thread per patch with a sequential loop over the 64 pixels (8×8 patch).
Dispatched 1D with 64-thread workgroups. Gradient computation runs as a
separate dispatch in the same compute pass.

Per-level GPU DIS cost:

| Resolution | GPU DIS | CPU DIS | Speedup |
|-----------|---------|---------|---------|
| 240×240 | 4.7ms | 7.8ms | **1.7x** |
| 480×480 | 5.2ms | 30.5ms | **5.9x** |
| 960×960 | 11.3ms | 124.7ms | **11.0x** |
| 1920×1920 | 38.7ms | 502.9ms | **13.0x** |

### 3. Densification (`densify.wgsl`)

Gather-based: one thread per output pixel. For each pixel, loops over all patches
that could cover it (bounded by patch size and stride). Dispatched 2D with 16×16
workgroups.

### 4. Variational Refinement

Four compute shaders:

1. `warp_by_flow.wgsl` — bilinear warp of target image by current flow
2. `precompute_coefficients.wgsl` — fused gradient computation + coefficient
   precomputation (packed as `vec4(a11, a12, a22, b1)` + separate `b2`)
3. `jacobi_step.wgsl` — double-buffered Jacobi iteration (ping-pong), 16×16
   workgroups, 7 inner iterations per outer iteration
4. `apply_flow_update.wgsl` — apply accumulated du/dv to the flow field

All outer iterations are encoded into a single command buffer with
`encoder.clear_buffer()` between iterations (no CPU–GPU sync until the end).

Per-level GPU variational cost:

| Resolution | GPU VAR | CPU VAR | Speedup |
|-----------|---------|---------|---------|
| 240×240 | 4.0ms | 13.4ms | **3.4x** |
| 480×480 | 4.8ms | 32.7ms | **6.8x** |
| 960×960 | 8.9ms | 74.3ms | **8.3x** |
| 1920×1920 | 27.6ms | 211.7ms | **7.7x** |

The CPU baseline is already well-optimized with rayon parallel rows + SSE2 SIMD
(4-wide) for the Jacobi inner loop.

### 5. Flow Upsampling (`upsample_flow.wgsl`)

One thread per output pixel, bilinear-samples the coarser flow field with clamped
boundary conditions, multiplies by 2.0. Dispatched 2D with 16×16 workgroups.
Used for both inter-level upsampling and the final upsample to full resolution.

## Textures vs Storage Buffers

Storage buffers (`array<f32>`) are used throughout for both images and mutable
intermediate data (flow fields, Jacobi ping-pong buffers, coefficients). The
asymmetric kernel centering in the pyramid shader makes explicit compute cleaner
than hardware texture samplers.

## Buffer Layout

| Buffer | Format | Size | Usage |
|--------|--------|------|-------|
| pyr_ref[L], pyr_tgt[L] | `storage<f32>` | W/2^L × H/2^L per level | GPU pyramid levels |
| pyr_intermediate | `storage<f32>` | max(W/2 × H) | Horizontal pass output |
| img_ref, img_tgt | `storage<f32>` | W×H per level | DIS input images |
| grad_x, grad_y | `storage<f32>` | W×H | Reference image gradients |
| flow_u, flow_v | `storage<f32>` | W×H | Current flow estimate |
| patch_results | `storage<vec3<f32>>` | N_patches | (flow_x, flow_y) per patch |
| a11, a12, a22, b1, b2 | `storage<f32>` | W×H each | Variational coefficients |
| du_0, dv_0, du_1, dv_1 | `storage<f32>` | W×H each | Jacobi ping-pong |
| warped + gradient images | `storage<f32>` | W×H each | Warped target + derivatives |

## GPU vs CPU Similarity

GPU and CPU produce functionally equivalent results.

| Preset | Condition | Endpoint Error |
|--------|-----------|----------------|
| fast (DIS only) | All frame gaps | **0.000 px** (bit-identical) |
| default (DIS + variational) | Gap 1 | 0.010 px (0.07% relative) |
| default | Gap 20 | 0.346 px (0.48% relative) |
| default | Gap 100 | 0.713 px (1.37% relative) |

The small variational differences arise from floating-point ordering differences
between GPU and CPU. These differences can cause DIS patches in later levels to
converge to different local minima in occluded regions, amplifying the error for
large frame gaps. In well-conditioned regions the error is negligible.

## Jacobi Kernel — WGSL Reference

Transliteration of `jacobi_pixel_scalar_to_row` from `variational.rs`:

```wgsl
@group(0) @binding(0) var<storage, read> du_old: array<f32>;
@group(0) @binding(1) var<storage, read> dv_old: array<f32>;
@group(0) @binding(2) var<storage, read_write> du_new: array<f32>;
@group(0) @binding(3) var<storage, read_write> dv_new: array<f32>;
@group(0) @binding(4) var<storage, read> flow_u: array<f32>;
@group(0) @binding(5) var<storage, read> flow_v: array<f32>;
@group(0) @binding(6) var<storage, read> a11: array<f32>;
@group(0) @binding(7) var<storage, read> a12: array<f32>;
@group(0) @binding(8) var<storage, read> a22: array<f32>;
@group(0) @binding(9) var<storage, read> b1: array<f32>;
@group(0) @binding(10) var<storage, read> b2: array<f32>;

struct Params { width: u32, height: u32, alpha: f32 }
@group(1) @binding(0) var<uniform> params: Params;

@compute @workgroup_size(16, 16)
fn jacobi_step(@builtin(global_invocation_id) gid: vec3<u32>) {
    let col = gid.x;
    let row = gid.y;
    if col >= params.width || row >= params.height { return; }
    let idx = row * params.width + col;
    let cu = flow_u[idx];
    let cv = flow_v[idx];

    // 4-neighbor Laplacian with boundary handling
    var lap_u = 0.0; var lap_v = 0.0; var nn = 0.0;
    if col > 0u {
        let n = idx - 1u;
        lap_u += flow_u[n] + du_old[n] - cu;
        lap_v += flow_v[n] + dv_old[n] - cv;
        nn += 1.0;
    }
    if col + 1u < params.width {
        let n = idx + 1u;
        lap_u += flow_u[n] + du_old[n] - cu;
        lap_v += flow_v[n] + dv_old[n] - cv;
        nn += 1.0;
    }
    if row > 0u {
        let n = idx - params.width;
        lap_u += flow_u[n] + du_old[n] - cu;
        lap_v += flow_v[n] + dv_old[n] - cv;
        nn += 1.0;
    }
    if row + 1u < params.height {
        let n = idx + params.width;
        lap_u += flow_u[n] + du_old[n] - cu;
        lap_v += flow_v[n] + dv_old[n] - cv;
        nn += 1.0;
    }

    let diag_u = a11[idx] + params.alpha * nn;
    let diag_v = a22[idx] + params.alpha * nn;

    du_new[idx] = select(
        du_old[idx],
        (b1[idx] + params.alpha * lap_u - a12[idx] * dv_old[idx]) / diag_u,
        abs(diag_u) > 1e-10
    );
    dv_new[idx] = select(
        dv_old[idx],
        (b2[idx] + params.alpha * lap_v - a12[idx] * du_old[idx]) / diag_v,
        abs(diag_v) > 1e-10
    );
}
```

## Timing Profiles

**Fisheye 3840×3840 high_quality** (164ms GPU, 1300ms CPU = 7.9x):

| Stage | GPU | CPU | Notes |
|-------|-----|-----|-------|
| Pyramid | 23ms | 97ms | GPU builds + seed readback |
| CPU levels (L7–L5) DIS | 4ms | 4ms | Below threshold, same code |
| CPU levels (L7–L5) VAR | 18ms | 18ms | L7 alone is 10ms (8 outer iters) |
| GPU levels (L4–L1) batch | 65ms | 1153ms | Single command buffer, one submit |
| Upsample/resize | ~2ms | 178ms | GPU final upsample, zero-copy return |

**Dino Dog Toy 2040×1536 high_quality** (59ms GPU, 293ms CPU = 5.0x):

| Stage | GPU | CPU | Notes |
|-------|-----|-----|-------|
| Pyramid | 8ms | 19ms | GPU builds + seed readback |
| CPU levels (L6–L3) DIS | 11ms | 11ms | Below threshold, same code |
| CPU levels (L6–L3) VAR | 16ms | 16ms | Same |
| GPU levels (L2–L1) batch | 18ms | 244ms | Single command buffer, one submit |
| Upsample/resize | ~1ms | 37ms | Transition upsample only |

Note: CPU variational on coarse levels is surprisingly expensive — the fisheye L7
(30×30) takes 10ms because `outer_iterations = 1 × (7+1) = 8`. This is 14% of
total GPU time and runs entirely on CPU.

## Future Work

**Workgroup shared memory for Jacobi.** The Jacobi shader reads 8 storage buffer
arrays per pixel per iteration. Caching `flow_u`, `flow_v`, `du_old`, `dv_old` in
workgroup shared memory with a 1-pixel halo (standard tiled-stencil pattern) could
reduce global memory traffic. Expected impact is moderate — the current shader
already achieves 7–13x speedup on large images.

## References

- `crates/sfmtool-core/src/optical_flow/` — Rust optical flow implementation
- `crates/sfmtool-core/src/optical_flow/gpu/` — GPU compute shaders and pipeline code
