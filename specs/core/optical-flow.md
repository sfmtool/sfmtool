# Optical Flow

A pure-Rust DIS (Dense Inverse Search) optical flow implementation in sfmtool-core,
with Python bindings via sfmtool-py and GPU acceleration via wgpu compute shaders.

## Motivation

Dense optical flow is useful as a candidate track generator for video-based SfM. A
Rust implementation (rather than wrapping OpenCV) gives us:

- Use in the Rust-only GUI for visualization and interactive features
- Control over the algorithm for SfM-specific optimizations
- Integration with the existing rayon-parallel matching pipeline
- GPU acceleration via wgpu compute shaders

## Algorithm: Dense Inverse Search (DIS)

Reference: Till Kroeger, Radu Timofte, Dengxin Dai, and Luc Van Gool, "Fast Optical Flow
using Dense Inverse Search," ECCV 2016. [arXiv:1603.03590](https://arxiv.org/abs/1603.03590)

### Overview

DIS is a coarse-to-fine patch-based method with three stages:

1. **Inverse search** for patch correspondences
2. **Dense displacement field** creation via patch aggregation across scales
3. **Variational refinement** (optional post-processing)

The "inverse search" trick avoids searching the target image for each reference patch
(expensive) by precomputing the gradient structure of the reference patch and computing
a closed-form update. This makes each patch update O(patch_size^2) rather than
O(search_area^2).

**Inverse search update rule** (Supplementary Section A, Eq. 12):

DIS uses the inverse compositional image alignment of Baker and Matthews (2001) to
avoid recomputing the Hessian at each iteration. For a template patch T centered at x
with current displacement estimate u, the update is:

    S' = nabla(T) . (dW/du)    (steepest descent images, from template gradients)
    H' = sum_x S'^T S'         (Hessian, precomputed once per patch)
    du = H'^{-1} sum_x S'^T . [I_{t+1}(W(x; u)) - T(x)]
    u <- u - du                (minus sign from inverse compositional formulation)

Because S' and H' depend only on the template (reference) patch gradients, they are
precomputed once and reused across all iterations. Only the image difference and the
bilinear extraction of the warped query patch are recomputed per iteration. Since the
warp is a simple translation, dW/du is the identity, so S' = nabla(T).

**Mean normalization** (Section 2.1): Both the template and warped query patch have
their mean intensity subtracted before computing the SSD, providing robustness against
illumination changes.

### Algorithm Steps

1. Build Gaussian image pyramids for both images (scale factor 2)
2. At the coarsest level, initialize flow to zero
3. At each pyramid level (coarsest to finest):
   a. Upsample flow from coarser level (x2 spatially, x2 magnitude)
   b. For each patch on a regular grid: inverse search via gradient descent
   c. Reject outlier patches: reset if displacement update exceeds patch size
   d. Densify sparse patch updates via photometric-error-weighted averaging (Eq. 3)
   e. Optionally apply variational refinement
4. Output the flow field at the finest level

### Parameters

| Symbol | Name | Description | Default |
|--------|------|-------------|---------|
| theta_ps | Patch size | Square patch edge length (pixels) | 8 |
| theta_ov | Patch overlap | Fraction of overlap between adjacent patches | 0.4 |
| theta_it | Iterations | Gradient descent iterations per patch | 12 |
| theta_sd | Scale factor | Downscaling factor between pyramid levels | 2 |
| theta_ss | Coarsest scale | Coarsest pyramid level (auto-computed) | image-dependent |
| theta_sf | Finest scale | Finest pyramid level (0 = full resolution) | theta_ss - 2 |

**Automatic coarsest scale:** `theta_ss = floor(log2(2 * width / (f * theta_ps)))`
where f = 5 (motions up to 1/5 of image width).

### Variational Refinement

The energy minimized is:

    E(U) = integral  delta * Psi(E_I) + gamma * Psi(E_G) + alpha * Psi(E_S) dx

where Psi(a^2) = sqrt(a^2 + epsilon^2) is a robust penalizer with epsilon = 0.001.

Two data terms (intensity constancy E_I and gradient constancy E_G) plus a smoothness
term E_S = ||nabla(u)||^2 + ||nabla(v)||^2.

**Solver:** Jacobi iteration (not the Gauss-Seidel SOR from the original paper). Jacobi
makes all pixels within an iteration independent, enabling SIMD, rayon parallelism, and
GPU compute shaders. It converges slower per iteration, so we use 7 inner iterations
(vs the paper's 5 for SOR).

| Symbol | Name | Default |
|--------|------|---------|
| delta | Intensity weight | 5 |
| gamma | Gradient weight | 10 |
| alpha | Smoothness weight | 10 |
| theta_vo | Outer iterations per scale | 1 x (s + 1), s = scale index |
| theta_vi | Inner Jacobi iterations | 7 |

### Presets

Three quality/speed presets correspond to the DIS paper's operating points:

| Preset | Patch | Overlap | Iterations | Finest Scale | Variational |
|--------|-------|---------|------------|--------------|-------------|
| `fast` | 8 | 0.3 | 16 | 3 (fixed) | disabled |
| `default_quality` | 8 | 0.4 | 12 | theta_ss - 2 | enabled |
| `high_quality` | 12 | 0.75 | 16 | 1 (fixed) | enabled |

## Architecture

### Module Structure

```
sfmtool-core/src/optical_flow/
├── mod.rs          # Public API: FlowField, compute_optical_flow(), presets
├── dis.rs          # DIS algorithm core
├── pyramid.rs      # Gaussian image pyramid
├── variational.rs  # Variational refinement
├── interp.rs       # Bilinear interpolation, image warping, flow densification
└── gpu/            # GPU compute shader implementation (wgpu)
    ├── mod.rs
    ├── dis_pipeline.rs
    ├── pyramid_pipeline.rs
    └── shaders/
        ├── apply_flow_update.wgsl
        ├── blur_downsample.wgsl
        ├── compute_gradients.wgsl
        ├── densify.wgsl
        ├── inverse_search.wgsl
        ├── jacobi_step.wgsl
        ├── precompute_coefficients.wgsl
        ├── upsample_flow.wgsl
        └── warp_by_flow.wgsl
```

### Key Types

**`FlowField`** stores per-pixel (dx, dy) displacements in split layout — separate
`data_u` and `data_v` arrays (both `Vec<f32>`, row-major). The split layout makes
contiguous SIMD loads possible. Methods include `get`/`set`, bilinear `sample` at
fractional coordinates, `advect_points` for batch keypoint advection, and
`upsample_2x`/`downsample_2x` for pyramid operations.

**`GrayImage`** is a simple f32 grayscale image (row-major, normalized to [0, 1]).
Coordinate convention: pixel centers at (0.5, 0.5), matching the .sfmr/.sift format
and COLMAP convention. Sampling at (0.5, 0.5) returns the exact top-left pixel value.

**`DisFlowParams`** holds all algorithm parameters, with constructor methods for each
preset (`fast`, `default_quality`, `high_quality`). Includes `gpu_min_pixels` (default
50,000) to control the per-level CPU/GPU routing threshold.

**`ImagePyramid`** builds a Gaussian pyramid via fused 6-tap blur + 2x downsample at
each level (sigma = 1.0, separable, even tap count for between-pixel centering).

### Internal Modules

**`dis.rs`** runs DIS at a single pyramid level: grid creation, patch inverse search
(parallelized via rayon), outlier rejection, densification, and optionally variational
refinement. Densification uses photometric-error-weighted averaging where each patch's
weight is `w = 1 / max(1, |intensity_difference|)`.

**`variational.rs`** minimizes the energy functional via Jacobi iteration with
double-buffered ping-pong. The Jacobi solver enables per-row parallelism (rayon) and
per-pixel SIMD (SSE2 4-wide for interior rows).

**`interp.rs`** provides bilinear sampling, image warping by flow fields, and flow
densification from sparse patch results.

### Public API

The two main entry points are `compute_optical_flow` (zero initialization) and
`compute_optical_flow_with_init` (starts from a provided flow estimate). Both accept
an optional `GpuFlowContext` for GPU acceleration — see
[gpu-optical-flow.md](gpu-optical-flow.md).

`compose_flow` composes two flow fields: `result(x) = flow_ab(x) + flow_bc(x + flow_ab(x))`,
used for chaining adjacent-frame flows into long-range estimates.

### Python Bindings

The Python module `sfmtool._sfmtool` exposes:

- `compute_optical_flow(img_a, img_b, preset, use_gpu)` — returns `(flow_u, flow_v)`
  as two `(H, W)` float32 arrays
- `compute_optical_flow_with_init(img_a, img_b, initial_flow_u, initial_flow_v, preset, use_gpu)` — same with initial flow
- `compose_flow(flow_ab_u, flow_ab_v, flow_bc_u, flow_bc_v)` — returns composed `(flow_u, flow_v)`
- `advect_points(points, flow_u, flow_v)` — advects `(N, 2)` float32 points through a flow field
- `gpu_available()` — returns whether GPU acceleration is available

Images accept `uint8` or `float32` numpy arrays. Flow fields use separate u/v arrays
(not interleaved), matching the internal split layout.

## Initial Flow Support

When computing flow between distant frames (e.g., 10-40 frames apart), large
displacements can exceed the coarse-to-fine pyramid's capture range. Adjacent-frame
flows are cheap and accurate, and can be chained via `compose_flow` to produce
approximate long-range estimates.

Given flow fields F_12 (frame 1 to 2) and F_23 (frame 2 to 3), the composed flow is:

    F_13(x) = F_12(x) + F_23(x + F_12(x))

Error accumulates as ~sqrt(N) * sigma for N chained frames with per-frame error sigma.
At sigma ~ 0.5px (default preset), 10 frames gives ~1.6px accumulated error — within
the DIS patch size, so the solver can refine from there.

`compute_optical_flow_with_init` downsamples the initial flow to the coarsest pyramid
level and starts refinement from there, rather than from zero. This lets the solver
compute the residual correction, converging faster for large displacements.

Chaining breaks down at occlusion boundaries (advected to garbage), out-of-frame points
(lost permanently), and systematic errors at texture boundaries (accumulate linearly).

## Performance

The implementation uses four layers of optimization:

**SIMD (SSE2).** The DIS inverse search inner loop processes 4 pixels per iteration
with vectorized bilinear sampling and Hessian accumulation. The Jacobi solver and
image gradient computation also have SSE2 paths for interior rows, with scalar
fallbacks for borders.

**Jacobi solver.** The variational refinement uses double-buffered Jacobi iteration
rather than the Gauss-Seidel SOR from the original paper. All pixels within an
iteration are independent, which is what makes SIMD, rayon, and GPU parallelism
possible. 7 inner iterations compensate for Jacobi's slower per-iteration convergence.

**Rayon parallelism.** Per-patch inverse search uses `par_iter` (each patch reads
shared images and writes its own result). Jacobi solver rows use `into_par_iter`
(each row writes only its own output slice, no cross-row dependencies thanks to
double-buffering).

**GPU compute shaders.** All five stages (pyramid, inverse search, densification,
variational refinement, flow upsampling) are mapped to wgpu compute shaders with
persistent buffer pools, merged submissions, and GPU-resident flow between levels.
See [gpu-optical-flow.md](gpu-optical-flow.md).

**Split FlowField layout.** `FlowField` uses separate `data_u`/`data_v` arrays rather
than interleaved storage. This makes flow component loads contiguous, benefiting SIMD
(4-wide loads), the Jacobi solver (operates on u and v independently), and the Python
bindings (return separate `(H, W)` arrays without reshaping).

## Benchmarks

Criterion benchmarks in `crates/sfmtool-core/benches/optical_flow.rs` on the test
datasets (dino_dog_toy 2040x1536, seattle_backyard 360x640):

| Group | What it measures |
|-------|-----------------|
| `optical_flow/end_to_end` | Full pipeline, fast + default presets |
| `optical_flow/pyramid` | Gaussian pyramid construction (8 levels) |
| `optical_flow/dis_refine` | Single-level DIS refinement at full resolution |
| `optical_flow/variational` | Variational refinement at full resolution |
| `optical_flow/bilinear` | 1M bilinear samples |

### Cross-Validation Against OpenCV

Python tests compare against OpenCV DIS on real image pairs:

| Test | Threshold | Worst actual |
|------|-----------|-------------|
| Seattle Backyard direction correlation (default) | > 0.95 | 0.989 |
| Seattle Backyard median EPE (default) | < 2.0 px | 1.25 |
| Fast preset direction correlation | > 0.7 | 0.838 |
| Seoul Bull direction correlation (default) | > 0.2 | 0.299 |
| Magnitude ratio (all pairs) | < 4.0 | 3.22 |

## Dependencies

No new crate dependencies. Uses `rayon` for parallelism, `image` for loading test
images in benchmarks, and `criterion` (dev-dependency) for benchmarks.
