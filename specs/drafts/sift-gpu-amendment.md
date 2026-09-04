# GPU SIFT (amendment)

**Status:** Draft

Amends [`../core/features/sift.md`](../core/features/sift.md), which specifies
the shipped CPU SIFT implementation and points back here.

`sfmtool`'s own SIFT runs on the CPU: a scalar/SSE2/AVX2 kernel set parallelized
with rayon, with the DoG fused into detection and each Gaussian blur fused into
row stripes. That is fast enough that extraction over 1196 4K frames takes about
141 s on a 24-core laptop CPU, but it is still the largest single cost in a
`sfm sift --extract` run, and the machine already carries a GPU that the viewer
and the optical-flow kernels drive through `wgpu`. This draft proposes a GPU
backend for the same algorithm.

## What would move to the GPU

Compute shaders for the four data-parallel stages, reusing the existing `wgpu`
infrastructure ([`features/optical_flow/gpu/`](../../crates/sfmtool-core/src/features/optical_flow/gpu/mod.rs)
is the working precedent for a `wgpu` compute path in this crate, behind the
crate's `gpu` feature):

- **Separable Gaussian blur** — the scale-space chain, two passes per level.
- **Difference of Gaussians** — pointwise, and cheap enough to fuse into the
  extrema pass rather than materialize, exactly as the CPU path does.
- **Extrema detection** — the 26-neighbour test plus the contrast/edge
  rejections, with the surviving candidates compacted into an append buffer.
- **Descriptor** — the 4x4x8 trilinear-interpolated histogram, one workgroup per
  keypoint.

Subpixel localization and orientation assignment are the awkward stages: both are
iterative and sparse, so they may stay on the CPU in the first cut, with only the
dense stages offloaded.

## What has to be decided

- **Output parity.** The CPU path guarantees byte-identical `.sift` output across
  thread counts and SIMD tiers, and the cross-validation suite is built on that.
  A GPU path cannot promise bit-identity against the CPU one (different
  summation orders, different transcendental implementations), so the backend
  needs its own acceptance criterion — most likely the OpenCV-style tolerance
  band already used for cross-validation, applied against the CPU backend.
- **Where the backend is selected.** `sfm sift --extract` already chooses between
  `colmap`, `opencv` and `sfmtool` backends; a GPU path is either a fifth backend
  or a flag on the `sfmtool` one. The latter is nicer if parity is close, worse if
  the outputs differ enough that a workspace's `feature_tool_xxh128` must
  distinguish them — and it must, if they do.
- **Whether the CPU path's memory-traffic work carries over.** The stripe fusion
  that makes the CPU path fast is a cache-hierarchy argument; the GPU's
  constraints are different and the fusion boundaries would be redrawn.

The design would be written up as `specs/core/features/gpu-sift.md` and this
draft deleted once it ships.
