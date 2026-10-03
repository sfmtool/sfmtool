# Workgroup shared memory for the GPU Jacobi step (amendment)

**Status:** Draft

Amends [`../core/features/gpu-optical-flow.md`](../core/features/gpu-optical-flow.md),
whose § "Jacobi Kernel — the binding contract" describes the shipped shader and
points back here.

[`jacobi_step.wgsl`](../../crates/sfmtool-core/src/features/optical_flow/gpu/shaders/jacobi_step.wgsl)
runs one thread per pixel in 16×16 workgroups and reads every value it needs,
including the four neighbours of `flow_u`, `flow_v`, `du_old` and `dv_old`, from
global storage buffers. Each neighbour value is read by up to five threads in the
same workgroup.

## Proposal

Load an 18×18 tile of `flow_u`, `flow_v`, `du_old` and `dv_old` (the 16×16
workgroup plus a 1-pixel halo) into `var<workgroup>` arrays at the start of the
pass, synchronize with `workgroupBarrier()`, and read the neighbour stencil from
those arrays. The shader's early return for threads outside the image has to move
below the barrier, because every thread in a workgroup must reach it.
The coefficient arrays (`coefficients`, `b2_buf`) are read once per pixel and stay
in global memory. The bindings do not change, so the eight-buffer limit described
in the standing spec still holds.

The result must stay bit-identical to the current shader, since the CPU/GPU parity
tests in `gpu/tests.rs` compare against `jacobi_pixel_scalar_to_row`.

## Expected benefit and how to measure it

The benefit is expected to be moderate. The current shader is already 7–13x
faster than the CPU path on 960×960 and larger levels (see the per-level tables in
the standing spec), and GPU levels are 65 ms of the 164 ms fisheye 3840×3840
`high_quality` run, of which the Jacobi step is one part. Adopt the change only if
it reduces the GPU-levels batch time on the fisheye and Dino Dog Toy
`high_quality` runs by a measurable amount on more than one GPU.
