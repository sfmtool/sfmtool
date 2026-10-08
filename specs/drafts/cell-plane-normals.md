# Cell Plane Normals

**Status:** Draft. Built: the kernel, its binding and a measurement against the two checked-in ground truths. Not decided: whether a consumer reads it, and with which gate. See [Open questions](#open-questions).

Amends: [cluster-patches-piecewise-refinement.md](cluster-patches-piecewise-refinement.md) § "From displacements to a normal", which this draft builds. Shares its determinacy verdict with [piece-gated-grid-normal.md](piece-gated-grid-normal.md). Measured in [cluster-patches-piecewise-refinement-measurements.md](cluster-patches-piecewise-refinement-measurements.md#cell-plane-normals-against-the-ground-truths-2026-10-08).

## Purpose

A cluster of a `.matches` file is a set of sightings of one small surface patch, one per image, refined against one of them, the reference member. The piecewise refinement cuts the patch into a three-by-three split of cells and stores, for every kept member, where each cell's content lies relative to where the member's affine shape places it. Once camera poses exist, each cell's location in each member is a pixel and so a ray, and the rays of one cell across the members meet near one world point. This kernel triangulates up to nine such points per cluster, fits a plane through them, and returns the plane's normal as the patch normal, with a verdict saying which of the normal's two axes the cells fix. It reads no image, so a seed stage that has poses and the cluster file but renders nothing can call it.

## Rust API

The kernel is [`cell_plane_normals`](../../crates/sfmtool-core/src/patch/cell_plane_normals.rs) in `sfmtool_core::patch::cell_plane_normals`, bound as `sfmtool._sfmtool.analysis.cell_plane_normals` in [cell_plane_normals.rs](../../crates/sfmtool-py/src/analysis/cell_plane_normals.rs).

```rust
pub struct CellPlaneClusters<'a> {
    pub cluster_starts: &'a [u32],                 // C + 1
    pub reference_members: &'a [u32],              // C, NO_REFERENCE for none
    pub member_images: &'a [u32],                  // K
    pub member_status: &'a [ClusterMemberStatus],  // K
    pub member_positions: &'a [[f32; 2]],          // K
    pub member_shapes: &'a [[[f32; 2]; 2]],        // K
    pub cell_shift_px: &'a [[[f32; 2]; 9]],        // K, row-major cells
    pub cell_status: &'a [[ClusterCellStatus; 9]], // K
    pub patch_size: f64,                           // refine_options.patch_size
    pub resolution: u32,                           // refine_options.resolution
}

pub struct CellPlaneCameras<'a> {
    pub cameras: &'a [CameraIntrinsics],
    pub image_camera: &'a [u32],                   // per image
    pub cam_from_world: &'a [Option<RigidTransform>], // per image, None = unposed
}

pub struct CellPlaneParams {
    pub include_refused_outlier: bool,   // false
    pub min_rays: usize,                 // 2
    pub min_triangulation_angle_deg: f64, // 2.0
    pub ray_noise_floor_px: f64,         // 0.05
    pub irls_iters: u32,                 // 3
    pub tukey_c: f64,                    // 4.685
    pub min_cells: usize,                // 3
    pub det_aniso: f64,                  // 0.10
}

pub enum NormalDeterminacy {
    BothAxes,
    OneAxis { free_axis: [f64; 3] },
    None,
}

pub struct CellPlaneNormal {
    pub normal: [f64; 3],                // NaN under None
    pub determinacy: NormalDeterminacy,
    pub view_dir: [f64; 3],
    pub cell_positions: [[f64; 3]; 9],   // NaN where not triangulated
    pub cell_rays: [u32; 9],
    pub cell_status: [CellPlaneStatus; 9],
    pub cell_weight: [f64; 9],
    pub cell_residual_px: [f64; 9],
    pub plane_rms: f64,
    pub anisotropy: f64,
    pub n_eff: f64,
}

pub fn cell_plane_normals(
    clusters: &CellPlaneClusters<'_>,
    cameras: &CellPlaneCameras<'_>,
    params: &CellPlaneParams,
) -> Vec<CellPlaneNormal>;
```

**Why this shape.** The inputs are the cluster file's member-parallel arrays borrowed as they are, so a caller holding a `MatchesData` or the Python `MatchesFile` passes its columns without building per-cluster objects, and the cameras are the per-camera intrinsics, per-image camera index and per-image `cam_from_world` pose that the bundle adjustment and resection take. An image without a pose is `None`, so a partly posed seed passes the file's whole image list. The output keeps every cell's position, ray count and status, because a wrong normal is diagnosed by asking which cells were used and how each was seen.

**Why `patch/`.** The kernel is a reading of the cluster-patches file's cell geometry: the grid split `[0, R/3, R − R/3, R]` and the map from a grid coordinate to a member's pixel. Both are defined in `patch/`, where `grid_bounds` is visible to the module without being widened. The binding is registered in the `analysis` submodule beside `estimate_adjacency_surfel_normals`, the other normal estimator a caller with poses chooses between.

**Example.** The measurement script reads a `--piecewise` cluster file and a ground-truth `.sfmr` and calls

```python
out = analysis.cell_plane_normals(
    mf.cluster_starts, mf.reference_members, mf.member_images, mf.member_status,
    mf.member_positions(), mf.member_affine_shapes(),
    mf.member_cell_shift_px, mf.member_cell_status,
    mf.refine_options["patch_size"], mf.refine_options["resolution"],
    list(gt.cameras), image_camera, quaternions_wxyz, translations)
```

and reads `out["normal"]`, `out["determinacy"]` (codes into `out["determinacy_names"]`) and `out["free_axis"]`.

## Theory

### A cell's rays

A member's grid coordinate `u`, centred on the patch grid, lies at pixel `p + (patch_size / R) · S · u`, where `p` is the member's stored position and `S` its stored shape; this is the refinement's sampling map. Cell `c`'s centre is `u = c`, and its content in a kept member lies at `u = c + d` for the stored displacement `d`. The reference member is the template, so its cell lies at `c` by construction and it contributes a ray with no displacement. A kept member contributes when its image is posed and its cell status is `fitted`; `refused_outlier` cells, whose displacement passed the ZNCC and curvature bars but which the member's affine fit gave no weight, enter on request. Each pixel is turned into a world ray through the camera model's `pixel_to_ray` and the pose.

### Triangulating a cell

The cell's position is the least-squares nearest point of its rays, `x = (Σ Pᵢ)⁻¹ Σ Pᵢ oᵢ` with `Pᵢ = I − dᵢdᵢᵀ`. A cell is not triangulated with fewer than `min_rays` rays (the reference's counts), when no pair of its rays subtends `min_triangulation_angle_deg`, or when the point lies behind a ray's camera.

Its noise is read in pixels: each ray's perpendicular distance to `x`, divided by the depth along the ray and multiplied by the camera's focal length, is that ray's residual. With `n` rays the point has `2n − 3` degrees of freedom left, so the cell's residual is `σ = √(Σ eᵢ² / (2n − 3))`, floored at `ray_noise_floor_px`. The position's covariance is `σ² Λ⁻¹` with `Λ = Σ (fᵢ/ρᵢ)² Pᵢ`, the information of `n` rays of angular noise `σ/f` at depths `ρᵢ`. More rays, a wider baseline and a smaller residual all shrink it.

### The plane

The plane through the positioned cells is the weighted least-variance plane: weighted centroid, weighted scatter, and the eigenvector of its smallest eigenvalue. A cell's weight is the inverse of its position's variance along the current normal, `1 / (nᵀ Σ n)`, times its Tukey weight. The Tukey loop runs `irls_iters` passes on the standardised residuals `zⱼ = |(xⱼ − x̄) · n| / √(nᵀ Σⱼ n)` with robust scale `max(1.4826 · median z, 1)` and cut-off `tukey_c` scales. The scale's floor is one, because the residuals are already in units of each cell's own modelled noise, and a smaller scale would trust the cells more than their rays allow. The normal is turned to face the mean direction from the cells toward the cameras of their rays.

### Which axes the cells fix

A cell is live when its weight is at least a quarter of the cluster's largest. With fewer than `min_cells` live cells, or no spread among them, the verdict is none and the normal is `NaN`; the kernel never substitutes a prior for a normal it did not measure. Otherwise the live cells' scatter has eigenvalues `λ₀ ≤ λ₁ ≤ λ₂`, and the anisotropy `λ₁ / λ₂` says whether they spread in two directions. At `det_aniso` or above both axes are fixed. Below it the cells lie on a line along the eigenvector of `λ₂`: the normal must be perpendicular to that line, but its rotation about the line is free. The kernel then reports the line as `free_axis` and returns the normal perpendicular to it that is nearest the mean viewing direction, so the free rotation stays at that prior. A three-by-three grid of cells reads an anisotropy near one, one row of cells near zero, and two rows `3/8`.

This is the verdict [piece-gated-grid-normal.md](piece-gated-grid-normal.md) describes for its piece estimator. The [adjacency surfel normals](../core/analysis/adjacency-surfel-normals.md) report a boolean `determined` from the same kind of diagnostics (effective support, anisotropy), not a free axis, so the three-way `NormalDeterminacy` is new here and is the type the piece-gated estimator would share.

## Implementation notes

- The cell centres are those of the piecewise refinement's `CellLayout`, computed from the same `grid_bounds`. Its displacements are in grid px, so the kernel needs the file's `patch_size` and `resolution`, never the image's pixels.
- The geometry is `f64` throughout; the stored positions, shapes and displacements are `f32` and are widened on read.
- The determinacy reads the live cells' spread with equal weights: the verdict asks where the cells are, not how sure the fit is of each.
- Per-cluster work is independent and runs in parallel with no randomness and a fixed pass count; a test checks one thread and four give equal output.

## Testing

[tests.rs](../../crates/sfmtool-core/src/patch/cell_plane_normals/tests.rs) builds a planar cluster from known poses: each member's shape is the plane's affine at the patch centre, and its displacements are the exact remainder, the homography's second-order term. The kernel recovers the normal within 0.5° with both axes fixed and every cell in the plane; with every member's shape sheared by 6% and the displacements measured from the sheared shapes it still does, and with those displacements zeroed it does not. Cells refused outside the middle row give one axis whose free axis is the row and a normal perpendicular to it in the plane of the axis and the viewing direction. Two triangulated cells give no normal. A cell reached only by the reference's ray, and one reached only by one kept member when the reference's image is unposed, is not triangulated. A cell whose rays meet off the plane is refused by the plane fit. A narrow baseline triangulates nothing. `refused_outlier` cells enter only on request. The Python test checks the binding's shapes and dtypes, a fronto-parallel plane, and its argument checks.

## Open questions

- **A gate on the normal's precision.** The measurement shows the error follows the precision the cell weights predict, `1 / √(Σ wⱼ (dⱼ · e)²)` along the weaker in-plane axis `e`. Should the kernel return it, and should a consumer gate on it or on a wider minimum triangulation angle?
- **Support.** Most cells are triangulated from two rays, because most clusters have at most one kept member. Whether the normal is worth reading on a two-ray cluster at all is open.
- **The ground truth.** The checked-in ground truths' normals are themselves estimates, and their keypoints are not the cluster files' detections, so the comparison runs on a few dozen clusters. A ground truth whose normals are measured independently would settle the open questions above.
