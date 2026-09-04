# Adjacency surfel normals: robust plane fit over image-space neighbours

## Overview

Given a point cloud and its
[observation adjacency graph](observation-adjacency-graph.md), estimate a
surface normal at each selected point by fitting a plane **through the point
itself** to the *directions* of its graph neighbours. Because the plane is
anchored at the point (surfel semantics) and fitted on unit directions, each
neighbour contributes its angular deviation once, rather than in proportion
to its distance — distant neighbours supply the angular leverage the fit
needs without dominating it, and the neighbours' own position noise enters
only through angle. A Tukey-redescending IRLS loop makes the fit robust to
neighbours that belong to other surfaces.

Alongside each normal the kernel reports how well-determined it is —
effective support, angular coverage, in-plane anisotropy — and a boolean
verdict from thresholds on those. Callers route on the verdict (e.g. keep
the normal, try to acquire more neighbours, or fall back and mark the point
low-confidence); the kernel itself never substitutes a fallback normal, so
an unfittable point comes back `NaN`, not something that looks estimated.

## Fit

Per selected point `p` with neighbour set `N(p)` from the adjacency CSR
(plus any caller-supplied extra neighbour positions, see below):

1. **Displacements.** `d_q = position[q] − position[p]` for each `q ∈ N(p)`;
   rows with `‖d_q‖ ≤ 1e−12` are dropped. `n_support` is the count that
   remains. Points with `n_support < 2` get no normal.
2. **IRLS** (`irls_iters` passes, default 3), all rows starting at weight 1:
   - Normal = eigenvector of the smallest eigenvalue of the weighted scatter
     `Σ w_q d̂_q d̂_qᵀ` over the unit directions `d̂_q`.
   - Scale-free tilt residual `r_q = |d̂_q · n|` (the sine of the angle by
     which `q` sits off the plane).
   - Robust scale `σ = 1.4826 · median(r_q)`, floored at
     `sin(sigma_floor_deg)` (default 2°; the floor also replaces a
     non-finite median).
   - Tukey biweight: `u = r_q / (tukey_c · σ)` (default `c = 4.685`),
     `w_q = (1 − u²)²` for `u < 1`, else 0.
   - **Stall exit, per point:** if the redescended weights of a point sum to
     ≤ 1e−12, the point keeps the weights it had and stops iterating — the
     alternative is an all-zero scatter matrix whose eigenvectors are an
     arbitrary axis frame. A stalled point's recorded `σ` is the last one
     computed while it was still active.
3. **Final solve** with the converged weights, then a sign convention: flip
   the normal so `n · view_dir[p] ≥ 0`, where `view_dir` is the caller's
   per-point reference direction (typically the mean unit direction toward
   the observing cameras).

Medians are numpy-convention (mean of the two middle values on even
counts). The fit involves no randomness and a fixed pass count, so results
are deterministic and independent of thread scheduling.

## Diagnostics and the determinacy verdict

Computed with the final weights `w_q` and normal:

- `n_eff = (Σw)² / Σw²` — effective neighbour count after redescending.
- `anisotropy = λ_mid / λ_max` of the final scatter — how two-dimensional
  the in-plane spread is (a rank-1 direction line cannot pin a plane).
- `sectors` — angular coverage: build a **normal-free** orthonormal basis
  `(e1, e2)` of the plane orthogonal to `view_dir[p]` (seed axis `z`, or
  `x` when `|view_z| > 0.95`; degenerate `view_dir` falls back to `z`), bin
  each neighbour's `atan2(d·e2, d·e1)` into `n_sectors` equal sectors
  (default 8), and count the sectors occupied by **live** rows — those with
  `w_q ≥ 0.25 · max_q w_q` for that point. Basing the sectors on the view
  plane rather than the fitted plane keeps the coverage measure from being
  circular with the normal it qualifies.
- `sigma_deg = asin(min(σ, 1))`, `resid_deg = asin(min(rms, 1))` where
  `rms` is the weighted RMS of the final residuals — the fit's scale and
  misfit as angles.
- `n_support` — surviving neighbour rows.

**`determined`** = `n_support ≥ 2` ∧ `n_eff ≥ det_n_eff` (default 4) ∧
`sectors ≥ det_sectors` (default 3) ∧ `anisotropy ≥ det_aniso` (default
0.10). The verdict routes points; it is not a quality score.

## Extra neighbours

Callers may pass synthesized neighbour **positions** for specific points
(e.g. congealed helper patches acquired for points whose graph
neighbourhood is under-determined). They enter the fit exactly like graph
neighbours — displacement from `p`, unit direction, IRLS weight — and count
toward every diagnostic. Extras for unselected points are ignored.

## Output

Dense arrays over the whole cloud, `NaN` (or `false`) for unselected and
unfitted points:

- `normals` — `(n, 3)` f64, unit, sign-aligned to `view_dir`; `NaN` rows
  where `n_support < 2` or the point was not selected.
- `n_eff`, `anisotropy`, `sectors`, `sigma_deg`, `resid_deg`, `n_support`
  — f64 per point.
- `determined` — bool per point.

## API

The kernel lives in
[adjacency_surfel_normals.rs](../../../crates/sfmtool-core/src/analysis/adjacency_surfel_normals.rs),
bound as `sfmtool._sfmtool.analysis.estimate_adjacency_surfel_normals`.

```rust
pub struct AdjacencySurfelParams {
    pub irls_iters: u32,       // 3
    pub tukey_c: f64,          // 4.685
    pub sigma_floor_deg: f64,  // 2.0
    pub n_sectors: u32,        // 8
    pub det_n_eff: f64,        // 4.0
    pub det_sectors: u32,      // 3
    pub det_aniso: f64,        // 0.10
}

pub fn estimate_adjacency_surfel_normals(
    positions: &[[f64; 3]],          // per point
    offsets: &[u32],                 // adjacency CSR, n_points + 1
    neighbours: &[u32],
    view_dirs: &[[f64; 3]],          // per point; sign + sector reference
    selected: &[bool],               // which points to fit
    extras: &ExtraNeighbours,        // CSR of extra positions, may be empty
    params: &AdjacencySurfelParams,
) -> AdjacencySurfelNormals
```

The per-point work is independent; the implementation parallelizes over
selected points with per-point 3×3 symmetric eigendecompositions (choosing
eigenvectors by explicit eigenvalue comparison, never by an assumed
ordering).

The Python binding takes numpy arrays (accepting `SfmrReconstruction`
accessor dtypes and the adjacency builder's outputs directly), `extras` as
an optional `dict[point_index → (k, 3) float64 array]`, keyword parameters
with the defaults above, and returns the output arrays as a dict keyed as
in the output list.

## Testing

Sibling `tests.rs` under `analysis/adjacency_surfel_normals/` covers: an
exact plane of neighbours recovering its normal (and the sign flipping to
the `view_dir` side); a plane plus gross off-surface outliers, where the
Tukey loop drives the outliers' weights to zero and recovers the clean
normal; the sigma floor engaging on near-perfect data (median residual
below `sin 2°`); the per-point stall exit keeping a usable normal when all
residuals redescend to zero weight; `n_support < 2` yielding `NaN` and
`determined = false`; a collinear neighbour line failing the anisotropy
gate and a half-plane-only neighbourhood failing the sector gate while a
well-spread one passes; extras entering the fit (an under-determined point
becoming determined when extras fill its empty sectors) and extras for
unselected points being ignored; and unselected points staying `NaN`
end-to-end.
