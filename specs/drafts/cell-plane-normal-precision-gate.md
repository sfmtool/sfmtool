# Cell Plane Normal Precision Gate

**Status:** Draft. Decided:
- no consumer reads a cell plane normal until a gate on its precision has been measured: the verdict says which of the normal's axes the cells fix, not how well, and the measurements show the error follows a precision the kernel can compute but does not return;
- the gate is measured against a ground truth able to tell the cell normal from the alternatives, which the two checked-in ground truths are not.

Not decided: whether the gate reads the predicted precision or a wider minimum triangulation angle; its bar; whether the cell displacements earn their place over the members' affine shapes; whether the seed's writer derives its frames from the cells. See [Open questions](#open-questions).

Amends:
- [core/patch/cell-plane-normals.md](../core/patch/cell-plane-normals.md): a precision output per cluster, and the gate on it (its Non-goals link here)
- [core/patch/cluster-patch-refinement.md](../core/patch/cluster-patch-refinement.md) § "Piecewise refinement": a first consumer of the stored cells (its Non-goals link here)

Related drafts: [piece-gated-grid-normal.md](piece-gated-grid-normal.md) is the rendering estimator that shares the cell plane normal's verdict.

## Purpose

A cluster-patches file written with `--piecewise` stores, for every kept member of every cluster, where each of nine cells of the patch lies relative to the member's affine shape. Once camera poses exist, the cell plane normal kernel turns those displacements into a patch normal per cluster, with a verdict saying which of the normal's axes the cells fix. Both exist; nothing reads them. The seed's writer still derives the normals of its debug snapshots from a tilt solve on the whole-member affine shapes, and a hypothesis release carries no patch frames.

The measurements on the two checked-in ground truths ([cluster-patch-refinement-measurements.md](../core/patch/cluster-patch-refinement-measurements.md#cell-plane-normals-after-the-audit-2026-10-08)) say why a consumer cannot simply read the normal. Over the clusters the kernel calls both-axes, the median error is about two-thirds of the mean viewing direction's (19.1° against 29.3° on `SeoulBull`, 20.6° against 30.0° on `KerryPark480`), but the p90 is near 60°, no better than the viewing direction. The error is not spread evenly: it follows the precision the cell weights predict. This draft proposes returning that precision, gating on it, and only then giving the normal a consumer.

## Proposal

### Return the predicted precision

The plane fit already weights each live cell by the inverse of its position's variance along the normal. The precision those weights predict for the normal, along the plane's weaker in-plane axis `e`, is

```
σ_n = 1 / √(Σⱼ wⱼ (dⱼ · e)²)
```

with `dⱼ` the live cell's offset from the weighted centroid. The measurement computed it in its script from the kernel's outputs; the kernel would return it per cluster, as `normal_precision_deg` beside `anisotropy` and `n_eff`, `NaN` where the verdict is none. On the first measurement, clusters predicted under 5° had a median error of 7.7° and 11.1°, and those over 15° of 27.9° and 29.7°, against a viewing-direction error near 30° throughout ([measurement](../core/patch/cluster-patch-refinement-measurements.md#cell-plane-normals-against-the-ground-truths-2026-10-08)).

### Gate on it

A consumer reads a cluster's normal only when the predicted precision is under a bar, and otherwise treats the cluster as having no measured normal. The bar is set from a sweep of the outcome, the error against a ground truth able to judge it, not fixed in advance. The gate belongs to the consumer, not to the kernel: the kernel reports the verdict and the precision, and a consumer with different needs chooses its own bar.

The alternative gate is a wider minimum triangulation angle. At 5° instead of 2°, the first measurement's both-axes clusters had median errors of 10.7° and 10.8°, a subset about as accurate as the precision gate's, at the cost of the both-axes share (11.7% and 5.8% of all clusters). The angle gate reads one geometric quantity and needs no new output; the precision gate reads the cell weights, which also carry the ray count and the intersection residual. The sweep measures both on the same clusters.

### The seed's writer derives its frames from the cells

With a gate measured, the seed's writer builds each released point's patch frame from the cells: the gated cell plane normal where the cluster has one, and the viewing direction otherwise. Today the writer does not read the cells at all. A photometric score of the seed's candidates is the consumer that needs those frames.

## Open questions

- **Precision or angle.** Which gate, and at what bar, is decided by the sweep above.
- **Displacements or shapes.** With the same cells, setting every displacement to zero, so the cells sit where the members' affine shapes place them, gives a median error within 4° of the cell normal at the current defaults (1.0° better on `SeoulBull`, 3.6° worse on `KerryPark480`, [measurement](../core/patch/cluster-patch-refinement-measurements.md#cell-plane-normals-at-the-current-defaults-2026-10-09)), better on one ground truth and worse on the other. If the displacements add nothing measurable over the shapes, the frames could be built from the shapes alone, and the piecewise stage would earn its cost only through its per-cell statuses. The checked-in ground truths cannot settle this: their keypoints are not the cluster files' detections, so under 1% of clusters with a reference match a ground-truth point within 1 px, too few to separate two estimators a few degrees apart; and their normals are themselves estimates, so where a cell normal disagrees with a truth near the viewing direction they cannot say which is wrong. A ground truth built from the same detections as the cluster files, or one whose normals are measured independently of any patch estimator (a surveyed plane, a calibration target), would settle it, and the same ground truth is what the gate's sweep needs.
- **Support.** Most triangulated cells are seen by two rays, because most clusters have at most one kept member. Whether a two-ray cluster's normal is worth reading at all, or whether the gate should also require a kept-member count, is open.
- **The `move_shape` loop.** The loop that lets the cell fit move a member's shape is on by default, since a blind human review preferred the shapes it moves to the cascade's ([review](../core/patch/cluster-patch-refinement-measurements.md#human-review-of-moved-shapes-2026-10-09)). Whether the moved shapes change the cell plane normals is settled: on files written at the current defaults the loop moves 6 and 60 members of `SeoulBull` and `KerryPark480`, the both-axes normals of the 3 and 17 clusters they touch turn by a median of 7.2° and 1.3°, and no figure against the ground truths changes ([measurement](../core/patch/cluster-patch-refinement-measurements.md#cell-plane-normals-at-the-current-defaults-2026-10-09)), so the numbers above hold for the default. Whether the loop should start from the cascade's shape or from the detection's, which would say whether the cascade still earns its cost, is open.
