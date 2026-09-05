# Source Clusters: What a Member's Admission Left Behind

A reconstruction member is drawn from a cluster selection and holds a subset of
its clusters. The rest are not worse evidence; a smaller feature localizes
better, and the reason it was left behind is that the selection the member was
built from wanted a small number of large ones. This kernel names those
clusters, reads each one's feature radius, and bands them by radius so a caller
can bring them back a band at a time.

## The join is exact and carries no geometry

A member observation and a selection row both carry the image index and the
feature index. Both index the same tables, so a member row IS a selection row by
`(image, feature)`; nothing has to be re-detected, re-matched or compared in
pixels.

The join packs each key into one 64-bit integer, `(image << 32) | feature`,
sorts the selection's keys with ties left in selection row order, and binary
searches each member key against them. A member row takes the FIRST selection
row carrying its key. Selections may repeat a key across clusters, so which one
is taken is part of the contract rather than an accident: the first is the one
in the lowest-numbered cluster, and any later cluster carrying the same key is
still available as a candidate.

The count of member rows that found a selection row at all comes back beside the
result, so a caller can see whether the two tables really describe the same
capture.

## A candidate is a cluster two placed frames see that no row admitted

A cluster the join marked admitted belongs to the member already. A cluster only
one placed frame sees states a bearing and no depth: one ray is not a
measurement. A candidate is neither: at least two of the frames the caller names
see it, and no member row admitted it.

Candidates come back ascending by cluster id, and the selected observations come
back in selection row order, so the result is a function of the selection and
the member alone rather than of the order anything was enumerated in.

## Radius is read off the stored affine

Each selection row carries an absolute 2x3 affine: the leading 2x2 is the
feature's shape in that image's pixels and the last column is its keypoint. A
row's radius is the refine radius times the MEAN of the affine's two column
norms, computed as half the refine radius times their sum. A cluster's radius is
its widest row's; a cluster with no rows reads `0.0`.

That is the same reading the selection's own admission was drawn with, so
"radius" means the same thing on both sides of the join.

It is also one implementation rather than a convention each caller restates.
The reading lives in its own module and the join calls it, so a caller that
wants the radii alone, or the coarsest clusters ordered by them, gets exactly
the numbers the bands are measured against.

The reading is width-agnostic on purpose. A `.matches` backbone stores the
shapes as `f32` and a selection read back through Python may hold them as
`f64`; every value widens to `f64` where it is read, before the first multiply,
so both spellings of the same shape produce the same radius bit for bit.

## The coarsest-N cut

A caller that wants a small alias-free working set takes the `n` clusters of
largest radius. The ordering is radius descending with ties broken by ascending
cluster id, and `NaN` -- a shape with no extent -- sorts last, so a cluster that
has an extent is never displaced by one that does not.

The cut comes back sorted ASCENDING by id. That is what makes it composable: it
is a cluster-id set to restrict a selection by, not a ranking. Fewer than `n`
clusters yields all of them.

## Bands are octaves of the admission floor

The floor is the smallest radius the member's admission holds. It is the unit
the bands are measured in, because an absolute pixel radius is not commensurable
across captures whose sensors and refine radii differ by an order of magnitude.

`band_edges` runs decreasing in units of that floor: band `k` holds the radii in
`[floor * band_edges[k + 1], floor * band_edges[k])`, half open, so a radius
exactly on an edge belongs to the band below it. The first edge may be infinite,
which leaves the top band open above the floor: a member's admission is a
group-local selection over its own images rather than a capture-wide radius bar,
so a cluster coarser than the floor can sit outside it. A radius under the last
edge falls in no band and reads `-1`.

Where the admission is empty the floor is `NaN`. Every band comparison against
`NaN` is false, so nothing lands in a band and the caller reads the refusal off
the floor itself rather than off a band that silently swallowed everything.

The banding is also exposed on its own, so a caller holding a radius array can
apply the same rule without running the join.

## What calls it

A caller that holds a partial reconstruction drawn from a selection and wants
the clusters that selection holds but the reconstruction never admitted: it
runs the join once against the selection, then brings the candidates back one
band at a time, coarsest first.

## The binding

The kernel lives in
[source_clusters.rs](../../../crates/sfmtool-core/src/analysis/source_clusters.rs),
bound as `sfmtool._sfmtool.analysis.source_clusters` and `assign_bands`.

```python
from sfmtool._sfmtool.analysis import assign_bands, source_clusters

out = source_clusters(
    cluster_starts,    # (n_cluster + 1,) uint32 CSR boundaries
    member_images,     # (n_member,) uint32
    member_features,   # (n_member,) uint32
    member_positions,     # (n_member, 2) float64
    member_affine_shapes, # (n_member, 2, 2) float64
    refine_radius,     # float
    n_images,          # int
    obs_image,         # (n_obs,) uint32, the member's own
    obs_feature,       # (n_obs,) uint32, the member's own
    frames,            # (n_frames,) uint32, the placed images
    band_edges,        # (n_band + 1,) float64, decreasing
)
out["candidates"]        # (n_cand,) uint32 ascending
out["candidate_radius"]  # (n_cand,) float64
out["candidate_band"]    # (n_cand,) int64, -1 past the last edge
out["obs_uv"]            # (n_selected, 2) float64
out["obs_shape"]         # (n_selected, 2, 2) float64

bands = assign_bands(radius, floor, band_edges)
```

The counts `n_file_clusters`, `n_admitted`, `n_rows_matched` and `n_selected`
come back beside the arrays, along with `admission_radius` and
`admission_floor_px`.

The band grid itself is not built here. It is a property of the population a
caller is banding, and the caller passes the edges it decided on.

The radius reading and the cut over it live in
[cluster_radii.rs](../../../crates/sfmtool-core/src/analysis/cluster_radii.rs)
(`member_radii`, `cluster_radii`, `coarsest_clusters`, and the
`*_from_matches` forms), bound as `sfmtool._sfmtool.analysis.cluster_radii` and
`coarsest_clusters`. Both bindings take their input in either of two forms, the
shape the intrinsics estimate uses: a `MatchesFile` -- a selection included --
which states the shapes and the refine radius itself, or those arrays spelled
out.

```python
from sfmtool._sfmtool.analysis import cluster_radii, coarsest_clusters

radius = cluster_radii(matches)              # (n_clusters,) float64
keep = coarsest_clusters(matches, 3000)      # (<= 3000,) uint32, ascending

radius = cluster_radii(
    cluster_starts,        # (n_clusters + 1,) uint32 CSR boundaries
    member_affine_shapes,  # (n_member, 2, 2) float32
    refine_radius,         # float
)
keep = coarsest_clusters(cluster_starts, 3000, member_affine_shapes, refine_radius)
```

The file forms need the cluster backbone, its member affine shapes, and the
`cluster_patches/` section the refine radius is recorded in; a file missing any
of those is refused by name rather than answered from a default.
