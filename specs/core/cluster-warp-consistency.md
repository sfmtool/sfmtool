# Cluster Warp Consistency (Weak-Perspective Factorization Residual)

_Status: **implemented** (2026-07-10). A reconstruction-free per-member
consistency signal for cluster patches: how well each member's refined
affine warp agrees with a single jointly-fitted weak-perspective camera per
image and one planar tangent frame per cluster. Computed during
`sfm cluster-patches` (after [cluster-patch
refinement](cluster-patch-refinement.md)) and stored as
`cluster_patches/member_consistency_residual` in the `.matches` file — a
**signal, not a gate**: consumers pick their own threshold, mirroring how
`member_zncc` / `member_shift_px` enable re-vetting without re-running._

## Problem

Cluster-patch refinement vets members photometrically (ZNCC, shift), which
is blind to a failure mode the 3D validation experiments exposed: a
**wrong-match member on repetitive texture** can align photometrically
(ZNCC ≈ 1) while being geometrically incompatible with the rest of its
cluster. On dino_dog_toy, multi-member clusters absent from the SfM solution
triangulated with median 344 px reprojection error — contamination the
photometric gates structurally cannot see, because they compare appearance,
not geometry. Detecting it without requiring poses (the clusters exist
*before* any solve) needs a geometric consistency test built purely from
the warps themselves.

## The model

An affine warp between two views of a surface point is the **Jacobian of
the local image-to-image map** (the classical affine-correspondence
result — see [References](#references)). Model every image `k` as a
**scaled-orthographic (weak-perspective) camera** `M_k` (2×3) and every
refined cluster `c` as a planar patch with tangent frame `T_c` (3×2),
parameterized so the reference member's warp is the identity. Each stored
member warp (the 2×2 linear part of `member_affines`) must then factor as

```
J_ck = M_k · T_c            (reference member: J = I by construction)
```

**Per cluster this is inherently ambiguous.** Each weak-perspective view
adds exactly as many camera unknowns (3 rotation + 1 scale) as its warp
adds measurements (4), so a cluster's plane keeps 2 free slant DOF for
*any* member count — the patch-scale instance of the classical
orthographic-SfM ambiguities. What makes the problem over-determined is
that the cameras are **shared**: every cluster in an image constrains the
same `M_k`, so the joint bilinear system over all clusters (hundreds of
thousands of 2×2 measurements against a few hundred camera parameters) is
massively redundant — the same redundancy the Tomasi–Kanade factorization
exploits, with local affine frames in place of tracked points.

## Algorithm

`warp_consistency_residuals` in
`crates/sfmtool-core/src/patch/cluster_refine/consistency.rs`:

1. **Fit set.** Per refinable cluster: the reference (`J = I`) plus every
   `Kept` member whose warp 2×2 clears `|det| ≥ 1e-6`; clusters with ≥ 2
   such members participate. Everything else reports NaN.
2. **Alternating least squares** on `J_ck = M_k T_c`: tangent half-step
   `T_c = (Σ MᵀM + λI)⁻¹ Σ MᵀJ` per cluster, camera half-step
   `M_k = (Σ JTᵀ)(Σ TTᵀ + λI)⁻¹` per image (`λ = 1e-9` ridge), both
   rayon-parallel over their independent solves with fixed-order
   accumulation. Up to 500 sweeps, early-stopped when the RMS change over
   10 sweeps falls below 1e-9.
3. **Deterministic restarts.** Bilinear factorization with structured
   missing data has local minima, so the ALS runs from 4 deterministic
   inits (orthographic identity perturbed by SplitMix64 noise at escalating
   amplitudes 0.15 / 0.5 / 1.0 / 2.0); the lowest-RMS solution wins (RMS
   computed sequentially so selection is schedule-independent). The whole
   kernel is deterministic and pose-free.
4. **Residual.** Per fit member, the relative misfit
   `‖M_k·T_c − J_ck‖_F / ‖J_ck‖_F`, stored as f32. No metric upgrade is
   performed — the residual is invariant to the factorization's global
   `GL(3)` gauge, and rotations/normals are not this signal's business
   (recovered normals measured *worse than fronto-parallel* against
   embed-patches ground truth — the slant signal is second-order and below
   the warp noise floor; see the experiment notes below).

Cost: ≈ seconds for 358k memberships / 85 images (dino) — a small post-pass
on the cluster-patches operation.

### Model limits (accepted for v1)

Weak perspective ignores perspective variation across each image; on
close-range wide-FOV captures this inflates residuals for wide-baseline
members of *good* clusters (the metric-upgrade Gram matrix measured
slightly non-PSD on dino). That taxes large clusters' scores but leaves the
signal's ranking power intact (AUC below). Paraperspective cameras and
member-level (rather than max-based) consumption are the known follow-ups.

## Evidence (dino_dog_toy, 110k clusters, 85 views)

Ground truth: triangulation of each cluster with the (held-out) solve's
poses; good = in front + max reprojection ≤ 2 px, bad = behind camera or
> 5 px. Camera fit on a random half of the clusters, statistics on the
other half (split-half holdout — scoring is honest out-of-sample):

- **n = 2 clusters: AUC 0.944.** At a max-residual threshold of 0.20 the
  filter would reject 5.3 % of good pairs while catching 80.3 % of bad
  ones; at 0.30, 1.8 % / 70.8 % (bad fraction 16.9 % → 5.7 %).
- **n ≥ 3 clusters: AUC 0.849.** At 0.30: 10.3 % of good rejected, 60.6 %
  of bad caught (bad fraction 26.6 % → 13.7 %); against BA-verified
  in-solution clusters the rejection cost is 18.4 %. Contaminated absent
  multi-member clusters score median residual ≈ 0.8 (90 % > 0.2) vs ≈ 0.05
  for verified pairs.
- Blind vs oracle-camera initialization changes none of this materially —
  the signal does not depend on pose quality, and no pose data enters the
  computation.

Enrichment framing: at threshold 0.30 a RANSAC-style consumer keeps ~90 %
of good clusters while its outlier fraction halves — hence the stored
per-member signal with caller-chosen thresholds instead of a baked-in gate.

## Storage

`cluster_patches/member_consistency_residual.{K}.float32.zst` — member-
parallel like the other signal arrays; NaN where the member did not enter
the fit (non-kept status, degenerate warp, unrefinable or < 2-member fit
cluster). See [matches-file-format.md](../formats/matches-file-format.md).
Added to format **version 3 without a version bump** (no public release
has shipped version 3 files; readers of pre-addition dev files must
regenerate them).

## References

Verified against the papers (PDFs fetched 2026-07-10; short quotes checked
verbatim):

- **The affine correspondence as the local map's Jacobian, and its use in
  SfM.** C. Raposo and J. P. Barreto, "Theory and Practice of
  Structure-from-Motion using Affine Correspondences," CVPR 2016,
  pp. 5470–5478. §2: "the affine transformation A is the Jacobian of the
  homography defined in point x" (result credited to Köser et al.);
  abstract: "Affine Correspondences (ACs) are more informative than Point
  Correspondences (PCs) that are used as input in mainstream algorithms
  for Structure-from-Motion (SfM)."
  <https://openaccess.thecvf.com/content_cvpr_2016/papers/Raposo_Theory_and_Practice_CVPR_2016_paper.pdf>
- **The shared-camera factorization and its redundancy under orthography.**
  C. Tomasi and T. Kanade, "Shape and motion from image streams under
  orthography: a factorization method," IJCV 9(2):137–154, 1992. §3.1:
  "Rank Theorem. Without noise, the registered measurement matrix Ŵ is at
  most of rank three."; §3.3 (the metric constraints): "the rows of the
  true rotation matrix R are unit vectors and the first F are orthogonal to
  the corresponding F in the second half of R." This spec's ALS solves the
  same shared-camera bilinear structure with 2×2 affine frames as the
  measurements; the metric upgrade is deliberately omitted (residual-only,
  gauge-invariant).
  <https://people.eecs.berkeley.edu/~yang/courses/cs294-6/papers/TomasiC_Shape%20and%20motion%20from%20image%20streams%20under%20orthography.pdf>
- **Known-pose normal estimation from ACs** (context for why normals are
  *not* extracted here): D. Barath, I. Eichhardt, L. Hajder, "Optimal
  Multi-View Surface Normal Estimation Using Affine Correspondences," IEEE
  TIP 28(7), 2019 — normals from ACs assume known poses; our pose-free
  experiments measured warp-derived normals at ≈ 75° median error
  (worse than assuming fronto-parallel), consistent with slant being
  second-order in the Jacobian near fronto-parallel.
  DOI 10.1109/TIP.2019.2895542;
  <https://pubmed.ncbi.nlm.nih.gov/30703027/>

## Open questions

1. **Member-level consumption.** The evidence tables score clusters by max
   member residual; per-member pruning (drop the one inconsistent member,
   keep the cluster) should cut the good-cluster rejection cost
   substantially. The stored array already supports it.
2. **Paraperspective cameras** (Poelman & Kanade) to reduce the model error
   that taxes wide-baseline members on close-range captures.
3. **Derived-pairs integration:** expose the signal through
   `pairs_from_matches` so pair consumers (DB export, verification) can
   pre-filter without reading the cluster section directly.
