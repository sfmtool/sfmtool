# ZNCC-weighted refinement-basis selection (amendment)

**Status:** Draft

Amends
[`../core/patch/patch-normal-refine-view-subset.md`](../core/patch/patch-normal-refine-view-subset.md),
which specifies the shipped D-optimal view-subset selection and points back here.

The shipped pick is purely geometric: it maximizes `det(M_S)` over
`M_S = Σ_{i ∈ S} wᵢ wᵢᵀ`, where `wᵢ` is view `i`'s tangent-plane information
direction, and every view contributes with weight 1. That maximizes *observability*
of the two tilt degrees of freedom and nothing else. This draft proposes weighting
each view's contribution by how well it actually matches the consensus.

## Why it is not a nicety

Geometry-only D-optimal deliberately selects the *most oblique* views, because an
oblique view carries the most information about a normal. Those are also the
photometrically worst views a patch has: most foreshortened, most aliased, most
likely to be partially occluded. So the selection rule is, by construction,
biased toward the noisiest members of an already-vetted set.

That is the most likely driver of the large per-point normal divergence the
subset shows against the all-views baseline — on the Spain Soapmaker sweep the
normal Δ p95 is far above the median (`K = 5`: median 6.4°, p95 36°), which is
the signature of a minority of points picking a bad basis rather than a uniform
loss of precision. Until that is fixed, any geometric subset trades robustness
for observability, and the choice of `K` is a balance rather than a calibrated
optimum.

## The proposal

Weight each view's information contribution by its per-view ZNCC-to-consensus
(its signal-to-noise proxy):

```
M_S = Σ_{i ∈ S} s(zncc_i) · wᵢ wᵢᵀ
```

so the greedy maximizes information *per unit of photometric noise* rather than
raw information. The shape of `s(·)` is undecided; the plausible candidates are
the ZNCC itself, clamped at zero, and a steeper monotone map that effectively
excludes views below a floor. Which one to use should be settled by the existing
validation harness
([`../../scripts/validate_refine_subset.py`](../../scripts/validate_refine_subset.py)),
whose normal-agreement p95 against the all-views baseline is the metric this is
trying to move.

## What it needs first

The scores have to reach the kernel. `select_views` computes a per-view score for
every admitted view, and `_embed_patches.py` now keeps those scores and threads
them into `localize_keypoints` for the consensus-basis cap (see
[`../core/patch/keypoint-localization-consensus-basis.md`](../core/patch/keypoint-localization-consensus-basis.md)),
so the plumbing pattern already exists and works. What remains is that the
normal-refinement path runs per round with the point set compacted between
rounds — point indices are renumbered — so the score map has to be re-keyed each
round rather than passed through once. Whether the right score is
`select_views`'s own view score or the refinement's per-view ZNCC-to-consensus,
measured at refine time and therefore available without any plumbing at all, is
the first thing to decide.

## Open questions

- Whether the weight belongs in the *selection* or in the refinement's own
  residual weighting, where a bad view already costs less. The selection is
  cheaper to change and does not touch the objective.
- Whether a per-view floor (drop below `zncc_min`, then pick geometrically among
  the survivors) captures most of the benefit for a fraction of the design cost.
- Whether the weighted pick changes the right default `K`, which is currently 8
  in the pipeline and 0 (disabled) at the crate level.
