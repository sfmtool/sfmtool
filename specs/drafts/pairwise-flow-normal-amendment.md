# Pairwise Flow Normal: Tests Not Yet Written and Open Questions

**Status:** Draft. The step is built, and Python, Track View, the wire and
`refine_normals` run it; what remains is three tests and the questions about
its settings and rules. Nothing here changes the step until it is built or
settled.

Amends: [core/bench/pairwise-flow-normal.md](../core/bench/pairwise-flow-normal.md).
The measurements cited are in
[core/bench/pairwise-flow-normal-measurements.md](../core/bench/pairwise-flow-normal-measurements.md).

## Purpose

`pairwise_flow_normal` estimates a patch's normal from the optical flow between
pairs of the point's views rendered on a common plane, choosing the largest
support on which the pairs agree, and says when the normal is undetermined or
unstable. Rust code calls it, Python through its binding, Track View's
*Pairwise Flow Normal* and the wire's `fit_bench_track_normal` on one track,
and `refine_normals` on every point when asked, but several of its settings
were set on a handful of points or not measured at all. This draft lists the
tests the step does not have yet, and keeps the questions still open about it
in one place, so that a later measurement can settle them. A test that is
written, or a question that is settled, is deleted here and folded into the
spec.

## Tests not yet written

- **Exact geometry.** A textured plane seen by three or more cameras at
  different angles, from a given normal 20° off: `Turned` within a fraction of a
  degree. The built test starts 40° off and ends within a degree (0.01° on its
  scene); this one would bound the error more tightly.
- **The grid follows the finest view, end to end.** Patches covering about
  10, 43 and 1000 photograph pixels across in their finest view get 24, 43 and
  128 samples per side; at 128 the crop is 384 samples a side and the 1000 px
  view is read through its photograph's pyramid at level
  `round(log2 7.8)` = 3, and the step returns `Turned` near the true normal.
  The crops' own tests check the grid's rule; this would run the step on such
  photographs.
- **The residual σ.** A textured plane with a slight curvature over the disc
  and strong texture: the residual σ exceeds the texture σ at the larger radii,
  and the mean χ² there is lower than it would be with the texture σ alone.

## Open questions

- **seoul_bull stops improving after round 1.** On the seoul_bull points the
  prototype's rounds stopped at about 10° to 18° from the reference after round
  1, with pairs 2° to 5° apart, against 6.9° on the curated seoul_bull points for
  the first prototype. Point by point, the remaining errors are concentrated on
  layered and narrow-baseline points, not on the `ok` points: of the ten `ok`
  points, 1, 2, 109, 143 and 157 ended 1.2° to 7.1° from the reference, 10
  ended 16°, 18, 120 and 167 were `Unstable`, and 160, a single pair, ended 68°
  (before the one-round rule). What the step should do on a point with another
  layer in its crops is the open part. The Rust step has not been measured on
  seoul_bull.
- **kerry_park is still improving at round 3.** Whether to run more rounds, or
  take a stronger step per round.
- **Uncurated points.** On 40 evenly spaced kerry_park points the step left 18
  undetermined and turned some far from the stored normals (76.7°, 70.5°,
  42.9°) within the bar on the mean viewing direction
  (measurements, § "The Rust step on real points"). Which of those are wrong,
  and whether a rule catches them, is not looked at yet.
- **Which round's answer.** The step reports the last round's; whether to keep
  an earlier round's answer when the last has none, and whether to stop early
  when a round's answer moves less than its σ.
- **Start dependence.** The rounds were measured only from the mean viewing
  direction. Starts from the stored normal and from a hand-edited normal are not
  measured.
- **Separating undetermined points.** The outcomes do not yet tell the `A2`
  points (those whose grown-plane answer depends on its start) from good ones:
  at χ² ≤ 1, 8 of 18 kerry_park `A2` points turned in the prototype. They may
  be determined, with wide baselines; the `A2` label comes from the grown
  plane's instability, not from the truth.
- **The thresholds.** `max_sigma_deg` of 10° with `max_mean_chi2` of 1 or 2
  are the best settings tried so far on real points; neither was swept finely.
  `max_mean_chi2`, the agreement window and the radius list were also tuned
  against reference normals that the grown-plane step made, apart from two hand
  normals, and need checking against normals that step did not make.
- **Very wide pairs.** Whether to leave out pairs more than some angle apart
  (90° and more): on a common plane they line up, but their crops see the
  surface from very different sides.
- **The crops' defaults.** `crop_half_extents`, `min_crop_half_width_px`,
  `max_crop_samples` and the radius list are the values the real-point runs
  used, and `single_pair_radius` is proposed; none was measured against other
  values.
- **Points with one pair.** There is no agreement to test; the first prototype
  read them at 8 photograph pixels. The step reads one in one round and turns it
  only when its σ there is at most `max_sigma_deg`. The one-round rule rests on
  one point, seoul_bull 160, which drifted over the rounds. Open: whether a
  precise single pair should be turned at all, read at a fixed radius, or have
  its radius chosen by another rule.
- **Drift that stays in view.** The bar on the mean viewing direction stops an
  answer the views cannot see, but not every drift: seoul_bull point 207 ended
  112° from the reference within 80° of the mean viewing direction in the
  prototype.
- **The noise threshold.** The noise model the tensors are thresholded with
  (white Gaussian noise of the photometric residual's variance) and the
  significance level `noise_false_alarm`. The structure window has no setting
  of its own: it is the correlated square, so it is settled with the
  correlation length.
- **Which pairs.** A round reads at most 10 pairs, chosen by a shuffle with a
  fixed seed, as the prototype chose them at random. Better choices to
  measure: widest angles first, a spread of baselines, each view in a fixed
  number of pairs, or the pairs whose geometry predicts the most information
  about the tilt, as the grown-plane step's selection does. A choice should
  prefer pairs whose two views have similar scale: a very unequal pair is read
  at the coarser view's level and carries little information, which its σ
  shows. Every view of a round is rendered, including those in no chosen pair.
- **The correlation length.** 8 photograph px is a guess at how far flow errors
  stay correlated. It sets the scale of the texture σ and, through `L²`, of the
  residual σ; it could be measured from the residuals of the synthetic pairs.
- **The combined σ.** The pairs reuse views, so their information is not
  independent and the combined σ is too small; whether to discount it, as the
  grown-plane step halves each kept pair's information. The tests bound the
  error by the largest σ among the combined pairs for this reason.
- **A crease through the point.** On a crease through the point the rule
  shrinks the support to where the pairs still agree and turns the patch to
  about the bisector of the two faces, rather than reporting `Unstable`.
  Whether the bisector is the wanted answer for such a point, or whether it
  should be `Unstable`, or have its normal from one face, is open.
- **Layered points.** A foreground object against a background makes the flow
  see two layers. The chosen radius should stop before the other layer; the χ²
  rule does so only where the layers make the pairs disagree, which a narrow
  baseline may not. The robust weights reduce the weight of an outline whose
  flow belongs to the other layer, but do not remove the other layer from a
  disc.
- **GPU flow.** `compute_optical_flow` takes a `GpuFlowContext`
  ([core/features/gpu-optical-flow.md](../core/features/gpu-optical-flow.md));
  whether batches of small crops gain from it. A pair's square under 224
  samples a side is below the presets' `gpu_min_pixels` of 50,000; only pairs
  read at level 0 on a large patch, up to 384 a side, or on a tiny one whose
  grid is held at its lower bound, up to 512, reach it.
