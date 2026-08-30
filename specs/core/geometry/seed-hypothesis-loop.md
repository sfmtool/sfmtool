# Seed Hypothesis Loop

The seed stage develops structure hypotheses while the ones it commits
stay trustworthy and their coverage claims leave real evidence
unexplained, then ships the one best supported by capture-level
measurements. A hypothesis is one full seed exploration — probe, widen,
photometric verify, focal scan, release — over an admitted cluster
selection. The first hypothesis admits the whole selection; each later
hypothesis admits the coverage complement of everything the committed
hypotheses before it claimed. A capture that one hypothesis explains
produces exactly the single-hypothesis result; a capture whose cluster
evidence describes more than one rigid world produces one committed
hypothesis per world, and the arbitration picks the one the
capture-level measurements support.

## Capture-level measurements

The pairwise focal vote — and, where escalation confirms one, the
camera-model verdict — is computed once, over the full admitted
selection's pair graph, before any hypothesis runs. Every hypothesis
reads the same vote. The vote is a property of the capture, not of a
hypothesis: it is the independent referee the arbitration measures each
hypothesis's release against, so no hypothesis re-derives it from its
own (restricted) pair graph.

## Coverage claim

A committed hypothesis claims the image area its retained structure
samples, at the resolution the evidence itself samples it.

The retained structure is every cluster with a finite triangulated
position in the released geometry's full triangulation. The claim is
TRANSITIVE over cluster membership: a retained cluster is an explained
3D point, so its members stamp in **every** image they appear in —
posed or not. A hypothesis that poses a handful of a long capture's
frames still claims its structure's footprint capture-wide.

The claim is an occupancy grid per image, not a pixel bitmap. Each
image's cell size is the median nearest-neighbour distance among the
retained members' keypoints in that image — coverage measured at the
spacing the matcher actually sampled the scene, fine on dense texture
and coarse on sparse, with the pixel scale of the capture divided out.
A cell holding at least one retained member's keypoint is claimed.
Images with fewer than two retained members claim nothing (no spacing
exists to measure). Claims accumulate across committed hypotheses as a
per-image union of claimed cells (each hypothesis stamps into its own
grid geometry; a later test evaluates against every accumulated grid).

## Complement admission

The next hypothesis admits the source selection minus the claimed
clusters. A cluster is claimed when more than half of its members fall
in claimed cells of their images. The complement is expressed as a
cluster-id restriction of the stage's selection handle
(`select_clusters` with `restrict_cluster_ids` — see
[cluster-selection.md](../../formats/cluster-selection.md)), so a complement is
itself an ordinary derived selection: it carries provenance, and the
downstream stages — the seed's restriction stage, the finalization —
read it exactly like the unrestricted one. No stage applies a claim
predicate of its own; the selection file is the admission.

## Materiality

A complement is explored only when the claim actually bit: when it
retains less than half of the clusters the previous pass admitted. A
complement that is most of the admission again means the committed
hypothesis's structure barely overlaps the evidence pool — on a
single-world capture that is the signature of an under-posed seed, and
exploring it enumerates independently-seedable frame windows of the
same world rather than finding another one.

The one exception is rescue: when the hypothesis just committed does
NOT qualify (the arbitration's gates below), its complement is explored
regardless of materiality — an untrusted hypothesis's claim is not
evidence about the rest of the capture, and the capture still gets its
look at the other world. The exception is spent once per capture: one
rescue exploration, not a chain of them — a run of untrusted
hypotheses says the capture seeds poorly everywhere, and enumerating
its windows is not development.

## Loop

The first pass explores the full admission. The loop then derives the
committed hypothesis's claim, forms the complement selection, and —
when the complement is material or the rescue exception holds —
reruns the exploration on it. A pass that commits no seed — no seed
group, or a release that poses no image — ends the loop, as does an
immaterial complement under a qualified hypothesis, an empty
complement, or a claim that claimed nothing. Termination is
structural: a committed hypothesis claims at least the clusters whose
members it retained, so the complement strictly shrinks with every
committed pass.

## Arbitration

Each committed hypothesis records its released focal, released inlier
fraction, coverage reach, scan spread, confidence flags, and the
log-focal distance between its release and the bias-corrected
capture-level vote. A hypothesis qualifies when the existing
structure-trust gates all hold: the commit bar (posed count, reach,
scan spread), the release inside the corrected vote band, and no
flat-scan or edge-scan verdict. Coverage reach is measured on the
CAPTURE-LEVEL covisibility graph — the full admission's — for every
hypothesis alike: reach asks how much of the capture a solve connects
to, and a complement's smaller admission must not deflate the answer
for a solve that genuinely spans it.

Two hypotheses are DISTINCT when they share at least two posed images
and disagree about the geometry there: over the image pairs posed in
both, the median difference between the two hypotheses' relative
rotations exceeds 5° — above the seed stage's own pose-noise scale.
Hypotheses with disjoint posed sets, or shared frames in agreement,
are provisionally the same world seeded from different windows.
Combination's cross-resection certificates (below) are a second source
of the same verdict, for the world-split the shared-frame test cannot
see: a pair whose linking frames systematically resect into one
structure but not the other is two FRAME-DISJOINT worlds, whatever
their posed sets said — the reclassification stops the weld and feeds
the flag, but never re-runs the arbitration (the incumbent already won
among the qualified). Neither test sees a CONTENT-SPLIT pair — two
structure populations visible from the same viewpoints, sky against
ground — because a frame of such a capture genuinely resects into both
worlds; there the shared-frame rotation disagreement is the only
detector, and a certificate that passes both ways is not evidence of
one world.

The shipping rule:

- The earliest qualified hypothesis is the incumbent. A qualified
  challenger displaces it only when the two are DISTINCT and the
  challenger ranks higher (released inlier fraction, coverage reach as
  tiebreak). A non-distinct challenger never displaces a qualified
  incumbent, whatever its numbers — inlier fractions measured on
  different admissions of the same world reward the smaller solve.
- When no hypothesis qualifies, the first hypothesis ships with its
  confidence flags — the single-hypothesis behavior, unchanged.
- When only a later hypothesis qualifies, it ships (rescue).

`confidence_flags` gains `multiple_hypotheses` only when two or more
hypotheses qualify AND at least one qualified pair is distinct — the
capture's cluster evidence supports more than one rigid world, and the
unchosen one is real structure, not noise.

## Combination

Losing hypotheses are working capital, not waste. Their point clouds
are never merged across gauges: same-world hypotheses agree about
rotations and disagree only about the depth their narrow windows never
observed, so shared structure carries too little triangulation angle to
fix a similarity transform and a cross-gauge alignment imports exactly
that unobserved freedom. Their FRAMES, though, are certified-seedable
viewpoints of the winner's own world.

After arbitration, the winner grows by resection over one pool of
certified frames — the other committed non-distinct hypotheses' posed
frames, plus BRIDGES. A bridge is a frame posed by neither hypothesis
that is covisible with both retained cluster sets (found by membership
counting over the full admission, no solving), at the growth ladder's
own pool floor on each side: a frame that cannot clear the floor
toward a structure cannot resect there, so counting membership is the
whole pre-filter and no candidate costs a solve until it has cleared
it twice. It earns its place in the pool by resecting into BOTH
structures, with asymmetric roles. The DONOR-side resection is a
CERTIFICATE only — it proves the frame genuinely views the donor's
world, and its donor-gauge pose orders the walk (donor frames nearest
an accepted bridge first) — and is then discarded; the donor's
depths never contribute a measurement. That certificate is what admits
the frame to the pool. The winner-side resection is the load-bearing
one and the WALK makes it, against the structure the walk currently
has and through the same gate and verification as every rung: a
candidate the released structure cannot reach yet is exactly what a
rung of growth is for, so a winner-side resection taken against the
release admits nothing and refuses nothing. Donor-posed frames carry
their certificate already: the donor posed them.

The walk is one growth loop over the pool: each rung takes the best
candidate that clears the ladder's observation floor against the
CURRENT structure, and between rungs the structure grows — clusters
that gain a second winner-gauge view triangulate in, which is what
lets a bridge open a window whose frames saw none of the winner's
original structure. Frames covisible with neither hypothesis are not
in the pool: welding committed hypotheses is this stage's job,
growing the capture is the completion's. When the frames that count
well toward both structures systematically fail one side's
certificate, the pair is reclassified DISTINCT (see the arbitration)
and the weld aborts, withdrawing that donor's frames.

The population that verdict reads is every frame clearing the floor
toward both structures — the bridge candidates and the donor's own
frames alike, since a donor frame holds its donor-side certificate
already and only its winner side is open. A frame certifying into one
structure and not the other is a DISCORDANT trial. SYSTEMATIC means
TOTAL: one side certifies NOTHING while the other certifies, over
enough discordants for the one-sidedness itself to beat chance. A
rate DIFFERENCE cannot carry the verdict, because the two sides are
not exchangeable trials — a complement hypothesis's thin structure
certifies a candidate more readily than the full admission's rich one,
so same-world pairs produce lopsided rates of their own.

Every acceptance runs the gate scaled by the winner's own consensus
(the widen ladder's rule) with per-rung verification; the grown state
is retriangulated and bundle-adjusted at the winner's released focal,
with the release basin guard unchanged. The combination must leave the
winner no worse on the gates: the additions are reverted whole
(keep-best) when a gate that held before stops holding, or when the
combined release raises a confidence flag the winner did not carry — a
collapse that keeps the focal inside its basin registers on the
consensus flag and nowhere else. Frames of DISTINCT hypotheses are
never resected into the winner: they belong to another world.

## Product

The winning hypothesis — after combination — finalizes through the
restriction stage against its own selection handle, with the
restriction covering the combined posed set, and writes
`sfmr/seed-final.sfmr`; losing hypotheses are not finalized.

Every committed hypothesis's released estimate — winner included — is
written as a release-grade reconstruction (poses and points, no
bitmaps) under `sfmr/seed-hypotheses/<stamp>-h<k>.sfmr`, so the
developed alternatives stay inspectable after every run. A new run
under a new stamp accumulates alongside old ones, like the round
snapshots. The artifacts are written under the capture's own camera
model: a fisheye capture's hypotheses densify and reproject through
the equidistant context, never the pinhole default.

Every committed rotation-only hypothesis also commits a relaxed sibling,
a finite member built from the observations its rotation model refused
(see [seed-relaxation.md](seed-relaxation.md)); the rotation-only entry
and its release file are unchanged by it.

`tool_options` gains hypothesis records only when more than one
hypothesis committed, so a single-hypothesis capture's metadata is
byte-identical to a run without the loop: `hypothesis_count`, the
winner's index, and per-hypothesis released focal, inlier fraction,
posed count, and flags.
