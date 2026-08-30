# Copyright The SfM Tool Authors
# SPDX-License-Identifier: Apache-2.0

"""The seed's rung 2: the SELECTION pass over a committed candidate set.

Rung 1 commits a set of candidate reconstructions and measures every member
with the battery of `scripts/seed_candidate_eval.py`
(`specs/core/geometry/seed-candidate-evaluation.md`).  This pass consumes that stored
evidence -- it re-derives none of it -- and produces a CLEANED SET beside the
release, never in place:

* **Rank.**  Members are ordered within the capture by the fit channels
  together with the non-member translation-direction delta and the
  hold-out depth-agreement rho.  The output is an ORDERING plus a verdict per
  member; no member is ever stamped the winner and redundancy is never a
  reason to drop one.
* **Refuse.**  A member is refused only on defect evidence: a gated channel
  past its population gate with healthy conditioning, or a diverging settling
  refit.  Every refusal names the readings it was taken on.  A CORROBORATING
  channel (scale coherence, the lens deviation) qualifies a verdict other
  evidence already carries and is never a sole ground for one.
* **Trim.**  When the defect localizes to named frames, those frames are
  dropped instead of the member: the points that fall below two supporting
  observations go with them, the cut may not break a link the member's own
  frames had, and the trimmed member is written as a new `.sfmr` beside the
  original.  Three things have to hold.  The evidence must localize -- a channel that speaks about the
  whole member (a held-out image's verdict on it, a diverging refit) is never
  repaired by dropping one of its frames.  Each frame cut must be named by a
  POSE-SHAPED per-frame reading; a coherence reading says something is off,
  not which frame is wrong.  And the surviving core must then pass the member
  gates in its own right, on channels RE-MEASURED over the frames it kept: a
  reading taken while the cut frames were still in the member describes
  geometry the core no longer holds, and cannot answer for it.  The arrays
  rung 1 ships beside the release set are what the core is stated from.
  Where any of them fails the verdict is a refusal, and the
  frames a trim would have dropped are recorded instead.
* **Coverage.**  The capture coverage of the SURVIVING set is computed from
  the frames those members posed, with its gaps.  It is never inferred from
  quality.

Gates are quantiles of a fleet population of the same channels, derived once
by the `derive-gates` mode and passed in as a file.  Each gated channel
carries an ABSOLUTE bar, a FLOOR (a quantile of the members no absolute bar
accuses, which is the channel's noise scale) and a CAPTURE-RELATIVE bar on the
member's reading over its own capture's median.  The absolute bar and the
floor are drawn PER CAMERA-MODEL FAMILY where the family holds enough
readings for its quantile to be a quantile; a fisheye rig and a phone pan do
not share a hold-out delta distribution, and a bar over both is the majority
family's bar imposed on the minority.  A family too small falls back to the
fleet's bar, and the reading says so.  A capture-relative reading
fires only when the absolute reading also clears the floor: dividing by a
capture median guarantees somebody is that capture's loudest member.  A
capture whose median member fails the absolute bars is majority-defective, and
its capture-relative readings are non-measurements -- its median is a broken
member.  A worst-over-median frame spread gates at a stricter quantile than a
member median, because a defect confined to two frames of a dozen only shows
at the extreme of such a reading.

Both of those readings are RANKS.  A fleet quantile ranks the member among the
fleet's members and a capture median ranks it inside its own capture, and a
population of sound members still has a loudest one, so a rank alone hands the
top of a sound population a firing.  A channel may therefore carry a REFUSAL
FLOOR beside its bars: the magnitude the ranks are read next to, taken at the
stricter quantile over the channel's WHOLE gate-eligible population -- not the
clean subset, because a population defined by the bar cannot exonerate a member
that bar accuses.  A reading inside that spread refuses nothing, whichever
reading fired.  Every gated channel carries one, of either model family: a
worst-frame reading is a rank whatever family produced it.

A per-frame reading that COUNTS the evidence a frame's geometry rests on,
instead of measuring that geometry, is read against a SUPPORT FLOOR: the low
decile of the fleet's own per-frame counts, per camera-model family where the
family's own decile is distinguishable from the fleet's.  A frame under it is
one the member posed on almost nothing, so whatever its geometry reads, it
reads on too little to be a reading.  Such a frame is named for a cut whatever
the member's own verdict, because insufficiency is not an accusation: an
unaccused member that loses its starved frames keeps its verdict, and if no
core survives the cut it keeps its frames too.

A reading taken below a conditioning floor (rung 1's resection inlier floor,
triangulation-angle bar, two-view parallax bar, warp vergence floor and
surface support floor) is a NON-MEASUREMENT: it is reported, and it is never
gate-eligible.  So is a channel whose referee is a capture-level quantity when
that capture's own members outvote the referee: half the fleet's
omnidirectional captures disagree with their focal vote unanimously, and a
family cannot be an outlier against a referee it outnumbers.

The catalogue, the gate quantiles and the trim test are all fleet
measurements, written up beside the study corpus.

Usage
-----
Flatten the channels of one or more release directories into a table::

    pixi run -e dev python scripts/exp_seed_rung2.py channels \\
        --out channels.tsv <release-dir> [<release-dir> ...]

Derive the fleet gates from those same release directories::

    pixi run -e dev python scripts/exp_seed_rung2.py derive-gates \\
        --out gates.json <release-dir> [<release-dir> ...]

Run the selection pass over one release directory::

    pixi run -e dev python scripts/exp_seed_rung2.py select \\
        --gates gates.json [--out DIR] [--capture-frames N] <release-dir>

A "release directory" is a workspace's `sfmr/candidate_solves/`: a
`manifest.json` whose `hypotheses[i]` carry an `evaluation` block, plus one
`.sfmr` per committed member.  `select` writes `<release-dir>/rung2/` (or
`--out`) holding `rung2.json` and any trimmed members; it never modifies the
release.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import math
import sys
from collections import namedtuple
from pathlib import Path

import numpy as np

# ── Fixed structure of the pass ─────────────────────────────────────────────
#
# These are the pass's own arithmetic floors, not thresholds on evidence.
# Every threshold on evidence is a fleet quantile and lives in the gates file.

#: Frames a trimmed member must retain.  Below three there is no core to keep.
MIN_CORE_FRAMES = 3
#: Supporting observations a point needs to survive a trim.
MIN_POINT_OBS = 2
#: Points two frames must share before a trim counts them linked.
MIN_LINK_POINTS = 3
#: Largest share of a member's frames a trim may remove.  Past it the defect is
#: the member, not a few frames, and the verdict is a refusal.
MAX_TRIM_FRACTION = 0.5
#: The fleet quantile every gate is taken at, unless `derive-gates` is told
#: otherwise.  One quantile for every member-median channel: a per-channel
#: choice would be a knob fitted to the fleet's labels, and this pass has none.
DEFAULT_QUANTILE = 0.95
#: The quantile the WORST-OVER-MEDIAN frame-spread channels gate at.  A defect
#: confined to two frames of a dozen is diluted to nothing by a member-level
#: median, and the bulk of a spread channel's population is noise: its signal
#: lives at the extreme, so it takes a stricter bar than a member median does.
#: Measured against the ordinary quantile on the fleet: the two arms catch
#: exactly the same labelled defects at exactly the same precision, and the
#: looser one accuses ten more members and trims fifteen more for it.
DEFAULT_HARD_QUANTILE = 0.99
#: The quantile of the CLEAN population (members no absolute gate accuses) that
#: an absolute floor is taken at.  The floor is a noise scale, not an accusation
#: bar: it is what a capture-relative firing must additionally clear, and the
#: median clean member is what "louder than a sound member" means.  Chosen by
#: fleet measurement over p50 / p75 / p90: the median arm catches more of every
#: labelled defect population at the same precision, agrees with more of the
#: human ledger's discards, and refuses none of its keeps.
DEFAULT_FLOOR_QUANTILE = 50.0
#: The fleet percentile every SUPPORT floor is taken at -- the member-level
#: measurable-fraction floors and the per-frame observation count alike.  A
#: support floor is a bar on how much evidence a reading rests on, not on what
#: the reading says, so it is not one of the accusation quantiles: those rank a
#: member among the fleet's members, and this one asks whether there was
#: anything to rank.  The low decile is where the pass reads "the bottom of
#: what the fleet's own readings rest on".
SUPPORT_FLOOR_PERCENTILE = 10.0
#: A capture at or above this share of absolutely-accused members is
#: MAJORITY-DEFECTIVE: its median member fails the absolute gates, so its own
#: median is a broken member and its capture-relative readings say nothing.
MAJORITY_DEFECTIVE_FRACTION = 0.5

_ROT_ONLY = "rotation_only"


# ── Camera-model families ───────────────────────────────────────────────────
#
# An ABSOLUTE bar is a fleet quantile, and a fleet is not one population: a
# 480-pixel fisheye rig and a 4K phone pan do not produce the same hold-out
# delta distribution at equal correctness, so a bar drawn over both is the
# majority family's bar imposed on the minority.  The bar and the noise floor
# are therefore drawn per camera-model family -- but only where the family's
# own bar is DISTINGUISHABLE from the fleet's, because the family that makes
# up most of the fleet has the fleet's bar already and re-drawing it moves the
# bar by sampling noise alone, which flips whichever members happen to sit
# against it.  A capture-relative reading needs no split at all: it is a ratio
# inside one capture, which is already inside one family.

#: Bootstrap resamples behind a family bar's interval, and the sampler's seed.
#: Same population, same interval, every run.
FAMILY_BOOTSTRAP_N, FAMILY_BOOTSTRAP_SEED = 512, 0
#: The interval's coverage.  The family bar is drawn when the FLEET bar falls
#: outside it: the fleet's number is then not a plausible value of this
#: family's own quantile, which is what "this family reads differently" means.
FAMILY_BOOTSTRAP_COVERAGE = 95.0


def camera_family(hyp):
    """The lens family a member was released under.

    The released camera's model name, reduced to the distinction the readings
    actually separate on: whether the map is a fisheye one."""
    model = str(((hyp.get("camera") or {}).get("model")) or "").upper()
    if "FISHEYE" in model:
        return "fisheye"
    if "PINHOLE" in model:
        return "pinhole"
    return "other"


def family_min_n(quantile):
    """Readings a family needs before its own quantile is a quantile.

    Below ``1 / (1 - q)`` readings the ``q`` quantile is not an interpolation
    between two members of the population, it is the population's maximum, and
    a bar at the maximum accuses nobody.  So that is the floor: a family with
    fewer readings than this is read against the fleet's bar instead, and
    every reading taken that way says so."""
    q = float(quantile)
    return int(math.ceil(1.0 / max(1e-9, 1.0 - q))) if q < 1.0 else 0


def quantile_interval(values, quantile, low):
    """The bootstrap interval of one quantile of `values`.

    How far the bar would move if the same family had been sampled again."""
    vals = np.asarray([v for v in values if v is not None], float)
    if vals.size < 2:
        return None
    p = 100.0 * ((1.0 - quantile) if low else quantile)
    rng = np.random.default_rng(FAMILY_BOOTSTRAP_SEED)
    draw = rng.integers(0, vals.size, size=(FAMILY_BOOTSTRAP_N, vals.size))
    boot = np.percentile(vals[draw], p, axis=1)
    tail = (100.0 - FAMILY_BOOTSTRAP_COVERAGE) / 2.0
    return [
        float(np.percentile(boot, tail)),
        float(np.percentile(boot, 100.0 - tail)),
    ]


def family_bars(gate, family):
    """`(absolute, floor, fleet_floored)` for one member's family.

    A family with no bar of its own -- too small for a quantile, or one whose
    bar the fleet's is a plausible value of -- falls back to the fleet's, and
    the reading is flagged, so no verdict hides which population it was taken
    against."""
    fam = (gate.get("by_family") or {}).get(family) or {}
    ab = fam.get("absolute") if fam.get("distinct") else None
    if not ab or ab.get("value") is None:
        return (gate.get("absolute") or {}), (gate.get("floor") or {}), True
    return ab, (fam.get("floor") or gate.get("floor") or {}), False


def refusal_floor_of(gate, family):
    """`(floor, fleet_floored)` -- the magnitude bar below which `gate` cannot
    refuse.

    A family's own floor stands on the same condition its own bar does; a
    family without one reads against the fleet's, and says so."""
    fam = (gate.get("by_family") or {}).get(family) or {}
    if fam.get("distinct"):
        own = fam.get("refusal_floor")
        if own and own.get("value") is not None:
            return own, False
    return (gate.get("refusal_floor") or {}), True


# ── Channel access ──────────────────────────────────────────────────────────


def _num(value):
    """`value` as a finite float, or None."""
    if value is None or isinstance(value, bool):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _blk(ev, name):
    block = ev.get(name)
    return block if isinstance(block, dict) else {}


def _q(values, p):
    vals = [v for v in values if v is not None]
    return float(np.percentile(np.asarray(vals, float), p)) if vals else None


#: One gated defect channel.
#:
#: `eligible` answers "did this member produce a reading this channel's
#: conditioning floor lets us gate on?".  `low` marks a channel whose SMALL
#: values are the defect (a rho, an inlier fraction), so its gate is the
#: fleet's lower quantile and every comparison flips.  `modes` says which
#: readings of the channel may accuse: `abs` for a fleet bar on the value
#: itself, `rel` for the member's value over its own capture's median.  A
#: channel whose absolute level is a property of the CAPTURE rather than of
#: the member (image texture for a warp residual, scene volume for a surface
#: one) carries `rel` alone.  `corroborating` marks a channel that qualifies a
#: verdict other evidence already carries and can never be the sole ground for
#: one.  `hard` marks a worst-over-median frame spread, gated at the stricter
#: quantile.  `support` names the row field whose fleet quantile is this
#: channel's support floor.  `floored` marks a channel that carries a REFUSAL
#: FLOOR: a magnitude below which no reading of it is defect evidence,
#: whichever of the two readings fired.
Channel = namedtuple(
    "Channel",
    "key block field frame eligible low modes corroborating hard support floored",
    defaults=(("abs", "rel"), False, False, None, False),
)


def _loo_eligible(ev):
    loo = _blk(ev, "self_resection")
    return int(loo.get("gated_n") or 0) >= 1


def _loo_measured(ev):
    loo = _blk(ev, "self_resection")
    return int(loo.get("n_measured") or 0) >= 1


def _nm_conditioned(ev):
    nm = _blk(ev, "nonmember_resection")
    return int(nm.get("n_conditioned") or 0) >= 1


def _focal_measurable(ev):
    # A rotation-only layer ships the vote's own focal, so its deviation is
    # zero by construction: it is not a reading, and pooling it into the gate's
    # population would halve the fleet's median for free.
    fv = _blk(ev, "focal_vote")
    return (
        ev.get("model") != _ROT_ONLY
        and bool(fv.get("measurable"))
        and _num(fv.get("f_vote")) is not None
    )


def _support_measured(ev):
    return int(_blk(ev, "support").get("n_frames") or 0) >= 1


def _settling_measurable(ev):
    return bool(_blk(ev, "settling").get("measurable"))


def _warp_conditioned(ev):
    warp = _blk(ev, "warp_epipolar")
    return bool(warp.get("measurable")) and _num(warp.get("nf_cond_med")) is not None


def _measurable_of(name):
    def ok(ev):
        return bool(_blk(ev, name).get("measurable"))

    ok.__name__ = f"_{name}_measurable"
    return ok


def _spread_of(name, field="res"):
    def ok(ev):
        block = _blk(ev, name)
        return (
            bool(block.get("measurable"))
            and (_num(block.get(f"{field}_frame_med")) or 0.0) > 0.0
            and _num(block.get(f"{field}_frame_worst")) is not None
        )

    ok.__name__ = f"_{name}_spread"
    return ok


def _rot_holdout_eligible(ev):
    return int(_blk(ev, "rot_self_resection").get("gated_n") or 0) >= 1


def _rot_nm_witnessed(ev):
    return int(_blk(ev, "rot_nonmember_resection").get("n_witnessed") or 0) >= 1


def _rot_nm_measured(ev):
    return int(_blk(ev, "rot_nonmember_resection").get("n_measured") or 0) >= 1


def _rot_support_measured(ev):
    return ev.get("model") == _ROT_ONLY and bool(_blk(ev, "support").get("measurable"))


#: THE REFUSAL CATALOGUE.  Membership is the fleet measurement of
#: `rung2-firstpass/RUNG2_FIRSTPASS.md`, not a preference: each of these
#: separates a labelled defect population on the CURRENT channels, and the
#: families are kept because they catch disjoint failures (pose weld, wrong
#: block / inversion, scale weld, lens).
GATED_CHANNELS = (
    # -- pose: the hold-out's own worst gate-eligible frame.
    Channel(
        "loo_rot_worst",
        "self_resection",
        "rot_worst_gated",
        "rot_worst_gated_frame",
        _loo_eligible,
        False,
    ),
    Channel(
        "loo_trans_worst",
        "self_resection",
        "trans_worst_gated",
        "trans_worst_gated_frame",
        _loo_eligible,
        False,
    ),
    # -- structure: the depth agreement, in its two forms.  The low-quantile
    #    rho is the REFUSAL form and the median is the RANK form; they are
    #    different instruments and only the first is gated here.
    Channel(
        "loo_logdev_worst",
        "self_resection",
        "retri_logdev_worst",
        "retri_logdev_worst_frame",
        _loo_measured,
        False,
    ),
    Channel(
        "loo_rho_p10",
        "self_resection",
        "retri_rho_p10",
        "retri_rho_min_frame",
        _loo_measured,
        True,
    ),
    # -- generalization to the capture: the conditioned non-member readings.
    Channel(
        "nm_tdir_med",
        "nonmember_resection",
        "tdir_med_conditioned",
        None,
        _nm_conditioned,
        False,
    ),
    Channel(
        "nm_rot_med_cond",
        "nonmember_resection",
        "_rot_med_conditioned",
        None,
        _nm_conditioned,
        False,
    ),
    Channel(
        "nm_inlier_med",
        "nonmember_resection",
        "inlier_med",
        None,
        _nm_conditioned,
        True,
    ),
    # -- settling: how far the member moves when it is allowed to settle, read
    #    at the worst aggregate the spec prescribes for a refit.
    Channel(
        "settling_tdir_worst",
        "settling",
        "tdir_worst",
        "tdir_worst_frame",
        _settling_measurable,
        False,
    ),
    Channel(
        "settling_rot_worst",
        "settling",
        "rot_worst",
        "rot_worst_frame",
        _settling_measurable,
        False,
    ),
    # -- warp: the member's stored affine shapes against its own poses, over
    #    the pairs above the vergence floor.  The residual LEVEL is a property
    #    of the images (texture, blur, patch size) as much as of the solve, so
    #    only the capture-relative reading accuses.
    Channel(
        "warp_nf_cond_med",
        "warp_epipolar",
        "nf_cond_med",
        "nf_worst_frame",
        _warp_conditioned,
        False,
        modes=("rel",),
    ),
    # -- surface: whether the cloud looks like the surfaces it was taken of.
    #    Stranger membership is already a within-member comparison (the
    #    strangers' surface against the point's own place on it), so it carries
    #    an absolute reading too; the shape and adjacency readings are
    #    scene-dependent in absolute form and are capture-relative only.
    Channel(
        "stranger_res_med",
        "stranger_surface",
        "res_med",
        "res_worst_frame",
        _measurable_of("stranger_surface"),
        False,
        support="stranger_frac_measurable",
    ),
    Channel(
        "surface_sv_frame_med",
        "surface_variation",
        "sv_frame_med",
        "sv_worst_frame",
        _measurable_of("surface_variation"),
        False,
        modes=("rel",),
    ),
    # NOT IN THE CATALOGUE: the plain leave-one-out plane residual of the same
    # neighbourhood, at member level, in either of its published forms
    # (`point_res_med` over points, `res_frame_med` over frames).  Measured
    # capture-relative on the fleet as a candidate for this list: it accuses
    # ten members and catches nothing the catalogue does not already have,
    # costing three refusals of which two are GT-clean, so precision falls on
    # every labelled target (`bad` 0.557 -> 0.540, `worst_bad` 0.721 -> 0.698)
    # at unchanged recall.  Its per-frame form stays in `FRAME_CHANNELS`,
    # where it does localize.  It is measured and reported, and it does not
    # gate.
    Channel(
        "vetted_res_med",
        "range_vetted_surface",
        "res_med",
        "res_worst_frame",
        _measurable_of("range_vetted_surface"),
        False,
        modes=("rel",),
    ),
    # -- the weld readings: one frame standing far above its member's own
    #    median.  Gauge-free and capture-free by construction (a ratio inside
    #    one member), and gated at the stricter quantile.
    Channel(
        "stranger_res_spread",
        "stranger_surface",
        "_res_spread",
        "res_worst_frame",
        _spread_of("stranger_surface"),
        False,
        modes=("abs",),
        hard=True,
    ),
    Channel(
        "surface_res_spread",
        "surface_variation",
        "_res_spread",
        "res_worst_frame",
        _spread_of("surface_variation"),
        False,
        modes=("abs",),
        hard=True,
    ),
    # -- scale coherence: the weld reading, from the member's own frames.  It
    #    CORROBORATES: a frame whose support sits at a different distance from
    #    the rest says something is off, and which member or frame is wrong is
    #    a question only a pose-shaped channel answers.
    Channel(
        "sup_depth_logdev_worst",
        "support",
        "depth_log_dev_worst",
        "depth_log_dev_worst_frame",
        _support_measured,
        False,
        corroborating=True,
    ),
    # -- lens.  Also corroborating: the vote is a structure-free reading and
    #    the release a structural one, and a deviation alone does not say
    #    which of the two is wrong.
    Channel(
        "focal_dev",
        "focal_vote",
        "abs_fraction",
        None,
        _focal_measurable,
        False,
        corroborating=True,
    ),
    # ── the rotation-only family ────────────────────────────────────────────
    # A member that claims bearing without range is judged on the rotation-only
    # form of every channel above, and never passed through unjudged.  Only a
    # rotation-only member produces these readings, so each gate's population
    # is that family's on its own without anything having to say so.
    Channel(
        "rot_holdout_rot_worst",
        "rot_self_resection",
        "rot_worst_gated",
        "rot_worst_gated_frame",
        _rot_holdout_eligible,
        False,
        floored=True,
    ),
    Channel(
        "rot_holdout_dir_dev_worst",
        "rot_self_resection",
        "dir_dev_worst_gated",
        "dir_dev_worst_gated_frame",
        _rot_holdout_eligible,
        False,
        floored=True,
    ),
    Channel(
        "rot_nm_rot_med",
        "rot_nonmember_resection",
        "rot_med",
        None,
        _rot_nm_witnessed,
        False,
        floored=True,
    ),
    Channel(
        "rot_nm_inlier_med",
        "rot_nonmember_resection",
        "inlier_med",
        None,
        _rot_nm_measured,
        True,
        floored=True,
    ),
    Channel(
        "rot_settling_rot_worst",
        "rot_settling",
        "rot_worst",
        "rot_worst_frame",
        _measurable_of("rot_settling"),
        False,
        floored=True,
    ),
    # Under a pure rotation the warp is fully predicted, so this residual has
    # no surface term to hide in.  Its LEVEL is still the images', so only the
    # capture-relative reading accuses.
    Channel(
        "rot_warp_full_med",
        "rot_warp",
        "full_med",
        "full_worst_frame",
        _measurable_of("rot_warp"),
        False,
        modes=("rel",),
        floored=True,
    ),
    Channel(
        "rot_warp_spread",
        "rot_warp",
        "_full_spread",
        "full_worst_frame",
        _spread_of("rot_warp", "full"),
        False,
        modes=("abs",),
        hard=True,
        floored=True,
    ),
    # -- cycles: do the member's PAIRWISE relative rotations, each measured
    #    from its own shared rays, close around the covisibility graph's
    #    cycles?  An angle in degrees is a physical quantity and comparable
    #    across captures, so both readings accuse.
    Channel(
        "rot_cycle_res_med",
        "rot_cycles",
        "res_med",
        "res_worst_frame",
        _measurable_of("rot_cycles"),
        False,
        floored=True,
    ),
    Channel(
        "rot_cycle_res_worst",
        "rot_cycles",
        "res_worst",
        "res_worst_frame",
        _measurable_of("rot_cycles"),
        False,
        floored=True,
    ),
    # -- the exact photometric witness: the model's own prediction of the
    #    pairwise image map, checked against the images at the member's stored
    #    keypoints.  How well any window correlates is the images' property
    #    (texture, blur, exposure), so only the capture-relative reading
    #    accuses.
    Channel(
        "rot_photo_res_med",
        "rot_photometric",
        "res_med",
        "res_worst_frame",
        _measurable_of("rot_photometric"),
        False,
        modes=("rel",),
        floored=True,
    ),
    # -- support: a frame the member posed on almost nothing.  Whatever its
    #    rotation reads, it reads on too little to be a reading, and the
    #    channel NAMES that frame, so the repair is a trim.  The starvation is
    #    a ratio inside one member, so it carries no capture scale and gates
    #    absolutely, at the stricter quantile a frame spread takes.
    Channel(
        "rot_sup_obs_deficit",
        "support",
        "obs_deficit_worst",
        "obs_deficit_worst_frame",
        _rot_support_measured,
        False,
        modes=("abs",),
        hard=True,
        floored=True,
    ),
    Channel(
        "rot_sup_obs_min",
        "support",
        "obs_min",
        "obs_min_frame",
        _rot_support_measured,
        True,
        floored=True,
    ),
    # NOT IN THE CATALOGUE: the parallax residue.  A point at infinity is an
    # approximation that holds over a narrow-parallax subset and stops holding
    # over a wider one; a point that stops holding is a candidate for
    # GRADUATION to a finite depth, and a member is valued for its
    # orientations, not for how far its points stay at infinity.  So the
    # residue is measured per point, per frame and per member, recorded as
    # graduation evidence, and it refuses nothing.
)

#: The channels that qualify a verdict but can never be its sole ground, and
#: never name a frame to cut.
CORROBORATING = tuple(c.key for c in GATED_CHANNELS if c.corroborating)

#: Channels whose referee is a CAPTURE-level quantity rather than the member's
#: own geometry.  When the capture's members agree with each other and
#: collectively disagree with the referee, the referee is the outlier and the
#: channel is a non-measurement for every member of that capture.
CAPTURE_REFEREED = ("focal_dev",)

#: Channels that NAME a frame of the member, and so can localize a defect.
#: Everything else in the catalogue is a statement about the whole member --
#: a held-out image's verdict on it, its lens, its refit -- and a member-wide
#: statement is never repaired by dropping one of the member's frames.  A
#: corroborating channel names a frame in its record and still does not
#: localize: it qualifies, it does not accuse.
LOCALIZING = tuple(
    c.key for c in GATED_CHANNELS if c.frame is not None and not c.corroborating
)

#: One per-frame channel.  `eligible` is the per-frame record's own
#: eligibility field, evaluated on that record.
#:
#: `names` marks a reading that may put a frame on a cut list.  Two kinds do:
#: a POSE-SHAPED reading that says the frame's geometry is wrong (a hold-out
#: pose delta, a settling delta, a warp or surface residual), and a SUPPORT
#: reading that counts the evidence the frame's geometry rests on.  The
#: coherence readings are recorded beside them as corroboration -- a frame
#: whose support sits at the wrong distance is evidence that something is off,
#: not evidence about which frame -- and never name one.
#:
#: `support` marks the second kind.  It is not an accusation: it says the
#: member did not measure the frame, which is true whatever else the member
#: reads, so such a frame is named on a member no channel accuses as readily
#: as on one every channel does.  Its bar is a SUPPORT FLOOR, drawn at the
#: fleet's low decile and per camera-model family on the same condition every
#: family bar in this pass is drawn on.
#:
#: `relative` marks a reading whose LEVEL belongs to the capture, not to the
#: frame: it is gated on the frame's value over its own member's median frame,
#: which is a ratio inside one member and so carries no capture scale.  `low`
#: marks a reading whose SMALL values are the defect, so its bar is the
#: population's lower quantile and the comparison flips.
#:
#: `models` names the model families whose records carry the reading, where the
#: block alone does not say: a block only one family publishes is scoped by
#: that fact, and the support census, which both families publish, is scoped
#: here.
FrameChannel = namedtuple(
    "FrameChannel",
    "key block field eligible names relative low quantile support models",
    defaults=(False, None, False, None),
)

#: The per-frame channels.  A channel's population is the frames of the MODEL
#: FAMILY that produces the reading: a finite member and a rotation-only one
#: publish different blocks, and where they publish the same one (the support
#: block) they publish counts on different scales.
FRAME_CHANNELS = (
    FrameChannel(
        "frame_loo_rot", "self_resection", "rot_delta_deg", "gate_eligible", True, False
    ),
    FrameChannel(
        "frame_loo_trans",
        "self_resection",
        "trans_delta_frac",
        "gate_eligible",
        True,
        False,
    ),
    FrameChannel("frame_settling_rot", "settling", "rot_med", None, True, False),
    FrameChannel("frame_settling_tdir", "settling", "tdir_med", None, True, False),
    FrameChannel("frame_warp_nf", "warp_epipolar", "nf_med", None, True, True),
    FrameChannel("frame_stranger_res", "stranger_surface", "res_med", None, True, True),
    FrameChannel("frame_surface_res", "surface_variation", "res_med", None, True, True),
    FrameChannel("frame_depth_logdev", "support", "depth_log_dev", None, False, False),
    FrameChannel("frame_near_ratio", "support", "near_ratio", None, False, False),
    # the rotation-only family
    FrameChannel(
        "frame_rot_holdout_rot",
        "rot_self_resection",
        "rot_delta_deg",
        "gate_eligible",
        True,
        False,
    ),
    FrameChannel(
        "frame_rot_settling_rot", "rot_settling", "rot_med", None, True, False
    ),
    FrameChannel("frame_rot_warp", "rot_warp", "full_med", None, True, True),
    FrameChannel("frame_rot_photo", "rot_photometric", "res_med", None, True, True),
    # A cycle residual implicates every edge of its cycle, so the frame it is
    # attributed to is where to LOOK, not a frame to cut: it is recorded
    # beside the pose-shaped readings and never names one.
    FrameChannel(
        "frame_rot_cycle", "rot_cycles", "cycle_worst_deg", None, False, False
    ),
    # -- the two support readings of a member that claims bearing without
    #    range.  Starvation names a frame by COUNTING what the member posed on
    #    it, not by inferring anything about its geometry.  The deficit is the
    #    count against the member's own median frame, and the count is the
    #    count: a ratio says the frame is unlike its member's others, and only
    #    the count says how much the member held there.  Both belong to the
    #    rotation-only family, as their member-level forms do.
    FrameChannel(
        "frame_rot_obs_deficit",
        "support",
        "obs_deficit",
        None,
        True,
        False,
        models=(_ROT_ONLY,),
    ),
    FrameChannel(
        "frame_rot_obs",
        "support",
        "n_obs",
        None,
        True,
        False,
        low=True,
        quantile=1.0 - SUPPORT_FLOOR_PERCENTILE / 100.0,
        support=True,
        models=(_ROT_ONLY,),
    ),
)

#: The member channels a TRIMMED CORE is re-judged on, recomputed from the
#: member's own per-frame records over the frames the core kept.  Every one is
#: an aggregate rung 1 already published per frame, so re-judging a core is
#: consumption of stored evidence and not a re-measurement of geometry.  The
#: fleet gates are quantiles of the SAME statistic taken over each member's
#: whole frame set, so a core is judged exactly as a member would be.
#:
#: `(key, block, field, aggregate, low, eligibility, modes)`.
CORE_CHANNELS = (
    (
        "core_loo_rot_worst",
        "self_resection",
        "rot_delta_deg",
        "max",
        False,
        "gate_eligible",
        ("abs",),
    ),
    (
        "core_loo_trans_worst",
        "self_resection",
        "trans_delta_frac",
        "max",
        False,
        "gate_eligible",
        ("abs",),
    ),
    (
        "core_loo_logdev_worst",
        "self_resection",
        "retri_logdev_med",
        "max",
        False,
        None,
        ("abs",),
    ),
    ("core_loo_rho_p10", "self_resection", "retri_rho", "p10", True, None, ("abs",)),
    ("core_settling_rot_worst", "settling", "rot_med", "max", False, None, ("abs",)),
    ("core_settling_tdir_worst", "settling", "tdir_med", "max", False, None, ("abs",)),
    ("core_warp_nf_med", "warp_epipolar", "nf_med", "p50", False, None, ("rel",)),
    (
        "core_stranger_res_med",
        "stranger_surface",
        "res_med",
        "p50",
        False,
        None,
        ("rel",),
    ),
    (
        "core_surface_sv_med",
        "surface_variation",
        "sv_med",
        "p50",
        False,
        None,
        ("rel",),
    ),
    (
        "core_rot_holdout_rot_worst",
        "rot_self_resection",
        "rot_delta_deg",
        "max",
        False,
        "gate_eligible",
        ("abs",),
    ),
    (
        "core_rot_settling_rot_worst",
        "rot_settling",
        "rot_med",
        "max",
        False,
        None,
        ("abs",),
    ),
    ("core_rot_warp_med", "rot_warp", "full_med", "p50", False, None, ("rel",)),
    ("core_rot_photo_med", "rot_photometric", "res_med", "p50", False, None, ("rel",)),
    ("core_rot_obs_deficit", "support", "obs_deficit", "max", False, None, ("abs",)),
)

#: The rank families.  Each is (name, [(block, field, higher_is_better)]);
#: a family scores as the mean of the normalized ranks of its fields, and a
#: member's rank score is the mean over the families it could be measured on.
RANK_FAMILIES = (
    ("fit", (("fit", "inlier_2px", True), ("fit", "median_px", False))),
    ("nonmember_tdir", (("nonmember_resection", "tdir_med_conditioned", False),)),
    ("depth_agreement", (("self_resection", "retri_rho_med", True),)),
    # the rotation-only family's own two, alongside the fit family it shares
    ("rot_nonmember", (("rot_nonmember_resection", "rot_med", False),)),
    ("rot_direction_agreement", (("rot_self_resection", "dir_dev_med_deg", False),)),
)


def derived_fields(ev):
    """Readings rung 1 stores per image but does not aggregate.

    Rung 1 publishes `tdir_med_conditioned` but not its rotation twin, and the
    per-image records carry everything the aggregate needs.  Reducing them here
    is consumption of stored evidence, not a re-derivation of it: no geometry
    is recomputed and no artifact is re-read.
    """
    rot = [
        _num(rec.get("rot_delta_deg"))
        for rec in _blk(ev, "nonmember_resection").get("images") or []
        if rec.get("e_conditioned") and _num(rec.get("rot_delta_deg")) is not None
    ]
    out = {
        "nonmember_resection": {
            "_rot_med_conditioned": float(np.median(rot)) if rot else None
        }
    }
    # The frame spread of a surface residual: the member's worst frame over its
    # own median frame.  A ratio inside one member, so no capture scale and no
    # gauge survives into it.
    for block, field, key in (
        ("stranger_surface", "res", "_res_spread"),
        ("surface_variation", "res", "_res_spread"),
        ("rot_warp", "full", "_full_spread"),
    ):
        b = _blk(ev, block)
        med = _num(b.get(f"{field}_frame_med"))
        worst = _num(b.get(f"{field}_frame_worst"))
        out[block] = {key: (worst / med) if (med and med > 0) else None}
    return out


def member_channels(hyp):
    """Flatten one manifest hypothesis into `{channel: value}` plus context.

    Returns `(row, eligible)`: `row` carries every channel this pass reads and
    `eligible` marks the gated channels whose conditioning lets them be gated.
    """
    ev = hyp.get("evaluation") or {}
    fit, loo = _blk(ev, "fit"), _blk(ev, "self_resection")
    nm, settle = _blk(ev, "nonmember_resection"), _blk(ev, "settling")
    sup, fv = _blk(ev, "support"), _blk(ev, "focal_vote")
    warp, strg = _blk(ev, "warp_epipolar"), _blk(ev, "stranger_surface")
    svar, vet = _blk(ev, "surface_variation"), _blk(ev, "range_vetted_surface")
    rsr, rnm = _blk(ev, "rot_self_resection"), _blk(ev, "rot_nonmember_resection")
    rst, rwp = _blk(ev, "rot_settling"), _blk(ev, "rot_warp")
    par = _blk(ev, "parallax_residue")
    row = {
        "idx": int(hyp.get("idx", -1)),
        "model": hyp.get("model", "finite"),
        "family": camera_family(hyp),
        "release_file": hyp.get("release_file"),
        "posed": _num(hyp.get("posed")),
        "points": _num(hyp.get("points")),
        "eval_enabled": bool(ev.get("enabled")),
        # fit
        "fit_inlier_2px": _num(fit.get("inlier_2px")),
        "fit_inlier_4px": _num(fit.get("inlier_4px")),
        "fit_median_px": _num(fit.get("median_px")),
        "fit_p90_px": _num(fit.get("p90_px")),
        "fit_n_obs": _num(fit.get("n_obs")),
        # hold-out self-resection
        "loo_n_measured": _num(loo.get("n_measured")),
        "loo_gated_n": _num(loo.get("gated_n")),
        "loo_inlier_floor": _num(loo.get("inlier_floor")),
        "loo_tri_angle_bar_deg": _num(loo.get("tri_angle_bar_deg")),
        "loo_rho_med": _num(loo.get("retri_rho_med")),
        "loo_rho_p10": _num(loo.get("retri_rho_p10")),
        "loo_rho_min": _num(loo.get("retri_rho_min")),
        # non-member resection
        "nm_n_measured": _num(nm.get("n_measured")),
        "nm_n_conditioned": _num(nm.get("n_conditioned")),
        "nm_parallax_bar_bounds": _num(nm.get("parallax_bar_bounds")),
        "nm_inlier_med": _num(nm.get("inlier_med")),
        # settling
        "settling_measurable": bool(settle.get("measurable")),
        "settling_diverged": bool(settle.get("diverged")),
        "settling_residual_ratio": _num(settle.get("residual_ratio")),
        "settling_rot_worst": _num(settle.get("rot_worst")),
        "settling_tdir_worst": _num(settle.get("tdir_worst")),
        # support / coherence
        "sup_near_ratio_worst": _num(sup.get("near_ratio_worst")),
        "sup_depth_logdev_worst": _num(sup.get("depth_log_dev_worst")),
        "sup_obs_min": _num(sup.get("obs_min")),
        # focal vote
        "focal_signed_fraction": _num(fv.get("signed_fraction")),
        # warp epipolar consistency
        "warp_n_pairs": _num(warp.get("n_pairs")),
        "warp_n_conditioned": _num(warp.get("n_conditioned")),
        "warp_vergence_floor_deg": _num(warp.get("vergence_floor_deg")),
        "warp_verg_med": _num(warp.get("verg_med")),
        "warp_nf_med": _num(warp.get("nf_med")),
        "warp_epi_med": _num(warp.get("epi_med")),
        # surface
        "stranger_frac_measurable": _num(strg.get("frac_measurable")),
        "stranger_res_local_med": _num(strg.get("res_local_med")),
        "stranger_over_local_med": _num(strg.get("over_local_med")),
        "surface_point_sv_med": _num(svar.get("point_sv_med")),
        "vetted_frac_measurable": _num(vet.get("frac_measurable_med")),
        # rotation-only
        "rot_holdout_n_measured": _num(rsr.get("n_measured")),
        "rot_holdout_gated_n": _num(rsr.get("gated_n")),
        "rot_holdout_rot_med": _num(rsr.get("rot_med")),
        "rot_holdout_dir_dev_med": _num(rsr.get("dir_dev_med_deg")),
        "rot_holdout_spread_med_deg": _num(rsr.get("support_spread_med_deg")),
        "rot_nm_n_measured": _num(rnm.get("n_measured")),
        "rot_nm_n_witnessed": _num(rnm.get("n_witnessed")),
        "rot_settling_measurable": bool(rst.get("measurable")),
        "rot_settling_diverged": bool(rst.get("diverged")),
        "rot_settling_residual_ratio": _num(rst.get("residual_ratio")),
        "rot_warp_n_pairs": _num(rwp.get("n_pairs")),
        "rot_warp_frame_med": _num(rwp.get("full_frame_med")),
        "parallax_radial_null": _num(par.get("radial_null")),
        "parallax_rejected_frac_med": _num(par.get("rejected_frac_med")),
        "parallax_res_med_px": _num(par.get("res_med_px")),
        "parallax_n_measured": _num(par.get("n_measured")),
    }
    # The member's own per-frame observation counts, keyed by full relative
    # path, so a reading that names a frame can say what the member held there.
    row["_frame_obs"] = {
        rec.get("name"): _num(rec.get("n_obs"))
        for rec in sup.get("frames") or []
        if rec.get("name")
    }
    extra = derived_fields(ev)
    eligible = {}
    for c in GATED_CHANNELS:
        raw = extra.get(c.block, {}).get(c.field, _blk(ev, c.block).get(c.field))
        row[c.key] = _num(raw)
        eligible[c.key] = bool(c.eligible(ev)) and row[c.key] is not None
    # Rank inputs that are not gated channels.
    row["nm_tdir_med_all"] = _num(nm.get("tdir_med"))
    row["nm_rot_med_all"] = _num(nm.get("rot_med"))
    return row, eligible


def frame_rows(hyp):
    """Per-frame readings of one member, keyed by full relative frame path.

    A `member_relative` channel is stored twice: the raw reading, and the
    reading over the member's own median frame, which is the form its gate is
    a quantile of."""
    ev = hyp.get("evaluation") or {}
    out = {}
    for c in FRAME_CHANNELS:
        recs = _blk(ev, c.block).get("frames") or []
        med = None
        if c.relative:
            med = _q([_num(r.get(c.field)) for r in recs], 50)
        for rec in recs:
            name = rec.get("name")
            if not name:
                continue
            slot = out.setdefault(name, {})
            value = _num(rec.get(c.field))
            eligible = True if c.eligible is None else bool(rec.get(c.eligible))
            if rec.get("status") not in (None, "ok"):
                eligible = False
            if c.relative:
                slot[c.key + "__raw"] = value
                value = (
                    (value / med) if (value is not None and med and med > 0) else None
                )
            slot[c.key] = value
            slot[c.key + "__eligible"] = eligible and value is not None
    return out


def _agg_over(values, how):
    """`max`, `p50` or `p10` of the finite readings, or None when empty."""
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    if how == "max":
        return float(max(vals))
    return float(np.percentile(np.asarray(vals, float), 50 if how == "p50" else 10))


def core_channels(hyp, keep_names=None):
    """The core channels of one member over `keep_names` (all frames if None).

    The same aggregates rung 1 published at member level, recomputed from the
    per-frame records so that a trimmed core can be judged as a member."""
    ev = hyp.get("evaluation") or {}
    out = {}
    for key, block, field, how, _low, elig_field, _modes in CORE_CHANNELS:
        vals = []
        for rec in _blk(ev, block).get("frames") or []:
            name = rec.get("name")
            if keep_names is not None and name not in keep_names:
                continue
            if elig_field is not None and not rec.get(elig_field):
                continue
            if rec.get("status") not in (None, "ok"):
                continue
            vals.append(_num(rec.get(field)))
        out[key] = _agg_over(vals, how)
    return out


# ── Release directory ───────────────────────────────────────────────────────


def load_release(path):
    """`(manifest, hypotheses)` of a release directory."""
    path = Path(path)
    man = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    return man, list(man.get("hypotheses") or [])


def entry_name(release_dir):
    """A stable label for a release: the workspace directory it belongs to."""
    p = Path(release_dir).resolve()
    for parent in p.parents:
        if parent.name not in ("candidate_solves", "sfmr", "results"):
            return parent.name
    return p.name


# ── Gate derivation ─────────────────────────────────────────────────────────


def collect_population(release_dirs):
    """Pool the member and per-frame channel readings of a fleet."""
    members, frames = [], []
    for rd in release_dirs:
        try:
            _man, hyps = load_release(rd)
        except (OSError, ValueError) as exc:
            print(f"  skip {rd}: {type(exc).__name__}: {exc}", file=sys.stderr)
            continue
        entry = entry_name(rd)
        for hyp in hyps:
            row, eligible = member_channels(hyp)
            if not row["eval_enabled"]:
                continue
            row["entry"], row["release_dir"] = entry, str(rd)
            row["_eligible"] = eligible
            row.update(core_channels(hyp))
            members.append(row)
            for name, slot in frame_rows(hyp).items():
                frames.append(
                    {
                        "entry": entry,
                        "idx": row["idx"],
                        "model": row["model"],
                        "family": row["family"],
                        "frame": name,
                        **slot,
                    }
                )
    return members, frames


def _gate_stats(values, quantile, low=False):
    vals = sorted(v for v in values if v is not None)
    if not vals:
        return None
    arr = np.asarray(vals, float)
    # A low-side channel's gate sits at the mirrored quantile: the defect is
    # the fleet's bottom tail, so the same `quantile` means the same share of
    # the population accused.
    p = 100.0 * ((1.0 - quantile) if low else quantile)
    return {
        "value": float(np.percentile(arr, p)),
        "quantile": quantile,
        "low": bool(low),
        "n": int(arr.size),
        "p50": float(np.percentile(arr, 50)),
        "p75": float(np.percentile(arr, 75)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "max": float(arr.max()),
    }


def capture_median(members_of_capture, key, eligible_only=True):
    """The capture's own median reading of one channel, or None."""
    pop = [
        m[key]
        for m in members_of_capture
        if m.get(key) is not None
        and (not eligible_only or m.get("_eligible", {}).get(key, True))
    ]
    return float(np.median(pop)) if pop else None


def capture_ratio(value, med, low):
    """A member's reading over its capture's median, oriented so LARGER is
    worse whichever side of the channel the defect sits on."""
    if value is None or med is None:
        return None
    if low:
        return (med / value) if value > 0 else None
    return (value / med) if med > 0 else None


def absolute_hits(row, gates):
    """The non-corroborating channels whose ABSOLUTE reading accuses `row`.

    Read against the member's own camera-model family wherever that family is
    large enough to have a quantile of its own."""
    mg = gates.get("member_gates", {})
    out = []
    for c in GATED_CHANNELS:
        if c.corroborating or "abs" not in c.modes:
            continue
        gate, value = mg.get(c.key), row.get(c.key)
        if gate is None or value is None or not row["_eligible"].get(c.key):
            continue
        family = row.get("family", "other")
        ab, _floor, _fleet = family_bars(gate, family)
        bar = ab.get("value")
        if bar is None:
            continue
        # A reading the refusal floor silences is not an accusation, so it is
        # not one here either: this function is what "an accused member" means
        # for the clean population and for a majority-defective capture, and
        # those two have to mean what a verdict means.
        rbar = refusal_floor_of(gate, family)[0].get("value")
        if rbar is not None and (value >= rbar if c.low else value <= rbar):
            continue
        if value < bar if c.low else value > bar:
            out.append(c.key)
    return out


def majority_defective(members_of_capture, gates):
    """Does this capture's MEDIAN member fail the absolute gates?

    When it does the capture is majority-defective: its own median is a broken
    member's reading, so nothing measured against that median is a reading of
    a member."""
    if not members_of_capture:
        return False
    hits = [1 if absolute_hits(m, gates) else 0 for m in members_of_capture]
    return float(np.mean(hits)) >= MAJORITY_DEFECTIVE_FRACTION


def capture_referee_ok(members_of_capture, key, gate_value):
    """Is the capture-level referee of `key` usable for this capture?

    `False` when the capture's own members agree with each other and
    collectively sit past the gate: a whole family cannot be an outlier
    against a referee that the family itself outnumbers, so the referee is
    what failed.  Half the fleet's omnidirectional captures read this way
    against the focal vote.
    """
    pop = [m[key] for m in members_of_capture if m["_eligible"].get(key)]
    if not pop or gate_value is None:
        return True
    return float(np.median(pop)) <= gate_value


def _absolute_gate(members, c, quantile, family=None):
    """One channel's absolute bar, over its gate-eligible population.

    Over the whole fleet, or over one camera-model family of it."""
    pop = [
        m[c.key]
        for m in members
        if m["_eligible"].get(c.key) and (family is None or m.get("family") == family)
    ]
    if family is not None and len(pop) < family_min_n(quantile):
        return None
    return _gate_stats(pop, quantile, c.low)


def derive_gates(
    release_dirs,
    quantile=DEFAULT_QUANTILE,
    hard_quantile=DEFAULT_HARD_QUANTILE,
    floor_quantile=DEFAULT_FLOOR_QUANTILE,
):
    """Every gate this pass uses, as a quantile of the fleet's own readings.

    Each gated channel gets THREE numbers.

    * The **absolute** bar: the fleet quantile of the channel's gate-eligible
      population, at the ordinary quantile or, for a worst-over-median frame
      spread, at the stricter one -- and the same quantile again over each
      CAMERA-MODEL FAMILY that holds enough readings for the quantile to be
      one.  A member is read against its own family's bar; a family too small
      for its own falls back to the fleet's, and every reading taken that way
      is flagged.
    * The **floor**: a quantile of the CLEAN population -- the members no
      absolute bar accuses -- which is the channel's own noise scale, taken
      per family alongside the family's bar.  A capture-relative firing has to
      clear it, because dividing by a capture median guarantees somebody is
      that capture's loudest member.
    * The **relative** bar: the fleet quantile of the members' readings over
      their own capture's median, pooled over the captures whose median member
      passes the absolute bars.  A majority-defective capture's median is a
      broken member and contributes nothing here.

    A capture-refereed channel is additionally derived to a FIXPOINT: take the
    quantile, drop the captures whose own median sits past it (those members
    are readings of a broken referee, not of a member), re-take, repeat.
    """
    members, frames = collect_population(release_dirs)
    by_capture = {}
    # A capture's two model families are two populations: a finite member and a
    # rotation-only one share no gated channel, so a median over both would be
    # a median over whichever family happened to produce the reading.
    by_group = {}
    for m in members:
        by_capture.setdefault(m["entry"], []).append(m)
        by_group.setdefault((m["entry"], m["model"]), []).append(m)

    # -- 1. the absolute bars -------------------------------------------------
    member_gates = {}
    for c in GATED_CHANNELS:
        q = hard_quantile if c.hard else quantile
        stats = _absolute_gate(members, c, q)
        if stats is None:
            continue
        stats["source"] = (
            f"fleet quantile over gate-eligible members ({len(release_dirs)} releases)"
        )
        stats["hard"] = bool(c.hard)
        # THE REFUSAL FLOOR.  Both of a channel's readings are RANKS -- the
        # absolute bar ranks the member among the fleet's, the capture-relative
        # one ranks it inside its own capture -- and a population of sound
        # members still has a loudest.  The floor is the magnitude those ranks
        # are read beside: the top of what this channel's own population
        # produces, at the stricter quantile a worst-of-many statistic takes,
        # over the WHOLE gate-eligible population and not the clean subset,
        # because a population defined by the bar cannot exonerate a member
        # that bar accuses.
        refusal = _absolute_gate(members, c, hard_quantile) if c.floored else None
        if refusal is not None:
            refusal["source"] = (
                "fleet hard quantile over the channel's whole gate-eligible "
                "population: the magnitude a rank is read beside"
            )
        if c.key in CAPTURE_REFEREED:
            provisional, dropped = stats["value"], set()
            for _round in range(8):
                more = {
                    entry
                    for entry, ms in by_capture.items()
                    if not capture_referee_ok(ms, c.key, stats["value"])
                }
                if more == dropped:
                    break
                dropped = more
                pop2 = [
                    m[c.key]
                    for m in members
                    if m["_eligible"].get(c.key) and m["entry"] not in dropped
                ]
                nxt = _gate_stats(pop2, q, c.low)
                if nxt is None:
                    break
                stats = nxt
            stats["source"] = (
                "fleet quantile over gate-eligible members of the captures whose "
                "referee is usable (fixpoint)"
            )
            stats["provisional_value"] = provisional
            stats["captures_with_broken_referee"] = sorted(dropped)
        member_gates[c.key] = {"absolute": stats, "modes": list(c.modes)}
        if refusal is not None:
            member_gates[c.key]["refusal_floor"] = refusal
        # THE SAME BAR, PER CAMERA-MODEL FAMILY.  A family that cannot fill a
        # quantile of its own is left out here and falls back to the fleet's
        # bar at read time, flagged.  A capture-refereed channel keeps the
        # fleet fixpoint: the referee it is read against is capture-level, not
        # lens-level, so splitting the population would not split the referee.
        if c.key not in CAPTURE_REFEREED:
            by_family = {}
            for fam in sorted({m.get("family", "other") for m in members}):
                pop = [
                    m[c.key]
                    for m in members
                    if m["_eligible"].get(c.key) and m.get("family") == fam
                ]
                fstats = _absolute_gate(members, c, q, family=fam)
                if fstats is None:
                    by_family[fam] = {
                        "absolute": None,
                        "distinct": False,
                        "n": len(pop),
                        "reason": (
                            f"{len(pop)} readings below the {family_min_n(q)} a "
                            f"q{q:g} quantile needs to be one"
                        ),
                    }
                    continue
                fstats["source"] = (
                    f"{fam} quantile over that family's gate-eligible members"
                )
                fstats["hard"] = bool(c.hard)
                fstats["family"] = fam
                ci = quantile_interval(pop, q, c.low)
                fleet = stats["value"]
                distinct = bool(ci is not None and (fleet < ci[0] or fleet > ci[1]))
                fstats["interval"] = ci
                fstats["fleet_value"] = fleet
                rfam = (
                    _absolute_gate(members, c, hard_quantile, family=fam)
                    if c.floored
                    else None
                )
                if rfam is not None:
                    rfam["source"] = (
                        f"{fam} hard quantile over that family's whole "
                        "gate-eligible population"
                    )
                    rfam["family"] = fam
                by_family[fam] = {
                    "absolute": fstats,
                    "refusal_floor": rfam,
                    "distinct": distinct,
                    "n": len(pop),
                    "reason": (
                        "this family's bar and the fleet's are the same "
                        "measurement within the bar's own sampling interval"
                    )
                    if not distinct
                    else "the fleet bar is outside this family's own interval",
                }
            if by_family:
                member_gates[c.key]["by_family"] = by_family
        member_gates[c.key]["family_min_n"] = family_min_n(q)

    # -- 2. the floors, over the members no absolute bar accuses --------------
    provisional = {"member_gates": member_gates}
    clean = [m for m in members if not absolute_hits(m, provisional)]
    for c in GATED_CHANNELS:
        gate = member_gates.get(c.key)
        if gate is None:
            continue
        p = floor_quantile if not c.low else (100.0 - floor_quantile)

        def _floor(pop, p=p):
            return {
                "value": float(np.percentile(np.asarray(pop, float), p))
                if pop
                else None,
                "quantile": floor_quantile,
                "n_clean": len(pop),
                "source": (
                    "quantile of the members no absolute gate accuses: the "
                    "channel's own noise scale"
                ),
            }

        gate["floor"] = _floor([m[c.key] for m in clean if m["_eligible"].get(c.key)])
        # A family with a bar of its own gets a noise floor of its own: the
        # floor is that channel's noise scale, and noise scales differ between
        # lens families for the same reason the bars do.
        for fam, slot in (gate.get("by_family") or {}).items():
            if not slot.get("distinct"):
                continue
            slot["floor"] = _floor(
                [
                    m[c.key]
                    for m in clean
                    if m["_eligible"].get(c.key) and m.get("family") == fam
                ]
            )

    # -- 3. the capture-relative bars ----------------------------------------
    defective = sorted(
        key for key, ms in by_group.items() if majority_defective(ms, provisional)
    )
    for c in GATED_CHANNELS:
        gate = member_gates.get(c.key)
        if gate is None or "rel" not in c.modes:
            continue
        ratios = []
        for group, ms in by_group.items():
            if group in defective:
                continue
            med = capture_median(ms, c.key)
            for m in ms:
                if not m["_eligible"].get(c.key):
                    continue
                r = capture_ratio(m[c.key], med, c.low)
                if r is not None:
                    ratios.append(r)
        stats = _gate_stats(ratios, hard_quantile if c.hard else quantile, False)
        if stats is None:
            continue
        stats["source"] = (
            "fleet quantile of the member's reading over its own capture's "
            "median, over the captures whose median member passes the absolute "
            "gates"
        )
        gate["relative"] = stats

    # -- 4. the support floors -----------------------------------------------
    support_floors = {}
    for c in GATED_CHANNELS:
        if c.support is None:
            continue
        pop = [m[c.support] for m in members if m.get(c.support) is not None]
        support_floors[c.key] = {
            "field": c.support,
            "value": float(
                np.percentile(np.asarray(pop, float), SUPPORT_FLOOR_PERCENTILE)
            )
            if pop
            else None,
            "quantile": SUPPORT_FLOOR_PERCENTILE,
            "n": len(pop),
            "source": (
                f"fleet p{SUPPORT_FLOOR_PERCENTILE:g} of the channel's own "
                "measurable fraction"
            ),
        }

    # -- 5. per-frame gates, per MODEL FAMILY --------------------------------
    # A frame gate is a quantile of the frames that produce the reading, and
    # only one model family produces each: the two families publish different
    # blocks, and the one block they share (the support census) they fill on
    # different scales.  A SUPPORT floor is additionally drawn per CAMERA-MODEL
    # family, on the same condition every family bar in this pass is drawn on,
    # because it is a floor and this pass draws floors per lens family.
    frame_gates = {}
    for model in sorted({f["model"] for f in frames}):
        of_model = [f for f in frames if f["model"] == model]
        slot = {}
        for c in FRAME_CHANNELS:
            if c.models is not None and model not in c.models:
                continue
            q = c.quantile if c.quantile is not None else quantile
            pop = [f[c.key] for f in of_model if f.get(c.key + "__eligible")]
            stats = _gate_stats(pop, q, c.low)
            if stats is None:
                continue
            stats["source"] = (
                f"{model} quantile over gate-eligible frames "
                f"({len(release_dirs)} releases)"
                + (", read over the frame's own member median" if c.relative else "")
            )
            stats["names_frame"] = bool(c.names)
            stats["member_relative"] = bool(c.relative)
            stats["support"] = bool(c.support)
            if c.support:
                by_family = {}
                for fam in sorted({f.get("family", "other") for f in of_model}):
                    fpop = [
                        f[c.key]
                        for f in of_model
                        if f.get(c.key + "__eligible") and f.get("family") == fam
                    ]
                    if len(fpop) < family_min_n(q):
                        by_family[fam] = {
                            "absolute": None,
                            "distinct": False,
                            "n": len(fpop),
                            "reason": (
                                f"{len(fpop)} readings below the {family_min_n(q)} a "
                                f"q{q:g} quantile needs to be one"
                            ),
                        }
                        continue
                    fstats = _gate_stats(fpop, q, c.low)
                    fstats["source"] = (
                        f"{fam} quantile over that family's gate-eligible frames"
                    )
                    fstats["family"] = fam
                    ci = quantile_interval(fpop, q, c.low)
                    fleet = stats["value"]
                    distinct = bool(ci is not None and (fleet < ci[0] or fleet > ci[1]))
                    fstats["interval"] = ci
                    fstats["fleet_value"] = fleet
                    by_family[fam] = {
                        "absolute": fstats,
                        "distinct": distinct,
                        "n": len(fpop),
                        "reason": (
                            "this family's floor and the fleet's are the same "
                            "measurement within the floor's own sampling interval"
                        )
                        if not distinct
                        else "the fleet floor is outside this family's own interval",
                    }
                stats["by_family"] = by_family
                stats["family_min_n"] = family_min_n(q)
            slot[c.key] = stats
        frame_gates[model] = slot

    # -- 6. the gates a trimmed core is re-judged on --------------------------
    core_gates = {}
    for key, _block, _field, _how, low, _elig, modes in CORE_CHANNELS:
        entry = {"modes": list(modes), "low": bool(low)}
        if "abs" in modes:
            pop = [m.get(key) for m in members]
            stats = _gate_stats(pop, quantile, low)
            if stats is None:
                continue
            stats["source"] = (
                "fleet quantile of the same aggregate over each member's whole "
                "frame set"
            )
            entry["absolute"] = stats
        if "rel" in modes:
            ratios = []
            for group, ms in by_group.items():
                if group in defective:
                    continue
                med = capture_median(ms, key, eligible_only=False)
                for m in ms:
                    r = capture_ratio(m.get(key), med, low)
                    if r is not None:
                        ratios.append(r)
            stats = _gate_stats(ratios, quantile, False)
            if stats is None:
                continue
            stats["source"] = (
                "fleet quantile of the same aggregate over its own capture's median"
            )
            entry["relative"] = stats
        core_gates[key] = entry

    return {
        "schema": "seed-rung2-gates/4",
        "generated": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "quantile": quantile,
        "hard_quantile": hard_quantile,
        "floor_quantile": floor_quantile,
        "population": {
            "n_releases": len(release_dirs),
            "n_members": len(members),
            "n_clean_members": len(clean),
            "n_frames": len(frames),
            "entries": sorted({m["entry"] for m in members}),
            "n_by_model": {
                fam: sum(1 for m in members if m["model"] == fam)
                for fam in sorted({m["model"] for m in members})
            },
            "n_by_family": {
                fam: sum(1 for m in members if m.get("family") == fam)
                for fam in sorted({m.get("family", "other") for m in members})
            },
            "family_min_n": {
                "quantile": family_min_n(quantile),
                "hard_quantile": family_min_n(hard_quantile),
            },
            "majority_defective_groups": [f"{e}|{fam}" for e, fam in defective],
        },
        "member_gates": member_gates,
        "support_floors": support_floors,
        "frame_gates": frame_gates,
        "core_gates": core_gates,
    }


# ── Ranking ─────────────────────────────────────────────────────────────────


def _norm_ranks(values, higher_is_better):
    """Normalized ranks in `[0, 1]`, 1 = best.  None stays None."""
    have = [(i, v) for i, v in enumerate(values) if v is not None]
    out = [None] * len(values)
    if not have:
        return out
    if len(have) == 1:
        out[have[0][0]] = 1.0
        return out
    have.sort(key=lambda kv: kv[1], reverse=higher_is_better)
    for pos, (i, _v) in enumerate(have):
        out[i] = 1.0 - pos / (len(have) - 1)
    return out


def rank_members(rows):
    """Order `rows` (one model family) and annotate each with its rank score.

    The rule: within the capture, each rank family's fields become normalized
    ranks, a family scores as the mean of its fields, and a member's score is
    the mean over the families it could be measured on.  It is an ordering, not
    a verdict -- a member with no measurable family still gets a place, marked
    with the families it was scored on.
    """
    per_family = {}
    for name, fields in RANK_FAMILIES:
        parts = []
        for block, field, better in fields:
            key = {
                ("fit", "inlier_2px"): "fit_inlier_2px",
                ("fit", "median_px"): "fit_median_px",
                ("nonmember_resection", "tdir_med_conditioned"): "nm_tdir_med",
                ("self_resection", "retri_rho_p10"): "loo_rho_p10",
                ("self_resection", "retri_rho_med"): "loo_rho_med",
                ("rot_nonmember_resection", "rot_med"): "rot_nm_rot_med",
                (
                    "rot_self_resection",
                    "dir_dev_med_deg",
                ): "rot_holdout_dir_dev_med",
            }[(block, field)]
            parts.append(_norm_ranks([r.get(key) for r in rows], better))
        per_family[name] = [
            None
            if all(p[i] is None for p in parts)
            else float(np.mean([p[i] for p in parts if p[i] is not None]))
            for i in range(len(rows))
        ]
    order = []
    for i, row in enumerate(rows):
        got = {n: v[i] for n, v in per_family.items() if v[i] is not None}
        row["rank_families"] = sorted(got)
        row["rank_scores"] = {k: round(v, 4) for k, v in got.items()}
        row["rank_score"] = float(np.mean(list(got.values()))) if got else None
        order.append(i)
    order.sort(
        key=lambda i: (
            -(rows[i]["rank_score"] if rows[i]["rank_score"] is not None else -1.0),
            -(rows[i].get("fit_inlier_2px") or 0.0),
            rows[i]["idx"],
        )
    )
    for place, i in enumerate(order):
        rows[i]["rank"] = place
    return [rows[i] for i in order]


# ── Verdicts ────────────────────────────────────────────────────────────────


def member_evidence(
    row, gates, broken_referees=(), capture_meds=None, capture_defective=False
):
    """The defect evidence against one member: a list of named readings.

    A reading is evidence only when its conditioning made it gate-eligible.
    Everything else is reported as a non-measurement and can never refuse.
    `broken_referees` names the capture-refereed channels whose referee this
    capture's own members outvoted; those are non-measurements too, and so is
    every capture-relative reading of a majority-defective capture.
    """
    evidence, blocked = [], []
    mg = gates.get("member_gates", {})
    floors = gates.get("support_floors", {})
    capture_meds = capture_meds or {}
    for c in GATED_CHANNELS:
        value, gate = row.get(c.key), mg.get(c.key)
        if gate is None:
            continue
        frame = row.get("_frames", {}).get(c.frame)
        # What the member holds on the frame the reading names.  A reading is
        # not more or less true for it, but a worst frame the member posed on
        # a handful of observations is a different object from one it posed on
        # a hundred, and the verdict says which it was.
        frame_obs = (row.get("_frame_obs") or {}).get(frame)
        kind = "focal_vote_outlier" if c.key == "focal_dev" else "gated_worst_channel"
        if c.key in broken_referees:
            blocked.append(
                {
                    "channel": c.key,
                    "value": value,
                    "reason": "capture-level referee outvoted by its own members",
                }
            )
            continue
        if not row["_eligible"].get(c.key):
            blocked.append(
                {"channel": c.key, "value": value, "reason": "conditioning-limited"}
            )
            continue
        sup = floors.get(c.key)
        if sup and sup.get("value") is not None:
            have = row.get(sup["field"])
            if have is None or have < sup["value"]:
                blocked.append(
                    {
                        "channel": c.key,
                        "value": value,
                        "reason": (
                            f"support {have} below the fleet floor {sup['value']:.4g} "
                            f"on {sup['field']}"
                        ),
                    }
                )
                continue
        # THE BAR THIS MEMBER'S LENS FAMILY SETS, where the family is large
        # enough to set one, and the fleet's otherwise -- flagged, so a
        # verdict never hides which population it was taken against.
        family = row.get("family", "other")
        abs_gate, floor_gate, fleet_floored = family_bars(gate, family)
        bar = abs_gate.get("value")
        # THE REFUSAL FLOOR, ahead of either reading.  A fleet quantile of the
        # members' readings is a rank among members and a capture median is a
        # rank inside one capture; both hand the loudest member of a sound
        # population a firing.  A reading that sits inside the spread this
        # channel's own family produces is that loudest member, not a defect,
        # and it refuses nothing whichever reading fired.
        rfloor, rfloor_fleet = refusal_floor_of(gate, family)
        rbar = rfloor.get("value")
        if (
            rbar is not None
            and value is not None
            and (value >= rbar if c.low else value <= rbar)
        ):
            blocked.append(
                {
                    "channel": c.key,
                    "value": value,
                    "reason": (
                        f"inside the {family if not rfloor_fleet else 'fleet'} "
                        f"spread of this channel: refusal floor {rbar:.4g} at "
                        f"q{rfloor.get('quantile', 0):g} over "
                        f"{rfloor.get('n', 0)} readings"
                    ),
                }
            )
            continue
        past_abs = (
            "abs" in c.modes
            and value is not None
            and bar is not None
            and (value < bar if c.low else value > bar)
        )
        if past_abs:
            evidence.append(
                {
                    "channel": c.key,
                    "reading": "absolute",
                    "value": value,
                    "gate": bar,
                    "side": "below" if c.low else "above",
                    "quantile": abs_gate.get("quantile"),
                    "population_n": abs_gate.get("n"),
                    "family": family,
                    "fleet_floored": bool(fleet_floored),
                    "refusal_floor": rbar,
                    "frame": frame,
                    "frame_support": frame_obs,
                    "kind": kind,
                    "corroborating": bool(c.corroborating),
                }
            )
        # THE CAPTURE-RELATIVE READING, and the two things that silence it: a
        # capture whose own median member is broken has no usable denominator,
        # and a member quieter than the channel's noise floor is not loud, it
        # is merely its capture's loudest.
        rel_gate = gate.get("relative") or {}
        floor = floor_gate.get("value")
        med = capture_meds.get(c.key)
        ratio = capture_ratio(value, med, c.low)
        if "rel" not in c.modes or past_abs or ratio is None or not rel_gate:
            continue
        if capture_defective:
            blocked.append(
                {
                    "channel": c.key,
                    "value": value,
                    "ratio": ratio,
                    "reason": "capture is majority-defective: its median is not a bar",
                }
            )
            continue
        over_floor = floor is None or (value < floor if c.low else value > floor)
        if ratio > rel_gate["value"] and over_floor:
            evidence.append(
                {
                    "channel": c.key,
                    "reading": "capture-relative",
                    "value": value,
                    "ratio": ratio,
                    "capture_median": med,
                    "gate": rel_gate["value"],
                    "floor": floor,
                    "side": "above",
                    "quantile": rel_gate.get("quantile"),
                    "population_n": rel_gate.get("n"),
                    "family": family,
                    "fleet_floored": bool(fleet_floored),
                    "refusal_floor": rbar,
                    "frame": frame,
                    "frame_support": frame_obs,
                    "kind": kind,
                    "corroborating": bool(c.corroborating),
                }
            )
        elif ratio > rel_gate["value"]:
            blocked.append(
                {
                    "channel": c.key,
                    "value": value,
                    "ratio": ratio,
                    "reason": (
                        f"capture-relative reading past {rel_gate['value']:.4g} but "
                        f"absolute value inside the channel's noise floor {floor:.4g}"
                    ),
                }
            )
    for tag in ("settling", "rot_settling"):
        if row.get(f"{tag}_measurable") and row.get(f"{tag}_diverged"):
            evidence.append(
                {
                    "channel": f"{tag}_diverged",
                    "reading": "absolute",
                    "value": row.get(f"{tag}_residual_ratio"),
                    "gate": 1.0,
                    "quantile": None,
                    "population_n": None,
                    "frame": None,
                    "kind": "diverging_refit",
                    "corroborating": False,
                }
            )
    return evidence, blocked


def frame_bar(gate, family):
    """`(value, fleet_floored)` -- the bar one frame gate sets for a family.

    A support floor is drawn per camera-model family on the same condition
    every family bar in this pass is drawn on; a family without one of its own
    reads the fleet's, and says so."""
    fam = (gate.get("by_family") or {}).get(family) or {}
    if fam.get("distinct"):
        own = fam.get("absolute") or {}
        if own.get("value") is not None:
            return own.get("value"), False
    return gate.get("value"), True


def defect_frames(hyp, gates, model, family="other"):
    """Frames a member's own per-frame readings name, with their evidence.

    Returns `(named, corroborated)`: `named` holds the frames a NAMING
    per-frame channel puts on the cut list, and `corroborated` holds the
    coherence hits on all frames, recorded beside them.  A coherence reading
    says something is off; it does not say which frame is wrong, and it never
    puts a frame on the cut list.

    Each hit records whether it is a SUPPORT reading -- a count of the evidence
    the frame rests on rather than a measurement of its geometry -- because the
    two answer to different rules: a pose-shaped reading names a frame where
    the member's own defect must localize, and a support reading names a frame
    the member did not measure, which is true of a member nothing accuses.
    """
    fg = (gates.get("frame_gates") or {}).get(model) or {}
    named, corroborated = {}, {}
    for name, slot in frame_rows(hyp).items():
        hits, soft = [], []
        for c in FRAME_CHANNELS:
            gate, value = fg.get(c.key), slot.get(c.key)
            if gate is None or value is None or not slot.get(c.key + "__eligible"):
                continue
            bar, fleet_floored = frame_bar(gate, family)
            if bar is None or (value >= bar if c.low else value <= bar):
                continue
            hit = {
                "channel": c.key,
                "value": value,
                "raw": slot.get(c.key + "__raw"),
                "gate": bar,
                "side": "below" if c.low else "above",
                "quantile": gate["quantile"],
                "names_frame": bool(c.names),
                "support": bool(c.support),
                "fleet_floored": bool(fleet_floored),
            }
            (hits if c.names else soft).append(hit)
        if hits:
            named[name] = hits + soft
        elif soft:
            corroborated[name] = soft
    return named, corroborated


#: The members' own arrays, per release directory, loaded once.
_ARRAYS = {}


def member_arrays_of(release_dir):
    """`{idx: arrays}` for one release, or `{}` when it ships none.

    Rung 1 writes the arrays it measured beside the release set.  They are the
    member's own finished geometry -- keypoints, shapes, structure, poses,
    membership -- and they are what lets this pass state a trimmed core as a
    member and MEASURE it, instead of re-aggregating readings that were taken
    while the cut frames were still in."""
    key = str(Path(release_dir).resolve())
    if key in _ARRAYS:
        return _ARRAYS[key]
    import zlib

    path = Path(release_dir) / "member_arrays.npz"
    out = {}
    if path.is_file():
        with np.load(path, allow_pickle=False) as z:
            meta = json.loads(zlib.decompress(z["_meta"].tobytes()).decode("utf-8"))
            names = meta["names"]
            for tag, rec in meta["members"].items():
                d = {
                    "idx": int(rec["idx"]),
                    "model": rec["model"],
                    "camera": rec["camera"],
                    "f_eq": rec["f_eq"],
                    "names": names,
                }
                for field in (
                    "rvec",
                    "tvec",
                    "posed",
                    "pts",
                    "obs_c",
                    "obs_i",
                    "obs_uv",
                    "obs_f",
                    "obs_shape",
                    "keep",
                ):
                    k = f"{tag}__{field}"
                    d[field] = np.asarray(z[k]) if k in z.files else None
                out[d["idx"]] = d
    _ARRAYS[key] = out
    return out


def capture_floors(ev):
    """The conditioning floors rung 1 recorded for this member's capture.

    Conditioning is respected, never re-derived: a core measured on its own
    would take a floor over one member's frames, which is not the population
    the floor is a quantile of."""
    loo, rot = _blk(ev, "self_resection"), _blk(ev, "rot_self_resection")
    return {
        "inlier_floor": _num(loo.get("inlier_floor")),
        "rot_inlier_floor": _num(rot.get("inlier_floor")),
        "rot_support_spread_bar": _num(rot.get("support_spread_bar_deg")),
    }


def remeasure_core(arrays, hyp, keep_names, f_vote):
    """Re-run the battery on the frames a trim kept, as a member in its own
    right.

    Returns the fresh evaluation block, or None when the release ships no
    arrays to state the core from.  The two channels the capture supplies
    rather than the member -- the held-out images' verdict, which needs the
    capture's match graph -- come back unmeasurable, which is a reading about
    what could be asked of a core, not a reading about the core."""
    d = arrays.get(int(hyp.get("idx", -1)))
    if d is None:
        return None
    here = str(Path(__file__).resolve().parent)
    if here not in sys.path:
        sys.path.insert(0, here)
    import seed_candidate_eval as EV

    member = EV.member_from_arrays(d)
    core = member.restricted(keep_names, min_obs=MIN_POINT_OBS)
    if not int(core.posed.sum()):
        return None
    blocks = EV.evaluate(
        [core],
        f_vote,
        pair_obs=None,
        floors=capture_floors(hyp.get("evaluation") or {}),
    )
    return blocks.get(core.idx)


def core_evidence(values, gates, capture_meds, capture_defective):
    """Which member gates a trimmed core still fails.

    The core is judged exactly as a member is: the same aggregates, against
    fleet quantiles of those aggregates taken over whole members.  A core that
    fails is not a repaired member, and the trim does not stand."""
    cg = gates.get("core_gates", {})
    out = []
    for key, _b, _f, _how, low, _e, modes in CORE_CHANNELS:
        entry, value = cg.get(key), values.get(key)
        if entry is None or value is None:
            continue
        ab = entry.get("absolute")
        if "abs" in modes and ab:
            bar = ab["value"]
            if value < bar if low else value > bar:
                out.append(
                    {
                        "channel": key,
                        "reading": "absolute",
                        "value": value,
                        "gate": bar,
                        "side": "below" if low else "above",
                        "quantile": ab.get("quantile"),
                    }
                )
                continue
        rel = entry.get("relative")
        if "rel" in modes and rel and not capture_defective:
            ratio = capture_ratio(value, capture_meds.get(key), low)
            if ratio is not None and ratio > rel["value"]:
                out.append(
                    {
                        "channel": key,
                        "reading": "capture-relative",
                        "value": value,
                        "ratio": ratio,
                        "capture_median": capture_meds.get(key),
                        "gate": rel["value"],
                        "quantile": rel.get("quantile"),
                    }
                )
    return out


# ── Trimming ────────────────────────────────────────────────────────────────


def _link_components(track_image_indexes, track_point_indexes, labels):
    """The images' partition into blocks linked by shared points.

    Two images are linked when they share at least `MIN_LINK_POINTS` points;
    the blocks are the connected components of that graph, returned as
    frozensets of `labels`.  An image sharing enough points with nothing is a
    block of its own.
    """
    n_images = len(labels)
    tii = np.asarray(track_image_indexes, np.int64)
    tpi = np.asarray(track_point_indexes, np.int64)
    shared = {}
    if tii.size:
        order = np.argsort(tpi, kind="stable")
        tii, tpi = tii[order], tpi[order]
        bounds = np.flatnonzero(np.diff(tpi)) + 1
        for lo, hi in zip(
            np.concatenate(([0], bounds)), np.concatenate((bounds, [len(tpi)]))
        ):
            imgs = np.unique(tii[lo:hi])
            for a in range(len(imgs)):
                for b in range(a + 1, len(imgs)):
                    pair = (int(imgs[a]), int(imgs[b]))
                    shared[pair] = shared.get(pair, 0) + 1
    adj = {i: set() for i in range(n_images)}
    for (a, b), n in shared.items():
        if n >= MIN_LINK_POINTS:
            adj[a].add(b)
            adj[b].add(a)
    seen, blocks = set(), []
    for start in range(n_images):
        if start in seen:
            continue
        stack, block = [start], []
        seen.add(start)
        while stack:
            i = stack.pop()
            block.append(labels[i])
            for nxt in adj[i]:
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
        blocks.append(frozenset(block))
    return set(blocks)


def _restricted_to(blocks, labels):
    """`blocks` cut down to `labels`, dropping the blocks that empty out."""
    keep = set(labels)
    return {b & keep for b in blocks if b & keep}


def _links_preserved(recon, core, keep_names):
    """Did the trim break a link the member had?

    A trim may not fragment a member: every pair of surviving frames the
    member linked through shared points must still be linked in the core, and
    the point cull is what can break that.  The test is a comparison, not an
    absolute -- a member whose frames are bound by something other than shared
    structure is in as many pieces after the cut as before, and a cut that
    changes nothing about how its frames are tied together has broken nothing.
    """
    names = [str(n).replace("\\", "/") for n in recon.image_names]
    core_names = [str(n).replace("\\", "/") for n in core.image_names]
    tii = np.asarray(recon.track_image_indexes, np.int64)
    tpi = np.asarray(recon.track_point_indexes, np.int64)
    alive = np.asarray([n in keep_names for n in names], bool)
    mask = alive[tii] if tii.size else np.zeros(0, bool)
    before = _link_components(tii[mask], tpi[mask], names)
    after = _link_components(
        core.track_image_indexes, core.track_point_indexes, core_names
    )
    return _restricted_to(before, core_names) == after


def trim_member(sfmr_path, drop_names, out_path):
    """Drop `drop_names` from a member and write the surviving core.

    The frames go, then every point left with fewer than `MIN_POINT_OBS`
    supporting observations goes with them, then any frame the point cull left
    with nothing to see.  The bindings do the index remapping (both
    `subset_by_image_indices` and `filter_points_by_mask` return a fresh,
    consistently indexed reconstruction), so no array is rebuilt here.

    Returns a report dict; `ok` is False when no core survives, when the cut
    drops nothing, or when the cut breaks a link the member had, and nothing is
    written in those cases.
    """
    from sfmtool._sfmtool.reconstruction import SfmrReconstruction

    recon = SfmrReconstruction.load(str(sfmr_path))
    names = [str(n).replace("\\", "/") for n in recon.image_names]
    drop = {str(n).replace("\\", "/") for n in drop_names}
    keep = [i for i, n in enumerate(names) if n not in drop]
    report = {
        "ok": False,
        "frames_before": len(names),
        "points_before": int(recon.point_count),
        "obs_before": int(recon.observation_count),
        "frames_dropped": sorted(n for n in names if n in drop),
    }
    if len(keep) == len(names):
        # A named frame the release does not hold is a frame already gone.
        report["reason"] = "the release holds none of the named frames"
        return report
    if len(keep) < MIN_CORE_FRAMES:
        report["reason"] = f"core would hold {len(keep)} frames < {MIN_CORE_FRAMES}"
        return report
    core = recon.subset_by_image_indices(
        np.asarray(keep, np.uint32), drop_orphaned_points=True
    )
    counts = np.asarray(core.observation_counts, np.int64)
    core = core.filter_points_by_mask(counts >= MIN_POINT_OBS)
    # A frame the point cull emptied is no longer part of the core.
    tii = np.asarray(core.track_image_indexes, np.int64)
    alive = np.zeros(core.image_count, bool)
    if tii.size:
        alive[np.unique(tii)] = True
    if not alive.all():
        core = core.subset_by_image_indices(
            np.flatnonzero(alive).astype(np.uint32), drop_orphaned_points=True
        )
    core_names = [str(n).replace("\\", "/") for n in core.image_names]
    report["frames_dropped"] = sorted(set(names) - set(core_names))
    if core.image_count < MIN_CORE_FRAMES:
        report["reason"] = (
            f"core held {core.image_count} frames < {MIN_CORE_FRAMES} after the cull"
        )
        return report
    if not _links_preserved(recon, core, {n for n in names if n not in drop}):
        report["reason"] = "the cut breaks a link the member had"
        return report
    core.save(
        str(out_path),
        operation="seed_rung2_trim",
        tool_options={
            "rung2_trim": "frames dropped on per-frame defect evidence",
            "frames_dropped": "|".join(report["frames_dropped"]),
            "min_point_obs": str(MIN_POINT_OBS),
        },
    )
    report.update(
        ok=True,
        output=Path(out_path).name,
        frames_after=int(core.image_count),
        points_after=int(core.point_count),
        obs_after=int(core.observation_count),
        frames_kept=core_names,
        links_preserved=True,
    )
    return report


# ── Coverage ────────────────────────────────────────────────────────────────


def _spans(indices):
    out, run = [], []
    for i in sorted(set(int(v) for v in indices)):
        if run and i == run[-1] + 1:
            run.append(i)
        else:
            if run:
                out.append([run[0], run[-1]])
            run = [i]
    if run:
        out.append([run[0], run[-1]])
    return out


def _surviving_frames(hyps, verdicts, model=None):
    """The frames the surviving members of one model family hold."""
    by_idx = {int(h.get("idx", -1)): h for h in hyps}
    out = {}
    for v in verdicts:
        hyp = by_idx.get(v["idx"])
        if hyp is None or v["verdict"] == "refuse":
            continue
        if model is not None and v.get("model") != model:
            continue
        posed = [int(k) for k in hyp.get("posed_frames") or []]
        dropped = set((v.get("trim") or {}).get("frames_dropped") or [])
        if dropped:
            names = [
                r.get("name")
                for r in _blk(hyp.get("evaluation") or {}, "support").get("frames")
                or []
            ]
            if len(names) == len(posed):
                posed = [k for k, n in zip(posed, names) if n not in dropped]
        out[v["idx"]] = set(posed)
    return out


def arbitration_report(hyps, verdicts):
    """Where a surviving FINITE member and a surviving ROTATION-ONLY member
    cover the same frames.

    Nothing is chosen here.  A far-field layer and a near-field solve of the
    same frames are two readings of one capture, and the overlap plus both
    sets of channels is what a later pass arbitrates on."""
    fin = _surviving_frames(hyps, verdicts, "finite")
    rot = _surviving_frames(hyps, verdicts, _ROT_ONLY)
    fin_all = set().union(*fin.values()) if fin else set()
    rot_all = set().union(*rot.values()) if rot else set()
    pairs = []
    for r_idx, r_frames in sorted(rot.items()):
        for f_idx, f_frames in sorted(fin.items()):
            shared = r_frames & f_frames
            if not shared:
                continue
            pairs.append(
                {
                    "rotation_only": r_idx,
                    "finite": f_idx,
                    "shared_frames": len(shared),
                    "shared_of_rotation_only": len(shared) / max(len(r_frames), 1),
                    "shared_of_finite": len(shared) / max(len(f_frames), 1),
                }
            )
    return {
        "n_surviving_finite": len(fin),
        "n_surviving_rotation_only": len(rot),
        "frames_finite_only": _spans(fin_all - rot_all),
        "frames_rotation_only": _spans(rot_all - fin_all),
        "frames_both": _spans(fin_all & rot_all),
        "n_frames_both": len(fin_all & rot_all),
        "overlaps": pairs,
    }


def coverage_report(hyps, verdicts, capture_frames=None):
    """Capture coverage of the SURVIVING set, and its gaps.

    Coverage is the union of the frames the surviving members posed, expressed
    as spans over the capture's frame index space.  A trimmed member
    contributes only the frames its core kept.  Nothing here reads quality:
    a member that survived contributes exactly the frames it holds.
    """
    by_idx = {int(h.get("idx", -1)): h for h in hyps}
    committed, surviving = set(), set()
    for hyp in hyps:
        committed.update(int(k) for k in hyp.get("posed_frames") or [])
    for v in verdicts:
        hyp = by_idx.get(v["idx"])
        if hyp is None or v["verdict"] == "refuse":
            continue
        posed = [int(k) for k in hyp.get("posed_frames") or []]
        dropped = set((v.get("trim") or {}).get("frames_dropped") or [])
        if dropped:
            names = [
                r.get("name")
                for r in _blk(hyp.get("evaluation") or {}, "support").get("frames")
                or []
            ]
            if len(names) == len(posed):
                posed = [k for k, n in zip(posed, names) if n not in dropped]
        surviving.update(posed)
    lo = 0 if capture_frames else (min(committed) if committed else 0)
    hi = (
        (capture_frames - 1) if capture_frames else (max(committed) if committed else 0)
    )
    extent = set(range(lo, hi + 1))
    return {
        "extent": [lo, hi],
        "extent_source": "capture frame count"
        if capture_frames
        else "index span of the committed set",
        "n_extent": len(extent),
        "n_committed": len(committed),
        "n_surviving": len(surviving),
        "fraction_of_extent": (len(surviving) / len(extent)) if extent else None,
        "fraction_of_committed": (len(surviving) / len(committed))
        if committed
        else None,
        "spans": _spans(surviving),
        "gaps": _spans(extent - surviving),
        "lost_to_refusal": _spans(committed - surviving),
    }


# ── The pass ────────────────────────────────────────────────────────────────


def take_trim(cut, ctx, hyp, row):
    """Cut `cut` from one member and judge what is left, as a member.

    Returns the trim report.  `ok` says whether the core survived the cut and
    then passed the member gates; a core that fails carries its own evidence in
    `core_evidence`, and the released file is not left behind.
    """
    stem = Path(row["release_file"]).stem
    out_path = ctx["out_dir"] / f"{stem}-trimmed.sfmr"
    ctx["out_dir"].mkdir(parents=True, exist_ok=True)
    trim = trim_member(ctx["release_dir"] / (row["release_file"] or ""), cut, out_path)
    if not trim["ok"]:
        return trim
    # THE CORE IS MEASURED, not re-aggregated.  Every stored per-frame reading
    # was taken with the cut frames still in the member, so a core that
    # inherits them can read clean while the geometry it inherits is not.  The
    # battery runs again on the frames that survived, and the SAME member gates
    # the whole member faced are applied to what it says.
    kept = set(trim.get("frames_kept") or [])
    fam = row["model"]
    block = None
    try:
        block = remeasure_core(ctx["arrays"], hyp, kept, ctx["f_vote"])
    except Exception as exc:  # noqa: BLE001 — a failed re-measurement falls
        # back to the stored readings, it never kills the pass.
        trim["core_remeasure_error"] = f"{type(exc).__name__}: {exc}"
    if block is not None:
        crow, celig = member_channels(
            {
                "idx": row["idx"],
                "model": row["model"],
                "camera": hyp.get("camera"),
                "release_file": row["release_file"],
                "posed": int(trim.get("frames_after") or 0),
                "evaluation": block,
            }
        )
        crow["_eligible"] = celig
        core_ev, core_blocked = member_evidence(
            crow,
            ctx["gates"],
            ctx["broken"],
            ctx["capture_meds"].get(fam, {}),
            ctx["defective"].get(fam, False),
        )
        left = [e for e in core_ev if not e.get("corroborating")]
        trim["core_judged_on"] = "the surviving core, re-measured"
        trim["core_channels"] = {
            c.key: crow.get(c.key)
            for c in GATED_CHANNELS
            if crow.get(c.key) is not None
        }
        trim["core_corroborating"] = [e for e in core_ev if e.get("corroborating")]
        trim["core_conditioning_limited"] = core_blocked
    else:
        # No arrays beside the release: fall back to the aggregates of the
        # stored per-frame readings, and say so, because that is the weaker
        # reading.
        core = core_channels(hyp, kept)
        left = core_evidence(
            core,
            ctx["gates"],
            ctx["core_meds"].get(fam, {}),
            ctx["defective"].get(fam, False),
        )
        trim["core_judged_on"] = (
            "stored per-frame readings: the release ships no member arrays to "
            "state the core from"
        )
        trim["core_channels"] = core
    trim["core_evidence"] = left
    if left:
        trim["ok"] = False
        trim["reason"] = "the trimmed core still fails " + ", ".join(
            sorted({e["channel"] for e in left})
        )
        out_path.unlink(missing_ok=True)
    return trim


def select(release_dir, gates, out_dir=None, capture_frames=None, write=True):
    """Rank, refuse and trim one committed candidate set."""
    release_dir = Path(release_dir)
    man, hyps = load_release(release_dir)
    out_dir = Path(out_dir) if out_dir else release_dir / "rung2"
    # What a trimmed core is measured from, and the vote its lens is read
    # against.  Both are the release's own.
    arrays = member_arrays_of(release_dir)
    f_vote = _num((man.get("vote") or {}).get("f"))

    rows, by_idx = [], {}
    for hyp in hyps:
        row, eligible = member_channels(hyp)
        row["_eligible"] = eligible
        ev = hyp.get("evaluation") or {}
        row["_frames"] = {
            c.frame: _blk(ev, c.block).get(c.frame)
            for c in GATED_CHANNELS
            if c.frame is not None
        }
        row.update(core_channels(hyp))
        rows.append(row)
        by_idx[row["idx"]] = (hyp, row)

    # One ordering per model family: a rotation-only layer and a finite solve
    # do not share a measurable channel, and ranking them together would order
    # them on which channels exist rather than on what they say.
    ordering = {}
    for family in sorted({r["model"] for r in rows}):
        ordering[family] = [
            {"idx": r["idx"], "rank": r["rank"], "rank_score": r["rank_score"]}
            for r in rank_members([r for r in rows if r["model"] == family])
        ]

    # A capture-refereed channel whose referee this capture's own members
    # outvoted is a non-measurement for every member here.
    broken = tuple(
        key
        for key in CAPTURE_REFEREED
        if key in gates.get("member_gates", {})
        and not capture_referee_ok(
            rows, key, gates["member_gates"][key]["absolute"]["value"]
        )
    )
    # THE CAPTURE'S OWN MEDIANS, per model family: the denominator of every
    # capture-relative reading, and the verdict on whether they mean anything.
    families = sorted({r["model"] for r in rows})
    capture_meds, core_meds, defective = {}, {}, {}
    for fam in families:
        fam_rows = [r for r in rows if r["model"] == fam]
        capture_meds[fam] = {
            c.key: capture_median(fam_rows, c.key) for c in GATED_CHANNELS
        }
        core_meds[fam] = {
            key: capture_median(fam_rows, key, eligible_only=False)
            for key, *_rest in CORE_CHANNELS
        }
        defective[fam] = majority_defective(fam_rows, gates)

    #: Everything a trim needs to state a core and judge it as a member.
    ctx = {
        "release_dir": release_dir,
        "out_dir": out_dir,
        "gates": gates,
        "broken": broken,
        "arrays": arrays,
        "f_vote": f_vote,
        "capture_meds": capture_meds,
        "core_meds": core_meds,
        "defective": defective,
    }

    verdicts = []
    for hyp in hyps:
        _h, row = by_idx[int(hyp.get("idx", -1))]
        if not row["eval_enabled"]:
            verdicts.append(
                {
                    "idx": row["idx"],
                    "model": row["model"],
                    "release_file": row["release_file"],
                    "rank": row.get("rank"),
                    "rank_score": row.get("rank_score"),
                    "rank_families": row.get("rank_families", []),
                    "verdict": "keep",
                    "verdict_reason": "no evaluation block; nothing to refuse on",
                    "evidence": [],
                    "conditioning_limited": [],
                }
            )
            continue
        fam = row["model"]
        fam_defective = bool(defective.get(fam))
        evidence, blocked = member_evidence(
            row, gates, broken, capture_meds.get(fam, {}), fam_defective
        )
        rec = {
            "idx": row["idx"],
            "model": row["model"],
            "release_file": row["release_file"],
            "rank": row.get("rank"),
            "rank_score": row.get("rank_score"),
            "rank_families": row.get("rank_families", []),
            "rank_scores": row.get("rank_scores", {}),
            "posed": row.get("posed"),
            "verdict": "keep",
            "verdict_reason": "no gated channel past its population gate",
            "evidence": evidence,
            "conditioning_limited": blocked,
        }
        # A CORROBORATING channel never carries a verdict on its own.  Scale
        # coherence and a lens deviation qualify a defect other evidence
        # already shows; alone they say something is unusual, not that the
        # member is wrong, and neither of them names the frame that is.
        deciding = [e for e in evidence if not e.get("corroborating")]
        if evidence and not deciding:
            rec["verdict_reason"] = (
                "only corroborating channels fired ("
                + ", ".join(sorted({e["channel"] for e in evidence}))
                + "); they are never a sole ground"
            )
        if deciding:
            bad, soft = defect_frames(hyp, gates, row["model"], row["family"])
            n_posed = int(row.get("posed") or 0)
            rec["frame_evidence"] = {k: bad[k] for k in sorted(bad)}
            if soft:
                rec["frame_corroboration"] = {k: soft[k] for k in sorted(soft)}
            # THE TRIM TEST, in three parts.  First: does every deciding
            # channel NAME a frame?  A held-out image's verdict on the member
            # and a diverging refit are statements about the whole member, and
            # a member the capture cannot resect against is not repaired by
            # dropping one of its frames.  Second: does a POSE-SHAPED
            # per-frame reading put the defect in a minority of frames?  Third
            # -- after the cut -- does the surviving core pass the member
            # gates?  All three, or the verdict is a refusal.
            wide = sorted(
                {e["channel"] for e in deciding if e["channel"] not in LOCALIZING}
            )
            minority = (
                bool(bad)
                and bool(n_posed)
                and (len(bad) <= MAX_TRIM_FRACTION * n_posed)
            )
            if minority and not wide:
                trim = take_trim(sorted(bad), ctx, hyp, row)
                trim["frame_evidence"] = rec["frame_evidence"]
                rec["trim"] = trim
                if trim["ok"]:
                    rec["verdict"] = "trim"
                    rec["verdict_reason"] = (
                        f"defect localized to {len(trim['frames_dropped'])} of "
                        f"{n_posed} frames; the core keeps the member's own "
                        "links and was re-judged clean against the member gates"
                    )
                elif trim.get("core_evidence"):
                    rec["verdict"] = "refuse"
                    rec["verdict_reason"] = (
                        f"defect evidence, trim withdrawn: {trim['reason']}"
                    )
                    rec["trim_not_taken"] = sorted(bad)
                else:
                    rec["verdict"] = "refuse"
                    rec["verdict_reason"] = (
                        f"defect evidence, trim refused: {trim.get('reason')}"
                    )
            else:
                rec["verdict"] = "refuse"
                rec["verdict_reason"] = (
                    "member-wide evidence (" + ", ".join(wide) + ")"
                    if wide
                    else (
                        "defect does not localize to a frame minority"
                        if bad
                        else "defect evidence with no pose-shaped per-frame "
                        "localization"
                    )
                )
                # What a trim WOULD have dropped, recorded but not taken: the
                # named frames are still the most likely seat of the defect.
                if minority:
                    rec["trim_not_taken"] = sorted(bad)
        else:
            # NOTHING ACCUSES THIS MEMBER, and a frame it posed on almost
            # nothing is still not a frame it measured.  A SUPPORT reading
            # names such frames by counting what the member holds there, which
            # is a statement about the evidence and not about the geometry, so
            # it stands on a member no channel accuses.  The cut is taken on
            # those frames alone: a pose-shaped reading past a fleet frame bar
            # is a rank inside a sound member's own spread, and this pass does
            # not cut on a rank.
            bad, soft = defect_frames(hyp, gates, row["model"], row["family"])
            starved = {k: v for k, v in bad.items() if any(h["support"] for h in v)}
            if starved:
                n_posed = int(row.get("posed") or 0)
                rec["frame_support_evidence"] = {
                    k: [h for h in starved[k] if h["support"]] for k in sorted(starved)
                }
                if n_posed and len(starved) <= MAX_TRIM_FRACTION * n_posed:
                    trim = take_trim(sorted(starved), ctx, hyp, row)
                    trim["frame_evidence"] = rec["frame_support_evidence"]
                    if trim["ok"]:
                        rec["trim"] = trim
                        rec["verdict"] = "trim"
                        rec["verdict_reason"] = (
                            f"{len(trim['frames_dropped'])} of {n_posed} frames "
                            "hold too few observations to be a reading; the core "
                            "keeps the member's own links and was re-judged clean "
                            "against the member gates"
                        )
                    else:
                        # The member stands exactly as it did, with every
                        # frame it had.  Insufficient support on some of its
                        # frames is not defect evidence against the member, so
                        # it can never turn into one.
                        rec["trim_attempted"] = trim
                        rec["trim_not_taken"] = sorted(starved)
                        rec["verdict_reason"] = (
                            "no gated channel past its population gate; the "
                            f"support trim of {len(starved)} frames was not "
                            f"taken ({trim.get('reason')})"
                        )
                else:
                    rec["trim_not_taken"] = sorted(starved)
                    rec["verdict_reason"] = (
                        "no gated channel past its population gate; "
                        f"{len(starved)} of {n_posed} frames hold too few "
                        "observations to be a reading, which is more than a "
                        "trim may remove"
                    )
        verdicts.append(rec)

    out = {
        "schema": "seed-rung2/1",
        "generated": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "release_dir": str(release_dir),
        "entry": entry_name(release_dir),
        "source_stamp": man.get("stamp"),
        "gates": gates,
        "capture": {
            "families": families,
            "majority_defective": {k: bool(v) for k, v in defective.items()},
            "majority_defective_reason": (
                "a family whose median member fails the absolute gates has a "
                "broken member for a median, so no capture-relative reading of "
                "that family is read"
            ),
            "broken_referees": list(broken),
            "medians": {
                fam: {k: v for k, v in meds.items() if v is not None}
                for fam, meds in capture_meds.items()
            },
        },
        "ordering": ordering,
        "members": verdicts,
        "counts": {
            v: sum(1 for r in verdicts if r["verdict"] == v)
            for v in ("keep", "trim", "refuse")
        },
        "counts_by_model": {
            fam: {
                v: sum(
                    1 for r in verdicts if r["verdict"] == v and r.get("model") == fam
                )
                for v in ("keep", "trim", "refuse")
            }
            for fam in families
        },
        "coverage": coverage_report(hyps, verdicts, capture_frames),
        "arbitration": arbitration_report(hyps, verdicts),
    }
    if write:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "rung2.json").write_text(
            json.dumps(out, indent=2) + "\n", encoding="utf-8"
        )
    return out


# ── CLI ─────────────────────────────────────────────────────────────────────


def _write_tsv(path, rows):
    cols, seen = [], set()
    for row in rows:
        for k in row:
            if not k.startswith("_") and k not in seen:
                seen.add(k)
                cols.append(k)
    with Path(path).open("w", encoding="utf-8", newline="") as fh:
        fh.write("\t".join(cols) + "\n")
        for row in rows:
            fh.write(
                "\t".join("" if row.get(c) is None else str(row.get(c)) for c in cols)
                + "\n"
            )
    return cols


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="mode", required=True)

    p = sub.add_parser("channels", help="flatten release channels into a table")
    p.add_argument("--out", required=True)
    p.add_argument("--frames-out")
    p.add_argument("release", nargs="+")

    p = sub.add_parser("derive-gates", help="fleet quantile gates")
    p.add_argument("--out", required=True)
    p.add_argument("--quantile", type=float, default=DEFAULT_QUANTILE)
    p.add_argument("--hard-quantile", type=float, default=DEFAULT_HARD_QUANTILE)
    p.add_argument("--floor-quantile", type=float, default=DEFAULT_FLOOR_QUANTILE)
    p.add_argument("release", nargs="+")

    p = sub.add_parser("select", help="rank, refuse and trim one release")
    p.add_argument("--gates", required=True)
    p.add_argument("--out")
    p.add_argument("--capture-frames", type=int)
    p.add_argument("release")

    args = ap.parse_args(argv)
    if args.mode == "channels":
        members, frames = collect_population(args.release)
        cols = _write_tsv(args.out, members)
        print(f"{args.out}: {len(members)} members, {len(cols)} columns")
        if args.frames_out:
            _write_tsv(args.frames_out, frames)
            print(f"{args.frames_out}: {len(frames)} frame rows")
        return 0
    if args.mode == "derive-gates":
        gates = derive_gates(
            args.release, args.quantile, args.hard_quantile, args.floor_quantile
        )
        Path(args.out).write_text(json.dumps(gates, indent=2) + "\n", encoding="utf-8")
        pop = gates["population"]
        n_frame = sum(len(v) for v in gates["frame_gates"].values())
        print(
            f"{args.out}: {len(gates['member_gates'])} member gates, "
            f"{n_frame} frame gates over {len(gates['frame_gates'])} model "
            f"families, {len(gates['core_gates'])} core gates, "
            f"q={gates['quantile']}, hard q={gates['hard_quantile']}, "
            f"floor p{gates['floor_quantile']:g}, support floor "
            f"p{SUPPORT_FLOOR_PERCENTILE:g}, n={pop['n_members']} members "
            f"({pop['n_clean_members']} clean), "
            f"{len(pop['majority_defective_groups'])} majority-defective groups"
        )
        return 0
    gates = json.loads(Path(args.gates).read_text(encoding="utf-8"))
    out = select(args.release, gates, args.out, args.capture_frames)
    print(
        f"{out['entry']}: "
        + " ".join(f"{k}={v}" for k, v in out["counts"].items())
        + f"  coverage {out['coverage']['n_surviving']}/"
        f"{out['coverage']['n_committed']} committed frames"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
