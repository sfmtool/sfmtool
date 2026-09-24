// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! What every member does once a track-stage track stands: keep it on the
//! pixel, tilt it toward the neighbours' surface, grow it into the views it was
//! not found in, turn out the views that disagree with its geometry, and gate
//! it.
//!
//! **Anchoring.** A track-stage fit localizes every sighting, the queried one
//! included, against the consensus of the others. Beside a stronger feature the
//! whole patch slides toward it in every photograph at once. [`anchor`] slides
//! it back across its own plane until its centre in the queried photograph is
//! the pixel again and reads the track there, so every sighting moves by the
//! same in-plane displacement and the reading scores them where they now sit.

use nalgebra::Vector3;

use crate::bench::evaluate::{evaluate, EvaluateOptions};
use crate::bench::fit::{fit, FitOptions};
use crate::bench::geometry_search::{search_geometry, GeometrySearchOptions};
use crate::bench::stage::set_stage;
use crate::bench::steps::{
    add_observation, apply_thresholds, create_cluster, resize_patch, set_verdict, tilt_patch,
    translate_patch_to_pixel, ClusterSeed, ObservationSeed, Viewpoint,
};
use crate::bench::track::{EditableTrack, Provenance, StageKind, TrackMeasurement, Verdict};
use crate::bench::Bench;
use crate::numeric::median_in_place;
use crate::progress::Progress;

use super::{Ctx, FinishOptions, Refusal, RefusalStage, StageRecord, TiltRecord};

/// The median leave-one-out ZNCC over the `in` observations that carry one, or
/// negative infinity when none does.
pub(super) fn median_zncc(track: &EditableTrack) -> f64 {
    let mut z: Vec<f64> = track
        .observations
        .iter()
        .filter(|o| o.verdict == Verdict::In)
        .filter_map(|o| o.track.as_ref()?.zncc)
        .collect();
    if z.is_empty() {
        f64::NEG_INFINITY
    } else {
        median_in_place(&mut z)
    }
}

/// How many observations are `in`.
pub(super) fn in_count(track: &EditableTrack) -> usize {
    track.verdict_counts().0
}

/// A member's own reading of a track: `in` views weighted by the median ZNCC,
/// floored at zero.
pub(super) fn score_track(track: &EditableTrack) -> f64 {
    let z = median_zncc(track);
    if !z.is_finite() {
        return 0.0;
    }
    in_count(track) as f64 * z.max(0.0)
}

/// Whether the track is a bearing.
pub(super) fn at_infinity(track: &EditableTrack) -> bool {
    track.track().is_some_and(|p| p.at_infinity)
}

/// The track's position, when it is a finite point that has one.
pub(super) fn position(track: &EditableTrack) -> Option<Vector3<f64>> {
    if at_infinity(track) {
        return None;
    }
    Some(track.track()?.position?.coords)
}

/// A track-stage reading of observation `o`, `NaN` when it carries none.
fn reading(track: &EditableTrack, o: usize, key: fn(&TrackMeasurement) -> Option<f64>) -> f64 {
    track.observations[o]
        .track
        .as_ref()
        .and_then(key)
        .unwrap_or(f64::NAN)
}

/// How far observation `q`'s keypoint is from `pixel`, when it has one.
pub(super) fn query_offset(track: &EditableTrack, q: usize, pixel: [f64; 2]) -> Option<f64> {
    let kp = track.observations[q].track.as_ref()?.keypoint?;
    let dx = f64::from(kp[0]) - pixel[0];
    let dy = f64::from(kp[1]) - pixel[1];
    Some((dx * dx + dy * dy).sqrt())
}

/// The largest projection offset over the `in` views, or infinity when none
/// carries one.
fn max_projection_offset(track: &EditableTrack) -> f64 {
    track
        .observations
        .iter()
        .enumerate()
        .filter(|(_, o)| o.verdict == Verdict::In)
        .map(|(i, _)| reading(track, i, |m| m.projection_offset_px))
        .filter(|v| v.is_finite())
        .reduce(f64::max)
        .unwrap_or(f64::INFINITY)
}

/// Whether an `in` observation has no keypoint yet, which is the state of a
/// view the geometry search added before a fit places it.
fn unplaced(track: &EditableTrack) -> bool {
    track
        .observations
        .iter()
        .any(|o| o.verdict == Verdict::In && o.track.as_ref().and_then(|m| m.keypoint).is_none())
}

/// Read `track` as it stands, with the default reading.
pub(super) fn read(ctx: &Ctx<'_>, track: &EditableTrack) -> Result<EditableTrack, String> {
    evaluate(
        track,
        ctx.edited,
        ctx.views,
        &EvaluateOptions::default(),
        &Progress::none(),
    )
    .map(|(t, _)| t)
    .map_err(|e| e.to_string())
}

/// Fit `track` with the default fit.
pub(super) fn fit_track(ctx: &Ctx<'_>, track: &EditableTrack) -> Result<EditableTrack, String> {
    fit(
        track,
        ctx.edited,
        ctx.views,
        &FitOptions::default(),
        &Progress::none(),
    )
    .map(|(t, _)| t)
    .map_err(|e| e.to_string())
}

/// The thresholds' verdicts painted onto the unpinned observations.
pub(super) fn thresholds(track: &EditableTrack) -> EditableTrack {
    apply_thresholds(track).0
}

/// Slide the patch so its centre in observation `q` is `pixel`, then read it.
fn anchor(
    ctx: &Ctx<'_>,
    track: &EditableTrack,
    q: usize,
    pixel: [f64; 2],
) -> Result<EditableTrack, String> {
    let (moved, _) = translate_patch_to_pixel(track, ctx.edited, Viewpoint::Observation(q), pixel)
        .map_err(|e| e.to_string())?;
    read(ctx, &moved)
}

/// `refits` rounds of fit-then-anchor after an anchor, keeping the best
/// reading.
///
/// A round's fit is taken whatever the reading says when the track has an `in`
/// view with no keypoint, because only a fit places one; otherwise it is taken
/// only when the median ZNCC does not fall. The first anchor's refusal is the
/// call's; a later round's refusal ends the rounds.
pub(super) fn anchored_fit(
    ctx: &Ctx<'_>,
    track: &EditableTrack,
    q: usize,
    pixel: [f64; 2],
    refits: usize,
) -> Result<EditableTrack, String> {
    let mut best = anchor(ctx, track, q, pixel)?;
    for _ in 0..refits {
        let Ok(fitted) = fit_track(ctx, &best).and_then(|f| anchor(ctx, &f, q, pixel)) else {
            break;
        };
        if !unplaced(&best) && median_zncc(&fitted) < median_zncc(&best) {
            break;
        }
        best = fitted;
    }
    Ok(best)
}

/// A track-stage track from the queried pixel plus `(image, pixel)` sightings.
///
/// Every sighting is set `in`, by hand, and the cluster is upgraded without a
/// cluster reading: the upgrade triangulates the sightings as given and runs
/// the track-stage fit over the frame it builds. Observation 0 is the query.
pub(super) fn track_from_sightings(
    ctx: &Ctx<'_>,
    image: u32,
    pixel: [f64; 2],
    radius_px: f64,
    sightings: &[(u32, [f64; 2])],
) -> Result<EditableTrack, String> {
    let mut track = seed_cluster(ctx, image, pixel, radius_px)?;
    for &(other, px) in sightings {
        let seed = ObservationSeed {
            image: other,
            pixel: px,
            shape: None,
            provenance: Provenance::Sweep,
        };
        let (next, added) = add_observation(&track, &seed).map_err(|e| e.to_string())?;
        track = set_verdict(&next, added.observation, Verdict::In)
            .map_err(|e| e.to_string())?
            .0;
    }
    set_stage(
        &track,
        ctx.edited,
        ctx.views,
        StageKind::Track,
        &FitOptions::default(),
        &Progress::none(),
    )
    .map(|(t, _)| t)
    .map_err(|e| e.to_string())
}

/// A one-sighting cluster at `pixel` in `image`, `radius_px` from the pixel to
/// its edge, with that sighting set `in` by hand.
pub(super) fn seed_cluster(
    ctx: &Ctx<'_>,
    image: u32,
    pixel: [f64; 2],
    radius_px: f64,
) -> Result<EditableTrack, String> {
    let seed = ClusterSeed::from_pixel(image, ctx.image_stem(image), pixel, radius_px);
    let (bench, report) = create_cluster(&Bench::new(), &seed).map_err(|e| e.to_string())?;
    let track = bench
        .track(&report.label)
        .expect("the cluster was just put on the bench");
    set_verdict(track, 0, Verdict::In)
        .map(|(t, _)| t)
        .map_err(|e| e.to_string())
}

/// The patch tilted toward `normal` (flipped to face observation `q`'s camera)
/// and resized to `half`, each only where the step accepts it. A bearing is
/// returned as it is.
pub(super) fn shape_track(
    ctx: &Ctx<'_>,
    track: &EditableTrack,
    q: usize,
    normal: Option<Vector3<f64>>,
    half: Option<f64>,
) -> EditableTrack {
    if at_infinity(track) {
        return track.clone();
    }
    let mut track = track.clone();
    if let (Some(mut normal), Some(p)) = (normal, position(&track)) {
        let to_cam = ctx.cameras[track.observations[q].image as usize].center - p;
        if normal.dot(&to_cam) < 0.0 {
            normal = -normal;
        }
        if let Ok((tilted, _)) = tilt_patch(&track, ctx.edited, normal) {
            track = tilted;
        }
    }
    if let Some(half) = half.filter(|&h| h > 0.0) {
        if let Ok((sized, _)) = resize_patch(&track, ctx.edited, half, None) {
            track = sized;
        }
    }
    track
}

/// The distance-weighted mean normal of the observations near the pixel whose
/// depth is within 15% of `depth`, or `None` when there are none or they
/// cancel.
fn neighbour_normal(
    ctx: &Ctx<'_>,
    image: u32,
    pixel: [f64; 2],
    depth: f64,
    opts: &FinishOptions,
) -> Option<Vector3<f64>> {
    let near: Vec<_> = ctx
        .observations_near(image, pixel, opts.normal_prior_radius_px)
        .into_iter()
        .filter(|o| {
            o.stated_depth()
                .is_some_and(|d| (d / depth - 1.0).abs() < 0.15)
        })
        .take(opts.normal_prior_k)
        .collect();
    if near.is_empty() {
        return None;
    }
    let n: Vector3<f64> = near.iter().map(|o| o.normal / (1.0 + o.distance_px)).sum();
    let norm = n.norm();
    (norm > 1e-9).then(|| n / norm)
}

/// Turn out the views that disagree with the track's own geometry and refit,
/// for up to `clean_rounds` rounds.
fn clean(
    ctx: &Ctx<'_>,
    mut track: EditableTrack,
    q: usize,
    pixel: [f64; 2],
    opts: &FinishOptions,
    stages: &mut Vec<StageRecord>,
) -> EditableTrack {
    let mut removed = Vec::new();
    for _ in 0..opts.clean_rounds {
        let worst: Vec<usize> = (0..track.observations.len())
            .filter(|&i| i != q && track.observations[i].verdict == Verdict::In)
            .filter(|&i| {
                reading(&track, i, |m| m.seed_shift_px) > opts.clean_max_shift_px
                    || reading(&track, i, |m| m.projection_offset_px) > opts.clean_max_projection_px
            })
            .collect();
        if worst.is_empty() {
            break;
        }
        for &i in &worst {
            if let Ok((next, _)) = set_verdict(&track, i, Verdict::Out) {
                track = next;
            }
            removed.push(track.observations[i].image);
        }
        if in_count(&track) < 2 {
            break;
        }
        match anchored_fit(ctx, &track, q, pixel, opts.anchor_refits) {
            Ok(next) => track = next,
            Err(_) => break,
        }
    }
    stages.push(StageRecord::Clean {
        removed_images: removed,
    });
    track
}

/// Anchor, tilt toward the neighbours' normal, grow by the geometry search,
/// clean and gate a track-stage `track` whose observation `q` is the query.
pub(super) fn finish(
    ctx: &Ctx<'_>,
    track: EditableTrack,
    q: usize,
    pixel: [f64; 2],
    opts: &FinishOptions,
    stages: &mut Vec<StageRecord>,
) -> Result<EditableTrack, Refusal> {
    let mut track = anchored_fit(ctx, &track, q, pixel, opts.anchor_refits)
        .map_err(|e| Refusal::new(RefusalStage::Anchor, e))?;
    track = thresholds(&track);
    stages.push(StageRecord::Anchor {
        in_views: in_count(&track),
        zncc_median: median_zncc(&track),
    });

    if opts.normal_prior && !at_infinity(&track) {
        let image = track.observations[q].image;
        let depth = position(&track).map(|p| ctx.cameras[image as usize].depth(&p));
        let normal = depth
            .filter(|&d| d > 0.0)
            .and_then(|d| neighbour_normal(ctx, image, pixel, d, opts));
        if let Some(normal) = normal {
            let before = median_zncc(&track);
            let tilted = shape_track(ctx, &track, q, Some(normal), None);
            let record = anchored_fit(ctx, &tilted, q, pixel, opts.anchor_refits).map(|t| {
                let t = thresholds(&t);
                let after = median_zncc(&t);
                let kept = after >= before - opts.normal_prior_tolerance;
                if kept {
                    track = t;
                }
                TiltRecord {
                    degrees: None,
                    stopped: false,
                    zncc_before: before,
                    zncc_after: after,
                    kept,
                }
            });
            stages.push(StageRecord::NormalPrior(record));
        }
    }

    if opts.geometry_search && in_count(&track) >= 2 {
        let grown = search_geometry(
            &track,
            q,
            ctx.views,
            &GeometrySearchOptions::default(),
            &Progress::none(),
        )
        .map_err(|e| e.to_string())
        .and_then(|(grown, report)| {
            let found = (report.added(), report.self_agreement);
            if report.added() == 0 {
                return Ok((None, found));
            }
            let grown = thresholds(&read(ctx, &grown)?);
            let grown = anchored_fit(ctx, &grown, q, pixel, opts.anchor_refits)?;
            Ok((Some(grown), found))
        });
        match grown {
            Ok((grown, found)) => {
                stages.push(StageRecord::GeometrySearch(Ok(found)));
                if let Some(grown) = grown {
                    track = grown;
                }
            }
            Err(e) => stages.push(StageRecord::GeometrySearch(Err(e))),
        }
    }

    track = thresholds(&track);
    if opts.clean {
        track = clean(ctx, track, q, pixel, opts, stages);
    }
    gate(track, q, pixel, opts, stages)
}

/// The final gates: the queried sighting `in` and on the pixel, enough views,
/// a good median ZNCC, and every view near the point's projection.
fn gate(
    track: EditableTrack,
    q: usize,
    pixel: [f64; 2],
    opts: &FinishOptions,
    stages: &mut Vec<StageRecord>,
) -> Result<EditableTrack, Refusal> {
    let offset = query_offset(&track, q, pixel);
    let n_in = in_count(&track);
    let zncc = median_zncc(&track);
    let mut record = |worst: Option<f64>| {
        stages.push(StageRecord::Final {
            in_views: n_in,
            zncc_median: zncc,
            query_offset_px: offset,
            max_projection_offset_px: worst,
        })
    };
    let refuse = |reason: String| Err(Refusal::new(RefusalStage::Gate, reason));
    if track.observations[q].verdict != Verdict::In {
        record(None);
        return refuse("the queried sighting did not survive the final thresholds".into());
    }
    match offset {
        Some(d) if d <= opts.max_query_offset_px => {}
        _ => {
            record(None);
            return refuse(format!(
                "the fitted sighting sits {:.1} px from the pixel asked about (bar {:.1} px)",
                offset.unwrap_or(f64::NAN),
                opts.max_query_offset_px
            ));
        }
    }
    if n_in < opts.min_in_views {
        record(None);
        return refuse(format!(
            "only {n_in} views kept (bar {})",
            opts.min_in_views
        ));
    }
    if zncc < opts.min_zncc_median {
        record(None);
        return refuse(format!(
            "median ZNCC {zncc:.3} is under {}",
            opts.min_zncc_median
        ));
    }
    let worst = max_projection_offset(&track);
    record(Some(worst));
    if worst > opts.max_projection_offset_px {
        return refuse(format!(
            "a kept view sits {worst:.1} px from the point's projection (bar {:.1} px)",
            opts.max_projection_offset_px
        ));
    }
    Ok(track)
}
