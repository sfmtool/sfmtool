// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The four ways of finding a pixel's sightings in the other photographs.
//!
//! Each member ends in [`finish`], and each refuses with its own stage when it
//! finds nothing to finish. The three that try several candidates (clusters,
//! depth modes, surface hypotheses) fit every one, score each by its `in` views
//! weighted by the median ZNCC, and finish the best.

use nalgebra::{DMatrix, Vector3};

use crate::bench::search::{search_descriptors, Found, SearchOptions};
use crate::bench::stage::set_stage;
use crate::bench::steps::tilt_patch;
use crate::bench::track::{EditableTrack, StageKind};
use crate::features::kdforest::{radius_for_feature_count, ConstellationParams};
use crate::numeric::median_in_place;
use crate::progress::Progress;

use super::finish::{
    anchored_fit, at_infinity, finish, fit_track, in_count, median_zncc, position, read,
    score_track, seed_cluster, shape_track, thresholds, track_from_sightings,
};
use super::neighbourhood::{MatchesClusters, NearbyCluster, NearbyObservation};
use super::{
    CandidateKind, CandidateRecord, ClustersOptions, Ctx, HypothesisRecord, LateralRecord,
    LocalPriorRecord, Refusal, RefusalStage, StageRecord, TiltRecord, TrackAtPixelOptions,
    TransferOptions,
};

/// Other photographs' sightings of the pixel, as `(image, pixel)`.
type Sightings = Vec<(u32, [f64; 2])>;

/// One neighbour's keypoint in the queried image, its keypoint in another, and
/// the pair's weight.
type KeypointPair = ([f64; 2], [f64; 2], f64);

/// How far inside the frame a carried pixel has to land, in px.
const IN_FRAME_MARGIN_PX: f64 = 2.0;

/// Views seen through the reconstruction's own surface: when the observations
/// within this radius of the projection sit, at their median, nearer than
/// [`OCCLUSION_RATIO`] of the point's depth, something is in front of it.
const OCCLUSION_RADIUS_PX: f64 = 12.0;

/// See [`OCCLUSION_RADIUS_PX`].
const OCCLUSION_RATIO: f64 = 0.93;

/// `x` held inside `[lo, hi]` the way `numpy.clip` holds it: the lower bound
/// first, so bounds given in the wrong order give `hi` rather than a panic.
fn clip(x: f64, lo: f64, hi: f64) -> f64 {
    x.max(lo).min(hi)
}

/// The best-scoring of several fitted candidates.
struct Best {
    track: Option<EditableTrack>,
    score: f64,
}

impl Best {
    fn new() -> Self {
        Self {
            track: None,
            score: -1.0,
        }
    }

    /// Score a fitted candidate, keep it when it is the best so far, and write
    /// the score or the reason into `record`.
    fn offer(&mut self, fitted: Result<EditableTrack, String>, record: &mut CandidateRecord) {
        match fitted {
            Ok(track) => {
                let track = thresholds(&track);
                let s = score_track(&track);
                record.score = Some(s);
                if s > self.score {
                    self.track = Some(track);
                    self.score = s;
                }
            }
            Err(e) => record.error = Some(e),
        }
    }
}

/// Upgrade the sightings to a track and fit it, anchored on the pixel.
fn fit_sightings(
    ctx: &Ctx<'_>,
    image: u32,
    pixel: [f64; 2],
    radius_px: f64,
    sightings: &[(u32, [f64; 2])],
    refits: usize,
) -> Result<EditableTrack, String> {
    let track = track_from_sightings(ctx, image, pixel, radius_px, sightings)?;
    anchored_fit(ctx, &track, 0, pixel, refits)
}

// ---- Clusters ------------------------------------------------------------

/// The `.matches` clusters near the pixel, the pixel carried into each kept
/// member's image through the two members' affine shapes.
pub(super) fn clusters(
    ctx: &Ctx<'_>,
    image: u32,
    pixel: [f64; 2],
    options: &TrackAtPixelOptions,
    stages: &mut Vec<StageRecord>,
) -> Result<EditableTrack, Refusal> {
    let opts = &options.clusters;
    let Some(clusters) = ctx.sources.clusters else {
        return Err(Refusal::new(
            RefusalStage::Clusters,
            "no cluster-patches .matches clusters were given to read",
        ));
    };
    let near = clusters.near(image, pixel, opts.search_radius_px);
    stages.push(StageRecord::NearbyClusters { count: near.len() });
    if near.is_empty() {
        return Err(Refusal::new(
            RefusalStage::Clusters,
            format!(
                "no cluster has a member within {:.0} px of the pixel",
                opts.search_radius_px
            ),
        ));
    }
    let mut best = Best::new();
    let mut tried = Vec::new();
    for cluster in near.iter().take(opts.max_clusters) {
        let mut record = CandidateRecord {
            candidate: CandidateKind::Cluster {
                cluster: cluster.cluster,
                distance_px: cluster.distance_px,
            },
            sightings: 0,
            score: None,
            error: None,
        };
        match cluster_sightings(ctx, clusters, image, pixel, cluster, opts) {
            Some((sightings, scale)) if !sightings.is_empty() => {
                let radius_px = clip(
                    opts.radius_in_scales * scale,
                    opts.min_radius_px,
                    opts.max_radius_px,
                );
                record.sightings = sightings.len();
                let fitted = fit_sightings(
                    ctx,
                    image,
                    pixel,
                    radius_px,
                    &sightings,
                    options.finish.anchor_refits,
                );
                best.offer(fitted, &mut record);
            }
            _ => {
                record.error = Some("no kept member in another image to carry the pixel to".into())
            }
        }
        tried.push(record);
    }
    stages.push(StageRecord::Candidates { tried });
    let Some(track) = best.track else {
        return Err(Refusal::new(
            RefusalStage::Clusters,
            "no nearby cluster carried the pixel into another photograph",
        ));
    };
    finish(ctx, track, 0, pixel, &options.finish, stages)
}

/// The pixel carried into every other image the cluster's kept members (and
/// its reference) reach, one per image, with the queried member's scale.
///
/// The offset from the member to the pixel is a step in the queried image;
/// `shape_other * inverse(shape_query)` maps it to the matching step in the
/// other. Where two members share an image, the one with the higher ZNCC is
/// taken (a member with no ZNCC counts as 1). `None` when the member's shape is
/// degenerate or the pixel is further from it than the options allow.
fn cluster_sightings(
    ctx: &Ctx<'_>,
    clusters: &MatchesClusters,
    image: u32,
    pixel: [f64; 2],
    cluster: &NearbyCluster,
    opts: &ClustersOptions,
) -> Option<(Sightings, f64)> {
    let m = clusters.member(cluster.member);
    let s = m.shape;
    let det = s[0][0] * s[1][1] - s[0][1] * s[1][0];
    if det.abs() <= 1e-12 {
        return None;
    }
    let scale = det.abs().sqrt();
    let offset = [pixel[0] - m.position[0], pixel[1] - m.position[1]];
    if (offset[0] * offset[0] + offset[1] * offset[1]).sqrt() > opts.max_offset_in_scales * scale {
        return None;
    }
    // inverse(shape_query) @ offset.
    let step = [
        (s[1][1] * offset[0] - s[0][1] * offset[1]) / det,
        (-s[1][0] * offset[0] + s[0][0] * offset[1]) / det,
    ];
    let mut by_image: Vec<(u32, f64, [f64; 2])> = Vec::new();
    for k in cluster.members.clone() {
        let o = clusters.member(k);
        let Some(other) = o.image else {
            continue;
        };
        if other == image || !o.is_kept_or_reference() {
            continue;
        }
        let t = o.shape;
        let pred = [
            o.position[0] + t[0][0] * step[0] + t[0][1] * step[1],
            o.position[1] + t[1][0] * step[0] + t[1][1] * step[1],
        ];
        if !ctx.cameras[other as usize].in_frame(pred, IN_FRAME_MARGIN_PX) {
            continue;
        }
        let z = if o.zncc.is_finite() { o.zncc } else { 1.0 };
        match by_image.iter_mut().find(|e| e.0 == other) {
            Some(entry) => {
                if z > entry.1 {
                    entry.1 = z;
                    entry.2 = pred;
                }
            }
            None => by_image.push((other, z, pred)),
        }
    }
    Some((
        by_image.into_iter().map(|(i, _, p)| (i, p)).collect(),
        scale,
    ))
}

// ---- Neighbourhood depth modes ---------------------------------------------

/// The finite neighbours in front of the camera, sorted by depth and split
/// where one depth exceeds the one before it by more than `gap`.
pub(super) fn depth_modes(near: &[NearbyObservation], gap: f64) -> Vec<Vec<&NearbyObservation>> {
    let mut finite: Vec<(&NearbyObservation, f64)> = near
        .iter()
        .filter_map(|o| Some((o, o.positive_depth()?)))
        .collect();
    finite.sort_by(|a, b| a.1.total_cmp(&b.1));
    let mut modes = Vec::new();
    let mut current: Vec<(&NearbyObservation, f64)> = Vec::new();
    for entry in finite {
        if let Some(last) = current.last() {
            if entry.1 / last.1 > gap {
                modes.push(std::mem::take(&mut current));
            }
        }
        current.push(entry);
    }
    if !current.is_empty() {
        modes.push(current);
    }
    modes
        .into_iter()
        .map(|m| m.into_iter().map(|(o, _)| o).collect())
        .collect()
}

/// The median of the finite apparent half-widths among `neighbours`, or
/// `default` when none has one, held inside `[lo, hi]`.
fn half_px_of<'a>(
    neighbours: impl Iterator<Item = &'a NearbyObservation>,
    default: f64,
    lo: f64,
    hi: f64,
) -> f64 {
    let mut half: Vec<f64> = neighbours
        .map(|o| o.half_px)
        .filter(|h| h.is_finite())
        .collect();
    clip(
        if half.is_empty() {
            default
        } else {
            median_in_place(&mut half)
        },
        lo,
        hi,
    )
}

// ---- Transfer --------------------------------------------------------------

/// The neighbours' own matched keypoints, through a local affine map per
/// image.
pub(super) fn transfer(
    ctx: &Ctx<'_>,
    image: u32,
    pixel: [f64; 2],
    options: &TrackAtPixelOptions,
    stages: &mut Vec<StageRecord>,
) -> Result<EditableTrack, Refusal> {
    let opts = &options.transfer;
    let near = ctx.observations_near(image, pixel, opts.neighbour_radius_px);
    let mut modes: Vec<_> = depth_modes(&near, opts.depth_mode_gap)
        .into_iter()
        .filter(|m| m.len() >= opts.min_pairs)
        .collect();
    let nearest = |m: &[&NearbyObservation]| {
        m.iter()
            .map(|o| o.distance_px)
            .fold(f64::INFINITY, f64::min)
    };
    modes.sort_by(|a, b| nearest(a).total_cmp(&nearest(b)));
    stages.push(StageRecord::DepthModes {
        sizes: modes.iter().map(Vec::len).collect(),
    });
    if modes.is_empty() {
        return Err(Refusal::new(
            RefusalStage::Neighbourhood,
            format!(
                "fewer than {} reconstructed points of one surface lie within {:.0} px of the pixel",
                opts.min_pairs, opts.neighbour_radius_px
            ),
        ));
    }
    let mut best = Best::new();
    let mut tried = Vec::new();
    for mode in &modes {
        let (sightings, mut images) = transfer_sightings(ctx, image, pixel, mode, opts);
        images.sort_unstable();
        let mut record = CandidateRecord {
            candidate: CandidateKind::DepthMode {
                support: mode.len(),
                images,
            },
            sightings: sightings.len(),
            score: None,
            error: None,
        };
        if sightings.is_empty() {
            record.error = Some("no other photograph shares enough of these neighbours".into());
            tried.push(record);
            continue;
        }
        // The mode in depth order, as the modes are built: the first eight
        // by depth set the size, not the eight nearest the pixel.
        let radius_px = half_px_of(
            mode.iter().take(8).copied(),
            opts.default_radius_px,
            opts.min_radius_px,
            opts.max_radius_px,
        );
        let fitted = fit_sightings(
            ctx,
            image,
            pixel,
            radius_px,
            &sightings,
            options.finish.anchor_refits,
        );
        best.offer(fitted, &mut record);
        tried.push(record);
    }
    stages.push(StageRecord::Candidates { tried });
    let Some(track) = best.track else {
        return Err(Refusal::new(
            RefusalStage::Transfer,
            "no neighbourhood carried the pixel into another photograph",
        ));
    };
    finish(ctx, track, 0, pixel, &options.finish, stages)
}

/// The pixel carried into each other image by an affine map fitted to the
/// mode's keypoint pairs, and the images a map was fitted for.
///
/// For every other image, the pairs `(keypoint here, keypoint there)` of the
/// mode's nearest points seen there are weighted by nearness to the pixel. A
/// weighted least-squares affine is fitted, the pairs it misses by more than
/// `max_residual_px` are dropped and it is fitted again, for at most three
/// fits; the map is kept when it still rests on `min_pairs` pairs and does not
/// mirror the image.
fn transfer_sightings(
    ctx: &Ctx<'_>,
    image: u32,
    pixel: [f64; 2],
    mode: &[&NearbyObservation],
    opts: &TransferOptions,
) -> (Vec<(u32, [f64; 2])>, Vec<u32>) {
    let mut nearest: Vec<&NearbyObservation> = mode.to_vec();
    nearest.sort_by(|a, b| a.distance_px.total_cmp(&b.distance_px));
    nearest.truncate(opts.max_neighbours);
    // Keyed by image in the order the images are first met.
    let mut pairs: Vec<(u32, Vec<KeypointPair>)> = Vec::new();
    for o in nearest {
        for (other, kp) in ctx.observations.point_observations(o.point) {
            if other == image {
                continue;
            }
            let row = (o.keypoint, kp, 1.0 / (1.0 + o.distance_px));
            match pairs.iter_mut().find(|(i, _)| *i == other) {
                Some((_, rows)) => rows.push(row),
                None => pairs.push((other, vec![row])),
            }
        }
    }
    let mut sightings = Vec::new();
    let mut images = Vec::new();
    for (other, rows) in pairs {
        if rows.len() < opts.min_pairs {
            continue;
        }
        let src: Vec<[f64; 2]> = rows
            .iter()
            .map(|r| [r.0[0] - pixel[0], r.0[1] - pixel[1]])
            .collect();
        let dst: Vec<[f64; 2]> = rows.iter().map(|r| r.1).collect();
        let w: Vec<f64> = rows.iter().map(|r| r.2).collect();
        let mut keep = vec![true; rows.len()];
        let mut model = None;
        for _ in 0..3 {
            if keep.iter().filter(|&&k| k).count() < opts.min_pairs {
                model = None;
                break;
            }
            model = fit_affine(&src, &dst, &w, &keep);
            let Some((a, b)) = model else {
                break;
            };
            let new_keep: Vec<bool> = src
                .iter()
                .zip(&dst)
                .map(|(s, d)| {
                    let px = a[0][0] * s[0] + a[0][1] * s[1] + b[0] - d[0];
                    let py = a[1][0] * s[0] + a[1][1] * s[1] + b[1] - d[1];
                    (px * px + py * py).sqrt() <= opts.max_residual_px
                })
                .collect();
            if new_keep == keep {
                break;
            }
            keep = new_keep;
        }
        let Some((a, b)) = model else {
            continue;
        };
        if keep.iter().filter(|&&k| k).count() < opts.min_pairs {
            continue;
        }
        if a[0][0] * a[1][1] - a[0][1] * a[1][0] <= 0.0 {
            continue;
        }
        // The pixel is the origin of the source coordinates, so it lands at b.
        if !ctx.cameras[other as usize].in_frame(b, IN_FRAME_MARGIN_PX) {
            continue;
        }
        sightings.push((other, b));
        images.push(other);
    }
    (sightings, images)
}

/// The weighted least-squares affine `dst ~ A src + b` over the kept rows, as
/// `(A, b)`, or `None` when the solution is not finite.
///
/// Solved through the SVD with singular values under `eps * max(rows, 3)`
/// times the largest treated as zero, which is the minimum-norm solution
/// `numpy.linalg.lstsq` gives with its default cutoff.
pub(super) fn fit_affine(
    src: &[[f64; 2]],
    dst: &[[f64; 2]],
    w: &[f64],
    keep: &[bool],
) -> Option<([[f64; 2]; 2], [f64; 2])> {
    let rows: Vec<usize> = (0..src.len()).filter(|&i| keep[i]).collect();
    let n = rows.len();
    let mut x = DMatrix::<f64>::zeros(n, 3);
    let mut y = DMatrix::<f64>::zeros(n, 2);
    for (r, &i) in rows.iter().enumerate() {
        let sw = w[i].sqrt();
        x[(r, 0)] = src[i][0] * sw;
        x[(r, 1)] = src[i][1] * sw;
        x[(r, 2)] = sw;
        y[(r, 0)] = dst[i][0] * sw;
        y[(r, 1)] = dst[i][1] * sw;
    }
    let svd = x.svd(true, true);
    let (u, v_t) = (svd.u?, svd.v_t?);
    let s_max = svd.singular_values.iter().copied().fold(0.0, f64::max);
    let cutoff = f64::EPSILON * n.max(3) as f64 * s_max;
    // sol = V diag(1/s) U^T Y over the singular values above the cutoff.
    let mut sol = DMatrix::<f64>::zeros(3, 2);
    for (k, &s) in svd.singular_values.iter().enumerate() {
        if s <= cutoff {
            continue;
        }
        let uty = u.column(k).transpose() * &y;
        sol += v_t.row(k).transpose() * (uty / s);
    }
    if sol.iter().any(|v| !v.is_finite()) {
        return None;
    }
    Some((
        [[sol[(0, 0)], sol[(1, 0)]], [sol[(0, 1)], sol[(1, 1)]]],
        [sol[(2, 0)], sol[(2, 1)]],
    ))
}

// ---- Sweep -------------------------------------------------------------------

/// One surface hypothesis: where the pixel's ray meets a depth mode's plane.
struct Hypothesis {
    xyz: Vector3<f64>,
    normal: Vector3<f64>,
    half_px: f64,
    half: f64,
    record: HypothesisRecord,
}

/// A plane through each depth mode of the neighbours, met by the pixel's ray.
pub(super) fn sweep(
    ctx: &Ctx<'_>,
    image: u32,
    pixel: [f64; 2],
    options: &TrackAtPixelOptions,
    stages: &mut Vec<StageRecord>,
) -> Result<EditableTrack, Refusal> {
    let opts = &options.sweep;
    let hypotheses = surface_hypotheses(ctx, image, pixel, options);
    stages.push(StageRecord::Hypotheses {
        hypotheses: hypotheses.iter().map(|h| h.record.clone()).collect(),
    });
    if hypotheses.is_empty() {
        return Err(Refusal::new(
            RefusalStage::Prior,
            "no reconstructed point near the pixel says what surface it is on",
        ));
    }
    let mut best = Best::new();
    let mut tried = Vec::new();
    for h in &hypotheses {
        let mut views = visible_views(ctx, &h.xyz, &h.normal, image, opts.max_view_angle_deg);
        views.truncate(opts.seed_views);
        let sightings: Vec<(u32, [f64; 2])> = views.iter().map(|v| (v.0, v.1)).collect();
        let mut record = CandidateRecord {
            candidate: CandidateKind::Hypothesis {
                views: views.iter().map(|v| v.0).collect(),
            },
            sightings: sightings.len(),
            score: None,
            error: None,
        };
        if sightings.is_empty() {
            record.error = Some("no photograph sees the hypothesis".into());
            tried.push(record);
            continue;
        }
        let fitted = track_from_sightings(ctx, image, pixel, h.half_px, &sightings).and_then(|t| {
            let shaped = shape_track(ctx, &t, 0, Some(h.normal), Some(h.half));
            anchored_fit(ctx, &shaped, 0, pixel, options.finish.anchor_refits)
        });
        best.offer(fitted, &mut record);
        tried.push(record);
    }
    stages.push(StageRecord::Candidates { tried });
    let Some(track) = best.track else {
        return Err(Refusal::new(
            RefusalStage::Hypothesis,
            "no surface hypothesis could be fitted in any photograph",
        ));
    };
    finish(ctx, track, 0, pixel, &options.finish, stages)
}

/// One hypothesis per depth mode of the neighbours, nearest first.
///
/// A mode's plane passes through its points' distance-weighted centroid with
/// their distance-weighted mean normal, turned to face the camera. Where the
/// pixel's ray meets that plane at a grazing angle (the cosine under 0.15) the
/// ray is taken as far as the centroid instead.
fn surface_hypotheses(
    ctx: &Ctx<'_>,
    image: u32,
    pixel: [f64; 2],
    options: &TrackAtPixelOptions,
) -> Vec<Hypothesis> {
    let opts = &options.sweep;
    let cam = &ctx.cameras[image as usize];
    let ray = cam.ray(pixel);
    let center = cam.center;
    let near = ctx.observations_near(image, pixel, opts.prior_radius_px);
    let modes = depth_modes(&near, opts.depth_mode_gap);
    let mut out = Vec::new();
    for mode in &modes {
        if mode.len() < opts.min_mode_size && modes.len() > 1 {
            continue;
        }
        let mut same: Vec<&NearbyObservation> = mode.clone();
        same.sort_by(|a, b| a.distance_px.total_cmp(&b.distance_px));
        same.truncate(opts.prior_k);
        let weights: Vec<f64> = same.iter().map(|o| 1.0 / (1.0 + o.distance_px)).collect();
        let positions: Vec<Vector3<f64>> = same
            .iter()
            .map(|o| {
                ctx.observations
                    .position(o.point)
                    .map_or_else(Vector3::zeros, |p| p.coords)
            })
            .collect();
        let mut normal: Vector3<f64> = same.iter().zip(&weights).map(|(o, w)| o.normal * *w).sum();
        if normal.norm() < 1e-9 {
            continue;
        }
        normal /= normal.norm();
        let mean: Vector3<f64> = positions.iter().sum::<Vector3<f64>>() / positions.len() as f64;
        if normal.dot(&(center - mean)) < 0.0 {
            normal = -normal;
        }
        let weight_sum: f64 = weights.iter().sum();
        let centroid: Vector3<f64> = positions
            .iter()
            .zip(&weights)
            .map(|(p, w)| p * *w)
            .sum::<Vector3<f64>>()
            / weight_sum;
        let denom = normal.dot(&ray);
        let t = if denom.abs() > 0.15 {
            normal.dot(&(centroid - center)) / denom
        } else {
            (centroid - center).norm()
        };
        if t <= 0.0 {
            continue;
        }
        let xyz = center + ray * t;
        let depth = cam.depth(&xyz);
        let half_px = half_px_of(
            same.iter().copied(),
            opts.default_radius_px,
            opts.min_radius_px,
            opts.max_radius_px,
        );
        out.push(Hypothesis {
            xyz,
            normal,
            half_px,
            half: half_px * depth / cam.focal,
            record: HypothesisRecord {
                support: mode.len(),
                half_px,
                nearest_px: same[0].distance_px,
            },
        });
    }
    out.sort_by(|a, b| a.record.nearest_px.total_cmp(&b.record.nearest_px));
    out
}

/// The images other than `exclude` that plausibly see a surface point at `xyz`
/// facing `normal`, with its pixel there and the viewing angle, most nearly
/// face-on first.
///
/// A view is kept when the point projects inside the frame, the camera sits
/// within `max_view_angle_deg` of the normal, and the reconstruction's own
/// surface there is not in front of the point.
fn visible_views(
    ctx: &Ctx<'_>,
    xyz: &Vector3<f64>,
    normal: &Vector3<f64>,
    exclude: u32,
    max_view_angle_deg: f64,
) -> Vec<(u32, [f64; 2], f64)> {
    let cos_max = max_view_angle_deg.to_radians().cos();
    let mut out = Vec::new();
    for (other, cam) in ctx.cameras.iter().enumerate() {
        let other = other as u32;
        if other == exclude {
            continue;
        }
        let Some(px) = cam.project(xyz) else {
            continue;
        };
        if !cam.in_frame(px, IN_FRAME_MARGIN_PX) {
            continue;
        }
        let to_cam = (cam.center - xyz).normalize();
        let c = normal.dot(&to_cam);
        if c < cos_max {
            continue;
        }
        let angle = c.clamp(-1.0, 1.0).acos().to_degrees();
        let depth = cam.depth(xyz);
        let mut near: Vec<f64> = ctx
            .observations_near(other, px, OCCLUSION_RADIUS_PX)
            .iter()
            .filter_map(NearbyObservation::stated_depth)
            .collect();
        if !near.is_empty() && median_in_place(&mut near) < OCCLUSION_RATIO * depth {
            continue;
        }
        out.push((other, px, angle));
    }
    out.sort_by(|a, b| a.2.total_cmp(&b.2));
    out
}

// ---- Constellation ---------------------------------------------------------

/// The size, normal and depth structure the neighbours of the pixel suggest.
fn local_prior(
    ctx: &Ctx<'_>,
    image: u32,
    pixel: [f64; 2],
    options: &TrackAtPixelOptions,
) -> LocalPriorRecord {
    let opts = &options.constellation;
    let near = ctx.observations_near(image, pixel, opts.prior_radius_px);
    let mut prior = LocalPriorRecord {
        neighbours: near.len(),
        depth_modes: Vec::new(),
        edge: false,
        depth: None,
        half_px: None,
        normal: None,
        normal_spread_deg: None,
    };
    let finite: Vec<(&NearbyObservation, f64)> = near
        .iter()
        .filter_map(|o| Some((o, o.positive_depth()?)))
        .collect();
    let Some(&(_, nearest_depth)) = finite.first() else {
        return prior;
    };
    let mut depths: Vec<f64> = finite.iter().map(|f| f.1).collect();
    depths.sort_by(f64::total_cmp);
    let mut modes: Vec<Vec<f64>> = Vec::new();
    let mut start = 0;
    for i in 1..depths.len() {
        if depths[i] / depths[i - 1] > opts.depth_mode_gap {
            modes.push(depths[start..i].to_vec());
            start = i;
        }
    }
    modes.push(depths[start..].to_vec());
    prior.depth_modes = modes
        .iter()
        .map(|m| (median_in_place(&mut m.clone()), m.len()))
        .collect();
    prior.edge = modes.iter().filter(|m| m.len() >= 2).count() >= 2;

    // The surface the pixel is most likely on is its nearest neighbour's; the
    // size and normal come from that neighbour's depth mode only.
    let Some(mode) = modes
        .iter()
        .find(|m| m[0] <= nearest_depth && nearest_depth <= m[m.len() - 1])
    else {
        return prior;
    };
    let (lo, hi) = (mode[0], mode[mode.len() - 1]);
    let same: Vec<(&NearbyObservation, f64)> = finite
        .iter()
        .copied()
        .filter(|&(_, d)| lo <= d && d <= hi)
        .take(opts.prior_k)
        .collect();
    prior.depth = Some(median_in_place(
        &mut same.iter().map(|s| s.1).collect::<Vec<_>>(),
    ));
    let mut half: Vec<f64> = same
        .iter()
        .map(|s| s.0.half_px)
        .filter(|h| h.is_finite())
        .collect();
    if !half.is_empty() {
        prior.half_px = Some(median_in_place(&mut half));
    }
    let mean: Vector3<f64> = same
        .iter()
        .map(|(o, _)| o.normal / (1.0 + o.distance_px))
        .sum();
    if mean.norm() > 1e-9 {
        let mean = mean / mean.norm();
        prior.normal = Some(mean);
        let spread: f64 = same
            .iter()
            .map(|(o, _)| o.normal.dot(&mean).clamp(-1.0, 1.0).acos().to_degrees())
            .sum::<f64>()
            / same.len() as f64;
        prior.normal_spread_deg = Some(spread);
    }
    prior
}

/// The SIFT index's constellation query from the pixel, then the cluster
/// stage, the upgrade and a tilt toward the local prior's normal.
pub(super) fn constellation(
    ctx: &Ctx<'_>,
    image: u32,
    pixel: [f64; 2],
    options: &TrackAtPixelOptions,
    stages: &mut Vec<StageRecord>,
) -> Result<EditableTrack, Refusal> {
    let opts = &options.constellation;
    let refuse = |stage, reason: String| Err(Refusal::new(stage, reason));
    let Some(index) = ctx.sources.sift_index else {
        return refuse(
            RefusalStage::Constellation,
            "no SIFT index was given to search".into(),
        );
    };

    // Local prior and the cluster at the pixel.
    let prior = local_prior(ctx, image, pixel, options);
    let radius_px = clip(
        prior.half_px.unwrap_or(opts.default_radius_px),
        opts.min_radius_px,
        opts.max_radius_px,
    );
    let prior_normal = prior.normal;
    stages.push(StageRecord::LocalPrior { prior, radius_px });
    let mut track = match seed_cluster(ctx, image, pixel, radius_px) {
        Ok(t) => t,
        Err(e) => return refuse(RefusalStage::Constellation, e),
    };

    // Constellation search, then lateral searches from the best finds.
    let search_options = |radius: f32| SearchOptions {
        constellation: ConstellationParams {
            min_inliers: opts.min_inliers,
            ..ConstellationParams::DEFAULT
        },
        radius_px: radius,
        min_inliers: opts.min_inliers,
    };
    let radius_in = |i: u32| {
        let cam = &ctx.cameras[i as usize];
        radius_for_feature_count(
            cam.width,
            cam.height,
            index.keypoints[i as usize].positions.len(),
            opts.constellation_target,
        )
    };
    let search_radius = radius_in(image);
    let found = match search_descriptors(
        &track,
        0,
        &index.keypoints[image as usize],
        index.forest,
        &search_options(search_radius),
        &Progress::none(),
    ) {
        Ok((next, report)) => {
            track = next;
            report
        }
        Err(e) => return refuse(RefusalStage::Constellation, e.to_string()),
    };
    stages.push(StageRecord::Constellation {
        search_radius_px: f64::from(search_radius),
        keypoints: found.constellation,
        images: found.matches.iter().map(|m| (m.image, m.inliers)).collect(),
    });
    let mut added: Vec<(u32, usize, usize)> = found
        .matches
        .iter()
        .filter_map(|m| match m.found {
            Found::Added { observation } => Some((m.image, observation, m.inliers)),
            _ => None,
        })
        .collect();
    added.sort_by_key(|a| std::cmp::Reverse(a.2));
    let mut lateral = Vec::new();
    for &(other, observation, _) in added.iter().take(opts.lateral_searches) {
        let result = search_descriptors(
            &track,
            observation,
            &index.keypoints[other as usize],
            index.forest,
            &search_options(radius_in(other)),
            &Progress::none(),
        );
        lateral.push(LateralRecord {
            from: other,
            added: match result {
                Ok((next, more)) => {
                    track = next;
                    Ok(more.added())
                }
                Err(e) => Err(e.to_string()),
            },
        });
    }
    stages.push(StageRecord::Lateral { searches: lateral });
    if track.observations.len() < 2 {
        return refuse(
            RefusalStage::Constellation,
            format!(
                "the constellation of {} keypoints within {:.0} px matched no other image with \
                 {}+ inliers",
                found.constellation, search_radius, opts.min_inliers
            ),
        );
    }

    // Cluster evaluation and the thresholds' verdicts.
    track = match read(ctx, &track) {
        Ok(t) => thresholds(&t),
        Err(e) => return refuse(RefusalStage::ClusterEvaluate, e),
    };
    stages.push(StageRecord::ClusterEvaluate {
        observations: track.observations.len(),
        in_views: in_count(&track),
    });
    if in_count(&track) < 2 {
        return refuse(
            RefusalStage::ClusterEvaluate,
            format!(
                "{} candidate sighting(s) found, none registered against the queried patch well \
                 enough to keep",
                track.observations.len() - 1
            ),
        );
    }

    // Upgrade to the track stage.
    let (upgraded, staged) = match set_stage(
        &track,
        ctx.edited,
        ctx.views,
        StageKind::Track,
        &crate::bench::fit::FitOptions::default(),
        &Progress::none(),
    ) {
        Ok(done) => done,
        Err(e) => return refuse(RefusalStage::Upgrade, e.to_string()),
    };
    track = upgraded;
    stages.push(StageRecord::Upgrade {
        at_infinity: at_infinity(&track),
        reason: staged
            .fit
            .as_ref()
            .and_then(|f| f.classification)
            .map(|c| c.reason),
        zncc_median: median_zncc(&track),
    });

    // Tilt toward the prior's normal and refit; keep whichever reads better.
    if opts.normal_prior && !at_infinity(&track) {
        if let (Some(mut normal), Some(p)) = (prior_normal, position(&track)) {
            if normal.dot(&(ctx.cameras[image as usize].center - p)) < 0.0 {
                normal = -normal;
            }
            let record = tilt_patch(&track, ctx.edited, normal)
                .map_err(|e| e.to_string())
                .and_then(|(tilted, tilt)| Ok((fit_track(ctx, &tilted)?, tilt)))
                .map(|(tilted, tilt)| {
                    let before = median_zncc(&track);
                    let after = median_zncc(&tilted);
                    let kept = after >= before;
                    if kept {
                        track = tilted;
                    }
                    TiltRecord {
                        degrees: Some(tilt.degrees),
                        stopped: tilt.stopped.is_some(),
                        zncc_before: before,
                        zncc_after: after,
                        kept,
                    }
                });
            stages.push(StageRecord::PriorTilt(record));
        }
    }

    finish(ctx, track, 0, pixel, &options.finish, stages)
}
