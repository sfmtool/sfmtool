// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Re-reading every point of a track set from its own observations at one
//! geometry, and deciding per track what those observations support.
//!
//! [`super::triangulation::triangulate_batch`] answers where a track's rays
//! come closest and how well the depth was observed. It does not say whether
//! that answer should be used. Whether a track with parallel rays is a bearing,
//! whether a point behind a camera is demoted now or left for a later trim and
//! whether that demotion reads the track or the one observation that failed,
//! whether a single observation still carries a direction, and whether a fresh
//! estimate has to reproject inside a bound before it counts are the caller's
//! rules. This module holds them once, as options with an off position, so a
//! caller states its policy and the arithmetic is shared. With every option off
//! the operation is the batch triangulation solve.
//!
//! See `specs/core/reconstruction/point-estimation.md` for the design.

use nalgebra::{Point3, Quaternion, UnitQuaternion, Vector3};
use rayon::prelude::*;

use crate::camera::CameraIntrinsics;
use crate::numeric::median_in_place;
use crate::reconstruction::triangulation::triangulate_batch;

/// The direction a track with no usable ray at all falls back to: the camera
/// convention's forward direction.
pub const FALLBACK_DIRECTION: [f64; 3] = [0.0, 0.0, -1.0];

/// What a track with fewer than two usable observations becomes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FewObservations {
    /// The estimate is `NaN`: the track is not observed at this geometry. This
    /// is the off position.
    #[default]
    Absent,
    /// Its single ray is its direction, or [`FALLBACK_DIRECTION`] where it has
    /// none.
    Bearing,
}

/// The rules a track is judged by, each with an off position.
///
/// [`PointRules::default`] is every rule off, which makes the operation the
/// batch triangulation solve.
#[derive(Debug, Clone, Copy, Default)]
pub struct PointRules {
    /// The angular floor, in radians. A track whose widest ray pair subtends
    /// less than this is thin. `None` is off.
    pub floor_rad: Option<f64>,
    /// Demote a solved point that lands behind any camera observing it. Off,
    /// the point is kept and the in-front flag is reported.
    pub cheirality: bool,
    /// Read a cheirality failure per observation rather than per track: where
    /// the observations that see the solved point behind them are a strict
    /// minority of the track's usable observations, drop them, solve again on
    /// the survivors, and read the rules over the reduced track. Off, one
    /// observation behind the point demotes the whole track. Needs
    /// [`Self::cheirality`] on, which is the rule this one reads.
    pub prune_behind: bool,
    /// The pixel bound a fresh estimate has to reproject inside of. `None` is
    /// off. Reading it needs the observation form.
    pub bar_px: Option<f64>,
    /// What a track with fewer than two usable observations becomes.
    pub few: FewObservations,
}

/// Which rule decided a track.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum PointVerdict {
    /// The rays were solved and every rule in force admitted the result.
    Finite = 0,
    /// The caller marked the track a direction, so it was not solved.
    Marked = 1,
    /// The widest ray pair subtends less than the floor.
    Thin = 2,
    /// The solved point lands behind a camera that observes it.
    Behind = 3,
    /// The median observation reprojects past the bar.
    OverBar = 4,
    /// Fewer than two usable observations.
    Few = 5,
    /// A minority of the track's observations saw the solved point behind them
    /// and were dropped; the solve over the survivors was admitted.
    FinitePruned = 6,
}

impl PointVerdict {
    /// The verdict's wire code, the value the bindings hand to Python.
    pub fn code(self) -> u8 {
        self as u8
    }
}

/// How many tracks each rule took, and what the finite ones look like.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct PointCensus {
    /// Tracks the operation was given.
    pub seen: usize,
    /// Tracks solved and admitted.
    pub finite: usize,
    /// Tracks the caller had already marked directions.
    pub marked: usize,
    /// Tracks refused by the floor.
    pub thin: usize,
    /// Tracks refused by cheirality.
    pub behind: usize,
    /// Tracks refused by the reprojection bar.
    pub over_bar: usize,
    /// Tracks with fewer than two usable observations.
    pub few: usize,
    /// Tracks the cheirality prune rescued: a minority of their observations
    /// was dropped and the reduced solve was admitted. Disjoint from
    /// [`Self::finite`], so the finite population is the two together and
    /// [`Self::seen`] stays the sum of every bucket.
    pub finite_pruned: usize,
    /// Observations the cheirality prune dropped, summed over those tracks.
    pub pruned_obs: usize,
    /// Median widest-pair angle of the finite tracks, in degrees; `None` where
    /// nothing came back finite or the floor was off.
    pub triangulation_angle_median_deg: Option<f64>,
}

/// What the operation decided, one entry per track in the caller's own order.
#[derive(Debug, Clone, PartialEq)]
pub struct PointEstimates {
    /// `(x, y, z, w)` per track. `w = 1` is a position, `w = 0` a unit bearing,
    /// and every component is `NaN` for an absent track.
    pub xyzw: Vec<[f64; 4]>,
    /// One verdict per track.
    pub verdicts: Vec<PointVerdict>,
    /// Whether the solved point lay in front of every observing camera the
    /// estimate was kept on. False for a track that was never solved; for a
    /// pruned track it reads the survivors.
    pub in_front: Vec<bool>,
    /// One flag per observation the caller gave, true where the cheirality
    /// prune dropped that observation from its track. All false with
    /// [`PointRules::prune_behind`] off. An observation whose ray was not
    /// usable is not flagged: it was dropped before any rule was read.
    pub pruned: Vec<bool>,
    /// The counts behind those verdicts.
    pub census: PointCensus,
}

/// Unit world rays and matching camera centres, flattened CSR-style over tracks.
#[derive(Debug, Clone, Copy)]
pub struct RaySet<'a> {
    /// World-space rays, three components per observation.
    pub dirs: &'a [f64],
    /// Matching camera centres, three components per observation.
    pub centres: &'a [f64],
    /// `n_track + 1` CSR boundaries into those arrays.
    pub offsets: &'a [usize],
}

/// Pixels with the geometry they were seen through.
#[derive(Debug, Clone, Copy)]
pub struct ObservationSet<'a> {
    /// `n_obs * 2` observed pixels.
    pub uv: &'a [f64],
    /// `n_obs` image index per observation.
    pub obs_image: &'a [u32],
    /// `n_obs` track index per observation.
    pub obs_point: &'a [u32],
    /// `n_img * 4` world-to-camera rotations, WXYZ.
    pub quats_wxyz: &'a [f64],
    /// `n_img * 3` world-to-camera translations.
    pub translations: &'a [f64],
    /// How many tracks the result indexes.
    pub n_tracks: usize,
}

/// One track's rays and the state it came in with.
struct Track {
    /// Index into the caller's own track order.
    slot: usize,
    /// The track's usable world rays.
    dirs: Vec<Vector3<f64>>,
    /// The matching camera centres.
    centres: Vec<Point3<f64>>,
    /// The observation index of each usable ray, for the reprojection bar.
    rows: Vec<usize>,
    /// Whether the caller marked this track a direction.
    marked: bool,
}

/// Re-estimate every track of a ray set.
///
/// `marks` is the incoming direction flag per track; `None` is the rule off.
/// The reprojection bar needs pixels and a camera and is ignored in this form.
pub fn estimate_points_from_rays(
    rays: RaySet<'_>,
    marks: Option<&[bool]>,
    rules: PointRules,
) -> PointEstimates {
    assert_eq!(
        rays.dirs.len(),
        rays.centres.len(),
        "dirs and centres must have equal length"
    );
    let n_tracks = rays.offsets.len().saturating_sub(1);
    if let Some(m) = marks {
        assert_eq!(m.len(), n_tracks, "marks must have one entry per track");
    }
    let tracks: Vec<Track> = (0..n_tracks)
        .map(|t| {
            let (lo, hi) = (rays.offsets[t], rays.offsets[t + 1]);
            let mut dirs = Vec::with_capacity(hi - lo);
            let mut centres = Vec::with_capacity(hi - lo);
            let mut rows = Vec::with_capacity(hi - lo);
            for r in lo..hi {
                let d = Vector3::new(rays.dirs[3 * r], rays.dirs[3 * r + 1], rays.dirs[3 * r + 2]);
                if !d.x.is_finite() || !d.y.is_finite() || !d.z.is_finite() {
                    continue;
                }
                dirs.push(d);
                centres.push(Point3::new(
                    rays.centres[3 * r],
                    rays.centres[3 * r + 1],
                    rays.centres[3 * r + 2],
                ));
                rows.push(r);
            }
            Track {
                slot: t,
                dirs,
                centres,
                rows,
                marked: marks.is_some_and(|m| m[t]),
            }
        })
        .collect();
    decide(&tracks, n_tracks, rays.dirs.len() / 3, None, rules)
}

/// Re-estimate every track of an observation set, building the world rays
/// through `cam` and the observing image's pose.
///
/// The world ray of an observation is `R⁻¹ · pixel_to_ray(u, v)` and its camera
/// centre `-R⁻¹ t`, with `R` the image's world-to-camera rotation. An
/// observation whose ray is not finite is dropped from its track before any rule
/// is read.
pub fn estimate_points_from_observations(
    cam: &CameraIntrinsics,
    obs: ObservationSet<'_>,
    marks: Option<&[bool]>,
    rules: PointRules,
) -> PointEstimates {
    let n_obs = obs.obs_image.len();
    assert_eq!(obs.obs_point.len(), n_obs, "obs_image/obs_point mismatch");
    assert_eq!(obs.uv.len(), n_obs * 2, "uv must be n_obs * 2");
    if let Some(m) = marks {
        assert_eq!(m.len(), obs.n_tracks, "marks must have one entry per track");
    }
    let n_img = obs.quats_wxyz.len() / 4;
    assert_eq!(
        obs.translations.len(),
        n_img * 3,
        "translations must be n_img * 3"
    );
    let inv: Vec<UnitQuaternion<f64>> = (0..n_img)
        .map(|i| pose_rotation(obs, i).inverse())
        .collect();
    let centres: Vec<Point3<f64>> = (0..n_img)
        .map(|i| {
            let t = i * 3;
            Point3::from(
                -(inv[i]
                    * Vector3::new(
                        obs.translations[t],
                        obs.translations[t + 1],
                        obs.translations[t + 2],
                    )),
            )
        })
        .collect();

    // Tracks are grouped by a stable sort of the track index, so a track's rays
    // are accumulated in the order the caller listed them.
    let mut order: Vec<usize> = (0..n_obs).collect();
    order.sort_by_key(|&k| obs.obs_point[k]);

    let mut named = vec![false; obs.n_tracks];
    let mut tracks: Vec<Track> = Vec::new();
    let mut prev: Option<u32> = None;
    for &k in &order {
        let p = obs.obs_point[k];
        if prev != Some(p) {
            named[p as usize] = true;
            tracks.push(Track {
                slot: p as usize,
                dirs: Vec::new(),
                centres: Vec::new(),
                rows: Vec::new(),
                marked: marks.is_some_and(|m| m[p as usize]),
            });
            prev = Some(p);
        }
        let i = obs.obs_image[k] as usize;
        let d = cam.pixel_to_ray(obs.uv[2 * k], obs.uv[2 * k + 1]);
        let world = inv[i] * Vector3::new(d[0], d[1], d[2]);
        if !world.x.is_finite() || !world.y.is_finite() || !world.z.is_finite() {
            continue;
        }
        let last = tracks.last_mut().expect("a group was opened");
        last.dirs.push(world);
        last.centres.push(centres[i]);
        last.rows.push(k);
    }
    // A track no observation names has no usable ray, so it is a `few` track.
    for (slot, seen) in named.iter().enumerate() {
        if !seen {
            tracks.push(Track {
                slot,
                dirs: Vec::new(),
                centres: Vec::new(),
                rows: Vec::new(),
                marked: marks.is_some_and(|m| m[slot]),
            });
        }
    }
    decide(&tracks, obs.n_tracks, n_obs, Some((cam, obs)), rules)
}

/// A rule that decides a track without solving it.
#[derive(Clone, Copy)]
enum Early {
    /// Fewer than two usable rays.
    Few,
    /// The caller had already marked it a direction.
    Marked,
    /// Its widest ray pair is inside the floor.
    Thin,
}

/// What the solve, and the rules read after it, made of one open track.
struct Solved {
    /// The rule that decided the track.
    verdict: PointVerdict,
    /// The estimate: `w = 1` a position, `w = 0` a bearing.
    value: [f64; 4],
    /// Whether every observation the estimate was kept on sees it in front.
    front: bool,
    /// Observation rows the cheirality prune dropped, in the caller's own
    /// observation indexing and in increasing order.
    pruned: Vec<usize>,
    /// The widest-pair cosine of the surviving rays, where a prune re-read the
    /// floor over them. `None` leaves the full track's own cosine standing.
    cos_widest: Option<f64>,
}

/// The shared decision pass over prepared tracks.
///
/// `n_obs` is how many observations the caller handed in, which is the length
/// of the per-observation prune mask.
fn decide(
    tracks: &[Track],
    n_tracks: usize,
    n_obs: usize,
    reproject: Option<(&CameraIntrinsics, ObservationSet<'_>)>,
    rules: PointRules,
) -> PointEstimates {
    let mut xyzw = vec![[f64::NAN; 4]; n_tracks];
    let mut verdicts = vec![PointVerdict::Few; n_tracks];
    let mut in_front = vec![false; n_tracks];
    let mut pruned = vec![false; n_obs];

    // The widest pair is read once, here, and only where the floor asks for it:
    // it costs O(K²) in the track's observation count, and a caller with the
    // floor off has not asked for that pass.
    let cos_floor = rules.floor_rad.map(f64::cos);
    let early: Vec<(Option<Early>, Option<f64>)> = tracks
        .par_iter()
        .map(|t| {
            if t.dirs.len() < 2 {
                return (Some(Early::Few), None);
            }
            if t.marked {
                return (Some(Early::Marked), None);
            }
            match cos_floor {
                None => (None, None),
                Some(c) => {
                    let m = smallest_pairwise_cosine(&t.dirs);
                    (if m > c { Some(Early::Thin) } else { None }, Some(m))
                }
            }
        })
        .collect();

    let open: Vec<usize> = (0..tracks.len())
        .filter(|&k| early[k].0.is_none())
        .collect();
    let mut dirs = Vec::new();
    let mut centres = Vec::new();
    let mut offsets = Vec::with_capacity(open.len() + 1);
    for &k in &open {
        offsets.push(dirs.len());
        dirs.extend_from_slice(&tracks[k].dirs);
        centres.extend_from_slice(&tracks[k].centres);
    }
    offsets.push(dirs.len());
    let tris = triangulate_batch(&dirs, &centres, &offsets);

    let solved: Vec<Solved> = open
        .par_iter()
        .zip(tris.par_iter())
        .map(|(&k, tri)| {
            let t = &tracks[k];
            let p = tri.point.coords;
            let front = tri.in_front_of_all_cameras;
            if rules.cheirality && !front {
                if rules.prune_behind {
                    if let Some(s) = prune_behind(t, p, reproject, rules) {
                        return s;
                    }
                }
                return refused(PointVerdict::Behind, &t.dirs, false);
            }
            if let Some(bar) = rules.bar_px {
                if let Some((cam, obs)) = reproject {
                    if !clears_bar(cam, obs, &t.rows, &t.centres, p, bar) {
                        return refused(PointVerdict::OverBar, &t.dirs, front);
                    }
                }
            }
            Solved {
                verdict: PointVerdict::Finite,
                value: [p.x, p.y, p.z, 1.0],
                front,
                pruned: Vec::new(),
                cos_widest: None,
            }
        })
        .collect();

    let mut census = PointCensus {
        seen: n_tracks,
        ..Default::default()
    };
    for (k, t) in tracks.iter().enumerate() {
        match early[k].0 {
            Some(Early::Few) => {
                census.few += 1;
                verdicts[t.slot] = PointVerdict::Few;
                xyzw[t.slot] = match rules.few {
                    FewObservations::Absent => [f64::NAN; 4],
                    FewObservations::Bearing => match t.dirs.first() {
                        Some(d) => unit([d.x, d.y, d.z]),
                        None => unit(FALLBACK_DIRECTION),
                    },
                };
            }
            Some(Early::Marked) => {
                census.marked += 1;
                verdicts[t.slot] = PointVerdict::Marked;
                xyzw[t.slot] = bearing(&t.dirs);
            }
            Some(Early::Thin) => {
                census.thin += 1;
                verdicts[t.slot] = PointVerdict::Thin;
                xyzw[t.slot] = bearing(&t.dirs);
            }
            None => {}
        }
    }
    let mut angles: Vec<f64> = Vec::new();
    for (&k, s) in open.iter().zip(&solved) {
        let slot = tracks[k].slot;
        verdicts[slot] = s.verdict;
        xyzw[slot] = s.value;
        in_front[slot] = s.front;
        match s.verdict {
            PointVerdict::Finite => {
                census.finite += 1;
                if let Some(m) = early[k].1 {
                    angles.push(m.clamp(-1.0, 1.0).acos().to_degrees());
                }
            }
            PointVerdict::FinitePruned => {
                census.finite_pruned += 1;
                census.pruned_obs += s.pruned.len();
                for &r in &s.pruned {
                    pruned[r] = true;
                }
                // The surviving rays are what the estimate rests on, so the
                // angle it reports is theirs.
                if let Some(m) = s.cos_widest.or(early[k].1) {
                    angles.push(m.clamp(-1.0, 1.0).acos().to_degrees());
                }
            }
            PointVerdict::Behind => census.behind += 1,
            PointVerdict::OverBar => census.over_bar += 1,
            _ => unreachable!("a solved track carries a solved verdict"),
        }
    }
    census.triangulation_angle_median_deg = if angles.is_empty() {
        None
    } else {
        Some(median_in_place(&mut angles))
    };

    PointEstimates {
        xyzw,
        verdicts,
        in_front,
        pruned,
        census,
    }
}

/// A track a rule refused: the mean of its own rays, nothing pruned.
fn refused(verdict: PointVerdict, dirs: &[Vector3<f64>], front: bool) -> Solved {
    Solved {
        verdict,
        value: bearing(dirs),
        front,
        pruned: Vec::new(),
        cos_widest: None,
    }
}

/// Read the cheirality failure of `p` per observation and, where the failing
/// observations are a strict minority, solve again without them.
///
/// The failing set is the observations whose depth along their own ray is
/// non-positive, which is the same test the batch solve reports as the track's
/// in-front flag. A minority of them is a track whose observations still name a
/// majority; at a tie there is no majority to solve on, so a tie is not a
/// minority and the track keeps the whole-track verdict. A minority of a track
/// of two is impossible, so a rescue never leaves fewer than two rays.
///
/// The reduced track is then re-read by the rules the full one would have been:
/// the floor over the surviving pair angles, cheirality again over the new
/// solve, and the bar over the surviving observations. `None` where any of them
/// refuses, which leaves the track the bearing it would have been anyway, with
/// nothing pruned.
fn prune_behind(
    track: &Track,
    p: Vector3<f64>,
    reproject: Option<(&CameraIntrinsics, ObservationSet<'_>)>,
    rules: PointRules,
) -> Option<Solved> {
    let n = track.dirs.len();
    let behind: Vec<usize> = (0..n)
        .filter(|&i| (p - track.centres[i].coords).dot(&track.dirs[i]) <= 0.0)
        .collect();
    if behind.is_empty() || 2 * behind.len() >= n {
        return None;
    }
    let mut drop = vec![false; n];
    for &i in &behind {
        drop[i] = true;
    }
    let keep: Vec<usize> = (0..n).filter(|&i| !drop[i]).collect();
    let dirs: Vec<Vector3<f64>> = keep.iter().map(|&i| track.dirs[i]).collect();
    let centres: Vec<Point3<f64>> = keep.iter().map(|&i| track.centres[i]).collect();
    let rows: Vec<usize> = keep.iter().map(|&i| track.rows[i]).collect();

    let mut cos_widest = None;
    if let Some(c) = rules.floor_rad.map(f64::cos) {
        let m = smallest_pairwise_cosine(&dirs);
        if m > c {
            return None;
        }
        cos_widest = Some(m);
    }
    let tri = triangulate_batch(&dirs, &centres, &[0, dirs.len()]);
    let q = tri[0].point.coords;
    if !tri[0].in_front_of_all_cameras {
        return None;
    }
    if let Some(bar) = rules.bar_px {
        if let Some((cam, obs)) = reproject {
            if !clears_bar(cam, obs, &rows, &centres, q, bar) {
                return None;
            }
        }
    }
    Some(Solved {
        verdict: PointVerdict::FinitePruned,
        value: [q.x, q.y, q.z, 1.0],
        front: true,
        pruned: behind.iter().map(|&i| track.rows[i]).collect(),
        cos_widest,
    })
}

/// One image's world-to-camera rotation, taken as given.
///
/// The components are NOT renormalized: a caller holding a unit quaternion gets
/// its own rotation back bit for bit, which is what lets an optimizer route its
/// own re-estimation through this operation without moving its poses. The
/// bindings check the norm before handing anything over.
fn pose_rotation(obs: ObservationSet<'_>, image: usize) -> UnitQuaternion<f64> {
    let o = image * 4;
    UnitQuaternion::new_unchecked(Quaternion::new(
        obs.quats_wxyz[o],
        obs.quats_wxyz[o + 1],
        obs.quats_wxyz[o + 2],
        obs.quats_wxyz[o + 3],
    ))
}

/// The normalized mean of a track's rays as an `xyzw` bearing, or the fallback
/// direction where it has no ray.
fn bearing(dirs: &[Vector3<f64>]) -> [f64; 4] {
    if dirs.is_empty() {
        return unit(FALLBACK_DIRECTION);
    }
    let mut s = Vector3::<f64>::zeros();
    for d in dirs {
        s += d;
    }
    let n = dirs.len() as f64;
    unit([s.x / n, s.y / n, s.z / n])
}

/// `v` scaled to unit length, as an `xyzw` with `w = 0`.
fn unit(v: [f64; 3]) -> [f64; 4] {
    let n = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if n > 0.0 && n.is_finite() {
        [v[0] / n, v[1] / n, v[2] / n, 0.0]
    } else {
        [
            FALLBACK_DIRECTION[0],
            FALLBACK_DIRECTION[1],
            FALLBACK_DIRECTION[2],
            0.0,
        ]
    }
}

/// The smallest cosine over every pair of a track's rays, the diagonal
/// included: a pairwise statistic, read from the rays alone.
pub(crate) fn smallest_pairwise_cosine(dirs: &[Vector3<f64>]) -> f64 {
    let mut m = f64::INFINITY;
    for (a, x) in dirs.iter().enumerate() {
        for y in &dirs[a..] {
            let c = x.dot(y);
            if c < m {
                m = c;
            }
        }
    }
    m
}

/// Whether the median finite reprojection residual over `rows` sits inside the
/// bar.
///
/// `rows` are observation indices and `centres` the matching camera centres, so
/// the caller states which of a track's observations vote: the whole track, or
/// the survivors of a cheirality prune. Observations the camera model refuses to
/// project carry no residual and do not vote; a set where none of them projects
/// fails the bar.
fn clears_bar(
    cam: &CameraIntrinsics,
    obs: ObservationSet<'_>,
    rows: &[usize],
    centres: &[Point3<f64>],
    p: Vector3<f64>,
    bar: f64,
) -> bool {
    let mut res: Vec<f64> = Vec::with_capacity(rows.len());
    for (n, &k) in rows.iter().enumerate() {
        let i = obs.obs_image[k] as usize;
        let d = p - centres[n].coords;
        let xc = pose_rotation(obs, i) * d;
        if let Some((u, v)) = cam.ray_to_pixel([xc.x, xc.y, xc.z]) {
            let du = obs.uv[2 * k] - u;
            let dv = obs.uv[2 * k + 1] - v;
            let r = (du * du + dv * dv).sqrt();
            if r.is_finite() {
                res.push(r);
            }
        }
    }
    !res.is_empty() && median_in_place(&mut res) <= bar
}

#[cfg(test)]
mod tests;
