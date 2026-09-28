// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The far-field sweep: the distances in the far field a pixel's patch reads
//! at, each with its range and what it rests on.
//!
//! `specs/core/bench/far-field-sweep.md` is the design. The sweep reads the
//! patch from infinity in to a small disparity in every image it lands in,
//! keeps each peak of that reading, groups each peak's images by how their
//! middles agree with one another, and refits the peak on the largest other
//! group when the queried image stands apart from the rest.

use nalgebra::Vector3;

use crate::bench::fit::{fit, FitOptions};
use crate::bench::steps::{set_verdict, ClusterSeed};
use crate::bench::track::{EditableTrack, Thresholds, Verdict};
use crate::bench::track_at_pixel::{seed_cluster_with, upgrade_sightings, ViewCamera};
use crate::patch::normal_refine::ProjectedImage;
use crate::progress::Progress;
use crate::reconstruction::edited::EditedReconstruction;

use super::candidate::ray_angle;
use super::grey::GreyImages;
use super::patch_read::{read_patch_along_ray, PatchRead, RayPatch};
use super::range::camera_spread;

/// Which images the width of an image is judged against.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WideAmong {
    /// The widest of the images whose whole patch reads
    /// [`FarFieldOptions::min_whole`] somewhere in the sweep, so an image that
    /// sees something else at the pixel does not set the scale.
    Matching,
    /// The widest of every image the pixel lands in.
    All,
}

impl WideAmong {
    /// The name the Python binding and the harness spell it with.
    pub fn name(self) -> &'static str {
        match self {
            Self::Matching => "matching",
            Self::All => "all",
        }
    }
}

impl std::str::FromStr for WideAmong {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "matching" => Ok(Self::Matching),
            "all" => Ok(Self::All),
            other => Err(format!(
                "unknown wide_among {other:?} (expected matching|all)"
            )),
        }
    }
}

/// What the far-field sweep runs with. The defaults are the harness's.
#[derive(Debug, Clone, PartialEq)]
pub struct FarFieldOptions {
    /// The disparities read, in px of the image that moves the pixel most for
    /// a change of inverse distance, ascending from `0` (infinity).
    pub disparities: Vec<f64>,
    /// The patch's half-width, in px of the queried image.
    pub radius_px: f64,
    /// An image is wide when it moves the pixel at least this share as far as
    /// the widest image ([`Self::wide_among`] says of which).
    pub wide: f64,
    /// Which images the widest is taken among.
    pub wide_among: WideAmong,
    /// A peak's whole patch must read at least this, and an image agrees with
    /// a peak when its whole patch reads at least this there.
    pub min_whole: f64,
    /// A peak's middle must read at least this, unless the middle is flat.
    pub min_middle: f64,
    /// The queried patch's middle is flat, and the peaks are read on the
    /// whole patch, when its grey standard deviation is under this.
    pub middle_min_std: f64,
    /// At most this many peaks are kept, the highest.
    pub max_peaks: usize,
    /// A peak must stand at least this far above the reading around it.
    pub min_prominence: f64,
    /// Group each reading's images pairwise and refit it when the queried
    /// image stands apart.
    pub refit: bool,
    /// The average-linkage cut the images are grouped at.
    pub group_cut: f64,
    /// The most images grouped: the queried one and the best-reading others.
    pub group_max: usize,
    /// A refit that lands within this of the pixel leaves the reading where
    /// it is, in px.
    pub refit_px: f64,
    /// A refit that lands further than this from the pixel is rejected, in px.
    pub refit_max_px: f64,
    /// A refit with any sighting further than this from the refitted point is
    /// rejected, in px.
    pub refit_max_err_px: f64,
}

impl Default for FarFieldOptions {
    fn default() -> Self {
        Self {
            disparities: vec![0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0],
            radius_px: 8.0,
            wide: 0.5,
            wide_among: WideAmong::Matching,
            min_whole: 0.8,
            min_middle: 0.7,
            middle_min_std: 8.0,
            max_peaks: 3,
            min_prominence: 0.02,
            refit: true,
            group_cut: 0.9,
            group_max: 16,
            refit_px: 8.0,
            refit_max_px: 48.0,
            refit_max_err_px: 2.0,
        }
    }
}

/// What the grouping made of a reading's images and what the refit did.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Refit {
    /// Every image grouped with the queried one.
    Agrees,
    /// Some images grouped with the queried one; the reading keeps those.
    Grouped,
    /// The queried image stood alone and no two other images grouped either,
    /// so the sweep's own comparison stands.
    Unsplit,
    /// The refit on the largest other group landed within
    /// [`FarFieldOptions::refit_px`] of the pixel; the reading keeps that
    /// group.
    Stands,
    /// The refit landed further away, and the reading moved to that point.
    Moved,
    /// The refit landed too far away, or fitted its sightings too loosely; the
    /// reading is dropped.
    MovedRejected,
    /// The refit landed off the queried image; the reading is dropped.
    OffTheImage,
    /// The refit's track could not be built or fitted; the reading is dropped.
    Failed,
}

impl Refit {
    /// The outcome's name, as the harness spells it.
    pub fn name(self) -> &'static str {
        match self {
            Self::Agrees => "agrees",
            Self::Grouped => "grouped",
            Self::Unsplit => "unsplit",
            Self::Stands => "stands",
            Self::Moved => "moved",
            Self::MovedRejected => "moved, rejected",
            Self::OffTheImage => "off the image",
            Self::Failed => "failed",
        }
    }

    /// Whether a reading with this outcome is kept.
    pub fn keeps(self) -> bool {
        !matches!(self, Self::MovedRejected | Self::OffTheImage | Self::Failed)
    }
}

impl std::fmt::Display for Refit {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}

/// What a far-field reading rests on, for a comparison between candidates to
/// weigh.
#[derive(Debug, Clone, PartialEq)]
pub struct FarFieldMetrics {
    /// The whole patch's reading at the peak: the mean of the three best wide
    /// images.
    pub whole: f64,
    /// The middle's reading at the peak, the same way; `None` where no wide
    /// image reads there.
    pub middle: Option<f64>,
    /// How far the peak stands above the lowest reading between it and the
    /// nearest higher one, or above the lowest reading of all for the highest.
    pub prominence: f64,
    /// The peak's place among the kept peaks, 1 the highest.
    pub peak_rank: usize,
    /// How many peaks were kept.
    pub peaks: usize,
    /// Whether the queried middle was flat, so the peaks were read on the
    /// whole patch.
    pub middle_flat: bool,
    /// The queried middle's grey standard deviation.
    pub middle_std: f64,
    /// How many wide images read [`FarFieldOptions::min_whole`] at the peak,
    /// capped at one fewer than [`FarFieldOptions::group_max`].
    pub images: usize,
    /// The most any of those images moves the pixel per unit of inverse
    /// distance, in px.
    pub parallax_px: f64,
    /// The same for the widest image the pixel lands in, in px: the scale the
    /// disparities are counted in.
    pub widest_px: f64,
    /// The whole patch's reading at every disparity of the sweep, `None` where
    /// no wide image reads.
    pub profile_whole: Vec<Option<f64>>,
    /// The middle's reading at every disparity.
    pub profile_middle: Vec<Option<f64>>,
}

/// How a reading's images grouped by the ZNCC of their middles with each
/// other.
#[derive(Debug, Clone, PartialEq)]
pub struct FarFieldGrouping {
    /// The groups, each its images ascending, the queried image among them.
    pub groups: Vec<Vec<u32>>,
    /// Each other image's pairwise ZNCC with the queried one, in the order of
    /// the reading's views before the grouping.
    pub query_middle: Vec<f64>,
    /// The mean of those over the queried image's own group, `None` when it
    /// stands alone.
    pub group_middle: Option<f64>,
    /// How many images stand outside the queried image's group.
    pub left_out: usize,
}

/// One peak of the far-field sweep: a distance along the pixel's ray that its
/// patch reads at, and the images that read it there.
#[derive(Debug, Clone, PartialEq)]
pub struct FarFieldReading {
    /// The sweep's disparity at the peak, in px of the widest image; `0` is
    /// infinity.
    pub disparity: f64,
    /// The point: a place, or a unit direction when [`Self::at_infinity`].
    pub position: Vector3<f64>,
    /// Whether the reading is a bearing.
    pub at_infinity: bool,
    /// The sightings, `(image, pixel)`, the queried image's first.
    pub views: Vec<(u32, [f64; 2])>,
    /// Where the reading sits in the queried image: the pixel asked about, or
    /// where a moved reading's point lands.
    pub query_pixel: [f64; 2],
    /// How far [`Self::query_pixel`] is from the pixel asked about, in px.
    pub distance_px: f64,
    /// The largest reprojection error of the sightings, in px: zero for a
    /// sweep reading, which places them by construction.
    pub max_reproj_px: f64,
    /// The widest angle between two sightings' rays to the point, in degrees;
    /// zero for a bearing.
    pub max_ray_angle_deg: f64,
    /// The point's depth along the queried camera's axis; infinite for a
    /// bearing.
    pub depth: f64,
    /// The distances along the pixel's ray the reading allows, from the
    /// midpoints to the neighbouring disparities; `None` for a moved reading,
    /// whose range is its sightings'.
    pub range: Option<[f64; 2]>,
    /// What the reading rests on.
    pub metrics: FarFieldMetrics,
    /// How its images grouped, when the refit ran.
    pub grouping: Option<FarFieldGrouping>,
    /// What the grouping and the refit made of it, when the refit ran.
    pub refit: Option<Refit>,
    /// How far the refit landed from the pixel, in px, when it got that far.
    pub refit_px: Option<f64>,
}

/// What the far-field sweep found.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct FarFieldSweep {
    /// The kept readings, highest peak first.
    pub readings: Vec<FarFieldReading>,
    /// The peaks whose refit dropped them, each as it stood before the refit
    /// with the refit's outcome.
    pub dropped: Vec<FarFieldReading>,
}

/// Why the far-field sweep could not run.
#[derive(Debug, Clone, PartialEq)]
pub enum FarFieldError {
    /// The queried image is not one of the reconstruction's.
    NoSuchImage {
        /// The image asked about.
        image: u32,
        /// How many images the reconstruction holds.
        image_count: usize,
    },
    /// The pixel is not a place on the photograph.
    PixelOffImage {
        /// The pixel asked about.
        pixel: [f64; 2],
        /// The photograph's width, in px.
        width: u32,
        /// The photograph's height, in px.
        height: u32,
    },
    /// An input does not have one entry per image of the reconstruction.
    InputMismatch {
        /// Which input.
        input: &'static str,
        /// How many entries it has.
        got: usize,
        /// How many images the reconstruction holds.
        image_count: usize,
    },
    /// The progress handle was cancelled.
    Cancelled,
}

impl std::fmt::Display for FarFieldError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoSuchImage { image, image_count } => write!(
                f,
                "image {image} is not one of the reconstruction's {image_count} images"
            ),
            Self::PixelOffImage {
                pixel,
                width,
                height,
            } => write!(
                f,
                "the pixel ({:.1}, {:.1}) is not on the {width}x{height} photograph",
                pixel[0], pixel[1]
            ),
            Self::InputMismatch {
                input,
                got,
                image_count,
            } => write!(
                f,
                "{input} has {got} entries, but the reconstruction has {image_count} images"
            ),
            Self::Cancelled => write!(f, "the far-field sweep was cancelled"),
        }
    }
}

impl std::error::Error for FarFieldError {}

/// Read `pixel`'s patch in `image` from infinity in, in every image it lands
/// in, and return one reading per peak of that reading.
///
/// Disparities are counted in the image that moves the pixel most for a
/// change of inverse distance, among those it lands in at infinity: a
/// disparity `d` is the distance `R / d` along the pixel's ray, with `R` that
/// image's pixels per unit of inverse distance, and `d = 0` is infinity. The
/// patch is read ([`read_patch_along_ray`]) at every one of
/// [`FarFieldOptions::disparities`], and each peak of the reading over the
/// wide images is a reading. With [`FarFieldOptions::refit`], each reading's
/// images are grouped by the ZNCC of their middles with one another, and a
/// reading whose queried image stands alone is refitted on the largest other
/// group: it keeps its place, moves to where the fit lands, or is dropped.
///
/// `views` and `grey` hold one entry per image of `edited`, in its order;
/// `edited` is read only by the refit's fit. Nothing is committed.
pub fn far_field_sweep(
    edited: &EditedReconstruction,
    views: &[ProjectedImage<'_>],
    grey: &GreyImages,
    image: u32,
    pixel: [f64; 2],
    options: &FarFieldOptions,
    progress: &Progress<'_>,
) -> Result<FarFieldSweep, FarFieldError> {
    let image_count = edited.image_count();
    for (input, got) in [("views", views.len()), ("grey", grey.len())] {
        if got != image_count {
            return Err(FarFieldError::InputMismatch {
                input,
                got,
                image_count,
            });
        }
    }
    let Some(view) = views.get(image as usize) else {
        return Err(FarFieldError::NoSuchImage { image, image_count });
    };
    let (width, height) = (view.camera.width, view.camera.height);
    let on_photo = pixel.iter().all(|c| c.is_finite())
        && pixel[0] >= 0.0
        && pixel[1] >= 0.0
        && pixel[0] < f64::from(width)
        && pixel[1] < f64::from(height);
    if !on_photo {
        return Err(FarFieldError::PixelOffImage {
            pixel,
            width,
            height,
        });
    }
    let cancelled = |_| FarFieldError::Cancelled;
    progress.check_cancel().map_err(cancelled)?;
    let _phase = progress.phase("far-field sweep");

    let cameras: Vec<ViewCamera<'_>> = views.iter().map(ViewCamera::new).collect();
    let patch = RayPatch {
        image,
        pixel,
        radius_px: options.radius_px,
    };
    let others: Vec<u32> = (0..image_count as u32).filter(|&i| i != image).collect();
    let Some(at_infinity) =
        read_patch_along_ray(views, grey, &patch, &[f64::INFINITY], &others, false)
    else {
        return Ok(FarFieldSweep::default());
    };
    let cq = &cameras[image as usize];
    let ray = cq.ray(pixel).normalize();
    // Pixels of shift per unit of inverse distance: parallax is linear in
    // inverse distance, so the shift between infinity and a point far out,
    // times that point's distance, is the rate.
    let probe = 1e3 * camera_spread(views);
    let far_point = cq.center + ray * probe;
    let mut rates: Vec<(u32, f64)> = Vec::new();
    for (v, &j) in others.iter().enumerate() {
        if at_infinity.whole[[0, v]] <= -1.0 {
            continue;
        }
        let oc = &cameras[j as usize];
        if let (Some(a), Some(b)) = (
            oc.project_direction(&ray),
            oc.project_homogeneous(&far_point, 1.0),
        ) {
            rates.push((j, (b[0] - a[0]).hypot(b[1] - a[1]) * probe));
        }
    }
    let widest = rates
        .iter()
        .map(|&(_, r)| r)
        .fold(f64::NEG_INFINITY, f64::max);
    if rates.is_empty() || widest <= 0.0 {
        return Ok(FarFieldSweep::default());
    }
    let disp = &options.disparities;
    let distances: Vec<f64> = disp
        .iter()
        .map(|&d| if d == 0.0 { f64::INFINITY } else { widest / d })
        .collect();
    let images: Vec<u32> = rates.iter().map(|&(j, _)| j).collect();
    let rate: Vec<f64> = rates.iter().map(|&(_, r)| r).collect();
    progress.check_cancel().map_err(cancelled)?;
    let Some(read) = read_patch_along_ray(views, grey, &patch, &distances, &images, true) else {
        return Ok(FarFieldSweep::default());
    };

    let nv = images.len();
    let nd = disp.len();
    let matching: Vec<bool> = (0..nv)
        .map(|v| (0..nd).any(|k| read.whole[[k, v]] >= options.min_whole))
        .collect();
    if !matching.iter().any(|&m| m) {
        return Ok(FarFieldSweep::default());
    }
    let scale = match options.wide_among {
        WideAmong::Matching => (0..nv)
            .filter(|&v| matching[v])
            .map(|v| rate[v])
            .fold(f64::NEG_INFINITY, f64::max),
        WideAmong::All => widest,
    };
    let wide: Vec<bool> = (0..nv)
        .map(|v| matching[v] && rate[v] >= options.wide * scale)
        .collect();
    let bw: Vec<f64> = (0..nd)
        .map(|k| best3((0..nv).filter(|&v| wide[v]).map(|v| read.whole[[k, v]])))
        .collect();
    let bm: Vec<f64> = (0..nd)
        .map(|k| best3((0..nv).filter(|&v| wide[v]).map(|v| read.middle[[k, v]])))
        .collect();
    let flat = read.middle_std < options.middle_min_std;
    let key: Vec<f64> = (if flat { &bw } else { &bm })
        .iter()
        .map(|&x| if x.is_finite() { x } else { f64::NEG_INFINITY })
        .collect();
    let mut peaks: Vec<usize> = (0..nd.saturating_sub(1))
        .filter(|&k| {
            key[k].is_finite()
                && key[k] >= if k > 0 { key[k - 1] } else { f64::NEG_INFINITY }
                && key[k] > key[k + 1]
                && bw[k] >= options.min_whole
                && (flat || bm[k] >= options.min_middle)
        })
        .filter(|&k| prominence(&key, k) >= options.min_prominence)
        .collect();
    peaks.sort_by(|&a, &b| key[b].total_cmp(&key[a]));
    peaks.truncate(options.max_peaks);

    let listed =
        |x: &[f64]| -> Vec<Option<f64>> { x.iter().map(|&v| v.is_finite().then_some(v)).collect() };
    let mut sweep = FarFieldSweep::default();
    for (order, &k) in peaks.iter().enumerate() {
        progress.check_cancel().map_err(cancelled)?;
        let d = disp[k];
        let near = widest / (0.5 * (d + disp[k + 1]));
        let far = if k == 0 {
            f64::INFINITY
        } else {
            widest / (0.5 * (d + disp[k - 1]))
        };
        let mut agree: Vec<usize> = (0..nv)
            .filter(|&v| wide[v] && read.whole[[k, v]] >= options.min_whole)
            .collect();
        agree.sort_by(|&a, &b| read.whole[[k, b]].total_cmp(&read.whole[[k, a]]));
        agree.truncate(options.group_max.saturating_sub(1));
        let mut sight = vec![(image, pixel)];
        sight.extend(agree.iter().map(|&v| {
            (
                images[v],
                [read.centres[[k, v, 0]], read.centres[[k, v, 1]]],
            )
        }));
        let (position, at_inf, depth, angle) = if d == 0.0 {
            (ray, true, f64::INFINITY, 0.0)
        } else {
            let x = cq.center + ray * (widest / d);
            let angle = ray_angle(&cameras, &x, sight.iter().map(|s| s.0));
            (x, false, cq.depth(&x), angle)
        };
        let metrics = FarFieldMetrics {
            whole: bw[k],
            middle: bm[k].is_finite().then_some(bm[k]),
            prominence: prominence(&key, k),
            peak_rank: order + 1,
            peaks: peaks.len(),
            middle_flat: flat,
            middle_std: read.middle_std,
            images: agree.len(),
            parallax_px: agree.iter().map(|&v| rate[v]).fold(0.0, f64::max),
            widest_px: widest,
            profile_whole: listed(&bw),
            profile_middle: listed(&bm),
        };
        let reading = FarFieldReading {
            disparity: d,
            position,
            at_infinity: at_inf,
            views: sight,
            query_pixel: pixel,
            distance_px: 0.0,
            max_reproj_px: 0.0,
            max_ray_angle_deg: angle,
            depth,
            range: Some([near, far]),
            metrics,
            grouping: None,
            refit: None,
            refit_px: None,
        };
        if !options.refit {
            sweep.readings.push(reading);
            continue;
        }
        let refit = Refitter {
            edited,
            views,
            cameras: &cameras,
            image,
            pixel,
            options,
        };
        let (reading, kept) = refit.group(reading, &read, k, &agree);
        if kept {
            sweep.readings.push(reading);
        } else {
            sweep.dropped.push(reading);
        }
    }
    Ok(sweep)
}

/// The mean of the three highest of `values` that were read (above `-1`), or
/// `NaN` when none was.
pub(super) fn best3(values: impl Iterator<Item = f64>) -> f64 {
    let mut read: Vec<f64> = values.filter(|&v| v > -1.0).collect();
    if read.is_empty() {
        return f64::NAN;
    }
    read.sort_by(|a, b| b.total_cmp(a));
    read.truncate(3);
    read.iter().sum::<f64>() / read.len() as f64
}

/// How far `key[k]` stands above the lowest value between it and the nearest
/// higher one on either side, the higher of the two sides' lows when both have
/// one, or above the lowest finite value of all when neither does.
pub(super) fn prominence(key: &[f64], k: usize) -> f64 {
    let mut cols = Vec::new();
    for step in [-1isize, 1] {
        let mut low = key[k];
        let mut i = k as isize + step;
        while i >= 0 && (i as usize) < key.len() && key[i as usize] <= key[k] {
            low = low.min(key[i as usize]);
            i += step;
        }
        if i >= 0 && (i as usize) < key.len() {
            cols.push(low);
        }
    }
    let base = if cols.is_empty() {
        key.iter()
            .copied()
            .filter(|v| v.is_finite())
            .fold(f64::INFINITY, f64::min)
    } else {
        cols.into_iter().fold(f64::NEG_INFINITY, f64::max)
    };
    key[k] - base
}

/// The groups of indexes into the square `similar` (`n x n`, row-major),
/// merged while the two closest groups' mean similarity is at least `cut`:
/// average linkage.
///
/// Each merge joins the first pair, in the order the groups are listed, whose
/// mean is the highest; the later group is appended to the earlier one and
/// taken out of the list.
pub(super) fn average_linkage(similar: &[f64], n: usize, cut: f64) -> Vec<Vec<usize>> {
    let mut groups: Vec<Vec<usize>> = (0..n).map(|i| vec![i]).collect();
    while groups.len() > 1 {
        let mut best = f64::NEG_INFINITY;
        let mut pair = None;
        for x in 0..groups.len() {
            for y in x + 1..groups.len() {
                let mut sum = 0.0;
                for &a in &groups[x] {
                    for &b in &groups[y] {
                        sum += similar[a * n + b];
                    }
                }
                let m = sum / (groups[x].len() * groups[y].len()) as f64;
                if m > best {
                    best = m;
                    pair = Some((x, y));
                }
            }
        }
        let Some((x, y)) = pair.filter(|_| best >= cut) else {
            break;
        };
        let taken = groups.remove(y);
        groups[x].extend(taken);
    }
    groups
}

/// What the grouping and the refit of one reading read.
struct Refitter<'s, 'a> {
    edited: &'s EditedReconstruction,
    views: &'s [ProjectedImage<'a>],
    cameras: &'s [ViewCamera<'a>],
    image: u32,
    pixel: [f64; 2],
    options: &'s FarFieldOptions,
}

impl Refitter<'_, '_> {
    /// Group `reading`'s images by the ZNCC of their middles with each other
    /// (of the whole patches when the queried middle is flat), keep the
    /// queried image's group, and refit on the largest other group when the
    /// queried image stands alone. `agree` indexes the read's images in the
    /// order of `reading.views` after the query. Returns the reading and
    /// whether it is kept.
    fn group(
        &self,
        mut reading: FarFieldReading,
        read: &PatchRead,
        k: usize,
        agree: &[usize],
    ) -> (FarFieldReading, bool) {
        let samples = read
            .samples
            .as_ref()
            .expect("the sweep's read keeps its samples");
        let flat = reading.metrics.middle_flat;
        let pick = |values: &mut dyn Iterator<Item = f64>| -> Vec<f64> {
            values
                .zip(&samples.middle)
                .filter(|(_, &m)| flat || m)
                .map(|(v, _)| v)
                .collect()
        };
        let mut rows = vec![pick(&mut samples.template.iter().map(|&v| f64::from(v)))];
        for &v in agree {
            rows.push(pick(
                &mut samples
                    .values
                    .slice(ndarray::s![k, v, ..])
                    .iter()
                    .map(|&x| f64::from(x)),
            ));
        }
        for row in &mut rows {
            let mean = row.iter().sum::<f64>() / row.len() as f64;
            row.iter_mut().for_each(|x| *x -= mean);
            let norm = row.iter().map(|x| x * x).sum::<f64>().sqrt();
            row.iter_mut().for_each(|x| *x /= norm);
        }
        let n = rows.len();
        let mut similar = vec![0.0; n * n];
        for a in 0..n {
            for b in 0..n {
                similar[a * n + b] = rows[a].iter().zip(&rows[b]).map(|(x, y)| x * y).sum();
            }
        }
        let groups = average_linkage(&similar, n, self.options.group_cut);
        let mine = groups
            .iter()
            .find(|g| g.contains(&0))
            .expect("the query is in a group")
            .clone();
        let group_middle = (mine.len() > 1).then(|| {
            let others: Vec<f64> = mine
                .iter()
                .filter(|&&i| i != 0)
                .map(|&i| similar[i])
                .collect();
            others.iter().sum::<f64>() / others.len() as f64
        });
        reading.grouping = Some(FarFieldGrouping {
            groups: groups
                .iter()
                .map(|g| {
                    let mut images: Vec<u32> = g.iter().map(|&i| reading.views[i].0).collect();
                    images.sort_unstable();
                    images
                })
                .collect(),
            query_middle: similar[1..n].to_vec(),
            group_middle,
            left_out: n - mine.len(),
        });

        if mine.len() >= 2 {
            reading.refit = Some(if mine.len() == n {
                Refit::Agrees
            } else {
                Refit::Grouped
            });
            let mut keep = mine;
            keep.sort_unstable();
            reading.views = keep.iter().map(|&i| reading.views[i]).collect();
            return (reading, true);
        }
        // The largest other group, the first listed of that size.
        let mut rest: Vec<usize> = Vec::new();
        for g in groups.iter().filter(|g| !g.contains(&0)) {
            if g.len() > rest.len() {
                rest = g.clone();
            }
        }
        if rest.len() < 2 {
            reading.refit = Some(Refit::Unsplit);
            return (reading, true);
        }
        rest.sort_unstable();
        let sight: Vec<(u32, [f64; 2])> = rest.iter().map(|&i| reading.views[i]).collect();
        let Ok(track) = self.refit_track(&sight) else {
            reading.refit = Some(Refit::Failed);
            return (reading, false);
        };
        let Some(payload) = track.track() else {
            reading.refit = Some(Refit::Failed);
            return (reading, false);
        };
        let Some(point) = payload.position else {
            reading.refit = Some(Refit::Failed);
            return (reading, false);
        };
        let at_inf = payload.at_infinity;
        let x = point.coords;
        let cq = &self.cameras[self.image as usize];
        let land = if at_inf {
            cq.project_direction(&x)
        } else if cq.depth(&x) > 0.0 {
            cq.project_homogeneous(&x, 1.0)
        } else {
            None
        };
        let Some(land) = land.filter(|&l| cq.in_frame(l, 2.0)) else {
            reading.refit = Some(Refit::OffTheImage);
            return (reading, false);
        };
        let off = (land[0] - self.pixel[0]).hypot(land[1] - self.pixel[1]);
        reading.refit_px = Some(off);
        if off <= self.options.refit_px {
            reading.refit = Some(Refit::Stands);
            let mut views = vec![reading.views[0]];
            views.extend(rest.iter().map(|&i| reading.views[i]));
            reading.views = views;
            return (reading, true);
        }
        let mut views = vec![(self.image, land)];
        let mut errs = Vec::new();
        for o in &track.observations[1..] {
            let Some(m) = o.track.as_ref() else { continue };
            let Some(kp) = m.keypoint else { continue };
            if o.verdict != Verdict::In {
                continue;
            }
            views.push((o.image, [f64::from(kp[0]), f64::from(kp[1])]));
            errs.push(m.reprojection_error.unwrap_or(0.0));
        }
        let worst = errs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        if off > self.options.refit_max_px
            || views.len() < 3
            || worst > self.options.refit_max_err_px
        {
            reading.refit = Some(Refit::MovedRejected);
            return (reading, false);
        }
        reading.max_ray_angle_deg = if at_inf {
            0.0
        } else {
            ray_angle(self.cameras, &x, views.iter().map(|v| v.0))
        };
        reading.depth = if at_inf { f64::INFINITY } else { cq.depth(&x) };
        reading.position = x;
        reading.at_infinity = at_inf;
        reading.views = views;
        reading.query_pixel = land;
        reading.distance_px = off;
        reading.max_reproj_px = worst;
        reading.range = None;
        reading.refit = Some(Refit::Moved);
        (reading, true)
    }

    /// A track on `sight` with the queried pixel as observation 0, turned
    /// out, fitted twice.
    fn refit_track(&self, sight: &[(u32, [f64; 2])]) -> Result<EditableTrack, String> {
        let name = &self.edited.base.image_table.images[self.image as usize].name;
        let stem = std::path::Path::new(name)
            .file_stem()
            .map_or_else(|| name.clone(), |s| s.to_string_lossy().into_owned());
        let seed = ClusterSeed::from_pixel(self.image, stem, self.pixel, self.options.radius_px);
        let track = seed_cluster_with(&seed, &Thresholds::default())?;
        let track = upgrade_sightings(self.edited, self.views, track, sight)?;
        let (mut track, _) = set_verdict(&track, 0, Verdict::Out).map_err(|e| e.to_string())?;
        for _ in 0..2 {
            track = fit(
                &track,
                self.edited,
                self.views,
                &FitOptions::default(),
                &Progress::none(),
            )
            .map_err(|e| e.to_string())?
            .0;
        }
        Ok(track)
    }
}
