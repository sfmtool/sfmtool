// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Guided matching: the keypoints near a pixel, each matched by descriptor
//! among the keypoints of every other image whose rays pass close to its own,
//! as candidate tracks.
//!
//! `specs/core/bench/nearby-sources.md` is the design.

use std::path::Path;
use std::sync::OnceLock;

use nalgebra::Vector3;
use sfmtool_sift_format::{read_sift_features, SiftError, DESCRIPTOR_DIM};

use crate::bench::track_at_pixel::ViewCamera;
use crate::features::kdforest::ImageKeypoints;
use crate::patch::normal_refine::ProjectedImage;

use super::candidate::{
    candidate, check_query, pixel_distance, NearbyCandidate, NearbySource, NearbySourceError,
};
use super::triangulate::{meet_dropping_worst, meet_rays, Sighting, Sightings};

/// One image's `.sift` descriptors, row for row with its keypoints.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ImageDescriptors {
    data: Vec<u8>,
}

impl ImageDescriptors {
    /// Descriptors from their rows laid end to end,
    /// [`DESCRIPTOR_DIM`] bytes to the row.
    ///
    /// # Panics
    ///
    /// Panics if `data` is not a whole number of rows.
    pub fn new(data: Vec<u8>) -> Self {
        assert_eq!(
            data.len() % DESCRIPTOR_DIM,
            0,
            "descriptors come in rows of {DESCRIPTOR_DIM}"
        );
        Self { data }
    }

    /// Every descriptor of a `.sift` file.
    pub fn read(sift_path: &Path) -> Result<Self, SiftError> {
        Ok(Self::new(read_sift_features(sift_path)?.descriptors))
    }

    /// How many descriptors there are.
    pub fn len(&self) -> usize {
        self.data.len() / DESCRIPTOR_DIM
    }

    /// Whether there are none.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Descriptor `row`.
    pub fn row(&self, row: usize) -> &[u8] {
        &self.data[row * DESCRIPTOR_DIM..(row + 1) * DESCRIPTOR_DIM]
    }
}

/// The ray through every keypoint of every image, in its camera's frame, each
/// image's built the first time it is read.
///
/// A ray in the camera's frame depends on the lens and the keypoint, not on
/// the pose, so one set serves every version of a reconstruction that keeps
/// its cameras; a query turns them into the world frame with the pose it
/// holds. The cache is filled through a shared reference, so one set can be
/// read from several threads.
#[derive(Debug, Default)]
pub struct KeypointRays {
    images: Vec<OnceLock<Vec<Vector3<f64>>>>,
}

impl KeypointRays {
    /// An empty cache for `count` images.
    pub fn new(count: usize) -> Self {
        Self {
            images: (0..count).map(|_| OnceLock::new()).collect(),
        }
    }

    /// How many images the cache is for.
    pub fn len(&self) -> usize {
        self.images.len()
    }

    /// Whether the cache is for no images.
    pub fn is_empty(&self) -> bool {
        self.images.is_empty()
    }

    /// Image `image`'s rays, built from `views[image]`'s lens the first time.
    fn get(
        &self,
        views: &[ProjectedImage<'_>],
        keypoints: &[ImageKeypoints],
        image: usize,
    ) -> &[Vector3<f64>] {
        self.images[image].get_or_init(|| {
            let camera = views[image].camera;
            keypoints[image]
                .positions
                .iter()
                .map(|p| {
                    let d = camera.pixel_to_ray(f64::from(p[0]), f64::from(p[1]));
                    Vector3::new(d[0], d[1], d[2])
                })
                .collect()
        })
    }
}

/// What guided matching reads: every image's keypoints and descriptors, and
/// the rays through the keypoints.
#[derive(Clone, Copy)]
pub struct GuidedSource<'a> {
    /// Every image's `.sift` keypoints, one entry per image of the
    /// reconstruction, in its order.
    pub keypoints: &'a [ImageKeypoints],
    /// The same images' `.sift` descriptors, row for row with the keypoints.
    pub descriptors: &'a [ImageDescriptors],
    /// The rays through the keypoints, a cache for the same images.
    pub rays: &'a KeypointRays,
}

/// What [`guided_matches`] keeps. The defaults are the harness's.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GuidedOptions {
    /// How far from the pixel a keypoint of the queried image may be, in px
    /// (harness `guided_radius_px`).
    pub radius_px: f64,
    /// The most keypoints matched, the nearest (harness `guided_max`).
    pub max_keypoints: usize,
    /// Keypoints this near the pixel or nearer are skipped, in px; below zero
    /// none is (harness `guided_skip_px`).
    pub skip_px: f64,
    /// How close, in px of either image, a keypoint's ray must pass to the
    /// queried keypoint's to be a candidate match (harness
    /// `guided_epipolar_px`).
    pub epipolar_px: f64,
    /// The nearest candidate descriptor is a match only when its distance is at
    /// most this share of the second nearest's (harness `guided_ratio`).
    pub ratio: f32,
    /// The largest descriptor distance of a match (harness
    /// `guided_max_dist`).
    pub max_distance: f32,
    /// A match past [`Self::max_distance`] but within this is tried after the
    /// first triangulation (harness `guided_loose_dist`).
    pub loose_distance: f32,
    /// The fewest images a candidate needs, the queried one among them
    /// (harness `guided_min_views`).
    pub min_views: usize,
    /// The largest reprojection error any sighting may have, in px (harness
    /// `max_reproj_px`).
    pub max_reproj_px: f64,
}

impl Default for GuidedOptions {
    fn default() -> Self {
        Self {
            radius_px: 24.0,
            max_keypoints: 8,
            skip_px: -1.0,
            epipolar_px: 2.0,
            ratio: 0.8,
            max_distance: 250.0,
            loose_distance: 400.0,
            min_views: 2,
            max_reproj_px: 2.0,
        }
    }
}

/// The rows of `keypoints` within `radius_px` of `pixel` and further than
/// `skip_px`, nearest first, at most `limit`; rows at one distance keep their
/// order.
pub(super) fn keypoints_near(
    keypoints: &ImageKeypoints,
    pixel: [f64; 2],
    radius_px: f64,
    skip_px: f64,
    limit: usize,
) -> Vec<usize> {
    let mut rows: Vec<(usize, f64)> = keypoints
        .positions
        .iter()
        .enumerate()
        .map(|(k, p)| (k, pixel_distance([f64::from(p[0]), f64::from(p[1])], pixel)))
        .filter(|&(_, d)| d <= radius_px && d > skip_px)
        .collect();
    rows.sort_by(|a, b| a.1.total_cmp(&b.1));
    rows.truncate(limit);
    rows.into_iter().map(|(k, _)| k).collect()
}

/// A keypoint's position as a pixel.
fn at(keypoints: &ImageKeypoints, row: usize) -> [f64; 2] {
    let p = keypoints.positions[row];
    [f64::from(p[0]), f64::from(p[1])]
}

/// The distance between two descriptors, as the harness computes it in 32-bit
/// floats: every partial sum of the squared differences is an integer below
/// `2^24`, so it is exact, and only the square root rounds.
fn descriptor_distance(a: &[u8], b: &[u8]) -> f32 {
    let sum: i32 = a
        .iter()
        .zip(b)
        .map(|(&x, &y)| {
            let d = i32::from(x) - i32::from(y);
            d * d
        })
        .sum();
    (sum as f32).sqrt()
}

/// The keypoints within [`GuidedOptions::radius_px`] of `pixel` in `image`,
/// each matched by descriptor along the rays of every other image's
/// keypoints, as candidate tracks.
///
/// Up to [`GuidedOptions::max_keypoints`] keypoints are matched, the nearest
/// first. With the cameras posed, a keypoint's match in another image lies on
/// a keypoint whose ray passes within [`GuidedOptions::epipolar_px`] of the
/// keypoint's own ray, measured in px of either image at the rays' closest
/// approach, in front of both cameras. Among those candidates the nearest
/// descriptor is the match when its distance is at most
/// [`GuidedOptions::ratio`] of the second nearest's (a lone candidate always
/// passes) and at most [`GuidedOptions::max_distance`]. Each keypoint's
/// matches are triangulated with it, dropping the worst while three or more
/// remain, until every one is within [`GuidedOptions::max_reproj_px`]. A match
/// that passes the ratio test but is further than the distance bar, within
/// [`GuidedOptions::loose_distance`], is then added, the most distinct first,
/// when the triangulation with it still meets every sighting within the bar.
/// A keypoint with at least [`GuidedOptions::min_views`] sightings is a
/// candidate, sitting at the keypoint and named by its row in
/// [`NearbyCandidate::id`].
///
/// `views` holds one entry per image of the reconstruction, and `source` the
/// same images; only the views' cameras are read.
pub fn guided_matches(
    views: &[ProjectedImage<'_>],
    source: &GuidedSource<'_>,
    image: u32,
    pixel: [f64; 2],
    options: &GuidedOptions,
) -> Result<Vec<NearbyCandidate>, NearbySourceError> {
    check_query(
        views.len(),
        &[
            ("keypoints", source.keypoints.len()),
            ("descriptors", source.descriptors.len()),
            ("rays", source.rays.len()),
        ],
        views,
        image,
        pixel,
    )?;
    for (i, (kp, desc)) in source.keypoints.iter().zip(source.descriptors).enumerate() {
        if kp.len() != desc.len() {
            return Err(NearbySourceError::RowMismatch {
                image: i as u32,
                keypoints: kp.len(),
                descriptors: desc.len(),
            });
        }
    }
    let q = image as usize;
    let kq = &source.keypoints[q];
    let rows = keypoints_near(
        kq,
        pixel,
        options.radius_px,
        options.skip_px,
        options.max_keypoints,
    );
    if rows.is_empty() {
        return Ok(Vec::new());
    }
    let cameras: Vec<ViewCamera<'_>> = views.iter().map(ViewCamera::new).collect();
    let cq = &cameras[q];
    let world = |camera: &ViewCamera<'_>, ray: &Vector3<f64>| {
        let w = camera.world_direction(ray);
        w / w.norm()
    };
    let rays_q = source.rays.get(views, source.keypoints, q);
    let rq: Vec<Vector3<f64>> = rows.iter().map(|&k| world(cq, &rays_q[k])).collect();
    let dq: Vec<&[u8]> = rows.iter().map(|&k| source.descriptors[q].row(k)).collect();
    let mut sightings: Vec<Sightings> = rows.iter().map(|&k| vec![(image, at(kq, k))]).collect();
    let mut loose: Vec<Vec<(f32, Sighting)>> = rows.iter().map(|_| Vec::new()).collect();
    let mut candidates: Vec<usize> = Vec::new();
    for (j, cj) in cameras.iter().enumerate() {
        if j == q {
            continue;
        }
        let rays_j = source.rays.get(views, source.keypoints, j);
        if rays_j.is_empty() {
            continue;
        }
        let rj: Vec<Vector3<f64>> = rays_j.iter().map(|r| world(cj, r)).collect();
        let w0 = cq.center - cj.center;
        let e: Vec<f64> = rj.iter().map(|r| r.dot(&w0)).collect();
        let kj = &source.keypoints[j];
        for (a, ra) in rq.iter().enumerate() {
            // The closest approach of the queried ray and each keypoint ray.
            let d = ra.dot(&w0);
            candidates.clear();
            for (t, rt) in rj.iter().enumerate() {
                let b = ra.dot(rt);
                let den = 1.0 - b * b;
                if den <= 1e-8 {
                    continue;
                }
                let s = (b * e[t] - d) / den;
                let u = (e[t] - b * d) / den;
                if !(s > 0.0 && u > 0.0) {
                    continue;
                }
                let g = (cq.center + ra * s) - (cj.center + rt * u);
                let gap = (g.x * g.x + g.y * g.y + g.z * g.z).sqrt();
                let err = (cq.focal * gap / s.abs()).max(cj.focal * gap / u.abs());
                if err <= options.epipolar_px {
                    candidates.push(t);
                }
            }
            if candidates.is_empty() {
                continue;
            }
            // The nearest descriptor and the second nearest.
            let mut best = (f32::INFINITY, 0usize);
            let mut second = f32::INFINITY;
            for &t in &candidates {
                let dist = descriptor_distance(source.descriptors[j].row(t), dq[a]);
                if dist < best.0 {
                    second = best.0;
                    best = (dist, t);
                } else if dist < second {
                    second = dist;
                }
            }
            let ratio = if candidates.len() > 1 {
                best.0 / second
            } else {
                0.0
            };
            if ratio > options.ratio {
                continue;
            }
            let found = (j as u32, at(kj, best.1));
            if best.0 <= options.max_distance {
                sightings[a].push(found);
            } else if best.0 <= options.loose_distance {
                loose[a].push((ratio, found));
            }
        }
    }

    let mut out = Vec::new();
    for ((&k, sg), mut extra) in rows.iter().zip(sightings).zip(loose) {
        if sg.len() < 2 {
            continue;
        }
        let Some((mut kept, mut met)) =
            meet_dropping_worst(&cameras, image, sg, options.max_reproj_px)
        else {
            continue;
        };
        extra.sort_by(|x, y| x.0.total_cmp(&y.0));
        for (_, found) in extra {
            let mut tried = kept.clone();
            tried.push(found);
            if let Some(with) = meet_rays(&cameras, &tried) {
                if with.max_error_px() <= options.max_reproj_px {
                    kept = tried;
                    met = with;
                }
            }
        }
        if kept.len() < options.min_views {
            continue;
        }
        out.push(candidate(
            &cameras,
            NearbySource::Guided,
            Some(k as u32),
            met.position,
            kept,
            met.errors_px,
            pixel,
        ));
    }
    Ok(out)
}
