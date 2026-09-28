// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The shape every matching source returns: a candidate track near the pixel,
//! with its sightings, its point and how well the sightings meet there.
//!
//! `specs/core/bench/nearby-sources.md` is the design.

use nalgebra::Vector3;

use crate::bench::track_at_pixel::ViewCamera;
use crate::patch::normal_refine::ProjectedImage;

use super::range::{distance_range, DistanceRangeError};

/// Which matching source found a candidate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum NearbySource {
    /// The reconstruction's own points observed near the pixel
    /// ([`super::nearby_points`]).
    Points,
    /// The cluster-patches clusters near the pixel, vetted by triangulation
    /// ([`super::nearby_cluster_tracks`]).
    Clusters,
    /// The keypoints near the pixel matched by descriptor along their rays
    /// ([`super::guided_matches`]).
    Guided,
    /// The SIFT index's constellation query from the pixel, its seed
    /// positions triangulated ([`super::constellation_seeds`]).
    Constellation,
}

impl NearbySource {
    /// The source's name, as the specs and reports spell it.
    pub fn name(self) -> &'static str {
        match self {
            Self::Points => "points",
            Self::Clusters => "clusters",
            Self::Guided => "guided",
            Self::Constellation => "constellation",
        }
    }
}

impl std::fmt::Display for NearbySource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}

/// A candidate track near the pixel: a point that several photographs agree
/// on, with where each of them sees it.
///
/// Every matching source returns this shape, so the layers that compare them
/// read one kind of value whatever found it.
#[derive(Debug, Clone, PartialEq)]
pub struct NearbyCandidate {
    /// The source that found it.
    pub source: NearbySource,
    /// What the source names it by: the point for [`NearbySource::Points`],
    /// the cluster for [`NearbySource::Clusters`], the queried image's keypoint
    /// row for [`NearbySource::Guided`] and for a [`NearbySource::Constellation`]
    /// query made from a keypoint; `None` for one made from the pixel.
    pub id: Option<u32>,
    /// The point, in world coordinates.
    pub position: Vector3<f64>,
    /// The sightings, `(image, pixel)`, the queried image's first.
    pub sightings: Vec<(u32, [f64; 2])>,
    /// Each sighting's reprojection error at [`Self::position`], in px, in the
    /// order of [`Self::sightings`].
    pub errors_px: Vec<f64>,
    /// Where the candidate sits in the queried image: its sighting there.
    pub query_pixel: [f64; 2],
    /// How far [`Self::query_pixel`] is from the pixel asked about, in px.
    pub distance_px: f64,
    /// The largest of [`Self::errors_px`].
    pub max_reproj_px: f64,
    /// The widest angle between two sightings' rays to the point, in degrees.
    pub max_ray_angle_deg: f64,
    /// The point's depth along the queried camera's axis.
    pub depth: f64,
    /// The reconstruction's point the candidate is, for
    /// [`NearbySource::Points`].
    pub point: Option<u32>,
}

impl NearbyCandidate {
    /// How many images see the candidate.
    pub fn n_views(&self) -> usize {
        self.sightings.len()
    }

    /// The queried image: the first sighting's.
    pub fn image(&self) -> u32 {
        self.sightings[0].0
    }

    /// The point's distance along [`Self::query_pixel`]'s unit ray from the
    /// queried camera's centre, the distance [`distance_range`] takes.
    ///
    /// # Panics
    ///
    /// Panics if the queried image is not one of `views`.
    pub fn ray_distance(&self, views: &[ProjectedImage<'_>]) -> f64 {
        let cq = ViewCamera::new(&views[self.image() as usize]);
        let ray = cq.ray(self.query_pixel);
        (self.position - cq.center).dot(&(ray / ray.norm()))
    }

    /// The candidate's [`distance_range`] along [`Self::query_pixel`]'s ray.
    pub fn range(
        &self,
        views: &[ProjectedImage<'_>],
        tolerance_px: f64,
    ) -> Result<[f64; 2], DistanceRangeError> {
        let image = self.image();
        if image as usize >= views.len() {
            return Err(DistanceRangeError::NoSuchImage {
                image,
                image_count: views.len(),
            });
        }
        distance_range(
            views,
            image,
            self.query_pixel,
            &self.sightings,
            self.ray_distance(views),
            tolerance_px,
        )
    }
}

/// Why a matching source could not run.
#[derive(Debug, Clone, PartialEq)]
pub enum NearbySourceError {
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
    /// An image's descriptors are not row for row with its keypoints.
    RowMismatch {
        /// The image.
        image: u32,
        /// How many keypoints it has.
        keypoints: usize,
        /// How many descriptors it has.
        descriptors: usize,
    },
}

impl std::fmt::Display for NearbySourceError {
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
            Self::RowMismatch {
                image,
                keypoints,
                descriptors,
            } => write!(
                f,
                "image {image} has {keypoints} keypoints but {descriptors} descriptors"
            ),
        }
    }
}

impl std::error::Error for NearbySourceError {}

/// Check that every input has one entry per image, that `image` is one of
/// them and that `pixel` is on its photograph.
pub(super) fn check_query(
    image_count: usize,
    inputs: &[(&'static str, usize)],
    views: &[ProjectedImage<'_>],
    image: u32,
    pixel: [f64; 2],
) -> Result<(), NearbySourceError> {
    for &(input, got) in inputs {
        if got != image_count {
            return Err(NearbySourceError::InputMismatch {
                input,
                got,
                image_count,
            });
        }
    }
    let Some(view) = views.get(image as usize) else {
        return Err(NearbySourceError::NoSuchImage { image, image_count });
    };
    let (width, height) = (view.camera.width, view.camera.height);
    let on_photo = pixel.iter().all(|c| c.is_finite())
        && pixel[0] >= 0.0
        && pixel[1] >= 0.0
        && pixel[0] < f64::from(width)
        && pixel[1] < f64::from(height);
    if !on_photo {
        return Err(NearbySourceError::PixelOffImage {
            pixel,
            width,
            height,
        });
    }
    Ok(())
}

/// The pixel distance between `a` and `b`.
pub(super) fn pixel_distance(a: [f64; 2], b: [f64; 2]) -> f64 {
    let (dx, dy) = (a[0] - b[0], a[1] - b[1]);
    (dx * dx + dy * dy).sqrt()
}

/// The widest angle between the rays from two of `images`' camera centres to
/// `x`, in degrees.
pub(super) fn ray_angle(
    cameras: &[ViewCamera<'_>],
    x: &Vector3<f64>,
    images: impl Iterator<Item = u32>,
) -> f64 {
    let dirs: Vec<Vector3<f64>> = images
        .map(|i| (cameras[i as usize].center - x).normalize())
        .collect();
    let mut best = 0.0f64;
    for (a, da) in dirs.iter().enumerate() {
        for db in &dirs[a + 1..] {
            best = best.max(da.dot(db).clamp(-1.0, 1.0).acos().to_degrees());
        }
    }
    best
}

/// A candidate from its sightings, the queried image's first, and their
/// errors at `position`.
pub(super) fn candidate(
    cameras: &[ViewCamera<'_>],
    source: NearbySource,
    id: Option<u32>,
    position: Vector3<f64>,
    sightings: Vec<(u32, [f64; 2])>,
    errors_px: Vec<f64>,
    pixel: [f64; 2],
) -> NearbyCandidate {
    let (image, query_pixel) = sightings[0];
    let max_reproj_px = errors_px.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let max_ray_angle_deg = ray_angle(cameras, &position, sightings.iter().map(|s| s.0));
    NearbyCandidate {
        source,
        id,
        position,
        query_pixel,
        distance_px: pixel_distance(query_pixel, pixel),
        max_reproj_px,
        max_ray_angle_deg,
        depth: cameras[image as usize].depth(&position),
        point: None,
        sightings,
        errors_px,
    }
}
