// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Reading a pixel's patch along its ray: the patch sampled on a plane facing
//! the queried camera at each of a list of distances, in each of a list of
//! images, scored against the queried photograph's own patch.

use nalgebra::Vector3;
use ndarray::{Array2, Array3};

use crate::bench::track_at_pixel::ViewCamera;
use crate::patch::normal_refine::ProjectedImage;

use super::grey::{sample_grey, GreyImages};

/// The samples along each side of a read's square grid. The grid's middle
/// sample, at index `PATCH_GRID * PATCH_GRID / 2`, is the pixel itself.
pub const PATCH_GRID: usize = 11;

/// How far from the grid's middle sample, along each side, the **middle** of
/// the patch reaches: `PATCH_GRID / 4`, so the middle is the central 5 x 5 of
/// the 11 x 11 grid, about half its width.
const MIDDLE_HALF: usize = PATCH_GRID / 4;

/// The queried pixel's patch: the image it is in, the pixel, and the patch's
/// half-width.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RayPatch {
    /// The queried image.
    pub image: u32,
    /// The pixel in it, `(x, y)` with pixel centres at half-integers.
    pub pixel: [f64; 2],
    /// From the pixel to the grid's edge, along each axis, in px of the
    /// queried image.
    pub radius_px: f64,
}

/// What reading a patch along its ray measured.
///
/// Every table is indexed `[distance, image]`, in the order the distances and
/// the images were asked for.
#[derive(Debug, Clone, PartialEq)]
pub struct PatchRead {
    /// The images read, as asked for.
    pub images: Vec<u32>,
    /// The distances along the pixel's ray the patch was read at, as asked
    /// for; `f64::INFINITY` is the patch at infinity.
    pub distances: Vec<f64>,
    /// The ZNCC of the whole grid with the queried patch's, `-1` where the
    /// patch cannot be read there (or reads as flat).
    pub whole: Array2<f64>,
    /// The ZNCC of the middle of the grid, from the same samples, `-1` where
    /// [`Self::whole`] is unread.
    pub middle: Array2<f64>,
    /// Where the pixel itself lands in each read, `[distance, image, xy]`,
    /// `NaN` where unread.
    pub centres: Array3<f64>,
    /// The grey standard deviation of the queried patch's middle, on the 0 to
    /// 255 scale: under about 8, the middle is too flat to correlate.
    pub middle_std: f64,
    /// The samples themselves, when they were asked for.
    pub samples: Option<PatchSamples>,
}

/// The samples behind a [`PatchRead`].
#[derive(Debug, Clone, PartialEq)]
pub struct PatchSamples {
    /// The queried patch's grey samples, row-major over the grid.
    pub template: Vec<f32>,
    /// Each read's samples, `[distance, image, sample]`, `NaN` where unread.
    pub values: Array3<f32>,
    /// Which samples of the grid are its middle.
    pub middle: Vec<bool>,
}

impl PatchRead {
    /// The index of `image` among [`Self::images`], if it was read.
    pub fn image_index(&self, image: u32) -> Option<usize> {
        self.images.iter().position(|&i| i == image)
    }
}

/// The ZNCC of `a` and `b`, or `-1` when either is flat.
fn zncc(a: impl Iterator<Item = f64> + Clone, b: impl Iterator<Item = f64> + Clone) -> f64 {
    let n = a.clone().count() as f64;
    let ma = a.clone().sum::<f64>() / n;
    let mb = b.clone().sum::<f64>() / n;
    let (mut ab, mut aa, mut bb) = (0.0, 0.0, 0.0);
    for (x, y) in a.zip(b) {
        let (x, y) = (x - ma, y - mb);
        ab += x * y;
        aa += x * x;
        bb += y * y;
    }
    let den = aa.sqrt() * bb.sqrt();
    if den > 1e-6 {
        ab / den
    } else {
        -1.0
    }
}

/// The population standard deviation of `x`.
fn std_dev(x: impl Iterator<Item = f64> + Clone) -> f64 {
    let n = x.clone().count() as f64;
    let m = x.clone().sum::<f64>() / n;
    (x.map(|v| (v - m) * (v - m)).sum::<f64>() / n).sqrt()
}

/// The grid's middle samples.
pub(super) fn middle_mask() -> Vec<bool> {
    let c = PATCH_GRID / 2;
    (0..PATCH_GRID * PATCH_GRID)
        .map(|k| {
            let (row, col) = (k / PATCH_GRID, k % PATCH_GRID);
            row.abs_diff(c) <= MIDDLE_HALF && col.abs_diff(c) <= MIDDLE_HALF
        })
        .collect()
}

/// Read `patch` at each of `distances` along its ray, in each of `images`.
///
/// The patch is a square grid of [`PATCH_GRID`] samples a side, `radius_px`
/// from the pixel to its edge, sampled in the queried photograph. Each grid
/// sample's ray is cut by the plane that faces the queried camera at the
/// distance, the plane `(X - C) . r = t` for the pixel's own unit ray `r` from
/// the camera centre `C`, and the point is projected into the other image and
/// sampled there. At infinity the rays themselves are projected, as
/// directions. Each read is compared with the queried patch by ZNCC, over the
/// whole grid and over its middle (the central samples, about half its width)
/// from the same samples: a match of the whole patch that the middle does not
/// share is carried by the parts away from the pixel.
///
/// A read is skipped, and scores `-1`, when any grid point is behind the other
/// camera, when the pixel lands less than half the radius inside the other
/// photograph's frame, or when any sample is off it. Every photograph is read
/// in grey and blurred ([`GreyImages`]).
///
/// Returns `None` when the queried patch runs off its photograph or is flat,
/// with nothing to correlate. `keep_samples` keeps the samples in
/// [`PatchRead::samples`].
///
/// # Panics
///
/// Panics if `patch.image` or an entry of `images` is not an index into
/// `views` and `grey`.
pub fn read_patch_along_ray(
    views: &[ProjectedImage<'_>],
    grey: &GreyImages,
    patch: &RayPatch,
    distances: &[f64],
    images: &[u32],
    keep_samples: bool,
) -> Option<PatchRead> {
    let n = PATCH_GRID;
    let centre = (n * n) / 2;
    let q = patch.image as usize;
    let cam = ViewCamera::new(&views[q]);
    let r = patch.radius_px;
    let step = 2.0 * r / (n - 1) as f64;
    let offsets: Vec<f64> = (0..n)
        .map(|i| if i == n - 1 { r } else { i as f64 * step - r })
        .collect();
    let grid: Vec<[f64; 2]> = (0..n * n)
        .map(|k| {
            [
                patch.pixel[0] + offsets[k % n],
                patch.pixel[1] + offsets[k / n],
            ]
        })
        .collect();

    let query_grey = grey.get(views, q);
    let template: Vec<f32> = grid
        .iter()
        .map(|p| sample_grey(query_grey, p[0], p[1]))
        .collect::<Option<_>>()?;
    let as_f64 = |v: &[f32]| v.iter().map(|&x| f64::from(x)).collect::<Vec<_>>();
    let template64 = as_f64(&template);
    if std_dev(template64.iter().copied()) < 1e-3 {
        return None;
    }
    let middle = middle_mask();
    let middle_std = std_dev(
        template64
            .iter()
            .zip(&middle)
            .filter(|(_, &m)| m)
            .map(|(&v, _)| v),
    );

    let rays: Vec<Vector3<f64>> = grid.iter().map(|&p| cam.ray(p).normalize()).collect();
    let axis = rays[centre];
    let along: Vec<f64> = rays.iter().map(|ray| ray.dot(&axis)).collect();

    let (nd, ni) = (distances.len(), images.len());
    let mut whole = Array2::from_elem((nd, ni), -1.0);
    let mut mid = Array2::from_elem((nd, ni), -1.0);
    let mut centres = Array3::from_elem((nd, ni, 2), f64::NAN);
    let mut values = keep_samples.then(|| Array3::from_elem((nd, ni, n * n), f32::NAN));
    let others: Vec<ViewCamera<'_>> = images
        .iter()
        .map(|&i| ViewCamera::new(&views[i as usize]))
        .collect();

    let mut points = vec![Vector3::zeros(); n * n];
    let mut pixels: Vec<Option<[f64; 2]>> = vec![None; n * n];
    let mut sampled = vec![0.0f32; n * n];
    for (di, &t) in distances.iter().enumerate() {
        for (vi, oc) in others.iter().enumerate() {
            for (k, ray) in rays.iter().enumerate() {
                points[k] = if t.is_finite() {
                    oc.to_camera_homogeneous(&(cam.center + ray * (t / along[k])), 1.0)
                } else {
                    oc.to_camera_homogeneous(ray, 0.0)
                };
            }
            if points.iter().any(|p| -p.z <= 1e-9) {
                continue;
            }
            for (k, p) in points.iter().enumerate() {
                pixels[k] = oc.camera_ray_to_pixel(p);
            }
            let Some(c) = pixels[centre] else { continue };
            if !oc.in_frame(c, r * 0.5) {
                continue;
            }
            let other_grey = grey.get(views, images[vi] as usize);
            let all = pixels.iter().zip(sampled.iter_mut()).all(|(p, s)| {
                match p.and_then(|p| sample_grey(other_grey, p[0], p[1])) {
                    Some(v) => {
                        *s = v;
                        true
                    }
                    None => false,
                }
            });
            if !all {
                continue;
            }
            let vals = as_f64(&sampled);
            whole[[di, vi]] = zncc(template64.iter().copied(), vals.iter().copied());
            let pick = |x: &[f64]| {
                x.iter()
                    .zip(&middle)
                    .filter(|(_, &m)| m)
                    .map(|(&v, _)| v)
                    .collect::<Vec<_>>()
            };
            let (tm, vm) = (pick(&template64), pick(&vals));
            mid[[di, vi]] = zncc(tm.iter().copied(), vm.iter().copied());
            centres[[di, vi, 0]] = c[0];
            centres[[di, vi, 1]] = c[1];
            if let Some(values) = values.as_mut() {
                for (k, &v) in sampled.iter().enumerate() {
                    values[[di, vi, k]] = v;
                }
            }
        }
    }
    Some(PatchRead {
        images: images.to_vec(),
        distances: distances.to_vec(),
        whole,
        middle: mid,
        centres,
        middle_std,
        samples: values.map(|values| PatchSamples {
            template,
            values,
            middle,
        }),
    })
}
