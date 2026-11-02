// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The statistics of one camera's per-image parameters over its images: what
//! the viewer's Camera Intrinsics panel shows beside the shared parameters, and
//! what its MCP `get_camera_intrinsics` reports.
//!
//! `specs/gui/camera-intrinsics.md` § "2. Parameters" is how they are shown.
//! The images they are taken over are the ones
//! [`drop_image_adjustments`](super::drop_image_adjustments::drop_image_adjustments)
//! folds the mean of, so the mean stated here is the value a drop moves into
//! the camera.

use std::fmt;

use super::bundle_adjust::is_posed;
use super::data::ImageSet;

/// The statistics of one number over a set of images.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ComponentStatistics {
    /// The mean.
    pub mean: f64,
    /// The population standard deviation, the root of the mean squared
    /// distance from [`Self::mean`].
    pub std: f64,
    /// The smallest value.
    pub min: f64,
    /// The largest value.
    pub max: f64,
    /// How many values lie more than two standard deviations from the mean.
    /// None does when the standard deviation is zero.
    pub beyond_two_std: usize,
}

impl ComponentStatistics {
    /// The statistics of `values`, or `None` for an empty slice.
    ///
    /// # Example
    ///
    /// ```
    /// use sfmtool_core::reconstruction::image_adjustment_statistics::ComponentStatistics;
    /// let s = ComponentStatistics::of(&[1.0, 2.0, 3.0]).unwrap();
    /// assert_eq!((s.mean, s.min, s.max, s.beyond_two_std), (2.0, 1.0, 3.0, 0));
    /// ```
    pub fn of(values: &[f64]) -> Option<Self> {
        if values.is_empty() {
            return None;
        }
        let n = values.len() as f64;
        let mean = values.iter().sum::<f64>() / n;
        let std = (values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n).sqrt();
        let min = values.iter().copied().fold(f64::INFINITY, f64::min);
        let max = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let beyond_two_std = values
            .iter()
            .filter(|v| (*v - mean).abs() > 2.0 * std)
            .count();
        Some(Self {
            mean,
            std,
            min,
            max,
            beyond_two_std,
        })
    }
}

/// The statistics of one camera's per-image parameters, one entry per
/// component, each in the unit the `.sfmr` file stores it in.
///
/// A parameter is `Some` when the camera varies it per image (its flag in
/// [`ImageSet::camera_per_image_adjustments`] is set) and the camera has an
/// image, and `None` otherwise, so a parameter the camera varies whose values
/// are all zero reads as statistics of zero rather than as absent.
#[derive(Debug, Clone, PartialEq)]
pub struct ImageAdjustmentStatistics {
    /// The images the statistics are over, in index order: the camera's
    /// posed images, or all its images where none is posed.
    pub images: Vec<usize>,
    /// The focal scale `s`, as a fraction of the camera's focal, taken in log
    /// space as the bundle adjustment re-centres it and a drop folds it:
    /// `mean` is the geometric mean of `1 + s` less one, `exp(mean ln(1 + s))
    /// − 1`; `std` is the standard deviation of `ln(1 + s)`, which for small
    /// values reads as a fraction of the focal as `s` does; `min` and `max` are
    /// the smallest and largest `s`; and `beyond_two_std` counts the images
    /// whose `ln(1 + s)` lies more than two of those deviations from its mean.
    pub focal: Option<ComponentStatistics>,
    /// The principal-point offset `[a_x, a_y]`, in the short-side unit.
    pub principal_point: Option<[ComponentStatistics; 2]>,
    /// The radial correction's coefficients `[e_0, e_1]`, in units of the
    /// normalized radius `ρ`.
    pub radial_bspline: Option<[ComponentStatistics; 2]>,
    /// The rolling-shutter rates `[ωx, ωy, ωz]`, in radians per short side of
    /// readout.
    pub rolling_shutter: Option<[ComponentStatistics; 3]>,
}

/// A camera index past the image set's cameras.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UnknownCamera {
    /// The index given.
    pub camera: usize,
    /// The number of cameras the image set has.
    pub count: usize,
}

impl fmt::Display for UnknownCamera {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "camera {} does not exist; the reconstruction has {} camera(s)",
            self.camera, self.count
        )
    }
}

impl std::error::Error for UnknownCamera {}

/// The images of camera `camera` a statement about its per-image parameters is
/// over: its posed images, whose observations the parameters change, or all
/// its images where none is posed. An unposed image's adjustment is
/// bookkeeping that keeps its intrinsics where they were.
pub(crate) fn adjustment_images(image_set: &ImageSet, camera: usize) -> Vec<usize> {
    let members: Vec<usize> = (0..image_set.images.len())
        .filter(|&i| image_set.images[i].camera_index as usize == camera)
        .collect();
    let posed: Vec<usize> = members
        .iter()
        .copied()
        .filter(|&i| {
            let image = &image_set.images[i];
            is_posed(&image.quaternion_wxyz, &image.translation_xyz)
        })
        .collect();
    if posed.is_empty() {
        members
    } else {
        posed
    }
}

/// The statistics of camera `camera`'s per-image parameters over its images
/// ([`ImageAdjustmentStatistics`]).
///
/// The images are its posed ones, or all of them where none is posed: the set
/// [`drop_image_adjustments`](super::drop_image_adjustments::drop_image_adjustments)
/// takes the mean over, so the focal's geometric mean and the other means are
/// the values a drop folds into the camera.
///
/// # Errors
///
/// [`UnknownCamera`] for an index past the cameras.
///
/// # Example
///
/// ```no_run
/// use sfmtool_core::reconstruction::image_adjustment_statistics::image_adjustment_statistics;
/// # fn run(recon: &sfmtool_core::SfmrReconstruction) -> Result<(), Box<dyn std::error::Error>> {
/// let statistics = image_adjustment_statistics(&recon.image_set, 0)?;
/// if let Some(focal) = statistics.focal {
///     println!("focal scale {:+.3}% ± {:.3}%", 100.0 * focal.mean, 100.0 * focal.std);
/// }
/// # Ok(())
/// # }
/// ```
pub fn image_adjustment_statistics(
    image_set: &ImageSet,
    camera: usize,
) -> Result<ImageAdjustmentStatistics, UnknownCamera> {
    let count = image_set.cameras.len();
    if camera >= count {
        return Err(UnknownCamera { camera, count });
    }
    let flags = image_set.camera_per_image_adjustments(camera);
    let images = adjustment_images(image_set, camera);
    let column = |value: &dyn Fn(&super::SfmrImage) -> f64| -> Vec<f64> {
        images
            .iter()
            .map(|&i| value(&image_set.images[i]))
            .collect()
    };
    let of = |value: &dyn Fn(&super::SfmrImage) -> f64| ComponentStatistics::of(&column(value));

    let focal = if flags.focal {
        ComponentStatistics::of(&column(&|image| image.intrinsic_adjustment.focal.ln_1p())).map(
            |log| ComponentStatistics {
                mean: log.mean.exp_m1(),
                std: log.std,
                min: log.min.exp_m1(),
                max: log.max.exp_m1(),
                beyond_two_std: log.beyond_two_std,
            },
        )
    } else {
        None
    };
    let principal_point = if flags.principal_point {
        both(
            of(&|image| image.intrinsic_adjustment.principal_point[0]),
            of(&|image| image.intrinsic_adjustment.principal_point[1]),
        )
    } else {
        None
    };
    let radial_bspline = if flags.radial_bspline {
        both(
            of(&|image| image.intrinsic_adjustment.radial_bspline[0]),
            of(&|image| image.intrinsic_adjustment.radial_bspline[1]),
        )
    } else {
        None
    };
    let rolling_shutter = if flags.rolling_shutter {
        match (
            of(&|image| image.intrinsic_adjustment.rolling_shutter[0]),
            of(&|image| image.intrinsic_adjustment.rolling_shutter[1]),
            of(&|image| image.intrinsic_adjustment.rolling_shutter[2]),
        ) {
            (Some(x), Some(y), Some(z)) => Some([x, y, z]),
            _ => None,
        }
    } else {
        None
    };
    Ok(ImageAdjustmentStatistics {
        images,
        focal,
        principal_point,
        radial_bspline,
        rolling_shutter,
    })
}

/// Two components' statistics, present when both are.
fn both(
    a: Option<ComponentStatistics>,
    b: Option<ComponentStatistics>,
) -> Option<[ComponentStatistics; 2]> {
    Some([a?, b?])
}

#[cfg(test)]
mod tests;
