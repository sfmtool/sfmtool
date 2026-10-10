// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Each observation's readings on its own `R×R` render: the eight optional
//! `tracks/` columns flagged by `tracks/metadata.json`'s
//! `has_observation_readings`, and the reading options recorded beside them.
//!
//! See `specs/formats/sfmr-file-format.md` § "Observation readings".

use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};

use crate::entries::ReadingColumn;

/// The `tracks/metadata.json` flag that says the eight observation-reading
/// columns are present.
pub const HAS_OBSERVATION_READINGS: &str = "has_observation_readings";

/// The `tracks/metadata.json` key holding the [`ObservationReadingOptions`]
/// the readings were taken with.
pub const OBSERVATION_READING_OPTIONS: &str = "observation_reading_options";

/// One observation's readings on its own `R×R` render: the render through the
/// point's patch re-anchored on the observation's keypoint, at the point's
/// patch resolution `R`, with the sampler the sampler rule picks for it.
///
/// A row is a record of the render it names. The angle, tilt and zoom fields
/// describe the geometry of that render, not the file's current geometry, so
/// a reader can tell which render the readings were taken on.
///
/// `NaN` in [`Self::ellipse_axes`] means the observation was not measured;
/// its angles and zoom are then `NaN` and its flags `false`. `NaN` in a score
/// means the score was not read, as where the point has no stored bitmap. The
/// reference observation's scores are `1`.
///
/// Two rows are equal where every value is equal or `NaN` in both, so two
/// rows with nothing measured are equal.
#[derive(Debug, Clone, Copy)]
pub struct ObservationReading {
    /// `[semi-major, semi-minor]` axes of the whole render's ZNCC
    /// self-similarity ellipse, in grid px. The semi-major axis is the
    /// radius.
    pub ellipse_axes: [f32; 2],
    /// Per axis, whether the true length may be larger.
    pub ellipse_axes_is_at_least: [bool; 2],
    /// The angle of the ellipse's major axis from the patch's `u` axis
    /// towards its `v` axis, radians in `[0, π)`; `NaN` where the ellipse is
    /// a circle and has no direction.
    pub ellipse_major_angle: f32,
    /// `cos θ = −n · d̂` of the render: `n` the re-anchored patch's outward
    /// normal, `d̂` the unit ray from the camera centre through the
    /// observation's keypoint. Positive where the patch faces the camera.
    pub cos_view_angle: f32,
    /// The angle of the render's tilt direction (`d̂` projected into the
    /// patch plane) from the patch's `u` axis towards its `v` axis, radians in
    /// `[0, π)`. `NaN` where the view faces the patch head on.
    pub tilt_angle: f32,
    /// `[least, most]` zoom of the render: `[1/σ_major, 1/σ_minor]` of the
    /// Jacobian of the patch grid into the photograph at the patch centre,
    /// grid px per photograph px.
    pub zoom: [f32; 2],
    /// The render's plain ZNCC against the point's stored bitmap.
    pub plain_bitmap_zncc: f32,
    /// The render's blur-matched ZNCC against the point's stored bitmap.
    pub blur_matched_bitmap_zncc: f32,
}

impl ObservationReading {
    /// A row with nothing measured: every value `NaN`, every flag `false`.
    pub const NOT_MEASURED: Self = Self {
        ellipse_axes: [f32::NAN; 2],
        ellipse_axes_is_at_least: [false; 2],
        ellipse_major_angle: f32::NAN,
        cos_view_angle: f32::NAN,
        tilt_angle: f32::NAN,
        zoom: [f32::NAN; 2],
        plain_bitmap_zncc: f32::NAN,
        blur_matched_bitmap_zncc: f32::NAN,
    };

    /// Whether the self-similarity reading was taken: the semi-major axis is
    /// a number.
    pub fn is_measured(&self) -> bool {
        !self.ellipse_axes[0].is_nan()
    }

    /// The same row with both scores `NaN`: what a writer that changes the
    /// point's reference observation and does not render the observation
    /// again keeps.
    pub fn without_scores(self) -> Self {
        Self {
            plain_bitmap_zncc: f32::NAN,
            blur_matched_bitmap_zncc: f32::NAN,
            ..self
        }
    }

    fn floats(&self) -> [f32; 9] {
        [
            self.ellipse_axes[0],
            self.ellipse_axes[1],
            self.ellipse_major_angle,
            self.cos_view_angle,
            self.tilt_angle,
            self.zoom[0],
            self.zoom[1],
            self.plain_bitmap_zncc,
            self.blur_matched_bitmap_zncc,
        ]
    }
}

impl Default for ObservationReading {
    fn default() -> Self {
        Self::NOT_MEASURED
    }
}

impl PartialEq for ObservationReading {
    fn eq(&self, other: &Self) -> bool {
        self.ellipse_axes_is_at_least == other.ellipse_axes_is_at_least
            && self
                .floats()
                .iter()
                .zip(other.floats())
                .all(|(a, b)| *a == b || (a.is_nan() && b.is_nan()))
    }
}

/// The options the readings were taken with, recorded in
/// `tracks/metadata.json` so a reader can tell whether the stored radii are
/// comparable with its own, and work out each render's sampler from its
/// stored zoom.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ObservationReadingOptions {
    /// The self-similarity reading's `max_radius` `r`, in grid px: the
    /// largest shift searched and the largest radius read.
    pub max_radius: u32,
    /// The template spread, in grey levels, under which a channel carries no
    /// texture.
    pub flat_floor: f64,
    /// `n`, the noise between two views, in grey levels.
    pub noise: f64,
    /// `ε`, the ZNCC deficit two views of the same surface show, as a
    /// fraction.
    pub relative_tolerance: f64,
    /// The sampler rule's threshold `a` the renders were made under; `None`
    /// (`null`) where every render used one sampler whatever its zoom.
    pub anisotropic_threshold: Option<f64>,
}

/// The eight observation-reading columns as stored, parallel to the other
/// `tracks/*` arrays, with the options they were read with
/// ([`crate::SfmrData::observation_readings`]).
#[derive(Debug, Clone, PartialEq)]
pub struct ObservationReadingColumns {
    /// `(M, 2)` `tracks/zncc_self_similarity_ellipse_axes`.
    pub zncc_self_similarity_ellipse_axes: Array2<f32>,
    /// `(M, 2)` `tracks/zncc_self_similarity_ellipse_axes_is_at_least`, `0`
    /// or `1`.
    pub zncc_self_similarity_ellipse_axes_is_at_least: Array2<u8>,
    /// `(M,)` `tracks/zncc_self_similarity_ellipse_major_angle`.
    pub zncc_self_similarity_ellipse_major_angle: Array1<f32>,
    /// `(M,)` `tracks/zncc_self_similarity_cos_view_angle`.
    pub zncc_self_similarity_cos_view_angle: Array1<f32>,
    /// `(M,)` `tracks/zncc_self_similarity_tilt_angle`.
    pub zncc_self_similarity_tilt_angle: Array1<f32>,
    /// `(M, 2)` `tracks/zncc_self_similarity_zoom`.
    pub zncc_self_similarity_zoom: Array2<f32>,
    /// `(M,)` `tracks/plain_bitmap_zncc`.
    pub plain_bitmap_zncc: Array1<f32>,
    /// `(M,)` `tracks/blur_matched_bitmap_zncc`.
    pub blur_matched_bitmap_zncc: Array1<f32>,
    /// What the readings were taken with.
    pub options: ObservationReadingOptions,
}

impl ObservationReadingColumns {
    /// The columns holding `rows`, one per observation.
    pub fn from_rows(rows: &[ObservationReading], options: ObservationReadingOptions) -> Self {
        let m = rows.len();
        let pairs = |f: &dyn Fn(&ObservationReading) -> [f32; 2]| {
            Array2::from_shape_vec((m, 2), rows.iter().flat_map(f).collect())
                .expect("two values per row")
        };
        let singles = |f: &dyn Fn(&ObservationReading) -> f32| rows.iter().map(f).collect();
        Self {
            zncc_self_similarity_ellipse_axes: pairs(&|r| r.ellipse_axes),
            zncc_self_similarity_ellipse_axes_is_at_least: Array2::from_shape_vec(
                (m, 2),
                rows.iter()
                    .flat_map(|r| r.ellipse_axes_is_at_least.map(u8::from))
                    .collect(),
            )
            .expect("two values per row"),
            zncc_self_similarity_ellipse_major_angle: singles(&|r| r.ellipse_major_angle),
            zncc_self_similarity_cos_view_angle: singles(&|r| r.cos_view_angle),
            zncc_self_similarity_tilt_angle: singles(&|r| r.tilt_angle),
            zncc_self_similarity_zoom: pairs(&|r| r.zoom),
            plain_bitmap_zncc: singles(&|r| r.plain_bitmap_zncc),
            blur_matched_bitmap_zncc: singles(&|r| r.blur_matched_bitmap_zncc),
            options,
        }
    }

    /// The number of rows, `M`.
    pub fn len(&self) -> usize {
        self.plain_bitmap_zncc.len()
    }

    /// Whether there are no rows.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Row `j`.
    ///
    /// # Panics
    ///
    /// Panics if `j` is out of range.
    pub fn row(&self, j: usize) -> ObservationReading {
        let pair = |a: &Array2<f32>| [a[[j, 0]], a[[j, 1]]];
        let flags = &self.zncc_self_similarity_ellipse_axes_is_at_least;
        ObservationReading {
            ellipse_axes: pair(&self.zncc_self_similarity_ellipse_axes),
            ellipse_axes_is_at_least: [flags[[j, 0]] != 0, flags[[j, 1]] != 0],
            ellipse_major_angle: self.zncc_self_similarity_ellipse_major_angle[j],
            cos_view_angle: self.zncc_self_similarity_cos_view_angle[j],
            tilt_angle: self.zncc_self_similarity_tilt_angle[j],
            zoom: pair(&self.zncc_self_similarity_zoom),
            plain_bitmap_zncc: self.plain_bitmap_zncc[j],
            blur_matched_bitmap_zncc: self.blur_matched_bitmap_zncc[j],
        }
    }

    /// Every row, in order.
    pub fn rows(&self) -> Vec<ObservationReading> {
        (0..self.len()).map(|j| self.row(j)).collect()
    }

    /// Check that every column has `observation_count` rows of its shape, and
    /// that every flag is `0` or `1`.
    pub fn validate(&self, observation_count: usize) -> Result<(), String> {
        let m = observation_count;
        let pairs = [
            (
                "zncc_self_similarity_ellipse_axes",
                self.zncc_self_similarity_ellipse_axes.shape(),
            ),
            (
                "zncc_self_similarity_ellipse_axes_is_at_least",
                self.zncc_self_similarity_ellipse_axes_is_at_least.shape(),
            ),
            (
                "zncc_self_similarity_zoom",
                self.zncc_self_similarity_zoom.shape(),
            ),
        ];
        for (name, shape) in pairs {
            if shape != [m, 2] {
                return Err(format!("{name} shape {shape:?} != [{m}, 2]"));
            }
        }
        let singles = [
            (
                "zncc_self_similarity_ellipse_major_angle",
                self.zncc_self_similarity_ellipse_major_angle.len(),
            ),
            (
                "zncc_self_similarity_cos_view_angle",
                self.zncc_self_similarity_cos_view_angle.len(),
            ),
            (
                "zncc_self_similarity_tilt_angle",
                self.zncc_self_similarity_tilt_angle.len(),
            ),
            ("plain_bitmap_zncc", self.plain_bitmap_zncc.len()),
            (
                "blur_matched_bitmap_zncc",
                self.blur_matched_bitmap_zncc.len(),
            ),
        ];
        for (name, len) in singles {
            if len != m {
                return Err(format!("{name} len {len} != observation_count {m}"));
            }
        }
        if let Some(flag) = self
            .zncc_self_similarity_ellipse_axes_is_at_least
            .iter()
            .find(|&&f| f > 1)
        {
            return Err(format!(
                "zncc_self_similarity_ellipse_axes_is_at_least holds {flag}, not 0 or 1"
            ));
        }
        Ok(())
    }

    /// The columns with their rows taken in the order of `perm`: row `i` of
    /// the result is row `perm[i]` of `self`.
    pub fn select(&self, perm: &[usize]) -> Self {
        let rows = self.rows();
        let picked: Vec<ObservationReading> = perm.iter().map(|&j| rows[j]).collect();
        Self::from_rows(&picked, self.options)
    }

    /// The stored bytes of `column`, little-endian, row-major.
    ///
    /// # Panics
    ///
    /// Panics if a column is not contiguous.
    pub(crate) fn bytes(&self, column: ReadingColumn) -> &[u8] {
        fn f32s(a: &[f32]) -> &[u8] {
            bytemuck::cast_slice(a)
        }
        match column {
            ReadingColumn::BlurMatchedBitmapZncc => {
                f32s(self.blur_matched_bitmap_zncc.as_slice().unwrap())
            }
            ReadingColumn::PlainBitmapZncc => f32s(self.plain_bitmap_zncc.as_slice().unwrap()),
            ReadingColumn::CosViewAngle => {
                f32s(self.zncc_self_similarity_cos_view_angle.as_slice().unwrap())
            }
            ReadingColumn::EllipseAxes => {
                f32s(self.zncc_self_similarity_ellipse_axes.as_slice().unwrap())
            }
            ReadingColumn::EllipseAxesIsAtLeast => self
                .zncc_self_similarity_ellipse_axes_is_at_least
                .as_slice()
                .unwrap(),
            ReadingColumn::EllipseMajorAngle => f32s(
                self.zncc_self_similarity_ellipse_major_angle
                    .as_slice()
                    .unwrap(),
            ),
            ReadingColumn::TiltAngle => {
                f32s(self.zncc_self_similarity_tilt_angle.as_slice().unwrap())
            }
            ReadingColumn::Zoom => f32s(self.zncc_self_similarity_zoom.as_slice().unwrap()),
        }
    }
}

/// Whether `tracks/metadata.json` flags the observation-reading columns. A
/// missing flag is `false`.
pub(crate) fn observation_readings_flagged(tracks_meta: &serde_json::Value) -> bool {
    tracks_meta
        .get(HAS_OBSERVATION_READINGS)
        .and_then(|v| v.as_bool())
        .unwrap_or(false)
}

/// The reading options `tracks/metadata.json` records beside the columns.
pub(crate) fn observation_reading_options(
    tracks_meta: &serde_json::Value,
) -> Result<ObservationReadingOptions, String> {
    let value = tracks_meta
        .get(OBSERVATION_READING_OPTIONS)
        .cloned()
        .ok_or_else(|| {
            format!(
                "tracks/metadata.json sets {HAS_OBSERVATION_READINGS} without \
                 {OBSERVATION_READING_OPTIONS}"
            )
        })?;
    serde_json::from_value(value).map_err(|e| format!("{OBSERVATION_READING_OPTIONS}: {e}"))
}

/// The columns being read back, one at a time, by [`ReadingColumn`].
#[derive(Default)]
pub(crate) struct ReadingColumnsBuilder {
    floats: Vec<(ReadingColumn, Vec<f32>)>,
    flags: Option<Vec<u8>>,
}

impl ReadingColumnsBuilder {
    /// Hold `column`'s values, `observation_count · width` of them.
    pub(crate) fn push_f32(&mut self, column: ReadingColumn, values: Vec<f32>) {
        self.floats.push((column, values));
    }

    /// Hold the flags column's values.
    pub(crate) fn push_flags(&mut self, values: Vec<u8>) {
        self.flags = Some(values);
    }

    /// The columns, once all eight are held.
    pub(crate) fn finish(
        mut self,
        observation_count: usize,
        options: ObservationReadingOptions,
    ) -> Result<ObservationReadingColumns, String> {
        let m = observation_count;
        let mut take = |column: ReadingColumn| -> Result<Vec<f32>, String> {
            let at = self
                .floats
                .iter()
                .position(|(c, _)| *c == column)
                .ok_or_else(|| format!("{} was not read", column.stem()))?;
            Ok(self.floats.swap_remove(at).1)
        };
        let pair = |v: Vec<f32>, column: ReadingColumn| {
            Array2::from_shape_vec((m, 2), v).map_err(|e| format!("{}: {e}", column.stem()))
        };
        let columns = ObservationReadingColumns {
            zncc_self_similarity_ellipse_axes: pair(
                take(ReadingColumn::EllipseAxes)?,
                ReadingColumn::EllipseAxes,
            )?,
            zncc_self_similarity_ellipse_axes_is_at_least: Array2::from_shape_vec(
                (m, 2),
                self.flags
                    .ok_or("zncc_self_similarity_ellipse_axes_is_at_least was not read")?,
            )
            .map_err(|e| format!("zncc_self_similarity_ellipse_axes_is_at_least: {e}"))?,
            zncc_self_similarity_ellipse_major_angle: Array1::from_vec(take(
                ReadingColumn::EllipseMajorAngle,
            )?),
            zncc_self_similarity_cos_view_angle: Array1::from_vec(take(
                ReadingColumn::CosViewAngle,
            )?),
            zncc_self_similarity_tilt_angle: Array1::from_vec(take(ReadingColumn::TiltAngle)?),
            zncc_self_similarity_zoom: pair(take(ReadingColumn::Zoom)?, ReadingColumn::Zoom)?,
            plain_bitmap_zncc: Array1::from_vec(take(ReadingColumn::PlainBitmapZncc)?),
            blur_matched_bitmap_zncc: Array1::from_vec(take(ReadingColumn::BlurMatchedBitmapZncc)?),
            options,
        };
        columns.validate(m)?;
        Ok(columns)
    }
}
