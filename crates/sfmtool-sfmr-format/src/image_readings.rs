// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Each image's sharpness and gradient structure tensor (version 15+).
//!
//! Two required per-image columns, `images/sharpness` and
//! `images/gradient_structure_tensors_xx_yy_xy`, and the `image_to_gray`
//! object in `images/metadata.json` that says which grey conversion both were
//! computed through. The readings are for comparing the images of one
//! reconstruction with each other: they depend on scene content and are not
//! comparable across scenes. See `specs/formats/sfmr-file-format.md`
//! § "Image readings".

use ndarray::{Array1, Array2};

/// The first `.sfmr` version that stores the image readings: the columns
/// `images/sharpness` and `images/gradient_structure_tensors_xx_yy_xy` and the
/// `image_to_gray` key of `images/metadata.json`, all three required from it
/// on.
///
/// A file below it carries none of them, and the reader refuses one that
/// does.
pub const SFMR_IMAGE_READINGS_VERSION: u32 = 15;

/// The image readings of a `.sfmr` file: every image's sharpness and gradient
/// structure tensor, and the grey conversion both were computed through.
///
/// One conversion covers every row, because readings computed through
/// different conversions do not compare. In a file of version
/// [`SFMR_IMAGE_READINGS_VERSION`] or later the three are always present; a
/// file below it has none, and [`crate::SfmrData::image_readings`] is then
/// `None`.
#[derive(Debug, Clone, PartialEq)]
pub struct ImageReadingColumns {
    /// The `image_to_gray` object the rows were computed with:
    /// `{"formula": <string>, "quantize_bits": <1 to 16>}`, `quantize_bits`
    /// optional. Stored in `images/metadata.json`. This crate checks the
    /// object's shape; whether `formula` follows the grammar is checked where
    /// it is parsed, by `SfmrReconstruction::from_sfmr_data` in
    /// `sfmtool-core`.
    pub image_to_gray: serde_json::Value,
    /// `(N,)` per image, its sharpness: the population variance of the
    /// Laplacian over the interior of its reading image, in grey units
    /// squared. Finite and not negative. On disk
    /// `images/sharpness.{N}.float32.zst`.
    pub sharpness: Array1<f32>,
    /// `(N, 3)` per image, `[Exx, Eyy, Exy]`: the means of `Gx²`, `Gy²` and
    /// `Gx·Gy` over the interior of its reading image, not normalised.
    /// Finite; `Exx` and `Eyy` not negative. On disk
    /// `images/gradient_structure_tensors_xx_yy_xy.{N}.3.float32.zst`.
    pub gradient_structure_tensors_xx_yy_xy: Array2<f32>,
}

/// Check the image readings of a file of `image_count` images: the
/// `image_to_gray` object well formed (a non-empty string `formula`, an
/// optional integer `quantize_bits` from 1 to 16 and no other key), the
/// sharpness `(N,)` and the tensors `(N, 3)`, every value finite, and every
/// sharpness, `Exx` and `Eyy` not negative.
///
/// Read, write and verify all route through this, so a file the writer
/// accepts is one the reader and the verifier accept.
pub fn validate_image_readings(
    readings: &ImageReadingColumns,
    image_count: usize,
) -> Result<(), String> {
    sfmtool_sift_format::validate_image_to_gray(&readings.image_to_gray)
        .map_err(|e| format!("images/metadata.json {e}"))?;
    if readings.sharpness.len() != image_count {
        return Err(format!(
            "images/sharpness has {} rows, not {image_count}",
            readings.sharpness.len()
        ));
    }
    let tensors = &readings.gradient_structure_tensors_xx_yy_xy;
    if tensors.shape() != [image_count, 3] {
        return Err(format!(
            "images/gradient_structure_tensors_xx_yy_xy has shape {:?}, not [{image_count}, 3]",
            tensors.shape()
        ));
    }
    for (i, &s) in readings.sharpness.iter().enumerate() {
        if !s.is_finite() || s < 0.0 {
            return Err(format!(
                "images/sharpness row {i} is {s}, which is not finite and not negative"
            ));
        }
    }
    for (i, row) in tensors.rows().into_iter().enumerate() {
        let (xx, yy, xy) = (row[0], row[1], row[2]);
        if !(xx.is_finite() && yy.is_finite() && xy.is_finite()) || xx < 0.0 || yy < 0.0 {
            return Err(format!(
                "images/gradient_structure_tensors_xx_yy_xy row {i} is [{xx}, {yy}, {xy}], \
                 which is not finite with Exx and Eyy not negative"
            ));
        }
    }
    Ok(())
}

/// Why a file of `version` is refused for carrying image readings, or `None`
/// when it carries none it should not.
///
/// `stored` says whether the archive holds `images/sharpness`,
/// `images/gradient_structure_tensors_xx_yy_xy` and the `image_to_gray` key.
/// The reader and the verifier both ask this, so they refuse the same files.
pub(crate) fn image_readings_presence_error(version: u32, stored: [bool; 3]) -> Option<String> {
    const WHAT: [&str; 3] = [
        "images/sharpness",
        "images/gradient_structure_tensors_xx_yy_xy",
        "the image_to_gray key of images/metadata.json",
    ];
    if version < SFMR_IMAGE_READINGS_VERSION {
        let carried: Vec<&str> = (0..3).filter(|&k| stored[k]).map(|k| WHAT[k]).collect();
        return (!carried.is_empty()).then(|| {
            format!(
                "a version {version} file carries {}, which version \
                 {SFMR_IMAGE_READINGS_VERSION} introduced",
                carried.join(" and ")
            )
        });
    }
    let missing: Vec<&str> = (0..3).filter(|&k| !stored[k]).map(|k| WHAT[k]).collect();
    (!missing.is_empty()).then(|| {
        format!(
            "a version {version} file carries {}, {} and {}; this one lacks {}",
            WHAT[0],
            WHAT[1],
            WHAT[2],
            missing.join(" and ")
        )
    })
}

/// The `f32` values of a little-endian `float32` entry, or a message when its
/// length is not `expected` values.
pub(crate) fn f32_values(raw: &[u8], name: &str, expected: usize) -> Result<Vec<f32>, String> {
    if raw.len() != expected * 4 {
        return Err(format!(
            "{name} holds {} bytes, not {}",
            raw.len(),
            expected * 4
        ));
    }
    Ok(raw
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b))
        .collect())
}
