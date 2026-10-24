// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! [`SiftParams`] from the `feature_options` object a workspace records for the
//! `sfmtool` feature tool.
//!
//! `.sfm-workspace.json` and every `.sift` file the `sfmtool` tool writes carry
//! the extraction settings as a JSON object whose keys are the output-defining
//! [`SiftParams`] fields by name, plus `gray_formula` for
//! [`SiftParams::image_to_gray`]. [`SiftParams::from_feature_options`] is the one
//! reader of that object: the Python binding `extract_sift` converts its
//! `params` dict to the same JSON value and calls it, and
//! [`build_sift_files`](super::build_sift_files) calls it on the workspace's
//! object, so the two cannot read a key differently.

use std::fmt;

use super::{gray, SiftParams};

/// A `feature_options` object [`SiftParams::from_feature_options`] refused.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FeatureOptionsError {
    /// The key at fault, or `None` when the value as a whole is wrong (not an
    /// object, or the descriptor grid pair).
    pub key: Option<String>,
    /// What is wrong with it, as the end of a sentence that starts with the key.
    pub problem: String,
}

impl FeatureOptionsError {
    fn key(key: &str, problem: impl Into<String>) -> Self {
        Self {
            key: Some(key.to_owned()),
            problem: problem.into(),
        }
    }
}

impl fmt::Display for FeatureOptionsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.key {
            Some(key) => write!(f, "feature option '{key}' {}", self.problem),
            None => write!(f, "{}", self.problem),
        }
    }
}

impl std::error::Error for FeatureOptionsError {}

impl SiftParams {
    /// Apply a `feature_options` JSON object onto [`SiftParams::default`].
    ///
    /// The keys are `octave_layers`, `sigma`, `blur_radius_factor`,
    /// `input_sigma`, `double_image`, `contrast_threshold`, `edge_threshold`,
    /// `max_num_features` (an integer, or `null` for no cap),
    /// `orientation_bins`, `peak_ratio`, `descriptor_width`, `descriptor_bins`,
    /// `descriptor_magnification`, `descriptor_clamp` and `gray_formula` (a
    /// string [`gray::parse_gray_formula`] reads). A key left out keeps its
    /// default; `null` for the whole value means no overrides.
    ///
    /// # Errors
    ///
    /// [`FeatureOptionsError`] for a value that is not an object, an unknown
    /// key, a value of the wrong type, a value that would break the pipeline
    /// (`octave_layers` or `orientation_bins` below 1, a `sigma`,
    /// `blur_radius_factor`, `descriptor_magnification` or `descriptor_clamp`
    /// that is not positive and finite, an `input_sigma` that is negative or not
    /// finite), and a descriptor grid other than 4 x 8, since the descriptor is
    /// always 128 values.
    pub fn from_feature_options(
        options: &serde_json::Value,
    ) -> Result<SiftParams, FeatureOptionsError> {
        let mut out = SiftParams::default();
        let map = match options {
            serde_json::Value::Null => return Ok(out),
            serde_json::Value::Object(map) => map,
            _ => {
                return Err(FeatureOptionsError {
                    key: None,
                    problem: "the feature options must be a JSON object".to_owned(),
                })
            }
        };

        let wrong_type = |key: &str| FeatureOptionsError::key(key, "has the wrong type");
        let as_u32 = |key: &str, value: &serde_json::Value| {
            value
                .as_u64()
                .and_then(|v| u32::try_from(v).ok())
                .ok_or_else(|| wrong_type(key))
        };
        let as_f64 =
            |key: &str, value: &serde_json::Value| value.as_f64().ok_or_else(|| wrong_type(key));

        for (key, value) in map {
            let key = key.as_str();
            match key {
                "octave_layers" => out.octave_layers = as_u32(key, value)?,
                "sigma" => out.sigma = as_f64(key, value)?,
                "blur_radius_factor" => out.blur_radius_factor = as_f64(key, value)?,
                "input_sigma" => out.input_sigma = as_f64(key, value)?,
                "double_image" => {
                    out.double_image = value.as_bool().ok_or_else(|| wrong_type(key))?
                }
                "contrast_threshold" => out.contrast_threshold = as_f64(key, value)?,
                "edge_threshold" => out.edge_threshold = as_f64(key, value)?,
                "max_num_features" => {
                    // `null` means unlimited; an integer caps the feature count.
                    out.max_num_features = if value.is_null() {
                        None
                    } else {
                        Some(
                            value
                                .as_u64()
                                .and_then(|v| usize::try_from(v).ok())
                                .ok_or_else(|| {
                                    FeatureOptionsError::key(
                                        key,
                                        "must be a non-negative integer or None",
                                    )
                                })?,
                        )
                    };
                }
                "orientation_bins" => out.orientation_bins = as_u32(key, value)?,
                "peak_ratio" => out.peak_ratio = as_f64(key, value)?,
                "descriptor_width" => out.descriptor_width = as_u32(key, value)?,
                "descriptor_bins" => out.descriptor_bins = as_u32(key, value)?,
                "descriptor_magnification" => out.descriptor_magnification = as_f64(key, value)?,
                "descriptor_clamp" => out.descriptor_clamp = as_f64(key, value)?,
                "gray_formula" => {
                    let formula = value
                        .as_str()
                        .ok_or_else(|| FeatureOptionsError::key(key, "must be a string"))?;
                    out.image_to_gray = gray::parse_gray_formula(formula).map_err(|e| {
                        FeatureOptionsError::key(key, format!("is not a gray formula: {e}"))
                    })?;
                }
                other => {
                    return Err(FeatureOptionsError {
                        key: None,
                        problem: format!("unknown feature option '{other}'"),
                    });
                }
            }
        }

        // Reject degenerate values that would otherwise panic deep in the
        // pipeline (e.g. `orientation_bins = 0` -> `rem_euclid(0)`,
        // `octave_layers = 0` -> `k = 2^inf` -> a huge kernel allocation).
        let bad = |key: &str, problem: &str| Err(FeatureOptionsError::key(key, problem));
        if out.octave_layers < 1 {
            return bad("octave_layers", "must be >= 1");
        }
        if out.orientation_bins < 1 {
            return bad("orientation_bins", "must be >= 1");
        }
        if !(out.sigma.is_finite() && out.sigma > 0.0) {
            return bad("sigma", "must be a positive, finite number");
        }
        if !(out.blur_radius_factor.is_finite() && out.blur_radius_factor > 0.0) {
            return bad("blur_radius_factor", "must be a positive, finite number");
        }
        if !(out.input_sigma.is_finite() && out.input_sigma >= 0.0) {
            return bad("input_sigma", "must be a non-negative, finite number");
        }
        if !(out.descriptor_magnification.is_finite() && out.descriptor_magnification > 0.0) {
            return bad(
                "descriptor_magnification",
                "must be a positive, finite number",
            );
        }
        if !(out.descriptor_clamp.is_finite() && out.descriptor_clamp > 0.0) {
            return bad("descriptor_clamp", "must be a positive, finite number");
        }

        // The descriptor emits a fixed 128-D vector, so the grid dimensions are
        // not configurable. Reject other values rather than ignoring them.
        if out.descriptor_width != 4 || out.descriptor_bins != 8 {
            return Err(FeatureOptionsError {
                key: None,
                problem: format!(
                    "descriptor_width and descriptor_bins are fixed at 4 and 8 (the sfmtool \
                     descriptor produces a 128-D vector); got {} and {}",
                    out.descriptor_width, out.descriptor_bins
                ),
            });
        }

        Ok(out)
    }
}
