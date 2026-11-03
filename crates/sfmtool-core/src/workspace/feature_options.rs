// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The feature options a workspace records, and how a version 1 record becomes
//! a version 2 one.
//!
//! `.sfm-workspace.json`, and the `workspace.contents` of every `.sfmr` written
//! from it, record the feature extraction as `feature_tool`, `feature_type`,
//! `feature_options` and `feature_prefix_dir`. In a version 2 workspace,
//! `feature_options` carries the `image_to_gray` object at its top level, for
//! every feature tool; a version 1 workspace records the `sfmtool` tool's
//! conversion as a `gray_formula` string and the other tools' not at all. See
//! `specs/workspace/workspace.md`.
//!
//! [`upgrade_feature_options`] is the conversion `sfm ws upgrade` makes, and
//! the viewer's *Build SIFT Files* makes the same one in memory for a version 1
//! record. [`recorded_image_to_gray`] reads the conversion a record of either
//! version states, which is what `sfm undistort` records for its output.
//! [`feature_prefix_dir`] names the folder a new record's `.sift` files go in.

use std::fmt;

use crate::features::sift::{ImageToGray, ImageToGrayError, DEFAULT_GRAY_FORMULA};

/// The `.sfm-workspace.json` version a writer writes.
pub const WORKSPACE_VERSION: u32 = 2;

/// The `feature_tool` of the toolkit's own SIFT extractor.
pub const SFMTOOL_TOOL: &str = "sfmtool";
/// The `feature_tool` of the COLMAP extractor.
pub const COLMAP_TOOL: &str = "colmap";
/// The `feature_tool` of the OpenCV extractor.
pub const OPENCV_TOOL: &str = "opencv";
/// The `feature_tool` `sfm undistort` records for the `.sift` files it writes,
/// which it makes from its source's rather than by extraction.
pub const UNDISTORT_TOOL: &str = "sfmtool-undistort";

/// The BT.601 luma weights, the grey conversion of OpenCV's `cvtColor`.
const BT601_GRAY_FORMULA: &str = "0.299*R + 0.587*G + 0.114*B";

/// The feature options that describe hardware or performance choices rather
/// than the features, and so do not name a folder. Python's
/// `get_feature_tool_xxh128` leaves out the same keys.
const NON_DEFINING_OPTION_KEYS: &[&str] = &["use_gpu"];

/// Why a record's conversion could not be read or upgraded.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FeatureOptionsError {
    /// `feature_options` is not a JSON object.
    NotAnObject,
    /// The options already carry a top-level `image_to_gray`, so they are
    /// version 2 settings.
    AlreadyVersion2,
    /// The record is `sfmtool-undistort`'s, whose `.sift` files are made from
    /// its source's and cannot be extracted again.
    Undistort,
    /// A feature tool this build knows no conversion for.
    UnknownTool(String),
    /// The `sfmtool-undistort` record has no `source_feature_tool` or
    /// `source_feature_options` to read its source's conversion from.
    NoSource,
    /// The `gray_formula` is not a string or does not parse.
    GrayFormula(String),
    /// The `image_to_gray` object does not parse.
    ImageToGray(ImageToGrayError),
}

impl fmt::Display for FeatureOptionsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotAnObject => write!(f, "feature_options is not a JSON object"),
            Self::AlreadyVersion2 => write!(
                f,
                "feature_options already carry 'image_to_gray', so they are version 2 settings"
            ),
            Self::Undistort => write!(
                f,
                "the feature tool is 'sfmtool-undistort', whose .sift files are made from its \
                 source's and cannot be extracted again; run 'sfm undistort' again to write a \
                 version 2 workspace"
            ),
            Self::UnknownTool(tool) => write!(
                f,
                "the feature tool '{tool}' is not one of 'sfmtool', 'colmap' or 'opencv'"
            ),
            Self::NoSource => write!(
                f,
                "the 'sfmtool-undistort' feature_options have no 'source_feature_tool' and \
                 'source_feature_options'"
            ),
            Self::GrayFormula(problem) => write!(f, "'gray_formula' {problem}"),
            Self::ImageToGray(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for FeatureOptionsError {}

/// The `image_to_gray` a version 2 record of `feature_tool` carries when it is
/// made from the tool's defaults: BT.709 luma, unrounded, for `sfmtool` (a
/// setting, which a workspace may change); BT.709 luma rounded to 8 bits for
/// `colmap` and BT.601 luma rounded to 8 bits for `opencv`, the conversions
/// those tools make, which a workspace cannot change. `None` for any other
/// tool. The tool name is matched without regard to case.
///
/// ```
/// use sfmtool_core::workspace::tool_image_to_gray;
///
/// let opencv = tool_image_to_gray("opencv").unwrap();
/// assert_eq!(opencv.formula(), "0.299*R + 0.587*G + 0.114*B");
/// assert_eq!(opencv.quantize_bits(), Some(8));
/// assert!(tool_image_to_gray("sfmtool-undistort").is_none());
/// ```
pub fn tool_image_to_gray(feature_tool: &str) -> Option<ImageToGray> {
    let (formula, bits) = match feature_tool.to_ascii_lowercase().as_str() {
        SFMTOOL_TOOL => (DEFAULT_GRAY_FORMULA, None),
        COLMAP_TOOL => (DEFAULT_GRAY_FORMULA, Some(8)),
        OPENCV_TOOL => (BT601_GRAY_FORMULA, Some(8)),
        _ => return None,
    };
    Some(ImageToGray::new(formula, bits).expect("a tool's conversion parses"))
}

/// Whether `feature_options` are version 2 settings: whether they carry an
/// `image_to_gray` key at their top level. The key's value is not checked.
pub fn has_image_to_gray(feature_options: &serde_json::Value) -> bool {
    feature_options.get("image_to_gray").is_some()
}

/// The conversion a record of either version states for `feature_tool`'s
/// `feature_options`.
///
/// It is the top-level `image_to_gray` object when there is one; otherwise a
/// `gray_formula` string, read with no rounding; otherwise the tool's
/// conversion ([`tool_image_to_gray`]). An `sfmtool-undistort` record without
/// its own object states its source's, read from `source_feature_tool` and
/// `source_feature_options` the same way. This is what `sfm undistort` records
/// as its output's `image_to_gray`, and what a reader takes a version 1 file's
/// conversion to be.
///
/// # Errors
///
/// [`FeatureOptionsError`] for options that are not an object, an object or a
/// formula that does not parse, an `sfmtool-undistort` record with no source,
/// and a tool with no conversion of its own and no `gray_formula`.
pub fn recorded_image_to_gray(
    feature_tool: &str,
    feature_options: &serde_json::Value,
) -> Result<ImageToGray, FeatureOptionsError> {
    let map = feature_options
        .as_object()
        .ok_or(FeatureOptionsError::NotAnObject)?;
    if let Some(value) = map.get("image_to_gray") {
        return ImageToGray::from_json(value).map_err(FeatureOptionsError::ImageToGray);
    }
    if let Some(value) = map.get("gray_formula") {
        return gray_formula_conversion(value);
    }
    if feature_tool.eq_ignore_ascii_case(UNDISTORT_TOOL) {
        let tool = map.get("source_feature_tool").and_then(|t| t.as_str());
        let options = map.get("source_feature_options");
        return match (tool, options) {
            (Some(tool), Some(options)) => recorded_image_to_gray(tool, options),
            _ => Err(FeatureOptionsError::NoSource),
        };
    }
    tool_image_to_gray(feature_tool)
        .ok_or_else(|| FeatureOptionsError::UnknownTool(feature_tool.to_owned()))
}

/// Make version 1 `feature_options` of `feature_tool` into version 2 ones,
/// keeping every other option as it is.
///
/// For `sfmtool`, a `gray_formula` becomes `image_to_gray: {"formula":
/// <gray_formula>}`, and options without one gain the tool's default. For
/// `colmap` and `opencv`, the options gain the tool's conversion; a
/// `gray_formula`, which neither tool reads, is dropped. This is the
/// conversion `sfm ws upgrade` makes, and the viewer makes for a version 1
/// `sfmtool` record before it builds `.sift` files from it.
///
/// ```
/// use sfmtool_core::workspace::upgrade_feature_options;
///
/// let v1 = serde_json::json!({"peak_ratio": 0.7, "gray_formula": "G"});
/// let v2 = upgrade_feature_options("sfmtool", &v1)?;
/// assert_eq!(v2, serde_json::json!({"peak_ratio": 0.7, "image_to_gray": {"formula": "G"}}));
/// # Ok::<(), sfmtool_core::workspace::FeatureOptionsError>(())
/// ```
///
/// # Errors
///
/// [`FeatureOptionsError::AlreadyVersion2`] for options that already carry
/// `image_to_gray`, [`FeatureOptionsError::Undistort`] for an
/// `sfmtool-undistort` record, [`FeatureOptionsError::UnknownTool`] for any
/// other tool than the three, [`FeatureOptionsError::GrayFormula`] for a
/// `gray_formula` that does not parse, and
/// [`FeatureOptionsError::NotAnObject`].
pub fn upgrade_feature_options(
    feature_tool: &str,
    feature_options: &serde_json::Value,
) -> Result<serde_json::Value, FeatureOptionsError> {
    let mut map = feature_options
        .as_object()
        .ok_or(FeatureOptionsError::NotAnObject)?
        .clone();
    if map.contains_key("image_to_gray") {
        return Err(FeatureOptionsError::AlreadyVersion2);
    }
    let tool = feature_tool.to_ascii_lowercase();
    let image_to_gray = match tool.as_str() {
        SFMTOOL_TOOL => match map.remove("gray_formula") {
            Some(value) => gray_formula_conversion(&value)?,
            None => tool_image_to_gray(SFMTOOL_TOOL).expect("sfmtool has a conversion"),
        },
        COLMAP_TOOL | OPENCV_TOOL => {
            map.remove("gray_formula");
            tool_image_to_gray(&tool).expect("the tool has a conversion")
        }
        UNDISTORT_TOOL => return Err(FeatureOptionsError::Undistort),
        _ => return Err(FeatureOptionsError::UnknownTool(feature_tool.to_owned())),
    };
    map.insert("image_to_gray".to_owned(), image_to_gray.to_json());
    Ok(serde_json::Value::Object(map))
}

/// The folder, relative to an image's directory, that a new record's `.sift`
/// files go in: `features/{feature_type}-{hash}`, the hash being the XXH3-128
/// of the compact JSON of `feature_tool`, `feature_type` and `feature_options`
/// with every object's keys sorted and without `use_gpu`, as 32 lowercase hex
/// digits.
///
/// This is the rule `get_feature_tool_xxh128` in `sfmtool.sift.file`
/// computes, so for the same options the two usually agree; they need not,
/// since a folder name is computed only when a folder is created and is read
/// back from the record after that. Options that differ always name different
/// folders.
pub fn feature_prefix_dir(
    feature_tool: &str,
    feature_type: &str,
    feature_options: &serde_json::Value,
) -> String {
    let mut hashed = feature_options.clone();
    if let Some(map) = hashed.as_object_mut() {
        for key in NON_DEFINING_OPTION_KEYS {
            map.remove(*key);
        }
    }
    // `serde_json` keeps an object's keys sorted (its `preserve_order`
    // feature is off) and writes compact JSON, so this is Python's
    // `json.dumps(..., sort_keys=True, separators=(",", ":"))` of the same
    // values. A test holds the `sfmtool` defaults' result to the prefix of a
    // workspace that `sfm ws init` wrote.
    let canonical = serde_json::json!({
        "feature_tool": feature_tool,
        "feature_type": feature_type,
        "feature_options": hashed,
    })
    .to_string();
    let hash = xxhash_rust::xxh3::xxh3_128(canonical.as_bytes());
    format!("features/{feature_type}-{hash:032x}")
}

/// The conversion a `gray_formula` value states, or why it states none.
fn gray_formula_conversion(value: &serde_json::Value) -> Result<ImageToGray, FeatureOptionsError> {
    let formula = value
        .as_str()
        .ok_or_else(|| FeatureOptionsError::GrayFormula(format!("is not a string: {value}")))?;
    ImageToGray::from_gray_formula(formula).map_err(|e| {
        FeatureOptionsError::GrayFormula(match e {
            ImageToGrayError::Formula { formula, error } => {
                format!("{formula:?} does not parse: {error}")
            }
            other => other.to_string(),
        })
    })
}

#[cfg(test)]
mod tests;
