// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The `.sfm-workspace.json` of a new workspace for the `sfmtool` feature
//! tool: [`sfmtool_workspace_marker`] gives its text and
//! [`write_sfmtool_workspace_marker`] writes it.
//!
//! This is the one writer of a new `sfmtool` workspace: `sfm ws init` calls
//! it through its binding, so the options a new workspace records cannot
//! differ between Rust and Python. The text is what Python's `json.dump(...,
//! indent=2)` writes for the record, with the keys in the order `sfm ws init`
//! has always written them and `mipmap_options` last, so the file is the one
//! Python's writer would write for the same record on Linux and macOS.

use std::path::Path;

use sfmtool_archive_io::WORKSPACE_MARKER;

use super::feature_options::{
    feature_prefix_dir, feature_type_for_tool, write_python_float, write_python_string,
    SFMTOOL_TOOL, WORKSPACE_VERSION,
};
use super::mipmap_options::{mipmap_options_json, NEW_WORKSPACE_MIPMAP_OPTIONS};
use crate::features::sift::{ImageToGray, SiftParams};

/// The `double_image_below_long_side` a new workspace records: an image
/// whose long side is 4096 pixels or more is not doubled before octave 0. The
/// reason is in `specs/core/features/sift.md` § "1. Scale-space pyramid".
/// [`SiftParams::default`] has no limit, which is what a record without the
/// key means.
pub const NEW_WORKSPACE_DOUBLE_IMAGE_BELOW_LONG_SIDE: u32 = 4096;

/// The choices `sfm ws init` offers for a new `sfmtool` workspace.
#[derive(Debug, Clone)]
pub struct SfmtoolWorkspaceOptions {
    /// The cap on the features kept per image: 8192, the default of
    /// [`SiftParams::max_num_features`].
    pub max_num_features: usize,
    /// The grey conversion: BT.709 luma, unrounded, by default.
    pub image_to_gray: ImageToGray,
}

impl Default for SfmtoolWorkspaceOptions {
    fn default() -> Self {
        let params = SiftParams::default();
        Self {
            max_num_features: params.max_num_features.unwrap_or(8192),
            image_to_gray: params.image_to_gray,
        }
    }
}

impl SfmtoolWorkspaceOptions {
    /// The SIFT parameters a new workspace records: [`SiftParams::default`]
    /// with `double_image_below_long_side` at
    /// [`NEW_WORKSPACE_DOUBLE_IMAGE_BELOW_LONG_SIDE`] and these options'
    /// feature cap and grey conversion.
    pub fn sift_params(&self) -> SiftParams {
        SiftParams {
            double_image_below_long_side: Some(NEW_WORKSPACE_DOUBLE_IMAGE_BELOW_LONG_SIDE),
            max_num_features: Some(self.max_num_features),
            image_to_gray: self.image_to_gray.clone(),
            ..SiftParams::default()
        }
    }

    /// The `feature_options` object a new workspace records:
    /// [`sift_params`](Self::sift_params) by field name, the object
    /// [`SiftParams::from_feature_options`] reads back to the same
    /// parameters.
    pub fn feature_options(&self) -> serde_json::Value {
        let p = self.sift_params();
        serde_json::json!({
            "octave_layers": p.octave_layers,
            "sigma": p.sigma,
            "blur_radius_factor": p.blur_radius_factor,
            "input_sigma": p.input_sigma,
            "double_image": p.double_image,
            "double_image_below_long_side": p.double_image_below_long_side,
            "contrast_threshold": p.contrast_threshold,
            "edge_threshold": p.edge_threshold,
            "max_num_features": p.max_num_features,
            "orientation_bins": p.orientation_bins,
            "peak_ratio": p.peak_ratio,
            "descriptor_width": p.descriptor_width,
            "descriptor_bins": p.descriptor_bins,
            "descriptor_magnification": p.descriptor_magnification,
            "descriptor_clamp": p.descriptor_clamp,
            "image_to_gray": p.image_to_gray.to_json(),
        })
    }
}

/// The keys of the record in the order `sfm ws init` writes them: the
/// record's own, then the feature options', then `image_to_gray`'s, then
/// `mipmap_options`'. A key
/// not listed would follow the listed ones in sorted order.
const KEY_ORDER: &[&str] = &[
    "version",
    "feature_tool",
    "feature_type",
    "feature_options",
    "feature_prefix_dir",
    "mipmap_options",
    "octave_layers",
    "sigma",
    "blur_radius_factor",
    "input_sigma",
    "double_image",
    "double_image_below_long_side",
    "contrast_threshold",
    "edge_threshold",
    "max_num_features",
    "orientation_bins",
    "peak_ratio",
    "descriptor_width",
    "descriptor_bins",
    "descriptor_magnification",
    "descriptor_clamp",
    "image_to_gray",
    "formula",
    "quantize_bits",
    "out_of_core_long_side",
    "jpeg_quality",
    "full_resolution_jpeg_quality",
];

/// The text of a new `sfmtool` workspace's `.sfm-workspace.json`: version
/// [`WORKSPACE_VERSION`], the `sfmtool` tool, its `feature_type`, the
/// options' [`feature_options`](SfmtoolWorkspaceOptions::feature_options),
/// the `feature_prefix_dir` [`feature_prefix_dir`] names from them, and the
/// `mipmap_options` object of [`NEW_WORKSPACE_MIPMAP_OPTIONS`], which turns
/// the out-of-core tier on for the new workspace.
///
/// It is written as Python's `json.dump(record, f, indent=2)` writes it: two
/// spaces an indent level, `", "` and `": "` separators with no space before
/// a line break, floats in Python's `repr` form, every character outside
/// printable ASCII as a `\uXXXX` escape, `\n` line breaks and no line break
/// at the end.
///
/// ```
/// use sfmtool_core::workspace::{sfmtool_workspace_marker, SfmtoolWorkspaceOptions};
///
/// let text = sfmtool_workspace_marker(&SfmtoolWorkspaceOptions::default());
/// let record: serde_json::Value = serde_json::from_str(&text).unwrap();
/// assert_eq!(record["feature_options"]["double_image_below_long_side"], 4096);
/// assert_eq!(record["mipmap_options"]["out_of_core_long_side"], 4096);
/// assert!(text.starts_with("{\n  \"version\": 2,\n  \"feature_tool\": \"sfmtool\",\n"));
/// ```
pub fn sfmtool_workspace_marker(options: &SfmtoolWorkspaceOptions) -> String {
    let feature_options = options.feature_options();
    let feature_type = feature_type_for_tool(SFMTOOL_TOOL, &feature_options);
    let prefix_dir = feature_prefix_dir(SFMTOOL_TOOL, &feature_type, &feature_options);
    let record = serde_json::json!({
        "version": WORKSPACE_VERSION,
        "feature_tool": SFMTOOL_TOOL,
        "feature_type": feature_type,
        "feature_options": feature_options,
        "feature_prefix_dir": prefix_dir,
        "mipmap_options": mipmap_options_json(&NEW_WORKSPACE_MIPMAP_OPTIONS),
    });
    let mut text = String::new();
    write_indented(&record, 0, &mut text);
    text
}

/// Write a new `sfmtool` workspace's `.sfm-workspace.json` into
/// `workspace_dir`, creating the directory and its parents when they do not
/// exist and replacing a workspace file already there; its folder of
/// features and its named image lists stay on disk. Returns the text written,
/// [`sfmtool_workspace_marker`]'s.
///
/// # Errors
///
/// The [`std::io::Error`] of creating the directory or writing the file.
pub fn write_sfmtool_workspace_marker(
    workspace_dir: &Path,
    options: &SfmtoolWorkspaceOptions,
) -> std::io::Result<String> {
    let text = sfmtool_workspace_marker(options);
    std::fs::create_dir_all(workspace_dir)?;
    std::fs::write(workspace_dir.join(WORKSPACE_MARKER), &text)?;
    Ok(text)
}

/// Append `value` to `out` as Python's `json.dumps(value, indent=2)` writes
/// it at nesting `level`, with an object's keys in [`KEY_ORDER`].
fn write_indented(value: &serde_json::Value, level: usize, out: &mut String) {
    use serde_json::Value;
    let newline = |level: usize, out: &mut String| {
        out.push('\n');
        out.extend(std::iter::repeat_n(' ', 2 * level));
    };
    match value {
        Value::Null => out.push_str("null"),
        Value::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        Value::Number(n) => {
            if let Some(u) = n.as_u64() {
                out.push_str(&u.to_string());
            } else if let Some(i) = n.as_i64() {
                out.push_str(&i.to_string());
            } else if let Some(f) = n.as_f64() {
                write_python_float(f, out);
            }
        }
        Value::String(s) => write_python_string(s, out),
        Value::Array(items) if items.is_empty() => out.push_str("[]"),
        Value::Array(items) => {
            out.push('[');
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                newline(level + 1, out);
                write_indented(item, level + 1, out);
            }
            newline(level, out);
            out.push(']');
        }
        Value::Object(map) if map.is_empty() => out.push_str("{}"),
        Value::Object(map) => {
            let rank = |key: &str| KEY_ORDER.iter().position(|k| *k == key);
            let mut entries: Vec<(&String, &Value)> = map.iter().collect();
            entries.sort_by(|a, b| match (rank(a.0), rank(b.0)) {
                (Some(x), Some(y)) => x.cmp(&y),
                (Some(_), None) => std::cmp::Ordering::Less,
                (None, Some(_)) => std::cmp::Ordering::Greater,
                (None, None) => a.0.cmp(b.0),
            });
            out.push('{');
            for (i, (key, item)) in entries.into_iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                newline(level + 1, out);
                write_python_string(key, out);
                out.push_str(": ");
                write_indented(item, level + 1, out);
            }
            newline(level, out);
            out.push('}');
        }
    }
}

#[cfg(test)]
mod tests;
