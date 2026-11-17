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
/// than the features, and so do not name a folder ([`feature_prefix_dir`]).
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

/// The `feature_type` a workspace records for `feature_tool` with
/// `feature_options`, the first part of its folder's name: `sift-opencv`,
/// `sift-sfmtool`, and for `colmap` `sift-colmap` followed by `-dsp` when
/// `domain_size_pooling` is set and `-max<N>` when `max_num_features` is
/// present and not 8192; any other tool gives `sift-<tool>`. The tool name is
/// matched in lower case.
///
/// This is the one implementation: `sfm ws init` and the extractors name a
/// record's `feature_type` with it through its binding,
/// `sfmtool.workspace.feature_type_for_tool`, and the workspace upgrade names
/// a converted folder with it. An option's value is read as Python reads it
/// there: any value but `false`, `null`, `0`, `""`, `[]` and `{}` sets
/// `domain_size_pooling`, and `max_num_features` is written as Python writes
/// it, so `8192.0` counts as 8192 and `1000.5` is written `max1000.5`.
///
/// ```
/// use sfmtool_core::workspace::feature_type_for_tool;
///
/// let options = serde_json::json!({"domain_size_pooling": true, "max_num_features": 4000});
/// assert_eq!(feature_type_for_tool("COLMAP", &options), "sift-colmap-dsp-max4000");
/// assert_eq!(feature_type_for_tool("sfmtool", &options), "sift-sfmtool");
/// ```
pub fn feature_type_for_tool(feature_tool: &str, feature_options: &serde_json::Value) -> String {
    use serde_json::Value;
    let tool = feature_tool.to_lowercase();
    match tool.as_str() {
        OPENCV_TOOL => "sift-opencv".to_owned(),
        SFMTOOL_TOOL => "sift-sfmtool".to_owned(),
        COLMAP_TOOL => {
            let mut name = "sift-colmap".to_owned();
            let truthy = |value: &Value| match value {
                Value::Null => false,
                Value::Bool(b) => *b,
                Value::Number(n) => n.as_f64() != Some(0.0),
                Value::String(s) => !s.is_empty(),
                Value::Array(a) => !a.is_empty(),
                Value::Object(o) => !o.is_empty(),
            };
            if feature_options
                .get("domain_size_pooling")
                .is_some_and(truthy)
            {
                name.push_str("-dsp");
            }
            match feature_options.get("max_num_features") {
                None | Some(Value::Null) => {}
                Some(Value::Number(n)) if n.as_f64() == Some(8192.0) => {}
                Some(value) => {
                    name.push_str("-max");
                    match value {
                        Value::String(s) => name.push_str(s),
                        Value::Bool(true) => name.push_str("True"),
                        Value::Bool(false) => name.push_str("False"),
                        Value::Number(n) if n.is_f64() => {
                            write_python_float(n.as_f64().unwrap_or_default(), &mut name)
                        }
                        other => write_python_json(other, &mut name),
                    }
                }
            }
            name
        }
        _ => format!("sift-{tool}"),
    }
}

/// The folder, relative to an image's directory, that a new record's `.sift`
/// files go in: `features/{feature_type}-{hash}`, the hash being the XXH3-128
/// of the compact JSON of `feature_tool`, `feature_type` and `feature_options`
/// with every object's keys sorted and without the top-level `use_gpu`, as 32
/// lowercase hex digits.
///
/// This is the one implementation of the folder name: Python calls it as
/// `sfmtool.workspace.feature_prefix_dir`. The JSON is written byte for byte
/// as Python's `json.dumps(..., sort_keys=True, separators=(",", ":"))` writes
/// it, which named every folder before this function did, so the same options
/// give the name an earlier `sfm ws init` or `sfm sift --extract` gave: a
/// float in Python's `repr` form (`3.0`, `1e-05`, `1e+16`), and every
/// character outside printable ASCII as a `\uXXXX` escape. Options that differ
/// always name different folders. A NaN or infinite float, an integer outside
/// the 64-bit range and a non-string key are not `serde_json::Value`s, so they
/// name no folder; the Python binding refuses them.
///
/// ```
/// use sfmtool_core::workspace::feature_prefix_dir;
///
/// let options = serde_json::json!({"peak_ratio": 0.8, "use_gpu": true});
/// let folder = feature_prefix_dir("sfmtool", "sift-sfmtool", &options);
/// assert!(folder.starts_with("features/sift-sfmtool-"));
/// ```
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
    let record = serde_json::json!({
        "feature_tool": feature_tool,
        "feature_type": feature_type,
        "feature_options": hashed,
    });
    let mut canonical = String::new();
    write_python_json(&record, &mut canonical);
    let hash = xxhash_rust::xxh3::xxh3_128(canonical.as_bytes());
    format!("features/{feature_type}-{hash:032x}")
}

/// Append `value` to `out` as Python's `json.dumps(value, sort_keys=True,
/// separators=(",", ":"))` writes it.
///
/// `serde_json`'s own writer differs from Python's in two ways that change the
/// bytes: it writes a float's exponent without a sign or a leading zero
/// (`1e-5` for Python's `1e-05`), and it writes a character outside ASCII as
/// UTF-8 where Python writes `\uXXXX`. Keys are sorted here rather than taken
/// in the map's order, which a `preserve_order` build would change; `String`
/// sorts by code point, as Python's `str` does.
fn write_python_json(value: &serde_json::Value, out: &mut String) {
    use serde_json::Value;
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
        Value::Array(items) => {
            out.push('[');
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                write_python_json(item, out);
            }
            out.push(']');
        }
        Value::Object(map) => {
            let mut entries: Vec<(&String, &Value)> = map.iter().collect();
            entries.sort_by(|a, b| a.0.cmp(b.0));
            out.push('{');
            for (i, (key, item)) in entries.into_iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                write_python_string(key, out);
                out.push(':');
                write_python_json(item, out);
            }
            out.push('}');
        }
    }
}

/// Append `f` as Python's `repr(float)` writes it: the shortest digits that
/// read back as `f`, the closest to `f` of those, positional with at least one digit after the point when
/// the decimal exponent is from -4 to 15, and otherwise `d.ddde±XX`, the
/// exponent signed and of at least two digits. A `serde_json` number is never
/// NaN or infinite, but those are written as Python writes them too.
fn write_python_float(f: f64, out: &mut String) {
    if f.is_nan() {
        out.push_str("NaN");
        return;
    }
    if f.is_infinite() {
        out.push_str(if f > 0.0 { "Infinity" } else { "-Infinity" });
        return;
    }
    // `{:e}` writes the fewest digits that read back as `f`, as
    // `[-]d[.ddd]e[-]x`. When several strings of that length read back, it
    // need not pick the one `repr` picks, which is the closest to `f`, ties
    // going to the even digit (`1997913866023742.2`, where `{:e}` gives
    // `...423e15`). `{:.N$e}` rounds the exact value that way, so it is
    // `repr`'s digits whenever it reads back as `f`.
    let shortest = format!("{f:e}");
    let digit_count = shortest
        .split('e')
        .next()
        .map_or(1, |m| m.chars().filter(char::is_ascii_digit).count());
    let rounded = format!("{f:.prec$e}", prec = digit_count.saturating_sub(1));
    let chosen = if rounded.parse::<f64>() == Ok(f) {
        rounded
    } else {
        shortest
    };
    let (mantissa, exponent) = chosen.split_once('e').unwrap_or((&chosen, "0"));
    let exponent: i32 = exponent.parse().unwrap_or(0);
    let (negative, mantissa) = match mantissa.strip_prefix('-') {
        Some(rest) => (true, rest),
        None => (false, mantissa),
    };
    let digits: String = mantissa.chars().filter(|c| *c != '.').collect();
    if negative {
        out.push('-');
    }
    if !(-4..16).contains(&exponent) {
        out.push_str(&digits[..1]);
        if digits.len() > 1 {
            out.push('.');
            out.push_str(&digits[1..]);
        }
        let sign = if exponent < 0 { '-' } else { '+' };
        out.push_str(&format!("e{sign}{:02}", exponent.unsigned_abs()));
    } else if exponent >= 0 {
        let point = exponent as usize + 1;
        if digits.len() <= point {
            out.push_str(&digits);
            out.extend(std::iter::repeat_n('0', point - digits.len()));
            out.push_str(".0");
        } else {
            out.push_str(&digits[..point]);
            out.push('.');
            out.push_str(&digits[point..]);
        }
    } else {
        out.push_str("0.");
        out.extend(std::iter::repeat_n('0', (-exponent - 1) as usize));
        out.push_str(&digits);
    }
}

/// Append `s` as a JSON string the way Python's `json.dumps` writes it with
/// its default `ensure_ascii=True`: printable ASCII as it is, `"`, `\` and
/// five control characters with their short escapes, and every other
/// character as `\uXXXX` in lowercase hex, one outside the Basic Multilingual
/// Plane as its UTF-16 surrogate pair.
fn write_python_string(s: &str, out: &mut String) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{8}' => out.push_str("\\b"),
            '\u{c}' => out.push_str("\\f"),
            ' '..='~' => out.push(c),
            _ => {
                let mut units = [0u16; 2];
                for unit in c.encode_utf16(&mut units) {
                    out.push_str(&format!("\\u{unit:04x}"));
                }
            }
        }
    }
    out.push('"');
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
