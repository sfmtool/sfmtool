// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The `mipmap_options` object of a workspace's `.sfm-workspace.json`, which
//! turns the out-of-core tier on and sets its threshold and the JPEG
//! qualities of its mip level files.

use std::fmt;
use std::path::{Path, PathBuf};

use crate::camera::mipmap::{MipmapOptions, DEFAULT_OUT_OF_CORE_LONG_SIDE};

use super::WORKSPACE_FILE_NAME;

/// Why a workspace file's settings could not be read. Each names the file.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum WorkspaceConfigError {
    /// The file could not be read.
    Io { path: PathBuf, error: String },
    /// The file is not valid JSON, or its top level is not an object.
    Json { path: PathBuf, error: String },
    /// The `mipmap_options` object is invalid: `key` is the key at fault
    /// (`None` when the value of `mipmap_options` itself is not an object),
    /// and `problem` says what is wrong, quoting the value.
    MipmapOptions {
        path: PathBuf,
        key: Option<String>,
        problem: String,
    },
}

impl fmt::Display for WorkspaceConfigError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io { path, error } => write!(f, "{}: {error}", path.display()),
            Self::Json { path, error } => write!(f, "{}: {error}", path.display()),
            Self::MipmapOptions {
                path,
                key: Some(key),
                problem,
            } => write!(f, "{}: mipmap_options.{key} {problem}", path.display()),
            Self::MipmapOptions {
                path,
                key: None,
                problem,
            } => write!(f, "{}: mipmap_options {problem}", path.display()),
        }
    }
}

impl std::error::Error for WorkspaceConfigError {}

/// The `mipmap_options` a new workspace records: photographs with a side
/// over [`DEFAULT_OUT_OF_CORE_LONG_SIDE`] are out of core, and every mip
/// level file is encoded at quality 95. The reasons are in
/// `specs/core/camera/mipmaps.md` § "The workspace setting".
pub const NEW_WORKSPACE_MIPMAP_OPTIONS: MipmapOptions = MipmapOptions {
    out_of_core_long_side: DEFAULT_OUT_OF_CORE_LONG_SIDE,
    jpeg_quality: 95,
    full_resolution_jpeg_quality: 95,
};

/// `options` as the `mipmap_options` object of `.sfm-workspace.json` holds
/// them, which [`read_mipmap_options`] reads back to `options`.
///
/// ```
/// use sfmtool_core::workspace::{mipmap_options_json, NEW_WORKSPACE_MIPMAP_OPTIONS};
///
/// let object = mipmap_options_json(&NEW_WORKSPACE_MIPMAP_OPTIONS);
/// assert_eq!(object["out_of_core_long_side"], 4096);
/// assert_eq!(object["jpeg_quality"], 95);
/// assert_eq!(object["full_resolution_jpeg_quality"], 95);
/// ```
pub fn mipmap_options_json(options: &MipmapOptions) -> serde_json::Value {
    serde_json::json!({
        "out_of_core_long_side": options.out_of_core_long_side,
        "jpeg_quality": options.jpeg_quality,
        "full_resolution_jpeg_quality": options.full_resolution_jpeg_quality,
    })
}

/// The keys of `mipmap_options`, each with its smallest and largest value.
const KEYS: [(&str, u64, u64); 3] = [
    ("out_of_core_long_side", 1, u32::MAX as u64),
    ("jpeg_quality", 1, 100),
    ("full_resolution_jpeg_quality", 1, 100),
];

/// The `mipmap_options` object of `<root>/.sfm-workspace.json`: `None` when
/// the file has none.
///
/// The object holds exactly the three keys of [`MipmapOptions`], each a JSON
/// integer: `out_of_core_long_side` at least 1, and the two qualities 1 to
/// 100. A missing key, an unknown key, a value that is not a JSON integer
/// (`95.0`, `"95"`, `true`) or one out of range is an error naming the file,
/// the key and the value, as is a file that cannot be read or parsed.
pub fn read_mipmap_options(root: &Path) -> Result<Option<MipmapOptions>, WorkspaceConfigError> {
    let path = root.join(WORKSPACE_FILE_NAME);
    let text = std::fs::read_to_string(&path).map_err(|e| WorkspaceConfigError::Io {
        path: path.clone(),
        error: e.to_string(),
    })?;
    let json: serde_json::Value = serde_json::from_str(text.trim_start_matches('\u{feff}'))
        .map_err(|e| WorkspaceConfigError::Json {
            path: path.clone(),
            error: e.to_string(),
        })?;
    let Some(top) = json.as_object() else {
        return Err(WorkspaceConfigError::Json {
            path,
            error: "the top level is not a JSON object".to_string(),
        });
    };
    match top.get("mipmap_options") {
        None => Ok(None),
        Some(value) => parse_mipmap_options(value)
            .map(Some)
            .map_err(|(key, problem)| WorkspaceConfigError::MipmapOptions { path, key, problem }),
    }
}

/// The options in `value`, or the key at fault and what is wrong with it.
fn parse_mipmap_options(
    value: &serde_json::Value,
) -> Result<MipmapOptions, (Option<String>, String)> {
    let Some(object) = value.as_object() else {
        return Err((None, format!("is {value}, not a JSON object")));
    };
    if let Some(unknown) = object
        .keys()
        .find(|k| !KEYS.iter().any(|(name, _, _)| name == k))
    {
        return Err((
            Some(unknown.clone()),
            "is not a key of mipmap_options, whose keys are out_of_core_long_side, \
             jpeg_quality and full_resolution_jpeg_quality"
                .to_string(),
        ));
    }
    let mut values = [0u64; 3];
    for ((name, min, max), out) in KEYS.iter().zip(values.iter_mut()) {
        let Some(v) = object.get(*name) else {
            return Err((Some(name.to_string()), "is missing".to_string()));
        };
        let range = format!("an integer from {min} to {max}");
        let range = if *max == u32::MAX as u64 {
            format!("an integer of at least {min}")
        } else {
            range
        };
        // `as_u64` is `Some` only for a JSON integer written without a
        // fraction or exponent, so 95.0 is refused.
        match v.as_u64() {
            Some(n) if (*min..=*max).contains(&n) => *out = n,
            _ => {
                return Err((
                    Some(name.to_string()),
                    format!("is {v}; it must be {range}"),
                ))
            }
        }
    }
    Ok(MipmapOptions {
        out_of_core_long_side: values[0] as u32,
        jpeg_quality: values[1] as u8,
        full_resolution_jpeg_quality: values[2] as u8,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn read(file: &str) -> Result<Option<MipmapOptions>, WorkspaceConfigError> {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join(WORKSPACE_FILE_NAME), file).unwrap();
        read_mipmap_options(dir.path())
    }

    fn with_options(options: &str) -> Result<Option<MipmapOptions>, WorkspaceConfigError> {
        read(&format!(
            r#"{{"version": 2, "feature_tool": "sfmtool", "mipmap_options": {options}}}"#
        ))
    }

    #[test]
    fn a_file_without_the_object_has_no_options() {
        assert_eq!(read(r#"{"version": 2}"#).unwrap(), None);
    }

    #[test]
    fn a_valid_object_is_read() {
        let options = with_options(
            r#"{"out_of_core_long_side": 64, "jpeg_quality": 90,
                "full_resolution_jpeg_quality": 98}"#,
        )
        .unwrap()
        .unwrap();
        assert_eq!(
            options,
            MipmapOptions {
                out_of_core_long_side: 64,
                jpeg_quality: 90,
                full_resolution_jpeg_quality: 98
            }
        );
        assert_eq!(options.quality_for_level(0), 98);
        assert_eq!(options.quality_for_level(3), 90);
    }

    #[test]
    fn each_invalid_object_is_an_error_naming_the_file_the_key_and_the_value() {
        let good = [
            ("out_of_core_long_side", "64"),
            ("jpeg_quality", "95"),
            ("full_resolution_jpeg_quality", "95"),
        ];
        let object = |replace: &str, value: Option<&str>, extra: &str| {
            let members: Vec<String> = good
                .iter()
                .filter_map(|(k, v)| {
                    if *k == replace {
                        value.map(|value| format!("\"{k}\": {value}"))
                    } else {
                        Some(format!("\"{k}\": {v}"))
                    }
                })
                .chain((!extra.is_empty()).then(|| extra.to_string()))
                .collect();
            format!("{{{}}}", members.join(", "))
        };
        for (options, key, shown) in [
            (
                object("jpeg_quality", None, ""),
                "jpeg_quality",
                "is missing",
            ),
            (
                object("", None, r#""tile_size": 32"#),
                "tile_size",
                "is not a key",
            ),
            (
                object("jpeg_quality", Some("95.0"), ""),
                "jpeg_quality",
                "is 95.0",
            ),
            (
                object("full_resolution_jpeg_quality", Some("\"95\""), ""),
                "full_resolution_jpeg_quality",
                "is \"95\"",
            ),
            (
                object("jpeg_quality", Some("true"), ""),
                "jpeg_quality",
                "is true",
            ),
            (
                object("jpeg_quality", Some("0"), ""),
                "jpeg_quality",
                "is 0",
            ),
            (
                object("full_resolution_jpeg_quality", Some("101"), ""),
                "full_resolution_jpeg_quality",
                "is 101",
            ),
            (
                object("out_of_core_long_side", Some("0"), ""),
                "out_of_core_long_side",
                "is 0",
            ),
            (
                object("out_of_core_long_side", Some("-5"), ""),
                "out_of_core_long_side",
                "is -5",
            ),
        ] {
            let e = with_options(&options).unwrap_err();
            let message = e.to_string();
            assert!(
                matches!(&e, WorkspaceConfigError::MipmapOptions { key: Some(k), .. } if k == key),
                "{options}: {message}"
            );
            assert!(message.contains(WORKSPACE_FILE_NAME), "{message}");
            assert!(
                message.contains(&format!("mipmap_options.{key} {shown}")),
                "{options}: {message}"
            );
        }
        let e = with_options("[64, 95, 95]").unwrap_err();
        assert!(
            matches!(e, WorkspaceConfigError::MipmapOptions { key: None, .. }),
            "{e}"
        );
    }

    #[test]
    fn an_unreadable_or_unparsable_file_is_an_error_naming_it() {
        let e = read("{not json").unwrap_err();
        assert!(matches!(e, WorkspaceConfigError::Json { .. }), "{e}");
        assert!(e.to_string().contains(WORKSPACE_FILE_NAME), "{e}");
        let dir = tempfile::tempdir().unwrap();
        let e = read_mipmap_options(dir.path()).unwrap_err();
        assert!(matches!(e, WorkspaceConfigError::Io { .. }), "{e}");
    }
}
