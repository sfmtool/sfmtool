// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The `mipmap_options` object of a workspace's `.sfm-workspace.json`, which
//! turns the out-of-core tier on and sets its threshold and the JPEG
//! qualities of its mip level files: [`read_mipmap_options`] reads it, and
//! [`enable_mipmap_options`] adds it to a file that has none.

use std::fmt;
use std::path::{Path, PathBuf};

use crate::camera::mipmap::{MipmapOptions, DEFAULT_OUT_OF_CORE_LONG_SIDE};

use super::{WORKSPACE_FILE_NAME, WORKSPACE_LAST_RELEASE_READING_VERSION_1, WORKSPACE_VERSION};

/// Why a workspace file's settings could not be read or written. Each names
/// the file.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum WorkspaceConfigError {
    /// The file could not be read or written.
    Io { path: PathBuf, error: String },
    /// The file is not valid JSON, or its top level is not an object.
    Json { path: PathBuf, error: String },
    /// The `mipmap_options` object is invalid, the one in the file or the one
    /// [`enable_mipmap_options`] was given to write into it: `key` is the key
    /// at fault (`None` when the value of `mipmap_options` itself is not an
    /// object), and `problem` says what is wrong, quoting the value.
    MipmapOptions {
        path: PathBuf,
        key: Option<String>,
        problem: String,
    },
    /// [`enable_mipmap_options`] was given a file that already has a
    /// `mipmap_options` object, which it leaves as it is.
    MipmapOptionsPresent { path: PathBuf },
    /// [`enable_mipmap_options`] was given a file whose `version` is not
    /// [`WORKSPACE_VERSION`]: `version` is the value the file holds, as JSON,
    /// or `None` when it has no `version`.
    Version {
        path: PathBuf,
        version: Option<String>,
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
            Self::MipmapOptionsPresent { path } => write!(
                f,
                "{} already has mipmap_options; edit the object in the file to change its values",
                path.display()
            ),
            Self::Version { path, version } => {
                match version {
                    Some(v) => write!(f, "{} is version {v}", path.display())?,
                    None => write!(f, "{} has no version", path.display())?,
                }
                write!(
                    f,
                    ", and mipmap_options is added only to a version {WORKSPACE_VERSION} \
                     workspace file"
                )?;
                if version.as_deref() == Some("1") {
                    write!(
                        f,
                        "; upgrade the workspace with sfmtool \
                         {WORKSPACE_LAST_RELEASE_READING_VERSION_1}'s `sfm ws upgrade` first"
                    )?;
                }
                Ok(())
            }
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
/// assert_eq!(object["out_of_core_long_side"], 1536);
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

/// Why a `mipmap_options` object is invalid: the key at fault, `None` when
/// the object is not a JSON object, and what is wrong, quoting the value.
/// It shows as `mipmap_options.<key> <problem>`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct InvalidMipmapOptions {
    pub key: Option<String>,
    pub problem: String,
}

impl fmt::Display for InvalidMipmapOptions {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.key {
            Some(key) => write!(f, "mipmap_options.{key} {}", self.problem),
            None => write!(f, "mipmap_options {}", self.problem),
        }
    }
}

impl std::error::Error for InvalidMipmapOptions {}

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
            .map_err(|e| invalid_options(path, e)),
    }
}

/// The error naming `path` for the invalid object `e` describes.
fn invalid_options(path: PathBuf, e: InvalidMipmapOptions) -> WorkspaceConfigError {
    WorkspaceConfigError::MipmapOptions {
        path,
        key: e.key,
        problem: e.problem,
    }
}

/// The options a `mipmap_options` object holds, checked as
/// [`read_mipmap_options`] checks the file's: exactly the three keys of
/// [`MipmapOptions`], each a JSON integer, `out_of_core_long_side` at least 1
/// and the two qualities 1 to 100.
///
/// ```
/// use sfmtool_core::workspace::parse_mipmap_options;
///
/// let object = serde_json::json!({
///     "out_of_core_long_side": 2048, "jpeg_quality": 90,
///     "full_resolution_jpeg_quality": 101,
/// });
/// let e = parse_mipmap_options(&object).unwrap_err();
/// assert_eq!(
///     e.to_string(),
///     "mipmap_options.full_resolution_jpeg_quality is 101; it must be an integer from 1 to 100"
/// );
/// ```
pub fn parse_mipmap_options(
    value: &serde_json::Value,
) -> Result<MipmapOptions, InvalidMipmapOptions> {
    let invalid = |key: Option<String>, problem: String| InvalidMipmapOptions { key, problem };
    let Some(object) = value.as_object() else {
        return Err(invalid(None, format!("is {value}, not a JSON object")));
    };
    if let Some(unknown) = object
        .keys()
        .find(|k| !KEYS.iter().any(|(name, _, _)| name == k))
    {
        return Err(invalid(
            Some(unknown.clone()),
            "is not a key of mipmap_options, whose keys are out_of_core_long_side, \
             jpeg_quality and full_resolution_jpeg_quality"
                .to_string(),
        ));
    }
    let mut values = [0u64; 3];
    for ((name, min, max), out) in KEYS.iter().zip(values.iter_mut()) {
        let Some(v) = object.get(*name) else {
            return Err(invalid(Some(name.to_string()), "is missing".to_string()));
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
                return Err(invalid(
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

/// Add `options` to `<root>/.sfm-workspace.json` as its `mipmap_options`
/// object, which turns the out-of-core tier on for the workspace; make no
/// mipmap.
///
/// The object is added as the last key of the file, after
/// `feature_prefix_dir` in every file sfmtool writes, and nothing else in the
/// file changes: its other keys, their order and their text stay as they
/// are. The object is written as [`mipmap_options_json`] gives it, with the
/// two-space indent and the line breaks of `json.dump(..., indent=2)`, `\r\n`
/// when the file has them, so a file `json.dump` wrote reads as the one it
/// would write with the object last. The file is replaced through a
/// temporary file beside it, so a failure leaves it as it was.
///
/// # Errors
///
/// Each names the file, which is left as it is:
/// [`MipmapOptions`](WorkspaceConfigError::MipmapOptions) for `options` that
/// [`parse_mipmap_options`] refuses (a threshold of 0, a quality of 0 or over
/// 100); [`Io`](WorkspaceConfigError::Io) for a file that cannot be read or
/// replaced; [`Json`](WorkspaceConfigError::Json) for one that is not a JSON
/// object; [`Version`](WorkspaceConfigError::Version) for one whose `version`
/// is not [`WORKSPACE_VERSION`]; and
/// [`MipmapOptionsPresent`](WorkspaceConfigError::MipmapOptionsPresent) for
/// one that already has the object, whatever it holds.
///
/// ```
/// use sfmtool_core::workspace::{
///     enable_mipmap_options, read_mipmap_options, NEW_WORKSPACE_MIPMAP_OPTIONS,
/// };
///
/// let root = tempfile::tempdir().unwrap();
/// std::fs::write(root.path().join(".sfm-workspace.json"), r#"{"version": 2}"#).unwrap();
/// enable_mipmap_options(root.path(), &NEW_WORKSPACE_MIPMAP_OPTIONS).unwrap();
/// assert_eq!(read_mipmap_options(root.path()).unwrap(), Some(NEW_WORKSPACE_MIPMAP_OPTIONS));
/// // A second call is refused, since the file now has the object.
/// assert!(enable_mipmap_options(root.path(), &NEW_WORKSPACE_MIPMAP_OPTIONS).is_err());
/// ```
pub fn enable_mipmap_options(
    root: &Path,
    options: &MipmapOptions,
) -> Result<(), WorkspaceConfigError> {
    use std::io::Write as _;

    let path = root.join(WORKSPACE_FILE_NAME);
    let object = mipmap_options_json(options);
    parse_mipmap_options(&object).map_err(|e| invalid_options(path.clone(), e))?;
    let io = |path: &Path, e: std::io::Error| WorkspaceConfigError::Io {
        path: path.to_path_buf(),
        error: e.to_string(),
    };
    let text = std::fs::read_to_string(&path).map_err(|e| io(&path, e))?;
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
    match top.get("version") {
        Some(v) if v.as_u64() == Some(u64::from(WORKSPACE_VERSION)) => {}
        version => {
            return Err(WorkspaceConfigError::Version {
                path,
                version: version.map(|v| v.to_string()),
            })
        }
    }
    if top.contains_key("mipmap_options") {
        return Err(WorkspaceConfigError::MipmapOptionsPresent { path });
    }

    // The text is a JSON object with nothing after it but white space, so
    // its last non-space character is the object's closing brace. The object
    // holds `version`, so the new key follows a comma.
    let newline = if text.contains("\r\n") { "\r\n" } else { "\n" };
    let body = text.trim_end();
    let after = &text[body.len()..];
    let Some(before) = body.strip_suffix('}') else {
        return Err(WorkspaceConfigError::Json {
            path,
            error: "the top level is not a JSON object".to_string(),
        });
    };
    let before = before.trim_end();
    let members: Vec<String> = KEYS
        .iter()
        .map(|(key, _, _)| format!("{newline}    \"{key}\": {}", object[*key]))
        .collect();
    let members = members.join(",");
    let new_text = format!(
        "{before},{newline}  \"mipmap_options\": {{{members}{newline}  }}{newline}}}{after}"
    );
    sfmtool_archive_io::write_atomically(&path, |file| file.write_all(new_text.as_bytes()))
        .map_err(|e| io(&path, e))
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

    /// Enable `options` in a workspace whose file is `text`, and return the
    /// result and the names of the files in the root afterwards.
    fn enable(
        text: &str,
        options: &MipmapOptions,
    ) -> (Result<String, WorkspaceConfigError>, Vec<String>) {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join(WORKSPACE_FILE_NAME);
        std::fs::write(&path, text).unwrap();
        let result = enable_mipmap_options(dir.path(), options)
            .map(|()| std::fs::read_to_string(&path).unwrap());
        let names = std::fs::read_dir(dir.path())
            .unwrap()
            .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
            .collect();
        (result, names)
    }

    #[test]
    fn enabling_a_new_workspace_s_file_without_the_object_writes_that_file() {
        // A new workspace's file is what `json.dump(..., indent=2)` writes,
        // with the object last; with the object cut out, enabling the new
        // workspace's options writes it back byte for byte, with either line
        // break.
        let marker = crate::workspace::sfmtool_workspace_marker(&Default::default());
        let cut = marker.find(",\n  \"mipmap_options\"").unwrap();
        let without = format!("{}\n}}", &marker[..cut]);
        assert!(read(&without).unwrap().is_none());
        for (without, marker) in [
            (without.clone(), marker.clone()),
            (without.replace('\n', "\r\n"), marker.replace('\n', "\r\n")),
        ] {
            let (written, names) = enable(&without, &NEW_WORKSPACE_MIPMAP_OPTIONS);
            assert_eq!(written.unwrap(), marker);
            assert_eq!(names, [WORKSPACE_FILE_NAME]);
        }
    }

    #[test]
    fn enabling_writes_the_options_given_after_every_other_key_unchanged() {
        let options = MipmapOptions {
            out_of_core_long_side: 2048,
            jpeg_quality: 80,
            full_resolution_jpeg_quality: 100,
        };
        // Keys out of sorted order and written on one line, and a line break
        // after the closing brace. Every file has a `version`, so none is
        // empty and the object always follows a comma.
        for text in [
            r#"{"version": 2, "zeta": [1, 2.50], "alpha": {"b": 1, "a": "é"}}"#,
            "{\"version\":2}\n",
        ] {
            let (written, _) = enable(text, &options);
            let written = written.unwrap();
            let body = text.trim_end();
            let kept = body[..body.len() - 1].trim_end();
            assert!(written.starts_with(kept), "{written}");
            assert!(written.ends_with(&text[body.len()..]), "{written:?}");
            let dir = tempfile::tempdir().unwrap();
            std::fs::write(dir.path().join(WORKSPACE_FILE_NAME), &written).unwrap();
            assert_eq!(read_mipmap_options(dir.path()).unwrap(), Some(options));
        }
    }

    #[test]
    fn a_file_with_the_object_is_refused_and_left_as_it_is() {
        for object in [
            r#"{"out_of_core_long_side": 64, "jpeg_quality": 90,
                "full_resolution_jpeg_quality": 98}"#,
            "null",
        ] {
            let text = format!(r#"{{"version": 2, "mipmap_options": {object}}}"#);
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join(WORKSPACE_FILE_NAME);
            std::fs::write(&path, &text).unwrap();
            let e = enable_mipmap_options(dir.path(), &NEW_WORKSPACE_MIPMAP_OPTIONS).unwrap_err();
            assert_eq!(
                e,
                WorkspaceConfigError::MipmapOptionsPresent { path: path.clone() }
            );
            assert!(e.to_string().contains("already has mipmap_options"), "{e}");
            assert_eq!(std::fs::read_to_string(&path).unwrap(), text);
        }
    }

    #[test]
    fn invalid_options_and_a_file_of_another_version_are_refused_and_it_is_left() {
        let options = |long_side, quality, full| MipmapOptions {
            out_of_core_long_side: long_side,
            jpeg_quality: quality,
            full_resolution_jpeg_quality: full,
        };
        let v2 = r#"{"version": 2}"#;
        for (text, options, shown) in [
            (
                v2,
                options(0, 95, 95),
                "mipmap_options.out_of_core_long_side is 0; it must be an integer of at least 1",
            ),
            (
                v2,
                options(64, 0, 95),
                "mipmap_options.jpeg_quality is 0; it must be an integer from 1 to 100",
            ),
            (
                v2,
                options(64, 95, 101),
                "mipmap_options.full_resolution_jpeg_quality is 101",
            ),
            (
                r#"{"version": 1}"#,
                NEW_WORKSPACE_MIPMAP_OPTIONS,
                "is version 1, and mipmap_options is added only to a version 2 workspace \
                 file; upgrade the workspace with sfmtool 0.4.1's",
            ),
            (
                r#"{"version": 3}"#,
                NEW_WORKSPACE_MIPMAP_OPTIONS,
                "is version 3, and",
            ),
            (
                r#"{"version": "2"}"#,
                NEW_WORKSPACE_MIPMAP_OPTIONS,
                "is version \"2\", and",
            ),
            ("{}", NEW_WORKSPACE_MIPMAP_OPTIONS, "has no version, and"),
            ("[2]", NEW_WORKSPACE_MIPMAP_OPTIONS, "not a JSON object"),
            (
                "{not json",
                NEW_WORKSPACE_MIPMAP_OPTIONS,
                "key must be a string",
            ),
        ] {
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join(WORKSPACE_FILE_NAME);
            std::fs::write(&path, text).unwrap();
            let message = enable_mipmap_options(dir.path(), &options)
                .unwrap_err()
                .to_string();
            assert!(message.contains(WORKSPACE_FILE_NAME), "{message}");
            assert!(message.contains(shown), "{text}: {message}");
            assert_eq!(std::fs::read_to_string(&path).unwrap(), text);
        }
        let dir = tempfile::tempdir().unwrap();
        let e = enable_mipmap_options(dir.path(), &NEW_WORKSPACE_MIPMAP_OPTIONS).unwrap_err();
        assert!(matches!(e, WorkspaceConfigError::Io { .. }), "{e}");
        assert_eq!(std::fs::read_dir(dir.path()).unwrap().count(), 0);
    }
}
