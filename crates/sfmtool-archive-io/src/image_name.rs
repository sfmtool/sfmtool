// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The rule an image name obeys.
//!
//! `.sfmr`, `.matches` and `.kdf` each store the images they cover as a list of
//! paths relative to the workspace directory, and a reader joins each one to
//! the workspace it resolves. The rule is here once so that every format
//! refuses the same names, on write and on read, and a name one format accepts
//! is one the others accept.

/// Check that `name` is an image name: a path relative to the workspace
/// directory that names a file inside it, written the same way on every
/// platform.
///
/// An image name is not empty, and:
///
/// - it separates its components with `/`, and holds no `\`;
/// - it does not start with `/`, so it is not absolute and has no UNC prefix
///   (`//server/share`);
/// - it does not start with an ASCII letter followed by `:`, the Windows drive
///   prefix (`C:`, `C:/x.jpg`, `C:x.jpg`), which makes a path absolute or
///   relative to another drive's current directory;
/// - every component is non-empty and is neither `.` nor `..`, so it has no
///   leading, trailing or doubled `/` and cannot climb out of the workspace;
/// - it holds no control character (Unicode category `Cc`: U+0000 to U+001F
///   and U+007F to U+009F). An operating system ends a path at U+0000, and
///   Windows refuses U+0001 to U+001F in a file name, so a name holding one
///   would not name the file it spells on every platform.
///
/// Nothing else is refused. A colon after the first component's second
/// character, a space or a non-ASCII character is part of a valid file name on
/// at least one platform, and stays inside the workspace on all of them. The
/// check does not change a name: a name that fails it is an error, not a name
/// to be rewritten.
///
/// Returns the reason a name is refused; [`validate_image_names`] adds the
/// name and its index.
///
/// # Example
///
/// ```
/// use sfmtool_archive_io::validate_image_name;
/// assert!(validate_image_name("frames/DSC_0001.JPG").is_ok());
/// assert!(validate_image_name("../outside.jpg").is_err());
/// ```
pub fn validate_image_name(name: &str) -> Result<(), String> {
    if name.is_empty() {
        return Err("it is empty".to_string());
    }
    if let Some(c) = name.chars().find(|c| c.is_control()) {
        return Err(format!(
            "it holds the control character U+{:04X}",
            u32::from(c)
        ));
    }
    if name.contains('\\') {
        return Err("it holds a '\\'; an image name separates its components with '/'".into());
    }
    if name.starts_with('/') {
        return Err("it starts with '/', so it is not relative to the workspace".into());
    }
    let bytes = name.as_bytes();
    if bytes.len() >= 2 && bytes[0].is_ascii_alphabetic() && bytes[1] == b':' {
        return Err(format!(
            "it starts with the Windows drive prefix '{}', so it is not relative to the \
             workspace",
            &name[..2]
        ));
    }
    for component in name.split('/') {
        match component {
            "" => return Err("it has an empty component (a trailing or doubled '/')".to_string()),
            "." | ".." => return Err(format!("it has a '{component}' component")),
            _ => {}
        }
    }
    Ok(())
}

/// Check every name in `names` with [`validate_image_name`], stopping at the
/// first that fails, with an error that names its index and the name.
///
/// # Example
///
/// ```
/// use sfmtool_archive_io::validate_image_names;
/// let error = validate_image_names(&["a.jpg", "b//c.jpg"]).unwrap_err();
/// assert!(error.starts_with("image name 1 \"b//c.jpg\""));
/// ```
pub fn validate_image_names<S: AsRef<str>>(names: &[S]) -> Result<(), String> {
    for (index, name) in names.iter().enumerate() {
        let name = name.as_ref();
        validate_image_name(name).map_err(|reason| {
            format!(
                "image name {index} {name:?} is not a path relative to the workspace \
                 directory: {reason}"
            )
        })?;
    }
    Ok(())
}

#[cfg(test)]
mod tests;
