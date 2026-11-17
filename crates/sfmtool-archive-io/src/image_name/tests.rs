// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The image-name rule: the names it accepts, and each form it refuses with
//! the reason it gives.

use super::{validate_image_name, validate_image_names};

#[test]
fn accepts_relative_slash_paths() {
    for name in [
        "a.jpg",
        "frames/DSC_0001.JPG",
        "a/b/c/d/e/f/g/h/i/j/k/l/m/n/o/p/frame_0001.jpg",
        "photos of the park/IMG 0001.jpg",
        "Fotos/Straße/bild_ü.jpg",
        "写真/画像.png",
        "..hidden/.x.jpg",
        "a/.../b.jpg",
        "a/x./b.jpg",
        "dir:with:colons/12:30:00.jpg",
        "ab:c/x.jpg",
        "1:/x.jpg",
    ] {
        assert_eq!(validate_image_name(name), Ok(()), "{name:?}");
    }
}

/// Every refused form, each with a fragment of the reason it must give, so a
/// name refused for the wrong reason fails as well as one let through.
#[test]
fn refuses_each_form_with_its_reason() {
    for (name, reason) in [
        ("", "it is empty"),
        ("../x.jpg", "'..' component"),
        ("a/../../x.jpg", "'..' component"),
        ("a/..", "'..' component"),
        ("./a/b.jpg", "'.' component"),
        ("a/./b.jpg", "'.' component"),
        (".", "'.' component"),
        ("a//b.jpg", "empty component"),
        ("a/b/", "empty component"),
        ("a\\b.jpg", "holds a '\\'"),
        ("\\\\server\\share\\x.jpg", "holds a '\\'"),
        ("/abs/x.jpg", "starts with '/'"),
        ("//server/share/x.jpg", "starts with '/'"),
        ("C:/x.jpg", "drive prefix 'C:'"),
        ("C:", "drive prefix 'C:'"),
        ("d:x.jpg", "drive prefix 'd:'"),
        ("a\0b.jpg", "control character U+0000"),
        ("a\nb.jpg", "control character U+000A"),
        ("a\u{7f}.jpg", "control character U+007F"),
        ("a\u{85}.jpg", "control character U+0085"),
    ] {
        let error = validate_image_name(name).expect_err(name);
        assert!(error.contains(reason), "{name:?}: {error}");
    }
}

#[test]
fn names_the_first_refused_name_and_its_index() {
    assert_eq!(validate_image_names::<&str>(&[]), Ok(()));
    assert_eq!(validate_image_names(&["a.jpg", "b/c.jpg"]), Ok(()));
    let error = validate_image_names(&[
        "a.jpg".to_string(),
        "b/c.jpg".to_string(),
        "../d.jpg".to_string(),
        "/e.jpg".to_string(),
    ])
    .unwrap_err();
    assert_eq!(
        error,
        "image name 2 \"../d.jpg\" is not a path relative to the workspace directory: it has a \
         '..' component"
    );
}
