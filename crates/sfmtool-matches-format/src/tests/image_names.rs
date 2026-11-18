// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! `images/names.json.zst` holds image names: the writer refuses a name that
//! is not one, and so do the readers and the verifier of a file some other
//! writer produced.

use super::*;

/// One of each form the image-name rule refuses.
const REFUSED: &[&str] = &[
    "",
    "../x.jpg",
    "a/../b.jpg",
    "./a/b.jpg",
    "a/./b.jpg",
    "a//b.jpg",
    "a/b/",
    "a\\b.jpg",
    "/abs/x.jpg",
    "//server/share/x.jpg",
    "C:/x.jpg",
    "C:x.jpg",
    "a\0b.jpg",
];

/// The error must name the refused name and its index, image 1 of three.
fn names_image_1(error: &str, name: &str) -> bool {
    error.contains(&format!("image name 1 {name:?}"))
}

#[test]
fn write_refuses_each_form_and_keeps_accepted_names() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("names.matches");
    for &name in REFUSED {
        let mut data = make_test_data();
        data.image_names[1] = name.to_string();
        let error = write_matches(&path, &data, 3).unwrap_err().to_string();
        assert!(names_image_1(&error, name), "{name:?}: {error}");
        assert!(!path.exists(), "{name:?} left a file behind");
    }

    let accepted = vec![
        "photos of the park/IMG 0001.jpg".to_string(),
        "Fotos/Straße/写真 ü.jpg".to_string(),
        "a/b/c/d/e/f/g/h/i/j/k/l/m/n/o/p/frame_0001.jpg".to_string(),
    ];
    let mut data = make_test_data();
    data.image_names = accepted.clone();
    write_matches(&path, &data, 3).unwrap();
    assert_eq!(read_matches(&path).unwrap().image_names, accepted);
}

#[test]
fn readers_and_verify_refuse_each_form() {
    let dir = tempfile::tempdir().unwrap();
    let src = dir.path().join("valid.matches");
    write_matches(&src, &make_test_data(), 3).unwrap();
    let valid_entries = load_archive_entries(&src);
    let path = dir.path().join("bad.matches");
    for &name in REFUSED {
        let mut entries = valid_entries.clone();
        mutate_entry(&mut entries, "images/names.json.zst", |bytes| {
            *bytes = serde_json::to_vec(&["frames/frame_000.jpg", name, "c.jpg"]).unwrap();
        });
        // The hashes are recomputed, so the name is the only thing wrong.
        rebuild_matches_archive(&entries, &path);

        let error = read_matches(&path).err().unwrap().to_string();
        assert!(names_image_1(&error, name), "{name:?}: {error}");
        let error = read_matches_image_names(&path).unwrap_err().to_string();
        assert!(names_image_1(&error, name), "{name:?}: {error}");
        let (valid, errors) = verify_matches(&path).unwrap();
        assert!(!valid, "{name:?}");
        assert_eq!(errors.len(), 1, "{name:?}: {errors:?}");
        assert!(names_image_1(&errors[0], name), "{name:?}: {errors:?}");
    }
}

/// Copy the archive at `src` to `dst` without its entry `name`.
fn copy_without_entry(src: &std::path::Path, dst: &std::path::Path, name: &str) {
    let mut archive = zip::ZipArchive::new(std::fs::File::open(src).unwrap()).unwrap();
    let mut out = zip::ZipWriter::new(std::fs::File::create(dst).unwrap());
    for i in 0..archive.len() {
        let entry = archive.by_index_raw(i).unwrap();
        if entry.name().unwrap() != name {
            out.raw_copy_file(entry).unwrap();
        }
    }
    out.finish().unwrap();
}

/// The unchecked reader returns the names entry as it is, a name the rule
/// refuses included, where the checked reader refuses it; a file without the
/// entry is an error.
#[test]
fn the_unchecked_reader_returns_names_the_rule_refuses() {
    let dir = tempfile::tempdir().unwrap();
    let src = dir.path().join("valid.matches");
    write_matches(&src, &make_test_data(), 3).unwrap();
    let valid_entries = load_archive_entries(&src);
    let path = dir.path().join("bad.matches");
    for &name in REFUSED {
        let names = ["frames/frame_000.jpg", name, "c.jpg"];
        let mut entries = valid_entries.clone();
        mutate_entry(&mut entries, "images/names.json.zst", |bytes| {
            *bytes = serde_json::to_vec(&names).unwrap();
        });
        rebuild_matches_archive(&entries, &path);
        assert!(read_matches_image_names(&path).is_err(), "{name:?}");
        assert_eq!(read_matches_image_names_unchecked(&path).unwrap(), names);
    }
    copy_without_entry(&src, &path, "images/names.json.zst");
    assert!(read_matches_image_names_unchecked(&path).is_err());
}
