// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! `images/names.json.zst` holds image names: the writer refuses a name that
//! is not one, and so do the reader and the verifier of a file some other
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
    let dir = std::env::temp_dir().join("sfmr_test_image_names_write");
    // A run that failed part-way leaves its files here.
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("names.sfmr");
    for &name in REFUSED {
        let mut data = make_test_data();
        data.image_names[1] = name.to_string();
        let error = write_sfmr(&path, &mut data).unwrap_err().to_string();
        assert!(names_image_1(&error, name), "{name:?}: {error}");
        assert!(!path.exists(), "{name:?} left a file behind");
    }

    let accepted = [
        "photos of the park/IMG 0001.jpg".to_string(),
        "Fotos/Straße/写真 ü.jpg".to_string(),
        "a/b/c/d/e/f/g/h/i/j/k/l/m/n/o/p/frame_0001.jpg".to_string(),
    ];
    let mut data = make_test_data();
    data.image_names = accepted.to_vec();
    write_sfmr(&path, &mut data).unwrap();
    assert_eq!(read_sfmr(&path).unwrap().image_names, accepted);
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn read_and_verify_refuse_each_form() {
    let dir = std::env::temp_dir().join("sfmr_test_image_names_read");
    // A run that failed part-way leaves its files here.
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let src = dir.join("valid.sfmr");
    write_sfmr(&src, &mut make_test_data()).unwrap();
    let path = dir.join("bad.sfmr");
    for &name in REFUSED {
        let names = ["frame_001.jpg", name, "frame_003.jpg"];
        rewrite_entries(&src, &path, |entry, _| {
            (entry == "images/names.json.zst").then(|| serde_json::to_vec(&names).unwrap())
        });
        let error = read_sfmr(&path).err().unwrap().to_string();
        assert!(names_image_1(&error, name), "{name:?}: {error}");
        let (valid, errors) = verify_sfmr(&path).unwrap();
        assert!(
            !valid && errors.iter().any(|e| names_image_1(e, name)),
            "{name:?}: {errors:?}"
        );
    }
    std::fs::remove_dir_all(&dir).unwrap();
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
/// refuses included, where the reader refuses the file; a file without the
/// entry is an error.
#[test]
fn the_unchecked_reader_returns_names_the_rule_refuses() {
    let dir = std::env::temp_dir().join("sfmr_test_image_names_unchecked");
    // A run that failed part-way leaves its files here.
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let src = dir.join("valid.sfmr");
    write_sfmr(&src, &mut make_test_data()).unwrap();
    let path = dir.join("bad.sfmr");
    for &name in REFUSED {
        let names = ["frame_001.jpg", name, "frame_003.jpg"];
        rewrite_entries(&src, &path, |entry, _| {
            (entry == "images/names.json.zst").then(|| serde_json::to_vec(&names).unwrap())
        });
        assert!(read_sfmr(&path).is_err(), "{name:?}");
        assert_eq!(read_sfmr_image_names_unchecked(&path).unwrap(), names);
    }
    copy_without_entry(&src, &path, "images/names.json.zst");
    assert!(read_sfmr_image_names_unchecked(&path).is_err());
    std::fs::remove_dir_all(&dir).unwrap();
}
