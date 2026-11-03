// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

use std::path::Path;

use ndarray::{Array2, Array3};
use sfmtool_archive_io::parse_hash_bytes;
use sfmtool_sift_format::*;

use super::*;
use crate::{ObservationSource, SfmrReconstruction};

/// The BT.601 conversion rounded to 8 bits, which differs from the demo's.
fn bt601() -> ImageToGray {
    ImageToGray::new("0.299*R + 0.587*G + 0.114*B", Some(8)).unwrap()
}

/// A `w` x `h` RGB photograph of a pattern that differs per `seed`.
fn photograph(w: u32, h: u32, seed: u32) -> ImageU8 {
    let data = (0..w * h)
        .flat_map(|i| {
            let (x, y) = (i % w, i / w);
            let v = ((x * 7 + y * 13 + seed * 31) % 256) as u8;
            [v, v.wrapping_mul(3), 255 - v]
        })
        .collect();
    ImageU8::new(w, h, 3, data)
}

/// Write `image` as a PNG at `workspace / name` and return its XXH128.
fn write_photograph(workspace: &Path, name: &str, image: &ImageU8) -> [u8; 16] {
    let path = workspace.join(name);
    image.write(&path, 95).unwrap();
    let (hash, _) = crate::features::sift::xxh128_of_file(&path).unwrap();
    parse_hash_bytes(&hash).unwrap()
}

/// Write the `.sift` of `name` under `workspace`: version 2 with `readings`
/// computed through `image_to_gray`, or version 1 when `readings` is `None`,
/// recorded as extracted from a photograph hashing to `image_hash`. Returns
/// the file's content hash.
fn write_sift_for(
    workspace: &Path,
    name: &str,
    image_hash: [u8; 16],
    readings: Option<(&ImageToGray, ImageReadings)>,
) -> [u8; 16] {
    let mut feature_options = serde_json::json!({});
    if let Some((conversion, _)) = readings {
        feature_options["image_to_gray"] = conversion.to_json();
    }
    let data = SiftData {
        feature_tool_metadata: FeatureToolMetadata {
            feature_tool: "sfmtool".into(),
            feature_type: "sift".into(),
            feature_options,
        },
        metadata: SiftMetadata {
            version: if readings.is_some() { 2 } else { 1 },
            image_name: name.into(),
            image_file_xxh128: sfmtool_archive_io::format_hash(u128::from_be_bytes(image_hash)),
            image_file_size: 1,
            image_width: 64,
            image_height: 48,
            feature_count: 0,
            sharpness: readings.map(|(_, r)| f64::from(r.sharpness)),
            gradient_structure_tensor_xx_yy_xy: readings
                .map(|(_, r)| r.gradient_structure_tensor_xx_yy_xy.map(f64::from)),
        },
        content_hash: SiftContentHash::default(),
        positions_xy: Array2::zeros((0, 2)),
        affine_shapes: Array3::zeros((0, 2, 2)),
        descriptors: Array2::zeros((0, 128)),
        thumbnail_y_x_rgb: Array3::zeros((THUMBNAIL_SIZE, THUMBNAIL_SIZE, 3)),
    };
    let path = workspace.join(format!("{name}.sift"));
    write_sift(&path, &data, 3).unwrap();
    let (_, _, hash) = read_sift_metadata(&path).unwrap();
    parse_hash_bytes(&hash.content_xxh128).unwrap()
}

/// The demo reconstruction in `workspace` with no image readings at all, as
/// one read from a file below version 15 is.
fn without_readings(workspace: &Path) -> SfmrReconstruction {
    let mut recon = SfmrReconstruction::demo(12);
    recon.workspace_dir = workspace.to_path_buf();
    recon.image_set.image_to_gray = None;
    for image in &mut recon.image_set.images {
        image.readings = None;
    }
    recon
}

fn set_sift_content_hash(recon: &mut SfmrReconstruction, index: usize, hash: [u8; 16]) {
    if let ObservationSource::SiftFiles {
        sift_content_hashes,
        ..
    } = &mut recon.point_set.observations
    {
        sift_content_hashes[index] = hash;
    }
}

fn names(recon: &SfmrReconstruction) -> Vec<String> {
    recon
        .image_set
        .images
        .iter()
        .map(|i| i.name.clone())
        .collect()
}

#[test]
fn the_reconstructions_own_rows_come_first() {
    let dir = tempfile::tempdir().unwrap();
    let mut recon = SfmrReconstruction::demo(12);
    recon.workspace_dir = dir.path().to_path_buf();
    let gathered = recon.gather_image_readings(&Progress::none()).unwrap();
    assert_eq!(gathered.count(ImageReadingSource::Reconstruction), 8);
    assert_eq!(gathered.image_to_gray, ImageToGray::default());
    for (row, image) in gathered.rows.iter().zip(&recon.image_set.images) {
        assert_eq!(Some(*row), image.readings);
    }
}

#[test]
fn each_source_in_order_and_a_conversion_mismatch_recomputes() {
    let dir = tempfile::tempdir().unwrap();
    let mut recon = without_readings(dir.path());
    let conversion = bt601();
    recon.metadata.workspace.contents.feature_options =
        serde_json::json!({"image_to_gray": conversion.to_json()});
    let names = names(&recon);
    let images: Vec<ImageU8> = (0..names.len() as u32)
        .map(|i| photograph(64, 48, i))
        .collect();
    let computed = |i: usize, through: &ImageToGray| {
        let image = ImageU8::read_rgb(&dir.path().join(&names[i])).unwrap();
        image_readings_from_rgb(image.width(), image.height(), image.data(), through).unwrap()
    };

    // Image 0 already holds a row; its .sift would say otherwise.
    let own = ImageReadings {
        sharpness: 0.125,
        gradient_structure_tensor_xx_yy_xy: [0.5, 0.25, -0.125],
    };
    recon.image_set.images[0].readings = Some(own);
    recon.image_set.image_to_gray = Some(conversion.clone());
    for (i, name) in names.iter().enumerate() {
        let image_hash = write_photograph(dir.path(), name, &images[i]);
        let stored = ImageReadings {
            sharpness: 0.01 * (i + 1) as f32,
            gradient_structure_tensor_xx_yy_xy: [0.003, 0.002, -0.0001 * i as f32],
        };
        // Images 1 and 2 have a version 2 .sift under the file's conversion,
        // image 3 one under another conversion, image 4 a version 1 .sift,
        // and the rest one under the file's conversion too.
        let hash = match i {
            3 => write_sift_for(
                dir.path(),
                name,
                image_hash,
                Some((&ImageToGray::default(), stored)),
            ),
            4 => write_sift_for(dir.path(), name, image_hash, None),
            _ => write_sift_for(dir.path(), name, image_hash, Some((&conversion, stored))),
        };
        set_sift_content_hash(&mut recon, i, hash);
    }

    let gathered = recon.gather_image_readings(&Progress::none()).unwrap();
    assert_eq!(gathered.image_to_gray, conversion);
    assert_eq!(gathered.sources[0], ImageReadingSource::Reconstruction);
    assert_eq!(gathered.rows[0], own);
    assert_eq!(gathered.sources[1], ImageReadingSource::Sift);
    assert_eq!(gathered.rows[1].sharpness, 0.02);
    assert_eq!(
        gathered.rows[2].gradient_structure_tensor_xx_yy_xy[2],
        -0.0002
    );
    // The .sift under another conversion is not copied: the row is computed
    // again from the photograph through the file's conversion.
    assert_eq!(gathered.sources[3], ImageReadingSource::Photograph);
    assert_eq!(gathered.rows[3], computed(3, &conversion));
    assert_ne!(gathered.rows[3].sharpness, 0.04);
    // A version 1 .sift records no readings but does record the photograph.
    assert_eq!(gathered.sources[4], ImageReadingSource::Photograph);
    assert_eq!(gathered.rows[4], computed(4, &conversion));

    // A row the value holds under no recorded conversion is not used: the
    // file's conversion is then the workspace's, and image 0's .sift has it.
    recon.image_set.image_to_gray = None;
    let gathered = recon.gather_image_readings(&Progress::none()).unwrap();
    assert_eq!(gathered.sources[0], ImageReadingSource::Sift);
}

#[test]
fn an_image_with_no_source_fails_the_gather_naming_it() {
    let dir = tempfile::tempdir().unwrap();
    let mut recon = without_readings(dir.path());
    let names = names(&recon);
    // Image 0 is complete: photograph and a version 1 .sift that records it.
    let hash = write_photograph(dir.path(), &names[0], &photograph(64, 48, 0));
    let sift_hash = write_sift_for(dir.path(), &names[0], hash, None);
    set_sift_content_hash(&mut recon, 0, sift_hash);
    // Image 1's photograph has changed since its .sift was written.
    write_photograph(dir.path(), &names[1], &photograph(64, 48, 1));
    let sift_hash = write_sift_for(dir.path(), &names[1], [9; 16], None);
    set_sift_content_hash(&mut recon, 1, sift_hash);
    // Image 2's .sift is another extraction; the others have nothing.
    write_sift_for(dir.path(), &names[2], [9; 16], None);

    let Err(ImageReadingsError::Unavailable { images, .. }) =
        recon.gather_image_readings(&Progress::none())
    else {
        panic!("the gather succeeded without a source for every image");
    };
    let named: Vec<&str> = images.iter().map(|i| i.name.as_str()).collect();
    assert_eq!(named, &names[1..]);
    assert!(images[0].photograph.contains("do not hash"), "{images:?}");
    assert!(images[0].sift.contains("version 1"), "{images:?}");
    assert!(images[1].sift.contains("another extraction"), "{images:?}");
    assert!(images[2].sift.contains("no such file"), "{images:?}");
    assert!(images[2].photograph.contains("no such file"), "{images:?}");
    assert!(
        images[2].sift.contains(&format!("{}.sift", names[3])),
        "{images:?}"
    );

    // A save of the value fails the same way and writes nothing.
    let out = dir.path().join("out.sfmr");
    let err = recon.save(&out).unwrap_err().to_string();
    for name in &names[1..] {
        assert!(err.contains(name.as_str()), "{err}");
    }
    assert!(!out.exists());
}

#[test]
fn an_embedded_patches_value_without_sift_files_reads_its_photographs() {
    let dir = tempfile::tempdir().unwrap();
    let mut recon = without_readings(dir.path());
    let names = names(&recon);
    let hashes: Vec<[u8; 16]> = names
        .iter()
        .enumerate()
        .map(|(i, name)| write_photograph(dir.path(), name, &photograph(40, 30, i as u32)))
        .collect();
    recon.point_set.observations = ObservationSource::EmbeddedPatches {
        keypoints_xy: Array2::zeros((recon.point_set.tracks.len(), 2)),
        image_file_hashes: hashes,
    };
    let gathered = recon.fill_image_readings(&Progress::none()).unwrap();
    // No .sift and no recorded conversion: BT.709 unrounded.
    assert_eq!(gathered.image_to_gray, ImageToGray::default());
    assert_eq!(gathered.count(ImageReadingSource::Photograph), names.len());
    assert!(recon.image_set.has_image_readings());
    assert_eq!(recon.image_set.images[5].readings, Some(gathered.rows[5]));
}

#[test]
fn the_conversion_for_a_save_follows_the_order_the_format_gives() {
    let dir = tempfile::tempdir().unwrap();
    let mut recon = without_readings(dir.path());
    // Nothing recorded and a colmap record: its version 1 conversion.
    recon.metadata.workspace.contents.feature_tool = "colmap".into();
    recon.metadata.workspace.contents.feature_options = serde_json::json!({});
    assert_eq!(
        recon.image_to_gray_for_save(),
        ImageToGray::new("0.2126*R + 0.7152*G + 0.0722*B", Some(8)).unwrap()
    );
    // A version 2 .sift of a later image comes before the tool's conversion.
    let names = names(&recon);
    write_sift_for(
        dir.path(),
        &names[3],
        [0; 16],
        Some((&bt601(), ImageReadings::ZERO)),
    );
    assert_eq!(recon.image_to_gray_for_save(), bt601());
    // The workspace record comes before the .sift files.
    recon.metadata.workspace.contents.feature_options =
        serde_json::json!({"image_to_gray": {"formula": "G"}});
    assert_eq!(recon.image_to_gray_for_save().formula(), "G");
    // And the value's own conversion before everything.
    recon.image_set.image_to_gray = Some(ImageToGray::default());
    assert_eq!(recon.image_to_gray_for_save(), ImageToGray::default());
    // An unknown tool with nothing else falls back to BT.709 unrounded.
    let mut bare = without_readings(Path::new(""));
    bare.metadata.workspace.contents.feature_tool = "other".into();
    bare.metadata.workspace.contents.feature_options = serde_json::json!({});
    assert_eq!(bare.image_to_gray_for_save(), ImageToGray::default());
}

#[test]
fn a_save_writes_the_gathered_rows_and_leaves_the_value_unchanged() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join(".sfm-workspace.json"), "{}").unwrap();
    let mut recon = without_readings(dir.path());
    for (i, name) in names(&recon).iter().enumerate() {
        let stored = ImageReadings {
            sharpness: 0.5 + i as f32,
            gradient_structure_tensor_xx_yy_xy: [0.25, 0.125, 0.0],
        };
        let hash = write_sift_for(
            dir.path(),
            name,
            [0; 16],
            Some((&ImageToGray::default(), stored)),
        );
        set_sift_content_hash(&mut recon, i, hash);
    }
    let out = dir.path().join("out.sfmr");
    recon.save(&out).unwrap();
    assert!(!recon.image_set.has_image_readings(), "the save keeps none");
    let written = sfmtool_sfmr_format::read_sfmr(&out)
        .unwrap()
        .image_readings
        .unwrap();
    assert_eq!(
        written.sharpness.to_vec(),
        [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5]
    );
    assert_eq!(written.image_to_gray, ImageToGray::default().to_json());
}
