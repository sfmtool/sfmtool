// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

use std::path::Path;

use sfmtool_archive_io::parse_hash_bytes;

use crate::progress::Progress;
use crate::{ObservationSource, SfmrReconstruction};

/// A `.sift` for `name` under `dir`, recorded as extracted from a photograph
/// hashing to `image_hash`, and the content hash it was written with.
fn write_sift_for(dir: &Path, name: &str, image_hash: &str) -> [u8; 16] {
    use ndarray::{Array2, Array3};
    use sfmtool_sift_format::*;
    let data = SiftData {
        feature_tool_metadata: FeatureToolMetadata {
            feature_tool: "sfmtool".into(),
            feature_type: "sift".into(),
            feature_options: serde_json::json!({}),
        },
        metadata: SiftMetadata {
            version: 1,
            image_name: name.into(),
            image_file_xxh128: image_hash.into(),
            image_file_size: 1,
            image_width: 64,
            image_height: 64,
            feature_count: 0,
            sharpness: None,
            gradient_structure_tensor_xx_yy_xy: None,
        },
        content_hash: SiftContentHash::default(),
        positions_xy: Array2::zeros((0, 2)),
        affine_shapes: Array3::zeros((0, 2, 2)),
        descriptors: Array2::zeros((0, 128)),
        thumbnail_y_x_rgb: Array3::zeros((THUMBNAIL_SIZE, THUMBNAIL_SIZE, 3)),
    };
    let path = dir.join(format!("{name}.sift"));
    write_sift(&path, &data, 3).unwrap();
    let (_, _, hash) = read_sift_metadata(&path).unwrap();
    parse_hash_bytes(&hash.content_xxh128).unwrap()
}

fn set_sift_content_hash(recon: &mut SfmrReconstruction, index: usize, hash: [u8; 16]) {
    if let ObservationSource::SiftFiles {
        sift_content_hashes,
        ..
    } = &mut recon.point_set.observations
    {
        sift_content_hashes[index] = hash;
    } else {
        panic!("the demo is a sift_files reconstruction");
    }
}

#[test]
fn an_embedded_patches_reconstruction_gives_its_stored_column() {
    let mut recon = SfmrReconstruction::demo(12);
    let n = recon.image_set.images.len();
    let hashes: Vec<[u8; 16]> = (0..n).map(|i| [i as u8 + 1; 16]).collect();
    recon.point_set.observations = ObservationSource::EmbeddedPatches {
        keypoints_xy: ndarray::Array2::zeros((recon.point_set.tracks.len(), 2)),
        image_file_hashes: hashes.clone(),
    };
    let found = recon.find_image_file_hashes(&Progress::none()).unwrap();
    assert_eq!(found, hashes.into_iter().map(Some).collect::<Vec<_>>());
}

#[test]
fn a_sift_files_reconstruction_reads_each_matching_sift_file() {
    let dir = tempfile::tempdir().unwrap();
    let mut recon = SfmrReconstruction::demo(12);
    recon.workspace_dir = dir.path().to_path_buf();
    let names: Vec<String> = recon
        .image_set
        .images
        .iter()
        .map(|image| image.name.clone())
        .collect();
    let first_hash = "00112233445566778899aabbccddeeff";
    let second_hash = "ffeeddccbbaa99887766554433221100";
    let first = write_sift_for(dir.path(), &names[0], first_hash);
    set_sift_content_hash(&mut recon, 0, first);
    // Image 1's .sift is another extraction than the one the reconstruction
    // records: its content hash is left at the demo's zeros.
    write_sift_for(dir.path(), &names[1], second_hash);
    // Image 2's .sift records a value that is not a hash.
    let third = write_sift_for(dir.path(), &names[2], "not a hash");
    set_sift_content_hash(&mut recon, 2, third);
    // Every other image has no .sift file.

    let found = recon.find_image_file_hashes(&Progress::none()).unwrap();
    assert_eq!(found.len(), names.len());
    assert_eq!(found[0], Some(parse_hash_bytes(first_hash).unwrap()));
    assert_eq!(found[1], None, "another extraction");
    assert_eq!(found[2], None, "not a hash");
    assert!(found[3..].iter().all(Option::is_none), "no file");
}

#[test]
fn a_reconstruction_with_no_workspace_knows_no_hash() {
    let recon = SfmrReconstruction::demo(12);
    let found = recon.find_image_file_hashes(&Progress::none()).unwrap();
    assert_eq!(found.len(), recon.image_set.images.len());
    assert!(found.iter().all(Option::is_none));
}

#[test]
fn reading_the_sift_files_can_be_cancelled() {
    let dir = tempfile::tempdir().unwrap();
    let mut recon = SfmrReconstruction::demo(12);
    recon.workspace_dir = dir.path().to_path_buf();
    let flag = std::sync::atomic::AtomicBool::new(true);
    let progress = Progress::none().cancelled_by(&flag);
    assert!(recon.find_image_file_hashes(&progress).is_err());
}
