use super::*;

#[test]
fn test_observations_for_point() {
    let recon = SfmrReconstruction::demo(1000);
    // Demo creates 1000 points, each observed by 2 cameras
    assert_eq!(recon.observation_offsets.len(), recon.points.len() + 1);
    assert_eq!(
        *recon.observation_offsets.last().unwrap(),
        recon.tracks.len()
    );

    for i in 0..recon.points.len() {
        let obs = recon.observations_for_point(i);
        assert_eq!(obs.len(), 2, "point {i} should have 2 observations");
        for o in obs {
            assert_eq!(o.point_index, i as u32);
        }
    }
}

#[test]
fn test_track_image_indices() {
    let recon = SfmrReconstruction::demo(1000);
    // Point 0 is observed by cameras 0 and 1 in the demo
    let images = recon.track_image_indices(0);
    assert_eq!(images.len(), 2);
}

#[test]
fn test_subset_keep_all_images_is_identity() {
    let recon = SfmrReconstruction::demo(1000);
    let indices: Vec<u32> = (0..recon.images.len() as u32).collect();
    let subset = recon.subset_by_image_indices(&indices, false).unwrap();
    assert_eq!(subset.images.len(), recon.images.len());
    assert_eq!(subset.points.len(), recon.points.len());
    assert_eq!(subset.tracks.len(), recon.tracks.len());
    assert_eq!(subset.observation_counts, recon.observation_counts);
}

#[test]
fn test_subset_keeps_all_points_by_default() {
    let recon = SfmrReconstruction::demo(1000);
    // Keep only image 0. In the demo, point i is observed by images
    // (i % 8) and ((i + 1) % 8), so ~2 points out of every 8 touch image 0.
    let subset = recon.subset_by_image_indices(&[0], false).unwrap();

    assert_eq!(subset.images.len(), 1);
    // Default: all points kept even if their track dropped to zero.
    assert_eq!(subset.points.len(), recon.points.len());
    assert_eq!(subset.observation_counts.len(), recon.points.len());

    // Observations that survived are the ones referencing image 0.
    let expected_surviving: usize = recon.tracks.iter().filter(|t| t.image_index == 0).count();
    assert_eq!(subset.tracks.len(), expected_surviving);
    // Every surviving track now references the new image index 0.
    for obs in &subset.tracks {
        assert_eq!(obs.image_index, 0);
    }
    // Per-point observation_counts sum to the surviving track count.
    assert_eq!(
        subset.observation_counts.iter().sum::<u32>() as usize,
        expected_surviving
    );
    // And some points have zero observations.
    assert!(subset.observation_counts.contains(&0));
}

#[test]
fn test_subset_drops_orphaned_points_when_requested() {
    let recon = SfmrReconstruction::demo(1000);
    let subset = recon.subset_by_image_indices(&[0], true).unwrap();

    assert_eq!(subset.images.len(), 1);
    // All surviving points have at least one observation.
    assert!(subset.observation_counts.iter().all(|&c| c > 0));
    assert_eq!(
        subset.points.len(),
        subset.observation_counts.iter().filter(|&&c| c > 0).count()
    );
    // Point IDs in tracks are contiguous.
    let max_pt = subset
        .tracks
        .iter()
        .map(|t| t.point_index)
        .max()
        .unwrap_or(0);
    assert!((max_pt as usize) < subset.points.len());
    // Observation offsets round-trip.
    assert_eq!(
        *subset.observation_offsets.last().unwrap(),
        subset.tracks.len()
    );
}

#[test]
fn test_subset_rejects_out_of_bounds_and_duplicates() {
    let recon = SfmrReconstruction::demo(1000);
    let n = recon.images.len() as u32;
    assert!(recon.subset_by_image_indices(&[n], false).is_err());
    assert!(recon.subset_by_image_indices(&[0, 0], false).is_err());
}

#[test]
fn test_subset_filters_rig_frame_data() {
    use ndarray::{Array1, Array2};
    use sfmr_format::{FramesMetadata, RigDefinition, RigsMetadata};

    // Start from the demo (8 images) and attach a trivial rig/frame
    // structure: one single-sensor rig, one frame per image.
    let mut recon = SfmrReconstruction::demo(1000);
    let n_images = recon.images.len();
    let rig_def = RigDefinition {
        name: "rig0".to_string(),
        sensor_count: 1,
        sensor_offset: 0,
        ref_sensor_name: "sensor0".to_string(),
        sensor_names: vec!["sensor0".to_string()],
    };
    recon.rig_frame_data = Some(RigFrameData {
        rigs_metadata: RigsMetadata {
            rig_count: 1,
            sensor_count: 1,
            rigs: vec![rig_def],
        },
        sensor_camera_indexes: Array1::from_vec(vec![0u32]),
        sensor_quaternions_wxyz: Array2::from_shape_vec((1, 4), vec![1.0, 0.0, 0.0, 0.0]).unwrap(),
        sensor_translations_xyz: Array2::from_shape_vec((1, 3), vec![0.0, 0.0, 0.0]).unwrap(),
        frames_metadata: FramesMetadata {
            frame_count: n_images as u32,
        },
        rig_indexes: Array1::from_vec(vec![0u32; n_images]),
        image_sensor_indexes: Array1::from_vec(vec![0u32; n_images]),
        image_frame_indexes: Array1::from_vec((0..n_images as u32).collect()),
    });

    // Keep images 0, 3, 5 — three frames survive and must be remapped to 0,1,2.
    let subset = recon.subset_by_image_indices(&[0, 3, 5], false).unwrap();
    let rf = subset.rig_frame_data.as_ref().unwrap();
    assert_eq!(rf.frames_metadata.frame_count, 3);
    assert_eq!(rf.rig_indexes.len(), 3);
    assert_eq!(rf.image_frame_indexes.to_vec(), vec![0, 1, 2]);
    // Sensor definitions are unchanged.
    assert_eq!(rf.rigs_metadata.rig_count, 1);
    assert_eq!(rf.sensor_camera_indexes.to_vec(), vec![0]);
}

#[test]
fn test_from_sfmr_data_rejects_embedded_patches() {
    // SfmrReconstruction does not yet model patch observations, so loading an
    // embedded_patches file must fail with a clear error rather than half-load.
    let mut data = SfmrReconstruction::demo(10).to_sfmr_data();
    data.metadata.feature_source = FEATURE_SOURCE_EMBEDDED_PATCHES.to_string();
    let err = SfmrReconstruction::from_sfmr_data(data).err().unwrap();
    assert!(
        format!("{err}").contains("embedded_patches"),
        "unexpected error: {err}"
    );
}

#[test]
fn test_to_sfmr_data_is_sift_files() {
    // A reconstruction round-trips as a sift_files v4 file.
    let data = SfmrReconstruction::demo(10).to_sfmr_data();
    assert_eq!(data.metadata.feature_source, FEATURE_SOURCE_SIFT_FILES);
    assert!(data.feature_indexes.is_some());
    assert!(data.keypoints_xy.is_none());
    assert!(data.image_file_hashes.is_none());
}
