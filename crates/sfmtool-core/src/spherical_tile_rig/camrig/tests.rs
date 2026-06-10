use super::*;
use crate::sphere_points::RelaxConfig;
use crate::spherical_tile_rig::SphericalTileRigParams;

fn sample_rig(n: usize) -> SphericalTileRig {
    SphericalTileRig::new(&SphericalTileRigParams {
        centre: [1.0, -2.0, 0.5],
        n,
        arc_per_pixel: 2.0 * std::f64::consts::PI / 512.0,
        overlap_factor: 1.2,
        atlas_cols: None,
        relax: Some(RelaxConfig {
            seed: Some(7),
            ..Default::default()
        }),
    })
    .unwrap()
}

fn assert_rigs_match(a: &SphericalTileRig, b: &SphericalTileRig) {
    assert_eq!(a.len(), b.len());
    assert_eq!(a.patch_size(), b.patch_size());
    assert_eq!(a.atlas_cols(), b.atlas_cols());
    for k in 0..3 {
        assert!((a.centre()[k] - b.centre()[k]).abs() < 1e-12, "centre");
    }
    assert!((a.half_fov_rad() - b.half_fov_rad()).abs() < 1e-12);
    assert!((a.measured_max_nn_angle() - b.measured_max_nn_angle()).abs() < 1e-12);
    assert!((a.measured_max_coverage_angle() - b.measured_max_coverage_angle()).abs() < 1e-12);
    for t in 0..a.len() {
        let (da, db) = (a.direction(t), b.direction(t));
        let (ra, ua) = a.basis(t);
        let (rb, ub) = b.basis(t);
        for k in 0..3 {
            assert!((da[k] - db[k]).abs() < 1e-9, "tile {t} direction");
            assert!((ra[k] - rb[k]).abs() < 1e-9, "tile {t} e_right");
            assert!((ua[k] - ub[k]).abs() < 1e-9, "tile {t} e_up");
        }
    }
}

#[test]
fn camrig_struct_round_trip() {
    let rig = sample_rig(320);
    let data = rig.to_camrig("test_rig");
    data.validate()
        .expect("produced .camrig data must be valid");
    assert_eq!(data.metadata.rig_type, "spherical_tiles");
    assert_eq!(data.metadata.camera_count, 1);
    assert_eq!(data.cameras.len(), 1);
    assert!(data.translations_xyz.iter().all(|&t| t == 0.0));
    assert!(data.sensor_image_patterns.is_empty());

    let back = SphericalTileRig::from_camrig(&data).unwrap();
    assert_rigs_match(&rig, &back);
}

#[test]
fn camrig_file_round_trip() {
    let rig = sample_rig(128);
    let dir = std::env::temp_dir().join("camrig_conv_file_test");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("sph.camrig");

    rig.write_camrig(&path, "sph").unwrap();
    let back = SphericalTileRig::read_camrig(&path).unwrap();
    assert_rigs_match(&rig, &back);

    std::fs::remove_dir_all(&dir).ok();
}

fn expect_conversion_error(data: &CamRigData) -> CamRigConversionError {
    match SphericalTileRig::from_camrig(data) {
        Ok(_) => panic!("expected from_camrig to fail"),
        Err(e) => e,
    }
}

#[test]
fn from_camrig_rejects_wrong_rig_type() {
    let mut data = sample_rig(80).to_camrig("x");
    data.metadata.rig_type = "cubemap".into();
    assert!(matches!(
        expect_conversion_error(&data),
        CamRigConversionError::NotSphericalTiles(_)
    ));
}

#[test]
fn from_camrig_rejects_missing_attribute() {
    let mut data = sample_rig(80).to_camrig("x");
    data.metadata.rig_attributes = serde_json::json!({ "centre": [0.0, 0.0, 0.0] });
    assert!(matches!(
        expect_conversion_error(&data),
        CamRigConversionError::BadAttribute(_)
    ));
}

#[test]
fn from_camrig_rejects_non_co_centric_translations() {
    let mut data = sample_rig(80).to_camrig("x");
    data.translations_xyz[[3, 0]] = 0.5;
    assert!(matches!(
        expect_conversion_error(&data),
        CamRigConversionError::Invalid(_)
    ));
}

#[test]
fn from_camrig_rejects_multiple_cameras() {
    let mut data = sample_rig(80).to_camrig("x");
    let extra = data.cameras[0].clone();
    data.cameras.push(extra);
    data.metadata.camera_count = 2;
    assert!(matches!(
        expect_conversion_error(&data),
        CamRigConversionError::Invalid(_)
    ));
}

#[test]
fn from_camrig_rejects_bad_shared_camera() {
    // A non-square camera is not a valid tile camera.
    let mut data = sample_rig(80).to_camrig("x");
    data.cameras[0].height += 1;
    assert!(matches!(
        expect_conversion_error(&data),
        CamRigConversionError::Invalid(_)
    ));
}

#[test]
fn from_camrig_ignores_informational_attributes() {
    // patch_size / half_fov_rad in rig_attributes are informational: the
    // shared camera is authoritative, so garbage values there are ignored.
    let rig = sample_rig(80);
    let mut data = rig.to_camrig("x");
    if let Some(obj) = data.metadata.rig_attributes.as_object_mut() {
        obj.insert("patch_size".into(), serde_json::json!(99999));
        obj.insert("half_fov_rad".into(), serde_json::json!(123.0));
    }
    let back = SphericalTileRig::from_camrig(&data).unwrap();
    assert_eq!(back.patch_size(), rig.patch_size());
    assert!((back.half_fov_rad() - rig.half_fov_rad()).abs() < 1e-12);
}

#[test]
fn from_camrig_rejects_inconsistent_shapes() {
    let mut data = sample_rig(80).to_camrig("x");
    // sensor_count disagrees with the array rows — validate() must catch
    // this before from_camrig indexes quaternion rows.
    data.metadata.sensor_count = 79;
    assert!(matches!(
        expect_conversion_error(&data),
        CamRigConversionError::Invalid(_)
    ));
}
