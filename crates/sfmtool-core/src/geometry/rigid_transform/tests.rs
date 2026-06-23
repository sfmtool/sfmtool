use super::*;
use approx::assert_relative_eq;
use std::f64::consts::FRAC_PI_2;

#[test]
fn identity_transform() {
    let t = RigidTransform::identity();

    assert_eq!(t.rotation, RotQuaternion::identity());

    assert_relative_eq!(t.translation.x, 0.0, epsilon = 1e-12);
    assert_relative_eq!(t.translation.y, 0.0, epsilon = 1e-12);
    assert_relative_eq!(t.translation.z, 0.0, epsilon = 1e-12);

    let origin = t.inverse_translation_origin();
    assert_relative_eq!(origin.x, 0.0, epsilon = 1e-12);
    assert_relative_eq!(origin.y, 0.0, epsilon = 1e-12);
    assert_relative_eq!(origin.z, 0.0, epsilon = 1e-12);
}

#[test]
fn from_axis_angle_rotation_only() {
    let rt = RigidTransform::from_axis_angle(Vector3::z(), FRAC_PI_2).unwrap();
    // Translation should be zero
    assert_relative_eq!(rt.translation.x, 0.0, epsilon = 1e-12);
    assert_relative_eq!(rt.translation.y, 0.0, epsilon = 1e-12);
    assert_relative_eq!(rt.translation.z, 0.0, epsilon = 1e-12);
    // Rotating (1,0,0) by 90° around Z should give (0,1,0)
    let result = rt.transform_point(&Point3::new(1.0, 0.0, 0.0));
    assert_relative_eq!(result.x, 0.0, epsilon = 1e-10);
    assert_relative_eq!(result.y, 1.0, epsilon = 1e-10);
    assert_relative_eq!(result.z, 0.0, epsilon = 1e-10);
}

#[test]
fn from_axis_angle_zero_axis_error() {
    assert!(RigidTransform::from_axis_angle(Vector3::zeros(), 1.0).is_err());
}

#[test]
fn from_wxyz_translation_round_trip() {
    let q = [0.5, 0.5, 0.5, 0.5];
    let t = [1.0, 2.0, 3.0];
    let rt = RigidTransform::from_wxyz_translation(q, t);

    let q_out = rt.rotation.to_wxyz_array();
    assert_relative_eq!(q_out[0], q[0], epsilon = 1e-12);
    assert_relative_eq!(q_out[1], q[1], epsilon = 1e-12);
    assert_relative_eq!(q_out[2], q[2], epsilon = 1e-12);
    assert_relative_eq!(q_out[3], q[3], epsilon = 1e-12);

    assert_relative_eq!(rt.translation.x, t[0], epsilon = 1e-12);
    assert_relative_eq!(rt.translation.y, t[1], epsilon = 1e-12);
    assert_relative_eq!(rt.translation.z, t[2], epsilon = 1e-12);
}

#[test]
fn inverse_translation_origin_identity_rotation() {
    // Identity rotation, t = [0, 0, 5]
    // origin = -R^T * t = -I * [0,0,5] = [0, 0, -5]
    let rt = RigidTransform::new(RotQuaternion::identity(), Vector3::new(0.0, 0.0, 5.0));
    let origin = rt.inverse_translation_origin();
    assert_relative_eq!(origin.x, 0.0, epsilon = 1e-12);
    assert_relative_eq!(origin.y, 0.0, epsilon = 1e-12);
    assert_relative_eq!(origin.z, -5.0, epsilon = 1e-12);
}

#[test]
fn inverse_translation_origin_with_rotation() {
    // 90 degrees around Y: R rotates X->Z, Z->-X
    // R = [[0,0,1],[0,1,0],[-1,0,0]]
    // With t = [1, 0, 0]:
    // origin = -R^T * t = -[[0,0,-1],[0,1,0],[1,0,0]] * [1,0,0] = -[0,0,1] = [0,0,-1]
    let rot = RotQuaternion::from_axis_angle(Vector3::y(), FRAC_PI_2).unwrap();
    let rt = RigidTransform::new(rot, Vector3::new(1.0, 0.0, 0.0));
    let origin = rt.inverse_translation_origin();
    assert_relative_eq!(origin.x, 0.0, epsilon = 1e-10);
    assert_relative_eq!(origin.y, 0.0, epsilon = 1e-10);
    assert_relative_eq!(origin.z, -1.0, epsilon = 1e-10);
}

#[test]
fn transform_point_identity_preserves() {
    let rt = RigidTransform::identity();
    let point = Point3::new(5.0, -3.0, 7.0);
    let result = rt.transform_point(&point);
    assert_relative_eq!(result.x, point.x, epsilon = 1e-12);
    assert_relative_eq!(result.y, point.y, epsilon = 1e-12);
    assert_relative_eq!(result.z, point.z, epsilon = 1e-12);
}

#[test]
fn transform_point_with_rotation_and_translation() {
    // 90 degrees around Z: (1,0,0) -> (0,1,0)
    // Then add translation (10, 20, 30)
    // Result: (0+10, 1+20, 0+30) = (10, 21, 30)
    let rot = RotQuaternion::from_axis_angle(Vector3::z(), FRAC_PI_2).unwrap();
    let rt = RigidTransform::new(rot, Vector3::new(10.0, 20.0, 30.0));
    let result = rt.transform_point(&Point3::new(1.0, 0.0, 0.0));
    assert_relative_eq!(result.x, 10.0, epsilon = 1e-10);
    assert_relative_eq!(result.y, 21.0, epsilon = 1e-10);
    assert_relative_eq!(result.z, 30.0, epsilon = 1e-10);
}

#[test]
fn partial_eq_same() {
    let rt1 = RigidTransform::new(
        RotQuaternion::from_axis_angle(Vector3::z(), 1.0).unwrap(),
        Vector3::new(1.0, 2.0, 3.0),
    );
    let rt2 = rt1.clone();
    assert_eq!(rt1, rt2);
}

#[test]
fn partial_eq_different() {
    let rt1 = RigidTransform::new(RotQuaternion::identity(), Vector3::new(1.0, 2.0, 3.0));
    let rt2 = RigidTransform::new(RotQuaternion::identity(), Vector3::new(1.0, 2.0, 4.0));
    assert_ne!(rt1, rt2);
}

#[test]
fn partial_eq_quaternion_sign_ambiguity() {
    let q = RotQuaternion::from_axis_angle(Vector3::new(1.0, 2.0, 3.0), 1.0).unwrap();
    let neg_q = RotQuaternion::new(-q.w(), -q.x(), -q.y(), -q.z());
    let t = Vector3::new(1.0, 2.0, 3.0);

    let rt1 = RigidTransform::new(q, t);
    let rt2 = RigidTransform::new(neg_q, t);
    assert_eq!(rt1, rt2);
}

#[test]
fn default_is_identity() {
    let rt: RigidTransform = Default::default();
    assert_eq!(rt, RigidTransform::identity());
}

#[test]
fn display_format() {
    let rt = RigidTransform::identity();
    let s = format!("{rt}");
    assert!(s.contains("RigidTransform"));
    assert!(s.contains("rotation="));
    assert!(s.contains("translation="));
}

#[test]
fn to_rotation_matrix_delegates() {
    let rot = RotQuaternion::from_axis_angle(Vector3::z(), FRAC_PI_2).unwrap();
    let rt = RigidTransform::new(rot.clone(), Vector3::zeros());
    let mat_rt = rt.to_rotation_matrix();
    let mat_quat = rot.to_rotation_matrix();
    assert_relative_eq!(mat_rt, mat_quat, epsilon = 1e-12);
}
