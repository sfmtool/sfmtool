use super::*;
use approx::assert_relative_eq;

#[test]
fn test_look_at_basic() {
    let camera = Camera::look_at(Point3::new(0.0, -5.0, 0.0), Point3::origin(), Vector3::z());

    // Camera should be looking toward +Y (forward = +Y)
    let forward = camera.forward();
    assert_relative_eq!(forward.x, 0.0, epsilon = 1e-10);
    assert_relative_eq!(forward.y, 1.0, epsilon = 1e-10);
    assert_relative_eq!(forward.z, 0.0, epsilon = 1e-10);

    // Up should be +Z
    let up = camera.up();
    assert_relative_eq!(up.x, 0.0, epsilon = 1e-10);
    assert_relative_eq!(up.y, 0.0, epsilon = 1e-10);
    assert_relative_eq!(up.z, 1.0, epsilon = 1e-10);
}

#[test]
fn test_view_matrix_identity_at_origin() {
    // Camera at origin looking down -Z with Y up
    let camera = Camera {
        position: Point3::origin(),
        orientation: UnitQuaternion::identity(),
        target_distance: 5.0,
    };

    let _view = camera.view_matrix();

    // Point in front of camera (negative Z in world = negative Z in camera)
    let world_point = Point3::new(0.0, 0.0, -5.0);
    let camera_point = camera.world_to_camera(&world_point);

    assert_relative_eq!(camera_point.x, 0.0, epsilon = 1e-10);
    assert_relative_eq!(camera_point.y, 0.0, epsilon = 1e-10);
    assert_relative_eq!(camera_point.z, -5.0, epsilon = 1e-10);
}

#[test]
fn test_round_trip() {
    let camera = Camera::look_at(
        Point3::new(1.0, 2.0, 3.0),
        Point3::new(4.0, 5.0, 6.0),
        Vector3::z(),
    );

    let world_point = Point3::new(10.0, 20.0, 30.0);
    let camera_point = camera.world_to_camera(&world_point);
    let back_to_world = camera.camera_to_world(&camera_point);

    assert_relative_eq!(back_to_world.x, world_point.x, epsilon = 1e-10);
    assert_relative_eq!(back_to_world.y, world_point.y, epsilon = 1e-10);
    assert_relative_eq!(back_to_world.z, world_point.z, epsilon = 1e-10);
}
