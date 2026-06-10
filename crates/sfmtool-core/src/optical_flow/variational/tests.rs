use super::*;

#[test]
fn test_variational_refine_doesnt_crash() {
    let img = GrayImage::checkerboard(16, 16);
    let mut flow = FlowField::new(16, 16);
    let params = VariationalParams {
        delta: 5.0,
        gamma: 10.0,
        alpha: 10.0,
        jacobi_iterations: 3,
        outer_iterations: 1,
    };
    variational_refine(&img, &img, &mut flow, &params);

    // Flow should remain near zero for identical images
    let mut max_flow = 0.0f32;
    for row in 0..16 {
        for col in 0..16 {
            let (dx, dy) = flow.get(col, row);
            max_flow = max_flow.max(dx.abs()).max(dy.abs());
        }
    }
    assert!(
        max_flow < 0.5,
        "Expected near-zero flow, got max {}",
        max_flow
    );
}

#[test]
fn test_psi_deriv() {
    // psi_deriv(0) = 1 / (2 * sqrt(eps^2)) = 1 / (2 * eps) = 500
    let val = psi_deriv(0.0);
    assert!(val > 400.0 && val < 600.0, "psi_deriv(0) = {}", val);

    // For large values, psi_deriv should be small
    let val_large = psi_deriv(100.0);
    assert!(val_large < 0.1, "psi_deriv(100) = {}", val_large);
}

#[test]
fn test_variational_small_image_noop() {
    // Images smaller than 3x3 should be skipped
    let img = GrayImage::new_constant(2, 2, 0.5);
    let mut flow = FlowField::new(2, 2);
    let params = VariationalParams {
        delta: 5.0,
        gamma: 10.0,
        alpha: 10.0,
        jacobi_iterations: 5,
        outer_iterations: 1,
    };
    variational_refine(&img, &img, &mut flow, &params);
    // Should not crash, flow should remain zero
    let (dx, dy) = flow.get(0, 0);
    assert!((dx).abs() < 1e-6);
    assert!((dy).abs() < 1e-6);
}
