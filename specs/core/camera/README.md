# Camera Specifications

Camera models, the distortion kernels behind them, and the image-space
operations they support. Implemented in `crates/sfmtool-core/src/camera/`.

| Document | Description |
|----------|-------------|
| [camera-model-registry.md](camera-model-registry.md) | The single table that defines every camera model and both `SfmrCamera` conversions. |
| [sfmtool-pinhole-kernels.md](sfmtool-pinhole-kernels.md) | The `SFMTOOL_PINHOLE` model: a B-spline radial basis over a perspective base, and its forward/inverse kernels. |
| [sfmtool-fisheye-kernels.md](sfmtool-fisheye-kernels.md) | The `SFMTOOL_FISHEYE` model, sharing that B-spline machinery over a fisheye base, plus the monotonicity check. |
| [projection-jacobian.md](projection-jacobian.md) | Analytic ray-to-pixel derivatives — the shared basis for bundle adjustment and pose refinement. |
| [image-warping.md](image-warping.md) | Applying distort/undistort to whole images: warp maps, remapping, and pyramids. |
| [ray-grid-projection.md](ray-grid-projection.md) | Splitting patch warps into model-free geometry and a camera-owned projection stage, to cut the per-pixel cost. |
| [epipolar-curves.md](epipolar-curves.md) | Epipolar geometry as curves rather than lines, for fisheye and other non-perspective cameras. |
