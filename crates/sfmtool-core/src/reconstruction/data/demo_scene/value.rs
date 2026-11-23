// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The demo scene's reconstruction: every column of an `embedded_patches`
//! value, written down from the scene ([`DemoScene::reconstruction`]).
//!
//! The poses, the points, the keypoints, the normals, the patch frames and
//! the measurement's endpoints come from the scene's geometry and are exact
//! up to their storage type. The thumbnails, the image readings and the
//! patch bitmaps are computed from the decoded JPEGs, the pixels a reader of
//! the written files sees, and each image's file hash is taken over its JPEG
//! bytes, so the value is the one a load of the written workspace gives.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::Arc;

use nalgebra::{Point3, Vector3};
use ndarray::{Array1, Array2, Array3, Array4};

use sfmtool_sfmr_format::{
    ContentHash, DepthStatistics, DistanceMeasurementsData, ImageDepthStats, MeasurementsData,
    ObservedDepthStats, SfmrMetadata, WorkspaceContents, WorkspaceMetadata,
    FEATURE_SOURCE_EMBEDDED_PATCHES, LENGTH_UNIT_M,
};

use crate::camera::image::ImageU8Pyramid;
use crate::camera::sampler::RegionPool;
use crate::camera::TiledPhotograph;
use crate::features::sift::{image_readings_from_rgb, ImageToGray};
use crate::geometry::RigidTransform;
use crate::patch::keypoint_subpixel::KeypointSubpixelParams;
use crate::patch::normal_refine::ProjectedImage;
use crate::patch::stored_bitmap::{
    render_patch_cloud_bitmaps, CloudRenderError, UnreferencedPoints,
};
use crate::patch::PatchCloud;
use crate::progress::Progress;
use crate::reconstruction::measurement_endpoint::{
    resolve_measurement_endpoint, MeasurementEndpointOptions,
};
use crate::reconstruction::thumbnail::thumbnail_from_rgb;
use crate::reconstruction::EditedReconstruction;
use crate::workspace::{
    feature_prefix_dir, feature_type_for_tool, SfmtoolWorkspaceOptions, SFMTOOL_TOOL,
};
use crate::{SfmrReconstruction, THUMBNAIL_SIZE};

use super::super::{
    compute_observation_offsets, ImageSet, ObservationSource, Point3D, PointSet, SfmrImage,
    TrackObservation,
};
use super::points::{demo_poses, project, sample_points, DemoPose, SampledPoint};
use super::render::dot;
use super::texture::{CUBE_A, CUBE_B};
use super::{demo_camera, DemoScene, DemoSceneError};

/// The buckets of each image's depth histogram, as the sphere demo has.
const DEPTH_HISTOGRAM_BUCKETS: u32 = 128;

/// The measurement's label.
const MEASUREMENT_LABEL: &str = "cube edge";

/// The measurement's endpoints: the base and the top of the cube's vertical
/// edge where the faces `-a` and `-b` meet, `-0.5·a − 0.5·b` at heights 0
/// and 1, each with the image it is marked in: the base corner in `demo_2`
/// (0°, 1.2 m) and the top corner in `demo_1` (−27.5°, 3.0 m).
fn measurement_corners() -> [(u32, [f64; 3]); 2] {
    let corner = |z: f64| {
        [
            -0.5 * CUBE_A[0] - 0.5 * CUBE_B[0],
            -0.5 * CUBE_A[1] - 0.5 * CUBE_B[1],
            z,
        ]
    };
    [(2, corner(0.0)), (1, corner(1.0))]
}

/// `v` scaled by `h`, rounded to `f32`: a patch frame's half-vector.
fn half_vector(v: [f64; 3], h: f64) -> [f32; 3] {
    v.map(|c| (c * h) as f32)
}

impl DemoScene {
    /// The decoded photographs as the tiled sources the patch kernels read,
    /// in image order: each of [`photographs`](Self::photographs) through
    /// a full pyramid ([`ImageU8Pyramid::build`] to
    /// [`ImageU8Pyramid::full_levels`]), as [`TiledPhotograph::from_image`]
    /// builds it.
    pub fn tiled_photographs(&self) -> Vec<TiledPhotograph> {
        self.photographs
            .iter()
            .map(|image| {
                let levels = ImageU8Pyramid::full_levels(image.width(), image.height());
                TiledPhotograph::from_pyramid(Arc::new(ImageU8Pyramid::build(image, levels)))
            })
            .collect()
    }

    /// The reconstruction of the scene, an `embedded_patches` value with no
    /// files behind it (`workspace_dir` empty).
    ///
    /// The points are sampled on the ground and the cube's faces, and each is
    /// kept with the cameras that observe it when two or more do, in a phase
    /// `points` that counts the points sampled. Each observation's keypoint is
    /// the point's projection, each point's patch frame lies in its surface
    /// and its normal is the frame's, `normalize(u × v)`. The thumbnails and
    /// image readings are computed from the decoded photographs, and the
    /// patch bitmaps rendered from them, from each point's reference
    /// observation by the reference-view rule, in a phase `patch bitmaps` that
    /// counts the patches. The one distance measurement, the cube's 1 m
    /// vertical edge, is kept when both its endpoints resolve to a texel of an
    /// observed patch, and left out otherwise.
    ///
    /// # Errors
    ///
    /// [`DemoSceneError::Cancelled`] when `progress` is cancelled, and
    /// [`DemoSceneError::Photograph`] when a photograph could not be read for
    /// the bitmaps.
    ///
    /// ```
    /// use sfmtool_core::progress::Progress;
    /// use sfmtool_core::reconstruction::{DemoScene, DemoSceneOptions};
    ///
    /// let options = DemoSceneOptions {
    ///     image_width: 160,
    ///     image_height: 120,
    ///     focal_length_px: 140.0,
    ///     ..DemoSceneOptions::default()
    /// };
    /// let scene = DemoScene::render(&options, &Progress::none())?;
    /// let recon = scene.reconstruction(&Progress::none())?;
    /// assert_eq!(recon.image_count(), 5);
    /// assert!(recon.point_count() > 100);
    /// # Ok::<(), sfmtool_core::reconstruction::DemoSceneError>(())
    /// ```
    pub fn reconstruction(
        &self,
        progress: &Progress<'_>,
    ) -> Result<SfmrReconstruction, DemoSceneError> {
        let options = &self.options;
        let camera = demo_camera(options);
        let poses = demo_poses();
        let sampled = {
            let phase = progress.phase("points");
            sample_points(options, &camera, &poses, &phase)?
        };
        let mut recon = self.geometry(&poses, &sampled);
        recon.recompute_point_errors().map_err(|e| match e {
            crate::reconstruction::ReconstructionError::Cancelled => DemoSceneError::Cancelled,
            // Every observation has its keypoint inline, so no `.sift` file
            // is read.
            e => unreachable!("the demo scene's errors are scored from its own keypoints: {e}"),
        })?;
        recon
            .recompute_depth_statistics()
            .expect("the demo scene's columns agree with each other");
        {
            let phase = progress.phase("patch bitmaps");
            self.render_bitmaps(&mut recon, &poses, &phase)?;
        }
        if let Some(measurement) = cube_edge_measurement(&recon, &camera, &poses) {
            recon.image_set.measurements = MeasurementsData {
                distance: Some(measurement),
            };
            recon
                .image_set
                .measurements
                .set_metadata_counts(&mut recon.metadata);
        }
        Ok(recon)
    }

    /// The value with every column but the bitmaps, the reprojection errors,
    /// the depth statistics and the measurement: the images and their
    /// photograph-derived columns, and the points, tracks, keypoints and
    /// patch frames of `sampled`.
    fn geometry(&self, poses: &[DemoPose], sampled: &[SampledPoint]) -> SfmrReconstruction {
        let options = &self.options;
        let camera = demo_camera(options);
        let image_to_gray = ImageToGray::default();
        let (width, height) = (options.image_width, options.image_height);

        let images: Vec<SfmrImage> = poses
            .iter()
            .zip(&self.image_names)
            .zip(&self.photographs)
            .map(|((pose, name), photograph)| SfmrImage {
                name: name.clone(),
                camera_index: 0,
                quaternion_wxyz: pose.rotation,
                translation_xyz: pose.translation,
                intrinsic_adjustment: Default::default(),
                readings: Some(
                    image_readings_from_rgb(width, height, photograph.data(), &image_to_gray)
                        .expect("a decoded photograph is RGB of its size, with finite readings"),
                ),
            })
            .collect();
        let image_count = images.len();

        let side = THUMBNAIL_SIZE;
        let mut thumbnails = Array4::<u8>::zeros((image_count, side, side, 3));
        for (i, photograph) in self.photographs.iter().enumerate() {
            let row = thumbnail_from_rgb(photograph.data(), width as usize, height as usize);
            thumbnails
                .index_axis_mut(ndarray::Axis(0), i)
                .as_slice_mut()
                .expect("a freshly allocated array is contiguous")
                .copy_from_slice(&row);
        }

        let h = options.patch_half_extent;
        let point_count = sampled.len();
        let mut points = Vec::with_capacity(point_count);
        let mut patch_u = Array2::<f32>::zeros((point_count, 3));
        let mut patch_v = Array2::<f32>::zeros((point_count, 3));
        let mut tracks = Vec::new();
        let mut observation_counts = Vec::with_capacity(point_count);
        let mut keypoints = Vec::new();
        for (p, point) in sampled.iter().enumerate() {
            let (_, u_axis, v_axis) = point.surface.frame();
            let (u, v) = (half_vector(u_axis, h), half_vector(v_axis, h));
            patch_u.row_mut(p).assign(&Array1::from(u.to_vec()));
            patch_v.row_mut(p).assign(&Array1::from(v.to_vec()));
            let normal = sfmtool_sfmr_format::patch_normal(u, v)
                .expect("a patch frame along two orthogonal unit axes gives a normal");
            let colour = point
                .surface
                .colour(point.position, options.seed)
                .map(|c| c.round().clamp(0.0, 255.0) as u8);
            points.push(Point3D {
                position: Point3::from(point.position),
                w: 1.0,
                color: colour,
                rms_reprojection_error: 0.0,
                normal: Vector3::from(normal),
            });
            for &image in &point.images {
                let [x, y] = project(&camera, &poses[image as usize], point.position)
                    .expect("an observing camera has the point in front of it");
                tracks.push(TrackObservation {
                    image_index: image,
                    point_index: p as u32,
                });
                keypoints.extend([x as f32, y as f32]);
            }
            observation_counts.push(point.images.len() as u32);
        }
        let observation_count = tracks.len();
        let keypoints_xy = Array2::from_shape_vec((observation_count, 2), keypoints)
            .expect("two coordinates per observation");

        let image_file_hashes = self
            .encoded_photographs
            .iter()
            .map(|bytes| xxhash_rust::xxh3::xxh3_128(bytes).to_be_bytes())
            .collect();

        let workspace = SfmtoolWorkspaceOptions::default();
        let feature_options = workspace.feature_options();
        let feature_type = feature_type_for_tool(SFMTOOL_TOOL, &feature_options);
        let feature_prefix_dir = feature_prefix_dir(SFMTOOL_TOOL, &feature_type, &feature_options);
        let metadata = SfmrMetadata {
            version: sfmtool_sfmr_format::SFMR_FORMAT_VERSION,
            operation: "demo".into(),
            tool: "sfmtool".into(),
            tool_version: env!("CARGO_PKG_VERSION").into(),
            tool_options: BTreeMap::new(),
            workspace: WorkspaceMetadata {
                absolute_path: String::new(),
                relative_path: ".".into(),
                contents: WorkspaceContents {
                    feature_tool: SFMTOOL_TOOL.into(),
                    feature_type,
                    feature_options,
                    feature_prefix_dir,
                },
            },
            timestamp: String::new(),
            image_count: image_count as u32,
            point_count: point_count as u32,
            infinity_point_count: 0,
            observation_count: observation_count as u32,
            camera_count: 1,
            rig_count: None,
            sensor_count: None,
            frame_count: None,
            distance_measurement_count: None,
            world_space_unit: Some("m".into()),
            feature_source: FEATURE_SOURCE_EMBEDDED_PATCHES.to_string(),
        };

        // Filled by `recompute_depth_statistics`.
        let empty_stats = ImageDepthStats {
            histogram_min_z: None,
            histogram_max_z: None,
            observed: ObservedDepthStats {
                count: 0,
                infinity_count: 0,
                min_z: None,
                max_z: None,
                median_z: None,
                mean_z: None,
            },
        };
        let observation_offsets = compute_observation_offsets(&observation_counts);
        let mut recon = SfmrReconstruction {
            workspace_dir: PathBuf::new(),
            metadata,
            content_hash: ContentHash::default(),
            image_set: ImageSet {
                cameras: vec![camera],
                per_image_adjustments: Vec::new(),
                images,
                thumbnails_y_x_rgb: Some(Arc::new(thumbnails)),
                image_to_gray: Some(image_to_gray),
                depth_statistics: DepthStatistics {
                    num_histogram_buckets: DEPTH_HISTOGRAM_BUCKETS,
                    images: vec![empty_stats; image_count],
                },
                depth_histogram_counts: vec![
                    vec![0u32; DEPTH_HISTOGRAM_BUCKETS as usize];
                    image_count
                ],
                rig_frame_data: None,
                measurements: MeasurementsData::default(),
            },
            point_set: PointSet {
                infinity_point_count: 0,
                patch_u_halfvec_xyz: Some(patch_u),
                patch_v_halfvec_xyz: Some(patch_v),
                patch_bitmaps_y_x_rgba: None,
                patch_bitmaps_for_display: false,
                has_normals: true,
                normal_confidence: None,
                point_constraints: None,
                observation_confidence: None,
                observation_readings: None,
                reference_observations: Some(vec![-1; point_count]),
                display_only_references: None,
                observations: ObservationSource::EmbeddedPatches { image_file_hashes },
                keypoints_xy: Some(keypoints_xy),
                points,
                tracks,
                observation_counts,
                observation_offsets,
                image_feature_to_point: Vec::new(),
                max_track_feature_index: Vec::new(),
            },
        };
        recon.rebuild_derived_fields();
        recon
    }

    /// Render every point's bitmap from the decoded photographs, from the
    /// observation the reference-view rule picks, at the options' bitmap
    /// side, and store the bitmaps and the picks: what `--add-patch-bitmaps`
    /// does for a value with patch frames and no references.
    fn render_bitmaps(
        &self,
        recon: &mut SfmrReconstruction,
        poses: &[DemoPose],
        progress: &Progress<'_>,
    ) -> Result<(), DemoSceneError> {
        let cloud = PatchCloud::from_stored_patch_frames(recon)
            .expect("the demo scene's patch frames have one row per point")
            .expect("the demo scene stores patch frames");
        let sources = self.tiled_photographs();
        let transforms: Vec<RigidTransform> = poses
            .iter()
            .map(|pose| {
                let q = pose.rotation;
                let t = pose.translation;
                RigidTransform::from_wxyz_translation([q.w, q.i, q.j, q.k], [t.x, t.y, t.z])
            })
            .collect();
        let camera = &recon.image_set.cameras[0];
        let views: Vec<Option<ProjectedImage<'_>>> = transforms
            .iter()
            .zip(&sources)
            .map(|(cam_from_world, source)| {
                Some(ProjectedImage {
                    camera,
                    cam_from_world,
                    source,
                })
            })
            .collect();
        let params = KeypointSubpixelParams {
            resolution: self.options.patch_bitmap_resolution,
            ..KeypointSubpixelParams::default()
        };
        let column = render_patch_cloud_bitmaps(
            &cloud,
            recon,
            &views,
            &RegionPool::for_kernel_call(),
            &params,
            UnreferencedPoints::Pick,
            None,
            progress,
        )
        .map_err(|e| match e {
            CloudRenderError::Cancelled => DemoSceneError::Cancelled,
            CloudRenderError::Photograph(e) => DemoSceneError::Photograph(
                ::image::ImageError::IoError(std::io::Error::other(e.to_string())),
            ),
            // The cloud, the keypoints and one view per image all come from
            // `recon`, so they agree with it.
            e => unreachable!("the demo scene's bitmaps of its own frames: {e}"),
        })?;
        recon.point_set.patch_bitmaps_y_x_rgba = Some(Arc::new(column.bitmaps));
        recon.point_set.reference_observations = Some(column.reference_observations);
        Ok(())
    }
}

/// The cube-edge measurement of `recon`, or `None` when an endpoint does not
/// resolve to a finite texel of an observed patch.
///
/// Each endpoint is its corner's projection into its image, rounded to `f32`
/// as `tracks/keypoints_xy` is, and is resolved by
/// [`resolve_measurement_endpoint`] with the default options, as a reader of
/// the measurement resolves it.
fn cube_edge_measurement(
    recon: &SfmrReconstruction,
    camera: &crate::camera::CameraIntrinsics,
    poses: &[DemoPose],
) -> Option<DistanceMeasurementsData> {
    let corners = measurement_corners();
    let mut endpoints = Array3::<f32>::zeros((1, 2, 2));
    for (k, &(image, corner)) in corners.iter().enumerate() {
        let [x, y] = project(camera, &poses[image as usize], corner)?;
        let inside = (0.0..f64::from(camera.width)).contains(&x)
            && (0.0..f64::from(camera.height)).contains(&y);
        if !inside {
            return None;
        }
        endpoints[[0, k, 0]] = x as f32;
        endpoints[[0, k, 1]] = y as f32;
    }
    let edited = EditedReconstruction::new(Arc::new(recon.clone()));
    let options = MeasurementEndpointOptions::default();
    for (k, &(image, _)) in corners.iter().enumerate() {
        let pixel = [
            f64::from(endpoints[[0, k, 0]]),
            f64::from(endpoints[[0, k, 1]]),
        ];
        resolve_measurement_endpoint(&edited, image, pixel, &options)
            .ok()?
            .position()
            .finite()?;
    }
    let length = {
        let d = [0, 1, 2].map(|i| corners[1].1[i] - corners[0].1[i]);
        dot(d, d).sqrt()
    };
    Some(DistanceMeasurementsData {
        distances: Array1::from(vec![length as f32]),
        units: Array1::from(vec![LENGTH_UNIT_M]),
        image_indexes: Array2::from_shape_vec((1, 2), corners.map(|(image, _)| image).to_vec())
            .expect("two images for one measurement"),
        endpoints_xy: endpoints,
        labels: vec![MEASUREMENT_LABEL.to_owned()],
    })
}

#[cfg(test)]
mod tests;
