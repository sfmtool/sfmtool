// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

use std::path::Path;
use std::sync::OnceLock;

use super::super::points::{demo_poses, sample_points};
use super::super::render::{camera_rotation, CAMERA_CENTRES};
use super::super::tests::{reference_trace, scene};
use super::super::texture::Surface;
use super::super::DemoSceneOptions;
use super::*;
use crate::camera::image::ImageU8;
use crate::reconstruction::measurement_endpoint::resolve_distance_measurements;

/// The surface a point of the demo scene lies on, by its position: the
/// ground at `z = 0` outside the cube's footprint, and otherwise the cube's
/// face whose plane it is nearest. For the tests, which check each point's
/// columns against its surface.
fn surface_of(x: [f64; 3]) -> Surface {
    let pa = dot(x, CUBE_A);
    let pb = dot(x, CUBE_B);
    if x[2].abs() < 1e-12 && !(pa.abs() < 0.5 && pb.abs() < 0.5) {
        return Surface::Ground;
    }
    [
        (Surface::Top, (x[2] - 1.0).abs()),
        (Surface::MinusA, (pa + 0.5).abs()),
        (Surface::MinusB, (pb + 0.5).abs()),
        (Surface::PlusA, (pa - 0.5).abs()),
        (Surface::PlusB, (pb - 0.5).abs()),
    ]
    .into_iter()
    .min_by(|a, b| a.1.total_cmp(&b.1))
    .map(|(s, _)| s)
    .expect("five faces")
}

/// The default scene's reconstruction, built once for the tests that only
/// read it.
fn recon() -> &'static SfmrReconstruction {
    static RECON: OnceLock<SfmrReconstruction> = OnceLock::new();
    RECON.get_or_init(|| scene().reconstruction(&Progress::none()).unwrap())
}

/// The pixel where camera `image` sees `x`, computed from the camera's centre
/// and the rows of its rotation rather than the stored pose: `p = R (x − C)`,
/// `u = cx + f · p_x / −p_z`, `v = cy − f · p_y / −p_z`.
fn projection(image: u32, x: [f64; 3]) -> [f64; 2] {
    let centre = CAMERA_CENTRES[image as usize];
    let rows = camera_rotation(centre);
    let d = [0, 1, 2].map(|k| x[k] - centre[k]);
    let p = rows.map(|row| dot(row, d));
    assert!(p[2] < 0.0, "{x:?} is behind camera {image}");
    let f = 560.0;
    [320.0 + f * p[0] / -p[2], 240.0 - f * p[1] / -p[2]]
}

/// The position of point `p` of `recon`.
fn position(recon: &SfmrReconstruction, p: usize) -> [f64; 3] {
    let x = recon.point_set.points[p].position;
    [x.x, x.y, x.z]
}

/// Point `p`'s observations in `recon`, as `(image, keypoint)`.
fn observations(recon: &SfmrReconstruction, p: usize) -> Vec<(u32, [f64; 2])> {
    let offsets = &recon.point_set.observation_offsets;
    let keypoints = recon.point_set.keypoints_xy.as_ref().unwrap();
    (offsets[p]..offsets[p + 1])
        .map(|j| {
            (
                recon.point_set.tracks[j].image_index,
                [f64::from(keypoints[[j, 0]]), f64::from(keypoints[[j, 1]])],
            )
        })
        .collect()
}

#[test]
fn the_defaults_keep_the_default_point_count_and_for_point_count_keeps_near_it() {
    assert_eq!(
        recon().point_count(),
        DemoSceneOptions::DEFAULT_POINT_COUNT as usize
    );
    // The sampling alone, without the photographs: the count kept is the
    // reconstruction's, which is built from it.
    for asked in [300u32, 1000, 3000, 10_000] {
        let options = DemoSceneOptions::for_point_count(asked).unwrap();
        let kept = sample_points(
            &options,
            &demo_camera(&options),
            &demo_poses(),
            &Progress::none(),
        )
        .unwrap()
        .len();
        let off = kept as f64 / f64::from(asked) - 1.0;
        assert!(off.abs() <= 0.05, "{asked} asked, {kept} kept");
    }
}

#[test]
fn every_keypoint_is_its_points_projection() {
    let recon = recon();
    assert_eq!(
        recon.point_set.tracks.len(),
        recon.point_set.keypoints_xy.as_ref().unwrap().nrows()
    );
    for p in 0..recon.point_count() {
        let observed = observations(recon, p);
        assert!(observed.len() >= 2, "point {p}");
        for (image, keypoint) in observed {
            let expected = projection(image, position(recon, p));
            for k in 0..2 {
                // The rounding of a coordinate under 640 to f32 is at most
                // 3 · 10⁻⁵ px.
                assert!(
                    (keypoint[k] - expected[k]).abs() < 1e-4,
                    "point {p} in image {image}: {keypoint:?} against {expected:?}"
                );
            }
            assert!((0.0..640.0).contains(&keypoint[0]) && (0.0..480.0).contains(&keypoint[1]));
        }
    }
}

#[test]
fn no_observations_segment_meets_the_cube() {
    let recon = recon();
    for p in 0..recon.point_count() {
        let x = position(recon, p);
        for (image, _) in observations(recon, p) {
            let centre = CAMERA_CENTRES[image as usize];
            let segment = [0, 1, 2].map(|k| x[k] - centre[k]);
            let (surface, lambda) = reference_trace(centre, segment).expect("the point is hit");
            assert!(
                lambda >= 1.0 - 1e-6,
                "point {p} at {x:?} in image {image}: the segment meets {surface:?} at {lambda}"
            );
        }
    }
}

#[test]
fn a_ground_point_behind_the_cube_is_hidden_from_the_middle_camera_and_seen_from_an_outer_one() {
    let recon = recon();
    let middle = CAMERA_CENTRES[2];
    let mut hidden = 0;
    for p in 0..recon.point_count() {
        let x = position(recon, p);
        if surface_of(x) != Surface::Ground {
            continue;
        }
        let images: Vec<u32> = observations(recon, p).iter().map(|o| o.0).collect();
        let behind_cube = matches!(
            reference_trace(middle, [0, 1, 2].map(|k| x[k] - middle[k])),
            Some((surface, lambda)) if surface != Surface::Ground && lambda < 1.0
        );
        if !behind_cube {
            continue;
        }
        // The cube is the only reason camera 2 does not see it: it projects
        // inside the image, and the camera views the ground above the cut.
        let [u, v] = projection(2, x);
        let to_camera = [0, 1, 2].map(|k| middle[k] - x[k]);
        let cos = to_camera[2] / dot(to_camera, to_camera).sqrt();
        if !((0.0..640.0).contains(&u) && (0.0..480.0).contains(&v))
            || cos < 80f64.to_radians().cos()
        {
            continue;
        }
        assert!(!images.contains(&2), "point {p} at {x:?}: {images:?}");
        if images.contains(&0) || images.contains(&4) {
            hidden += 1;
        }
    }
    assert!(hidden >= 5, "{hidden} ground points hidden by the cube");
}

#[test]
fn the_derived_normals_are_the_surface_normals() {
    let recon = recon();
    let u = recon.point_set.patch_u_halfvec_xyz.as_ref().unwrap();
    let v = recon.point_set.patch_v_halfvec_xyz.as_ref().unwrap();
    let mut on = std::collections::BTreeMap::new();
    for (p, point) in recon.point_set.points.iter().enumerate() {
        let surface = surface_of(position(recon, p));
        *on.entry(surface as u8).or_insert(0) += 1;
        let expected = surface.normal();
        let row = |a: &Array2<f32>| [a[[p, 0]], a[[p, 1]], a[[p, 2]]];
        let derived = sfmtool_sfmr_format::patch_normal(row(u), row(v)).unwrap();
        for k in 0..3 {
            assert!(
                (f64::from(point.normal[k]) - expected[k]).abs() < 1e-6,
                "point {p} on {surface:?}: normal {:?}",
                point.normal
            );
            assert_eq!(point.normal[k], derived[k], "point {p}");
        }
        // The frame lies in the surface, square, at the half-extent.
        for a in [row(u), row(v)] {
            let a = a.map(f64::from);
            assert!(dot(a, expected).abs() < 1e-7, "point {p}");
            assert!((dot(a, a).sqrt() - 0.09).abs() < 1e-7, "point {p}");
        }
    }
    // The ground, the top and the faces -a and -b keep points; +a and +b do
    // not.
    let surfaces: Vec<u8> = on.keys().copied().collect();
    assert_eq!(
        surfaces,
        [
            Surface::Ground,
            Surface::Top,
            Surface::MinusA,
            Surface::MinusB
        ]
        .map(|s| s as u8)
    );
}

#[test]
fn the_measurement_is_the_cube_edge_and_resolves_to_1_m() {
    let recon = recon();
    let m = recon
        .image_set
        .measurements
        .distance
        .as_ref()
        .expect("a distance measurement");
    assert_eq!(m.len(), 1);
    assert_eq!(m.distances[0], 1.0);
    assert_eq!(m.units[0], LENGTH_UNIT_M);
    assert_eq!(m.labels, ["cube edge"]);
    assert_eq!(recon.metadata.distance_measurement_count, Some(1));
    let edited = EditedReconstruction::new(Arc::new(recon.clone()));
    let options = MeasurementEndpointOptions::default();
    for (k, (image, corner)) in measurement_corners().into_iter().enumerate() {
        assert_eq!(m.image_indexes[[0, k]], image);
        let pixel = [
            f64::from(m.endpoints_xy[[0, k, 0]]),
            f64::from(m.endpoints_xy[[0, k, 1]]),
        ];
        let expected = projection(image, corner);
        for c in 0..2 {
            assert!((pixel[c] - expected[c]).abs() < 1e-4, "endpoint {k}");
        }
        let resolution = resolve_measurement_endpoint(&edited, image, pixel, &options).unwrap();
        let chosen = resolution.chosen();
        let at = chosen.position.finite().unwrap();
        let off = [0, 1, 2].map(|i| at[i] - corner[i]);
        assert!(
            dot(off, off).sqrt() < 1e-5,
            "endpoint {k} resolves to {at:?}"
        );
        assert!(chosen.patch_coords.iter().all(|c| c.abs() <= 1.0));
        // The texel is in a patch the image observes.
        let observed = observations(recon, chosen.point as usize);
        assert_eq!(observed[chosen.observation].0, image, "endpoint {k}");
    }
    let resolved = resolve_distance_measurements(&edited, &options);
    let distance = resolved[0].scene_distance.unwrap();
    assert!((distance - 1.0).abs() < 1e-6, "{distance}");
}

/// A copy of the default scene with other options that do not change its
/// photographs, so it need not be rendered again.
fn with_options(options: DemoSceneOptions) -> DemoScene {
    let scene = scene();
    let copy = |images: &[ImageU8]| -> Vec<ImageU8> {
        images
            .iter()
            .map(|i| ImageU8::new(i.width(), i.height(), i.channels(), i.data().to_vec()))
            .collect()
    };
    DemoScene {
        options,
        image_names: scene.image_names.clone(),
        rendered_photographs: copy(&scene.rendered_photographs),
        encoded_photographs: scene.encoded_photographs.clone(),
        photographs: copy(&scene.photographs),
    }
}

#[test]
fn an_option_set_that_leaves_a_corner_in_no_patch_leaves_the_measurement_out() {
    // A half-extent of 0.03 m: the face points nearest the corners are
    // 0.0625 m from both edges, and the ground points nearest the base corner
    // at least 0.0375 m along x, so no patch holds a corner.
    let recon = with_options(DemoSceneOptions {
        patch_half_extent: 0.03,
        ..DemoSceneOptions::default()
    })
    .reconstruction(&Progress::none())
    .unwrap();
    assert!(recon.point_count() > 400);
    assert!(recon.image_set.measurements.distance.is_none());
    assert_eq!(recon.metadata.distance_measurement_count, None);
}

/// Write the default scene's workspace into `dir`, returning the `.sfmr`
/// path.
fn write(dir: &Path) -> PathBuf {
    scene().write_workspace(dir, &Progress::none()).unwrap()
}

#[test]
fn the_value_saves_verifies_and_loads_back_equal_with_the_workspace_directory_set() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path().join("demo");
    let path = write(&dir);
    assert_eq!(path, dir.join("demo.sfmr"));
    let (valid, errors) = sfmtool_sfmr_format::verify_sfmr(&path).unwrap();
    assert!(valid, "{errors:?}");
    assert_eq!(
        std::fs::read_to_string(dir.join(".sfm-workspace.json")).unwrap(),
        crate::workspace::sfmtool_workspace_marker(&SfmtoolWorkspaceOptions::default())
    );

    let loaded = SfmrReconstruction::load(&path, &Progress::none()).unwrap();
    assert_eq!(
        loaded.workspace_dir.canonicalize().unwrap(),
        dir.canonicalize().unwrap()
    );
    let value = recon();
    assert!(value.workspace_dir.as_os_str().is_empty());
    let hash = value.content_xxh128().unwrap();
    assert_eq!(format!("{:?}", loaded.content_hash), format!("{hash:?}"));
    assert_eq!(
        format!("{:?}", loaded.content_xxh128().unwrap()),
        format!("{hash:?}")
    );
    assert_eq!(loaded.point_set.points, value.point_set.points);
    assert_eq!(loaded.point_set.tracks, value.point_set.tracks);
    assert_eq!(loaded.point_set.keypoints_xy, value.point_set.keypoints_xy);
    assert_eq!(
        loaded.point_set.patch_bitmaps_y_x_rgba,
        value.point_set.patch_bitmaps_y_x_rgba
    );
    assert_eq!(
        loaded.point_set.reference_observations,
        value.point_set.reference_observations
    );
    assert_eq!(loaded.image_set.measurements, value.image_set.measurements);

    // The image file hashes are those of the written files, which are the
    // scene's JPEGs.
    let hashes = loaded.image_file_hashes().unwrap();
    for (i, name) in scene().image_names().iter().enumerate() {
        let bytes = std::fs::read(dir.join(name)).unwrap();
        assert!(bytes == scene().encoded_photographs()[i], "{name}");
        assert_eq!(
            hashes[i],
            xxhash_rust::xxh3::xxh3_128(&bytes).to_be_bytes(),
            "{name}"
        );
    }

    // A second write into the directory, which now holds files, is refused.
    let refused = scene().write_workspace(&dir, &Progress::none());
    assert!(
        matches!(refused, Err(DemoSceneError::Directory(_))),
        "{refused:?}"
    );
}

#[test]
fn the_thumbnails_readings_and_bitmaps_are_those_of_the_written_files_decoded_again() {
    let tmp = tempfile::tempdir().unwrap();
    let path = write(tmp.path());
    let loaded = SfmrReconstruction::load(&path, &Progress::none()).unwrap();
    let value = recon();
    let thumbnails = loaded.image_set.thumbnails_y_x_rgb.as_ref().unwrap();
    let mut decoded = Vec::new();
    for (i, image) in loaded.image_set.images.iter().enumerate() {
        let photograph =
            ImageU8::decode_rgb(&std::fs::read(loaded.workspace_dir.join(&image.name)).unwrap())
                .unwrap();
        let (w, h) = (photograph.width(), photograph.height());
        let thumbnail = thumbnail_from_rgb(photograph.data(), w as usize, h as usize);
        assert!(
            thumbnails
                .index_axis(ndarray::Axis(0), i)
                .as_slice()
                .unwrap()
                == thumbnail,
            "image {i}"
        );
        let readings =
            image_readings_from_rgb(w, h, photograph.data(), &ImageToGray::default()).unwrap();
        assert_eq!(
            value.image_set.images[i].readings,
            Some(readings),
            "image {i}"
        );
        decoded.push(TiledPhotograph::from_image(photograph));
    }

    // The bitmaps rendered again from the decoded files, from the stored
    // reference observations.
    let poses = demo_poses();
    let transforms: Vec<RigidTransform> = poses
        .iter()
        .map(|pose| {
            let (q, t) = (pose.rotation, pose.translation);
            RigidTransform::from_wxyz_translation([q.w, q.i, q.j, q.k], [t.x, t.y, t.z])
        })
        .collect();
    let camera = &loaded.image_set.cameras[0];
    let views: Vec<Option<ProjectedImage<'_>>> = transforms
        .iter()
        .zip(&decoded)
        .map(|(cam_from_world, source)| {
            Some(ProjectedImage {
                camera,
                cam_from_world,
                source,
            })
        })
        .collect();
    let cloud = PatchCloud::from_stored_patch_frames(&loaded)
        .unwrap()
        .unwrap();
    let again = render_patch_cloud_bitmaps(
        &cloud,
        &loaded,
        &views,
        &RegionPool::for_kernel_call(),
        &KeypointSubpixelParams {
            resolution: 24,
            ..KeypointSubpixelParams::default()
        },
        UnreferencedPoints::Skip,
        None,
        &Progress::none(),
    )
    .unwrap();
    let references = loaded.point_set.reference_observations.as_ref().unwrap();
    // One point near the lower edge of both its views has no view whose tile
    // the reference-view rule takes, and so no bitmap.
    let unreferenced = references.iter().filter(|&&r| r < 0).count();
    assert!(
        unreferenced <= 1,
        "{unreferenced} points without a reference"
    );
    assert_eq!(&again.reference_observations, references);
    assert!(
        Some(&again.bitmaps) == loaded.point_set.patch_bitmaps_y_x_rgba.as_deref(),
        "the bitmaps rendered again differ"
    );
}

/// The values the default scene's photograph-derived columns take on one
/// encoder path: a sample of thumbnail texels, every image's readings, and a
/// sample of bitmap texels with the reference observation each bitmap is
/// rendered from.
struct DerivedDigest {
    /// The RGB of each texel of [`THUMBNAIL_TEXELS`] in each image, image by
    /// image.
    thumbnails: [[u8; 3]; 15],
    /// Each image's `[sharpness, Exx, Eyy, Exy]`.
    readings: [[f32; 4]; 5],
    /// The RGB of texel `(12, 8)` of every 36th point's bitmap.
    bitmaps: [[u8; 3]; 15],
    /// The reference observation of every 36th point.
    references: [i32; 15],
}

/// The thumbnail texels [`DerivedDigest::thumbnails`] samples, `(row,
/// column)`, three per image.
const THUMBNAIL_TEXELS: [(usize, usize); 3] = [(100, 30), (64, 64), (110, 90)];

/// The digest of `recon`'s photograph-derived columns.
fn digest(recon: &SfmrReconstruction) -> DerivedDigest {
    let thumbnails = recon.image_set.thumbnails_y_x_rgb.as_ref().unwrap();
    let bitmaps = recon.point_set.patch_bitmaps_y_x_rgba.as_ref().unwrap();
    let references = recon.point_set.reference_observations.as_ref().unwrap();
    let rgb = |f: &dyn Fn(usize) -> u8| [f(0), f(1), f(2)];
    DerivedDigest {
        thumbnails: std::array::from_fn(|n| {
            let (image, (row, col)) = (n / 3, THUMBNAIL_TEXELS[n % 3]);
            rgb(&|c| thumbnails[[image, row, col, c]])
        }),
        readings: std::array::from_fn(|i| {
            let r = recon.image_set.images[i].readings.unwrap();
            let [xx, yy, xy] = r.gradient_structure_tensor_xx_yy_xy;
            [r.sharpness, xx, yy, xy]
        }),
        bitmaps: std::array::from_fn(|n| rgb(&|c| bitmaps[[36 * n, 12, 8, c]])),
        references: std::array::from_fn(|n| references[36 * n]),
    }
}

/// The digest of the AVX2 encoder path, which an x86 CPU with AVX2 takes:
/// computed on x86-64 Windows.
const DIGEST_AVX2: DerivedDigest = DerivedDigest {
    thumbnails: [
        [129, 109, 78],
        [132, 86, 80],
        [122, 102, 73],
        [127, 108, 78],
        [99, 63, 59],
        [107, 88, 64],
        [109, 92, 66],
        [72, 87, 121],
        [105, 87, 63],
        [90, 75, 54],
        [83, 98, 136],
        [110, 93, 69],
        [112, 95, 69],
        [71, 87, 121],
        [145, 121, 89],
    ],
    readings: [
        [0.0014819059, 8.196122e-5, 0.00032258782, 4.109168e-7],
        [0.0022913304, 0.00018535783, 0.00039324723, -9.206131e-7],
        [0.0016041422, 9.7398035e-5, 0.00031975744, -9.906911e-7],
        [0.0022708257, 0.0001769497, 0.00040021038, 3.7516481e-6],
        [0.0015165627, 8.808131e-5, 0.00031664356, -1.0469641e-6],
    ],
    bitmaps: [
        [145, 118, 87],
        [89, 73, 57],
        [93, 78, 56],
        [98, 82, 59],
        [79, 65, 49],
        [99, 83, 60],
        [152, 129, 96],
        [131, 110, 80],
        [88, 73, 52],
        [99, 82, 57],
        [65, 87, 66],
        [120, 74, 69],
        [119, 76, 70],
        [69, 79, 108],
        [69, 83, 118],
    ],
    references: [2, 1, 3, 3, 3, 1, 1, 0, 1, 1, 0, 0, 0, 3, 3],
};

/// The digest of the scalar encoder path, which every other CPU takes:
/// computed on x86-64 Windows with `jpeg-encoder`'s `simd` feature off. Its
/// sampled texels and references are the AVX2 path's, and its readings
/// differ from them in the fourth significant digit.
const DIGEST_SCALAR: DerivedDigest = DerivedDigest {
    thumbnails: [
        [129, 109, 78],
        [132, 86, 80],
        [122, 102, 73],
        [127, 108, 78],
        [99, 63, 59],
        [107, 88, 64],
        [109, 92, 66],
        [72, 87, 121],
        [105, 87, 63],
        [90, 75, 54],
        [83, 98, 136],
        [110, 93, 69],
        [112, 95, 69],
        [71, 87, 121],
        [145, 121, 89],
    ],
    readings: [
        [0.0014817407, 8.195104e-5, 0.0003226103, 4.0607074e-7],
        [0.0022916205, 0.00018536896, 0.00039319132, -9.2496066e-7],
        [0.0016036866, 9.738611e-5, 0.0003197048, -9.828054e-7],
        [0.0022713316, 0.00017696018, 0.00040021195, 3.74318e-6],
        [0.001516891, 8.8088316e-5, 0.0003166438, -1.04725e-6],
    ],
    bitmaps: [
        [145, 118, 87],
        [89, 73, 57],
        [93, 78, 56],
        [98, 82, 59],
        [79, 65, 49],
        [99, 83, 60],
        [152, 129, 96],
        [131, 110, 80],
        [88, 73, 52],
        [99, 82, 57],
        [65, 87, 66],
        [120, 74, 69],
        [119, 76, 70],
        [69, 79, 108],
        [69, 83, 118],
    ],
    references: [2, 1, 3, 3, 3, 1, 1, 0, 1, 1, 0, 0, 0, 3, 3],
};

/// Whether two images' readings agree as closely as decodings of one JPEG
/// that differ by a level at some samples make them: the sharpness, `Exx`
/// and `Eyy` within 1% of each other, and `Exy`, which is near 0, within 1%
/// of `√(Exx · Eyy)`.
///
/// A difference of one level at every sample bounds the change only loosely
/// (the square root of `Exx` by `1/255` of a grey unit, about half its value
/// here), since the decoders differ at few samples. Measured on the default
/// scene, `jpeg-decoder`'s scalar path and its SSSE3 one, which differ by one
/// level at some samples, give readings within `6.6 · 10⁻⁴` of each other,
/// and `Exy` within `5 · 10⁻⁵` of `√(Exx · Eyy)`; 1% is fifteen times that.
/// The two encoder paths' readings differ by less, at most `3 · 10⁻⁴`, so
/// the readings stored for the two paths agree within this tolerance too.
fn readings_within_one_level(got: [f32; 4], want: [f32; 4]) -> bool {
    let [got, want] = [got, want].map(|r| r.map(f64::from));
    let near = |k: usize| (got[k] - want[k]).abs() <= 0.01 * want[k].abs();
    let scale = (want[1] * want[2]).max(0.0).sqrt();
    near(0) && near(1) && near(2) && (got[3] - want[3]).abs() <= 0.01 * scale
}

#[test]
fn the_photograph_derived_columns_match_the_values_stored_for_the_encoders_path() {
    let got = digest(recon());
    let want = if super::super::tests::encoder_takes_avx2() {
        &DIGEST_AVX2
    } else {
        &DIGEST_SCALAR
    };
    let within_a_level =
        |a: [u8; 3], b: [u8; 3]| a.iter().zip(&b).all(|(x, y)| x.abs_diff(*y) <= 1);
    for (n, (a, b)) in got.thumbnails.iter().zip(&want.thumbnails).enumerate() {
        assert!(
            within_a_level(*a, *b),
            "thumbnail texel {n}: {a:?} against {b:?}"
        );
    }
    for (i, (a, b)) in got.readings.iter().zip(&want.readings).enumerate() {
        assert!(
            readings_within_one_level(*a, *b),
            "image {i}: {a:?} against {b:?}"
        );
    }
    // The reference-view rule picks between two near-equal views of a point
    // differently when the pixels differ by a level: of the 524 points, point
    // 359 changes its pick between the two encoder paths, and point 382
    // between the x86 decoder's SSSE3 path and its scalar one. Neither is
    // sampled, so on those paths every sampled pick agrees; the one sampled
    // point allowed to differ is for a path not measured, such as arm64's
    // decoder. A point whose pick differs has the other view's bitmap, so its
    // texel is not compared, and every other one's is within a level.
    let mut picks_differ = 0;
    for n in 0..15 {
        let (a, b) = (got.bitmaps[n], want.bitmaps[n]);
        if got.references[n] != want.references[n] {
            picks_differ += 1;
            continue;
        }
        assert!(
            within_a_level(a, b),
            "bitmap {}: {a:?} against {b:?}",
            36 * n
        );
    }
    assert!(
        picks_differ <= 1,
        "{picks_differ} sampled points changed their pick"
    );
}

#[test]
fn write_workspace_reports_its_phases_stops_when_cancelled_and_refuses_a_bad_directory() {
    use crate::progress::Event;
    use std::sync::Mutex;

    let events = Mutex::new(Vec::new());
    let sink = |event: Event<'_>| match event {
        Event::Enter { phase, depth: 0 } => events.lock().unwrap().push(format!("enter {phase}")),
        Event::Count {
            done,
            total: Some(total),
            unit,
        } if done == total && (unit == "points" || unit == "files") => {
            events.lock().unwrap().push(format!("{done} {unit}"))
        }
        _ => {}
    };
    let tmp = tempfile::tempdir().unwrap();
    scene()
        .write_workspace(&tmp.path().join("demo"), &Progress::to(&sink))
        .unwrap();
    let events = events.into_inner().unwrap();
    let entered: Vec<&str> = events
        .iter()
        .filter(|e| e.starts_with("enter "))
        .map(String::as_str)
        .collect();
    assert_eq!(
        entered,
        ["enter points", "enter patch bitmaps", "enter write"]
    );
    assert!(events.iter().any(|e| e.ends_with(" points")), "{events:?}");
    assert_eq!(events.last().map(String::as_str), Some("7 files"));

    // Cancelled, nothing is written.
    let cancelled = std::sync::atomic::AtomicBool::new(true);
    let dir = tmp.path().join("cancelled");
    let result = scene().write_workspace(&dir, &Progress::none().cancelled_by(&cancelled));
    assert!(
        matches!(result, Err(DemoSceneError::Cancelled)),
        "{result:?}"
    );
    assert!(!dir.exists());
    assert!(matches!(
        scene().reconstruction(&Progress::none().cancelled_by(&cancelled)),
        Err(DemoSceneError::Cancelled)
    ));

    // A file at the path, a directory inside a workspace, and a path under a
    // file are refused with an error.
    let file = tmp.path().join("a_file");
    std::fs::write(&file, b"x").unwrap();
    let result = scene().write_workspace(&file, &Progress::none());
    assert!(
        matches!(result, Err(DemoSceneError::Directory(_))),
        "{result:?}"
    );
    let result = scene().write_workspace(&file.join("below"), &Progress::none());
    assert!(matches!(result, Err(DemoSceneError::Io(_))), "{result:?}");
    let result =
        scene().write_workspace(&tmp.path().join("demo").join("nested"), &Progress::none());
    assert!(
        matches!(result, Err(DemoSceneError::Directory(_))),
        "{result:?}"
    );
}
