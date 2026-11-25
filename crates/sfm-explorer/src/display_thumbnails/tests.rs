// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Display thumbnails: the file's own column shared, and for a file without
//! one every row built at once, from the `.sift` first, the photograph second
//! and the grey placeholder last.

use std::path::Path;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;

use ndarray::{Array2, Array3, Array4};
use sfmtool_core::camera::PhotographCache;
use sfmtool_core::progress::Progress;
use sfmtool_core::{ObservationSource, SfmrReconstruction, THUMBNAIL_SIZE};

use super::{row_for, BuiltFrom, DisplayThumbnails, PLACEHOLDER_GREY};
use crate::scene::SceneNode;

/// The flat colour photograph `image` is written as.
pub(crate) fn colour_of(image: usize) -> [u8; 3] {
    [
        (image * 30 % 256) as u8,
        200,
        (255 - image * 20 % 256) as u8,
    ]
}

/// The flat grey the `.sift` thumbnail of `image` is written as, which no
/// photograph's colour matches.
pub(crate) fn sift_grey_of(image: usize) -> u8 {
    10 + image as u8
}

/// The demo reconstruction in workspace `dir`, with no thumbnails, and a flat
/// colour PNG photograph under `images/` for every image not in `missing`.
pub(crate) fn workspace_with_photographs(dir: &Path, missing: &[usize]) -> SfmrReconstruction {
    let mut recon = SfmrReconstruction::demo(16);
    recon.workspace_dir = dir.to_path_buf();
    std::fs::create_dir_all(dir.join("images")).unwrap();
    for (i, image) in recon.image_set.images.iter_mut().enumerate() {
        image.name = format!("images/photo_{i:02}.png");
        if missing.contains(&i) {
            continue;
        }
        let [r, g, b] = colour_of(i);
        let photo = image::RgbImage::from_pixel(96, 54, image::Rgb([r, g, b]));
        photo.save(dir.join(&image.name)).unwrap();
    }
    assert!(recon.image_set.thumbnails_y_x_rgb.is_none());
    recon
}

/// Write a `.sift` beside each image in `images`, its thumbnail flat
/// [`sift_grey_of`], and record its content hash in `recon`'s
/// `sift_content_hashes` when `images` names it in `verified` too. A `.sift`
/// whose hash is not recorded is one the reconstruction does not vouch for.
pub(crate) fn write_sift_thumbnails(
    recon: &mut SfmrReconstruction,
    images: &[usize],
    verified: &[usize],
) {
    for &image in images {
        let path = recon.sift_path_for_image(image);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        let data = sfmtool_sift_format::SiftData {
            feature_tool_metadata: sfmtool_sift_format::FeatureToolMetadata {
                feature_tool: "test".into(),
                feature_type: "sift".into(),
                feature_options: serde_json::json!({"image_to_gray": {"formula": "G"}}),
            },
            metadata: sfmtool_sift_format::SiftMetadata {
                version: sfmtool_sift_format::SIFT_FORMAT_VERSION,
                image_name: recon.image_set.images[image].name.clone(),
                image_file_xxh128: "0".repeat(32),
                image_file_size: 1,
                image_width: 96,
                image_height: 54,
                feature_count: 0,
                sharpness: 0.0,
                gradient_structure_tensor_xx_yy_xy: [0.0; 3],
            },
            content_hash: sfmtool_sift_format::SiftContentHash::default(),
            positions_xy: Array2::zeros((0, 2)),
            affine_shapes: Array3::zeros((0, 2, 2)),
            descriptors: Array2::zeros((0, 128)),
            thumbnail_y_x_rgb: Array3::from_elem(
                (THUMBNAIL_SIZE, THUMBNAIL_SIZE, 3),
                sift_grey_of(image),
            ),
        };
        sfmtool_sift_format::write_sift(&path, &data, 3).unwrap();
        if !verified.contains(&image) {
            continue;
        }
        let (_, _, hash) = sfmtool_sift_format::read_sift_metadata(&path).unwrap();
        let digest = u128::from_str_radix(&hash.content_xxh128, 16).unwrap();
        if let ObservationSource::SiftFiles {
            sift_content_hashes,
            ..
        } = &mut recon.point_set.observations
        {
            sift_content_hashes[image] = digest.to_be_bytes();
        }
    }
}

pub(crate) fn is_flat(row: ndarray::ArrayView3<'_, u8>, rgb: [u8; 3]) -> bool {
    row.shape() == [THUMBNAIL_SIZE, THUMBNAIL_SIZE, 3]
        && row
            .as_slice()
            .expect("a contiguous row")
            .chunks(3)
            .all(|p| p == rgb)
}

/// Build without a sink, which is what a caller that wants the rows and not
/// the report does.
fn build(recon: &SfmrReconstruction) -> (Option<Arc<DisplayThumbnails>>, BuiltFrom) {
    DisplayThumbnails::build(
        recon,
        &PhotographCache::new(Default::default()),
        &Progress::none(),
    )
    .expect("nothing cancels it")
}

#[test]
fn a_file_with_thumbnails_shares_its_own_column() {
    let mut recon = SfmrReconstruction::demo(16);
    let n = recon.image_set.images.len();
    let column = Arc::new(Array4::<u8>::from_shape_fn(
        (n, THUMBNAIL_SIZE, THUMBNAIL_SIZE, 3),
        |(i, _, _, c)| (i * 10 + c) as u8,
    ));
    recon.image_set.thumbnails_y_x_rgb = Some(Arc::clone(&column));

    let node = SceneNode::demo(recon);
    let display = node.display_thumbnails.as_ref().expect("the file's column");
    assert_eq!(display.len(), n);
    // Shared rather than copied: the row is a view into the image set's own array.
    let row = display.row("image_003.jpg").expect("a row");
    assert_eq!(
        row.as_ptr(),
        column.index_axis(ndarray::Axis(0), 3).as_ptr()
    );
}

/// Each row comes from the first source that can supply it: a `.sift` the
/// reconstruction vouches for, then the photograph, then the grey.
#[test]
fn a_row_comes_from_the_sift_first_the_photograph_second_and_the_grey_last() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path().to_path_buf();
    let mut recon = workspace_with_photographs(&dir, &[2, 5]);
    // 0, 1 and 2 have a vouched-for .sift, 2 without its photograph; 3 has a
    // .sift the reconstruction does not vouch for; 5 has neither.
    write_sift_thumbnails(&mut recon, &[0, 1, 2, 3], &[0, 1, 2]);
    let n = recon.image_set.images.len();

    let (display, from) = build(&recon);
    let display = display.expect("rows to show");
    assert_eq!(
        from,
        BuiltFrom {
            sift: 3,
            photographs: n - 4,
            awaiting: 0,
            placeholders: 1,
        }
    );
    let row = |i: usize| display.row(&format!("images/photo_{i:02}.png")).unwrap();
    for i in [0, 1, 2] {
        let g = sift_grey_of(i);
        assert!(is_flat(row(i), [g, g, g]), "row {i} is its .sift copy");
    }
    assert!(
        is_flat(row(3), colour_of(3)),
        "an unvouched .sift is passed over"
    );
    assert!(
        is_flat(row(4), colour_of(4)),
        "row 4 is its photograph's resize"
    );
    assert!(is_flat(row(5), [PLACEHOLDER_GREY; 3]));
}

#[test]
fn a_capture_with_nothing_to_build_from_has_no_display_column() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path().to_path_buf();
    let all: Vec<usize> = (0..8).collect();
    let recon = workspace_with_photographs(&dir, &all);
    let (display, from) = build(&recon);
    assert!(display.is_none());
    assert_eq!(from.placeholders, recon.image_set.images.len());
    // Every panel falls back to its placeholder.
    assert!(row_for(None, &recon, 0).is_none());
}

#[test]
fn a_build_that_is_cancelled_says_so() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path().to_path_buf();
    let recon = workspace_with_photographs(&dir, &[]);
    let stop = AtomicBool::new(true);
    let progress = Progress::none().cancelled_by(&stop);
    let cache = PhotographCache::new(Default::default());
    assert!(DisplayThumbnails::build(&recon, &cache, &progress).is_err());
}

#[test]
fn rows_follow_the_image_name_across_a_renumbering() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path().to_path_buf();
    let recon = workspace_with_photographs(&dir, &[]);
    let display = build(&recon).0.expect("photographs");

    // Delete Image renumbers the image set: what was image 3 is image 2 now, and
    // the column built for the first version still draws it.
    let subset = recon
        .subset_by_image_indices(&[0, 1, 3, 4, 5, 6, 7], false)
        .unwrap();
    let row = row_for(Some(&display), &subset, 2).expect("a row");
    assert!(is_flat(row, colour_of(3)));
}

#[test]
fn an_image_the_column_was_not_built_for_falls_back_to_the_values_column() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path().to_path_buf();
    let recon = workspace_with_photographs(&dir, &[]);
    let display = build(&recon).0.expect("photographs");

    let mut other = recon.clone();
    other.image_set.images[5].name = "elsewhere/new.png".into();
    let n = other.image_set.images.len();
    other.image_set.thumbnails_y_x_rgb = Some(Arc::new(Array4::from_elem(
        (n, THUMBNAIL_SIZE, THUMBNAIL_SIZE, 3),
        7u8,
    )));
    assert!(is_flat(
        row_for(Some(&display), &other, 5).unwrap(),
        [7, 7, 7]
    ));
    // The images the column knows are still drawn from it.
    assert!(is_flat(
        row_for(Some(&display), &other, 4).unwrap(),
        colour_of(4)
    ));
}

/// In a workspace whose `mipmap_options` put its 300 x 200 photographs out of
/// core: the row of one whose mipmap is current is made from its coarsest
/// mip level file, level 1, read through the viewer's cache; the row of one
/// whose mipmap is not is the grey placeholder, awaiting that photograph's
/// mipmap, and the open neither decodes it whole nor makes its mipmap.
#[test]
fn an_out_of_core_photograph_is_read_from_its_coarsest_mip_level_file_when_it_is_current() {
    use crate::state::build_mipmaps::tests::{mipmap_options, out_of_core_photographs_of};
    use sfmtool_core::camera::image::ImageU8;
    use sfmtool_core::camera::mipmap::{build_mipmaps, mipmap_location};
    use sfmtool_core::camera::PHOTOGRAPH_TILE_SIZE;
    use sfmtool_core::reconstruction::thumbnail::thumbnail_from_rgb;

    let tmp = tempfile::tempdir().unwrap();
    let mut recon = SfmrReconstruction::demo(16);
    let paths = out_of_core_photographs_of(&mut recon, tmp.path(), &[0, 1]);
    let built = build_mipmaps(
        &paths[0],
        None,
        &mipmap_options(),
        PHOTOGRAPH_TILE_SIZE,
        &Progress::none(),
        &mut |_| {},
    )
    .unwrap();
    let thumbnail = |image: &ImageU8| {
        thumbnail_from_rgb(
            image.data(),
            image.width() as usize,
            image.height() as usize,
        )
    };
    let coarsest = built.level(1).unwrap().decode().unwrap();
    let cache = PhotographCache::new(Default::default());
    let (display, from) = DisplayThumbnails::build(&recon, &cache, &Progress::none()).unwrap();
    let display = display.expect("two photographs");
    assert_eq!((from.photographs, from.awaiting), (1, 1));
    let row = |i: usize| -> Vec<u8> {
        let row = display.row(&recon.image_set.images[i].name).unwrap();
        row.iter().copied().collect()
    };
    let whole_0 = thumbnail(&ImageU8::read_rgb(&paths[0]).unwrap());
    assert!(row(0) == thumbnail(&coarsest));
    assert!(
        row(0) != whole_0,
        "the coarsest level's row is not level 0's"
    );
    assert!(is_flat(
        display.row(&recon.image_set.images[1].name).unwrap(),
        [PLACEHOLDER_GREY; 3]
    ));
    let awaiting: Vec<&Path> = display.awaiting().collect();
    assert_eq!(
        awaiting,
        [recon.workspace_dir.join(&recon.image_set.images[1].name)]
    );
    let level_1 = mipmap_location(&paths[1]).unwrap().level_path(1, 95);
    assert!(!level_1.exists(), "the open made a mipmap");
    let stats = cache.stats();
    assert_eq!((stats.decodes, stats.mipmaps_made), (0, 0));
}

/// A display column over `recon`'s images, each row flat [`colour_of`] its
/// image, but the rows of `awaiting`, which are grey and await their
/// photographs (`workspace_dir` joined with the image name), as an open
/// leaves the rows of out-of-core photographs whose mipmaps are not current.
pub(crate) fn column_awaiting(
    recon: &SfmrReconstruction,
    awaiting: &[usize],
) -> Arc<DisplayThumbnails> {
    let images = &recon.image_set.images;
    let column = Array4::from_shape_fn(
        (images.len(), THUMBNAIL_SIZE, THUMBNAIL_SIZE, 3),
        |(i, _, _, c)| {
            if awaiting.contains(&i) {
                PLACEHOLDER_GREY
            } else {
                colour_of(i)[c]
            }
        },
    );
    Arc::new(DisplayThumbnails::new(
        super::names_to_rows(recon),
        Arc::new(column),
        awaiting
            .iter()
            .map(|&i| (recon.workspace_dir.join(&images[i].name), i))
            .collect(),
    ))
}

/// A flat `rgb` row, as a fill brings back.
pub(crate) fn flat_row(rgb: [u8; 3]) -> Arc<Array3<u8>> {
    Arc::new(Array3::from_shape_fn(
        (THUMBNAIL_SIZE, THUMBNAIL_SIZE, 3),
        |(_, _, c)| rgb[c],
    ))
}

/// Filling rows in gives a new column in which only the rows that awaited
/// the photographs filled changed, each marked with the fill that changed
/// it, every other row shared with the column before; a fill of photographs
/// no row awaits gives nothing.
#[test]
fn filling_rows_in_replaces_only_the_rows_that_await_them() {
    let mut recon = SfmrReconstruction::demo(16);
    recon.workspace_dir = std::path::PathBuf::from("ws");
    let name = |i: usize| recon.image_set.images[i].name.clone();
    let path = |i: usize| recon.workspace_dir.join(name(i));
    let built = column_awaiting(&recon, &[2, 5]);

    assert!(built.with_rows(&[(path(3), flat_row([1, 2, 3]))]).is_none());
    let once = built
        .with_rows(&[
            (path(2), flat_row([1, 2, 3])),
            (path(3), flat_row([9, 9, 9])),
        ])
        .expect("row 2 awaited");
    assert!(is_flat(once.row(&name(2)).unwrap(), [1, 2, 3]));
    assert!(is_flat(once.row(&name(3)).unwrap(), colour_of(3)));
    assert!(is_flat(once.row(&name(5)).unwrap(), [PLACEHOLDER_GREY; 3]));
    assert_eq!(once.awaiting().collect::<Vec<_>>(), [path(5).as_path()]);
    assert_eq!(once.fills(), 1);
    assert!(once.same_build(&built));
    // The fill copied nothing: the names, the awaiting rows and the filled
    // rows are shared, and the column before it still draws row 2 as it was.
    assert!(Arc::ptr_eq(&once.shared, &built.shared));
    assert!(is_flat(built.row(&name(2)).unwrap(), [PLACEHOLDER_GREY; 3]));
    assert!(built.awaits(&path(2)));
    assert!(!built.filled_after(0, &name(2)));
    assert_eq!(
        once.row(&name(3)).unwrap().as_ptr(),
        built.row(&name(3)).unwrap().as_ptr()
    );
    assert!(once.filled_after(0, &name(2)));
    assert!(!once.filled_after(0, &name(5)));

    let twice = once.with_rows(&[(path(5), flat_row([4, 5, 6]))]).unwrap();
    assert!(is_flat(twice.row(&name(5)).unwrap(), [4, 5, 6]));
    assert!(twice.filled_after(1, &name(5)));
    assert!(!twice.filled_after(1, &name(2)), "filled by the first fill");
    assert_eq!(twice.awaiting().count(), 0);
}
