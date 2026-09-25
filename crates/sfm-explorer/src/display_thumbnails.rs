// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The thumbnails the viewer draws: a node's own, never the value's.
//!
//! See `specs/gui/multi-panel-image-browser.md` § "Thumbnail loading". A
//! frustum's far plane, an Image Browser cell and its colour barcode, and a
//! Track View row each show a 128 x 128 picture of an image. When the file
//! carries a thumbnail column those are its rows, and the display column here
//! is the image table's own [`Arc`], shared rather than copied. When it does
//! not, [`DisplayThumbnails::build`] fills every row in as part of the open
//! ([`crate::state::open`]): from the image's `.sift` first, which already holds
//! the row reduced, from the photograph second, and as a flat grey placeholder
//! only for an image neither can supply.
//!
//! **Display thumbnails belong to the node.** They are held on the
//! [`crate::scene::SceneNode`] beside its history and never enter an
//! `ImageTable`, so nothing built here can reach a save or a content hash: a
//! file opened without thumbnails is saved without them, whatever was drawn.
//!
//! **They are keyed by image name.** Delete Image renumbers the image table
//! and Undo renumbers it back, so a column addressed by index would need
//! rebuilding on every such step; addressed by the workspace-relative name the
//! table carries, one column built when the node opens serves every version.

use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use ndarray::{Array4, ArrayView3, Axis};
use rayon::prelude::*;
use sfmtool_core::progress::{Cancelled, Progress};
use sfmtool_core::reconstruction::thumbnail::{display_thumbnail_row, ThumbnailSource};
use sfmtool_core::{SfmrReconstruction, THUMBNAIL_SIZE};

#[cfg(test)]
pub(crate) mod tests;

pub(crate) use sfmtool_core::reconstruction::thumbnail::PLACEHOLDER_GREY;

/// Bytes in one RGB thumbnail row.
const ROW_BYTES: usize = THUMBNAIL_SIZE * THUMBNAIL_SIZE * 3;

/// One node's display thumbnails.
pub struct DisplayThumbnails {
    /// Row of each image name the column was built for.
    rows_by_name: HashMap<String, usize>,
    /// The rows: the file's own column, or the one [`Self::build`] filled in.
    column: Arc<Array4<u8>>,
}

/// Where the rows of a built column came from, for the open's Action Log note.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub(crate) struct BuiltFrom {
    /// Rows read from the image's verified `.sift`.
    pub(crate) sift: usize,
    /// Rows decoded and resized from the photograph.
    pub(crate) photographs: usize,
    /// Rows neither could supply, which show the grey placeholder.
    pub(crate) placeholders: usize,
}

impl DisplayThumbnails {
    /// The display column of a reconstruction that carries thumbnails: its
    /// own column, shared. `None` when it carries none.
    pub(crate) fn embedded(recon: &SfmrReconstruction) -> Option<Arc<Self>> {
        let column = recon.image_table.thumbnails_y_x_rgb.as_ref()?;
        Some(Arc::new(Self {
            rows_by_name: names_to_rows(recon),
            column: Arc::clone(column),
        }))
    }

    /// Build a row for every image of `recon`, which carries no thumbnails.
    ///
    /// Each row is the image's `.sift` thumbnail when a `.sift` verifiably
    /// belongs to it
    /// ([`verified_sift_thumbnail`](sfmtool_core::reconstruction::thumbnail::verified_sift_thumbnail)),
    /// otherwise its photograph
    /// decoded and resized by area averaging, otherwise the flat
    /// [`PLACEHOLDER_GREY`]. The rows are built in parallel; `progress` gets an
    /// `images` count as each lands, and is polled for cancellation before
    /// each one.
    ///
    /// `Ok(None)` when there is nothing to show: the reconstruction has no
    /// images, or not one row came from either source. The node then has no
    /// display column, the frustums draw their outlines without image quads,
    /// and the panels draw their placeholder.
    pub(crate) fn build(
        recon: &SfmrReconstruction,
        progress: &Progress<'_>,
    ) -> Result<(Option<Arc<Self>>, BuiltFrom), Cancelled> {
        let images = &recon.image_table.images;
        let total = images.len();
        let landed = AtomicUsize::new(0);
        let rows: Vec<(Vec<u8>, ThumbnailSource)> = (0..total)
            .into_par_iter()
            .map(|index| {
                if progress.is_cancelled() {
                    return (Vec::new(), ThumbnailSource::Placeholder);
                }
                let row = build_row(recon, index);
                let n = landed.fetch_add(1, Ordering::Relaxed) + 1;
                progress.count(n as u64, Some(total as u64), "images");
                row
            })
            .collect();
        progress.check_cancel()?;

        let mut from = BuiltFrom::default();
        let mut flat = Vec::with_capacity(total * ROW_BYTES);
        for (pixels, source) in rows {
            match source {
                ThumbnailSource::Sift => from.sift += 1,
                ThumbnailSource::Photograph => from.photographs += 1,
                ThumbnailSource::Placeholder => from.placeholders += 1,
            }
            flat.extend_from_slice(&pixels);
        }
        if from.placeholders == total {
            return Ok((None, from));
        }
        let column = Array4::from_shape_vec((total, THUMBNAIL_SIZE, THUMBNAIL_SIZE, 3), flat)
            .expect("one THUMBNAIL_SIZE x THUMBNAIL_SIZE x 3 row per image");
        let display = Self {
            rows_by_name: names_to_rows(recon),
            column: Arc::new(column),
        };
        Ok((Some(Arc::new(display)), from))
    }

    /// The thumbnail row of the image named `name`, or `None` for a name the
    /// column was not built for.
    pub(crate) fn row(&self, name: &str) -> Option<ArrayView3<'_, u8>> {
        let &row = self.rows_by_name.get(name)?;
        Some(self.column.index_axis(Axis(0), row))
    }

    /// How many rows the column has.
    #[cfg(test)]
    pub(crate) fn len(&self) -> usize {
        self.column.shape()[0]
    }
}

/// Image name to row, over `recon`'s image table as it stands.
fn names_to_rows(recon: &SfmrReconstruction) -> HashMap<String, usize> {
    recon
        .image_table
        .images
        .iter()
        .enumerate()
        .map(|(row, image)| (image.name.clone(), row))
        .collect()
}

/// Image `index`'s row: the verified `.sift` copy, the photograph, or the grey
/// ([`display_thumbnail_row`], which `sfm web-export` fills a column with too).
fn build_row(recon: &SfmrReconstruction, index: usize) -> (Vec<u8>, ThumbnailSource) {
    let (pixels, source) = display_thumbnail_row(recon, index);
    if source == ThumbnailSource::Placeholder {
        log::warn!(
            "Display thumbnail: neither a .sift nor the photograph {} could be read; \
             showing a placeholder",
            recon
                .workspace_dir
                .join(&recon.image_table.images[index].name)
                .display()
        );
    }
    (pixels, source)
}

/// The row a panel draws for image `index` of `recon`, the value a node shows.
///
/// The node's display column `display` first, by the image's name. An image
/// that column was not built for falls back to `recon`'s own thumbnail column
/// when it carries one, so a version that gained an image from a file with
/// thumbnails still shows it. `None` means draw the placeholder: there is no
/// picture to show.
pub(crate) fn row_for<'a>(
    display: Option<&'a DisplayThumbnails>,
    recon: &'a SfmrReconstruction,
    index: usize,
) -> Option<ArrayView3<'a, u8>> {
    let name = &recon.image_table.images.get(index)?.name;
    if let Some(row) = display.and_then(|display| display.row(name)) {
        return Some(row);
    }
    let column = recon.image_table.thumbnails_y_x_rgb.as_ref()?;
    (index < column.shape()[0]).then(|| column.index_axis(Axis(0), index))
}
