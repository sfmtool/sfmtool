// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The thumbnails the viewer draws: a node's own, never the value's.
//!
//! See `specs/gui/multi-panel-image-browser.md` § "Thumbnail loading". A
//! frustum's far plane, an Image Browser cell and its colour barcode, and a
//! Track View row each show a 128 x 128 picture of an image. When the file
//! carries a thumbnail column those are its rows, and the display column here
//! is the image set's own [`Arc`], shared rather than copied. When it does
//! not, [`DisplayThumbnails::build`] fills every row in as part of the open
//! ([`crate::state::open`]): from the image's `.sift` first, which already holds
//! the row reduced, from the photograph second (from its coarsest mip level
//! file when it is out of core and that file is current, read through the
//! viewer's photograph cache, and from a whole decode when it is in core), and
//! as a flat grey placeholder for an image neither can supply.
//!
//! **A row can wait for a mipmap.** An out-of-core photograph whose mipmap is
//! not current gets the grey placeholder at open, since the open neither
//! makes a mipmap nor decodes such a photograph whole; the column remembers
//! the row as awaiting that photograph. Once *Build mipmaps* has answered the
//! photograph, [`crate::state::thumbnail_fills`] reads its row on the rayon
//! pool and gives the node a column with the row filled in
//! ([`DisplayThumbnails::with_rows`]), a new value that shares every other
//! row, so the panels that keep what they read can tell which rows changed.
//!
//! **Display thumbnails belong to the node.** They are held on the
//! [`crate::scene::SceneNode`] beside its history and never enter an
//! `ImageSet`, so nothing built or filled in here can reach a save or a
//! content hash: a file opened without thumbnails is saved without them,
//! whatever was drawn.
//!
//! **They are keyed by image name.** Delete Image renumbers the image set
//! and Undo renumbers it back, so a column addressed by index would need
//! rebuilding on every such step; addressed by the workspace-relative name the
//! table carries, one column built when the node opens serves every version.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, OnceLock};

use ndarray::{Array3, Array4, ArrayView3, Axis};
use rayon::prelude::*;
use sfmtool_core::camera::PhotographCache;
use sfmtool_core::progress::{Cancelled, Progress};
use sfmtool_core::reconstruction::thumbnail::{
    display_thumbnail_row, photograph_thumbnail_row, NotCurrentMipmap, ThumbnailSource,
};
use sfmtool_core::{SfmrReconstruction, THUMBNAIL_SIZE};

#[cfg(test)]
pub(crate) mod tests;

pub(crate) use sfmtool_core::reconstruction::thumbnail::PLACEHOLDER_GREY;

/// Bytes in one RGB thumbnail row.
const ROW_BYTES: usize = THUMBNAIL_SIZE * THUMBNAIL_SIZE * 3;

/// One node's display thumbnails.
///
/// A fill ([`Self::with_rows`]) makes a new value that shares everything
/// with the old one but its count of fills: the rows, the names, the rows
/// awaiting a mipmap and the filled rows are all behind [`Arc`]s, and a
/// value sees only the fills up to its own count.
#[derive(Clone)]
pub(crate) struct DisplayThumbnails {
    /// The rows: the file's own column, or the one [`Self::build`] filled in.
    column: Arc<Array4<u8>>,
    /// What every value of this column shares.
    shared: Arc<Shared>,
    /// The fill this value sees up to: 0 for the column as the open built
    /// it, otherwise the number [`Self::with_rows`] gave the latest fill it
    /// made.
    fills: u64,
}

/// What every value of one display column shares: written when the column
/// is built, but for the filled rows, each of which is written once.
struct Shared {
    /// Row of each image name the column was built for.
    rows_by_name: HashMap<String, usize>,
    /// The rows drawn as the placeholder while their out-of-core
    /// photograph's mipmap is made, by the photograph's path
    /// (`workspace_dir` joined with the image name), as the open left them.
    awaiting: HashMap<PathBuf, usize>,
    /// Per row of a column with rows awaiting a mipmap, the row filled in
    /// with the number of the fill that filled it, set once; empty for a
    /// column with none.
    filled: Vec<OnceLock<(u64, Arc<Array3<u8>>)>>,
    /// The number of the latest fill made of this column, by any value.
    fills_made: AtomicU64,
}

/// Where the rows of a built column came from, for the open's Action Log note.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub(crate) struct BuiltFrom {
    /// Rows read from the image's verified `.sift`.
    pub(crate) sift: usize,
    /// Rows decoded and resized from the photograph.
    pub(crate) photographs: usize,
    /// Rows drawn as the placeholder until their out-of-core photograph's
    /// mipmap is made.
    pub(crate) awaiting: usize,
    /// Rows neither could supply, which show the grey placeholder.
    pub(crate) placeholders: usize,
}

impl DisplayThumbnails {
    /// The display column of a reconstruction that carries thumbnails: its
    /// own column, shared. `None` when it carries none.
    pub(crate) fn embedded(recon: &SfmrReconstruction) -> Option<Arc<Self>> {
        let column = recon.image_set.thumbnails_y_x_rgb.as_ref()?;
        Some(Arc::new(Self::new(
            names_to_rows(recon),
            Arc::clone(column),
            HashMap::new(),
        )))
    }

    /// A column of `column`'s rows, named by `rows_by_name`, of which the
    /// rows of `awaiting` await their photograph's mipmap.
    fn new(
        rows_by_name: HashMap<String, usize>,
        column: Arc<Array4<u8>>,
        awaiting: HashMap<PathBuf, usize>,
    ) -> Self {
        let rows = if awaiting.is_empty() {
            0
        } else {
            column.shape()[0]
        };
        Self {
            column,
            shared: Arc::new(Shared {
                rows_by_name,
                awaiting,
                filled: (0..rows).map(|_| OnceLock::new()).collect(),
                fills_made: AtomicU64::new(0),
            }),
            fills: 0,
        }
    }

    /// Build a row for every image of `recon`, which carries no thumbnails.
    ///
    /// Each row is the image's `.sift` thumbnail when a `.sift` verifiably
    /// belongs to it
    /// ([`verified_sift_thumbnail`](sfmtool_core::reconstruction::thumbnail::verified_sift_thumbnail)),
    /// otherwise its photograph resized by area averaging, otherwise the flat
    /// [`PLACEHOLDER_GREY`]
    /// ([`display_thumbnail_row`]). An out-of-core photograph whose mipmap is
    /// current is read from its coarsest mip level file through
    /// `photographs`; one whose mipmap is not gets the placeholder and is
    /// remembered as awaiting its mipmap
    /// ([`NotCurrentMipmap::Placeholder`]), since the open makes no mipmap
    /// and decodes no such photograph whole. The rows are built in parallel;
    /// `progress` gets an `images` count as each lands, and is polled for
    /// cancellation before each one and while a photograph is read.
    ///
    /// `Ok(None)` when there is nothing to show: the reconstruction has no
    /// images, or not one row came from either source or awaits a mipmap.
    /// The node then has no display column, the frustums draw their outlines
    /// without image quads, and the panels draw their placeholder.
    pub(crate) fn build(
        recon: &SfmrReconstruction,
        photographs: &PhotographCache,
        progress: &Progress<'_>,
    ) -> Result<(Option<Arc<Self>>, BuiltFrom), Cancelled> {
        let images = &recon.image_set.images;
        let total = images.len();
        let landed = AtomicUsize::new(0);
        let rows: Vec<(Vec<u8>, ThumbnailSource)> = (0..total)
            .into_par_iter()
            .map(|index| {
                if progress.is_cancelled() {
                    return (Vec::new(), ThumbnailSource::Placeholder);
                }
                let Ok(row) = build_row(recon, index, photographs, progress) else {
                    return (Vec::new(), ThumbnailSource::Placeholder);
                };
                let n = landed.fetch_add(1, Ordering::Relaxed) + 1;
                progress.count(n as u64, Some(total as u64), "images");
                row
            })
            .collect();
        progress.check_cancel()?;

        let mut from = BuiltFrom::default();
        let mut awaiting = HashMap::new();
        let mut flat = Vec::with_capacity(total * ROW_BYTES);
        for (index, (pixels, source)) in rows.into_iter().enumerate() {
            match source {
                ThumbnailSource::Sift => from.sift += 1,
                ThumbnailSource::Photograph => from.photographs += 1,
                ThumbnailSource::AwaitingMipmap => {
                    from.awaiting += 1;
                    awaiting.insert(recon.workspace_dir.join(&images[index].name), index);
                }
                ThumbnailSource::Placeholder => from.placeholders += 1,
            }
            flat.extend_from_slice(&pixels);
        }
        if from.placeholders == total {
            return Ok((None, from));
        }
        let column = Array4::from_shape_vec((total, THUMBNAIL_SIZE, THUMBNAIL_SIZE, 3), flat)
            .expect("one THUMBNAIL_SIZE x THUMBNAIL_SIZE x 3 row per image");
        let display = Self::new(names_to_rows(recon), Arc::new(column), awaiting);
        Ok((Some(Arc::new(display)), from))
    }

    /// The thumbnail row of the image named `name`, or `None` for a name the
    /// column was not built for.
    pub(crate) fn row(&self, name: &str) -> Option<ArrayView3<'_, u8>> {
        let &row = self.shared.rows_by_name.get(name)?;
        if let Some((_, filled)) = self.filled(row) {
            return Some(filled.view());
        }
        Some(self.column.index_axis(Axis(0), row))
    }

    /// Row `row` filled in, with the number of the fill that filled it, when
    /// a fill this value sees filled it.
    fn filled(&self, row: usize) -> Option<&(u64, Arc<Array3<u8>>)> {
        let filled = self.shared.filled.get(row)?.get()?;
        (filled.0 <= self.fills).then_some(filled)
    }

    /// The photographs whose rows await their mipmap.
    pub(crate) fn awaiting(&self) -> impl Iterator<Item = &Path> {
        self.shared
            .awaiting
            .iter()
            .filter(|&(_, &row)| self.filled(row).is_none())
            .map(|(path, _)| path.as_path())
    }

    /// Whether the row of the photograph at `path` awaits its mipmap.
    pub(crate) fn awaits(&self, path: &Path) -> bool {
        self.shared
            .awaiting
            .get(path)
            .is_some_and(|&row| self.filled(row).is_none())
    }

    /// This column with the row of each photograph of `rows` that awaits its
    /// mipmap replaced by the row given for it, which then awaits nothing;
    /// `None` when no photograph of `rows` awaits here.
    ///
    /// A new value, sharing everything with this one but its count of
    /// fills, so a fill copies none of the column, the names or the rows
    /// filled before; a panel that kept what it read of the old value finds
    /// the rows that changed with [`Self::filled_after`]. The old value
    /// still draws its rows as they were.
    pub(crate) fn with_rows(&self, rows: &[(PathBuf, Arc<Array3<u8>>)]) -> Option<Self> {
        if !rows.iter().any(|(path, _)| self.awaits(path)) {
            return None;
        }
        let fills = self.shared.fills_made.fetch_add(1, Ordering::Relaxed) + 1;
        for (path, row) in rows {
            if !self.awaits(path) {
                continue;
            }
            let index = self.shared.awaiting[path];
            // Set once: the node's value is the only one filled, so a row it
            // sees awaiting has not been set.
            let _ = self.shared.filled[index].set((fills, Arc::clone(row)));
        }
        Some(Self {
            column: Arc::clone(&self.column),
            shared: Arc::clone(&self.shared),
            fills,
        })
    }

    /// The fill this value sees up to, which only grows from one value of a
    /// column to the next: what a panel that kept what it read compares
    /// against [`Self::filled_after`].
    pub(crate) fn fills(&self) -> u64 {
        self.fills
    }

    /// Whether `self` and `other` are the same built column, `other` with
    /// rows filled in or not: the only difference between them is then the
    /// rows [`Self::filled_after`] names.
    pub(crate) fn same_build(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.column, &other.column)
    }

    /// Whether the row of the image named `name` was filled in by a fill this
    /// value sees that came after fill `fills`.
    pub(crate) fn filled_after(&self, fills: u64, name: &str) -> bool {
        self.shared
            .rows_by_name
            .get(name)
            .and_then(|&row| self.filled(row))
            .is_some_and(|&(at, _)| at > fills)
    }

    /// How many rows the column has.
    #[cfg(test)]
    pub(crate) fn len(&self) -> usize {
        self.column.shape()[0]
    }
}

/// The row a photograph whose display row awaited its mipmap gets once
/// *Build mipmaps* has answered it: read from its coarsest mip level file
/// when its mipmap is current, and decoded whole when it is not (its build
/// failed, or its levels were held only in memory and let go), or the grey
/// placeholder when neither can be read
/// ([`photograph_thumbnail_row`] with [`NotCurrentMipmap::DecodeOriginal`]).
/// Makes no mipmap. Runs off the GUI thread: on the open's worker, or on the
/// rayon pool.
pub(crate) fn filled_row(path: &Path, photographs: &PhotographCache) -> Arc<Array3<u8>> {
    let (pixels, source) = photograph_thumbnail_row(
        path,
        photographs,
        NotCurrentMipmap::DecodeOriginal,
        &Progress::none(),
    )
    .unwrap_or_else(|Cancelled| {
        (
            vec![PLACEHOLDER_GREY; ROW_BYTES],
            ThumbnailSource::Placeholder,
        )
    });
    if source == ThumbnailSource::Placeholder {
        log::warn!(
            "Display thumbnail: the photograph {} could not be read; showing a placeholder",
            path.display()
        );
    }
    let row = Array3::from_shape_vec((THUMBNAIL_SIZE, THUMBNAIL_SIZE, 3), pixels)
        .expect("one THUMBNAIL_SIZE x THUMBNAIL_SIZE x 3 row");
    Arc::new(row)
}

/// Image name to row, over `recon`'s image set as it stands.
fn names_to_rows(recon: &SfmrReconstruction) -> HashMap<String, usize> {
    recon
        .image_set
        .images
        .iter()
        .enumerate()
        .map(|(row, image)| (image.name.clone(), row))
        .collect()
}

/// Image `index`'s row: the verified `.sift` copy, the photograph, or the grey,
/// awaiting the photograph's mipmap or for good ([`display_thumbnail_row`],
/// which `sfm web-export` fills a column with too).
fn build_row(
    recon: &SfmrReconstruction,
    index: usize,
    photographs: &PhotographCache,
    progress: &Progress<'_>,
) -> Result<(Vec<u8>, ThumbnailSource), Cancelled> {
    let (pixels, source) = display_thumbnail_row(
        recon,
        index,
        photographs,
        NotCurrentMipmap::Placeholder,
        progress,
    )?;
    if source == ThumbnailSource::Placeholder {
        log::warn!(
            "Display thumbnail: neither a .sift nor the photograph {} could be read; \
             showing a placeholder",
            recon
                .workspace_dir
                .join(&recon.image_set.images[index].name)
                .display()
        );
    }
    Ok((pixels, source))
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
    let name = &recon.image_set.images.get(index)?.name;
    if let Some(row) = display.and_then(|display| display.row(name)) {
        return Some(row);
    }
    let column = recon.image_set.thumbnails_y_x_rgb.as_ref()?;
    (index < column.shape()[0]).then(|| column.index_axis(Axis(0), index))
}
