// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The thumbnails the viewer draws: a node's own, never the value's.
//!
//! See `specs/gui/multi-panel-image-browser.md` § "Thumbnail loading". A
//! frustum's far plane, an Image Browser cell and its colour barcode, and a
//! Track View row each show a 128 x 128 picture of an image. When the file
//! carries a thumbnail column those are its rows, and the display column here
//! is the image table's own [`Arc`], shared rather than copied. When it does
//! not, the rows are built from the source photographs off the GUI thread, on
//! a worker pool of their own, and fill in as they finish.
//!
//! **Display thumbnails belong to the node.** They are held on the
//! [`crate::scene::SceneNode`] beside its history and never enter an
//! `ImageTable`, so nothing synthesised here can reach a save: a file opened
//! without thumbnails is saved without them, whatever was drawn.
//!
//! **They are keyed by image name.** Delete Image renumbers the image table
//! and Undo renumbers it back, so a column addressed by index would need
//! rebuilding on every such step; addressed by the workspace-relative name the
//! table carries, one column built when the node opens serves every version.
//!
//! **Synthesis is not a document operation.** It changes no value, costs no
//! version, and does not occupy the one background-task slot edits use
//! ([`crate::background`]), so a large capture never locks out editing while
//! its thumbnails are built.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, LazyLock, Mutex, OnceLock, Weak};
use std::time::{Duration, Instant};

use ndarray::{Array4, ArrayView3, Axis};
use sfmtool_core::reconstruction::thumbnail::thumbnail_from_rgb;
use sfmtool_core::{SfmrReconstruction, THUMBNAIL_SIZE};

#[cfg(test)]
pub(crate) mod tests;

/// The grey a row is filled with when its photograph cannot be read.
///
/// A neutral mid-grey rather than black: black reads as a dark photograph,
/// while a flat grey reads as "no picture here".
pub(crate) const PLACEHOLDER_GREY: u8 = 128;

/// Bytes in one RGB thumbnail row.
const ROW_BYTES: usize = THUMBNAIL_SIZE * THUMBNAIL_SIZE * 3;

/// The worker pool synthesis runs on, shared by every node.
///
/// Its own pool rather than rayon's global one, so a capture of thousands of
/// full-size photographs neither starves the kernels a background task runs
/// nor waits behind them. One thread is left for the GUI.
static POOL: LazyLock<rayon::ThreadPool> = LazyLock::new(|| {
    let threads = std::thread::available_parallelism()
        .map(|n| n.get().saturating_sub(1))
        .unwrap_or(1)
        .max(1);
    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .thread_name(|i| format!("display-thumbnails-{i}"))
        .build()
        .expect("a thread pool of at least one thread")
});

/// One node's display thumbnails.
pub struct DisplayThumbnails {
    /// Row of each image name the column was built for.
    rows_by_name: HashMap<String, usize>,
    /// Where the rows come from.
    source: Source,
}

enum Source {
    /// The file's own column: every row is final from the start.
    Embedded(Arc<Array4<u8>>),
    /// Rows built from the photographs, filling in.
    Synthesized(Synthesis),
}

/// The state a synthesis shares with its workers.
struct Synthesis {
    /// The photograph of each row.
    paths: Vec<PathBuf>,
    /// Each row once it is final: the photograph's resize, or the flat grey
    /// placeholder for one that could not be read.
    rows: Vec<OnceLock<Box<[u8]>>>,
    /// Whether a worker has taken each row, so no row is built twice.
    claimed: Vec<AtomicBool>,
    /// The next row in Image Browser order not yet handed out.
    next: AtomicUsize,
    /// Rows asked for ahead of order (the cells on screen), taken last-first.
    urgent: Mutex<Vec<usize>>,
    /// How many rows are final.
    ready: AtomicUsize,
    /// How many photographs could not be read.
    unreadable: AtomicUsize,
    /// When synthesis began, for the Action Log line.
    started: Instant,
    /// How long it took, once it has finished.
    elapsed: OnceLock<Duration>,
    /// Whether the finished line has been handed out.
    reported: AtomicBool,
    /// Tells the event loop a row is ready to draw.
    wake: Option<Arc<dyn Fn() + Send + Sync>>,
}

impl DisplayThumbnails {
    /// The display column of a reconstruction that carries thumbnails: its
    /// own column, shared. `None` when it carries none.
    pub(crate) fn embedded(recon: &SfmrReconstruction) -> Option<Arc<Self>> {
        let column = recon.image_table.thumbnails_y_x_rgb.as_ref()?;
        Some(Arc::new(Self {
            rows_by_name: names_to_rows(recon),
            source: Source::Embedded(Arc::clone(column)),
        }))
    }

    /// Start building display thumbnails for `recon` from its photographs.
    ///
    /// `None` when there is nothing to build from: the reconstruction has no
    /// images, its workspace did not resolve, or not one of its photographs is
    /// on disk. The node then has no display column, the frustums draw their
    /// outlines without image quads, and the panels draw their placeholder.
    ///
    /// Otherwise the rows are handed to [`POOL`] in image order and each
    /// finished row calls `wake`, so it is drawn on the next frame.
    pub(crate) fn synthesize(
        recon: &SfmrReconstruction,
        wake: Option<Arc<dyn Fn() + Send + Sync>>,
    ) -> Option<Arc<Self>> {
        let workspace = &recon.workspace_dir;
        if recon.image_table.images.is_empty()
            || workspace.as_os_str().is_empty()
            || !workspace.is_dir()
        {
            return None;
        }
        let paths: Vec<PathBuf> = recon
            .image_table
            .images
            .iter()
            .map(|image| workspace.join(&image.name))
            .collect();
        if !paths.iter().any(|path| path.is_file()) {
            return None;
        }
        let n = paths.len();
        let display = Arc::new(Self {
            rows_by_name: names_to_rows(recon),
            source: Source::Synthesized(Synthesis {
                paths,
                rows: (0..n).map(|_| OnceLock::new()).collect(),
                claimed: (0..n).map(|_| AtomicBool::new(false)).collect(),
                next: AtomicUsize::new(0),
                urgent: Mutex::new(Vec::new()),
                ready: AtomicUsize::new(0),
                unreadable: AtomicUsize::new(0),
                started: Instant::now(),
                elapsed: OnceLock::new(),
                reported: AtomicBool::new(false),
                wake,
            }),
        });
        for _ in 0..POOL.current_num_threads().min(n) {
            let weak = Arc::downgrade(&display);
            POOL.spawn(move || work(weak));
        }
        Some(display)
    }

    /// The thumbnail row of the image named `name`, when it is final.
    ///
    /// `None` for a name the column was not built for, and for a row that is
    /// still being built; the caller draws its placeholder and asks again on
    /// a later frame.
    pub(crate) fn row(&self, name: &str) -> Option<ArrayView3<'_, u8>> {
        let &row = self.rows_by_name.get(name)?;
        match &self.source {
            Source::Embedded(column) => Some(column.index_axis(Axis(0), row)),
            Source::Synthesized(synthesis) => {
                let bytes = synthesis.rows[row].get()?;
                Some(
                    ArrayView3::from_shape((THUMBNAIL_SIZE, THUMBNAIL_SIZE, 3), bytes)
                        .expect("a row is THUMBNAIL_SIZE x THUMBNAIL_SIZE x 3"),
                )
            }
        }
    }

    /// How many rows are final. Grows until it reaches [`Self::len`], so a
    /// consumer that remembers it knows when there is something new to draw.
    pub(crate) fn ready(&self) -> usize {
        match &self.source {
            Source::Embedded(_) => self.len(),
            Source::Synthesized(synthesis) => synthesis.ready.load(Ordering::Acquire),
        }
    }

    /// How many rows the column has.
    pub(crate) fn len(&self) -> usize {
        match &self.source {
            Source::Embedded(column) => column.shape()[0],
            Source::Synthesized(synthesis) => synthesis.rows.len(),
        }
    }

    /// Whether every row is final.
    pub(crate) fn is_complete(&self) -> bool {
        self.ready() == self.len()
    }

    /// Whether the rows were built from photographs rather than read from the
    /// file.
    pub(crate) fn is_synthesized(&self) -> bool {
        matches!(self.source, Source::Synthesized(_))
    }

    /// Build the rows of `names` ahead of the rest, in the order given.
    ///
    /// What the Image Browser calls with the cells it is showing, so what is on
    /// screen fills in first. Each call replaces the previous request; rows
    /// already taken are skipped.
    pub(crate) fn prioritize<'a>(&self, names: impl IntoIterator<Item = &'a str>) {
        let Source::Synthesized(synthesis) = &self.source else {
            return;
        };
        let mut wanted: Vec<usize> = names
            .into_iter()
            .filter_map(|name| self.rows_by_name.get(name).copied())
            .filter(|&row| !synthesis.claimed[row].load(Ordering::Relaxed))
            .collect();
        // Taken from the back, so the first name asked for is built first.
        wanted.reverse();
        *synthesis.urgent.lock().expect("urgent list lock") = wanted;
    }

    /// The Action Log line for a synthesis that has finished, handed out once.
    pub(crate) fn take_finished_line(&self, label: &str) -> Option<String> {
        let Source::Synthesized(synthesis) = &self.source else {
            return None;
        };
        let elapsed = *synthesis.elapsed.get()?;
        if synthesis.reported.swap(true, Ordering::AcqRel) {
            return None;
        }
        let n = synthesis.rows.len();
        let unreadable = synthesis.unreadable.load(Ordering::Relaxed);
        let plural = if n == 1 { "" } else { "s" };
        let mut line = format!(
            "Built {n} display thumbnail{plural} for {label} from its photographs in {:.1} s",
            elapsed.as_secs_f64()
        );
        if unreadable > 0 {
            line.push_str(&format!(
                "; {unreadable} could not be read and show a grey placeholder"
            ));
        }
        Some(line)
    }

    /// Block until every row is final. For tests, which have no frame loop.
    #[cfg(test)]
    pub(crate) fn wait(&self) {
        let deadline = Instant::now() + Duration::from_secs(60);
        while !self.is_complete() {
            assert!(
                Instant::now() < deadline,
                "display thumbnails did not finish"
            );
            std::thread::sleep(Duration::from_millis(5));
        }
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

/// One worker: take rows until none are left or the node has let go.
fn work(display: Weak<DisplayThumbnails>) {
    loop {
        // Upgraded per row, so a node closed mid-way stops its workers after
        // the row each is on.
        let Some(display) = display.upgrade() else {
            return;
        };
        let Source::Synthesized(synthesis) = &display.source else {
            return;
        };
        let Some(row) = next_row(synthesis) else {
            return;
        };
        let path = &synthesis.paths[row];
        let pixels = match build_row(path) {
            Some(pixels) => pixels,
            None => {
                // Logged once: each row is built once.
                log::warn!(
                    "Display thumbnail: could not read {}; showing a placeholder",
                    path.display()
                );
                synthesis.unreadable.fetch_add(1, Ordering::Relaxed);
                vec![PLACEHOLDER_GREY; ROW_BYTES].into_boxed_slice()
            }
        };
        let _ = synthesis.rows[row].set(pixels);
        let ready = synthesis.ready.fetch_add(1, Ordering::AcqRel) + 1;
        if ready == synthesis.rows.len() {
            let _ = synthesis.elapsed.set(synthesis.started.elapsed());
        }
        if let Some(wake) = synthesis.wake.as_ref() {
            wake();
        }
    }
}

/// The next unclaimed row: the most recent urgent request first, then image
/// order.
fn next_row(synthesis: &Synthesis) -> Option<usize> {
    loop {
        let urgent = synthesis.urgent.lock().expect("urgent list lock").pop();
        let row = match urgent {
            Some(row) => row,
            None => {
                let row = synthesis.next.fetch_add(1, Ordering::Relaxed);
                if row >= synthesis.rows.len() {
                    return None;
                }
                row
            }
        };
        if !synthesis.claimed[row].swap(true, Ordering::AcqRel) {
            return Some(row);
        }
    }
}

/// The display row of one photograph: decoded without applying EXIF
/// orientation (the `image` crate's default, and the extractors'
/// `IMREAD_IGNORE_ORIENTATION`), then resized to 128 x 128 by area averaging.
fn build_row(path: &std::path::Path) -> Option<Box<[u8]>> {
    let rgb = image::open(path).ok()?.to_rgb8();
    let (width, height) = (rgb.width() as usize, rgb.height() as usize);
    if width == 0 || height == 0 {
        return None;
    }
    Some(thumbnail_from_rgb(rgb.as_raw(), width, height).into_boxed_slice())
}

/// The row a panel draws for image `index` of `recon`, the value a node shows.
///
/// The node's display column `display` first, by the image's name. An image
/// that column was not built for falls back to `recon`'s own thumbnail column
/// when it carries one, so a version that gained an image from a file with
/// thumbnails still shows it. `None` means draw the placeholder: the row is
/// still being built, or there is no picture to show.
pub(crate) fn row_for<'a>(
    display: Option<&'a DisplayThumbnails>,
    recon: &'a SfmrReconstruction,
    index: usize,
) -> Option<ArrayView3<'a, u8>> {
    let name = &recon.image_table.images.get(index)?.name;
    if let Some(display) = display {
        if display.rows_by_name.contains_key(name) {
            return display.row(name);
        }
    }
    let column = recon.image_table.thumbnails_y_x_rgb.as_ref()?;
    (index < column.shape()[0]).then(|| column.index_axis(Axis(0), index))
}
