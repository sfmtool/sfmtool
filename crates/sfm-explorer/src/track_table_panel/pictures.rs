// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The Track Table's picture columns: what each one shows, the picture size
//! the header's *Size* slider sets, and the texture cache the cells draw from.
//!
//! See `specs/gui/track-table.md` § "The Patch column". The picture columns
//! are pinned beside the Point column and set the row height. A cell's picture
//! is made only for a row that is drawn, so only for the rows on screen, and
//! its texture is kept in [`PictureTextures`], keyed by the point and the
//! version the rows belong to, so that scrolling back over rows already seen
//! uploads nothing. The cache is bounded by entry count: the pictures of rows
//! that have left the view are kept until it is full and then released, the
//! longest unseen first.

use std::collections::HashMap;

use sfmtool_core::EditedReconstruction;

use crate::document::VersionSerial;
use crate::scene::ReconId;

/// The smallest picture size the *Size* slider sets, in points. A row is then
/// [`PICTURE_PAD`] taller on each side, which still holds a line of body text
/// with room around it.
pub(super) const MIN_PICTURE_SIZE: f32 = 32.0;
/// The largest picture size the *Size* slider sets, in points.
pub(super) const MAX_PICTURE_SIZE: f32 = 128.0;
/// The picture size before the slider is moved, in points.
pub(super) const DEFAULT_PICTURE_SIZE: f32 = 64.0;
/// Space between a picture and its row's top and bottom edges, so that the
/// pictures of two rows do not touch.
pub(super) const PICTURE_PAD: f32 = 2.0;
/// How many times the picture size a picture's hover view draws it at.
pub(super) const HOVER_SCALE: f32 = 3.0;
/// The largest a picture's hover view draws it, in points: three times the
/// largest picture size would cover most of a small window.
pub(super) const MAX_HOVER_SIZE: f32 = 256.0;

/// The side of a picture's hover view for pictures `size` points square.
pub(super) fn hover_size(size: f32) -> f32 {
    (HOVER_SCALE * size).min(MAX_HOVER_SIZE)
}
/// How many pictures [`PictureTextures`] keeps before it releases the ones
/// drawn longest ago. Ample for the rows a window shows at the smallest size,
/// and small beside the GPU's memory: a 64-texel patch is 16 KiB.
pub(super) const TEXTURE_CAPACITY: usize = 1024;

/// A column of pictures, drawn pinned beside the Point column. Picture columns
/// cannot be sorted or bounded, so they are not track table columns.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub(super) enum PictureColumn {
    /// The point's patch bitmap, as Track View's header slot draws it.
    Patch,
}

impl PictureColumn {
    /// Every picture column, in the order they are drawn.
    pub(super) const ALL: [PictureColumn; 1] = [PictureColumn::Patch];

    /// The heading the column is shown under.
    pub(super) fn title(self) -> &'static str {
        match self {
            PictureColumn::Patch => "Patch",
        }
    }

    /// The column's name in the panel query.
    pub(super) fn wire_name(self) -> &'static str {
        match self {
            PictureColumn::Patch => "patch",
        }
    }

    /// What the column shows, for its heading's hover text and its entry in
    /// the Columns menu.
    pub(super) fn description(self) -> &'static str {
        match self {
            PictureColumn::Patch => {
                "The point's patch bitmap, the picture Track View shows beside a viewed point: \
                 the bitmap the file stores, or the one the open rendered for a file with patch \
                 frames and no bitmaps. A picture column cannot be sorted or bounded."
            }
        }
    }

    /// Whether `edited` can supply the column: for Patch, whether its points
    /// carry a patch bitmap, stored or rendered by the open for display.
    pub(super) fn is_available(self, edited: &EditedReconstruction) -> bool {
        match self {
            PictureColumn::Patch => edited.has_patch_bitmaps(),
        }
    }

    /// The picture of `point` in `edited`, or `None` where it has none: for
    /// Patch, an all-zero bitmap row, which is how a point with no stored
    /// patch is written.
    pub(super) fn image(
        self,
        edited: &EditedReconstruction,
        point: u32,
    ) -> Option<egui::ColorImage> {
        match self {
            PictureColumn::Patch => {
                crate::patch_pictures::stored_patch_image(edited.point(point)?.patch_bitmap()?)
            }
        }
    }
}

/// One cell's picture, named by the version its row belongs to, so a picture
/// is never drawn for a version it was not made from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(super) struct PictureKey {
    pub(super) column: PictureColumn,
    pub(super) node: ReconId,
    pub(super) point: u32,
    pub(super) document: VersionSerial,
}

/// What a cell has to draw.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum CellPicture {
    /// The picture.
    Texture(egui::TextureId),
    /// The point has no picture in this column: *no patch*.
    Nothing,
    /// No picture is held for the cell and none is made for it: an empty
    /// frame. The rows of a version the node has left are drawn so, until
    /// the new version's rows arrive.
    Waiting,
}

/// One cached picture.
struct Entry {
    /// The texture, or `None` for a point with no picture.
    texture: Option<egui::TextureHandle>,
    /// The frame that last drew it.
    used: u64,
}

/// The textures of the pictures the table has drawn, bounded by
/// [`TEXTURE_CAPACITY`] entries.
///
/// Each frame that draws the table calls [`Self::begin_frame`], then asks for
/// the visible cells' pictures, then calls [`Self::end_frame`], which releases
/// the pictures drawn longest ago once the cache holds more than its
/// capacity. Releasing takes it down to three quarters of its capacity, so a
/// sort of the entries is paid once per quarter of the capacity of new
/// pictures rather than on every frame. A picture drawn this frame is never
/// released.
#[derive(Default)]
pub(super) struct PictureTextures {
    entries: HashMap<PictureKey, Entry>,
    /// The frame now being drawn.
    frame: u64,
    /// How many pictures have been made, for the tests that hold which rows
    /// ask for one.
    #[cfg(test)]
    pub(super) made: Vec<PictureKey>,
}

impl std::fmt::Debug for PictureTextures {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PictureTextures")
            .field("entries", &self.entries.len())
            .field("frame", &self.frame)
            .finish()
    }
}

impl PictureTextures {
    /// Start a frame of the table.
    pub(super) fn begin_frame(&mut self) {
        self.frame += 1;
    }

    /// The cell `key`'s picture. When none is held and `make` is given, the
    /// picture is made with it, uploaded and kept; with no `make`, as for
    /// the rows of a version the node has left, nothing is made and the cell
    /// is [`CellPicture::Waiting`].
    pub(super) fn picture(
        &mut self,
        ctx: &egui::Context,
        key: PictureKey,
        make: Option<&dyn Fn() -> Option<egui::ColorImage>>,
    ) -> CellPicture {
        let frame = self.frame;
        let entry = match self.entries.get_mut(&key) {
            Some(entry) => entry,
            None => {
                let Some(make) = make else {
                    return CellPicture::Waiting;
                };
                #[cfg(test)]
                self.made.push(key);
                let texture = make().map(|image| {
                    ctx.load_texture(
                        format!("track_table_{}_{}", key.column.wire_name(), key.point),
                        image,
                        egui::TextureOptions::NEAREST,
                    )
                });
                self.entries
                    .entry(key)
                    .or_insert(Entry { texture, used: 0 })
            }
        };
        entry.used = frame;
        match &entry.texture {
            Some(texture) => CellPicture::Texture(texture.id()),
            None => CellPicture::Nothing,
        }
    }

    /// End the frame: once more than [`TEXTURE_CAPACITY`] pictures are held,
    /// release the ones drawn longest ago, down to three quarters of it,
    /// keeping every picture this frame drew.
    pub(super) fn end_frame(&mut self) {
        if self.entries.len() <= TEXTURE_CAPACITY {
            return;
        }
        let mut used: Vec<u64> = self.entries.values().map(|e| e.used).collect();
        used.sort_unstable();
        let release = self.entries.len() - TEXTURE_CAPACITY * 3 / 4;
        // Entries drawn at or before this frame go; ties at the cut go with it,
        // except this frame's own.
        let cut = used[release - 1].min(self.frame.saturating_sub(1));
        self.entries.retain(|_, entry| entry.used > cut);
    }

    /// How many pictures are held.
    #[cfg(test)]
    pub(super) fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the cell `key`'s picture is held.
    #[cfg(test)]
    pub(super) fn holds(&self, key: &PictureKey) -> bool {
        self.entries.contains_key(key)
    }
}

/// Paint one picture cell in `cell`: the picture, `size` points square and
/// centred, dimmed when `dimmed`; *no patch* where the point has none; or an
/// empty frame.
pub(super) fn paint_picture(
    ui: &egui::Ui,
    cell: egui::Rect,
    size: f32,
    picture: CellPicture,
    dimmed: bool,
) {
    let painter = ui.painter();
    let visuals = ui.visuals();
    let rect = egui::Rect::from_center_size(cell.center(), egui::vec2(size, size));
    match picture {
        CellPicture::Texture(texture) => {
            let tint = if dimmed {
                // The rows are of the version before an edit: the picture is
                // drawn at half brightness, as their text is drawn weak.
                egui::Color32::from_gray(128)
            } else {
                egui::Color32::WHITE
            };
            painter.image(
                texture,
                rect,
                egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)),
                tint,
            );
        }
        CellPicture::Nothing => {
            painter.text(
                cell.center(),
                egui::Align2::CENTER_CENTER,
                "no patch",
                egui::TextStyle::Small.resolve(ui.style()),
                visuals.weak_text_color(),
            );
        }
        CellPicture::Waiting => {
            painter.rect_stroke(
                rect,
                2.0,
                visuals.widgets.noninteractive.bg_stroke,
                egui::StrokeKind::Inside,
            );
        }
    }
}
