// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The Track Table's picture columns: what each one shows, the picture size
//! the header's *Size* slider sets, and the texture cache the cells draw from.
//!
//! See `specs/gui/track-table.md` § "The Patch column" and § "The Crop
//! column". The picture columns are pinned beside the Point column and set the
//! row height. A cell's picture is made only for a row that is drawn, so only
//! for the rows on screen, and its texture is kept in [`PictureTextures`],
//! keyed by the column, the point and the version the rows belong to, so that
//! scrolling back over rows already seen uploads nothing. The cache is bounded
//! by entry count: the pictures of rows that have left the view are kept until
//! it is full and then released, the longest unseen first. A cell with nothing
//! to draw keeps the reason in the cache as a picture is kept, and a cell whose
//! picture is not ready yet, a crop whose photograph is decoding or whose
//! reference is being picked, keeps nothing and is asked again.

use std::collections::HashMap;

use sfmtool_core::EditedReconstruction;

use crate::document::VersionSerial;
use crate::patch_pictures::crop::{CropPicture, DrawnCrop, Outline, CELL_TEXTURE_OPTIONS};
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
///
/// Declared in the order they are drawn, which is the order they sort in.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub(super) enum PictureColumn {
    /// The point's reference photograph cut around its patch's outline, as
    /// Track View cuts a row's crop.
    Crop,
    /// The point's patch bitmap, as Track View's header slot draws it.
    Patch,
}

impl PictureColumn {
    /// Every picture column, in the order they are drawn: the photograph as
    /// it is before the picture made from it, as in Track View.
    pub(super) const ALL: [PictureColumn; 2] = [PictureColumn::Crop, PictureColumn::Patch];

    /// The heading the column is shown under.
    pub(super) fn title(self) -> &'static str {
        match self {
            PictureColumn::Crop => "Crop",
            PictureColumn::Patch => "Patch",
        }
    }

    /// The column's name in the panel query.
    pub(super) fn wire_name(self) -> &'static str {
        match self {
            PictureColumn::Crop => "crop",
            PictureColumn::Patch => "patch",
        }
    }

    /// What the column shows, for its heading's hover text and its entry in
    /// the Columns menu.
    pub(super) fn description(self) -> &'static str {
        match self {
            PictureColumn::Crop => {
                "The point's reference photograph cut around its patch's outline, with the \
                 outline drawn over it, as Track View cuts a row's crop. A point that stores no \
                 reference gets one picked by the reference-view rule. A picture column cannot \
                 be sorted or bounded."
            }
            PictureColumn::Patch => {
                "The point's patch bitmap, the picture Track View shows beside a viewed point: \
                 the bitmap the file stores, or the one the open rendered for a file with patch \
                 frames and no bitmaps. A picture column cannot be sorted or bounded."
            }
        }
    }

    /// Whether `edited` can supply the column: for Crop, whether its points
    /// carry a patch frame and its observations a keypoint, which the outline
    /// is made of; for Patch, whether its points carry a patch bitmap, stored
    /// or rendered by the open for display.
    pub(super) fn is_available(self, edited: &EditedReconstruction) -> bool {
        match self {
            PictureColumn::Crop => edited.has_patch_frames() && edited.has_keypoints(),
            PictureColumn::Patch => edited.has_patch_bitmaps(),
        }
    }
}

/// The Patch picture of `point` in `edited`: its patch bitmap, or *no patch*
/// for an all-zero bitmap row, which is how a point with no stored patch is
/// written.
pub(super) fn patch_image(edited: &EditedReconstruction, point: u32) -> Made {
    match edited
        .point(point)
        .and_then(|view| crate::patch_pictures::stored_patch_image(view.patch_bitmap()?))
    {
        Some(image) => Made::Image(image),
        None => Made::Absent(Absent::NoPatch),
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

/// Why a cell has no picture, which it says in place of one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Absent {
    /// The point has no patch: an all-zero bitmap row in the Patch column,
    /// zero patch half-vectors in the Crop column. *no patch*.
    NoPatch,
    /// The Crop column's photograph cannot be read. *unreadable*.
    Unreadable,
    /// The point stores no reference observation and the reference-view rule
    /// picked none. *no reference*.
    NoReference,
    /// No sample of the patch's outline projects into the photograph.
    /// *off image*.
    OffImage,
}

impl Absent {
    /// What the cell says.
    pub(super) fn text(self) -> &'static str {
        match self {
            Absent::NoPatch => "no patch",
            Absent::Unreadable => "unreadable",
            Absent::NoReference => "no reference",
            Absent::OffImage => "off image",
        }
    }
}

/// What a cell has to draw.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum CellPicture {
    /// The picture.
    Texture(egui::TextureId),
    /// The point has no picture in this column, for the reason given, which
    /// the cell says.
    Absent(Absent),
    /// No picture is held for the cell and none is made for it yet: an empty
    /// frame. A crop whose photograph is decoding or whose reference is being
    /// picked is drawn so, and so are the rows of a version the node has left,
    /// until the new version's rows arrive.
    Waiting,
}

/// What making a cell's picture gave.
pub(super) enum Made {
    /// A picture to upload.
    Image(egui::ColorImage),
    /// A crop to upload, with the outline it was cut around, which its hover
    /// view is cut around too, and the image it was cut from.
    Crop {
        picture: Box<CropPicture>,
        outline: Box<Outline>,
        image: usize,
    },
    /// Nothing to draw, for the reason given: kept, as a picture is.
    Absent(Absent),
    /// Not yet: the crop's photograph is decoding or its reference is being
    /// picked. Nothing is kept, and the cell is asked again on the next frame
    /// that draws it.
    NotYet,
}

/// One cached cell.
enum Held {
    /// A picture's texture.
    Texture(egui::TextureHandle),
    /// A crop, uploaded with its outline, and what its hover view is cut
    /// from.
    Crop {
        drawn: DrawnCrop,
        outline: Box<Outline>,
        image: usize,
    },
    /// The reason the cell has no picture.
    Absent(Absent),
}

/// One cached picture.
struct Entry {
    /// What the cell draws.
    held: Held,
    /// The frame that last drew it.
    used: u64,
}

/// The outline stroke over a crop in a cell: white over a wider dark stroke,
/// as Track View draws the marks over a hover view, since the table has no
/// verdict to colour it by.
const CELL_OUTLINE: [egui::Stroke; 2] = [
    egui::Stroke {
        width: 2.5,
        color: egui::Color32::from_black_alpha(170),
    },
    egui::Stroke {
        width: 1.0,
        color: egui::Color32::from_gray(240),
    },
];

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
    /// The hover view of the crop the pointer last rested on, cut when it
    /// first rested there: `None` inside for a crop whose hover view cut
    /// nothing.
    hover: Option<(PictureKey, Option<DrawnCrop>)>,
    /// The key of every picture made, in order, for the tests that hold which
    /// rows ask for one.
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
    /// picture is made with it, uploaded and kept, or the reason there is
    /// none kept; a [`Made::NotYet`] keeps nothing and the cell is
    /// [`CellPicture::Waiting`]. With no `make`, as for the rows of a version
    /// the node has left, nothing is made and the cell is
    /// [`CellPicture::Waiting`].
    pub(super) fn picture(
        &mut self,
        ctx: &egui::Context,
        key: PictureKey,
        make: Option<&dyn Fn() -> Made>,
    ) -> CellPicture {
        let frame = self.frame;
        let entry = match self.entries.get_mut(&key) {
            Some(entry) => entry,
            None => {
                let Some(make) = make else {
                    return CellPicture::Waiting;
                };
                let name = format!("track_table_{}_{}", key.column.wire_name(), key.point);
                let held = match make() {
                    Made::NotYet => return CellPicture::Waiting,
                    Made::Absent(absent) => Held::Absent(absent),
                    Made::Image(image) => {
                        Held::Texture(ctx.load_texture(name, image, egui::TextureOptions::NEAREST))
                    }
                    Made::Crop {
                        picture,
                        outline,
                        image,
                    } => Held::Crop {
                        drawn: DrawnCrop::new(
                            ctx,
                            *picture,
                            CELL_OUTLINE[1].color,
                            name,
                            CELL_TEXTURE_OPTIONS,
                        ),
                        outline,
                        image,
                    },
                };
                #[cfg(test)]
                self.made.push(key);
                self.entries.entry(key).or_insert(Entry { held, used: 0 })
            }
        };
        entry.used = frame;
        match &entry.held {
            Held::Texture(texture) => CellPicture::Texture(texture.id()),
            Held::Crop { drawn, .. } => CellPicture::Texture(drawn.texture_id()),
            Held::Absent(absent) => CellPicture::Absent(*absent),
        }
    }

    /// The crop held for the cell `key`, with the outline it was cut around
    /// and the image it was cut from, for drawing its outline and cutting its
    /// hover view.
    pub(super) fn crop(&self, key: &PictureKey) -> Option<(&DrawnCrop, &Outline, usize)> {
        match &self.entries.get(key)?.held {
            Held::Crop {
                drawn,
                outline,
                image,
            } => Some((drawn, outline, *image)),
            _ => None,
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

    /// The hover view of the crop held for `key`, cut around the same outline
    /// from the photograph `pyramid` finds for its image, the first time it
    /// is asked for, and kept while the pointer rests on that cell. `None`
    /// when no crop is held for `key` or its photograph has left the cache.
    pub(super) fn crop_context(
        &mut self,
        ctx: &egui::Context,
        key: PictureKey,
        pyramid: &dyn Fn(
            usize,
        )
            -> Option<std::sync::Arc<sfmtool_core::camera::image::ImageU8Pyramid>>,
    ) -> Option<&DrawnCrop> {
        if self.hover.as_ref().map(|(held, _)| *held) != Some(key) {
            let (_, outline, image) = self.crop(&key)?;
            let src = pyramid(image)?;
            let drawn = crate::patch_pictures::crop::context(outline, &src).map(|picture| {
                let caption = crate::patch_pictures::crop::point_context_caption(&picture);
                DrawnCrop::new(
                    ctx,
                    picture,
                    CELL_OUTLINE[1].color,
                    format!("track_table_crop_context_{}", key.point),
                    egui::TextureOptions::LINEAR,
                )
                .with_caption(caption)
            });
            self.hover = Some((key, drawn));
        }
        self.hover.as_ref().and_then(|(_, drawn)| drawn.as_ref())
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
/// centred, dimmed when `dimmed`, with a crop's outline over it when `crop`
/// is the crop it is; the reason where the point has none; or an empty frame.
pub(super) fn paint_picture(
    ui: &egui::Ui,
    cell: egui::Rect,
    size: f32,
    picture: CellPicture,
    crop: Option<&DrawnCrop>,
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
            match crop {
                Some(crop) => crop.paint_cell_with(painter, rect, tint, &CELL_OUTLINE),
                None => {
                    painter.image(
                        texture,
                        rect,
                        egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)),
                        tint,
                    );
                }
            }
        }
        CellPicture::Absent(absent) => {
            painter.text(
                cell.center(),
                egui::Align2::CENTER_CENTER,
                absent.text(),
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
