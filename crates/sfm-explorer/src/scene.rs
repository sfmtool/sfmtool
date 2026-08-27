// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The scene graph: the reconstructions the viewer has loaded, and the typed
//! references that address entities inside them.
//!
//! See `specs/gui/scene-graph.md`. All five phases of that design are in
//! place: the identity types, the node with its per-node display state, its
//! similarity transform and its tint, and a scene that holds **any number** of
//! nodes — `File > Open` appends, and the Scene Graph panel
//! ([`crate::scene_graph`]) is the control surface for them.
//!
//! ## Where refs are used, and where local indices survive
//!
//! A [`ReconId`] is never reused, so a ref can go stale but can never alias a
//! different reconstruction. That is what makes them worth threading, and it
//! draws the line for the rest of the crate:
//!
//! - Anything **stored across frames** — [`crate::state::AppState`] selection
//!   and hover, the SIFT/full-res caches, every panel-local texture cache, the
//!   camera-view target — is keyed by [`ImageRef`] / [`PointRef`] /
//!   [`CameraRef`].
//! - Anything **scoped to one call** is a plain index local to the
//!   reconstruction it was read from: panel `show` arguments, panel responses,
//!   track image lists, and the GPU uniforms. Panels take the owning
//!   [`ReconId`] alongside the reconstruction and mint refs where they need to
//!   remember something; `dock.rs` unwraps refs back to local indices on the
//!   way in.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU32, Ordering};

use sfmtool_core::{Se3Transform, SfmrReconstruction};

/// Source of [`ReconId`] values. Monotonic and never reset, so ids are unique
/// for the life of the process rather than merely within one `AppState`.
static NEXT_RECON_ID: AtomicU32 = AtomicU32::new(0);

/// Identity of one loaded reconstruction. Monotonically assigned per session,
/// never reused — so stale cache entries can never alias a new load.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ReconId(u32);

impl ReconId {
    /// Mint the next unused id. The only way to make one: ids are handed out,
    /// never chosen, which is what guarantees they are not reused.
    pub fn next() -> Self {
        Self(NEXT_RECON_ID.fetch_add(1, Ordering::Relaxed))
    }

    /// A chosen id, for tests that need a fixture and an assertion to agree on
    /// one. Test-only on purpose: in the app, ids are handed out by
    /// [`ReconId::next`] and never picked.
    #[cfg(test)]
    pub const fn from_raw(raw: u32) -> Self {
        Self(raw)
    }
}

/// A camera/image within a specific reconstruction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ImageRef {
    pub recon: ReconId,
    pub image: u32,
}

impl ImageRef {
    pub fn new(recon: ReconId, image: usize) -> Self {
        Self {
            recon,
            image: image as u32,
        }
    }

    /// The image index within its own reconstruction.
    pub fn index(self) -> usize {
        self.image as usize
    }

    /// This ref's local index, but only if it points into `recon`.
    pub fn index_in(self, recon: ReconId) -> Option<usize> {
        (self.recon == recon).then(|| self.index())
    }
}

/// A 3D point within a specific reconstruction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PointRef {
    pub recon: ReconId,
    pub point: u32,
}

impl PointRef {
    pub fn new(recon: ReconId, point: usize) -> Self {
        Self {
            recon,
            point: point as u32,
        }
    }

    /// The point index within its own reconstruction.
    pub fn index(self) -> usize {
        self.point as usize
    }

    /// This ref's local index, but only if it points into `recon`.
    pub fn index_in(self, recon: ReconId) -> Option<usize> {
        (self.recon == recon).then(|| self.index())
    }
}

/// A camera intrinsics record within a specific reconstruction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CameraRef {
    pub recon: ReconId,
    pub camera: u32,
}

impl CameraRef {
    pub fn new(recon: ReconId, camera: usize) -> Self {
        Self {
            recon,
            camera: camera as u32,
        }
    }

    /// The camera index within its own reconstruction.
    pub fn index(self) -> usize {
        self.camera as usize
    }

    /// This ref's local index, but only if it points into `recon`.
    pub fn index_in(self, recon: ReconId) -> Option<usize> {
        (self.recon == recon).then(|| self.index())
    }
}

/// One entry of the per-node tint palette: a display name and its color.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TintColor {
    /// What the `Tint ▸` menu calls it.
    pub name: &'static str,
    pub rgb: [u8; 3],
}

/// The colors a node can be tinted with.
///
/// The Okabe–Ito qualitative palette (Okabe & Ito, *Color Universal Design*,
/// 2002 — <https://jfly.uni-koeln.de/color/>), which is designed to stay
/// mutually distinguishable under the common forms of color blindness. Its
/// eighth entry, black, is left out: a black tint is not an identity color but
/// a way to make a node vanish into this viewer's dark background, and the eye
/// toggle already does that honestly.
///
/// A fixed set rather than a free color picker (the spec's open question,
/// settled here): the job of a tint is *telling two nodes apart*, which a
/// pre-vetted mutually-distinguishable set does by construction and a picker
/// leaves to the user — who can, and eventually will, pick two blues.
///
/// A `static` rather than a `const` so `&TINT_PALETTE[i]` is a real `'static`
/// reference for a runtime index; [`NodeTint`] stores one.
pub static TINT_PALETTE: [TintColor; 7] = [
    TintColor {
        name: "Orange",
        rgb: [230, 159, 0],
    },
    TintColor {
        name: "Sky Blue",
        rgb: [86, 180, 233],
    },
    TintColor {
        name: "Bluish Green",
        rgb: [0, 158, 115],
    },
    TintColor {
        name: "Yellow",
        rgb: [240, 228, 66],
    },
    TintColor {
        name: "Blue",
        rgb: [0, 114, 178],
    },
    TintColor {
        name: "Vermillion",
        rgb: [213, 94, 0],
    },
    TintColor {
        name: "Reddish Purple",
        rgb: [204, 121, 167],
    },
];

/// The color a **selected camera's sibling images** are marked in: every
/// frustum whose image uses the selected intrinsics, and the same images'
/// thumbnails in the Image Browser.
///
/// Declared here, once, because two panels have to say the same thing in the
/// same color for the highlight to read as one statement about one set of
/// images — `scene_renderer::upload::frustums` packs it into the per-image
/// color buffer and `image_browser` strokes a border with it.
///
/// A violet, chosen to sit clear of the three highlights already on a frustum:
/// white (the node's own), cyan (the selected image) and orange (the selected
/// point's track). The three of them are **ranked, not mixed** — selected image
/// first, track member next, camera sibling last — so this is the color a
/// frustum keeps only when neither of the other two claims it.
pub const SIBLING_HIGHLIGHT_RGB: [u8; 3] = [170, 130, 255];

/// How far a tinted node's colors are pulled toward the tint.
///
/// The shaders composite as `mix(original, tint.rgb, tint.a)`, so this is that
/// `a`: 0 keeps the original colors (the per-recon uniform block's stated
/// convention for "no tint"), 1 flattens the node to one flat color. 0.7 is far
/// enough that a node reads as "the orange one" at a glance, and short enough
/// that photo-derived point colors keep the shading that tells you what you are
/// looking at.
pub const TINT_STRENGTH: f32 = 0.7;

/// A node's comparison tint: its own colors, or one palette entry mixed into
/// them.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum NodeTint {
    /// Draw the reconstruction's own point, thumbnail and patch colors.
    #[default]
    Original,
    /// Mix [`TINT_STRENGTH`] of this color into all of them.
    Tint(&'static TintColor),
}

impl NodeTint {
    /// The `tint_color` vec4 the shaders read: `rgb` in 0..1 with the strength
    /// in `a`, and all-zero — hence `a == 0`, hence "original colors" — for
    /// [`NodeTint::Original`].
    pub fn to_uniform(self) -> [f32; 4] {
        match self {
            NodeTint::Original => [0.0; 4],
            NodeTint::Tint(color) => [
                color.rgb[0] as f32 / 255.0,
                color.rgb[1] as f32 / 255.0,
                color.rgb[2] as f32 / 255.0,
                TINT_STRENGTH,
            ],
        }
    }

    /// The tint's own color, for the Scene panel's swatch and menu. `None` when
    /// the node is drawn in its original colors.
    pub fn rgb(self) -> Option<[u8; 3]> {
        match self {
            NodeTint::Original => None,
            NodeTint::Tint(color) => Some(color.rgb),
        }
    }
}

/// One loaded reconstruction and the view state that belongs to it.
pub struct SceneNode {
    pub id: ReconId,
    /// Display label: the file stem, or `"demo"` for demo data, disambiguated
    /// with `" (2)"`, `" (3)"`… when a stem is already taken (see
    /// [`unique_label`]).
    pub label: String,
    /// Source path; `None` for demo data.
    pub path: Option<PathBuf>,
    pub recon: SfmrReconstruction,
    /// This node's data needs (re-)upload to the GPU. Replaces the former
    /// global `AppState::points_need_upload`.
    pub needs_upload: bool,

    // ── Per-node display state (Scene Graph panel) ──
    /// Master eye for the whole node. Off = nothing of it is drawn.
    pub visible: bool,
    /// Whether pointer interaction (hover + click pick) in the 3D viewport
    /// reaches this node. Off = display-only: it still renders and occludes,
    /// Alt+click depth targeting still works on it, and it can still be
    /// selected from the Scene panel — it just stops capturing picks.
    pub interactive: bool,
    /// Group eye: the node's 3D points.
    pub show_points: bool,
    /// Group eye: the node's camera frustums and image quads.
    ///
    /// Named for what it draws: a `.sfmr` *image* is the posed view a frustum
    /// and its quad stand for, while a *camera* is the intrinsics record any
    /// number of images can share.
    pub show_camera_images: bool,
    /// Group eye: the node's patch surfels (inert without patch data).
    pub show_patches: bool,
    /// Sub-toggle of [`SceneNode::show_points`]: the `w = 0` directions.
    pub show_points_at_infinity: bool,
    /// Comparison tint: a color mixed into everything this node draws, so two
    /// reconstructions sharing one space can be told apart at a glance. Set
    /// from the Scene panel's `Tint ▸` menu.
    ///
    /// Display state like the eyes: it never touches the reconstruction, and it
    /// is deliberately powerless over the highlight colors — selection, hover
    /// and the track orange stay themselves (see the shaders).
    pub tint: NodeTint,

    /// Similarity transform (uniform scale · rotation · translation) mapping
    /// this node's native coordinates into the shared world space. Identity on
    /// load, set by the Scene panel's `Align to…`.
    ///
    /// **View state only**: it reaches the GPU as the per-recon `model` matrix
    /// and the CPU world-space paths that mirror it (track rays, bounds,
    /// camera-view entry), and never touches the [`SfmrReconstruction`] in
    /// memory nor the `.sfmr` on disk. Baking a transform into a file stays
    /// `sfm xform`'s job.
    pub transform: Se3Transform,
}

impl SceneNode {
    /// A node for `recon`, labeled `label` and sourced from `path`.
    fn new(label: String, path: Option<PathBuf>, recon: SfmrReconstruction) -> Self {
        Self {
            id: ReconId::next(),
            label,
            path,
            recon,
            needs_upload: true,
            visible: true,
            interactive: true,
            show_points: true,
            show_camera_images: true,
            show_patches: true,
            show_points_at_infinity: true,
            tint: NodeTint::Original,
            transform: Se3Transform::identity(),
        }
    }

    /// A node derived in-session from another node (a resection, for one),
    /// carrying `recon` under `label`. It came from no file — `Reload from
    /// Disk` is greyed on it, exactly as it is on demo data — and everything
    /// there is to know about where it came from is in its label and its
    /// reconstruction's metadata.
    pub fn derived(label: String, recon: SfmrReconstruction) -> Self {
        Self::new(label, None, recon)
    }

    /// A node for a reconstruction read from `path`, labeled with its file stem.
    pub fn from_path(path: &Path, recon: SfmrReconstruction) -> Self {
        Self::new(label_for_path(path), Some(path.to_path_buf()), recon)
    }

    /// A node for generated demo data, which came from no file.
    pub fn demo(recon: SfmrReconstruction) -> Self {
        Self::new("demo".to_string(), None, recon)
    }

    /// The file name shown in the window title, or `None` for demo data.
    pub fn file_name(&self) -> Option<String> {
        let path = self.path.as_ref()?;
        Some(
            path.file_name()
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_else(|| path.display().to_string()),
        )
    }

    /// Whether this node carries everything the patch surfel pass needs: patch
    /// frames *and* the bitmaps to texture them with.
    pub fn has_patch_data(&self) -> bool {
        let r = &self.recon;
        r.patch_u_halfvec_xyz.is_some()
            && r.patch_v_halfvec_xyz.is_some()
            && r.patch_bitmaps_y_x_rgba.is_some()
    }

    /// Whether this node has been moved out of its own frame — i.e. its
    /// transform is not the identity.
    ///
    /// Compared exactly rather than with a tolerance: the only two ways a
    /// transform is set are `Align to…` (which never returns an exact identity
    /// by accident) and `Reset Transform` (which assigns
    /// [`Se3Transform::identity`] itself).
    pub fn has_transform(&self) -> bool {
        let t = &self.transform;
        t.scale != 1.0
            || t.translation != nalgebra::Vector3::zeros()
            || t.rotation != sfmtool_core::RotQuaternion::identity()
    }

    /// Copy the per-node display state (eyes, interaction cursor, tint,
    /// transform) from `other`.
    ///
    /// Used by `Reload from Disk`, which is a fresh read of the same file and
    /// so should not also reset how the node is being displayed — an alignment
    /// the user fitted before refreshing the file included, and the tint that
    /// is how they were telling it apart from the file beside it.
    pub fn copy_display_from(&mut self, other: &SceneNode) {
        self.visible = other.visible;
        self.interactive = other.interactive;
        self.show_points = other.show_points;
        self.show_camera_images = other.show_camera_images;
        self.show_patches = other.show_patches;
        self.show_points_at_infinity = other.show_points_at_infinity;
        self.tint = other.tint;
        self.transform = other.transform.clone();
    }
}

/// The label a path alone suggests: its file stem, or the whole path when it
/// has no stem to speak of.
pub fn label_for_path(path: &Path) -> String {
    path.file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| path.display().to_string())
}

/// `base`, or the first free `"base (n)"` when `base` is already in the scene.
///
/// Two files of the same name in different directories are a routine way to
/// compare solver runs, so the tree has to be able to tell them apart even
/// though their stems agree.
pub fn unique_label(scene: &[SceneNode], base: &str) -> String {
    if !scene.iter().any(|n| n.label == base) {
        return base.to_string();
    }
    (2..)
        .map(|n| format!("{base} ({n})"))
        .find(|candidate| !scene.iter().any(|n| &n.label == candidate))
        .expect("the candidate sequence is infinite")
}

/// Look up a node by id.
///
/// A free function over the scene slice rather than an `AppState` method, so
/// callers can hold the node while mutating other `AppState` fields — the SIFT
/// and full-res caches, notably, which is the whole point of the split.
pub fn node_by_id(scene: &[SceneNode], id: ReconId) -> Option<&SceneNode> {
    scene.iter().find(|n| n.id == id)
}

/// The node the panels follow, given the scene and `AppState::selected_recon`.
///
/// Same split-borrow rationale as [`node_by_id`]: the scene slice and the
/// selection are passed separately so the caller keeps `&mut` access to the
/// rest of `AppState`.
pub fn selected_node(scene: &[SceneNode], selected: Option<ReconId>) -> Option<&SceneNode> {
    node_by_id(scene, selected?)
}

/// A node's 3D points in the **shared world space** — its own positions put
/// through its transform.
///
/// The CPU counterpart of the per-recon `model` matrix, for the framing paths
/// that work on point positions rather than on the GPU: `Z` zoom-to-fit, the
/// Scene panel's per-node `Zoom to Fit`, and the viewport's first-show framing.
pub fn world_points(node: &SceneNode) -> Vec<nalgebra::Point3<f64>> {
    node.recon
        .points
        .iter()
        .map(|p| node.transform.apply_to_point(&p.position))
        .collect()
}

/// The centres of the node's images taken through camera `index`, in the
/// **shared world space** — the camera-row counterpart of [`world_points`].
///
/// What the Scene panel's double-click on a camera row frames. Empty when no
/// image uses that camera, which `zoom_to_fit_points` treats as nothing to
/// frame rather than as a degenerate one.
pub fn camera_world_centres(node: &SceneNode, index: usize) -> Vec<nalgebra::Point3<f64>> {
    node.recon
        .images
        .iter()
        .filter(|image| image.camera_index as usize == index)
        .map(|image| node.transform.apply_to_point(&image.camera_center()))
        .collect()
}

/// The images of `node` taken through the **selected** camera, as indices
/// local to it.
///
/// The set every panel marks in [`SIBLING_HIGHLIGHT_RGB`]: selecting a camera
/// is a statement about a set of images, and the rest of the viewer should say
/// which ones.
///
/// Empty in the two cases where the highlight would say nothing: no camera of
/// this node is selected, and — the interesting one — *every* image in the node
/// uses it. Highlighting the whole of a single-camera reconstruction is correct
/// and uninformative, so it is suppressed rather than drawn.
pub fn camera_sibling_images(node: &SceneNode, selected: Option<CameraRef>) -> Vec<usize> {
    let Some(index) = selected.and_then(|camera| camera.index_in(node.id)) else {
        return Vec::new();
    };
    let siblings: Vec<usize> = node
        .recon
        .images
        .iter()
        .enumerate()
        .filter(|(_, image)| image.camera_index as usize == index)
        .map(|(i, _)| i)
        .collect();
    if siblings.len() == node.recon.images.len() {
        return Vec::new();
    }
    siblings
}

/// What the viewport's top-left stats overlay reports.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SceneStats {
    /// How many nodes contributed — the count the overlay leads with once it
    /// is greater than one.
    pub recons: usize,
    pub points: usize,
    pub points_at_infinity: usize,
    pub images: usize,
}

/// Whether a node is drawn at all: its own master eye, **and** the scene's solo
/// override if one is active.
///
/// The single definition of effective node visibility, and deliberately the
/// only one. Solo is a *view mode* layered over the eyes rather than a bulk edit
/// of them (`solo` names at most one node, and un-soloing restores whatever the
/// user had set), so every consumer has to compose the two the same way — the
/// draw loop and the scene bounds through the `NodeDisplay` mirror `app.rs`
/// builds from this, the stats overlay through [`visible_stats`].
///
/// Note what it is *not* the AND of: the global HUD layer toggles and the
/// per-group eyes are per-layer, applied where each layer is drawn. This is the
/// whole-node question.
pub fn is_visible(node: &SceneNode, solo: Option<ReconId>) -> bool {
    node.visible && solo.is_none_or(|id| id == node.id)
}

/// Sum the scene's entity counts over the nodes that are actually drawn.
///
/// Hidden nodes drop out entirely: the overlay describes what is on screen, and
/// a node switched off — by its own eye or by another node's solo — is not.
pub fn visible_stats(scene: &[SceneNode], solo: Option<ReconId>) -> SceneStats {
    let mut stats = SceneStats::default();
    for node in scene.iter().filter(|n| is_visible(n, solo)) {
        stats.recons += 1;
        stats.points += node.recon.points.len();
        stats.points_at_infinity += node.recon.metadata.infinity_point_count as usize;
        stats.images += node.recon.images.len();
    }
    stats
}

/// The first 8 hex characters of a reconstruction's content hash — the part
/// that goes into a displayed `pt3d_<hash>_<index>` id.
///
/// Zero-filled when the reconstruction carries no hash, so an id is always the
/// same shape and always parseable.
pub fn hash_prefix(recon: &SfmrReconstruction) -> String {
    let hash = &recon.content_hash.content_xxh128;
    if hash.len() >= 8 {
        hash[..8].to_string()
    } else {
        "00000000".to_string()
    }
}

/// The copyable point id the Point Track panel shows, `pt3d_<hash>_<index>`.
///
/// Because the hash is per-reconstruction content, these ids are already
/// unambiguous across simultaneously loaded files.
pub fn point_id(recon: &SfmrReconstruction, point_idx: usize) -> String {
    format!("pt3d_{}_{}", hash_prefix(recon), point_idx)
}
