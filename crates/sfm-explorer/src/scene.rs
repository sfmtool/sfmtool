// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The scene graph: the reconstructions the viewer has loaded, and the typed
//! references that address entities inside them.
//!
//! See `specs/gui/gui-scene-graph.md`. Phases 1–4 of that design are in place:
//! the identity types, the node with its per-node display state and its
//! similarity transform, and a scene that holds **any number** of nodes —
//! `File > Open` appends, and the Scene Graph panel ([`crate::scene_graph`]) is
//! the control surface for them. Node tints (phase 5) are still renderer-side
//! defaults.
//!
//! ## Where refs are used, and where local indices survive
//!
//! A [`ReconId`] is never reused, so a ref can go stale but can never alias a
//! different reconstruction. That is what makes them worth threading, and it
//! draws the line for the rest of the crate:
//!
//! - Anything **stored across frames** — [`crate::state::AppState`] selection
//!   and hover, the SIFT/full-res caches, every panel-local texture cache, the
//!   camera-view target — is keyed by [`ImageRef`] / [`PointRef`].
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

/// One loaded reconstruction and the view state that belongs to it.
///
/// The spec's `tint` is still missing: it arrives with phase 5, when the
/// renderer first has something other than the original colors to apply.
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
    pub show_cameras: bool,
    /// Group eye: the node's patch surfels (inert without patch data).
    pub show_patches: bool,
    /// Sub-toggle of [`SceneNode::show_points`]: the `w = 0` directions.
    pub show_points_at_infinity: bool,

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
            show_cameras: true,
            show_patches: true,
            show_points_at_infinity: true,
            transform: Se3Transform::identity(),
        }
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

    /// Copy the per-node display state (eyes, interaction cursor, transform)
    /// from `other`.
    ///
    /// Used by `Reload from Disk`, which is a fresh read of the same file and
    /// so should not also reset how the node is being displayed — an alignment
    /// the user fitted before refreshing the file included.
    pub fn copy_display_from(&mut self, other: &SceneNode) {
        self.visible = other.visible;
        self.interactive = other.interactive;
        self.show_points = other.show_points;
        self.show_cameras = other.show_cameras;
        self.show_patches = other.show_patches;
        self.show_points_at_infinity = other.show_points_at_infinity;
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

/// Sum the scene's entity counts over the nodes whose master eye is on.
///
/// Hidden nodes drop out entirely: the overlay describes what is on screen, and
/// a node switched off is not.
pub fn visible_stats(scene: &[SceneNode]) -> SceneStats {
    let mut stats = SceneStats::default();
    for node in scene.iter().filter(|n| n.visible) {
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
