// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The scene graph: the reconstructions the viewer has loaded, and the typed
//! references that address entities inside them.
//!
//! See `specs/gui/gui-scene-graph.md`. This is phase 1 of that design: the
//! identity types and the node exist, and every cache and selection is keyed by
//! them, but the scene still holds **at most one** node — `File > Open`
//! replaces rather than appends, and there is no Scene Graph panel yet.
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

use sfmtool_core::SfmrReconstruction;

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
/// Phase 1 carries only what the single-reconstruction viewer already needed.
/// The spec's remaining fields — `visible`, `interactive`, the per-group eyes,
/// `tint` and `transform` — arrive with phases 3 and 4, when there is a Scene
/// Graph panel to drive them and per-node GPU resources to apply them to; they
/// would be dead code until then.
pub struct SceneNode {
    pub id: ReconId,
    /// Display label: the file stem, or `"demo"` for demo data. Collision
    /// disambiguation (`" (2)"`, `" (3)"`…) arrives with multi-load in phase 3.
    ///
    /// Nothing displays it yet — the Scene Graph panel that does is phase 3 —
    /// but the label belongs to node creation, which is what phase 1 unifies.
    #[allow(dead_code)]
    pub label: String,
    /// Source path; `None` for demo data.
    pub path: Option<PathBuf>,
    pub recon: SfmrReconstruction,
    /// This node's data needs (re-)upload to the GPU. Replaces the former
    /// global `AppState::points_need_upload`.
    pub needs_upload: bool,
}

impl SceneNode {
    /// A node for a reconstruction read from `path`, labeled with its file stem.
    pub fn from_path(path: &Path, recon: SfmrReconstruction) -> Self {
        let label = path
            .file_stem()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| path.display().to_string());
        Self {
            id: ReconId::next(),
            label,
            path: Some(path.to_path_buf()),
            recon,
            needs_upload: true,
        }
    }

    /// A node for generated demo data, which came from no file.
    pub fn demo(recon: SfmrReconstruction) -> Self {
        Self {
            id: ReconId::next(),
            label: "demo".to_string(),
            path: None,
            recon,
            needs_upload: true,
        }
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
