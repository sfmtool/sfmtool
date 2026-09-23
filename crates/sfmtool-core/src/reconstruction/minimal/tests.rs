// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

use std::collections::BTreeMap;
use std::path::Path;
use std::sync::Arc;

use ndarray::{Array2, Array4};
use sfmtool_sfmr_format::{LineageEntry, LineageMap, LINEAGE_KIND_BASE};

use super::SaveStamp;
use crate::progress::Progress;
use crate::{SfmrReconstruction, THUMBNAIL_SIZE};

const STAMP: SaveStamp<'static> = SaveStamp {
    operation: "minimal",
    tool: "sfm-explorer",
    tool_version: "9.9.9",
};

/// A demo value with everything `to_minimal` drops or clears: both heavy
/// columns, a patch frame per point, a lineage entry, an absolute path and an
/// inherited option, over a workspace at `workspace`, which is
/// created with its marker so a file written beside it resolves it.
fn heavy(workspace: &Path) -> SfmrReconstruction {
    std::fs::create_dir_all(workspace).unwrap();
    std::fs::write(workspace.join(".sfm-workspace.json"), "{}").unwrap();
    let mut recon = SfmrReconstruction::demo(12);
    recon.workspace_dir = workspace.to_path_buf();
    let images = recon.image_table.images.len();
    let points = recon.point_set.points.len();
    recon.image_table.thumbnails_y_x_rgb = Some(Arc::new(Array4::from_elem(
        (images, THUMBNAIL_SIZE, THUMBNAIL_SIZE, 3),
        77u8,
    )));
    let mut u = Array2::<f32>::zeros((points, 3));
    let mut v = Array2::<f32>::zeros((points, 3));
    for p in 0..points {
        u[[p, 0]] = 0.05;
        v[[p, 1]] = 0.05;
    }
    recon.point_set.patch_u_halfvec_xyz = Some(u);
    recon.point_set.patch_v_halfvec_xyz = Some(v);
    recon.point_set.patch_bitmaps_y_x_rgba =
        Some(Arc::new(Array4::from_elem((points, 4, 4, 4), 200u8)));
    recon.metadata.lineage.push(LineageEntry {
        hash: "0123456789abcdef0123456789abcdef".to_string(),
        kind: LINEAGE_KIND_BASE.to_string(),
        map: LineageMap::Dense {
            rows: (0..points as u32).map(Some).collect(),
        },
    });
    recon.metadata.workspace.absolute_path = "/some/other/machine".to_string();
    recon
        .metadata
        .tool_options
        .insert("inherited".to_string(), serde_json::json!(true));
    recon
}

#[test]
fn stamp_save_records_the_tool_and_the_workspace_from_the_output() {
    let dir = tempfile::tempdir().unwrap();
    let workspace = dir.path().join("ws");
    let mut recon = heavy(&workspace);
    recon.stamp_save(&dir.path().join("out").join("a.sfmr"), &STAMP);

    assert_eq!(recon.metadata.operation, "minimal");
    assert_eq!(recon.metadata.tool, "sfm-explorer");
    assert_eq!(recon.metadata.tool_version, "9.9.9");
    assert_eq!(recon.metadata.workspace.relative_path, "../ws");
    assert_eq!(
        recon.metadata.workspace.absolute_path,
        workspace.to_string_lossy()
    );
    assert_eq!(recon.metadata.point_count as usize, recon.point_count());
    // Nothing a minimal save clears is touched by the stamp alone.
    assert_eq!(recon.metadata.lineage.len(), 1);
    assert!(recon.metadata.tool_options.contains_key("inherited"));
}

#[test]
fn to_minimal_drops_the_heavy_columns_and_the_incidental_metadata() {
    let dir = tempfile::tempdir().unwrap();
    let recon = heavy(&dir.path().join("ws"));
    let out = dir.path().join("published").join("a.sfmr");
    let mut options = BTreeMap::new();
    options.insert("transforms".to_string(), serde_json::json!(["Minimal"]));
    let minimal = recon.to_minimal(&out, &STAMP, options.clone());

    assert!(minimal.image_table.thumbnails_y_x_rgb.is_none());
    assert!(minimal.point_set.patch_bitmaps_y_x_rgba.is_none());
    assert!(!minimal.point_set.patch_bitmaps_for_display);
    // The geometry stays, so bitmaps can be rendered back onto it.
    assert!(minimal.point_set.patch_u_halfvec_xyz.is_some());
    assert!(minimal.point_set.patch_v_halfvec_xyz.is_some());
    assert!(minimal.metadata.workspace.absolute_path.is_empty());
    assert_eq!(minimal.metadata.workspace.relative_path, "../ws");
    assert!(minimal.metadata.lineage.is_empty());
    assert_eq!(minimal.metadata.tool_options, options);
    assert_eq!(minimal.metadata.operation, "minimal");
    // Not the content the input was read as.
    assert!(minimal.content_hash.content_xxh128.is_empty());
    // The input is untouched.
    assert!(recon.image_table.thumbnails_y_x_rgb.is_some());
    assert_eq!(recon.metadata.lineage.len(), 1);

    minimal.save(&out).unwrap();
    let read = SfmrReconstruction::load(&out, &Progress::none()).unwrap();
    assert!(read.image_table.thumbnails_y_x_rgb.is_none());
    assert!(read.point_set.patch_bitmaps_y_x_rgba.is_none());
    assert!(read.point_set.patch_u_halfvec_xyz.is_some());
    assert!(read.metadata.workspace.absolute_path.is_empty());
    assert!(read.metadata.lineage.is_empty());
}

/// A workspace reached by a path that is not its real one: the relative path is
/// still the step between the two directories, not a walk from the root. A host
/// whose temporary directory is a symlink or a short alias hands the two sides
/// in different forms, which is the same situation as an unresolved component
/// here.
#[test]
fn the_relative_path_is_resolved_before_it_is_measured() {
    let dir = tempfile::tempdir().unwrap();
    let workspace = dir.path().join("ws");
    let mut recon = heavy(&workspace);
    // The same directory, named through a detour that only resolving removes.
    recon.workspace_dir = workspace.join("sub").join("..");
    std::fs::create_dir_all(workspace.join("sub")).unwrap();
    let out = dir.path().join("published");
    std::fs::create_dir_all(&out).unwrap();
    recon.stamp_save(&out.join("a.sfmr"), &STAMP);

    assert_eq!(recon.metadata.workspace.relative_path, "../ws");
}

/// The same value written minimal from two machines' worth of absolute paths,
/// to the same place relative to its workspace, is the same content.
#[test]
fn a_minimal_file_does_not_depend_on_where_the_workspace_is() {
    let first = tempfile::tempdir().unwrap();
    let second = tempfile::tempdir().unwrap();
    let hash_of = |root: &Path| {
        let recon = heavy(&root.join("ws"));
        let out = root.join("published").join("a.sfmr");
        recon
            .to_minimal(&out, &STAMP, BTreeMap::new())
            .content_xxh128()
            .unwrap()
            .content_xxh128
    };
    assert_eq!(hash_of(first.path()), hash_of(second.path()));
}

/// A patch bitmap column rendered for display never reaches a file or a hash:
/// the value keeps the identity of the file it was read from.
#[test]
fn a_display_bitmap_column_is_left_out_of_the_save_and_the_hash() {
    let dir = tempfile::tempdir().unwrap();
    let mut recon = heavy(&dir.path().join("ws"));
    let bitmaps = recon.point_set.patch_bitmaps_y_x_rgba.take();
    let without = recon.content_xxh128().unwrap();

    recon.point_set.patch_bitmaps_y_x_rgba = bitmaps;
    let with_own = recon.content_xxh128().unwrap();
    assert_ne!(with_own.content_xxh128, without.content_xxh128);

    recon.point_set.patch_bitmaps_for_display = true;
    assert!(recon.to_sfmr_data().patch_bitmaps_y_x_rgba.is_none());
    assert_eq!(
        recon.content_xxh128().unwrap().content_xxh128,
        without.content_xxh128
    );

    let out = dir.path().join("display.sfmr");
    recon.stamp_save(&out, &STAMP);
    // Hashed with the display column in place, which is what the session holds.
    let in_memory = recon.content_xxh128().unwrap();
    recon.save(&out).unwrap();
    let read = SfmrReconstruction::load(&out, &Progress::none()).unwrap();
    assert!(read.point_set.patch_bitmaps_y_x_rgba.is_none());
    assert!(!read.point_set.patch_bitmaps_for_display);
    assert_eq!(read.content_hash.content_xxh128, in_memory.content_xxh128);
}

/// Every pass that selects rows carries the mark with the column.
#[test]
fn a_row_selection_carries_the_display_mark() {
    let dir = tempfile::tempdir().unwrap();
    let mut recon = heavy(&dir.path().join("ws"));
    recon.point_set.patch_bitmaps_for_display = true;
    let keep: Vec<bool> = (0..recon.point_count()).map(|p| p % 2 == 0).collect();
    let filtered = recon.filter_points_by_mask(&keep);
    assert!(filtered.point_set.patch_bitmaps_y_x_rgba.is_some());
    assert!(filtered.point_set.patch_bitmaps_for_display);
}
