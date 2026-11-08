// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The files in a workspace root that configure the workspace as a whole.
//!
//! [`named_image_lists`] reads and validates `.sfm-named-image-lists.json`,
//! the named sets of the workspace's images, and resolves one list to the
//! workspace's image files or to a reconstruction's image names.
//! [`feature_options`] holds the rules of the feature options a
//! `.sfm-workspace.json` records: each feature tool's grey conversion, the
//! conversion of a version 1 record to version 2, and the name of the folder
//! a new record's `.sift` files go in. [`find_workspace_for_path`], the
//! upward search for `.sfm-workspace.json`, is `sfmtool-archive-io`'s, re-exported
//! here; reading that file is the caller's (see `specs/workspace/workspace.md`).

pub mod feature_options;
pub mod named_image_lists;

pub use sfmtool_archive_io::find_workspace_for_path;

pub use feature_options::{
    feature_prefix_dir, has_image_to_gray, recorded_image_to_gray, tool_image_to_gray,
    upgrade_feature_options, FeatureOptionsError, COLMAP_TOOL, OPENCV_TOOL, SFMTOOL_TOOL,
    UNDISTORT_TOOL, WORKSPACE_VERSION,
};
pub use named_image_lists::{
    is_named_image_list_name, split_named_image_list_value, DuplicateMatch, NamedImageList,
    NamedImageListEntry, NamedImageListMember, NamedImageListSlice, NamedImageListSliceError,
    NamedImageLists, NamedImageListsError, NAMED_IMAGE_LISTS_FILE_NAME, NAMED_IMAGE_LISTS_VERSION,
    WORKSPACE_FILE_NAME,
};
