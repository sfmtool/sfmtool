// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The files in a workspace root that configure the workspace as a whole.
//!
//! [`named_image_lists`] reads and validates `.sfm-named-image-lists.json`,
//! the named sets of the workspace's images, and resolves one list to the
//! workspace's image files or to a reconstruction's image names. Finding the
//! workspace root, by the upward search for `.sfm-workspace.json`, is the
//! caller's (see `specs/workspace/workspace.md`).

pub mod named_image_lists;

pub use named_image_lists::{
    is_named_image_list_name, DuplicateMatch, NamedImageList, NamedImageListEntry,
    NamedImageListMember, NamedImageLists, NamedImageListsError, NAMED_IMAGE_LISTS_FILE_NAME,
    NAMED_IMAGE_LISTS_VERSION, WORKSPACE_FILE_NAME,
};
