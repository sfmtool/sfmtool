// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The count formatters, on their own.
//!
//! No frame and no fixture: these are the two spellings of a number the panel
//! uses — compact on the node row where the width is contested, exact on the
//! group rows where it is not — and they are pinned as the pure functions they
//! are. What draws them is covered by the panel's own tests, in
//! `scene_graph/tests.rs`.

use super::{compact_count, with_thousands};

#[test]
fn counts_are_formatted_compactly_on_the_node_row() {
    assert_eq!(compact_count(0), "0");
    assert_eq!(compact_count(999), "999");
    assert_eq!(compact_count(1_000), "1.0K");
    assert_eq!(compact_count(12_345), "12.3K");
    assert_eq!(compact_count(1_204_551), "1.2M");
}

#[test]
fn exact_counts_carry_thousands_separators_on_the_group_rows() {
    assert_eq!(with_thousands(0), "0");
    assert_eq!(with_thousands(999), "999");
    assert_eq!(with_thousands(1_000), "1,000");
    assert_eq!(with_thousands(1_204_551), "1,204,551");
}
