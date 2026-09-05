// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The reader's own guards, on the paths a corrupt file reaches.

use super::capped_capacity;

#[test]
fn capped_capacity_clamps_to_available_bytes() {
    // A hostile/corrupt count must be clamped to what the file could
    // actually hold, so it never reaches `Vec::with_capacity` as a value
    // that overflows the allocator (regression: an unclamped `u64::MAX`
    // count panicked with "capacity overflow").
    assert_eq!(capped_capacity(u64::MAX, 1000, 24), 1000 / 24);
    assert_eq!(capped_capacity(u64::MAX, 0, 24), 0);
    // A plausible count below the file-size bound is preserved exactly.
    assert_eq!(capped_capacity(5, 1000, 24), 5);
    // A zero minimum record size must not divide by zero.
    assert_eq!(capped_capacity(7, 1000, 0), 7);
}
