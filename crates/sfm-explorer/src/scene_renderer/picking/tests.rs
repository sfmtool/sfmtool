// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Pick encode/decode round-trip tests, including the boundary values that a
//! 2-bit tag plus a 30-bit global index makes reachable.

use super::*;

const A: ReconId = ReconId::from_raw(0);
const B: ReconId = ReconId::from_raw(1);
const C: ReconId = ReconId::from_raw(2);

/// Two reconstructions, points `[0, 10)` and `[10, 13)`, images `[0, 2)` and
/// `[2, 5)` — the shape [`super::super::SceneRenderer::assign_pick_bases`]
/// produces, built here by hand so decode is tested on its own.
fn two_nodes() -> PickTables {
    let mut tables = PickTables::default();
    tables.push(A, 0, 10, 0, 2);
    tables.push(B, 10, 3, 2, 3);
    tables
}

#[test]
fn a_point_id_round_trips_through_its_reconstruction() {
    let tables = two_nodes();

    for (global, recon, local) in [(0, A, 0), (9, A, 9), (10, B, 0), (12, B, 2)] {
        let id = encode(PICK_TAG_POINT, global);
        assert_eq!(
            tables.resolve(id),
            Some(PickTarget::Point(PointRef::new(recon, local))),
            "global point index {global}",
        );
    }
}

#[test]
fn an_image_id_round_trips_through_its_reconstruction() {
    let tables = two_nodes();

    for (global, recon, local) in [(0, A, 0), (1, A, 1), (2, B, 0), (4, B, 2)] {
        let id = encode(PICK_TAG_FRUSTUM, global);
        assert_eq!(
            tables.resolve(id),
            Some(PickTarget::Image(ImageRef::new(recon, local))),
            "global image index {global}",
        );
    }
}

#[test]
fn the_two_index_spaces_are_independent() {
    let tables = two_nodes();

    // Global index 2 is A's third point but B's first image: the tag, not the
    // index, decides which table is consulted.
    assert_eq!(
        tables.resolve(encode(PICK_TAG_POINT, 2)),
        Some(PickTarget::Point(PointRef::new(A, 2))),
    );
    assert_eq!(
        tables.resolve(encode(PICK_TAG_FRUSTUM, 2)),
        Some(PickTarget::Image(ImageRef::new(B, 0))),
    );
}

#[test]
fn the_last_index_of_a_range_resolves_and_the_next_one_does_not() {
    let mut tables = PickTables::default();
    // A single node with a gap after it: index `count` belongs to nobody.
    tables.push(A, 0, 4, 0, 4);

    assert_eq!(
        tables.resolve(encode(PICK_TAG_POINT, 3)),
        Some(PickTarget::Point(PointRef::new(A, 3))),
    );
    assert_eq!(tables.resolve(encode(PICK_TAG_POINT, 4)), None);
}

#[test]
fn the_top_of_the_index_space_is_addressable() {
    let base = PICK_INDEX_CAPACITY - 4;
    let mut tables = PickTables::default();
    tables.push(A, base, 4, base, 4);

    let top = PICK_INDEX_CAPACITY - 1;
    let id = encode(PICK_TAG_POINT, top);
    // The tag must survive the largest representable index untouched.
    assert_eq!(id & PICK_TAG_MASK, PICK_TAG_POINT);
    assert_eq!(id & PICK_INDEX_MASK, top);
    assert_eq!(
        tables.resolve(id),
        Some(PickTarget::Point(PointRef::new(A, 3))),
    );
}

#[test]
fn the_none_value_and_the_all_ones_sentinel_decode_to_nothing() {
    let tables = two_nodes();

    // Zero is the pick texture's clear value: tag 0, index 0.
    assert_eq!(PICK_TAG_NONE, 0);
    assert_eq!(tables.resolve(PICK_TAG_NONE), None);
    // The uniform sentinel decodes to the reserved tag 3, so a readback that
    // somehow contained it cannot be mistaken for an entity.
    assert_eq!(PICK_INDEX_NONE & PICK_TAG_MASK, 0b11 << PICK_TAG_SHIFT);
    assert_eq!(tables.resolve(PICK_INDEX_NONE), None);
}

#[test]
fn the_first_entity_of_the_first_node_is_distinct_from_nothing() {
    let tables = two_nodes();

    // Base 0, local 0 — the case an 8-bit-tag scheme with tag 0 would have
    // collided with the background value.
    let id = encode(PICK_TAG_FRUSTUM, 0);
    assert_ne!(id, PICK_TAG_NONE);
    assert_eq!(
        tables.resolve(id),
        Some(PickTarget::Image(ImageRef::new(A, 0))),
    );
}

#[test]
fn an_index_past_every_range_resolves_to_nothing() {
    let tables = two_nodes();

    assert_eq!(tables.resolve(encode(PICK_TAG_POINT, 13)), None);
    assert_eq!(tables.resolve(encode(PICK_TAG_FRUSTUM, 5)), None);
}

#[test]
fn empty_tables_resolve_nothing() {
    let tables = PickTables::default();

    assert_eq!(tables.resolve(encode(PICK_TAG_POINT, 0)), None);
    assert_eq!(tables.resolve(encode(PICK_TAG_FRUSTUM, 7)), None);
}

#[test]
fn a_node_with_no_entities_never_captures_an_index() {
    let mut tables = PickTables::default();
    // B contributes an empty range at the same base as C's first entity, which
    // is exactly what an imageless reconstruction between two others produces.
    tables.push(A, 0, 2, 0, 2);
    tables.push(B, 2, 0, 2, 0);
    tables.push(C, 2, 2, 2, 2);

    assert_eq!(
        tables.resolve(encode(PICK_TAG_POINT, 2)),
        Some(PickTarget::Point(PointRef::new(C, 0))),
    );
    assert_eq!(
        tables.resolve(encode(PICK_TAG_FRUSTUM, 3)),
        Some(PickTarget::Image(ImageRef::new(C, 1))),
    );
}
