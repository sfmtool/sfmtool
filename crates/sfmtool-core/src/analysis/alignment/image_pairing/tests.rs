// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

use super::*;

fn hash(n: u8) -> Option<[u8; 16]> {
    Some([n; 16])
}

fn image(name: &str, file_hash: Option<[u8; 16]>) -> ImageIdentity<'_> {
    ImageIdentity { name, file_hash }
}

#[test]
fn images_with_the_same_file_pair_whatever_their_names() {
    let source = [image("a/0.jpg", hash(1)), image("a/1.jpg", hash(2))];
    let target = [image("b/x.jpg", hash(2)), image("b/y.jpg", hash(1))];
    let pairing = pair_images(&source, &target);
    assert_eq!(pairing.pairs, vec![(0, 1), (1, 0)]);
    assert_eq!((pairing.by_file_hash, pairing.by_name), (2, 0));
    assert!(pairing.same_name_different_file.is_empty());
}

#[test]
fn the_same_name_on_different_files_is_not_paired_and_is_reported() {
    let source = [
        image("frame_000001.jpg", hash(1)),
        image("frame_000002.jpg", hash(2)),
    ];
    let target = [
        image("frame_000001.jpg", hash(7)),
        image("frame_000002.jpg", hash(2)),
    ];
    let pairing = pair_images(&source, &target);
    assert_eq!(pairing.pairs, vec![(1, 1)]);
    assert_eq!((pairing.by_file_hash, pairing.by_name), (1, 0));
    assert_eq!(pairing.same_name_different_file, vec![(0, 0)]);
}

#[test]
fn renamed_files_are_paired_by_hash_and_their_old_names_reported() {
    // Two frames swapped names: by name alone each would pair with the other's
    // file. By hash they pair across the names, and both names are reported.
    let source = [image("0.jpg", hash(1)), image("1.jpg", hash(2))];
    let target = [image("1.jpg", hash(1)), image("0.jpg", hash(2))];
    let pairing = pair_images(&source, &target);
    assert_eq!(pairing.pairs, vec![(0, 0), (1, 1)]);
    assert_eq!(pairing.by_file_hash, 2);
    assert_eq!(pairing.same_name_different_file, vec![(0, 1), (1, 0)]);
}

#[test]
fn an_image_without_a_hash_pairs_by_name() {
    let source = [
        image("a.jpg", None),
        image("b.jpg", hash(2)),
        image("c.jpg", None),
    ];
    let target = [
        image("a.jpg", hash(1)),
        image("b.jpg", None),
        image("c.jpg", None),
    ];
    let pairing = pair_images(&source, &target);
    assert_eq!(pairing.pairs, vec![(0, 0), (1, 1), (2, 2)]);
    assert_eq!((pairing.by_file_hash, pairing.by_name), (0, 3));
    assert!(pairing.same_name_different_file.is_empty());
}

#[test]
fn an_image_paired_by_hash_is_not_paired_again_by_name() {
    // Source "a.jpg" is the target's "renamed.jpg" by hash, so the target's
    // hashless "a.jpg" finds no partner left.
    let source = [image("a.jpg", hash(1))];
    let target = [image("renamed.jpg", hash(1)), image("a.jpg", None)];
    let pairing = pair_images(&source, &target);
    assert_eq!(pairing.pairs, vec![(0, 0)]);
    assert_eq!((pairing.by_file_hash, pairing.by_name), (1, 0));
}

#[test]
fn a_hash_held_by_several_images_pairs_only_by_name_among_them() {
    // The same file under two names on both sides: the names say which is
    // which, and an image of the group with no same-named partner is left
    // unpaired, even though its file is on the other side.
    let source = [
        image("left.jpg", hash(5)),
        image("right.jpg", hash(5)),
        image("extra.jpg", hash(5)),
    ];
    let target = [
        image("right.jpg", hash(5)),
        image("left.jpg", hash(5)),
        image("other.jpg", hash(5)),
    ];
    let pairing = pair_images(&source, &target);
    assert_eq!(pairing.pairs, vec![(0, 1), (1, 0)]);
    assert_eq!((pairing.by_file_hash, pairing.by_name), (2, 0));
    assert!(pairing.same_name_different_file.is_empty());
}

#[test]
fn a_hash_repeated_on_one_side_only_pairs_by_name_within_it() {
    let source = [image("one.jpg", hash(5)), image("two.jpg", hash(5))];
    let target = [image("two.jpg", hash(5))];
    let pairing = pair_images(&source, &target);
    assert_eq!(pairing.pairs, vec![(1, 0)]);

    // With no same-named image in the group, nothing pairs.
    let target = [image("three.jpg", hash(5))];
    let pairing = pair_images(&source, &target);
    assert!(pairing.pairs.is_empty());
}

#[test]
fn a_repeated_name_pairs_once() {
    let source = [image("a.jpg", None), image("a.jpg", None)];
    let target = [image("a.jpg", None), image("a.jpg", None)];
    let pairing = pair_images(&source, &target);
    assert_eq!(pairing.pairs, vec![(0, 0)]);
    assert_eq!(pairing.by_name, 1);
}

#[test]
fn the_summary_gives_the_counts_and_the_names_shared_by_different_files() {
    let source = [
        image("a.jpg", hash(1)),
        image("b.jpg", None),
        image("c.jpg", hash(3)),
    ];
    let target = [
        image("x.jpg", hash(1)),
        image("b.jpg", None),
        image("c.jpg", hash(9)),
    ];
    let pairing = pair_images(&source, &target);
    assert_eq!(
        pairing.summary(),
        "images paired: 1 by file hash, 1 by name; 1 name shared by different files"
    );
    assert_eq!(
        pair_images(&source[..2], &target[..2]).summary(),
        "images paired: 1 by file hash, 1 by name"
    );
}
