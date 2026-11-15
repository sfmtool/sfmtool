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
    assert!(pairing.by_name_different_files.is_empty());
}

#[test]
fn the_same_name_on_different_files_pairs_by_name_and_is_reported() {
    let source = [
        image("frame_000001.jpg", hash(1)),
        image("frame_000002.jpg", hash(2)),
    ];
    let target = [
        image("frame_000001.jpg", hash(7)),
        image("frame_000002.jpg", hash(2)),
    ];
    let pairing = pair_images(&source, &target);
    assert_eq!(pairing.pairs, vec![(0, 0), (1, 1)]);
    assert_eq!((pairing.by_file_hash, pairing.by_name), (1, 1));
    assert_eq!(pairing.by_name_different_files, vec![(0, 0)]);
}

#[test]
fn the_same_images_re_encoded_pair_by_name() {
    // The same photographs at another resolution: every file differs, every
    // name matches, and every image pairs by name as it would with no hashes.
    let names = ["0.jpg", "1.jpg", "2.jpg", "3.jpg"];
    let source: Vec<_> = (0..4u8)
        .map(|i| image(names[i as usize], hash(i)))
        .collect();
    let target: Vec<_> = (0..4u8)
        .map(|i| image(names[i as usize], hash(100 + i)))
        .collect();
    let pairing = pair_images(&source, &target);
    let all: Vec<(u32, u32)> = (0..4).map(|i| (i, i)).collect();
    assert_eq!(pairing.pairs, all);
    assert_eq!((pairing.by_file_hash, pairing.by_name), (0, 4));
    assert_eq!(pairing.by_name_different_files, all);
}

#[test]
fn renamed_files_pair_by_hash_before_any_name_is_read() {
    // Two frames swapped names: by name alone each would pair with the other's
    // file. By hash they pair across the names, which leaves nothing to pair
    // by name.
    let source = [image("0.jpg", hash(1)), image("1.jpg", hash(2))];
    let target = [image("1.jpg", hash(1)), image("0.jpg", hash(2))];
    let pairing = pair_images(&source, &target);
    assert_eq!(pairing.pairs, vec![(0, 0), (1, 1)]);
    assert_eq!((pairing.by_file_hash, pairing.by_name), (2, 0));
    assert!(pairing.by_name_different_files.is_empty());
}

#[test]
fn renamed_same_files_and_same_named_different_files_pair_together() {
    let source = [
        image("a.jpg", hash(1)),
        image("b.jpg", hash(2)),
        image("c.jpg", hash(3)),
        image("d.jpg", hash(4)),
    ];
    let target = [
        image("moved/a.jpg", hash(1)),
        image("b.jpg", hash(20)),
        image("c.jpg", hash(3)),
        image("a.jpg", hash(40)),
        image("d.jpg", None),
    ];
    let pairing = pair_images(&source, &target);
    // a.jpg is the target's moved/a.jpg by hash, so the target's a.jpg, a
    // different file, finds no partner left; b.jpg pairs by name across two
    // files; c.jpg pairs by hash; d.jpg by name, its target hash not known.
    assert_eq!(pairing.pairs, vec![(0, 0), (1, 1), (2, 2), (3, 4)]);
    assert_eq!((pairing.by_file_hash, pairing.by_name), (2, 2));
    assert_eq!(pairing.by_name_different_files, vec![(1, 1)]);
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
    assert!(pairing.by_name_different_files.is_empty());
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
    // which. An image of the group with no same-named partner in the other
    // side's group is not paired by its hash, even though its file is on the
    // other side; it is left to the pairing by name.
    let source = [
        image("left.jpg", hash(5)),
        image("right.jpg", hash(5)),
        image("extra.jpg", hash(5)),
    ];
    let target = [
        image("right.jpg", hash(5)),
        image("left.jpg", hash(5)),
        image("other.jpg", hash(5)),
        image("extra.jpg", hash(6)),
    ];
    let pairing = pair_images(&source, &target);
    assert_eq!(pairing.pairs, vec![(0, 1), (1, 0), (2, 3)]);
    assert_eq!((pairing.by_file_hash, pairing.by_name), (2, 1));
    assert_eq!(pairing.by_name_different_files, vec![(2, 3)]);
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
fn the_summary_gives_the_counts_and_the_name_pairs_of_different_files() {
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
        "images paired: 1 by file hash, 2 by name, 1 of them different files"
    );
    assert_eq!(
        pair_images(&source[..2], &target[..2]).summary(),
        "images paired: 1 by file hash, 1 by name"
    );
}

/// The pairs a pairing by name alone makes: the first image of each name on
/// each side.
fn pairs_by_name_alone(
    source: &[ImageIdentity<'_>],
    target: &[ImageIdentity<'_>],
) -> Vec<(usize, usize)> {
    let target_by_name = first_by_name(target, 0..target.len());
    let mut seen = HashSet::new();
    source
        .iter()
        .enumerate()
        .filter(|(_, image)| seen.insert(image.name))
        .filter_map(|(s, image)| target_by_name.get(image.name).map(|&t| (s, t)))
        .collect()
}

/// A deterministic sequence of small numbers, each below the bound asked for.
struct Draws(u64);

impl Draws {
    fn below(&mut self, n: u64) -> usize {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((self.0 >> 33) % n) as usize
    }

    /// Up to five images, each with one of three names and one of three
    /// hashes or none, so that names repeat, hashes repeat, and the two
    /// disagree in every way.
    fn side(&mut self) -> Vec<ImageIdentity<'static>> {
        const NAMES: [&str; 3] = ["a.jpg", "b.jpg", "c.jpg"];
        (0..self.below(6))
            .map(|_| {
                let name = NAMES[self.below(3)];
                let file_hash = [None, hash(1), hash(2), hash(3)][self.below(4)];
                image(name, file_hash)
            })
            .collect()
    }
}

#[test]
fn every_pair_by_name_alone_is_kept_unless_a_hash_pair_took_an_image() {
    let mut draws = Draws(0x9e37_79b9_7f4a_7c15);
    for case in 0..2000 {
        let source = draws.side();
        let target = draws.side();
        let pairing = pair_images(&source, &target);
        let partner_of_source: HashMap<usize, usize> = pairing
            .pairs
            .iter()
            .map(|&(s, t)| (s as usize, t as usize))
            .collect();
        let partner_of_target: HashMap<usize, usize> =
            partner_of_source.iter().map(|(&s, &t)| (t, s)).collect();
        let same_file = |s: usize, t: usize| {
            source[s].file_hash.is_some() && source[s].file_hash == target[t].file_hash
        };
        for (s, t) in pairs_by_name_alone(&source, &target) {
            let kept = partner_of_source.get(&s) == Some(&t);
            let taken_by_hash = partner_of_source
                .get(&s)
                .is_some_and(|&other| same_file(s, other))
                || partner_of_target
                    .get(&t)
                    .is_some_and(|&other| same_file(other, t));
            assert!(
                kept || taken_by_hash,
                "case {case}: name pair ({s}, {t}) lost in {source:?} / {target:?}"
            );
        }
        let different_files: Vec<(u32, u32)> = pairing
            .pairs
            .iter()
            .copied()
            .filter(|&(s, t)| {
                matches!(
                    (source[s as usize].file_hash, target[t as usize].file_hash),
                    (Some(a), Some(b)) if a != b
                )
            })
            .collect();
        assert_eq!(
            pairing.by_name_different_files, different_files,
            "case {case}"
        );
        assert_eq!(pairing.by_file_hash + pairing.by_name, pairing.pairs.len());
    }
}
