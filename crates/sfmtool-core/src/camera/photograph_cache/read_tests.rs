// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The photographs `get` and `get_many` answer, which hold no tile and read
//! each through the cache.

use std::sync::atomic::{AtomicBool, AtomicUsize};

use crate::camera::sampler::{read_source_rect, test_pool, LevelRect};
use crate::progress::Progress;

use super::tests::with_budget;
use super::*;

/// Write a `w x h` PNG at `dir/name`, its pixels set by `seed`, and return its
/// path.
fn write_photo(dir: &Path, name: &str, w: u32, h: u32, seed: u8) -> PathBuf {
    let path = dir.join(name);
    let image = image::RgbImage::from_fn(w, h, |x, y| {
        image::Rgb([x as u8 ^ seed, y as u8, seed.wrapping_mul(3)])
    });
    image.save(&path).unwrap();
    path
}

/// Twenty 48 x 32 photographs in `dir`, each with pixels of its own.
fn twenty(dir: &Path) -> Vec<PathBuf> {
    (0..20u8)
        .map(|i| write_photo(dir, &format!("{i:02}.png"), 48, 32, i * 11))
        .collect()
}

/// Level 0 of `photograph`, read whole through a render entry point.
fn level_0(
    photograph: &TiledPhotograph,
    progress: &Progress<'_>,
) -> Result<Vec<u8>, PhotographReadError> {
    let (x1, y1) = photograph.layout().level_size(0);
    let rect = LevelRect {
        x0: 0,
        y0: 0,
        x1,
        y1,
    };
    let read = read_source_rect(photograph, test_pool(), 0, rect, progress)?;
    Ok(read.image().data().to_vec())
}

/// The file's own pixels, as a decode reads them.
fn pixels_of(path: &Path) -> Vec<u8> {
    ImageU8::read_rgb(path).unwrap().data().to_vec()
}

/// The bytes the cache keeps of one 48 x 32 photograph.
fn bytes_of_one(dir: &Path) -> u64 {
    let path = write_photo(dir, "one.png", 48, 32, 0);
    let cache = PhotographCache::new(with_budget(1 << 30));
    cache.get(&path, &Progress::none()).unwrap();
    cache.stats().bytes
}

/// A kernel's read phase, `get_many` over twenty photographs, decodes each
/// into tiles once, and when the tiles fit the budget its renders decode
/// nothing more: every answer holds no tile and reads the cache's.
#[test]
fn a_read_phase_decodes_every_photograph_so_its_renders_decode_nothing() {
    let dir = tempfile::tempdir().unwrap();
    let paths = twenty(dir.path());
    let refs: Vec<&Path> = paths.iter().map(PathBuf::as_path).collect();
    let cache = PhotographCache::new(with_budget(1 << 30));

    let (found, tally) = cache.get_many(&refs, &Progress::none()).unwrap();
    assert_eq!(tally.read, 20);
    assert_eq!(cache.stats().decodes, 20);
    for (photograph, path) in found.iter().zip(&paths) {
        let photograph = photograph.as_ref().expect("readable");
        assert_eq!(photograph.held_tiles(), 0, "it reads the cache's tiles");
        assert_eq!(
            level_0(photograph, &Progress::none()).unwrap(),
            pixels_of(path)
        );
    }
    assert_eq!(cache.stats().decodes, 20, "the renders decoded nothing");
}

/// A `get_many` over twenty photographs under a budget of three of them holds
/// no pyramid and no tile of its own: the cache keeps only tiles, within its
/// budget, and a read of a photograph whose tiles went decodes it again, with
/// the same pixels.
#[test]
fn a_get_many_over_twenty_photographs_holds_no_pyramid() {
    let dir = tempfile::tempdir().unwrap();
    let paths = twenty(dir.path());
    let refs: Vec<&Path> = paths.iter().map(PathBuf::as_path).collect();
    let budget = 3 * bytes_of_one(dir.path());
    let cache = PhotographCache::new(with_budget(budget));

    let (found, tally) = cache.get_many(&refs, &Progress::none()).unwrap();
    assert_eq!(tally.read, 20);
    let stats = cache.stats();
    assert!(stats.bytes <= budget, "{stats:?}");
    assert!(found
        .iter()
        .all(|p| p.as_ref().is_some_and(|p| p.held_tiles() == 0)));

    for (photograph, path) in found.iter().zip(&paths) {
        let photograph = photograph.as_ref().unwrap();
        assert_eq!(
            level_0(photograph, &Progress::none()).unwrap(),
            pixels_of(path)
        );
    }
    let stats = cache.stats();
    assert!(stats.bytes <= budget, "{stats:?}");
    assert!(
        stats.decodes >= 20 + 17,
        "the photographs whose tiles went were decoded again: {stats:?}"
    );
}

/// A read of a photograph whose file was removed after its tiles were evicted
/// fails with `NotFound`, naming it; one whose file was replaced fails with
/// `Changed`, naming it, rather than read the new file's pixels.
#[test]
fn a_read_after_eviction_of_a_removed_or_replaced_file_fails_naming_it() {
    let dir = tempfile::tempdir().unwrap();
    let a = write_photo(dir.path(), "a.png", 48, 32, 1);
    let b = write_photo(dir.path(), "b.png", 48, 32, 2);
    let c = write_photo(dir.path(), "c.png", 48, 32, 3);
    // Each decode keeps only its own tiles.
    let cache = PhotographCache::new(with_budget(1));
    let read_a = cache.get(&a, &Progress::none()).unwrap();
    let read_b = cache.get(&b, &Progress::none()).unwrap();
    cache.get(&c, &Progress::none()).unwrap();

    std::fs::remove_file(&a).unwrap();
    let error = level_0(&read_a, &Progress::none()).unwrap_err();
    assert_eq!(error, PhotographReadError::NotFound { path: a.clone() });
    assert!(error.to_string().contains("a.png"), "{error}");

    // The same size with other pixels: a file whose length or modification
    // time is not the one the answer was decoded from.
    write_photo(dir.path(), "b.png", 48, 32, 200);
    let error = level_0(&read_b, &Progress::none()).unwrap_err();
    assert_eq!(error, PhotographReadError::Changed { path: b.clone() });
    assert!(error.to_string().contains("b.png"), "{error}");
}

/// A cache whose decodes after the first `after` set `flag`, as a caller's
/// cancel would arrive during them.
fn cancelling_cache(budget: u64, after: usize, flag: &Arc<AtomicBool>) -> PhotographCache {
    let mut cache = PhotographCache::new(with_budget(budget));
    let started = AtomicUsize::new(0);
    let flag = Arc::clone(flag);
    cache.set_slot_hook(Box::new(move |_, event| {
        if event == SlotEvent::DecodeStarted && started.fetch_add(1, Ordering::SeqCst) >= after {
            flag.store(true, Ordering::SeqCst);
        }
    }));
    cache
}

/// A render that decodes a photograph whose tiles were evicted stops with the
/// cancellation that arrives while it decodes, and leaves the photograph
/// readable for the next render.
#[test]
fn a_render_cancelled_while_it_decodes_an_evicted_tile_stops() {
    let dir = tempfile::tempdir().unwrap();
    let a = write_photo(dir.path(), "a.png", 48, 32, 1);
    let b = write_photo(dir.path(), "b.png", 48, 32, 2);
    let flag = Arc::new(AtomicBool::new(false));
    let cache = cancelling_cache(1, 2, &flag);
    let read_a = cache.get(&a, &Progress::none()).unwrap();
    cache.get(&b, &Progress::none()).unwrap();
    assert!(!flag.load(Ordering::SeqCst));

    let error = level_0(&read_a, &Progress::none().cancelled_by(&flag)).unwrap_err();
    assert_eq!(error, PhotographReadError::Cancelled);
    assert_eq!(
        cache.stats().decodes,
        3,
        "the cancel came during the decode"
    );
    assert_eq!(level_0(&read_a, &Progress::none()).unwrap(), pixels_of(&a));
}

/// Under a budget of 0 the answer holds every tile of the decode that made
/// it, so two renders of it decode the photograph once.
#[test]
fn under_a_budget_of_0_two_renders_of_one_answer_decode_once() {
    let dir = tempfile::tempdir().unwrap();
    let a = write_photo(dir.path(), "a.png", 48, 32, 1);
    let cache = PhotographCache::new(with_budget(0));
    let read_a = cache.get(&a, &Progress::none()).unwrap();
    assert_eq!(read_a.held_tiles(), tile_count(read_a.layout()));
    for _ in 0..2 {
        assert_eq!(level_0(&read_a, &Progress::none()).unwrap(), pixels_of(&a));
    }
    assert_eq!(cache.stats().decodes, 1);
}
