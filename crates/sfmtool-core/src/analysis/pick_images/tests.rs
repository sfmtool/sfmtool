// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The walk on synthetic captures: a scene of points with random descriptors
//! seen along a camera path with known motion per frame, written as version 2
//! `.sift` files by a pool of workers only for the frames the source is asked
//! to prefetch or wait for.

mod capture;
mod density;
mod pilot;

use std::collections::BTreeSet;
use std::path::PathBuf;
use std::sync::atomic::AtomicBool;
use std::sync::OnceLock;

use capture::{sharp, sift_files, Asked, Content, Fault, Synth, SynthSource, ONE_PERCENT};

use super::*;
use crate::numeric::median;
use crate::progress::Progress;

/// The match floor of [`options`]: a fraction of the matches a step in the
/// band keeps at the synthetic captures' densities, which are far below a
/// real frame's features.
const FLOOR: usize = 60;

/// The options of the walk tests: the defaults with the match floor at
/// [`FLOOR`] and the pilot vote off, so that a sequence with no camera is
/// walked without the local density term and no test of the walk pays for a
/// pilot it does not assert on. The pilot's tests use the default, on.
fn options() -> PickImagesOptions {
    PickImagesOptions {
        min_step_matches: FLOOR,
        pilot_vote: false,
        ..PickImagesOptions::default()
    }
}

/// How many workers the test source runs unless a test says otherwise.
const THREADS: usize = 4;

struct Run {
    result: Result<PickImagesReport, PickImagesError>,
    /// Every frame waited for, in order.
    asked: Vec<FrameId>,
    /// The frames of each wait.
    waits: Vec<Vec<FrameId>>,
    /// Every frame asked to be prefetched.
    prefetched: BTreeSet<FrameId>,
    sequences: Vec<CaptureSequence>,
    dir: tempfile::TempDir,
}

impl Run {
    fn report(&self) -> &PickImagesReport {
        match &self.result {
            Ok(report) => report,
            Err(e) => panic!("pick_images failed: {e}"),
        }
    }

    fn walk(&self, sequence: usize) -> &SequenceReport {
        &self.report().sequences[sequence]
    }
}

fn run_with(
    synths: &[&Synth],
    options: &PickImagesOptions,
    fault: Fault,
    cancel: Option<&AtomicBool>,
) -> Run {
    run_on(synths, options, fault, cancel, THREADS)
}

fn run_on(
    synths: &[&Synth],
    options: &PickImagesOptions,
    fault: Fault,
    cancel: Option<&AtomicBool>,
    threads: usize,
) -> Run {
    let dir = tempfile::tempdir().expect("a temporary folder");
    let sequences: Vec<CaptureSequence> = synths
        .iter()
        .enumerate()
        .map(|(i, s)| s.sequence(&dir.path().join(format!("seq{i}"))))
        .collect();
    let flag = AtomicBool::new(false);
    let flag = cancel.unwrap_or(&flag);
    let progress = Progress::none().cancelled_by(flag);
    let mut source = SynthSource::new(synths, threads, fault).cancelled_by(flag);
    let result = pick_images(&sequences, &mut source, options, &progress);
    let Asked { waits, prefetched } = source.into_asked();
    Run {
        result,
        asked: waits.iter().flatten().copied().collect(),
        waits,
        prefetched,
        sequences,
        dir,
    }
}

fn run(synths: &[&Synth]) -> Run {
    run_with(synths, &options(), Fault::None, None)
}

fn strides(walk: &SequenceReport, keep: impl Fn(&WalkStep) -> bool) -> Vec<f64> {
    walk.steps
        .iter()
        .filter(|s| keep(s))
        .map(|s| (s.to - s.from) as f64)
        .collect()
}

/// The frames waited for are exactly the frames the report lists as read, in
/// the same order and each once; every frame read has `.sift` files, written
/// in the run; and the other frames with `.sift` files were prefetched, as
/// many as the report counts as extracted and never read.
fn assert_read_is_asked(run: &Run) {
    let report = run.report();
    let read: Vec<FrameId> = report.frames.iter().map(|f| f.id).collect();
    assert_eq!(read, run.asked);
    let unique: BTreeSet<FrameId> = run.asked.iter().copied().collect();
    assert_eq!(
        unique.len(),
        run.asked.len(),
        "a frame was waited for twice"
    );
    assert_eq!(report.frames_extracted, vec![true; read.len()]);
    let mut unread = 0;
    for (s, sequence) in run.sequences.iter().enumerate() {
        let folder = run.dir.path().join(format!("seq{s}"));
        let files: BTreeSet<PathBuf> = sift_files(&folder).into_iter().collect();
        for (t, frame) in sequence.frames.iter().enumerate() {
            let id = FrameId {
                sequence: s,
                frame: t,
            };
            let has = frame.images.iter().all(|i| files.contains(&i.sift_path));
            if unique.contains(&id) {
                assert!(has, "{id:?} was read but has no .sift files");
            } else if has {
                assert!(run.prefetched.contains(&id), "{id:?} has files unasked");
                unread += 1;
            } else {
                assert!(frame.images.iter().all(|i| !files.contains(&i.sift_path)));
            }
        }
        let read_here = read.iter().filter(|id| id.sequence == s).count();
        assert_eq!(report.sequences[s].frames_read, read_here);
    }
    assert_eq!(report.frames_extracted_unread, unread);
}

/// The band at [`options`]' target.
fn band() -> (f64, f64) {
    let o = options();
    (
        o.target_step_percent * (1.0 - o.step_band),
        o.target_step_percent * (1.0 + o.step_band),
    )
}

#[test]
fn a_steady_video_is_chosen_at_about_the_target_step() {
    let synth = Synth::steady(1, 300, ONE_PERCENT);
    let run = run(&[&synth]);
    let report = run.report();
    let walk = run.walk(0);
    assert_eq!(walk.chosen.first(), Some(&0));
    assert_eq!(walk.chosen.last(), Some(&299));
    // The last step is below the band at the last frame, which a step does
    // not pass.
    let last = walk.steps.last().expect("steps");
    assert_eq!((last.to, last.kind), (299, StepKind::AtLimit), "{last:?}");
    // With no recent step, the walk measures the next frame first and aims
    // from its rate: the first frame waited for after frame 0 is frame 1.
    assert_eq!(run.asked[0].frame, 0);
    assert_eq!(run.asked[1].frame, 1);
    let steps: Vec<f64> = walk.steps.iter().map(|s| s.stats.step_percent).collect();
    let median_step = median(&steps);
    assert!(
        (7.0..=13.0).contains(&median_step),
        "median step {median_step}"
    );
    let in_band = walk
        .steps
        .iter()
        .filter(|s| s.kind == StepKind::InBand)
        .count();
    assert!(
        in_band * 10 >= walk.steps.len() * 9,
        "{in_band} of {}",
        walk.steps.len()
    );
    let stride = median(&strides(walk, |_| true));
    assert!((8.0..=12.0).contains(&stride), "median stride {stride}");
    assert!(walk.swaps.is_empty(), "{:?}", walk.swaps);
    // Each chosen frame is read with the frames within 2 positions of it, and
    // the walk reads few frames beyond those.
    assert!(
        walk.frames_read <= walk.chosen.len() * 7,
        "{} read",
        walk.frames_read
    );
    assert!(report.frames.len() < 300);
    assert_read_is_asked(&run);
    // A frame never read lies inside a step in the band.
    let asked: BTreeSet<usize> = run.asked.iter().map(|id| id.frame).collect();
    for t in (0..300).filter(|t| !asked.contains(t)) {
        assert!(walk
            .steps
            .iter()
            .any(|s| s.from < t && t < s.to && s.kind == StepKind::InBand));
    }
}

#[test]
fn the_stride_adapts_to_the_speed() {
    // 0.5% per frame for 200 frames, then 2% per frame for 150.
    let mut x = 0.0;
    let xs: Vec<f64> = (0..350)
        .map(|t| {
            let here = x;
            x += if t < 200 { 0.5 } else { 2.0 } * ONE_PERCENT;
            here
        })
        .collect();
    let length = *xs.last().unwrap();
    let synth = Synth::wall(2, -12.0, length + 12.0, 80.0).along(xs);
    let run = run(&[&synth]);
    let walk = run.walk(0);
    let slow = median(&strides(walk, |s| s.to < 190));
    let fast = median(&strides(walk, |s| s.from > 215));
    assert!((15.0..=25.0).contains(&slow), "slow stride {slow}");
    assert!((3.0..=7.0).contains(&fast), "fast stride {fast}");
    let in_band = walk
        .steps
        .iter()
        .filter(|s| s.kind == StepKind::InBand)
        .count();
    assert!(in_band * 10 >= walk.steps.len() * 8);
    assert_read_is_asked(&run);
}

/// The report's reading of `frame`, which was read.
fn reading(run: &Run, frame: usize) -> FrameReading {
    run.report()
        .frames
        .iter()
        .find(|f| f.id.frame == frame)
        .expect("the frame was read")
        .clone()
}

fn was_read(run: &Run, id: &FrameId) -> bool {
    run.report().frames.iter().any(|f| f.id == *id)
}

/// The point density of the captures the tests of blur against the
/// neighbours walk: about 250 features a frame, so a step in the band keeps
/// some 200 matches, several times the [`FLOOR`] a step beyond the next
/// frame keeps, at a quarter of [`Synth::steady`]'s cost per frame.
const NEIGHBOUR_DENSITY: f64 = 20.0;

/// A steady capture of 200 frames at [`NEIGHBOUR_DENSITY`] and its clean
/// run, which the tests that change one frame of it read to choose the
/// frame.
fn clean_steady() -> &'static (Synth, Run) {
    static CLEAN: OnceLock<(Synth, Run)> = OnceLock::new();
    CLEAN.get_or_init(|| {
        let synth = Synth::steady_with_density(34, 200, ONE_PERCENT, NEIGHBOUR_DENSITY);
        let run = run(&[&synth]);
        (synth, run)
    })
}

/// The first step after `after` to a frame in the band and, when `stride`
/// is given, that many frames long: its `from` and `to`.
fn step_in_band(run: &Run, after: usize, stride: Option<usize>) -> (usize, usize) {
    run.walk(0)
        .steps
        .iter()
        .find(|s| {
            s.to > after && s.kind == StepKind::InBand && stride.is_none_or(|k| s.to - s.from == k)
        })
        .map(|s| (s.from, s.to))
        .expect("a step in the band")
}

/// The statistics of the pair from `a` to `b` when the run matched it.
fn measured(run: &Run, a: usize, b: usize) -> Option<&FramePairStats> {
    run.report()
        .pairs
        .iter()
        .find(|p| p.a.frame == a && p.b.frame == b)
        .map(|p| &p.stats)
}

/// Whether the step from `from` to `n` may replace a candidate whose own
/// step was `own`: it holds, keeping the floor beyond the next frame, and is
/// no longer than the band and no shorter than the band or `own`.
fn swap_fits(stats: &FramePairStats, from: usize, n: usize, own: f64) -> bool {
    let (low, high) = band();
    (n == from + 1 || stats.matches >= options().min_step_matches)
        && stats.step_percent >= own.min(low)
        && stats.step_percent <= high
}

#[test]
fn a_frame_blurred_against_its_neighbours_is_swapped_for_the_sharpest_that_fits() {
    // A frame the walk chose, made less sharp than its neighbours by more
    // than 0.85. The two frames 2 positions away are the sharpest, the later
    // one sharper: one of their steps is always in the band, and neither is
    // the frame a nearest-first order would try first.
    let (clean, first) = clean_steady();
    let mut synth = clean.clone();
    let (_, soft) = step_in_band(first, 40, None);
    synth.shots[soft].readings[0].0 = 0.7;
    synth.shots[soft + 2].readings[0].0 = 1.3;
    synth.shots[soft - 2].readings[0].0 = 1.2;
    let run = run(&[&synth]);
    let walk = run.walk(0);
    let swap = walk
        .swaps
        .iter()
        .find(|s| s.candidate == soft)
        .unwrap_or_else(|| panic!("{soft} was not swapped: {:?}", walk.swaps));
    assert!(!walk.chosen.contains(&soft));
    assert!(walk.chosen.contains(&swap.chosen));
    // Every frame within two positions was read.
    let id = |frame| FrameId { sequence: 0, frame };
    let window: Vec<usize> = (soft - 2..=soft + 2).filter(|&n| n != soft).collect();
    assert!(window.iter().all(|&n| was_read(&run, &id(n))));
    // The chosen frame is the sharpest neighbour whose step fits; every
    // sharper one was measured and does not fit.
    let from = walk
        .steps
        .iter()
        .find(|s| s.to == swap.chosen)
        .unwrap()
        .from;
    let own = measured(&run, from, soft).unwrap().step_percent;
    let sharpness = |t: usize| synth.shots[t].readings[0].0;
    let mut ranked: Vec<usize> = window.iter().copied().filter(|&n| n > from).collect();
    ranked.sort_by(|&x, &y| {
        sharpness(y)
            .total_cmp(&sharpness(x))
            .then(x.abs_diff(soft).cmp(&y.abs_diff(soft)))
            .then(x.cmp(&y))
    });
    let at = ranked.iter().position(|&n| n == swap.chosen).unwrap();
    assert!(swap_fits(
        measured(&run, from, swap.chosen).unwrap(),
        from,
        swap.chosen,
        own
    ));
    for &n in &ranked[..at] {
        let stats = measured(&run, from, n).unwrap();
        assert!(!swap_fits(stats, from, n, own), "{n}: {stats:?}");
    }
    assert_read_is_asked(&run);
}

#[test]
fn a_frame_sharper_than_most_is_still_compared_with_its_neighbours() {
    // A frame the walk chose, sharper than every other frame of the capture
    // but its neighbours, which are sharper still by more than the ratio: it
    // is blurred against them and swapped, however sharp it is.
    let (clean, first) = clean_steady();
    let mut synth = clean.clone();
    let (_, frame) = step_in_band(first, 40, None);
    synth.shots[frame].readings[0].0 = 1.3;
    for n in (frame - 2..=frame + 2).filter(|&n| n != frame) {
        synth.shots[n].readings[0].0 = 2.0;
    }
    let run = run(&[&synth]);
    let walk = run.walk(0);
    assert!(!walk.chosen.contains(&frame), "{frame} was chosen");
    let swap = walk
        .swaps
        .iter()
        .find(|s| s.candidate == frame)
        .unwrap_or_else(|| panic!("{frame} was not swapped: {:?}", walk.swaps));
    assert!(swap.chosen.abs_diff(frame) <= 2);
    assert!(walk.chosen.contains(&swap.chosen));
    assert_read_is_asked(&run);
}

#[test]
fn a_uniformly_blurred_stretch_is_not_swapped_against_its_neighbours() {
    // A stretch whose frames are all equally soft, at 0.7 of the rest: each
    // frame chosen in it is compared with its neighbours, which share the
    // blur, so none is swapped for one, and the stretch is crossed at the
    // band's spacing.
    let mut synth = Synth::steady_with_density(36, 150, ONE_PERCENT, NEIGHBOUR_DENSITY);
    for t in 50..=110 {
        synth.shots[t].readings[0].0 = 0.7 + 0.005 * ((t * 7) % 5) as f32;
    }
    let run = run(&[&synth]);
    let walk = run.walk(0);
    // The frames whose neighbours within two positions are all in the
    // stretch.
    let inner = 52..=108;
    let inside: Vec<usize> = walk
        .chosen
        .iter()
        .copied()
        .filter(|c| inner.contains(c))
        .collect();
    assert!(inside.len() >= 4, "{inside:?}");
    let id = |frame| FrameId { sequence: 0, frame };
    for &c in &inside {
        assert!((c - 2..=c + 2).all(|n| was_read(&run, &id(n))), "{c}");
    }
    assert!(
        walk.swaps.iter().all(|s| !inner.contains(&s.candidate)),
        "{:?}",
        walk.swaps
    );
    let across = median(&strides(walk, |s| inner.contains(&s.from)));
    assert!(across >= 8.0, "median stride {across} across the stretch");
    assert_read_is_asked(&run);
}

#[test]
fn a_neighbour_whose_step_does_not_fit_is_not_swapped_in() {
    // At about 2.5% a frame, a step of 3 to 5 frames is in the band and one
    // of 2 or 6 is not. The candidate is blurred against its neighbours. The two
    // frames 2 away are the sharpest and out of the band; the frame after it
    // is the next sharpest and shows one point in 5, so its step is short of
    // the floor; and the frame before it qualifies.
    let mut synth = Synth::steady_with_density(37, 100, 2.7 * ONE_PERCENT, NEIGHBOUR_DENSITY);
    let (from, soft) = step_in_band(&run(&[&synth]), 30, Some(4));
    synth.shots[soft].readings[0].0 = 0.7;
    synth.shots[soft - 2].readings[0].0 = 1.3;
    synth.shots[soft + 2].readings[0].0 = 1.3;
    synth.shots[soft + 1].readings[0].0 = 1.25;
    synth.shots[soft + 1].content = Content::Subset { every: 5 };
    synth.shots[soft - 1].readings[0].0 = 1.1;
    let swapped = run(&[&synth]);
    let walk = swapped.walk(0);
    assert!(walk.chosen.contains(&from));
    assert!(
        walk.swaps.contains(&Swap {
            candidate: soft,
            chosen: soft - 1,
        }),
        "{:?}",
        walk.swaps
    );
    let (low, high) = band();
    let step = |n: usize| measured(&swapped, from, n).unwrap().clone();
    assert!(step(soft - 2).step_percent < low, "{:?}", step(soft - 2));
    assert!(step(soft + 2).step_percent > high, "{:?}", step(soft + 2));
    let few = step(soft + 1);
    assert!((low..=high).contains(&few.step_percent), "{few:?}");
    assert!(few.matches < FLOOR, "{few:?}");
    assert_read_is_asked(&swapped);

    // With the frame before it as soft as the candidate, no neighbour
    // qualifies, and the candidate is kept.
    synth.shots[soft - 1].readings[0].0 = 0.7;
    let kept = run(&[&synth]);
    let walk = kept.walk(0);
    assert!(walk.chosen.contains(&soft), "{soft} was not kept");
    assert!(walk.swaps.iter().all(|s| s.candidate != soft));
    assert_read_is_asked(&kept);
}

#[test]
fn gradients_along_one_direction_do_not_blur_a_sharp_frame() {
    // A frame the walk chose, given a strongly directional tensor at its
    // usual sharpness, as a sharp frame of a slope or of a line-like subject
    // reads. It is still chosen and not swapped: blur is judged from the
    // sharpness alone.
    let (clean, first) = clean_steady();
    let mut streaked = clean.clone();
    let (_, frame) = step_in_band(first, 40, None);
    streaked.shots[frame].readings[0].1 = [1.0, 0.2, 0.0];
    let run = run(&[&streaked]);
    let walk = run.walk(0);
    assert!(walk.chosen.contains(&frame), "{frame} was not chosen");
    assert!(walk.swaps.iter().all(|s| s.candidate != frame));
    let read = reading(&run, frame);
    assert!(read.sensors[0].axis_ratio.unwrap() > 1.5);
    assert!(read.sensors[0].coherence > 0.6);
    assert_read_is_asked(&run);
}

/// Every accepted step beyond the next frame keeps `floor`; and the walk
/// measured at least one pair beyond the next frame from a chosen frame that
/// was short of it, so the floor was tested.
fn assert_steps_keep_the_floor(run: &Run, floor: usize) {
    let walk = run.walk(0);
    for step in &walk.steps {
        assert!(
            step.to == step.from + 1 || step.stats.matches >= floor,
            "{step:?}"
        );
    }
    let chosen: BTreeSet<usize> = walk.chosen.iter().copied().collect();
    assert!(
        run.report()
            .pairs
            .iter()
            .any(|p| chosen.contains(&p.a.frame)
                && p.b.frame > p.a.frame + 1
                && p.stats.matches < floor),
        "no pair short of the floor was measured"
    );
}

/// A capture of `frames` frames at `speed` along x and [`NEIGHBOUR_DENSITY`]
/// whose frames show the points for which `(frame + point) % 40 < 20`:
/// frames `k` apart share `(20 − k) / 40` of the points in view, so the
/// matches fall steadily with the distance, about 6 for each frame further,
/// from about 115 at the next frame.
fn fading(seed: u64, frames: usize, speed: f64) -> Synth {
    let mut synth = Synth::steady_with_density(seed, frames, speed, NEIGHBOUR_DENSITY);
    for shot in &mut synth.shots {
        shot.content = Content::Window {
            period: 40,
            lifetime: 20,
        };
    }
    synth
}

/// [`options`] with the match floor at `floor`.
fn with_floor(floor: usize) -> PickImagesOptions {
    PickImagesOptions {
        min_step_matches: floor,
        ..options()
    }
}

#[test]
fn a_capture_whose_neighbours_match_below_the_floor_keeps_every_frame() {
    // Frames 3% apart, a third of the target, whose next frames keep fewer
    // matches than a floor of 140: every step stops at the next frame, and
    // each is the only pair its step measures, since a step to the next
    // frame short of the floor leaves no rate to aim further with.
    let synth = fading(38, 30, 3.0 * ONE_PERCENT);
    let floor = 140;
    let run = run_with(&[&synth], &with_floor(floor), Fault::None, None);
    let walk = run.walk(0);
    assert_eq!(walk.chosen, (0..30).collect::<Vec<usize>>());
    assert!(
        walk.steps
            .iter()
            .all(|s| s.kind == StepKind::AtFloor && s.stats.matches < floor),
        "{:?}",
        walk.steps
    );
    let pairs: Vec<(usize, usize)> = run
        .report()
        .pairs
        .iter()
        .map(|p| (p.a.frame, p.b.frame))
        .collect();
    assert_eq!(pairs, (0..29).map(|t| (t, t + 1)).collect::<Vec<_>>());
    assert_read_is_asked(&run);
}

#[test]
fn the_floor_stops_a_step_short_of_the_target() {
    // At 1% a frame the target is about 10 frames away, where the matches
    // fall to some 60. With the floor at 60 the walk steps at the target;
    // with it at 85 every step stops some 5 frames out, at the farthest
    // frame that keeps the floor.
    let synth = fading(39, 100, ONE_PERCENT);
    let at_target = run_with(&[&synth], &with_floor(60), Fault::None, None);
    let stride = median(&strides(at_target.walk(0), |_| true));
    assert!(stride >= 8.0, "median stride {stride}");
    assert_read_is_asked(&at_target);

    let floor = 85;
    let run = run_with(&[&synth], &with_floor(floor), Fault::None, None);
    let walk = run.walk(0);
    let stride = median(&strides(walk, |_| true));
    assert!((3.0..=7.0).contains(&stride), "median stride {stride}");
    assert_steps_keep_the_floor(&run, floor);
    let at_floor: Vec<&WalkStep> = walk
        .steps
        .iter()
        .filter(|s| s.kind == StepKind::AtFloor)
        .collect();
    assert!(
        at_floor.len() * 2 >= walk.steps.len(),
        "{} of {} at the floor",
        at_floor.len(),
        walk.steps.len()
    );
    // A step at the floor is the farthest that keeps it: the frame after it
    // was measured short of the floor.
    for step in at_floor {
        let next = measured(&run, step.from, step.to + 1).expect("the frame after was measured");
        assert!(next.matches < floor, "{step:?} {next:?}");
    }
    assert_read_is_asked(&run);
}

#[test]
fn a_stretch_that_matches_nothing_is_walked_one_frame_at_a_time() {
    // Frames that match nothing right after the first frame and in the
    // middle: the step to the next frame is always taken, so every one of
    // them is chosen, from each only its next frame is measured, and the
    // walk takes its full stride again after each stretch.
    let mut synth = Synth::steady(6, 200, ONE_PERCENT);
    let (start, middle) = (1..=8, 100..=115);
    for t in start.clone().chain(middle.clone()) {
        synth.shots[t].content = Content::Junk;
    }
    let run = run(&[&synth]);
    let walk = run.walk(0);
    for t in (0..=9).chain(middle.clone()) {
        assert!(walk.chosen.contains(&t), "{t} was not chosen");
    }
    let junk = |t: usize| start.contains(&t) || middle.contains(&t);
    for step in walk.steps.iter().filter(|s| junk(s.from)) {
        assert_eq!(step.to, step.from + 1, "{step:?}");
        assert_eq!(step.kind, StepKind::AtFloor, "{step:?}");
        assert!(step.stats.matches < FLOOR, "{step:?}");
    }
    for pair in run.report().pairs.iter().filter(|p| junk(p.a.frame)) {
        assert_eq!(pair.b.frame, pair.a.frame + 1, "{pair:?}");
    }
    assert!(walk.swaps.iter().all(|s| !junk(s.candidate)));
    let between = median(&strides(walk, |s| s.from > 20 && s.to < 95));
    let after = median(&strides(walk, |s| s.from > 125));
    assert!(between >= 8.0 && after >= 8.0, "{between} {after}");
    assert_read_is_asked(&run);
}

#[test]
fn a_rig_takes_its_motion_from_the_reference_sensor_and_blur_from_either() {
    let mut steady = Synth::steady_with_density(12, 200, ONE_PERCENT, NEIGHBOUR_DENSITY);
    steady.sensors = 2;
    let first = run(&[&steady]);
    // The second sensor sees only random features of another size, so the
    // walk matched the reference sensor's images alone.
    let walk = first.walk(0);
    assert!(walk.swaps.is_empty());
    let stride = median(&strides(walk, |_| true));
    assert!((8.0..=12.0).contains(&stride));
    assert_eq!(first.report().frames[0].sensors.len(), 2);
    assert_read_is_asked(&first);

    // The second sensor's image of a chosen frame much less sharp than its
    // neighbours' makes the frame blurred against them, and it is swapped.
    // In the same run, an earlier chosen frame is blurred against its
    // neighbours on the second sensor, but the one neighbour sharper there by
    // the ratio is less sharp than it by the ratio on the reference sensor:
    // no neighbour qualifies, and it is kept, so the walk after it is the
    // clean run's.
    let (_, frame) = step_in_band(&first, 60, None);
    let (_, guarded) = step_in_band(&first, 20, None);
    assert!(guarded + 5 < frame, "{guarded} {frame}");
    steady.shots[guarded].readings[0].0 = 1.1;
    steady.shots[guarded].readings[1].0 = 3.6;
    for n in [guarded - 2, guarded - 1, guarded + 2] {
        steady.shots[n].readings[1].0 = 4.0;
    }
    steady.shots[guarded + 1].readings[0].0 = 0.85;
    steady.shots[guarded + 1].readings[1].0 = 6.5;
    steady.shots[frame].readings[1] = (1.0, sharp(frame, 1).1);
    let blurred = run(&[&steady]);
    let walk = blurred.walk(0);
    assert!(walk.chosen.contains(&guarded), "{guarded} was not kept");
    assert!(walk.swaps.iter().all(|s| s.candidate != guarded));
    let id = |frame| FrameId { sequence: 0, frame };
    assert!(was_read(&blurred, &id(guarded + 1)));
    assert!(!walk.chosen.contains(&frame));
    assert!(walk.swaps.iter().any(|s| s.candidate == frame));

    // The second sensor's image of another chosen frame below its
    // neighbours', the reference sensor's not: the frame is blurred against
    // its neighbours on the second sensor alone, judged against that
    // sensor's neighbours, and swapped. Of the two nearest neighbours, the
    // one after it gains more on its least improved sensor (1.12 against
    // 1.08) and the one before it more on its most improved sensor (1.5
    // against 1.39), so the frame after it is taken.
    steady.shots[frame].readings[1] = sharp(frame, 1);
    for n in guarded - 2..=guarded + 2 {
        steady.shots[n].readings = vec![sharp(n, 0), sharp(n, 1)];
    }
    let (_, other) = step_in_band(&first, 130, None);
    steady.shots[other].readings[0].0 = 1.0;
    steady.shots[other].readings[1].0 = 3.6;
    steady.shots[other + 1].readings[0].0 = 1.12;
    steady.shots[other + 1].readings[1].0 = 5.0;
    steady.shots[other - 1].readings[0].0 = 1.08;
    steady.shots[other - 1].readings[1].0 = 5.4;
    let local = run(&[&steady]);
    let walk = local.walk(0);
    assert!(!walk.chosen.contains(&other));
    assert!(
        walk.swaps.contains(&Swap {
            candidate: other,
            chosen: other + 1,
        }),
        "{:?}",
        walk.swaps
    );
    assert_read_is_asked(&local);
}

/// The report without the one count that depends on timing, as its JSON
/// value, in which a NaN (a step with no matches) equals itself.
fn decisions(run: &Run) -> serde_json::Value {
    let report = PickImagesReport {
        frames_extracted_unread: 0,
        ..run.report().clone()
    };
    serde_json::to_value(report).expect("the report serializes")
}

#[test]
fn the_same_capture_gives_the_same_report() {
    let synth = Synth::steady(13, 200, ONE_PERCENT);
    let a = run(&[&synth]);
    let b = run(&[&synth]);
    assert_eq!(decisions(&a), decisions(&b));
    assert_eq!(a.waits, b.waits);
}

/// A capture with something of everything the walk does: a change of speed,
/// frames blurred against their neighbours, a stretch whose matches fall
/// fast with the distance, where steps stop at the floor, and a stretch that
/// matches nothing. [`assert_eventful`] checks that a walk of it shows each.
fn eventful() -> Synth {
    let mut x = 0.0;
    let xs: Vec<f64> = (0..200)
        .map(|t| {
            let here = x;
            x += if t < 130 { 1.0 } else { 1.6 } * ONE_PERCENT;
            here
        })
        .collect();
    let length = *xs.last().unwrap();
    let mut synth = Synth::wall(32, -12.0, length + 12.0, NEIGHBOUR_DENSITY).along(xs);
    // Two of every three frames soft, so a candidate that lands on one is
    // swapped for the sharp frame beside it.
    for t in (20..=50).filter(|t| t % 3 != 1) {
        synth.shots[t].readings[0].0 = 0.7;
    }
    for t in 70..=110 {
        synth.shots[t].content = Content::Window {
            period: 40,
            lifetime: 12,
        };
    }
    for t in 160..=170 {
        synth.shots[t].content = Content::Junk;
    }
    synth
}

/// What a walk of [`eventful`] is meant to show, each named with whether it
/// does: the stride follows the change of speed, a frame blurred against
/// its neighbours is swapped, a step beyond the next frame stops at the
/// floor in the stretch whose matches fall fast, and the junk frames are
/// walked one frame at a time.
fn eventful_events(run: &Run) -> Vec<(&'static str, bool)> {
    let walk = run.walk(0);
    let rates = |keep: &dyn Fn(&WalkStep) -> bool| {
        let r: Vec<f64> = walk
            .steps
            .iter()
            .filter(|s| keep(s))
            .map(|s| s.rate)
            .collect();
        median(&r)
    };
    vec![
        (
            "the stride follows the change of speed",
            rates(&|s| s.from >= 135 && s.to < 160) > 1.3 * rates(&|s| s.from >= 52 && s.to <= 68),
        ),
        (
            "a frame blurred against its neighbours is swapped",
            walk.swaps.iter().any(|s| (20..=50).contains(&s.candidate)),
        ),
        (
            "a step beyond the next frame stops at the floor",
            walk.steps.iter().any(|s| {
                s.kind == StepKind::AtFloor && s.to > s.from + 1 && (70..=110).contains(&s.from)
            }),
        ),
        (
            "the junk frames are walked one frame at a time",
            (160..=170).all(|t| walk.chosen.contains(&t))
                && walk.steps.iter().any(|s| {
                    s.kind == StepKind::AtFloor
                        && s.to == s.from + 1
                        && (160..=169).contains(&s.from)
                }),
        ),
    ]
}

/// Every event of [`eventful_events`] happened.
fn assert_eventful(run: &Run) {
    let missing: Vec<&str> = eventful_events(run)
        .into_iter()
        .filter(|(_, happened)| !happened)
        .map(|(name, _)| name)
        .collect();
    assert!(missing.is_empty(), "missing: {missing:?}");
}

#[test]
fn the_report_is_the_same_for_any_lookahead_and_thread_count() {
    let synth = eventful();
    let off = PickImagesOptions {
        prefetch: false,
        ..options()
    };
    let reference = run_on(&[&synth], &off, Fault::None, None, 1);
    // The capture exercises what it is meant to.
    assert_eventful(&reference);
    assert!(reference.prefetched.is_empty());
    assert_eq!(reference.report().frames_extracted_unread, 0);
    assert_read_is_asked(&reference);
    for (prefetch, threads) in [(false, 8), (true, 1), (true, 8)] {
        let options = PickImagesOptions {
            prefetch,
            ..options()
        };
        let run = run_on(&[&synth], &options, Fault::None, None, threads);
        let case = format!("prefetch {prefetch}, {threads} threads");
        assert_eq!(decisions(&run), decisions(&reference), "{case}");
        assert_eq!(run.waits, reference.waits, "{case}");
        assert_eq!(run.prefetched.is_empty(), !prefetch, "{case}");
        assert_read_is_asked(&run);
    }
}

/// The frames each step measured from its `from`, in the order measured.
fn tries(run: &Run, from: usize) -> Vec<usize> {
    run.report()
        .pairs
        .iter()
        .filter(|p| p.a.frame == from)
        .map(|p| p.b.frame)
        .collect()
}

#[test]
fn the_walk_reads_ahead_and_a_frame_failing_unread_stops_nothing() {
    // A steady video: every step after the first starts at the candidate the
    // walk predicted while it judged the last candidate's neighbours.
    let synth = Synth::steady(33, 200, ONE_PERCENT);
    let steady = run(&[&synth]);
    let walk = steady.walk(0);
    let id = |frame| FrameId { sequence: 0, frame };
    for step in &walk.steps[1..] {
        let first = tries(&steady, step.from)[0];
        assert!(steady.prefetched.contains(&id(first)), "{step:?}");
    }
    assert_read_is_asked(&steady);

    // Steps stopped by the floor: a try of the floor search after the first
    // was predicted.
    let fades = fading(39, 100, ONE_PERCENT);
    let run = run_with(&[&fades], &with_floor(85), Fault::None, None);
    let walk = run.walk(0);
    assert!(
        walk.steps.iter().any(|s| tries(&run, s.from)
            .iter()
            .skip(2)
            .any(|&t| run.prefetched.contains(&id(t)))),
        "no floor-search try was predicted"
    );
    // At most the next candidate and its four neighbours per step, and one
    // try per step of the floor search, are predicted and not read.
    let searched: usize = walk.steps.iter().map(|s| tries(&run, s.from).len()).sum();
    assert!(
        run.report().frames_extracted_unread <= 5 * walk.steps.len() + searched,
        "{} extracted and never read",
        run.report().frames_extracted_unread
    );
    assert_read_is_asked(&run);

    // A capture whose neighbouring frames match below the floor, which the
    // walk reads frame by frame: every frame after the first step's was
    // predicted.
    let below = fading(38, 30, 3.0 * ONE_PERCENT);
    let every = run_with(&[&below], &with_floor(140), Fault::None, None);
    for t in 2..30 {
        assert!(every.prefetched.contains(&id(t)), "{t} was not predicted");
    }
    assert_read_is_asked(&every);

    // A worker failing on a frame predicted and never read stops nothing.
    let never_read = *run
        .prefetched
        .iter()
        .find(|id| !was_read(&run, id))
        .expect("a frame predicted and never read");
    let failing = run_with(
        &[&fades],
        &with_floor(85),
        Fault::FailFrame(never_read.frame),
        None,
    );
    assert_eq!(decisions(&failing), decisions(&run));
}

#[test]
fn several_sequences_are_walked_in_turn() {
    let one = Synth::steady(14, 150, ONE_PERCENT);
    let two = Synth::steady(15, 80, ONE_PERCENT);
    let lone = Synth::steady(15, 1, ONE_PERCENT);
    let run = run(&[&one, &two, &lone]);
    let report = run.report();
    assert_eq!(run.walk(0).chosen.last(), Some(&149));
    assert_eq!(run.walk(1).chosen.last(), Some(&79));
    // A lone frame is chosen: there is no step to take from it.
    assert_eq!(run.walk(2).chosen, vec![0]);
    assert!(run.walk(2).steps.is_empty());
    // The sequences are read one after another.
    let order: Vec<usize> = run.asked.iter().map(|id| id.sequence).collect();
    assert!(order.windows(2).all(|w| w[0] <= w[1]), "{order:?}");
    assert!(report.pairs.iter().all(|p| p.a.sequence == p.b.sequence));
    assert_read_is_asked(&run);
}

// Errors.

fn error(run: &Run) -> &PickImagesError {
    match &run.result {
        Ok(_) => panic!("pick_images succeeded"),
        Err(e) => e,
    }
}

#[test]
fn every_option_out_of_range_is_refused() {
    let synth = Synth::steady(16, 10, ONE_PERCENT);
    let cases: Vec<(&str, PickImagesOptions)> = vec![
        (
            "target_step_percent",
            PickImagesOptions {
                target_step_percent: 0.0,
                ..options()
            },
        ),
        (
            "target_step_percent",
            PickImagesOptions {
                target_step_percent: f64::NAN,
                ..options()
            },
        ),
        (
            "max_density_gain",
            PickImagesOptions {
                max_density_gain: 0.99,
                ..options()
            },
        ),
        (
            "max_density_gain",
            PickImagesOptions {
                max_density_gain: f64::NAN,
                ..options()
            },
        ),
        (
            "max_density_gain",
            PickImagesOptions {
                max_density_gain: f64::INFINITY,
                ..options()
            },
        ),
        (
            "step_band",
            PickImagesOptions {
                step_band: 0.0,
                ..options()
            },
        ),
        (
            "step_band",
            PickImagesOptions {
                step_band: 1.0,
                ..options()
            },
        ),
        (
            "neighbour_blur_ratio",
            PickImagesOptions {
                neighbour_blur_ratio: 0.0,
                ..options()
            },
        ),
        (
            "forest.num_trees",
            PickImagesOptions {
                forest: crate::features::kdforest::KdForestParams {
                    num_trees: 0,
                    ..crate::features::kdforest::KdForestParams::balanced()
                },
                ..options()
            },
        ),
        (
            "pair.ratio",
            PickImagesOptions {
                pair: FramePairOptions {
                    ratio: 0.0,
                    ..FramePairOptions::default()
                },
                ..options()
            },
        ),
        (
            "min_step_matches",
            PickImagesOptions {
                min_step_matches: 0,
                ..options()
            },
        ),
    ];
    for (expected, options) in cases {
        let run = run_with(&[&synth], &options, Fault::None, None);
        match error(&run) {
            PickImagesError::Options { name, .. } => assert_eq!(name, expected),
            other => panic!("{expected}: {other:?}"),
        }
        assert!(run.waits.is_empty() && run.prefetched.is_empty());
    }
    // The bounds themselves are accepted.
    let edges = PickImagesOptions {
        min_step_matches: 1,
        max_density_gain: 1.0,
        ..options()
    };
    assert!(run_with(&[&synth], &edges, Fault::None, None)
        .result
        .is_ok());
}

fn sequence_error(sequences: &[CaptureSequence]) -> String {
    let synth = Synth::steady(17, 10, ONE_PERCENT);
    let synths: Vec<&Synth> = sequences.iter().map(|_| &synth).collect();
    let mut source = SynthSource::new(&synths, THREADS, Fault::None);
    let result = pick_images(sequences, &mut source, &options(), &Progress::none());
    let asked = source.into_asked();
    match result {
        Err(PickImagesError::Sequence { reason }) => {
            assert!(asked.waits.is_empty() && asked.prefetched.is_empty());
            reason
        }
        other => panic!("{other:?}"),
    }
}

#[test]
fn a_malformed_sequence_is_refused() {
    let dir = tempfile::tempdir().unwrap();
    let mut rig = Synth::steady(18, 10, ONE_PERCENT);
    rig.sensors = 2;
    let good = rig.sequence(dir.path());

    assert!(sequence_error(&[]).contains("no sequences"));
    let empty = CaptureSequence {
        frames: Vec::new(),
        ..good.clone()
    };
    assert!(sequence_error(&[empty]).contains("has no frames"));
    let mut short = good.clone();
    short.frames[4].images.pop();
    assert!(sequence_error(&[short]).contains("has 1 images"));
    let mut none = good.clone();
    for frame in &mut none.frames {
        frame.images.clear();
    }
    assert!(sequence_error(&[none]).contains("has no images"));
    let mut patterns = good.clone();
    patterns.patterns.pop();
    assert!(sequence_error(&[patterns]).contains("1 patterns"));
    let mut order = good.clone();
    order.frames[3].number = order.frames[2].number;
    assert!(sequence_error(&[order]).contains("increasing"));
    let mut shared = good.clone();
    shared.frames[5].images[1].sift_path = shared.frames[5].images[0].sift_path.clone();
    assert!(sequence_error(&[shared]).contains("two images"));
    assert!(sequence_error(&[good.clone(), good]).contains("two images"));
}

#[test]
fn version_1_files_in_a_folder_are_refused_before_anything_is_extracted() {
    let synth = Synth::steady(19, 40, ONE_PERCENT);
    let dir = tempfile::tempdir().unwrap();
    let sequences = vec![synth.sequence(&dir.path().join("seq0"))];
    std::fs::create_dir_all(dir.path().join("seq0")).unwrap();
    // A version 1 file of an image of the capture, already in its folder.
    let path = &sequences[0].frames[7].images[0].sift_path;
    synth.write(7, 0, path, 1);
    let mut source = SynthSource::new(&[&synth], THREADS, Fault::None);
    let err = pick_images(&sequences, &mut source, &options(), &Progress::none()).unwrap_err();
    match &err {
        PickImagesError::Version1Files(found) => {
            assert_eq!(found.files, vec![path.clone()]);
            assert!(err.to_string().contains("remove those files"));
        }
        other => panic!("{other:?}"),
    }
    let asked = source.into_asked();
    assert!(asked.waits.is_empty() && asked.prefetched.is_empty());
}

#[test]
fn a_version_1_file_read_is_refused() {
    let synth = Synth::steady(20, 40, ONE_PERCENT);
    let run = run_with(&[&synth], &options(), Fault::WriteVersion1, None);
    match error(&run) {
        PickImagesError::Version1Sift { path } => {
            assert_eq!(path, &run.sequences[0].frames[0].images[0].sift_path);
        }
        other => panic!("{other:?}"),
    }
}

#[test]
fn a_folder_that_cannot_be_listed_is_refused() {
    let synth = Synth::steady(21, 10, ONE_PERCENT);
    let dir = tempfile::tempdir().unwrap();
    let blocker = dir.path().join("not_a_folder");
    std::fs::write(&blocker, b"a file").unwrap();
    let sequences = vec![synth.sequence(&blocker)];
    let mut source = SynthSource::new(&[&synth], THREADS, Fault::None);
    match pick_images(&sequences, &mut source, &options(), &Progress::none()) {
        Err(PickImagesError::ListFolder { folder, .. }) => assert_eq!(folder, blocker),
        other => panic!("{other:?}"),
    }
    let asked = source.into_asked();
    assert!(asked.waits.is_empty() && asked.prefetched.is_empty());
}

#[test]
fn a_worker_s_failure_on_a_frame_read_is_the_error_naming_the_frame() {
    let synth = Synth::steady(22, 150, ONE_PERCENT);
    let clean = run(&[&synth]);
    // The first frame, and a frame the walk reads later.
    let later = clean.asked[clean.asked.len() / 2];
    for frame in [clean.asked[0], later] {
        let run = run_with(&[&synth], &options(), Fault::FailFrame(frame.frame), None);
        match error(&run) {
            PickImagesError::Source(FrameSourceError::Extract { frame: f, error }) => {
                assert_eq!(*f, frame);
                let message = error.to_string();
                assert!(message.contains("the disk is full"), "{message}");
                let name = format!("cam0_{:04}.jpg.sift", frame.frame + 1);
                assert!(message.contains(&name), "{message}");
            }
            other => panic!("{other:?}"),
        }
        assert!(run.waits.last().unwrap().contains(&frame));
    }
}

#[test]
fn a_missing_sift_file_is_a_read_error() {
    let synth = Synth::steady(23, 40, ONE_PERCENT);
    let run = run_with(&[&synth], &options(), Fault::WriteNothing, None);
    match error(&run) {
        PickImagesError::ReadSift { path, .. } => {
            assert_eq!(path, &run.sequences[0].frames[0].images[0].sift_path);
        }
        other => panic!("{other:?}"),
    }
}

#[test]
fn frames_of_different_sizes_are_a_pair_error() {
    let synth = Synth::steady(24, 40, ONE_PERCENT);
    // Frame 1, matched with frame 0 by the first step, at another size.
    let run = run_with(&[&synth], &options(), Fault::Resize(1), None);
    match error(&run) {
        PickImagesError::Pair { a, b, error, .. } => {
            assert_eq!((a.frame, b.frame), (0, 1));
            assert!(matches!(error, FramePairError::SizeMismatch { .. }));
        }
        other => panic!("{other:?}"),
    }
}

#[test]
fn a_cancelled_run_stops() {
    let synth = Synth::steady(25, 300, ONE_PERCENT);
    let flag = AtomicBool::new(true);
    let run = run_with(&[&synth], &options(), Fault::None, Some(&flag));
    assert!(matches!(error(&run), PickImagesError::Cancelled));
    assert!(run.waits.is_empty() && run.prefetched.is_empty());

    for call in [0, 1, 3, 40] {
        let flag = AtomicBool::new(false);
        let run = run_with(
            &[&synth],
            &options(),
            Fault::CancelOnWait(call),
            Some(&flag),
        );
        assert!(matches!(error(&run), PickImagesError::Cancelled));
        assert_eq!(run.waits.len(), call + 1);
    }
}
