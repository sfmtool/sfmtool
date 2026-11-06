// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The walk on synthetic captures: a scene of points with random descriptors
//! seen along a camera path with known motion per frame, written as version 2
//! `.sift` files by a pool of workers only for the frames the source is asked
//! to prefetch or wait for.

mod capture;

use std::collections::BTreeSet;
use std::path::PathBuf;
use std::sync::atomic::AtomicBool;
use std::sync::OnceLock;

use capture::{sharp, sift_files, Asked, Content, Fault, Synth, SynthSource, ONE_PERCENT};

use super::*;
use crate::features::sift::ImageReadings;
use crate::numeric::median;
use crate::progress::Progress;

fn options() -> PickImagesOptions {
    PickImagesOptions {
        baseline_frames: 30,
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

    fn walk(&self, sequence: usize) -> &WalkReport {
        self.report().sequences[sequence]
            .walk
            .as_ref()
            .expect("the sequence was walked")
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

fn strides(walk: &WalkReport, keep: impl Fn(&WalkStep) -> bool) -> Vec<f64> {
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
    }
    assert_eq!(report.frames_extracted_unread, unread);
}

#[test]
fn a_steady_video_is_chosen_at_about_the_target_step() {
    let synth = Synth::steady(1, 300, ONE_PERCENT);
    let run = run(&[&synth]);
    let report = run.report();
    assert_eq!(report.verdict, Verdict::Solvable);
    let sequence = &report.sequences[0];
    assert!(sequence.video.is_video);
    assert!(sequence.solvability.as_ref().unwrap().solvable);
    assert!((sequence.baseline.rate_median - 1.0).abs() < 0.15);
    let walk = run.walk(0);
    assert_eq!(walk.chosen.first(), Some(&0));
    assert_eq!(walk.chosen.last(), Some(&299));
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
    assert!(walk.breaks.is_empty() && walk.gaps.is_empty() && walk.swaps.is_empty());
    assert!(walk.kept_blurred.is_empty() && walk.weak_points.is_empty());
    // The walk reads few frames beyond those it chooses.
    assert!(
        walk.frames_used <= walk.chosen.len() * 3,
        "{} used",
        walk.frames_used
    );
    assert!(report.frames.len() < 300);
    assert_read_is_asked(&run);
    // A frame never read lies between two chosen frames of one segment.
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

/// The first chosen frame after the middle of the steady walk that is not a
/// sampled frame, so blurring it moves no baseline median.
fn unsampled_choice(run: &Run, after: usize) -> usize {
    let samples = &run.report().sequences[0].baseline.samples;
    *run.walk(0)
        .chosen
        .iter()
        .find(|&&c| c > after && !samples.contains(&c) && !samples.contains(&(c + 1)))
        .expect("a chosen frame that is not sampled")
}

/// The report's reading of `frame`, which was read.
fn judged(run: &Run, frame: usize) -> FrameReading {
    run.report()
        .frames
        .iter()
        .find(|f| f.id.frame == frame)
        .expect("the frame was read")
        .clone()
}

#[test]
fn a_blurred_frame_is_swapped_for_a_sharp_neighbour() {
    // A frame the walk chose, made less sharp than 0.6 of the median.
    let mut blurred = Synth::steady(3, 300, ONE_PERCENT);
    let soft = unsampled_choice(&run(&[&blurred]), 100);
    blurred.shots[soft].readings[0].0 = 0.3;
    let run = run(&[&blurred]);
    let walk = run.walk(0);
    assert!(!walk.chosen.contains(&soft), "{soft} was chosen");
    let swap = walk
        .swaps
        .iter()
        .find(|s| s.candidate == soft)
        .unwrap_or_else(|| panic!("{soft} was not swapped: {:?}", walk.swaps));
    assert!(swap.chosen.abs_diff(soft) <= 2);
    assert!(walk.chosen.contains(&swap.chosen));
    assert!(judged(&run, soft).sensors[0].sharpness_ratio < 0.6);
    assert!(judged(&run, soft).blurred);
    assert_read_is_asked(&run);
}

/// A run with a sparser baseline than [`options`]'s, 12 samples, so a short
/// capture still has stretches of frames the baseline does not read.
fn run_sparse(synths: &[&Synth]) -> Run {
    let sparse = PickImagesOptions {
        baseline_frames: 12,
        ..options()
    };
    run_with(synths, &sparse, Fault::None, None)
}

/// The point density of the captures the tests of blur against the
/// neighbours walk: about 250 features a frame, so a step in the band keeps
/// some 200 matches, several times the `2 × min_matches` a step beyond the
/// next frame needs, at a quarter of [`Synth::steady`]'s cost per frame.
const NEIGHBOUR_DENSITY: f64 = 20.0;

/// A steady capture of 200 frames at [`NEIGHBOUR_DENSITY`] and its clean
/// run, which the tests that change one frame of it read to choose the
/// frame.
fn clean_steady() -> &'static (Synth, Run) {
    static CLEAN: OnceLock<(Synth, Run)> = OnceLock::new();
    CLEAN.get_or_init(|| {
        let synth = Synth::steady_with_density(34, 200, ONE_PERCENT, NEIGHBOUR_DENSITY);
        let run = run_sparse(&[&synth]);
        (synth, run)
    })
}

/// A step after `after` to a frame in the band and, when `stride` is given,
/// that many frames long, with every frame within two positions of its frame
/// or of the frame after it `free`: its `from` and `to`.
fn step_with_free_window(
    run: &Run,
    after: usize,
    stride: Option<usize>,
    free: impl Fn(usize) -> bool,
) -> (usize, usize) {
    run.walk(0)
        .steps
        .iter()
        .find(|s| {
            s.to > after
                && s.kind == StepKind::InBand
                && stride.is_none_or(|k| s.to - s.from == k)
                && (s.to - 2..=s.to + 3).all(&free)
        })
        .map(|s| (s.from, s.to))
        .expect("a step whose frame has the window asked for")
}

/// [`step_with_free_window`] with no sampled frame in the window, so
/// changing the readings there moves no baseline median.
fn unsampled_window_step(run: &Run, after: usize, stride: Option<usize>) -> (usize, usize) {
    let samples = &run.report().sequences[0].baseline.samples;
    step_with_free_window(run, after, stride, |t| !samples.contains(&t))
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
/// step was `own`: it matches well, has no few matches beyond the next frame,
/// and is no longer than the band and no shorter than the band or `own`.
fn swap_fits(stats: &FramePairStats, from: usize, n: usize, own: f64) -> bool {
    let o = options();
    let (low, high) = (
        o.target_step_percent * (1.0 - o.step_band),
        o.target_step_percent * (1.0 + o.step_band),
    );
    baseline::matches_well(stats, &o)
        && (n == from + 1 || stats.matches >= 2 * o.pair.min_matches)
        && stats.step_percent >= own.min(low)
        && stats.step_percent <= high
}

#[test]
fn a_frame_blurred_against_its_neighbours_is_swapped_for_the_sharpest_that_fits() {
    // A frame the walk chose, made less sharp than its neighbours by more
    // than 0.85 while staying above 0.6 of the median, so only the comparison
    // with its neighbours finds it blurred. The two frames 2 positions away
    // are the sharpest, the later one sharper: one of their steps is always
    // in the band, and neither is the frame a nearest-first order would
    // try first.
    let (clean, first) = clean_steady();
    let mut synth = clean.clone();
    let (_, soft) = unsampled_window_step(first, 40, None);
    synth.shots[soft].readings[0].0 = 0.7;
    synth.shots[soft + 2].readings[0].0 = 1.3;
    synth.shots[soft - 2].readings[0].0 = 1.2;
    let run = run_sparse(&[&synth]);
    let walk = run.walk(0);
    let reading = judged(&run, soft);
    assert!(!reading.blurred, "{reading:?}");
    assert!((0.6..1.0).contains(&reading.sensors[0].sharpness_ratio));
    let swap = walk
        .swaps
        .iter()
        .find(|s| s.candidate == soft)
        .unwrap_or_else(|| panic!("{soft} was not swapped: {:?}", walk.swaps));
    assert_eq!(swap.reason, SwapReason::Neighbours);
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
fn a_frame_sharper_than_the_median_is_not_compared_with_its_neighbours() {
    // A frame the walk chose, above its median sharpness and far below its
    // neighbours: compared with them it would be blurred, but a frame
    // sharper than the median is not compared, and its neighbours are not
    // read. No stage before the walk reads a frame near it.
    let (clean, first) = clean_steady();
    let mut synth = clean.clone();
    let by_baseline: BTreeSet<usize> = first
        .report()
        .frames
        .iter()
        .filter(|f| f.stage != Stage::Walk)
        .map(|f| f.id.frame)
        .collect();
    let (_, frame) = step_with_free_window(first, 40, None, |t| !by_baseline.contains(&t));
    let window: Vec<usize> = (frame - 2..=frame + 2).filter(|&n| n != frame).collect();
    synth.shots[frame].readings[0].0 = 1.3;
    for &n in &window {
        synth.shots[n].readings[0].0 = 2.0;
    }
    let run = run_sparse(&[&synth]);
    let walk = run.walk(0);
    assert!(walk.chosen.contains(&frame), "{frame} was not chosen");
    assert!(walk.swaps.iter().all(|s| s.candidate != frame));
    assert!(judged(&run, frame).sensors[0].sharpness_ratio > 1.0);
    for n in window {
        let id = FrameId {
            sequence: 0,
            frame: n,
        };
        assert!(!run.asked.contains(&id), "{n} was read");
        assert!(!run.prefetched.contains(&id), "{n} was prefetched");
    }
    assert_read_is_asked(&run);
}

#[test]
fn a_uniformly_blurred_stretch_is_not_swapped_against_its_neighbours() {
    // A stretch whose frames are all equally soft, at 0.7 of the median: each
    // frame chosen in it is compared with its neighbours, which share the
    // blur, so none is swapped for one.
    let mut synth = Synth::steady_with_density(36, 150, ONE_PERCENT, NEIGHBOUR_DENSITY);
    for t in 50..=110 {
        synth.shots[t].readings[0].0 = 0.7 + 0.005 * ((t * 7) % 5) as f32;
    }
    let run = run_sparse(&[&synth]);
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
        let reading = judged(&run, c);
        assert!(!reading.blurred && reading.sensors[0].sharpness_ratio < 1.0);
        assert!((c - 2..=c + 2).all(|n| was_read(&run, &id(n))), "{c}");
    }
    assert!(
        walk.swaps.iter().all(|s| !inner.contains(&s.candidate)),
        "{:?}",
        walk.swaps
    );
    assert_read_is_asked(&run);
}

#[test]
fn a_neighbour_whose_step_does_not_fit_is_not_swapped_in() {
    // At about 2.5% a frame, a step of 3 to 5 frames is in the band and one
    // of 2 or 6 is not. The candidate is blurred against its neighbours. The two
    // frames 2 away are the sharpest and out of the band; the frame after it
    // is the next sharpest and shows one point in 5, so its step has few
    // matches; and the frame before it qualifies.
    let mut synth = Synth::steady_with_density(37, 100, 2.7 * ONE_PERCENT, NEIGHBOUR_DENSITY);
    let (from, soft) = unsampled_window_step(&run_sparse(&[&synth]), 30, Some(4));
    synth.shots[soft].readings[0].0 = 0.7;
    synth.shots[soft - 2].readings[0].0 = 1.3;
    synth.shots[soft + 2].readings[0].0 = 1.3;
    synth.shots[soft + 1].readings[0].0 = 1.25;
    synth.shots[soft + 1].content = Content::Subset { every: 5 };
    synth.shots[soft - 1].readings[0].0 = 1.1;
    let swapped = run_sparse(&[&synth]);
    let walk = swapped.walk(0);
    assert!(walk.chosen.contains(&from));
    assert!(
        walk.swaps.contains(&Swap {
            candidate: soft,
            chosen: soft - 1,
            reason: SwapReason::Neighbours,
        }),
        "{:?}",
        walk.swaps
    );
    let o = options();
    let low = o.target_step_percent * (1.0 - o.step_band);
    let high = o.target_step_percent * (1.0 + o.step_band);
    let step = |n: usize| measured(&swapped, from, n).unwrap().clone();
    assert!(step(soft - 2).step_percent < low, "{:?}", step(soft - 2));
    assert!(step(soft + 2).step_percent > high, "{:?}", step(soft + 2));
    let few = step(soft + 1);
    assert!((low..=high).contains(&few.step_percent), "{few:?}");
    assert!(few.matches < 2 * o.pair.min_matches, "{few:?}");
    assert_read_is_asked(&swapped);

    // With the frame before it as soft as the candidate, no neighbour
    // qualifies, and the candidate is kept.
    synth.shots[soft - 1].readings[0].0 = 0.7;
    let kept = run_sparse(&[&synth]);
    let walk = kept.walk(0);
    assert!(walk.chosen.contains(&soft), "{soft} was not kept");
    assert!(walk.swaps.iter().all(|s| s.candidate != soft));
    assert_read_is_asked(&kept);
}

#[test]
fn gradients_along_one_direction_do_not_blur_a_sharp_frame() {
    // A frame the walk chose, given a strongly directional tensor at its
    // usual sharpness, as a sharp frame of a slope or of a line-like subject
    // reads. It is still chosen, not swapped, and not blurred.
    let mut streaked = Synth::steady(3, 300, ONE_PERCENT);
    let frame = unsampled_choice(&run(&[&streaked]), 100);
    streaked.shots[frame].readings[0].1 = [1.0, 0.2, 0.0];
    let run = run(&[&streaked]);
    let walk = run.walk(0);
    assert!(walk.chosen.contains(&frame), "{frame} was not chosen");
    assert!(walk.swaps.iter().all(|s| s.candidate != frame));
    let reading = judged(&run, frame);
    assert!(reading.sensors[0].axis_ratio.unwrap() > 1.5);
    assert!(reading.sensors[0].coherence > 0.6);
    assert!(!reading.blurred);
    assert_read_is_asked(&run);
}

#[test]
fn a_blurred_run_keeps_its_sharpest_frames() {
    let mut synth = Synth::steady(4, 300, ONE_PERCENT);
    let run_frames = 140..=170;
    for t in run_frames.clone() {
        synth.shots[t].readings[0].0 = 0.2 + 0.01 * ((t * 7) % 10) as f32;
    }
    let run = run(&[&synth]);
    let report = run.report();
    let walk = run.walk(0);
    assert!(walk.breaks.is_empty());
    assert!(!walk.kept_blurred.is_empty());
    assert!(report.sequences[0]
        .baseline
        .flagged_regions
        .iter()
        .any(|r| r.blurred));
    let sharpness = |t: usize| synth.shots[t].readings[0].0;
    for kept in &walk.kept_blurred {
        assert!(run_frames.contains(&kept.frame));
        assert!(walk.chosen.contains(&kept.frame));
        // The sharpest of the frames searched whose step from the last chosen
        // frame matched well within the band.
        let step = walk.steps.iter().find(|s| s.to == kept.frame).unwrap();
        let (low, high) = band(step);
        assert!((low..=high).contains(&step.stats.step_percent), "{step:?}");
        let in_band = |n: usize| {
            report.pairs.iter().any(|p| {
                p.a.frame == step.from
                    && p.b.frame == n
                    && baseline::matches_well(&p.stats, &options())
                    && (low..=high).contains(&p.stats.step_percent)
            })
        };
        let best = (kept.searched_first..=kept.searched_last)
            .filter(|&n| in_band(n))
            .map(sharpness)
            .fold(f32::MIN, f32::max);
        assert_eq!(sharpness(kept.frame), best);
    }
    // The walk keeps going through the run, and leaves it.
    assert!(walk.chosen.iter().any(|c| run_frames.contains(c)));
    assert_eq!(walk.chosen.last(), Some(&299));
    assert_read_is_asked(&run);
}

/// Every accepted step with fewer than twice `min_matches` matches goes to
/// the next frame or fills in a gap; and the walk measured at least one such
/// pair beyond the next frame from a chosen frame, so the rule was tested.
fn assert_few_match_steps_only_where_allowed(run: &Run) {
    let few = |matches: usize| matches < 2 * options().pair.min_matches;
    let walk = run.walk(0);
    for step in &walk.steps {
        if few(step.stats.matches) {
            assert!(
                step.to == step.from + 1 || step.kind == StepKind::GapFill,
                "{step:?}"
            );
        }
    }
    let chosen: BTreeSet<usize> = walk.chosen.iter().copied().collect();
    assert!(
        run.report().pairs.iter().any(|p| p.stage == Stage::Walk
            && chosen.contains(&p.a.frame)
            && p.b.frame > p.a.frame + 1
            && p.stats.matches >= options().pair.min_matches
            && few(p.stats.matches)),
        "no pair with few matches was measured"
    );
}

#[test]
fn a_blurred_frame_is_not_swapped_for_a_neighbour_with_few_matches() {
    // The frame the walk would choose is blurred, and its sharp neighbours
    // show only one point in 20, so a step to one of them has few matches.
    let mut synth = Synth::steady(29, 300, ONE_PERCENT);
    let soft = unsampled_choice(&run(&[&synth]), 100);
    synth.shots[soft].readings[0].0 = 0.3;
    for t in soft - 4..=soft + 4 {
        if t != soft {
            synth.shots[t].content = Content::Subset { every: 20 };
        }
    }
    let run = run(&[&synth]);
    let walk = run.walk(0);
    assert!(!walk.chosen.contains(&soft));
    assert!(walk.breaks.is_empty());
    assert_few_match_steps_only_where_allowed(&run);
    assert_read_is_asked(&run);
}

#[test]
fn a_blurred_run_does_not_take_a_sharp_frame_with_few_matches() {
    // The frames around the one the walk would choose are blurred, so no
    // neighbour can be swapped in, and the walk reads the blurred run. The
    // only sharp frames in it past the next frame show one point in 20, so a
    // step to one of them has few matches.
    let mut synth = Synth::steady(30, 300, ONE_PERCENT);
    let first = run(&[&synth]);
    let target = unsampled_choice(&first, 100);
    let walk = first.walk(0);
    let from = walk.chosen[walk.chosen.iter().position(|&c| c == target).unwrap() - 1];
    for t in from + 2..=from + 5 {
        synth.shots[t].content = Content::Subset { every: 20 };
    }
    for t in from + 6..=target + 4 {
        synth.shots[t].readings[0].0 = 0.3;
    }
    let run = run(&[&synth]);
    let walk = run.walk(0);
    assert!(walk.chosen.contains(&from));
    assert!(walk.breaks.is_empty());
    assert_few_match_steps_only_where_allowed(&run);
    assert_read_is_asked(&run);
}

#[test]
fn a_blurred_run_does_not_keep_a_sharper_blurred_frame_with_few_matches() {
    // Every frame from the next one to past the frame the walk would choose
    // is blurred, so the walk keeps the sharpest frame in the band. The
    // frames past the next one that are less blurred than that frame show
    // one point in 20, so a step to one of them has few matches.
    let mut synth = Synth::steady(31, 300, ONE_PERCENT);
    let first = run(&[&synth]);
    let target = unsampled_choice(&first, 100);
    let walk = first.walk(0);
    let from = walk.chosen[walk.chosen.iter().position(|&c| c == target).unwrap() - 1];
    for t in from + 1..=target + 4 {
        synth.shots[t].readings[0].0 = 0.2;
        if t > from + 1 && t != target {
            synth.shots[t].readings[0].0 = 0.3;
            synth.shots[t].content = Content::Subset { every: 20 };
        }
    }
    let run = run(&[&synth]);
    let walk = run.walk(0);
    assert!(walk.chosen.contains(&from));
    assert!(walk.breaks.is_empty());
    assert_few_match_steps_only_where_allowed(&run);
    assert!(
        walk.kept_blurred.iter().any(|k| k.frame == target),
        "{:?}",
        walk.kept_blurred
    );
    assert_read_is_asked(&run);
}

/// The band a step was held to: halved with the target in a flagged region.
fn band(step: &WalkStep) -> (f64, f64) {
    let o = options();
    let target = o.target_step_percent * if step.flagged { 0.5 } else { 1.0 };
    (target * (1.0 - o.step_band), target * (1.0 + o.step_band))
}

#[test]
fn a_long_blurred_stretch_is_crossed_at_the_bands_spacing() {
    let mut synth = Synth::steady(26, 300, ONE_PERCENT);
    let stretch = 80..=220;
    for t in stretch.clone() {
        synth.shots[t].readings[0].0 = 0.2 + 0.01 * ((t * 7) % 10) as f32;
    }
    let run = run(&[&synth]);
    let walk = run.walk(0);
    let inside: Vec<&WalkStep> = walk
        .steps
        .iter()
        .filter(|s| stretch.contains(&s.from) && stretch.contains(&s.to))
        .collect();
    assert!(inside.len() >= 10, "{} steps in the stretch", inside.len());
    for step in &inside {
        let (low, high) = band(step);
        assert!(
            (low..=high).contains(&step.stats.step_percent),
            "a step across the blurred stretch left its band: {step:?}"
        );
    }
    // Every frame chosen in the stretch is blurred, so it is kept though
    // blurred, at the band's spacing rather than one frame at a time. The
    // stretch matches well, so it is flagged for blur alone and the band is
    // not halved there: about 10 frames a step at 1% a frame.
    assert!(inside
        .iter()
        .all(|s| walk.kept_blurred.iter().any(|k| k.frame == s.to)));
    let regions = &run.report().sequences[0].baseline.flagged_regions;
    assert!(regions.iter().any(|r| r.blurred));
    assert!(regions.iter().all(|r| !r.weak_matches), "{regions:?}");
    assert!(inside.iter().all(|s| !s.flagged));
    assert!(
        median(&strides(walk, |s| stretch.contains(&s.from)
            && stretch.contains(&s.to)))
            >= 7.0
    );
    assert!(walk.breaks.is_empty());
    assert_read_is_asked(&run);
}

#[test]
fn only_the_sharpness_judges_blur() {
    let reading = |sharpness: f32, tensor: [f32; 3]| ImageReadings::new(sharpness, tensor).unwrap();
    let readings: Vec<ImageReadings> = (0..9)
        .map(|i| reading(0.9 + 0.025 * i as f32, [1.0, 1.0, 0.0]))
        .collect();
    let base = baseline::sensor_baseline(&readings);
    assert!((base.sharpness_median - 1.0).abs() < 1e-6);
    let judge = |r: ImageReadings| baseline::judge(&r, &base, 0.6);
    assert!(judge(reading(0.59, [1.0, 1.0, 0.0])).blurred);
    assert!(!judge(reading(0.61, [1.0, 1.0, 0.0])).blurred);
    // Gradients in one direction only, at the median sharpness: the readings
    // are reported, and the image is not blurred.
    let streak = judge(reading(1.0, [1.0, 0.01, 0.0]));
    assert!(streak.axis_ratio.unwrap() > 4.0);
    assert!(streak.coherence > 0.9);
    assert!(!streak.blurred);
}

#[test]
fn a_bad_patch_is_filled_in_at_shorter_steps() {
    let mut synth = Synth::steady(5, 300, ONE_PERCENT);
    let patch = 120..=180;
    for t in patch.clone() {
        synth.shots[t].content = Content::Window {
            period: 40,
            lifetime: 4,
        };
    }
    let run = run(&[&synth]);
    let walk = run.walk(0);
    assert!(walk.breaks.is_empty(), "{:?}", walk.breaks);
    assert!(!walk.gaps.is_empty());
    for gap in &walk.gaps {
        for step in &walk.steps[gap.first_step..gap.end_step] {
            assert_eq!(step.kind, StepKind::GapFill);
            assert!(step.to - step.from <= 2, "{step:?}");
            assert!(step.from >= 110 && step.to <= 182, "{step:?}");
        }
    }
    // The patch is flagged for weak matches, and the steps from inside it
    // have their band halved.
    let regions = &run.report().sequences[0].baseline.flagged_regions;
    let weak: Vec<&FlaggedRegion> = regions.iter().filter(|r| r.weak_matches).collect();
    assert!(!weak.is_empty(), "{regions:?}");
    for step in &walk.steps {
        let in_weak = weak
            .iter()
            .any(|r| r.first <= step.from && step.from <= r.last);
        assert_eq!(step.flagged, in_weak, "{step:?}");
    }
    assert!(walk.steps.iter().any(|s| s.flagged));
    // Dense across the patch, sparse elsewhere.
    let inside = walk.chosen.iter().filter(|c| patch.contains(c)).count();
    assert!(inside >= 20, "{inside} frames chosen in the patch");
    let outside = median(&strides(walk, |s| s.to < 110 || s.from > 190));
    assert!(outside >= 8.0);
    assert_read_is_asked(&run);
}

#[test]
fn a_break_starts_a_new_segment_and_weak_points_are_reported() {
    let mut synth = Synth::steady(6, 300, ONE_PERCENT);
    for t in (2..=30).chain(150..=175) {
        synth.shots[t].content = Content::Junk;
    }
    for t in 220..=260 {
        synth.shots[t].content = Content::Subset { every: 20 };
    }
    let run = run(&[&synth]);
    let walk = run.walk(0);
    assert_eq!(
        walk.breaks,
        vec![
            Break {
                after: 1,
                resumes_at: Some(31)
            },
            Break {
                after: 149,
                resumes_at: Some(176)
            }
        ]
    );
    assert_eq!(
        walk.segments,
        vec![
            Segment { start: 0, len: 2 },
            Segment {
                start: 2,
                len: walk
                    .chosen
                    .iter()
                    .filter(|&&c| (31..150).contains(&c))
                    .count()
            },
            Segment {
                start: 2 + walk
                    .chosen
                    .iter()
                    .filter(|&&c| (31..150).contains(&c))
                    .count(),
                len: walk.chosen.iter().filter(|&&c| c >= 176).count()
            },
        ]
    );
    assert!(!walk
        .chosen
        .iter()
        .any(|c| (2..=30).contains(c) || (150..=175).contains(c)));
    assert!(walk
        .weak_points
        .contains(&WeakPoint::ShortSegment { segment: 0 }));
    let few: Vec<&WalkStep> = walk
        .weak_points
        .iter()
        .filter_map(|w| match w {
            WeakPoint::Step {
                step,
                few_matches: true,
                ..
            } => Some(&walk.steps[*step]),
            _ => None,
        })
        .collect();
    assert!(!few.is_empty());
    for step in few {
        assert!(step.to >= 220 && step.from <= 260, "{step:?}");
        // A step with few matches is settled on only at the next frame, or
        // when halving a failed step finds nothing better.
        assert!(
            step.to == step.from + 1 || step.kind == StepKind::GapFill,
            "{step:?}"
        );
    }
    // Every frame of a break was read: the walk looked for the first that
    // matches its next frame.
    let asked: BTreeSet<usize> = run.asked.iter().map(|id| id.frame).collect();
    assert!((1..=31).chain(149..=176).all(|t| asked.contains(&t)));
    assert_read_is_asked(&run);
}

#[test]
fn shuffled_frames_are_not_a_video() {
    let steady = Synth::steady(7, 300, ONE_PERCENT);
    let mut order: Vec<usize> = (0..300).collect();
    // A fixed shuffle: neighbours in the order are far apart on the path.
    order.sort_by_key(|&t| (t * 7919) % 300);
    let xs: Vec<f64> = order.iter().map(|&t| ONE_PERCENT * t as f64).collect();
    let shuffled = Synth::wall(7, -12.0, ONE_PERCENT * 300.0 + 12.0, 80.0).along(xs);
    let run = run(&[&steady, &shuffled]);
    let report = run.report();
    assert_eq!(report.verdict, Verdict::NotAVideo);
    assert!(report.sequences[0].video.is_video);
    let photos = &report.sequences[1].video;
    assert!(!photos.is_video);
    assert!(!photos.step_small && !photos.neighbours_match);
    for sequence in &report.sequences {
        assert!(sequence.solvability.is_none() && sequence.walk.is_none());
    }
    // Nothing is read, or asked for, after the first batch.
    assert!(report.frames.iter().all(|f| f.stage == Stage::BaselineNear));
    assert!(run.prefetched.iter().all(|id| was_read(&run, id)));
    assert_read_is_asked(&run);
}

#[test]
fn frames_whose_neighbours_are_far_apart_are_not_a_video() {
    // Frames 6% of the long side apart: the motion adds up, the step is
    // steady and neighbours match closely, but a step that large is no
    // video's.
    let synth = Synth::steady(28, 120, 6.0 * ONE_PERCENT);
    let run = run(&[&synth]);
    let report = run.report();
    let video = &report.sequences[0].video;
    assert!(
        video.motion_adds_up && video.step_steady && video.neighbours_match,
        "{video:?}"
    );
    assert!(!video.step_small, "{video:?}");
    assert!(!video.is_video);
    assert_eq!(report.verdict, Verdict::NotAVideo);
    assert_read_is_asked(&run);
}

#[test]
fn a_sequence_of_three_frames_is_not_a_video() {
    let synth = Synth::steady(8, 3, ONE_PERCENT);
    let run = run(&[&synth]);
    assert_eq!(run.report().verdict, Verdict::NotAVideo);
    assert!(!run.report().sequences[0].video.is_video);
}

fn assert_not_solvable(run: &Run) -> &SolvabilityCheck {
    let report = run.report();
    assert_eq!(report.verdict, Verdict::NotSolvable);
    assert!(report.sequences[0].video.is_video);
    assert!(report.sequences[0].walk.is_none());
    assert!(report.frames.iter().all(|f| f.stage != Stage::Walk));
    assert!(report.frames.iter().any(|f| f.stage == Stage::BaselineFar));
    assert_read_is_asked(run);
    report.sequences[0].solvability.as_ref().unwrap()
}

#[test]
fn a_camera_that_only_turns_is_not_solvable() {
    let mut synth = Synth::surround(9, 2500).along((0..300).map(|_| 0.0));
    for (t, shot) in synth.shots.iter_mut().enumerate() {
        shot.yaw = (0.5 * t as f64).to_radians();
    }
    let run = run(&[&synth]);
    let check = assert_not_solvable(&run);
    assert!(check.panorama && !check.moving_scene && !check.barely_moves);
    assert!(check.h_over_f_30_median.unwrap() > 0.97);
    assert!(check.rotation_error_30_median.unwrap() < 0.01);
}

#[test]
fn a_camera_that_barely_moves_is_not_solvable() {
    let synth = Synth::steady(10, 300, 0.25 * ONE_PERCENT);
    let run = run(&[&synth]);
    let check = assert_not_solvable(&run);
    assert!(check.barely_moves && !check.panorama && !check.moving_scene);
    assert!(check.total_motion_percent < 100.0);
}

#[test]
fn a_moving_scene_is_not_solvable() {
    let synth = Synth::steady(11, 300, ONE_PERCENT).moving(11, 0.03);
    let run = run(&[&synth]);
    let check = assert_not_solvable(&run);
    assert!(check.moving_scene && !check.panorama);
    assert!(check.f_inlier_share_10_median.unwrap() < 0.3);
}

#[test]
fn a_rig_takes_its_motion_from_the_reference_sensor_and_blur_from_either() {
    let mut steady = Synth::steady_with_density(12, 200, ONE_PERCENT, NEIGHBOUR_DENSITY);
    steady.sensors = 2;
    let first = run_sparse(&[&steady]);
    let report = first.report();
    assert_eq!(report.verdict, Verdict::Solvable);
    // The second sensor sees only random features of another size, so the
    // walk matched the reference sensor's images alone.
    let walk = first.walk(0);
    assert!(walk.breaks.is_empty() && walk.swaps.is_empty());
    let stride = median(&strides(walk, |_| true));
    assert!((8.0..=12.0).contains(&stride));
    // Each sensor against its own median: sharpness near 1 and near 5.
    let sensors = &report.sequences[0].baseline.sensors;
    assert_eq!(sensors.len(), 2);
    assert!((sensors[0].sharpness_median - 1.0).abs() < 0.1);
    assert!((sensors[1].sharpness_median - 5.0).abs() < 0.5);
    assert_eq!(report.sequences[0].baseline.blurred_samples, 0);
    assert_read_is_asked(&first);

    // The second sensor's image of a chosen frame blurred makes the frame
    // blurred, and it is swapped. In the same run, an earlier chosen frame
    // is blurred against its neighbours on the second sensor, but the one
    // neighbour sharper there by the ratio is less sharp than it by the ratio
    // on the reference sensor, though not blurred: no neighbour qualifies,
    // and it is kept, so the walk after it is the clean run's.
    let frame = unsampled_choice(&first, 60);
    let (_, guarded) = unsampled_window_step(&first, 20, None);
    assert!(guarded + 5 < frame, "{guarded} {frame}");
    steady.shots[guarded].readings[0].0 = 1.1;
    steady.shots[guarded].readings[1].0 = 3.6;
    for n in [guarded - 2, guarded - 1, guarded + 2] {
        steady.shots[n].readings[1].0 = 4.0;
    }
    steady.shots[guarded + 1].readings[0].0 = 0.85;
    steady.shots[guarded + 1].readings[1].0 = 6.5;
    steady.shots[frame].readings[1] = (1.0, sharp(frame, 1).1);
    let run = run_sparse(&[&steady]);
    let walk = run.walk(0);
    assert!(walk.chosen.contains(&guarded), "{guarded} was not kept");
    assert!(walk.swaps.iter().all(|s| s.candidate != guarded));
    let id = |frame| FrameId { sequence: 0, frame };
    assert!(was_read(&run, &id(guarded + 1)));
    assert!(!judged(&run, guarded + 1).blurred);
    assert!(!walk.chosen.contains(&frame));
    assert!(walk.swaps.iter().any(|s| s.candidate == frame));
    let read = run
        .report()
        .frames
        .iter()
        .find(|f| f.id.frame == frame)
        .unwrap();
    assert!(read.blurred && !read.sensors[0].blurred && read.sensors[1].blurred);

    // The second sensor's image of another chosen frame below its own
    // neighbours', the reference sensor's above its median: the frame is
    // blurred against its neighbours on the second sensor alone, judged
    // against that sensor's neighbours, and swapped. Of the two nearest
    // neighbours, the one after it gains more on its least improved sensor
    // (1.36 against 1.31) and the one before it more on the reference
    // sensor (1.55 against 1.36) and on its most improved sensor (1.55
    // against 1.39), so the frame after it is taken.
    steady.shots[frame].readings[1] = sharp(frame, 1);
    for n in guarded - 2..=guarded + 2 {
        steady.shots[n].readings = vec![sharp(n, 0), sharp(n, 1)];
    }
    let (_, other) = unsampled_window_step(&first, 130, None);
    steady.shots[other].readings[0].0 = 1.1;
    steady.shots[other].readings[1].0 = 3.6;
    steady.shots[other + 1].readings[0].0 = 1.5;
    steady.shots[other + 1].readings[1].0 = 5.0;
    steady.shots[other - 1].readings[0].0 = 1.7;
    steady.shots[other - 1].readings[1].0 = 4.7;
    let local = run_sparse(&[&steady]);
    let walk = local.walk(0);
    assert!(!walk.chosen.contains(&other));
    assert!(
        walk.swaps.contains(&Swap {
            candidate: other,
            chosen: other + 1,
            reason: SwapReason::Neighbours,
        }),
        "{:?}",
        walk.swaps
    );
    let read = judged(&local, other);
    assert!(!read.blurred, "{read:?}");
    assert!(read.sensors[0].sharpness_ratio > 1.0 && read.sensors[1].sharpness_ratio > 0.6);
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

fn was_read(run: &Run, id: &FrameId) -> bool {
    run.report().frames.iter().any(|f| f.id == *id)
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
/// a blurred frame, a frame blurred against its neighbours, a blurred run, a
/// bad patch filled in and a break.
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
    // Every third frame blurred, so a candidate that lands on one is
    // swapped for a sharp neighbour.
    for t in (15..=45).step_by(3) {
        synth.shots[t].readings[0].0 = 0.3;
    }
    for t in 52..=72 {
        synth.shots[t].readings[0].0 = 0.2 + 0.01 * ((t * 7) % 10) as f32;
    }
    for t in (118..=129).filter(|t| t % 3 != 1) {
        synth.shots[t].readings[0].0 = 0.7;
    }
    for t in 85..=115 {
        synth.shots[t].content = Content::Window {
            period: 40,
            lifetime: 4,
        };
    }
    for t in 160..=170 {
        synth.shots[t].content = Content::Junk;
    }
    synth
}

#[test]
fn the_report_is_the_same_for_any_lookahead_and_thread_count() {
    let synth = eventful();
    let off = PickImagesOptions {
        prefetch: false,
        ..options()
    };
    let reference = run_on(&[&synth], &off, Fault::None, None, 1);
    let walk = reference.walk(0);
    // The capture exercises what it is meant to.
    for reason in [SwapReason::Median, SwapReason::Neighbours] {
        assert!(
            walk.swaps.iter().any(|s| s.reason == reason),
            "no swap for {reason:?}"
        );
    }
    assert!(!walk.gaps.is_empty(), "no gap");
    assert!(!walk.breaks.is_empty(), "no break");
    assert!(!walk.kept_blurred.is_empty(), "nothing kept though blurred");
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

#[test]
fn the_baseline_and_a_break_are_read_ahead_and_a_frame_failing_unread_stops_nothing() {
    // A steady video with a stretch that matches nothing, which the walk
    // reads one frame at a time. The one clean run serves both halves.
    let mut synth = Synth::steady(33, 200, ONE_PERCENT);
    for t in 120..=140 {
        synth.shots[t].content = Content::Junk;
    }
    let run = run(&[&synth]);
    let report = run.report();
    let walk = run.walk(0);
    assert_eq!(walk.breaks.len(), 1, "{:?}", walk.breaks);
    // Every frame of both batches of the baseline was prefetched.
    for frame in report.frames.iter().filter(|f| f.stage != Stage::Walk) {
        assert!(run.prefetched.contains(&frame.id), "{:?}", frame.id);
    }
    // Through the break, every frame waited for after the first pair that
    // failed was prefetched, and nothing reaches past the break by more than
    // the source's workers.
    let after = walk.breaks[0].after;
    let resumes = walk.breaks[0].resumes_at.expect("the walk resumes");
    let id = |frame| FrameId { sequence: 0, frame };
    let first = run
        .waits
        .iter()
        .position(|w| w.contains(&id(after + 2)))
        .expect("the break's first pair was waited for");
    let last = run
        .waits
        .iter()
        .position(|w| w.contains(&id(resumes + 1)))
        .expect("the resumed pair was waited for");
    let through: Vec<FrameId> = run.waits[first + 1..=last].concat();
    assert!(!through.is_empty());
    for frame in &through {
        assert!(run.prefetched.contains(frame), "{frame:?}");
    }
    let walked_ahead = run.prefetched.iter().filter(|f| !was_read(&run, f)).count();
    assert!(
        walked_ahead <= THREADS,
        "{walked_ahead} predicted and never read"
    );
    assert_read_is_asked(&run);

    // The break read ahead leaves frames past its end prefetched and unread;
    // a worker failing on one of them stops nothing.
    let never_read = *run
        .prefetched
        .iter()
        .find(|id| !was_read(&run, id))
        .expect("a frame predicted and never read");
    let failing = run_with(
        &[&synth],
        &options(),
        Fault::FailFrame(never_read.frame),
        None,
    );
    assert_eq!(decisions(&failing), decisions(&run));
}

#[test]
fn several_sequences_share_the_baseline_and_are_walked_in_turn() {
    let one = Synth::steady(14, 300, ONE_PERCENT);
    let two = Synth::steady(15, 150, ONE_PERCENT);
    let run = run(&[&one, &two]);
    let report = run.report();
    assert_eq!(report.verdict, Verdict::Solvable);
    // 30 samples shared 2 to 1.
    assert_eq!(report.sequences[0].baseline.samples.len(), 20);
    assert_eq!(report.sequences[1].baseline.samples.len(), 10);
    assert_eq!(run.walk(0).chosen.last(), Some(&299));
    assert_eq!(run.walk(1).chosen.last(), Some(&149));
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
            "blur_ratio",
            PickImagesOptions {
                blur_ratio: 1.0,
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
            "baseline_frames",
            PickImagesOptions {
                baseline_frames: 1,
                ..options()
            },
        ),
        (
            "min_inlier_share",
            PickImagesOptions {
                min_inlier_share: 1.5,
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
            "pair.min_matches",
            PickImagesOptions {
                pair: FramePairOptions {
                    min_matches: 6,
                    ..FramePairOptions::default()
                },
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
        baseline_frames: 2,
        min_inlier_share: 1.0,
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
    // A frame of the first batch, of the second, and of the walk.
    for stage in [Stage::BaselineNear, Stage::BaselineFar, Stage::Walk] {
        let frame = clean
            .report()
            .frames
            .iter()
            .find(|f| f.stage == stage && f.id.frame > 0)
            .expect("a frame of the stage")
            .id;
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
    // Frame 1, matched with frame 0 in the first batch, at another size.
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
