// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The walk on synthetic captures: a scene of points with random descriptors
//! seen along a camera path with known motion per frame, written as version 2
//! `.sift` files only when the source is asked for them.

mod capture;

use std::collections::BTreeSet;
use std::sync::atomic::AtomicBool;

use capture::{sharp, sift_files, Content, Fault, Synth, SynthSource, ONE_PERCENT};

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

struct Run {
    result: Result<PickImagesReport, PickImagesError>,
    asked: Vec<FrameId>,
    calls: Vec<Vec<FrameId>>,
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
    let dir = tempfile::tempdir().expect("a temporary folder");
    let sequences: Vec<CaptureSequence> = synths
        .iter()
        .enumerate()
        .map(|(i, s)| s.sequence(&dir.path().join(format!("seq{i}"))))
        .collect();
    let flag = AtomicBool::new(false);
    let flag = cancel.unwrap_or(&flag);
    let progress = Progress::none().cancelled_by(flag);
    let (result, asked, calls) = {
        let mut source = SynthSource::new(synths, &sequences);
        source.fault = fault;
        source.cancel = Some(flag);
        let result = pick_images(&sequences, &mut source, options, &progress);
        (result, source.asked, source.calls)
    };
    Run {
        result,
        asked,
        calls,
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

/// The frames the source was asked for are exactly the frames the report
/// lists as read, in the same order and each once, and exactly those frames
/// have `.sift` files.
fn assert_read_is_asked(run: &Run) {
    let read: Vec<FrameId> = run.report().frames.iter().map(|f| f.id).collect();
    assert_eq!(read, run.asked);
    let unique: BTreeSet<FrameId> = run.asked.iter().copied().collect();
    assert_eq!(unique.len(), run.asked.len(), "a frame was asked for twice");
    for (s, sequence) in run.sequences.iter().enumerate() {
        let folder = run.dir.path().join(format!("seq{s}"));
        let expected: Vec<_> = run
            .asked
            .iter()
            .filter(|id| id.sequence == s)
            .flat_map(|id| {
                sequence.frames[id.frame]
                    .images
                    .iter()
                    .map(|i| i.sift_path.clone())
            })
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect();
        assert_eq!(sift_files(&folder), expected);
    }
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

#[test]
fn a_blurred_frame_is_swapped_for_a_sharp_neighbour() {
    // Blurred in every direction, and along one direction only, each a
    // frame the walk chose before it was blurred. The swap of the first
    // changes the walk after it, so the second is taken from that walk.
    let mut blurred = Synth::steady(3, 300, ONE_PERCENT);
    let soft = unsampled_choice(&run(&[&blurred]), 100);
    blurred.shots[soft].readings[0].0 = 0.3;
    let streaked = unsampled_choice(&run(&[&blurred]), 200);
    blurred.shots[streaked].readings[0].1 = [1.0, 0.2, 0.0];
    let run = run(&[&blurred]);
    let walk = run.walk(0);
    for frame in [soft, streaked] {
        assert!(!walk.chosen.contains(&frame), "{frame} was chosen");
        let swap = walk
            .swaps
            .iter()
            .find(|s| s.candidate == frame)
            .unwrap_or_else(|| panic!("{frame} was not swapped: {:?}", walk.swaps));
        assert!(swap.chosen.abs_diff(frame) <= 2);
        assert!(walk.chosen.contains(&swap.chosen));
    }
    let judged = |frame| {
        run.report()
            .frames
            .iter()
            .find(|f| f.id.frame == frame)
            .expect("the frame was read")
            .clone()
    };
    assert!(judged(soft).sensors[0].sharpness_ratio < 0.6);
    assert!(judged(streaked).sensors[0].axis_ratio_z.unwrap() > 2.0);
    assert!(judged(soft).blurred && judged(streaked).blurred);
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
fn a_spread_of_zero_is_floored_at_the_smallest_deviation() {
    let reading = |xx: f32| ImageReadings::new(1.0, [xx, 1.0, 0.0]).unwrap();
    // Most readings equal the median, so the MAD is 0; the floor is 1.4826
    // times the smallest positive deviation, here 0.01 in the axis ratio.
    let readings: Vec<ImageReadings> = (0..10)
        .map(|i| reading(if i < 7 { 1.0 } else { 0.01f32.exp() }))
        .collect();
    let base = baseline::sensor_baseline(&readings);
    assert_eq!(base.axis_ratio_mad, 0.0);
    assert!((base.axis_ratio_spread - 1.4826 * 0.01).abs() < 1e-6);
    let judged = |xx: f32| baseline::judge(&reading(xx), &base, 0.6);
    // A deviation as small as the deviants' is not blurred; one well beyond
    // two spreads is.
    assert!(!judged(0.01f32.exp()).blurred);
    assert!(judged(0.05f32.exp()).blurred);
    assert!(judged(0.05f32.exp()).axis_ratio_z.unwrap() > 2.0);

    // Every reading the same: no spread, so no z-score and no blur by it.
    let same = baseline::sensor_baseline(&[reading(1.0); 10]);
    assert_eq!(same.axis_ratio_spread, 0.0);
    let streak = baseline::judge(&reading(5.0), &same, 0.6);
    assert_eq!(streak.axis_ratio_z, None);
    assert!(!streak.blurred);
}

#[test]
fn readings_mostly_at_the_median_do_not_flag_the_rest() {
    // Three frames in five read exactly [1, 1, 0] and two a tensor 2% apart:
    // with no floor under a MAD of 0, those two in five would all be blurred.
    let mut synth = Synth::steady(27, 300, ONE_PERCENT);
    for (t, shot) in synth.shots.iter_mut().enumerate() {
        shot.readings[0].1 = if t % 5 < 2 {
            [1.02, 1.0, 0.0]
        } else {
            [1.0, 1.0, 0.0]
        };
    }
    let run = run(&[&synth]);
    let report = run.report();
    assert_eq!(report.sequences[0].baseline.sensors[0].axis_ratio_mad, 0.0);
    assert_eq!(report.sequences[0].baseline.blurred_samples, 0);
    assert!(report.frames.iter().all(|f| !f.blurred));
    let walk = run.walk(0);
    assert!(walk.swaps.is_empty() && walk.kept_blurred.is_empty());
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
    // Nothing is read after the first batch.
    assert!(report.frames.iter().all(|f| f.stage == Stage::BaselineNear));
    assert_eq!(run.calls.len(), 1);
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
    let mut steady = Synth::steady(12, 300, ONE_PERCENT);
    steady.sensors = 2;
    let first = run(&[&steady]);
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
    // blurred, and it is swapped.
    let frame = unsampled_choice(&first, 120);
    steady.shots[frame].readings[1] = (1.0, sharp(frame, 1).1);
    let run = run(&[&steady]);
    let walk = run.walk(0);
    assert!(!walk.chosen.contains(&frame));
    assert!(walk.swaps.iter().any(|s| s.candidate == frame));
    let read = run
        .report()
        .frames
        .iter()
        .find(|f| f.id.frame == frame)
        .unwrap();
    assert!(read.blurred && !read.sensors[0].blurred && read.sensors[1].blurred);
}

#[test]
fn the_same_capture_gives_the_same_report() {
    let synth = Synth::steady(13, 200, ONE_PERCENT);
    let a = run(&[&synth]);
    let b = run(&[&synth]);
    assert_eq!(a.report(), b.report());
    assert_eq!(a.calls, b.calls);
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
        assert!(run.calls.is_empty());
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
    let mut source = SynthSource::new(&synths, sequences);
    match pick_images(sequences, &mut source, &options(), &Progress::none()) {
        Err(PickImagesError::Sequence { reason }) => {
            assert!(source.calls.is_empty());
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
    let synths = [&synth];
    let mut source = SynthSource::new(&synths, &sequences);
    let err = pick_images(&sequences, &mut source, &options(), &Progress::none()).unwrap_err();
    match &err {
        PickImagesError::Version1Files(found) => {
            assert_eq!(found.files, vec![path.clone()]);
            assert!(err.to_string().contains("remove those files"));
        }
        other => panic!("{other:?}"),
    }
    assert!(source.calls.is_empty());
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
    let synths = [&synth];
    let mut source = SynthSource::new(&synths, &sequences);
    match pick_images(&sequences, &mut source, &options(), &Progress::none()) {
        Err(PickImagesError::ListFolder { folder, .. }) => assert_eq!(folder, blocker),
        other => panic!("{other:?}"),
    }
    assert!(source.calls.is_empty());
}

#[test]
fn the_sources_failure_is_returned() {
    let synth = Synth::steady(22, 300, ONE_PERCENT);
    for call in [0, 1, 2] {
        let run = run_with(&[&synth], &options(), Fault::FailOnCall(call), None);
        match error(&run) {
            PickImagesError::Source { message } => {
                assert_eq!(message, "the extractor ran out of disk");
            }
            other => panic!("{other:?}"),
        }
        assert_eq!(run.calls.len(), call + 1);
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
    let dir = tempfile::tempdir().unwrap();
    let sequences = vec![synth.sequence(dir.path())];
    struct Resized<'s>(SynthSource<'s>, &'s [CaptureSequence]);
    impl FrameSource for Resized<'_> {
        fn ensure_sift_files(
            &mut self,
            frames: &[FrameId],
            progress: &Progress,
        ) -> Result<(), FrameSourceError> {
            self.0.ensure_sift_files(frames, progress)?;
            // Frame 1, matched with frame 0 in the first batch, at another
            // size.
            let path = &self.1[0].frames[1].images[0].sift_path;
            capture::write_sift_file(path, &[[1.0, 1.0]], vec![0; 128], 320, 240, sharp(1, 0), 2);
            Ok(())
        }
    }
    let synths = [&synth];
    let mut source = Resized(SynthSource::new(&synths, &sequences), &sequences);
    match pick_images(&sequences, &mut source, &options(), &Progress::none()) {
        Err(PickImagesError::Pair { a, b, error, .. }) => {
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
    assert!(run.calls.is_empty());

    for call in [0, 1, 3] {
        let flag = AtomicBool::new(false);
        let run = run_with(
            &[&synth],
            &options(),
            Fault::CancelOnCall(call),
            Some(&flag),
        );
        assert!(matches!(error(&run), PickImagesError::Cancelled));
        assert_eq!(run.calls.len(), call + 1);
    }
}
