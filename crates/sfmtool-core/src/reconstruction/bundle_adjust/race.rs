// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! The scanlines race with its early stop: the two solves of a camera whose
//! rolling-shutter scanlines are `UNKNOWN` run on two threads in lockstep, one
//! LM iteration of each at a time, and the trailing one is stopped once it
//! cannot plausibly catch up.
//!
//! The rule is [`EarlyStopRule`], a function of the values the two solves
//! report through their `Progress` (`KEPT_VALUE`, `NOISE_VALUE`, `COST_VALUE`)
//! and of nothing else, so its decisions never depend on thread timing. It is
//! the rule `scripts/scanlines_race_early_stop.py` replays, with its
//! definitions, and `specs/core/reconstruction/bundle-adjust-measurements.md`
//! measured. [`race`] runs the two solves and feeds the rule; blocking a solve
//! in its progress sink changes nothing it computes.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Condvar, Mutex, MutexGuard};
use std::time::Duration;

use crate::camera::RollingShutterScanlines;
use crate::geometry::bundle_adjust::{COST_VALUE, KEPT_VALUE, NOISE_VALUE};
use crate::progress::{Event, Level, Progress};
use crate::progress_note;

/// How the reconstruction-level adjustment stops the trailing solve of a
/// scanlines race early ([`super::BundleAdjustOptions::scanlines_early_stop`]).
///
/// After each pair of LM iterations, one of each solve, the gap is
/// `D = (C_trailing − C_leading)/σ̄²`, each `C` the cost the solve's damping
/// ladder compares and `σ̄²` the mean of the two solves' noise levels squared,
/// each measured at the start of the round. The trailing solve is stopped when,
/// for [`Self::count`] consecutive pairs with the same solve trailing, `D`
/// exceeds both [`Self::gap_threshold`] and what the trailing solve would
/// remove in the rest of its iteration budget at its recent rate.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ScanlinesEarlyStopOptions {
    /// Whether the rule runs. Off, the two solves run one after the other,
    /// each to the end of its schedule, and the final costs decide.
    pub enabled: bool,
    /// The gap `D`, in units of `σ̄²`, that the trailing solve must trail by.
    pub gap_threshold: f64,
    /// How many consecutive pairs the rule must hold for, with the same solve
    /// trailing.
    pub count: usize,
}

impl Default for ScanlinesEarlyStopOptions {
    /// On, with a threshold of 100 and a count of 3, the values the
    /// measurement recommended.
    fn default() -> Self {
        Self {
            enabled: true,
            gap_threshold: 100.0,
            count: 3,
        }
    }
}

/// How many of the trailing solve's last decreases the rate is the mean of.
const RATE_WINDOW: usize = 3;

/// The names of the phases the two solves of a race run in, rows first.
pub(super) const SOLVE_PHASES: [&str; 2] = ["solve with rows", "solve with columns"];

/// The two settings a race solves on, rows first.
pub(super) const SETTINGS: [RollingShutterScanlines; 2] = [
    RollingShutterScanlines::Rows,
    RollingShutterScanlines::Columns,
];

/// One schedule round of one solve, as its `Progress` reported it.
#[derive(Debug, Default, Clone, PartialEq)]
pub(super) struct RoundTrace {
    /// The observations its trim kept.
    pub kept: f64,
    /// Its noise level `σ_n`, px, measured at its start.
    pub noise: Option<f64>,
    /// The cost before its first iteration and after each, px².
    pub costs: Vec<f64>,
}

/// One solve of a race, as its `Progress` reported it.
#[derive(Debug, Default, Clone, PartialEq)]
pub(super) struct SolveTrace {
    /// Its rounds so far, in order.
    pub rounds: Vec<RoundTrace>,
    /// Whether it has returned, which ends its last round.
    pub finished: bool,
}

impl SolveTrace {
    /// Take in one value the solve reported. A round starts at its kept count,
    /// the first value it reports.
    pub fn record(&mut self, name: &str, value: f64) {
        match name {
            KEPT_VALUE => self.rounds.push(RoundTrace {
                kept: value,
                ..RoundTrace::default()
            }),
            NOISE_VALUE => {
                if let Some(round) = self.rounds.last_mut() {
                    round.noise = Some(value);
                }
            }
            COST_VALUE => {
                if let Some(round) = self.rounds.last_mut() {
                    round.costs.push(value);
                }
            }
            _ => {}
        }
    }

    /// Whether the solve has finished round `r` (counted from 0): it has
    /// started a later one, or returned.
    fn past(&self, r: usize) -> bool {
        self.rounds.len() > r + 1 || self.finished
    }
}

/// Where the rule stopped the trailing solve.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct StopPoint {
    /// The solve stopped: 0 for rows, 1 for columns.
    pub stopped: usize,
    /// The round of the pair it fired at, counted from 0.
    pub round: usize,
    /// The pair's iteration within the round, counted from 1.
    pub iteration: usize,
    /// The gap `D` at the pair, in units of `sigma2`.
    pub gap: f64,
    /// Each solve's cost at the pair, rows first, in units of `sigma2`.
    pub costs: [f64; 2],
    /// The observations both solves' rounds kept.
    pub kept: usize,
    /// The mean of the two solves' round-start noise levels squared, px².
    pub sigma2: f64,
}

/// The early-stopping rule, fed pair by pair in order.
///
/// The pairs of a round are `i = 0, 1, …` up to the longer of the two
/// solves' iteration counts in it; a solve that has finished the round keeps
/// its last cost. The pair at `i = 0`, a round's start, is skipped: it neither
/// counts nor breaks a run. A pair holds when both rounds kept the same count
/// of observations, both measured a noise level, the trailing solve has a
/// rate, `D > gap_threshold` and `D > rate × left`, `left` being the
/// iterations left in the round plus `max_iters` per later round, and the
/// trailing solve still has an iteration to run: a pair at which it has
/// finished its last round, and keeps its last cost, does not hold. Rounds are
/// compared while both solves have one; once either has none the rule has no
/// more pairs.
#[derive(Debug, Clone)]
pub(super) struct EarlyStopRule {
    options: ScanlinesEarlyStopOptions,
    max_iters: usize,
    rounds: usize,
    /// The next pair to judge, `(round, i)`.
    next: (usize, usize),
    /// How many consecutive pairs have held, and with which solve trailing.
    run: usize,
    trailing_before: Option<usize>,
}

impl EarlyStopRule {
    /// The rule for a schedule of `rounds` rounds (as the kernel runs it) at
    /// `max_iters` iterations per round.
    pub fn new(options: ScanlinesEarlyStopOptions, max_iters: usize, rounds: usize) -> Self {
        Self {
            options,
            max_iters,
            rounds,
            next: (0, 0),
            run: 0,
            trailing_before: None,
        }
    }

    /// Judge every pair `traces` now allow, in order, and return where the rule
    /// fires, if it does. A pair is judged once both solves have reported it,
    /// or finished its round. Called again after it has fired, it judges on.
    pub fn advance(&mut self, traces: &[SolveTrace; 2]) -> Option<StopPoint> {
        loop {
            let (r, i) = self.next;
            if r >= self.rounds {
                return None;
            }
            for trace in traces {
                if trace.rounds.len() <= r {
                    // A solve that returned without this round leaves no more
                    // pairs; one that has not reached it yet is waited for.
                    if trace.finished {
                        self.next = (self.rounds, 0);
                    }
                    return None;
                }
            }
            let has = |s: usize| traces[s].rounds[r].costs.len() > i;
            if !(0..2).all(|s| has(s) || traces[s].past(r)) {
                return None;
            }
            if !(has(0) || has(1)) {
                self.next = (r + 1, 0);
                continue;
            }
            self.next = (r, i + 1);
            if i == 0 {
                continue;
            }
            if let Some(stop) = self.judge(traces, r, i) {
                return Some(stop);
            }
        }
    }

    /// Judge pair `i ≥ 1` of round `r`.
    fn judge(&mut self, traces: &[SolveTrace; 2], r: usize, i: usize) -> Option<StopPoint> {
        let rounds = [&traces[0].rounds[r], &traces[1].rounds[r]];
        let evidence = (|| {
            if rounds.iter().any(|round| round.costs.is_empty()) {
                return None;
            }
            let sigma2 = 0.5 * (rounds[0].noise?.powi(2) + rounds[1].noise?.powi(2));
            if sigma2.is_nan() || sigma2 <= 0.0 {
                return None;
            }
            let at = |s: usize| {
                let costs = &rounds[s].costs;
                costs[i.min(costs.len() - 1)] / sigma2
            };
            let c = [at(0), at(1)];
            let trailing = if c[0] > c[1] { 0 } else { 1 };
            let gap = (c[0] - c[1]).abs();
            let rate = rate(&rounds[trailing].costs, i, sigma2)?;
            let left = self.max_iters.saturating_sub(i) + self.max_iters * (self.rounds - 1 - r);
            let holds = !spent(&traces[trailing], r, i)
                && rounds[0].kept == rounds[1].kept
                && gap > self.options.gap_threshold
                && gap > rate * left as f64;
            holds.then_some((trailing, gap, c, sigma2))
        })();
        match evidence {
            Some((trailing, gap, costs, sigma2)) => {
                self.run = if self.trailing_before == Some(trailing) {
                    self.run + 1
                } else {
                    1
                };
                self.trailing_before = Some(trailing);
                (self.run >= self.options.count).then_some(StopPoint {
                    stopped: trailing,
                    round: r,
                    iteration: i,
                    gap,
                    costs,
                    kept: rounds[0].kept as usize,
                    sigma2,
                })
            }
            None => {
                self.run = 0;
                self.trailing_before = None;
                None
            }
        }
    }
}

/// Whether `trace` has no iteration left to run at pair `i` of round `r`: it
/// returned with `r` its last round, having finished that round before
/// iteration `i`. Stopping it there would save nothing, and the race reads it
/// only once it has returned, so a stop there would come after its thread had
/// ended.
fn spent(trace: &SolveTrace, r: usize, i: usize) -> bool {
    trace.finished && trace.rounds.len() == r + 1 && trace.rounds[r].costs.len() <= i
}

/// The mean of the last [`RATE_WINDOW`] decreases of `costs` up to step `i`
/// (or its last step, where it ended sooner), in units of `sigma2`; `None`
/// with no step yet.
fn rate(costs: &[f64], i: usize, sigma2: f64) -> Option<f64> {
    let i = i.min(costs.len().saturating_sub(1));
    let first = (i + 1).saturating_sub(RATE_WINDOW).max(1);
    let steps: Vec<f64> = (first..=i)
        .map(|k| (costs[k - 1] - costs[k]) / sigma2)
        .collect();
    (!steps.is_empty()).then(|| steps.iter().sum::<f64>() / steps.len() as f64)
}

/// An [`Event`] kept until it can be passed on.
enum Held {
    Enter(&'static str, u8),
    Leave(&'static str, u8, Duration, Option<String>),
    Message(Level, u8, String),
    Status(String),
    Count(u64, Option<u64>, &'static str),
    Fraction(f32),
    Value(&'static str, f64),
}

impl Held {
    fn of(event: Event<'_>) -> Self {
        match event {
            Event::Enter { phase, depth } => Held::Enter(phase, depth),
            Event::Leave {
                phase,
                depth,
                took,
                note,
            } => Held::Leave(phase, depth, took, note.map(str::to_string)),
            Event::Message { level, depth, text } => Held::Message(level, depth, text.to_string()),
            Event::Status { text } => Held::Status(text.to_string()),
            Event::Count { done, total, unit } => Held::Count(done, total, unit),
            Event::Fraction { of_whole } => Held::Fraction(of_whole),
            Event::Value { name, value } => Held::Value(name, value),
        }
    }

    fn forward(&self, to: &Progress<'_>) {
        to.forward(match self {
            Held::Enter(phase, depth) => Event::Enter {
                phase,
                depth: *depth,
            },
            Held::Leave(phase, depth, took, note) => Event::Leave {
                phase,
                depth: *depth,
                took: *took,
                note: note.as_deref(),
            },
            Held::Message(level, depth, text) => Event::Message {
                level: *level,
                depth: *depth,
                text,
            },
            Held::Status(text) => Event::Status { text },
            Held::Count(done, total, unit) => Event::Count {
                done: *done,
                total: *total,
                unit,
            },
            Held::Fraction(of_whole) => Event::Fraction {
                of_whole: *of_whole,
            },
            Held::Value(name, value) => Event::Value {
                name,
                value: *value,
            },
        });
    }
}

/// What the two threads of a race share.
struct Shared {
    traces: [SolveTrace; 2],
    rule: EarlyStopRule,
    stop: Option<StopPoint>,
    /// The caller asked the operation to stop.
    cancelled: bool,
}

struct Race<'p> {
    shared: Mutex<Shared>,
    changed: Condvar,
    /// Each solve's own cancel flag: set for the stopped solve, and for both
    /// when the caller cancels.
    flags: [AtomicBool; 2],
    /// The caller's `Progress` for this race.
    part: Progress<'p>,
    /// The columns solve's events, held until the rows solve has returned so
    /// that a collector reading one phase at a time at each depth sees the
    /// rows solve's phase whole, then the columns solve's; `None` once they
    /// are passed on as they come.
    held: Mutex<Option<Vec<Held>>>,
}

impl Race<'_> {
    fn lock(&self) -> MutexGuard<'_, Shared> {
        self.shared.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Judge what the traces now allow, stopping the trailing solve where the
    /// rule fires, and wake every waiting solve.
    fn changed(&self, shared: &mut Shared) {
        // Once the caller has cancelled, the traces stop where each solve
        // happened to be, so no further pair is judged.
        if shared.stop.is_none() && !shared.cancelled {
            if let Some(stop) = shared.rule.advance(&shared.traces) {
                self.flags[stop.stopped].store(true, Ordering::Relaxed);
                shared.stop = Some(stop);
            }
        }
        self.changed.notify_all();
    }

    /// What solve `s`'s `Progress` reports goes through here.
    fn tap(&self, s: usize, event: Event<'_>) {
        if s == 0 {
            self.part.forward(event);
        } else {
            let mut held = self.held.lock().unwrap_or_else(|e| e.into_inner());
            match held.as_mut() {
                Some(events) => events.push(Held::of(event)),
                None => self.part.forward(event),
            }
        }
        let mut shared = self.lock();
        if self.part.is_cancelled() && !shared.cancelled {
            shared.cancelled = true;
            for flag in &self.flags {
                flag.store(true, Ordering::Relaxed);
            }
            self.changed.notify_all();
        }
        let Event::Value { name, value } = event else {
            return;
        };
        if shared.stop.is_some() || shared.cancelled {
            return;
        }
        shared.traces[s].record(name, value);
        self.changed(&mut shared);
        let other = 1 - s;
        // Where this solve now stands, and what it waits for: at a round's
        // start until the other has finished the round before, and after each
        // of its iterations until the other has reported the same iteration
        // or finished the round.
        let r = shared.traces[s].rounds.len().saturating_sub(1);
        let ready: Box<dyn Fn(&Shared) -> bool> = match name {
            KEPT_VALUE if r > 0 => Box::new(move |sh: &Shared| sh.traces[other].past(r - 1)),
            COST_VALUE => {
                let i = shared.traces[s].rounds[r].costs.len() - 1;
                if i == 0 {
                    return;
                }
                Box::new(move |sh: &Shared| {
                    let t = &sh.traces[other];
                    t.past(r) || t.rounds.get(r).is_some_and(|round| round.costs.len() > i)
                })
            }
            _ => return,
        };
        while !(shared.stop.is_some() || shared.cancelled || ready(&shared)) {
            shared = self.changed.wait(shared).unwrap_or_else(|e| e.into_inner());
        }
    }

    /// Solve `s` has returned.
    fn finished(&self, s: usize) {
        let mut shared = self.lock();
        shared.traces[s].finished = true;
        self.changed(&mut shared);
    }

    /// The rows solve has returned and closed its phase: pass on what the
    /// columns solve said meanwhile, and the rest of it as it comes.
    fn release_held(&self) {
        let mut held = self.held.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(events) = held.take() {
            for event in &events {
                event.forward(&self.part);
            }
        }
    }
}

/// Marks a solve returned when its thread ends, a panic included, so the other
/// never waits on it.
struct Finish<'r, 'p>(&'r Race<'p>, usize);

impl Drop for Finish<'_, '_> {
    fn drop(&mut self) {
        self.0.finished(self.1);
    }
}

/// Run the solve with rows and the solve with columns of one race on two
/// threads in lockstep under `options`' rule, each in its phase of `part`, and
/// return both results, rows first, with where the trailing one was stopped.
///
/// `solve` runs one setting's solve and reports through the `Progress` it is
/// handed; it is the same call it would be alone, since the only thing the
/// race does to it is hold it in its progress sink and, for the trailing
/// solve, set its cancel flag. A stopped solve's result is what it reached and
/// is the caller's to discard. `rounds` is how many rounds each solve runs
/// ([`crate::geometry::bundle_adjust::rounds_run`]). A cancellation of `part`
/// stops both.
///
/// Two hazards for a caller. Each solve sees `part`'s cancel flag only through
/// its tap, which passes it on to the solve's own flag at the solve's next
/// event, so a cancellation lands at most one event late. And the race blocks
/// threads: each solve in its progress sink while it waits for the other, and
/// the calling thread, which runs the rows solve, until the columns solve's
/// own `std` thread returns. The solves' parallel work runs on the rayon pool,
/// so called from the only worker of a one-thread pool the race would
/// deadlock, that worker waiting on solves whose work waits on it. No caller
/// runs the race on a pool worker: the viewer calls the adjustment on a `std`
/// thread, the Python binding on the calling thread with the interpreter
/// released.
pub(super) fn race<T: Send>(
    part: &Progress<'_>,
    options: ScanlinesEarlyStopOptions,
    max_iters: usize,
    rounds: usize,
    solve: impl Fn(RollingShutterScanlines, &Progress<'_>) -> T + Sync,
) -> ([T; 2], Option<StopPoint>) {
    let race = Race {
        shared: Mutex::new(Shared {
            traces: Default::default(),
            rule: EarlyStopRule::new(options, max_iters, rounds),
            stop: None,
            cancelled: false,
        }),
        changed: Condvar::new(),
        flags: [AtomicBool::new(false), AtomicBool::new(false)],
        part: *part,
        held: Mutex::new(Some(Vec::new())),
    };
    let race = &race;
    let solve = &solve;
    let run = move |s: usize| {
        let tap = move |event: Event<'_>| race.tap(s, event);
        let tapped = race.part.tapped(&tap).cancelled_by(&race.flags[s]);
        let mut phase = tapped.phase(SOLVE_PHASES[s]);
        let result = {
            let _finish = Finish(race, s);
            solve(SETTINGS[s], &phase)
        };
        // The rule judges a pair only while the trailing solve still has an
        // iteration to run, which it waits for in its progress sink, so a stop
        // of this solve was set before it left its last wait, and reading it
        // here gives the same answer whatever the other thread is doing. A
        // cancelled race notes no stop: it is refused, and where each solve
        // was when the cancellation reached it is a matter of timing.
        let stop = {
            let shared = race.lock();
            let cancelled = shared.cancelled || race.part.is_cancelled();
            shared.stop.filter(|stop| stop.stopped == s && !cancelled)
        };
        if let Some(stop) = stop {
            progress_note!(
                phase,
                "stopped at round {}, iteration {}",
                stop.round + 1,
                stop.iteration
            );
        }
        drop(phase);
        if s == 0 {
            race.release_held();
        }
        result
    };
    let (rows, columns) = std::thread::scope(|scope| {
        let columns = scope.spawn(move || run(1));
        let rows = run(0);
        (
            rows,
            columns
                .join()
                .unwrap_or_else(|e| std::panic::resume_unwind(e)),
        )
    });
    // A panic in the rows solve leaves the columns events held; nothing is
    // left to read them then, so there is nothing to release here.
    let stop = race.lock().stop;
    ([rows, columns], stop)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A round that kept `kept` observations at noise `noise` px, with `costs`.
    fn round(kept: f64, noise: f64, costs: &[f64]) -> RoundTrace {
        RoundTrace {
            kept,
            noise: Some(noise),
            costs: costs.to_vec(),
        }
    }

    fn finished(rounds: Vec<RoundTrace>) -> SolveTrace {
        SolveTrace {
            rounds,
            finished: true,
        }
    }

    fn options(threshold: f64, count: usize) -> ScanlinesEarlyStopOptions {
        ScanlinesEarlyStopOptions {
            enabled: true,
            gap_threshold: threshold,
            count,
        }
    }

    /// Two one-round solves at noise 1 px over 500 kept observations: rows
    /// at 1000 after a first step from 2000, columns falling from 1500 by
    /// `step` an iteration, so its rate is `step` throughout.
    fn flat_against_falling(step: f64, iterations: usize) -> [SolveTrace; 2] {
        let mut rows = vec![2000.0];
        rows.extend(std::iter::repeat_n(1000.0, iterations));
        let columns: Vec<f64> = (0..=iterations)
            .map(|k| 1500.0 + step - step * k as f64)
            .collect();
        [
            finished(vec![round(500.0, 1.0, &rows)]),
            finished(vec![round(500.0, 1.0, &columns)]),
        ]
    }

    #[test]
    fn the_rule_fires_on_the_count_th_pair_that_holds() {
        // Columns trails by 500 from the first iteration at a rate of
        // nothing, so every pair holds and the third is iteration 3.
        let traces = flat_against_falling(0.0, 6);
        let stop = EarlyStopRule::new(options(100.0, 3), 6, 1)
            .advance(&traces)
            .expect("fires");
        assert_eq!((stop.stopped, stop.round, stop.iteration), (1, 0, 3));
        assert_eq!(stop.gap, 500.0);
        assert_eq!(stop.costs, [1000.0, 1500.0]);
        assert_eq!((stop.kept, stop.sigma2), (500, 1.0));
        // A count of 1 fires at the first pair that holds.
        let stop = EarlyStopRule::new(options(100.0, 1), 6, 1)
            .advance(&traces)
            .unwrap();
        assert_eq!(stop.iteration, 1);
    }

    #[test]
    fn the_threshold_holds_the_rule_off() {
        let traces = flat_against_falling(0.0, 8);
        assert!(EarlyStopRule::new(options(500.0, 3), 8, 1)
            .advance(&traces)
            .is_none());
        assert!(EarlyStopRule::new(options(499.0, 3), 8, 1)
            .advance(&traces)
            .is_some());
    }

    #[test]
    fn the_budget_term_holds_the_rule_off_while_the_trailer_falls_fast() {
        // Columns falls by 20 an iteration and trails by 520 − 20·i. With 60
        // iterations a round, 20·(60 − i) exceeds the gap at every pair.
        let traces = flat_against_falling(20.0, 10);
        assert!(EarlyStopRule::new(options(100.0, 3), 60, 1)
            .advance(&traces)
            .is_none());
        // With 10, 20·(10 − i) is under it at every pair.
        let stop = EarlyStopRule::new(options(100.0, 3), 10, 1)
            .advance(&traces)
            .expect("fires");
        assert_eq!(stop.iteration, 3);
        // Two more rounds in the schedule add 10 iterations each to every
        // budget, which then exceeds the gap again.
        assert!(EarlyStopRule::new(options(100.0, 3), 10, 3)
            .advance(&traces)
            .is_none());
    }

    #[test]
    fn a_change_of_trailing_solve_restarts_the_count() {
        // Columns trails at iterations 1 and 2, rows at 3, 4 and 5.
        let rows = [1000.0, 1000.0, 1000.0, 1600.0, 1600.0, 1600.0];
        let columns = [1500.0, 1500.0, 1500.0, 1000.0, 1000.0, 1000.0];
        let traces = [
            finished(vec![round(500.0, 1.0, &rows)]),
            finished(vec![round(500.0, 1.0, &columns)]),
        ];
        let stop = EarlyStopRule::new(options(100.0, 3), 5, 1)
            .advance(&traces)
            .expect("fires");
        assert_eq!((stop.stopped, stop.iteration), (0, 5));
    }

    #[test]
    fn a_rounds_start_is_skipped_and_neither_counts_nor_breaks_a_run() {
        // Two rounds of two iterations each: columns trails at both pairs of
        // round 1 and at the first of round 2, so the third holding pair is
        // round 2's first iteration, the round-start pair between them
        // skipped although rows trails by a jump there.
        let rows = [
            round(500.0, 1.0, &[2000.0, 1000.0, 1000.0]),
            round(500.0, 1.0, &[90000.0, 1000.0, 1000.0]),
        ];
        let columns = [
            round(500.0, 1.0, &[1500.0, 1500.0, 1500.0]),
            round(500.0, 1.0, &[1500.0, 1500.0, 1500.0]),
        ];
        let traces = [finished(rows.to_vec()), finished(columns.to_vec())];
        let stop = EarlyStopRule::new(options(100.0, 3), 2, 2)
            .advance(&traces)
            .expect("fires");
        assert_eq!((stop.stopped, stop.round, stop.iteration), (1, 1, 1));
    }

    #[test]
    fn no_stop_while_the_two_rounds_kept_different_counts() {
        let mut traces = flat_against_falling(0.0, 8);
        traces[1].rounds[0].kept = 499.0;
        assert!(EarlyStopRule::new(options(100.0, 3), 8, 1)
            .advance(&traces)
            .is_none());
    }

    #[test]
    fn a_solve_that_finished_its_round_keeps_its_last_cost() {
        // Rows converges after two iterations; columns runs on, trailing.
        let traces = [
            finished(vec![round(500.0, 1.0, &[2000.0, 1000.0, 1000.0])]),
            finished(vec![round(500.0, 1.0, &[1500.0, 1500.0, 1500.0, 1500.0])]),
        ];
        let stop = EarlyStopRule::new(options(100.0, 3), 4, 1)
            .advance(&traces)
            .expect("fires");
        assert_eq!((stop.stopped, stop.iteration), (1, 3));
        assert_eq!(stop.costs, [1000.0, 1500.0]);
    }

    #[test]
    fn the_gap_is_in_units_of_the_mean_round_start_noise_squared() {
        // Noise 1 and 3 px: σ̄² = 5, so a cost difference of 500 is a gap of
        // 100, which does not exceed a threshold of 100.
        let mut traces = flat_against_falling(0.0, 8);
        traces[1].rounds[0].noise = Some(3.0);
        assert!(EarlyStopRule::new(options(100.0, 3), 8, 1)
            .advance(&traces)
            .is_none());
        let stop = EarlyStopRule::new(options(99.0, 3), 8, 1)
            .advance(&traces)
            .unwrap();
        assert_eq!((stop.gap, stop.sigma2), (100.0, 5.0));
    }

    #[test]
    fn the_rule_waits_for_both_solves_and_judges_each_pair_once() {
        // Fed value by value, in two different interleavings, the rule fires
        // at the same pair as on the finished traces.
        let whole = flat_against_falling(0.0, 6);
        let want = EarlyStopRule::new(options(100.0, 3), 6, 1).advance(&whole);
        for rows_first in [true, false] {
            let mut rule = EarlyStopRule::new(options(100.0, 3), 6, 1);
            let mut traces: [SolveTrace; 2] = Default::default();
            let mut got = None;
            for k in 0..7 {
                for s in if rows_first { [0, 1] } else { [1, 0] } {
                    if k == 0 {
                        traces[s].record(KEPT_VALUE, 500.0);
                        traces[s].record(NOISE_VALUE, 1.0);
                    }
                    traces[s].record(COST_VALUE, whole[s].rounds[0].costs[k]);
                    if let Some(stop) = rule.advance(&traces) {
                        got.get_or_insert(stop);
                    }
                }
            }
            assert_eq!(got, want, "rows first: {rows_first}");
        }
    }

    /// One scripted solve: one round of 500 kept observations at noise 1 px,
    /// reporting `costs` (the start and one per iteration), stopping where it
    /// is cancelled, sleeping `delay` before each iteration. It returns the
    /// costs it reported.
    fn scripted(
        costs: &[f64],
        delay: Duration,
        after_cancel: Duration,
        progress: &Progress<'_>,
    ) -> Vec<f64> {
        progress.value(KEPT_VALUE, 500.0);
        progress.value(NOISE_VALUE, 1.0);
        let mut said = Vec::new();
        for (k, &cost) in costs.iter().enumerate() {
            if k > 0 {
                if progress.is_cancelled() {
                    std::thread::sleep(after_cancel);
                    break;
                }
                std::thread::sleep(delay);
            }
            progress.value(COST_VALUE, cost);
            said.push(cost);
        }
        said
    }

    /// What [`scripted_race`] returns: each solve's costs, the stop, and the
    /// notes the phases closed with.
    type Scripted = (
        [Vec<f64>; 2],
        Option<StopPoint>,
        Vec<(&'static str, Option<String>)>,
    );

    /// The race of two scripted solves, one round of `max_iters` 5, the rule
    /// at a threshold of 100 and a count of 3, with one solve delayed.
    fn scripted_race(rows: &[f64], columns: &[f64], delayed: usize) -> Scripted {
        let notes = Mutex::new(Vec::new());
        let sink = |event: Event<'_>| {
            if let Event::Leave { phase, note, .. } = event {
                notes
                    .lock()
                    .unwrap()
                    .push((phase, note.map(str::to_string)));
            }
        };
        let part = Progress::to(&sink);
        let (results, stop) = race(&part, options(100.0, 3), 5, 1, |setting, p| {
            let s = usize::from(setting == RollingShutterScanlines::Columns);
            let delay = Duration::from_millis(if s == delayed { 100 } else { 0 });
            let costs = if s == 0 { rows } else { columns };
            scripted(costs, delay, Duration::ZERO, p)
        });
        (results, stop, notes.into_inner().unwrap())
    }

    #[test]
    fn a_cancelled_race_judges_no_more_pairs_and_notes_no_stop() {
        // The caller cancels when the rows solve reports its third iteration,
        // the pair at which the rule would stop columns. Rows runs each
        // iteration 50 ms behind, so columns has reported its third first and
        // the pair could be judged from the frozen traces once rows returns.
        // One solve then sleeps before it returns. Either way no stop is judged after the
        // cancellation, the call is cancelled, and the phases close alike.
        let rows = [2000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0];
        let columns = [1500.0; 6];
        let mut seen = Vec::new();
        for slow in [0, 1] {
            let flag = AtomicBool::new(false);
            let costs = std::sync::atomic::AtomicUsize::new(0);
            let notes = Mutex::new(Vec::new());
            // Only the rows solve's events arrive live: the columns solve's
            // are held until the rows solve has closed its phase.
            let sink = |event: Event<'_>| match event {
                Event::Value {
                    name: COST_VALUE, ..
                } => {
                    if costs.fetch_add(1, Ordering::Relaxed) + 1 == 4 {
                        flag.store(true, Ordering::Relaxed);
                    }
                }
                Event::Leave { phase, note, .. } => notes
                    .lock()
                    .unwrap()
                    .push((phase, note.map(str::to_string))),
                _ => {}
            };
            let part = Progress::to(&sink).cancelled_by(&flag);
            let (_, stop) = race(&part, options(100.0, 3), 5, 1, |setting, p| {
                let s = usize::from(setting == RollingShutterScanlines::Columns);
                let after = Duration::from_millis(if s == slow { 200 } else { 0 });
                let costs = if s == 0 { &rows[..] } else { &columns[..] };
                let delay = Duration::from_millis(if s == 0 { 50 } else { 0 });
                scripted(costs, delay, after, p)
            });
            assert!(part.is_cancelled(), "slow: {slow}");
            assert_eq!(stop, None, "slow: {slow}");
            seen.push(notes.into_inner().unwrap());
        }
        let want: Vec<(&str, Option<String>)> =
            vec![("solve with rows", None), ("solve with columns", None)];
        assert_eq!(seen, [want.clone(), want]);
    }

    #[test]
    fn a_trailer_that_has_finished_its_last_round_is_not_stopped() {
        // Columns trails by 500 at a rate of nothing and ends its only round
        // after two iterations; rows runs on. The third holding pair would be
        // iteration 3, where columns has nothing left to run, so the rule does
        // not fire, both solves run to the end, and whichever thread is slower
        // the answer and the phases' notes are the same.
        let rows = [2000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0];
        let columns = [1500.0, 1500.0, 1500.0];
        let traces = [
            finished(vec![round(500.0, 1.0, &rows)]),
            finished(vec![round(500.0, 1.0, &columns)]),
        ];
        assert_eq!(
            EarlyStopRule::new(options(100.0, 3), 5, 1).advance(&traces),
            None
        );
        let want = ([rows.to_vec(), columns.to_vec()], None);
        for delayed in [0, 1] {
            let (results, stop, notes) = scripted_race(&rows, &columns, delayed);
            assert_eq!((results, stop), want.clone(), "delayed: {delayed}");
            assert_eq!(
                notes,
                [("solve with rows", None), ("solve with columns", None)],
                "delayed: {delayed}"
            );
        }
    }

    #[test]
    fn a_stopped_trailer_is_stopped_and_noted_the_same_whichever_thread_is_slower() {
        // Columns would run five iterations; the rule stops it at the third,
        // and it reports nothing after.
        let rows = [2000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0];
        let columns = [1500.0; 6];
        for delayed in [0, 1] {
            let (results, stop, notes) = scripted_race(&rows, &columns, delayed);
            let stop = stop.expect("fires");
            assert_eq!((stop.stopped, stop.round, stop.iteration), (1, 0, 3));
            assert_eq!(results[0], rows.to_vec(), "delayed: {delayed}");
            assert_eq!(results[1], columns[..4].to_vec(), "delayed: {delayed}");
            assert_eq!(
                notes,
                [
                    ("solve with rows", None),
                    (
                        "solve with columns",
                        Some("stopped at round 1, iteration 3".to_string())
                    )
                ],
                "delayed: {delayed}"
            );
        }
    }

    #[test]
    fn the_rate_is_the_mean_of_the_last_three_decreases() {
        let costs = [10.0, 9.0, 7.0, 4.0, 0.0];
        assert_eq!(rate(&costs, 0, 1.0), None);
        assert_eq!(rate(&costs, 1, 1.0), Some(1.0));
        assert_eq!(rate(&costs, 2, 1.0), Some(1.5));
        assert_eq!(rate(&costs, 4, 1.0), Some(3.0));
        // Past the end it reads the last steps.
        assert_eq!(rate(&costs, 9, 2.0), Some(1.5));
    }
}
