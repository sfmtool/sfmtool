// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! One frame of a value moving toward a goal: speeding up at a fixed rate,
//! capped at a top speed, and slowing so that it stops exactly at the goal.
//!
//! Shared by two motions in the viewer: **Maintain Z-up**, which turns the
//! viewport's `world_up` back to +Z with the angle left as the value
//! (`viewer_3d::righting`), and the camera model proposal's **Preview
//! retriangulation**, which runs the blend `s` between a point's stored
//! position and its proposed one
//! (`crate::camera_model_proposal::preview::PreviewMotion`). See
//! `specs/gui/viewport-navigation.md` § "Maintain Z-up" for the speed profile.
//! The velocity is the only state carried between frames, and it carries a
//! sign, so a goal that flips while the value moves turns it round without a
//! jump.

/// The longest frame time one step integrates over, in seconds. A stalled
/// frame then moves the value by no more than a normal frame at this rate.
pub(crate) const MAX_DT: f64 = 0.05;

/// The speed that, slowing down at `acceleration` one frame of `dt` at a time,
/// stops after covering exactly `distance`.
///
/// The continuous answer is `sqrt(2·a·distance)`. Stepping it frame by frame
/// arrives with a speed of about `sqrt(2·a²·dt²)` left over, which is then
/// dropped to zero in one frame; this form accounts for the steps, so the last
/// step before the goal is taken at about `a·dt`, the same speed the first step
/// of a move is taken at.
fn stopping_speed(distance: f64, dt: f64, acceleration: f64) -> f64 {
    let a_dt = acceleration * dt;
    (2.0 * acceleration * distance + 0.25 * a_dt * a_dt).sqrt() - 0.5 * a_dt
}

/// One frame of moving `value` toward `goal`: the new value and the velocity to
/// carry into the next frame.
///
/// `velocity` is the last frame's, signed, zero on the first frame of a move.
/// The step accelerates toward the goal at `acceleration`, caps the speed at
/// `max_speed`, and slows so that the value stops exactly at the goal, which it
/// then returns with a velocity of zero. When the velocity points away from the
/// goal it first slows through zero at `acceleration` before turning, so the
/// value never jumps. `dt` is clamped to [`MAX_DT`].
pub(crate) fn step(
    value: f64,
    velocity: f64,
    goal: f64,
    dt: f64,
    max_speed: f64,
    acceleration: f64,
) -> (f64, f64) {
    let distance = goal - value;
    if distance == 0.0 && velocity == 0.0 {
        return (goal, 0.0);
    }
    let dt = dt.clamp(0.0, MAX_DT);
    // The direction toward the goal, and the speed along it: negative while
    // the value is still moving away.
    let toward = if distance >= 0.0 { 1.0 } else { -1.0 };
    let along = velocity * toward;
    if along < 0.0 {
        // Moving away: slow through zero first, at the same rate, without
        // reaching for the goal in this frame.
        let next = (along + acceleration * dt).min(0.0);
        return (value + toward * next * dt, toward * next);
    }
    let remaining = distance.abs();
    let next = (along + acceleration * dt)
        .min(stopping_speed(remaining, dt, acceleration))
        .min(max_speed);
    let moved = next * dt;
    if moved >= remaining {
        return (goal, 0.0);
    }
    (value + toward * moved, toward * next)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The preview's constants, which the draft's timings are stated for.
    const PREVIEW_MAX: f64 = 2.0;
    const PREVIEW_ACCELERATION: f64 = 10.0;

    /// Every value `step` passes through from `from` to `goal` at a fixed
    /// frame time, the arrival included.
    fn run(from: f64, goal: f64, dt: f64) -> Vec<f64> {
        let (mut value, mut velocity) = (from, 0.0);
        let mut values = vec![value];
        while value != goal || velocity != 0.0 {
            (value, velocity) = step(value, velocity, goal, dt, PREVIEW_MAX, PREVIEW_ACCELERATION);
            values.push(value);
            assert!(values.len() < 10_000, "the move never arrived");
        }
        values
    }

    /// A full move of the preview's blend, 0 to 1, takes 0.7 s: two 0.2 s
    /// ramps at 10/s² and 0.3 s at the 2/s cap.
    #[test]
    fn a_full_preview_move_takes_about_seven_tenths_of_a_second() {
        let dt = 1.0 / 60.0;
        let values = run(0.0, 1.0, dt);
        let seconds = (values.len() - 1) as f64 * dt;
        assert!((0.65..=0.75).contains(&seconds), "took {seconds} s");
    }

    /// The value reaches the goal and stops there, never past it.
    #[test]
    fn the_value_stops_at_the_goal_without_overshoot() {
        for goal in [1.0, 0.0, 0.3] {
            let from = 1.0 - goal;
            let values = run(from, goal, 1.0 / 60.0);
            let (low, high) = (from.min(goal), from.max(goal));
            assert!(values.iter().all(|v| (low..=high).contains(v)));
            assert_eq!(*values.last().unwrap(), goal);
        }
    }

    /// A goal that flips mid-motion turns the value round without a jump: no
    /// frame moves it further than the top speed allows, and its speed changes
    /// by no more than the acceleration allows.
    #[test]
    fn a_goal_flipped_mid_motion_reverses_without_a_jump() {
        let dt = 1.0 / 60.0;
        let (mut value, mut velocity) = (0.0, 0.0);
        let mut values = vec![value];
        for frame in 0..200 {
            let goal = if frame < 20 { 1.0 } else { 0.0 };
            (value, velocity) = step(value, velocity, goal, dt, PREVIEW_MAX, PREVIEW_ACCELERATION);
            values.push(value);
        }
        assert_eq!(value, 0.0);
        let speeds: Vec<f64> = values.windows(2).map(|w| (w[1] - w[0]) / dt).collect();
        assert!(speeds.iter().all(|s| s.abs() <= PREVIEW_MAX + 1e-9));
        for pair in speeds.windows(2) {
            assert!(
                (pair[1] - pair[0]).abs() <= PREVIEW_ACCELERATION * dt * 1.2,
                "the speed jumped from {} to {}",
                pair[0],
                pair[1]
            );
        }
        // It went past where it turned: the reversal slowed through zero.
        let turned_at = values[20];
        assert!(values.iter().any(|&v| v > turned_at));
    }

    /// At rest on the goal, nothing moves.
    #[test]
    fn at_the_goal_it_stays() {
        assert_eq!(step(1.0, 0.0, 1.0, 1.0 / 60.0, 2.0, 10.0), (1.0, 0.0));
    }
}
