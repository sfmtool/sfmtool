// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Turning the viewport's `world_up` back to +Z while **Maintain Z-up** is on.
//!
//! [`step`] picks each frame's turning speed from the angle left and the last
//! frame's speed, so the speed is the only state carried between frames. The
//! speed profile, the timings [`MAX_SPEED`] and [`ACCELERATION`] give, and the
//! two cases that need a choice of how to turn are in
//! `specs/gui/viewport-navigation.md` § "Maintain Z-up".

use nalgebra::{Unit, UnitQuaternion, Vector3};

/// The fastest `world_up` turns, in radians per second.
pub(super) const MAX_SPEED: f64 = 4.0;

/// How fast the turning speed changes, in radians per second squared, both
/// speeding up and slowing down.
pub(super) const ACCELERATION: f64 = 20.0;

/// The longest frame time one step integrates over, in seconds. A stalled
/// frame then moves the view by no more than a normal frame at this rate.
const MAX_DT: f64 = 0.05;

/// An angle this small between `world_up` and +Z counts as level.
const LEVEL: f64 = 1e-6;

/// The speed that, slowing down at [`ACCELERATION`] one frame of `dt` at a
/// time, stops after turning exactly `angle`.
///
/// The continuous answer is `sqrt(2·a·angle)`. Stepping it frame by frame
/// arrives with a speed of about `sqrt(2·a²·dt²)` left over, which is then
/// dropped to zero in one frame; this form accounts for the steps, so the
/// last step before level is taken at about `a·dt`, the same speed the first
/// step of a turn is taken at.
fn stopping_speed(angle: f64, dt: f64) -> f64 {
    let a_dt = ACCELERATION * dt;
    (2.0 * ACCELERATION * angle + 0.25 * a_dt * a_dt).sqrt() - 0.5 * a_dt
}

/// One frame of turning `up` toward +Z: the new `world_up` and the speed to
/// carry into the next frame.
///
/// `up` turns along the great circle to +Z, which is the smallest rotation
/// that levels it. When `up` points almost straight down that circle is
/// undefined, and the turn is taken about `forward`, the view direction, so
/// that the view rolls over rather than pitching. `speed` is the speed the
/// last frame turned at, zero on the first frame of a turn.
///
/// Returns +Z and a speed of zero once `up` is level.
pub(super) fn step(
    up: Vector3<f64>,
    forward: Vector3<f64>,
    speed: f64,
    dt: f64,
) -> (Vector3<f64>, f64) {
    let z = Vector3::z();
    let angle = up.angle(&z);
    if angle <= LEVEL {
        return (z, 0.0);
    }
    let dt = dt.clamp(0.0, MAX_DT);
    let next_speed = (speed.max(0.0) + ACCELERATION * dt)
        .min(stopping_speed(angle, dt))
        .min(MAX_SPEED);
    let turn = next_speed * dt;
    if turn >= angle {
        return (z, 0.0);
    }
    let axis = Unit::try_new(up.cross(&z), 1e-9)
        .or_else(|| Unit::try_new(forward - up * forward.dot(&up), 1e-9))
        .unwrap_or_else(|| Unit::new_normalize(up.cross(&Vector3::x())));
    let turned = UnitQuaternion::from_axis_angle(&axis, turn) * up;
    (turned.normalize(), next_speed)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// What stepping `up` to level at a fixed frame time looked like.
    struct Run {
        /// Seconds until `up` was +Z.
        seconds: f64,
        /// The speed of each frame, in order, ending with the zero of arrival.
        speeds: Vec<f64>,
        /// The angle to +Z after each frame.
        angles: Vec<f64>,
        /// The largest amount `up` moved off the plane perpendicular to
        /// `forward`, which is zero for a turn that is a pure roll.
        off_roll: f64,
    }

    fn run(up: Vector3<f64>, forward: Vector3<f64>, dt: f64) -> Run {
        let mut up = up;
        let mut speed = 0.0;
        let mut frames = 0;
        let mut speeds = Vec::new();
        let mut angles = Vec::new();
        let mut off_roll: f64 = 0.0;
        while up != Vector3::z() {
            (up, speed) = step(up, forward, speed, dt);
            frames += 1;
            speeds.push(speed);
            angles.push(up.angle(&Vector3::z()));
            off_roll = off_roll.max(up.dot(&forward).abs());
            assert!(frames < 10_000, "the turn never reached level");
        }
        Run {
            seconds: frames as f64 * dt,
            speeds,
            angles,
            off_roll,
        }
    }

    #[test]
    fn upside_down_levels_in_about_a_second() {
        let run = run(-Vector3::z(), Vector3::y(), 1.0 / 60.0);
        assert!(
            (0.95..=1.05).contains(&run.seconds),
            "took {} s",
            run.seconds
        );
    }

    #[test]
    fn a_quarter_turn_levels_in_a_little_over_half_a_second() {
        let run = run(Vector3::x(), Vector3::y(), 1.0 / 60.0);
        assert!(
            (0.55..=0.65).contains(&run.seconds),
            "took {} s",
            run.seconds
        );
    }

    #[test]
    fn upside_down_rolls_about_the_view_direction() {
        let run = run(-Vector3::z(), Vector3::y(), 1.0 / 60.0);
        assert!(
            run.off_roll < 1e-9,
            "up left the roll plane by {}",
            run.off_roll
        );
    }

    #[test]
    fn the_turn_starts_and_stops_without_a_jump_in_speed() {
        let dt = 1.0 / 60.0;
        let run = run(-Vector3::z(), Vector3::y(), dt);
        assert!(run.speeds.iter().all(|&s| s <= MAX_SPEED));
        assert!(run.speeds.contains(&MAX_SPEED));
        // How fast the view actually turned on each frame, from rest before
        // the first frame to rest after the last. The frame that arrives
        // reports a speed of zero but still turns the rest of the way, so the
        // angles are the measure, not the reported speeds.
        let mut angles = vec![std::f64::consts::PI];
        angles.extend(&run.angles);
        let mut turned = vec![0.0];
        turned.extend(angles.windows(2).map(|pair| (pair[0] - pair[1]) / dt));
        turned.push(0.0);
        for pair in turned.windows(2) {
            assert!(
                (pair[1] - pair[0]).abs() <= ACCELERATION * dt * 1.2,
                "the speed jumped from {} to {} rad/s",
                pair[0],
                pair[1]
            );
        }
    }

    #[test]
    fn the_angle_only_ever_shrinks() {
        let run = run(
            Vector3::new(0.3, -0.8, -0.5).normalize(),
            Vector3::y(),
            1.0 / 60.0,
        );
        assert!(run.angles.windows(2).all(|pair| pair[1] < pair[0]));
    }

    #[test]
    fn the_time_to_level_does_not_depend_on_the_frame_rate() {
        let up = -Vector3::z();
        let slow = run(up, Vector3::y(), 1.0 / 30.0).seconds;
        let fast = run(up, Vector3::y(), 1.0 / 144.0).seconds;
        assert!(
            (slow - fast).abs() < 0.05,
            "{slow} s at 30 Hz, {fast} s at 144 Hz"
        );
    }

    #[test]
    fn level_stays_level() {
        assert_eq!(
            step(Vector3::z(), Vector3::y(), 0.0, 1.0 / 60.0),
            (Vector3::z(), 0.0)
        );
    }
}
