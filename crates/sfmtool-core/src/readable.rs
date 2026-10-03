// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! A number as a sentence written for a person prints it.
//!
//! Rust's `{}` and `{:.3}` print an `f64` in positional notation at any
//! magnitude, so a refusal that echoes a caller's `1e-300` spells out three
//! hundred zeros, and a label that reports a scale of `1e300` prints a
//! three-hundred-digit integer. [`Readable`] prints such a number in exponent
//! notation instead and every other number exactly as `{}` would, so it can
//! replace `{}` in a message without changing any ordinary one.
//!
//! ```
//! use sfmtool_core::readable::Readable;
//!
//! assert_eq!(format!("{}", Readable(46.5)), "46.5");
//! assert_eq!(format!("{:.3}", Readable(0.25)), "0.250");
//! assert_eq!(format!("{}", Readable(1e-300)), "1e-300");
//! assert_eq!(format!("{:.3}", Readable(1e300)), "1.000e300");
//! ```

use std::fmt;

/// At or above this magnitude a number prints in exponent notation, with or
/// without a precision: sixteen digits before the point is past what a person
/// reads at a glance, and past what an `f64` holds exactly.
const LARGE: f64 = 1e15;

/// Below this magnitude, and above zero, a number printed **without** a
/// precision prints in exponent notation, where `{}` would print the zeros
/// after the point. A number printed with a precision stays positional at any
/// small magnitude, since the precision already bounds its width (`1e-300`
/// at `{:.3}` is `0.000`).
const SMALL: f64 = 1e-5;

/// `f64` with a [`fmt::Display`] that switches to exponent notation where
/// positional notation would run to dozens of digits. See the
/// [module docs](self).
///
/// A precision is honoured in both notations: `{:.3}` gives three digits after
/// the point either way. Zero, NaN and the infinities print as `{}` prints them.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Readable(pub f64);

impl fmt::Display for Readable {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let x = self.0;
        let magnitude = x.abs();
        let exponent = x.is_finite()
            && (magnitude >= LARGE
                || (f.precision().is_none() && magnitude != 0.0 && magnitude < SMALL));
        match (exponent, f.precision()) {
            (true, Some(digits)) => write!(f, "{x:.digits$e}"),
            (true, None) => write!(f, "{x:e}"),
            (false, Some(digits)) => write!(f, "{x:.digits$}"),
            (false, None) => write!(f, "{x}"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::Readable;

    #[test]
    fn ordinary_numbers_print_as_display_prints_them() {
        for x in [0.0, -0.0, 1.0, 46.5, -3.25, 0.001, 123456.789, 1e14] {
            assert_eq!(format!("{}", Readable(x)), format!("{x}"));
            assert_eq!(format!("{:.3}", Readable(x)), format!("{x:.3}"));
        }
        for x in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert_eq!(format!("{}", Readable(x)), format!("{x}"));
        }
    }

    #[test]
    fn extreme_numbers_print_in_exponent_notation() {
        assert_eq!(format!("{}", Readable(1e-300)), "1e-300");
        assert_eq!(format!("{}", Readable(-2.5e-7)), "-2.5e-7");
        assert_eq!(format!("{}", Readable(1e300)), "1e300");
        assert_eq!(format!("{:.3}", Readable(1e300)), "1.000e300");
        assert_eq!(format!("{:.1}", Readable(-1e308)), "-1.0e308");
        // A precision bounds a small number's width already.
        assert_eq!(format!("{:.3}", Readable(1e-300)), "0.000");
        for x in [1e-300, 1e300, f64::MAX, f64::MIN_POSITIVE] {
            assert!(format!("{}", Readable(x)).len() < 30, "{x:e}");
            assert!(format!("{:.3}", Readable(x)).len() < 30, "{x:e}");
        }
    }
}
