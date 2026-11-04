// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! A slice of a named image list by position, Python's `[start:stop:step]`,
//! and the split of a `-r` or `--named-image-list` value into a list name and
//! a slice. The rules are in `specs/workspace/named-image-lists.md` §
//! "Slices".

use std::fmt;

use super::{is_named_image_list_name, NAME_RULE};

/// The form a slice takes, as an error about a malformed one shows it.
const SLICE_FORM: &str = "a slice is written NAME[start:stop] or NAME[start:stop:step]";

/// A slice of a named image list's images by position, Python's
/// `[start:stop:step]`. Each part is `None` when it was left empty, and the
/// step is never `Some(0)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct NamedImageListSlice {
    start: Option<i64>,
    stop: Option<i64>,
    step: Option<i64>,
}

/// Why the text of a slice, or a `-r` value holding one, cannot be read.
/// Every variant's message names the value as given.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NamedImageListSliceError {
    /// The value has no `[` and is not a valid list name.
    InvalidName { value: String },
    /// The value has a `[`, and the text before it is not a valid list name:
    /// empty, a range expression, or a name that breaks the name rule.
    NotOnAName { value: String, before: String },
    /// The slice's text is not `[start:stop]` or `[start:stop:step]`;
    /// `problem` says what is wrong with it.
    Malformed { value: String, problem: String },
    /// The step is 0.
    ZeroStep { value: String },
}

impl fmt::Display for NamedImageListSliceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidName { value } => {
                write!(
                    f,
                    "'{value}' is not a valid named image list name: {NAME_RULE}"
                )
            }
            Self::NotOnAName { value, before } => {
                write!(
                    f,
                    "'{value}' is not a valid slice of a named image list: a slice follows the \
                     name of a named image list, as in NAME[start:stop:step]; "
                )?;
                if before.is_empty() {
                    write!(f, "no name comes before the '['")
                } else if !before.chars().any(char::is_alphabetic) {
                    write!(
                        f,
                        "'{before}' is a range, not a name, and a list is narrowed by a range \
                         given beside its name (-r NAME -r RANGE)"
                    )
                } else {
                    write!(
                        f,
                        "'{before}' is not a valid named image list name: {NAME_RULE}"
                    )
                }
            }
            Self::Malformed { value, problem } => write!(
                f,
                "'{value}' is not a valid slice of a named image list: {problem}; {SLICE_FORM}"
            ),
            Self::ZeroStep { value } => write!(f, "'{value}': slice step cannot be zero"),
        }
    }
}

impl std::error::Error for NamedImageListSliceError {}

impl NamedImageListSlice {
    /// The slice `[start:stop:step]`; `ZeroStep` when `step` is `Some(0)`.
    pub fn new(
        start: Option<i64>,
        stop: Option<i64>,
        step: Option<i64>,
    ) -> Result<Self, NamedImageListSliceError> {
        let slice = Self { start, stop, step };
        if step == Some(0) {
            return Err(NamedImageListSliceError::ZeroStep {
                value: slice.to_string(),
            });
        }
        Ok(slice)
    }

    /// Read the bracketed text of a slice, `[` and `]` included, such as
    /// `[:30]` or `[::3]`. Errors name `text`.
    pub fn parse(text: &str) -> Result<Self, NamedImageListSliceError> {
        parse_brackets(text, text)
    }

    /// The part before the first `:`; `None` when it was left empty.
    pub fn start(&self) -> Option<i64> {
        self.start
    }

    /// The part between the first and second `:`; `None` when it was left
    /// empty.
    pub fn stop(&self) -> Option<i64> {
        self.stop
    }

    /// The part after the second `:`; `None` when it was left empty or not
    /// written. Never `Some(0)`.
    pub fn step(&self) -> Option<i64> {
        self.step
    }

    /// The indexes the slice keeps in an array of `len` items, in the
    /// slice's order: Python's `range(*slice(start, stop, step).indices(len))`.
    ///
    /// A negative `start` or `stop` counts from the end, a bound beyond
    /// either end is clamped to it, and a negative step walks backward from
    /// the last item by default.
    pub fn indices(&self, len: usize) -> Vec<usize> {
        // i128 holds every sum of an i64 bound and a usize length.
        let len = len as i128;
        let step = i128::from(self.step.unwrap_or(1));
        let (lower, upper) = if step < 0 { (-1, len - 1) } else { (0, len) };
        let bound = |part: Option<i64>, default: i128| match part {
            None => default,
            Some(value) => {
                let value = i128::from(value);
                let value = if value < 0 { value + len } else { value };
                value.clamp(lower, upper)
            }
        };
        let start = bound(self.start, if step < 0 { upper } else { lower });
        let stop = bound(self.stop, if step < 0 { lower } else { upper });
        let mut indices = Vec::new();
        let mut i = start;
        while (step > 0 && i < stop) || (step < 0 && i > stop) {
            // `lower <= i <= upper` and `i` is never -1 inside the loop,
            // so it is an index of the array.
            indices.push(i as usize);
            i += step;
        }
        indices
    }
}

impl fmt::Display for NamedImageListSlice {
    /// `[start:stop]`, or `[start:stop:step]` when the step was written, each
    /// part empty when it is `None`.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let part = |p: Option<i64>| p.map(|v| v.to_string()).unwrap_or_default();
        write!(f, "[{}:{}", part(self.start), part(self.stop))?;
        if let Some(step) = self.step {
            write!(f, ":{step}")?;
        }
        write!(f, "]")
    }
}

/// Split a `-r` or `--named-image-list` value into the list's name and its
/// slice: `walk1` is the name `walk1` and no slice, and `walk1[:30]` the
/// name `walk1` and the slice `[:30]`.
///
/// The text before the first `[` is the name, and must be a valid list name
/// ([`is_named_image_list_name`]); the text from the `[` to the end is the
/// slice. Errors name `value`.
pub fn split_named_image_list_value(
    value: &str,
) -> Result<(&str, Option<NamedImageListSlice>), NamedImageListSliceError> {
    let Some(open) = value.find('[') else {
        if !is_named_image_list_name(value) {
            return Err(NamedImageListSliceError::InvalidName {
                value: value.to_string(),
            });
        }
        return Ok((value, None));
    };
    let (name, brackets) = value.split_at(open);
    if !is_named_image_list_name(name) {
        return Err(NamedImageListSliceError::NotOnAName {
            value: value.to_string(),
            before: name.to_string(),
        });
    }
    Ok((name, Some(parse_brackets(brackets, value)?)))
}

/// Read `text`, which starts with `[`, as a slice; errors name `value`.
fn parse_brackets(
    text: &str,
    value: &str,
) -> Result<NamedImageListSlice, NamedImageListSliceError> {
    let malformed = |problem: String| NamedImageListSliceError::Malformed {
        value: value.to_string(),
        problem,
    };
    let Some(rest) = text.strip_prefix('[') else {
        return Err(malformed("it does not start with '['".to_string()));
    };
    let Some(close) = rest.find(']') else {
        return Err(malformed("it has no closing ']'".to_string()));
    };
    let (inner, after) = (&rest[..close], &rest[close + 1..]);
    if !after.is_empty() {
        return Err(malformed(format!(
            "'{after}' follows the ']', and a value takes one slice only"
        )));
    }
    if inner.chars().any(char::is_whitespace) {
        return Err(malformed(
            "it holds whitespace, and a slice is written without spaces".to_string(),
        ));
    }
    if inner.is_empty() {
        return Err(malformed("the brackets are empty".to_string()));
    }
    let parts: Vec<&str> = inner.split(':').collect();
    if parts.len() == 1 {
        // The slice that keeps the one image Python's `a[i]` would give.
        let one = match inner.parse::<i64>() {
            Ok(-1) => "; [-1:] is the one image at index -1".to_string(),
            Ok(i) => match i.checked_add(1) {
                Some(next) => format!("; [{i}:{next}] is the one image at index {i}"),
                None => String::new(),
            },
            Err(_) => String::new(),
        };
        return Err(malformed(format!(
            "[{inner}] is a single index, which is not accepted, since a slice keeps a list of \
             images{one}"
        )));
    }
    if parts.len() > 3 {
        return Err(malformed("it has more than two ':'".to_string()));
    }
    let mut numbers = [None; 3];
    for (slot, (part, what)) in numbers
        .iter_mut()
        .zip(parts.iter().zip(["start", "stop", "step"]))
    {
        *slot = parse_part(part, what).map_err(malformed)?;
    }
    let [start, stop, step] = numbers;
    if step == Some(0) {
        return Err(NamedImageListSliceError::ZeroStep {
            value: value.to_string(),
        });
    }
    Ok(NamedImageListSlice { start, stop, step })
}

/// One part of a slice: empty, or ASCII digits with an optional leading `+`
/// or `-` that fit in an `i64`.
fn parse_part(part: &str, what: &str) -> Result<Option<i64>, String> {
    if part.is_empty() {
        return Ok(None);
    }
    let digits = part.strip_prefix(['+', '-']).unwrap_or(part);
    if digits.is_empty() || !digits.bytes().all(|b| b.is_ascii_digit()) {
        return Err(format!("the {what} '{part}' is not an integer"));
    }
    part.parse::<i64>()
        .map(Some)
        .map_err(|_| format!("the {what} '{part}' does not fit in a 64-bit integer"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn slice(text: &str) -> NamedImageListSlice {
        NamedImageListSlice::parse(text).unwrap_or_else(|e| panic!("{text}: {e}"))
    }

    /// Python's `list(range(*slice(start, stop, step).indices(len)))`, written
    /// out from CPython's `PySlice_Unpack` and `PySlice_AdjustIndices` as a
    /// check independent of `indices`: an empty bound becomes the largest or
    /// smallest integer, and is then clamped like any other.
    fn python_indices(
        start: Option<i64>,
        stop: Option<i64>,
        step: Option<i64>,
        len: i64,
    ) -> Vec<usize> {
        let step = step.unwrap_or(1);
        let start = start.unwrap_or(if step < 0 { i64::MAX } else { 0 });
        let stop = stop.unwrap_or(if step < 0 { i64::MIN } else { i64::MAX });
        let adjust = |v: i64| -> i64 {
            if v < 0 {
                match v.checked_add(len) {
                    Some(v) if v >= 0 => v,
                    _ if step < 0 => -1,
                    _ => 0,
                }
            } else if v >= len {
                if step < 0 {
                    len - 1
                } else {
                    len
                }
            } else {
                v
            }
        };
        let (start, stop) = (adjust(start), adjust(stop));
        let mut out = Vec::new();
        let mut i = start;
        while (step > 0 && i < stop) || (step < 0 && i > stop) {
            out.push(i as usize);
            i += step;
        }
        out
    }

    #[test]
    fn parse_reads_each_part() {
        assert_eq!(
            slice("[:30]"),
            NamedImageListSlice::new(None, Some(30), None).unwrap()
        );
        assert_eq!(
            slice("[-30:]"),
            NamedImageListSlice::new(Some(-30), None, None).unwrap()
        );
        assert_eq!(
            slice("[::3]"),
            NamedImageListSlice::new(None, None, Some(3)).unwrap()
        );
        assert_eq!(
            slice("[+2:-1:-2]"),
            NamedImageListSlice::new(Some(2), Some(-1), Some(-2)).unwrap()
        );
        assert_eq!(
            slice("[:]"),
            NamedImageListSlice::new(None, None, None).unwrap()
        );
        assert_eq!(
            slice("[::]"),
            NamedImageListSlice::new(None, None, None).unwrap()
        );
        let s = slice("[1:2:3]");
        assert_eq!((s.start(), s.stop(), s.step()), (Some(1), Some(2), Some(3)));
        assert_eq!(
            slice(&format!("[{}:{}]", i64::MIN, i64::MAX)),
            NamedImageListSlice::new(Some(i64::MIN), Some(i64::MAX), None).unwrap()
        );
    }

    #[test]
    fn display_writes_the_parts_back() {
        for (text, shown) in [
            ("[:30]", "[:30]"),
            ("[-30:]", "[-30:]"),
            ("[::3]", "[::3]"),
            ("[+2:-1:-2]", "[2:-1:-2]"),
            ("[:]", "[:]"),
            ("[::]", "[:]"),
            ("[5:6]", "[5:6]"),
        ] {
            assert_eq!(slice(text).to_string(), shown, "{text}");
        }
    }

    #[test]
    fn malformed_text_says_what_is_wrong() {
        for (text, problem) in [
            ("[a:3]", "the start 'a' is not an integer"),
            ("[1.5:]", "the start '1.5' is not an integer"),
            ("[:3x]", "the stop '3x' is not an integer"),
            ("[::-]", "the step '-' is not an integer"),
            ("[:+]", "the stop '+' is not an integer"),
            ("[:٣]", "the stop '٣' is not an integer"),
            ("[1:2:3:4]", "it has more than two ':'"),
            ("[5]", "[5] is a single index, which is not accepted"),
            ("[5]", "[5:6] is the one image at index 5"),
            ("[-2]", "[-2:-1] is the one image at index -2"),
            ("[-1]", "[-1:] is the one image at index -1"),
            ("[x]", "[x] is a single index"),
            ("[]", "the brackets are empty"),
            ("[:30", "it has no closing ']'"),
            ("[: 30]", "it holds whitespace"),
            ("[:30]x", "'x' follows the ']'"),
            ("[:30][::2]", "'[::2]' follows the ']'"),
            (
                "[99999999999999999999:]",
                "the start '99999999999999999999' does not fit",
            ),
            (":30]", "it does not start with '['"),
        ] {
            let e = NamedImageListSlice::parse(text).unwrap_err();
            assert!(
                matches!(e, NamedImageListSliceError::Malformed { .. }),
                "{text}: {e:?}"
            );
            let message = e.to_string();
            assert!(message.contains(problem), "{text}: {message}");
            assert!(
                message.starts_with(&format!("'{text}' is not a valid slice")),
                "{text}: {message}"
            );
            assert!(message.contains("NAME[start:stop:step]"), "{message}");
        }
    }

    #[test]
    fn a_step_of_zero_is_python_s_error() {
        for text in ["[::0]", "[1:2:0]", "[::-0]", "[::+0]"] {
            let e = NamedImageListSlice::parse(text).unwrap_err();
            assert_eq!(
                e,
                NamedImageListSliceError::ZeroStep {
                    value: text.to_string()
                }
            );
            assert_eq!(
                e.to_string(),
                format!("'{text}': slice step cannot be zero")
            );
        }
        assert!(matches!(
            NamedImageListSlice::new(None, None, Some(0)),
            Err(NamedImageListSliceError::ZeroStep { .. })
        ));
    }

    #[test]
    fn indices_follow_python_on_examples() {
        let ten = 10;
        let cases: &[(&str, Vec<usize>)] = &[
            ("[:3]", vec![0, 1, 2]),
            ("[-3:]", vec![7, 8, 9]),
            ("[::3]", vec![0, 3, 6, 9]),
            ("[1::3]", vec![1, 4, 7]),
            ("[::-1]", vec![9, 8, 7, 6, 5, 4, 3, 2, 1, 0]),
            ("[::-3]", vec![9, 6, 3, 0]),
            ("[8:2:-2]", vec![8, 6, 4]),
            ("[:30]", (0..10).collect()),
            ("[-30:]", (0..10).collect()),
            ("[40:]", vec![]),
            ("[5:2]", vec![]),
            ("[2:5:-1]", vec![]),
            ("[-1:-11:-1]", vec![9, 8, 7, 6, 5, 4, 3, 2, 1, 0]),
            ("[-100:100:-1]", vec![]),
            ("[100:-100:-1]", vec![9, 8, 7, 6, 5, 4, 3, 2, 1, 0]),
            ("[5:6]", vec![5]),
            ("[:]", (0..10).collect()),
        ];
        for (text, expected) in cases {
            assert_eq!(&slice(text).indices(ten), expected, "{text}");
        }
        assert_eq!(slice("[::-1]").indices(0), Vec::<usize>::new());
        assert_eq!(slice("[:]").indices(0), Vec::<usize>::new());
        let extreme = NamedImageListSlice::new(Some(i64::MIN), Some(i64::MAX), Some(i64::MAX))
            .unwrap()
            .indices(5);
        assert_eq!(extreme, vec![0]);
        let extreme = NamedImageListSlice::new(Some(i64::MAX), Some(i64::MIN), Some(i64::MIN))
            .unwrap()
            .indices(5);
        assert_eq!(extreme, vec![4]);
    }

    #[test]
    fn indices_follow_python_on_every_small_slice() {
        let bounds = [
            None,
            Some(-12),
            Some(-7),
            Some(-6),
            Some(-1),
            Some(0),
            Some(1),
            Some(5),
            Some(6),
            Some(11),
        ];
        let steps = [
            None,
            Some(-7),
            Some(-2),
            Some(-1),
            Some(1),
            Some(2),
            Some(3),
            Some(7),
        ];
        for len in [0, 1, 2, 6] {
            for start in bounds {
                for stop in bounds {
                    for step in steps {
                        let s = NamedImageListSlice::new(start, stop, step).unwrap();
                        assert_eq!(
                            s.indices(len as usize),
                            python_indices(start, stop, step, len),
                            "{s} of {len}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn split_takes_the_name_and_the_slice() {
        assert_eq!(
            split_named_image_list_value("walk1").unwrap(),
            ("walk1", None)
        );
        assert_eq!(
            split_named_image_list_value("walk1[:30]").unwrap(),
            ("walk1", Some(slice("[:30]")))
        );
        assert_eq!(
            split_named_image_list_value("rig-sharp.v2[1::2]").unwrap(),
            ("rig-sharp.v2", Some(slice("[1::2]")))
        );
    }

    #[test]
    fn split_errors_name_the_value() {
        let e = split_named_image_list_value("walk 1").unwrap_err();
        assert!(matches!(e, NamedImageListSliceError::InvalidName { .. }));
        assert_eq!(
            e.to_string(),
            format!("'walk 1' is not a valid named image list name: {NAME_RULE}")
        );

        let e = split_named_image_list_value("10-20[::2]").unwrap_err();
        assert!(matches!(e, NamedImageListSliceError::NotOnAName { .. }));
        let message = e.to_string();
        assert!(
            message.starts_with("'10-20[::2]' is not a valid slice"),
            "{message}"
        );
        assert!(message.contains("a slice follows the name of a named image list"));
        assert!(message.contains("'10-20' is a range"), "{message}");
        assert!(message.contains("-r NAME -r RANGE"), "{message}");

        let message = split_named_image_list_value("[:30]")
            .unwrap_err()
            .to_string();
        assert!(
            message.contains("no name comes before the '['"),
            "{message}"
        );

        let message = split_named_image_list_value("wa lk[:30]")
            .unwrap_err()
            .to_string();
        assert!(
            message.contains("'wa lk' is not a valid named image list name"),
            "{message}"
        );

        let e = split_named_image_list_value("walk1[a:3]").unwrap_err();
        assert!(
            e.to_string().starts_with(
                "'walk1[a:3]' is not a valid slice of a named image list: the start 'a'"
            ),
            "{e}"
        );
        assert_eq!(
            split_named_image_list_value("walk1[::0]").unwrap_err(),
            NamedImageListSliceError::ZeroStep {
                value: "walk1[::0]".to_string()
            }
        );
        // The first '[' starts the slice, so a second bracket is text after it.
        let message = split_named_image_list_value("walk1[:3][::2]")
            .unwrap_err()
            .to_string();
        assert!(message.contains("'[::2]' follows the ']'"), "{message}");
    }
}
