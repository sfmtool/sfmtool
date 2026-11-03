// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! Image-to-gray conversion formula parser and evaluator.
//!
//! SIFT operates on a single-channel float image, and several of its parameters
//! (notably the contrast threshold `|D| < 0.03`) are defined in that value domain.
//! The mapping from a decoded RGB source image to the gray sample is therefore
//! pinned by a formula recorded in the `.sift` metadata. This module parses and
//! evaluates that formula.
//!
//! See `specs/formats/sift-file-format.md` §"Image-to-gray conversion" for the
//! grammar. In brief: an arithmetic expression over the variables `R`, `G`, `B`
//! and decimal numeric literals (a leading `-` allowed on a literal), with the
//! operators `+`, `*`, `**` (precedence `**` > `*` > `+`; `**` right-associative)
//! and parentheses for grouping. Evaluation is in IEEE-754 single precision
//! and the output is used as-is (not clamped).
//!
//! [`ImageToGray`] is the formula together with the optional `quantize_bits`
//! that rounds its value to a bit depth: the `image_to_gray` object
//! `{"formula": …, "quantize_bits": …}` that a version 2 `.sift` file and a
//! version 2 workspace record.
//!
//! [`ImageToGray::gray_row`] is sfmtool's one conversion of 8-bit RGB pixels
//! to grey. The detector's input ([`gray_from_rgb`]) and the per-image
//! readings ([`image_readings_from_rgb`](super::image_readings_from_rgb)) both
//! make their grey samples with it, row by row, so the two see the same
//! `f32` values.
//!
//! **Why single precision.** The inputs are 8-bit samples, 256 levels, about
//! 2.4 significant digits; an `f32` holds 24 bits, so each operation's
//! rounding (a relative `2^-24`, `6e-8`) is about 1/65,000 of a grey level.
//! Evaluated in `f32` rather than `f64`, the default BT.709 formula's sample
//! moves by at most 2 ulps, and rounded to 8 bits it lands on another level
//! for 1,197 of the 2^24 colours, each an exact or near tie. A row of `f32`
//! is twice the lanes of `f64` per vector instruction and half the memory:
//! a 3840 x 2160 frame converts in about 15 ms on one thread, against 26 ms
//! for the same row loops in `f64`.

use super::GrayImage;

/// The default image-to-gray formula: BT.709 luma, matching COLMAP's
/// `Bitmap::CloneAsGrey`.
pub const DEFAULT_GRAY_FORMULA: &str = "0.2126*R + 0.7152*G + 0.0722*B";

/// An error produced while parsing an image-to-gray formula.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParseError {
    /// Human-readable description of what went wrong.
    pub message: String,
    /// Byte offset into the source string where the error was detected.
    pub position: usize,
}

impl std::fmt::Display for ParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} (at position {})", self.message, self.position)
    }
}

impl std::error::Error for ParseError {}

/// A parsed image-to-gray formula, ready to evaluate per pixel.
///
/// The expression tree is small. [`GrayFormula::eval`] walks it for one pixel;
/// a whole row is evaluated a node at a time by [`ImageToGray::gray_row`],
/// which gives the same `f32` at each pixel.
#[derive(Debug, Clone, PartialEq)]
pub struct GrayFormula {
    root: Expr,
}

impl GrayFormula {
    /// Evaluate the formula at the given (r, g, b) channel values, each in
    /// `[0, 1]`, in single precision: every literal is the `f32` nearest its
    /// decimal text and every operation's result is rounded to `f32`.
    ///
    /// The result is *not* clamped — value-domain parameters assume inputs on a
    /// `[0, 1]` scale, but a formula is free to produce values outside it.
    pub fn eval(&self, r: f32, g: f32, b: f32) -> f32 {
        self.root.eval(r, g, b)
    }

    /// Evaluate the formula at every index of equal-length channel slices
    /// into `out`, giving the same `f32` as [`GrayFormula::eval`] at each
    /// index: the same operations in the same order, a whole row per node of
    /// the tree rather than a tree walk per pixel. `scratch` holds the
    /// intermediate rows, kept between calls so a caller converting row after
    /// row allocates them once.
    fn eval_slices(
        &self,
        channels: (&[f32], &[f32], &[f32]),
        out: &mut [f32],
        scratch: &mut Vec<Vec<f32>>,
    ) {
        let (r, g, b) = channels;
        debug_assert!(r.len() == out.len() && g.len() == out.len() && b.len() == out.len());
        self.root.eval_slices(channels, out, scratch);
    }
}

impl Default for GrayFormula {
    /// The default formula is [`DEFAULT_GRAY_FORMULA`] (BT.709 luma), which always
    /// parses.
    fn default() -> Self {
        parse_gray_formula(DEFAULT_GRAY_FORMULA).expect("default gray formula must parse")
    }
}

/// Expression AST node.
#[derive(Debug, Clone, PartialEq)]
enum Expr {
    /// A numeric literal (a leading `-` is folded into the value).
    Literal(f32),
    /// The red channel variable `R`.
    R,
    /// The green channel variable `G`.
    G,
    /// The blue channel variable `B`.
    B,
    /// `lhs + rhs`.
    Add(Box<Expr>, Box<Expr>),
    /// `lhs * rhs`.
    Mul(Box<Expr>, Box<Expr>),
    /// `base ** exponent`.
    Pow(Box<Expr>, Box<Expr>),
}

impl Expr {
    fn eval(&self, r: f32, g: f32, b: f32) -> f32 {
        match self {
            Expr::Literal(v) => *v,
            Expr::R => r,
            Expr::G => g,
            Expr::B => b,
            Expr::Add(a, c) => a.eval(r, g, b) + c.eval(r, g, b),
            Expr::Mul(a, c) => a.eval(r, g, b) * c.eval(r, g, b),
            Expr::Pow(a, c) => a.eval(r, g, b).powf(c.eval(r, g, b)),
        }
    }

    fn eval_slices(
        &self,
        channels: (&[f32], &[f32], &[f32]),
        out: &mut [f32],
        scratch: &mut Vec<Vec<f32>>,
    ) {
        /// `out = op(a, c)` elementwise, `a` evaluated into `out` first and
        /// `c` into a row taken from `scratch` and returned to it.
        fn binary(
            (a, c): (&Expr, &Expr),
            channels: (&[f32], &[f32], &[f32]),
            out: &mut [f32],
            scratch: &mut Vec<Vec<f32>>,
            op: impl Fn(f32, f32) -> f32,
        ) {
            a.eval_slices(channels, out, scratch);
            let mut rhs = scratch.pop().unwrap_or_default();
            rhs.resize(out.len(), 0.0);
            c.eval_slices(channels, &mut rhs, scratch);
            for (o, v) in out.iter_mut().zip(&rhs) {
                *o = op(*o, *v);
            }
            scratch.push(rhs);
        }
        let (r, g, b) = channels;
        match self {
            Expr::Literal(v) => out.fill(*v),
            Expr::R => out.copy_from_slice(r),
            Expr::G => out.copy_from_slice(g),
            Expr::B => out.copy_from_slice(b),
            Expr::Add(a, c) => binary((a, c), channels, out, scratch, |x, y| x + y),
            Expr::Mul(a, c) => binary((a, c), channels, out, scratch, |x, y| x * y),
            Expr::Pow(a, c) => binary((a, c), channels, out, scratch, f32::powf),
        }
    }
}

/// Parse an image-to-gray formula from its textual form.
///
/// See the module docs for the grammar. Returns a [`ParseError`] on malformed
/// input (unknown identifier, unbalanced parentheses, trailing tokens, etc.).
pub fn parse_gray_formula(src: &str) -> Result<GrayFormula, ParseError> {
    let tokens = tokenize(src)?;
    // The parser, the evaluation and the drop of the tree all recurse once per
    // level of nesting, and a sum of n terms nests n deep, so a bound on the
    // token count bounds the stack they use.
    if tokens.len() > MAX_FORMULA_TOKENS {
        return Err(ParseError {
            message: format!(
                "the formula has {} tokens, more than the {MAX_FORMULA_TOKENS} allowed",
                tokens.len()
            ),
            position: tokens[MAX_FORMULA_TOKENS].position,
        });
    }
    let mut parser = Parser { tokens, pos: 0 };
    let root = parser.parse_add()?;
    // Anything left over after a complete expression is an error.
    if let Some(tok) = parser.peek() {
        return Err(ParseError {
            message: format!("unexpected trailing token {:?}", tok.kind),
            position: tok.position,
        });
    }
    Ok(GrayFormula { root })
}

/// The most tokens a formula may have. A real conversion has a few dozen; the
/// bound keeps the recursive parse and evaluation within a thread's stack.
pub const MAX_FORMULA_TOKENS: usize = 1000;

/// The most bits [`ImageToGray::quantize_bits`] may round to.
pub const MAX_QUANTIZE_BITS: u8 = 16;

/// An `image_to_gray` value [`ImageToGray`] refused.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ImageToGrayError {
    /// The value is not a JSON object.
    NotAnObject,
    /// The object has no `formula` key.
    MissingFormula,
    /// `formula` is not a string.
    FormulaNotAString,
    /// `formula` does not parse; holds the formula and the parser's error.
    Formula {
        /// The formula as given.
        formula: String,
        /// Why it does not parse.
        error: ParseError,
    },
    /// `quantize_bits` is not an integer from 1 to [`MAX_QUANTIZE_BITS`];
    /// holds the value as given, in JSON.
    QuantizeBits(String),
    /// The object has a key other than `formula` and `quantize_bits`.
    UnknownKey(String),
}

impl std::fmt::Display for ImageToGrayError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotAnObject => write!(f, "image_to_gray must be a JSON object"),
            Self::MissingFormula => write!(f, "image_to_gray has no 'formula'"),
            Self::FormulaNotAString => write!(f, "image_to_gray 'formula' must be a string"),
            Self::Formula { formula, error } => {
                write!(
                    f,
                    "image_to_gray formula {formula:?} does not parse: {error}"
                )
            }
            Self::QuantizeBits(value) => write!(
                f,
                "image_to_gray 'quantize_bits' must be an integer from 1 to \
                 {MAX_QUANTIZE_BITS}, got {value}"
            ),
            Self::UnknownKey(key) => write!(
                f,
                "image_to_gray has an unknown key {key:?} (only 'formula' and \
                 'quantize_bits' are defined)"
            ),
        }
    }
}

impl std::error::Error for ImageToGrayError {}

/// The conversion from an image's colour samples to its grey samples: a
/// formula over `R`, `G` and `B` on `[0, 1]`, and an optional bit depth its
/// value is rounded to.
///
/// It is the `image_to_gray` object `{"formula": …, "quantize_bits": …}`.
/// Without `quantize_bits` the grey sample is the formula's value `v`, not
/// clamped. With `quantize_bits = b` it is
/// `round(clamp(v, 0, 1) · (2^b − 1)) / (2^b − 1)`, an exact half rounded up
/// ([`ImageToGray::gray`]). The formula's text is kept as given, so
/// [`ImageToGray::to_json`] writes back the object it was read from.
///
/// # Example
///
/// ```
/// use sfmtool_core::features::sift::ImageToGray;
///
/// let value = serde_json::json!({"formula": "0.2126*R + 0.7152*G + 0.0722*B", "quantize_bits": 8});
/// let to_gray = ImageToGray::from_json(&value)?;
/// assert_eq!(to_gray.quantize_bits(), Some(8));
/// // Full-scale white is exactly 1 after rounding to 8 bits.
/// assert_eq!(to_gray.gray(1.0, 1.0, 1.0), 1.0);
/// assert_eq!(to_gray.to_json(), value);
///
/// // A version 1 record's `gray_formula` string: no rounding.
/// let v1 = ImageToGray::from_gray_formula("G")?;
/// assert_eq!(v1.quantize_bits(), None);
/// # Ok::<(), sfmtool_core::features::sift::gray::ImageToGrayError>(())
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct ImageToGray {
    formula_text: String,
    formula: GrayFormula,
    quantize_bits: Option<u8>,
}

impl ImageToGray {
    /// The conversion of `formula`, rounded to `quantize_bits` bits when given.
    ///
    /// # Errors
    ///
    /// [`ImageToGrayError::Formula`] when the formula does not parse, and
    /// [`ImageToGrayError::QuantizeBits`] for a bit count outside 1 to
    /// [`MAX_QUANTIZE_BITS`].
    pub fn new(formula: &str, quantize_bits: Option<u8>) -> Result<Self, ImageToGrayError> {
        if let Some(b) = quantize_bits {
            if !(1..=MAX_QUANTIZE_BITS).contains(&b) {
                return Err(ImageToGrayError::QuantizeBits(b.to_string()));
            }
        }
        let parsed = parse_gray_formula(formula).map_err(|error| ImageToGrayError::Formula {
            formula: formula.to_owned(),
            error,
        })?;
        Ok(Self {
            formula_text: formula.to_owned(),
            formula: parsed,
            quantize_bits,
        })
    }

    /// The conversion a `gray_formula` string records: the formula, with no
    /// rounding. This is how a version 1 record, and the `sfmtool` tool's
    /// `gray_formula` feature option, state the conversion.
    ///
    /// # Errors
    ///
    /// [`ImageToGrayError::Formula`] when the formula does not parse.
    pub fn from_gray_formula(formula: &str) -> Result<Self, ImageToGrayError> {
        Self::new(formula, None)
    }

    /// Read the `image_to_gray` JSON object: a string `formula` and, optionally,
    /// an integer `quantize_bits` from 1 to [`MAX_QUANTIZE_BITS`].
    ///
    /// # Errors
    ///
    /// [`ImageToGrayError`] for a value that is not an object, a missing or
    /// non-string `formula`, a formula that does not parse, a `quantize_bits`
    /// that is not an integer in range (`null` included), and any other key.
    pub fn from_json(value: &serde_json::Value) -> Result<Self, ImageToGrayError> {
        let map = value.as_object().ok_or(ImageToGrayError::NotAnObject)?;
        if let Some(key) = map
            .keys()
            .find(|k| k.as_str() != "formula" && k.as_str() != "quantize_bits")
        {
            return Err(ImageToGrayError::UnknownKey(key.clone()));
        }
        let formula = map
            .get("formula")
            .ok_or(ImageToGrayError::MissingFormula)?
            .as_str()
            .ok_or(ImageToGrayError::FormulaNotAString)?;
        let quantize_bits = match map.get("quantize_bits") {
            None => None,
            Some(v) => Some(
                v.as_u64()
                    .and_then(|b| u8::try_from(b).ok())
                    .filter(|b| (1..=MAX_QUANTIZE_BITS).contains(b))
                    .ok_or_else(|| ImageToGrayError::QuantizeBits(v.to_string()))?,
            ),
        };
        Self::new(formula, quantize_bits)
    }

    /// The `image_to_gray` JSON object: `formula` as given, and `quantize_bits`
    /// only when present.
    pub fn to_json(&self) -> serde_json::Value {
        let mut map = serde_json::Map::new();
        map.insert("formula".into(), self.formula_text.clone().into());
        if let Some(b) = self.quantize_bits {
            map.insert("quantize_bits".into(), b.into());
        }
        serde_json::Value::Object(map)
    }

    /// The formula's text, as given.
    pub fn formula(&self) -> &str {
        &self.formula_text
    }

    /// The parsed formula.
    pub fn gray_formula(&self) -> &GrayFormula {
        &self.formula
    }

    /// The bit depth the formula's value is rounded to, or `None` for none.
    pub fn quantize_bits(&self) -> Option<u8> {
        self.quantize_bits
    }

    /// The grey sample of channel values `r`, `g`, `b` on `[0, 1]`: the
    /// formula's value in single precision, then [`quantize`] when
    /// `quantize_bits` is present.
    pub fn gray(&self, r: f32, g: f32, b: f32) -> f32 {
        let v = self.formula.eval(r, g, b);
        match self.quantize_bits {
            Some(bits) => quantize(v, bits),
            None => v,
        }
    }

    /// The grey samples of one row of 8-bit RGB pixels, `[R, G, B]` per pixel,
    /// into `out`, one per pixel: each channel divided by 255 in `f32` and the
    /// pixel made grey by [`ImageToGray::gray`], giving the same `f32` at each
    /// pixel.
    ///
    /// This is sfmtool's one conversion from RGB to grey: [`gray_from_rgb`]
    /// and the reading image of
    /// [`image_readings_from_rgb`](super::image_readings_from_rgb) both call
    /// it. The formula is evaluated a node of its tree at a time over the
    /// whole row, so the work is a few loops over `f32` rows that the compiler
    /// vectorises, rather than a tree walk per pixel. `scratch` holds the row
    /// buffers, kept between calls so a caller converting row after row
    /// allocates them once.
    ///
    /// # Panics
    ///
    /// Panics if `rgb` does not hold `3 * out.len()` bytes.
    pub fn gray_row(&self, rgb: &[u8], out: &mut [f32], scratch: &mut GrayRowScratch) {
        assert_eq!(rgb.len(), 3 * out.len(), "a row of n pixels is 3 n bytes");
        let n = out.len();
        let [r, g, b] = &mut scratch.channels;
        r.resize(n, 0.0);
        g.resize(n, 0.0);
        b.resize(n, 0.0);
        for (x, p) in rgb.as_chunks::<3>().0.iter().enumerate() {
            r[x] = UNIT_SAMPLES[usize::from(p[0])];
            g[x] = UNIT_SAMPLES[usize::from(p[1])];
            b[x] = UNIT_SAMPLES[usize::from(p[2])];
        }
        self.formula
            .eval_slices((r, g, b), out, &mut scratch.formula);
        if let Some(bits) = self.quantize_bits {
            for v in out.iter_mut() {
                *v = quantize(*v, bits);
            }
        }
    }
}

/// Each 8-bit sample `v` as `v / 255` in `f32`, the division rounded once.
static UNIT_SAMPLES: [f32; 256] = {
    let mut table = [0f32; 256];
    let mut v = 0;
    while v < 256 {
        table[v] = v as f32 / 255.0;
        v += 1;
    }
    table
};

/// The row buffers [`ImageToGray::gray_row`] works in, kept between rows.
#[derive(Debug, Default)]
pub struct GrayRowScratch {
    channels: [Vec<f32>; 3],
    formula: Vec<Vec<f32>>,
}

impl Default for ImageToGray {
    /// [`DEFAULT_GRAY_FORMULA`] (BT.709 luma) with no rounding.
    fn default() -> Self {
        Self::from_gray_formula(DEFAULT_GRAY_FORMULA).expect("default gray formula must parse")
    }
}

/// `v` rounded to `bits` bits over `[0, 1]`:
/// `round(clamp(v, 0, 1) · (2^bits − 1)) / (2^bits − 1)`, an exact half
/// rounded up.
///
/// In single precision: the product and the quotient are each rounded to
/// `f32`, and `2^bits − 1`, at most 65,535, is exact. The argument of the
/// rounding is never negative, so rounding half away from zero
/// ([`f32::round`]) is rounding half up. An infinity clamps to `0` or `1`; a
/// NaN stays NaN. `bits` is expected in 1 to [`MAX_QUANTIZE_BITS`], which
/// [`ImageToGray`] checks; a value outside it is treated as the nearer end of
/// that range.
pub fn quantize(v: f32, bits: u8) -> f32 {
    let levels = ((1u32 << bits.clamp(1, MAX_QUANTIZE_BITS)) - 1) as f32;
    (v.clamp(0.0, 1.0) * levels).round() / levels
}

/// A lexical token together with its byte position in the source.
#[derive(Debug, Clone, PartialEq)]
struct Token {
    kind: TokKind,
    position: usize,
}

#[derive(Debug, Clone, PartialEq)]
enum TokKind {
    Number(f32),
    Ident(char),
    Plus,
    Star,
    StarStar,
    LParen,
    RParen,
}

/// Split the source into tokens. Whitespace is insignificant.
///
/// A `-` is only accepted immediately preceding a numeric literal (there is no
/// subtraction operator); the sign is folded into the resulting `Number`.
fn tokenize(src: &str) -> Result<Vec<Token>, ParseError> {
    let bytes = src.as_bytes();
    let mut tokens = Vec::new();
    let mut i = 0;
    while i < bytes.len() {
        let c = bytes[i] as char;
        match c {
            c if c.is_whitespace() => {
                i += 1;
            }
            '+' => {
                tokens.push(Token {
                    kind: TokKind::Plus,
                    position: i,
                });
                i += 1;
            }
            '*' => {
                if i + 1 < bytes.len() && bytes[i + 1] == b'*' {
                    tokens.push(Token {
                        kind: TokKind::StarStar,
                        position: i,
                    });
                    i += 2;
                } else {
                    tokens.push(Token {
                        kind: TokKind::Star,
                        position: i,
                    });
                    i += 1;
                }
            }
            '(' => {
                tokens.push(Token {
                    kind: TokKind::LParen,
                    position: i,
                });
                i += 1;
            }
            ')' => {
                tokens.push(Token {
                    kind: TokKind::RParen,
                    position: i,
                });
                i += 1;
            }
            'R' | 'G' | 'B' => {
                tokens.push(Token {
                    kind: TokKind::Ident(c),
                    position: i,
                });
                i += 1;
            }
            // A numeric literal, optionally signed with a leading `-`.
            '-' | '0'..='9' | '.' => {
                let start = i;
                if c == '-' {
                    i += 1;
                }
                let mut saw_digit = false;
                let mut saw_dot = false;
                while i < bytes.len() {
                    let d = bytes[i] as char;
                    if d.is_ascii_digit() {
                        saw_digit = true;
                        i += 1;
                    } else if d == '.' && !saw_dot {
                        saw_dot = true;
                        i += 1;
                    } else {
                        break;
                    }
                }
                if !saw_digit {
                    return Err(ParseError {
                        message: "malformed numeric literal".to_string(),
                        position: start,
                    });
                }
                let text = &src[start..i];
                // The f32 nearest the decimal text, rounded once.
                let value: f32 = text.parse().map_err(|_| ParseError {
                    message: format!("invalid numeric literal {:?}", text),
                    position: start,
                })?;
                tokens.push(Token {
                    kind: TokKind::Number(value),
                    position: start,
                });
            }
            _ => {
                return Err(ParseError {
                    message: format!("unexpected character {:?}", c),
                    position: i,
                });
            }
        }
    }
    Ok(tokens)
}

/// Recursive-descent parser over the token stream.
///
/// Grammar (lowest to highest precedence):
/// ```text
/// add  := mul ('+' mul)*
/// mul  := pow ('*' pow)*
/// pow  := atom ('**' pow)?        // right-associative
/// atom := Number | 'R' | 'G' | 'B' | '(' add ')'
/// ```
struct Parser {
    tokens: Vec<Token>,
    pos: usize,
}

impl Parser {
    fn peek(&self) -> Option<&Token> {
        self.tokens.get(self.pos)
    }

    fn next(&mut self) -> Option<Token> {
        let tok = self.tokens.get(self.pos).cloned();
        if tok.is_some() {
            self.pos += 1;
        }
        tok
    }

    fn parse_add(&mut self) -> Result<Expr, ParseError> {
        let mut lhs = self.parse_mul()?;
        while matches!(self.peek().map(|t| &t.kind), Some(TokKind::Plus)) {
            self.next();
            let rhs = self.parse_mul()?;
            lhs = Expr::Add(Box::new(lhs), Box::new(rhs));
        }
        Ok(lhs)
    }

    fn parse_mul(&mut self) -> Result<Expr, ParseError> {
        let mut lhs = self.parse_pow()?;
        while matches!(self.peek().map(|t| &t.kind), Some(TokKind::Star)) {
            self.next();
            let rhs = self.parse_pow()?;
            lhs = Expr::Mul(Box::new(lhs), Box::new(rhs));
        }
        Ok(lhs)
    }

    fn parse_pow(&mut self) -> Result<Expr, ParseError> {
        let base = self.parse_atom()?;
        if matches!(self.peek().map(|t| &t.kind), Some(TokKind::StarStar)) {
            self.next();
            // Right-associative: the exponent is itself a `pow`.
            let exp = self.parse_pow()?;
            Ok(Expr::Pow(Box::new(base), Box::new(exp)))
        } else {
            Ok(base)
        }
    }

    fn parse_atom(&mut self) -> Result<Expr, ParseError> {
        let tok = self.next().ok_or_else(|| ParseError {
            message: "unexpected end of formula".to_string(),
            position: self.tokens.last().map(|t| t.position + 1).unwrap_or(0),
        })?;
        match tok.kind {
            TokKind::Number(v) => Ok(Expr::Literal(v)),
            TokKind::Ident('R') => Ok(Expr::R),
            TokKind::Ident('G') => Ok(Expr::G),
            TokKind::Ident('B') => Ok(Expr::B),
            TokKind::Ident(c) => Err(ParseError {
                message: format!("unknown variable {:?}", c),
                position: tok.position,
            }),
            TokKind::LParen => {
                let inner = self.parse_add()?;
                match self.next() {
                    Some(Token {
                        kind: TokKind::RParen,
                        ..
                    }) => Ok(inner),
                    other => Err(ParseError {
                        message: "expected closing parenthesis".to_string(),
                        position: other.map(|t| t.position).unwrap_or(tok.position),
                    }),
                }
            }
            other => Err(ParseError {
                message: format!("expected a value, found {:?}", other),
                position: tok.position,
            }),
        }
    }
}

/// Convert an interleaved 8-bit RGB image to a [`GrayImage`] through
/// `image_to_gray`.
///
/// `rgb` is `3 * width * height` bytes in row-major `[R, G, B, R, G, B, ...]`
/// order. Each row is made grey by [`ImageToGray::gray_row`]: each channel
/// divided by 255, the formula evaluated in single precision, and the value
/// rounded to `quantize_bits` bits when the conversion has them; without them
/// it is used as-is and is **not** clamped. Bands of rows are converted in
/// parallel with rayon, and each sample is the same whatever the thread count.
///
/// # Panics
///
/// Panics if `rgb` is not `3 * width * height` bytes.
pub fn gray_from_rgb(
    width: u32,
    height: u32,
    rgb: &[u8],
    image_to_gray: &ImageToGray,
) -> GrayImage {
    use rayon::prelude::*;

    let w = width as usize;
    let h = height as usize;
    assert_eq!(rgb.len(), 3 * w * h, "rgb must be 3 * width * height bytes");

    let mut data = vec![0.0f32; w * h];
    if !data.is_empty() {
        data.par_chunks_mut(w * GRAY_BAND_ROWS)
            .enumerate()
            .for_each(|(band, out_rows)| {
                let mut scratch = GrayRowScratch::default();
                for (i, out_row) in out_rows.chunks_exact_mut(w).enumerate() {
                    let y = band * GRAY_BAND_ROWS + i;
                    let src = &rgb[y * w * 3..(y + 1) * w * 3];
                    image_to_gray.gray_row(src, out_row, &mut scratch);
                }
            });
    }
    GrayImage::new(width, height, data)
}

/// The rows a rayon task converts with one set of row buffers, so the buffers
/// are allocated once per band rather than once per row.
pub(crate) const GRAY_BAND_ROWS: usize = 16;

#[cfg(test)]
mod tests;
