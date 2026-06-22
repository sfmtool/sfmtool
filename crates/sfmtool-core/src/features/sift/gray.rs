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
//! and parentheses for grouping. Evaluation is in IEEE-754 double precision and
//! the output is used as-is (not clamped).

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
/// The expression tree is small and evaluation is a cheap recursive walk over
/// the f64 channel values; for hot loops it is shared by reference across all
/// pixels rather than re-parsed.
#[derive(Debug, Clone, PartialEq)]
pub struct GrayFormula {
    root: Expr,
}

impl GrayFormula {
    /// Evaluate the formula at the given (r, g, b) channel values, each in `[0, 1]`.
    ///
    /// The result is *not* clamped — value-domain parameters assume inputs on a
    /// `[0, 1]` scale, but a formula is free to produce values outside it.
    pub fn eval(&self, r: f64, g: f64, b: f64) -> f64 {
        self.root.eval(r, g, b)
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
    Literal(f64),
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
    fn eval(&self, r: f64, g: f64, b: f64) -> f64 {
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
}

/// Parse an image-to-gray formula from its textual form.
///
/// See the module docs for the grammar. Returns a [`ParseError`] on malformed
/// input (unknown identifier, unbalanced parentheses, trailing tokens, etc.).
pub fn parse_gray_formula(src: &str) -> Result<GrayFormula, ParseError> {
    let tokens = tokenize(src)?;
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

/// A lexical token together with its byte position in the source.
#[derive(Debug, Clone, PartialEq)]
struct Token {
    kind: TokKind,
    position: usize,
}

#[derive(Debug, Clone, PartialEq)]
enum TokKind {
    Number(f64),
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
                let value: f64 = text.parse().map_err(|_| ParseError {
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

/// Convert an interleaved 8-bit RGB image to a [`GrayImage`] using `formula`.
///
/// `rgb` is `3 * width * height` bytes in row-major `[R, G, B, R, G, B, ...]`
/// order. Each channel is normalized to `[0, 1]` (divided by 255) before the
/// formula is evaluated. The output is used as-is and is **not** clamped. Rows
/// are converted in parallel with rayon.
pub fn gray_from_rgb(width: u32, height: u32, rgb: &[u8], formula: &GrayFormula) -> GrayImage {
    use rayon::prelude::*;

    let w = width as usize;
    let h = height as usize;
    assert_eq!(rgb.len(), 3 * w * h, "rgb must be 3 * width * height bytes");

    let mut data = vec![0.0f32; w * h];
    data.par_chunks_mut(w)
        .enumerate()
        .for_each(|(row, out_row)| {
            let row_base = row * w * 3;
            for (col, out) in out_row.iter_mut().enumerate() {
                let p = row_base + col * 3;
                let r = rgb[p] as f64 / 255.0;
                let g = rgb[p + 1] as f64 / 255.0;
                let b = rgb[p + 2] as f64 / 255.0;
                *out = formula.eval(r, g, b) as f32;
            }
        });

    GrayImage::new(width, height, data)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_precedence_pow_over_mul_over_add() {
        // 2 + 3 * 4 ** 2 = 2 + 3 * 16 = 2 + 48 = 50
        let f = parse_gray_formula("2 + 3 * 4 ** 2").unwrap();
        assert_eq!(f.eval(0.0, 0.0, 0.0), 50.0);
    }

    #[test]
    fn test_pow_right_associative() {
        // 2 ** 3 ** 2 = 2 ** 9 = 512 (right-assoc), not (2**3)**2 = 64
        let f = parse_gray_formula("2 ** 3 ** 2").unwrap();
        assert_eq!(f.eval(0.0, 0.0, 0.0), 512.0);
    }

    #[test]
    fn test_parentheses_override_precedence() {
        // (2 + 3) * 4 = 20
        let f = parse_gray_formula("(2 + 3) * 4").unwrap();
        assert_eq!(f.eval(0.0, 0.0, 0.0), 20.0);
    }

    #[test]
    fn test_negative_literal() {
        // -2 * 3 + 10 = 4
        let f = parse_gray_formula("-2 * 3 + 10").unwrap();
        assert_eq!(f.eval(0.0, 0.0, 0.0), 4.0);
    }

    #[test]
    fn test_default_colmap_formula() {
        let f = parse_gray_formula(DEFAULT_GRAY_FORMULA).unwrap();
        // Pure white -> 0.2126 + 0.7152 + 0.0722 = 1.0
        assert!((f.eval(1.0, 1.0, 1.0) - 1.0).abs() < 1e-12);
        // Pure red -> 0.2126
        assert!((f.eval(1.0, 0.0, 0.0) - 0.2126).abs() < 1e-12);
        // A sample value
        let v = f.eval(0.5, 0.25, 0.75);
        let expected = 0.2126 * 0.5 + 0.7152 * 0.25 + 0.0722 * 0.75;
        assert!((v - expected).abs() < 1e-12);
    }

    #[test]
    fn test_channel_selection() {
        let f = parse_gray_formula("G").unwrap();
        assert_eq!(f.eval(0.1, 0.7, 0.3), 0.7);
    }

    #[test]
    fn test_gray_from_rgb_small_image() {
        // 2x1 image: white pixel then black pixel.
        let rgb = vec![255u8, 255, 255, 0, 0, 0];
        let f = parse_gray_formula(DEFAULT_GRAY_FORMULA).unwrap();
        let img = gray_from_rgb(2, 1, &rgb, &f);
        assert_eq!(img.width(), 2);
        assert_eq!(img.height(), 1);
        assert!((img.get_pixel(0, 0) - 1.0).abs() < 1e-6);
        assert!((img.get_pixel(1, 0) - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_gray_from_rgb_no_upper_clamp() {
        // A formula that exceeds 1.0 must not be clamped.
        let f = parse_gray_formula("2 * R").unwrap();
        let rgb = vec![255u8, 0, 0];
        let img = gray_from_rgb(1, 1, &rgb, &f);
        assert!((img.get_pixel(0, 0) - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_parse_errors() {
        assert!(parse_gray_formula("R +").is_err());
        assert!(parse_gray_formula("R + + G").is_err());
        assert!(parse_gray_formula("(R + G").is_err());
        assert!(parse_gray_formula("R G").is_err());
        assert!(parse_gray_formula("X").is_err());
        assert!(parse_gray_formula("R - G").is_err()); // no subtraction operator
    }
}
