// Copyright The SfM Tool Authors
// SPDX-License-Identifier: Apache-2.0

//! A JSON value that keeps every object member, in the order the file wrote
//! them.
//!
//! `serde_json::Value` keeps one member per key, the last, so a lists file
//! that defines one list name twice would lose the first definition without a
//! word. This tree keeps both, so the validator can report the repeat, and it
//! keeps the file's order, which is the order the lists are reported in.

use serde::de::{Deserialize, Deserializer, MapAccess, SeqAccess, Visitor};
use std::fmt;

/// A parsed JSON value whose objects are lists of `(key, value)` members.
#[derive(Debug, Clone, PartialEq)]
pub(super) enum Json {
    Null,
    Bool(bool),
    Number(serde_json::Number),
    String(String),
    Array(Vec<Json>),
    Object(Vec<(String, Json)>),
}

impl Json {
    /// The JSON type's name, as an error message says what a value is.
    pub(super) fn type_name(&self) -> &'static str {
        match self {
            Json::Null => "null",
            Json::Bool(_) => "a boolean",
            Json::Number(_) => "a number",
            Json::String(_) => "a string",
            Json::Array(_) => "an array",
            Json::Object(_) => "an object",
        }
    }
}

impl<'de> Deserialize<'de> for Json {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        deserializer.deserialize_any(JsonVisitor)
    }
}

struct JsonVisitor;

impl<'de> Visitor<'de> for JsonVisitor {
    type Value = Json;

    fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("any JSON value")
    }

    fn visit_unit<E>(self) -> Result<Json, E> {
        Ok(Json::Null)
    }

    fn visit_bool<E>(self, v: bool) -> Result<Json, E> {
        Ok(Json::Bool(v))
    }

    fn visit_i64<E>(self, v: i64) -> Result<Json, E> {
        Ok(Json::Number(v.into()))
    }

    fn visit_u64<E>(self, v: u64) -> Result<Json, E> {
        Ok(Json::Number(v.into()))
    }

    fn visit_f64<E: serde::de::Error>(self, v: f64) -> Result<Json, E> {
        // JSON text cannot spell a NaN or an infinity, so this always holds
        // for what serde_json hands over.
        serde_json::Number::from_f64(v)
            .map(Json::Number)
            .ok_or_else(|| E::custom("a number that is not finite"))
    }

    fn visit_str<E>(self, v: &str) -> Result<Json, E> {
        Ok(Json::String(v.to_string()))
    }

    fn visit_string<E>(self, v: String) -> Result<Json, E> {
        Ok(Json::String(v))
    }

    fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<Json, A::Error> {
        let mut items = Vec::new();
        while let Some(item) = seq.next_element()? {
            items.push(item);
        }
        Ok(Json::Array(items))
    }

    fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<Json, A::Error> {
        let mut members = Vec::new();
        while let Some((key, value)) = map.next_entry::<String, Json>()? {
            members.push((key, value));
        }
        Ok(Json::Object(members))
    }
}
