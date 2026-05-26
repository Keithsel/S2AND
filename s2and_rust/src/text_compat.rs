use pyo3::prelude::*;
use pyo3::types::PyAny;
use pyo3::Bound;
use std::collections::{HashMap, HashSet};

pub(crate) fn ensure_unidecode_for_text(
    unidecode_fn: &Bound<'_, PyAny>,
    text: &str,
    unidecode_char_map: &mut HashMap<char, String>,
) -> PyResult<()> {
    if text.is_ascii() {
        return Ok(());
    }
    for ch in text.chars() {
        if ch.is_ascii() || unidecode_char_map.contains_key(&ch) {
            continue;
        }
        let mapped: String = unidecode_fn.call1((ch.to_string(),))?.extract()?;
        unidecode_char_map.insert(ch, mapped);
    }
    Ok(())
}

pub(crate) fn normalize_ascii_text_compat(text: &str, special_case_apostrophes: bool) -> String {
    let mut normalized = String::with_capacity(text.len());
    let mut prev_space = true;
    for byte in text.bytes() {
        let lowered = byte.to_ascii_lowercase();
        if lowered.is_ascii_alphabetic() {
            normalized.push(lowered as char);
            prev_space = false;
        } else if special_case_apostrophes && lowered == b'\'' {
            continue;
        } else if !prev_space {
            normalized.push(' ');
            prev_space = true;
        }
    }
    while normalized.ends_with(' ') {
        normalized.pop();
    }
    normalized
}

pub(crate) fn normalize_text_compat_from_map(
    text: &str,
    special_case_apostrophes: bool,
    unidecode_char_map: &HashMap<char, String>,
) -> String {
    if text.is_empty() {
        return String::new();
    }
    if text.is_ascii() {
        return normalize_ascii_text_compat(text, special_case_apostrophes);
    }

    let mut transliterated = String::with_capacity(text.len());
    for ch in text.chars() {
        if ch.is_ascii() {
            transliterated.push(ch.to_ascii_lowercase());
            continue;
        }
        if let Some(mapped) = unidecode_char_map.get(&ch) {
            for mapped_ch in mapped.chars() {
                for lowered in mapped_ch.to_lowercase() {
                    transliterated.push(lowered);
                }
            }
        } else {
            for lowered in ch.to_lowercase() {
                transliterated.push(lowered);
            }
        }
    }

    let source = if special_case_apostrophes {
        transliterated.replace('\'', "")
    } else {
        transliterated
    };
    let mut normalized = String::with_capacity(source.len());
    let mut prev_space = true;
    for ch in source.chars() {
        if ch.is_ascii_alphabetic() {
            normalized.push(ch);
            prev_space = false;
        } else if !prev_space {
            normalized.push(' ');
            prev_space = true;
        }
    }
    while normalized.ends_with(' ') {
        normalized.pop();
    }
    normalized
}

pub(crate) fn first_normalized_token_python_compat(
    first_normalized: &str,
    middle_normalized: &str,
    name_prefixes: &HashSet<String>,
) -> String {
    let joined = format!("{} {}", first_normalized, middle_normalized);
    let mut parts: Vec<String> = joined.split(' ').map(|token| token.to_string()).collect();
    if let Some(prefix) = parts.first() {
        if name_prefixes.contains(prefix) {
            parts.remove(0);
        }
    }
    parts.get(0).cloned().unwrap_or_default()
}

fn is_name_dash(ch: char) -> bool {
    matches!(
        ch,
        '-' | '\u{2010}'
            | '\u{2011}'
            | '\u{2012}'
            | '\u{2013}'
            | '\u{2014}'
            | '\u{2212}'
            | '\u{FE58}'
            | '\u{FE63}'
            | '\u{FF0D}'
    )
}

pub(crate) fn contains_name_dash(value: &str) -> bool {
    value.chars().any(is_name_dash)
}

pub(crate) fn split_first_middle_hyphen_aware_compat(
    first_raw: &str,
    middle_raw: &str,
    name_prefixes: &HashSet<String>,
    unidecode_char_map: &HashMap<char, String>,
) -> (String, String) {
    let has_dash_in_first = contains_name_dash(first_raw);
    let first_noapos = normalize_text_compat_from_map(first_raw, true, unidecode_char_map);
    let middle_norm = normalize_text_compat_from_map(middle_raw, false, unidecode_char_map);

    let mut f_parts: Vec<String> = first_noapos
        .split_whitespace()
        .map(|token| token.to_string())
        .collect();
    let m_parts: Vec<String> = middle_norm
        .split_whitespace()
        .map(|token| token.to_string())
        .collect();
    if let Some(prefix) = f_parts.first() {
        if name_prefixes.contains(prefix) {
            f_parts.remove(0);
        }
    }

    if f_parts.is_empty() {
        return (String::new(), m_parts.join(" "));
    }
    if has_dash_in_first {
        return (f_parts.join(" "), m_parts.join(" "));
    }
    let first = f_parts[0].clone();
    let middle = f_parts[1..]
        .iter()
        .chain(m_parts.iter())
        .cloned()
        .collect::<Vec<_>>()
        .join(" ");
    (first, middle)
}

pub(crate) fn compute_block_compat(name: &str) -> String {
    if name.is_empty() {
        return String::new();
    }
    let name_parts: Vec<&str> = name.split(' ').collect();
    if name_parts.len() == 1 {
        return name_parts[0].to_string();
    }
    let Some(first_initial) = name_parts[0].chars().next() else {
        return String::new();
    };
    format!("{} {}", first_initial, name_parts[name_parts.len() - 1])
}
