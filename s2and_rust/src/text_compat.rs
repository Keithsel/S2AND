use pyo3::prelude::PyResult;
use std::collections::{HashMap, HashSet};

pub(crate) fn ensure_unidecode_for_text(
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
        unidecode_char_map.insert(ch, text_unidecode_char(ch).to_string());
    }
    Ok(())
}

fn text_unidecode_char(ch: char) -> &'static str {
    match ch as u32 {
        0x0080 | 0x0082 | 0x0083 | 0x0084 | 0x0085 | 0x0086 | 0x0087 | 0x0088 | 0x0089 | 0x008A
        | 0x008B | 0x008C | 0x008E | 0x0091 | 0x0092 | 0x0093 | 0x0094 | 0x0095 | 0x0096
        | 0x0097 | 0x0098 | 0x0099 | 0x009A | 0x009B | 0x009C | 0x009E | 0x009F | 0x02E5
        | 0x02E6 | 0x02E7 | 0x02E8 | 0x02E9 | 0x02EA | 0x02EB | 0xFDF0 | 0xFDF1 | 0xFDF2
        | 0xFDF3 | 0xFDF4 | 0xFDF5 | 0xFDF6 | 0xFDF7 | 0xFDF8 | 0xFDF9 | 0xFDFA | 0xFDFB => "",
        0x02FF | 0x03FF | 0x04FF | 0x05FF | 0x06FF | 0x07FF | 0x09FF | 0x0AFF | 0x0BFF | 0x0CFF
        | 0x0DFF | 0x0EFF | 0x0FFF | 0x10FF | 0x11FF | 0x13FF | 0x16FF | 0x17FF | 0x18FF
        | 0x1EFF | 0x1FFF | 0x20FF | 0x21FF | 0x22FF | 0x23FF | 0x24FF | 0x25FF | 0x26FF
        | 0x27FF | 0x2EFF | 0x2FFF | 0x30FF | 0x31FF | 0x32FF | 0x33FF | 0x4DFF | 0x9FFF
        | 0xA4FF | 0xD7FF | 0xFAFF | 0xFDFF => "[?] ",
        0x25F4 | 0x25F5 | 0x25F6 | 0x25F7 => "#",
        0x02EF | 0x02F0 | 0x02F1 | 0x02F2 | 0x02F3 | 0x02F4 | 0x02F5 | 0x02F6 | 0x02F7 | 0x02F8
        | 0x02F9 | 0x02FA | 0x02FB | 0x02FC | 0x02FD | 0x02FE | 0x03F4 | 0x03F5 | 0x03F6
        | 0x03F7 | 0x03F8 | 0x03F9 | 0x03FC | 0x03FD | 0x03FE | 0x0AF0 | 0x0AF1 | 0x0AF9
        | 0x13F5 | 0x13F8 | 0x13F9 | 0x13FA | 0x13FB | 0x13FC | 0x13FD | 0x1EFA | 0x1EFB
        | 0x1EFC | 0x1EFD | 0x1EFE | 0x25F8 | 0x25F9 | 0x25FA | 0x25FB | 0x25FC | 0x25FD
        | 0x25FE | 0xFDFC | 0xFDFD => "[?]",
        _ => unidecode::unidecode_char(ch),
    }
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

pub(crate) fn normalize_text_compat_native(text: &str, special_case_apostrophes: bool) -> String {
    normalize_text_compat_with_map(text, special_case_apostrophes, None)
}

pub(crate) fn normalize_text_compat_from_map(
    text: &str,
    special_case_apostrophes: bool,
    unidecode_char_map: &HashMap<char, String>,
) -> String {
    normalize_text_compat_with_map(text, special_case_apostrophes, Some(unidecode_char_map))
}

fn normalize_text_compat_with_map(
    text: &str,
    special_case_apostrophes: bool,
    unidecode_char_map: Option<&HashMap<char, String>>,
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
        let mapped = unidecode_char_map
            .and_then(|char_map| char_map.get(&ch).map(String::as_str))
            .unwrap_or_else(|| text_unidecode_char(ch));
        for mapped_ch in mapped.chars() {
            transliterated.push(mapped_ch.to_ascii_lowercase());
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

pub(crate) fn contains_non_ascii_name_dash(value: &str) -> bool {
    value.chars().any(|ch| ch != '-' && is_name_dash(ch))
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

#[cfg(test)]
mod tests {
    use super::{normalize_text_compat_native, text_unidecode_char};

    #[test]
    fn text_unidecode_char_matches_python_text_unidecode_compat_overrides() {
        assert_eq!(text_unidecode_char('\u{0080}'), "");
        assert_eq!(text_unidecode_char('\u{02EF}'), "[?]");
        assert_eq!(text_unidecode_char('\u{02FF}'), "[?] ");
        assert_eq!(text_unidecode_char('\u{25F4}'), "#");
        assert_eq!(text_unidecode_char('\u{FDFD}'), "[?]");
    }

    #[test]
    fn normalize_text_compat_preserves_python_boundaries_for_overrides() {
        assert_eq!(
            normalize_text_compat_native("a\u{0080}b \u{03F4}c \u{02FF}d", false),
            "ab c d",
        );
    }
}
