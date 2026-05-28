fn is_orcid_dash(ch: char) -> bool {
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

pub(crate) fn normalize_orcid_owned(value: &str) -> Option<String> {
    let chars: Vec<char> = value.trim().chars().collect();
    for start in 0..chars.len() {
        if !chars[start].is_ascii_digit() {
            continue;
        }
        if start > 0
            && (chars[start - 1].is_ascii_digit()
                || chars[start - 1] == 'X'
                || chars[start - 1] == 'x')
        {
            continue;
        }
        let mut compact = String::with_capacity(16);
        let mut idx = start;
        let mut valid = true;
        for (group_index, group_len) in [4usize, 4, 4, 3].iter().enumerate() {
            for _ in 0..*group_len {
                if idx >= chars.len() || !chars[idx].is_ascii_digit() {
                    valid = false;
                    break;
                }
                compact.push(chars[idx]);
                idx += 1;
            }
            if !valid {
                break;
            }
            if group_index < 3 && idx < chars.len() && is_orcid_dash(chars[idx]) {
                idx += 1;
            }
        }
        if !valid || idx >= chars.len() {
            continue;
        }
        let check_digit = chars[idx];
        if !(check_digit.is_ascii_digit() || check_digit == 'X' || check_digit == 'x') {
            continue;
        }
        compact.push(check_digit.to_ascii_uppercase());
        idx += 1;
        let bytes = compact.as_bytes();
        if bytes.len() == 16
            && bytes[..15].iter().all(|byte| byte.is_ascii_digit())
            && (bytes[15].is_ascii_digit() || bytes[15] == b'X')
            && (idx == chars.len()
                || !(chars[idx].is_ascii_digit() || chars[idx] == 'X' || chars[idx] == 'x'))
        {
            return Some(format!(
                "{}-{}-{}-{}",
                &compact[0..4],
                &compact[4..8],
                &compact[8..12],
                &compact[12..16]
            ));
        }
    }
    None
}

pub(crate) fn normalize_orcid_compact_owned(value: &str) -> Option<String> {
    normalize_orcid_owned(value).map(|orcid| orcid.replace('-', ""))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn orcid_normalization_canonicalizes_common_forms() {
        assert_eq!(
            normalize_orcid_owned(" https://orcid.org/0000-0002-1825-0097 "),
            Some("0000-0002-1825-0097".to_string())
        );
        assert_eq!(
            normalize_orcid_owned("ORCID: 000000021825009x"),
            Some("0000-0002-1825-009X".to_string())
        );
        assert_eq!(
            normalize_orcid_owned("https://orcid.org/0000\u{2010}0002\u{2010}1825\u{2010}0097"),
            Some("0000-0002-1825-0097".to_string())
        );
        for dash in [
            '-', '\u{2010}', '\u{2011}', '\u{2012}', '\u{2013}', '\u{2014}', '\u{2212}',
            '\u{FE58}', '\u{FE63}', '\u{FF0D}',
        ] {
            let value = format!("0000{dash}0002{dash}1825{dash}0097");
            assert_eq!(
                normalize_orcid_owned(&value),
                Some("0000-0002-1825-0097".to_string())
            );
        }
        assert_eq!(
            normalize_orcid_compact_owned("ORCID: 000000021825009x"),
            Some("000000021825009X".to_string())
        );
        assert_eq!(normalize_orcid_owned("s000-0000-1879-1075X"), None);
        assert_eq!(normalize_orcid_owned("0000-0002-1825"), None);
    }
}
