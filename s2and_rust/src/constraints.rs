use std::collections::{HashMap, HashSet};

pub(crate) fn count_initials(s: &str) -> HashMap<char, usize> {
    let mut counts = HashMap::new();
    for part in s.split(' ') {
        if !part.is_empty() {
            if let Some(ch) = part.chars().next() {
                *counts.entry(ch).or_insert(0) += 1;
            }
        }
    }
    counts
}

pub(crate) fn lasts_equivalent_for_constraint(l1: &str, l2: &str) -> bool {
    if l1 == l2 {
        return true;
    }
    l1.replace(' ', "") == l2.replace(' ', "")
}

pub(crate) fn same_prefix_tokens(a: &str, b: &str) -> bool {
    let mut ita = a.split_whitespace();
    let mut itb = b.split_whitespace();
    loop {
        match (ita.next(), itb.next()) {
            (Some(x), Some(y)) => {
                if !(x.starts_with(y) || y.starts_with(x)) {
                    return false;
                }
            }
            _ => return true,
        }
    }
}

pub(crate) fn name_tuple_contains(
    map: &HashMap<String, HashSet<String>>,
    a: &str,
    b: &str,
) -> bool {
    map.get(a).map_or(false, |vals| vals.contains(b))
        || map.get(b).map_or(false, |vals| vals.contains(a))
}

fn first_name_forms(value: &str) -> (String, String, String) {
    let normalized = value.to_string();
    let parts: Vec<&str> = normalized.split_whitespace().collect();
    let joined = parts.join("");
    let token = parts
        .first()
        .map_or_else(|| normalized.clone(), |part| (*part).to_string());
    (normalized, joined, token)
}

pub(crate) fn first_names_name_compatible(
    first_1: &str,
    first_2: &str,
    name_tuples: &HashMap<String, HashSet<String>>,
) -> bool {
    if same_prefix_tokens(first_1, first_2) {
        return true;
    }
    let forms_1 = first_name_forms(first_1);
    let forms_2 = first_name_forms(first_2);
    name_tuple_contains(name_tuples, &forms_1.0, &forms_2.0)
        || name_tuple_contains(name_tuples, &forms_1.1, &forms_2.1)
        || name_tuple_contains(name_tuples, &forms_1.2, &forms_2.2)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn name_tuple_compatibility_does_not_apply_extra_case_normalization() {
        let name_tuples =
            HashMap::from([("Bill".to_string(), HashSet::from(["William".to_string()]))]);

        assert!(first_names_name_compatible("Bill", "William", &name_tuples));
        assert!(!first_names_name_compatible(
            "bill",
            "william",
            &name_tuples
        ));
    }
}
