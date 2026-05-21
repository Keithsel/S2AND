from __future__ import annotations

from itertools import combinations

from s2and.subblocking import _sorted_subblock_merge_candidates
from s2and.text import same_prefix_tokens


def _legacy_sorted_subblock_merge_candidates(output, maximum_size, first_k_letter_counts_sorted):
    small_enough_keys = [k for k, v in output.items() if len(v) < maximum_size]
    small_enough_pairs_counts = []
    for pair in list(combinations(small_enough_keys, 2)):
        if len(output[pair[0]]) + len(output[pair[1]]) < maximum_size:
            pair_0_split = pair[0].split("|")
            pair_1_split = pair[1].split("|")

            first_name_1 = pair_0_split[0]
            first_name_2 = pair_1_split[0]

            if len(pair_0_split) > 1:
                middle_name_1 = pair_0_split[1].split("=")[1]
            else:
                middle_name_1 = None

            if len(pair_1_split) > 1:
                middle_name_2 = pair_1_split[1].split("=")[1]
            else:
                middle_name_2 = None

            if len(first_name_1) > 1 and len(first_name_2) > 1:
                name_for_splits_1 = first_name_1
                name_for_splits_2 = first_name_2
            elif (
                len(first_name_1) == 1
                and len(first_name_2) == 1
                and middle_name_1 is not None
                and middle_name_2 is not None
            ):
                name_for_splits_1 = middle_name_1
                name_for_splits_2 = middle_name_2
            else:
                continue

            if name_for_splits_1 == name_for_splits_2:
                if middle_name_1 is not None and middle_name_2 is not None:
                    score = 0
                    for i in range(1, len(middle_name_1)):
                        if middle_name_1[:i] == middle_name_2[:i]:
                            score = i
                else:
                    score = 0
                small_enough_pairs_counts.append((pair, 1e10 + score))
            elif same_prefix_tokens(name_for_splits_1, name_for_splits_2):
                score = min(len(name_for_splits_1), len(name_for_splits_2))
                small_enough_pairs_counts.append((pair, 1e5 + score))
            else:
                lookup_1 = name_for_splits_1.split(" ")[0]
                lookup_2 = name_for_splits_2.split(" ")[0]
                if lookup_1 in first_k_letter_counts_sorted and lookup_2 in first_k_letter_counts_sorted[lookup_1]:
                    small_enough_pairs_counts.append((pair, first_k_letter_counts_sorted[lookup_1][lookup_2]))

    return sorted(small_enough_pairs_counts, key=lambda x: (x[1], x[0][0], x[0][1]), reverse=True)


def test_sorted_subblock_merge_candidates_matches_legacy_edge_cases() -> None:
    output = {
        "alex": ["a1", "a2"],
        "alex|middle=w": ["a3"],
        "alexander": ["a4"],
        "a|middle=wei": ["s1"],
        "a|middle=weijun": ["s2"],
        "a|middle=li": ["s3", "s4"],
        "b": ["skip"],
        "bo": ["b1"],
        "carol": ["c1", "c2", "c3"],
        "david": ["d1", "d2", "d3", "d4"],
    }
    counts = {
        "alex": {"bo": 7},
        "bo": {"alex": 99},
        "wei": {"li": 5},
        "li": {"wei": 11},
    }

    observed = _sorted_subblock_merge_candidates(output, maximum_size=5, first_k_letter_counts_sorted=counts)
    expected = _legacy_sorted_subblock_merge_candidates(output, maximum_size=5, first_k_letter_counts_sorted=counts)

    assert observed == expected
    assert (("carol", "david"), 0) not in observed
    assert any(pair == ("a|middle=wei", "a|middle=weijun") for pair, _score in observed)


def test_sorted_subblock_merge_candidates_matches_legacy_many_keys() -> None:
    output = {}
    counts = {}
    for index in range(80):
        if index % 3 == 0:
            key = f"al{index}"
        elif index % 3 == 1:
            key = f"a|middle=mi{index}"
        else:
            key = f"bo {index}"
        output[key] = [f"s{index}_{j}" for j in range(index % 4 + 1)]
    for left in range(0, 80, 5):
        counts.setdefault(f"al{left}", {})["bo"] = left + 1

    assert _sorted_subblock_merge_candidates(output, 7, counts) == _legacy_sorted_subblock_merge_candidates(
        output,
        7,
        counts,
    )
