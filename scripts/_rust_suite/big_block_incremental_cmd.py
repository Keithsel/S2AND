from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

# When this module is executed directly (including via the compare-mode subprocess
# path), ensure `scripts/` is importable so `import _rust_suite.*` works.
_SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from _rust_suite.common import (  # type: ignore  # noqa: E402, I001
    PROJECT_ROOT,
    ProcessTreeRSSMonitor,
    cluster_membership_digest as _cluster_membership_digest,
    collect_rust_extension_identity,
    extract_marked_json_payload,
    get_result_markers,
    signature_to_cluster_fingerprint_map as _signature_to_cluster_fingerprint_map,
)

RESULT_JSON_START, RESULT_JSON_END = get_result_markers("big_block")
DEFAULT_TOTAL_RAM_BYTES = 32 * 1024 * 1024 * 1024


def _optional_fraction(numerator: float, denominator: float) -> float | None:
    if denominator <= 0:
        return None
    return round(float(numerator) / float(denominator), 6)


def _optional_speedup(baseline_seconds: float, candidate_seconds: float) -> float | None:
    if candidate_seconds <= 0:
        return None
    return round(float(baseline_seconds) / float(candidate_seconds), 6)


def _resolve_existing_file(base_dir: Path, candidates: list[str]) -> Path:
    for candidate in candidates:
        path = base_dir / candidate
        if path.exists():
            return path
    raise FileNotFoundError(f"Missing required file under {base_dir}: tried {candidates}")


def _resolve_optional_existing_file(explicit_path: str) -> Path | None:
    if explicit_path:
        path = Path(explicit_path)
        if not path.exists():
            raise FileNotFoundError(f"Configured file does not exist: {path}")
        return path
    return None


def _load_subset_payload(
    subset_dir: Path,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]], str | None]:
    signatures_path = _resolve_existing_file(
        subset_dir,
        [
            "signatures.json",
            f"{subset_dir.name}_signatures.json",
        ],
    )
    papers_path = _resolve_existing_file(
        subset_dir,
        [
            "papers.json",
            f"{subset_dir.name}_papers.json",
        ],
    )
    with signatures_path.open("r", encoding="utf-8") as infile:
        signatures = json.load(infile)
    with papers_path.open("r", encoding="utf-8") as infile:
        papers = json.load(infile)

    meta_path = subset_dir / "meta.json"
    target_block = None
    if meta_path.exists():
        with meta_path.open("r", encoding="utf-8") as infile:
            meta = json.load(infile)
        target_block = str(meta.get("target_block")) if meta.get("target_block") else None

    return signatures, papers, target_block


def _resolve_target_block(signatures: dict[str, dict[str, Any]], target_block: str | None) -> str:
    if target_block:
        return target_block
    block_counts = Counter(
        str(payload.get("author_info", {}).get("block", ""))
        for payload in signatures.values()
        if payload.get("author_info", {}).get("block") is not None
    )
    if not block_counts:
        raise RuntimeError("No signature blocks found while resolving target block")
    return str(block_counts.most_common(1)[0][0])


def _select_block_signature_ids(
    signatures: dict[str, dict[str, Any]],
    papers: dict[str, dict[str, Any]],
    target_block: str,
    total_signatures: int,
    random_seed: int,
) -> list[str]:
    in_block = [
        signature_id
        for signature_id, payload in signatures.items()
        if str(payload.get("author_info", {}).get("block", "")) == target_block
    ]
    if len(in_block) < total_signatures:
        raise ValueError(
            f"Requested total_signatures={total_signatures} but block '{target_block}' "
            f"has only {len(in_block)} signatures."
        )
    rng = random.Random(random_seed)
    in_block_sorted = sorted(in_block)
    rng.shuffle(in_block_sorted)
    selected: list[str] = []
    skipped_invalid_paper = 0
    for signature_id in in_block_sorted:
        paper_id = str(signatures[signature_id].get("paper_id", ""))
        paper_payload = papers.get(paper_id)
        if paper_payload is None:
            skipped_invalid_paper += 1
            continue
        if not _paper_has_block_safe_author_names(paper_payload):
            skipped_invalid_paper += 1
            continue
        selected.append(signature_id)
        if len(selected) >= total_signatures:
            break
    if len(selected) < total_signatures:
        raise ValueError(
            f"Requested total_signatures={total_signatures}, but only {len(selected)} block-safe signatures were found "
            f"in block '{target_block}' (skipped_invalid_paper={skipped_invalid_paper})."
        )
    return selected


def _paper_has_block_safe_author_names(paper_payload: dict[str, Any]) -> bool:
    from s2and.text import compute_block

    authors = paper_payload.get("authors")
    if not isinstance(authors, list):
        return True
    for author in authors:
        if isinstance(author, dict):
            author_name = author.get("author_name", "")
        else:
            author_name = str(author)
        if str(author_name).strip() == "":
            return False
        try:
            compute_block(str(author_name))
        except Exception:
            return False
    return True


def _effective_seed_cluster_count(seed_signature_count: int, requested_seed_clusters: int) -> int:
    if seed_signature_count < 2:
        raise ValueError("Need at least 2 seed signatures")
    if requested_seed_clusters <= 0:
        raise ValueError("requested_seed_clusters must be > 0")
    effective = min(requested_seed_clusters, seed_signature_count // 2)
    if effective <= 0:
        raise ValueError("Unable to form seed clusters with at least 2 signatures each")
    return effective


def _build_cluster_seeds(seed_signature_ids: list[str], seed_cluster_count: int) -> dict[str, dict[str, str]]:
    if seed_cluster_count <= 0:
        raise ValueError("seed_cluster_count must be > 0")
    if len(seed_signature_ids) < 2 * seed_cluster_count:
        raise ValueError(
            "seed_signature_ids must have at least 2 signatures per seed cluster; "
            f"got signatures={len(seed_signature_ids)} clusters={seed_cluster_count}"
        )

    cluster_seeds: dict[str, dict[str, str]] = {}
    base_size = len(seed_signature_ids) // seed_cluster_count
    remainder = len(seed_signature_ids) % seed_cluster_count
    cursor = 0
    for cluster_idx in range(seed_cluster_count):
        cluster_size = base_size + (1 if cluster_idx < remainder else 0)
        members = seed_signature_ids[cursor : cursor + cluster_size]
        cursor += cluster_size
        if len(members) < 2:
            raise ValueError(f"Cluster {cluster_idx} has <2 members; invalid seed partition")
        root = members[0]
        cluster_seeds[root] = {member: "require" for member in members[1:]}
    return cluster_seeds


def _cluster_size_summary(cluster_to_signatures: dict[str, list[str]]) -> dict[str, Any]:
    sizes = sorted((len(signatures) for signatures in cluster_to_signatures.values()), reverse=True)
    return {
        "cluster_count": int(len(sizes)),
        "largest_clusters_top10": [int(size) for size in sizes[:10]],
        "max_cluster_size": int(sizes[0]) if sizes else 0,
        "min_cluster_size": int(sizes[-1]) if sizes else 0,
    }


def _truth_enabled(args: argparse.Namespace) -> bool:
    return bool(str(getattr(args, "truth_bundle_root", "")).strip() or str(getattr(args, "truth_dataset", "")).strip())


def _auto_truth_table_key(dataset_name: str) -> str:
    if dataset_name == "h_wang":
        return "hwang_eval_path"
    if dataset_name == "s_lee":
        return "s_lee_eval_path"
    if dataset_name == "s_park":
        return "s_park_eval_path"
    if dataset_name in {"a_khan", "a_silva", "j_smith", "s_gupta"}:
        return f"extra_eval_paths.{dataset_name}"
    return f"extra_eval_paths.{dataset_name}"


def _resolve_truth_asset_path(bundle_root: Path, asset_group: str, table_key: str) -> Path:
    bundle_path = bundle_root / "bundle.json"
    if not bundle_path.exists():
        raise FileNotFoundError(f"Truth bundle is missing bundle.json: {bundle_path}")
    with bundle_path.open("r", encoding="utf-8") as infile:
        bundle = json.load(infile)
    files = bundle.get("assets", {}).get(asset_group, {}).get("files", {})
    relpath = files.get(table_key)
    if relpath is None:
        available = ", ".join(sorted(str(key) for key in files))
        raise ValueError(f"Truth table key {table_key!r} is not in {asset_group}; available keys: {available}")
    path = bundle_root / str(relpath)
    if not path.exists():
        raise FileNotFoundError(f"Truth asset does not exist: {path}")
    return path


def _load_truth_bundle_inputs(args: argparse.Namespace) -> dict[str, Any]:
    import pandas as pd

    bundle_root = Path(str(args.truth_bundle_root))
    dataset_name = str(args.truth_dataset)
    table_key = str(args.truth_table_key or _auto_truth_table_key(dataset_name))
    labels_path = _resolve_truth_asset_path(bundle_root, "featureless_rows", table_key)
    assignments_path = bundle_root / "splits" / "combined_query_split_assignments.csv"
    raw_dir = bundle_root / "raw" / dataset_name
    components_path = bundle_root / "components" / f"{dataset_name}_members.parquet"
    signatures_path = raw_dir / "signatures.json"
    papers_path = raw_dir / "papers.json"
    for path in [assignments_path, components_path, signatures_path, papers_path]:
        if not path.exists():
            raise FileNotFoundError(f"Truth bundle input does not exist: {path}")

    labels = pd.read_parquet(
        labels_path,
        columns=[
            "dataset",
            "query_group_id",
            "query_signature_id",
            "candidate_component_key",
            "label",
            "retrieval_rank",
            "source_key",
        ],
    )
    labels = labels[labels["dataset"].astype(str) == dataset_name].copy()
    if labels.empty:
        raise ValueError(f"No labels for dataset={dataset_name!r} in {labels_path}")

    assignments = pd.read_csv(
        assignments_path,
        usecols=["query_group_id", "source_key", "split"],
        dtype=str,
    ).rename(columns={"split": "assigned_split"})
    label_rows_before_assignment = len(labels)
    labels = labels.merge(assignments, on=["query_group_id", "source_key"], how="left", indicator=True)
    missing_assignment = labels["_merge"].ne("both")
    if bool(missing_assignment.any()):
        examples = labels.loc[missing_assignment, ["query_group_id", "source_key"]].head(5).to_dict("records")
        raise ValueError(
            "Truth labels are missing split assignments: "
            f"missing={int(missing_assignment.sum())} total={label_rows_before_assignment} examples={examples}"
        )
    labels = labels.drop(columns=["_merge"])
    truth_split = str(args.truth_split)
    if truth_split != "any":
        labels = labels[labels["assigned_split"].astype(str) == truth_split].copy()
    retrieval_rank = pd.to_numeric(labels["retrieval_rank"], errors="coerce")
    bad_retrieval_rank = retrieval_rank.isna()
    if bool(bad_retrieval_rank.any()):
        examples = labels.loc[bad_retrieval_rank, ["query_group_id", "source_key", "retrieval_rank"]].head(5).to_dict(
            "records"
        )
        raise ValueError(
            "Truth labels contain invalid retrieval_rank values: "
            f"invalid={int(bad_retrieval_rank.sum())} examples={examples}"
        )
    labels["retrieval_rank"] = retrieval_rank.astype(int)
    labels = labels[labels["retrieval_rank"] <= int(args.truth_max_candidates_per_query)].copy()
    if labels.empty:
        raise ValueError(
            "No truth rows remain after split/rank filtering: "
            f"dataset={dataset_name!r} split={truth_split!r} max_rank={args.truth_max_candidates_per_query}"
        )

    with signatures_path.open("r", encoding="utf-8") as infile:
        signatures = json.load(infile)
    with papers_path.open("r", encoding="utf-8") as infile:
        papers = json.load(infile)

    def _query_block(signature_id: Any) -> str:
        return str(signatures[str(signature_id)].get("author_info", {}).get("block", ""))

    query_blocks = {
        str(signature_id): _query_block(signature_id)
        for signature_id in labels["query_signature_id"].astype(str).drop_duplicates()
        if str(signature_id) in signatures
    }
    labels = labels[labels["query_signature_id"].astype(str).isin(query_blocks)].copy()
    if labels.empty:
        raise ValueError("No truth rows have query signatures present in the raw signature metadata")
    labels["_query_block"] = labels["query_signature_id"].astype(str).map(query_blocks)
    target_block = str(args.target_block or labels["_query_block"].value_counts().idxmax())
    labels = labels[labels["_query_block"].astype(str) == target_block].copy()
    if labels.empty:
        raise ValueError(f"No truth rows remain for target block {target_block!r}")

    query_groups = sorted(labels["query_group_id"].astype(str).drop_duplicates().tolist())
    rng = random.Random(int(args.random_seed))
    rng.shuffle(query_groups)
    selected_query_groups = set(query_groups[: int(args.truth_query_limit)])
    labels = labels[labels["query_group_id"].astype(str).isin(selected_query_groups)].copy()

    components = sorted(labels["candidate_component_key"].astype(str).drop_duplicates().tolist())
    members = pd.read_parquet(components_path, columns=["candidate_component_key", "member_index", "signature_id"])
    members = members[members["candidate_component_key"].astype(str).isin(components)].copy()
    member_index = pd.to_numeric(members["member_index"], errors="coerce")
    bad_member_index = member_index.isna()
    if bool(bad_member_index.any()):
        examples = members.loc[bad_member_index, ["candidate_component_key", "member_index", "signature_id"]].head(
            5
        ).to_dict("records")
        raise ValueError(
            "Truth component members contain invalid member_index values: "
            f"invalid={int(bad_member_index.sum())} examples={examples}"
        )
    members["member_index"] = member_index.astype(int)
    members = members.sort_values(["candidate_component_key", "member_index"], kind="stable")
    if int(args.truth_max_component_members) > 0:
        members = members.groupby("candidate_component_key", sort=False).head(int(args.truth_max_component_members))

    query_signature_ids = set(labels["query_signature_id"].astype(str).tolist())
    seed_signature_to_component: dict[str, str] = {}
    component_to_seed_signatures: dict[str, list[str]] = {}
    for row in members.itertuples(index=False):
        signature_id = str(row.signature_id)
        component_key = str(row.candidate_component_key)
        if signature_id in query_signature_ids or signature_id not in signatures:
            continue
        seed_signature_to_component[signature_id] = component_key
        component_to_seed_signatures.setdefault(component_key, []).append(signature_id)

    seedable_components = set(component_to_seed_signatures)
    labels = labels[labels["candidate_component_key"].astype(str).isin(seedable_components)].copy()
    if labels.empty:
        raise ValueError("No truth rows remain after removing components with no seed signatures")

    truth_queries: dict[str, dict[str, Any]] = {}
    for query_signature_id, group in labels.groupby(labels["query_signature_id"].astype(str), sort=False):
        candidate_components = set(group["candidate_component_key"].astype(str).tolist())
        positive_components = set(group.loc[group["label"].astype(int) == 1, "candidate_component_key"].astype(str))
        truth_queries[str(query_signature_id)] = {
            "query_group_id": str(group["query_group_id"].iloc[0]),
            "candidate_components": sorted(candidate_components),
            "positive_components": sorted(positive_components),
        }

    cluster_seeds_require = {
        signature_id: component_key for signature_id, component_key in seed_signature_to_component.items()
    }
    selected_signature_ids = sorted(seed_signature_to_component) + sorted(truth_queries)
    selected_paper_ids = {
        str(signatures[signature_id].get("paper_id", ""))
        for signature_id in selected_signature_ids
        if signature_id in signatures and signatures[signature_id].get("paper_id") is not None
    }
    selected_papers = {paper_id: payload for paper_id, payload in papers.items() if str(paper_id) in selected_paper_ids}
    selected_signatures = {signature_id: signatures[signature_id] for signature_id in selected_signature_ids}

    specter_path = bundle_root / "embeddings" / dataset_name / "specter2.pkl"
    return {
        "signatures": selected_signatures,
        "papers": selected_papers,
        "target_block": target_block,
        "selected_signature_ids": selected_signature_ids,
        "cluster_seeds_require": cluster_seeds_require,
        "seed_signature_to_component": seed_signature_to_component,
        "truth_queries": truth_queries,
        "truth_bundle_root": str(bundle_root),
        "truth_dataset": dataset_name,
        "truth_table_key": table_key,
        "truth_labels_path": str(labels_path),
        "truth_split": truth_split,
        "truth_query_count": len(truth_queries),
        "truth_seed_component_count": len(seedable_components),
        "truth_seed_signature_count": len(seed_signature_to_component),
        "specter_path": str(specter_path) if specter_path.exists() else "",
    }


def _signature_to_predicted_cluster(pred_clusters: dict[str, list[str]]) -> dict[str, str]:
    out: dict[str, str] = {}
    for cluster_id, signature_ids in pred_clusters.items():
        for signature_id in signature_ids:
            out[str(signature_id)] = str(cluster_id)
    return out


def _evaluate_truth_link_quality(
    pred_clusters: dict[str, list[str]],
    truth_context: dict[str, Any],
) -> dict[str, Any]:
    signature_to_cluster = _signature_to_predicted_cluster(pred_clusters)
    seed_signature_to_component = {
        str(signature_id): str(component_key)
        for signature_id, component_key in truth_context["seed_signature_to_component"].items()
    }
    truth_queries = truth_context["truth_queries"]
    cluster_members = {str(cluster_id): [str(sig) for sig in sigs] for cluster_id, sigs in pred_clusters.items()}
    counts = Counter()
    multi_component_links = 0
    linked_positive_and_negative = 0
    examples: list[dict[str, Any]] = []

    for query_signature_id, truth in truth_queries.items():
        positive_components = set(str(component) for component in truth["positive_components"])
        candidate_components = set(str(component) for component in truth["candidate_components"])
        cluster_id = signature_to_cluster.get(str(query_signature_id))
        linked_components: set[str] = set()
        if cluster_id is not None:
            linked_components = {
                seed_signature_to_component[signature_id]
                for signature_id in cluster_members.get(cluster_id, [])
                if signature_id in seed_signature_to_component
            }
        linked_candidate_components = linked_components.intersection(candidate_components)
        positive_links = linked_candidate_components.intersection(positive_components)
        negative_links = linked_candidate_components.difference(positive_components)
        has_positive = bool(positive_components)
        linked = bool(linked_candidate_components)

        if len(linked_candidate_components) > 1:
            multi_component_links += 1
        if positive_links and negative_links:
            linked_positive_and_negative += 1

        if linked and positive_links:
            outcome = "correct_link"
        elif linked and has_positive:
            outcome = "wrong_link"
        elif linked:
            outcome = "false_link"
        elif has_positive:
            outcome = "false_abstain"
        else:
            outcome = "correct_abstain"
        counts[outcome] += 1

        if outcome != "correct_link" and len(examples) < 10:
            examples.append(
                {
                    "query_signature_id": str(query_signature_id),
                    "query_group_id": str(truth["query_group_id"]),
                    "outcome": outcome,
                    "positive_components": sorted(positive_components),
                    "linked_components": sorted(linked_candidate_components),
                }
            )

    evaluated = int(sum(counts.values()))
    linked_count = int(counts["correct_link"] + counts["wrong_link"] + counts["false_link"])
    positive_query_count = int(counts["correct_link"] + counts["wrong_link"] + counts["false_abstain"])
    no_positive_query_count = int(counts["correct_abstain"] + counts["false_link"])
    precision = _optional_fraction(float(counts["correct_link"]), float(linked_count))
    recall = _optional_fraction(float(counts["correct_link"]), float(positive_query_count))
    if precision is None or recall is None or precision + recall <= 0:
        f1 = None
    else:
        f1 = round(2 * precision * recall / (precision + recall), 6)

    return {
        "source": "s2and_and_big_blocks_linker_dataset_20260513",
        "dataset": str(truth_context["truth_dataset"]),
        "table_key": str(truth_context["truth_table_key"]),
        "split": str(truth_context["truth_split"]),
        "target_block": str(truth_context["target_block"]),
        "evaluated_queries": evaluated,
        "positive_query_count": positive_query_count,
        "no_positive_query_count": no_positive_query_count,
        "linked_query_count": linked_count,
        "correct_link": int(counts["correct_link"]),
        "wrong_link": int(counts["wrong_link"]),
        "false_abstain": int(counts["false_abstain"]),
        "correct_abstain": int(counts["correct_abstain"]),
        "false_link": int(counts["false_link"]),
        "link_precision": precision,
        "link_recall": recall,
        "link_f1": f1,
        "accuracy": _optional_fraction(float(counts["correct_link"] + counts["correct_abstain"]), float(evaluated)),
        "false_abstain_rate": _optional_fraction(float(counts["false_abstain"]), float(positive_query_count)),
        "false_link_rate": _optional_fraction(float(counts["false_link"]), float(no_positive_query_count)),
        "multi_component_link_count": multi_component_links,
        "linked_positive_and_negative_count": linked_positive_and_negative,
        "non_correct_examples": examples,
    }


def _promoted_measurement_fields(
    *,
    broad_seed_query_pairs: int,
    promoted_telemetry: dict[str, Any],
    phase_b_residual_count: int,
    phase_b_required_bytes: int,
) -> dict[str, Any]:
    promoted_candidate_rows = int(promoted_telemetry.get("candidate_row_count", 0))
    promoted_scored_pairs = int(promoted_telemetry.get("pair_count", 0))
    promoted_measurement_available = bool(promoted_telemetry) and (
        promoted_candidate_rows > 0 or promoted_scored_pairs > 0
    )
    if promoted_measurement_available:
        residual_tail_pair_count = int(phase_b_residual_count * max(0, phase_b_residual_count - 1) // 2)
        promoted_scored_pair_reduction = int(broad_seed_query_pairs - promoted_scored_pairs)
        promoted_scored_pair_reduction_fraction = _optional_fraction(
            promoted_scored_pair_reduction,
            broad_seed_query_pairs,
        )
        promoted_candidate_row_ratio = _optional_fraction(promoted_candidate_rows, broad_seed_query_pairs)
        promoted_scored_pair_ratio = _optional_fraction(promoted_scored_pairs, broad_seed_query_pairs)
        residual_tail_matrix_bytes = int(phase_b_required_bytes)
    else:
        residual_tail_pair_count = None
        residual_tail_matrix_bytes = None
        promoted_scored_pair_reduction = None
        promoted_scored_pair_reduction_fraction = None
        promoted_candidate_row_ratio = None
        promoted_scored_pair_ratio = None

    return {
        "measurement_contract": (
            "promoted_rust_predict_incremental" if promoted_measurement_available else "legacy_predict_incremental"
        ),
        "promoted_measurement_available": promoted_measurement_available,
        "broad_seed_query_pairs": int(broad_seed_query_pairs),
        "promoted_candidate_rows": promoted_candidate_rows,
        "promoted_scored_pairs": promoted_scored_pairs,
        "promoted_scored_pair_reduction": promoted_scored_pair_reduction,
        "promoted_scored_pair_reduction_fraction": promoted_scored_pair_reduction_fraction,
        "promoted_candidate_row_ratio_vs_broad_pairs": promoted_candidate_row_ratio,
        "promoted_scored_pair_ratio_vs_broad_pairs": promoted_scored_pair_ratio,
        "residual_tail_pair_count": residual_tail_pair_count,
        "residual_tail_matrix_bytes": residual_tail_matrix_bytes,
    }


def _print_single_summary(result: dict[str, Any]) -> None:
    print("Incremental measurement summary:")
    print(f"1. Backend: {result['backend']} | contract: {result['measurement_contract']}")
    print(
        "2. Runtime: "
        f"predict={result['predict_seconds']}s | total={result['total_runtime_seconds']}s | "
        f"peak RSS={result['peak_rss_gb']} GB"
    )
    print(
        "3. Workload: "
        f"seeds={result['seed_signatures']} | unassigned={result['unassigned_signatures']} | "
        f"broad seed/query pairs={result['broad_seed_query_pairs']}"
    )
    if result.get("promoted_measurement_available"):
        print(
            "4. Promoted candidate work: "
            f"candidate rows={result['promoted_candidate_rows']} | "
            f"scored pairs={result['promoted_scored_pairs']} | "
            f"pair reduction={result['promoted_scored_pair_reduction_fraction']}"
        )
        print(
            "5. Promoted decisions: "
            f"links={result['promoted_link_count']} | abstains={result['promoted_abstain_count']} | "
            f"query batches={result['promoted_query_batch_count']}"
        )
    else:
        print("4. Promoted candidate work: not available for this run")
    if result.get("promoted_measurement_available"):
        print(
            "6. Residual tail: "
            f"queries={result['phase_b_residual_count']} | pairs={result['residual_tail_pair_count']} | "
            f"matrix bytes={result['residual_tail_matrix_bytes']} | phase_b_mode={result['phase_b_mode']}"
        )
    else:
        print(
            "6. Exact incremental path: "
            f"phase_b_required_bytes={result['phase_b_required_bytes']} | phase_b_mode={result['phase_b_mode']}"
        )
    truth_quality = result.get("truth_quality")
    if isinstance(truth_quality, dict):
        print(
            "7. Truth-bundle quality: "
            f"precision={truth_quality.get('link_precision')} | recall={truth_quality.get('link_recall')} | "
            f"f1={truth_quality.get('link_f1')} | accuracy={truth_quality.get('accuracy')} | "
            f"evaluated={truth_quality.get('evaluated_queries')}"
        )


def _set_runtime_env(
    *,
    backend: str,
    n_jobs: int,
) -> dict[str, str | None]:
    if backend not in {"python", "rust", "auto"}:
        raise ValueError(f"Unsupported backend={backend}")
    prior_values = {
        "S2AND_BACKEND": os.environ.get("S2AND_BACKEND"),
        "S2AND_SKIP_FASTTEXT": os.environ.get("S2AND_SKIP_FASTTEXT"),
        "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
    }
    os.environ["S2AND_BACKEND"] = backend
    os.environ["S2AND_SKIP_FASTTEXT"] = "1"
    os.environ["OMP_NUM_THREADS"] = str(max(1, n_jobs))
    return prior_values


def _restore_runtime_env(prior_values: dict[str, str | None]) -> None:
    for name, prior_value in prior_values.items():
        if prior_value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = prior_value


def _validate_args(args: argparse.Namespace) -> None:
    if _truth_enabled(args):
        if not str(args.truth_bundle_root).strip():
            raise ValueError("--truth-bundle-root is required when using truth-bundle evaluation")
        if not str(args.truth_dataset).strip():
            raise ValueError("--truth-dataset is required when using truth-bundle evaluation")
        if args.truth_query_limit <= 0:
            raise ValueError("--truth-query-limit must be > 0")
        if args.truth_max_candidates_per_query <= 0:
            raise ValueError("--truth-max-candidates-per-query must be > 0")
        if args.truth_max_component_members < 0:
            raise ValueError("--truth-max-component-members must be >= 0")
    if args.total_signatures <= 0:
        raise ValueError("--total-signatures must be > 0")
    if getattr(args, "cluster_seeds_path", ""):
        if args.seed_signatures < 0:
            raise ValueError("--seed-signatures must be >= 0 when --cluster-seeds-path is supplied")
        if args.seed_cluster_count < 0:
            raise ValueError("--seed-cluster-count must be >= 0 when --cluster-seeds-path is supplied")
    else:
        if args.seed_signatures <= 0:
            raise ValueError("--seed-signatures must be > 0")
        if args.seed_signatures >= args.total_signatures:
            raise ValueError("--seed-signatures must be < --total-signatures")
        if args.seed_cluster_count <= 0:
            raise ValueError("--seed-cluster-count must be > 0")
    if args.n_jobs <= 0:
        raise ValueError("--n-jobs must be > 0")
    if args.batching_threshold <= 0:
        raise ValueError("--batching-threshold must be > 0")
    if args.total_signatures > 4000 and not bool(args.full_run):
        raise ValueError("Refusing large run without explicit confirmation. Use --full-run for >4000 signatures.")


def _summarize_incremental_seed_state(
    *,
    selected_signature_ids: list[str],
    anddata: Any,
) -> dict[str, Any]:
    selected_signature_set = {str(signature_id) for signature_id in selected_signature_ids}
    cluster_seeds_require = {
        str(signature_id): cluster_id
        for signature_id, cluster_id in getattr(anddata, "cluster_seeds_require", {}).items()
    }
    invalid_seed_signatures = sorted(
        signature_id for signature_id in cluster_seeds_require if signature_id not in selected_signature_set
    )
    if invalid_seed_signatures:
        raise ValueError(
            "External cluster seeds reference signatures outside the selected subset: "
            f"{invalid_seed_signatures[:10]}"
        )

    cluster_seeds_disallow = {
        (str(signature_id_a), str(signature_id_b))
        for signature_id_a, signature_id_b in getattr(anddata, "cluster_seeds_disallow", set())
    }
    invalid_disallow_signatures = sorted(
        {
            signature_id
            for pair in cluster_seeds_disallow
            for signature_id in pair
            if signature_id not in selected_signature_set
        }
    )
    if invalid_disallow_signatures:
        raise ValueError(
            "External cluster seed disallow constraints reference signatures outside the selected subset: "
            f"{invalid_disallow_signatures[:10]}"
        )

    altered_cluster_signatures = getattr(anddata, "altered_cluster_signatures", None)
    altered_signature_ids = (
        sorted(str(signature_id) for signature_id in altered_cluster_signatures) if altered_cluster_signatures else []
    )
    invalid_altered_signatures = sorted(
        signature_id for signature_id in altered_signature_ids if signature_id not in selected_signature_set
    )
    if invalid_altered_signatures:
        raise ValueError(
            "Altered cluster signatures reference signatures outside the selected subset: "
            f"{invalid_altered_signatures[:10]}"
        )

    seed_signature_ids = [
        str(signature_id) for signature_id in selected_signature_ids if str(signature_id) in cluster_seeds_require
    ]
    unassigned_signature_ids = [
        str(signature_id) for signature_id in selected_signature_ids if str(signature_id) not in cluster_seeds_require
    ]
    return {
        "seed_signature_ids": seed_signature_ids,
        "unassigned_signature_ids": unassigned_signature_ids,
        "seed_cluster_count": int(len({cluster_seeds_require[signature_id] for signature_id in seed_signature_ids})),
        "altered_signature_ids": altered_signature_ids,
    }


def _run_single_impl(args: argparse.Namespace) -> dict[str, Any]:
    from s2and import model as model_module
    from s2and.data import ANDData
    from s2and.production_model import load_production_model

    rust_extension_identity: dict[str, Any] | None = None
    if args.backend in {"rust", "auto"}:
        rust_extension_identity = collect_rust_extension_identity(
            require_release=bool(args.require_rust_release),
            fail_if_unavailable=False,
        )

    model_path = Path(args.model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model artifact not found: {model_path}")

    truth_context: dict[str, Any] | None = None
    if _truth_enabled(args):
        truth_context = _load_truth_bundle_inputs(args)
        subset_dir = Path(str(truth_context["truth_bundle_root"]))
        signatures = dict(truth_context["signatures"])
        papers = dict(truth_context["papers"])
        target_block = str(truth_context["target_block"])
        selected_signature_ids = list(truth_context["selected_signature_ids"])
        cluster_seeds = None
        cluster_seeds_source = (
            f"truth-bundle:{truth_context['truth_dataset']}:{truth_context['truth_table_key']}:"
            f"{truth_context['truth_split']}"
        )
        explicit_specter_path = str(getattr(args, "specter_path", ""))
        if explicit_specter_path:
            specter_path = _resolve_optional_existing_file(explicit_specter_path)
        else:
            specter_path = _resolve_optional_existing_file(str(truth_context.get("specter_path", "")))
        altered_cluster_signatures = None
        altered_cluster_signatures_source = "unset"
    else:
        subset_dir = Path(args.subset_dir)
        if not subset_dir.exists():
            raise FileNotFoundError(f"Subset directory not found: {subset_dir}")
        signatures, papers, meta_target_block = _load_subset_payload(subset_dir)
        specter_path = _resolve_optional_existing_file(str(getattr(args, "specter_path", "")))
        cluster_seeds_path = _resolve_optional_existing_file(str(getattr(args, "cluster_seeds_path", "")))
        altered_cluster_signatures_path = _resolve_optional_existing_file(
            str(getattr(args, "altered_cluster_signatures_path", ""))
        )
        target_block = _resolve_target_block(signatures, args.target_block or meta_target_block)
        selected_signature_ids = _select_block_signature_ids(
            signatures=signatures,
            papers=papers,
            target_block=target_block,
            total_signatures=int(args.total_signatures),
            random_seed=int(args.random_seed),
        )
        if cluster_seeds_path is None:
            seed_signature_ids = selected_signature_ids[: int(args.seed_signatures)]
            effective_seed_cluster_count = _effective_seed_cluster_count(
                seed_signature_count=len(seed_signature_ids),
                requested_seed_clusters=int(args.seed_cluster_count),
            )
            cluster_seeds = _build_cluster_seeds(seed_signature_ids, effective_seed_cluster_count)
            cluster_seeds_source = "synthetic"
        else:
            cluster_seeds = str(cluster_seeds_path)
            cluster_seeds_source = str(cluster_seeds_path)

        altered_cluster_signatures: list[str] | str | None
        altered_cluster_signatures_source: str
        if altered_cluster_signatures_path is None:
            altered_cluster_signatures = None
            altered_cluster_signatures_source = "unset"
        else:
            altered_cluster_signatures = str(altered_cluster_signatures_path)
            altered_cluster_signatures_source = str(altered_cluster_signatures_path)

    filtered_signatures = {signature_id: signatures[signature_id] for signature_id in selected_signature_ids}
    selected_paper_ids = {
        str(filtered_signatures[signature_id]["paper_id"])
        for signature_id in selected_signature_ids
        if filtered_signatures[signature_id].get("paper_id") is not None
    }
    filtered_papers = {paper_id: payload for paper_id, payload in papers.items() if str(paper_id) in selected_paper_ids}

    total_start = time.perf_counter()
    with ProcessTreeRSSMonitor(interval_seconds=0.05) as monitor:
        anddata_start = time.perf_counter()
        anddata = ANDData(
            signatures=filtered_signatures,
            papers=filtered_papers,
            name=f"big_block_incremental_{args.backend}",
            mode="inference",
            clusters=None,
            specter_embeddings=str(specter_path) if specter_path is not None else None,
            cluster_seeds=cluster_seeds,
            altered_cluster_signatures=altered_cluster_signatures,
            block_type="s2",
            train_pairs=None,
            val_pairs=None,
            test_pairs=None,
            train_pairs_size=1000,
            val_pairs_size=1000,
            test_pairs_size=1000,
            n_jobs=int(args.n_jobs),
            load_name_counts=False,
            preprocess=True,
            random_seed=int(args.random_seed),
            name_tuples="filtered",
            use_orcid_id=bool(int(args.use_orcid_id)),
            use_sinonym_overwrite=False,
            compute_reference_features=False,
        )
        anddata_build_seconds = time.perf_counter() - anddata_start
        if truth_context is not None:
            anddata.cluster_seeds_require = dict(truth_context["cluster_seeds_require"])
            anddata.cluster_seeds_disallow = set()
            anddata.max_seed_cluster_id = len(set(anddata.cluster_seeds_require.values()))
            anddata._cluster_seeds_version = int(getattr(anddata, "_cluster_seeds_version", 0)) + 1
        seed_state = _summarize_incremental_seed_state(selected_signature_ids=selected_signature_ids, anddata=anddata)
        actual_seed_signature_ids = list(seed_state["seed_signature_ids"])
        actual_unassigned_signature_ids = list(seed_state["unassigned_signature_ids"])
        actual_seed_cluster_count = int(seed_state["seed_cluster_count"])

        clusterer = load_production_model(str(model_path))
        model_module._ensure_lightgbm_fitted(clusterer.classifier)
        model_module._ensure_lightgbm_fitted(clusterer.nameless_classifier)
        clusterer.use_cache = False
        clusterer.n_jobs = int(args.n_jobs)

        predict_start = time.perf_counter()
        incremental_result = clusterer.predict_incremental(
            selected_signature_ids,
            anddata,
            batching_threshold=int(args.batching_threshold),
            total_ram_bytes=int(args.total_ram_bytes),
        )
        pred_clusters = incremental_result["clusters"]
        promoted_telemetry = dict(incremental_result.get("incremental_linker_telemetry", {}))
        predict_seconds = time.perf_counter() - predict_start

    truth_quality = _evaluate_truth_link_quality(pred_clusters, truth_context) if truth_context is not None else None
    total_runtime_seconds = time.perf_counter() - total_start
    assigned_signatures = int(sum(len(signatures_in_cluster) for signatures_in_cluster in pred_clusters.values()))
    broad_seed_query_pairs = int(len(actual_seed_signature_ids) * len(actual_unassigned_signature_ids))
    phase_b_residual_count = int(incremental_result.get("phase_b_residual_count", 0))
    phase_b_required_bytes = int(incremental_result["phase_b_required_bytes"])
    measurement_fields = _promoted_measurement_fields(
        broad_seed_query_pairs=broad_seed_query_pairs,
        promoted_telemetry=promoted_telemetry,
        phase_b_residual_count=phase_b_residual_count,
        phase_b_required_bytes=phase_b_required_bytes,
    )

    result = {
        "mode": "single",
        "backend": args.backend,
        "subset_dir": str(subset_dir),
        "model_path": str(model_path),
        "target_block": target_block,
        "use_orcid_id": bool(int(args.use_orcid_id)),
        "total_ram_bytes": int(args.total_ram_bytes),
        "total_signatures": int(len(selected_signature_ids)),
        "seed_signatures_requested": int(args.seed_signatures),
        "seed_signatures": int(len(actual_seed_signature_ids)),
        "unassigned_signatures": int(len(actual_unassigned_signature_ids)),
        "seed_clusters_requested": int(args.seed_cluster_count),
        "seed_clusters_effective": int(actual_seed_cluster_count),
        "cluster_seeds_source": cluster_seeds_source,
        "specter_embeddings_source": str(specter_path) if specter_path is not None else "unset",
        "altered_cluster_signatures_source": altered_cluster_signatures_source,
        "quality_evidence_available": truth_quality is not None,
        "truth_quality": truth_quality,
        "batching_threshold": int(args.batching_threshold),
        "n_jobs": int(args.n_jobs),
        "random_seed": int(args.random_seed),
        "estimated_incremental_pairs": broad_seed_query_pairs,
        "anddata_build_seconds": round(anddata_build_seconds, 3),
        "predict_seconds": round(predict_seconds, 3),
        "total_runtime_seconds": round(total_runtime_seconds, 3),
        "peak_rss_gb": round(monitor.peak_gb, 3),
        "predicted_cluster_count": int(len(pred_clusters)),
        "assigned_signatures": assigned_signatures,
        "cluster_size_summary": _cluster_size_summary(pred_clusters),
        "cluster_membership_digest": _cluster_membership_digest(pred_clusters),
        "phase_b_mode": str(incremental_result["phase_b_mode"]),
        "phase_b_budget_bytes": int(incremental_result["phase_b_budget_bytes"]),
        "phase_b_required_bytes": phase_b_required_bytes,
        "phase_b_residual_count": phase_b_residual_count,
        **measurement_fields,
        "promoted_link_count": int(promoted_telemetry.get("link_count", 0)),
        "promoted_abstain_count": int(promoted_telemetry.get("abstain_count", 0)),
        "promoted_query_batch_count": int(promoted_telemetry.get("query_batch_count", 0)),
        "promoted_query_batch_size_max": int(promoted_telemetry.get("query_batch_size_max", 0)),
        "promoted_memory_predicted_peak_delta_bytes_max": int(
            promoted_telemetry.get("memory_predicted_peak_delta_bytes_max", 0)
        ),
        "promoted_memory_observed_peak_delta_bytes_max": int(
            promoted_telemetry.get("memory_observed_peak_delta_bytes_max", 0)
        ),
        "rust_extension_identity": rust_extension_identity,
        "rust_cluster_seed_sync_calls": int(getattr(anddata, "_rust_cluster_seeds_sync_calls", 0)),
        "rust_cluster_seed_sync_attempted": int(getattr(anddata, "_rust_cluster_seeds_sync_attempted", 0)),
        "rust_cluster_seed_sync_succeeded": int(getattr(anddata, "_rust_cluster_seeds_sync_succeeded", 0)),
        "rust_cluster_seed_sync_skipped_unchanged": int(
            getattr(anddata, "_rust_cluster_seeds_sync_skipped_unchanged", 0)
        ),
        "rust_cluster_seed_sync_seconds_total": round(
            float(getattr(anddata, "_rust_cluster_seeds_sync_seconds_total", 0.0)),
            6,
        ),
        "rust_cluster_seed_sync_seconds_max": round(
            float(getattr(anddata, "_rust_cluster_seeds_sync_seconds_max", 0.0)),
            6,
        ),
    }
    if int(args.emit_signature_map) == 1:
        result["signature_to_cluster_fingerprint"] = _signature_to_cluster_fingerprint_map(pred_clusters)
    if args.single_write_json:
        output_path = Path(args.single_write_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as outfile:
            json.dump(result, outfile, indent=2, sort_keys=True)
        print(f"Wrote single-run JSON: {output_path}")
    return result


def _run_single(args: argparse.Namespace) -> dict[str, Any]:
    prior_runtime_env = _set_runtime_env(
        backend=args.backend,
        n_jobs=int(args.n_jobs),
    )
    try:
        return _run_single_impl(args)
    finally:
        _restore_runtime_env(prior_runtime_env)


def _extract_single_result(stdout_text: str) -> dict[str, Any]:
    return extract_marked_json_payload(stdout_text, RESULT_JSON_START, RESULT_JSON_END)


def _run_subprocess_single(
    script_path: Path,
    args: argparse.Namespace,
    *,
    batching_threshold_override: int | None = None,
    backend_override: str | None = None,
    emit_signature_map: bool = True,
) -> dict[str, Any]:
    batching_threshold = (
        int(args.batching_threshold) if batching_threshold_override is None else int(batching_threshold_override)
    )
    backend = str(args.backend if backend_override is None else backend_override)
    cmd = [
        sys.executable,
        str(script_path),
        "--mode",
        "single",
        "--backend",
        backend,
        "--subset-dir",
        args.subset_dir,
        "--target-block",
        args.target_block,
        "--total-signatures",
        str(args.total_signatures),
        "--seed-signatures",
        str(args.seed_signatures),
        "--seed-cluster-count",
        str(args.seed_cluster_count),
        "--batching-threshold",
        str(batching_threshold),
        "--n-jobs",
        str(args.n_jobs),
        "--random-seed",
        str(args.random_seed),
        "--total-ram-bytes",
        str(args.total_ram_bytes),
        "--use-orcid-id",
        str(int(args.use_orcid_id)),
        "--model-path",
        args.model_path,
        "--specter-path",
        args.specter_path,
        "--cluster-seeds-path",
        args.cluster_seeds_path,
        "--altered-cluster-signatures-path",
        args.altered_cluster_signatures_path,
        "--truth-bundle-root",
        str(getattr(args, "truth_bundle_root", "")),
        "--truth-dataset",
        str(getattr(args, "truth_dataset", "")),
        "--truth-table-key",
        str(getattr(args, "truth_table_key", "")),
        "--truth-split",
        str(getattr(args, "truth_split", "test")),
        "--truth-query-limit",
        str(int(getattr(args, "truth_query_limit", 20))),
        "--truth-max-candidates-per-query",
        str(int(getattr(args, "truth_max_candidates_per_query", 25))),
        "--truth-max-component-members",
        str(int(getattr(args, "truth_max_component_members", 20))),
        "--require-rust-release",
        str(int(args.require_rust_release)),
        "--emit-signature-map",
        "1" if emit_signature_map else "0",
        "--full-run",
    ]
    completed = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(
            f"Subprocess single-run failed (returncode={completed.returncode}).\n"
            f"STDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}"
        )
    return _extract_single_result(completed.stdout)


def _partition_diff_summary(
    baseline_result: dict[str, Any],
    candidate_result: dict[str, Any],
) -> dict[str, Any]:
    baseline_partition = baseline_result.get("signature_to_cluster_fingerprint", {})
    candidate_partition = candidate_result.get("signature_to_cluster_fingerprint", {})
    signature_partition_diff_count = int(
        sum(
            1
            for signature_id, fingerprint in baseline_partition.items()
            if candidate_partition.get(signature_id) != fingerprint
        )
    )
    signature_partition_diff_fraction = float(signature_partition_diff_count / max(1, len(baseline_partition)))
    return {
        "cluster_equivalent": baseline_result["cluster_membership_digest"]
        == candidate_result["cluster_membership_digest"],
        "signature_partition_diff_count": signature_partition_diff_count,
        "signature_partition_diff_fraction": round(signature_partition_diff_fraction, 6),
        "signature_partition_denominator": int(len(baseline_partition)),
    }


def _compact_result(result: dict[str, Any]) -> dict[str, Any]:
    compact = dict(result)
    compact.pop("signature_to_cluster_fingerprint", None)
    return compact


def _runtime_delta_summary(
    baseline_result: dict[str, Any],
    candidate_result: dict[str, Any],
) -> dict[str, Any]:
    baseline_predict = float(baseline_result["predict_seconds"])
    candidate_predict = float(candidate_result["predict_seconds"])
    baseline_total = float(baseline_result["total_runtime_seconds"])
    candidate_total = float(candidate_result["total_runtime_seconds"])
    baseline_peak = float(baseline_result["peak_rss_gb"])
    candidate_peak = float(candidate_result["peak_rss_gb"])
    return {
        "predict_speedup_vs_baseline": _optional_speedup(baseline_predict, candidate_predict),
        "total_speedup_vs_baseline": _optional_speedup(baseline_total, candidate_total),
        "predict_delta_seconds": round(candidate_predict - baseline_predict, 3),
        "total_delta_seconds": round(candidate_total - baseline_total, 3),
        "peak_rss_delta_gb": round(candidate_peak - baseline_peak, 3),
        "peak_rss_delta_fraction": _optional_fraction(candidate_peak - baseline_peak, baseline_peak),
    }


def _run_compare_promoted(args: argparse.Namespace) -> dict[str, Any]:
    script_path = Path(__file__).resolve()
    legacy_python_threshold = max(int(args.total_signatures), int(args.batching_threshold)) + 1

    legacy_python_result = _run_subprocess_single(
        script_path,
        args,
        batching_threshold_override=legacy_python_threshold,
        backend_override="python",
        emit_signature_map=True,
    )
    promoted_rust_result = _run_subprocess_single(
        script_path,
        args,
        backend_override="rust",
        emit_signature_map=True,
    )

    partition_diff = _partition_diff_summary(legacy_python_result, promoted_rust_result)
    runtime_delta = _runtime_delta_summary(legacy_python_result, promoted_rust_result)
    legacy_quality = legacy_python_result.get("truth_quality")
    promoted_quality = promoted_rust_result.get("truth_quality")
    quality_evidence_available = isinstance(legacy_quality, dict) and isinstance(promoted_quality, dict)
    quality_delta: dict[str, Any] | None = None
    if quality_evidence_available:
        quality_delta = {
            "link_precision_delta": (
                None
                if legacy_quality.get("link_precision") is None or promoted_quality.get("link_precision") is None
                else round(float(promoted_quality["link_precision"]) - float(legacy_quality["link_precision"]), 6)
            ),
            "link_recall_delta": (
                None
                if legacy_quality.get("link_recall") is None or promoted_quality.get("link_recall") is None
                else round(float(promoted_quality["link_recall"]) - float(legacy_quality["link_recall"]), 6)
            ),
            "link_f1_delta": (
                None
                if legacy_quality.get("link_f1") is None or promoted_quality.get("link_f1") is None
                else round(float(promoted_quality["link_f1"]) - float(legacy_quality["link_f1"]), 6)
            ),
            "accuracy_delta": (
                None
                if legacy_quality.get("accuracy") is None or promoted_quality.get("accuracy") is None
                else round(float(promoted_quality["accuracy"]) - float(legacy_quality["accuracy"]), 6)
            ),
        }
    summary = {
        "mode": "compare_promoted",
        "comparison": "legacy_python_vs_promoted_rust",
        "legacy_output_parity_is_release_gate": False,
        "quality_evidence_available": quality_evidence_available,
        "quality_evidence_note": (
            "Truth-bundle link/abstain quality is present."
            if quality_evidence_available
            else (
                "This script reports runtime, RSS, candidate reduction, residual-tail, and partition-diff telemetry. "
                "Pass --truth-bundle-root and --truth-dataset for link/abstain quality."
            )
        ),
        "subset_dir": args.subset_dir,
        "target_block": legacy_python_result["target_block"],
        "total_signatures": int(args.total_signatures),
        "seed_signatures": int(args.seed_signatures),
        "seed_clusters_requested": int(args.seed_cluster_count),
        "seed_clusters_effective": int(legacy_python_result["seed_clusters_effective"]),
        "legacy_python_batching_threshold": int(legacy_python_threshold),
        "promoted_rust_batching_threshold": int(args.batching_threshold),
        "n_jobs": int(args.n_jobs),
        "random_seed": int(args.random_seed),
        "total_ram_bytes": int(args.total_ram_bytes),
        **partition_diff,
        **runtime_delta,
        "truth_quality_delta": quality_delta,
        "legacy_python": _compact_result(legacy_python_result),
        "promoted_rust": _compact_result(promoted_rust_result),
    }

    print("Promoted incremental comparison summary:")
    print(
        "1. Legacy Python: "
        f"predict={legacy_python_result['predict_seconds']}s | "
        f"total={legacy_python_result['total_runtime_seconds']}s | "
        f"peak RSS={legacy_python_result['peak_rss_gb']} GB"
    )
    print(
        "2. Promoted Rust: "
        f"predict={promoted_rust_result['predict_seconds']}s | "
        f"total={promoted_rust_result['total_runtime_seconds']}s | "
        f"peak RSS={promoted_rust_result['peak_rss_gb']} GB"
    )
    print(
        "3. Speedup vs legacy Python: "
        f"predict={summary['predict_speedup_vs_baseline']}x | total={summary['total_speedup_vs_baseline']}x"
    )
    if promoted_rust_result.get("promoted_measurement_available"):
        print(
            "4. Promoted candidate reduction: "
            f"{promoted_rust_result['broad_seed_query_pairs']} broad pairs -> "
            f"{promoted_rust_result['promoted_scored_pairs']} scored pairs "
            f"(reduction={promoted_rust_result['promoted_scored_pair_reduction_fraction']})"
        )
        print(
            "5. Promoted residual tail: "
            f"queries={promoted_rust_result['phase_b_residual_count']} | "
            f"pairs={promoted_rust_result['residual_tail_pair_count']} | "
            f"bytes={promoted_rust_result['residual_tail_matrix_bytes']}"
        )
    else:
        print("4. Promoted candidate reduction: not available in Rust result")
    print(
        "6. Legacy partition diff, descriptive only: "
        f"cluster_equivalent={summary['cluster_equivalent']} | "
        f"signature_diff={summary['signature_partition_diff_count']}/"
        f"{summary['signature_partition_denominator']}"
    )
    if quality_evidence_available:
        print(
            "7. Truth-bundle quality (legacy -> promoted): "
            f"F1={legacy_quality.get('link_f1')} -> {promoted_quality.get('link_f1')} | "
            f"accuracy={legacy_quality.get('accuracy')} -> {promoted_quality.get('accuracy')}"
        )
    else:
        print("7. Truth-bundle quality: not evaluated")

    if args.write_json:
        output_path = Path(args.write_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as outfile:
            json.dump(summary, outfile, indent=2, sort_keys=True)
        print(f"Wrote promoted compare JSON: {output_path}")

    return summary


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Measure big-block incremental inference with process-tree peak RSS, wall-clock, "
            "promoted-linker candidate reduction, and residual-tail telemetry."
        )
    )
    parser.add_argument(
        "--mode",
        choices=["single", "compare_promoted"],
        default="single",
        help=(
            "single measures one backend; compare_promoted compares legacy Python to promoted Rust without "
            "treating partition differences as failures."
        ),
    )
    parser.add_argument("--backend", choices=["python", "rust", "auto"], default="rust")
    parser.add_argument("--subset-dir", default=str(PROJECT_ROOT / "scratch" / "inventors_topblock_15k"))
    parser.add_argument(
        "--target-block",
        default="",
        help="Optional block key. Empty -> meta target block or largest block.",
    )
    parser.add_argument("--total-signatures", type=int, default=4000)
    parser.add_argument("--seed-signatures", type=int, default=3000)
    parser.add_argument("--seed-cluster-count", type=int, default=500)
    parser.add_argument("--batching-threshold", type=int, default=1500)
    parser.add_argument("--n-jobs", type=int, default=8)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--use-orcid-id", type=int, choices=[0, 1], default=1)
    parser.add_argument("--specter-path", default="")
    parser.add_argument("--cluster-seeds-path", default="")
    parser.add_argument("--altered-cluster-signatures-path", default="")
    parser.add_argument(
        "--truth-bundle-root",
        default="",
        help="Optional self-contained linker bundle root for link/abstain quality evaluation.",
    )
    parser.add_argument(
        "--truth-dataset",
        default="",
        help="Dataset name inside --truth-bundle-root, for example a_khan.",
    )
    parser.add_argument(
        "--truth-table-key",
        default="",
        help="Optional bundle featureless_rows table key. Empty derives an eval table from --truth-dataset.",
    )
    parser.add_argument(
        "--truth-split",
        default="test",
        help="combined_query_split_assignments split to use for truth rows, or 'any'.",
    )
    parser.add_argument(
        "--truth-query-limit",
        type=int,
        default=20,
        help="Maximum labeled query signatures sampled from the truth bundle.",
    )
    parser.add_argument(
        "--truth-max-candidates-per-query",
        type=int,
        default=25,
        help="Maximum truth retrieval rank retained per query.",
    )
    parser.add_argument(
        "--truth-max-component-members",
        type=int,
        default=20,
        help="Maximum seed signatures retained per candidate component; 0 keeps all members.",
    )
    parser.add_argument(
        "--total-ram-bytes",
        type=int,
        default=DEFAULT_TOTAL_RAM_BYTES,
        help="RAM budget for promoted query batching.",
    )
    parser.add_argument(
        "--model-path",
        default=str(PROJECT_ROOT / "s2and" / "data" / "production_model_v1.21"),
    )
    parser.add_argument(
        "--emit-signature-map",
        type=int,
        choices=[0, 1],
        default=0,
        help="Single-mode debug: include signature-to-cluster fingerprint map in JSON output.",
    )
    parser.add_argument("--write-json", default="", help="Compare-mode output JSON path.")
    parser.add_argument("--single-write-json", default="", help="Single-mode output JSON path.")
    parser.add_argument(
        "--fail-on-cluster-mismatch",
        type=int,
        choices=[0, 1],
        default=0,
        help=(
            "Only applies to compare_promoted. Keep 0 for promoted-linker measurement; set 1 when "
            "explicitly checking partition equivalence."
        ),
    )
    parser.add_argument("--require-rust-release", type=int, choices=[0, 1], default=0)
    parser.add_argument(
        "--full-run",
        action="store_true",
        help="Required for runs larger than 4000 signatures.",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    _validate_args(args)

    if args.mode == "single":
        result = _run_single(args)
        _print_single_summary(result)
        print(RESULT_JSON_START)
        print(json.dumps(result, indent=2, sort_keys=True))
        print(RESULT_JSON_END)
        return

    _run_compare_promoted(args)


if __name__ == "__main__":
    main()
