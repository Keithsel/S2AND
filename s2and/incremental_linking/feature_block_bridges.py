"""FeatureBlock bridges to ANDData, service payloads, and mini-ANDData fixtures."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from s2and.incremental_linking.feature_block_arrow import (
    RAW_PLANNER_ARROW_MAX_RECORD_BATCH_ROWS,
    write_feature_block_arrow_tables,
)
from s2and.incremental_linking.feature_block_contract import (
    FeatureBlock,
    FeatureBlockPaper,
    FeatureBlockPaperAuthor,
    FeatureBlockSignature,
    _feature_block_specter_from_mapping,
    _optional_bool,
    _optional_int,
    _optional_str,
    _strict_string_tuple,
    feature_block_signature_order_from_raw_candidate_plan,
    filter_cluster_seed_disallows_for_signature_subset,
)


def write_feature_block_arrow_from_anddata(
    dataset: Any,
    output_dir: str | Path,
    *,
    signature_ids: Sequence[Any] | None = None,
    query_signature_ids: Sequence[Any] = (),
    cluster_seeds_require: Mapping[Any, Any] | None = None,
    include_specter: bool = True,
    include_empty_cluster_seeds: bool = False,
    max_record_batch_rows: Mapping[str, int] | int | None = RAW_PLANNER_ARROW_MAX_RECORD_BATCH_ROWS,
    overwrite: bool = True,
) -> dict[str, str]:
    """Build a `FeatureBlock` from `ANDData` and write complete Arrow IPC tables."""

    feature_block = feature_block_from_anddata(
        dataset,
        signature_ids=signature_ids,
        query_signature_ids=query_signature_ids,
        cluster_seeds_require=cluster_seeds_require,
        include_specter=include_specter,
    )
    return write_feature_block_arrow_tables(
        feature_block,
        output_dir,
        include_empty_cluster_seeds=include_empty_cluster_seeds,
        max_record_batch_rows=max_record_batch_rows,
        overwrite=overwrite,
    )


def feature_block_from_anddata(
    dataset: Any,
    *,
    signature_ids: Sequence[Any] | None = None,
    query_signature_ids: Sequence[Any] = (),
    cluster_seeds_require: Mapping[Any, Any] | None = None,
    include_specter: bool = True,
) -> FeatureBlock:
    """Build a `FeatureBlock` view from an existing `ANDData`-like object."""

    resolved_signature_ids = tuple(
        str(value) for value in (dataset.signatures.keys() if signature_ids is None else signature_ids)
    )
    signatures = tuple(
        _feature_block_signature_from_anddata(dataset, signature_id) for signature_id in resolved_signature_ids
    )
    papers = _feature_block_papers_from_anddata(dataset, signatures)
    paper_authors = _feature_block_paper_authors_from_papers(
        papers_by_id={paper.paper_id: paper for paper in papers}, dataset=dataset
    )

    signature_id_set = set(resolved_signature_ids)
    source_cluster_seeds = dict(
        getattr(dataset, "cluster_seeds_require", {}) if cluster_seeds_require is None else cluster_seeds_require
    )
    require_pairs = tuple(
        (str(signature_id), str(component_id))
        for signature_id, component_id in source_cluster_seeds.items()
        if str(signature_id) in signature_id_set
    )
    disallow_pairs = filter_cluster_seed_disallows_for_signature_subset(
        getattr(dataset, "cluster_seeds_disallow", set()),
        signature_id_set,
    )
    specter_paper_ids, specter_embeddings = _feature_block_specter_from_anddata(
        dataset,
        papers,
        include_specter=include_specter,
    )
    return FeatureBlock(
        signatures=signatures,
        papers=papers,
        paper_authors=paper_authors,
        cluster_seeds_require=require_pairs,
        cluster_seeds_disallow=disallow_pairs,
        query_signature_ids=tuple(str(value) for value in query_signature_ids),
        specter_paper_ids=specter_paper_ids,
        specter_embeddings=specter_embeddings,
    )


def feature_block_from_raw_payloads(
    *,
    signatures: Mapping[str, Mapping[str, Any]],
    papers: Mapping[str, Mapping[str, Any]],
    raw_candidate_plan: Mapping[str, Any],
    cluster_seeds_require: Mapping[Any, Any],
    cluster_seeds_disallow: Iterable[tuple[Any, Any]] | None = None,
    specter_embeddings: Mapping[Any, Any] | tuple[Any, Any] | None = None,
) -> FeatureBlock:
    """Build a mini `FeatureBlock` directly from raw JSON-shaped payloads."""

    signature_order = feature_block_signature_order_from_raw_candidate_plan(raw_candidate_plan)
    selected_signature_ids = signature_order.signature_ids
    signature_rows = tuple(
        _feature_block_signature_from_raw_payload(signature_id, signatures[str(signature_id)])
        for signature_id in selected_signature_ids
    )
    paper_ids = tuple(dict.fromkeys(row.paper_id for row in signature_rows))
    missing_paper_ids = [paper_id for paper_id in paper_ids if str(paper_id) not in papers]
    if missing_paper_ids:
        raise ValueError(f"raw payload papers are missing signature paper_ids: {missing_paper_ids[:10]}")
    paper_rows = tuple(_feature_block_paper_from_raw_payload(paper_id, papers[str(paper_id)]) for paper_id in paper_ids)
    paper_author_rows = tuple(
        row
        for paper_id in paper_ids
        for row in _feature_block_paper_authors_from_raw_payload(paper_id, papers[str(paper_id)])
    )
    selected_signature_id_set = set(selected_signature_ids)
    require_pairs = tuple(
        (str(signature_id), str(component_id))
        for signature_id, component_id in cluster_seeds_require.items()
        if str(signature_id) in selected_signature_id_set
    )
    disallow_pairs = filter_cluster_seed_disallows_for_signature_subset(
        cluster_seeds_disallow or (),
        selected_signature_id_set,
    )
    specter_paper_ids, specter_matrix = _feature_block_specter_from_mapping(
        paper_ids,
        specter_embeddings,
    )
    return FeatureBlock(
        signatures=signature_rows,
        papers=paper_rows,
        paper_authors=paper_author_rows,
        cluster_seeds_require=require_pairs,
        cluster_seeds_disallow=disallow_pairs,
        query_signature_ids=signature_order.query_signature_ids,
        specter_paper_ids=specter_paper_ids,
        specter_embeddings=specter_matrix,
    )


def feature_block_to_mini_anddata(
    feature_block: FeatureBlock,
    *,
    name: str = "feature_block_mini",
    n_jobs: int = 1,
    preprocess: bool = True,
    name_tuples: set[tuple[str, str]] | str | None = "filtered",
    load_name_counts: bool | dict[str, Any] = False,
) -> Any:
    """Materialize a small compatibility `ANDData` from a `FeatureBlock`.

    This is a bridge for scoring wrappers that still call existing pairwise and
    constraint code. It is intended for query plus retrieved candidate members,
    not for full-block Arrow-to-Python materialization.
    """

    from s2and.data import ANDData

    dataset = ANDData(
        signatures=_feature_block_signatures_payload(feature_block),
        papers=_feature_block_papers_payload(feature_block),
        name=name,
        mode="inference",
        specter_embeddings=_feature_block_specter_payload(feature_block),
        load_name_counts=load_name_counts,
        n_jobs=n_jobs,
        preprocess=preprocess,
        name_tuples=name_tuples,
        use_orcid_id=True,
        use_sinonym_overwrite=False,
    )
    dataset.cluster_seeds_require = dict(feature_block.cluster_seeds_require)
    dataset.cluster_seeds_disallow = set(feature_block.cluster_seeds_disallow)

    return dataset


def _source_author_ids_payload(values: Sequence[str]) -> list[str]:
    return [str(value) for value in values]


def _raw_author_info(raw_signature: Mapping[str, Any]) -> Mapping[str, Any]:
    value = raw_signature.get("author_info", {})
    if not isinstance(value, Mapping):
        raise TypeError("raw signature author_info must be a mapping")
    return value


def _raw_orcid(author_info: Mapping[str, Any]) -> str | None:
    source = author_info.get("source_id_source")
    source_ids = author_info.get("source_ids") or []
    if source == "ORCID" and source_ids:
        return _optional_str(source_ids[0])
    return _optional_str(author_info.get("orcid"))


def _feature_block_signature_from_raw_payload(
    signature_id: str,
    raw_signature: Mapping[str, Any],
) -> FeatureBlockSignature:
    author_info = _raw_author_info(raw_signature)
    return FeatureBlockSignature(
        signature_id=str(raw_signature.get("signature_id", signature_id)),
        paper_id=str(raw_signature["paper_id"]),
        author_first=_optional_str(author_info.get("first")),
        author_middle=_optional_str(author_info.get("middle")),
        author_last=_optional_str(author_info.get("last")),
        author_suffix=_optional_str(author_info.get("suffix")),
        author_affiliations=_strict_string_tuple(
            author_info.get("affiliations"),
            field_name="signatures.author_info.affiliations",
        ),
        author_orcid=_raw_orcid(author_info),
        author_position=_optional_int(author_info.get("position"), field_name="signatures.author_info.position"),
        author_block=_optional_str(author_info.get("block")),
        author_email=_optional_str(author_info.get("email")),
        source_author_ids=_strict_string_tuple(
            raw_signature.get("sourced_author_ids"),
            field_name="signatures.sourced_author_ids",
            skip_none=True,
        ),
    )


def _feature_block_paper_from_raw_payload(
    paper_id: str,
    raw_paper: Mapping[str, Any],
) -> FeatureBlockPaper:
    return FeatureBlockPaper(
        paper_id=str(raw_paper.get("paper_id", paper_id)),
        title=_optional_str(raw_paper.get("title")),
        abstract=_optional_str(raw_paper.get("abstract")),
        venue=_optional_str(raw_paper.get("venue")),
        journal_name=_optional_str(raw_paper.get("journal_name")),
        year=_optional_int(raw_paper.get("year"), field_name="papers.year"),
        predicted_language=_optional_str(raw_paper.get("predicted_language")),
        is_reliable=_optional_bool(raw_paper.get("is_reliable"), field_name="papers.is_reliable"),
    )


def _feature_block_paper_authors_from_raw_payload(
    paper_id: str,
    raw_paper: Mapping[str, Any],
) -> tuple[FeatureBlockPaperAuthor, ...]:
    rows: list[FeatureBlockPaperAuthor] = []
    for index, author in enumerate(raw_paper.get("authors") or ()):
        if not isinstance(author, Mapping):
            raise ValueError(f"papers.authors entries must be mappings for paper_id={paper_id!r}")
        position = _optional_int(author.get("position"), field_name="papers.authors.position")
        rows.append(
            FeatureBlockPaperAuthor(
                paper_id=str(paper_id),
                position=index if position is None else position,
                author_name=str(author.get("author_name") or author.get("name") or ""),
            )
        )
    return tuple(rows)


def _feature_block_signatures_payload(feature_block: FeatureBlock) -> dict[str, dict[str, Any]]:
    payload: dict[str, dict[str, Any]] = {}
    for row in feature_block.signatures:
        source_ids = [row.author_orcid] if row.author_orcid else []
        author_info: dict[str, Any] = {
            "first": row.author_first or "",
            "middle": row.author_middle or "",
            "last": row.author_last or "",
            "suffix": row.author_suffix or "",
            "affiliations": list(row.author_affiliations),
            "email": row.author_email or "",
            "source_ids": source_ids,
            "position": None if row.author_position is None else int(row.author_position),
            "block": row.author_block or "",
        }
        if row.author_orcid:
            author_info["source_id_source"] = "ORCID"
        payload[row.signature_id] = {
            "signature_id": row.signature_id,
            "paper_id": row.paper_id,
            "author_info": author_info,
            "sourced_author_ids": _source_author_ids_payload(row.source_author_ids),
        }
    return payload


def _feature_block_papers_payload(feature_block: FeatureBlock) -> dict[str, dict[str, Any]]:
    authors_by_paper_id: dict[str, list[dict[str, Any]]] = {}
    for row in feature_block.paper_authors:
        authors_by_paper_id.setdefault(row.paper_id, []).append(
            {
                "position": int(row.position),
                "author_name": row.author_name,
            }
        )
    return {
        row.paper_id: {
            "paper_id": row.paper_id,
            "title": row.title or "",
            "abstract": row.abstract or "",
            "venue": row.venue or "",
            "journal_name": row.journal_name or "",
            "year": row.year,
            "predicted_language": row.predicted_language,
            "is_reliable": row.is_reliable,
            "references": [],
            "authors": sorted(authors_by_paper_id.get(row.paper_id, []), key=lambda item: int(item["position"])),
        }
        for row in feature_block.papers
    }


def _feature_block_specter_payload(feature_block: FeatureBlock) -> dict[str, np.ndarray] | None:
    if feature_block.specter_embeddings is None:
        return None
    return {
        paper_id: np.ascontiguousarray(feature_block.specter_embeddings[index], dtype=np.float32)
        for index, paper_id in enumerate(feature_block.specter_paper_ids)
    }


def _feature_block_signature_from_anddata(dataset: Any, signature_id: str) -> FeatureBlockSignature:
    signature = dataset.signatures[str(signature_id)]
    return FeatureBlockSignature(
        signature_id=str(signature_id),
        paper_id=str(signature.paper_id),
        author_first=_optional_str(getattr(signature, "author_info_first", None)),
        author_middle=_optional_str(getattr(signature, "author_info_middle", None)),
        author_last=_optional_str(getattr(signature, "author_info_last", None)),
        author_suffix=_optional_str(getattr(signature, "author_info_suffix", None)),
        author_affiliations=_strict_string_tuple(
            getattr(signature, "author_info_affiliations", None),
            field_name="signatures.author_info_affiliations",
        ),
        author_orcid=_optional_str(getattr(signature, "author_info_orcid", None)),
        author_position=_optional_int(
            getattr(signature, "author_info_position", None),
            field_name="signatures.author_info_position",
        ),
        author_block=_optional_str(getattr(signature, "author_info_block", None)),
        author_email=_optional_str(getattr(signature, "author_info_email", None)),
        source_author_ids=_strict_string_tuple(
            getattr(signature, "sourced_author_ids", None),
            field_name="signatures.sourced_author_ids",
            skip_none=True,
        ),
    )


def _feature_block_papers_from_anddata(
    dataset: Any,
    signatures: Sequence[FeatureBlockSignature],
) -> tuple[FeatureBlockPaper, ...]:
    papers: list[FeatureBlockPaper] = []
    seen: set[str] = set()
    for signature in signatures:
        paper_id = str(signature.paper_id)
        if paper_id in seen:
            continue
        paper = getattr(dataset, "papers", {}).get(paper_id)
        if paper is None:
            raise ValueError(f"ANDData papers are missing signature paper_id: {paper_id!r}")
        seen.add(paper_id)
        papers.append(
            FeatureBlockPaper(
                paper_id=paper_id,
                title=_optional_str(getattr(paper, "title", None)),
                abstract="Has Abstract" if bool(getattr(paper, "has_abstract", False)) else "",
                venue=_optional_str(getattr(paper, "venue", None)),
                journal_name=_optional_str(getattr(paper, "journal_name", None)),
                year=_optional_int(getattr(paper, "year", None), field_name="papers.year"),
                predicted_language=_optional_str(getattr(paper, "predicted_language", None)),
                is_reliable=_optional_bool(getattr(paper, "is_reliable", None), field_name="papers.is_reliable"),
            )
        )
    return tuple(papers)


def _feature_block_paper_authors_from_papers(
    *,
    papers_by_id: Mapping[str, FeatureBlockPaper],
    dataset: Any,
) -> tuple[FeatureBlockPaperAuthor, ...]:
    rows: list[FeatureBlockPaperAuthor] = []
    dataset_papers = getattr(dataset, "papers", {})
    for paper_id in papers_by_id:
        paper = dataset_papers[str(paper_id)]
        for index, author in enumerate(getattr(paper, "authors", None) or ()):
            position = _optional_int(getattr(author, "position", index), field_name="papers.authors.position")
            rows.append(
                FeatureBlockPaperAuthor(
                    paper_id=paper_id,
                    position=index if position is None else position,
                    author_name=str(getattr(author, "author_name", "") or ""),
                )
            )
    return tuple(rows)


def _feature_block_specter_from_anddata(
    dataset: Any,
    papers: Sequence[FeatureBlockPaper],
    *,
    include_specter: bool,
) -> tuple[tuple[str, ...], np.ndarray | None]:
    if not include_specter:
        return (), None
    specter = getattr(dataset, "specter_embeddings", None)
    if not specter:
        return (), None
    paper_ids: list[str] = []
    vectors: list[np.ndarray] = []
    expected_dim: int | None = None
    for paper in papers:
        vector = specter.get(str(paper.paper_id))
        if vector is None:
            continue
        array = np.asarray(vector, dtype=np.float32)
        if array.ndim != 1:
            raise ValueError(f"SPECTER vector for paper_id={paper.paper_id!r} must be 1D, got shape={array.shape}")
        if expected_dim is None:
            expected_dim = int(array.shape[0])
        elif int(array.shape[0]) != expected_dim:
            raise ValueError(
                "SPECTER vectors in a FeatureBlock must have equal dimensions: "
                f"expected {expected_dim}, got {array.shape[0]} for paper_id={paper.paper_id!r}"
            )
        paper_ids.append(str(paper.paper_id))
        vectors.append(array)
    if not vectors:
        return (), None
    return tuple(paper_ids), np.ascontiguousarray(np.vstack(vectors), dtype=np.float32)
