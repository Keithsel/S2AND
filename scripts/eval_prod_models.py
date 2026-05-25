# mypy: ignore-errors

"""
Evaluate production S2AND models (SPECTER1 vs SPECTER2) on various datasets.


In this script we try to answer the question: if we deploy SPECTER2, will S2AND care?
Both with retraining and without retraining.

This is done with s2and-mini. Ai2 employee, find it at s3://ai2-s2-research/s2and/s2and-mini/

With retraining (random seed 42):

Performance with SPECTERv1 data, on arnetminer (B3): (0.922, 0.985, 0.952)
Performance with SPECTERv2 data, on arnetminer (B3): (0.93, 0.988, 0.958)

Performance with SPECTERv1 data, on inspire (B3): (0.958, 0.974, 0.966)
Performance with SPECTERv2 data, on inspire (B3): (0.995, 0.959, 0.977)

Performance with SPECTERv1 data, on kisti (B3): (0.951, 0.971, 0.961)
Performance with SPECTERv2 data, on kisti (B3): (0.946, 0.98, 0.963)

Performance with SPECTERv1 data, on pubmed (B3): (0.849, 0.988, 0.913)
Performance with SPECTERv2 data, on pubmed (B3): (0.86, 0.988, 0.92)

Performance with SPECTERv1 data, on qian (B3): (0.936, 0.943, 0.94)
Performance with SPECTERv2 data, on qian (B3): (0.95, 0.964, 0.957)

Performance with SPECTERv1 data, on zbmath (B3): (0.966, 0.984, 0.975)
Performance with SPECTERv2 data, on zbmath (B3): (0.975, 0.991, 0.983)

---

Without retraining (production model artifacts, random seed 42, verified 2026-05-21):

Performance with SPECTERv1 data, on arnetminer (B3): (0.988, 0.972, 0.98)
Performance with SPECTERv2 data, on arnetminer (B3): (0.946, 0.982, 0.963)

Performance with SPECTERv1 data, on inspire (B3): (0.994, 0.954, 0.973)
Performance with SPECTERv2 data, on inspire (B3): (0.998, 0.927, 0.961)

Performance with SPECTERv1 data, on kisti (B3): (0.964, 0.937, 0.95)
Performance with SPECTERv2 data, on kisti (B3): (0.96, 0.96, 0.96)

Performance with SPECTERv1 data, on pubmed (B3): (1.0, 0.895, 0.945)
Performance with SPECTERv2 data, on pubmed (B3): (1.0, 0.892, 0.943)

Performance with SPECTERv1 data, on qian (B3): (0.991, 0.937, 0.963)
Performance with SPECTERv2 data, on qian (B3): (0.978, 0.964, 0.971)

Performance with SPECTERv1 data, on zbmath (B3): (0.966, 0.986, 0.975)
Performance with SPECTERv2 data, on zbmath (B3): (0.961, 0.992, 0.976)


Usage:
    # Evaluate on inventors_s2and (default)
    uv run python scripts/eval_prod_models.py

    # Evaluate on inventors_s2and
    uv run python scripts/eval_prod_models.py --dataset inventors_s2and

    # Evaluate on s2and_mini datasets
    uv run python scripts/eval_prod_models.py --dataset mini
    # Uses s2and/data/s2and_mini_arrow automatically when complete Arrow artifacts exist.

    # Evaluate released benchmark Arrow bundles directly
    uv run python scripts/eval_prod_models.py --dataset full --use-arrow \
      --arrow-data-root s2and/data/s2and-release-arrow

    # Retrain from scratch instead of using prod models
    uv run python scripts/eval_prod_models.py --train

    # Override seed / n_jobs
    uv run python scripts/eval_prod_models.py --seed 42 --n_jobs 8
"""

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate prod S2AND models (SPECTER1 vs SPECTER2)")
    parser.add_argument(
        "--dataset",
        choices=["inventors_s2and", "mini", "full"],
        default="inventors_s2and",
        help="Which dataset(s) to evaluate on (default: inventors_s2and)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Random seed (default: 1, matches transfer_experiment_internal)",
    )
    parser.add_argument("--n_jobs", type=int, default=4, help="Number of parallel jobs (default: 4)")
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=None,
        help="Optional subset of dataset names to evaluate, e.g. --datasets zbmath qian.",
    )
    parser.add_argument(
        "--specter-suffixes",
        nargs="*",
        choices=list(MODELS.keys()),
        default=None,
        help="Optional subset of embedding suffixes to evaluate.",
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Retrain models from scratch instead of loading prod pickles",
    )
    parser.add_argument(
        "--use-arrow",
        action="store_true",
        help=(
            "Force production-model evaluation through direct Arrow/Rust predict_from_arrow_paths. "
            "Arrow is used automatically for mini evals when complete artifacts exist. Not supported with --train."
        ),
    )
    parser.add_argument(
        "--no-arrow",
        action="store_true",
        help="Disable automatic Arrow/Rust evaluation even when Arrow artifacts exist.",
    )
    parser.add_argument(
        "--arrow-data-root",
        default=None,
        help=(
            "Arrow data root. Defaults to s2and/data/s2and_mini_arrow for --dataset mini and "
            "s2and/data/s2and-release-arrow for --dataset full."
        ),
    )
    return parser


def _resolve_requested_datasets(
    default_datasets: list[str],
    requested_datasets: list[str] | None,
    dataset_label: str,
) -> list[str]:
    if not requested_datasets:
        return list(default_datasets)
    requested = [str(dataset_name) for dataset_name in requested_datasets]
    unknown_datasets = sorted(set(requested) - set(default_datasets))
    if unknown_datasets:
        raise ValueError(f"Unknown dataset(s) for --dataset {dataset_label}: {unknown_datasets}")
    return requested


def _resolve_requested_specter_suffixes(default_suffixes: list[str], requested_suffixes: list[str] | None) -> list[str]:
    if not requested_suffixes:
        return list(default_suffixes)
    return [str(suffix) for suffix in requested_suffixes]


def _default_arrow_data_root(project_root_path: str, dataset_label: str) -> str | None:
    if dataset_label == "mini":
        return os.path.join(project_root_path, "s2and", "data", "s2and_mini_arrow")
    if dataset_label == "full":
        return os.path.join(project_root_path, "s2and", "data", "s2and-release-arrow")
    return None


def _supports_arrow_eval(dataset_label: str) -> bool:
    return dataset_label in {"mini", "full"}


# specter suffix -> production model artifact
# v1.1 was trained on specter1 features; v1.21 bundles the v1.2 SPECTER2 pairwise model.
MODELS = {
    "_specter.pickle": "production_model_v1.1.pickle",
    "_specter2.pkl": "production_model_v1.21",
}
specter_suffixes = list(MODELS.keys())


def resolve_dataset_file(data_root: str, dataset_name: str, preferred_name: str, fallback_name: str) -> str:
    """Try preferred filename, then fallback, raising FileNotFoundError if neither exists."""
    preferred_path = os.path.join(data_root, dataset_name, preferred_name)
    if os.path.exists(preferred_path):
        return preferred_path
    fallback_path = os.path.join(data_root, dataset_name, fallback_name)
    if os.path.exists(fallback_path):
        return fallback_path
    raise FileNotFoundError(f"Missing dataset file. Tried '{preferred_path}' and '{fallback_path}'.")


def resolve_arrow_dataset_paths(arrow_root: str, dataset_name: str, specter_suffix: str) -> dict[str, str]:
    from s2and.incremental_linking.feature_block import RAW_PLANNER_ARROW_BATCH_INDEX_KEYS

    dataset_root = os.path.join(arrow_root, dataset_name)
    specter_name = "specter2.arrow" if specter_suffix == "_specter2.pkl" else "specter.arrow"

    paths = {
        "signatures": os.path.join(dataset_root, "signatures.arrow"),
        "papers": os.path.join(dataset_root, "papers.arrow"),
        "paper_authors": os.path.join(dataset_root, "paper_authors.arrow"),
        "specter": os.path.join(dataset_root, specter_name),
        "clusters": os.path.join(dataset_root, f"{dataset_name}_clusters.json"),
    }
    name_counts_index_path = _resolve_eval_name_counts_index_path(Path(dataset_root))
    if name_counts_index_path is not None:
        paths["name_counts_index"] = name_counts_index_path
    missing = {key: path for key, path in paths.items() if not os.path.exists(path)}
    if missing:
        formatted = ", ".join(f"{key}={path}" for key, path in missing.items())
        raise FileNotFoundError(f"Missing Arrow dataset files for {dataset_name}: {formatted}")
    if "name_counts_index" not in paths:
        raise FileNotFoundError(
            f"Missing Arrow name_counts_index for {dataset_name}. "
            "Production mini eval models use name-count features and Arrow eval must use the index."
        )
    for arrow_key, index_key in RAW_PLANNER_ARROW_BATCH_INDEX_KEYS.items():
        arrow_path = paths.get(arrow_key)
        if arrow_path is None:
            continue
        arrow_stem = os.path.splitext(os.path.basename(arrow_path))[0]
        for candidate in (
            os.path.join(os.path.dirname(arrow_path), f"{arrow_stem}.{index_key}.bin"),
            os.path.join(dataset_root, f"{index_key}.bin"),
        ):
            if os.path.exists(candidate):
                paths[index_key] = candidate
                break
    return paths


def _resolve_eval_name_counts_index_path(dataset_root: Path) -> str | None:
    manifest_path = dataset_root / "manifest.json"
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Arrow manifest is not valid JSON: {manifest_path}") from exc
        manifest_paths = manifest.get("paths", {})
        if isinstance(manifest_paths, dict):
            path_value = manifest_paths.get("name_counts_index")
            if path_value is not None:
                resolved = Path(str(path_value))
                if not resolved.is_absolute():
                    resolved = dataset_root / resolved
                if resolved.exists():
                    return str(resolved)
                raise FileNotFoundError(
                    f"Arrow manifest {manifest_path} specifies name_counts_index path that does not exist: "
                    f"{path_value}"
                )
    for candidate in (
        dataset_root / "name_counts_index",
        dataset_root.parent / "name_counts_index",
        dataset_root.parent.parent / "name_counts_index",
    ):
        if candidate.exists():
            return str(candidate)
    return None


def arrow_datasets_available(arrow_root: str | None, datasets: list[str], specter_suffixes: list[str]) -> bool:
    return first_missing_arrow_dataset_error(arrow_root, datasets, specter_suffixes) is None


def first_missing_arrow_dataset_error(
    arrow_root: str | None,
    datasets: list[str],
    specter_suffixes: list[str],
) -> FileNotFoundError | None:
    if arrow_root is None:
        return FileNotFoundError("Missing Arrow data root")
    for dataset_name in datasets:
        for specter_suffix in specter_suffixes:
            try:
                resolve_arrow_dataset_paths(arrow_root, dataset_name, specter_suffix)
            except FileNotFoundError as exc:
                return FileNotFoundError(
                    f"Missing Arrow files for dataset={dataset_name!r}, specter_suffix={specter_suffix!r}: {exc}"
                )
    return None


def read_arrow_s2_blocks(signatures_arrow_path: str) -> dict[str, list[str]]:
    import pyarrow as pa

    with pa.memory_map(signatures_arrow_path, "r") as source:
        table = pa.ipc.open_file(source).read_all().select(["signature_id", "author_block"])
    block_dict: dict[str, list[str]] = defaultdict(list)
    signature_ids = table.column("signature_id").to_pylist()
    author_blocks = table.column("author_block").to_pylist()
    for signature_id, author_block in zip(signature_ids, author_blocks, strict=True):
        block_dict[str(author_block)].append(str(signature_id))
    return dict(block_dict)


def split_blocks_like_anddata(
    blocks_dict: dict[str, list[str]],
    *,
    random_seed: int,
    num_clusters_for_block_size: int = 1,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
) -> tuple[dict[str, list[str]], dict[str, list[str]], dict[str, list[str]]]:
    import numpy as np
    from sklearn.cluster import KMeans
    from sklearn.model_selection import train_test_split

    block_ids = []
    block_sizes = []
    # Match ANDData.split_blocks_helper exactly. This seeded stratified split is
    # order-sensitive; sorting changes pinned production-eval test sets.
    for block_id in blocks_dict:
        signatures = blocks_dict[block_id]
        block_ids.append(block_id)
        block_sizes.append(len(signatures))
    if len(block_ids) == 0:
        return {}, {}, {}
    y_group = (
        KMeans(n_clusters=num_clusters_for_block_size, random_state=random_seed, n_init=10)
        .fit(np.array(block_sizes).reshape(-1, 1))
        .labels_
    )
    train_blocks, val_test_blocks, _, val_test_length = train_test_split(
        block_ids,
        y_group,
        test_size=val_ratio + test_ratio,
        stratify=y_group,
        random_state=random_seed,
    )
    val_blocks, test_blocks = train_test_split(
        val_test_blocks,
        test_size=test_ratio / (val_ratio + test_ratio),
        stratify=val_test_length,
        random_state=random_seed,
    )
    return (
        {block_id: blocks_dict[block_id] for block_id in train_blocks},
        {block_id: blocks_dict[block_id] for block_id in val_blocks},
        {block_id: blocks_dict[block_id] for block_id in test_blocks},
    )


def read_signature_to_cluster_id(clusters_path: str) -> dict[str, str]:
    with open(clusters_path, encoding="utf-8") as infile:
        clusters = json.load(infile)
    signature_to_cluster_id = {}
    for cluster_id, cluster_info in clusters.items():
        for signature_id in cluster_info["signature_ids"]:
            signature_to_cluster_id[str(signature_id)] = str(cluster_id)
    return signature_to_cluster_id


def construct_cluster_to_signatures(
    signature_to_cluster_id: dict[str, str],
    block_dict: dict[str, list[str]],
) -> dict[str, list[str]]:
    cluster_to_signatures: dict[str, list[str]] = defaultdict(list)
    missing_signature_ids: list[str] = []
    for signatures in block_dict.values():
        for signature_id in signatures:
            signature_key = str(signature_id)
            cluster_id = signature_to_cluster_id.get(signature_key)
            if cluster_id is None:
                missing_signature_ids.append(signature_key)
                continue
            cluster_to_signatures[cluster_id].append(signature_key)
    if missing_signature_ids:
        raise ValueError(
            "clusters.json is missing cluster assignments for "
            f"{len(missing_signature_ids)} evaluated signature(s): {missing_signature_ids[:10]}"
        )
    return dict(cluster_to_signatures)


def cluster_eval_arrow(
    arrow_paths: dict[str, str],
    clusterer,
    *,
    random_seed: int,
    n_jobs: int,
) -> tuple[dict[str, tuple], dict[str, tuple[float, float, float]]]:
    import numpy as np

    from s2and.eval import b3_precision_recall_fscore, pairwise_precision_recall_fscore

    _train_block_dict, _val_block_dict, test_block_dict = split_blocks_like_anddata(
        read_arrow_s2_blocks(arrow_paths["signatures"]),
        random_seed=random_seed,
    )
    signature_to_cluster_id = read_signature_to_cluster_id(arrow_paths["clusters"])
    cluster_to_signatures = construct_cluster_to_signatures(signature_to_cluster_id, test_block_dict)
    predict_arrow_paths = {key: value for key, value in arrow_paths.items() if key != "clusters"}
    pred_clusters, _ = clusterer.predict_from_arrow_paths(
        test_block_dict,
        predict_arrow_paths,
        total_ram_bytes=1_000_000_000_000,
        load_name_counts=True,
        name_tuples="filtered",
    )
    (
        b3_p,
        b3_r,
        b3_f1,
        b3_metrics_per_signature,
        pred_bigger_ratios,
        true_bigger_ratios,
    ) = b3_precision_recall_fscore(cluster_to_signatures, pred_clusters)
    metrics: dict[str, tuple] = {"B3 (P, R, F1)": (b3_p, b3_r, b3_f1)}
    metrics["Cluster (P, R F1)"] = pairwise_precision_recall_fscore(
        cluster_to_signatures, pred_clusters, test_block_dict, "clusters"
    )
    metrics["Cluster Macro (P, R, F1)"] = pairwise_precision_recall_fscore(
        cluster_to_signatures, pred_clusters, test_block_dict, "cmacro"
    )

    def _mean_or_nan(xs):
        if len(xs) == 0:
            return float("nan")
        return float(np.round(np.mean(xs), 2))

    metrics["Pred bigger ratio (mean, count)"] = (_mean_or_nan(pred_bigger_ratios), len(pred_bigger_ratios))
    metrics["True bigger ratio (mean, count)"] = (_mean_or_nan(true_bigger_ratios), len(true_bigger_ratios))
    return metrics, b3_metrics_per_signature


# feature categories (all except reference_features)
features_to_use = [
    "name_similarity",
    "affiliation_similarity",
    "email_similarity",
    "coauthor_similarity",
    "venue_similarity",
    "year_diff",
    "title_similarity",
    # "reference_features",
    "misc_features",
    "name_counts",
    "embedding_similarity",
    "journal_similarity",
    "advanced_name_similarity",
]

# nameless model: no name-based features (prevents model overreliance on names)
nameless_features_to_use = [
    f for f in features_to_use if f not in {"name_similarity", "advanced_name_similarity", "name_counts"}
]


def main() -> None:
    import numpy as np

    from s2and.consts import DEFAULT_CHUNK_SIZE, FEATURIZER_VERSION, PROJECT_ROOT_PATH
    from s2and.data import ANDData
    from s2and.eval import cluster_eval
    from s2and.featurizer import FeaturizationInfo, featurize
    from s2and.model import Clusterer, PairwiseModeler
    from s2and.production_model import load_production_model
    from s2and.warnings_utils import suppress_sklearn_feature_name_warnings

    args = _build_parser().parse_args()
    suppress_sklearn_feature_name_warnings()
    n_jobs = args.n_jobs
    random_seed = args.seed
    train_flag = bool(args.train)
    if args.use_arrow and args.no_arrow:
        raise ValueError("Pass only one of --use-arrow or --no-arrow")
    if args.use_arrow and train_flag:
        raise ValueError("--use-arrow is for production-model evaluation and cannot be combined with --train")
    os.environ["OMP_NUM_THREADS"] = str(n_jobs)

    if args.dataset == "mini":
        data_original = os.path.join(PROJECT_ROOT_PATH, "s2and", "data", "s2and_mini")
        arrow_data_root = args.arrow_data_root or _default_arrow_data_root(PROJECT_ROOT_PATH, args.dataset)
        # aminer has too much variance; medline is pairwise only
        datasets = ["arnetminer", "inspire", "kisti", "pubmed", "qian", "zbmath"]
    elif args.dataset == "full":
        data_original = os.path.join(PROJECT_ROOT_PATH, "s2and", "data")
        arrow_data_root = args.arrow_data_root or _default_arrow_data_root(PROJECT_ROOT_PATH, args.dataset)
        datasets = ["arnetminer", "inspire", "kisti", "pubmed", "qian", "zbmath"]
    else:
        data_original = os.path.join(PROJECT_ROOT_PATH, "s2and", "data")
        arrow_data_root = args.arrow_data_root
        if args.use_arrow:
            raise ValueError("--use-arrow currently supports --dataset mini and --dataset full only")
        datasets = ["inventors_s2and"]
    datasets = _resolve_requested_datasets(datasets, args.datasets, args.dataset)
    active_specter_suffixes = _resolve_requested_specter_suffixes(specter_suffixes, args.specter_suffixes)
    missing_arrow_error = (
        first_missing_arrow_dataset_error(arrow_data_root, datasets, active_specter_suffixes)
        if _supports_arrow_eval(args.dataset) and not train_flag
        else FileNotFoundError("Arrow eval is unavailable for this configuration")
    )
    arrow_available = _supports_arrow_eval(args.dataset) and not train_flag and missing_arrow_error is None
    if args.use_arrow and missing_arrow_error is not None:
        raise missing_arrow_error
    use_arrow = bool(args.use_arrow or (args.dataset == "mini" and arrow_available and not args.no_arrow))

    print(
        f"Config: dataset={args.dataset}, seed={random_seed}, n_jobs={n_jobs}, "
        f"train={train_flag}, use_arrow={use_arrow}"
    )
    print(f"Datasets: {datasets}")
    print(f"SPECTER suffixes: {active_specter_suffixes}")
    if use_arrow:
        print(f"Arrow data root: {arrow_data_root}")
    print()

    featurization_info = FeaturizationInfo(features_to_use=features_to_use, featurizer_version=FEATURIZER_VERSION)
    nameless_featurization_info = FeaturizationInfo(
        features_to_use=nameless_features_to_use,
        featurizer_version=FEATURIZER_VERSION,
    )

    results = {}
    for specter_suffix in active_specter_suffixes:
        clusterer = None

        if not train_flag:
            model_name = MODELS[specter_suffix]
            model_path = os.path.join(PROJECT_ROOT_PATH, "s2and", "data", model_name)
            if not os.path.exists(model_path):
                raise FileNotFoundError(
                    f"Missing model artifact at {model_path}. "
                    "Either use --train to retrain, or place the model artifact in s2and/data/."
                )
            print(f"=== specter_suffix: {specter_suffix}, model: {model_name} ===")
            clusterer = load_production_model(model_path)
            clusterer.use_cache = False
            clusterer.n_jobs = n_jobs
        else:
            print(f"=== specter_suffix: {specter_suffix}, training from scratch ===")

        cluster_metrics_all = []
        for dataset_name in datasets:
            print(f"-- dataset: {dataset_name} --")
            if use_arrow:
                if clusterer is None:
                    raise RuntimeError("Arrow evaluation requires a loaded production Clusterer")
                arrow_paths = resolve_arrow_dataset_paths(arrow_data_root, dataset_name, specter_suffix)
                cluster_metrics, _b3_metrics_per_signature = cluster_eval_arrow(
                    arrow_paths,
                    clusterer,
                    random_seed=random_seed,
                    n_jobs=n_jobs,
                )
                print(cluster_metrics)
                cluster_metrics_all.append(cluster_metrics)
                continue

            signatures_path = resolve_dataset_file(
                data_original, dataset_name, f"{dataset_name}_signatures.json", "signatures.json"
            )
            papers_path = resolve_dataset_file(
                data_original, dataset_name, f"{dataset_name}_papers.json", "papers.json"
            )
            clusters_path = resolve_dataset_file(
                data_original, dataset_name, f"{dataset_name}_clusters.json", "clusters.json"
            )
            embeddings_path = resolve_dataset_file(
                data_original, dataset_name, f"{dataset_name}{specter_suffix}", specter_suffix.lstrip("_")
            )
            anddata = ANDData(
                signatures=signatures_path,
                papers=papers_path,
                name=dataset_name,
                mode="train",
                specter_embeddings=embeddings_path,
                clusters=clusters_path,
                block_type="s2",
                train_pairs=None,
                val_pairs=None,
                test_pairs=None,
                train_pairs_size=100000,
                val_pairs_size=10000,
                test_pairs_size=10000,
                n_jobs=n_jobs,
                load_name_counts=True,
                preprocess=True,
                random_seed=random_seed,
                name_tuples="filtered",
            )

            if train_flag:
                train, val, _test = featurize(
                    anddata,
                    featurization_info,
                    n_jobs=n_jobs,
                    use_cache=False,
                    chunk_size=DEFAULT_CHUNK_SIZE,
                    nameless_featurizer_info=nameless_featurization_info,
                    nan_value=np.nan,
                )
                X_train, y_train, nameless_X_train = train
                X_val, y_val, nameless_X_val = val

                pairwise_modeler = PairwiseModeler(
                    n_iter=25,
                    estimator=None,
                    search_space=None,
                    monotone_constraints=featurization_info.lightgbm_monotone_constraints,
                    random_state=random_seed,
                )
                pairwise_modeler.fit(X_train, y_train, X_val, y_val)

                nameless_pairwise_modeler = PairwiseModeler(
                    n_iter=25,
                    estimator=None,
                    search_space=None,
                    monotone_constraints=nameless_featurization_info.lightgbm_monotone_constraints,
                    random_state=random_seed,
                )
                nameless_pairwise_modeler.fit(nameless_X_train, y_train, nameless_X_val, y_val)

                clusterer = Clusterer(
                    featurization_info,
                    pairwise_modeler.classifier,
                    n_jobs=n_jobs,
                    use_cache=False,
                    nameless_classifier=nameless_pairwise_modeler.classifier,
                    nameless_featurizer_info=nameless_featurization_info,
                    random_state=random_seed,
                    use_default_constraints_as_supervision=False,
                )
                clusterer.fit(anddata)

            if clusterer is None:
                raise RuntimeError("Clusterer was not initialized. Check --train flag and model artifact path.")

            cluster_metrics, _b3_metrics_per_signature = cluster_eval(
                anddata,
                clusterer,
                split="test",
                use_s2_clusters=False,
            )
            print(cluster_metrics)
            cluster_metrics_all.append(cluster_metrics)

        results[specter_suffix] = cluster_metrics_all
        b3s = [m["B3 (P, R, F1)"][-1] for m in cluster_metrics_all]
        print(f"B3 F1s: {b3s}, mean: {sum(b3s) / len(b3s):.3f}")
        print()

    # summary
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    if "_specter.pickle" in results and "_specter2.pkl" in results:
        result_specter1 = results["_specter.pickle"]
        result_specter2 = results["_specter2.pkl"]

        for i, dataset_name in enumerate(datasets):
            print(f"Performance with SPECTERv1 data, on {dataset_name} (B3): {result_specter1[i]['B3 (P, R, F1)']}")
            print(f"Performance with SPECTERv2 data, on {dataset_name} (B3): {result_specter2[i]['B3 (P, R, F1)']}")
            print()
    else:
        for specter_suffix, metrics_by_dataset in results.items():
            for i, dataset_name in enumerate(datasets):
                print(
                    f"Performance with {specter_suffix} data, on {dataset_name} (B3): "
                    f"{metrics_by_dataset[i]['B3 (P, R, F1)']}"
                )
            print()


if __name__ == "__main__":
    main()
