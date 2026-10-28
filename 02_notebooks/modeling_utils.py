"""Shared helpers for the modelling notebooks (04_*).

Both modelling notebooks use the same data loading, the same prompt-grouped train/test split and the
same evaluation loop, so their results are directly comparable.

Why grouped? About one in five first prompts occurs more than once in the data. With a random split,
identical prompts end up in both train and test, which inflates the scores (especially for models that
use text embeddings). Here all rows with the same (lower-cased, stripped) prompt stay on the same side.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.metrics.pairwise import cosine_distances
from sklearn.model_selection import StratifiedGroupKFold, cross_validate

RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_SPLITS = 5

TARGETS = [
    "target_refusal",
    "target_capability",
    "target_needs_clarification",
]
VERSIONS = ["v01", "v02", "v03", "v04"]

INPUT_FILE = "llm_sustainability_strict_v1-2-0_dataframe.csv"
EMBEDDINGS_FILE = "llm_sustainability_strict_v1-1-0_embeddings.npy"
CATEGORICAL_COLUMNS = ["prompt_style", "task_type", "topic_cat"]

SCORING = {
    "roc_auc": "roc_auc",
    "average_precision": "average_precision",
    "f1": "f1",
    "precision": "precision",
    "recall": "recall",
}


# ----------------------------
# PATHS AND DATA
# ----------------------------

def find_project_root(start=None, marker="01_data"):
    start = Path.cwd() if start is None else Path(start)
    for candidate in [start, *start.parents]:
        if (candidate / marker).exists():
            return candidate
    raise FileNotFoundError(f"Could not find project root containing {marker!r}")


def load_modeling_data(project_root=None):
    """Load the feature table, the embeddings and the prompt groups used for splitting."""
    root = find_project_root() if project_root is None else Path(project_root)
    features_dir = root / "01_data" / "03_features"

    df = pd.read_csv(features_dir / INPUT_FILE)
    embeddings = np.load(features_dir / EMBEDDINGS_FILE)

    # The embeddings are aligned with the table by row position (both come from notebook 02).
    assert len(df) == len(embeddings), (
        f"Row mismatch: table has {len(df)} rows, embeddings have {len(embeddings)}. "
        "Re-run notebook 02 so both files come from the same run."
    )

    for col in CATEGORICAL_COLUMNS:
        df[col] = df[col].astype("category")
    for target in TARGETS:
        df[target] = df[target].astype(int)

    groups = pd.factorize(df["clean_prompt"].fillna("").str.strip().str.lower())[0]

    duplicated_share = float(pd.Series(groups).duplicated(keep=False).mean())
    print(f"Rows: {len(df)} | embedding dimensions: {embeddings.shape[1]}")
    print(f"Unique prompt groups: {groups.max() + 1} | rows whose prompt occurs more than once: {duplicated_share:.1%}")
    return df, embeddings, groups


def embedding_column_names(embeddings):
    return [f"embedding_{i}" for i in range(embeddings.shape[1])]


# ----------------------------
# FEATURE SETS
# ----------------------------

def build_feature_selections(embedding_cols):
    structural_num = ["log_first_prompt_tokens", "log_question_count", "orthographic_error_rate"]
    semantic_num = structural_num + ["embedding_novelty", "topic_prob"]
    binary = ["has_role_instruction", "has_audience_or_level_instruction", "has_format_instruction"]

    return {
        # v01: structural features
        "v01": {"cat": [], "num": structural_num, "bin": binary, "emb": []},
        # v02: structural + engineered semantic features
        "v02": {"cat": ["task_type", "topic_cat"], "num": semantic_num, "bin": binary, "emb": []},
        # v03: v02 + dense embeddings
        "v03": {"cat": ["task_type", "topic_cat"], "num": semantic_num, "bin": binary, "emb": list(embedding_cols)},
        # v04: dense embeddings only
        "v04": {"cat": [], "num": [], "bin": [], "emb": list(embedding_cols)},
    }


def get_features_cols(features):
    cols = (
        features.get("num", [])
        + features.get("emb", [])
        + features.get("bin", [])
        + features.get("cat", [])
    )
    return list(dict.fromkeys(cols))


# ----------------------------
# SPLITS
# ----------------------------

def grouped_train_test_split(y, groups, test_size=TEST_SIZE, random_state=RANDOM_STATE):
    """Stratified holdout split in which identical prompts never appear on both sides."""
    n_splits = round(1 / test_size)
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    train_idx, test_idx = next(splitter.split(np.zeros(len(y)), y, groups))
    assert set(groups[train_idx]).isdisjoint(groups[test_idx]), "A prompt group appears in train and test"
    return train_idx, test_idx


def prepare_split(df, embeddings, embedding_cols, target, groups):
    """Split for one target and add embedding novelty (computed from the training embeddings only)."""
    train_idx, test_idx = grouped_train_test_split(df[target].to_numpy(), groups)

    train_df = df.iloc[train_idx].copy()
    test_df = df.iloc[test_idx].copy()
    train_emb = embeddings[train_idx]
    test_emb = embeddings[test_idx]

    centroid = train_emb.mean(axis=0, keepdims=True)
    train_df["embedding_novelty"] = cosine_distances(train_emb, centroid).ravel()
    test_df["embedding_novelty"] = cosine_distances(test_emb, centroid).ravel()

    train_full = pd.concat(
        [train_df, pd.DataFrame(train_emb, columns=embedding_cols, index=train_df.index)], axis=1
    )
    test_full = pd.concat(
        [test_df, pd.DataFrame(test_emb, columns=embedding_cols, index=test_df.index)], axis=1
    )

    return {
        "x_train_full": train_full,
        "x_test_full": test_full,
        "train_idx": train_idx,
        "test_idx": test_idx,
        "train_groups": groups[train_idx],
    }


# ----------------------------
# EXPERIMENTS
# ----------------------------

def run_experiments(df, embeddings, groups, feature_selections, make_pipeline, targets=TARGETS, verbose=True):
    """Fit one pipeline per target and feature version; cross-validation and test metrics.

    make_pipeline(features) must return an sklearn Pipeline with the steps 'preprocess' and 'model'.
    Returns: datasets, fitted_pipelines, features_by_version, results (DataFrame).
    """
    embedding_cols = embedding_column_names(embeddings)
    cv = StratifiedGroupKFold(n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    datasets, fitted_pipelines, features_by_version, rows = {}, {}, {}, []

    for target in targets:
        split = prepare_split(df, embeddings, embedding_cols, target, groups)
        datasets[target] = split
        fitted_pipelines[target] = {}
        features_by_version[target] = {}

        y_train = split["x_train_full"][target]
        y_test = split["x_test_full"][target]

        for version, features in feature_selections.items():
            cols = get_features_cols(features)
            X_train = split["x_train_full"][cols]
            X_test = split["x_test_full"][cols]

            pipeline = make_pipeline(features)

            cv_results = cross_validate(
                pipeline, X_train, y_train,
                cv=cv, groups=split["train_groups"], scoring=SCORING, n_jobs=1,
            )

            pipeline.fit(X_train, y_train)
            test_pred = pipeline.predict(X_test)
            test_proba = pipeline.predict_proba(X_test)[:, 1]

            row = {
                "target": target,
                "version": version,
                "cv_roc_auc": cv_results["test_roc_auc"].mean(),
                "cv_average_precision": cv_results["test_average_precision"].mean(),
                "cv_f1": cv_results["test_f1"].mean(),
                "cv_precision": cv_results["test_precision"].mean(),
                "cv_recall": cv_results["test_recall"].mean(),
                "test_roc_auc": roc_auc_score(y_test, test_proba),
                "test_average_precision": average_precision_score(y_test, test_proba),
                "test_f1": f1_score(y_test, test_pred, zero_division=0),
                "test_precision": precision_score(y_test, test_pred, zero_division=0),
                "test_recall": recall_score(y_test, test_pred, zero_division=0),
            }
            rows.append(row)

            fitted_pipelines[target][version] = pipeline
            features_by_version[target][version] = cols
            if verbose:
                print(f"{target} {version}: test ROC-AUC {row['test_roc_auc']:.3f}, "
                      f"AP {row['test_average_precision']:.3f}")

    return datasets, fitted_pipelines, features_by_version, pd.DataFrame(rows)


COMPACT_COLUMNS = [
    "target", "version", "cv_roc_auc", "test_roc_auc", "test_average_precision",
    "test_f1", "test_precision", "test_recall",
]


def save_results_table(results, filename, project_root=None):
    root = find_project_root() if project_root is None else Path(project_root)
    path = root / "03_reports" / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    results.round(4).to_csv(path, index=False)
    return path


# ----------------------------
# SHAP HELPERS
# ----------------------------

def original_feature_name(feature, features):
    """Map a transformed column name (e.g. 'cat__task_type_coding') to the original feature."""
    if feature.startswith("num__"):
        return feature.removeprefix("num__")
    if feature.startswith("bin__"):
        return feature.removeprefix("bin__")
    if feature.startswith("emb__"):
        return "embeddings"
    if feature.startswith("cat__"):
        rest = feature.removeprefix("cat__")
        for col in sorted(features.get("cat", []), key=len, reverse=True):
            if rest == col or rest.startswith(col + "_"):
                return col
        return rest
    return feature


def group_shap_importance(feature_names, shap_values, features):
    """Mean |SHAP| per transformed column, aggregated to the original features.

    importance_sum adds up all columns of a feature (384 for embeddings), so a block with many
    columns can dominate by count alone. importance_mean is the average per column.
    """
    shap_values = np.asarray(shap_values)
    if shap_values.ndim == 3:
        shap_values = shap_values[:, :, 1]

    fi = pd.DataFrame({
        "feature": feature_names,
        "importance": np.abs(shap_values).mean(axis=0),
    })
    fi["original_feature"] = fi["feature"].apply(lambda f: original_feature_name(f, features))

    grouped = (
        fi.groupby("original_feature")
        .agg(
            importance_sum=("importance", "sum"),
            importance_mean=("importance", "mean"),
            n_columns=("importance", "count"),
        )
        .sort_values("importance_sum", ascending=False)
    )
    grouped["importance_share"] = grouped["importance_sum"] / grouped["importance_sum"].sum()
    return fi, grouped
