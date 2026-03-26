"""Classification & Hypothesis Testing
Loads the session-level features CSV, trains Logistic Regression and
Random Forest classifiers, evaluates with per-class metrics, runs a
feature ablation study, and produces publication-ready figures.
Usage:
    python -m models.classifiers
    python -m models.classifiers --input data/processed/features.csv
"""

import argparse
import json
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # File-only backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    auc,
    classification_report,
    confusion_matrix,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize
from api.app.config import PROJECT_ROOT, RANDOM_SEED

#  Paths 
DEFAULT_INPUT = PROJECT_ROOT / "data" / "processed" / "features.csv"
FIGURES_DIR = PROJECT_ROOT / "report" / "figures"
METRICS_DIR = PROJECT_ROOT / "report" / "metrics"

logger = logging.getLogger("honeypot.classifiers")

#  Feature Groups 
# TIMING: derived purely from inter-request timestamps
TIMING_FEATURES = [
    "mean_gap_time",
    "std_gap_time",
    "max_delay",
    "session_duration",
    "burstiness",
]

# STRUCTURAL: content diversity, error behavior, traversal patterns
# request_rate lives here (correlated twin of burstiness — separate groups)
STRUCTURAL_FEATURES = [
    "request_rate",
    "recovery_time",
    "session_entropy",
    "payload_entropy",
    "path_repetition",
]

ALL_FEATURES = TIMING_FEATURES + STRUCTURAL_FEATURES

# LR drops request_rate (Pearson r = 0.998 with burstiness → coefficient inflation)
ALL_FEATURES_LR = [f for f in ALL_FEATURES if f != "request_rate"]

# Ablation experiment: 3 feature sets × 2 models = 6 runs
# LR always excludes request_rate; RF keeps all features
ABLATION_SETS = {
    "ALL": {
        "LR": ALL_FEATURES_LR,
        "RF": ALL_FEATURES,
    },
    "TIMING": {
        "LR": TIMING_FEATURES,
        "RF": TIMING_FEATURES,
    },
    "STRUCTURAL": {
        "LR": [f for f in STRUCTURAL_FEATURES if f != "request_rate"],
        "RF": STRUCTURAL_FEATURES,
    },
}

# Consistent palette
PALETTE = {"bot": "#e74c3c", "human": "#3498db", "llm_agent": "#2ecc71"}


#  DATA LOADING & PREPARATION

def load_features(path: Path) -> pd.DataFrame:
    """Load features CSV and verify expected columns exist."""
    df = pd.read_csv(path)
    required = set(ALL_FEATURES + ["session_id", "actor_type"])
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    print(f"  Loaded {len(df)} sessions from {path}")
    return df


def split_data(df: pd.DataFrame, seed: int = RANDOM_SEED):
    """Stratified 80/20 train/test split.
    Returns raw DataFrames (pre-imputation) + encoded labels + encoder."""
    le = LabelEncoder()
    y = le.fit_transform(df["actor_type"])

    # Split by index so the original DataFrame can be sliced 
    train_idx, test_idx = train_test_split(
        np.arange(len(df)), test_size=0.2, stratify=y, random_state=seed
    )
    train_df = df.iloc[train_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)
    y_train = y[train_idx]
    y_test = y[test_idx]

    return train_df, test_df, y_train, y_test, le


def prepare_features(train_df: pd.DataFrame, test_df: pd.DataFrame,
                     feature_cols: list[str]):
    """Select features → add missing indicators → impute (median) → scale.
    All fitting uses train data only. Returns numpy arrays + feature names."""
    X_train = train_df[feature_cols].copy()
    X_test = test_df[feature_cols].copy()

    # Missing indicators: binary columns for features with NaN in training data
    # Added BEFORE imputation so the model can learn "value was missing"
    nan_cols = [c for c in feature_cols if X_train[c].isna().any()]
    for col in nan_cols:
        X_train[f"{col}_missing"] = X_train[col].isna().astype(int)
        X_test[f"{col}_missing"] = X_test[col].isna().astype(int)

    feature_names = list(X_train.columns)

    # Median imputation — robust to skew in timing features
    imputer = SimpleImputer(strategy="median")
    X_train_arr = imputer.fit_transform(X_train)
    X_test_arr = imputer.transform(X_test)

    # Standard scaling — one pipeline for both
    scaler = StandardScaler()
    X_train_arr = scaler.fit_transform(X_train_arr)
    X_test_arr = scaler.transform(X_test_arr)

    return X_train_arr, X_test_arr, feature_names


#  MODEL FACTORIES

def make_lr(seed: int = RANDOM_SEED) -> LogisticRegression:
    """Logistic Regression: interpretability baseline."""
    return LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=seed,
        solver="lbfgs",
    )


def make_rf(seed: int = RANDOM_SEED) -> RandomForestClassifier:
    """Random Forest: nonlinear performance baseline."""
    return RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=seed,
        n_jobs=-1,
        min_samples_leaf=2,
    )

#  EVALUATION

def evaluate_model(model, X_test, y_test, class_names: list[str]):
    """Run predictions and compute classification report + confusion matrix."""
    y_pred = model.predict(X_test)
    report = classification_report(
    y_test, y_pred, target_names=class_names, output_dict=True,
    labels=list(range(len(class_names))), zero_division=0,
)
    cm = confusion_matrix(y_test, y_pred)
    return y_pred, report, cm


def save_metrics(data: dict, path: Path) -> None:
    """Write metrics dict to JSON."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"  Saved: {path}")


#  PLOTS

def plot_confusion_matrix(cm, class_names: list[str],
                          title: str, path: Path) -> None:
    """Annotated heatmap confusion matrix."""
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names, ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix — {title}")
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_roc_curves(models_dict: dict, y_test, class_names: list[str],
                    path: Path) -> None:
    """One-vs-rest ROC curves for each model, side by side."""
    n_classes = len(class_names)
    y_bin = label_binarize(y_test, classes=range(n_classes))

    fig, axes = plt.subplots(1, len(models_dict), figsize=(7 * len(models_dict), 6))
    if len(models_dict) == 1:
        axes = [axes]

    colors = list(PALETTE.values())

    for ax, (name, (model, X_test)) in zip(axes, models_dict.items()):
        y_proba = model.predict_proba(X_test)
        for i, (cls, color) in enumerate(zip(class_names, colors)):
            fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, color=color, lw=2,
                    label=f"{cls} (AUC = {roc_auc:.3f})")
        ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.4)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(f"ROC — {name}")
        ax.legend(loc="lower right", fontsize=9)

    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_feature_importance(lr_model, rf_model,
                            lr_features: list[str], rf_features: list[str],
                            path: Path) -> None:
    """LR mean |coefficient| vs RF Gini importance."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, max(6, len(rf_features) * 0.4)))

    # LR: mean absolute coefficient across the 3 class rows
    lr_imp = np.abs(lr_model.coef_).mean(axis=0)
    lr_order = np.argsort(lr_imp)
    ax1.barh(
        [lr_features[i] for i in lr_order], lr_imp[lr_order], color="#3498db"
    )
    ax1.set_title("Logistic Regression — Mean |Coefficient|")
    ax1.set_xlabel("Importance")

    # RF: Gini importance
    rf_imp = rf_model.feature_importances_
    rf_order = np.argsort(rf_imp)
    ax2.barh(
        [rf_features[i] for i in rf_order], rf_imp[rf_order], color="#2ecc71"
    )
    ax2.set_title("Random Forest — Gini Importance")
    ax2.set_xlabel("Importance")

    fig.suptitle("Feature Importance Comparison", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_ablation(results: dict, path: Path) -> None:
    """Grouped bar chart: macro F1 by feature set and model."""
    sets = ["ALL", "TIMING", "STRUCTURAL"]
    models = ["LR", "RF"]

    x = np.arange(len(sets))
    width = 0.3

    fig, ax = plt.subplots(figsize=(8, 5))
    bar_colors = {"LR": "#3498db", "RF": "#2ecc71"}

    for i, model in enumerate(models):
        f1s = [results[f"{s}_{model}"]["macro_f1"] for s in sets]
        bars = ax.bar(x + i * width, f1s, width,
                      label=model, color=bar_colors[model])
        for bar, f1 in zip(bars, f1s):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.008,
                f"{f1:.3f}", ha="center", va="bottom", fontsize=9,
            )

    ax.set_xlabel("Feature Set")
    ax.set_ylabel("Macro F1")
    ax.set_title("Feature Ablation — Macro F1 by Feature Set and Model")
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(sets)
    ax.set_ylim(0, 1.08)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_killer_viz_with_errors(test_df: pd.DataFrame, y_test,
                                y_pred, class_names: list[str],
                                path: Path) -> None:
    """Killer viz (log gap × entropy) with misclassified points marked."""
    fig, ax = plt.subplots(figsize=(10, 7))

    correct_mask = y_test == y_pred

    # Correctly classified: standard scatter by class
    for i, cls in enumerate(class_names):
        mask = (y_test == i) & correct_mask
        subset = test_df.iloc[np.where(mask)[0]]
        ax.scatter(
            np.log10(subset["mean_gap_time"].clip(lower=1)),
            subset["payload_entropy"],
            c=PALETTE[cls], label=f"{cls} (correct)",
            alpha=0.7, edgecolors="white", linewidths=0.5, s=60,
        )

    # Misclassified: red × markers on top
    wrong_mask = ~correct_mask
    if wrong_mask.any():
        subset = test_df.iloc[np.where(wrong_mask)[0]]
        ax.scatter(
            np.log10(subset["mean_gap_time"].clip(lower=1)),
            subset["payload_entropy"],
            c="red", marker="X", s=120, linewidths=2,
            label=f"Misclassified ({wrong_mask.sum()})", zorder=5,
        )

    ax.set_xlabel("log₁₀(mean_gap_time)", fontsize=11)
    ax.set_ylabel("payload_entropy", fontsize=11)
    ax.set_title("Test Set — Predictions with Misclassifications", fontsize=13)
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"  Saved: {path}")


#  ABLATION STUDY

def run_ablation(train_df: pd.DataFrame, test_df: pd.DataFrame,
                 y_train, y_test, class_names: list[str],
                 seed: int = RANDOM_SEED) -> dict:
    """Train LR + RF on each feature subset. Returns results dict."""
    results = {}

    for set_name, feature_map in ABLATION_SETS.items():
        for model_name, feature_cols in feature_map.items():
            X_train, X_test, feat_names = prepare_features(
                train_df, test_df, feature_cols
            )
            model = make_lr(seed) if model_name == "LR" else make_rf(seed)
            model.fit(X_train, y_train)
            _, report, _ = evaluate_model(model, X_test, y_test, class_names)

            key = f"{set_name}_{model_name}"
            results[key] = {
                "feature_set": set_name,
                "model": model_name,
                "features_used": feature_cols,
                "n_features_with_indicators": len(feat_names),
                "macro_f1": report["macro avg"]["f1-score"],
                "accuracy": report["accuracy"],
                "per_class": {
                    cls: {
                        "precision": report[cls]["precision"],
                        "recall": report[cls]["recall"],
                        "f1": report[cls]["f1-score"],
                    }
                    for cls in class_names
                },
            }
            print(f"    {key}: macro F1 = {report['macro avg']['f1-score']:.4f}")

    return results


#  ERROR ANALYSIS

def analyze_errors(test_df: pd.DataFrame, y_test, y_pred,
                   class_names: list[str]) -> pd.DataFrame:
    """Extract and characterize misclassified test samples."""
    mask = y_test != y_pred

    if mask.sum() == 0:
        print("  Zero misclassifications on the test set.")
        return pd.DataFrame()

    error_idx = np.where(mask)[0]
    errors = test_df.iloc[error_idx].copy()
    errors["true_label"] = [class_names[y_test[i]] for i in error_idx]
    errors["predicted"] = [class_names[y_pred[i]] for i in error_idx]

    total = len(y_test)
    n_errors = mask.sum()
    print(f"  Misclassified: {n_errors} / {total} ({n_errors / total * 100:.1f}%)")

    # Summarize confusion pairs
    print("\n  Misclassified samples (raw feature values):")
    print(f"  {'True':<12} {'Predicted':<12} {'mean_gap':>10} "
          f"{'pay_ent':>8} {'burst':>6} {'recovery':>10} {'req_cnt':>8}")
    print(f"  {'-'*12} {'-'*12} {'-'*10} {'-'*8} {'-'*6} {'-'*10} {'-'*8}")

    for _, row in errors.iterrows():
        gap = f"{row['mean_gap_time']:.0f}ms" if pd.notna(row["mean_gap_time"]) else "NaN"
        pay = f"{row['payload_entropy']:.2f}" if pd.notna(row["payload_entropy"]) else "NaN"
        bur = f"{row['burstiness']:.2f}" if pd.notna(row["burstiness"]) else "NaN"
        rec = f"{row['recovery_time']:.0f}ms" if pd.notna(row["recovery_time"]) else "NaN"
        cnt = f"{row['request_count']:.0f}"
        print(f"  {row['true_label']:<12} {row['predicted']:<12} "
              f"{gap:>10} {pay:>8} {bur:>6} {rec:>10} {cnt:>8}")

    # Pattern detection: are errors concentrated among short sessions?
    short_session = errors["request_count"] <= 2
    if short_session.any():
        pct = short_session.sum() / len(errors) * 100
        print(f"\n  Pattern: {short_session.sum()}/{len(errors)} "
              f"({pct:.0f}%) misclassified samples have ≤2 requests "
              f"(NaN in 5+ features)")

    # Pattern detection: human-agent boundary
    ha_boundary = (
        ((errors["true_label"] == "human") & (errors["predicted"] == "llm_agent"))
        | ((errors["true_label"] == "llm_agent") & (errors["predicted"] == "human"))
    )
    if ha_boundary.any():
        print(f"  Pattern: {ha_boundary.sum()}/{len(errors)} misclassifications "
              f"are on the human↔agent boundary")
    return errors


#  SUMMARY

def print_summary(lr_report: dict, rf_report: dict,
                  lr_pred, rf_pred, y_test,
                  ablation: dict,
                  lr_features: list[str], rf_features: list[str],
                  lr_model, rf_model,
                  class_names: list[str]) -> None:
    """Print the results summary and headline finding."""
    lr_f1 = lr_report["macro avg"]["f1-score"]
    rf_f1 = rf_report["macro avg"]["f1-score"]
    best_name = "RF" if rf_f1 >= lr_f1 else "LR"
    best_report = rf_report if best_name == "RF" else lr_report
    best_f1 = max(lr_f1, rf_f1)
    best_pred = rf_pred if best_name == "RF" else lr_pred

    # Agent recall, this is basically a business-critical metric
    agent_recall = best_report["llm_agent"]["recall"]

    # Hardest confusion pair (largest off-diagonal cell)
    cm = confusion_matrix(y_test, best_pred)
    cm_off = cm.copy()
    np.fill_diagonal(cm_off, 0)
    worst_i, worst_j = np.unravel_index(cm_off.argmax(), cm_off.shape)
    worst_count = cm_off[worst_i, worst_j]
    worst_total = (y_test == worst_i).sum()

    # Top 3 features from best model
    if best_name == "RF":
        importances = rf_model.feature_importances_
        feat_names = rf_features
    else:
        importances = np.abs(lr_model.coef_).mean(axis=0)
        feat_names = lr_features
    top3_idx = np.argsort(importances)[-3:][::-1]
    top3 = [(feat_names[i], importances[i]) for i in top3_idx]

    # Ablation F1 scores
    timing_best = max(ablation["TIMING_LR"]["macro_f1"],
                      ablation["TIMING_RF"]["macro_f1"])
    struct_best = max(ablation["STRUCTURAL_LR"]["macro_f1"],
                      ablation["STRUCTURAL_RF"]["macro_f1"])

    # Ablation verdict
    if timing_best > struct_best + 0.05:
        verdict = "Timing features are the primary discriminator."
    elif struct_best > timing_best + 0.05:
        verdict = "Structural features dominate — timing alone is insufficient."
    else:
        verdict = "Both timing and structural features contribute comparably."

    # LR vs RF comparison interpretation
    gap = abs(rf_f1 - lr_f1)
    if gap < 0.03:
        lr_rf_interp = ("LR and RF perform comparably — features are "
                        "approximately linearly separable.")
    elif rf_f1 > lr_f1:
        lr_rf_interp = (f"RF outperforms LR by {gap:.1%} — nonlinear "
                        "interactions improve classification.")
    else:
        lr_rf_interp = (f"LR outperforms RF by {gap:.1%} — linear "
                        "boundaries suffice; RF may be overfitting.")

    #  Print 
    print(f"\n{'*'*60}")
    print("RESULTS SUMMARY")
    print(f"{'*'*60}")

    print(f"\n  Best model: {best_name} (macro F1 = {best_f1:.4f})")
    print(f"  LR macro F1: {lr_f1:.4f}  |  RF macro F1: {rf_f1:.4f}")
    print(f"  Interpretation: {lr_rf_interp}")

    print(f"\n  Per-class metrics ({best_name}):")
    print(f"  {'Class':<12} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print(f"  {'-'*12} {'-'*10} {'-'*10} {'-'*10}")
    for cls in class_names:
        p = best_report[cls]["precision"]
        r = best_report[cls]["recall"]
        f = best_report[cls]["f1-score"]
        tag = ""
        print(f"  {cls:<12} {p:>10.3f} {r:>10.3f} {f:>10.3f}{tag}")

    print(f"\n  Agent recall: {agent_recall:.1%}")

    if worst_count > 0:
        print(f"\n  Hardest pair: {worst_count}/{worst_total} "
              f"{class_names[worst_i]} predicted as {class_names[worst_j]} "
              f"({worst_count / worst_total:.1%})")

    print(f"\n  Top 3 features ({best_name}):")
    for name, imp in top3:
        print(f"    {name}: {imp:.4f}")

    print(f"\n  Ablation (macro F1):")
    print(f"    ALL:        LR={ablation['ALL_LR']['macro_f1']:.4f}  "
          f"RF={ablation['ALL_RF']['macro_f1']:.4f}")
    print(f"    TIMING:     LR={ablation['TIMING_LR']['macro_f1']:.4f}  "
          f"RF={ablation['TIMING_RF']['macro_f1']:.4f}")
    print(f"    STRUCTURAL: LR={ablation['STRUCTURAL_LR']['macro_f1']:.4f}  "
          f"RF={ablation['STRUCTURAL_RF']['macro_f1']:.4f}")
    print(f"  Verdict: {verdict}")

    print(f"\n{'*'*60}")
    print("HEADLINE FINDING")
    print(f"{'*'*60}")
    print(f"\n  Timing-based behavioral features achieve {best_f1:.1%} macro F1")
    print(f"  in distinguishing LLM agents from humans and script bots,")
    print(f"  with {top3[0][0]} and {top3[1][0]} as the top discriminators.")
    print(f"  Feature ablation: timing-only = {timing_best:.1%}, "
          f"structural-only = {struct_best:.1%}.")
    print(f"  {verdict}")
    print(f"  Agent recall = {agent_recall:.1%}.")
    print(f"\n{'*'*60}\n")


#  MAIN PIPELINE

def run_pipeline(input_path: Path) -> dict:
    """Execute the full classification pipeline.
    Returns a results dict for programmatic use"""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n{'*'*60}")
    print("Classification & Hypothesis Testing")
    print(f"{'*'*60}")
    print(f"  Input:  {input_path}")
    print(f"  Seed:   {RANDOM_SEED}\n")

    #  Load + Split 
    print("[1/7] Loading features and splitting data...")
    df = load_features(input_path)
    train_df, test_df, y_train, y_test, le = split_data(df)
    class_names = list(le.classes_)

    print(f"  Split: {len(y_train)} train / {len(y_test)} test")
    for cls in class_names:
        cls_i = le.transform([cls])[0]
        print(f"    {cls}: {(y_train == cls_i).sum()} train, "
              f"{(y_test == cls_i).sum()} test")

    #  Train Baseline Models (ALL features) 
    print("\n[2/7] Training baseline models...")
    X_train_lr, X_test_lr, lr_feat = prepare_features(
        train_df, test_df, ALL_FEATURES_LR
    )
    X_train_rf, X_test_rf, rf_feat = prepare_features(
        train_df, test_df, ALL_FEATURES
    )

    lr_model = make_lr()
    lr_model.fit(X_train_lr, y_train)
    print(f"  LR trained ({len(lr_feat)} features incl. indicators)")

    rf_model = make_rf()
    rf_model.fit(X_train_rf, y_train)
    print(f"  RF trained ({len(rf_feat)} features incl. indicators)")

    #  Evaluate 
    print("\n[3/7] Evaluating models...")
    lr_pred, lr_report, lr_cm = evaluate_model(
        lr_model, X_test_lr, y_test, class_names
    )
    rf_pred, rf_report, rf_cm = evaluate_model(
        rf_model, X_test_rf, y_test, class_names
    )
    print(f"  LR macro F1: {lr_report['macro avg']['f1-score']:.4f}")
    print(f"  RF macro F1: {rf_report['macro avg']['f1-score']:.4f}")

    plot_confusion_matrix(
        lr_cm, class_names, "Logistic Regression",
        FIGURES_DIR / "confusion_matrix_lr.png",
    )
    plot_confusion_matrix(
        rf_cm, class_names, "Random Forest",
        FIGURES_DIR / "confusion_matrix_rf.png",
    )

    save_metrics(lr_report, METRICS_DIR / "lr_results.json")
    save_metrics(rf_report, METRICS_DIR / "rf_results.json")

    #  ROC Curves 
    print("\n[4/7] Plotting ROC curves...")
    plot_roc_curves(
        {
            "Logistic Regression": (lr_model, X_test_lr),
            "Random Forest": (rf_model, X_test_rf),
        },
        y_test, class_names,
        FIGURES_DIR / "roc_curves.png",
    )

    #  Feature Importance 
    print("\n[5/7] Plotting feature importance...")
    plot_feature_importance(
        lr_model, rf_model, lr_feat, rf_feat,
        FIGURES_DIR / "feature_importance.png",
    )

    #  Ablation Study 
    print("\n[6/7] Running feature ablation study...")
    ablation = run_ablation(
        train_df, test_df, y_train, y_test, class_names
    )
    save_metrics(ablation, METRICS_DIR / "ablation_results.json")
    plot_ablation(ablation, FIGURES_DIR / "ablation_comparison.png")

    #  Error Analysis + Summary 
    print("\n[7/7] Error analysis and summary...")

    # Use best model's predictions for error analysis
    best_f1_lr = lr_report["macro avg"]["f1-score"]
    best_f1_rf = rf_report["macro avg"]["f1-score"]
    best_pred = rf_pred if best_f1_rf >= best_f1_lr else lr_pred

    errors_df = analyze_errors(test_df, y_test, best_pred, class_names)

    plot_killer_viz_with_errors(
        test_df, y_test, best_pred, class_names,
        FIGURES_DIR / "killer_viz_with_errors.png",
    )

    print_summary(
        lr_report, rf_report, lr_pred, rf_pred, y_test,
        ablation, lr_feat, rf_feat, lr_model, rf_model, class_names,
    )

    # Return results for programmatic use
    return {
        "lr_report": lr_report,
        "rf_report": rf_report,
        "ablation": ablation,
        "class_names": class_names,
        "errors": errors_df,
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the classification pipeline."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Path to features CSV (default: {DEFAULT_INPUT})",
    )
    args = parser.parse_args()
    run_pipeline(args.input)