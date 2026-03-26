"""Robustness Testing" Saves the phase 5 model artifacts, 
loads them for evaluation on perturbed data,
and provides the evaluation harness used by every robustness experiment.
Usage:
    python -m models.robustness --save-artifacts
    python -m models.robustness --experiment tarpit --input data/processed/features_tarpit.csv
"""

import argparse
import json
import logging
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

from api.app.config import PROJECT_ROOT, RANDOM_SEED
from models.classifiers import (
    ALL_FEATURES,
    ALL_FEATURES_LR,
    TIMING_FEATURES,
    load_features,
    split_data,
    prepare_features,
    make_lr,
    make_rf,
    evaluate_model,
    plot_confusion_matrix,
)
logger = logging.getLogger("honeypot.robustness")

#    Paths 
DEFAULT_INPUT   = PROJECT_ROOT / "data" / "processed" / "features.csv"
ARTIFACTS_DIR   = PROJECT_ROOT / "models" / "artifacts"
FIGURES_DIR     = PROJECT_ROOT / "report" / "figures"
METRICS_DIR     = PROJECT_ROOT / "report" / "metrics"

# Artifact filenames — referenced by both save and load
ARTIFACT_PATHS = {
    "rf":          ARTIFACTS_DIR / "rf_baseline.joblib",
    "lr":          ARTIFACTS_DIR / "lr_baseline.joblib",
    "label_enc":   ARTIFACTS_DIR / "label_encoder.joblib",
    "imputer_rf":  ARTIFACTS_DIR / "imputer_rf.joblib",
    "scaler_rf":   ARTIFACTS_DIR / "scaler_rf.joblib",
    "imputer_lr":  ARTIFACTS_DIR / "imputer_lr.joblib",
    "scaler_lr":   ARTIFACTS_DIR / "scaler_lr.joblib",
}

# Timing features to perturb in the noise curve experiment
TIMING_NOISE_COLS = ["mean_gap_time", "std_gap_time", "max_delay", "recovery_time"]
# Noise sigma levels (ms) for the robustness curve sweep
SIGMA_LEVELS = [0, 100, 500, 1000, 2000, 5000]


#    ARTIFACT SAVE

def save_artifacts(input_path: Path = DEFAULT_INPUT) -> None:
    """Retrain Phase 5 models (same seed, same split) and save all artifacts to disk.
    Prints a verification table"""

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n{'*'*60}")
    print(" Saving Model Artifacts")
    print(f"{'*'*60}")
    print(f"  Input:  {input_path}")
    print(f"  Output: {ARTIFACTS_DIR}\n")

    # Load + split 
    print("[1/4] Loading features and splitting data...")
    df = load_features(input_path)
    train_df, test_df, y_train, y_test, le = split_data(df, seed=RANDOM_SEED)
    class_names = list(le.classes_)
    print(f"  Split: {len(y_train)} train / {len(y_test)} test")

    # Prepare features for both models
    # prepare_features() returns fitted imputer + scaler baked into the arrays.
    # the fitted objects are needed separately, so its internals are replicated here.
    print("\n[2/4] Fitting preprocessors on training data...")
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler

    def fit_preprocessors(train_df, feature_cols):
        """Fit imputer and scaler on training data. Return arrays + fitted objects."""
        X_train = train_df[feature_cols].copy()
        # Missing indicators: binary flag for each column with NaN in training data
        nan_cols = [c for c in feature_cols if X_train[c].isna().any()]
        for col in nan_cols:
            X_train[f"{col}_missing"] = X_train[col].isna().astype(int)
        feature_names = list(X_train.columns)
        imputer = SimpleImputer(strategy="median")
        X_arr = imputer.fit_transform(X_train)
        scaler = StandardScaler()
        X_arr = scaler.fit_transform(X_arr)
        return X_arr, imputer, scaler, feature_names

    X_train_rf, imp_rf, scl_rf, rf_feat = fit_preprocessors(train_df, ALL_FEATURES)
    X_train_lr, imp_lr, scl_lr, lr_feat = fit_preprocessors(train_df, ALL_FEATURES_LR)
    print(f"  RF features: {len(rf_feat)}  |  LR features: {len(lr_feat)}")

    #    Train models (identical hyperparameters to Phase 5)
    print("\n[3/4] Training baseline models...")
    rf = make_rf(seed=RANDOM_SEED)
    rf.fit(X_train_rf, y_train)
    lr = make_lr(seed=RANDOM_SEED)
    lr.fit(X_train_lr, y_train)
    print("  RF trained  |  LR trained")

    # Save all 7 artifacts
    print("\n[4/4] Saving artifacts...")
    joblib.dump(rf,    ARTIFACT_PATHS["rf"])
    joblib.dump(lr,    ARTIFACT_PATHS["lr"])
    joblib.dump(le,    ARTIFACT_PATHS["label_enc"])
    joblib.dump(imp_rf, ARTIFACT_PATHS["imputer_rf"])
    joblib.dump(scl_rf, ARTIFACT_PATHS["scaler_rf"])
    joblib.dump(imp_lr, ARTIFACT_PATHS["imputer_lr"])
    joblib.dump(scl_lr, ARTIFACT_PATHS["scaler_lr"])
    for key, path in ARTIFACT_PATHS.items():
        print(f"  Saved: {path.name}")

    # Verification: evaluate on test set, confirm F1 matches Phase 5
    print("\n  Verification :")
    print(f"  {'Model':<6} {'Expected':>10} {'Actual':>10} {'Match':>8}")
    print(f"  {'-'*6} {'-'*10} {'-'*10} {'-'*8}")

    # Apply saved preprocessors to test set (transform only — no refit)
    X_test_rf = _apply_preprocessors(test_df, ALL_FEATURES, imp_rf, scl_rf)
    X_test_lr = _apply_preprocessors(test_df, ALL_FEATURES_LR, imp_lr, scl_lr)

    _, rf_report, _ = evaluate_model(rf, X_test_rf, y_test, class_names)
    _, lr_report, _ = evaluate_model(lr, X_test_lr, y_test, class_names)

    rf_f1 = rf_report["macro avg"]["f1-score"]
    lr_f1 = lr_report["macro avg"]["f1-score"]

    rf_match = "✓" if abs(rf_f1 - 1.0000) < 0.001 else "✗ CHECK"
    lr_match = "✓" if abs(lr_f1 - 0.9916) < 0.005 else "✗ CHECK"

    print(f"  {'RF':<6} {1.0000:>10.4f} {rf_f1:>10.4f} {rf_match:>8}")
    print(f"  {'LR':<6} {0.9916:>10.4f} {lr_f1:>10.4f} {lr_match:>8}")
    print(f"\n  Artifacts saved to: {ARTIFACTS_DIR}")
    print(f"{'*'*60}\n")


#   ARTIFACT LOAD

def load_artifacts() -> dict:
    """Load all Phase 5 artifacts from disk.
    Returns a dict with keys: rf, lr, label_enc,
    imputer_rf, scaler_rf, imputer_lr, scaler_lr."""
    missing = [k for k, p in ARTIFACT_PATHS.items() if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing artifacts: {missing}. Run: python -m models.robustness --save-artifacts"
        )
    return {key: joblib.load(path) for key, path in ARTIFACT_PATHS.items()}


#    PREPROCESSING HELPER

def _apply_preprocessors(
    df: pd.DataFrame,
    feature_cols: list[str],
    imputer,
    scaler,
) -> np.ndarray:
    """Apply saved imputer and scaler to a new DataFrame.
    Adds missing indicators for NaN columns, then aligns to the exact
    feature set the imputer was fit on (padding absent indicators with 0)."""
    X = df[feature_cols].copy()

    # Add missing indicators for any NaN columns in this data
    nan_cols = [c for c in feature_cols if X[c].isna().any()]
    for col in nan_cols:
        X[f"{col}_missing"] = X[col].isna().astype(int)

    # Align to the exact columns the imputer was fit on.
    # If a missing indicator column was created at fit time but this data
    # has no NaNs (so the column was never added above), fill it with 0.
    expected_cols = list(imputer.feature_names_in_)
    for col in expected_cols:
        if col not in X.columns:
            X[col] = 0  # no NaNs in this data → indicator is all-zero

    # Reorder to match fit-time column order exactly
    X = X[expected_cols]

    X_arr = imputer.transform(X)
    X_arr = scaler.transform(X_arr)
    return X_arr


#    EXPERIMENT EVALUATOR

def evaluate_on_new_data(
    input_path: Path,
    experiment_name: str,
    save_figures: bool = True,
) -> dict:
    """Load saved artifacts. Apply to new features CSV. Return metrics dict.
    This is the core harness for every robustness experiment.
    Args:
        input_path:      Path to a features CSV produced by features/pipeline.py
        experiment_name: Label for figures and metrics (e.g., "tarpit", "concurrent")
        save_figures:    Whether to save confusion matrix PNGs to report/figures/
    Returns:
        Dict with rf_f1, lr_f1, rf_report, lr_report, class_names, agent_recall
    """
    print(f"\n{'*'*60}")
    print(f"Robustness Evaluation — {experiment_name}")
    print(f"{'*'*60}")
    print(f"  Input: {input_path}")

    #  Load artifacts 
    artifacts = load_artifacts()
    rf        = artifacts["rf"]
    lr        = artifacts["lr"]
    le        = artifacts["label_enc"]
    class_names = list(le.classes_)

    # Load new features CSV 
    df = load_features(input_path)
    y = le.transform(df["actor_type"])
    print(f"  Sessions: {len(df)}  |  Classes: {class_names}")

    # Apply saved preprocessors (NO refitting)
    X_rf = _apply_preprocessors(df, ALL_FEATURES,    artifacts["imputer_rf"], artifacts["scaler_rf"])
    X_lr = _apply_preprocessors(df, ALL_FEATURES_LR, artifacts["imputer_lr"], artifacts["scaler_lr"])

    # Evaluate
    rf_pred, rf_report, rf_cm = evaluate_model(rf, X_rf, y, class_names)
    lr_pred, lr_report, lr_cm = evaluate_model(lr, X_lr, y, class_names)

    rf_f1 = rf_report["macro avg"]["f1-score"]
    lr_f1 = lr_report["macro avg"]["f1-score"]

    # Agent recall 
    agent_idx   = list(le.classes_).index("llm_agent")
    agent_recall = rf_report["llm_agent"]["recall"]

    #  Print results 
    print(f"\n  RF macro F1:  {rf_f1:.4f}")
    print(f"  LR macro F1:  {lr_f1:.4f}")
    print(f"  Agent recall: {agent_recall:.1%}")
    print(f"\n  Per-class (RF):")
    print(f"  {'Class':<12} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print(f"  {'-'*12} {'-'*10} {'-'*10} {'-'*10}")
    for cls in class_names:
        p = rf_report[cls]["precision"]
        r = rf_report[cls]["recall"]
        f = rf_report[cls]["f1-score"]
        tag = ""
        print(f"  {cls:<12} {p:>10.3f} {r:>10.3f} {f:>10.3f}{tag}")

    #    Save confusion matrices
    if save_figures:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        plot_confusion_matrix(
            rf_cm, class_names,
            f"RF — {experiment_name}",
            FIGURES_DIR / f"confusion_matrix_rf_{experiment_name}.png",
        )
        plot_confusion_matrix(
            lr_cm, class_names,
            f"LR — {experiment_name}",
            FIGURES_DIR / f"confusion_matrix_lr_{experiment_name}.png",
        )

    return {
        "experiment":   experiment_name,
        "rf_f1":        round(rf_f1, 4),
        "lr_f1":        round(lr_f1, 4),
        "agent_recall": round(agent_recall, 4),
        "rf_report":    rf_report,
        "lr_report":    lr_report,
        "class_names":  class_names,
        "n_sessions":   len(df),
    }


#  NOISE CURVE EXPERIMENT 

def inject_timing_noise(df: pd.DataFrame, sigma_ms: float, seed: int) -> pd.DataFrame:
    """Add zero-mean Gaussian noise to timing features. Clip at 0 (gaps can't be negative).
    Uses a seeded RNG for reproducibility."""
    rng = np.random.default_rng(seed)
    noisy = df.copy()
    for col in TIMING_NOISE_COLS:
        if col not in noisy.columns:
            continue
        mask = noisy[col].notna()
        noise = rng.normal(0, sigma_ms, mask.sum())
        noisy.loc[mask, col] = (noisy.loc[mask, col] + noise).clip(lower=0)
    return noisy


def run_noise_curve(input_path: Path = DEFAULT_INPUT) -> dict:
    """Sweep Gaussian noise sigma from 0 to 5000ms across timing features.
    Re-classifies with saved RF and LR at each level. Returns results dict
    and saves robustness_curve.png to report/figures/."""
    print(f"\n{'*'*60}")
    print("Robustness Curve — F1 vs Timing Noise")
    print(f"{'*'*60}")

    artifacts = load_artifacts()
    rf = artifacts["rf"]
    lr = artifacts["lr"]
    le = artifacts["label_enc"]
    class_names = list(le.classes_)

    df = load_features(input_path)
    y  = le.transform(df["actor_type"])

    results = {}
    print(f"\n  {'Sigma (ms)':>12} {'RF F1':>10} {'LR F1':>10}")
    print(f"  {'-'*12} {'-'*10} {'-'*10}")

    for sigma in SIGMA_LEVELS:
        noisy_df = inject_timing_noise(df, sigma_ms=sigma, seed=RANDOM_SEED)

        X_rf = _apply_preprocessors(noisy_df, ALL_FEATURES,    artifacts["imputer_rf"], artifacts["scaler_rf"])
        X_lr = _apply_preprocessors(noisy_df, ALL_FEATURES_LR, artifacts["imputer_lr"], artifacts["scaler_lr"])

        _, rf_report, _ = evaluate_model(rf, X_rf, y, class_names)
        _, lr_report, _ = evaluate_model(lr, X_lr, y, class_names)

        rf_f1 = round(rf_report["macro avg"]["f1-score"], 4)
        lr_f1 = round(lr_report["macro avg"]["f1-score"], 4)
        results[sigma] = {"rf_f1": rf_f1, "lr_f1": lr_f1}
        print(f"  {sigma:>12} {rf_f1:>10.4f} {lr_f1:>10.4f}")

    #  Plot the curve 
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    sigmas = list(results.keys())
    rf_f1s = [results[s]["rf_f1"] for s in sigmas]
    lr_f1s = [results[s]["lr_f1"] for s in sigmas]

    ax.plot(sigmas, rf_f1s, marker="o", color="#2ecc71", linewidth=2, label="Random Forest")
    ax.plot(sigmas, lr_f1s, marker="s", color="#3498db", linewidth=2, label="Logistic Regression")
    ax.axhline(0.90, color="red", linestyle="--", linewidth=1, alpha=0.7, label="0.90 threshold")
    ax.set_xlabel("Timing Noise σ (ms)")
    ax.set_ylabel("Macro F1")
    ax.set_title("Robustness Curve: Macro F1 vs Injected Timing Noise")
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path = FIGURES_DIR / "robustness_curve.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"\n  Saved: {out_path}")

    return results

def build_robustness_report() -> dict:
    """Compile all experiment results into a comparison table, bar chart,
    and robustness verdict. Handles missing experiment files gracefully —
    experiments not yet run are marked 'pending' in the table."""

    import warnings
    warnings.filterwarnings("ignore")

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n{'*' * 60}")
    print("Robustness Report")
    print(f"{'*' * 60}\n")

    # Baseline (Phase 5 results from classifiers.py metrics) 
    baseline = {"rf_f1": 1.0000, "lr_f1": 0.9916, "agent_recall": 1.0000, "n_sessions": 120}

    #  Load experiment results 
    # Each experiment writes its metrics to METRICS_DIR/robustness_{name}.json
    experiment_files = {
        "tarpit":    METRICS_DIR / "robustness_tarpit.json",
        "concurrent": METRICS_DIR / "robustness_concurrent.json",
        "ood_model": METRICS_DIR / "robustness_ood_model.json",
        "ood_prompt": METRICS_DIR / "robustness_ood_prompt.json",
    }

    experiment_labels = {
        "tarpit":    "Tarpitting (429→200+delay)",
        "concurrent": "Concurrent Simulation",
        "ood_model": "OOD Agent (Mistral)",
        "ood_prompt": "OOD Agent (Neutral Prompt)",
    }

    results = {"baseline": baseline}
    for name, path in experiment_files.items():
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            results[name] = {
                "rf_f1": data.get("rf_f1", 0.0),
                "lr_f1": data.get("lr_f1", 0.0),
                "agent_recall": data.get("agent_recall", 0.0),
                "n_sessions": data.get("n_sessions", 0),
            }
        else:
            results[name] = None  # pending

    #  Print comparison table 
    print(f"  {'Condition':<32} {'RF F1':>8} {'LR F1':>8} {'Agent Recall':>14} {'vs Baseline':>12}")
    print(f"  {'-'*32} {'-'*8} {'-'*8} {'-'*14} {'-'*12}")

    def _delta(val, base=1.0000):
        diff = val - base
        return f"{diff:+.4f}"

    # Baseline row
    print(f"  {'Phase 5 Baseline':<32} {baseline['rf_f1']:>8.4f} {baseline['lr_f1']:>8.4f} "
          f"{baseline['agent_recall']:>14.1%} {'(baseline)':>12}")

    for name, label in experiment_labels.items():
        r = results.get(name)
        if r is None:
            print(f"  {label:<32} {'PENDING':>8} {'PENDING':>8} {'PENDING':>14} {'':>12}")
        else:
            # OOD experiments only have agent sessions — macro F1 is not comparable
            is_agent_only = name in ("ood_model", "ood_prompt")
            if is_agent_only:
                print(f"  {label:<32} {'N/A':>8} {'N/A':>8} "
                      f"{r['agent_recall']:>14.1%} {'(agent recall only)':>18}")
            else:
                delta = _delta(r["rf_f1"], baseline["rf_f1"])
                print(f"  {label:<32} {r['rf_f1']:>8.4f} {r['lr_f1']:>8.4f} "
                      f"{r['agent_recall']:>14.1%} {delta:>12}")

    # Robustness verdict 
    completed = {k: v for k, v in results.items() if v is not None and k != "baseline"}
    # Excludes agent-only experiments from macro F1 verdict because their F1 is not comparable
    comparable = {k: v for k, v in completed.items() if k not in ("ood_model", "ood_prompt")}
    if completed:
        min_rf = min(v["rf_f1"] for v in comparable.values()) if comparable else 1.0
        min_agent_recall = min(v["agent_recall"] for v in completed.values())
       
        print(f"\n  Min RF F1 across all experiments: {min_rf:.4f}")
        print(f"  Min agent recall across all experiments: {min_agent_recall:.1%}")

        if min_rf >= 0.95:
            verdict = "ROBUST"
            verdict_detail = (
                "The classifier maintains >95% macro F1 across all tested perturbations. "
                "Timing-based behavioral detection is robust under adversarial conditions, "
                "concurrent load, and out-of-distribution LLM agents."
            )
        elif min_rf >= 0.90:
            worst = min(completed, key=lambda k: completed[k]["rf_f1"])
            verdict = "PARTIALLY ROBUST"
            verdict_detail = (
                f"The classifier maintains >90% macro F1 across most conditions. "
                f"The weakest condition is '{experiment_labels[worst]}' "
                f"(RF F1 = {completed[worst]['rf_f1']:.4f}), which represents "
                f"the primary limit of timing-based detection."
            )
        else:
            worst = min(completed, key=lambda k: completed[k]["rf_f1"])
            verdict = "PARTIALLY FRAGILE"
            verdict_detail = (
                f"The classifier degrades significantly under '{experiment_labels[worst]}' "
                f"(RF F1 = {completed[worst]['rf_f1']:.4f}). Structural features provide "
                f"partial fallback but timing corruption is a real evasion vector."
            )
    else:
        verdict = "PENDING"
        verdict_detail = "No experiments completed yet."

    print(f"\n{'*' * 60}")
    print(f"ROBUSTNESS VERDICT: {verdict}")
    print(f"{'*' * 60}")
    print(f"\n  {verdict_detail}\n")

    #  Comparison bar chart 
    completed_for_plot = {k: v for k, v in results.items() if v is not None}
    if len(completed_for_plot) >= 2:
        labels_plot = []
        rf_vals = []
        lr_vals = []

        label_map = {
            "baseline":  "Baseline\n(Phase 5)",
            "tarpit":    "Tarpitting",
            "concurrent": "Concurrent",
            "ood_model": "OOD\n(Mistral)",
            "ood_prompt": "OOD\n(Neutral)",
        }

        for key in ["baseline", "tarpit", "concurrent", "ood_model", "ood_prompt"]:
            if key in completed_for_plot:
                labels_plot.append(label_map[key])
                rf_vals.append(completed_for_plot[key]["rf_f1"])
                lr_vals.append(completed_for_plot[key]["lr_f1"])

        x = np.arange(len(labels_plot))
        width = 0.35

        fig, ax = plt.subplots(figsize=(10, 5))
        bars_rf = ax.bar(x - width/2, rf_vals, width, label="Random Forest",
                         color="#2ecc71", alpha=0.85)
        bars_lr = ax.bar(x + width/2, lr_vals, width, label="Logistic Regression",
                         color="#3498db", alpha=0.85)

        # Value labels on bars
        for bar in bars_rf:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                    f"{bar.get_height():.4f}", ha="center", va="bottom", fontsize=8)
        for bar in bars_lr:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                    f"{bar.get_height():.4f}", ha="center", va="bottom", fontsize=8)

        ax.axhline(0.90, color="red", linestyle="--", linewidth=1.2,
                   alpha=0.7, label="0.90 robustness threshold")
        ax.axhline(0.95, color="orange", linestyle=":", linewidth=1.0,
                   alpha=0.7, label="0.95 target threshold")

        ax.set_xlabel("Experiment Condition")
        ax.set_ylabel("Macro F1")
        ax.set_title("Phase 6 Robustness: Macro F1 Across All Conditions")
        ax.set_xticks(x)
        ax.set_xticklabels(labels_plot)
        ax.set_ylim(0.85, 1.02)
        ax.legend(loc="lower right", fontsize=9)
        ax.grid(True, alpha=0.3, axis="y")
        fig.tight_layout()

        out_path = FIGURES_DIR / "robustness_comparison.png"
        fig.savefig(out_path, dpi=300)
        plt.close(fig)
        print(f"  Saved: {out_path}")

    #  Save combined results 
    combined_path = METRICS_DIR / "robustness_results.json"
    save_robustness_results(results, combined_path)

    return {"verdict": verdict, "results": results}


#  COMPARISON TABLE

def save_robustness_results(results: dict, path: Path) -> None:
    """Write all experiment results to a single JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  Saved: {path}")


#  CLI 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=" Robustness testing.")
    parser.add_argument("--save-artifacts", action="store_true",
                        help="Retrain Phase 5 models and save artifacts to models/artifacts/")
    parser.add_argument("--experiment", type=str,
                        choices=["tarpit", "concurrent", "ood_model", "ood_prompt", "noise_curve"],
                        help="Which robustness experiment to run")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT,
                        help="Path to features CSV (used by --experiment)")
    parser.add_argument("--report", action="store_true",
                        help="Compile all experiment results into the robustness report")
    args = parser.parse_args()

    if args.save_artifacts:
        save_artifacts(args.input)
    elif args.experiment == "noise_curve":
        run_noise_curve(args.input)
    elif args.experiment:
        results = evaluate_on_new_data(args.input, experiment_name=args.experiment)
        out = METRICS_DIR / f"robustness_{args.experiment}.json"
        save_robustness_results(results, out)
    elif args.report:
        build_robustness_report()
    else:
        parser.print_help()
