"""Supervised position classifier using iconographic features.

Uses the JSON features embedded in the Indus corpus to train a classifier
that predicts whether a sign occurs in start, middle, or end position.
Tests whether visual/iconographic properties correlate with positional function.

Models tested: Random Forest, Logistic Regression.
Output: classification report, feature importance, confusion matrix.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.preprocessing import StandardScaler


def parse_feature_array(feature_json: str) -> list[float]:
    """Parse JSON feature array into a list of floats."""
    try:
        arr = json.loads(feature_json)
        if isinstance(arr, list):
            return [float(v) if v is not None else 0.0 for v in arr]
    except (json.JSONDecodeError, TypeError, ValueError):
        pass
    return []


def build_position_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Label each token as start, middle, or end based on position within line."""
    labels = []
    for _, group in df.groupby("line_id"):
        max_pos = group["position"].max()
        for _, row in group.iterrows():
            pos = row["position"]
            if pos == 1:
                labels.append("start")
            elif pos == max_pos:
                labels.append("end")
            else:
                labels.append("middle")
    df = df.copy()
    df["position_label"] = labels
    return df


def main():
    parser = argparse.ArgumentParser(description="Supervised position classifier")
    parser.add_argument("--input", default="data/raw/lost_language/corpus_indus_real.csv")
    parser.add_argument("--output-dir", default="reports/tables")
    parser.add_argument("--min-count-per-label", type=int, default=10)
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    df = build_position_labels(df)

    # Parse features and determine max length
    parsed_features = []
    max_len = 0
    for f in df["features"]:
        arr = parse_feature_array(f)
        parsed_features.append(arr)
        max_len = max(max_len, len(arr))

    # Build feature matrix with padding
    feature_matrix = []
    for arr in parsed_features:
        padded = arr + [0.0] * (max_len - len(arr))
        # Add summary stats as additional features
        if arr:
            padded.extend([
                float(len(arr)),
                float(np.mean(arr)),
                float(np.std(arr)) if len(arr) > 1 else 0.0,
                float(np.max(arr)),
                float(np.min(arr)),
                float(np.sum(arr)),
            ])
        else:
            padded.extend([0.0] * 6)
        feature_matrix.append(padded)

    X = np.array(feature_matrix)
    y = df["position_label"].to_numpy()

    # Filter labels with enough samples
    label_counts = pd.Series(y).value_counts()
    valid_labels = label_counts[label_counts >= args.min_count_per_label].index.tolist()
    mask = np.isin(y, valid_labels)
    X = X[mask]
    y = y[mask]

    if len(valid_labels) < 2:
        print(f"Insufficient label diversity after filtering: {valid_labels}")
        return

    # Remove constant features
    feature_vars = np.var(X, axis=0)
    non_constant = feature_vars > 1e-12
    if not np.any(non_constant):
        print("All features are constant. Cannot train classifier.")
        return

    X = X[:, non_constant]

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print(f"Samples: {len(y)}, Features: {X.shape[1]}, Labels: {valid_labels}")
    print(f"Class distribution:\n{pd.Series(y).value_counts().to_string()}")

    # Random Forest
    print("\n" + "=" * 60)
    print("RANDOM FOREST CLASSIFIER (5-fold CV)")
    print("=" * 60)
    rf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, class_weight="balanced")
    y_pred_rf = cross_val_predict(rf, X_scaled, y, cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42))
    print(classification_report(y, y_pred_rf, digits=3))

    # Feature importance (fit on full data for importance)
    rf.fit(X_scaled, y)
    feature_names = [f"feat_{i}" for i in range(max_len)] + ["len", "mean", "std", "max", "min", "sum"]
    feature_names = [feature_names[i] for i in range(len(feature_names)) if non_constant[i]]
    importance = pd.DataFrame({
        "feature": feature_names,
        "importance": rf.feature_importances_,
    }).sort_values("importance", ascending=False)
    print("\nTop 10 important features:")
    print(importance.head(10).to_string(index=False))

    # Confusion matrix
    cm = confusion_matrix(y, y_pred_rf, labels=valid_labels)
    cm_df = pd.DataFrame(cm, index=[f"true_{l}" for l in valid_labels], columns=[f"pred_{l}" for l in valid_labels])
    print("\nConfusion matrix:")
    print(cm_df.to_string())

    # Logistic Regression
    print("\n" + "=" * 60)
    print("LOGISTIC REGRESSION (5-fold CV)")
    print("=" * 60)
    lr = LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")
    y_pred_lr = cross_val_predict(lr, X_scaled, y, cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42))
    print(classification_report(y, y_pred_lr, digits=3))

    # Save outputs
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    importance.to_csv(out_dir / "position_classifier_feature_importance.csv", index=False)
    cm_df.to_csv(out_dir / "position_classifier_confusion_matrix.csv")

    # Save predictions for error analysis
    pred_df = pd.DataFrame({
        "true": y,
        "rf_pred": y_pred_rf,
        "lr_pred": y_pred_lr,
    })
    pred_df.to_csv(out_dir / "position_classifier_predictions.csv", index=False)
    print(f"\nSaved results to {out_dir}")


if __name__ == "__main__":
    main()
