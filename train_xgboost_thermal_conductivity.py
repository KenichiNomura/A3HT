#!/usr/bin/env python3
"""Train an XGBoost regressor on generated glassy-carbon ML features."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

try:
    import xgboost as xgb
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "xgboost is not installed in the current Python environment. "
        "Install it in the same environment you will use for training."
    ) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--features-csv",
        default=str(Path(__file__).resolve().parent / "ml_features.csv"),
        help="input feature table from build_ml_features.py",
    )
    parser.add_argument(
        "--target-column",
        default="target_final_thermal_conductivity",
        help="target column in the feature CSV",
    )
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).resolve().parent / "xgboost_thermal_conductivity_model"),
        help="directory for model and reports",
    )
    parser.add_argument("--test-fraction", type=float, default=0.2, help="holdout fraction when enough samples exist")
    parser.add_argument("--seed", type=int, default=7, help="random seed")
    parser.add_argument("--n-estimators", type=int, default=400, help="number of trees")
    parser.add_argument("--max-depth", type=int, default=4, help="tree depth")
    parser.add_argument("--learning-rate", type=float, default=0.05, help="boosting learning rate")
    parser.add_argument("--subsample", type=float, default=0.9, help="row subsampling")
    parser.add_argument("--colsample-bytree", type=float, default=0.9, help="feature subsampling")
    parser.add_argument("--reg-alpha", type=float, default=0.0, help="L1 regularization")
    parser.add_argument("--reg-lambda", type=float, default=1.0, help="L2 regularization")
    return parser.parse_args()


def load_feature_table(path: Path, target_column: str) -> Tuple[List[str], List[str], np.ndarray, np.ndarray]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    if not rows:
        raise SystemExit(f"feature file is empty: {path}")
    if target_column not in rows[0]:
        raise SystemExit(f"target column not found: {target_column}")

    run_ids = [row.get("run_id", f"row_{i}") for i, row in enumerate(rows)]
    candidate_columns = [c for c in rows[0].keys() if c not in {"run_id", target_column}]
    numeric_columns: List[str] = []
    matrix_cols: List[np.ndarray] = []
    for col in candidate_columns:
        values = []
        ok = True
        for row in rows:
            raw = row[col]
            try:
                value = float(raw)
            except (TypeError, ValueError):
                ok = False
                break
            values.append(np.nan if math.isnan(value) else value)
        if not ok:
            continue
        numeric_columns.append(col)
        matrix_cols.append(np.array(values, dtype=float))

    X = np.column_stack(matrix_cols) if matrix_cols else np.empty((len(rows), 0), dtype=float)
    y = np.array([float(row[target_column]) for row in rows], dtype=float)
    return run_ids, numeric_columns, X, y


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot == 0.0:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def make_split(n_rows: int, test_fraction: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    indices = list(range(n_rows))
    rng = random.Random(seed)
    rng.shuffle(indices)
    if n_rows < 5:
        return np.array(indices, dtype=int), np.array([], dtype=int)
    test_size = max(1, int(round(n_rows * test_fraction)))
    test_idx = np.array(sorted(indices[:test_size]), dtype=int)
    train_idx = np.array(sorted(indices[test_size:]), dtype=int)
    return train_idx, test_idx


def write_predictions(path: Path, run_ids: Sequence[str], y_true: np.ndarray, y_pred: np.ndarray, split: str) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["run_id", "split", "y_true", "y_pred", "residual"])
        writer.writeheader()
        for run_id, true, pred in zip(run_ids, y_true, y_pred):
            writer.writerow(
                {
                    "run_id": run_id,
                    "split": split,
                    "y_true": float(true),
                    "y_pred": float(pred),
                    "residual": float(pred - true),
                }
            )


def write_feature_importance(path: Path, booster: "xgb.Booster", feature_names: Sequence[str]) -> None:
    gain_map = booster.get_score(importance_type="gain")
    weight_map = booster.get_score(importance_type="weight")
    cover_map = booster.get_score(importance_type="cover")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["feature", "gain", "weight", "cover"])
        writer.writeheader()
        rows = []
        for idx, feature in enumerate(feature_names):
            key = f"f{idx}"
            rows.append(
                {
                    "feature": feature,
                    "gain": float(gain_map.get(key, 0.0)),
                    "weight": float(weight_map.get(key, 0.0)),
                    "cover": float(cover_map.get(key, 0.0)),
                }
            )
        rows.sort(key=lambda row: row["gain"], reverse=True)
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    features_csv = Path(args.features_csv).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    run_ids, feature_names, X, y = load_feature_table(features_csv, args.target_column)
    if X.shape[0] < 2:
        raise SystemExit("need at least 2 rows to train anything")
    if X.shape[1] == 0:
        raise SystemExit("no numeric feature columns were found")

    train_idx, test_idx = make_split(X.shape[0], args.test_fraction, args.seed)

    model = xgb.XGBRegressor(
        objective="reg:squarederror",
        random_state=args.seed,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        reg_alpha=args.reg_alpha,
        reg_lambda=args.reg_lambda,
        n_jobs=1,
    )

    if test_idx.size > 0:
        model.fit(X[train_idx], y[train_idx], eval_set=[(X[test_idx], y[test_idx])], verbose=False)
    else:
        model.fit(X[train_idx], y[train_idx], verbose=False)

    train_pred = model.predict(X[train_idx])
    train_metrics = {
        "rmse": rmse(y[train_idx], train_pred),
        "mae": mae(y[train_idx], train_pred),
        "r2": r2_score(y[train_idx], train_pred),
        "count": int(train_idx.size),
    }
    metrics = {"train": train_metrics}

    if test_idx.size > 0:
        test_pred = model.predict(X[test_idx])
        metrics["test"] = {
            "rmse": rmse(y[test_idx], test_pred),
            "mae": mae(y[test_idx], test_pred),
            "r2": r2_score(y[test_idx], test_pred),
            "count": int(test_idx.size),
        }
        write_predictions(output_dir / "test_predictions.csv", [run_ids[i] for i in test_idx], y[test_idx], test_pred, "test")
    else:
        metrics["test"] = None

    write_predictions(output_dir / "train_predictions.csv", [run_ids[i] for i in train_idx], y[train_idx], train_pred, "train")
    model.save_model(str(output_dir / "xgboost_model.json"))
    write_feature_importance(output_dir / "feature_importance.csv", model.get_booster(), feature_names)

    metadata = {
        "features_csv": str(features_csv),
        "target_column": args.target_column,
        "row_count": int(X.shape[0]),
        "feature_count": int(X.shape[1]),
        "feature_names": feature_names,
        "train_run_ids": [run_ids[i] for i in train_idx],
        "test_run_ids": [run_ids[i] for i in test_idx],
        "metrics": metrics,
        "params": {
            "n_estimators": args.n_estimators,
            "max_depth": args.max_depth,
            "learning_rate": args.learning_rate,
            "subsample": args.subsample,
            "colsample_bytree": args.colsample_bytree,
            "reg_alpha": args.reg_alpha,
            "reg_lambda": args.reg_lambda,
            "seed": args.seed,
        },
        "warning": None if test_idx.size > 0 else "Fewer than 5 rows available; model was trained without a holdout test split.",
    }
    (output_dir / "training_summary.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Model written to {output_dir / 'xgboost_model.json'}")
    print(f"Rows: {X.shape[0]}")
    print(f"Features: {X.shape[1]}")
    print(f"Train RMSE: {train_metrics['rmse']:.6f}")
    if metrics["test"] is not None:
        print(f"Test RMSE: {metrics['test']['rmse']:.6f}")
    else:
        print("Test RMSE: N/A (not enough rows for holdout split)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
