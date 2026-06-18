#!/usr/bin/env python3
"""
LAB 16 - CPU Baseline Benchmark (LightGBM)
Credit Card Fraud Detection - gradient boosting trên CPU instance.

Chạy:
    python3 benchmark.py

Script tự tìm file creditcard.csv (tải từ Kaggle: mlg-ulb/creditcardfraud).
Nếu không tìm thấy dataset, dùng dữ liệu synthetic mất cân bằng để vẫn chạy được
(khuyến nghị dùng dữ liệu Kaggle thật cho bài nộp).
"""
import json
import os
import time
import glob

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, accuracy_score, f1_score, precision_score, recall_score,
    precision_recall_curve, average_precision_score,
)

RESULT_FILE = "benchmark_result.json"


def load_dataset():
    """Trả về (X, y, source). Ưu tiên creditcard.csv, fallback synthetic."""
    candidates = [
        "creditcard.csv",
        os.path.expanduser("~/ml-benchmark/creditcard.csv"),
    ]
    candidates += glob.glob(os.path.expanduser("~/ml-benchmark/*.csv"))
    for path in candidates:
        if os.path.exists(path):
            print(f"[data] Đọc dataset Kaggle: {path}")
            t0 = time.perf_counter()
            df = pd.read_csv(path)
            load_time = time.perf_counter() - t0
            y = df["Class"].astype(int)
            X = df.drop(columns=["Class"])
            return X, y, f"kaggle:{os.path.basename(path)}", load_time

    print("[data] KHÔNG tìm thấy creditcard.csv -> dùng dữ liệu SYNTHETIC "
          "(chỉ để test, hãy tải dataset Kaggle thật cho bài nộp).")
    from sklearn.datasets import make_classification
    t0 = time.perf_counter()
    X_arr, y_arr = make_classification(
        n_samples=284_807, n_features=30, n_informative=12,
        n_redundant=8, weights=[0.998, 0.002], flip_y=0.001, random_state=42,
    )
    X = pd.DataFrame(X_arr, columns=[f"V{i}" for i in range(30)])
    y = pd.Series(y_arr, name="Class")
    load_time = time.perf_counter() - t0
    return X, y, "synthetic", load_time


def main():
    print("=" * 60)
    print(" LAB 16 - CPU Baseline: LightGBM Fraud Detection")
    print("=" * 60)

    X, y, source, load_time = load_dataset()
    n_pos = int(y.sum())
    print(f"[data] shape={X.shape}, fraud={n_pos} ({n_pos / len(y) * 100:.3f}%), "
          f"load_time={load_time:.3f}s")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.2, stratify=y_train, random_state=42
    )

    model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=3000,
        learning_rate=0.02,
        num_leaves=63,
        min_child_samples=50,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        force_col_wise=True,
        verbose=-1,
        n_jobs=-1,
        random_state=42,
    )

    print("[train] Bắt đầu training (early stopping trên AUC)...")
    t0 = time.perf_counter()
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        eval_metric="auc",
        callbacks=[lgb.early_stopping(200), lgb.log_evaluation(200)],
    )
    train_time = time.perf_counter() - t0
    best_iter = int(model.best_iteration_ or model.n_estimators)
    print(f"[train] Xong: {train_time:.2f}s, best_iteration={best_iter}")

    # Chọn ngưỡng phân loại tối ưu F1 trên tập validation (dữ liệu rất mất cân bằng
    # nên ngưỡng 0.5 mặc định không phù hợp)
    val_proba = model.predict_proba(X_val, num_iteration=best_iter)[:, 1]
    prec_v, rec_v, thr_v = precision_recall_curve(y_val, val_proba)
    f1_v = 2 * prec_v * rec_v / (prec_v + rec_v + 1e-12)
    best_idx = int(np.argmax(f1_v))
    threshold = float(thr_v[best_idx]) if best_idx < len(thr_v) else 0.5

    # Đánh giá trên test set
    proba = model.predict_proba(X_test, num_iteration=best_iter)[:, 1]
    pred = (proba >= threshold).astype(int)
    metrics = {
        "auc_roc": float(roc_auc_score(y_test, proba)),
        "auc_pr": float(average_precision_score(y_test, proba)),
        "accuracy": float(accuracy_score(y_test, pred)),
        "f1": float(f1_score(y_test, pred, zero_division=0)),
        "precision": float(precision_score(y_test, pred, zero_division=0)),
        "recall": float(recall_score(y_test, pred, zero_division=0)),
        "threshold": threshold,
    }

    # Inference latency (1 dòng) - trung bình qua nhiều lần
    one_row = X_test.iloc[[0]]
    for _ in range(10):  # warm-up
        model.predict_proba(one_row, num_iteration=best_iter)
    n_iter = 200
    t0 = time.perf_counter()
    for _ in range(n_iter):
        model.predict_proba(one_row, num_iteration=best_iter)
    latency_1row_ms = (time.perf_counter() - t0) / n_iter * 1000

    # Throughput (1000 dòng)
    batch = X_test.iloc[:1000]
    t0 = time.perf_counter()
    model.predict_proba(batch, num_iteration=best_iter)
    batch_time = time.perf_counter() - t0
    throughput_1000 = 1000 / batch_time

    result = {
        "dataset_source": source,
        "n_samples": int(len(y)),
        "n_features": int(X.shape[1]),
        "fraud_count": n_pos,
        "load_time_sec": round(load_time, 3),
        "train_time_sec": round(train_time, 3),
        "best_iteration": best_iter,
        **{k: round(v, 6) for k, v in metrics.items()},
        "inference_latency_1row_ms": round(latency_1row_ms, 4),
        "inference_throughput_1000rows_per_sec": round(throughput_1000, 1),
        "machine_type": os.environ.get("MACHINE_TYPE", "e2-standard-8"),
    }

    print("\n" + "=" * 60)
    print(" KẾT QUẢ BENCHMARK")
    print("=" * 60)
    for k, v in result.items():
        print(f"  {k:42s}: {v}")

    with open(RESULT_FILE, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n[done] Đã ghi metrics vào {os.path.abspath(RESULT_FILE)}")


if __name__ == "__main__":
    main()
