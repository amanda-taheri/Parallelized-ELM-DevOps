"""
Paper-aligned benchmark runner for the uploaded ELM project.

Default mode is multi-class KDD99 incident classification because the paper reports
KDD99 as 41 features, 23 classes, and 494,021 instances.
Use --task binary only for DevOps normal/anomaly experiments; those results should
not be compared directly with the paper's KDD99 table.
"""

import argparse
import os
import time

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

import settings
from src.elm_online import OnlineParallelELM


PAPER_BATCH_SIZES = [200, 300, 400, 500, 1000, 2000]


def load_kdd_data(file_path, task="multiclass", max_samples=0, random_state=42):
    df = pd.read_csv(file_path, compression="gzip", header=None, names=settings.KDD_COLUMNS)

    if max_samples and 0 < max_samples < len(df):
        df = df.sample(n=max_samples, random_state=random_state).reset_index(drop=True)

    X_df = df.drop("label", axis=1).copy()
    for col in ["protocol_type", "service", "flag"]:
        X_df[col] = LabelEncoder().fit_transform(X_df[col].astype(str))

    if task == "binary":
        y = (df["label"] != "normal.").astype(int).values
        label_names = np.array([0, 1])
    else:
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(df["label"].astype(str))
        label_names = label_encoder.classes_

    return X_df.values.astype(float), y, label_names


def make_minibatches(X, y, batch_size):
    for start in range(0, len(X), batch_size):
        end = min(start + batch_size, len(X))
        if end > start:
            yield X[start:end], y[start:end]


def train_stream(model, X_train, y_train, batch_size, workers, X_eval, y_eval):
    X_group, y_group = [], []
    for X_batch, y_batch in make_minibatches(X_train, y_train, batch_size):
        X_group.append(X_batch)
        y_group.append(y_batch)
        if len(X_group) == workers:
            model.learn_batches(X_group, y_group, X_eval=X_eval, y_eval=y_eval)
            X_group, y_group = [], []
    if X_group:
        model.learn_batches(X_group, y_group, X_eval=X_eval, y_eval=y_eval)


def run_once(X_train, y_train, X_test, y_test, classes, batch_size, workers, args):
    hidden_size = args.hidden_size or min(batch_size, X_train.shape[1])

    model = OnlineParallelELM(
        input_size=X_train.shape[1],
        hidden_size=hidden_size,
        n_workers=workers,
        activation=args.activation,
        kb_size=args.kb_size,
        kb_distance_threshold=args.kb_distance_threshold,
        min_reliability=args.min_reliability,
        classes=classes,
    )

    eval_size = min(args.eval_size, len(X_test))
    X_eval, y_eval = X_test[:eval_size], y_test[:eval_size]

    start = time.perf_counter()
    train_stream(model, X_train, y_train, batch_size, workers, X_eval, y_eval)
    train_time = time.perf_counter() - start

    y_pred = model.predict(X_test)
    result = {
        "batch_size": batch_size,
        "hidden_size": hidden_size,
        "workers": workers,
        "kb_vectors": len(model.synthesizer.knowledge_base),
        "accuracy": accuracy_score(y_test, y_pred),
        "precision_weighted": precision_score(y_test, y_pred, average="weighted", zero_division=0),
        "f1_weighted": f1_score(y_test, y_pred, average="weighted", zero_division=0),
        "train_time_sec": train_time,
    }
    if len(classes) == 2:
        result["anomaly_recall"] = recall_score(y_test, y_pred, zero_division=0)
    return result


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=settings.DATA_PATH)
    parser.add_argument("--task", choices=["multiclass", "binary"], default="multiclass")
    parser.add_argument("--max-samples", type=int, default=0, help="0 means full dataset")
    parser.add_argument("--test-size", type=float, default=0.20)
    parser.add_argument("--workers", type=int, default=2, choices=[2, 3])
    parser.add_argument("--batch-size", type=int, default=0, help="0 means paper batch sizes")
    parser.add_argument("--hidden-size", type=int, default=0, help="0 means min(batch_size, feature_dim)")
    parser.add_argument("--kb-size", type=int, default=20)
    parser.add_argument("--kb-distance-threshold", type=float, default=2.0)
    parser.add_argument("--min-reliability", type=float, default=None)
    parser.add_argument("--eval-size", type=int, default=10000)
    parser.add_argument("--activation", choices=["sigmoid", "relu", "linear"], default="sigmoid")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--output", default="assets/paper_aligned_results.csv")
    return parser.parse_args()


def main():
    args = parse_args()
    X, y, label_names = load_kdd_data(args.data, args.task, args.max_samples, args.random_state)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y,
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    batch_sizes = PAPER_BATCH_SIZES if args.batch_size == 0 else [args.batch_size]
    classes = np.unique(y_train)

    print("=" * 70)
    print("Paper-aligned Online Parallel ELM")
    print(f"Task: {args.task} | Samples: {len(X):,} | Features: {X.shape[1]} | Classes: {len(classes)}")
    print(f"Train/Test: {len(X_train):,}/{len(X_test):,} | Workers: {args.workers}")
    print("=" * 70)

    results = []
    for batch_size in batch_sizes:
        result = run_once(X_train, y_train, X_test, y_test, classes, batch_size, args.workers, args)
        results.append(result)
        print(
            f"batch={result['batch_size']:>4} hidden={result['hidden_size']:>3} "
            f"acc={result['accuracy']:.4f} precision={result['precision_weighted']:.4f} "
            f"f1={result['f1_weighted']:.4f} time={result['train_time_sec']:.3f}s "
            f"kb={result['kb_vectors']}"
        )

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    pd.DataFrame(results).to_csv(args.output, index=False)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
