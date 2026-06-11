"""
Paper-aligned benchmark runner for the ELM implementations.

It keeps the original project behavior that Amanda wanted:
- runs comparative benchmarks,
- saves charts under assets/,
- saves CSV results,
- updates the auto-generated benchmark section in README.md.

Main modes:
- --mode models: compare Sequential ELM, Sequential SVD-ELM, and Online Parallel ELM.
- --mode sweep: test Online Parallel ELM across the paper batch sizes.
- --mode both: do both and update the README with both result groups.

Paper-aligned defaults:
- task: multiclass KDD incident/attack-type classification
- train/test split: 80/20
- batch sizes: 200, 300, 400, 500, 1000, 2000
- hidden nodes: min(batch_size, feature_dimension) when --hidden-size 0
"""

import argparse
import os
import platform
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Iterable, List, Optional, Sequence, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

import settings
from src.elm_base import ELMBase
from src.elm_online import OnlineParallelELM
from src.elm_svd import ELMSVD


PAPER_BATCH_SIZES = [200, 300, 400, 500, 1000, 2000]
MODEL_RESULTS_CSV = os.path.join(settings.ASSETS_DIR, "benchmark_results.csv")
SWEEP_RESULTS_CSV = os.path.join(settings.ASSETS_DIR, "batch_sweep_results.csv")


@dataclass
class BenchmarkResult:
    model: str
    accuracy: float
    precision_weighted: float
    recall_weighted: float
    f1_weighted: float
    anomaly_recall: Optional[float]
    train_time_sec: float
    predict_time_sec: float
    throughput_samples_sec: float
    peak_cpu_percent: float
    peak_ram_mb: float
    report: str
    confusion: np.ndarray


class ResourceMonitor:
    """Lightweight CPU/RAM monitor for each benchmark run."""

    def __init__(self, interval: float = settings.RESOURCE_SAMPLE_INTERVAL):
        self.interval = interval
        self.process = psutil.Process(os.getpid())
        self.cpu_samples: List[float] = []
        self.ram_samples: List[float] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._sample, daemon=True)

    def __enter__(self):
        psutil.cpu_percent(interval=None)
        self.process.cpu_percent(interval=None)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, traceback):
        self._stop.set()
        self._thread.join()

    def _sample(self):
        while not self._stop.is_set():
            self.cpu_samples.append(psutil.cpu_percent(interval=None))
            self.ram_samples.append(self.process.memory_info().rss / (1024**2))
            time.sleep(self.interval)

    @property
    def peak_cpu(self) -> float:
        return max(self.cpu_samples) if self.cpu_samples else 0.0

    @property
    def peak_ram(self) -> float:
        return max(self.ram_samples) if self.ram_samples else 0.0


def load_kdd_data(
    file_path: str,
    task: str = "multiclass",
    max_samples: int = 0,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load KDD Cup 99 data.

    task="multiclass" keeps the attack/incident labels, which is closer to the paper.
    task="binary" converts labels into normal/anomaly for DevOps anomaly detection.
    """
    print(f"Loading dataset: {file_path}")
    df = pd.read_csv(file_path, compression="gzip", header=None, names=settings.KDD_COLUMNS)

    if max_samples and 0 < max_samples < len(df):
        df = df.sample(n=max_samples, random_state=random_state).reset_index(drop=True)

    X_df = df.drop("label", axis=1).copy()
    for col in ["protocol_type", "service", "flag"]:
        X_df[col] = LabelEncoder().fit_transform(X_df[col].astype(str))

    if task == "binary":
        y = (df["label"] != "normal.").astype(int).values
        label_names = np.array(["Normal", "Anomaly"])
    else:
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(df["label"].astype(str))
        label_names = label_encoder.classes_

    return X_df.values.astype(float), y, label_names


def resolve_hidden_size(requested_hidden_size: int, batch_size: int, feature_dim: int) -> int:
    """Paper rule: hidden nodes = min(batch_size, feature_dimension)."""
    if requested_hidden_size and requested_hidden_size > 0:
        return requested_hidden_size
    return min(batch_size, feature_dim)


def make_one_hot_targets(y: np.ndarray, classes: np.ndarray) -> np.ndarray:
    y = np.asarray(y)
    if len(classes) <= 2:
        class_to_index = {label: idx for idx, label in enumerate(classes)}
        return np.array([class_to_index[label] for label in y], dtype=float).reshape(-1, 1)

    class_to_index = {label: idx for idx, label in enumerate(classes)}
    target = np.zeros((len(y), len(classes)), dtype=float)
    for row, label in enumerate(y):
        target[row, class_to_index[label]] = 1.0
    return target


def labels_from_output(raw_output, classes: np.ndarray) -> np.ndarray:
    """Convert model scores or labels into final class labels."""
    raw_output = np.asarray(raw_output)

    # Some models already return final class labels.
    if raw_output.ndim == 1 and np.all(np.isin(raw_output, classes)):
        if len(classes) > 2:
            return raw_output.astype(classes.dtype, copy=False)
        # For binary, one-dimensional output may also be a continuous score.
        unique_values = np.unique(raw_output)
        if set(unique_values.tolist()).issubset(set(classes.tolist())):
            return raw_output.astype(classes.dtype, copy=False)

    if raw_output.ndim > 1 and raw_output.shape[1] > 1:
        return classes[np.argmax(raw_output, axis=1)]

    scores = raw_output.reshape(-1)
    if len(classes) == 2:
        encoded = (scores > 0.5).astype(int)
        return classes[encoded]

    raise ValueError("Multi-class predictions must be class labels or a 2D score matrix.")


def classification_report_labels(classes: np.ndarray, label_names: np.ndarray) -> List[str]:
    labels = []
    for label in classes:
        try:
            labels.append(str(label_names[int(label)]))
        except Exception:
            labels.append(str(label))
    return labels


def evaluate_model(
    name: str,
    train_fn: Callable[[], None],
    predict_fn: Callable[[np.ndarray], np.ndarray],
    X_train: np.ndarray,
    y_test: np.ndarray,
    X_test: np.ndarray,
    classes: np.ndarray,
    label_names: np.ndarray,
) -> BenchmarkResult:
    print(f"\nRunning: {name}")
    with ResourceMonitor() as monitor:
        train_start = time.perf_counter()
        train_fn()
        train_time = time.perf_counter() - train_start

        predict_start = time.perf_counter()
        y_pred = labels_from_output(predict_fn(X_test), classes)
        predict_time = time.perf_counter() - predict_start

    target_names = classification_report_labels(classes, label_names)
    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    anomaly_recall = None
    if len(classes) == 2 and 1 in classes:
        anomaly_recall = recall_score(y_test, y_pred, pos_label=1, zero_division=0)

    throughput = len(X_train) / train_time if train_time > 0 else 0.0
    report = classification_report(
        y_test,
        y_pred,
        labels=classes,
        target_names=target_names,
        zero_division=0,
    )
    cm = confusion_matrix(y_test, y_pred, labels=classes)

    print(f"Accuracy: {acc:.4f}")
    print(f"Precision weighted: {precision:.4f}")
    print(f"Recall weighted: {recall:.4f}")
    print(f"F1 weighted: {f1:.4f}")
    if anomaly_recall is not None:
        print(f"Anomaly Recall: {anomaly_recall:.4f}")
    print(f"Train Time: {train_time:.3f}s")
    print(f"Throughput: {throughput:,.0f} samples/sec")
    print(f"Peak CPU: {monitor.peak_cpu:.1f}% | Peak RAM: {monitor.peak_ram:.1f} MB")

    return BenchmarkResult(
        model=name,
        accuracy=acc,
        precision_weighted=precision,
        recall_weighted=recall,
        f1_weighted=f1,
        anomaly_recall=anomaly_recall,
        train_time_sec=train_time,
        predict_time_sec=predict_time,
        throughput_samples_sec=throughput,
        peak_cpu_percent=monitor.peak_cpu,
        peak_ram_mb=monitor.peak_ram,
        report=report,
        confusion=cm,
    )


def make_minibatches(X: np.ndarray, y: np.ndarray, batch_size: int) -> Iterable[Tuple[np.ndarray, np.ndarray]]:
    for start in range(0, len(X), batch_size):
        end = min(start + batch_size, len(X))
        if end > start:
            yield X[start:end], y[start:end]


def train_online_parallel(
    model: OnlineParallelELM,
    X_train: np.ndarray,
    y_train: np.ndarray,
    batch_size: int,
    workers: int,
    X_eval: np.ndarray,
    y_eval: np.ndarray,
) -> None:
    """Paper-style stream training: parallelize across independent mini-batches."""
    X_group, y_group = [], []
    for X_batch, y_batch in make_minibatches(X_train, y_train, batch_size):
        X_group.append(X_batch)
        y_group.append(y_batch)
        if len(X_group) == workers:
            if hasattr(model, "learn_batches"):
                model.learn_batches(X_group, y_group, X_eval=X_eval, y_eval=y_eval)
            else:
                # Backward-compatible fallback. Copy the patched src/elm_online.py for exact paper mode.
                for xb, yb in zip(X_group, y_group):
                    model.learn_batch(xb, yb, X_eval=X_eval, y_eval=y_eval)
            X_group, y_group = [], []

    if X_group:
        if hasattr(model, "learn_batches"):
            model.learn_batches(X_group, y_group, X_eval=X_eval, y_eval=y_eval)
        else:
            for xb, yb in zip(X_group, y_group):
                model.learn_batch(xb, yb, X_eval=X_eval, y_eval=y_eval)


def run_model_comparison(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    classes: np.ndarray,
    label_names: np.ndarray,
    args,
) -> List[BenchmarkResult]:
    hidden_size = resolve_hidden_size(args.hidden_size, args.batch_size, X_train.shape[1])
    eval_size = min(args.eval_size, len(X_test))
    X_eval = X_test[:eval_size]
    y_eval = y_test[:eval_size]

    baseline = ELMBase(
        input_size=X_train.shape[1],
        hidden_size=hidden_size,
        activation=args.activation,
    )
    svd_model = ELMSVD(
        input_size=X_train.shape[1],
        hidden_size=hidden_size,
        activation=args.activation,
    )
    parallel_kwargs = dict(
        input_size=X_train.shape[1],
        hidden_size=hidden_size,
        n_workers=args.workers,
        activation=args.activation,
        kb_size=args.kb_size,
        kb_distance_threshold=args.kb_distance_threshold,
        min_reliability=args.min_reliability,
    )
    try:
        parallel_model = OnlineParallelELM(**parallel_kwargs, classes=classes)
    except TypeError:
        parallel_model = OnlineParallelELM(**parallel_kwargs)
        if hasattr(parallel_model, "set_classes"):
            parallel_model.set_classes(classes)

    baseline_targets = make_one_hot_targets(y_train, classes)

    return [
        evaluate_model(
            "Sequential ELM",
            lambda: baseline.fit(X_train, baseline_targets),
            baseline.predict,
            X_train,
            y_test,
            X_test,
            classes,
            label_names,
        ),
        evaluate_model(
            "Sequential SVD-ELM",
            lambda: svd_model.fit(X_train, y_train),
            svd_model.predict,
            X_train,
            y_test,
            X_test,
            classes,
            label_names,
        ),
        evaluate_model(
            "Online Parallel ELM",
            lambda: train_online_parallel(
                parallel_model,
                X_train,
                y_train,
                args.batch_size,
                args.workers,
                X_eval,
                y_eval,
            ),
            parallel_model.predict,
            X_train,
            y_test,
            X_test,
            classes,
            label_names,
        ),
    ]


def run_batch_sweep(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    classes: np.ndarray,
    args,
) -> pd.DataFrame:
    batch_sizes = PAPER_BATCH_SIZES if args.batch_size == 0 else [args.batch_size]
    rows = []
    eval_size = min(args.eval_size, len(X_test))
    X_eval = X_test[:eval_size]
    y_eval = y_test[:eval_size]

    print("\n" + "=" * 70)
    print("PAPER BATCH-SIZE SWEEP: Online Parallel ELM")
    print("=" * 70)

    for batch_size in batch_sizes:
        hidden_size = resolve_hidden_size(args.hidden_size, batch_size, X_train.shape[1])
        parallel_kwargs = dict(
            input_size=X_train.shape[1],
            hidden_size=hidden_size,
            n_workers=args.workers,
            activation=args.activation,
            kb_size=args.kb_size,
            kb_distance_threshold=args.kb_distance_threshold,
            min_reliability=args.min_reliability,
        )
        try:
            model = OnlineParallelELM(**parallel_kwargs, classes=classes)
        except TypeError:
            model = OnlineParallelELM(**parallel_kwargs)
            if hasattr(model, "set_classes"):
                model.set_classes(classes)

        train_start = time.perf_counter()
        train_online_parallel(model, X_train, y_train, batch_size, args.workers, X_eval, y_eval)
        train_time = time.perf_counter() - train_start

        predict_start = time.perf_counter()
        y_pred = labels_from_output(model.predict(X_test), classes)
        predict_time = time.perf_counter() - predict_start

        row = {
            "Batch Size": batch_size,
            "Hidden Size": hidden_size,
            "Workers": args.workers,
            "KB Vectors": len(model.synthesizer.knowledge_base),
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision Weighted": precision_score(y_test, y_pred, average="weighted", zero_division=0),
            "Recall Weighted": recall_score(y_test, y_pred, average="weighted", zero_division=0),
            "F1 Weighted": f1_score(y_test, y_pred, average="weighted", zero_division=0),
            "Train Time (s)": train_time,
            "Predict Time (s)": predict_time,
            "Throughput (samples/s)": len(X_train) / train_time if train_time > 0 else 0.0,
        }
        if len(classes) == 2 and 1 in classes:
            row["Anomaly Recall"] = recall_score(y_test, y_pred, pos_label=1, zero_division=0)

        rows.append(row)
        print(
            f"batch={batch_size:>4} hidden={hidden_size:>3} "
            f"acc={row['Accuracy']:.4f} f1={row['F1 Weighted']:.4f} "
            f"time={train_time:.3f}s kb={row['KB Vectors']}"
        )

    return pd.DataFrame(rows)


def results_frame(results: Sequence[BenchmarkResult]) -> pd.DataFrame:
    rows = []
    for r in results:
        row = {
            "Model": r.model,
            "Accuracy": r.accuracy,
            "Precision Weighted": r.precision_weighted,
            "Recall Weighted": r.recall_weighted,
            "F1 Weighted": r.f1_weighted,
            "Train Time (s)": r.train_time_sec,
            "Predict Time (s)": r.predict_time_sec,
            "Throughput (samples/s)": r.throughput_samples_sec,
            "Peak CPU (%)": r.peak_cpu_percent,
            "Peak RAM (MB)": r.peak_ram_mb,
        }
        if r.anomaly_recall is not None:
            row["Anomaly Recall"] = r.anomaly_recall
        rows.append(row)
    return pd.DataFrame(rows)


def save_results(model_results: Optional[Sequence[BenchmarkResult]], sweep_df: Optional[pd.DataFrame]) -> None:
    os.makedirs(settings.ASSETS_DIR, exist_ok=True)
    if model_results:
        df = results_frame(model_results)
        df.to_csv(MODEL_RESULTS_CSV, index=False)
        print(f"Saved model metrics: {MODEL_RESULTS_CSV}")
    if sweep_df is not None and not sweep_df.empty:
        sweep_df.to_csv(SWEEP_RESULTS_CSV, index=False)
        print(f"Saved batch sweep metrics: {SWEEP_RESULTS_CSV}")


def _bar_chart(labels, series, title: str, ylabel: str, output_path: str, ylim=None) -> None:
    x = np.arange(len(labels))
    width = 0.8 / max(1, len(series))

    fig, ax = plt.subplots(figsize=(11, 5.5))
    for idx, (name, values) in enumerate(series.items()):
        offset = (idx - (len(series) - 1) / 2) * width
        ax.bar(x + offset, values, width, label=name)

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=10, ha="right")
    if ylim is not None:
        ax.set_ylim(*ylim)
    if len(series) > 1:
        ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def save_model_plots(results: Sequence[BenchmarkResult], classes: np.ndarray, label_names: np.ndarray) -> None:
    if not results:
        return

    df = results_frame(results)
    labels = df["Model"].tolist()

    _bar_chart(
        labels,
        {
            "Accuracy": df["Accuracy"].tolist(),
            "F1 weighted": df["F1 Weighted"].tolist(),
            "Precision weighted": df["Precision Weighted"].tolist(),
        },
        "Classification Metrics Comparison",
        "Score",
        os.path.join(settings.ASSETS_DIR, "accuracy.png"),
        ylim=(0, 1.05),
    )

    _bar_chart(
        labels,
        {"Train time": df["Train Time (s)"].tolist()},
        "Training Time Comparison",
        "Seconds",
        os.path.join(settings.ASSETS_DIR, "performance.png"),
    )

    _bar_chart(
        labels,
        {
            "Peak CPU (%)": df["Peak CPU (%)"].tolist(),
            "Peak RAM (MB)": df["Peak RAM (MB)"].tolist(),
        },
        "Peak Resource Usage",
        "Observed value",
        os.path.join(settings.ASSETS_DIR, "resources.png"),
    )

    best = max(results, key=lambda item: item.accuracy)
    tick_labels = classification_report_labels(classes, label_names)
    matrix_size = max(7, min(18, 0.35 * len(tick_labels) + 6))
    fig, ax = plt.subplots(figsize=(matrix_size, matrix_size))
    im = ax.imshow(best.confusion, interpolation="nearest")
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(f"Confusion Matrix: {best.model}")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(np.arange(len(tick_labels)))
    ax.set_yticks(np.arange(len(tick_labels)))
    ax.set_xticklabels(tick_labels, rotation=90, fontsize=8)
    ax.set_yticklabels(tick_labels, fontsize=8)

    # Annotate only small matrices; 23-class KDD matrices become unreadable if fully annotated.
    if len(tick_labels) <= 12:
        threshold = best.confusion.max() / 2 if best.confusion.size else 0
        for i in range(best.confusion.shape[0]):
            for j in range(best.confusion.shape[1]):
                value = best.confusion[i, j]
                ax.text(
                    j,
                    i,
                    str(value),
                    ha="center",
                    va="center",
                    color="white" if value > threshold else "black",
                    fontsize=8,
                )

    fig.tight_layout()
    fig.savefig(os.path.join(settings.ASSETS_DIR, "confusion_matrix.png"), dpi=300)
    plt.close(fig)


def save_sweep_plot(sweep_df: Optional[pd.DataFrame]) -> None:
    if sweep_df is None or sweep_df.empty:
        return

    fig, ax1 = plt.subplots(figsize=(10, 5.5))
    x = sweep_df["Batch Size"].astype(str).tolist()
    ax1.plot(x, sweep_df["Accuracy"], marker="o", label="Accuracy")
    ax1.plot(x, sweep_df["F1 Weighted"], marker="o", label="F1 weighted")
    ax1.set_title("Online Parallel ELM Across Paper Batch Sizes")
    ax1.set_xlabel("Batch size")
    ax1.set_ylabel("Score")
    ax1.set_ylim(0, 1.05)
    ax1.grid(axis="y", alpha=0.25)
    ax1.legend(loc="lower right")

    ax2 = ax1.twinx()
    ax2.plot(x, sweep_df["Train Time (s)"], marker="s", linestyle="--", label="Train time")
    ax2.set_ylabel("Train time (s)")
    ax2.legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(os.path.join(settings.ASSETS_DIR, "batch_sweep.png"), dpi=300)
    plt.close(fig)


def save_plots(
    model_results: Optional[Sequence[BenchmarkResult]],
    sweep_df: Optional[pd.DataFrame],
    classes: np.ndarray,
    label_names: np.ndarray,
) -> None:
    os.makedirs(settings.ASSETS_DIR, exist_ok=True)
    if model_results:
        save_model_plots(model_results, classes, label_names)
    if sweep_df is not None and not sweep_df.empty:
        save_sweep_plot(sweep_df)
    print(f"Saved charts in: {settings.ASSETS_DIR}/")


def markdown_model_table(results: Sequence[BenchmarkResult]) -> str:
    if not results:
        return ""

    has_anomaly = any(r.anomaly_recall is not None for r in results)
    header = "| Model | Accuracy | Precision | Recall | F1 | Train Time | Throughput | Peak CPU | Peak RAM |"
    align = "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
    if has_anomaly:
        header = "| Model | Accuracy | Precision | Recall | F1 | Anomaly Recall | Train Time | Throughput | Peak CPU | Peak RAM |"
        align = "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"

    lines = [header, align]
    for r in results:
        base = (
            f"| {r.model} | {r.accuracy:.4f} | {r.precision_weighted:.4f} | "
            f"{r.recall_weighted:.4f} | {r.f1_weighted:.4f} | "
        )
        if has_anomaly:
            anomaly_text = "-" if r.anomaly_recall is None else f"{r.anomaly_recall:.4f}"
            base += f"{anomaly_text} | "
        base += (
            f"{r.train_time_sec:.3f}s | {r.throughput_samples_sec:,.0f}/s | "
            f"{r.peak_cpu_percent:.1f}% | {r.peak_ram_mb:.1f} MB |"
        )
        lines.append(base)
    return "\n".join(lines)


def markdown_sweep_table(sweep_df: Optional[pd.DataFrame]) -> str:
    if sweep_df is None or sweep_df.empty:
        return ""

    lines = [
        "| Batch Size | Hidden Size | Accuracy | Precision | Recall | F1 | Train Time | KB Vectors |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, row in sweep_df.iterrows():
        lines.append(
            f"| {int(row['Batch Size'])} | {int(row['Hidden Size'])} | "
            f"{row['Accuracy']:.4f} | {row['Precision Weighted']:.4f} | "
            f"{row['Recall Weighted']:.4f} | {row['F1 Weighted']:.4f} | "
            f"{row['Train Time (s)']:.3f}s | {int(row['KB Vectors'])} |"
        )
    return "\n".join(lines)


def update_readme(
    model_results: Optional[Sequence[BenchmarkResult]],
    sweep_df: Optional[pd.DataFrame],
    args,
    total_samples: int,
    train_samples: int,
    test_samples: int,
    feature_dim: int,
    n_classes: int,
) -> None:
    if not os.path.exists(settings.README_PATH):
        print(f"README not found: {settings.README_PATH}. Skipping README update.")
        return

    start_marker = "<!-- BENCHMARK_RESULTS_START -->"
    end_marker = "<!-- BENCHMARK_RESULTS_END -->"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    hidden_text = (
        str(args.hidden_size)
        if args.hidden_size and args.hidden_size > 0
        else "min(batch_size, feature_dim)"
    )
    batch_text = "paper sweep: 200, 300, 400, 500, 1000, 2000" if args.batch_size == 0 else str(args.batch_size)
    task_text = "multi-class incident/attack-type classification" if args.task == "multiclass" else "binary normal/anomaly detection"

    model_section = ""
    if model_results:
        best_accuracy = max(model_results, key=lambda item: item.accuracy)
        fastest = min(model_results, key=lambda item: item.train_time_sec)
        model_section = f"""
### Model Comparison

{markdown_model_table(model_results)}

Best accuracy: **{best_accuracy.model}** ({best_accuracy.accuracy:.4f})  
Fastest training: **{fastest.model}** ({fastest.train_time_sec:.3f}s)

<p align="center">
  <img src="assets/accuracy.png" width="48%" />
  <img src="assets/performance.png" width="48%" />
  <img src="assets/resources.png" width="48%" />
  <img src="assets/confusion_matrix.png" width="48%" />
</p>
"""

    sweep_section = ""
    if sweep_df is not None and not sweep_df.empty:
        best_sweep = sweep_df.loc[sweep_df["Accuracy"].idxmax()]
        sweep_section = f"""
### Paper Batch-Size Sweep: Online Parallel ELM

{markdown_sweep_table(sweep_df)}

Best sweep accuracy: **batch size {int(best_sweep['Batch Size'])}** ({best_sweep['Accuracy']:.4f})

<p align="center">
  <img src="assets/batch_sweep.png" width="80%" />
</p>
"""

    section = f"""\
{start_marker}
## Auto-Generated Benchmark Results

Last updated: `{timestamp}`  
System: `{platform.system()} {platform.machine()}`  
Dataset: `{args.data}`  
Task: **{task_text}**  
Samples used: `{total_samples:,}` total, `{train_samples:,}` train, `{test_samples:,}` test  
Features: `{feature_dim}` | Classes: `{n_classes}`  
Configuration: hidden neurons = `{hidden_text}`, batch = `{batch_text}`, workers = `{args.workers}`, split = `{1 - args.test_size:.0%}/{args.test_size:.0%}`
{model_section}{sweep_section}
{end_marker}"""

    with open(settings.README_PATH, "r", encoding="utf-8") as file:
        readme = file.read()

    if start_marker in readme and end_marker in readme:
        before = readme.split(start_marker)[0].rstrip()
        after = readme.split(end_marker, 1)[1].lstrip()
        updated = f"{before}\n\n{section}\n\n{after}"
    else:
        insert_before = "## 📚 Reference"
        if insert_before in readme:
            updated = readme.replace(insert_before, f"{section}\n\n{insert_before}")
        else:
            updated = f"{readme.rstrip()}\n\n{section}\n"

    with open(settings.README_PATH, "w", encoding="utf-8") as file:
        file.write(updated)

    print(f"Updated README: {settings.README_PATH}")


def run_benchmark(args) -> None:
    os.makedirs(settings.ASSETS_DIR, exist_ok=True)

    if not os.path.exists(args.data):
        raise FileNotFoundError(f"Dataset not found: {args.data}")

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
    classes = np.unique(y_train)

    # If the user asks for the full paper sweep, use batch_size=0 internally.
    if args.mode in ["models", "both"] and args.batch_size == 0:
        args.batch_size = 2000

    print("=" * 70)
    print("ELM BENCHMARK")
    print("=" * 70)
    print(f"Task: {args.task}")
    print(f"Samples: {len(X):,} | Train: {len(X_train):,} | Test: {len(X_test):,}")
    print(f"Features: {X_train.shape[1]} | Classes: {len(classes)}")
    print(f"Hidden Rule: {'paper min(batch, feature)' if args.hidden_size == 0 else args.hidden_size}")
    print(f"Batch Size: {args.batch_size} | Workers: {args.workers}")
    print("=" * 70)

    model_results = None
    sweep_df = None

    if args.mode in ["models", "both"]:
        model_results = run_model_comparison(X_train, y_train, X_test, y_test, classes, label_names, args)

    if args.mode in ["sweep", "both"]:
        # For both mode we still want the full paper sweep, so pass a copy-like object.
        sweep_args = argparse.Namespace(**vars(args))
        if args.paper_sweep:
            sweep_args.batch_size = 0
        sweep_df = run_batch_sweep(X_train, y_train, X_test, y_test, classes, sweep_args)

    save_results(model_results, sweep_df)
    save_plots(model_results, sweep_df, classes, label_names)
    update_readme(
        model_results,
        sweep_df,
        args,
        len(X),
        len(X_train),
        len(X_test),
        X_train.shape[1],
        len(classes),
    )
    print("\nDone. Charts, CSV files, and README benchmark section were updated.")


def parse_args():
    parser = argparse.ArgumentParser(description="Compare ELM benchmark variants and update README charts.")
    parser.add_argument("--data", default=settings.DATA_PATH, help="Path to KDD .gz dataset.")
    parser.add_argument(
        "--task",
        choices=["multiclass", "binary"],
        default="multiclass",
        help="multiclass is closer to the paper; binary is DevOps normal/anomaly detection.",
    )
    parser.add_argument(
        "--mode",
        choices=["models", "sweep", "both"],
        default="both",
        help="models compares ELM variants; sweep tests paper batch sizes; both does both.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Limit rows for faster local benchmarks. Use 0 for the full dataset.",
    )
    parser.add_argument(
        "--hidden-size",
        type=int,
        default=0,
        help="0 applies the paper rule: min(batch_size, feature_dimension).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2000,
        help="Online mini-batch size for model comparison. Use --mode sweep for all paper batch sizes.",
    )
    parser.add_argument("--workers", type=int, default=2, help="Paper experiments used 2 or 3 parallel processes.")
    parser.add_argument("--test-size", type=float, default=0.20, help="Paper-aligned default is 0.20.")
    parser.add_argument("--eval-size", type=int, default=10_000)
    parser.add_argument("--kb-size", type=int, default=settings.KB_SIZE)
    parser.add_argument("--kb-distance-threshold", type=float, default=settings.KB_DISTANCE_THRESHOLD)
    parser.add_argument("--min-reliability", type=float, default=settings.MIN_RELIABILITY)
    parser.add_argument("--activation", choices=["sigmoid", "relu", "linear"], default=settings.ACTIVATION)
    parser.add_argument("--random-state", type=int, default=settings.RANDOM_STATE)
    parser.add_argument(
        "--no-paper-sweep",
        dest="paper_sweep",
        action="store_false",
        help="In --mode both, only sweep the selected --batch-size instead of all paper batch sizes.",
    )
    parser.set_defaults(paper_sweep=True)
    return parser.parse_args()


if __name__ == "__main__":
    run_benchmark(parse_args())
