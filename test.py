"""
Benchmark runner for the ELM implementations.

It compares:
- Traditional sequential ELM with random weights
- Sequential SVD-ELM
- Online Parallel ELM with Weight Synthesizer and Knowledge Base

Outputs:
- assets/accuracy.png
- assets/performance.png
- assets/resources.png
- assets/benchmark_results.csv
- an auto-generated benchmark section in README.md
"""

import argparse
import os
import platform
import threading
import time
from dataclasses import dataclass
from datetime import datetime

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from src.elm_base import ELMBase
from src.elm_online import OnlineParallelELM
from src.elm_svd import ELMSVD


DATA_PATH = "data/kddcup.data_10_percent.gz"
ASSETS_DIR = "assets"
README_PATH = "README.md"
RESULTS_CSV = os.path.join(ASSETS_DIR, "benchmark_results.csv")


KDD_COLUMNS = [
    "duration", "protocol_type", "service", "flag", "src_bytes",
    "dst_bytes", "land", "wrong_fragment", "urgent", "hot",
    "num_failed_logins", "logged_in", "num_compromised", "root_shell",
    "su_attempted", "num_root", "num_file_creations", "num_shells",
    "num_access_files", "num_outbound_cmds", "is_host_login",
    "is_guest_login", "count", "srv_count", "serror_rate",
    "srv_serror_rate", "rerror_rate", "srv_rerror_rate",
    "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate",
    "dst_host_count", "dst_host_srv_count", "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
    "dst_host_srv_serror_rate", "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate", "label",
]


@dataclass
class BenchmarkResult:
    model: str
    accuracy: float
    anomaly_recall: float
    train_time_sec: float
    predict_time_sec: float
    throughput_samples_sec: float
    peak_cpu_percent: float
    peak_ram_mb: float
    report: str
    confusion: np.ndarray


class ResourceMonitor:
    def __init__(self, interval=0.1):
        self.interval = interval
        self.process = psutil.Process(os.getpid())
        self.cpu_samples = []
        self.ram_samples = []
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
            self.ram_samples.append(self.process.memory_info().rss / (1024 ** 2))
            time.sleep(self.interval)

    @property
    def peak_cpu(self):
        return max(self.cpu_samples) if self.cpu_samples else 0.0

    @property
    def peak_ram(self):
        return max(self.ram_samples) if self.ram_samples else 0.0


def load_kdd_data(file_path, max_samples=None, random_state=42):
    print(f"Loading dataset: {file_path}")
    df = pd.read_csv(file_path, compression="gzip", header=None, names=KDD_COLUMNS)

    if max_samples and max_samples > 0 and max_samples < len(df):
        df = df.sample(n=max_samples, random_state=random_state).reset_index(drop=True)

    y = (df["label"] != "normal.").astype(int).values
    X_df = df.drop("label", axis=1).copy()

    for col in ["protocol_type", "service", "flag"]:
        X_df[col] = LabelEncoder().fit_transform(X_df[col])

    return X_df.values.astype(float), y


def to_binary_predictions(raw_predictions):
    raw_predictions = np.asarray(raw_predictions)
    if raw_predictions.ndim > 1 and raw_predictions.shape[1] > 1:
        return np.argmax(raw_predictions, axis=1)
    return (raw_predictions.reshape(-1) > 0.5).astype(int)


def evaluate_model(name, train_fn, predict_fn, X_train, y_train, X_test, y_test):
    print(f"\nRunning: {name}")
    with ResourceMonitor() as monitor:
        train_start = time.perf_counter()
        train_fn()
        train_time = time.perf_counter() - train_start

        predict_start = time.perf_counter()
        y_pred = to_binary_predictions(predict_fn(X_test))
        predict_time = time.perf_counter() - predict_start

    acc = accuracy_score(y_test, y_pred)
    anomaly_recall = recall_score(y_test, y_pred, zero_division=0)
    throughput = len(X_train) / train_time if train_time > 0 else 0.0
    report = classification_report(
        y_test,
        y_pred,
        target_names=["Normal", "Anomaly"],
        zero_division=0,
    )
    cm = confusion_matrix(y_test, y_pred)

    print(f"Accuracy: {acc:.4f}")
    print(f"Anomaly Recall: {anomaly_recall:.4f}")
    print(f"Train Time: {train_time:.3f}s")
    print(f"Throughput: {throughput:,.0f} samples/sec")
    print(f"Peak CPU: {monitor.peak_cpu:.1f}% | Peak RAM: {monitor.peak_ram:.1f} MB")

    return BenchmarkResult(
        model=name,
        accuracy=acc,
        anomaly_recall=anomaly_recall,
        train_time_sec=train_time,
        predict_time_sec=predict_time,
        throughput_samples_sec=throughput,
        peak_cpu_percent=monitor.peak_cpu,
        peak_ram_mb=monitor.peak_ram,
        report=report,
        confusion=cm,
    )


def train_online_parallel(model, X_train, y_train, batch_size, X_eval, y_eval):
    for start in range(0, len(X_train), batch_size):
        end = start + batch_size
        model.learn_batch(
            X_train[start:end],
            y_train[start:end],
            X_eval=X_eval,
            y_eval=y_eval,
        )


def run_benchmark(args):
    os.makedirs(ASSETS_DIR, exist_ok=True)

    if not os.path.exists(args.data):
        raise FileNotFoundError(f"Dataset not found: {args.data}")

    X, y = load_kdd_data(args.data, args.max_samples, args.random_state)
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

    eval_size = min(args.eval_size, len(X_test))
    X_eval = X_test[:eval_size]
    y_eval = y_test[:eval_size]

    print("=" * 70)
    print("ELM BENCHMARK")
    print("=" * 70)
    print(f"Samples: {len(X):,} | Train: {len(X_train):,} | Test: {len(X_test):,}")
    print(f"Hidden Neurons: {args.hidden_size} | Batch Size: {args.batch_size}")
    print(f"Workers: {args.workers}")

    baseline = ELMBase(
        input_size=X_train.shape[1],
        hidden_size=args.hidden_size,
        activation=args.activation,
    )
    svd_model = ELMSVD(
        input_size=X_train.shape[1],
        hidden_size=args.hidden_size,
        activation=args.activation,
    )
    parallel_model = OnlineParallelELM(
        input_size=X_train.shape[1],
        hidden_size=args.hidden_size,
        n_workers=args.workers,
        activation=args.activation,
        kb_size=args.kb_size,
        min_reliability=args.min_reliability,
    )

    results = [
        evaluate_model(
            "Sequential ELM",
            lambda: baseline.fit(X_train, y_train),
            baseline.predict,
            X_train,
            y_train,
            X_test,
            y_test,
        ),
        evaluate_model(
            "Sequential SVD-ELM",
            lambda: svd_model.fit(X_train, y_train),
            svd_model.predict,
            X_train,
            y_train,
            X_test,
            y_test,
        ),
        evaluate_model(
            "Online Parallel ELM",
            lambda: train_online_parallel(
                parallel_model,
                X_train,
                y_train,
                args.batch_size,
                X_eval,
                y_eval,
            ),
            parallel_model.predict,
            X_train,
            y_train,
            X_test,
            y_test,
        ),
    ]

    save_results(results)
    save_plots(results)
    update_readme(results, args, len(X), len(X_train), len(X_test))
    print("\nDone. Charts and README benchmark section were updated.")


def results_frame(results):
    return pd.DataFrame([
        {
            "Model": r.model,
            "Accuracy": r.accuracy,
            "Anomaly Recall": r.anomaly_recall,
            "Train Time (s)": r.train_time_sec,
            "Predict Time (s)": r.predict_time_sec,
            "Throughput (samples/s)": r.throughput_samples_sec,
            "Peak CPU (%)": r.peak_cpu_percent,
            "Peak RAM (MB)": r.peak_ram_mb,
        }
        for r in results
    ])


def save_results(results):
    df = results_frame(results)
    df.to_csv(RESULTS_CSV, index=False)
    print(f"Saved metrics: {RESULTS_CSV}")


def save_plots(results):
    sns.set_theme(style="whitegrid")
    df = results_frame(results)

    plt.figure(figsize=(10, 5))
    metric_df = df.melt(
        id_vars="Model",
        value_vars=["Accuracy", "Anomaly Recall"],
        var_name="Metric",
        value_name="Score",
    )
    ax = sns.barplot(data=metric_df, x="Model", y="Score", hue="Metric")
    ax.set_ylim(0, 1.05)
    ax.set_title("Accuracy and Anomaly Recall")
    ax.set_xlabel("")
    ax.set_ylabel("Score")
    plt.xticks(rotation=10)
    plt.tight_layout()
    plt.savefig(os.path.join(ASSETS_DIR, "accuracy.png"), dpi=300)
    plt.close()

    plt.figure(figsize=(10, 5))
    ax = sns.barplot(data=df, x="Model", y="Train Time (s)", color="#4C78A8")
    ax.set_title("Training Time Comparison")
    ax.set_xlabel("")
    ax.set_ylabel("Seconds")
    plt.xticks(rotation=10)
    plt.tight_layout()
    plt.savefig(os.path.join(ASSETS_DIR, "performance.png"), dpi=300)
    plt.close()

    plt.figure(figsize=(10, 5))
    resource_df = df.melt(
        id_vars="Model",
        value_vars=["Peak CPU (%)", "Peak RAM (MB)"],
        var_name="Resource",
        value_name="Value",
    )
    ax = sns.barplot(data=resource_df, x="Model", y="Value", hue="Resource")
    ax.set_title("Peak Resource Usage")
    ax.set_xlabel("")
    ax.set_ylabel("Observed Value")
    plt.xticks(rotation=10)
    plt.tight_layout()
    plt.savefig(os.path.join(ASSETS_DIR, "resources.png"), dpi=300)
    plt.close()

    best = max(results, key=lambda item: item.accuracy)
    plt.figure(figsize=(7, 6))
    sns.heatmap(
        best.confusion,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=["Normal", "Anomaly"],
        yticklabels=["Normal", "Anomaly"],
    )
    plt.title(f"Confusion Matrix: {best.model}")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(os.path.join(ASSETS_DIR, "confusion_matrix.png"), dpi=300)
    plt.close()

    print(f"Saved charts in: {ASSETS_DIR}/")


def markdown_table(results):
    lines = [
        "| Model | Accuracy | Anomaly Recall | Train Time | Throughput | Peak CPU | Peak RAM |",
        "| :--- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in results:
        lines.append(
            f"| {r.model} | {r.accuracy:.4f} | {r.anomaly_recall:.4f} | "
            f"{r.train_time_sec:.3f}s | {r.throughput_samples_sec:,.0f}/s | "
            f"{r.peak_cpu_percent:.1f}% | {r.peak_ram_mb:.1f} MB |"
        )
    return "\n".join(lines)


def update_readme(results, args, total_samples, train_samples, test_samples):
    if not os.path.exists(README_PATH):
        return

    start_marker = "<!-- BENCHMARK_RESULTS_START -->"
    end_marker = "<!-- BENCHMARK_RESULTS_END -->"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    best_accuracy = max(results, key=lambda item: item.accuracy)
    fastest = min(results, key=lambda item: item.train_time_sec)

    section = f"""\
{start_marker}
## Auto-Generated Benchmark Results

Last updated: `{timestamp}`

Dataset: `{args.data}`  
Samples used: `{total_samples:,}` total, `{train_samples:,}` train, `{test_samples:,}` test  
Configuration: `{args.hidden_size}` hidden neurons, `{args.batch_size}` online batch size, `{args.workers}` workers

{markdown_table(results)}

Best accuracy: **{best_accuracy.model}** ({best_accuracy.accuracy:.4f})  
Fastest training: **{fastest.model}** ({fastest.train_time_sec:.3f}s)

<p align="center">
  <img src="assets/accuracy.png" width="48%" />
  <img src="assets/performance.png" width="48%" />
  <img src="assets/resources.png" width="48%" />
  <img src="assets/confusion_matrix.png" width="48%" />
</p>
{end_marker}"""

    with open(README_PATH, "r", encoding="utf-8") as file:
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

    with open(README_PATH, "w", encoding="utf-8") as file:
        file.write(updated)

    print(f"Updated README: {README_PATH}")


def parse_args():
    default_workers = max(1, min(4, psutil.cpu_count(logical=False) or 1))
    parser = argparse.ArgumentParser(description="Compare ELM benchmark variants.")
    parser.add_argument("--data", default=DATA_PATH, help="Path to KDD .gz dataset.")
    parser.add_argument(
        "--max-samples",
        type=int,
        default=120_000,
        help="Limit rows for repeatable local benchmarks. Use 0 for full dataset.",
    )
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=10_000)
    parser.add_argument("--workers", type=int, default=default_workers)
    parser.add_argument("--test-size", type=float, default=0.3)
    parser.add_argument("--eval-size", type=int, default=10_000)
    parser.add_argument("--kb-size", type=int, default=20)
    parser.add_argument("--min-reliability", type=float, default=None)
    parser.add_argument("--activation", choices=["sigmoid", "relu", "linear"], default="sigmoid")
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    run_benchmark(parse_args())
