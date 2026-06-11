"""
Central project settings.

Change values here to control the training app and benchmark defaults.
Command-line arguments in test.py can still override these values.
"""

import os
import psutil


DATA_PATH = "data/kddcup.data_10_percent.gz"
ASSETS_DIR = "assets"
README_PATH = "README.md"
RESULTS_CSV = os.path.join(ASSETS_DIR, "benchmark_results.csv")

N_HIDDEN = 4096
BATCH_SIZE = 40_000
N_WORKERS = 4
ACTIVATION = "sigmoid"

TEST_SIZE = 0.30
RANDOM_STATE = 42

KB_SIZE = 20
KB_DISTANCE_THRESHOLD = 2.0
MIN_RELIABILITY = None

BENCHMARK_MAX_SAMPLES = 120_000
BENCHMARK_EVAL_SIZE = 10_000
RESOURCE_SAMPLE_INTERVAL = 0.1

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


def available_workers():
    physical_cores = psutil.cpu_count(logical=False) or psutil.cpu_count(logical=True) or 1
    return max(1, min(N_WORKERS, physical_cores))
