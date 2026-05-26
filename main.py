"""
Parallelized ELM - DevOps Anomaly Detection Tool
Author: Amanda Taheri
Features: System Monitoring, Parallel Processing, Online Learning

"""

import numpy as np
import pandas as pd
import time
import os
import psutil
import platform
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Import our custom modules
import settings
from src.elm_online import OnlineParallelELM

def get_system_info():
    """Extracts hardware and system information."""
    info = {
        "OS": platform.system(),
        "OS Version": platform.version(),
        "CPU": platform.processor(),
        "Physical Cores": psutil.cpu_count(logical=False),
        "Total Cores": psutil.cpu_count(logical=True),
        "Total RAM": f"{round(psutil.virtual_memory().total / (1024**3), 2)} GB"
    }
    return info

def get_resource_usage():
    """Captures current CPU and RAM usage."""
    return {
        "cpu_percent": psutil.cpu_percent(interval=None),
        "ram_percent": psutil.virtual_memory().percent
    }

def load_and_preprocess(file_path):
    print(f"Analyzing Data Source: {file_path}")

    df = pd.read_csv(file_path, compression='gzip', header=None, names=settings.KDD_COLUMNS)
    df['label'] = df['label'].apply(lambda x: 0 if x == 'normal.' else 1)
    
    le = LabelEncoder()
    for col in ['protocol_type', 'service', 'flag']:
        df[col] = le.fit_transform(df[col])
        
    return df.drop('label', axis=1).values, df['label'].values

def main():
    # 1. Header & System Info
    print("="*60)
    print("      PARALLELIZED ELM - DEVOPS ANOMALY DETECTION TOOL")
    print("="*60)
    
    sys_info = get_system_info()
    for key, value in sys_info.items():
        print(f"{key}: {value}")
    print("-" * 60)

    # 2. Dynamic File Input
    data_path = input(f"Please enter the path to your .gz dataset (default: {settings.DATA_PATH}): ")
    if not data_path:
        data_path = settings.DATA_PATH
    
    if not os.path.exists(data_path):
        print(f"❌ Error: File not found at {data_path}")
        return

    # 3. Data Preparation
    X, y = load_and_preprocess(data_path)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=settings.TEST_SIZE,
        random_state=settings.RANDOM_STATE,
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 4. Model Setup
    model = OnlineParallelELM(
        input_size=X_train_scaled.shape[1],
        hidden_size=settings.N_HIDDEN,
        n_workers=settings.available_workers(),
        activation=settings.ACTIVATION,
        kb_size=settings.KB_SIZE,
        kb_distance_threshold=settings.KB_DISTANCE_THRESHOLD,
        min_reliability=settings.MIN_RELIABILITY,
    )
    
    # 5. Training with Resource Monitoring
    print(f"\n Training started on {settings.available_workers()} workers...")
    start_time = time.time()

    for i in range(0, X_train_scaled.shape[0], settings.BATCH_SIZE):
        X_batch = X_train_scaled[i:i+settings.BATCH_SIZE]
        y_batch = y_train[i:i+settings.BATCH_SIZE]
        
        model.learn_batch(X_batch, y_batch)
        
        # Monitoring every 5 batches
        if (i // settings.BATCH_SIZE) % 5 == 0:
            usage = get_resource_usage()
            print(f"   [Progress: {i}/{X_train_scaled.shape[0]}] -> CPU: {usage['cpu_percent']}% | RAM: {usage['ram_percent']}%")
            
    total_time = time.time() - start_time
    
    # 6. Final Dashboard
    print("\n" + "="*60)
    print("                FINAL PERFORMANCE REPORT")
    print("="*60)
    print(f" Total Execution Time: {total_time:.2f} seconds")
    
    raw_predictions = model.predict(X_test_scaled)
    # Convert continuous outputs to binary classes (0 or 1)
    y_pred = (raw_predictions > 0.5).astype(int)
    
    print("\n Classification Accuracy Metrics:")
    print(classification_report(y_test, y_pred))
    
    print("\n Resource Efficiency:")
    final_usage = get_resource_usage()
    print(f"   Peak CPU Load during evaluation: {final_usage['cpu_percent']}%")
    print(f"   Final Memory Footprint: {final_usage['ram_percent']}%")
    print("="*60)

if __name__ == "__main__":
    main()
