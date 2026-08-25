# -*- coding: utf-8 -*-
"""
random_forest_baseline.py

Classical ML baseline for FogMLS's generated dataset, demonstrating the
dataset carries genuine, learnable scheduling signal -- without requiring
deep learning (which is the subject of a separate, forthcoming paper).

Reshapes the slot-level dataset into task-level rows (one row per task,
matching the granularity used by comparable classical-ML fog-scheduling
papers), adds job position within the queue as an explicit feature
(shown to be highly informative -- FogMLS's per-position decisions
depend on live capacity depletion sequence, which position captures),
then trains a Random Forest classifier.

Usage:
    python3 random_forest_baseline.py dataset-<timestamp>.txt
"""

import sys
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

MAX_JOBS = 200
JOB_ATTRIBUTES = 6  # TaskType, TaskCodeSize, TaskDataSize, FogCap1/2/3
SLOT_LEVEL_FEATURES = 11


def parse_and_reshape(filepath):
    """Loads the slot-level dataset and reshapes to one row per task,
    adding job position within the queue as an explicit feature."""
    data = np.loadtxt(filepath, delimiter=',', skiprows=1)
    X = data[:, :MAX_JOBS * JOB_ATTRIBUTES].reshape(-1, MAX_JOBS, JOB_ATTRIBUTES)
    y_start = MAX_JOBS * JOB_ATTRIBUTES + SLOT_LEVEL_FEATURES
    Y = data[:, y_start:y_start + MAX_JOBS]

    rows_X, rows_Y = [], []
    for s in range(X.shape[0]):
        for j in range(MAX_JOBS):
            task = X[s, j]
            label = Y[s, j]
            if task.sum() == 0 and label == 0:
                continue  # padding, not a real task
            rows_X.append(np.append(task, j))  # append position index
            rows_Y.append(label)

    return np.array(rows_X), np.array(rows_Y, dtype=int)


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 random_forest_baseline.py <dataset_file>")
        sys.exit(1)

    filepath = sys.argv[1]
    print(f"[RF] Loading and reshaping {filepath}...")
    X, Y = parse_and_reshape(filepath)
    print(f"[RF] Task-level dataset: {X.shape[0]} rows, {X.shape[1]} features")

    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    Y_train, Y_test = Y[:split], Y[split:]
    print(f"[RF] Train: {len(X_train)}, Test: {len(X_test)}")

    print("[RF] Training Random Forest (100 trees)...")
    rf = RandomForestClassifier(n_estimators=100, max_depth=25,
                                 random_state=42, n_jobs=-1)
    rf.fit(X_train, Y_train)

    pred = rf.predict(X_test)
    acc = accuracy_score(Y_test, pred)

    print()
    print("=" * 50)
    print("  RANDOM FOREST BASELINE - RESULTS")
    print("=" * 50)
    print(f"  Overall accuracy: {100*acc:.2f}%")
    print()
    names = ['Nothing', 'Cloud', 'Fog-1', 'Fog-2', 'Fog-3']
    print(classification_report(Y_test, pred, target_names=names,
                                 labels=[0, 1, 2, 3, 4], zero_division=0))


if __name__ == '__main__':
    main()
