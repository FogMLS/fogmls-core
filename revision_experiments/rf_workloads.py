# -*- coding: utf-8 -*-
"""
rf_workloads.py

R1-1, RF_SMART part. For each workload folder made by r1_1_workloads.py:
  1. train a Random Forest on that workload's own GA dataset
     (GA_dataset_seed42.txt, kept by r1_1_workloads.py)
  2. run RF_SMART_RM with that model on unseen seeds (43, 44, 45)

The model learns from seed 42, so seed 42 is NOT used for testing.

Usage (from ~/fogmls_revision, same folder as r1_1_workloads.py):
    python3 rf_workloads.py A4
    python3 rf_workloads.py C1 --seeds 43 44
Results are added to ~/r1_1_workloads/results/runs.csv
(sched = RF_SMART); see them with: python3 r1_1_workloads.py --summary
Safe to stop and re-run: finished runs are skipped.
"""

import csv
import os
import re
import shutil
import subprocess
import sys

from r1_1_workloads import (WORKLOADS, ROOT, RESULTS, CSV_FILE,
                            patch_config, run_one, append_csv)

RF_SCRIPT = 'random_forest_baseline.py'
DATASET = 'GA_dataset_seed42.txt'
DEFAULT_SEEDS = [43, 44, 45]


def train(name, d):
    model = os.path.join(d, 'random_forest_model.joblib')
    log_path = os.path.join(RESULTS, f'rf_train_{name}.txt')
    if os.path.exists(model) and os.path.exists(log_path):
        print(f'  model for {name} already trained, skipping training',
              flush=True)
        return
    if not os.path.exists(os.path.join(d, DATASET)):
        sys.exit(f'{name}: {DATASET} not found in {d}. Run the GA seed 42 '
                 f'for this workload first.')
    shutil.copy(os.path.join(os.getcwd(), RF_SCRIPT), os.path.join(d, RF_SCRIPT))
    print(f'  training Random Forest for {name} ...', flush=True)
    p = subprocess.run([sys.executable, RF_SCRIPT, DATASET], cwd=d,
                       capture_output=True, text=True)
    with open(log_path, 'w') as f:
        f.write(p.stdout + p.stderr)
    m = re.search(r'Overall accuracy: ([\d.]+)%', p.stdout)
    if not m or not os.path.exists(model):
        sys.exit(f'{name}: training failed, see {log_path}')
    print(f'  {name}: RF test accuracy {m.group(1)}% '
          f'(chronological 80/20 split)', flush=True)


def main():
    args = sys.argv[1:]
    seeds = DEFAULT_SEEDS
    if '--seeds' in args:
        i = args.index('--seeds')
        seeds = [int(a) for a in args[i + 1:]]
        args = args[:i]
    names = [a for a in args if not a.startswith('--')]
    if not names or any(n not in WORKLOADS for n in names):
        sys.exit(f'give workload name(s) from {list(WORKLOADS)}')

    done = set()
    if os.path.exists(CSV_FILE):
        for r in csv.DictReader(open(CSV_FILE)):
            done.add(f"{r['workload']}_{r['sched']}_{r['seed']}")

    for name in names:
        w = WORKLOADS[name]
        d = os.path.join(ROOT, name)
        print(f"\n=== {name}: {w['desc']} (RF_SMART) ===", flush=True)
        if not os.path.isdir(d):
            sys.exit(f'{d} not found; run r1_1_workloads.py for {name} first')
        train(name, d)
        for seed in seeds:
            key = f'{name}_RF_SMART_{seed}'
            if key in done:
                print(f'  skip {key}', flush=True)
                continue
            r = run_one(name, w, d, 'RF_SMART', 9, seed)
            append_csv(r)
            print(f"  {key}: latency {r['latency']:.4f}, cloud "
                  f"{r['cloud']:.0f}/{r['total']:.0f}, {r['wall']} s",
                  flush=True)


if __name__ == '__main__':
    main()