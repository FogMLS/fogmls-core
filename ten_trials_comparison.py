# -*- coding: utf-8 -*-
"""
ten_trials_comparison.py

Rigorous 10-independent-trial statistical comparison of all 5
schedulers (DEFAULT, GA, RR, RF, RF_SMART) at the validated baseline
configuration (3 fog nodes, 9 sensors/cluster).

Each trial uses a DIFFERENT random seed (42-51), so results are
genuinely independent, not 10 identical copies of the same
deterministic run -- required for a meaningful paired statistical test.

CHECKPOINTED: safe to interrupt and re-run, automatically resumes.
NO TIMEOUT: runs are allowed to take as long as they genuinely need.

PREREQUISITE: RF_RM and RF_SMART_RM require 'random_forest_model.joblib'
to already exist (trained once, at seed 42, on the standard baseline
dataset) -- this script does NOT train it; run random_forest_baseline.py
first if that file is missing.

Usage:
    python3 ten_trials_comparison.py
    python3 ten_trials_comparison.py --restart

Output:
    ten_trials_results.txt       -- raw per-trial results
    ten_trials_checkpoint.json   -- resume tracking
    ten_trials_log_<sched>_<trial>.txt -- individual run logs
    ten_trials_statistics.txt    -- final summary with means, std devs,
                                     and paired t-tests vs GA (written
                                     only once all 50 trials complete)
"""

import subprocess
import re
import sys
import shutil
import time
import json
import os

CONFIG_FILE = 'config.py'
CONFIG_BACKUP = 'config.py.ten_trials_backup'
CHECKPOINT_FILE = 'ten_trials_checkpoint.json'
RESULTS_FILE = 'ten_trials_results.txt'
STATS_FILE = 'ten_trials_statistics.txt'

BASELINE_SENSORS = 9
BASELINE_MAX_JOBS = 200
BASELINE_MAX_RS = 5

SEEDS = list(range(42, 52))

SCHEDULERS = [
    ('DEFAULT', 0),
    ('GA', 1),
    ('RR', 4),
    ('RF', 7),
    ('RF_SMART', 9),
]


def load_checkpoint():
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, 'r') as f:
            return json.load(f)
    return {'completed_runs': [], 'results': []}


def save_checkpoint(checkpoint):
    with open(CHECKPOINT_FILE, 'w') as f:
        json.dump(checkpoint, f, indent=2)


def run_key(sched_name, seed):
    return f"{sched_name}_seed{seed}"


def get_original_config():
    with open(CONFIG_FILE, 'r') as f:
        content = f.read()
    sensors = int(re.search(r'^SENSORS_PER_CLUSTER = (\d+)', content, re.MULTILINE).group(1))
    max_jobs = int(re.search(r'^MAX_JOBS = (\d+)', content, re.MULTILINE).group(1))
    rm_type = int(re.search(r'^RM_TYPE = (\d+)', content, re.MULTILINE).group(1))
    max_rs = int(re.search(r'^MAX_RS = (\d+)', content, re.MULTILINE).group(1))
    seed = int(re.search(r'^RANDOM_SEED = (\d+)', content, re.MULTILINE).group(1))
    return sensors, max_jobs, rm_type, max_rs, seed


def set_config(sensors, max_jobs, rm_type, max_rs, seed):
    with open(CONFIG_FILE, 'r') as f:
        content = f.read()
    content = re.sub(r'^SENSORS_PER_CLUSTER = \d+', f'SENSORS_PER_CLUSTER = {sensors}', content, flags=re.MULTILINE)
    content = re.sub(r'^MAX_JOBS = \d+', f'MAX_JOBS = {max_jobs}', content, flags=re.MULTILINE)
    content = re.sub(r'^RM_TYPE = \d+', f'RM_TYPE = {rm_type}', content, flags=re.MULTILINE)
    content = re.sub(r'^MAX_RS = \d+', f'MAX_RS = {max_rs}', content, flags=re.MULTILINE)
    content = re.sub(r'^RANDOM_SEED = \d+', f'RANDOM_SEED = {seed}', content, flags=re.MULTILINE)
    with open(CONFIG_FILE, 'w') as f:
        f.write(content)


def parse_output(log_text):
    result = {}
    m = re.search(r'avg TimeTaken for Tasks:([\d.]+)', log_text)
    result['avg_latency'] = float(m.group(1)) if m else None
    m = re.search(r'Cloud executed :(\d+) Tasks', log_text)
    result['cloud_tasks'] = int(m.group(1)) if m else None
    return result


def append_result_line(result):
    line = (f"{result['scheduler']:>10} {result['seed']:>6} "
            f"{str(result.get('avg_latency')):>14} "
            f"{str(result.get('cloud_tasks')):>12} "
            f"{result['wall_time_seconds']:>12.1f}\n")
    write_header = not os.path.exists(RESULTS_FILE)
    with open(RESULTS_FILE, 'a') as f:
        if write_header:
            f.write(f"{'Scheduler':>10} {'Seed':>6} {'AvgLatency':>14} "
                    f"{'CloudTasks':>12} {'WallTime(s)':>12}\n")
        f.write(line)


def run_one(sched_name, rm_type, seed, orig):
    print(f"\n{'='*60}")
    print(f"  Running {sched_name}, seed={seed}")
    print(f"{'='*60}")
    sys.stdout.flush()

    set_config(BASELINE_SENSORS, BASELINE_MAX_JOBS, rm_type, BASELINE_MAX_RS, seed)

    start = time.time()
    proc = subprocess.run(
        [sys.executable, 'IoTsimulation.py'],
        capture_output=True, text=True
    )
    wall_time = time.time() - start

    log_text = proc.stdout + proc.stderr
    log_filename = f'ten_trials_log_{sched_name}_seed{seed}.txt'
    with open(log_filename, 'w') as f:
        f.write(log_text)

    set_config(*orig)

    parsed = parse_output(log_text)
    parsed['wall_time_seconds'] = wall_time
    parsed['scheduler'] = sched_name
    parsed['seed'] = seed

    print(f"  Done in {wall_time:.1f}s. avg_latency={parsed.get('avg_latency')}, "
          f"cloud_tasks={parsed.get('cloud_tasks')}")
    sys.stdout.flush()

    return parsed


def compute_statistics(checkpoint):
    """Compute mean, std dev, and paired t-test (GA vs each other
    scheduler) using only the standard library (no scipy dependency,
    for maximum portability across environments)."""
    import math

    by_scheduler = {}
    for r in checkpoint['results']:
        by_scheduler.setdefault(r['scheduler'], []).append(r['avg_latency'])

    def mean(xs):
        return sum(xs) / len(xs)

    def stdev(xs, m):
        if len(xs) < 2:
            return 0.0
        return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))

    def paired_ttest(xs, ys):
        diffs = [a - b for a, b in zip(xs, ys)]
        n = len(diffs)
        d_mean = mean(diffs)
        d_std = stdev(diffs, d_mean)
        if d_std == 0:
            return float('inf') if d_mean != 0 else 0.0, n - 1
        t_stat = d_mean / (d_std / math.sqrt(n))
        return t_stat, n - 1

    lines = []
    lines.append("=" * 70)
    lines.append("  10-TRIAL STATISTICAL COMPARISON -- FINAL RESULTS")
    lines.append("=" * 70)
    lines.append("")

    for sched in ['DEFAULT', 'GA', 'RR', 'RF', 'RF_SMART']:
        vals = by_scheduler.get(sched, [])
        if not vals:
            continue
        m = mean(vals)
        s = stdev(vals, m)
        lines.append(f"{sched}: mean={m:.4f}, std={s:.4f}, n={len(vals)}")
        lines.append(f"  values: {vals}")

    lines.append("")
    lines.append("Paired t-tests vs GA (positive t = GA has LOWER latency,")
    lines.append("i.e. GA is better; larger |t| = stronger evidence):")

    ga_vals = by_scheduler.get('GA', [])
    for sched in ['DEFAULT', 'RR', 'RF', 'RF_SMART']:
        vals = by_scheduler.get(sched, [])
        if not vals or not ga_vals or len(vals) != len(ga_vals):
            continue
        t_stat, df = paired_ttest(vals, ga_vals)
        lines.append(f"  GA vs {sched}: t({df}) = {t_stat:.4f}")

    result_text = '\n'.join(lines)
    with open(STATS_FILE, 'w') as f:
        f.write(result_text)
    print(result_text)


def main():
    restart = '--restart' in sys.argv
    if restart:
        for f in [CHECKPOINT_FILE, RESULTS_FILE, STATS_FILE]:
            if os.path.exists(f):
                os.remove(f)
        print("Restart requested -- cleared existing checkpoint and results.")

    if not os.path.exists('random_forest_model.joblib'):
        print("[ERROR] random_forest_model.joblib not found. Run "
              "random_forest_baseline.py first to train and save the "
              "model before running RF trials.")
        sys.exit(1)

    checkpoint = load_checkpoint()
    completed = set(checkpoint['completed_runs'])
    if completed:
        print(f"Resuming: {len(completed)} run(s) already completed.")

    orig = get_original_config()
    print(f"Original config.py state: {orig}")
    shutil.copy(CONFIG_FILE, CONFIG_BACKUP)

    total_runs = len(SCHEDULERS) * len(SEEDS)
    run_number = 0

    try:
        for sched_name, rm_type in SCHEDULERS:
            for seed in SEEDS:
                run_number += 1
                key = run_key(sched_name, seed)

                if key in completed:
                    print(f"[{run_number}/{total_runs}] SKIPPING "
                          f"{sched_name} seed={seed} -- already done.")
                    continue

                print(f"\n[{run_number}/{total_runs}] Starting "
                      f"{sched_name} seed={seed}...")
                sys.stdout.flush()

                result = run_one(sched_name, rm_type, seed, orig)

                checkpoint['completed_runs'].append(key)
                checkpoint['results'].append(result)
                save_checkpoint(checkpoint)
                append_result_line(result)
                print(f"  Checkpoint saved. Progress: "
                      f"{len(checkpoint['completed_runs'])}/{total_runs}")
                sys.stdout.flush()

    finally:
        set_config(*orig)
        print(f"\nFinal safety check: config.py restored to original state.")

    if len(checkpoint['completed_runs']) == total_runs:
        compute_statistics(checkpoint)
    else:
        print(f"\n{len(checkpoint['completed_runs'])}/{total_runs} runs "
              f"complete. Re-run this script to continue.")


if __name__ == '__main__':
    main()