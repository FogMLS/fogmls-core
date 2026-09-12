# -*- coding: utf-8 -*-
"""
scalability_test_nodes.py

Node-count scaling experiment, following LEAF's exact precedent (Fog 1
through Fog 6: 1, 2, 3, 4, 5, 6 fog nodes). Sensors per cluster held
fixed at 9 (the validated baseline); only fog node count varies.

Beyond 3 nodes, FogDevice automatically cycles through the 3 existing
hardware profiles (confirmed safe -- devices.py prints a warning, does
not error), matching the same "replicate existing types" approach
already used for the Intelligent Surveillance case study.

CHECKPOINTED: safe to interrupt and re-run -- automatically skips any
run already completed and continues from where it left off.

NO TIMEOUT on individual runs (removed after confirming GA@5-nodes
genuinely needs more than 2 hours to converge -- a real, important
finding about GA's non-linear scaling cost, not a bug). Runs are
allowed to take as long as they genuinely need.

Usage:
    python3 scalability_test_nodes.py
    python3 scalability_test_nodes.py --restart

Output:
    scalability_nodes_results.txt
    scalability_nodes_checkpoint.json
    scalability_nodes_log_<scheduler>_<nodes>nodes.txt
"""

import subprocess
import re
import sys
import shutil
import time
import json
import os

CONFIG_FILE = 'config.py'
CONFIG_BACKUP = 'config.py.scalability_nodes_backup'
CHECKPOINT_FILE = 'scalability_nodes_checkpoint.json'
RESULTS_FILE = 'scalability_nodes_results.txt'

SENSORS_PER_CLUSTER_FIXED = 9

# node_count -> (MAX_RS, MAX_JOBS). MAX_RS = node_count + 2.
# MAX_JOBS = SENSORS_PER_CLUSTER_FIXED * node_count * 3 stages.
NODE_LEVELS = [
    (1, 3, 27),
    (2, 4, 54),
    (3, 5, 81),   # matches the already-fully-validated baseline
    (4, 6, 108),
    (5, 7, 135),
    (6, 8, 162),
]

# RM_TYPE values: DEFAULT_RM = 0, GA_RM = 1, RR_RM = 4
SCHEDULERS = [
    ('DEFAULT', 0),
    ('GA', 1),
    ('RR', 4),
]


def load_checkpoint():
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, 'r') as f:
            return json.load(f)
    return {'completed_runs': [], 'results': []}


def save_checkpoint(checkpoint):
    with open(CHECKPOINT_FILE, 'w') as f:
        json.dump(checkpoint, f, indent=2)


def run_key(nodes, sched_name):
    return f"{sched_name}_{nodes}"


def get_original_config():
    with open(CONFIG_FILE, 'r') as f:
        content = f.read()
    sensors = int(re.search(r'^SENSORS_PER_CLUSTER = (\d+)', content, re.MULTILINE).group(1))
    max_jobs = int(re.search(r'^MAX_JOBS = (\d+)', content, re.MULTILINE).group(1))
    rm_type = int(re.search(r'^RM_TYPE = (\d+)', content, re.MULTILINE).group(1))
    max_rs = int(re.search(r'^MAX_RS = (\d+)', content, re.MULTILINE).group(1))
    return sensors, max_jobs, rm_type, max_rs


def set_config(sensors_per_cluster, max_jobs, rm_type, max_rs):
    with open(CONFIG_FILE, 'r') as f:
        content = f.read()

    content = re.sub(r'^SENSORS_PER_CLUSTER = \d+',
                      f'SENSORS_PER_CLUSTER = {sensors_per_cluster}',
                      content, flags=re.MULTILINE)
    content = re.sub(r'^MAX_JOBS = \d+',
                      f'MAX_JOBS = {max_jobs}',
                      content, flags=re.MULTILINE)
    content = re.sub(r'^RM_TYPE = \d+',
                      f'RM_TYPE = {rm_type}',
                      content, flags=re.MULTILINE)
    content = re.sub(r'^MAX_RS = \d+',
                      f'MAX_RS = {max_rs}',
                      content, flags=re.MULTILINE)

    with open(CONFIG_FILE, 'w') as f:
        f.write(content)


def verify_config(expected_sensors, expected_max_jobs, expected_rm_type, expected_max_rs):
    actual = get_original_config()
    expected = (expected_sensors, expected_max_jobs, expected_rm_type, expected_max_rs)
    ok = actual == expected
    if not ok:
        print(f"  [WARNING] config.py mismatch! Expected {expected}, got {actual}.")
    return ok


def parse_output(log_text):
    result = {}
    m = re.search(r'avg TimeTaken for Tasks:([\d.]+)', log_text)
    result['avg_latency'] = float(m.group(1)) if m else None
    m = re.search(r'Cloud executed :(\d+) Tasks', log_text)
    result['cloud_tasks'] = int(m.group(1)) if m else None
    m = re.search(r'Total Number of Tasks:(\d+)', log_text)
    result['total_tasks'] = int(m.group(1)) if m else None
    m = re.search(r'Avg time per slot: ([\d.]+) seconds', log_text)
    result['ga_avg_slot_time'] = float(m.group(1)) if m else None
    m = re.search(r'\[WARNING\] No unique specs', log_text)
    result['used_cycled_hardware'] = bool(m)
    return result


def append_result_line(result):
    line = (f"{result['nodes']:>6} {result['scheduler']:>10} "
            f"{str(result.get('avg_latency')):>12} "
            f"{str(result.get('cloud_tasks')):>12} "
            f"{str(result.get('ga_avg_slot_time')):>12} "
            f"{result['wall_time_seconds']:>12.1f} "
            f"{str(result.get('used_cycled_hardware')):>8}\n")
    write_header = not os.path.exists(RESULTS_FILE)
    with open(RESULTS_FILE, 'a') as f:
        if write_header:
            f.write(f"{'Nodes':>6} {'Scheduler':>10} {'AvgLatency':>12} "
                    f"{'CloudTasks':>12} {'GA_SlotTime':>12} "
                    f"{'WallTime(s)':>12} {'Cycled':>8}\n")
        f.write(line)


def run_one(nodes, max_rs, max_jobs, sched_name, rm_type, orig):
    orig_sensors, orig_max_jobs, orig_rm_type, orig_max_rs = orig

    print(f"\n{'='*60}")
    print(f"  Running {sched_name} at {nodes} fog node(s) "
          f"(MAX_RS={max_rs}, MAX_JOBS={max_jobs})")
    print(f"  NOTE: no timeout -- this run will take as long as it needs.")
    print(f"{'='*60}")
    sys.stdout.flush()

    set_config(SENSORS_PER_CLUSTER_FIXED, max_jobs, rm_type, max_rs)
    verify_config(SENSORS_PER_CLUSTER_FIXED, max_jobs, rm_type, max_rs)

    start = time.time()
    proc = subprocess.run(
        [sys.executable, 'IoTsimulation.py'],
        capture_output=True, text=True
        # NO timeout parameter -- runs to completion regardless of duration
    )
    wall_time = time.time() - start

    log_text = proc.stdout + proc.stderr
    log_filename = f'scalability_nodes_log_{sched_name}_{nodes}nodes.txt'
    with open(log_filename, 'w') as f:
        f.write(log_text)

    set_config(orig_sensors, orig_max_jobs, orig_rm_type, orig_max_rs)
    verify_config(orig_sensors, orig_max_jobs, orig_rm_type, orig_max_rs)

    parsed = parse_output(log_text)
    parsed['wall_time_seconds'] = wall_time
    parsed['nodes'] = nodes
    parsed['max_rs'] = max_rs
    parsed['max_jobs'] = max_jobs
    parsed['scheduler'] = sched_name
    parsed['log_file'] = log_filename

    print(f"  Done in {wall_time:.1f}s wall-clock ({wall_time/60:.1f} min). "
          f"avg_latency={parsed.get('avg_latency')}, "
          f"cloud_tasks={parsed.get('cloud_tasks')}, "
          f"ga_slot_time={parsed.get('ga_avg_slot_time')}, "
          f"used_cycled_hardware={parsed.get('used_cycled_hardware')}")
    print(f"  config.py restored to original state.")
    sys.stdout.flush()

    return parsed


def main():
    restart = '--restart' in sys.argv
    if restart:
        for f in [CHECKPOINT_FILE, RESULTS_FILE]:
            if os.path.exists(f):
                os.remove(f)
        print("Restart requested -- cleared existing checkpoint and results.")

    checkpoint = load_checkpoint()
    completed = set(checkpoint['completed_runs'])
    if completed:
        print(f"Resuming: {len(completed)} run(s) already completed: "
              f"{sorted(completed)}")

    orig = get_original_config()
    print(f"Original config.py state: SENSORS_PER_CLUSTER={orig[0]}, "
          f"MAX_JOBS={orig[1]}, RM_TYPE={orig[2]}, MAX_RS={orig[3]}")
    shutil.copy(CONFIG_FILE, CONFIG_BACKUP)

    total_runs = len(NODE_LEVELS) * len(SCHEDULERS)
    run_number = 0

    try:
        for nodes, max_rs, max_jobs in NODE_LEVELS:
            for sched_name, rm_type in SCHEDULERS:
                run_number += 1
                key = run_key(nodes, sched_name)

                if key in completed:
                    print(f"\n[{run_number}/{total_runs}] SKIPPING "
                          f"{sched_name} at {nodes} node(s) -- already done.")
                    continue

                print(f"\n[{run_number}/{total_runs}] Starting "
                      f"{sched_name} at {nodes} node(s)...")
                sys.stdout.flush()

                result = run_one(nodes, max_rs, max_jobs, sched_name,
                                  rm_type, orig)

                checkpoint['completed_runs'].append(key)
                checkpoint['results'].append(result)
                save_checkpoint(checkpoint)
                append_result_line(result)
                print(f"  Checkpoint saved. Progress: "
                      f"{len(checkpoint['completed_runs'])}/{total_runs}")
                sys.stdout.flush()

    finally:
        set_config(*orig)
        verify_config(*orig)
        print(f"\nFinal safety check: config.py restored to original state.")

    if len(checkpoint['completed_runs']) == total_runs:
        print("\n" + "=" * 70)
        print("  ALL RUNS COMPLETE -- NODE SCALABILITY TEST FULL SUMMARY")
        print("=" * 70)
        with open(RESULTS_FILE, 'r') as f:
            print(f.read())
    else:
        print(f"\n{len(checkpoint['completed_runs'])}/{total_runs} runs "
              f"complete. Re-run this script to continue.")


if __name__ == '__main__':
    main()