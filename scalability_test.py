# -*- coding: utf-8 -*-
"""
scalability_test.py

Runs the sensor-count scaling experiment automatically: GA and DEFAULT_RM
at 4 sensor levels (9, 27, 45, 63 sensors/cluster), measuring real GA
runtime and scheduling behavior at each level.

CHECKPOINTED: results are saved to disk immediately after each individual
run completes, not just at the end. If interrupted (crash, terminal
closed, etc.), simply re-run this script -- it will automatically skip
any runs already completed and continue from where it left off.

Each run is a fresh subprocess (avoids Python import-caching issues when
config.py changes mid-session). config.py is safely restored to its
original state after every single run, not just at the very end -- so
even a mid-run interruption leaves config.py in a known-good state.

Usage:
    python3 scalability_test.py

To start over from scratch (ignore any existing checkpoint):
    python3 scalability_test.py --restart

Output:
    scalability_checkpoint.json -- machine-readable progress, safe to
        inspect, do not edit by hand while the script is running.
    scalability_results.txt -- human-readable summary, updated after
        every run.
    scalability_log_<scheduler>_<sensors>sensors.txt -- raw output of
        each individual run, for inspection if something looks wrong.
"""

import subprocess
import re
import sys
import shutil
import time
import json
import os

CONFIG_FILE = 'config.py'
CONFIG_BACKUP = 'config.py.scalability_backup'
CHECKPOINT_FILE = 'scalability_checkpoint.json'
RESULTS_FILE = 'scalability_results.txt'

# (sensors_per_cluster, max_jobs) -- MAX_JOBS = sensors * 3 nodes * 3 stages
SCALE_LEVELS = [
    (9, 81),
    (27, 243),
    (45, 405),
    (63, 567),
]

# RM_TYPE values: DEFAULT_RM = 0, GA_RM = 1
SCHEDULERS = [
    ('DEFAULT', 0),
    ('GA', 1),
]


def load_checkpoint():
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, 'r') as f:
            return json.load(f)
    return {'completed_runs': [], 'results': []}


def save_checkpoint(checkpoint):
    with open(CHECKPOINT_FILE, 'w') as f:
        json.dump(checkpoint, f, indent=2)


def run_key(sensors, sched_name):
    return f"{sched_name}_{sensors}"


def get_original_config():
    """Reads config.py's CURRENT values before any modification, so we
    know the true original state to restore to (not a value from a
    previous, possibly-interrupted run)."""
    with open(CONFIG_FILE, 'r') as f:
        content = f.read()
    sensors = int(re.search(r'^SENSORS_PER_CLUSTER = (\d+)', content, re.MULTILINE).group(1))
    max_jobs = int(re.search(r'^MAX_JOBS = (\d+)', content, re.MULTILINE).group(1))
    rm_type = int(re.search(r'^RM_TYPE = (\d+)', content, re.MULTILINE).group(1))
    return sensors, max_jobs, rm_type


def set_config(sensors_per_cluster, max_jobs, rm_type):
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

    with open(CONFIG_FILE, 'w') as f:
        f.write(content)


def verify_config(expected_sensors, expected_max_jobs, expected_rm_type):
    actual_sensors, actual_max_jobs, actual_rm_type = get_original_config()
    ok = (actual_sensors == expected_sensors and
          actual_max_jobs == expected_max_jobs and
          actual_rm_type == expected_rm_type)
    if not ok:
        print(f"  [WARNING] config.py mismatch after write! "
              f"Expected sensors={expected_sensors}, max_jobs={expected_max_jobs}, "
              f"rm_type={expected_rm_type}. Got sensors={actual_sensors}, "
              f"max_jobs={actual_max_jobs}, rm_type={actual_rm_type}.")
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

    return result


def append_result_line(result):
    line = (f"{result['sensors_per_cluster']:>8} {result['scheduler']:>10} "
            f"{str(result.get('avg_latency')):>12} "
            f"{str(result.get('cloud_tasks')):>12} "
            f"{str(result.get('ga_avg_slot_time')):>12} "
            f"{result['wall_time_seconds']:>12.1f}\n")
    write_header = not os.path.exists(RESULTS_FILE)
    with open(RESULTS_FILE, 'a') as f:
        if write_header:
            f.write(f"{'Sensors':>8} {'Scheduler':>10} {'AvgLatency':>12} "
                    f"{'CloudTasks':>12} {'GA_SlotTime':>12} {'WallTime(s)':>12}\n")
        f.write(line)


def run_one(sensors, max_jobs, sched_name, rm_type, orig_sensors, orig_max_jobs, orig_rm_type):
    print(f"\n{'='*60}")
    print(f"  Running {sched_name} at {sensors} sensors/cluster "
          f"(MAX_JOBS={max_jobs})")
    print(f"{'='*60}")

    set_config(sensors, max_jobs, rm_type)
    verify_config(sensors, max_jobs, rm_type)

    start = time.time()
    proc = subprocess.run(
        [sys.executable, 'IoTsimulation.py'],
        capture_output=True, text=True, timeout=7200
    )
    wall_time = time.time() - start

    log_text = proc.stdout + proc.stderr
    log_filename = f'scalability_log_{sched_name}_{sensors}sensors.txt'
    with open(log_filename, 'w') as f:
        f.write(log_text)

    # Restore original config IMMEDIATELY after every single run, not
    # just at the very end -- so a mid-experiment interruption still
    # leaves config.py in a known-good state.
    set_config(orig_sensors, orig_max_jobs, orig_rm_type)
    verify_config(orig_sensors, orig_max_jobs, orig_rm_type)

    parsed = parse_output(log_text)
    parsed['wall_time_seconds'] = wall_time
    parsed['sensors_per_cluster'] = sensors
    parsed['max_jobs'] = max_jobs
    parsed['scheduler'] = sched_name
    parsed['log_file'] = log_filename

    print(f"  Done in {wall_time:.1f}s wall-clock. "
          f"avg_latency={parsed.get('avg_latency')}, "
          f"cloud_tasks={parsed.get('cloud_tasks')}, "
          f"ga_slot_time={parsed.get('ga_avg_slot_time')}")
    print(f"  config.py restored to original state.")

    return parsed


def main():
    restart = '--restart' in sys.argv
    if restart and os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)
        if os.path.exists(RESULTS_FILE):
            os.remove(RESULTS_FILE)
        print("Restart requested -- cleared existing checkpoint and results.")

    checkpoint = load_checkpoint()
    completed = set(checkpoint['completed_runs'])

    if completed:
        print(f"Resuming from checkpoint: {len(completed)} run(s) already "
              f"completed: {sorted(completed)}")

    orig_sensors, orig_max_jobs, orig_rm_type = get_original_config()
    print(f"Original config.py state: SENSORS_PER_CLUSTER={orig_sensors}, "
          f"MAX_JOBS={orig_max_jobs}, RM_TYPE={orig_rm_type}")
    shutil.copy(CONFIG_FILE, CONFIG_BACKUP)
    print(f"Backed up {CONFIG_FILE} to {CONFIG_BACKUP} (extra safety copy)")

    total_runs = len(SCALE_LEVELS) * len(SCHEDULERS)
    run_number = 0

    try:
        for sensors, max_jobs in SCALE_LEVELS:
            for sched_name, rm_type in SCHEDULERS:
                run_number += 1
                key = run_key(sensors, sched_name)

                if key in completed:
                    print(f"\n[{run_number}/{total_runs}] SKIPPING "
                          f"{sched_name} at {sensors} sensors -- already "
                          f"completed (checkpoint).")
                    continue

                print(f"\n[{run_number}/{total_runs}] Starting "
                      f"{sched_name} at {sensors} sensors...")

                result = run_one(sensors, max_jobs, sched_name, rm_type,
                                  orig_sensors, orig_max_jobs, orig_rm_type)

                checkpoint['completed_runs'].append(key)
                checkpoint['results'].append(result)
                save_checkpoint(checkpoint)
                append_result_line(result)
                print(f"  Checkpoint saved. Progress: "
                      f"{len(checkpoint['completed_runs'])}/{total_runs}")

    finally:
        # Final safety restore, in case anything above went wrong.
        set_config(orig_sensors, orig_max_jobs, orig_rm_type)
        verify_config(orig_sensors, orig_max_jobs, orig_rm_type)
        print(f"\nFinal safety check: config.py confirmed restored to "
              f"SENSORS_PER_CLUSTER={orig_sensors}, MAX_JOBS={orig_max_jobs}, "
              f"RM_TYPE={orig_rm_type}")

    if len(checkpoint['completed_runs']) == total_runs:
        print("\n" + "=" * 70)
        print("  ALL RUNS COMPLETE -- SCALABILITY TEST FULL SUMMARY")
        print("=" * 70)
        with open(RESULTS_FILE, 'r') as f:
            print(f.read())
        print(f"Full results in {RESULTS_FILE}")
        print(f"Checkpoint file {CHECKPOINT_FILE} can be safely deleted now.")
    else:
        print(f"\n{len(checkpoint['completed_runs'])}/{total_runs} runs "
              f"complete. Re-run this script to continue.")


if __name__ == '__main__':
    main()
