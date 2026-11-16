# -*- coding: utf-8 -*-
"""
r1_1_workloads.py

Reviewer 1, comment 1: test DEFAULT, GA and RR under different arrival
rates, task mixes and node profiles (5 seeds each).

Each workload runs in its OWN folder (copy of the .py files), so your
~/fogmls_revision/config.py is never touched.
Checkpointed: stop and re-run any time, finished runs are skipped.

Usage (from ~/fogmls_revision):
    python3 r1_1_workloads.py                 # all workloads
    python3 r1_1_workloads.py --only A1 C1    # only these
    python3 r1_1_workloads.py --summary       # just print the table

Output (in ~/r1_1_workloads/results):
    runs.csv       one line per run
    log_<workload>_<sched>_<seed>.txt
    summary.txt    mean, sd and gaps per workload
"""

import csv
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time

SRC = os.getcwd()                                   # ~/fogmls_revision
ROOT = os.path.expanduser('~/r1_1_workloads')
RESULTS = os.path.join(ROOT, 'results')
CKPT = os.path.join(RESULTS, 'checkpoint.json')
CSV_FILE = os.path.join(RESULTS, 'runs.csv')

FILES = ['IoTsimulation.py', 'devices.py', 'config.py',
         'GeneticAlg.py', 'ParticleSwarmAlg.py']

SEEDS = [42, 43, 44, 45, 46]
SCHEDULERS = [('DEFAULT', 0), ('GA', 1), ('RR', 4)]
KEEP_GA_DATASET_SEED = 42   # keep this GA dataset (needed later for RF)

SENSOR_LINE = "s_type = 1 if j % 2 == 0 else 2"

# name: description, config patches, expected slot time, optional sensor line
WORKLOADS = {
    'BASE': dict(desc='Baseline (sanity check, should match the 10-trial runs)',
                 cfg={}, slot=35.0),
    'A1': dict(desc='Arrival rate up: slot 28 SU',
               cfg={'SLOT_TIME0': -9.5, 'MAX_SIMULATION_TIME': 140000},
               slot=28.0),
    'A3': dict(desc='Arrival rate up (milder): slot 31 SU',
               cfg={'SLOT_TIME0': -6.5, 'MAX_SIMULATION_TIME': 155000},
               slot=31.0),
    'A4': dict(desc='Arrival rate up (slightly): slot 33 SU',
               cfg={'SLOT_TIME0': -4.5, 'MAX_SIMULATION_TIME': 165000},
               slot=33.0),
    'A5': dict(desc='Arrival rate up (just above baseline): slot 34 SU',
               cfg={'SLOT_TIME0': -3.5, 'MAX_SIMULATION_TIME': 170000},
               slot=34.0),
    'A6': dict(desc='Arrival rate down (slightly): slot 38 SU',
               cfg={'SLOT_TIME0': 0.5, 'MAX_SIMULATION_TIME': 190000},
               slot=38.0),
    'A2': dict(desc='Arrival rate down: slot 42 SU',
               cfg={'SLOT_TIME0': 4.5, 'MAX_SIMULATION_TIME': 210000},
               slot=42.0),
    'B1': dict(desc='Task mix: camera-heavy (6 camera, 3 traffic)',
               cfg={}, slot=35.0,
               sensor='s_type = 1 if j % 3 != 2 else 2'),
    'B2': dict(desc='Task mix: traffic-heavy (6 traffic, 3 camera)',
               cfg={}, slot=35.0,
               sensor='s_type = 2 if j % 3 != 2 else 1'),
    'D': dict(desc='Task mix: wide task-size range',
              cfg={'NO_INSTRUCTIONS3_MIN': 50, 'NO_INSTRUCTIONS3_MAX': 200,
                   'NO_INSTRUCTIONS4_MIN': 60, 'NO_INSTRUCTIONS4_MAX': 220,
                   'SLOT_TIME0': -20}, slot=35.0),
    'C1': dict(desc='Node profile: three identical nodes (CPU 80)',
               cfg={'CPU_SPEED1': 80, 'CPU_SPEED2': 80, 'CPU_SPEED3': 80,
                    'FD_BANDWIDTH1': 800, 'FD_BANDWIDTH2': 800,
                    'FD_BANDWIDTH3': 800,
                    'FD_RAM1': 4000, 'FD_RAM2': 4000, 'FD_RAM3': 4000,
                    'SLOT_TIME0': 16.25}, slot=35.0),
    'C2': dict(desc='Node profile: very different nodes (CPU 20/80/320)',
               cfg={'CPU_SPEED1': 20, 'CPU_SPEED2': 80, 'CPU_SPEED3': 320,
                    'SLOT_TIME0': -40}, slot=35.0),
}


def patch_config(path, patches):
    with open(path) as f:
        text = f.read()
    for name, val in patches.items():
        text, n = re.subn(rf'^{name} = .*$', f'{name} = {val}', text,
                          flags=re.MULTILINE)
        if n != 1:
            raise RuntimeError(f'{name}: expected 1 match in config.py, got {n}')
    with open(path, 'w') as f:
        f.write(text)


def patch_sensor(path, new_line):
    with open(path) as f:
        text = f.read()
    if text.count(SENSOR_LINE) != 1:
        raise RuntimeError('sensor-type line not found exactly once')
    with open(path, 'w') as f:
        f.write(text.replace(SENSOR_LINE, new_line))


def build_folder(name, w):
    d = os.path.join(ROOT, name)
    os.makedirs(d, exist_ok=True)
    for fn in FILES:
        shutil.copy(os.path.join(SRC, fn), os.path.join(d, fn))
    patch_config(os.path.join(d, 'config.py'), w['cfg'])
    if 'sensor' in w:
        patch_sensor(os.path.join(d, 'IoTsimulation.py'), w['sensor'])
    return d


def num(pattern, text):
    m = re.search(pattern, text)
    return float(m.group(1)) if m else None


def parse(log):
    return dict(
        slot=num(r'SLOT DURATION: ([\d.]+)', log),
        latency=num(r'avg TimeTaken for Tasks:([\d.]+)', log),
        cloud=num(r'Cloud executed :(\d+) Tasks', log),
        total=num(r'Total Number of Tasks:(\d+)', log),
        energy=num(r'Total Fog layer Power Utilised:([\d.]+)', log),
        slots=num(r'Total Number of Slots:(\d+)', log),
    )


def load_ckpt():
    if os.path.exists(CKPT):
        with open(CKPT) as f:
            return json.load(f)
    return {}


def save_ckpt(c):
    with open(CKPT, 'w') as f:
        json.dump(c, f, indent=1)


def clean(d, keep_dataset_as=None):
    for fn in os.listdir(d):
        if fn.startswith('dataset-') and keep_dataset_as:
            shutil.move(os.path.join(d, fn), os.path.join(d, keep_dataset_as))
            keep_dataset_as = None
        elif fn.startswith(('tasks-', 'slots-', 'dataset-')) or fn in (
                'canonical_label_stats.txt', 'ga_convergence_output.txt'):
            os.remove(os.path.join(d, fn))


def run_one(name, w, d, sched, rm, seed):
    patch_config(os.path.join(d, 'config.py'),
                 {'RM_TYPE': rm, 'RANDOM_SEED': seed})
    t0 = time.time()
    p = subprocess.run([sys.executable, 'IoTsimulation.py'], cwd=d,
                       capture_output=True, text=True)
    wall = time.time() - t0
    log = p.stdout + p.stderr
    with open(os.path.join(RESULTS, f'log_{name}_{sched}_{seed}.txt'), 'w') as f:
        f.write(log)
    keep = None
    if sched == 'GA' and seed == KEEP_GA_DATASET_SEED:
        keep = f'GA_dataset_seed{seed}.txt'
    clean(d, keep)
    r = parse(log)
    r.update(workload=name, sched=sched, seed=seed, wall=round(wall, 1))
    if r['latency'] is None:
        raise RuntimeError(f'{name} {sched} {seed}: no result in log, '
                           f'see log_{name}_{sched}_{seed}.txt')
    if r['slot'] is None or abs(r['slot'] - w['slot']) > 0.01:
        raise RuntimeError(f"{name}: slot duration {r['slot']} but expected "
                           f"{w['slot']}. Stopping, check SLOT_TIME0.")
    return r


def append_csv(r):
    new = not os.path.exists(CSV_FILE)
    cols = ['workload', 'sched', 'seed', 'latency', 'cloud', 'total',
            'cloud_pct', 'energy', 'slots', 'slot', 'wall']
    r['cloud_pct'] = round(100 * r['cloud'] / r['total'], 2)
    with open(CSV_FILE, 'a', newline='') as f:
        wr = csv.DictWriter(f, fieldnames=cols)
        if new:
            wr.writeheader()
        wr.writerow({k: r.get(k) for k in cols})


def mean(x):
    return sum(x) / len(x)


def sd(x):
    m = mean(x)
    return math.sqrt(sum((v - m) ** 2 for v in x) / (len(x) - 1)) if len(x) > 1 else 0.0


def summary():
    if not os.path.exists(CSV_FILE):
        print('no results yet')
        return
    rows = list(csv.DictReader(open(CSV_FILE)))
    lines = []
    hdr = (f"{'Workload':8} {'Sched':8} {'n':>2} {'Latency':>9} {'sd':>7} "
           f"{'Cloud%':>7} {'Energy':>12} {'Tasks':>8}")
    lines.append(hdr)
    for wn in WORKLOADS:
        by = {}
        for r in rows:
            if r['workload'] == wn:
                by.setdefault(r['sched'], []).append(r)
        for s in ['DEFAULT', 'GA', 'RR', 'RF_SMART']:
            if s not in by:
                continue
            L = [float(r['latency']) for r in by[s]]
            C = [float(r['cloud_pct']) for r in by[s]]
            E = [float(r['energy']) for r in by[s] if r['energy']]
            T = [float(r['total']) for r in by[s]]
            lines.append(f"{wn:8} {s:8} {len(L):>2} {mean(L):9.3f} {sd(L):7.3f} "
                         f"{mean(C):7.2f} {mean(E) if E else 0:12.0f} {mean(T):8.0f}")
        if 'GA' in by and 'DEFAULT' in by and len(by['GA']) == len(by['DEFAULT']):
            g = [float(a['latency']) - float(b['latency'])
                 for a, b in zip(by['GA'], by['DEFAULT'])]
            lines.append(f"{'':8} gap GA-DEFAULT: mean {mean(g):.3f}, "
                         f"range {min(g):.3f} to {max(g):.3f}")
        if 'GA' in by and 'RR' in by and len(by['GA']) == len(by['RR']):
            g = [float(a['latency']) - float(b['latency'])
                 for a, b in zip(by['GA'], by['RR'])]
            lines.append(f"{'':8} gap GA-RR     : mean {mean(g):.3f}, "
                         f"range {min(g):.3f} to {max(g):.3f}")
        if 'RF_SMART' in by:
            for other in ('GA', 'DEFAULT'):
                if other not in by:
                    continue
                o = {r['seed']: float(r['latency']) for r in by[other]}
                g = [float(r['latency']) - o[r['seed']]
                     for r in by['RF_SMART'] if r['seed'] in o]
                if g:
                    lines.append(f"{'':8} gap RF_SMART-{other} (same seeds, "
                                 f"n={len(g)}): mean {mean(g):.3f}, "
                                 f"range {min(g):.3f} to {max(g):.3f}")
        lines.append('')
    text = '\n'.join(lines)
    with open(os.path.join(RESULTS, 'summary.txt'), 'w') as f:
        f.write(text)
    print(text)


def main():
    os.makedirs(RESULTS, exist_ok=True)
    if '--summary' in sys.argv:
        summary()
        return
    names = list(WORKLOADS)
    if '--only' in sys.argv:
        names = [a for a in sys.argv[sys.argv.index('--only') + 1:]
                 if not a.startswith('--')]
        bad = [n for n in names if n not in WORKLOADS]
        if bad:
            sys.exit(f'unknown workload(s): {bad}. Choose from {list(WORKLOADS)}')

    scheds = [s for s, _ in SCHEDULERS]
    if '--sched' in sys.argv:
        scheds = [a for a in sys.argv[sys.argv.index('--sched') + 1:]
                  if not a.startswith('--') and a in dict(SCHEDULERS)]
        if not scheds:
            sys.exit('--sched needs DEFAULT, GA or RR')

    done = set()
    if os.path.exists(CSV_FILE):
        for r in csv.DictReader(open(CSV_FILE)):
            done.add(f"{r['workload']}_{r['sched']}_{r['seed']}")
    ck = load_ckpt()
    for name in names:
        w = WORKLOADS[name]
        print(f"\n=== {name}: {w['desc']} ===", flush=True)
        d = build_folder(name, w)
        for sched, rm in SCHEDULERS:
            if sched not in scheds:
                continue
            for seed in SEEDS:
                key = f'{name}_{sched}_{seed}'
                if key in ck or key in done:
                    print(f'  skip {key}', flush=True)
                    continue
                r = run_one(name, w, d, sched, rm, seed)
                ck[key] = r
                save_ckpt(ck)
                append_csv(r)
                print(f"  {key}: latency {r['latency']:.4f}, cloud "
                      f"{r['cloud']:.0f}/{r['total']:.0f}, "
                      f"{r['wall']} s", flush=True)
    summary()


if __name__ == '__main__':
    main()