# -*- coding: utf-8 -*-
"""
scale_rerun.py - rerun of the Section 6.5 scalability study with the corrected GA.
Sensor scaling: 9/27/45/63 sensors per cluster, 3 fog nodes
Node scaling:   1/2/4/5/6 fog nodes, 9 sensors per cluster (3 nodes = the 9-sensor run)
Schedulers: DEFAULT, GA, RR.  Seeds: 42, 43, 44.
Usage (from ~/fogmls_fixed):
    python3 scale_rerun.py --list | --task N | --summary
"""
import csv, fcntl, math, os, re, shutil, subprocess, sys, time

SRC = os.path.expanduser('~/fogmls_fixed')
ROOT = os.path.expanduser('~/scale_rerun')
RESULTS = os.path.join(ROOT, 'results')
CSV_FILE = os.path.join(RESULTS, 'scale_runs.csv')
FILES = ['IoTsimulation.py', 'devices.py', 'config.py',
         'GeneticAlg.py', 'ParticleSwarmAlg.py']
SEEDS = [42, 43, 44]
SCHEDULERS = [('DEFAULT', 0), ('GA', 1), ('RR', 4)]
SENSOR_LEVELS = [(9, 81), (27, 243), (45, 405), (63, 567)]
NODE_LEVELS = [(1, 3, 27), (2, 4, 54), (3, 5, 81),
               (4, 6, 108), (5, 7, 135), (6, 8, 162)]
COLS = ['kind', 'level', 'sched', 'seed', 'latency', 'cloud', 'total',
        'cloud_pct', 'energy', 'slots', 'slot', 'ga_slot', 'wall', 'cycled']


def build_tasks():
    tasks = []
    for s, mj in SENSOR_LEVELS:
        for sched, rm in SCHEDULERS:
            for seed in SEEDS:
                tasks.append(('S', s, sched, rm, seed,
                              {'SENSORS_PER_CLUSTER': s, 'MAX_JOBS': mj}))
    for n, rs, mj in NODE_LEVELS:
        if n == 3:
            continue
        for sched, rm in SCHEDULERS:
            for seed in SEEDS:
                tasks.append(('N', n, sched, rm, seed,
                              {'SENSORS_PER_CLUSTER': 9, 'MAX_RS': rs, 'MAX_JOBS': mj}))

    def cost(t):
        if t[2] == 'GA' and t[0] == 'N' and t[1] >= 5:
            return 0
        if t[2] == 'GA':
            return 1
        return 2
    tasks.sort(key=cost)
    return tasks


def patch_config(path, patches):
    with open(path) as f:
        text = f.read()
    for name, val in patches.items():
        text, n = re.subn(rf'^{name} = .*$', f'{name} = {val}', text, flags=re.MULTILINE)
        if n != 1:
            raise RuntimeError(f'{name}: expected 1 match in config.py, got {n}')
    with open(path, 'w') as f:
        f.write(text)


def num(pattern, text):
    m = re.search(pattern, text)
    return float(m.group(1)) if m else None


def parse(log):
    return dict(
        slot=num(r'SLOT DURATION: ([\d.]+)', log),
        latency=num(r'avg TimeTaken for Tasks:([\d.]+)', log),
        cloud=num(r'Cloud executed\s*:(\d+) Tasks', log),
        total=num(r'Total Number of Tasks:(\d+)', log),
        energy=num(r'Total Fog layer Power Utilised:([\d.]+)', log),
        slots=num(r'Total Number of Slots:(\d+)', log),
        ga_slot=num(r'Avg time per slot: ([\d.]+) seconds', log),
        cycled=int('[WARNING] No unique specs' in log))


def done_keys():
    if not os.path.exists(CSV_FILE):
        return set()
    with open(CSV_FILE) as f:
        return {(r['kind'], r['level'], r['sched'], r['seed']) for r in csv.DictReader(f)}


def append_csv(row):
    os.makedirs(RESULTS, exist_ok=True)
    with open(CSV_FILE, 'a', newline='') as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        wr = csv.DictWriter(f, fieldnames=COLS)
        if os.fstat(f.fileno()).st_size == 0:
            wr.writeheader()
        wr.writerow({k: row.get(k) for k in COLS})
        f.flush()


def run_task(idx):
    kind, level, sched, rm, seed, cfg = build_tasks()[idx]
    if (kind, str(level), sched, str(seed)) in done_keys():
        print(f'task {idx} already done, skipping')
        return
    tag = f'{kind}{level}_{sched}_{seed}'
    d = os.path.join(ROOT, 'work', tag)
    if os.path.exists(d):
        shutil.rmtree(d)
    os.makedirs(d)
    os.makedirs(RESULTS, exist_ok=True)
    for fn in FILES:
        shutil.copy(os.path.join(SRC, fn), os.path.join(d, fn))
    patches = dict(cfg)
    patches.update({'RM_TYPE': rm, 'RANDOM_SEED': seed})
    patch_config(os.path.join(d, 'config.py'), patches)
    print(f'task {idx}: {tag}  {patches}', flush=True)
    t0 = time.time()
    p = subprocess.run([sys.executable, 'IoTsimulation.py'], cwd=d,
                       capture_output=True, text=True)
    wall = time.time() - t0
    log = p.stdout + p.stderr
    with open(os.path.join(RESULTS, f'log_{tag}.txt'), 'w') as f:
        f.write(log)
    r = parse(log)
    if r['latency'] is None:
        raise RuntimeError(f'{tag}: no result, see results/log_{tag}.txt (folder kept: {d})')
    r.update(kind=kind, level=level, sched=sched, seed=seed, wall=round(wall, 1))
    r['cloud_pct'] = round(100 * r['cloud'] / r['total'], 2)
    append_csv(r)
    shutil.rmtree(d)
    print(f'done {tag}: latency {r["latency"]:.4f}, cloud {int(r["cloud"])}, '
          f'tasks {int(r["total"])}, slot {r["slot"]}, ga_slot {r["ga_slot"]}, '
          f'wall {wall:.0f}s', flush=True)


def mean(x):
    return sum(x) / len(x)


def sd(x):
    m = mean(x)
    return math.sqrt(sum((v - m) ** 2 for v in x) / (len(x) - 1)) if len(x) > 1 else 0.0


def summary():
    if not os.path.exists(CSV_FILE):
        print('no results yet')
        return
    with open(CSV_FILE) as f:
        rows = list(csv.DictReader(f))

    def get(kind, level, sched):
        if kind == 'N' and level == 3:
            kind, level = 'S', 9
        return sorted([r for r in rows if r['kind'] == kind and int(r['level']) == level
                       and r['sched'] == sched], key=lambda r: int(r['seed']))

    lines = []

    def block(title, kind, levels, label):
        lines.append(title)
        lines.append(f"{label:>6} {'Sched':8} {'n':>2} {'Latency':>9} {'sd':>7} "
                     f"{'Cloud%':>7} {'Tasks':>9} {'Slots':>6} {'GA s/slot':>10} {'Wall s':>8}")
        for lvl in levels:
            by = {s: get(kind, lvl, s) for s, _ in SCHEDULERS}
            for s, _ in SCHEDULERS:
                rr = by[s]
                if not rr:
                    continue
                L = [float(r['latency']) for r in rr]
                C = [float(r['cloud_pct']) for r in rr]
                T = [float(r['total']) for r in rr]
                S = [float(r['slots']) for r in rr]
                W = [float(r['wall']) for r in rr]
                G = [float(r['ga_slot']) for r in rr if r['ga_slot'] not in ('', 'None')]
                lines.append(f"{lvl:>6} {s:8} {len(L):>2} {mean(L):9.3f} {sd(L):7.3f} "
                             f"{mean(C):7.2f} {mean(T):9.0f} {mean(S):6.0f} "
                             f"{(f'{mean(G):.4f}' if G else '-'):>10} {mean(W):8.0f}")
            for other in ('DEFAULT', 'RR'):
                a = {r['seed']: float(r['latency']) for r in by['GA']}
                b = {r['seed']: float(r['latency']) for r in by[other]}
                g = [a[k] - b[k] for k in a if k in b]
                if g:
                    lines.append(f"{'':6} gap GA-{other} (n={len(g)}): mean {mean(g):.3f}, "
                                 f"range {min(g):.3f} to {max(g):.3f}")
        lines.append('')

    block('SENSOR SCALING (3 fog nodes)', 'S', [s for s, _ in SENSOR_LEVELS], 'Sens')
    block('NODE SCALING (9 sensors per cluster)', 'N', [n for n, _, _ in NODE_LEVELS], 'Nodes')
    text = '\n'.join(lines)
    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, 'summary.txt'), 'w') as f:
        f.write(text)
    print(text)


if __name__ == '__main__':
    if '--list' in sys.argv:
        for i, t in enumerate(build_tasks()):
            print(i, t[0], t[1], t[2], t[4])
        print('total', len(build_tasks()))
    elif '--summary' in sys.argv:
        summary()
    elif '--task' in sys.argv:
        run_task(int(sys.argv[sys.argv.index('--task') + 1]))
    else:
        print(__doc__)
