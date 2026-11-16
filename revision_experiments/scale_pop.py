# -*- coding: utf-8 -*-
"""
scale_pop.py - sensitivity test: GA population size at the scaling points where
GA lost to DEFAULT (45 and 63 sensors, 4 nodes).  GA only, 3 seeds.
Population 8 (main setting) is NOT rerun: it is already in scale_runs.csv.
Main GA setting stays 8 chromosomes / 10 iterations.

Usage (from ~/fogmls_fixed):
    python3 scale_pop.py --list | --task N | --summary
"""
import csv, fcntl, os, re, shutil, subprocess, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import scale_rerun as sr

CSV_FILE = os.path.join(sr.RESULTS, 'pop_runs.csv')
COLS = ['kind', 'level', 'pop', 'seed', 'latency', 'cloud', 'total',
        'cloud_pct', 'energy', 'slots', 'slot', 'ga_slot', 'wall']
POPS = [16, 32]
SEEDS = [42, 43, 44]
LEVELS = [('S', 45, {'SENSORS_PER_CLUSTER': 45, 'MAX_JOBS': 405}),
          ('S', 63, {'SENSORS_PER_CLUSTER': 63, 'MAX_JOBS': 567}),
          ('N', 4, {'SENSORS_PER_CLUSTER': 9, 'MAX_RS': 6, 'MAX_JOBS': 108})]


def build_tasks():
    t = []
    for pop in sorted(POPS, reverse=True):
        for kind, level, cfg in LEVELS:
            for seed in SEEDS:
                t.append((kind, level, pop, seed, cfg))
    return t


def done_keys():
    if not os.path.exists(CSV_FILE):
        return set()
    with open(CSV_FILE) as f:
        return {(r['kind'], r['level'], r['pop'], r['seed']) for r in csv.DictReader(f)}


def append_csv(row):
    os.makedirs(sr.RESULTS, exist_ok=True)
    with open(CSV_FILE, 'a', newline='') as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        wr = csv.DictWriter(f, fieldnames=COLS)
        if os.fstat(f.fileno()).st_size == 0:
            wr.writeheader()
        wr.writerow({k: row.get(k) for k in COLS})
        f.flush()


def run_task(idx):
    kind, level, pop, seed, cfg = build_tasks()[idx]
    if (kind, str(level), str(pop), str(seed)) in done_keys():
        print(f'task {idx} already done, skipping')
        return
    tag = f'P{kind}{level}_pop{pop}_{seed}'
    d = os.path.join(sr.ROOT, 'work', tag)
    if os.path.exists(d):
        shutil.rmtree(d)
    os.makedirs(d)
    os.makedirs(sr.RESULTS, exist_ok=True)
    for fn in sr.FILES:
        shutil.copy(os.path.join(sr.SRC, fn), os.path.join(d, fn))
    patches = dict(cfg)
    patches.update({'RM_TYPE': 1, 'RANDOM_SEED': seed, 'N_CHROMOSOMES': pop})
    sr.patch_config(os.path.join(d, 'config.py'), patches)
    print(f'task {idx}: {tag} {patches}', flush=True)
    t0 = time.time()
    p = subprocess.run([sys.executable, 'IoTsimulation.py'], cwd=d,
                       capture_output=True, text=True)
    wall = time.time() - t0
    log = p.stdout + p.stderr
    with open(os.path.join(sr.RESULTS, f'log_{tag}.txt'), 'w') as f:
        f.write(log)
    r = sr.parse(log)
    if r['latency'] is None:
        raise RuntimeError(f'{tag}: no result, see results/log_{tag}.txt (folder kept: {d})')
    r.update(kind=kind, level=level, pop=pop, seed=seed, wall=round(wall, 1))
    r['cloud_pct'] = round(100 * r['cloud'] / r['total'], 2)
    append_csv(r)
    shutil.rmtree(d)
    print(f'done {tag}: latency {r["latency"]:.4f}, tasks {int(r["total"])}, '
          f'ga_slot {r["ga_slot"]}, wall {wall:.0f}s', flush=True)


def summary():
    if not os.path.exists(CSV_FILE):
        print('no results yet')
        return
    with open(CSV_FILE) as f:
        pop_rows = list(csv.DictReader(f))
    with open(sr.CSV_FILE) as f:
        base_rows = list(csv.DictReader(f))
    lines = [f"{'Setting':10} {'Method':14} {'n':>2} {'Latency':>9} {'sd':>7} "
             f"{'Tasks':>9} {'GA s/slot':>10} {'Wall s':>8}"]

    def add(setting, method, rr):
        if not rr:
            return None
        L = [float(r['latency']) for r in rr]
        T = [float(r['total']) for r in rr]
        W = [float(r['wall']) for r in rr]
        G = [float(r['ga_slot']) for r in rr if r['ga_slot'] not in ('', 'None')]
        lines.append(f"{setting:10} {method:14} {len(L):>2} {sr.mean(L):9.3f} "
                     f"{sr.sd(L):7.3f} {sr.mean(T):9.0f} "
                     f"{(f'{sr.mean(G):.4f}' if G else '-'):>10} {sr.mean(W):8.0f}")
        return {r['seed']: float(r['latency']) for r in rr}

    for kind, level, _ in LEVELS:
        name = f"{level} {'sens' if kind == 'S' else 'nodes'}"
        base = [r for r in base_rows if r['kind'] == kind and int(r['level']) == level]
        dflt = add(name, 'DEFAULT', [r for r in base if r['sched'] == 'DEFAULT'])
        res = {8: add(name, 'GA pop 8', [r for r in base if r['sched'] == 'GA'])}
        for pop in sorted(POPS):
            res[pop] = add(name, f'GA pop {pop}',
                           [r for r in pop_rows if r['kind'] == kind
                            and int(r['level']) == level and int(r['pop']) == pop])
        for pop, d in res.items():
            if d and dflt:
                g = [d[k] - dflt[k] for k in d if k in dflt]
                if g:
                    lines.append(f"{'':10} gap GA pop {pop} minus DEFAULT (n={len(g)}): "
                                 f"mean {sr.mean(g):.3f}, range {min(g):.3f} to {max(g):.3f}")
        lines.append('')
    text = '\n'.join(lines)
    with open(os.path.join(sr.RESULTS, 'pop_summary.txt'), 'w') as f:
        f.write(text)
    print(text)


if __name__ == '__main__':
    if '--list' in sys.argv:
        for i, t in enumerate(build_tasks()):
            print(i, t[0], t[1], 'pop', t[2], 'seed', t[3])
        print('total', len(build_tasks()))
    elif '--summary' in sys.argv:
        summary()
    elif '--task' in sys.argv:
        run_task(int(sys.argv[sys.argv.index('--task') + 1]))
    else:
        print(__doc__)
