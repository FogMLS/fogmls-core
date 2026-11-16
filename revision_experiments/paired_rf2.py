import csv, re, os, statistics as st
R = './'
rows = list(csv.DictReader(open(R + 'runs.csv')))
def get(w, s):
    return {int(r['seed']): float(r['latency']) for r in rows if r['workload'] == w and r['sched'] == s}
def getlog(w, s):
    out = {}
    for seed in range(42, 52):
        p = R + 'log_%s_%s_%d.txt' % (w, s, seed)
        if os.path.exists(p):
            m = re.search(r'avg TimeTaken for Tasks:([\d.]+)', open(p).read())
            if m:
                out[seed] = float(m.group(1))
    return out
def show(label, a, b):
    seeds = sorted(set(a) & set(b))
    d = [a[k] - b[k] for k in seeds]
    if d:
        sd = st.stdev(d) if len(d) > 1 else 0
        print(f"  {label}: n={len(d)} mean {st.mean(d):+.3f} sd {sd:.3f} range {min(d):+.3f} to {max(d):+.3f}")
rf = dict(get('FX10_BASE', 'RF_SMART'))
for s in range(44, 52):
    rf.update(get('FXRF_%d' % s, 'RF_SMART'))
rf = {k: v for k, v in rf.items() if 43 <= k <= 51}
print('MAIN seeds', sorted(rf))
for o in ('DEFAULT', 'RR'):
    ref = dict(getlog('FX10_BASE', o)); ref.update(get('FX10_BASE', o)); ref.update(get('BASE', o))
    show('RF_SMART minus ' + o, rf, ref)
for n in ['A4', 'A6', 'B1', 'B2', 'D', 'C1', 'C2']:
    rfw = {}
    for s in (43, 44, 45):
        rfw.update(get('FXW_%s_%d' % (n, s), 'RF_SMART'))
    print(n)
    for o in ('DEFAULT', 'RR'):
        show('RF_SMART minus ' + o, rfw, get(n, o))
