import csv, statistics as st
rows = list(csv.DictReader(open('runs.csv')))
def get(w, s):
    return {int(r['seed']): float(r['latency']) for r in rows if r['workload'] == w and r['sched'] == s}
def show(label, a, b):
    seeds = sorted(set(a) & set(b))
    d = [a[k] - b[k] for k in seeds]
    if d:
        sd = st.stdev(d) if len(d) > 1 else 0
        print(f"  {label}: n={len(d)} mean {st.mean(d):+.3f} sd {sd:.3f} range {min(d):+.3f} to {max(d):+.3f}")
# main table: RF seed 43 from FX10_BASE, 44 to 51 from FXRF_<s>
rf = dict(get('FX10_BASE', 'RF_SMART'))
for s in range(44, 52):
    rf.update(get('FXRF_%d' % s, 'RF_SMART'))
rf = {k: v for k, v in rf.items() if 43 <= k <= 51}
print('MAIN (seeds', sorted(rf), ')')
for o in ('GA', 'DEFAULT', 'RR'):
    show('RF_SMART minus ' + o, rf, get('FX10_BASE', o))
for n in ['A4', 'A6', 'B1', 'B2', 'D', 'C1', 'C2']:
    rfw = {}
    for s in (43, 44, 45):
        rfw.update(get('FXW_%s_%d' % (n, s), 'RF_SMART'))
    print(n)
    for o in ('GA', 'DEFAULT', 'RR'):
        show('RF_SMART minus ' + o, rfw, get('FX_' + n, o))
