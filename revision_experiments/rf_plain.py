import os, sys, csv
import r1_1_workloads as m
from r1_1_workloads import WORKLOADS, ROOT, CSV_FILE, run_one, append_csv
m.WORKLOADS['FX10_BASE'] = dict(desc='Corrected GA, baseline, 10 seeds', cfg={}, slot=35.0)
for s in range(44, 52):
    m.WORKLOADS['FXRF_%d' % s] = dict(desc='RF plain, corrected labels, seed %d' % s, cfg={}, slot=35.0)

seed = int(sys.argv[1])
name = 'FX10_BASE' if seed == 43 else 'FXRF_%d' % seed
d = os.path.join(ROOT, name)
if not os.path.exists(os.path.join(d, 'random_forest_model.joblib')):
    sys.exit('no trained model in %s' % d)
done = set()
if os.path.exists(CSV_FILE):
    for r in csv.DictReader(open(CSV_FILE)):
        done.add('%s_%s_%s' % (r['workload'], r['sched'], r['seed']))
key = '%s_RF_%d' % (name, seed)
if key in done:
    sys.exit('skip ' + key)
r = run_one(name, WORKLOADS[name], d, 'RF', 7, seed)
append_csv(r)
print('%s: latency %.4f, cloud %.0f/%.0f, %s s' % (key, r['latency'], r['cloud'], r['total'], r['wall']), flush=True)
