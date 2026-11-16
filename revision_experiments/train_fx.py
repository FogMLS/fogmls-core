import os
import rf_fixed2 as f
import rf_workloads as r
for n in f.NAMES:
    r.train('FX_' + n, os.path.join(f.m.ROOT, 'FX_' + n))
