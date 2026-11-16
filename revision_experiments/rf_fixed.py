import r1_1_workloads as m
m.WORKLOADS['FX10_BASE'] = dict(desc='Corrected GA, baseline, 10 seeds', cfg={}, slot=35.0)
for s in range(44, 52):
    m.WORKLOADS['FXRF_%d' % s] = dict(desc='RF_SMART, corrected labels, seed %d' % s, cfg={}, slot=35.0)
import rf_workloads as r
if __name__ == '__main__':
    r.main()
