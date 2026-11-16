import r1_1_workloads as m
for s in range(43, 52):
    m.WORKLOADS['FXPS_%d' % s] = dict(desc='RF_SMART, PSO-label classifier (5 percent), seed %d' % s, cfg={}, slot=35.0)
import rf_workloads as r
if __name__ == '__main__':
    r.main()
