import r1_1_workloads as m
NAMES = ['A3', 'A5', 'A2']
for n in NAMES:
    m.WORKLOADS['FX_' + n] = dict(m.WORKLOADS[n])
    for s in (43, 44, 45):
        m.WORKLOADS['FXW_%s_%d' % (n, s)] = dict(m.WORKLOADS[n])
import rf_workloads as r
if __name__ == '__main__':
    r.main()
