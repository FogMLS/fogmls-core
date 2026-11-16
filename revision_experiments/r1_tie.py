import os, re, sys
import r1_1_workloads as m
import rf_workloads as r

TOL = {'FXT0': '0.0', 'FXT10': '0.1'}
for k, v in TOL.items():
    m.WORKLOADS[k] = dict(desc='Corrected GA, tie tolerance ' + v, cfg={}, slot=35.0)
m.SEEDS = [42]
m.SCHEDULERS = [('GA', 1)]

_orig = m.build_folder
def bf(name, w):
    d = _orig(name, w)
    if name in TOL:
        p = os.path.join(d, 'GeneticAlg.py')
        t = open(p).read()
        t, n = re.subn(r'^TIE_RELATIVE_TOLERANCE = [\d.]+',
                       'TIE_RELATIVE_TOLERANCE = ' + TOL[name], t, flags=re.M)
        if n != 1:
            sys.exit('TIE_RELATIVE_TOLERANCE line found %d times, expected 1' % n)
        open(p, 'w').write(t)
    return d
m.build_folder = bf

if __name__ == '__main__':
    sys.argv = ['r1_tie.py', '--only', 'FXT0', 'FXT10']
    m.main()
    for name in TOL:
        r.train(name, os.path.join(m.ROOT, name))
