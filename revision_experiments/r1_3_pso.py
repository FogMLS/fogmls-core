import os, re, sys
import r1_1_workloads as m
import rf_workloads as r

TOL = {'PSOT05': '0.05', 'PSOT10': '0.1'}
for k, v in TOL.items():
    m.WORKLOADS[k] = dict(desc='PSO, tie tolerance ' + v + ' (GA and PSO constants)', cfg={}, slot=35.0)
m.SEEDS = list(range(42, 52))
m.SCHEDULERS = [('PSO', 6)]

def patch_tol(path, value):
    t = open(path).read()
    if path.endswith('ParticleSwarmAlg.py') and not re.search(r'^PSO_DETERMINISTIC = False', t, flags=re.M):
        sys.exit('PSO_DETERMINISTIC is not False')
    t, n = re.subn(r'^TIE_RELATIVE_TOLERANCE = [\d.]+',
                   'TIE_RELATIVE_TOLERANCE = ' + value, t, flags=re.M)
    if n != 1:
        sys.exit('%s: tolerance line found %d times' % (path, n))
    open(path, 'w').write(t)

_orig_bf = m.build_folder
def bf(name, w):
    d = _orig_bf(name, w)
    if name in TOL:
        patch_tol(os.path.join(d, 'ParticleSwarmAlg.py'), TOL[name])
        patch_tol(os.path.join(d, 'GeneticAlg.py'), TOL[name])
    return d
m.build_folder = bf

_orig_run = m.run_one
def run_one(name, w, d, sched, rm, seed):
    res = _orig_run(name, w, d, 'GA', rm, seed)
    res['sched'] = 'PSO'
    return res
m.run_one = run_one

if __name__ == '__main__':
    name = sys.argv[1]
    sys.argv = ['r1_3_pso.py', '--only', name]
    m.main()
    r.train(name, os.path.join(m.ROOT, name))
