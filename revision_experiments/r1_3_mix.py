import sys
import r1_1_workloads as m
m.WORKLOADS['PSOMIX'] = dict(desc='PSO, paper setting (GA constant 0.05, PSO constant 0.1)', cfg={}, slot=35.0)
m.SEEDS = list(range(42, 52))
m.SCHEDULERS = [('PSO', 6)]
_orig_run = m.run_one
def run_one(name, w, d, sched, rm, seed):
    res = _orig_run(name, w, d, 'GA', rm, seed)
    res['sched'] = 'PSO'
    return res
m.run_one = run_one
if __name__ == '__main__':
    sys.argv = ['r1_3_mix.py', '--only', 'PSOMIX']
    m.main()
