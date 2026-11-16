import sys
import r1_1_workloads as m
which = sys.argv[1]
m.SEEDS = [42]
if which == 'GA':
    m.WORKLOADS['TIMEGA'] = dict(desc='timing GA', cfg={}, slot=35.0)
    m.SCHEDULERS = [('GA', 1)]
    sys.argv = ['r1_time.py', '--only', 'TIMEGA']
else:
    m.WORKLOADS['TIMEPSO'] = dict(desc='timing PSO', cfg={}, slot=35.0)
    m.SCHEDULERS = [('PSO', 6)]
    sys.argv = ['r1_time.py', '--only', 'TIMEPSO']
m.main()
