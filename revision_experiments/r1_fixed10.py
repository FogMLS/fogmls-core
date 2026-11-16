import r1_1_workloads as m
m.SEEDS = list(range(42, 52))
m.WORKLOADS.update({'FX10_BASE': dict(desc='Corrected GA, baseline, 10 seeds', cfg={}, slot=35.0)})
if __name__ == '__main__':
    m.main()
