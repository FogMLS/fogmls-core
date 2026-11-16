import r1_1_workloads as m
new = {
    'FX_BASE': dict(desc='Corrected GA, baseline', cfg={}, slot=35.0),
    'FX_W1': dict(desc='Corrected GA, latency only', cfg={'ALPHA': 1.0, 'BETA': 0.0}, slot=35.0),
    'FX_W2': dict(desc='Corrected GA, energy only', cfg={'ALPHA': 0.0, 'BETA': 1.0}, slot=35.0),
    'FX_W3': dict(desc='Corrected GA, equal thirds', cfg={'ALPHA': '1/3', 'BETA': '1/3', 'GAMMA': '1/3'}, slot=35.0),
}
for k, w in list(m.WORKLOADS.items()):
    if k != 'BASE':
        new['FX_' + k] = dict(w)
m.WORKLOADS.update(new)
if __name__ == '__main__':
    m.main()
