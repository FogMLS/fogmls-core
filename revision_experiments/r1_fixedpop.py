import r1_1_workloads as m
m.WORKLOADS.update({
    'FXP4': dict(desc='Corrected GA, population 4', cfg={'N_CHROMOSOMES': 4}, slot=35.0),
    'FXP16': dict(desc='Corrected GA, population 16', cfg={'N_CHROMOSOMES': 16}, slot=35.0),
    'FXP32': dict(desc='Corrected GA, population 32', cfg={'N_CHROMOSOMES': 32}, slot=35.0),
})
if __name__ == '__main__':
    m.main()
