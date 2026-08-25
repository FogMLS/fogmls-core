# -*- coding: utf-8 -*-
"""
main_fogmls.py

Runs the YAFS comparison scenario matching FogMLS's real topology and
Round Robin scheduler.
"""

import random
import time
import numpy as np

from yafs.core import Sim
from yafs.population import Statical
from yafs.distribution import deterministic_distribution

from topology_fogmls import build_topology
from application_fogmls import create_application
from placement_fogmls import FogPlacement
from selection_fogmls import RoundRobinSelection
from fogmls_shared_config import RANDOM_SEED, SLOT_TIME

MAX_SIMULATION_TIME = 175000


def slot_monitor(env, selector, slot_time, max_time):
    last_gen = 0
    last_fog = 0
    last_cloud = 0
    slot_no = 0
    total_slots = int(max_time / slot_time) if slot_time > 0 else 0

    while env.now < max_time:
        yield env.timeout(slot_time)
        slot_no += 1

        gen_delta = selector.total_generated - last_gen
        fog_delta = selector.total_fog_assigned - last_fog
        cloud_delta = selector.total_cloud_escalated - last_cloud
        progress = (env.now / max_time) * 100.0

        if slot_no % 500 == 0 or slot_no == total_slots:
            print(f"[SIM] Slot {slot_no}/{total_slots} ({progress:.1f}%) | "
                  f"gen={gen_delta} fog={fog_delta} cloud={cloud_delta} | "
                  f"SimTime: {env.now:.0f}/{max_time}")

        last_gen = selector.total_generated
        last_fog = selector.total_fog_assigned
        last_cloud = selector.total_cloud_escalated


def main(simulated_time):
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    t = build_topology()

    app = create_application()

    placement = FogPlacement("FogRoundRobin")
    placement.scaleService({"Worker": 1})

    pop = Statical("Statical")
    pop.set_sink_control({
        "model": "actuator-device", "number": 1,
        "module": app.get_sink_modules()
    })

    dCam = deterministic_distribution(name="Deterministic-Cam", time=SLOT_TIME)
    dTraf = deterministic_distribution(name="Deterministic-Traf", time=SLOT_TIME)

    pop.set_src_control({
        "model": "sensor-camera", "number": 1,
        "message": app.get_message("M.CAM"), "distribution": dCam
    })
    pop.set_src_control({
        "model": "sensor-traffic", "number": 1,
        "message": app.get_message("M.TRAF"), "distribution": dTraf
    })

    selector = RoundRobinSelection()

    s = Sim(t, default_results_path="Results_FogMLS_RR")
    s.deploy_app2(app, placement, pop, selector)
    s.env.process(slot_monitor(s.env, selector, SLOT_TIME, simulated_time))
    s.run(simulated_time, show_progress_monitor=False)


def print_summary(simulated_time):
    import pandas as pd

    df = pd.read_csv("Results_FogMLS_RR.csv")
    comp = df[df['type'] == 'COMP_M'].copy()
    comp['total_latency'] = comp['time_out'] - comp['time_emit']

    fog_ids = sorted(comp['DES.dst'].unique())

    print("")
    print("Simulation Completed")
    print("-----------Results--------------------")
    print("Total Simulation Time:" + str(simulated_time))
    print("Total Number of Tasks:" + str(len(comp)))
    print("")
    print("avg TimeTaken for Tasks:" + str(comp['total_latency'].mean()))
    print("median TimeTaken for Tasks:" + str(comp['total_latency'].median()))
    print("min/max TimeTaken for Tasks:" + str(comp['total_latency'].min()) +
          " / " + str(comp['total_latency'].max()))
    print("")
    print("Per Fog Node:")
    for fid in fog_ids:
        sub = comp[comp['DES.dst'] == fid]
        print("  DES " + str(fid) + ": " + str(len(sub)) +
              " tasks, avg latency " + str(round(sub['total_latency'].mean(), 4)))
    print("")


if __name__ == '__main__':
    import logging.config
    import os
    logging.config.fileConfig(os.getcwd() + '/logging.ini')

    SIM_TIME = 175000
    start = time.time()
    main(simulated_time=SIM_TIME)
    print("--- %s seconds ---" % (time.time() - start))
    print_summary(SIM_TIME)
