# -*- coding: utf-8 -*-
"""
main_ga.py

Runs the GA comparison scenario matching FogMLS's real per-slot loop
exactly. Uses GA's own DelayFitness formula to measure per-task total
time -- the exact same metric FogMLS itself reports (compute +
transmission + swap + propagation), not a re-derived approximation.

This does NOT route through YAFS's tuple/event machinery, since GA is
fundamentally a "collect all jobs in a slot, then solve together"
algorithm -- forcing this through a continuous, per-tuple routing
architecture (like was needed for Round Robin) would fight the tool
rather than validate it. Instead this directly mirrors FogMLS's real
per-slot loop, using the SAME GA code (patched to the confirmed,
HPC-verified 88.11% configuration).
"""

import random
from ga_core import Resource, Job, GeneticAlgorithm, GAConfig, compute_swap_time, compute_exec_duration
from fogmls_shared_config import (
    SENSORS_PER_CLUSTER, RANDOM_SEED, NUM_FOG_NODES as NUM_CLUSTERS,
    CPU_SPEED, FD_RAM, CLOUD_CPU_SPEED as CLOUD_CPU, CLOUD_RAM,
    SLOT_TIME, MAX_SIMULATION_TIME, BUSY_POWER,
    NO_INSTRUCTIONS_BY_ID as NO_INSTR, DATA_SIZE_BY_ID as DATA_SIZE,
    CAMERA_SEN, TRAFFIC_SEN, VEHICLE_IDENTIFICATION as VEHICLE_ID,
    TRAFFIC_LOCATION as TRAFFIC_LOC, ACTION_ON_VEHICLE as ACTION_VEH,
    ACTION_CLEAR_TRAFFIC as ACTION_TRAF
)

FD_BANDWIDTH = [400, 800, 1600]  # real SU values (not pre-divided -- GA uses direct formula, no YAFS unit quirk)
CLOUD_BANDWIDTH = 1600  # real SU value


class FogJob:
    """Tracks a job through its 3-stage lifecycle, matching FogMLS's
    real Job/Task class behavior."""
    def __init__(self, task_type, slot_created):
        self.task_type = task_type
        self.slot_created = slot_created
        self.no_valid_task = False
        self._roll_instructions_and_data()

    def _roll_instructions_and_data(self):
        imin, imax = NO_INSTR[self.task_type]
        dmin, dmax = DATA_SIZE[self.task_type]
        self.no_instructions = random.randint(imin, imax) if imax > imin else imin
        self.data_bytes = random.randint(dmin, dmax) if dmax > dmin else dmin

    def get_noInstructions(self):
        return self.no_instructions

    def get_dataBytes(self):
        return self.data_bytes

    def advance_stage(self):
        """Matches FogMLS's devices.py stage transition logic exactly.
        IMPORTANT: no_valid_task is set only AFTER stage 3 has been
        processed and timed this slot -- not at the moment of
        transitioning INTO stage 3, otherwise stage 3 gets filtered
        out of the queue before it's ever actually executed/timed."""
        if self.task_type == CAMERA_SEN:
            self.task_type = VEHICLE_ID
        elif self.task_type == TRAFFIC_SEN:
            self.task_type = TRAFFIC_LOC
        elif self.task_type == VEHICLE_ID:
            self.task_type = ACTION_VEH
        elif self.task_type == TRAFFIC_LOC:
            self.task_type = ACTION_TRAF
        elif self.task_type in (ACTION_VEH, ACTION_TRAF):
            # This job's stage-3 (action) task was JUST processed and
            # timed this slot -- chain terminates now.
            self.no_valid_task = True
        self._roll_instructions_and_data()


def main():
    random.seed(RANDOM_SEED)

    fog_resources = [Resource(CPU_SPEED[i], FD_BANDWIDTH[i], FD_RAM[i], SLOT_TIME, BUSY_POWER[i])
                     for i in range(NUM_CLUSTERS)]
    cloud = Resource(CLOUD_CPU, CLOUD_BANDWIDTH, CLOUD_RAM, float('inf'))
    cfg = GAConfig()

    total_sensors = NUM_CLUSTERS * SENSORS_PER_CLUSTER
    jobQ = []
    total_slots = int(MAX_SIMULATION_TIME / SLOT_TIME)

    task_times = []
    cloud_count = 0
    fog_count = 0
    total_tasks_completed = 0

    for slot in range(1, total_slots + 1):
        # Generate new stage-1 jobs, matching FogMLS's alternating CAM/TRAF per sensor
        for s in range(total_sensors):
            task_type = CAMERA_SEN if (s % 2 == 0) else TRAFFIC_SEN
            jobQ.append(FogJob(task_type, slot))

        if len(jobQ) > 0:
            jbs = [Job(j.get_noInstructions(), j.get_dataBytes(), ref=j) for j in jobQ]
            ga = GeneticAlgorithm(fog_resources, jbs, cloud, cfg)
            ga.GenerateOptimalChromosome()
            best = ga.returnBESTCRM()

            for idx, job in enumerate(jobQ):
                FR_ID = best.getRs(idx)
                if FR_ID is not None and FR_ID < len(fog_resources):
                    res = fog_resources[FR_ID]
                    swap_time = compute_swap_time(job.get_noInstructions(), job.get_dataBytes(), res.getRAM())
                    duration = (job.get_noInstructions() / res.get_CPUSpeed() +
                                job.get_dataBytes() / res.getBANDWIDTH() +
                                cfg.LATENCY_ED_FR + swap_time)
                    fog_count += 1
                else:
                    swap_time = compute_swap_time(job.get_noInstructions(), job.get_dataBytes(), cloud.getRAM())
                    duration = (job.get_noInstructions() / cloud.get_CPUSpeed() +
                                job.get_dataBytes() / cloud.getBANDWIDTH() +
                                cfg.LATENCY_CS_FR + swap_time)
                    cloud_count += 1

                task_times.append(duration)
                total_tasks_completed += 1
                job.advance_stage()

            jobQ = [j for j in jobQ if not j.no_valid_task]

        if slot % 500 == 0 or slot == total_slots:
            print(f"[SLOT {slot}/{total_slots}] ({100*slot/total_slots:.1f}%) tasks so far: {total_tasks_completed}")

    print()
    print("Simulation Completed")
    print("-----------Results (GA, YAFS-topology-matched)--------------------")
    print("Total Simulation Time:", MAX_SIMULATION_TIME)
    print("Total Number of Tasks:", total_tasks_completed)
    print()
    print("avg TimeTaken for Tasks:", sum(task_times) / len(task_times))
    print("min/max TimeTaken for Tasks:", min(task_times), "/", max(task_times))
    print()
    print("Fog executed:", fog_count, " Cloud executed:", cloud_count)
    print("Cloud fraction:", cloud_count / total_tasks_completed)


if __name__ == '__main__':
    main()
