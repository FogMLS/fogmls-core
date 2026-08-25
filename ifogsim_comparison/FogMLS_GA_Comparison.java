package org.fog.test.perfeval;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import org.fog.test.perfeval.GACore.Chromosome;
import org.fog.test.perfeval.GACore.GeneticAlgorithm;
import org.fog.test.perfeval.GACore.Job;
import org.fog.test.perfeval.GACore.Resource;

/**
 * FogMLS_GA_Comparison.java
 *
 * GA-based comparison, mirroring FogMLS's real slot-based main loop
 * structure directly (IoTsimulation.py), using the real ported GA
 * algorithm (GACore.java, patched to the confirmed HPC-verified
 * 88.11% configuration: order-aware random init + canonical
 * labeling) and real topology specs. Same strategy used successfully
 * for the YAFS GA comparison (main_ga.py) -- does not route through
 * iFogSim's tuple/CloudSim event engine, since GA is fundamentally a
 * "collect all jobs in a slot, solve together" algorithm that
 * doesn't fit a continuous per-tuple architecture.
 */
public class FogMLS_GA_Comparison {

    static int NUM_CLUSTERS = 3;
    static int SENSORS_PER_CLUSTER = 9;
    static double[] CPU_SPEED = {40, 80, 160};
    static double[] FD_BANDWIDTH = {400, 800, 1600};
    static double[] FD_RAM = {2000, 4000, 8000};
    static double[] BUSY_POWER = {105.0, 110.0, 115.339};

    static double CLOUD_CPU = 48000;
    static double CLOUD_RAM = 1000000;
    static double CLOUD_BANDWIDTH = 1600;

    static double SLOT_TIME = 35; // matches FogMLS at 9 sensors/cluster
    static int MAX_SIMULATION_TIME = 175000;
    static long RANDOM_SEED = 42;

    static final int CAMERA_SEN = 1, TRAFFIC_SEN = 2, VEHICLE_ID = 3, TRAFFIC_LOC = 4, ACTION_VEH = 5, ACTION_TRAF = 6;

    static Random rng = new Random(RANDOM_SEED);

    static int[] instrMin = new int[7], instrMax = new int[7];
    static int[] dataMin = new int[7], dataMax = new int[7];
    static {
        instrMin[CAMERA_SEN] = 8; instrMax[CAMERA_SEN] = 14;
        instrMin[TRAFFIC_SEN] = 10; instrMax[TRAFFIC_SEN] = 16;
        instrMin[VEHICLE_ID] = 100; instrMax[VEHICLE_ID] = 140;
        instrMin[TRAFFIC_LOC] = 110; instrMax[TRAFFIC_LOC] = 150;
        instrMin[ACTION_VEH] = 0; instrMax[ACTION_VEH] = 0;
        instrMin[ACTION_TRAF] = 0; instrMax[ACTION_TRAF] = 0;

        dataMin[CAMERA_SEN] = 800; dataMax[CAMERA_SEN] = 1200;
        dataMin[TRAFFIC_SEN] = 400; dataMax[TRAFFIC_SEN] = 600;
        dataMin[VEHICLE_ID] = 1200; dataMax[VEHICLE_ID] = 1800;
        dataMin[TRAFFIC_LOC] = 1600; dataMax[TRAFFIC_LOC] = 2400;
        dataMin[ACTION_VEH] = 0; dataMax[ACTION_VEH] = 0;
        dataMin[ACTION_TRAF] = 0; dataMax[ACTION_TRAF] = 0;
    }

    static class FogJob {
        int taskType;
        boolean noValidTask = false;
        double noInstructions, dataBytes;

        FogJob(int taskType) {
            this.taskType = taskType;
            rollInstructionsAndData();
        }

        void rollInstructionsAndData() {
            int imin = instrMin[taskType], imax = instrMax[taskType];
            int dmin = dataMin[taskType], dmax = dataMax[taskType];
            noInstructions = (imax > imin) ? (imin + rng.nextInt(imax - imin + 1)) : imin;
            dataBytes = (dmax > dmin) ? (dmin + rng.nextInt(dmax - dmin + 1)) : dmin;
        }

        void advanceStage() {
            if (taskType == CAMERA_SEN) taskType = VEHICLE_ID;
            else if (taskType == TRAFFIC_SEN) taskType = TRAFFIC_LOC;
            else if (taskType == VEHICLE_ID) taskType = ACTION_VEH;
            else if (taskType == TRAFFIC_LOC) taskType = ACTION_TRAF;
            else if (taskType == ACTION_VEH || taskType == ACTION_TRAF) noValidTask = true;
            rollInstructionsAndData();
        }
    }

    public static void main(String[] args) {
        List<Resource> fogResources = new ArrayList<>();
        for (int i = 0; i < NUM_CLUSTERS; i++)
            fogResources.add(new Resource(CPU_SPEED[i], FD_BANDWIDTH[i], FD_RAM[i], SLOT_TIME, BUSY_POWER[i]));
        Resource cloud = new Resource(CLOUD_CPU, CLOUD_BANDWIDTH, CLOUD_RAM, Double.POSITIVE_INFINITY, 0);

        int totalSensors = NUM_CLUSTERS * SENSORS_PER_CLUSTER;
        List<FogJob> jobQ = new ArrayList<>();
        int totalSlots = (int) (MAX_SIMULATION_TIME / SLOT_TIME);

        List<Double> taskTimes = new ArrayList<>();
        int cloudCount = 0, fogCount = 0, totalCompleted = 0;

        for (int slot = 1; slot <= totalSlots; slot++) {
            for (int s = 0; s < totalSensors; s++) {
                int taskType = (s % 2 == 0) ? CAMERA_SEN : TRAFFIC_SEN;
                jobQ.add(new FogJob(taskType));
            }

            if (!jobQ.isEmpty()) {
                List<Job> jbs = new ArrayList<>();
                for (FogJob j : jobQ) jbs.add(new Job(j.noInstructions, j.dataBytes, j));

                GeneticAlgorithm ga = new GeneticAlgorithm(fogResources, jbs, cloud);
                ga.generateOptimalChromosome();
                Chromosome best = ga.returnBestCRM();

                for (int idx = 0; idx < jobQ.size(); idx++) {
                    FogJob job = jobQ.get(idx);
                    Integer FR_ID = best.getRs(idx);
                    double duration;
                    if (FR_ID != null && FR_ID < fogResources.size()) {
                        Resource res = fogResources.get(FR_ID);
                        double swapTime = GACore.computeSwapTime(job.noInstructions, job.dataBytes, res.getRAM());
                        duration = job.noInstructions / res.getCPUSpeed() + job.dataBytes / res.getBANDWIDTH()
                                + GACore.LATENCY_ED_FR + swapTime;
                        fogCount++;
                    } else {
                        double swapTime = GACore.computeSwapTime(job.noInstructions, job.dataBytes, cloud.getRAM());
                        duration = job.noInstructions / cloud.getCPUSpeed() + job.dataBytes / cloud.getBANDWIDTH()
                                + GACore.LATENCY_CS_FR + swapTime;
                        cloudCount++;
                    }
                    taskTimes.add(duration);
                    totalCompleted++;
                    job.advanceStage();
                }

                List<FogJob> remaining = new ArrayList<>();
                for (FogJob j : jobQ) if (!j.noValidTask) remaining.add(j);
                jobQ = remaining;
            }

            if (slot % 500 == 0 || slot == totalSlots) {
                System.out.printf("[SLOT %d/%d] (%.1f%%) tasks so far: %d%n",
                        slot, totalSlots, 100.0 * slot / totalSlots, totalCompleted);
            }
        }

        double sum = 0, min = Double.POSITIVE_INFINITY, max = 0;
        for (double t : taskTimes) { sum += t; min = Math.min(min, t); max = Math.max(max, t); }

        System.out.println();
        System.out.println("Simulation Completed");
        System.out.println("-----------Results (GA, iFogSim-topology-matched)--------------------");
        System.out.println("Total Simulation Time: " + MAX_SIMULATION_TIME);
        System.out.println("Total Number of Tasks: " + totalCompleted);
        System.out.println();
        System.out.println("avg TimeTaken for Tasks: " + (sum / taskTimes.size()));
        System.out.println("min/max TimeTaken for Tasks: " + min + " / " + max);
        System.out.println();
        System.out.println("Fog executed: " + fogCount + "  Cloud executed: " + cloudCount);
        System.out.println("Cloud fraction: " + ((double) cloudCount / totalCompleted));
    }
}
