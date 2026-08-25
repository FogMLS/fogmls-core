package org.fog.test.perfeval;

/**
 * FogMLSConfig
 *
 * SINGLE SOURCE OF TRUTH, matching FogMLS's real config.py structure
 * completely -- every value that could need changing, or that more
 * than one class needs to agree on, lives here.
 */
public class FogMLSConfig {

	// ============================================================
	// RESOURCE MANAGEMENT OPTIONS (matches FogMLS's config.py exactly)
	// ============================================================
	public static final int DEFAULT_RM = 0;
	public static final int GA_RM = 1;
	public static final int RR_RM = 4;

	// Change THIS ONE line to switch scheduler
	public static int RM_TYPE = RR_RM;

	// ============================================================
	// CHANGE THESE VALUES TO CHANGE THE WORKLOAD
	// ============================================================
	public static int SENSORS_PER_CLUSTER = 9;
	public static long RANDOM_SEED = 42;

	// ============================================================
	// SIMULATION DURATION (matches FogMLS's config.py exactly)
	// ============================================================
	public static int MAX_SIMULATION_TIME = 175000;

	// ============================================================
	// FITNESS PARAMETERS (matches FogMLS's config.py exactly)
	// ============================================================
	public static double ALPHA = 0.5;
	public static double BETA = 0.5;
	public static double GAMMA = 0.0;

	// ============================================================
	// GA PARAMETERS (matches FogMLS's config.py exactly)
	// ============================================================
	public static int N_CHROMOSOMES = 8;
	public static int MAX_ITERATIONS = 10;
	public static int TOURNAMENT_SIZE = 3;

	// ============================================================
	// RAM SWAP (matches FogMLS's config.py exactly)
	// ============================================================
	public static double RAM_SWAP_UNIT = 1.0;

	// ============================================================
	// FOG / CLOUD HARDWARE (matches FogMLS's real config.py)
	// ============================================================
	public static int NUM_FOG_NODES = 3;
	public static double[] CPU_SPEED = {40, 80, 160};
	public static double[] FD_BANDWIDTH = {400, 800, 1600};
	public static double[] FD_RAM = {2000, 4000, 8000};
	public static double[] BUSY_POWER = {105.0, 110.0, 115.339};
	public static double[] IDLE_POWER = {80.0, 82.0, 85.0};

	public static double CLOUD_CPU_SPEED = 48000;
	public static double CLOUD_RAM = 1000000;
	public static double CLOUD_BANDWIDTH = 1600;
	public static double CLOUD_COST_UNIT_TIME = 0.01;

	// ============================================================
	// NETWORK LATENCY (matches FogMLS's config.py exactly)
	// ============================================================
	public static double LATENCY_ED_FR = 1.0;   // fog propagation delay
	public static double LATENCY_CS_FR = 50.0;  // cloud propagation delay

	// ============================================================
	// TASK INSTRUCTION / DATA SIZE RANGES (matches FogMLS's real
	// config.py exactly -- keyed by task type name)
	// ============================================================
	public static int[] RANGE_CAM_INSTR = {8, 14};
	public static int[] RANGE_TRAF_INSTR = {10, 16};
	public static int[] RANGE_VEHID_INSTR = {100, 140};
	public static int[] RANGE_TRAFLOC_INSTR = {110, 150};
	public static int[] RANGE_ACTVEH_INSTR = {0, 0};
	public static int[] RANGE_ACTTRAF_INSTR = {0, 0};

	public static int[] RANGE_CAM_BYTES = {800, 1200};
	public static int[] RANGE_TRAF_BYTES = {400, 600};
	public static int[] RANGE_VEHID_BYTES = {1200, 1800};
	public static int[] RANGE_TRAFLOC_BYTES = {1600, 2400};
	public static int[] RANGE_ACTVEH_BYTES = {0, 0};
	public static int[] RANGE_ACTTRAF_BYTES = {0, 0};

	// ============================================================
	// AUTO-CALCULATED SLOT_TIME -- copied exactly from FogMLS's
	// config.py formula. Recompute after changing SENSORS_PER_CLUSTER.
	// ============================================================
	public static double SLOT_TIME;
	static {
		int maxInstrAnyTask = Math.max(
			Math.max(RANGE_CAM_INSTR[1], RANGE_TRAF_INSTR[1]),
			Math.max(RANGE_VEHID_INSTR[1], RANGE_TRAFLOC_INSTR[1])
		);
		double minCpuSpeed = Math.min(CPU_SPEED[0], Math.min(CPU_SPEED[1], CPU_SPEED[2]));
		double slotTime0 = -2.5;
		double intervalGap = maxInstrAnyTask / minCpuSpeed;
		SLOT_TIME = intervalGap * SENSORS_PER_CLUSTER + intervalGap + slotTime0;
	}
}
