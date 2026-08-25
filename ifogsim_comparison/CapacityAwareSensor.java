package org.fog.test.perfeval;

import java.util.HashMap;
import java.util.Map;
import java.util.Random;

import org.cloudbus.cloudsim.UtilizationModelFull;
import org.cloudbus.cloudsim.core.CloudSim;
import org.fog.application.AppEdge;
import org.fog.entities.Sensor;
import org.fog.entities.Tuple;
import org.fog.utils.FogEvents;
import org.fog.utils.FogUtils;
import org.fog.utils.distribution.Distribution;

/**
 * CapacityAwareSensor
 *
 * Replicates FogMLS's real capacity-aware Round Robin scheduler
 * (IoTsimulation.py case 4). Holds the SHARED scheduling state
 * (rrIndex, per-slot busy-time budget) used by BOTH this class
 * (Stage 1) AND CapacityAwareFogDevice (Stage 2), matching FogMLS's
 * real design where ONE shared rr_index and ONE shared per-node
 * capacity budget governs ALL stages in a slot.
 *
 * IMPORTANT FIX: the base Sensor.transmit() only adds a fixed
 * propagation delay (getLatency()) and completely omits bandwidth-
 * based transmission delay (data_bytes / bandwidth). FogMLS's real
 * formula is data_bytes/bandwidth + propagation_delay. This class
 * does NOT call super.transmit() -- it replicates the tuple-creation
 * logic directly, adding the missing bandwidth term explicitly.
 */
public class CapacityAwareSensor extends Sensor {

	public static int[] FOG_NODE_IDS;
	public static int CLOUD_DEVICE_ID;
	public static double[] FOG_CPU_SPEED;
	public static double[] FOG_BANDWIDTH;
	public static double[] FOG_RAM;
	public static double CLOUD_BANDWIDTH;
	public static double SLOT_TIME = 35;

	// SHARED state -- used by CapacityAwareFogDevice too.
	private static int rrIndex = 0;
	private static double currentSlotIndex = -1;
	private static Map<Integer, Double> slotBusyTime = new HashMap<Integer, Double>();

	public static int transmitCallCount = 0;
	public static int fogAssignedCount = 0;
	public static int cloudAssignedCount = 0;

	// Shared emit-time tracking: records when a task was first
	// dispatched (Stage 1 transmit, or Stage 2/3 sendToSelf), NOT
	// when it arrives at a device. Used to compute TRUE end-to-end
	// latency (transmission + compute + swap), matching FogMLS's
	// real per-task total time.
	public static Map<Integer, Double> emitTimeByCloudletId = new HashMap<Integer, Double>();

	private String taskType;
	private double taskInstructions;
	private double taskDataBytes;
	private static Random rng = new Random(42);

	public CapacityAwareSensor(String name, String tupleType, int userId, String appId,
			Distribution transmitDistribution, double instructions, double dataBytes) {
		super(name, tupleType, userId, appId, transmitDistribution);
		this.taskType = tupleType;
		this.taskInstructions = instructions;
		this.taskDataBytes = dataBytes;
	}

	private static void maybeResetSlot(double now) {
		double slotIdx = Math.floor(now / SLOT_TIME);
		if (slotIdx != currentSlotIndex) {
			currentSlotIndex = slotIdx;
			for (int fid : FOG_NODE_IDS) slotBusyTime.put(fid, 0.0);
			rrIndex = 0;
		}
	}

	public static double propagationDelayFor(int deviceId) {
		return (deviceId == CLOUD_DEVICE_ID) ? FogMLSConfig.LATENCY_CS_FR : FogMLSConfig.LATENCY_ED_FR;
	}

	public static double bandwidthOf(int deviceId) {
		if (deviceId == CLOUD_DEVICE_ID) return CLOUD_BANDWIDTH;
		for (int i = 0; i < FOG_NODE_IDS.length; i++) {
			if (FOG_NODE_IDS[i] == deviceId) return FOG_BANDWIDTH[i];
		}
		return 1.0;
	}

	private static double computeSwapTime(double noInstructions, double dataBytes, double ram) {
		double swapRamNo = (noInstructions + dataBytes) / ram;
		if (swapRamNo > 1) return swapRamNo; // RAM_SWAP_UNIT = 1
		return 1.0;
	}

	/**
	 * Real FogMLS Round Robin (IoTsimulation.py case 4): rotate
	 * through fog nodes starting from the shared rrIndex, check
	 * whether the candidate has enough remaining SLOT_TIME budget
	 * this slot, escalate to cloud if none do.
	 */
	public static synchronized int chooseDevice(double instructions, double dataBytes) {
		double now = CloudSim.clock();
		maybeResetSlot(now);

		for (int attempt = 0; attempt < FOG_NODE_IDS.length; attempt++) {
			int idx = (rrIndex + attempt) % FOG_NODE_IDS.length;
			int fogId = FOG_NODE_IDS[idx];

			double swapTime = computeSwapTime(instructions, dataBytes, FOG_RAM[idx]);
			double execDuration = instructions / FOG_CPU_SPEED[idx] + swapTime;

			double busy = slotBusyTime.getOrDefault(fogId, 0.0);
			if (busy + execDuration <= SLOT_TIME) {
				slotBusyTime.put(fogId, busy + execDuration);
				rrIndex = (idx + 1) % FOG_NODE_IDS.length;
				fogAssignedCount++;
				return fogId;
			}
		}

		rrIndex = (rrIndex + 1) % FOG_NODE_IDS.length;
		cloudAssignedCount++;
		return CLOUD_DEVICE_ID;
	}

	@Override
	public void transmit() {
		transmitCallCount++;

		int chosenDevice = chooseDevice(taskInstructions, taskDataBytes);
		setGatewayDeviceId(chosenDevice);

		AppEdge edge = null;
		for (AppEdge e : getApp().getEdges()) {
			if (e.getSource().equals(getTupleType())) edge = e;
		}

		Tuple tuple = new Tuple(getAppId(), FogUtils.generateTupleId(), Tuple.UP,
				(long) taskInstructions, 1, (long) taskDataBytes, 0,
				new UtilizationModelFull(), new UtilizationModelFull(), new UtilizationModelFull());
		tuple.setUserId(getUserId());
		tuple.setTupleType(getTupleType());
		tuple.setDestModuleName(edge.getDestination());
		tuple.setSrcModuleName(getSensorName());
		tuple.setDestinationDeviceId(chosenDevice);

		emitTimeByCloudletId.put(tuple.getCloudletId(), CloudSim.clock());

		double bandwidth = bandwidthOf(chosenDevice);
		double transmissionDelay = taskDataBytes / bandwidth + propagationDelayFor(chosenDevice);

		send(chosenDevice, transmissionDelay, FogEvents.TUPLE_ARRIVAL, tuple);
	}
}
