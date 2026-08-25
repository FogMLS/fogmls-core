package org.fog.test.perfeval;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

import org.cloudbus.cloudsim.Storage;
import org.cloudbus.cloudsim.UtilizationModelFull;
import org.cloudbus.cloudsim.core.CloudSim;
import org.cloudbus.cloudsim.core.CloudSimTags;
import org.cloudbus.cloudsim.core.SimEvent;
import org.fog.application.Application;
import org.fog.application.AppLoop;
import org.fog.entities.FogDevice;
import org.fog.entities.FogDeviceCharacteristics;
import org.fog.entities.Tuple;
import org.fog.policy.AppModuleAllocationPolicy;
import org.fog.utils.FogEvents;
import org.fog.utils.FogUtils;
import org.fog.utils.TimeKeeper;

/**
 * CapacityAwareFogDevice
 *
 * Bypasses iFogSim's built-in Cloudlet/VM scheduling (fractional
 * time-sharing among concurrently arriving tuples -- confirmed to
 * compound badly under FogMLS's real bursty, slot-aligned arrival
 * pattern) and instead tracks task completion SEQUENTIALLY, directly,
 * matching FogMLS's real per-device queue model.
 *
 * FIXED during validation: this class previously only intercepted
 * the "worker"->"processor" transition, matching a SIMPLIFIED
 * 2-stage application that did not reflect FogMLS's real 3-stage
 * chain. Now intercepts ALL worker->worker self-loop stage
 * transitions (CAM->VEHID, TRAF->TRAFLOC, VEHID->ACTVEH,
 * TRAFLOC->ACTTRAF), and constructs NEW tuples with randomized
 * instructions/data per FogMLS's real ranges (since
 * Cloudlet.cloudletFileSize is private final and cannot be modified
 * on iFogSim's own auto-generated resultant tuples).
 */
public class CapacityAwareFogDevice extends FogDevice {

	private enum CustomTag implements CloudSimTags {
		DEFERRED_STAGE2_DECISION,
		SEQUENTIAL_TUPLE_COMPLETE
	}

	public static int transmitCallCount = 0;
	public static int executeTupleCallCount = 0;
	public static int completionCallCount = 0;
	public static Map<String, Double> ourTotalTime = new HashMap<String, Double>();
	public static Map<String, Integer> ourCounts = new HashMap<String, Integer>();

	private static Random rng = new Random(42);

	// Real FogMLS ranges per stage transition type, keyed by the
	// INCOMING tuple type (what triggers the next stage).
	private static final Map<String, int[]> NEXT_STAGE_INSTR = new HashMap<String, int[]>();
	private static final Map<String, int[]> NEXT_STAGE_BYTES = new HashMap<String, int[]>();
	private static final Map<String, String> NEXT_STAGE_NAME = new HashMap<String, String>();
	static {
		NEXT_STAGE_NAME.put("CAM", "VEHID");
		NEXT_STAGE_INSTR.put("CAM", new int[]{100, 140});
		NEXT_STAGE_BYTES.put("CAM", new int[]{1200, 1800});

		NEXT_STAGE_NAME.put("TRAF", "TRAFLOC");
		NEXT_STAGE_INSTR.put("TRAF", new int[]{110, 150});
		NEXT_STAGE_BYTES.put("TRAF", new int[]{1600, 2400});

		NEXT_STAGE_NAME.put("VEHID", "ACTVEH");
		NEXT_STAGE_INSTR.put("VEHID", new int[]{0, 0});
		NEXT_STAGE_BYTES.put("VEHID", new int[]{0, 0});

		NEXT_STAGE_NAME.put("TRAFLOC", "ACTTRAF");
		NEXT_STAGE_INSTR.put("TRAFLOC", new int[]{0, 0});
		NEXT_STAGE_BYTES.put("TRAFLOC", new int[]{0, 0});
	}

	public CapacityAwareFogDevice(String name, FogDeviceCharacteristics characteristics,
			AppModuleAllocationPolicy vmAllocationPolicy, List<Storage> storageList,
			double schedulingInterval, double uplinkBandwidth, double downlinkBandwidth,
			double uplinkLatency, double ratePerMips) throws Exception {
		super(name, characteristics, vmAllocationPolicy, storageList, schedulingInterval,
				uplinkBandwidth, downlinkBandwidth, uplinkLatency, ratePerMips);
	}

	@Override
	protected void executeTuple(SimEvent ev, String moduleName) {
		executeTupleCallCount++;
		try {
			Tuple tuple = (Tuple) ev.getData();

			double mips = getHost().getTotalMips();
			double ram = getHost().getRam();

			double instructions = tuple.getCloudletLength();
			double dataBytes = tuple.getCloudletFileSize();
			double swapRatio = (instructions + dataBytes) / ram;
			double swapTime = (swapRatio > 1) ? (FogMLSConfig.RAM_SWAP_UNIT * swapRatio) : FogMLSConfig.RAM_SWAP_UNIT;

			// Shared BUDGET model, not a sequential queue: each task
			// gets its own independent duration once it has already
			// passed the capacity check at the selector.
			double execDuration = instructions / mips + swapTime;

			double now = CloudSim.clock();
			double completionTime = now + execDuration;

			TimeKeeper.getInstance().tupleStartedExecution(tuple);

			send(getId(), completionTime - now, CustomTag.SEQUENTIAL_TUPLE_COMPLETE, tuple);
		} catch (Exception e) {
			System.out.println("EXCEPTION in executeTuple: " + e);
			e.printStackTrace();
		}
	}

	private static long randRange(int min, int max) {
		if (max <= min) return min;
		return min + rng.nextInt(max - min + 1);
	}

	private void updateTimingsOnSendingPublic(Tuple resTuple) {
		String srcModule = resTuple.getSrcModuleName();
		String destModule = resTuple.getDestModuleName();
		Application app = getApplicationMap().get(resTuple.getAppId());
		for (AppLoop loop : app.getLoops()) {
			if (loop.hasEdge(srcModule, destModule) && loop.isStartModule(srcModule)) {
				int tupleId = TimeKeeper.getInstance().getUniqueId();
				resTuple.setActualTupleId(tupleId);
				if (!TimeKeeper.getInstance().getLoopIdToTupleIds().containsKey(loop.getLoopId()))
					TimeKeeper.getInstance().getLoopIdToTupleIds().put(loop.getLoopId(), new ArrayList<Integer>());
				TimeKeeper.getInstance().getLoopIdToTupleIds().get(loop.getLoopId()).add(tupleId);
				TimeKeeper.getInstance().getEmitTimes().put(tupleId, CloudSim.clock());
			}
		}
	}

	private void handleSequentialCompletion(Tuple tuple) {
		completionCallCount++;
		try {
			Double startTime = CapacityAwareSensor.emitTimeByCloudletId.remove(tuple.getCloudletId());
			double duration = (startTime != null) ? (CloudSim.clock() - startTime) : -1.0;

			String type = tuple.getTupleType();
			ourTotalTime.put(type, ourTotalTime.getOrDefault(type, 0.0) + duration);
			ourCounts.put(type, ourCounts.getOrDefault(type, 0) + 1);

			TimeKeeper.getInstance().tupleEndedExecution(tuple);

			if (NEXT_STAGE_NAME.containsKey(type)) {
				String nextType = NEXT_STAGE_NAME.get(type);
				int[] instrRange = NEXT_STAGE_INSTR.get(type);
				int[] bytesRange = NEXT_STAGE_BYTES.get(type);
				long nextInstr = randRange(instrRange[0], instrRange[1]);
				long nextBytes = randRange(bytesRange[0], bytesRange[1]);

				Tuple next = new Tuple(tuple.getAppId(), FogUtils.generateTupleId(), Tuple.UP,
						nextInstr, 1, nextBytes, 0,
						new UtilizationModelFull(), new UtilizationModelFull(), new UtilizationModelFull());
				next.setUserId(tuple.getUserId());
				next.setTupleType(nextType);
				next.setDestModuleName("worker");
				next.setSrcModuleName("worker");
				next.setModuleCopyMap(new HashMap<String, Integer>(tuple.getModuleCopyMap()));

				updateTimingsOnSendingPublic(next);
				sendToSelf(next);
			} else if (type.equals("ACTVEH") || type.equals("ACTTRAF")) {
				Application application = getApplicationMap().get(tuple.getAppId());
				List<Tuple> resultantTuples = application.getResultantTuples(
						tuple.getDestModuleName(), tuple, getId(), tuple.getVmId());
				for (Tuple resTuple : resultantTuples) {
					updateTimingsOnSendingPublic(resTuple);
					sendToSelf(resTuple);
				}
			}
		} catch (Exception e) {
			System.out.println("EXCEPTION in handleSequentialCompletion: " + e);
			e.printStackTrace();
		}
	}

	@Override
	protected void sendToSelf(Tuple tuple) {
		if ("worker".equals(tuple.getSrcModuleName()) && "worker".equals(tuple.getDestModuleName())
				&& CapacityAwareSensor.FOG_NODE_IDS != null) {
			double slotTime = CapacityAwareSensor.SLOT_TIME;
			double now = CloudSim.clock();
			double wait = slotTime - (now % slotTime);
			if (wait <= 0) wait = slotTime;

			send(getId(), wait, CustomTag.DEFERRED_STAGE2_DECISION, tuple);
			return;
		}
		super.sendToSelf(tuple);
	}

	@Override
	public void processEvent(SimEvent ev) {
		if (ev.getTag() == CustomTag.DEFERRED_STAGE2_DECISION) {
			Tuple tuple = (Tuple) ev.getData();

			CapacityAwareSensor.emitTimeByCloudletId.put(tuple.getCloudletId(), CloudSim.clock());

			int targetDevice = CapacityAwareSensor.chooseDevice(tuple.getCloudletLength(), tuple.getCloudletFileSize());
			double bandwidth = CapacityAwareSensor.bandwidthOf(targetDevice);
			double transmissionDelay = tuple.getCloudletFileSize() / bandwidth + CapacityAwareSensor.propagationDelayFor(targetDevice);
			send(targetDevice, transmissionDelay, FogEvents.TUPLE_ARRIVAL, tuple);
			return;
		}
		if (ev.getTag() == CustomTag.SEQUENTIAL_TUPLE_COMPLETE) {
			Tuple tuple = (Tuple) ev.getData();
			handleSequentialCompletion(tuple);
			return;
		}
		super.processEvent(ev);
	}
}
