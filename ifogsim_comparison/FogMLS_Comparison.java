package org.fog.test.perfeval;

import java.util.ArrayList;
import java.util.Calendar;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;

import org.cloudbus.cloudsim.Host;
import org.cloudbus.cloudsim.Log;
import org.cloudbus.cloudsim.Pe;
import org.cloudbus.cloudsim.Storage;
import org.cloudbus.cloudsim.core.CloudSim;
import org.cloudbus.cloudsim.power.PowerHost;
import org.cloudbus.cloudsim.sdn.overbooking.BwProvisionerOverbooking;
import org.cloudbus.cloudsim.sdn.overbooking.PeProvisionerOverbooking;
import org.cloudbus.cloudsim.provisioners.RamProvisionerSimple;
import org.fog.application.AppEdge;
import org.fog.application.AppLoop;
import org.fog.application.Application;
import org.fog.application.selectivity.FractionalSelectivity;
import org.fog.entities.Actuator;
import org.fog.entities.FogBroker;
import org.fog.entities.FogDevice;
import org.fog.entities.FogDeviceCharacteristics;
import org.fog.entities.Sensor;
import org.fog.entities.Tuple;
import org.fog.placement.Controller;
import org.fog.placement.ModuleMapping;
import org.fog.placement.ModulePlacementMapping;
import org.fog.policy.AppModuleAllocationPolicy;
import org.fog.scheduler.StreamOperatorScheduler;
import org.fog.utils.Config;
import org.fog.utils.FogLinearPowerModel;
import org.fog.utils.FogUtils;
import org.fog.utils.TimeKeeper;
import org.fog.utils.distribution.DeterministicDistribution;

/**
 * FogMLS_Comparison.java
 *
 * Round Robin comparison, routed through iFogSim's real, full
 * CloudSim event-driven engine (unlike the GA comparison, which uses
 * a standalone batch loop). Uses CapacityAwareSensor and
 * CapacityAwareFogDevice to replicate FogMLS's real capacity-aware
 * Round Robin scheduler exactly.
 */
public class FogMLS_Comparison {
	static List<FogDevice> fogDevices = new ArrayList<FogDevice>();
	static List<Sensor> sensors = new ArrayList<Sensor>();
	static List<Actuator> actuators = new ArrayList<Actuator>();

	static int NUM_CLUSTERS = 3;
	static int SENSORS_PER_CLUSTER = 9;
	static double SLOT_TIME = 35; // matches FogMLS at 9 sensors/cluster

	static long[] FOG_CPU = {40, 80, 160};
	static long[] FOG_BW = {400, 800, 1600};
	static long[] FOG_RAM_VALUES = {2000, 4000, 8000};
	static double[] FOG_BUSY_POWER = {105.0, 110.0, 115.339};
	static double[] FOG_IDLE_POWER = {80.0, 82.0, 85.0};

	static long CLOUD_CPU = 48000;
	static long CLOUD_BW = 1600;

	public static void main(String[] args) {
		Log.printLine("Starting FogMLS_Comparison in iFogSim...");

		try {
			Log.disable();
			int num_user = 1;
			Calendar calendar = Calendar.getInstance();
			boolean trace_flag = false;

			CloudSim.init(num_user, calendar, trace_flag);

			Config.MAX_SIMULATION_TIME = 175000;

			String appId = "fogmls_app";

			FogBroker broker = new FogBroker("broker");

			Application application = createApplication(appId, broker.getId());
			application.setUserId(broker.getId());

			createFogDevices(broker.getId(), appId);

			ModuleMapping moduleMapping = ModuleMapping.createModuleMapping();
			for (FogDevice dev : fogDevices) {
				if (dev.getName().startsWith("fog-") || dev.getName().equals("cloud")) {
					moduleMapping.addModuleToDevice("worker", dev.getName());
				}
			}

			Controller controller = new Controller("master-controller", fogDevices, sensors, actuators);

			controller.submitApplication(application, 0,
					new ModulePlacementMapping(fogDevices, application, moduleMapping));

			TimeKeeper.getInstance().setSimulationStartTime(Calendar.getInstance().getTimeInMillis());

			CloudSim.startSimulation();
			// NOTE: execution never reaches here -- Controller's
			// STOP_SIMULATION handler calls System.exit(0) internally
			// after printing results (including our summary, patched
			// in to call printCleanSummaryStatic() from there).
			CloudSim.stopSimulation();

			Log.printLine("FogMLS_Comparison finished!");
		} catch (Exception e) {
			e.printStackTrace();
			Log.printLine("Unwanted errors happen");
		}
	}

	public static void printCleanSummaryStatic() {
		int totalTasks = 0;
		double sum = 0.0, min = Double.POSITIVE_INFINITY, max = 0.0;
		for (String type : CapacityAwareFogDevice.ourCounts.keySet()) {
			int count = CapacityAwareFogDevice.ourCounts.get(type);
			double total = CapacityAwareFogDevice.ourTotalTime.get(type);
			totalTasks += count;
			sum += total;
		}
		double overallAvg = totalTasks > 0 ? sum / totalTasks : 0.0;

		System.out.println("");
		System.out.println("Simulation Completed");
		System.out.println("-----------Results (RR, iFogSim-topology-matched)--------------------");
		System.out.println("Total Simulation Time:" + Config.MAX_SIMULATION_TIME);
		System.out.println("Total Number of Tasks:" + totalTasks);
		System.out.println("");
		System.out.println("avg TimeTaken for Tasks:" + overallAvg);
		System.out.println("");
		System.out.println("Per Task Type:");
		for (String type : CapacityAwareFogDevice.ourCounts.keySet()) {
			int count = CapacityAwareFogDevice.ourCounts.get(type);
			double avg = CapacityAwareFogDevice.ourTotalTime.get(type) / count;
			System.out.println("  " + type + ": " + count + " tasks, avg time " + avg);
		}
		System.out.println("");
		System.out.println("Sensor transmit() call count: " + CapacityAwareSensor.transmitCallCount);
		System.out.println("Fog assigned count: " + CapacityAwareSensor.fogAssignedCount);
		System.out.println("Cloud assigned count: " + CapacityAwareSensor.cloudAssignedCount);
		double cloudFrac = (double) CapacityAwareSensor.cloudAssignedCount /
				(CapacityAwareSensor.fogAssignedCount + CapacityAwareSensor.cloudAssignedCount);
		System.out.println("Cloud fraction: " + cloudFrac);
	}

	private static void createFogDevices(int userId, String appId) {
		FogDevice cloud = createFogDevice("cloud", CLOUD_CPU, 1000000, CLOUD_BW, CLOUD_BW, 0, 0.01, 16*103, 16*83.25);
		cloud.setParentId(-1);
		fogDevices.add(cloud);

		int[] fogNodeIds = new int[NUM_CLUSTERS];
		double[] fogCpuSpeeds = new double[NUM_CLUSTERS];
		FogDevice[] fogNodesArr = new FogDevice[NUM_CLUSTERS];

		for (int i = 0; i < NUM_CLUSTERS; i++) {
			FogDevice fogNode = createFogDevice("fog-" + i, FOG_CPU[i], (int) FOG_RAM_VALUES[i], FOG_BW[i], FOG_BW[i], 1,
					0.0, FOG_BUSY_POWER[i], FOG_IDLE_POWER[i]);
			fogNode.setParentId(cloud.getId());
			fogNode.setUplinkLatency(50);
			fogDevices.add(fogNode);
			fogNodesArr[i] = fogNode;
			fogNodeIds[i] = fogNode.getId();
			fogCpuSpeeds[i] = FOG_CPU[i];
		}

		CapacityAwareSensor.FOG_NODE_IDS = fogNodeIds;
		CapacityAwareSensor.FOG_CPU_SPEED = fogCpuSpeeds;
		double[] fogBandwidths = new double[NUM_CLUSTERS];
		double[] fogRams = new double[NUM_CLUSTERS];
		for (int i = 0; i < NUM_CLUSTERS; i++) {
			fogBandwidths[i] = FOG_BW[i];
			fogRams[i] = FOG_RAM_VALUES[i];
		}
		CapacityAwareSensor.FOG_BANDWIDTH = fogBandwidths;
		CapacityAwareSensor.FOG_RAM = fogRams;
		CapacityAwareSensor.CLOUD_BANDWIDTH = CLOUD_BW;
		CapacityAwareSensor.CLOUD_DEVICE_ID = cloud.getId();
		CapacityAwareSensor.SLOT_TIME = SLOT_TIME;

		for (int i = 0; i < NUM_CLUSTERS; i++) {
			FogDevice fogNode = fogNodesArr[i];
			for (int j = 0; j < SENSORS_PER_CLUSTER; j++) {
				String sensorId = i + "-" + j;
				boolean isCam = (j % 2 == 0);

				double instr = isCam ? 11 : 13;
				double dataBytes = isCam ? 1000 : 500;
				CapacityAwareSensor sensor = new CapacityAwareSensor("s-" + sensorId,
						isCam ? "CAM" : "TRAF", userId, appId,
						new DeterministicDistribution(SLOT_TIME), instr, dataBytes);
				sensors.add(sensor);
				sensor.setGatewayDeviceId(fogNode.getId());
				sensor.setLatency(1.0);

				Actuator actuator = new Actuator("a-" + sensorId, userId, appId, "ACTUATOR");
				actuators.add(actuator);
				actuator.setGatewayDeviceId(fogNode.getId());
				actuator.setLatency(1.0);
			}
		}
	}

	private static FogDevice createFogDevice(String nodeName, long mips,
			int ram, long upBw, long downBw, int level, double ratePerMips, double busyPower, double idlePower) {

		List<Pe> peList = new ArrayList<Pe>();
		peList.add(new Pe(0, new PeProvisionerOverbooking(mips)));

		int hostId = FogUtils.generateEntityId();
		long storage = 1000000;
		int bw = 10000;

		PowerHost host = new PowerHost(
				hostId,
				new RamProvisionerSimple(ram),
				new BwProvisionerOverbooking(bw),
				storage,
				peList,
				new StreamOperatorScheduler(peList),
				new FogLinearPowerModel(busyPower, idlePower)
		);

		List<Host> hostList = new ArrayList<Host>();
		hostList.add(host);

		String arch = "x86";
		String os = "Linux";
		String vmm = "Xen";
		double time_zone = 10.0;
		double cost = 3.0;
		double costPerMem = 0.05;
		double costPerStorage = 0.001;
		double costPerBw = 0.0;
		LinkedList<Storage> storageList = new LinkedList<Storage>();

		FogDeviceCharacteristics characteristics = new FogDeviceCharacteristics(
				arch, os, vmm, host, time_zone, cost, costPerMem,
				costPerStorage, costPerBw);

		FogDevice fogdevice = null;
		try {
			fogdevice = new CapacityAwareFogDevice(nodeName, characteristics,
					new AppModuleAllocationPolicy(hostList), storageList, 10, upBw, downBw, 0, ratePerMips);
		} catch (Exception e) {
			e.printStackTrace();
		}

		fogdevice.setLevel(level);
		return fogdevice;
	}

	@SuppressWarnings({"serial"})
	private static Application createApplication(String appId, int userId) {

		Application application = Application.createApplication(appId, userId);

		application.addAppModule("worker", 10);

		application.addAppEdge("CAM", "worker", 11, 1000, "CAM", Tuple.UP, AppEdge.SENSOR);
		application.addAppEdge("TRAF", "worker", 13, 500, "TRAF", Tuple.UP, AppEdge.SENSOR);

		application.addAppEdge("worker", "worker", 120, 1500, "VEHID", Tuple.UP, AppEdge.MODULE);
		application.addAppEdge("worker", "worker", 130, 2000, "TRAFLOC", Tuple.UP, AppEdge.MODULE);

		application.addAppEdge("worker", "worker", 0, 0, "ACTVEH", Tuple.UP, AppEdge.MODULE);
		application.addAppEdge("worker", "worker", 0, 0, "ACTTRAF", Tuple.UP, AppEdge.MODULE);

		application.addAppEdge("worker", "ACTUATOR", 1, 10, "DONE", Tuple.DOWN, AppEdge.ACTUATOR);

		application.addTupleMapping("worker", "CAM", "VEHID", new FractionalSelectivity(1.0));
		application.addTupleMapping("worker", "TRAF", "TRAFLOC", new FractionalSelectivity(1.0));
		application.addTupleMapping("worker", "VEHID", "ACTVEH", new FractionalSelectivity(1.0));
		application.addTupleMapping("worker", "TRAFLOC", "ACTTRAF", new FractionalSelectivity(1.0));
		application.addTupleMapping("worker", "ACTVEH", "DONE", new FractionalSelectivity(1.0));
		application.addTupleMapping("worker", "ACTTRAF", "DONE", new FractionalSelectivity(1.0));

		final AppLoop loop1 = new AppLoop(new ArrayList<String>(){{add("CAM");add("worker");add("ACTUATOR");}});
		List<AppLoop> loops = new ArrayList<AppLoop>(){{add(loop1);}};
		application.setLoops(loops);

		return application;
	}
}
