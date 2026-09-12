# -*- coding: utf-8 -*-
import os
import time
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_XLA_FLAGS"] = "--tf_xla_enable_xla_devices=false"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import random
import numpy as np
from config import RANDOM_SEED
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

from config import (
    AI_RM, ALPHA, BETA, DEFAULT_RM, DEVICE_TO_CLASS, DAGGER_RM, RF_RM, RF_BATCH_RM, RF_SMART_RM,
    FEATURES_PER_JOB, GA_RM,
    GAMMA, JOB_ATTRIBUTES, MAX_JOBS, MAX_RS,
    MAX_SIMULATION_TIME, NUM_FOG_NODES,
    RM_TYPE, RR_RM, SENSORS_PER_CLUSTER, SLOT_TIME,
    SLOT_LEVEL_FEATURES,
    CPU_SPEED1, CPU_SPEED2, CPU_SPEED3,
    FD_RAM1, FD_RAM2, FD_RAM3,
    FD_BANDWIDTH1, FD_BANDWIDTH2, FD_BANDWIDTH3,
    LATENCY_ED_FR, LATENCY_CS_FR
)
from devices import (
    Actuator, Actuators, ClusterRs, jobQ, Sensor,
    FogDevice, Sensors, cs, Job, logentry,
    jobsTimetracker, tasksTimetracker, resource_mg_string,
    compute_swap_time, compute_exec_duration
)
from GeneticAlg import Chromosome, GeneticAlgorithm
from ParticleSwarmAlg import ParticleSwarmOptimization

# Maximum task positions per slot
MaxTasksInSlot = MAX_JOBS
# Number of fog devices
numOfFDs = MAX_RS - 2
# Number of sensors per cluster
numSens = SENSORS_PER_CLUSTER

# Reset fog node counter for clean initialization.
# Prevents wrong hardware specs if running multiple times
# in the same Python session.
FogDevice.reset_count()

# Create fog devices and connect sensors and actuators
for i in range(numOfFDs):
    fr = FogDevice("ClusterFR:")
    ClusterRs.append(fr)

for fr in ClusterRs:
    for j in range(numSens):
        s_type = 1 if j % 2 == 0 else 2
        ed = Sensor("Sensor_" + str(fr.id) + "_" + str(j), s_type)
        ed.setClusterId(fr.id)
        Sensors.append(ed)
    act = Actuator("Actuator" + str(fr.id))
    act.setClusterId(fr.id)
    Actuators.append(act)

# ================================================================
# EXPLICIT DEVICE-TO-CLASS LABEL MAPPING
# ================================================================
# Populate the mapping AFTER all devices are created.
# This eliminates the fragile dependency on device creation order.
# Class labels: 0=Nothing, 1=Cloud, 2=Fog-1, 3=Fog-2, ...
DEVICE_TO_CLASS[cs.id] = 1  # Cloud is always class 1
for fog_dev in ClusterRs:
    # fog_class_index is 0-based, class labels start at 2
    DEVICE_TO_CLASS[fog_dev.id] = fog_dev.fog_class_index + 2

print("[SIM] Device-to-class mapping:")
print("  Cloud (device " + str(cs.id) + ") -> class 1")
for fog_dev in ClusterRs:
    cls = DEVICE_TO_CLASS[fog_dev.id]
    print("  " + fog_dev.name + " (device " + str(fog_dev.id) +
          ") -> class " + str(cls) + " (Fog-" +
          str(fog_dev.fog_class_index + 1) + ")")

# Validate mapping covers all expected classes
mapped_classes = sorted(DEVICE_TO_CLASS.values())
expected_classes = list(range(1, MAX_RS))
if mapped_classes != expected_classes:
    print("[ERROR] Device-to-class mapping incomplete!")
    print("  Mapped: " + str(mapped_classes))
    print("  Expected: " + str(expected_classes))
    raise RuntimeError("Label mapping does not cover all resource classes.")


def getClusterFromID(id):
    for device in ClusterRs:
        if device.id == id:
            return device
    return None


def computeRewards():
    watch_dog = 0
    for device in ClusterRs:
        watch_dog = watch_dog + device.getIntervelMetricsDetails()
    rewards = [watch_dog for x in range(MAX_JOBS)]
    return rewards


# Print topology
print("Cluster-Fds (FogDevices)")
for device in ClusterRs:
    print("   " + device.name +
          "    CPU Speed(MIPS):" + str(device.cpu_speed) +
          " RAM(MB)" + str(device.ram))
    print("   " + "Cluster BandWidth:" + str(device.bandwidth))
print("Sensor Devices")
for device in Sensors:
    print("   " + device.name +
          ' [Id:' + str(device.id) +
          "] (Connected to ClusterFR:" +
          str(device.connected_cl_id) + ")")
print("Actuator Devices")
for device in Actuators:
    print("   " + device.name +
          ' [Id:' + str(device.id) +
          "]  (Connected to ClusterFR:" +
          str(device.connected_cl_id) + ")")

print('Resource management strategy: ' + resource_mg_string[RM_TYPE])
print('SLOT DURATION: ' + str(SLOT_TIME))
print("Simulation Started")

sl_no = 0
ga_slot_times = []  # timing measurement, safe to ignore/remove after use
sim_time = 0.

# Write dataset header.
# Features: TaskType, TaskCodeSize, TaskDataSize per task position.
# Capacity features are excluded from dataset -- used internally
# by GA only for scheduling decisions.
feature_headers = ','.join([
    'TaskType,TaskCodeSize,TaskDataSize,FogCap1,FogCap2,FogCap3'
] * MaxTasksInSlot)
slot_level_headers = ('Fog1_CPU,Fog1_RAM,Fog1_BW,'
                       'Fog2_CPU,Fog2_RAM,Fog2_BW,'
                       'Fog3_CPU,Fog3_RAM,Fog3_BW,'
                       'Latency_ED,Latency_CS')
feature_headers = feature_headers + ',' + slot_level_headers
label_headers = ','.join(['Resource_Id'] * MaxTasksInSlot)
logentry.writeToDataset(feature_headers + ',' + label_headers + '\n')

# Load AI model only if needed
if RM_TYPE == AI_RM or RM_TYPE == DAGGER_RM:
    from model_arch import build_model
    import joblib
    model, _ = build_model()
    weights_file = 'DL_weights_' + str(MAX_JOBS) + '.weights.h5'
    scaler_file = 'DL_scaler_' + str(MAX_JOBS) + '.joblib'
    model.load_weights(weights_file)
    scaler = joblib.load(scaler_file)
    print('[SIM] AI model loaded from ' + weights_file)
    print('[SIM] Scaler loaded from ' + scaler_file)
    # Compile inference function once using tf.function.
    # model.predict() has per-call overhead for batch setup (~107ms/slot).
    # tf.function traces the computation graph once, then executes
    # directly on GPU with no Python overhead (~1-2ms/slot).
    import tensorflow as tf
    @tf.function(reduce_retracing=True)
    def predict_fn(x):
        return model(x, training=False)
    # Warmup call to trigger tracing before simulation starts
    _dummy = np.zeros((1, MAX_JOBS * FEATURES_PER_JOB), dtype=np.float32)
    _dummy_scaled = scaler.transform(_dummy).astype(np.float32)
    predict_fn(tf.constant(_dummy_scaled))
    print('[SIM] Inference function compiled and warmed up.')
else:
    model = None
    scaler = None

if RM_TYPE == RF_RM or RM_TYPE == RF_BATCH_RM or RM_TYPE == RF_SMART_RM:
    import joblib
    rf_model = joblib.load('random_forest_model.joblib')
    print('[SIM] Random Forest model loaded from random_forest_model.joblib')
else:
    rf_model = None

if RM_TYPE != DEFAULT_RM:
    print('Alpha:' + str(ALPHA) +
          ', Beta:' + str(BETA) +
          ', Gamma:' + str(GAMMA))

# Main simulation loop
while MAX_SIMULATION_TIME > sim_time:

    # Reset fog devices and cloud for new slot
    for device in ClusterRs:
        device.clean(sim_time)
    cs.clear(sim_time)

    sl_no += 1

    # Generate tasks from all sensors
    for device in Sensors:
        job = device.CreateUploadJob(sim_time, sl_no)
        if job is not None:
            jobQ.append(job)

    msg = ("\nSlot No:" + str(sl_no) +
           " Tasks:" + str(len(jobQ)) + "\n")
    logentry.WriteStringToSlotsLog(msg)
    logentry.WriteStringToSlotsLog(
        "-------------------------------------------------------------\n")

    len_ofjobQ = len(jobQ)

    # Collectors for dataset row
    slot_features = []
    slot_labels = []

    if len_ofjobQ > 0:
        match RM_TYPE:

            case 0:
                # Default resource management
                for job in jobQ:
                    clustrFogNode = getClusterFromID(
                        job.getDestinatinFogID())
                    if clustrFogNode is not None:
                        clustrFogNode.ExecutesJob(job, sim_time, sl_no)

            case 6:
                # PSO resource manager with dataset collection.
                # DELIBERATELY IDENTICAL to case 1 (GA) in every
                # respect except the optimizer itself -- same capacity
                # tracking, same feature extraction, same execution
                # logic, same labeling, same row format. This
                # isolation is required for the generality experiment:
                # any difference in downstream ML accuracy between
                # GA-labeled and PSO-labeled datasets must be
                # attributable to the search mechanism alone.
                PSO = ParticleSwarmOptimization(ClusterRs, jobQ)
                PSO.GenerateOptimalChromosome()
                Cr = PSO.returnBESTCRM()

                ga_cap = [dev.get_Ex_capacity() for dev in ClusterRs]

                slot_level_values = [
                    CPU_SPEED1, FD_RAM1, FD_BANDWIDTH1,
                    CPU_SPEED2, FD_RAM2, FD_BANDWIDTH2,
                    CPU_SPEED3, FD_RAM3, FD_BANDWIDTH3,
                    LATENCY_ED_FR, LATENCY_CS_FR
                ]
                slot_level_str = ','.join(str(v) for v in slot_level_values)

                for j in range(min(len(jobQ), MaxTasksInSlot)):
                    slot_features.append(jobQ[j].get_type())
                    slot_features.append(jobQ[j].get_noInstructions())
                    slot_features.append(jobQ[j].get_dataBytes())

                    for cap in ga_cap:
                        slot_features.append(cap)

                    FR_ID = Cr.getRs(j)
                    if FR_ID is not None and FR_ID < len(ClusterRs):
                        swap_time = Cr.GetSwapTime(FR_ID, j)
                        exec_duration = compute_exec_duration(
                            jobQ[j].get_noInstructions(),
                            ClusterRs[FR_ID].get_CPUSpeed(),
                            swap_time)
                        ga_cap[FR_ID] = max(0, ga_cap[FR_ID] - exec_duration)
                        ClusterRs[FR_ID].ExecutesJob(
                            jobQ[j], sim_time, sl_no)
                    elif FR_ID is not None and FR_ID == len(ClusterRs):
                        cs.ExecutesJob(jobQ[j], sim_time, sl_no)
                    else:
                        cs.ExecutesJob(jobQ[j], sim_time, sl_no)

                for j in range(min(len(jobQ), MaxTasksInSlot)):
                    slot_labels.append(jobQ[j].get_ex_class_label())

                actual_written = min(len_ofjobQ, MaxTasksInSlot)
                for r in range(MaxTasksInSlot - actual_written):
                    for _ in range(FEATURES_PER_JOB):
                        slot_features.append(0)
                    slot_labels.append(0)

                row = (','.join(str(val) for val in slot_features) +
                       ',' + slot_level_str + ',' +
                       ','.join(str(val) for val in slot_labels))
                logentry.writeToDataset(row + '\n')

            case 7:
                # Random Forest deployed as a live, real-time scheduler.
                # RF was trained on TASK-LEVEL rows (6 base features +
                # position index within the slot), so -- unlike the DL
                # model -- no padding/reshaping into a slot-level vector
                # is needed. Each task's live capacity is refreshed
                # immediately before its own prediction, matching how
                # the training data's capacity columns were generated
                # (sequential, live, per-task).
                n_jobs_this_slot = min(len(jobQ), MaxTasksInSlot)
                rf_cap = [dev.get_Ex_capacity() for dev in ClusterRs]

                for j in range(n_jobs_this_slot):
                    row_features = [
                        jobQ[j].get_type(),
                        jobQ[j].get_noInstructions(),
                        jobQ[j].get_dataBytes(),
                        rf_cap[0], rf_cap[1], rf_cap[2],
                        j,  # position within slot
                    ]
                    FR_ID = int(rf_model.predict([row_features])[0])

                    if FR_ID == 0:
                        clustrFogNode = getClusterFromID(
                            jobQ[j].getDestinatinFogID())
                        if clustrFogNode is not None:
                            clustrFogNode.ExecutesJob(jobQ[j], sim_time, sl_no)
                    elif FR_ID == 1:
                        cs.ExecutesJob(jobQ[j], sim_time, sl_no)
                    else:
                        fog_idx = FR_ID - 2
                        if 0 <= fog_idx < len(ClusterRs):
                            ClusterRs[fog_idx].ExecutesJob(jobQ[j], sim_time, sl_no)
                            swap_time = compute_swap_time(
                                jobQ[j].get_noInstructions(),
                                jobQ[j].get_dataBytes(),
                                ClusterRs[fog_idx].getRAM())
                            exec_duration = compute_exec_duration(
                                jobQ[j].get_noInstructions(),
                                ClusterRs[fog_idx].get_CPUSpeed(),
                                swap_time)
                            rf_cap[fog_idx] = max(0, rf_cap[fog_idx] - exec_duration)
                        else:
                            cs.ExecutesJob(jobQ[j], sim_time, sl_no)

            case 8:
                # Random Forest, BATCHED inference (all tasks in a slot
                # predicted at once, using STALE start-of-slot capacity
                # for every task). Faster (one predict() call per slot
                # instead of up to 200), but methodologically incorrect
                # -- built specifically to measure the real cost, in
                # scheduling quality, of this shortcut, against RF_RM's
                # correct-but-slower sequential version.
                n_jobs_this_slot = min(len(jobQ), MaxTasksInSlot)
                rf_cap_batch = [dev.get_Ex_capacity() for dev in ClusterRs]

                batch_rows = []
                for j in range(n_jobs_this_slot):
                    batch_rows.append([
                        jobQ[j].get_type(),
                        jobQ[j].get_noInstructions(),
                        jobQ[j].get_dataBytes(),
                        rf_cap_batch[0], rf_cap_batch[1], rf_cap_batch[2],
                        j,
                    ])
                batch_predictions = rf_model.predict(batch_rows)

                for j in range(n_jobs_this_slot):
                    FR_ID = int(batch_predictions[j])

                    if FR_ID == 0:
                        clustrFogNode = getClusterFromID(
                            jobQ[j].getDestinatinFogID())
                        if clustrFogNode is not None:
                            clustrFogNode.ExecutesJob(jobQ[j], sim_time, sl_no)
                    elif FR_ID == 1:
                        cs.ExecutesJob(jobQ[j], sim_time, sl_no)
                    else:
                        fog_idx = FR_ID - 2
                        if 0 <= fog_idx < len(ClusterRs):
                            ClusterRs[fog_idx].ExecutesJob(jobQ[j], sim_time, sl_no)
                        else:
                            cs.ExecutesJob(jobQ[j], sim_time, sl_no)

            case 9:
                # Random Forest, CONFIDENCE-RANKED fallback -- an
                # improvement over case 7 (plain argmax). Instead of
                # committing unconditionally to RF's single top-choice
                # prediction, this gets RF's FULL probability
                # distribution over all classes, sorts candidates by
                # confidence, and tries each in order, actually
                # checking whether it fits the current live capacity
                # before committing -- similar in spirit to how RR
                # tries multiple nodes, but prioritized by RF's own
                # confidence ranking rather than round-robin order.
                # This targets the mechanism identified as RF_RM's
                # weakness: committing to a single, possibly-wrong
                # choice with no ability to reconsider.
                n_jobs_this_slot = min(len(jobQ), MaxTasksInSlot)
                rf_cap = [dev.get_Ex_capacity() for dev in ClusterRs]

                for j in range(n_jobs_this_slot):
                    row_features = [
                        jobQ[j].get_type(),
                        jobQ[j].get_noInstructions(),
                        jobQ[j].get_dataBytes(),
                        rf_cap[0], rf_cap[1], rf_cap[2],
                        j,
                    ]
                    proba = rf_model.predict_proba([row_features])[0]
                    ranked = sorted(
                        zip(rf_model.classes_, proba),
                        key=lambda x: x[1], reverse=True)

                    placed = False
                    for FR_ID, _ in ranked:
                        FR_ID = int(FR_ID)
                        if FR_ID == 0:
                            continue
                        elif FR_ID == 1:
                            cs.ExecutesJob(jobQ[j], sim_time, sl_no)
                            placed = True
                            break
                        else:
                            fog_idx = FR_ID - 2
                            if 0 <= fog_idx < len(ClusterRs):
                                swap_time = compute_swap_time(
                                    jobQ[j].get_noInstructions(),
                                    jobQ[j].get_dataBytes(),
                                    ClusterRs[fog_idx].getRAM())
                                exec_duration = compute_exec_duration(
                                    jobQ[j].get_noInstructions(),
                                    ClusterRs[fog_idx].get_CPUSpeed(),
                                    swap_time)
                                if exec_duration <= rf_cap[fog_idx]:
                                    ClusterRs[fog_idx].ExecutesJob(
                                        jobQ[j], sim_time, sl_no)
                                    rf_cap[fog_idx] = max(
                                        0, rf_cap[fog_idx] - exec_duration)
                                    placed = True
                                    break

                    if not placed:
                        cs.ExecutesJob(jobQ[j], sim_time, sl_no)

            case 1:
                # GA resource manager with dataset collection.
                ga_start_time = time.perf_counter()
                GA = GeneticAlgorithm(ClusterRs, jobQ)
                GA.GenerateOptimalChromosome()
                Cr = GA.returnBESTCRM()
                ga_end_time = time.perf_counter()
                ga_slot_times.append(ga_end_time - ga_start_time)

                # ga_cap tracks remaining TIME capacity per fog node.
                # Uses SLOT_TIME (not cpu_speed * SLOT_TIME) to match
                # the GA's internal capacity model and the simulator.
                ga_cap = [dev.get_Ex_capacity() for dev in ClusterRs]

                # Slot-level static features (fog specs + latency) --
                # same every row, written once per row after all
                # per-job features, before labels.
                slot_level_values = [
                    CPU_SPEED1, FD_RAM1, FD_BANDWIDTH1,
                    CPU_SPEED2, FD_RAM2, FD_BANDWIDTH2,
                    CPU_SPEED3, FD_RAM3, FD_BANDWIDTH3,
                    LATENCY_ED_FR, LATENCY_CS_FR
                ]
                slot_level_str = ','.join(str(v) for v in slot_level_values)

                for j in range(min(len(jobQ), MaxTasksInSlot)):
                    slot_features.append(jobQ[j].get_type())
                    slot_features.append(jobQ[j].get_noInstructions())
                    slot_features.append(jobQ[j].get_dataBytes())

                    # Live remaining capacity per fog node, AT THE
                    # MOMENT this task is being placed (snapshot before
                    # this job's execution updates it below). Mirrors
                    # MCEETO's inclusion of live node utilization.
                    for cap in ga_cap:
                        slot_features.append(cap)

                    # Execute task based on GA assignment
                    FR_ID = Cr.getRs(j)
                    if FR_ID is not None and FR_ID < len(ClusterRs):
                        swap_time = Cr.GetSwapTime(FR_ID, j)
                        exec_duration = compute_exec_duration(
                            jobQ[j].get_noInstructions(),
                            ClusterRs[FR_ID].get_CPUSpeed(),
                            swap_time)
                        ga_cap[FR_ID] = max(0, ga_cap[FR_ID] - exec_duration)
                        ClusterRs[FR_ID].ExecutesJob(
                            jobQ[j], sim_time, sl_no)
                    elif FR_ID is not None and FR_ID == len(ClusterRs):
                        cs.ExecutesJob(jobQ[j], sim_time, sl_no)
                    else:
                        # GA returned None - all fog nodes full and cloud
                        # assignment missing. Fallback to cloud to ensure
                        # every task gets a valid label (never Nothing).
                        cs.ExecutesJob(jobQ[j], sim_time, sl_no)

                # Collect labels using explicit class mapping.
                # get_ex_class_label() uses DEVICE_TO_CLASS instead of
                # relying on device IDs matching class indices.
                for j in range(min(len(jobQ), MaxTasksInSlot)):
                    slot_labels.append(jobQ[j].get_ex_class_label())

                # Pad unused task positions with zeros
                actual_written = min(len_ofjobQ, MaxTasksInSlot)
                for r in range(MaxTasksInSlot - actual_written):
                    for _ in range(FEATURES_PER_JOB):
                        slot_features.append(0)
                    slot_labels.append(0)

                # Write complete dataset row - no trailing comma
                row = (','.join(str(val) for val in slot_features) +
                       ',' + slot_level_str + ',' +
                       ','.join(str(val) for val in slot_labels))
                logentry.writeToDataset(row + '\n')

            case 2:
                # SEQUENTIAL/ITERATIVE inference - matches training
                # exactly. Training labels were generated with LIVE,
                # sequentially-updated capacity per job (job j's
                # FogCap1/2/3 reflects the true remaining capacity AFTER
                # jobs 0..j-1 were placed). The model was never trained
                # to handle a single stale, start-of-slot snapshot fed
                # uniformly to all 200 jobs -- doing so silently fed it
                # out-of-distribution inputs for every job after the
                # first, most severely affecting the smallest-capacity
                # node (Fog-1), which depletes fastest within a slot.
                #
                # Fix: predict and execute ONE job at a time, rebuilding
                # the full 1200-feature vector with CURRENT live
                # capacity before each prediction, using only that
                # job's own output head. Slower (up to MAX_JOBS model
                # calls per slot instead of 1) but matches the training
                # distribution correctly.
                n_jobs_this_slot = min(len(jobQ), MaxTasksInSlot)
                ai_cap = [dev.get_Ex_capacity() for dev in ClusterRs]

                # Static per-job task features never change during the
                # slot -- only the capacity columns need refreshing
                # each iteration.
                base_features = []
                for j in range(n_jobs_this_slot):
                    base_features.append(jobQ[j].get_type())
                    base_features.append(jobQ[j].get_noInstructions())
                    base_features.append(jobQ[j].get_dataBytes())
                    base_features.extend([0.0, 0.0, 0.0])  # cap placeholder
                pad_len = MAX_JOBS * JOB_ATTRIBUTES - len(base_features)
                base_features.extend([0] * pad_len)

                for j in range(n_jobs_this_slot):
                    # Refresh capacity columns for ALL jobs (0..n-1) to
                    # the CURRENT live state before this job's decision
                    # -- matches how training data's FogCap columns
                    # reflect live state at each job's own position.
                    for k in range(n_jobs_this_slot):
                        base_idx = k * JOB_ATTRIBUTES + 3
                        base_features[base_idx] = ai_cap[0]
                        base_features[base_idx + 1] = ai_cap[1]
                        base_features[base_idx + 2] = ai_cap[2]

                    input_array = scaler.transform(
                        np.array([base_features]))
                    results = predict_fn(
                        tf.constant(input_array.astype(np.float32)))
                    FR_ID = int(np.argmax(results[j][0]))

                    if FR_ID == 0:
                        clustrFogNode = getClusterFromID(
                            jobQ[j].getDestinatinFogID())
                        if clustrFogNode is not None:
                            clustrFogNode.ExecutesJob(jobQ[j], sim_time, sl_no)
                    elif FR_ID == 1:
                        cs.ExecutesJob(jobQ[j], sim_time, sl_no)
                    else:
                        fog_idx = FR_ID - 2
                        if 0 <= fog_idx < len(ClusterRs):
                            ClusterRs[fog_idx].ExecutesJob(jobQ[j], sim_time, sl_no)
                            swap_time = compute_swap_time(
                                jobQ[j].get_noInstructions(),
                                jobQ[j].get_dataBytes(),
                                ClusterRs[fog_idx].getRAM())
                            exec_duration = compute_exec_duration(
                                jobQ[j].get_noInstructions(),
                                ClusterRs[fog_idx].get_CPUSpeed(),
                                swap_time)
                            ai_cap[fog_idx] = max(0, ai_cap[fog_idx] - exec_duration)
                        else:
                            cs.ExecutesJob(jobQ[j], sim_time, sl_no)

            case 5:
                # DAgger correction data collection.
                #
                # AI's own predictions drive the REAL execution (same as
                # AI_RM/case 2), so the simulation state evolves exactly
                # as it would under real deployment -- including AI's
                # own compounding small mistakes. Separately, GA is run
                # on this SAME real state (GA only reads device state,
                # never mutates it, so this is safe) to get the
                # "correct" label for the states AI actually visits --
                # not just the states GA's own earlier decisions would
                # have created. This directly targets covariate shift:
                # the training set is expanded to cover situations only
                # reachable by AI's own (imperfect) policy, closing the
                # gap between training-time and deployment-time state
                # distributions (Ross, Gordon & Bagnell, AISTATS 2011).
                n_jobs_this_slot = min(len(jobQ), MaxTasksInSlot)

                ai_cap = [dev.get_Ex_capacity() for dev in ClusterRs]

                slot_features = []
                for j in range(n_jobs_this_slot):
                    slot_features.append(jobQ[j].get_type())
                    slot_features.append(jobQ[j].get_noInstructions())
                    slot_features.append(jobQ[j].get_dataBytes())
                    for cap in ai_cap:
                        slot_features.append(cap)
                pad_len = MAX_JOBS * JOB_ATTRIBUTES - len(slot_features)
                slot_features.extend([0] * pad_len)

                # GA labels the state AI is ABOUT to act on (read-only,
                # does not touch real ClusterRs/cs capacity). Derived
                # directly from Cr since get_ex_class_label() depends on
                # the job having actually been dispatched, which hasn't
                # happened yet at this point.
                GA = GeneticAlgorithm(ClusterRs, jobQ)
                GA.GenerateOptimalChromosome()
                Cr = GA.returnBESTCRM()
                slot_labels = []
                for j in range(n_jobs_this_slot):
                    ga_fr_id = Cr.getRs(j)
                    if ga_fr_id is not None and ga_fr_id < len(ClusterRs):
                        slot_labels.append(
                            DEVICE_TO_CLASS.get(ClusterRs[ga_fr_id].id, 0))
                    elif ga_fr_id is not None and ga_fr_id == len(ClusterRs):
                        slot_labels.append(DEVICE_TO_CLASS.get(cs.id, 1))
                    else:
                        slot_labels.append(1)  # fallback: cloud
                for r in range(MaxTasksInSlot - n_jobs_this_slot):
                    slot_labels.append(0)

                row = (','.join(str(v) for v in slot_features) + ',' +
                       str(CPU_SPEED1) + ',' + str(FD_RAM1) + ',' +
                       str(FD_BANDWIDTH1) + ',' +
                       str(CPU_SPEED2) + ',' + str(FD_RAM2) + ',' +
                       str(FD_BANDWIDTH2) + ',' +
                       str(CPU_SPEED3) + ',' + str(FD_RAM3) + ',' +
                       str(FD_BANDWIDTH3) + ',' +
                       str(LATENCY_ED_FR) + ',' + str(LATENCY_CS_FR) + ',' +
                       ','.join(str(v) for v in slot_labels))
                logentry.writeToDataset(row + '\n')

                # AI's OWN prediction drives real execution, so the
                # state that evolves is the one AI's own policy visits.
                input_array = scaler.transform(np.array([slot_features]))
                results = predict_fn(tf.constant(input_array.astype(np.float32)))
                for j in range(n_jobs_this_slot):
                    FR_ID = int(np.argmax(results[j][0]))
                    if FR_ID == 0:
                        clustrFogNode = getClusterFromID(jobQ[j].getDestinatinFogID())
                        if clustrFogNode is not None:
                            clustrFogNode.ExecutesJob(jobQ[j], sim_time, sl_no)
                    elif FR_ID == 1:
                        cs.ExecutesJob(jobQ[j], sim_time, sl_no)
                    else:
                        fog_idx = FR_ID - 2
                        if 0 <= fog_idx < len(ClusterRs):
                            ClusterRs[fog_idx].ExecutesJob(jobQ[j], sim_time, sl_no)
                        else:
                            cs.ExecutesJob(jobQ[j], sim_time, sl_no)

            case 3:
                # Reinforcement learning agent resource management
                list_tasks = []
                for j in range(len(jobQ)):
                    tp = jobQ[j].get_type()
                    sz = jobQ[j].get_codesize()
                    dsz = jobQ[j].get_dataBytes()
                    list_tasks.append(tp)
                    list_tasks.append(sz)
                    list_tasks.append(dsz)
                for r in range(MaxTasksInSlot - len_ofjobQ):
                    list_tasks.append(0)
                    list_tasks.append(0)
                    list_tasks.append(0)
                input_jobs = np.array([list_tasks])
                result = agent.act(input_jobs, sim_time, sl_no)
                for j in range(len(jobQ)):
                    FR_ID = np.argmax(result[j][0])
                    ClusterRs[FR_ID].ExecutesJob(jobQ[j], sim_time, sl_no)
                rewards = computeRewards()
                agent.remember(input_jobs, result, rewards)
                agent.train()
                if sl_no % 100 == 0:
                    agent.save("weights_" +
                               "{:04d}".format(sl_no) + ".weights.h5")

            case 4:
                # Capacity-aware Round Robin resource management.
                # Assigns tasks to fog nodes in rotation (0,1,2,0,1,2,...).
                # If the selected node is full, tries the next node.
                # If all fog nodes are full, escalates to cloud.
                # This gives Round Robin full cross-cluster access like GA,
                # making the comparison fair -- the only difference is that
                # GA optimises for latency and energy while RR distributes evenly.
                rr_index = 0
                for job in jobQ:
                    assigned = False
                    for attempt in range(len(ClusterRs)):
                        fog_idx = (rr_index + attempt) % len(ClusterRs)
                        fog_node = ClusterRs[fog_idx]
                        swap_time = compute_swap_time(
                            job.get_noInstructions(),
                            job.get_dataBytes(),
                            fog_node.getRAM())
                        exec_duration = compute_exec_duration(
                            job.get_noInstructions(),
                            fog_node.get_CPUSpeed(),
                            swap_time)
                        remaining = fog_node.get_remaining_capacity(sim_time)
                        if exec_duration <= remaining:
                            fog_node.ExecutesJob(job, sim_time, sl_no)
                            rr_index = (fog_idx + 1) % len(ClusterRs)
                            assigned = True
                            break
                    if not assigned:
                        cs.ExecutesJob(job, sim_time, sl_no)
                        rr_index = (rr_index + 1) % len(ClusterRs)

    else:
        # No jobs in queue
        if RM_TYPE == GA_RM:
            for _ in range(MaxTasksInSlot * FEATURES_PER_JOB):
                slot_features.append(0)
            for _ in range(MaxTasksInSlot):
                slot_labels.append(0)
            # Write empty slot row - no trailing comma
            row = ','.join(str(val) for val in slot_features + slot_labels)
            logentry.writeToDataset(row + '\n')

    # Remove completed jobs from queue
    new_JobQ = []
    for j in jobQ:
        if j.NoValidTask:
            jobsTimetracker.RecordTimeDetails(j)
        else:
            new_JobQ.append(j)
    jobQ = new_JobQ.copy()

    sim_time = sim_time + SLOT_TIME

    # Write slot summary to log file
    for device in ClusterRs:
        device.ComputeSlotLoad()

    cs.WriteSlotSummary()

    # Slot status terminal output
    jobs_generated = len(Sensors)
    fog_executed = sum(dev.slot_no_task_ex for dev in ClusterRs)
    cloud_executed = cs.slot_no_task_ex
    remaining_jobs = len(jobQ)
    total_slots = int(MAX_SIMULATION_TIME / SLOT_TIME) if SLOT_TIME > 0 else 0
    progress = (sim_time / MAX_SIMULATION_TIME) * 100.0

    print(f"""
    [SLOT {sl_no}]
      Generated Jobs        : {jobs_generated}
      Executed on Fog       : {fog_executed}
      Offloaded to Cloud    : {cloud_executed}
      Remaining in Queue   : {remaining_jobs}

    [SIM] Slot {sl_no}/{total_slots} ({progress:.1f}%) | SimTime: {sim_time:.0f}/{MAX_SIMULATION_TIME}
    """)


# Final cleanup
for device in ClusterRs:
    device.clean(sim_time)
cs.clear(sim_time)

print("Simulation Completed")
print("-----------Results--------------------")
if len(ga_slot_times) > 0:
    avg_ga_time = sum(ga_slot_times) / len(ga_slot_times)
    print("")
    print("=" * 50)
    print("  GA INFERENCE SPEED - RESULTS")
    print("=" * 50)
    print(f"  Slots timed: {len(ga_slot_times)}")
    print(f"  Avg time per slot: {avg_ga_time:.6f} seconds")
    print(f"  Min: {min(ga_slot_times):.6f}s | Max: {max(ga_slot_times):.6f}s")
    print("=" * 50)
    print("")
print("Total Simulation Time:" + str(MAX_SIMULATION_TIME))
print("Total Number of Slots:" + str(sl_no))

tasksTimetracker.PrintTimeDetails()
print()
fog_total_energy_utilised = 0.
for device in ClusterRs:
    device.PrintResult(sl_no)
    fog_total_energy_utilised += device.energy_utilised
    print()

cs.PrintNoTasksExecuted()
print()
print("Total Fog layer Power Utilised:" + str(fog_total_energy_utilised))
print("\nYou could see:")
print("              " + logentry.fnameTasks + " for details on each tasks")
print("              " + logentry.fnameSlots + " for summary of each slot")
print("---------------------------------------")