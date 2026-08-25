# -*- coding: utf-8 -*-
"""
fogmls_shared_config.py

Single source of truth for all FogMLS comparison config values.
Change values HERE ONLY -- every other file imports from this one.

SLOT_TIME is auto-calculated using FogMLS's own real formula (copied
directly from config.py), so changing SENSORS_PER_CLUSTER here
automatically produces the correct SLOT_TIME everywhere, exactly like
FogMLS's real config.py does.
"""

# ================================================================
# RESOURCE MANAGEMENT OPTIONS (matches FogMLS's config.py exactly)
# ================================================================
DEFAULT_RM = 0
GA_RM = 1
RR_RM = 4

# Set resource management approach -- change THIS ONE line to switch
RM_TYPE = RR_RM

# ================================================================
# CHANGE THESE VALUES TO CHANGE THE WORKLOAD
# ================================================================
SENSORS_PER_CLUSTER = 9      # sensors per fog cluster (FogMLS default: 9)
RANDOM_SEED = 42             # for reproducibility

# ================================================================
# SIMULATION DURATION (matches FogMLS's config.py exactly)
# ================================================================
MAX_SIMULATION_TIME = 175000

# ================================================================
# FITNESS PARAMETERS (matches FogMLS's config.py exactly)
# ================================================================
ALPHA = 0.5   # latency weight
BETA = 0.5    # energy weight
GAMMA = 0.0   # cost weight

# ================================================================
# GA PARAMETERS (matches FogMLS's config.py exactly)
# ================================================================
N_CHROMOSOMES = 8
MAX_ITERATIONS = 10
TOURNAMENT_SIZE = 3

# ================================================================
# RAM SWAP (matches FogMLS's config.py exactly)
# ================================================================
RAM_SWAP_UNIT = 1

# ================================================================
# FOG / CLOUD HARDWARE (matches FogMLS's real config.py)
# ================================================================
NUM_FOG_NODES = 3
CPU_SPEED = [40, 80, 160]              # Fog-1, Fog-2, Fog-3 (SU)
FD_BANDWIDTH_RAW = [400, 800, 1600]    # Fog-1, Fog-2, Fog-3 (SU)
FD_RAM = [2000, 4000, 8000]            # Fog-1, Fog-2, Fog-3 (MB)
BUSY_POWER = [105.0, 110.0, 115.339]   # Fog-1, Fog-2, Fog-3 (W)
IDLE_POWER = [80.0, 82.0, 85.0]        # Fog-1, Fog-2, Fog-3 (W)

CLOUD_CPU_SPEED = 48000
CLOUD_RAM = 1000000
CLOUD_BANDWIDTH_RAW = 1600
CLOUD_COST_UNIT_TIME = 0.01

LATENCY_ED_FR = 1     # sensor/fog propagation delay (SU)
LATENCY_CS_FR = 50    # fog/cloud propagation delay (SU)

# ================================================================
# TASK INSTRUCTION / DATA SIZE RANGES (matches FogMLS's config.py
# exactly -- full min/max). Keyed by task type name matching devices.py.
# ================================================================
NO_INSTRUCTIONS = {
    "CAM":     (8, 14),      # CAMERA_SEN
    "TRAF":    (10, 16),     # TRAFFIC_SEN
    "VEHID":   (100, 140),   # VEHICLE_IDENTIFICATION
    "TRAFLOC": (110, 150),   # TRAFFIC_LOCATION
    "ACTVEH":  (0, 0),       # ACTION_ON_VEHICLE
    "ACTTRAF": (0, 0),       # ACTION_CLEAR_TRAFFIC
}
DATA_SIZE = {
    "CAM":     (800, 1200),
    "TRAF":    (400, 600),
    "VEHID":   (1200, 1800),
    "TRAFLOC": (1600, 2400),
    "ACTVEH":  (0, 0),
    "ACTTRAF": (0, 0),
}

# ================================================================
# TASK TYPE CONSTANTS (matches FogMLS's config.py exactly)
# ================================================================
CAMERA_SEN = 1
TRAFFIC_SEN = 2
VEHICLE_IDENTIFICATION = 3
TRAFFIC_LOCATION = 4
ACTION_ON_VEHICLE = 5
ACTION_CLEAR_TRAFFIC = 6

# Integer-keyed versions of the ranges above, for code (like
# main_ga.py) using FogMLS's real numeric task-type convention.
NO_INSTRUCTIONS_BY_ID = {
    CAMERA_SEN: NO_INSTRUCTIONS["CAM"],
    TRAFFIC_SEN: NO_INSTRUCTIONS["TRAF"],
    VEHICLE_IDENTIFICATION: NO_INSTRUCTIONS["VEHID"],
    TRAFFIC_LOCATION: NO_INSTRUCTIONS["TRAFLOC"],
    ACTION_ON_VEHICLE: NO_INSTRUCTIONS["ACTVEH"],
    ACTION_CLEAR_TRAFFIC: NO_INSTRUCTIONS["ACTTRAF"],
}
DATA_SIZE_BY_ID = {
    CAMERA_SEN: DATA_SIZE["CAM"],
    TRAFFIC_SEN: DATA_SIZE["TRAF"],
    VEHICLE_IDENTIFICATION: DATA_SIZE["VEHID"],
    TRAFFIC_LOCATION: DATA_SIZE["TRAFLOC"],
    ACTION_ON_VEHICLE: DATA_SIZE["ACTVEH"],
    ACTION_CLEAR_TRAFFIC: DATA_SIZE["ACTTRAF"],
}

# Max-only values, used below purely for the SLOT_TIME formula
# (matches FogMLS's own config.py naming for this specific purpose).
NO_INSTRUCTIONS1_MAX = NO_INSTRUCTIONS["CAM"][1]
NO_INSTRUCTIONS2_MAX = NO_INSTRUCTIONS["TRAF"][1]
NO_INSTRUCTIONS3_MAX = NO_INSTRUCTIONS["VEHID"][1]
NO_INSTRUCTIONS4_MAX = NO_INSTRUCTIONS["TRAFLOC"][1]

# ================================================================
# AUTO-CALCULATED SLOT_TIME -- copied exactly from FogMLS's config.py
# formula. Do not edit this section directly; it derives from the
# values above, same as the real simulator does.
# ================================================================
_MAX_INSTRUCTIONS_ANY_TASK = max(
    NO_INSTRUCTIONS1_MAX, NO_INSTRUCTIONS2_MAX,
    NO_INSTRUCTIONS3_MAX, NO_INSTRUCTIONS4_MAX
)
_MIN_CPU_SPEED = min(CPU_SPEED)
_SLOT_TIME0 = -2.5
_INTERVAL_GAP = _MAX_INSTRUCTIONS_ANY_TASK / _MIN_CPU_SPEED

SLOT_TIME = _INTERVAL_GAP * SENSORS_PER_CLUSTER + _INTERVAL_GAP + _SLOT_TIME0

# ================================================================
# YAFS UNIT-CONVERSION NOTE:
# YAFS internally computes transmit_time = bytes / (BW * 1,000,000).
# Pre-dividing here so YAFS's internal x1,000,000 cancels out and
# matches FogMLS's real formula (data_bytes / bandwidth) exactly.
# Downstream files should use FD_BANDWIDTH / CLOUD_BANDWIDTH (already
# converted), never FD_BANDWIDTH_RAW / CLOUD_BANDWIDTH_RAW directly.
# ================================================================
FD_BANDWIDTH = [b / 1000000 for b in FD_BANDWIDTH_RAW]
CLOUD_BANDWIDTH = CLOUD_BANDWIDTH_RAW / 1000000

if __name__ == '__main__':
    print("SENSORS_PER_CLUSTER:", SENSORS_PER_CLUSTER)
    print("SLOT_TIME (auto-calculated):", SLOT_TIME)
