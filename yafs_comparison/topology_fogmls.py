# -*- coding: utf-8 -*-
"""
topology_fogmls.py

Builds a YAFS topology matching FogMLS's real hardware specs, confirmed
directly from Harry's config.py and devices.py (not guessed).

Structure (matches IoTsimulation.py exactly):
    27 sensors (9 per cluster x 3 clusters)
  + 3 actuators (1 per cluster)
  + 3 fog nodes (heterogeneous: Fog-1/2/3)
  + 1 cloud server
  connected as: sensor -> fog node (its own cluster) -> cloud

Node IDs:
    0        = Cloud
    1,2,3    = Fog-1, Fog-2, Fog-3
    10-18    = Sensors in Cluster 1 (connected to Fog-1)
    20-28    = Sensors in Cluster 2 (connected to Fog-2)
    30-38    = Sensors in Cluster 3 (connected to Fog-3)
    100,200,300 = Actuators for Cluster 1/2/3

All IPT (CPU) and BW (bandwidth) values are Harry's real SU numbers,
used directly. Ratios between fog nodes are preserved exactly
(40:80:160 CPU, 400:800:1600 bandwidth) -- that is what actually
matters for this validation, not absolute unit conversion.
"""

from yafs.topology import Topology
from fogmls_shared_config import (
    CLOUD_CPU_SPEED, CLOUD_RAM, CLOUD_BANDWIDTH,
    CPU_SPEED, FD_BANDWIDTH, FD_RAM,
    LATENCY_ED_FR, LATENCY_CS_FR,
    SENSORS_PER_CLUSTER, NUM_FOG_NODES
)


def create_json_topology():
    topology_json = {}
    topology_json["entity"] = []
    topology_json["link"] = []

    # --- CLOUD ---
    cloud_dev = {
        "id": 0, "model": "cloud", "mytag": "cloud",
        "IPT": CLOUD_CPU_SPEED, "RAM": CLOUD_RAM, "COST": 0, "WATT": 0.0
    }
    topology_json["entity"].append(cloud_dev)

    # --- FOG NODES (heterogeneous) ---
    fog_ids = [1, 2, 3]
    for i in range(NUM_FOG_NODES):
        fog_dev = {
            "id": fog_ids[i], "model": "fog-" + str(i + 1),
            "mytag": "fog",
            "IPT": CPU_SPEED[i], "RAM": FD_RAM[i], "COST": 0, "WATT": 0.0
        }
        topology_json["entity"].append(fog_dev)

        link_to_cloud = {
            "s": fog_ids[i], "d": 0,
            "BW": FD_BANDWIDTH[i], "PR": LATENCY_CS_FR
        }
        topology_json["link"].append(link_to_cloud)

    # Direct fog-to-fog links for every pair. Needed because Stage 2/3
    # tasks (VEHICLE_IDENTIFICATION, TRAFFIC_LOCATION, etc.) re-enter
    # the scheduler and can land on a DIFFERENT fog node than the
    # previous stage. FogMLS treats this as costing nothing extra.
    for i in range(NUM_FOG_NODES):
        for j in range(NUM_FOG_NODES):
            if i != j:
                link = {
                    "s": fog_ids[i], "d": fog_ids[j],
                    "BW": FD_BANDWIDTH[j], "PR": LATENCY_ED_FR
                }
                topology_json["link"].append(link)

    # --- SENSORS + ACTUATORS ---
    sensor_id_start = 10
    cluster_id_stride = SENSORS_PER_CLUSTER + 2
    actuator_ids = [100, 200, 300]
    for cluster_idx in range(NUM_FOG_NODES):
        fog_id = fog_ids[cluster_idx]
        cluster_base = sensor_id_start + cluster_idx * cluster_id_stride

        for j in range(SENSORS_PER_CLUSTER):
            sid = cluster_base + j
            model_name = "sensor-camera" if j % 2 == 0 else "sensor-traffic"
            sensor_dev = {
                "id": sid, "model": model_name,
                "IPT": 100, "RAM": 500, "COST": 0, "WATT": 0.0
            }
            topology_json["entity"].append(sensor_dev)

            # Direct link to ALL 3 fog nodes -- FogMLS's Round Robin
            # can assign to any fog node, not just this sensor's
            # cluster, and treats every fog node as equally "1 hop
            # away" (LATENCY_ED_FR=1) regardless of physical cluster.
            for target_idx in range(NUM_FOG_NODES):
                target_fog_id = fog_ids[target_idx]
                link = {
                    "s": sid, "d": target_fog_id,
                    "BW": FD_BANDWIDTH[target_idx], "PR": LATENCY_ED_FR
                }
                topology_json["link"].append(link)

        aid = actuator_ids[cluster_idx]
        actuator_dev = {
            "id": aid, "model": "actuator-device",
            "IPT": 100, "RAM": 500, "COST": 0, "WATT": 0.0
        }
        topology_json["entity"].append(actuator_dev)

    # Connect EVERY fog node directly to EVERY actuator.
    for fog_idx in range(NUM_FOG_NODES):
        fog_id = fog_ids[fog_idx]
        for act_idx in range(NUM_FOG_NODES):
            aid = actuator_ids[act_idx]
            link = {
                "s": fog_id, "d": aid,
                "BW": FD_BANDWIDTH[fog_idx], "PR": LATENCY_ED_FR
            }
            topology_json["link"].append(link)

    # Cloud's direct actuator connections too.
    for act_idx in range(NUM_FOG_NODES):
        aid = actuator_ids[act_idx]
        link = {
            "s": 0, "d": aid,
            "BW": CLOUD_BANDWIDTH, "PR": LATENCY_CS_FR
        }
        topology_json["link"].append(link)

    return topology_json


def build_topology():
    t = Topology()
    t_json = create_json_topology()
    t.load(t_json)
    return t


if __name__ == '__main__':
    t = build_topology()
    print("Topology built successfully.")
    print("Total nodes:", len(t.G.nodes()))
    print("Total links:", len(t.G.edges()))
