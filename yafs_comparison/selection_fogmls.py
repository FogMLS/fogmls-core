# -*- coding: utf-8 -*-
"""
selection_fogmls.py

Replicates FogMLS's real Round Robin scheduler (IoTsimulation.py case 4)
for Sensor -> Worker routing.
"""

import networkx as nx
from yafs.selection import Selection
from fogmls_shared_config import SLOT_TIME, NUM_FOG_NODES, FD_RAM

FOG_NODE_IDS = list(range(1, NUM_FOG_NODES + 1))
CLOUD_NODE_ID = 0


def compute_swap_time(no_instructions, data_bytes, ram):
    """Copied exactly from FogMLS's devices.py compute_swap_time."""
    swap_ram_no = (no_instructions + data_bytes) / ram
    if swap_ram_no > 1:
        return swap_ram_no  # RAM_SWAP_UNIT = 1
    return 1.0


class RoundRobinSelection(Selection):
    def __init__(self):
        super(RoundRobinSelection, self).__init__()
        self.rr_index = 0
        self.current_slot_index = -1
        self.slot_busy_time = {fid: 0.0 for fid in FOG_NODE_IDS}
        self.total_generated = 0
        self.total_fog_assigned = 0
        self.total_cloud_escalated = 0

    def _maybe_reset_slot(self, sim):
        slot_idx = int(sim.env.now // SLOT_TIME)
        if slot_idx != self.current_slot_index:
            self.current_slot_index = slot_idx
            self.slot_busy_time = {fid: 0.0 for fid in FOG_NODE_IDS}
            self.rr_index = 0

    def get_path(self, sim, app_name, message, topology_src, alloc_DES, alloc_module, traffic, from_des):
        DES_dst = alloc_module[app_name][message.dst]
        des_to_node = {des: alloc_DES[des] for des in DES_dst}

        if message.dst != "Worker":
            best_dev = None
            best_len = None
            for des, node in des_to_node.items():
                try:
                    plen = nx.shortest_path_length(sim.topology.G, source=topology_src, target=node)
                except nx.NetworkXNoPath:
                    continue
                if best_len is None or plen < best_len:
                    best_len = plen
                    best_dev = des
            path = list(nx.shortest_path(sim.topology.G, source=topology_src, target=des_to_node[best_dev]))
            return [path], [best_dev]

        self._maybe_reset_slot(sim)

        if message.name in ("M.CAM", "M.TRAF"):
            self.total_generated += 1

        chosen_des = None
        for attempt in range(len(FOG_NODE_IDS)):
            fog_idx = (self.rr_index + attempt) % len(FOG_NODE_IDS)
            fog_node_id = FOG_NODE_IDS[fog_idx]

            ipt = sim.topology.nodeAttributes[fog_node_id]["IPT"]
            ram = FD_RAM[fog_idx]
            swap_time = compute_swap_time(message.inst, message.bytes, ram)
            exec_duration = message.inst / float(ipt) + swap_time

            if self.slot_busy_time[fog_node_id] + exec_duration <= SLOT_TIME:
                for des, node in des_to_node.items():
                    if node == fog_node_id:
                        chosen_des = des
                        break
                self.slot_busy_time[fog_node_id] += exec_duration
                self.rr_index = (fog_idx + 1) % len(FOG_NODE_IDS)
                self.total_fog_assigned += 1
                break

        if chosen_des is None:
            for des, node in des_to_node.items():
                if node == CLOUD_NODE_ID:
                    chosen_des = des
                    break
            self.rr_index = (self.rr_index + 1) % len(FOG_NODE_IDS)
            self.total_cloud_escalated += 1

        dst_node = alloc_DES[chosen_des]
        path = list(nx.shortest_path(sim.topology.G, source=topology_src, target=dst_node))

        return [path], [chosen_des]
