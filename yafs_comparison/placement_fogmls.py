# -*- coding: utf-8 -*-
"""
placement_fogmls.py

Deploys one replica of the "Worker" module on each of the 3 fog nodes.
"""

from yafs.placement import Placement
from fogmls_shared_config import NUM_FOG_NODES


class FogPlacement(Placement):
    def initial_allocation(self, sim, app_name):
        app = sim.apps[app_name]
        services = app.services

        node_ids = list(range(1, NUM_FOG_NODES + 1)) + [0]

        for module in services:
            if module in self.scaleServices:
                for node_id in node_ids:
                    sim.deploy_module(app_name, module, services[module], [node_id])
