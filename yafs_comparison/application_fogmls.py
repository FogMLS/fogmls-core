# -*- coding: utf-8 -*-
"""
application_fogmls.py

Real 3-stage task chain matching FogMLS exactly, using RandomMessage
so each task instance gets a genuinely random instruction/byte count
within FogMLS's real ranges.

Real ranges (confirmed from config.py):
  CAMERA_SEN   instr 8-14   data 800-1200
  TRAFFIC_SEN  instr 10-16  data 400-600
  VEHICLE_IDENTIFICATION instr 100-140 data 1200-1800
  TRAFFIC_LOCATION       instr 110-150 data 1600-2400
  ACTION_ON_VEHICLE / ACTION_CLEAR_TRAFFIC: instr=0, data=0
"""

from yafs.application import Application, Message


def fractional_selectivity(**kwargs):
    """Always forward the message (threshold=1.0 = 100% forward). Added
    for compatibility -- not all YAFS versions ship this in yafs.application."""
    return True
from random_message import RandomMessage
from fogmls_shared_config import SLOT_TIME


def create_application():
    a = Application(name="FogMLS_RR")

    a.set_modules([
        {"Sensor": {"Type": Application.TYPE_SOURCE}},
        {"Worker": {"RAM": 10, "Type": Application.TYPE_MODULE}},
        {"Actuator": {"Type": Application.TYPE_SINK}}
    ])

    # Stage 1 (Sensor -> Worker)
    m_cam = RandomMessage("M.CAM", "Sensor", "Worker",
                           inst_range=(8, 14), bytes_range=(800, 1200))
    m_traf = RandomMessage("M.TRAF", "Sensor", "Worker",
                            inst_range=(10, 16), bytes_range=(400, 600))

    # Stage 2 (Worker -> Worker, self-loop, HEAVY)
    m_vehid = RandomMessage("M.VEHID", "Worker", "Worker",
                             inst_range=(100, 140), bytes_range=(1200, 1800))
    m_trafloc = RandomMessage("M.TRAFLOC", "Worker", "Worker",
                               inst_range=(110, 150), bytes_range=(1600, 2400))
    m_vehid.slot_align_time = SLOT_TIME
    m_trafloc.slot_align_time = SLOT_TIME

    # Stage 3 (Worker -> Worker, self-loop, terminal, exact 0 cost)
    m_actveh = Message("M.ACTVEH", "Worker", "Worker", instructions=0, bytes=0)
    m_acttraf = Message("M.ACTTRAF", "Worker", "Worker", instructions=0, bytes=0)
    m_actveh.slot_align_time = SLOT_TIME
    m_acttraf.slot_align_time = SLOT_TIME

    # Final result message (Worker -> Actuator, sink).
    m_done = Message("M.DONE", "Worker", "Actuator", instructions=0, bytes=0)

    a.add_source_messages(m_cam)
    a.add_source_messages(m_traf)

    a.add_service_module("Worker", m_cam, m_vehid, fractional_selectivity, threshold=1.0)
    a.add_service_module("Worker", m_traf, m_trafloc, fractional_selectivity, threshold=1.0)
    a.add_service_module("Worker", m_vehid, m_actveh, fractional_selectivity, threshold=1.0)
    a.add_service_module("Worker", m_trafloc, m_acttraf, fractional_selectivity, threshold=1.0)
    a.add_service_module("Worker", m_actveh, m_done, fractional_selectivity, threshold=1.0)
    a.add_service_module("Worker", m_acttraf, m_done, fractional_selectivity, threshold=1.0)

    return a
