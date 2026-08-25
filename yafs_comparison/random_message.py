# -*- coding: utf-8 -*-
"""
random_message.py

YAFS's Message normally carries one FIXED instruction/byte count per
message type. FogMLS instead draws a fresh random value within a
range for every single task (random.randint(min, max)).

Since YAFS creates a new task instance via copy.copy(message) on
EVERY emission, we override __copy__ so each copy gets its own random
inst/bytes values within the real FogMLS range.
"""

import copy
import random
from yafs.application import Message


class RandomMessage(Message):
    def __init__(self, name, src, dst, inst_range, bytes_range, seed_offset=0):
        mid_inst = (inst_range[0] + inst_range[1]) // 2
        mid_bytes = (bytes_range[0] + bytes_range[1]) // 2
        super(RandomMessage, self).__init__(name, src, dst,
                                             instructions=mid_inst, bytes=mid_bytes)
        self.inst_range = inst_range
        self.bytes_range = bytes_range

    def __copy__(self):
        # Generic attribute copy: copies EVERY attribute this specific
        # YAFS version's base Message class may have set (including
        # ones not known when this file was written, e.g.
        # original_DES_src on some versions), instead of a hand-picked
        # list that can silently miss version-specific internal fields.
        new = RandomMessage(self.name, self.src, self.dst,
                             self.inst_range, self.bytes_range)
        new.__dict__.update(self.__dict__)
        new.inst = random.randint(self.inst_range[0], self.inst_range[1])
        new.bytes = random.randint(self.bytes_range[0], self.bytes_range[1])
        new.path = copy.copy(self.path)
        return new
