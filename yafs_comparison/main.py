# -*- coding: utf-8 -*-
"""
main.py

Unified entry point, matching FogMLS's real design exactly: ONE
config file (fogmls_shared_config.py), ONE setting (RM_TYPE), ONE
file to run -- instead of separate main_fogmls.py (Round Robin) and
main_ga.py (GA) scripts needing to be run individually.

To switch between Round Robin and GA: edit RM_TYPE in
fogmls_shared_config.py (RR_RM = 4, GA_RM = 1), then just run:
    python3 main.py

Same as FogMLS's own config.py + IoTsimulation.py pattern, and
mirrors the equivalent Main.java entry point in ifogsim_comparison/.
"""

import time
from fogmls_shared_config import RM_TYPE, GA_RM, RR_RM, DEFAULT_RM

if __name__ == '__main__':
    if RM_TYPE == RR_RM:
        print("Resource management strategy: RR_RM (Round Robin)")
        import logging.config
        import os
        logging.config.fileConfig(os.getcwd() + '/logging.ini')
        import main_fogmls
        SIM_TIME = 175000
        start = time.time()
        main_fogmls.main(simulated_time=SIM_TIME)
        print("--- %s seconds ---" % (time.time() - start))
        main_fogmls.print_summary(SIM_TIME)
    elif RM_TYPE == GA_RM:
        print("Resource management strategy: GA_RM")
        import main_ga
        start = time.time()
        main_ga.main()
        print("--- %s seconds ---" % (time.time() - start))
    else:
        raise ValueError(
            "RM_TYPE=%d not supported in this comparison build. "
            "Only RR_RM (%d) and GA_RM (%d) are implemented. "
            "Set RM_TYPE in fogmls_shared_config.py to one of these."
            % (RM_TYPE, RR_RM, GA_RM))
