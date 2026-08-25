# FogMLS vs. YAFS Cross-Validation

Reproduces FogMLS's simulation results (GA and Round Robin schedulers) inside
YAFS, an independent, peer-reviewed fog computing simulator, to validate that
FogMLS's underlying computational model is realistic and consistent with
established tools.

## Setup

Requires YAFS installed (`pip install -e .` from a cloned
[YAFS repository](https://github.com/acsicuib/YAFS)).

**Important:** replace YAFS's own `src/yafs/core.py` with
`core_py_PATCHED_replace_yafs_src_yafs_core_py.py` from this folder before
running the Round Robin test. This patch fixes two real bugs found during
validation:
1. **Link-lock timing**: the original engine locks a shared network link for
   the full transmit+propagation duration; only transmit time should
   contribute to the shared lock.
2. **Consumer-queue serialization**: messages arriving together were forced
   to wait behind each other sequentially instead of being processed
   independently.

Place all other `.py` files in this folder into your YAFS example directory
(e.g. `YAFS/src/examples/FogMLS_Comparison/`).

**Version-compatibility note:** `application_fogmls.py` and
`random_message.py` include small compatibility fixes for YAFS versions where
`fractional_selectivity` is not exported from `yafs.application`, and where
the `Message` base class carries version-specific internal attributes not
present in earlier releases.

## Running

**Simplest option — unified entry point** (matches iFogSim comparison's
`Main.java` pattern): edit `RM_TYPE` in `fogmls_shared_config.py`
(`GA_RM` or `RR_RM`), then:
```bash
python3 main.py
```

**Or run each scheduler's script directly:**

**GA scheduler** (does not require the `core.py` patch — bypasses YAFS's
event engine by design, since GA makes batch decisions incompatible with
per-tuple event processing):
```bash
python3 main_ga.py
```

**Round Robin scheduler** (requires the patched `core.py`):
```bash
python3 main_fogmls.py
```

## Confirmed results (Aug 2026, independently reproduced on two separate
## HPC/sandbox environments)

| Scheduler | Avg latency |
|---|---|
| GA | 13.90 |
| Round Robin | 15.43-15.46 |
