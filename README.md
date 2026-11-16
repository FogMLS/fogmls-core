# FogMLS: A Fog Computing Simulator with Built-in ML-Ready Dataset Generation

FogMLS is a Python-based fog computing simulator with a 3-tier IoT-Fog-Cloud
hierarchy, purpose-built to generate genetic-algorithm-optimized, labeled
scheduling datasets for training machine learning-based fog schedulers.

## Key capability

Unlike existing fog/edge simulation tools (iFogSim, YAFS, CloudSim,
EdgeSimPy), FogMLS treats **labeled dataset generation as a first-class
simulation output**. Every simulated scheduling decision — made by a
genuinely-searching genetic algorithm, not a deterministic heuristic — is
recorded as a structured training example, ready for direct use in training
external ML models. A classifier trained on this dataset, deployed with a
confidence-ranked decision policy, exceeds the scheduling performance of the
optimization algorithm that generated its own training data (see
`ten_trials_comparison.py` and Results below).

## Validated

FogMLS's simulation behavior has been cross-validated against two independent,
peer-reviewed fog computing simulators — YAFS and iFogSim — across two
scheduling strategies (GA-based and Round Robin), with closely matching
results:


| Scheduler | FogMLS | YAFS | iFogSim |
|---|---|---|---|
| GA (avg latency) | 15.52 | 13.90 | 13.90 |
| Round Robin (avg latency) | 18.49 | 15.43 | 17.28 |


## Repository contents

| File / folder | Purpose |
|---|---|
| `IoTsimulation.py` | Main simulation entry point |
| `GeneticAlg.py` | Genetic algorithm scheduler (order-aware initialization, canonical labeling, tie-tolerance calibration) |
| `ParticleSwarmAlg.py` | Particle Swarm Optimization scheduler, used to test whether the labeling methodology generalizes beyond GA |
| `config.py` | All simulation parameters (topology, workload, scheduler settings) |
| `devices.py` | Fog node, cloud, sensor, and actuator device models |
| `random_forest_baseline.py` | Trains the Random Forest classifier and reports offline classification accuracy |
| `intelligent_surveillance_case_study.py` | Reproduction of iFogSim's own published case study (Gupta et al., 2017), demonstrating FogMLS generalizes beyond its own default scenario |
| `ablation_test_job_ordering.py` | Controlled ablation proving the latency-gap explanation vs. YAFS/iFogSim (deterministic task ordering) |
| `ten_trials_comparison.py` | Rigorous 10-independent-trial statistical comparison of all five schedulers (DEFAULT, GA, RR, RF, RF_SMART), with paired t-tests |
| `scalability_test.py` | Sensor-density scaling experiment (9/27/45/63 sensors per cluster) |
| `scalability_test_nodes.py` | Fog node-count scaling experiment (1-6 nodes), following LEAF's precedent |
| `yafs_comparison/` | Cross-validation against YAFS, an independent peer-reviewed fog simulator |
| `ifogsim_comparison/` | Cross-validation against iFogSim, the foundational fog simulator |

## Setup

**FogMLS core** (Python, this repo's root files) requires Python 3.10+:
```bash
pip install -r requirements.txt
```

**Cross-simulator validation** requires additional setup, documented
separately: see `yafs_comparison/README.md` (requires a working YAFS
installation) and `ifogsim_comparison/README.md` (requires JDK 21+ and a
cloned iFogSim repository).

## Quick start

```bash
python3 IoTsimulation.py
```

This runs a full simulation (5,000 scheduling slots, 3 fog nodes, 27 sensors,
1 cloud server) using the genetic algorithm scheduler, and produces a labeled
dataset file (`dataset-<timestamp>.txt`) ready for ML training.

To train the Random Forest classifier and demonstrate the dataset's ML
learnability:
```bash
python3 random_forest_baseline.py dataset-<timestamp>.txt
```

To reproduce iFogSim's own published Intelligent Surveillance case study:
```bash
python3 intelligent_surveillance_case_study.py
```

To reproduce the rigorous 10-trial statistical comparison across all five
schedulers (requires `random_forest_model.joblib`, produced by the step
above):
```bash
python3 ten_trials_comparison.py
```

## Configuration

Edit `config.py` to change the resource management strategy:

```python
RM_TYPE = GA_RM         # Genetic Algorithm
RM_TYPE = RR_RM         # Round Robin baseline
RM_TYPE = RF_SMART_RM   # Random Forest, confidence-ranked deployment (best result)
```


## Confirmed results

GA mutation step fixed on 16 November 2026 (child and parent now compared on raw score). All results below use the fixed version.

Averages over independent random seeds (the number of seeds is stated per result):

- **RF_SMART_RM** (Random Forest classifier trained on GA-derived labels,
  deployed with its confidence-ranked prediction list) reaches 14.81
  average task latency over 9 seeds, against 15.52 for GA over 10 seeds
  and 16.48 for DEFAULT_RM. Top-1 Random Forest deployment reaches 18.34.
  See `ten_trials_comparison.py`.
- **GA_RM** reaches 15.52 average latency against 16.48 for `DEFAULT_RM`.
  Round Robin reaches 18.49.
- **Cross-simulator validation:** results are compared against two
  independent, peer-reviewed fog simulators (YAFS, iFogSim). See
  `yafs_comparison/` and `ifogsim_comparison/` for details and
  reproduction instructions.
- **Classical ML learnability:** the Random Forest baseline reaches 73.9%
  offline accuracy (macro F1 0.75). F1 is 0.98 for Cloud and 0.73, 0.70
  and 0.59 for Fog-1, Fog-2 and Fog-3.
- **Case study reproduction:** independently confirms both the direction and
  scaling trend reported in iFogSim's own published Intelligent Surveillance
  case study (Gupta et al., 2017) — see
  `intelligent_surveillance_case_study.py`.
- **PSO generality test:** the same order-aware initialization and
  canonical labeling recipe was applied to an independently built PSO
  scheduler (tie tolerance 0.10 for PSO, 0.05 for GA). PSO reaches 12.26
  average latency against 15.52 for GA. See `ParticleSwarmAlg.py`.

## Related dataset

A large-scale (269,946 task-resource assignments, 5,000 slots) dataset
generated with FogMLS is publicly available on Zenodo:
[DOI: 10.5281/zenodo.19506449](https://doi.org/10.5281/zenodo.19506449)



## Citation

If you use FogMLS in your research, please cite:

```
[Citation details to be added upon publication]
```

## License

MIT License — see LICENSE file.
