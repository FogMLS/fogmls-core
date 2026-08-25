# -*- coding: utf-8 -*-
"""
ga_core.py

Direct Python port of FogMLS's real GeneticAlg.py (Chromosome +
GeneticAlgorithm classes), operating on simple Resource/Job proxy
objects instead of FogMLS's devices.py classes, so it can run inside
YAFS unmodified in logic. Formulas copied EXACTLY, not reimplemented
from memory, to guarantee identical behavior to the real FogMLS.

UPDATED: now ports FogMLS's CONFIRMED, HPC-verified GA configuration
(order-aware random initialization + canonical labeling), which
replaced an earlier plain-random-initialization version. This matches
the exact Python GeneticAlg.py that produced 88.11% accuracy and
15.51 avg latency on FogMLS's own real simulation.
"""

import random
from fogmls_shared_config import (
    ALPHA as _ALPHA, BETA as _BETA, GAMMA as _GAMMA,
    N_CHROMOSOMES as _N_CHROMOSOMES, MAX_ITERATIONS as _MAX_ITERATIONS,
    TOURNAMENT_SIZE as _TOURNAMENT_SIZE,
    LATENCY_ED_FR as _LATENCY_ED_FR, LATENCY_CS_FR as _LATENCY_CS_FR,
    CLOUD_COST_UNIT_TIME as _CLOUD_COST_UNIT_TIME,
    RAM_SWAP_UNIT as _RAM_SWAP_UNIT
)


class Resource:
    """Proxy matching FogMLS's FogDevice/Cloud interface (get_CPUSpeed,
    getBANDWIDTH, getRAM, get_Ex_capacity, busy_power)."""
    def __init__(self, cpu_speed, bandwidth, ram, ex_capacity, busy_power=0.0):
        self._cpu_speed = cpu_speed
        self._bandwidth = bandwidth
        self._ram = ram
        self._ex_capacity = ex_capacity
        self.busy_power = busy_power

    def get_CPUSpeed(self):
        return self._cpu_speed

    def getBANDWIDTH(self):
        return self._bandwidth

    def getRAM(self):
        return self._ram

    def get_Ex_capacity(self):
        return self._ex_capacity


class Job:
    """Proxy matching FogMLS's Job interface (get_noInstructions,
    get_dataBytes)."""
    def __init__(self, no_instructions, data_bytes, ref=None):
        self._no_instructions = no_instructions
        self._data_bytes = data_bytes
        self.ref = ref  # original task/tuple this job represents

    def get_noInstructions(self):
        return self._no_instructions

    def get_dataBytes(self):
        return self._data_bytes


def compute_swap_time(no_instructions, data_bytes, ram, ram_swap_unit=_RAM_SWAP_UNIT):
    """Copied exactly from FogMLS's devices.py compute_swap_time."""
    swap_ram_no = (no_instructions + data_bytes) / ram
    if swap_ram_no > 1:
        return ram_swap_unit * swap_ram_no
    return ram_swap_unit


def compute_exec_duration(no_instructions, cpu_speed, swap_time):
    """Copied exactly from FogMLS's devices.py compute_exec_duration."""
    return no_instructions / cpu_speed + swap_time


class GAConfig:
    """Matches FogMLS's config.py GA-related constants -- imported
    from fogmls_shared_config.py, single source of truth."""
    ALPHA = _ALPHA
    BETA = _BETA
    GAMMA = _GAMMA
    N_CHROMOSOMES = _N_CHROMOSOMES
    MAX_ITERATIONS = _MAX_ITERATIONS
    TOURNAMENT_SIZE = _TOURNAMENT_SIZE
    LATENCY_ED_FR = _LATENCY_ED_FR
    LATENCY_CS_FR = _LATENCY_CS_FR
    CLOUD_COST_UNIT_TIME = _CLOUD_COST_UNIT_TIME

    # Minimum fractional improvement (over pure greedy) that GA's
    # searched solution must achieve to be accepted as the canonical
    # label/schedule for a slot. Below this, greedy's consistent
    # assignment is used instead.
    CANONICAL_LABEL_THRESHOLD = 0.02

    # How many top-scoring valid fog nodes to consider when building
    # a random chromosome in order_aware_random_init().
    TOP_K_RANDOM_CANDIDATES = 2

    # Relative score difference below which two candidates are a
    # practical tie (deterministic break) vs meaningfully different
    # (random break, for real exploration).
    TIE_RELATIVE_TOLERANCE = 0.05


class Chromosome:
    """Direct port of FogMLS's GeneticAlg.py Chromosome class."""
    def __init__(self, rs, jbs, cloud, cfg, skip_init=False):
        self.n_r = len(rs)
        self.n_j = len(jbs)
        self.rs = rs
        self.jbs = jbs
        self.cloud = cloud
        self.cfg = cfg
        self.tbl = [[0 for _ in range(self.n_r + 1)] for _ in range(self.n_j)]
        self.Fitnes = 0.
        self.LatencyFitnes = 0.
        self.CostFitnes = 0.
        self.EnergyFitnes = 0.

        if not skip_init:
            self.greedy_init()

    @classmethod
    def create_empty(cls, rs, jbs, cloud, cfg):
        return cls(rs, jbs, cloud, cfg, skip_init=True)

    def copyfrom(self, p):
        self.LatencyFitnes = p.LatencyFitnes
        self.EnergyFitnes = p.EnergyFitnes
        self.CostFitnes = p.CostFitnes
        self.Fitnes = p.Fitnes
        for i in range(self.n_j):
            for j in range(self.n_r + 1):
                self.tbl[i][j] = p.tbl[i][j]

    def Fitness(self):
        self.Fitnes = 0.
        self.DelayFitness()
        self.EnergyFitness()
        self.CostFitness()
        self.Fitnes = (self.LatencyFitnes * self.cfg.ALPHA +
                       self.EnergyFitnes * self.cfg.BETA +
                       self.CostFitnes * self.cfg.GAMMA)

    def RawScore(self):
        """
        Combined weighted score using RAW (non population-relative)
        fitness components. Unlike Fitnes -- which is normalized
        against the current population's max values and therefore
        shifts meaning between generations -- this score is stable
        across generations and safe to use for elitism/canonical-
        labeling comparisons.
        """
        return (self.LatencyFitnes * self.cfg.ALPHA +
                self.EnergyFitnes * self.cfg.BETA +
                self.CostFitnes * self.cfg.GAMMA)

    def DelayFitness(self):
        self.LatencyFitnes = 0.
        for job_idx in range(self.n_j):
            FR_ID = self.getRs(job_idx)
            if FR_ID is None:
                pass
            elif FR_ID < len(self.rs):
                swap_time = self.GetSwapTime(FR_ID, job_idx)
                self.LatencyFitnes = (
                    self.LatencyFitnes +
                    self.jbs[job_idx].get_noInstructions() /
                    self.rs[FR_ID].get_CPUSpeed() +
                    self.jbs[job_idx].get_dataBytes() /
                    self.rs[FR_ID].getBANDWIDTH() +
                    self.cfg.LATENCY_ED_FR +
                    swap_time
                )
            else:
                swap_time = compute_swap_time(
                    self.jbs[job_idx].get_noInstructions(),
                    self.jbs[job_idx].get_dataBytes(),
                    self.cloud.getRAM())
                self.LatencyFitnes = (
                    self.LatencyFitnes +
                    self.jbs[job_idx].get_noInstructions() /
                    self.cloud.get_CPUSpeed() +
                    self.jbs[job_idx].get_dataBytes() /
                    self.cloud.getBANDWIDTH() +
                    self.cfg.LATENCY_CS_FR +
                    swap_time
                )

    def EnergyFitness(self):
        self.EnergyFitnes = 0.
        for i in range(self.n_j):
            FR_ID = self.getRs(i)
            if FR_ID is None:
                pass
            elif FR_ID < len(self.rs):
                self.EnergyFitnes = (
                    self.EnergyFitnes +
                    self.jbs[i].get_noInstructions() /
                    self.rs[FR_ID].get_CPUSpeed() *
                    self.rs[FR_ID].busy_power
                )

    def CostFitness(self):
        self.CostFitnes = 0.
        for i in range(self.n_j):
            FR_ID = self.getRs(i)
            if FR_ID is not None and FR_ID >= len(self.rs):
                self.CostFitnes = (
                    self.CostFitnes +
                    self.jbs[i].get_noInstructions() /
                    self.cloud.get_CPUSpeed() *
                    self.cfg.CLOUD_COST_UNIT_TIME
                )

    def getRs(self, j):
        for i in range(self.n_r + 1):
            if self.tbl[j][i] > 0:
                return i
        return None

    def greedy_init(self):
        fg_capacity = [self.rs[i].get_Ex_capacity() for i in range(self.n_r)]

        for jb_idx in range(self.n_j):
            for rs_idx in range(self.n_r + 1):
                self.tbl[jb_idx][rs_idx] = 0

            best_dev = None
            best_score = float('inf')
            for fg_i in range(self.n_r):
                swap_time = self.GetSwapTime(fg_i, jb_idx)
                exec_duration = compute_exec_duration(
                    self.jbs[jb_idx].get_noInstructions(),
                    self.rs[fg_i].get_CPUSpeed(),
                    swap_time)
                if fg_capacity[fg_i] >= exec_duration:
                    score = exec_duration / fg_capacity[fg_i]
                    if score < best_score:
                        best_score = score
                        best_dev = fg_i

            if best_dev is not None:
                swap_time = self.GetSwapTime(best_dev, jb_idx)
                exec_duration = compute_exec_duration(
                    self.jbs[jb_idx].get_noInstructions(),
                    self.rs[best_dev].get_CPUSpeed(),
                    swap_time)
                fg_capacity[best_dev] -= exec_duration
                self.tbl[jb_idx][best_dev] = exec_duration
            else:
                swap_time = self.GetSwapTime(0, jb_idx)
                exec_duration = compute_exec_duration(
                    self.jbs[jb_idx].get_noInstructions(),
                    self.cloud.get_CPUSpeed(),
                    swap_time)
                self.tbl[jb_idx][self.n_r] = exec_duration

        self.Fitness()

    def order_aware_random_init(self):
        """
        Order-aware random initialization, with deterministic
        tie-break. Direct port of FogMLS's confirmed GeneticAlg.py
        order_aware_random_init() -- the exact configuration that
        produced 88.11% accuracy (HPC-verified) and matches GA's real
        15.51 avg-latency simulation numbers exactly.

        Processes jobs in QUEUE ORDER (same as greedy_init) and tracks
        remaining capacity as it goes. For each job, scores all fog
        nodes with sufficient capacity using the same load-balanced
        score as greedy_init() (exec_duration / remaining_capacity,
        lower = better).

        Selection among valid nodes:
        - Take the TOP_K_RANDOM_CANDIDATES best-scoring valid nodes.
        - If their scores are a practical near-tie (within
          TIE_RELATIVE_TOLERANCE), break the tie DETERMINISTICALLY
          (always the lower device index).
        - Otherwise, choose randomly among the top-k candidates --
          preserves genuine search diversity for crossover/mutation.
        """
        fg_capacity = [self.rs[i].get_Ex_capacity() for i in range(self.n_r)]

        for jb_idx in range(self.n_j):
            for rs_idx in range(self.n_r + 1):
                self.tbl[jb_idx][rs_idx] = 0

            scored_devs = []
            for fg_i in range(self.n_r):
                swap_time = self.GetSwapTime(fg_i, jb_idx)
                exec_duration = compute_exec_duration(
                    self.jbs[jb_idx].get_noInstructions(),
                    self.rs[fg_i].get_CPUSpeed(),
                    swap_time)
                if fg_capacity[fg_i] >= exec_duration:
                    score = exec_duration / fg_capacity[fg_i]
                    scored_devs.append((score, fg_i, exec_duration))

            if scored_devs:
                scored_devs.sort(key=lambda x: x[0])
                top_k = scored_devs[:min(self.cfg.TOP_K_RANDOM_CANDIDATES,
                                          len(scored_devs))]

                if len(top_k) > 1 and top_k[0][0] > 0 and \
                   (top_k[-1][0] - top_k[0][0]) / top_k[0][0] <= self.cfg.TIE_RELATIVE_TOLERANCE:
                    # Practical near-tie: break deterministically,
                    # always the lower device index among tied
                    # candidates.
                    tied = [c for c in top_k
                            if top_k[0][0] > 0 and
                            (c[0] - top_k[0][0]) / top_k[0][0] <= self.cfg.TIE_RELATIVE_TOLERANCE]
                    chosen_score, chosen, chosen_exec = min(
                        tied, key=lambda x: x[1])
                else:
                    # Meaningfully different options: genuine random
                    # exploration among them.
                    chosen_score, chosen, chosen_exec = random.choice(top_k)

                fg_capacity[chosen] -= chosen_exec
                self.tbl[jb_idx][chosen] = chosen_exec
            else:
                # No fog node has capacity. Assign to cloud.
                swap_time = self.GetSwapTime(0, jb_idx)
                exec_duration = compute_exec_duration(
                    self.jbs[jb_idx].get_noInstructions(),
                    self.cloud.get_CPUSpeed(),
                    swap_time)
                self.tbl[jb_idx][self.n_r] = exec_duration

        self.Fitness()

    def crossover(self, pt, p2):
        for i in range(pt, self.n_j):
            for j in range(self.n_r + 1):
                self.tbl[i][j] = p2.tbl[i][j]
        self.Fitness()

    def mutation(self, p):
        for r in range(self.n_r):
            r_time_used = 0
            for j in range(self.n_j):
                r_time_used += self.tbl[j][r]
                if r_time_used > self.rs[r].get_Ex_capacity():
                    r_time_used -= self.tbl[j][r]
                    rsid = self.Find_FogDev(j)
                    self.tbl[j][rsid] = self.tbl[j][r]
                    self.tbl[j][r] = 0
        self.Fitness()
        if self.Fitnes > p.Fitnes:
            self.copyfrom(p)

    def Find_FogDev(self, jId):
        for r in range(self.n_r):
            r_time_used = 0
            for j in range(self.n_j):
                r_time_used = r_time_used + self.tbl[j][r]
            remaining = self.rs[r].get_Ex_capacity() - r_time_used
            swap_time = self.GetSwapTime(r, jId)
            exec_duration = compute_exec_duration(
                self.jbs[jId].get_noInstructions(),
                self.rs[r].get_CPUSpeed(),
                swap_time)
            if remaining >= exec_duration:
                return r
        return self.n_r

    def GetSwapTime(self, dev, jobid):
        return compute_swap_time(
            self.jbs[jobid].get_noInstructions(),
            self.jbs[jobid].get_dataBytes(),
            self.rs[dev].getRAM())


class GeneticAlgorithm:
    """Direct port of FogMLS's GeneticAlg.py GeneticAlgorithm class."""
    def __init__(self, rs, jbs, cloud, cfg=None):
        self.rs = rs
        self.jbs = jbs
        self.cloud = cloud
        self.cfg = cfg or GAConfig()

    def SelectChromosome(self, pop):
        candidates = random.sample(pop, min(self.cfg.TOURNAMENT_SIZE, len(pop)))
        return min(candidates, key=lambda c: c.Fitnes)

    def GA_crossover(self, p1, p2):
        c1 = Chromosome.create_empty(self.rs, self.jbs, self.cloud, self.cfg)
        c2 = Chromosome.create_empty(self.rs, self.jbs, self.cloud, self.cfg)
        c1.copyfrom(p1)
        c2.copyfrom(p2)
        crossover_point = random.randrange(0, len(self.jbs) - 1, 1) if len(self.jbs) > 1 else 0
        c1.crossover(crossover_point, c2)
        c2.crossover(crossover_point, p1)
        return [c1, c2]

    def NormalizePopulation(self, CRMs):
        max_L = max(c.LatencyFitnes for c in CRMs) or 1.0
        max_E = max(c.EnergyFitnes for c in CRMs) or 1.0
        max_C = max(c.CostFitnes for c in CRMs) or 1.0
        for c in CRMs:
            c.Fitnes = ((c.LatencyFitnes / max_L) * self.cfg.ALPHA +
                        (c.EnergyFitnes / max_E) * self.cfg.BETA +
                        (c.CostFitnes / max_C) * self.cfg.GAMMA)

    def GenerateOptimalChromosome(self):
        if len(self.jbs) == 0:
            self.BESTCRM = Chromosome.create_empty(self.rs, self.jbs, self.cloud, self.cfg)
            self.BESTCRM.Fitness()
            return

        CRMs = []
        greedy_only = None

        for i in range(self.cfg.N_CHROMOSOMES):
            if i == 0:
                # Seed the population with one greedy (smart) solution.
                ch = Chromosome(self.rs, self.jbs, self.cloud, self.cfg)
                # Keep a separate, untouched copy of the pure greedy
                # result -- fallback if GA's search doesn't find a
                # MEANINGFULLY better solution.
                greedy_only = Chromosome.create_empty(
                    self.rs, self.jbs, self.cloud, self.cfg)
                greedy_only.copyfrom(ch)
            else:
                # Rest of population: order-aware random assignment
                # for genuine diversity.
                ch = Chromosome.create_empty(
                    self.rs, self.jbs, self.cloud, self.cfg)
                ch.order_aware_random_init()
            CRMs.append(ch)

        self.NormalizePopulation(CRMs)
        CRMs.sort(key=lambda c: c.Fitnes)

        # Elitism: track the best chromosome found so far.
        best_ever = Chromosome.create_empty(
            self.rs, self.jbs, self.cloud, self.cfg)
        best_ever.copyfrom(CRMs[0])

        itr = 0
        while itr < self.cfg.MAX_ITERATIONS:
            itr += 1
            selected = [self.SelectChromosome(CRMs) for _ in range(self.cfg.N_CHROMOSOMES)]
            children = []
            for i in range(0, self.cfg.N_CHROMOSOMES - 1, 2):
                p1, p2 = selected[i], selected[i + 1]
                c = self.GA_crossover(p1, p2)
                c[0].mutation(p1)
                c[1].mutation(p2)
                children.append(c[0])
                children.append(c[1])

            # Inject the elite into this generation's pool, then trim
            # back down to N_CHROMOSOMES, keeping only the best.
            elite_copy = Chromosome.create_empty(
                self.rs, self.jbs, self.cloud, self.cfg)
            elite_copy.copyfrom(best_ever)
            children.append(elite_copy)

            self.NormalizePopulation(children)
            children.sort(key=lambda c: c.Fitnes)
            CRMs = children[:self.cfg.N_CHROMOSOMES]

            # Update elite only if the new best is genuinely better on
            # a stable, normalization-independent scale.
            if CRMs[0].RawScore() <= best_ever.RawScore():
                best_ever.copyfrom(CRMs[0])

        # Canonical labeling: use GA's genuine search result ONLY when
        # it's a meaningfully better solution than plain greedy.
        # Otherwise use greedy's consistent, repeatable assignment
        # instead -- removes arbitrary tie-break noise from the
        # training target while preserving every case where GA found
        # a REAL, meaningful improvement.
        greedy_score = greedy_only.RawScore()
        best_score = best_ever.RawScore()
        if greedy_score > 0:
            improvement = (greedy_score - best_score) / greedy_score
        else:
            improvement = 0.0

        if improvement > self.cfg.CANONICAL_LABEL_THRESHOLD:
            self.BESTCRM = best_ever
        else:
            self.BESTCRM = greedy_only

    def returnBESTCRM(self):
        return self.BESTCRM
