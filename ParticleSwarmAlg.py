# -*- coding: utf-8 -*-
"""
ParticleSwarmAlg.py

Particle Swarm Optimization scheduler for FogMLS.

PURPOSE: tests whether the "learnability cost of genuine search"
effect observed with the GA is a GENERAL property of stochastic
metaheuristics, or merely an artifact of our specific GA
implementation. This is the generality experiment that gates the
venue decision for Paper A.

DESIGN: deliberately mirrors GeneticAlg.py's structure exactly --
same Chromosome solution representation, same fitness functions,
same greedy seed, same order-aware random initialization, same
canonical-labeling fallback, same constants. The ONLY difference is
the SEARCH MECHANISM: PSO's velocity/position update replaces GA's
crossover/mutation. This isolation is essential: any accuracy
difference between GA-labeled and PSO-labeled datasets must be
attributable to the search mechanism, not to incidental
implementation differences.

TWO VARIANTS (mirroring the GA experiment exactly):
  1. PSO_DETERMINISTIC = True  -> no stochastic tie-breaking; particles
     always move toward the best-known position deterministically.
     Analogous to the "broken"/non-searching GA (93.49% accuracy).
  2. PSO_DETERMINISTIC = False -> genuine stochastic swarm search with
     random velocity coefficients. Analogous to the fixed,
     genuinely-searching GA (88.11% accuracy).

If the accuracy gap between these two variants is comparable to the
GA's 93.49% vs 88.11% gap, the learnability effect is general.
"""

import random
from config import (
    ALPHA, BETA, CLOUD_COST_UNIT_TIME, GAMMA,
    LATENCY_CS_FR, LATENCY_ED_FR, RAM_SWAP_UNIT,
    N_CHROMOSOMES, MAX_ITERATIONS
)
from devices import cs, compute_swap_time, compute_exec_duration

# Reuse GA's Chromosome class verbatim -- it is purely a solution
# representation plus fitness evaluation, with no GA-specific search
# logic inside it. Reusing it (rather than duplicating) guarantees
# PSO and GA are scored by IDENTICAL fitness functions, which is
# required for a controlled comparison.
from GeneticAlg import Chromosome

# ----------------------------------------------------------------
# PSO SEARCH-MECHANISM SWITCH -- the experimental variable.
# ----------------------------------------------------------------
# True  = deterministic variant (no stochastic component in the
#         velocity update; analogous to a non-searching GA)
# False = genuinely-searching variant (real stochastic swarm search)
PSO_DETERMINISTIC = False

# ----------------------------------------------------------------
# Canonical labeling -- IDENTICAL to GeneticAlg.py's value and
# semantics, so the two experiments differ only in search mechanism.
# ----------------------------------------------------------------
CANONICAL_LABEL_THRESHOLD = 0.02

# Order-aware random init parameters -- IDENTICAL to GeneticAlg.py.
TOP_K_RANDOM_CANDIDATES = 2
TIE_RELATIVE_TOLERANCE = 0.10

# ----------------------------------------------------------------
# Standard PSO coefficients (Kennedy & Eberhart 1995; Shi & Eberhart
# 1998 inertia weight). Values are the widely-used defaults.
# ----------------------------------------------------------------
PSO_INERTIA = 0.7298          # w  -- momentum from previous velocity
PSO_COGNITIVE = 1.49618       # c1 -- pull toward particle's own best
PSO_SOCIAL = 1.49618          # c2 -- pull toward swarm's global best

N_PARTICLES = N_CHROMOSOMES   # same population size as GA
PSO_ITERATIONS = MAX_ITERATIONS  # same search budget as GA


class Particle:
    """
    A PSO particle wrapping a Chromosome solution.

    Discrete-PSO encoding: since fog-node assignment is a discrete
    choice (which node runs job j), position is represented as the
    per-job assignment vector, and velocity as a per-job probability
    of changing that assignment. This is the standard discrete-PSO
    adaptation for assignment problems (cf. Kennedy & Eberhart's
    binary PSO), not a continuous-space PSO forced onto discrete
    variables.
    """

    def __init__(self, rs, jbs, seed_greedy=False):
        self.rs = rs
        self.jbs = jbs
        self.n_r = len(rs)
        self.n_j = len(jbs)

        # Current position (a full candidate solution).
        # FAIRNESS FIX: when PSO_DETERMINISTIC=True, ALL particles
        # seed identically via greedy_init -- a true zero-diversity
        # match to GA's original broken-bug condition (all 8
        # chromosomes identical). Without this, even the
        # "deterministic" variant retained real population diversity
        # from order_aware_random_init on non-seed particles, making
        # it an unfair, softer comparison than GA's real baseline.
        if seed_greedy or PSO_DETERMINISTIC:
            self.chromosome = Chromosome(rs, jbs)
        else:
            self.chromosome = Chromosome(rs, jbs, skip_init=True)
            self.chromosome.order_aware_random_init()

        # Personal best -- initialized to starting position
        self.pbest = Chromosome.create_empty(rs, jbs)
        self.pbest.copyfrom(self.chromosome)
        self.pbest_score = self.chromosome.RawScore()

        # Velocity: per-job change probability, initialized small
        self.velocity = [0.0 for _ in range(self.n_j)]

    def current_assignment(self):
        """Returns the per-job resource index vector (the 'position')."""
        return [self.chromosome.getRs(j) for j in range(self.n_j)]

    def update_velocity(self, gbest_assignment):
        """
        Standard PSO velocity update, adapted to discrete assignment:

            v = w*v + c1*r1*(pbest - x) + c2*r2*(gbest - x)

        For discrete positions, (pbest - x) and (gbest - x) are
        expressed as binary "differs / does not differ" indicators
        per job, so velocity accumulates as a per-job probability of
        moving toward the better-known assignment.

        In DETERMINISTIC mode, r1 and r2 are fixed at their expected
        value (0.5) instead of being drawn randomly -- removing the
        stochastic component while keeping the same search structure
        and the same expected step magnitude.
        """
        current = self.current_assignment()
        pbest_assign = [self.pbest.getRs(j) for j in range(self.n_j)]

        for j in range(self.n_j):
            if PSO_DETERMINISTIC:
                r1 = 0.5
                r2 = 0.5
            else:
                r1 = random.random()
                r2 = random.random()

            cognitive = PSO_COGNITIVE * r1 * (
                1.0 if pbest_assign[j] != current[j] else 0.0)
            social = PSO_SOCIAL * r2 * (
                1.0 if gbest_assignment[j] != current[j] else 0.0)

            self.velocity[j] = (PSO_INERTIA * self.velocity[j] +
                                cognitive + social)

    def update_position(self, gbest_assignment):
        """
        Applies velocity to move the particle, rebuilding the solution
        in QUEUE ORDER with live capacity tracking -- exactly as
        order_aware_random_init() does. This preserves the real
        sensor-queue-order structure that the GA experiment showed is
        essential for learnable labels.

        Per job, a high velocity means "move toward gbest/pbest".
        Where the particle does not move, it keeps a capacity-feasible
        assignment chosen by the same top-k rule the GA uses.
        """
        pbest_assign = [self.pbest.getRs(j) for j in range(self.n_j)]

        fg_capacity = [self.rs[i].get_Ex_capacity() for i in range(self.n_r)]

        for jb_idx in range(self.n_j):
            for rs_idx in range(self.n_r + 1):
                self.chromosome.tbl[jb_idx][rs_idx] = 0

            # Sigmoid-style move probability from velocity.
            v = self.velocity[jb_idx]
            move_prob = 1.0 / (1.0 + pow(2.718281828, -v)) if v > -50 else 0.0

            if PSO_DETERMINISTIC:
                # Deterministic: move iff probability exceeds 0.5.
                do_move = move_prob > 0.5
            else:
                do_move = random.random() < move_prob

            target = None
            if do_move:
                # Prefer gbest's choice, else pbest's.
                if gbest_assignment[jb_idx] is not None:
                    target = gbest_assignment[jb_idx]
                elif pbest_assign[jb_idx] is not None:
                    target = pbest_assign[jb_idx]

            assigned = False

            # Try the target assignment if it is capacity-feasible.
            if target is not None and target < self.n_r:
                swap_time = self.chromosome.GetSwapTime(target, jb_idx)
                exec_duration = compute_exec_duration(
                    self.jbs[jb_idx].get_noInstructions(),
                    self.rs[target].get_CPUSpeed(),
                    swap_time)
                if fg_capacity[target] >= exec_duration:
                    fg_capacity[target] -= exec_duration
                    self.chromosome.tbl[jb_idx][target] = exec_duration
                    assigned = True

            if not assigned:
                # Fall back to the SAME top-k selection rule used by
                # order_aware_random_init(), so non-moving jobs follow
                # identical logic to the GA experiment.
                scored_devs = []
                for fg_i in range(self.n_r):
                    swap_time = self.chromosome.GetSwapTime(fg_i, jb_idx)
                    exec_duration = compute_exec_duration(
                        self.jbs[jb_idx].get_noInstructions(),
                        self.rs[fg_i].get_CPUSpeed(),
                        swap_time)
                    if fg_capacity[fg_i] >= exec_duration:
                        score = exec_duration / fg_capacity[fg_i]
                        scored_devs.append((score, fg_i, exec_duration))

                if scored_devs:
                    scored_devs.sort(key=lambda x: x[0])
                    top_k = scored_devs[:min(TOP_K_RANDOM_CANDIDATES,
                                              len(scored_devs))]
                    best_score = top_k[0][0]

                    is_tie = (len(top_k) > 1 and best_score > 0 and
                              (top_k[-1][0] - best_score) / best_score
                              <= TIE_RELATIVE_TOLERANCE)

                    if is_tie or PSO_DETERMINISTIC:
                        # Deterministic tie-break: lowest device index.
                        tied = [c for c in top_k
                                if best_score <= 0 or
                                (c[0] - best_score) / best_score
                                <= TIE_RELATIVE_TOLERANCE]
                        if not tied:
                            tied = [top_k[0]]
                        chosen = min(tied, key=lambda x: x[1])
                    else:
                        chosen = random.choice(top_k)

                    _, dev, exec_duration = chosen
                    fg_capacity[dev] -= exec_duration
                    self.chromosome.tbl[jb_idx][dev] = exec_duration
                else:
                    # No fog node has capacity -- assign to cloud.
                    swap_time = self.chromosome.GetSwapTime(0, jb_idx)
                    exec_duration = compute_exec_duration(
                        self.jbs[jb_idx].get_noInstructions(),
                        cs.get_CPUSpeed(),
                        swap_time)
                    self.chromosome.tbl[jb_idx][self.n_r] = exec_duration

        self.chromosome.Fitness()

    def update_pbest(self):
        """Updates personal best if the current position is better."""
        score = self.chromosome.RawScore()
        if score <= self.pbest_score:
            self.pbest.copyfrom(self.chromosome)
            self.pbest_score = score


class ParticleSwarmOptimization:
    """
    PSO scheduler. Deliberately exposes the SAME public interface as
    GeneticAlgorithm (GenerateOptimalChromosome / returnBESTCRM), so
    IoTsimulation.py can swap between them with no other changes.
    """

    def __init__(self, rs, jbs):
        self.rs = rs
        self.jbs = jbs
        self.BESTCRM = None
        self.fitness_history = []

    def GenerateOptimalChromosome(self):
        if len(self.jbs) == 0:
            self.BESTCRM = Chromosome.create_empty(self.rs, self.jbs)
            self.BESTCRM.Fitness()
            return

        # Swarm initialization -- mirrors GA exactly: particle 0 is the
        # greedy seed (and is kept as the canonical-labeling fallback),
        # the rest use order-aware random init.
        particles = []
        greedy_only = None
        for i in range(N_PARTICLES):
            if i == 0:
                p = Particle(self.rs, self.jbs, seed_greedy=True)
                greedy_only = Chromosome.create_empty(self.rs, self.jbs)
                greedy_only.copyfrom(p.chromosome)
            else:
                p = Particle(self.rs, self.jbs, seed_greedy=False)
            particles.append(p)

        # Global best across the swarm.
        gbest = Chromosome.create_empty(self.rs, self.jbs)
        gbest.copyfrom(particles[0].chromosome)
        gbest_score = particles[0].chromosome.RawScore()
        for p in particles[1:]:
            s = p.chromosome.RawScore()
            if s < gbest_score:
                gbest.copyfrom(p.chromosome)
                gbest_score = s

        self.fitness_history = [gbest_score]

        for _ in range(PSO_ITERATIONS):
            gbest_assignment = [gbest.getRs(j) for j in range(len(self.jbs))]

            for p in particles:
                p.update_velocity(gbest_assignment)
                p.update_position(gbest_assignment)
                p.update_pbest()

                s = p.chromosome.RawScore()
                if s < gbest_score:
                    gbest.copyfrom(p.chromosome)
                    gbest_score = s

            self.fitness_history.append(gbest_score)

        # Canonical labeling -- IDENTICAL rule and threshold to the GA,
        # so the two experiments differ ONLY in search mechanism.
        greedy_score = greedy_only.RawScore()
        if greedy_score > 0:
            improvement = (greedy_score - gbest_score) / greedy_score
        else:
            improvement = 0.0

        if improvement > CANONICAL_LABEL_THRESHOLD:
            self.BESTCRM = gbest
        else:
            self.BESTCRM = greedy_only

    def returnBESTCRM(self):
        return self.BESTCRM
