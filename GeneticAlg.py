import random
from config import (
    ALPHA, BETA, CLOUD_COST_UNIT_TIME, GAMMA,
    LATENCY_CS_FR, LATENCY_ED_FR, RAM_SWAP_UNIT,
    N_CHROMOSOMES, MAX_ITERATIONS, TOURNAMENT_SIZE
)
from devices import cs, logentry, compute_swap_time, compute_exec_duration

# Minimum fractional improvement (over pure greedy) that GA's searched
# solution must achieve to be accepted as the canonical label/schedule
# for a slot. Below this, greedy's consistent assignment is used
# instead -- see canonical labeling logic in GenerateOptimalChromosome.
CANONICAL_LABEL_THRESHOLD = 0.02

# How many top-scoring valid fog nodes to consider when building a
# random chromosome in order_aware_random_init().
TOP_K_RANDOM_CANDIDATES = 2

# Relative score difference below which two candidates are considered
# a practical tie (broken deterministically) rather than meaningfully
# different options (broken randomly, for real exploration). Expressed
# as a fraction of the best candidate's score.
TIE_RELATIVE_TOLERANCE = 0.05

# Validate N_CHROMOSOMES is even at startup.
# Crossover loop processes chromosomes in pairs.
# An odd value silently drops the last chromosome each generation.
if N_CHROMOSOMES % 2 != 0:
    print("[WARNING] N_CHROMOSOMES = " + str(N_CHROMOSOMES) +
          " is odd. Last chromosome will be skipped in crossover. " +
          "Set N_CHROMOSOMES to an even number in config.py.")


class Chromosome:
    def __init__(self, rs, jbs, skip_init=False):
        self.n_r = len(rs)
        self.n_j = len(jbs)
        self.rs = rs
        self.jbs = jbs
        # tbl stores execution TIME consumed per task per resource.
        # All values are in TIME units (SU) for consistency with
        # the simulator's capacity tracking.
        self.tbl = [[0 for columns in range(self.n_r + 1)]
                    for rows in range(self.n_j)]
        self.Fitnes = 0.
        self.LatencyFitnes = 0.
        self.CostFitnes = 0.
        self.EnergyFitnes = 0.

        if not skip_init:
            self.greedy_init()
        
        #if not skip_init:
            #self.random_init()

    @classmethod
    def create_empty(cls, rs, jbs):
        """
        Creates a Chromosome without running greedy_init.
        Used in GA_crossover to avoid wasted initialization
        that would be immediately overwritten by copyfrom.
        """
        return cls(rs, jbs, skip_init=True)

    def printassignment(self):
        print(self.tbl)

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
        self.Fitnes = (self.LatencyFitnes * ALPHA +
                       self.EnergyFitnes * BETA +
                       self.CostFitnes * GAMMA)

    def RawScore(self):
        """
        Combined weighted score using RAW (non population-relative)
        fitness components. Unlike self.Fitnes -- which is normalized
        against the current population's max values and therefore
        shifts meaning between generations -- this score is stable
        across generations and safe to use for elitism comparisons.
        """
        return (self.LatencyFitnes * ALPHA +
                self.EnergyFitnes * BETA +
                self.CostFitnes * GAMMA)

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
                    LATENCY_ED_FR +
                    swap_time
                )
            else:
                self.LatencyFitnes = (
                    self.LatencyFitnes +
                    self.jbs[job_idx].get_noInstructions() /
                    cs.get_CPUSpeed() +
                    LATENCY_CS_FR
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
            else:
                pass

    def CostFitness(self):
        self.CostFitnes = 0.
        for i in range(self.n_j):
            FR_ID = self.getRs(i)
            if FR_ID is None:
                pass
            elif FR_ID < len(self.rs):
                pass
            else:
                self.CostFitnes = (
                    self.CostFitnes +
                    self.jbs[i].get_noInstructions() /
                    cs.get_CPUSpeed() *
                    CLOUD_COST_UNIT_TIME
                )

    def getRs(self, j):
        for i in range(self.n_r + 1):
            if self.tbl[j][i] > 0:
                return i
        return None

    def greedy_init(self):
        """
        Greedy initialization: assigns each task to the fog node
        with the best load-balanced score that has sufficient
        remaining capacity. Tasks that cannot fit on any fog node
        are assigned to the cloud server.

        Scoring: exec_duration / remaining_capacity
        This prefers nodes with more remaining capacity relative
        to the task size, distributing load evenly across all fog
        nodes instead of always preferring the fastest node.

        Previously the score was exec_duration alone, which always
        picked the fastest node (Fog-3) first, causing Fog-2 and
        Fog-3 to saturate simultaneously and producing nearly
        identical capacity values at assignment time. This made
        the DL model unable to discriminate between Fog-2 and
        Fog-3 assignments.

        CRITICAL: All capacity values are in TIME units (SU).
        fg_capacity[i] tracks remaining time on fog node i.
        exec_duration = instructions/speed + swap_time (TIME units).
        This matches the simulator's capacity model exactly.
        """
        # Initialize remaining time capacity per fog node.
        # Ex_capacity = SLOT_TIME (in SU).
        fg_capacity = []
        for fg_i in range(self.n_r):
            fg_capacity.append(self.rs[fg_i].get_Ex_capacity())

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
                    # Load-balanced score: prefer nodes with more
                    # remaining capacity relative to task duration.
                    # Lower score = better assignment.
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
                # No fog node has capacity. Assign to cloud.
                swap_time = self.GetSwapTime(0, jb_idx)
                exec_duration = compute_exec_duration(
                    self.jbs[jb_idx].get_noInstructions(),
                    cs.get_CPUSpeed(),
                    swap_time)
                self.tbl[jb_idx][self.n_r] = exec_duration

        self.Fitness()

    def crossover(self, pt, p2):
        for i in range(pt, self.n_j):
            for j in range(self.n_r + 1):
                self.tbl[i][j] = p2.tbl[i][j]
        self.Fitness()

    def mutation(self, p):
        """Capacity repair mutation. Checks each fog node for
        over-allocation and moves excess tasks to another node
        or cloud. All capacity comparisons use TIME units."""
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
        if self.RawScore() > p.RawScore():
            self.copyfrom(p)

    def Find_FogDev(self, jId):
        """Find a fog node with enough remaining TIME capacity
        for task jId. Returns fog index or self.n_r for cloud."""
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
        """Compute swap time using centralised function."""
        return compute_swap_time(
            self.jbs[jobid].get_noInstructions(),
            self.jbs[jobid].get_dataBytes(),
            self.rs[dev].getRAM())

    def random_init(self):
        """Random initialisation for comparison experiment only.
        Assigns each task to a random fog node or cloud."""
        for jb_idx in range(self.n_j):
            for rs_idx in range(self.n_r + 1):
                self.tbl[jb_idx][rs_idx] = 0
            # Random assignment
            rand_dev = random.randint(0, self.n_r)
            swap_time = self.GetSwapTime(min(rand_dev, self.n_r-1), jb_idx)
            exec_duration = compute_exec_duration(
                self.jbs[jb_idx].get_noInstructions(),
                self.rs[rand_dev].get_CPUSpeed() if rand_dev < self.n_r else cs.get_CPUSpeed(),
                swap_time)
            self.tbl[jb_idx][rand_dev] = exec_duration
        self.Fitness()

    def order_aware_random_init(self):
        """
        Order-aware random initialization, with deterministic tie-break.

        Processes jobs in QUEUE ORDER (same as greedy) and tracks
        remaining capacity as it goes. For each job, scores all fog
        nodes with sufficient capacity using the same load-balanced
        score as greedy_init() (exec_duration / remaining_capacity,
        lower = better).

        Selection among valid nodes:
        - Take the TOP_K_RANDOM_CANDIDATES best-scoring valid nodes.
        - If their scores are a GENUINE near-tie (within TIE_EPSILON),
          break the tie DETERMINISTICALLY (always the lower device
          index) -- this removes arbitrary coin-flip noise from cases
          that are truly equivalent, without reducing real exploration.
        - If the top candidates are NOT a near-tie (meaningfully
          different scores, both still "valid" but not equally good),
          choose randomly among them -- this preserves genuine search
          diversity for crossover/mutation to work with.

        Why this matters: job queue position corresponds to a real,
        fixed sensor identity (Sensors are iterated in the same order
        every slot in IoTsimulation.py), so respecting queue order
        keeps a legitimate, learnable pattern intact. The deterministic
        tie-break further removes label noise that comes purely from
        arbitrarily choosing between fitness-equivalent options, while
        still allowing genuine randomness where options actually differ.
        """
        fg_capacity = []
        for fg_i in range(self.n_r):
            fg_capacity.append(self.rs[fg_i].get_Ex_capacity())

        for jb_idx in range(self.n_j):
            for rs_idx in range(self.n_r + 1):
                self.tbl[jb_idx][rs_idx] = 0

            # Score all fog nodes with enough remaining capacity.
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
                top_k = scored_devs[:min(TOP_K_RANDOM_CANDIDATES,
                                          len(scored_devs))]

                if len(top_k) > 1 and top_k[0][0] > 0 and \
                   (top_k[-1][0] - top_k[0][0]) / top_k[0][0] <= TIE_RELATIVE_TOLERANCE:
                    # Practical near-tie: break deterministically, always
                    # the lower device index among the tied candidates.
                    tied = [c for c in top_k
                            if top_k[0][0] > 0 and
                            (c[0] - top_k[0][0]) / top_k[0][0] <= TIE_RELATIVE_TOLERANCE]
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
                    cs.get_CPUSpeed(),
                    swap_time)
                self.tbl[jb_idx][self.n_r] = exec_duration

        self.Fitness()

class GeneticAlgorithm:
    def __init__(self, rs, jbs):
        self.rs = rs
        self.jbs = jbs

    def SelectChromosome(self, pop):
        """
        Tournament selection. Randomly samples TOURNAMENT_SIZE candidates
        and returns the one with the lowest fitness score.
        """
        candidates = random.sample(pop, min(TOURNAMENT_SIZE, len(pop)))
        best = min(candidates, key=lambda c: c.Fitnes)
        return best

    def GA_crossover(self, p1, p2):
        """
        Single point crossover producing two children from two parents.
        """
        c1 = Chromosome.create_empty(self.rs, self.jbs)
        c2 = Chromosome.create_empty(self.rs, self.jbs)
        c1.copyfrom(p1)
        c2.copyfrom(p2)

        crossover_point = random.randrange(0, len(self.jbs) - 1, 1)
        c1.crossover(crossover_point, c2)
        c2.crossover(crossover_point, p1)
        return [c1, c2]

    def NormalizePopulation(self, CRMs):
        max_L = max(c.LatencyFitnes for c in CRMs) or 1.0
        max_E = max(c.EnergyFitnes for c in CRMs) or 1.0
        max_C = max(c.CostFitnes for c in CRMs) or 1.0
        for c in CRMs:
            c.Fitnes = ((c.LatencyFitnes / max_L) * ALPHA +
                        (c.EnergyFitnes / max_E) * BETA +
                        (c.CostFitnes / max_C) * GAMMA)

    def GenerateOptimalChromosome(self):
        CRMs = []
        greedy_only = None
        for i in range(N_CHROMOSOMES):
            if i == 0:
                # Seed the population with one greedy (smart) solution.
                ch = Chromosome(self.rs, self.jbs)
                # Keep a separate, untouched copy of the pure greedy
                # result. This is our fallback: if GA's search doesn't
                # find a MEANINGFULLY better solution, we return this
                # instead of a random-derived one, to preserve the
                # consistent, learnable pattern greedy provides.
                greedy_only = Chromosome.create_empty(self.rs, self.jbs)
                greedy_only.copyfrom(ch)
            else:
                # Rest of population: random assignment for genuine
                # diversity, so crossover/mutation have real variety
                # to search over instead of identical clones.
                ch = Chromosome(self.rs, self.jbs, skip_init=True)
                ch.order_aware_random_init()
            CRMs.append(ch)
    
        self.NormalizePopulation(CRMs)
        CRMs.sort(key=lambda c: c.Fitnes)
    
        # Fitness logging for convergence analysis.
        # Uses RawScore() (stable, normalization-independent) so the
        # trend is genuinely meaningful across iterations -- the
        # normalized Fitnes shifts scale between generations and is
        # not safe to plot as a convergence curve.
        self.fitness_history = [CRMs[0].RawScore()]

        # Elitism: track the best chromosome found so far (deep copy),
        # so later generations can never regress below it. Without this,
        # crossover/mutation can lose a good solution with no way to
        # recover it, causing fitness to drift worse over iterations.
        best_ever = Chromosome.create_empty(self.rs, self.jbs)
        best_ever.copyfrom(CRMs[0])
    
        itr = 0
        while itr < MAX_ITERATIONS:
            itr += 1
    
            selected = [self.SelectChromosome(CRMs)
                        for _ in range(N_CHROMOSOMES)]
    
            children = []
            for i in range(0, N_CHROMOSOMES - 1, 2):
                p1, p2 = selected[i], selected[i + 1]
                c = self.GA_crossover(p1, p2)
                c[0].mutation(p1)
                c[1].mutation(p2)
                children.append(c[0])
                children.append(c[1])

            # Inject the elite into this generation's pool so it competes
            # fairly under the same normalization, then trim back down to
            # N_CHROMOSOMES, keeping only the best.
            elite_copy = Chromosome.create_empty(self.rs, self.jbs)
            elite_copy.copyfrom(best_ever)
            children.append(elite_copy)

            self.NormalizePopulation(children)
            children.sort(key=lambda c: c.Fitnes)
            CRMs = children[:N_CHROMOSOMES]
    
            # Record best-ever raw score this iteration (stable metric)
            self.fitness_history.append(best_ever.RawScore())

            # Update elite only if the new best is genuinely better on
            # a stable, normalization-independent scale. Using the
            # normalized Fitnes here would be an apples-to-oranges
            # comparison, since normalization max values shift between
            # generations -- that was the root cause of the earlier
            # apparent "regression."
            if CRMs[0].RawScore() <= best_ever.RawScore():
                best_ever.copyfrom(CRMs[0])

        # Canonical labeling: use GA's genuine search result ONLY when
        # it's a meaningfully better solution than plain greedy. When
        # GA's result is only a near-tie with greedy (a marginal,
        # arbitrary tie-break among equally good options), use
        # greedy's consistent, repeatable assignment instead.
        #
        # Why: when multiple assignments are fitness-equivalent, GA's
        # choice among them is genuinely arbitrary, which makes the
        # dataset label unpredictable without making the schedule any
        # better. Canonicalizing to greedy's consistent pick in these
        # tie cases removes that arbitrary noise from the training
        # target while preserving every case where GA found a REAL,
        # meaningful improvement.
        greedy_score = greedy_only.RawScore()
        best_score = best_ever.RawScore()
        if greedy_score > 0:
            improvement = (greedy_score - best_score) / greedy_score
        else:
            improvement = 0.0

        if improvement > CANONICAL_LABEL_THRESHOLD:
            self.BESTCRM = best_ever
        else:
            self.BESTCRM = greedy_only
    
    def returnBESTCRM(self):
        return self.BESTCRM
