package org.fog.test.perfeval;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

/**
 * GACore.java
 *
 * Direct Java port of FogMLS's real GeneticAlg.py (Chromosome +
 * GeneticAlgorithm classes). Formulas copied exactly, not
 * reimplemented from memory, to guarantee identical behavior to the
 * real FogMLS. Same strategy used successfully for the YAFS GA port.
 *
 * UPDATED: ports the FogMLS GA configuration with order-aware random
 * initialization and canonical labeling. The mutation step compares
 * raw scores of child and parent (fixed 16 November 2026).
 */
public class GACore {

    public static double ALPHA = FogMLSConfig.ALPHA;
    public static double BETA = FogMLSConfig.BETA;
    public static double GAMMA = FogMLSConfig.GAMMA;
    public static int N_CHROMOSOMES = FogMLSConfig.N_CHROMOSOMES;
    public static int MAX_ITERATIONS = FogMLSConfig.MAX_ITERATIONS;
    public static int TOURNAMENT_SIZE = FogMLSConfig.TOURNAMENT_SIZE;
    public static double LATENCY_ED_FR = FogMLSConfig.LATENCY_ED_FR;
    public static double LATENCY_CS_FR = FogMLSConfig.LATENCY_CS_FR;
    public static double CLOUD_COST_UNIT_TIME = FogMLSConfig.CLOUD_COST_UNIT_TIME;

    // Minimum fractional improvement (over pure greedy) that GA's
    // searched solution must achieve to be accepted as the canonical
    // label/schedule for a slot. Below this, greedy's consistent
    // assignment is used instead.
    public static final double CANONICAL_LABEL_THRESHOLD = 0.02;

    // How many top-scoring valid fog nodes to consider when building
    // a random chromosome in orderAwareRandomInit().
    public static final int TOP_K_RANDOM_CANDIDATES = 2;

    // Relative score difference below which two candidates are a
    // practical tie (broken deterministically) rather than
    // meaningfully different options (broken randomly, for real
    // exploration). Expressed as a fraction of the best candidate's
    // score.
    public static final double TIE_RELATIVE_TOLERANCE = 0.05;

    static Random rng = new Random(42);

    public static class Resource {
        double cpuSpeed, bandwidth, ram, exCapacity, busyPower;
        public Resource(double cpuSpeed, double bandwidth, double ram, double exCapacity, double busyPower) {
            this.cpuSpeed = cpuSpeed; this.bandwidth = bandwidth; this.ram = ram;
            this.exCapacity = exCapacity; this.busyPower = busyPower;
        }
        public double getCPUSpeed() { return cpuSpeed; }
        public double getBANDWIDTH() { return bandwidth; }
        public double getRAM() { return ram; }
        public double getExCapacity() { return exCapacity; }
    }

    public static class Job {
        double noInstructions, dataBytes;
        public Object ref;
        public Job(double noInstructions, double dataBytes, Object ref) {
            this.noInstructions = noInstructions; this.dataBytes = dataBytes; this.ref = ref;
        }
        public double getNoInstructions() { return noInstructions; }
        public double getDataBytes() { return dataBytes; }
    }

    public static double computeSwapTime(double noInstructions, double dataBytes, double ram) {
        double swapRamNo = (noInstructions + dataBytes) / ram;
        double ramSwapUnit = FogMLSConfig.RAM_SWAP_UNIT;
        if (swapRamNo > 1) return ramSwapUnit * swapRamNo;
        return ramSwapUnit;
    }

    public static double computeExecDuration(double noInstructions, double cpuSpeed, double swapTime) {
        return noInstructions / cpuSpeed + swapTime;
    }

    public static class Chromosome {
        int n_r, n_j;
        List<Resource> rs;
        List<Job> jbs;
        Resource cloud;
        double[][] tbl;
        double Fitnes = 0, LatencyFitnes = 0, CostFitnes = 0, EnergyFitnes = 0;

        public Chromosome(List<Resource> rs, List<Job> jbs, Resource cloud, boolean skipInit) {
            this.rs = rs; this.jbs = jbs; this.cloud = cloud;
            this.n_r = rs.size(); this.n_j = jbs.size();
            this.tbl = new double[n_j][n_r + 1];
            if (!skipInit) greedyInit();
        }

        public static Chromosome createEmpty(List<Resource> rs, List<Job> jbs, Resource cloud) {
            return new Chromosome(rs, jbs, cloud, true);
        }

        public void copyfrom(Chromosome p) {
            LatencyFitnes = p.LatencyFitnes; EnergyFitnes = p.EnergyFitnes;
            CostFitnes = p.CostFitnes; Fitnes = p.Fitnes;
            for (int i = 0; i < n_j; i++)
                for (int j = 0; j < n_r + 1; j++)
                    tbl[i][j] = p.tbl[i][j];
        }

        public void fitness() {
            Fitnes = 0;
            delayFitness();
            energyFitness();
            costFitness();
            Fitnes = LatencyFitnes * ALPHA + EnergyFitnes * BETA + CostFitnes * GAMMA;
        }

        /**
         * Combined weighted score using RAW (non population-relative)
         * fitness components. Unlike Fitnes -- which is normalized
         * against the current population's max values and therefore
         * shifts meaning between generations -- this score is stable
         * across generations and safe to use for elitism/canonical-
         * labeling comparisons.
         */
        public double rawScore() {
            return LatencyFitnes * ALPHA + EnergyFitnes * BETA + CostFitnes * GAMMA;
        }

        void delayFitness() {
            LatencyFitnes = 0;
            for (int jobIdx = 0; jobIdx < n_j; jobIdx++) {
                Integer FR_ID = getRs(jobIdx);
                if (FR_ID == null) continue;
                if (FR_ID < rs.size()) {
                    Resource r = rs.get(FR_ID);
                    double swapTime = getSwapTime(FR_ID, jobIdx);
                    LatencyFitnes += jbs.get(jobIdx).getNoInstructions() / r.getCPUSpeed()
                            + jbs.get(jobIdx).getDataBytes() / r.getBANDWIDTH()
                            + LATENCY_ED_FR + swapTime;
                } else {
                    double swapTime = computeSwapTime(jbs.get(jobIdx).getNoInstructions(), jbs.get(jobIdx).getDataBytes(), cloud.getRAM());
                    LatencyFitnes += jbs.get(jobIdx).getNoInstructions() / cloud.getCPUSpeed()
                            + jbs.get(jobIdx).getDataBytes() / cloud.getBANDWIDTH()
                            + LATENCY_CS_FR + swapTime;
                }
            }
        }

        void energyFitness() {
            EnergyFitnes = 0;
            for (int i = 0; i < n_j; i++) {
                Integer FR_ID = getRs(i);
                if (FR_ID == null || FR_ID >= rs.size()) continue;
                Resource r = rs.get(FR_ID);
                EnergyFitnes += jbs.get(i).getNoInstructions() / r.getCPUSpeed() * r.busyPower;
            }
        }

        void costFitness() {
            CostFitnes = 0;
            for (int i = 0; i < n_j; i++) {
                Integer FR_ID = getRs(i);
                if (FR_ID != null && FR_ID >= rs.size()) {
                    CostFitnes += jbs.get(i).getNoInstructions() / cloud.getCPUSpeed() * CLOUD_COST_UNIT_TIME;
                }
            }
        }

        public Integer getRs(int j) {
            for (int i = 0; i < n_r + 1; i++)
                if (tbl[j][i] > 0) return i;
            return null;
        }

        void greedyInit() {
            double[] fgCapacity = new double[n_r];
            for (int i = 0; i < n_r; i++) fgCapacity[i] = rs.get(i).getExCapacity();

            for (int jbIdx = 0; jbIdx < n_j; jbIdx++) {
                for (int rsIdx = 0; rsIdx < n_r + 1; rsIdx++) tbl[jbIdx][rsIdx] = 0;

                int bestDev = -1;
                double bestScore = Double.POSITIVE_INFINITY;
                for (int fgI = 0; fgI < n_r; fgI++) {
                    double swapTime = getSwapTime(fgI, jbIdx);
                    double execDuration = computeExecDuration(jbs.get(jbIdx).getNoInstructions(), rs.get(fgI).getCPUSpeed(), swapTime);
                    if (fgCapacity[fgI] >= execDuration) {
                        double score = execDuration / fgCapacity[fgI];
                        if (score < bestScore) { bestScore = score; bestDev = fgI; }
                    }
                }

                if (bestDev != -1) {
                    double swapTime = getSwapTime(bestDev, jbIdx);
                    double execDuration = computeExecDuration(jbs.get(jbIdx).getNoInstructions(), rs.get(bestDev).getCPUSpeed(), swapTime);
                    fgCapacity[bestDev] -= execDuration;
                    tbl[jbIdx][bestDev] = execDuration;
                } else {
                    double swapTime = getSwapTime(0, jbIdx);
                    double execDuration = computeExecDuration(jbs.get(jbIdx).getNoInstructions(), cloud.getCPUSpeed(), swapTime);
                    tbl[jbIdx][n_r] = execDuration;
                }
            }
            fitness();
        }

        /**
         * Order-aware random initialization, with deterministic
         * tie-break. Direct port of Python's order_aware_random_init().
         *
         * Processes jobs in QUEUE ORDER (same as greedyInit) and
         * tracks remaining capacity as it goes. For each job, scores
         * all fog nodes with sufficient capacity using the same
         * load-balanced score as greedyInit() (execDuration /
         * remainingCapacity, lower = better).
         *
         * Selection among valid nodes:
         * - Take the TOP_K_RANDOM_CANDIDATES best-scoring valid nodes.
         * - If their scores are a practical near-tie (within
         *   TIE_RELATIVE_TOLERANCE), break the tie DETERMINISTICALLY
         *   (always the lower device index).
         * - Otherwise, choose randomly among the top-k candidates --
         *   preserves genuine search diversity for crossover/mutation.
         */
        public void orderAwareRandomInit(Random rngIn) {
            double[] fgCapacity = new double[n_r];
            for (int i = 0; i < n_r; i++) fgCapacity[i] = rs.get(i).getExCapacity();

            for (int jbIdx = 0; jbIdx < n_j; jbIdx++) {
                for (int r = 0; r < n_r + 1; r++) tbl[jbIdx][r] = 0;

                // Score all fog nodes with enough remaining capacity.
                List<double[]> scoredDevs = new ArrayList<>(); // {score, fgIndex, execDuration}
                for (int fgI = 0; fgI < n_r; fgI++) {
                    double swapTime = getSwapTime(fgI, jbIdx);
                    double execDuration = computeExecDuration(
                        jbs.get(jbIdx).getNoInstructions(),
                        rs.get(fgI).getCPUSpeed(),
                        swapTime);
                    if (fgCapacity[fgI] >= execDuration) {
                        double score = execDuration / fgCapacity[fgI];
                        scoredDevs.add(new double[]{score, fgI, execDuration});
                    }
                }

                if (!scoredDevs.isEmpty()) {
                    scoredDevs.sort((a, b) -> Double.compare(a[0], b[0]));
                    int k = Math.min(TOP_K_RANDOM_CANDIDATES, scoredDevs.size());
                    List<double[]> topK = scoredDevs.subList(0, k);

                    double bestScore = topK.get(0)[0];
                    double worstInTopK = topK.get(topK.size() - 1)[0];
                    double[] chosenEntry;

                    if (topK.size() > 1 && bestScore > 0 &&
                        (worstInTopK - bestScore) / bestScore <= TIE_RELATIVE_TOLERANCE) {
                        // Practical near-tie: break deterministically,
                        // always the lower device index among tied
                        // candidates.
                        List<double[]> tied = new ArrayList<>();
                        for (double[] c : topK) {
                            if (bestScore > 0 &&
                                (c[0] - bestScore) / bestScore <= TIE_RELATIVE_TOLERANCE) {
                                tied.add(c);
                            }
                        }
                        chosenEntry = tied.get(0);
                        for (double[] c : tied) {
                            if (c[1] < chosenEntry[1]) chosenEntry = c;
                        }
                    } else {
                        // Meaningfully different options: genuine
                        // random exploration among them.
                        chosenEntry = topK.get(rngIn.nextInt(topK.size()));
                    }

                    int chosen = (int) chosenEntry[1];
                    double chosenExec = chosenEntry[2];
                    fgCapacity[chosen] -= chosenExec;
                    tbl[jbIdx][chosen] = chosenExec;
                } else {
                    // No fog node has capacity. Assign to cloud.
                    double swapTime = getSwapTime(0, jbIdx);
                    double execDuration = computeExecDuration(
                        jbs.get(jbIdx).getNoInstructions(),
                        cloud.getCPUSpeed(),
                        swapTime);
                    tbl[jbIdx][n_r] = execDuration;
                }
            }
            fitness();
        }

        public void crossover(int pt, Chromosome p2) {
            for (int i = pt; i < n_j; i++)
                for (int j = 0; j < n_r + 1; j++)
                    tbl[i][j] = p2.tbl[i][j];
            fitness();
        }

        public void mutation(Chromosome p) {
            for (int r = 0; r < n_r; r++) {
                double rTimeUsed = 0;
                for (int j = 0; j < n_j; j++) {
                    rTimeUsed += tbl[j][r];
                    if (rTimeUsed > rs.get(r).getExCapacity()) {
                        rTimeUsed -= tbl[j][r];
                        int rsid = findFogDev(j);
                        tbl[j][rsid] = tbl[j][r];
                        tbl[j][r] = 0;
                    }
                }
            }
            fitness();
            if (rawScore() > p.rawScore()) copyfrom(p);
        }

        int findFogDev(int jId) {
            for (int r = 0; r < n_r; r++) {
                double rTimeUsed = 0;
                for (int j = 0; j < n_j; j++) rTimeUsed += tbl[j][r];
                double remaining = rs.get(r).getExCapacity() - rTimeUsed;
                double swapTime = getSwapTime(r, jId);
                double execDuration = computeExecDuration(jbs.get(jId).getNoInstructions(), rs.get(r).getCPUSpeed(), swapTime);
                if (remaining >= execDuration) return r;
            }
            return n_r;
        }

        double getSwapTime(int dev, int jobid) {
            return computeSwapTime(jbs.get(jobid).getNoInstructions(), jbs.get(jobid).getDataBytes(), rs.get(dev).getRAM());
        }
    }

    public static class GeneticAlgorithm {
        List<Resource> rs;
        List<Job> jbs;
        Resource cloud;
        Chromosome bestCRM;

        public GeneticAlgorithm(List<Resource> rs, List<Job> jbs, Resource cloud) {
            this.rs = rs; this.jbs = jbs; this.cloud = cloud;
        }

        Chromosome selectChromosome(List<Chromosome> pop) {
            List<Chromosome> shuffled = new ArrayList<>(pop);
            Collections.shuffle(shuffled, rng);
            List<Chromosome> candidates = shuffled.subList(0, Math.min(TOURNAMENT_SIZE, pop.size()));
            Chromosome best = candidates.get(0);
            for (Chromosome c : candidates) if (c.Fitnes < best.Fitnes) best = c;
            return best;
        }

        Chromosome[] gaCrossover(Chromosome p1, Chromosome p2) {
            Chromosome c1 = Chromosome.createEmpty(rs, jbs, cloud);
            Chromosome c2 = Chromosome.createEmpty(rs, jbs, cloud);
            c1.copyfrom(p1);
            c2.copyfrom(p2);
            int crossoverPoint = jbs.size() > 1 ? rng.nextInt(jbs.size() - 1) : 0;
            c1.crossover(crossoverPoint, c2);
            c2.crossover(crossoverPoint, p1);
            return new Chromosome[]{c1, c2};
        }

        void normalizePopulation(List<Chromosome> crms) {
            double maxL = 0, maxE = 0, maxC = 0;
            for (Chromosome c : crms) {
                maxL = Math.max(maxL, c.LatencyFitnes);
                maxE = Math.max(maxE, c.EnergyFitnes);
                maxC = Math.max(maxC, c.CostFitnes);
            }
            if (maxL == 0) maxL = 1.0;
            if (maxE == 0) maxE = 1.0;
            if (maxC == 0) maxC = 1.0;
            for (Chromosome c : crms) {
                c.Fitnes = (c.LatencyFitnes / maxL) * ALPHA + (c.EnergyFitnes / maxE) * BETA + (c.CostFitnes / maxC) * GAMMA;
            }
        }

        public void generateOptimalChromosome() {
            if (jbs.isEmpty()) {
                bestCRM = Chromosome.createEmpty(rs, jbs, cloud);
                bestCRM.fitness();
                return;
            }

            List<Chromosome> crms = new ArrayList<>();
            Chromosome greedyOnly = null;

            for (int i = 0; i < N_CHROMOSOMES; i++) {
                Chromosome ch;
                if (i == 0) {
                    // Seed the population with one greedy (smart) solution.
                    ch = new Chromosome(rs, jbs, cloud, false); // runs greedyInit()
                    // Keep a separate, untouched copy of the pure
                    // greedy result -- fallback if GA's search doesn't
                    // find a MEANINGFULLY better solution.
                    greedyOnly = Chromosome.createEmpty(rs, jbs, cloud);
                    greedyOnly.copyfrom(ch);
                } else {
                    // Rest of population: order-aware random
                    // assignment for genuine diversity.
                    ch = Chromosome.createEmpty(rs, jbs, cloud);
                    ch.orderAwareRandomInit(rng);
                }
                crms.add(ch);
            }

            normalizePopulation(crms);
            crms.sort((a, b) -> Double.compare(a.Fitnes, b.Fitnes));

            // Elitism: track the best chromosome found so far.
            Chromosome bestEver = Chromosome.createEmpty(rs, jbs, cloud);
            bestEver.copyfrom(crms.get(0));

            int itr = 0;
            while (itr < MAX_ITERATIONS) {
                itr++;
                List<Chromosome> selected = new ArrayList<>();
                for (int i = 0; i < N_CHROMOSOMES; i++) selected.add(selectChromosome(crms));

                List<Chromosome> children = new ArrayList<>();
                for (int i = 0; i < N_CHROMOSOMES - 1; i += 2) {
                    Chromosome p1 = selected.get(i), p2 = selected.get(i + 1);
                    Chromosome[] c = gaCrossover(p1, p2);
                    c[0].mutation(p1);
                    c[1].mutation(p2);
                    children.add(c[0]);
                    children.add(c[1]);
                }

                // Inject the elite into this generation's pool so it
                // competes fairly under the same normalization, then
                // trim back down to N_CHROMOSOMES, keeping only the
                // best.
                Chromosome eliteCopy = Chromosome.createEmpty(rs, jbs, cloud);
                eliteCopy.copyfrom(bestEver);
                children.add(eliteCopy);

                normalizePopulation(children);
                children.sort((a, b) -> Double.compare(a.Fitnes, b.Fitnes));
                crms = new ArrayList<>(children.subList(0, N_CHROMOSOMES));

                // Update elite only if the new best is genuinely
                // better on a stable, normalization-independent scale.
                if (crms.get(0).rawScore() <= bestEver.rawScore()) {
                    bestEver.copyfrom(crms.get(0));
                }
            }

            // Canonical labeling: use GA's genuine search result ONLY
            // when it's a meaningfully better solution than plain
            // greedy. When GA's result is only a near-tie with greedy
            // (a marginal, arbitrary tie-break among equally good
            // options), use greedy's consistent, repeatable
            // assignment instead.
            double greedyScore = greedyOnly.rawScore();
            double bestScore = bestEver.rawScore();
            double improvement = (greedyScore > 0)
                ? (greedyScore - bestScore) / greedyScore
                : 0.0;

            if (improvement > CANONICAL_LABEL_THRESHOLD) {
                bestCRM = bestEver;
            } else {
                bestCRM = greedyOnly;
            }
        }

        public Chromosome returnBestCRM() { return bestCRM; }
    }
}
