# FogMLS vs. iFogSim Cross-Validation

Reproduces FogMLS's simulation results (GA and Round Robin schedulers) inside
iFogSim, the foundational fog computing simulator, to validate that FogMLS's
underlying computational model is realistic and consistent with established
tools.

## Setup

Requires a cloned [iFogSim repository](https://github.com/Cloudslab/iFogSim)
with JDK installed (tested on OpenJDK 21).

**Important:** replace iFogSim's own `src/org/fog/placement/Controller.java`
with `Controller_PATCHED_replace_org_fog_placement_Controller_java.java` from
this folder. This patch fixes a real bug: iFogSim's `Controller.java` calls
`System.exit(0)` directly inside its `STOP_SIMULATION` event handler, which
silently kills the JVM before any code written after
`CloudSim.startSimulation()` in a normal `main()` method can execute. The
patch calls our results printer from *inside* that handler, before the exit.

Place all `.java` files from this folder into
`iFogSim/src/org/fog/test/perfeval/`.

## Running

Set `RM_TYPE` in `FogMLSConfig.java` (`GA_RM` or `RR_RM`), then:

```bash
cd iFogSim
CP=$(find jars -name "*.jar" | tr '\n' ':')
find src -name "*.java" > sources.txt
javac -d build -cp "build:$CP" @sources.txt
java -cp "build:$CP:." org.fog.test.perfeval.Main
```

## Confirmed results (Aug 2026, independently reproduced on two separate
## HPC/sandbox environments)

| Scheduler | Avg latency |
|---|---|
| GA | 13.898916... |
| Round Robin | 17.277-17.30 |
