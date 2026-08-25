package org.fog.test.perfeval;

/**
 * Main.java
 *
 * Unified entry point, matching FogMLS's real design exactly: ONE
 * config (FogMLSConfig.RM_TYPE), ONE class to run.
 *
 * To switch schedulers: edit RM_TYPE in FogMLSConfig.java, then
 * always run:
 *     java -cp ... org.fog.test.perfeval.Main
 */
public class Main {
	public static void main(String[] args) throws Exception {
		if (FogMLSConfig.RM_TYPE == FogMLSConfig.RR_RM) {
			System.out.println("Resource management strategy: RR_RM (Round Robin)");
			FogMLS_Comparison.main(args);
		} else if (FogMLSConfig.RM_TYPE == FogMLSConfig.GA_RM) {
			System.out.println("Resource management strategy: GA_RM");
			FogMLS_GA_Comparison.main(args);
		} else {
			throw new IllegalArgumentException(
				"RM_TYPE=" + FogMLSConfig.RM_TYPE + " not supported in this comparison build. " +
				"Only RR_RM (" + FogMLSConfig.RR_RM + ") and GA_RM (" + FogMLSConfig.GA_RM + ") " +
				"are implemented. Set RM_TYPE in FogMLSConfig.java to one of these.");
		}
	}
}
