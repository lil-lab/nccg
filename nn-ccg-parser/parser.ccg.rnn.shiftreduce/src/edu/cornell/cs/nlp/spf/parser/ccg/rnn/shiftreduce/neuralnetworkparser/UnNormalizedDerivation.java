package edu.cornell.cs.nlp.spf.parser.ccg.rnn.shiftreduce.neuralnetworkparser;

import edu.cornell.cs.nlp.spf.parser.ccg.shiftreduce.stacks.DerivationState;
import edu.cornell.cs.nlp.spf.parser.ccg.shiftreduce.steps.AbstractShiftReduceStep;

/** Packs a derivation state, its parser step and unnormalized score */
public class UnNormalizedDerivation<MR> {

	private final DerivationState<MR> dstate;
	private final double unNormalizedProb;
	private final AbstractShiftReduceStep<MR> step;
	
	public UnNormalizedDerivation(DerivationState<MR> dstate, double unNormalizedProb,  
										 AbstractShiftReduceStep<MR> step) {
		this.dstate = dstate;
		this.unNormalizedProb = unNormalizedProb;
		this.step = step;
	}
	
	public DerivationState<MR> getDState() {
		return this.dstate;
	}
	
	public double getUnNormalizedProb() {
		return this.unNormalizedProb;
	}
	
	public AbstractShiftReduceStep<MR> getStep() {
		return this.step;
	}
}
