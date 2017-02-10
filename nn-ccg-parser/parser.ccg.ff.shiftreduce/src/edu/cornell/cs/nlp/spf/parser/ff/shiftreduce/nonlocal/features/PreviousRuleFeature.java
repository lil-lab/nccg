package edu.cornell.cs.nlp.spf.parser.ff.shiftreduce.nonlocal.features;

import edu.cornell.cs.nlp.spf.base.hashvector.IHashVector;
import edu.cornell.cs.nlp.spf.parser.ccg.IParseStep;
import edu.cornell.cs.nlp.spf.parser.ccg.rules.RuleName;
import edu.cornell.cs.nlp.spf.parser.ccg.shiftreduce.stacks.DerivationState;
import edu.cornell.cs.nlp.spf.parser.ccg.shiftreduce.steps.IWeightedShiftReduceStep;
import edu.cornell.cs.nlp.spf.parser.ccg.shiftreduce.steps.ShiftReduceLexicalStep;

/** A non-local feature that takes into account the previous rule-name */
public class PreviousRuleFeature<MR> implements AbstractNonLocalFeature<MR> {

	private static final long serialVersionUID = -5230921231750562885L;
	private static final String	TAG				= "PREVRULE";
	
	@Override
	public void add(DerivationState<MR> state, IParseStep<MR> parseStep, IHashVector features, String[] buffer,
			int bufferIndex, String[] tags) {
		
		IWeightedShiftReduceStep<MR> step = state.returnStep();
		
		if(step == null) {
			return;
		}
		
		RuleName ruleName = step.getRuleName();
		if(ruleName.equals(ShiftReduceLexicalStep.LEXICAL_DERIVATION_STEP_RULENAME)) {
			features.add(TAG, "shift-lexical", 1.0);
		} else {
			features.add(TAG, ruleName.toString(), 1.0);
		}
	}
}
