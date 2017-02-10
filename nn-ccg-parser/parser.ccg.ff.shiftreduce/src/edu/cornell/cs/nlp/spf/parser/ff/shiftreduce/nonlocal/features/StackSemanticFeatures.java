package edu.cornell.cs.nlp.spf.parser.ff.shiftreduce.nonlocal.features;

import edu.cornell.cs.nlp.spf.base.hashvector.IHashVector;
import edu.cornell.cs.nlp.spf.mr.lambda.Lambda;
import edu.cornell.cs.nlp.spf.mr.lambda.Literal;
import edu.cornell.cs.nlp.spf.mr.lambda.LogicalConstant;
import edu.cornell.cs.nlp.spf.mr.lambda.LogicalExpression;
import edu.cornell.cs.nlp.spf.parser.ccg.IParseStep;
import edu.cornell.cs.nlp.spf.parser.ccg.shiftreduce.stacks.DerivationState;
import edu.cornell.cs.nlp.spf.parser.ccg.shiftreduce.stacks.DerivationStateHorizontalIterator;
import edu.uw.cs.lil.amr.lambda.AMRServices;

public class StackSemanticFeatures implements AbstractNonLocalFeature<LogicalExpression> {

	private static final long serialVersionUID = 2788018328296860850L;
	private final static String TYPETAG1 = "STACKTYPE1";
	private final static String TYPETAG2 = "STACKTYPE2";
	private final static String TYPETAG3 = "STACKTYPE3";
	
	private final static String HPREDICATE1 = "HPREDICATE1";
	private final static String HPREDICATE2 = "HPREDICATE2";
	private final static String HPREDICATE3 = "HPREDICATE3";
	
	private final static String NONE = "None";
	
	private void setHeadPredicate(LogicalExpression semantics, IHashVector features, String tag, String inactiveToken) {
		
		if (semantics != null) {
			final LogicalConstant instanceType;
			if (AMRServices.isSkolemTerm(semantics)) {
				instanceType = AMRServices
						.getTypingPredicate((Literal) semantics);
			} else if (AMRServices.isSkolemTermBody(semantics)) {
				instanceType = AMRServices
						.getTypingPredicate((Lambda) semantics);
			} else {
				return;
			}
			if (instanceType != null) {
				features.set(tag, instanceType.getBaseName(), 1.0);
			}
		} else {
			features.set(tag, inactiveToken, 1.0);
		}
	}
	
	@Override
	public void add(DerivationState<LogicalExpression> state, IParseStep<LogicalExpression> parseStep, IHashVector features, String[] buffer,
			int bufferIndex, String[] tags) {
		
		final LogicalExpression last, sndLast, thirdLast;
		
		if(state.getRightCategory() != null) {
			last = state.getRightCategory().getSemantics();
			sndLast = state.getLeftCategory().getSemantics();
			
			DerivationStateHorizontalIterator<LogicalExpression> hit = state.horizontalIterator();
			hit.next(); //equals to state
			
			if(hit.hasNext()) {
				thirdLast = hit.next().getLeftCategory().getSemantics();
			} else {
				thirdLast = null;
			}
		} else {
			if(state.getLeftCategory() != null) {
				last = state.getLeftCategory().getSemantics();
				sndLast = null;
				thirdLast = null;
			} else {
				last = null;
				sndLast = null;
				thirdLast = null;
			}
		}
		
		// Type features: Triggers on the type
		if(last == null) {
			features.add(TYPETAG1, NONE, 1.0);
		} else {
			features.add(TYPETAG1, last.getType().toString(), 1.0);
		}
		
		if(sndLast == null) {
			features.add(TYPETAG2, NONE, 1.0);
		} else {
			features.add(TYPETAG2, sndLast.getType().toString(), 1.0);
		}
		
		if(thirdLast == null) {
			features.add(TYPETAG3, NONE, 1.0);
		} else {
			features.add(TYPETAG3, thirdLast.getType().toString(), 1.0);
		}
		
		// Head Predicate Features: Triggers on the head predicate
		this.setHeadPredicate(last, features, HPREDICATE1, NONE);
		this.setHeadPredicate(sndLast, features, HPREDICATE2, NONE);
		this.setHeadPredicate(thirdLast, features, HPREDICATE3, NONE);
		
	}
}
