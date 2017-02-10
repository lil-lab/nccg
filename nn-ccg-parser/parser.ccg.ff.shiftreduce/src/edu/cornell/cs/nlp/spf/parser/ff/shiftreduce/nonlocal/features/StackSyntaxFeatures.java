package edu.cornell.cs.nlp.spf.parser.ff.shiftreduce.nonlocal.features;

import java.io.Serializable;
import java.util.stream.Collectors;

import edu.cornell.cs.nlp.spf.base.hashvector.IHashVector;
import edu.cornell.cs.nlp.spf.ccg.categories.syntax.Syntax;
import edu.cornell.cs.nlp.spf.parser.ccg.IParseStep;
import edu.cornell.cs.nlp.spf.parser.ccg.shiftreduce.stacks.DerivationState;
import edu.cornell.cs.nlp.spf.parser.ccg.shiftreduce.stacks.DerivationStateHorizontalIterator;

public class StackSyntaxFeatures<MR> implements AbstractNonLocalFeature<MR>, Serializable {

	private static final long serialVersionUID = -1053646662148990794L;
	
	private final static String TAG1 = "STACKSYNTAX1";
	private final static String TAG2 = "STACKSYNTAX2";
	private final static String TAG3 = "STACKSYNTAX3";
	private final static String ATTRIBTAG1 = "STACKATTRIB1";
	private final static String ATTRIBTAG2 = "STACKATTRIB2";
	private final static String ATTRIBTAG3 = "STACKATTRIB3";
	private final static String NoSyntax = "None";
	
	@Override
	public void add(DerivationState<MR> state, IParseStep<MR> parseStep, IHashVector features, String[] buffer,
			int bufferIndex, String[] tags) {
		
		final Syntax last, sndLast, thirdLast;
		
		if(state.getRightCategory() != null) {
			last = state.getRightCategory().getSyntax();
			sndLast = state.getLeftCategory().getSyntax();
			
			DerivationStateHorizontalIterator<MR> hit = state.horizontalIterator();
			hit.next(); //equals to state
			
			if(hit.hasNext()) {
				thirdLast = hit.next().getLeftCategory().getSyntax();
			} else {
				thirdLast = null;
			}
		} else {
			if(state.getLeftCategory() != null) {
				last = state.getLeftCategory().getSyntax();
				sndLast = null;
				thirdLast = null;
			} else {
				last = null;
				sndLast = null;
				thirdLast = null;
			}
		} 
		
		if(last == null) {
			features.add(TAG1, NoSyntax, 1.0);
			features.add(ATTRIBTAG1, NoSyntax, 1.0);
		} else {
			features.add(TAG1, last.stripAttributes().toString(), 1.0);
			final String attributeSeq = last.getAttributes().stream().sorted()
								.collect(Collectors.joining("+"));
			features.add(ATTRIBTAG1, attributeSeq, 1.0);
		}
		
		if(sndLast == null) {
			features.add(TAG2, NoSyntax, 1.0);
			features.add(ATTRIBTAG2, NoSyntax, 1.0);
		} else {
			features.add(TAG2, sndLast.stripAttributes().toString(), 1.0);
			final String attributeSeq = sndLast.getAttributes().stream().sorted()
										.collect(Collectors.joining("+"));
			features.add(ATTRIBTAG2, attributeSeq, 1.0);
		}
		
		if(thirdLast == null) {
			features.add(TAG3, NoSyntax, 1.0);
			features.add(ATTRIBTAG3, NoSyntax, 1.0);
		} else {
			features.add(TAG3, thirdLast.stripAttributes().toString(), 1.0);
			final String attributeSeq = thirdLast.getAttributes().stream().sorted()
										.collect(Collectors.joining("+"));
			features.add(ATTRIBTAG3, attributeSeq, 1.0);
		}
	}
}
