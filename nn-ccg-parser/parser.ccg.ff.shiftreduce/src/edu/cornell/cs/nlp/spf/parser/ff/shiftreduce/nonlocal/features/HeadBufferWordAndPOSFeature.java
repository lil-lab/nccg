package edu.cornell.cs.nlp.spf.parser.ff.shiftreduce.nonlocal.features;

import edu.cornell.cs.nlp.spf.base.hashvector.IHashVector;
import edu.cornell.cs.nlp.spf.parser.ccg.IParseStep;
import edu.cornell.cs.nlp.spf.parser.ccg.shiftreduce.stacks.DerivationState;

public class HeadBufferWordAndPOSFeature<MR> implements AbstractNonLocalFeature<MR> {

	private static final long serialVersionUID = -5644574694666143473L;
	private static final String WORD1 = "HDWORD1";
	private static final String WORD2 = "HDWORD2";
	private static final String NOWORD = "HDWORDNONE";
	
//	private static final String POS1 = "HDPOS1";
//	private static final String POS2 = "HDPOS2";
//	private static final String NOPOS = "HDPOSNONE";
	
	@Override
	public void add(DerivationState<MR> state, IParseStep<MR> parseStep, IHashVector features, String[] buffer,
			int bufferIndex, String[] tags) {
		
		final int n = buffer.length;
		
		if(bufferIndex == n) { //no words left
			features.add(WORD1, NOWORD, 1.0);
			features.add(WORD2, NOWORD, 1.0);
			
//			features.add(POS1, NOPOS, 1.0);
//			features.add(POS2, NOPOS, 1.0);
			
		} else if (bufferIndex == n -1) { //one word left
			features.add(WORD1, buffer[bufferIndex], 1.0);
			features.add(WORD2, NOWORD, 1.0);
			
//			features.add(POS1, tags[bufferIndex], 1.0);
//			features.add(POS2, NOPOS, 1.0);
		} else { //two or more words left
			features.add(WORD1, buffer[bufferIndex], 1.0);
			features.add(WORD2, buffer[bufferIndex + 1], 1.0);
			
//			features.add(POS1, tags[bufferIndex], 1.0);
//			features.add(POS2, tags[bufferIndex + 1], 1.0);
		}
	}
}