package edu.cornell.cs.nlp.spf.parser.ff.shiftreduce.nonlocal.features;

import java.io.Serializable;

import edu.cornell.cs.nlp.spf.base.hashvector.IHashVector;
import edu.cornell.cs.nlp.spf.parser.ccg.IParseStep;
import edu.cornell.cs.nlp.spf.parser.ccg.shiftreduce.stacks.DerivationState;

/** Every non-local feature used by feed forward shift reduce neural parser
 * must implement this interface. A general non-local feature depends upon
 * the word buffer and derivation state on the stack. 
 * @author Dipendra Misra
 *  */
public interface AbstractNonLocalFeature<MR> extends Serializable {

	/** Add feature based on derivation state representing the stack, sentence on the buffer along
	 * with bufferIndex in [0, sentence-len] which tells how much of the sentence has been read (0 means
	 * not read anything and sentence-length means the entire sentence has been read. The features
	 * are added to features datastructure. Each string in buffer represents string representation of a token.*/
	void add(DerivationState<MR> state, IParseStep<MR> parseStep, IHashVector features, String[] buffer, int bufferIndex, String[] tags);
}
