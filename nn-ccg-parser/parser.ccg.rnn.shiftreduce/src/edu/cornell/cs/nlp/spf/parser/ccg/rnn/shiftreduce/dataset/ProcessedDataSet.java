package edu.cornell.cs.nlp.spf.parser.ccg.rnn.shiftreduce.dataset;

import java.io.Serializable;
import java.util.List;

import com.google.common.base.Joiner;

import edu.cornell.cs.nlp.spf.base.token.TokenSeq;
import edu.cornell.cs.nlp.spf.parser.ParsingOp;
import edu.cornell.cs.nlp.spf.parser.ccg.shiftreduce.stacks.DerivationState;
import edu.cornell.cs.nlp.spf.parser.ccg.shiftreduce.stacks.DerivationStateHorizontalIterator;

/** Each point in the processed dataset represents a word-buffer, set of parsing actions 
 * and derivation state. */
public class ProcessedDataSet<MR> implements Serializable {

	private static final long serialVersionUID = 2661489662172043584L;

	/** sentence in the dataset from where its coming. */
	private final String	 		  sentence;
	
	/** current derivation (i.e. parser) state*/
	private final DerivationState<MR> dstate;
	
	/** list of actions taken so far */
	private final List<ParsingOp<MR>> actionHistory;
	
	/** words on the buffer */
	private final List<String> 		  buffer;
	
	/** list of actions taken so far */
	private final List<ParsingOp<MR>> possibleActions;
	
	/** ground truth action */
	private final int gTruthIx;
	
	/** Token seq representing the sentence. Needed for lexical entry embedding*/
	private final TokenSeq tk;
	
	
	public ProcessedDataSet(DerivationState<MR> dstate, List<ParsingOp<MR>> actionHistory, 
			                List<String> buffer, List<ParsingOp<MR>> possibleActions, int gTruthIx, 
			                List<String> sentence, TokenSeq tk) {
		
		this.dstate = dstate;
		this.actionHistory = actionHistory;
		this.buffer = buffer;
		this.possibleActions = possibleActions;
		this.gTruthIx = gTruthIx;
		this.sentence = Joiner.on(" ").join(sentence);
		this.tk = tk;
	}
	
	public DerivationState<MR> getState() {
		return this.dstate;
	}
	
	public List<ParsingOp<MR>> getActionHistory() {
		return this.actionHistory;
	}
	
	public List<String> getBuffer() {
		return this.buffer;
	}
	
	public List<ParsingOp<MR>> getPossibleActions() {
		return this.possibleActions;
	}
	
	public int getGTruthIx() {
		return this.gTruthIx;
	}
	
	public String getSentence() {
		return this.sentence;
	}
	
	public TokenSeq getTokenSeq() {
		return this.tk;
	}
	
	@Override
	public String toString() {
		
		StringBuffer s = new StringBuffer();
		s.append("{ Sentence: "+this.sentence+"\n buffer: "+Joiner.on(" ").join(this.buffer));
		s.append(", Actions: [");
		for(ParsingOp<MR> op: this.actionHistory) {
			s.append(op+"\n");
		}
		s.append("], ground-truth "+this.gTruthIx+", State: [");
		
		DerivationStateHorizontalIterator<MR> it = this.dstate.horizontalIterator();
		boolean first = true;
		while(it.hasNext()) {
			DerivationState<MR> dstate = it.next();
			if(first) {
				first = false;
				if(dstate.getRightCategory()!= null) {
					s.append(dstate.getRightCategory()+", ");
				}
			}
			
			s.append(dstate.getLeftCategory()+", ");			
		}
		
		s.append("], Possible Actions: [");
		for(ParsingOp<MR> op: this.possibleActions) {
			s.append(op+"\n");
		}
		s.append("]}");
		
		return s.toString();
	}
	
}
