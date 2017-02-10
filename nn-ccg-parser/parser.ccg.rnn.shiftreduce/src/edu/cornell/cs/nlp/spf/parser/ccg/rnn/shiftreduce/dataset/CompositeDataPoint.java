package edu.cornell.cs.nlp.spf.parser.ccg.rnn.shiftreduce.dataset;

import java.io.Serializable;
import java.util.LinkedList;
import java.util.List;

import com.google.common.base.Joiner;

import edu.cornell.cs.nlp.spf.base.token.TokenSeq;
import edu.cornell.cs.nlp.spf.parser.ParsingOp;
import edu.cornell.cs.nlp.spf.parser.ccg.shiftreduce.stacks.DerivationState;
import edu.cornell.cs.nlp.spf.parser.ccg.shiftreduce.stacks.DerivationStateHorizontalIterator;
import edu.cornell.cs.nlp.utils.composites.Pair;

/** Wraps all the decisions for a single sentence which share prefixes of state, action history
 * and word buffer, into one datapoint. 
 * 
 * @author  Dipendra Misra (dkm@cs.cornell.edu) */
public class CompositeDataPoint<MR> implements Serializable {
	
	private static final long serialVersionUID = -9116975645222558435L;

	/** sentence in the dataset from where its coming. */
	private final String sentence;
	
	/** the full parser state */
	private final DerivationState<MR> dstate;
	
	/** entire action history */
	private final List<ParsingOp<MR>> actionHistory;
	
	/** words and tags in the sentence in reverse. Pair is given by (word, tag) */
	private final List<Pair<String,String>>	buffer;

	/** list of decisions wrapped in this datapoint */
	private final List<CompositeDataPointDecision<MR>> decisions;
	
	public CompositeDataPoint(String sentence, TokenSeq tk, DerivationState<MR> dstate, 
							   List<ParsingOp<MR>> actionHistory, List<Pair<String, String>> buffer,
							   List<CompositeDataPointDecision<MR>> decisions) {
		this.sentence = sentence;
		this.dstate = dstate;
		this.actionHistory = actionHistory;
		this.buffer = buffer;
		this.decisions = decisions;
	}
	
	public String getSentence() {
		return this.sentence;
	}
	
	public DerivationState<MR> getState() {
		return this.dstate;
	}
	
	public List<ParsingOp<MR>> getActionHistory() {
		return this.actionHistory;
	}
	
	public List<Pair<String, String>> getBuffer() {
		return this.buffer;
	}
	
	public List<CompositeDataPointDecision<MR>> getDecisions() {
		return this.decisions;
	}
	
	@Override
	public String toString() {
		StringBuilder s = new StringBuilder();
		
		s.append("{ ");
		
		//convert state to string
		DerivationStateHorizontalIterator<MR> it = dstate.horizontalIterator();
		String stateString = " ]";
		boolean first = true;
		while(it.hasNext()) {
			DerivationState<MR> dstate = it.next();
			
			if(first && dstate.getRightCategory() != null) {
				first = false;
				stateString = dstate.getRightCategory().toString();
			} 
			
			if(dstate.getLeftCategory() != null) {
				stateString = dstate.getLeftCategory() + ", " + stateString;
			}
		}
		
		stateString = "[ " + stateString;
		s.append(stateString + "\n");
		
		//convert action history to string
		s.append("[ " + Joiner.on("\n").join(this.actionHistory) + " ]\n");
		
		//convert buffer to string
		s.append("[ " + Joiner.on(" ").join(this.buffer) + " ]\n");
		
		//convert decisions to string
		s.append("[ " + Joiner.on("\n").join(this.decisions) + " ]\n");

		s.append("}");
		
		return s.toString();
	}
	
	public static class Builder<MR> {
		
		private final String sentence;
		
		private final TokenSeq tk;
		
		private final DerivationState<MR> dstate;
		
		private final List<ParsingOp<MR>> actionHistory;
		
		private final List<Pair<String, String>> buffer;
		
		private final List<CompositeDataPointDecision<MR>> decisions;
		
		/** number of categories in the state */
		private final int numCategories;
		
		public Builder(String sentence, TokenSeq tk, DerivationState<MR> dstate,
				List<ParsingOp<MR>> actionHistory, List<Pair<String, String>> buffer) {
			this.sentence = sentence;
			this.tk = tk;
			this.dstate = dstate;
			this.actionHistory = actionHistory;
			this.buffer = buffer;
			this.decisions = new LinkedList<CompositeDataPointDecision<MR>>();
			
			int numCategories = 0;
			DerivationStateHorizontalIterator<MR> it = dstate.horizontalIterator();
			
			boolean first = true;
			while(it.hasNext()) {
				DerivationState<MR> tmpState = it.next();
				
				if(first && tmpState.getRightCategory() != null) {
					first = false;
					numCategories++;
				} 
				
				if(tmpState.getLeftCategory() != null) {
					numCategories++;
				}
			}
			
			this.numCategories = numCategories;
		}
		
		public Builder<MR> addDecision(CompositeDataPointDecision<MR> decision) {
			this.decisions.add(decision);
			return this;
		}
		
		public int numDecision() {
			return this.decisions.size();
		}
		
		public int numCategories() {
			return this.numCategories;
		}
		
		public CompositeDataPoint<MR> build(boolean optimize) {
			
			if(!optimize) {
				return this.build();
			}
			
			int maxSentenceIx = 0, maxActionHistoryIx = 0;
			
			for(CompositeDataPointDecision<MR> decision: this.decisions) {
				maxSentenceIx = Math.max(maxSentenceIx, decision.getSentenceIx());
				maxActionHistoryIx = Math.max(maxActionHistoryIx, decision.getActionHistoryIx());
			}
			
			//Truncate the buffer and action history
			List<Pair<String, String>> buffer = this.buffer.subList(0, maxSentenceIx);
			List<ParsingOp<MR>> actionHistory = this.actionHistory.subList(0, maxActionHistoryIx);
			
			return new CompositeDataPoint<MR>(this.sentence, this.tk, this.dstate, 
					  actionHistory, buffer, this.decisions);
		}
		
		public CompositeDataPoint<MR> build() {
			return new CompositeDataPoint<MR>(this.sentence, this.tk, this.dstate, 
											  this.actionHistory, this.buffer, this.decisions);
		}
	}
}	