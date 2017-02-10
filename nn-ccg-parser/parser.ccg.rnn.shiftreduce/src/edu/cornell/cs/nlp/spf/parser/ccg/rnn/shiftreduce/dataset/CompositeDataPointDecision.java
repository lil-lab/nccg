package edu.cornell.cs.nlp.spf.parser.ccg.rnn.shiftreduce.dataset;

import java.io.Serializable;
import java.util.List;

import edu.cornell.cs.nlp.spf.parser.ParsingOp;

/** This class represents a single decision in a composite data point. 
 * The action history, state and buffer are fixed in a composite datapoint. 
 * A composite data point decision represents an elementary parsing decision
 * and references the correct entries in the history, state and buffer using 
 * indices. */
public class CompositeDataPointDecision<MR> implements Serializable {
	
	private static final long serialVersionUID = 7444884583604251959L;

	/** list of actions taken so far. should not contain
	 * duplicate parsing ops. */
	private final List<ParsingOp<MR>> possibleActions;
	
	/** ground truth action. possibleActions[gTruthIx] represents
	 * the ground truth parsing operation. */
	private final int gTruthIx;
	
	/** index in the buffer till where to consider. The buffer is sentence
	 * in reverse. Say buffer is {x1, x2, x3, x4, x5} then sentenceIx is in {0...5}
	 * where 0 means that buffer is empty and 5 means the entire buffer is considered
	 * for this decision. */
	private final int sentenceIx;
	
	/** index in the action history till where to consider. If the action history
	 * has n actions {a1, a2, .. an} then actionHistoryIx is in {0..n} where 0 means
	 * that no action is considered and n means that all actions are considered. */
	private final int actionHistoryIx;
	
	/** index in the parser state till where to consider.  
	 * If the state has categories {c1, c2, c3 } then parserStateIx is in range {0..3} where
	 * 0 means that no category is considered and 3 means that all categories are considered. */
	private final int parserStateIx;
	
	public CompositeDataPointDecision(List<ParsingOp<MR>> possibleActions, int gTruthIx,
									  int sentenceIx, int actionHistoryIx, int parserStateIx) {
		
		if(gTruthIx < 0 || sentenceIx < 0 || actionHistoryIx < 0 || parserStateIx < 0) {
			throw new IllegalStateException("Indices in decision are negative.");
		}
		
		this.possibleActions = possibleActions;
		this.gTruthIx = gTruthIx;
		this.sentenceIx = sentenceIx;
		this.actionHistoryIx = actionHistoryIx;
		this.parserStateIx = parserStateIx;
	}
	
	public List<ParsingOp<MR>> getPossibleActions() {
		return this.possibleActions;
	}
	
	public int getGTruthIx() {
		return this.gTruthIx;
	}
	
	public int getSentenceIx() {
		return this.sentenceIx;
	}
	
	public int getActionHistoryIx() {
		return this.actionHistoryIx;
	}
	
	public int getParserStateIx() {
		return this.parserStateIx;
	}
	
	@Override
	public String toString() {
		StringBuilder s = new StringBuilder();
		s.append("[ GTruth :"+this.gTruthIx+"; " + this.possibleActions.get(this.gTruthIx) +"\n");
		int count = 0;
		for(ParsingOp<MR> op: this.possibleActions) {
			s.append((count++) + ". " + op + "\n");
		}
		
		s.append("SentenceIx " + this.sentenceIx + "; ");
		s.append("ActionHistoryIx " + this.actionHistoryIx + "; ");
		s.append("ParserStateIx " + this.parserStateIx + "]");
		
		return s.toString();
	}
}