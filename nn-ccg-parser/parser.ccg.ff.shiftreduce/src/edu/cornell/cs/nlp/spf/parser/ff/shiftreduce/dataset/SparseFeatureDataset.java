package edu.cornell.cs.nlp.spf.parser.ff.shiftreduce.dataset;

import java.util.List;

import edu.cornell.cs.nlp.spf.base.hashvector.IHashVector;
import edu.cornell.cs.nlp.spf.parser.ParsingOp;

/** Every decision is represented by a unique feature of this dataset. */
public class SparseFeatureDataset<MR> {

	private final List<IHashVector> possibleActionFeatures;
	
	private final int gTruthIx;
	
	private final String sentence;
	
	private final List<ParsingOp<MR>> possibleActions;
	
	public SparseFeatureDataset(List<IHashVector> possibleActionFeatures, int gTruthIx, 
								String sentence, List<ParsingOp<MR>> possibleActions) {
		
		this.possibleActionFeatures = possibleActionFeatures;
		this.gTruthIx = gTruthIx;
		this.sentence = sentence;
		this.possibleActions = possibleActions;
	}
	
	public List<IHashVector> getPossibleActionFeatures() {
		return this.possibleActionFeatures;
	}
	
	public int getGroundTruthIndex() {
		return this.gTruthIx;
	}
	
	public String getSentence() {
		return this.sentence;
	}
	
	public List<ParsingOp<MR>> getPossibleActions() {
		return this.possibleActions;
	}
}
