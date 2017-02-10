package edu.cornell.cs.nlp.spf.parser.ff.shiftreduce.dataset;

import java.util.Iterator;
import java.util.List;

import edu.cornell.cs.nlp.spf.base.hashvector.IHashVector;
import edu.cornell.cs.nlp.spf.mr.lambda.LogicalExpression;
import edu.cornell.cs.nlp.spf.parser.ParsingOp;
import edu.cornell.cs.nlp.spf.parser.ccg.shiftreduce.stacks.DerivationState;
import edu.cornell.cs.nlp.spf.parser.ccg.shiftreduce.stacks.DerivationStateHorizontalIterator;

/** Every decision is represented by a unique feature of this dataset. */
public class SparseFeatureAndStateDataset<MR> {

	private final List<IHashVector> possibleActionFeatures;
	
	private final int gTruthIx;
	
	private final String sentence;
	
	private final List<ParsingOp<MR>> possibleActions;
	
	private final IHashVector stateFeature;
	
	//TODO:  find a cleaner way to integrate these parameters into fewer, more readable ones
	private boolean setSemantics;
	private LogicalExpression last, sndLast, thirdLast;
	
	public SparseFeatureAndStateDataset(IHashVector stateFeature, List<IHashVector> possibleActionFeatures, int gTruthIx, 
								String sentence, List<ParsingOp<MR>> possibleActions) {
		
		this.stateFeature = stateFeature;
		this.possibleActionFeatures = possibleActionFeatures;
		this.gTruthIx = gTruthIx;
		this.sentence = sentence;
		this.possibleActions = possibleActions;
		
		this.setSemantics = false;
		this.last = null;
		this.sndLast = null;
		this.thirdLast = null;
	}
	
	public IHashVector getStateFeature() {
		return this.stateFeature;
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
	
	public void setSemantics(DerivationState<LogicalExpression> state) {
		
		if(this.setSemantics) {
			throw new RuntimeException("Cannot set semantics twice");
		}
		
		this.setSemantics = true;
		
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
		
		this.last = last;
		this.sndLast = sndLast;
		this.thirdLast = thirdLast;
	}
	
	public LogicalExpression getLastSemantics() {
		return this.last;
	}
	
	public LogicalExpression getSndLastSemantics() {
		return this.sndLast;
	}
	
	public LogicalExpression getThirdLastSemantics() {
		return this.thirdLast;
	}
	
	public boolean isSemanticsSet() {
		return this.setSemantics;
	}
	
	@Override
	public String toString() {
		StringBuilder s = new StringBuilder();
		
		s.append(this.sentence + "; gtruth " + this.gTruthIx + "\n");
		s.append(this.stateFeature + "; { \n ");
		
		Iterator<ParsingOp<MR>> actionIt = this.possibleActions.iterator();
		while(actionIt.hasNext()) {
			s.append(actionIt.next() +"\n");
		}
		
		s.append("}, { \n");
		
		Iterator<IHashVector> it = this.possibleActionFeatures.iterator();
		while(it.hasNext()) {
			s.append(it.next() + "\n");
		}
		s.append("}");
		
		return s.toString();
	}
	
	/** This is an ordering sensitive equals operation. That is, even though conceptually
	 * permutations of possibleActions does not change the datapoint we enforce that in equals
	 * for convenience. Moreoever, the way these datapoints are generated they follow fixed ordering
	 * on action features. */
	@Override
	public boolean equals(Object obj) {
		if (this == obj) {
			return true;
		}
		if (obj == null) {
			return false;
		}
		if (getClass() != obj.getClass()) {
			return false;
		}
		
		@SuppressWarnings("unchecked")
		final SparseFeatureAndStateDataset<MR> other = (SparseFeatureAndStateDataset<MR>)obj;

		if(this.gTruthIx  != other.gTruthIx) {
			return false;
		}
		
		if(!this.stateFeature.equals(other.stateFeature)) {
			return false;
		}
		
		if(this.possibleActionFeatures.size() != other.possibleActionFeatures.size()) {
			return false;
		}
		
		Iterator<IHashVector> it1 = this.possibleActionFeatures.iterator();
		Iterator<IHashVector> it2 = other.possibleActionFeatures.iterator();
		
		while(it1.hasNext()) {
			if(!it1.next().equals(it2.next())) {
				return false;
			}
		}
		
		if(!this.sentence.equals(other.sentence)) {
			return false;
		}
		
		Iterator<ParsingOp<MR>> actionIt1 = this.possibleActions.iterator();
		Iterator<ParsingOp<MR>> actionIt2 = other.possibleActions.iterator();
		
		while(actionIt1.hasNext()) {
			if(!actionIt1.next().equals(actionIt2.next())) {
				return false;
			}
		}
		
		if(this.setSemantics) {
			if((this.last == null && other.last != null) || (!this.last.equals(other.last))) {
				return false;
			}
			
			if((this.sndLast == null && other.sndLast != null) || (!this.sndLast.equals(other.sndLast))) {
				return false;
			}
			
			if((this.thirdLast == null && other.thirdLast != null) || (!this.thirdLast.equals(other.thirdLast))) {
				return false;
			}
		}
		
		return true;
	}
}
