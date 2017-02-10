package edu.cornell.cs.nlp.spf.parser.ccg.shiftreduce.stacks;

import java.util.LinkedList;
import java.util.List;

/** Packs Multiple Derivation State into one. 
 *  In order for many states to be packed into one, they have to be 
 *  root-equivalent to each other.
 *  
 *  @author Dipendra Misra
 * */
public class PackedState<MR> {
	
	private DerivationState<MR> bestState; //best state packed in this set, Cannot be NULL 
	
	private List<DerivationState<MR>> dStateLs; //list of packed states, don't have to be ordered
	
	public PackedState(DerivationState<MR> dstate) {
		this.bestState = dstate;
		this.dStateLs = new LinkedList<DerivationState<MR>>();
		this.dStateLs.add(dstate);
		dstate.setPackedState(this);
	}
	
	public void add(DerivationState<MR> dstate) {
		/* does not assert root-equivalence due to cost. 
		 * Programmers should make sure that its the case */
		
		assert bestState.wordsConsumed == dstate.wordsConsumed;
		
		this.dStateLs.add(dstate);
		
		if(bestState.score < dstate.score) {
			bestState = dstate;
		}
		
		dstate.setPackedState(this);
	}
	
	public DerivationState<MR> getBestState() {
		return this.bestState;
	}
	
	public double getBestScore() {
		return this.bestState.score;
	}
	
	public int numPacked() {
		return this.dStateLs.size();
	}
	
	public boolean containsState(DerivationState<MR> dstate) {
		return this.dStateLs.stream().anyMatch(x -> x == dstate);
	}
	
	public String print() {
		StringBuilder s = new StringBuilder();
		s.append("Number of dstates: "+this.dStateLs.size() + ", Best State: " + bestState.hashCode() + ", {");
		for(DerivationState<MR> dstate: dStateLs) {
			s.append(dstate.hashCode() + " ");
		}
		s.append("}");
		return s.toString();
	}
	
	@Override
	public int hashCode() {
		return this.bestState.hashCode(); //best state is the proxy for this packed state
	}
	
	/** Two different packed states have different root categories. Hence, by definition they cannot 
	 * have the same best derivation state. Therefore, we can use simple derivation state root equality 
	 * as proxy for equality.*/
	@Override
	public boolean equals(Object obj) {
		
		if(obj.getClass()!= this.getClass())
			return false;
		
		@SuppressWarnings("unchecked")
		PackedState<MR> other = (PackedState<MR>)obj;
	
		return this.bestState.equals(other.bestState); 
	}
}
