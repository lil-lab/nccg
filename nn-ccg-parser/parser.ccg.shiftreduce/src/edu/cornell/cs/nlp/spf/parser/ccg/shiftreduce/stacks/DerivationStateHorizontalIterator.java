package edu.cornell.cs.nlp.spf.parser.ccg.shiftreduce.stacks;

import java.util.Iterator;

/**
* Iterator for travelling from right to left in a derivation state.
* If there are n disjoint tree segments in this derivation state then 
* one will make n-1 such traversal. More precisely, if the root categories
* from left to right are: { a1 a2 a3 ...ak} an then derivation states from left to right
* have their left/right categories as: { [a1 a2] [a2 a3] [a3 a4] ... [ak-1 ak] }
* hence n-1 derivation states horizontally. 
* 
* This iterator does not explore the individual trees, only their roots are traversed. 
*  
* @author Dipendra Misra
*/
public class DerivationStateHorizontalIterator<MR> implements Iterator<DerivationState<MR>> {

	private DerivationState<MR> it;
	
	public DerivationStateHorizontalIterator(DerivationState<MR> dstate) {
		this.it = dstate;
	}
	
	@Override
	public boolean hasNext() {
		return (this.it == null) ? false : true;
	}

	@Override
	public DerivationState<MR> next() {
		DerivationState<MR> current = it;
		it = it.nextLeft;
		return current;
	}

}
