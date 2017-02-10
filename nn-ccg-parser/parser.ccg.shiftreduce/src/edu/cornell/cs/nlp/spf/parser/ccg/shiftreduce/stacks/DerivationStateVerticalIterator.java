package edu.cornell.cs.nlp.spf.parser.ccg.shiftreduce.stacks;

import java.util.Iterator;

/**
* Vertically traverses the rightmost tree in this derivation state.
* This iterator explores one tree, for traversing over the root of these
* trees use Horizontal Iterator. Example, consider there are three trees in the
* derivation state t1, t2, t3 such that 
* t1: (a (b c))
* t2: (r (p (s t) q)
* t3: (u (v w))
* 
* then top root categories are { a, r, u} and the horizontal iterator will have states
* with their left/right categories as {[a r], [r u]}. While vertical iterator will
* allow you to explore the rightmost tree that is t3. Thus the left/right categories that
* one will encounter on exploring this tree will be: { [r,u], [v w], [r v] }. The iteration 
* pattern can make things difficult so exercise caution. However, the fact that every node 
* results from binary or unary rule simplifies things. 
* 
* @author Dipendra Misra
*/

public class DerivationStateVerticalIterator<MR> implements Iterator<DerivationState<MR>> {

	private DerivationState<MR> end;
	private DerivationState<MR> it;

	public DerivationStateVerticalIterator(DerivationState<MR> dstate) {
		this.it = dstate;
		this.end = dstate.nextLeft; //termination iteration when you reach the previous tree
		
		if(dstate.nextLeft != null || dstate.getRightCategory() != null) {
			throw new RuntimeException("Currently can only iterate over complete tree");
		}
	}
	
	@Override
	public boolean hasNext() {
		return (this.it == this.end) ? false : true;
	}

	@Override
	public DerivationState<MR> next() {
		DerivationState<MR> current = it;
		it = it.parent;
		return current;
	}
	
}
