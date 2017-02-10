package edu.cornell.cs.nlp.spf.parser.ccg.rnn.shiftreduce.recursive;

import org.nd4j.linalg.api.ndarray.INDArray;

public interface AbstractRecursiveTreeNetwork {
	
	/** perform forward pas through the tree t */
	public INDArray feedForward(Tree t);
	
	/** perform backpropagation through the tree t. Assumes that a forward
	 *  pass over the tree has been made and vectors are initialized correctly
	 *  by the values in the forward pass */
	public void backProp(Tree t, INDArray loss);

}
