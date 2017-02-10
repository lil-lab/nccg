package edu.cornell.cs.nlp.spf.parser.ccg.rnn.shiftreduce.recursive;

import org.nd4j.linalg.api.ndarray.INDArray;

/** Defines a network that takes a tree and outputs the average of all leaf nodes.
 * Provides functionality for composition and backprop. */
public class AveragingNetwork {

	public static void averageAndSet(Tree t) {
	
		INDArray sumVectors = sum(t);
		double n = (double)t.numLeaves() + 0.000001;
		
		t.setVector(sumVectors.divi(n));
	}
	
	public static INDArray average(Tree t) {
		
		INDArray sumVectors = sum(t);
		double n = (double)t.numLeaves() + 0.000001;
		
		return sumVectors.divi(n);
	}
	
	private static INDArray sum(Tree t) {
		
		if(t.numChildren() == 0) { //leaf vector, leaf vectors are already initialized
			
			return t.getVector();
		}
		else if(t.numChildren() == 2) { //binary
			
			//do these recursive calls in parallel in future
			INDArray left  = sum(t.getChild(0)); 
			INDArray right = sum(t.getChild(1));
			
			INDArray sum = left.dup();
			sum.addi(right);
			
			return sum;
		}
		
		throw new IllegalStateException("Binarize the tree");
	}
	
	public static void backprop(Tree t, INDArray error) {
		
		double n = (double)t.numLeaves() + 0.000001;
		INDArray errorLeaf = error.divi(n);
		propagateError(t, errorLeaf);
	}
	
	private static void propagateError(Tree t, INDArray error) {
		
		if(t.numChildren() == 0) {
			t.addGradient(error);
		} else {
			
			Tree left = t.getChild(0);
			Tree right = t.getChild(1);
			
			propagateError(left, error);
			propagateError(right, error);
		}
	}
}
