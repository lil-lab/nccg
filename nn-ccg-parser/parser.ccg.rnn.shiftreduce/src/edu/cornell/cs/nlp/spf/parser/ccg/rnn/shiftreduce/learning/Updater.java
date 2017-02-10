package edu.cornell.cs.nlp.spf.parser.ccg.rnn.shiftreduce.learning;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.Sqrt;
import org.nd4j.linalg.factory.Nd4j;

/** Performs first-order optimization step on a given n-dimensional vector. */
public class Updater {

	private final double learningRate;
	private final double threshold;
	private final double regularizer;
	
	public Updater(double learningRate, double threshold, double regularizer) {
		this.threshold = threshold;
		this.learningRate = learningRate;
		this.regularizer = regularizer;
	}
	
	public INDArray update(INDArray vector, INDArray grad, INDArray sumSquareGradient) {
		
		//Add regularization
		grad.addi(vector.mul(this.regularizer));
	
		//Clip the gradient
		double norm = grad.normmaxNumber().doubleValue();
		
		if(norm > this.threshold) {
			grad.muli(this.threshold/norm);
		}
		
		//Perform AdaGrad based SGD step
		sumSquareGradient.addi(grad.mul(grad));
		
		INDArray invertedLearningRate = Nd4j.getExecutioner()
											.execAndReturn(new Sqrt(sumSquareGradient.dup()))
											.divi(this.learningRate);
		
		grad.divi(invertedLearningRate);
		
		vector.subi(grad);
		
		return vector;
	}
	
}
