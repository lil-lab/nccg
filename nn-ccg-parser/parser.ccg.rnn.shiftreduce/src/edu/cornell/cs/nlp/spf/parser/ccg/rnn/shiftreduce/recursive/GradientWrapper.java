package edu.cornell.cs.nlp.spf.parser.ccg.rnn.shiftreduce.recursive;

import java.io.Serializable;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/** Wraps a sum of gradient along with the number of terms in the summation */
public class GradientWrapper implements Serializable {
	
	private static final long serialVersionUID = -6937857682313816091L;
	private final INDArray sumGradient;
	private int freq;
	private final int dim;
	
	public GradientWrapper(int dim) {
		this.sumGradient = Nd4j.zeros(dim);
		this.freq = 0;
		this.dim = dim;
	}
	
	public void addGradient(INDArray gradient) {
		this.sumGradient.addi(gradient);
		this.freq++;
	}
	
	public INDArray getGradient() {
		return this.sumGradient;
	}

	public int numTerms() {
		return this.freq;
	}
	
	public void flush() {
		this.sumGradient.muli(0);
		this.freq = 0;
	}
	
	public  int getDimension() {
		return this.dim;
	}
}
