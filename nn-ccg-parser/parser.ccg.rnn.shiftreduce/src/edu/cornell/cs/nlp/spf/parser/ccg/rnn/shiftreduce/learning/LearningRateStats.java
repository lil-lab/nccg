package edu.cornell.cs.nlp.spf.parser.ccg.rnn.shiftreduce.learning;

/** A small class that measures the min and max learning rate. This to see if an adaptive learning rate
 * alg. is being over aggressive */
public class LearningRateStats {

	double minLearningRate;
	double maxLearningRate;
	
	public LearningRateStats() {
		this.minLearningRate = Double.MAX_VALUE;
		this.maxLearningRate = 0;
	}
	
	public void min(double learningRate) {
		this.minLearningRate = Math.min(this.minLearningRate, learningRate);
	}
	
	public void max(double learningRate) {
		this.maxLearningRate = Math.max(this.maxLearningRate, learningRate);	
	}
	
	@Override
	public String toString() {
		return " min learning rate " + this.minLearningRate + " max learning rate " + this.maxLearningRate;
	}
	
	public void unset() {		
		this.minLearningRate = Double.MAX_VALUE;
		this.maxLearningRate = 0;
	}
}
