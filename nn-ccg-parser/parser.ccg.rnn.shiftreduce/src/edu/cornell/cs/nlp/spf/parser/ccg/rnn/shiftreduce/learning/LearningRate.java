package edu.cornell.cs.nlp.spf.parser.ccg.rnn.shiftreduce.learning;

import java.io.Serializable;

/** A class that captures temporal change in learning rate */
public class LearningRate implements Serializable {
	
	private static final long serialVersionUID = 4808268899154898185L;
	private final double initialLearningRate;
	private final double learningRateDecay;
	
	private int time = 0;
	
	private Double currentLearningRate;
	
	public LearningRate(double initialLearningRate, double learningRateDecay) {
		this.initialLearningRate = initialLearningRate;
		this.learningRateDecay = learningRateDecay;
		this.currentLearningRate = initialLearningRate;
	}
	
	public void decay() {
		this.currentLearningRate = null;
		this.time++;
	}
	
	/** Returns learning rate given by 
	 * lr(t) = 1/(1+decay*t) lr(0) */
	public double getLearningRate() {
		
		if(this.currentLearningRate != null) {
			return this.currentLearningRate;
		} else {
			double learningRate = 
					this.initialLearningRate/(double)(1+ this.learningRateDecay*this.time);
			this.currentLearningRate = learningRate;
			return learningRate;
		}
	}

}
