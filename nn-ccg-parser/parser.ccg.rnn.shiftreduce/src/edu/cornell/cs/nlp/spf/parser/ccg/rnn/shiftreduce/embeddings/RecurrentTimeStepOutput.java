package edu.cornell.cs.nlp.spf.parser.ccg.rnn.shiftreduce.embeddings;

import java.util.Map;

import org.nd4j.linalg.api.ndarray.INDArray;

/** Wraps embedding and RNN state into one. Allows one to use the RNN state to continue
 * feed forwarding from the sequence that is encoded by the rnn state.*/
public class RecurrentTimeStepOutput {
	
	private final INDArray embedding;
	private final Map<String, INDArray>[] rnnState;
	
	public RecurrentTimeStepOutput(INDArray embedding, Map<String, INDArray>[] rnnState) {
		this.embedding = embedding;
		this.rnnState = rnnState;
	}
	
	public INDArray getEmbedding() {
		return this.embedding;
	}
	
	public Map<String, INDArray>[] getRNNState() {
		return this.rnnState;
	}

}
