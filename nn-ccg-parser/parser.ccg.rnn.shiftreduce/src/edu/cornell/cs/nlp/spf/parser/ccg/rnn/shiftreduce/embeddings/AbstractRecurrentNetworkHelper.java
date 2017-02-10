package edu.cornell.cs.nlp.spf.parser.ccg.rnn.shiftreduce.embeddings;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;

/** Interface to be implemented by any embedding class that uses recurrent network */
public interface AbstractRecurrentNetworkHelper {
	
	/** builds a multi-layer recurrent network from input nIn to nOut and initializes it */
	public MultiLayerNetwork buildRecurrentNetwork(int nIn, int nOut);
	
	/** Return all top layer embedding of a time-series (obj) */
	public Object getAllTopLayerEmbedding(Object obj);
	
	/** backprop the loss through the recurrent network and 
	 *  the structures below it. The loss corresponds to the final
	 *  and topmost layer of the RNN. */
	public void backprop(INDArray loss);
	
	/** backprop the loss through the recurrent network and 
	 *  the structures below it. The loss array corresponds
	 *  to the loss going into the top layer of the RNN. 
	 *  Length of the loss array is same as the length of RNN*/
	public void backprop(INDArray[] loss);
}
