package edu.cornell.cs.nlp.spf.parser.ccg.rnn.shiftreduce.embeddings;

/** parent interface for all form of embedding */
public interface AbstractEmbedding {
	
	/** size of embedding */
	public int getDimension();
	
	/** generic embedding function*/
	public Object getEmbedding(Object obj);

}
