package edu.cornell.cs.nlp.spf.parser.ccg.rnn.shiftreduce.wordembeddings;

import java.util.HashMap;

import edu.cornell.cs.nlp.spf.parser.ccg.rnn.shiftreduce.embeddings.AbstractEmbedding;

/** Defines Word Embedding  */
public interface WordEmbedding extends AbstractEmbedding {

	/** unknown word for handling OOV*/
	public static final String UNK = "$_UNK_$";
	
	/** dimensionality of word-vector */
	public int dimension();
	
	/** finds if the word is present in the corpus */
	public boolean find(String word);
	
	/** fetch word-embedding */
	public HashMap<Integer,Double> getWordEmbedding(String word);
	
	@Override
	public default int getDimension() {
		return this.dimension();
	}
	
	@Override
	public default Object getEmbedding(Object obj) {
		return this.getWordEmbedding((String)obj);
	}
}
