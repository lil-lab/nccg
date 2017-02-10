package edu.cornell.cs.nlp.spf.parser.ccg.rnn.shiftreduce.wordembeddings;

import java.io.IOException;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.HashMap;
import java.util.Hashtable;
import java.util.Iterator;
import java.util.List;

import org.junit.Assert;

/** Glove embeddings */
public class Glove implements WordEmbedding {
	
	private final Hashtable<String, HashMap<Integer,Double>> hashWordEmbeddings;
	private final int dim;
	
	/** reads word embeddings from a file of format  word: val1,val2,val3,.....valn\n ...
	 * @throws IOException */
	public Glove(String fileName) throws IOException {
		
		this.hashWordEmbeddings = new Hashtable<String, HashMap<Integer,Double>>();
		List<String> lines = Files.readAllLines(Paths.get(fileName),Charset.defaultCharset());
		
		Iterator<String> it = lines.iterator();
		int dim_ = -1; //intermediate
		
		while(it.hasNext()) {
			String line = it.next(); 
			//line format=>    word:val1,val2,....valk\n
			String[] args = line.split(":|,");
			String word = args[0];
			
			if(dim_ == -1) {
				dim_ = args.length - 1;
			}
			else {
				assert dim_ == args.length - 1;
			}
			
			final HashMap<Integer, Double> wordVector = new HashMap<Integer, Double>();
			//double[] wordVector = new double[dim_]; 
			for(int i = 1; i < dim_; i++) {
				wordVector.put(i, Double.parseDouble(args[i]));
				//wordVector[i] = Double.parseDouble(args[i]);
			}
			
			this.hashWordEmbeddings.put(word, wordVector);
		}
		
		this.dim = dim_;
		//embedding of unknown word
		final HashMap<Integer, Double> unkVector = new HashMap<Integer, Double>();
		/*double[] wordEmbedding = new double[this.dim];
		Arrays.fill(wordEmbedding, 0);
		this.hashWordEmbeddings.put(UNK, wordEmbedding);*/
		this.hashWordEmbeddings.put(UNK, unkVector);
	}
	
	public int dimension() {
		return this.dim;
	}
	
	public boolean find(String word) {
		HashMap<Integer, Double> result = this.hashWordEmbeddings.get(word);
		
		if(result == null)
			return false;
		else return true;
	}
	
	public HashMap<Integer, Double> getWordEmbedding(String word) {
		HashMap<Integer, Double> embedding = this.hashWordEmbeddings.get(word);
		
		if(embedding ==  null) {
			embedding = this.hashWordEmbeddings.get(UNK);
			Assert.assertNotNull(embedding);
		}
		
		return embedding;
	}

}
