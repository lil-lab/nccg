package edu.cornell.cs.nlp.spf.parser.ccg.rnn.shiftreduce.wordembeddings;

import java.util.HashMap;
import java.util.Hashtable;
import java.util.Iterator;
import java.util.LinkedHashSet;

import org.springframework.util.Assert;

/** implements word-embeddings using one-of-k coding */
public class OneOfKCoding implements WordEmbedding {

	private final LinkedHashSet<String> vocabulary;
	private final Hashtable<String,HashMap<Integer, Double>> hashWordEmbeddings;
	
	public OneOfKCoding(LinkedHashSet<String> vocabulary) {
		this.vocabulary = vocabulary; //must not contain unknown word
		this.hashWordEmbeddings = new Hashtable<String, HashMap<Integer, Double>>();
		
		//preprocess hashtable 
		Iterator<String> it = vocabulary.iterator();
		
		int index = 0;
		
		while(it.hasNext()) {
			HashMap<Integer, Double> wordEmbedding = new HashMap<Integer, Double>();
			wordEmbedding.put(index, 1.0);
			index++;
			
			this.hashWordEmbeddings.put(it.next(), wordEmbedding);
		}
		
		//embedding of unknown word
		HashMap<Integer, Double> wordEmbedding = new HashMap<Integer, Double>();
		this.hashWordEmbeddings.put(UNK, wordEmbedding);
	}
	
	public int dimension() {
		return this.vocabulary.size();
	}
	
	public boolean find(String word) {
		HashMap<Integer, Double> result = this.hashWordEmbeddings.get(word);
		
		if(result == null)
			return false;
		else return true;
	}
	
	public HashMap<Integer, Double> getWordEmbedding(String word) {
		
		HashMap<Integer, Double> embedding = this.hashWordEmbeddings.get(word);
		
		if(embedding == null) {
			embedding = this.hashWordEmbeddings.get(UNK);
			Assert.notNull(embedding);
		}
		
		return embedding;
	}
	
}
