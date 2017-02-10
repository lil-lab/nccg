package edu.cornell.cs.nlp.spf.parser.ccg.rnn.shiftreduce.wordembeddings;

import java.io.IOException;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Hashtable;
import java.util.Iterator;
import java.util.List;
import java.util.Map.Entry;

import org.junit.Assert;

/** word2vec embeddings */
public class Word2Vec implements WordEmbedding {
	
	private /*final*/ Hashtable<String, HashMap<Integer,Double>> hashWordEmbeddings;
	private final int dim;
	
	/** reads word embeddings from a file of format  word: val1,val2,val3,.....valn\n ...
	 * @throws IOException */
	public Word2Vec(String fileName) throws IOException {
		
		this.hashWordEmbeddings = new Hashtable<String, HashMap<Integer,Double>>();
		List<String> lines = Files.readAllLines(Paths.get(fileName),Charset.defaultCharset());
		
		Iterator<String> it = lines.iterator();
		int dim_ = -1; //intermediate
		
		while(it.hasNext()) {
			String line = it.next(); 
			//line format=>    word:val1,val2,....valk\n
			int wordIndex = line.lastIndexOf(':');
			String word = line.substring(0, wordIndex);
			String[] args = line.substring(wordIndex+1).split(",");
			
			if(dim_ == -1) {
				dim_ = args.length;
			}
			else {
				assert dim_ == args.length;
			}
			
			final HashMap<Integer, Double> wordVector = new HashMap<Integer, Double>();
			for(int i = 0; i < dim_; i++) {
				wordVector.put(i, Double.parseDouble(args[i]));
			}
			
			final HashMap<Integer, Double> normWordVector = wordVector;//this.normalizeEmbedding(wordVector, dim_);
			this.hashWordEmbeddings.put(word, normWordVector);
		}
		
		this.dim = dim_;
		System.out.println("Word Embedding Dimension is "+this.dim);
		//embedding of unknown word
		final HashMap<Integer, Double> unkVector = new HashMap<Integer, Double>();
		this.hashWordEmbeddings.put(UNK, unkVector);
		
		//this.hashWordEmbeddings = this.normalizeFeatures();
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
	
	/** take the word embedding and perform affine transformation so that every embedding has a mean 
	 *  of 0 and unit variance, along the features */
	public HashMap<Integer, Double> normalizeEmbedding(HashMap<Integer, Double> vec, int dim) {
		
		HashMap<Integer,Double> nVec = new HashMap<Integer,Double>();
		
		double mean = 0, std = 0;
	    for(int i=0; i< dim; i++) {
	    	Double res = vec.get(i);
	    	if(res != null)
	    		mean =  mean + res;
	    }
	    mean = mean/(double)dim;
	    
	    for(int i=0; i< dim; i++) {
	    	Double res = vec.get(i);
	    	if(res != null)
	    		std =  std + (res-mean)*(res-mean);
	    }
	    
	    std = Math.sqrt(std/(double)dim);
	    if(std == 0)
	    	std = 0.00001; 
	    
	    for(int i=0; i< dim; i++) {
	    	Double res = vec.get(i);
	    	if(res == null)
	    		res = 0.0;
	    	
	    	double nValue = (res - mean)/std;
	    	if(nValue != 0.0)
	    		nVec.put(i, nValue);
	    }
		
	    return nVec;
	}
	
	/** take the word embeddings and perform affine transformation so that every feature in an embedding has
	 *  a mean of 0 and unit variance, along the entire list of word embedding */
	public Hashtable<String, HashMap<Integer, Double>> normalizeFeatures() {
		
		Hashtable<String, HashMap<Integer,Double>> nEmbeddings = new Hashtable<String, HashMap<Integer,Double>>();
		
		double [] mean = new double[this.dim]; //feature wise mean
		Arrays.fill(mean, 0.0);
		double [] std = new double[this.dim]; //feature wise standard deviation
		Arrays.fill(std, 0.0);
		
		final int numPts = this.hashWordEmbeddings.size();
		
		for(Entry<String, HashMap<Integer, Double>> e: this.hashWordEmbeddings.entrySet()) {
			
			for(Entry<Integer, Double> val : e.getValue().entrySet()) {
				mean[val.getKey()] = mean[val.getKey()] + val.getValue();
			}
		}
		
	    for(int i=0; i< dim; i++) {
	    	mean[i] =  mean[i]/(double)numPts;
	    }
	    
	    for(Entry<String, HashMap<Integer, Double>> e: this.hashWordEmbeddings.entrySet()) {
			
			for(Entry<Integer, Double> val : e.getValue().entrySet()) {
				std[val.getKey()] = std[val.getKey()] + 
						(val.getValue() - mean[val.getKey()])*(val.getValue() - mean[val.getKey()]);
			}
		}
	    
	    for(int i=0; i< dim; i++) {
	    	std[i] =  Math.sqrt(std[i]/(double)numPts);
	    	if(std[i] == 0)
		    	std[i] = 0.00001;
	    }
	    
	    //apply the transformation
	    for(Entry<String, HashMap<Integer, Double>> e: this.hashWordEmbeddings.entrySet()) {
			
	    	HashMap<Integer, Double> nMap = new HashMap<Integer, Double>();
			for(Entry<Integer, Double> entry : e.getValue().entrySet()) {
				double nVal = (entry.getValue() - mean[entry.getKey()])/(double)std[entry.getKey()];
				nMap.put(entry.getKey(), nVal); 
			}
			nEmbeddings.put(e.getKey(), nMap);
		}
		
	    return nEmbeddings;
	}

}
