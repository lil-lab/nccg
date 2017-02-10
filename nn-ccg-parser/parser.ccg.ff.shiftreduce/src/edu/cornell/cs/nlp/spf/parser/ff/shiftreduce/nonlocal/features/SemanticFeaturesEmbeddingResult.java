package edu.cornell.cs.nlp.spf.parser.ff.shiftreduce.nonlocal.features;

import org.nd4j.linalg.api.ndarray.INDArray;

import edu.cornell.cs.nlp.spf.parser.ccg.rnn.shiftreduce.categoryembeddings.CategoryEmbeddingResult;

public class SemanticFeaturesEmbeddingResult {

	private final INDArray embedding;
	private final CategoryEmbeddingResult lastResult, sndLastResult, thirdLast;
	
	public SemanticFeaturesEmbeddingResult(INDArray embedding, CategoryEmbeddingResult lastResult, 
			CategoryEmbeddingResult sndLastResult, CategoryEmbeddingResult thirdLast) {
		
		this.embedding = embedding;
		this.lastResult = lastResult;
		this.sndLastResult = sndLastResult;
		this.thirdLast = thirdLast;
	}
	
	public INDArray getEmbedding() {
		return this.embedding;
	}
	
	public CategoryEmbeddingResult getLastResult() {
		return this.lastResult;
	}
	
	public CategoryEmbeddingResult getSndLastResult() {
		return this.sndLastResult;
	}
	
	public CategoryEmbeddingResult getThirdLastResult() {
		return this.thirdLast;
	}
}
