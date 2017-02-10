package edu.cornell.cs.nlp.spf.parser.ff.shiftreduce.neuralparser;

import org.nd4j.linalg.api.ndarray.INDArray;

import edu.cornell.cs.nlp.spf.base.hashvector.IHashVector;
import edu.cornell.cs.nlp.spf.parser.ccg.rules.LexicalResult;

public class LexicalEntryPreProcessResult<MR> {
	
	final private LexicalResult<MR> lexicalResult;
	final private IHashVector feature;
	final private INDArray embedding;
	
	public LexicalEntryPreProcessResult(LexicalResult<MR> lexicalResult, 
				IHashVector feature, INDArray embedding) {
		
		this.lexicalResult = lexicalResult;
		this.feature = feature;
		this.embedding = embedding;
	}
	
	public LexicalResult<MR> getLexicalResult() {
		return this.lexicalResult;
	}
	
	public IHashVector getFeature() {
		return this.feature;
	}
	
	public INDArray getEmbedding() {
		return this.embedding;
	}
}
