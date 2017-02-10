package edu.cornell.cs.nlp.spf.parser.ccg.rnn.shiftreduce.categoryembeddings;

import org.nd4j.linalg.api.ndarray.INDArray;

import edu.cornell.cs.nlp.spf.parser.ccg.rnn.shiftreduce.recursive.Tree;

public class CategoryEmbeddingResult {

	private final INDArray embedding;
	private final Tree syntacticTree;
	private final Tree semanticTree;
	
	public CategoryEmbeddingResult(INDArray embedding, Tree syntacticTree, Tree semanticTree) {
		
		this.embedding = embedding;
		this.syntacticTree = syntacticTree;
		this.semanticTree = semanticTree;
	}
	
	public INDArray getEmbedding() {
		return this.embedding;
	}
	
	public Tree getSyntacticTree() {
		return this.syntacticTree;
	}
	
	public Tree getSemanticTree() {
		return this.semanticTree;
	}
}
