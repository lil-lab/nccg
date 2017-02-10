package edu.cornell.cs.nlp.spf.parser.ccg.rnn.shiftreduce.parsingopembeddings;

import org.nd4j.linalg.api.ndarray.INDArray;

import edu.cornell.cs.nlp.spf.ccg.lexicon.LexicalEntry;
import edu.cornell.cs.nlp.spf.parser.ccg.rnn.shiftreduce.categoryembeddings.CategoryEmbeddingResult;

/** contains the result of embedding a parsing operation*/
public class ParsingOpEmbeddingResult {
	
	private final int ruleIndex;
	private final INDArray embedding;
	private final CategoryEmbeddingResult categoryResult;
	private final LexicalEntry<?> lexicalEntry;
	private final INDArray preOutput, x;
	
	public ParsingOpEmbeddingResult(int ruleIndex, INDArray embedding,
									CategoryEmbeddingResult categoryResult,
									LexicalEntry<?> lexicalEntry, INDArray preOutput, 
									INDArray x) {
		this.ruleIndex = ruleIndex;
		this.embedding = embedding;
		this.categoryResult = categoryResult;
		this.lexicalEntry = lexicalEntry;
		this.preOutput = preOutput;
		this.x = x;
	}
	
	public int ruleIndex() {
		return this.ruleIndex;
	}
	
	public INDArray getEmbedding() {
		return this.embedding;
	}
	
	public CategoryEmbeddingResult getCategoryResult() {
		return this.categoryResult;
	}
	
	public LexicalEntry<?> getLexicalEntry() {
		return this.lexicalEntry;
	}
	
	public INDArray getPreOutput() {
		return this.preOutput;
	}
	
	public INDArray getX() {
		return this.x;
	}
}
