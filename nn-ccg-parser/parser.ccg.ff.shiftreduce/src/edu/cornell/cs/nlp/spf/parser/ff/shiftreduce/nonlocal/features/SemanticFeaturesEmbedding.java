package edu.cornell.cs.nlp.spf.parser.ff.shiftreduce.nonlocal.features;

import java.io.Serializable;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

import edu.cornell.cs.nlp.spf.mr.lambda.LogicalExpression;
import edu.cornell.cs.nlp.spf.parser.ccg.rnn.shiftreduce.categoryembeddings.CategoryEmbeddingResult;
import edu.cornell.cs.nlp.spf.parser.ccg.rnn.shiftreduce.categoryembeddings.LogicalExpressionEmbedding;
import edu.cornell.cs.nlp.spf.parser.ccg.shiftreduce.stacks.DerivationState;
import edu.cornell.cs.nlp.spf.parser.ccg.shiftreduce.stacks.DerivationStateHorizontalIterator;

/** A non-local feature that does not implement AbstractNonLocalFeatures. 
 * Works by embedding semantics of root of last three trees on the stack. */
public class SemanticFeaturesEmbedding implements Serializable {

	private static final long serialVersionUID = -8822484297220347592L;
	//private static final long serialVersionUID = 3473037384686308555L;
	private LogicalExpressionEmbedding embedSemantics;
	private final int dim;
	
	public SemanticFeaturesEmbedding(int dim, double learningRate, double l2) {
		this.dim = dim;
		this.embedSemantics = new LogicalExpressionEmbedding(dim, learningRate, l2);
	}
	
	public SemanticFeaturesEmbeddingResult getSemanticEmbedding(DerivationState<LogicalExpression> state) {
		
		final LogicalExpression last, sndLast, thirdLast;
		
		if(state.getRightCategory() != null) {
			last = state.getRightCategory().getSemantics();
			sndLast = state.getLeftCategory().getSemantics();
			
			DerivationStateHorizontalIterator<LogicalExpression> hit = state.horizontalIterator();
			hit.next(); //equals to state
			
			if(hit.hasNext()) {
				thirdLast = hit.next().getLeftCategory().getSemantics();
			} else {
				thirdLast = null;
			}
		} else {
			if(state.getLeftCategory() != null) {
				last = state.getLeftCategory().getSemantics();
				sndLast = null;
				thirdLast = null;
			} else {
				last = null;
				sndLast = null;
				thirdLast = null;
			}
		}
		
		return this.getSemanticEmbedding(last, sndLast, thirdLast);
	}
	
	public SemanticFeaturesEmbeddingResult getSemanticEmbedding(LogicalExpression last, LogicalExpression sndLast, 
																LogicalExpression thirdLast) {
		
		//features go as last, sndlast, and third last
		CategoryEmbeddingResult lastResult = this.embedSemantics.getLogicalExpressionEmbedding(last);
		CategoryEmbeddingResult sndLastResult = this.embedSemantics.getLogicalExpressionEmbedding(sndLast);
		CategoryEmbeddingResult thirdLastResult = this.embedSemantics.getLogicalExpressionEmbedding(thirdLast);
		
		INDArray embedding = Nd4j.concat(1, lastResult.getEmbedding(), sndLastResult.getEmbedding(), thirdLastResult.getEmbedding());
		
		return new SemanticFeaturesEmbeddingResult(embedding, lastResult, sndLastResult, thirdLastResult);
	}
	
	public void backprop(INDArray error, SemanticFeaturesEmbeddingResult result) {
		
		final INDArray errorLast = error.get(NDArrayIndex.interval(0, this.dim));
		final INDArray errorSndLast = error.get(NDArrayIndex.interval(this.dim, 2 * this.dim));
		final INDArray errorThirdLast = error.get(NDArrayIndex.interval(2 * this.dim, 3 * this.dim));
		
		this.embedSemantics.backprop(result.getLastResult().getSemanticTree(), errorLast);
		this.embedSemantics.backprop(result.getSndLastResult().getSemanticTree(), errorSndLast);
		this.embedSemantics.backprop(result.getThirdLastResult().getSemanticTree(), errorThirdLast);
	}
	
	public LogicalExpressionEmbedding getSemanticEmbeddingObject() {
		return this.embedSemantics;
	}
	
	public int getDimension() {
		return this.embedSemantics.getDimension() * 3;
	}
}