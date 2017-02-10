package edu.cornell.cs.nlp.spf.parser.ccg.rnn.shiftreduce.learning;

import java.util.List;
import java.util.Spliterator;
import java.util.Spliterators;
import java.util.stream.Collectors;
import java.util.stream.StreamSupport;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import edu.cornell.cs.nlp.spf.data.sentence.Sentence;
import edu.cornell.cs.nlp.spf.parser.ParsingOp;
import edu.cornell.cs.nlp.spf.parser.ccg.rnn.shiftreduce.categoryembeddings.CategoryEmbedding;
import edu.cornell.cs.nlp.spf.parser.ccg.rnn.shiftreduce.dataset.CompositeDataPointDecision;
import edu.cornell.cs.nlp.spf.parser.ccg.rnn.shiftreduce.embeddings.AbstractRecurrentNetworkHelper;
import edu.cornell.cs.nlp.spf.parser.ccg.rnn.shiftreduce.embeddings.EmbedActionHistory;
import edu.cornell.cs.nlp.spf.parser.ccg.rnn.shiftreduce.embeddings.EmbedParserState;
import edu.cornell.cs.nlp.spf.parser.ccg.rnn.shiftreduce.embeddings.TopLayerMLP;
import edu.cornell.cs.nlp.spf.parser.ccg.rnn.shiftreduce.neuralnetworkparser.NeuralNetworkShiftReduceParser;
import edu.cornell.cs.nlp.spf.parser.ccg.rnn.shiftreduce.parsingopembeddings.ParsingOpEmbedding;
import edu.cornell.cs.nlp.spf.parser.ccg.rnn.shiftreduce.parsingopembeddings.ParsingOpEmbeddingResult;
import edu.cornell.cs.nlp.utils.composites.Pair;
import edu.cornell.cs.nlp.utils.log.ILogger;
import edu.cornell.cs.nlp.utils.log.LoggerFactory;

public class LearnerGradientCheck<MR> {
	
	public static final ILogger	LOG = LoggerFactory.create(LearnerGradientCheck.class);
	private final double epsilon;

	private final NeuralNetworkShiftReduceParser<Sentence, MR> parser;
	
	public LearnerGradientCheck(NeuralNetworkShiftReduceParser<Sentence, MR> parser) {
		this.parser = parser;
		this.epsilon = 0.000001;
	}
	
	private double calcLoss(List<Pair<Integer, CompositeDataPointDecision<MR>>> enumeratedDecisions, 
						    List<Double>[] allDecisionExponents) {
		
		double loss = 0;
		
		for(int i = 0; i < allDecisionExponents.length; i++) {
			
			List<Double> exponents = allDecisionExponents[i];
			CompositeDataPointDecision<MR> decision = enumeratedDecisions.get(i).second();
			
			double Z = 0;
			for(Double exponent: exponents) {
				Z = Z + Math.exp(exponent);
			}
			
			loss = loss - exponents.get(decision.getGTruthIx()) + Math.log(Z);
		}
		
		return loss;
	}
	
	/** Estimates gradient using symmetric difference quotient*/
	private double estimateGradient(INDArray vec, 
			List<Pair<Integer, CompositeDataPointDecision<MR>>> enumeratedDecisions, 
			List<ParsingOpEmbeddingResult>[] parsingOpEmbeddingResults, INDArray[] states) {
		
		TopLayerMLP topLayer = this.parser.getTopLayer();
		double orig = vec.getDouble(new int[]{0, 0});
		
		vec.putScalar(new int[]{0,  0}, orig + this.epsilon);
		List<Double>[] allDecisionExponents1 = topLayer.getEmbedding(parsingOpEmbeddingResults, states);
		
		double loss1 = this.calcLoss(enumeratedDecisions, allDecisionExponents1);
		
		vec.putScalar(new int[]{0,  0}, orig - this.epsilon);
		List<Double>[] allDecisionExponents2 = topLayer.getEmbedding(parsingOpEmbeddingResults, states);
		
		double loss2 = this.calcLoss(enumeratedDecisions, allDecisionExponents2);
				
		vec.putScalar(new int[]{0,  0}, orig);	
		
		double estimate = (loss1 - loss2)/(2*this.epsilon);
		return estimate;
	}

	public Pair<Double, Double> gradientCheck(List<Pair<Integer, CompositeDataPointDecision<MR>>> enumeratedDecisions, 
			INDArray[] topLayerA1, INDArray[] topLayerA2, INDArray[] topLayerA3) {
		
		final ParsingOpEmbedding<MR> embedParsingOp = this.parser.getEmbedParsingOp();
		final int numDecisions = enumeratedDecisions.size();
		INDArray[] states = new INDArray[numDecisions];
		
		@SuppressWarnings("unchecked")
		List<ParsingOpEmbeddingResult>[] parsingOpEmbeddingResults = new List[numDecisions];
		
		StreamSupport.stream(Spliterators
				.spliterator(enumeratedDecisions, Spliterator.IMMUTABLE), false)
				.forEach(enumeratedDecision -> {
					
					int ix = enumeratedDecision.first();
					CompositeDataPointDecision<MR> decision = enumeratedDecision.second();
					
					INDArray a1 = topLayerA1[decision.getActionHistoryIx()];
					INDArray a2 = topLayerA2[decision.getParserStateIx()];
					INDArray a3 = topLayerA3[decision.getSentenceIx()];
					
					INDArray x = Nd4j.concat(1, a1, a2, a3);//.transpose();
					LOG.debug("X %s", x);
					
					states[ix] = x;
					
					List<ParsingOp<MR>> options = decision.getPossibleActions();
					
					parsingOpEmbeddingResults[ix] = StreamSupport.stream(Spliterators
												.spliterator(options, Spliterator.IMMUTABLE), true)
												.map(op->embedParsingOp.getEmbedding(op))
												.collect(Collectors.toList());
				});
		
		//Gradient check for parsingOpEmbedding
		
		INDArray vec1 = parsingOpEmbeddingResults[0].get(0).getEmbedding();
		final double empiricalGradParsingOp = this.estimateGradient(vec1, enumeratedDecisions,
																		parsingOpEmbeddingResults, states);
		
		
		//Gradient check for states
		INDArray vec2 = states[0];
		final double empiricalGradX = this.estimateGradient(vec2, enumeratedDecisions, 
																		parsingOpEmbeddingResults, states);
		
		return Pair.of(empiricalGradParsingOp, empiricalGradX);
	}
	
	private double getLoss(List<Pair<Integer, CompositeDataPointDecision<MR>>> enumeratedDecisions, 
			List<Pair<AbstractRecurrentNetworkHelper, Object>> calcEmbedding) {
		
		final ParsingOpEmbedding<MR> embedParsingOp = this.parser.getEmbedParsingOp();
		final CategoryEmbedding<MR> categEmbedding = this.parser.getCategoryEmbedding();
		final EmbedActionHistory<MR> embedActionHistory = this.parser.getEmbedActionHistory();
		final EmbedParserState<MR> embedParserState = this.parser.getEmbedParserState();
		
		final TopLayerMLP topLayer = this.parser.getTopLayer();
		final int numDecisions = enumeratedDecisions.size();
		
		INDArray[] states = new INDArray[numDecisions];
		@SuppressWarnings("unchecked")
		List<ParsingOpEmbeddingResult>[] parsingOpEmbeddingResults = new List[numDecisions];
		
		// Compute embeddings of history, state and buffer
		List<Object> embeddings = StreamSupport.stream(Spliterators
					.spliterator(calcEmbedding, Spliterator.IMMUTABLE), true)
					.map(p->p.first().getAllTopLayerEmbedding(p.second()))
					.collect(Collectors.toList());
		
		final INDArray[] topLayerA1 = (INDArray[])embeddings.get(0);
		final INDArray[] topLayerA2 = (INDArray[])embeddings.get(1);
		final INDArray[] topLayerA3 = (INDArray[])embeddings.get(2);

		StreamSupport.stream(Spliterators
				.spliterator(enumeratedDecisions, Spliterator.IMMUTABLE), true)
				.forEach(enumeratedDecision -> {
					
					int ix = enumeratedDecision.first();
					CompositeDataPointDecision<MR> decision = enumeratedDecision.second();
					
					INDArray a1 = topLayerA1[decision.getActionHistoryIx()];
					INDArray a2 = topLayerA2[decision.getParserStateIx()];
					INDArray a3 = topLayerA3[decision.getSentenceIx()];
					
					INDArray x = Nd4j.concat(1, a1, a2, a3);
					states[ix] = x;
					
					List<ParsingOp<MR>> options = decision.getPossibleActions();
					
					parsingOpEmbeddingResults[ix] = StreamSupport.stream(Spliterators
												.spliterator(options, Spliterator.IMMUTABLE), true)
												.map(op->embedParsingOp.getEmbedding(op))
												.collect(Collectors.toList());
				});
		
		List<Double>[] allDecisionExponents1 = topLayer.getEmbedding(parsingOpEmbeddingResults, states);	
		final double loss = this.calcLoss(enumeratedDecisions, allDecisionExponents1);
		
		embedActionHistory.clearParsingOpEmbeddingResult();
		embedParserState.clearCategoryResults();
		categEmbedding.invalidateCache();
		
		return loss;
	}
	
	public void gradientCheckCategory(List<Pair<Integer, CompositeDataPointDecision<MR>>> enumeratedDecisions, 
			List<Pair<AbstractRecurrentNetworkHelper, Object>> calcEmbedding) {
		
		final ParsingOpEmbedding<MR> embedParsingOp = this.parser.getEmbedParsingOp();
		final CategoryEmbedding<MR> categEmbedding = this.parser.getCategoryEmbedding();
		final EmbedActionHistory<MR> embedActionHistory = this.parser.getEmbedActionHistory();
		final EmbedParserState<MR> embedParserState = this.parser.getEmbedParserState();
		
		embedActionHistory.clearParsingOpEmbeddingResult();
		embedParserState.clearCategoryResults();
		categEmbedding.invalidateCache();
		
		//Parsing Operation Gradient Test -- Action
		{
			INDArray vec = embedParsingOp.getActionVector();
			final double orig = vec.getDouble(new int[]{0, 0});
			
			vec.putScalar(new int[]{0, 0}, orig + this.epsilon);
			final double loss1 = this.getLoss(enumeratedDecisions, calcEmbedding);
					
			vec.putScalar(new int[]{0, 0}, orig - this.epsilon);
			final double loss2 = this.getLoss(enumeratedDecisions, calcEmbedding);
			
			final double empiricalGradParsingOp = (loss1 - loss2)/(2.0 * this.epsilon);
			
			vec.putScalar(new int[]{0, 0}, orig); 
			
			embedActionHistory.clearParsingOpEmbeddingResult();
			embedParserState.clearCategoryResults();
			categEmbedding.invalidateCache();
			
			embedParsingOp.empiricalAction = empiricalGradParsingOp;
		}
		
		//Parsing Operation Gradient Test -- Template
		{
			INDArray vec = embedParsingOp.getTemplateVector();
			final double orig = vec.getDouble(new int[]{0, 0});
			
			vec.putScalar(new int[]{0, 0}, orig + this.epsilon);
			final double loss1 = this.getLoss(enumeratedDecisions, calcEmbedding);
					
			vec.putScalar(new int[]{0, 0}, orig - this.epsilon);
			final double loss2 = this.getLoss(enumeratedDecisions, calcEmbedding);
			
			final double empiricalGradParsingOp = (loss1 - loss2)/(2.0 * this.epsilon);
			
			vec.putScalar(new int[]{0, 0}, orig); 
			
			embedActionHistory.clearParsingOpEmbeddingResult();
			embedParserState.clearCategoryResults();
			categEmbedding.invalidateCache();
			
			embedParsingOp.empiricalTemplate = empiricalGradParsingOp;
		}
		
		//Category Gradient Test -- Syntax
		{
			INDArray vec = categEmbedding.getSyntaxVector();
			final double orig = vec.getDouble(new int[]{0, 0});
			
			vec.putScalar(new int[]{0, 0}, orig + this.epsilon);
			final double loss1 = this.getLoss(enumeratedDecisions, calcEmbedding);
					
			vec.putScalar(new int[]{0, 0}, orig - this.epsilon);
			final double loss2 = this.getLoss(enumeratedDecisions, calcEmbedding);
			
			final double empiricalGradCategory = (loss1 - loss2)/(2.0 * this.epsilon);
			
			vec.putScalar(new int[]{0, 0}, orig); 
			
			embedActionHistory.clearParsingOpEmbeddingResult();
			embedParserState.clearCategoryResults();
			categEmbedding.invalidateCache();
			
			categEmbedding.empiricalSyntaxGrad = empiricalGradCategory;
		}
		
		//Category Gradient Test -- Semantic
		{
			INDArray vec = categEmbedding.getSemanticVector();
			final double orig = vec.getDouble(new int[]{0, 0});
			
			vec.putScalar(new int[]{0, 0}, orig + this.epsilon);
			final double loss1 = this.getLoss(enumeratedDecisions, calcEmbedding);
					
			vec.putScalar(new int[]{0, 0}, orig - this.epsilon);
			final double loss2 = this.getLoss(enumeratedDecisions, calcEmbedding);
			
			final double empiricalGradCategory = (loss1 - loss2)/(2.0 * this.epsilon);
			
			vec.putScalar(new int[]{0, 0}, orig); 
			
			embedActionHistory.clearParsingOpEmbeddingResult();
			embedParserState.clearCategoryResults();
			categEmbedding.invalidateCache();
			
			categEmbedding.empiricalSemanticGrad = empiricalGradCategory;
		}
		
		//Category Gradient Test -- Syntax Recursive Network, W
		{
			INDArray vec = categEmbedding.getSyntaxRecursiveW();
			final double orig = vec.getDouble(new int[]{0, 0});
			
			vec.putScalar(new int[]{0, 0}, orig + this.epsilon);
			final double loss1 = this.getLoss(enumeratedDecisions, calcEmbedding);
					
			vec.putScalar(new int[]{0, 0}, orig - this.epsilon);
			final double loss2 = this.getLoss(enumeratedDecisions, calcEmbedding);
			
			final double empiricalGradCategory = (loss1 - loss2)/(2.0 * this.epsilon);
			
			vec.putScalar(new int[]{0, 0}, orig); 
			
			embedActionHistory.clearParsingOpEmbeddingResult();
			embedParserState.clearCategoryResults();
			categEmbedding.invalidateCache();
			
			categEmbedding.empiricalSyntaxRecursiveW = empiricalGradCategory;
		}
		
		//Category Gradient Test -- Syntax Recursive Network, b
		{
			INDArray vec = categEmbedding.getSyntaxRecursiveb();
			final double orig = vec.getDouble(new int[]{0, 0});
			
			vec.putScalar(new int[]{0, 0}, orig + this.epsilon);
			final double loss1 = this.getLoss(enumeratedDecisions, calcEmbedding);
					
			vec.putScalar(new int[]{0, 0}, orig - this.epsilon);
			final double loss2 = this.getLoss(enumeratedDecisions, calcEmbedding);
			
			final double empiricalGradCategory = (loss1 - loss2)/(2.0 * this.epsilon);
			
			vec.putScalar(new int[]{0, 0}, orig); 
			
			embedActionHistory.clearParsingOpEmbeddingResult();
			embedParserState.clearCategoryResults();
			categEmbedding.invalidateCache();
			
			categEmbedding.empiricalSyntaxRecursiveb = empiricalGradCategory;
		}
		
		//Category Gradient Test -- Semantic Recursive Network, W
		{
			INDArray vec = categEmbedding.getSemanticRecursiveW();
			final double orig = vec.getDouble(new int[]{0, 0});
			
			vec.putScalar(new int[]{0, 0}, orig + this.epsilon);
			final double loss1 = this.getLoss(enumeratedDecisions, calcEmbedding);
					
			vec.putScalar(new int[]{0, 0}, orig - this.epsilon);
			final double loss2 = this.getLoss(enumeratedDecisions, calcEmbedding);
			
			final double empiricalGradCategory = (loss1 - loss2)/(2.0 * this.epsilon);
			
			vec.putScalar(new int[]{0, 0}, orig); 
			
			embedActionHistory.clearParsingOpEmbeddingResult();
			embedParserState.clearCategoryResults();
			categEmbedding.invalidateCache();
			
			categEmbedding.empiricalSemanticRecursiveW = empiricalGradCategory;
		}
		
		//Category Gradient Test -- Semantic Recursive Network, b
		{
			INDArray vec = categEmbedding.getSemanticRecursiveb();
			final double orig = vec.getDouble(new int[]{0, 0});
			
			vec.putScalar(new int[]{0, 0}, orig + this.epsilon);
			final double loss1 = this.getLoss(enumeratedDecisions, calcEmbedding);
					
			vec.putScalar(new int[]{0, 0}, orig - this.epsilon);
			final double loss2 = this.getLoss(enumeratedDecisions, calcEmbedding);
			
			final double empiricalGradCategory = (loss1 - loss2)/(2.0 * this.epsilon);
			
			vec.putScalar(new int[]{0, 0}, orig); 
			
			embedActionHistory.clearParsingOpEmbeddingResult();
			embedParserState.clearCategoryResults();
			categEmbedding.invalidateCache();
			
			categEmbedding.empiricalSemanticRecursiveb = empiricalGradCategory;
		}
	}
	
}
