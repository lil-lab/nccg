package edu.cornell.cs.nlp.spf.parser.ccg.rnn.shiftreduce.learning;

import java.util.LinkedList;
import java.util.List;
import java.util.Spliterator;
import java.util.Spliterators;
import java.util.stream.StreamSupport;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

import edu.cornell.cs.nlp.spf.ccg.categories.Category;
import edu.cornell.cs.nlp.spf.parser.ccg.rnn.shiftreduce.categoryembeddings.CategoryEmbedding;
import edu.cornell.cs.nlp.spf.parser.ccg.rnn.shiftreduce.categoryembeddings.CategoryEmbeddingResult;
import edu.cornell.cs.nlp.utils.composites.Pair;
import edu.cornell.cs.nlp.utils.log.ILogger;
import edu.cornell.cs.nlp.utils.log.LoggerFactory;

/** This class pre-trains parsing operations and category embeddings
 *  @author Dipendra Misra (dkm@cs.cornell.edu) */
public class ParsingOpPreTraining<MR> {

	public static final ILogger	LOG = LoggerFactory.create(ParsingOpPreTraining.class);
	
	private List<ParsingOpPreTrainingDataset<MR>> preTrainDataset;
	private List<ParsingOpPreTrainingDataset<MR>> testDataset;
	private int iterations; 
	private CategoryEmbedding<MR> categEmbedding;
	private INDArray W;
	private double learningRate;
	
	public ParsingOpPreTraining(List<ParsingOpPreTrainingDataset<MR>> preTrainDataset, 
								List<ParsingOpPreTrainingDataset<MR>> testDataset, 
			CategoryEmbedding<MR> categEmbedding, int numRules) {
		this.preTrainDataset = preTrainDataset;
		this.testDataset = testDataset;
		this.categEmbedding = categEmbedding;
		this.learningRate = 0.01;///(double)Math.max(this.preTrainDataset.size(), 1);
		int col = 2*categEmbedding.getDimension();
		this.W = Nd4j.zeros(new int[] {numRules, col});
		
		for(int i=0; i<numRules; i++) {
			for(int j=0; j<col; j++) {
				this.W.putScalar(new int[]{i, j}, Math.random());
			}
		}
		this.iterations = 25;
	}
	
	/** logs statistics about the pre-training dataset */
	public void logStats() {
		
		int posLabel = 0, negLabel =  0;
		for(ParsingOpPreTrainingDataset<MR> sample: this.preTrainDataset) {
			if(sample.getLabel()) {
				posLabel++;
			} else {
				negLabel++;
			}
		}
		
		LOG.info("Pretraining Dataset: Positive Examples %s, Negative Examples %s", posLabel, negLabel);
	}
	
	/** sample dataset. Tries to get a dataset with number of positive and negative examples equal to the "size" */
	public void sample(int size) {
		List<ParsingOpPreTrainingDataset<MR>> samplePreTrainDataset = 
									new LinkedList<ParsingOpPreTrainingDataset<MR>>();
		
		int addPosLabel = 0, negPosLabel = 0;
		for(ParsingOpPreTrainingDataset<MR> sample: this.preTrainDataset) {
			if(sample.getLabel() && addPosLabel < size) {
				samplePreTrainDataset.add(sample);
				addPosLabel++;
			} else if(!sample.getLabel() && negPosLabel < size) {
					samplePreTrainDataset.add(sample);
					negPosLabel++;
				}
		}
	
		this.preTrainDataset = samplePreTrainDataset;
	}
	
	/** Test the quality of pre-training */
	public void test() {
		
		int numPositive = 0, numNegative = 0;
		int numCorrectPositive = 0, numCorrectNegative = 0;
		
		for(ParsingOpPreTrainingDataset<MR> datapoint: this.testDataset) {
			
			Category<MR> categ1 = datapoint.getCategory1();
			Category<MR> categ2 = datapoint.getCategory2();
			INDArray a = datapoint.getRule();
			boolean label = datapoint.getLabel();
		
			CategoryEmbeddingResult c1 = this.categEmbedding.getCategoryEmbedding(categ1);
			CategoryEmbeddingResult c2 = this.categEmbedding.getCategoryEmbedding(categ2);
			
			INDArray concat = Nd4j.concat(1, c1.getEmbedding(), c2.getEmbedding()).transpose(); //check this
			INDArray aW = a.mmul(this.W);
			INDArray exponent = aW.mmul(concat); //must be a scalar
			double scalarExponent = exponent.getDouble(new int[]{0, 0});
			
			/* find probability that application can be applied given by 
			 * 1/{1 + exp{ * an * W [c1n; c2n]}} */
			double prob = 1/(1+ Math.exp(scalarExponent));
			if(label) { 
				numPositive++;
				if(prob > 0.5) {
					numCorrectPositive++;
				}
			} else {
				numNegative++;
				if(prob <= 0.5) {
					numCorrectNegative++;
				}
			}
			
		}
		
		double positivePrecision = numCorrectPositive/(double)(numCorrectPositive + numNegative - numCorrectNegative);
		double positiveRecall = numCorrectPositive/(double)numPositive;
		double negativePrecision = numCorrectNegative/(double)(numCorrectNegative + numPositive - numCorrectPositive);
		double negativeRecall = numCorrectNegative/(double)numNegative;
		double accuracy = (numCorrectPositive + numCorrectNegative)/(double)(numPositive + numNegative);
				
		LOG.info("Test Dataset: Positive Examples %s, Negative Examples %s", numPositive, numNegative);
		LOG.info("Positive Precision %s, Positive Recall %s ", positivePrecision, positiveRecall);
		LOG.info("Negative Precision %s, Negative Recall %s ", negativePrecision, negativeRecall);
		LOG.info("Accuracy on Test Dataset %s", accuracy);
	}
	
	public void pretrain() {
		
		final int size = this.preTrainDataset.size();
		final int dim = this.categEmbedding.getDimension();
		
		for(int i = 0; i < this.iterations; i++) {
			int ctr = 0;
			//compute gradient and perform updates
			for(ParsingOpPreTrainingDataset<MR> datapoint: this.preTrainDataset) {
				
				Category<MR> categ1 = datapoint.getCategory1();
				Category<MR> categ2 = datapoint.getCategory2();
				INDArray a = datapoint.getRule();
				boolean label = datapoint.getLabel();
			
				CategoryEmbeddingResult c1 = this.categEmbedding.getCategoryEmbedding(categ1);
				CategoryEmbeddingResult c2 = this.categEmbedding.getCategoryEmbedding(categ2);
				
				INDArray concat = Nd4j.concat(1, c1.getEmbedding(), c2.getEmbedding()).transpose(); //check this
				INDArray aW = a.mmul(this.W);
				INDArray exponent = aW.mmul(concat); //must be a scalar
				double scalarExponent = exponent.getDouble(new int[]{0, 0});
				
				//multiplier is given by -yn/{1 + exp{-yn * an * W [c1n; c2n]}}
				final double multiplier;
				if(label) {
					multiplier = 1/(1+ Math.exp(-scalarExponent));//;-1/(1+ Math.exp(-scalarExponent)); 
				} else {
					multiplier = -1/(1+ Math.exp(scalarExponent));//1/(1+ Math.exp(scalarExponent));
				}
				
				if(Double.isNaN(concat.normmaxNumber().doubleValue())) {
					LOG.info("NAN alert");
				}
				
				//gradient with respect to category 1
				INDArray cError = aW.mul(multiplier);
				
				if(Double.isNaN(cError.normmaxNumber().doubleValue())) {
					LOG.info("NAN alert");
				}
				
				LOG.info("Iteration %s, %s out of %s; Scalar Exponent %s, L2 norm of the Error %s",
						i, ++ctr, size, scalarExponent, cError.norm2(1));
				
				//gradient with respect to category 1, category 2
				INDArray c1Error = cError.get(NDArrayIndex.interval(0, dim));
				INDArray c2Error = cError.get(NDArrayIndex.interval(dim, 2*dim));
				
				/*this.categEmbedding.backprop(c1.getSyntacticTree(),
											 c1.getSemanticTree(), c1Error);

				this.categEmbedding.backprop(c2.getSyntacticTree(),
											 c2.getSemanticTree(), c2Error);*/
				
				List<Pair<CategoryEmbeddingResult, INDArray>> backpropToCategory = 
									new LinkedList<Pair<CategoryEmbeddingResult, INDArray>>();
				backpropToCategory.add(Pair.of(c1, c1Error));
				backpropToCategory.add(Pair.of(c2, c2Error));
				
				StreamSupport.stream(Spliterators
						.spliterator(backpropToCategory, Spliterator.IMMUTABLE), true)
						.forEach(p->this.categEmbedding.backprop(p.first().getSyntacticTree(),
								 							 p.first().getSemanticTree(), p.second()));
				
				//update the category embeddings and flush the gradients
				this.categEmbedding.updateParameters();
				this.categEmbedding.flushGradients();
				
				//gradient with respect to W 
				INDArray gradW = concat.mmul(a).transposei().muli(multiplier);
				this.W.subi(gradW.muli(this.learningRate));
			} 
			
		}
		
		this.categEmbedding.flushAdaGradHistory();
	}
	
}
