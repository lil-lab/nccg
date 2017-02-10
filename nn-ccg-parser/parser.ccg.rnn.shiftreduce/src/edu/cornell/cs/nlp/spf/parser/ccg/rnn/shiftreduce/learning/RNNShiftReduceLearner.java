package edu.cornell.cs.nlp.spf.parser.ccg.rnn.shiftreduce.learning;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.PrintWriter;
import java.io.UnsupportedEncodingException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;
import java.util.ListIterator;
import java.util.Spliterator;
import java.util.Spliterators;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.Predicate;
import java.util.stream.Collectors;
import java.util.stream.StreamSupport;

import org.json.JSONObject;
import org.nd4j.linalg.api.ndarray.INDArray;
//import org.nd4j.linalg.api.ops.impl.transforms.RectifedLinear;
//import org.nd4j.linalg.api.ops.impl.transforms.Step;
import org.nd4j.linalg.api.ops.impl.transforms.Sqrt;
import org.nd4j.linalg.api.ops.impl.transforms.Tanh;
import org.nd4j.linalg.api.ops.impl.transforms.TanhDerivative;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

import com.google.common.base.Joiner;

import edu.cornell.cs.nlp.spf.ccg.lexicon.CompositeImmutableLexicon;
import edu.cornell.cs.nlp.spf.ccg.lexicon.ILexiconImmutable;
import edu.cornell.cs.nlp.spf.data.IDataItem;
import edu.cornell.cs.nlp.spf.data.ILabeledDataItem;
import edu.cornell.cs.nlp.spf.data.collection.IDataCollection;
import edu.cornell.cs.nlp.spf.data.sentence.Sentence;
import edu.cornell.cs.nlp.spf.data.situated.sentence.SituatedSentence;
import edu.cornell.cs.nlp.spf.data.utils.IValidator;
import edu.cornell.cs.nlp.spf.explat.IResourceRepository;
import edu.cornell.cs.nlp.spf.explat.ParameterizedExperiment.Parameters;
import edu.cornell.cs.nlp.spf.explat.resources.IResourceObjectCreator;
import edu.cornell.cs.nlp.spf.explat.resources.usage.ResourceUsage;
import edu.cornell.cs.nlp.spf.mr.lambda.LogicalExpression;
import edu.cornell.cs.nlp.spf.parser.ParsingOp;
import edu.cornell.cs.nlp.spf.parser.ccg.model.IDataItemModel;
import edu.cornell.cs.nlp.spf.parser.ccg.model.Model;
import edu.cornell.cs.nlp.spf.parser.ccg.rnn.shiftreduce.categoryembeddings.CategoryEmbedding;
import edu.cornell.cs.nlp.spf.parser.ccg.rnn.shiftreduce.dataset.CompositeDataPoint;
import edu.cornell.cs.nlp.spf.parser.ccg.rnn.shiftreduce.dataset.CompositeDataPointDecision;
import edu.cornell.cs.nlp.spf.parser.ccg.rnn.shiftreduce.dataset.CreateCompositeDecisionDataset;
import edu.cornell.cs.nlp.spf.parser.ccg.rnn.shiftreduce.dataset.ProcessedDataSet;
import edu.cornell.cs.nlp.spf.parser.ccg.rnn.shiftreduce.embeddings.AbstractEmbedding;
import edu.cornell.cs.nlp.spf.parser.ccg.rnn.shiftreduce.embeddings.AbstractRecurrentNetworkHelper;
import edu.cornell.cs.nlp.spf.parser.ccg.rnn.shiftreduce.embeddings.EmbedActionHistory;
import edu.cornell.cs.nlp.spf.parser.ccg.rnn.shiftreduce.embeddings.EmbedParserState;
import edu.cornell.cs.nlp.spf.parser.ccg.rnn.shiftreduce.embeddings.EmbedWordBuffer;
import edu.cornell.cs.nlp.spf.parser.ccg.rnn.shiftreduce.embeddings.TopLayerMLP;
import edu.cornell.cs.nlp.spf.parser.ccg.rnn.shiftreduce.neuralnetworkparser.NeuralNetworkShiftReduceParser;
import edu.cornell.cs.nlp.spf.parser.ccg.rnn.shiftreduce.parsingopembeddings.ParsingOpEmbedding;
import edu.cornell.cs.nlp.spf.parser.ccg.rnn.shiftreduce.parsingopembeddings.ParsingOpEmbeddingResult;
import edu.cornell.cs.nlp.spf.parser.ccg.shiftreduce.stacks.DerivationState;
import edu.cornell.cs.nlp.spf.parser.ccg.shiftreduce.stacks.DerivationStateHorizontalIterator;
import edu.cornell.cs.nlp.spf.parser.filter.IParsingFilterFactory;
import edu.cornell.cs.nlp.spf.parser.joint.model.IJointModelImmutable;
import edu.uw.cs.lil.amr.data.AMRMeta;
import edu.cornell.cs.nlp.utils.collections.queue.DirectAccessBoundedPriorityQueue;
import edu.cornell.cs.nlp.utils.composites.Pair;
import edu.cornell.cs.nlp.utils.log.ILogger;
import edu.cornell.cs.nlp.utils.log.LogLevel;
import edu.cornell.cs.nlp.utils.log.LoggerFactory;

/** Neural Parser Training Algorithm for CCG Semantic Parsing 
 * @author Dipendra Misra (dkm@cs.cornell.edu)
 * */
public class RNNShiftReduceLearner<SAMPLE extends IDataItem<?>, DI extends ILabeledDataItem<SAMPLE, ?>, MR> {
	
	public static final ILogger	LOG = LoggerFactory.create(RNNShiftReduceLearner.class);

	private final IDataCollection<DI> trainingData;
	private final NeuralNetworkShiftReduceParser<Sentence, MR> parser;
	
	private final Integer epoch;
	private final Integer partitionFunctionApproximationK;
	private final LearningRate learningRate;
	/** Regularizer for matrix A and b */
	private final Double aRegularizer, bRegularizer;
	private final INDArray adaGradSumSquareA, adaGradSumSquareb;
	private final Double[] adaGradSumSquareActionBias;
	private final Integer beamSize;
	private final IParsingFilterFactory<DI, MR> parsingFilterFactory;
	
	private final String bootstrapFolderName;
	private final boolean saveModelAfterLearning;
	
	private boolean setDisplay;
	
	@SuppressWarnings("unused")
	private final LearnerGradientCheck<MR> gradientChecker;
	
	public RNNShiftReduceLearner(IDataCollection<DI> trainingData, 
			NeuralNetworkShiftReduceParser<Sentence, MR> parser, IValidator<DI,MR> validator, 
			Integer epoch, Double learningRate, Double learningRateDecay, Double l2, Integer beamSize, 
			Integer partitionFunctionApproximationK, IParsingFilterFactory<DI, MR> parsingFilterFactory, 
			CompositeImmutableLexicon<MR> compositeLexicon, ILexiconImmutable<MR> tempLexicon, 
			boolean preTrain, String folderName, boolean saveModelAfterLearning) {
		this.trainingData = trainingData;
		this.parser = parser;
		
		this.epoch = epoch;
		this.learningRate = new LearningRate(learningRate, learningRateDecay);
		this.aRegularizer = l2;
		this.bRegularizer = l2;
		this.adaGradSumSquareA = Nd4j.zeros(parser.getAffineA().shape()).addi(0.00001);
		this.adaGradSumSquareb = Nd4j.zeros(parser.getAffineb().shape()).addi(0.00001);
		this.adaGradSumSquareActionBias = new Double[parser.numRules()];
		Arrays.fill(this.adaGradSumSquareActionBias, 0.00001);
		
		this.partitionFunctionApproximationK = partitionFunctionApproximationK;
		
		this.beamSize = beamSize;
		this.parsingFilterFactory = parsingFilterFactory;
		LOG.setCustomLevel(LogLevel.INFO);
		
		this.bootstrapFolderName = folderName;
		this.saveModelAfterLearning = saveModelAfterLearning;
		
		this.setDisplay = false;
		this.gradientChecker = new LearnerGradientCheck<MR>(parser);
		
//		if(this.bootstrapFolderName != null) {
//			LOG.info("Bootstraping Parameters and Embeddings from Folder %s", this.bootstrapFolderName);
//			this.bootstrapModel(this.bootstrapFolderName);
//		}
		
		LOG.info("RNN Shift Reduce Learner: epoch %s, learningRate %s, l2 %s, beamSize %s,",
														this.epoch, this.learningRate, l2, this.beamSize);
		LOG.info("\t... partitionFunctionK %s, bootstrapFolderName %s ",
										  this.partitionFunctionApproximationK, this.bootstrapFolderName);
	}
	
	/** Pre trains action embedding which also includes categories */
	public void pretrain(Model<SAMPLE, MR> model) {
		
		LOG.info("Pretraining Categories");
		
		final CategoryEmbedding<MR> categEmbedding = this.parser.getCategoryEmbedding();
		final int numRules = this.parser.numRules();
		int count = 0;
		//Create Pre-training dataset
		List<ParsingOpPreTrainingDataset<MR>> preTrainDataset = new LinkedList<ParsingOpPreTrainingDataset<MR>>();
		for (final DI dataItem : this.trainingData) {
			
			final SAMPLE dataItemSample = dataItem.getSample();
			final IDataItemModel<MR> dataItemModel = model.createDataItemModel(dataItemSample);
			
			Sentence dataItemSentence = null;
			try {
				 dataItemSentence = (Sentence)dataItemSample;
			} catch(Exception e) {
				new RuntimeException("Error "+e.toString());
			}
			
			final Predicate<ParsingOp<MR>> pruningFilter = this.parsingFilterFactory.create(dataItem);
			
			List<ParsingOpPreTrainingDataset<MR>> sample = this.parser
					.createPreTrainingData(dataItemSentence, pruningFilter, dataItemModel, true, 
							  model.getLexicon(), this.beamSize);
			preTrainDataset.addAll(sample);
			if(++count == 50)
				break;
		}
		
		int size = preTrainDataset.size();
		if(size <= 10)
			throw new RuntimeException("Too few examples");
		
		int testSize = (int)(0.25*size);		
		List<ParsingOpPreTrainingDataset<MR>> testDataset = preTrainDataset.subList(0, testSize);
		preTrainDataset = preTrainDataset.subList(testSize, size);
		
		ParsingOpPreTraining<MR> preTrain = new ParsingOpPreTraining<MR>(preTrainDataset, testDataset, 
																		 categEmbedding, numRules);
		preTrain.logStats();
		preTrain.sample(100);
		preTrain.pretrain();
		preTrain.test();
	}
	
	/** Calculates log-likelihood and sum of likelihood statistics on the given batch*/
	private void calcBatchLikelihood(List<ProcessedDataSet<MR>> batch) {
		
		EmbedActionHistory<MR> embedActionHistory = this.parser.getEmbedActionHistory();			
		EmbedWordBuffer embedWordBuffer = this.parser.getEmbedWordBuffer();
		EmbedParserState<MR> embedParserState = this.parser.getEmbedParserState();
		ParsingOpEmbedding<MR> embedParsingOp = this.parser.getEmbedParsingOp();
		INDArray A = this.parser.getAffineA();
		INDArray b = this.parser.getAffineb();
		
		int exampleIndex = 0;
		
		double sumLogLikelihood = 0, sumLikelihood = 0;
		int correct = 0;
		
		for(ProcessedDataSet<MR> pt: batch) {

			DerivationState<MR> dstate = pt.getState();
			List<ParsingOp<MR>> parsingOps = pt.getActionHistory();
			List<String> wordBuffer = pt.getBuffer();
			List<ParsingOp<MR>> options = pt.getPossibleActions();
			int gTruthIx = pt.getGTruthIx();
			exampleIndex++;
			
			if(options.size() == 1) {
				continue;
			} else if(options.size() == 0) {
				throw new RuntimeException("No Actions?");
			}
			
			// Compute embeddings of history, state and buffer for this state
			INDArray a1 = embedActionHistory.getEmbedding(parsingOps);
			INDArray a2 = embedParserState.getEmbedding(dstate);
			INDArray a3 = embedWordBuffer.getEmbedding(wordBuffer);
			
			INDArray x = Nd4j.concat(1, Nd4j.concat(1, a1, a2), a3).transpose();
			
			//computes g(A[a1,a2,a3]+b)
			INDArray currentPreOutput = A.mmul(x).add(b).transpose(); 
			INDArray current = Nd4j.getExecutioner()
					   			   .execAndReturn(new /*LeakyReLU*/Tanh(currentPreOutput.dup()));
			
			//probability of taking each action
			Double[] scores = new Double[options.size()];
			double Z = 0; //partition function 
			
			List<ParsingOpEmbeddingResult> parsingOpEmbeddingResult = 
										   new LinkedList<ParsingOpEmbeddingResult>();
			
			List<Double> exponents = new LinkedList<Double>();
			List<INDArray> embedParseSteps = new LinkedList<INDArray>();
			Double maxExponent = Double.NEGATIVE_INFINITY;
			
			parsingOpEmbeddingResult = StreamSupport.stream(Spliterators
										.spliterator(options, Spliterator.IMMUTABLE), true)
										.map(op->embedParsingOp.getEmbedding(op))
										.collect(Collectors.toList());
			
			for(ParsingOpEmbeddingResult parseOpResult: parsingOpEmbeddingResult) {
				INDArray embedParseStep = parseOpResult.getEmbedding().transpose();
				
				//dot product
				double exponent = (current.mmul(embedParseStep)).getDouble(new int[]{0, 0});
				exponents.add(exponent);
				embedParseSteps.add(embedParseStep);
				
				if(exponent > maxExponent) {
					maxExponent = exponent;
				}
			}
			
			//max exponent trick
			Iterator<Double> exponentIt = exponents.iterator();
			int jl = 0;
			while(exponentIt.hasNext()) {
				
				double exponent = exponentIt.next();
				double score = Math.exp(exponent - maxExponent);
				
				scores[jl++] = score;

				Z = Z + score;
			}
			
			int maxScore = 0;
			for(int j=0; j < scores.length; j++) {
				scores[j] = scores[j]/Z;
				
				if(scores[j] > scores[maxScore]) {
					maxScore = j;
				}
			}
			
			//LOG.info("Score is %s, Z is %s, maxexp %s", Joiner.on(",").join(scores), Z, maxExponent);
			
			LOG.info("Example: %s, ground truth: %s, sentence: %s",
										exampleIndex, scores[gTruthIx], pt.getSentence());
			
			String categ = "";
			DerivationStateHorizontalIterator<MR> it = dstate.horizontalIterator();
			
			boolean first = true;
			while(it.hasNext()) {
				DerivationState<MR> dit = it.next();
				
				if(first && dit.getRightCategory() != null) {
					first = false;
					categ =  dit.getRightCategory().toString();
				}
				
				if(dit.getLeftCategory() != null) {
					categ =  dit.getLeftCategory() + ", " + categ;
				}
			}
			
			LOG.info("\t rule name: %s, buffer: %s, category: %s",
										options.get(gTruthIx).getRule(), Joiner.on(" ").join(wordBuffer),
										options.get(gTruthIx).getCategory());
			LOG.info("\t best parsing op rule: %s, category: %s, best score: %s \n",
					options.get(maxScore).getRule(), options.get(maxScore).getCategory(),
					scores[maxScore]);
			LOG.info("\t state categories: %s\n", categ);
			
			sumLikelihood = sumLikelihood + scores[gTruthIx];
			sumLogLikelihood = sumLogLikelihood + Math.log(scores[gTruthIx]); 
			
			if(scores[maxScore] == scores[gTruthIx]) {
				correct++;
			}
			
			embedActionHistory.clearParsingOpEmbeddingResult();
			embedParserState.clearCategoryResults();
		}
		LOG.info("Sum Log-Likelihood %s, Sum Likelihood %s, Correct %s out of %s",
					sumLogLikelihood, sumLikelihood, correct, batch.size());
	}
	
	/** Splits the dataset into a validation and train. The function performs learning
	 *  iterations on training by performing backprop through the entire computation graph.
	 *  Loss over validataion set is continuously computed. */
	public void fitDataSet(List<ProcessedDataSet<MR>> dataset) {
		
		int dataSize = dataset.size();
		int trainSize = (int)(0.8*dataSize);
		List<ProcessedDataSet<MR>> train = dataset.subList(0, trainSize);
		List<ProcessedDataSet<MR>> validation = dataset.subList(trainSize, dataSize);
		
		LOG.info("Fitting Dataset of size %s, train %s, validation %s", 
										dataset.size(), trainSize, dataSize - trainSize);
		
		//LOG.setCustomLevel(LogLevel.DEBUG);
		EmbedActionHistory<MR> embedActionHistory = this.parser.getEmbedActionHistory();			
		EmbedWordBuffer embedWordBuffer = this.parser.getEmbedWordBuffer();
		EmbedParserState<MR> embedParserState = this.parser.getEmbedParserState();
		CategoryEmbedding<MR> categEmbedding = this.parser.getCategoryEmbedding();
		ParsingOpEmbedding<MR> embedParsingOp = this.parser.getEmbedParsingOp();
		INDArray A = this.parser.getAffineA();
		INDArray b = this.parser.getAffineb();
		
		long time2 = 0, time4 = 0, totalTime = 0;
		double norm = 0, term = 0;
		
		// End of iteration, print the likelihood
		LOG.info("---------------- Beginning ------------");
		this.calcBatchLikelihood(train);
		LOG.info("---------------- Initial train ------------");
		this.calcBatchLikelihood(validation);
		LOG.info("---------------- Initial validation ------------");
		
		categEmbedding.printCategoryEmbeddings();
		
		for(int iter = 1; iter <= this.epoch; iter++) {
			LOG.info("=========================");
			LOG.info("Fit Dataset Iteration: %s", iter);
			int exampleIndex = 0;
			
			double sumLogLikelihood = 0, sumLikelihood = 0;
			
			for(ProcessedDataSet<MR> pt: train) {
				LOG.info("=========================");
				LOG.info("Example: %s", ++exampleIndex);
				
				long start1 = System.currentTimeMillis();
				
				DerivationState<MR> dstate = pt.getState();
				List<ParsingOp<MR>> parsingOps = pt.getActionHistory();
				List<String> wordBuffer = pt.getBuffer();
				List<ParsingOp<MR>> options = pt.getPossibleActions();
				int gTruthIx = pt.getGTruthIx();
				
				LOG.debug("Number options %s, Ground Truth %s,\n Buffer %s, Sentence %s",
						options.size(), gTruthIx, Joiner.on(", ").join(wordBuffer), pt.getSentence());
				
				if(options.size() == 1) {
					continue;
				} else if(options.size() == 0) {
					throw new RuntimeException("No Actions for this example.");
				}
				
				// Compute embeddings of history, state and buffer for this example
				
				List<Pair<AbstractEmbedding, Object>> calcEmbedding = 
									new LinkedList<Pair<AbstractEmbedding, Object>>();
				calcEmbedding.add(Pair.of(embedActionHistory, (Object)parsingOps));
				calcEmbedding.add(Pair.of(embedParserState, (Object)dstate));
				calcEmbedding.add(Pair.of(embedWordBuffer, (Object)wordBuffer));
				
				List<Object> embeddings = StreamSupport.stream(Spliterators
							.spliterator(calcEmbedding, Spliterator.IMMUTABLE), true)
							.map(p->p.first().getEmbedding(p.second()))
							.collect(Collectors.toList());
				
				INDArray a1 = (INDArray)embeddings.get(0);
				INDArray a2 = (INDArray)embeddings.get(1);
				INDArray a3 = (INDArray)embeddings.get(2);
				
				INDArray x = Nd4j.concat(1, Nd4j.concat(1, a1, a2), a3).transpose();
				
				//computes g(A[a1; a2; a3]+b)
				INDArray currentPreOutput = A.mmul(x).add(b).transpose(); 
				INDArray current = Nd4j.getExecutioner()
									   .execAndReturn(new /*LeakyReLU*/Tanh(currentPreOutput.dup()));
				
				INDArray embedGTruth = null;
				
				//some data-structures for computing gradients wrt A,b,x
				INDArray expYj = Nd4j.zeros(embedParsingOp.getDimension()); 
				
				//probability of taking each action
				Double[] scores = new Double[options.size()];
				double Z = 0; //partition function 
				int i = 0;
				
				String bbytes = "";
				List<ParsingOpEmbeddingResult> parsingOpEmbeddingResult = 
											   new LinkedList<ParsingOpEmbeddingResult>();
				
				List<Double> exponents = new LinkedList<Double>();
				List<INDArray> embedParseSteps = new LinkedList<INDArray>();
				Double maxExponent = Double.NEGATIVE_INFINITY;
				
				long start2 = System.currentTimeMillis();
				LOG.info("Time part 1 " +(start2 - start1));
								
				parsingOpEmbeddingResult = StreamSupport.stream(Spliterators
											.spliterator(options, Spliterator.IMMUTABLE), true)
											.map(op->embedParsingOp.getEmbedding(op))
											.collect(Collectors.toList());
				
				LOG.debug("X %s, current %s", Helper.printVector(x), Helper.printVector(current));
				
				for(ParsingOp<MR> op: options) {
					ParsingOpEmbeddingResult parseOpResult = parsingOpEmbeddingResult.get(i);
					INDArray embedParseStep = parseOpResult.getEmbedding().transpose();
					
					if(i == gTruthIx) {
						embedGTruth = embedParseStep;
					}
					
					//dot product
					double exponent = (current.mmul(embedParseStep)).getDouble(new int[]{0, 0});
					LOG.debug("Op %s", op);
					LOG.debug("Exponent %s. Parse Op Embedding is %s", exponent, 
																	Helper.printVector(embedParseStep));
					exponents.add(exponent);
					embedParseSteps.add(embedParseStep);
					
					if(exponent > maxExponent) {
						maxExponent = exponent;
					}
					
					i++;
				}
				
				//max exponent trick
				Iterator<Double> exponentIt = exponents.iterator();
				int jl = 0;
				for(INDArray embedParseStep: embedParseSteps) {
					
					double exponent = exponentIt.next();
					double score = Math.exp(exponent - maxExponent);
					
					bbytes = bbytes + ", score: "+score+" exponent: "+(exponent-maxExponent);
					
					scores[jl++] = score;
					expYj.addi(embedParseStep.transpose().muli(score));
					Z = Z + score;
				}
				
				assert i == jl;
				
				LOG.debug("Score Bytes %s ", bbytes);
				
				String scoreByte = "";
				for(int j=0; j < scores.length; j++) {
					scores[j] = scores[j]/Z;
					scoreByte = scoreByte + ", " + scores[j]; 
				}
				
				LOG.debug("ground truth %s, scores are %s and Z is %s ", gTruthIx, scoreByte, Z);
				
				sumLikelihood = sumLikelihood + scores[gTruthIx];
				sumLogLikelihood = sumLogLikelihood + Math.log(scores[gTruthIx]);
				
				LOG.info("Iteration: %s, Likelihood %s, NLL %s", 
						iter, scores[gTruthIx], -Math.log(scores[gTruthIx]));
				
				//LOG.info("A regularizer %s", 0.5*this.aRegularizer * Helper.squaredFrobeniusNorm(A));
				//LOG.info("b regularizer %s", 0.5*this.bRegularizer * Helper.squaredFrobeniusNorm(b));
				
				expYj.divi(Z);
				
				/* perform backpropagation through the entire computation graph
				 * 
				 * Loss is given by: negative log-likelihood + L2 regularization term + momentum term
				 *               
				 * gradient of loss with respect to A_pq will be: (y is the probability) 
				 * 				 { -pembed(y_i)_p + E_y_j[pembed(y_j)_p] } x_q
				 * with respect to b will be: -pembed(y_i) + E_y_j[pembed(y_j)]
				 * with respect to x_q: -\sum_p pembed(y_i)_p A_pq + E_y_j[\sum_p pembed(y_j)_p A_pq] */
				
				INDArray currentDerivative = Nd4j.getExecutioner()
												 .execAndReturn(new /*LeakyReLUDerivative*/TanhDerivative(currentPreOutput.dup()));
				
				INDArray gradb = embedGTruth.mul(-1).add(expYj.transpose()).mul(currentDerivative.transpose());
				INDArray gradA = gradb.mmul(x.transpose()); 
				INDArray gradX = gradb.transpose().mmul(A);
				
				//Add regularization term
				gradA.addi(A.mul(this.aRegularizer));
				gradb.addi(b.mul(this.bRegularizer));
				
				//Do gradient clipping
				double normA = gradA.normmaxNumber().doubleValue();
				double normb = gradb.normmaxNumber().doubleValue();
				double threshold = 1.0;
				
				if(normA > threshold) {
					gradA.muli(threshold/normA);
				}
				
				if(normb > threshold) {
					gradb.muli(threshold/normb);
				}
				
				LOG.debug("GradA norm %s, max number %s, min number %s",
								gradA.normmaxNumber().doubleValue(),
								gradA.maxNumber().doubleValue(), gradA.minNumber().doubleValue());
				LOG.debug("Gradb norm %s, max number %s, min number %s",
								gradb.normmaxNumber().doubleValue(),
								gradb.maxNumber().doubleValue(), gradb.minNumber().doubleValue());
				LOG.debug("Learning Rate %s", this.learningRate.getLearningRate());
				
				// update A,b using AdaGrad rule
				this.adaGradSumSquareA.addi(gradA.mul(gradA));
				this.adaGradSumSquareb.addi(gradb.mul(gradb));
				
				double initLearningRate = this.learningRate.getLearningRate();
				
				INDArray invertedLearningRateA = Nd4j.getExecutioner()
													.execAndReturn(new Sqrt(this.adaGradSumSquareA.dup()))
													.divi(initLearningRate);
				
				INDArray invertedLearningRateb = Nd4j.getExecutioner()
													 .execAndReturn(new Sqrt(this.adaGradSumSquareb.dup()))
													 .divi(initLearningRate);
				
				A.subi(gradA.div(invertedLearningRateA));
				b.subi(gradb.div(invertedLearningRateb));
				
				final long start3 = System.currentTimeMillis();
				long diff = (start3 - start2);
				LOG.info("Time part 2 " +diff);
				time2 = time2 + diff;
				
				//backprop through the action embedding
				ListIterator<ParsingOpEmbeddingResult> it = parsingOpEmbeddingResult.listIterator();
				
				List<Pair<ParsingOpEmbeddingResult, INDArray>> backpropParsingOp = new 
										LinkedList<Pair<ParsingOpEmbeddingResult, INDArray>>();
				
				while(it.hasNext()) {
					int ix = it.nextIndex();
					ParsingOpEmbeddingResult parseOpResult = it.next();
					
					final INDArray error;
					if(ix == gTruthIx) {
						 error = current.mul(-1 + scores[ix]);
					} else {
						 error = current.mul(scores[ix]);
					}
					norm = norm + error.maxNumber().doubleValue();
					term++;
					backpropParsingOp.add(Pair.of(parseOpResult, error));
				}
				
				StreamSupport.stream(Spliterators
						.spliterator(backpropParsingOp, Spliterator.IMMUTABLE), true).unordered()
						.forEach(p->embedParsingOp.backProp(p.second(), p.first()));
				
				//backprop through the recurrent networks and their leaves
				final int dimAction = embedActionHistory.getDimension();
				final int dimState = embedParserState.getDimension();
				final int dimBuffer = embedWordBuffer.getDimension();
				
				if(!Double.isFinite(gradX.sumNumber().doubleValue())) {
					
					LOG.info("Action History, a1 is %s", Helper.printVector(a1));
					LOG.info("Parse State, a2 is %s", Helper.printVector(a2));
					LOG.info("Word Embedding, a3 is %s", Helper.printVector(a3));
					LOG.info("Current is %s", Helper.printVector(current));
					LOG.info("A is %s", Helper.printMatrix(A));
					LOG.info("b is %s", Helper.printVector(b));
					LOG.info("Embed GTruth %s ", Helper.printVector(embedGTruth));
					LOG.info("expYj %s", Helper.printVector(expYj));
					LOG.info("gradX %s", Helper.printVector(gradX));
					
					for(INDArray embedParseStep: embedParseSteps) {
						LOG.info("Embed Parse Step %s", Helper.printVector(embedParseStep) );
					}
					throw new ArithmeticException("Found NaN");
				}
				
				final long start4 = System.currentTimeMillis();
				LOG.info("Time part 3 " +(start4 - start3));
				
				INDArray errorActionHistory = gradX.get(NDArrayIndex.interval(0, dimAction));
				INDArray errorState = gradX.get(NDArrayIndex.interval(dimAction, dimState + dimAction));
				INDArray errorWordBuffer = gradX.get(NDArrayIndex
									 .interval(dimState + dimAction, dimState + dimAction + dimBuffer));
				
				List<Pair<AbstractRecurrentNetworkHelper, INDArray>> backprop = 
									new LinkedList<Pair<AbstractRecurrentNetworkHelper, INDArray>>();
				backprop.add(Pair.of(embedActionHistory, errorActionHistory));
				backprop.add(Pair.of(embedParserState, errorState));
				backprop.add(Pair.of(embedWordBuffer, errorWordBuffer));
				
				StreamSupport.stream(Spliterators
							.spliterator(backprop, Spliterator.IMMUTABLE), true).unordered()
							.forEach(p-> p.first().backprop(p.second()));
				
				final long start5 = System.currentTimeMillis();
				long diff4 = (start5 - start4);
				LOG.info("Time part 4 " + diff4);
				time4 = time4 + diff4;
				
				//update the category vectors and flush the gradients (updates Recursive network)
				categEmbedding.updateParameters();
				categEmbedding.flushGradients();
				
				final long start6 = System.currentTimeMillis();
				LOG.info("Time part 5 " + (start6 - start5));
				
				//update the actions and flush the gradients
				embedParsingOp.updateParameters();
				embedParsingOp.flushGradients();
				
				//Clear the stored results
				embedActionHistory.clearParsingOpEmbeddingResult();
				embedParserState.clearCategoryResults();
				
				//Decay the learning rate, Recurrent NNs have their own decay scheme
				/*this.learningRate.decay(); //decay learning rate for A,b
				categEmbedding.decay(); //decay learning rate for Recursive network
				*/
				
				final long start7 = System.currentTimeMillis();
				LOG.info("Time part 6 " + (start7 - start6));
				
				totalTime = totalTime + (start7 - start1);
				
				// End of iteration, print the likelihood
				/*LOG.info("-------- example %s ------------", exampleIndex);
				this.calcBatchLikelihood(processedData.subList(0, 20));
				LOG.info("-------- example %s ------------", exampleIndex);*/
			}
			
			// End of iteration, print the likelihood for train and validation
			if(iter%2 == 0) {
				
				LOG.info("-------- train iteration %s  ------------", iter);
				this.calcBatchLikelihood(train);
				LOG.info("-------- train, end of iteration %s ------------", iter);
				
				LOG.info("-------- validation iteration %s  ------------", iter);
				this.calcBatchLikelihood(validation);
				LOG.info("-------- validation, end of iteration %s ------------", iter);
			}
			
			// Log the category
			categEmbedding.printCategoryEmbeddings();
		}
		
		LOG.info("Direct flow %s, term %s, average %s", norm, term, norm/(double)term);
		LOG.info("Action History norm %s, term %s, average %s ", embedActionHistory.norm, 
				embedActionHistory.term, embedActionHistory.norm/(double)embedActionHistory.term);
		LOG.info("Parser State norm %s, term %s, average %s ", embedParserState.norm, 
				embedParserState.term, embedParserState.norm/(double)embedParserState.term);
		
		double totalSteps = (double)(this.epoch * trainSize);
		LOG.info("Total Time taken %s. Total instance %s, Average %s", totalTime, totalSteps, totalTime/totalSteps);
		LOG.info("Task 2: Time taken %s. Total instance %s, Average %s", time2, totalSteps, time2/totalSteps);
		LOG.info("Task 4: Time taken %s. Total instance %s, Average %s", time4, totalSteps, time4/totalSteps);
		
		LOG.info("Category Embedding Vectors");
		categEmbedding.printCategoryEmbeddings();
		
		if(this.saveModelAfterLearning) {
			try {
				this.logModel("");
			} catch (FileNotFoundException | UnsupportedEncodingException e) {
				LOG.error("Failed to Log the model. Error: "+e);
			}
		}
		
		//Relcone the recurrent networks as updates have been made
		embedActionHistory.reclone();
		embedParserState.reclone();
	}
	
	/** Calculates log-likelihood and other statistics on the given batch. */
	private double calcCompositeBatchLikelihood(List<CompositeDataPoint<MR>> batch) {
		
		EmbedActionHistory<MR> embedActionHistory = this.parser.getEmbedActionHistory();			
		EmbedWordBuffer embedWordBuffer = this.parser.getEmbedWordBuffer();
		EmbedParserState<MR> embedParserState = this.parser.getEmbedParserState();
		ParsingOpEmbedding<MR> embedParsingOp = this.parser.getEmbedParsingOp();
		INDArray A = this.parser.getAffineA();
		INDArray b = this.parser.getAffineb();
		Double[] actionBias = this.parser.getActionBias();
				
		int exampleIndex = 0, numDecisions = 0;
		
		AtomicInteger correct = new AtomicInteger();
		double sumLogLikelihood = 0.0, sumLikelihood = 0.0;
		
		for(CompositeDataPoint<MR> pt: batch) {
			LOG.info("=========================");
			LOG.info("Example: %s", ++exampleIndex);
			
			DerivationState<MR> dstate = pt.getState();
			List<ParsingOp<MR>> parsingOps = pt.getActionHistory();
			List<Pair<String, String>> wordBuffer = pt.getBuffer();
			List<CompositeDataPointDecision<MR>> decisions = pt.getDecisions();
			
			if(this.setDisplay)
				LOG.info/*debug*/("Sentence %s", pt.getSentence());
			
			// Compute embeddings of history, state and buffer
			
			List<Pair<AbstractRecurrentNetworkHelper, Object>> calcEmbedding = 
								new LinkedList<Pair<AbstractRecurrentNetworkHelper, Object>>();
			calcEmbedding.add(Pair.of(embedActionHistory, (Object)parsingOps));
			calcEmbedding.add(Pair.of(embedParserState, (Object)dstate));
			calcEmbedding.add(Pair.of(embedWordBuffer, (Object)wordBuffer));
			
			List<Object> embeddings = StreamSupport.stream(Spliterators
						.spliterator(calcEmbedding, Spliterator.IMMUTABLE), true)
						.map(p->p.first().getAllTopLayerEmbedding(p.second()))
						.collect(Collectors.toList());
			
			INDArray[] topLayerA1 = (INDArray[])embeddings.get(0);
			INDArray[] topLayerA2 = (INDArray[])embeddings.get(1);
			INDArray[] topLayerA3 = (INDArray[])embeddings.get(2);
			
			double[] stats = new double[2]; 
			stats[0] = 0.0; stats[1] = 0.0; //0 is sum of log-likelihood, 1 is sum of likelihood
			
			numDecisions = numDecisions + decisions.size();
			
			///////
			List<Pair<CompositeDataPointDecision<MR>, Integer>> enumerated = new 
					LinkedList<Pair<CompositeDataPointDecision<MR>, Integer>>();
			int ix = 0;
			for(CompositeDataPointDecision<MR> decision: decisions) {
				enumerated.add(Pair.of(decision, ix++));
			}
			///////
			
			StreamSupport.stream(Spliterators
				.spliterator(enumerated/*decisions*/, Spliterator.IMMUTABLE), true)
				.forEach(iter/*decision*/ -> {
					
					CompositeDataPointDecision<MR> decision = iter.first();
					
					INDArray a1 = topLayerA1[decision.getActionHistoryIx()];
					INDArray a2 = topLayerA2[decision.getParserStateIx()];
					INDArray a3 = topLayerA3[decision.getSentenceIx()];
					
					INDArray x = Nd4j.concat(1, Nd4j.concat(1, a1, a2), a3).transpose();
					
					List<ParsingOp<MR>> options = decision.getPossibleActions();
					int gTruthIx = decision.getGTruthIx();
					
					//computes g(A[a1; a2; a3]+b)
					INDArray currentPreOutput = A.mmul(x).add(b).transpose(); 
					INDArray current = Nd4j.getExecutioner()
										   .execAndReturn(new /*RectifedLinear*/Tanh(currentPreOutput.dup()));
					
					//probability of taking each action
					Double[] scores = new Double[options.size()];
					double Z = 0; //partition function 
					
					List<ParsingOpEmbeddingResult> parsingOpEmbeddingResult = 
												   new LinkedList<ParsingOpEmbeddingResult>();
					
					List<Double> exponents = new LinkedList<Double>();
					List<INDArray> embedParseSteps = new LinkedList<INDArray>();
					Double maxExponent = Double.NEGATIVE_INFINITY;
									
					parsingOpEmbeddingResult = StreamSupport.stream(Spliterators
												.spliterator(options, Spliterator.IMMUTABLE), true)
												.map(op->embedParsingOp.getEmbedding(op))
												.collect(Collectors.toList());
					
					Iterator<ParsingOpEmbeddingResult> it = parsingOpEmbeddingResult.iterator();
					while(it.hasNext()) {
						ParsingOpEmbeddingResult parseOpResult = it.next();
						INDArray embedParseStep = parseOpResult.getEmbedding().transpose();
						
						//dot product
						double exponent = (current.mmul(embedParseStep)).getDouble(new int[]{0, 0});
						exponent = exponent + actionBias[parseOpResult.ruleIndex()];
						
						exponents.add(exponent);
						embedParseSteps.add(embedParseStep);
						
						if(exponent > maxExponent) {
							maxExponent = exponent;
						}
					}
					
					//max exponent trick
					Iterator<Double> exponentIt = exponents.iterator();
					int jl = 0;
					while(exponentIt.hasNext()) {
						
						double exponent = exponentIt.next();
						double score = Math.exp(exponent - maxExponent);
						
						scores[jl++] = score;
						Z = Z + score;
					}
					
					int maxScore = 0;
					for(int j=0; j < scores.length; j++) {
						scores[j] = scores[j]/Z;
						
						if(scores[j] > scores[maxScore]) {
							maxScore = j;
						}
					}
					
					synchronized(stats) {
						stats[0] = stats[0] + Math.log(scores[gTruthIx]);
						stats[1] = stats[1] + scores[gTruthIx];
					}
					
					if(scores[maxScore] == scores[gTruthIx]) {
						correct.incrementAndGet();
					}
					
					/////////////
					if(this.setDisplay) {
						if(scores[gTruthIx] < scores[maxScore]) { //currently printing the really bad ones
							LOG.info("Decision Index %s", iter.second());
							LOG.info("Right parsing action score %s -> %s ", scores[gTruthIx], options.get(gTruthIx));
							LOG.info("Highest scoring parsing action score %s -> %s", scores[maxScore], options.get(maxScore));
						}
					}
					/////////////	
				});
			
			LOG.info("Example Index %s, Sum of Likelihood %s, NLL %s, sentence: %s", 
											exampleIndex, stats[1], -stats[0], pt.getSentence());
			
			sumLogLikelihood = sumLogLikelihood + stats[0];
			sumLikelihood = sumLikelihood + stats[1];
			
			//Clear the stored results
			embedActionHistory.clearParsingOpEmbeddingResult();
			embedParserState.clearCategoryResults();
		}		
		
		LOG.info("Sum Log-Likelihood %s, Sum Likelihood %s, Correct %s out of %s",
				sumLogLikelihood, sumLikelihood, correct.get(), numDecisions);
		
		return sumLogLikelihood;
	}
	
	/** Splits the dataset into a validation and train. The function performs learning
	 *  iterations on training by performing backprop through the entire computation graph.
	 *  All decisions for a single sentence are wrapped into one using composite datapoint.
	 *  Loss over validataion set is continuously computed. */
	public void fitCompositeDataSet(List<CompositeDataPoint<MR>> dataset) {
		
		int dataSize = dataset.size();
		int trainSize = (int)(0.9*dataSize);
		List<CompositeDataPoint<MR>> train = dataset.subList(0, trainSize);
		List<CompositeDataPoint<MR>> validation = dataset.subList(trainSize, dataSize);
		
		LOG.info("Fitting Dataset of size %s, train %s, validation %s", 
										dataset.size(), trainSize, dataSize - trainSize);
				
		EmbedActionHistory<MR> embedActionHistory = this.parser.getEmbedActionHistory();			
		EmbedWordBuffer embedWordBuffer = this.parser.getEmbedWordBuffer();
		EmbedParserState<MR> embedParserState = this.parser.getEmbedParserState();
		CategoryEmbedding<MR> categEmbedding = this.parser.getCategoryEmbedding();
		ParsingOpEmbedding<MR> embedParsingOp = this.parser.getEmbedParsingOp();
		INDArray A = this.parser.getAffineA();
		INDArray b = this.parser.getAffineb();
		Double[] actionBias = this.parser.getActionBias();
		
		///////////////////
		LOG.info("Printing the dataset."); 
		int ix = 0;
		for(CompositeDataPoint<MR> pt: dataset) {
			LOG.info("Point %s, %s", ++ix, pt);
		}
		LOG.info("Done printing the dataset.");
		///////////////////
		
		//Induce dynamic lexical entry origin in parsing op
		embedParsingOp.induceDynamicOriginAndTemplate(dataset);
		categEmbedding.induceVectorsFromProcessedDataset(dataset);
		
		final int dimAction = embedActionHistory.getDimension();
		final int dimState = embedParserState.getDimension();
		final int dimBuffer = embedWordBuffer.getDimension();
		
		LOG.info("-------- train initialization  ------------");
		this.calcCompositeBatchLikelihood(train);
		LOG.info("-------- train, end of initialization ------------");
		
		LOG.info("-------- validation initialization  ------------");
		double prevValidationLogLikelihood = this.calcCompositeBatchLikelihood(validation);
		LOG.info("-------- validation, end of initialization ------------");
		
		try {
			this.logModel("init");
		} catch (FileNotFoundException | UnsupportedEncodingException e) {
			throw new RuntimeException("Could not save model. Exception " + e);
		}
		
		long time2 = 0, time4 = 0, totalTime = 0;
		
		for(int iter = 1; iter <= this.epoch; iter++) {
			LOG.info("=========================");
			LOG.info("Fit Dataset Iteration: %s", iter);
			int exampleIndex = 0;
			
			double sumLogLikelihood = 0.0, sumLikelihood = 0.0;
			
			for(CompositeDataPoint<MR> pt: train) {
				LOG.info("=========================");
				LOG.info("Example: %s", ++exampleIndex);
				
				long start1 = System.currentTimeMillis();
				
				DerivationState<MR> dstate = pt.getState();
				List<ParsingOp<MR>> parsingOps = pt.getActionHistory();
				List<Pair<String, String>> wordBuffer = pt.getBuffer();
				List<CompositeDataPointDecision<MR>> decisions = pt.getDecisions();
				
				LOG.debug("Sentence %s", pt.getSentence());
				
				// Compute embeddings of history, state and buffer
				List<Pair<AbstractRecurrentNetworkHelper, Object>> calcEmbedding = 
									new LinkedList<Pair<AbstractRecurrentNetworkHelper, Object>>();
				calcEmbedding.add(Pair.of(embedActionHistory, (Object)parsingOps));
				calcEmbedding.add(Pair.of(embedParserState, (Object)dstate));
				calcEmbedding.add(Pair.of(embedWordBuffer, (Object)wordBuffer));
				
				List<Object> embeddings = StreamSupport.stream(Spliterators
							.spliterator(calcEmbedding, Spliterator.IMMUTABLE), true)
							.map(p->p.first().getAllTopLayerEmbedding(p.second()))
							.collect(Collectors.toList());
				
				INDArray[] topLayerA1 = (INDArray[])embeddings.get(0);
				INDArray[] topLayerA2 = (INDArray[])embeddings.get(1);
				INDArray[] topLayerA3 = (INDArray[])embeddings.get(2);
				
				INDArray gradA = Nd4j.zeros(A.shape()), gradb = Nd4j.zeros(b.shape());
				
				Double[] gradActionBias = new Double[actionBias.length];
				Arrays.fill(gradActionBias, 0.0);
				
				INDArray[] errorActionHistory = new INDArray[topLayerA1.length];
				INDArray[] errorState = new INDArray[topLayerA2.length];
				INDArray[] errorWordBuffer = new INDArray[topLayerA3.length];
				
				int max = Math.max(topLayerA1.length, Math.max(topLayerA2.length, topLayerA3.length));
				for(int i = 0; i < max; i++) {
					if(i < topLayerA1.length) { 
						errorActionHistory[i] = Nd4j.zeros(dimAction);
					} 
					if(i < topLayerA2.length) {
						errorState[i] = Nd4j.zeros(dimState);
					}
					if(i < topLayerA3.length) {
						errorWordBuffer[i] = Nd4j.zeros(dimBuffer); 
					}
				}
				
				double[] stats = new double[2]; 
				stats[0] = 0.0; stats[1] = 0.0; //0 is sum of log-likelihood, 1 is sum of likelihood
				
				long start2 = System.currentTimeMillis();
				LOG.info("Time part 1 " +(start2 - start1));
				
				////////////////////////// Partition function approximation /////////////
				final int k = 29;//9;
				final Comparator<Pair<Double, ParsingOpEmbeddingResult>> cmp  = 
						new Comparator<Pair<Double, ParsingOpEmbeddingResult>>() {
					public int compare(Pair<Double, ParsingOpEmbeddingResult> left, 
									   Pair<Double, ParsingOpEmbeddingResult> right) {
		        		return Double.compare(left.first(), right.first()); 
		    		}   
				};
				/////////////////////////////////////////////////////////////////////////
				
				StreamSupport.stream(Spliterators
					.spliterator(decisions, Spliterator.IMMUTABLE), false)
					.forEach(decision -> {
						
						INDArray a1 = topLayerA1[decision.getActionHistoryIx()];
						INDArray a2 = topLayerA2[decision.getParserStateIx()];
						INDArray a3 = topLayerA3[decision.getSentenceIx()];
						
						INDArray x = Nd4j.concat(1, a1, a2, a3).transpose();

						List<ParsingOp<MR>> options = decision.getPossibleActions();
						int gTruthIx = decision.getGTruthIx();
						//LOG.info("Number of options %s", options.size());
						
						//computes g(A[a1; a2; a3]+b)
						INDArray currentPreOutput = A.mmul(x).addi(b).transposei(); 
						INDArray current = Nd4j.getExecutioner()
											   .execAndReturn(new /*LeakyReLU*/Tanh(currentPreOutput.dup()));
						
						INDArray embedGTruth = null;
						
						//some data-structures for computing gradients wrt A,b,x
						INDArray expYj = Nd4j.zeros(embedParsingOp.getDimension()); 
						
						//probability of taking each action
						Double[] scores = new Double[options.size()];
						double Z = 0; //partition function 
						int i = 0;
						
						List<ParsingOpEmbeddingResult> parsingOpEmbeddingResult = 
													   new LinkedList<ParsingOpEmbeddingResult>();
						
						List<Double> exponents = new LinkedList<Double>();
						List<INDArray> embedParseSteps = new LinkedList<INDArray>();
						Double maxExponent = Double.NEGATIVE_INFINITY;
										
						parsingOpEmbeddingResult = StreamSupport.stream(Spliterators
													.spliterator(options, Spliterator.IMMUTABLE), true)
													.map(op->embedParsingOp.getEmbedding(op))
													.collect(Collectors.toList());
						
						LOG.debug("X %s, current %s", x, current);
						
						///////////////////////// Debug ////////////////////////
						DirectAccessBoundedPriorityQueue<Pair<Double, ParsingOpEmbeddingResult>> partitionFunctionAppr 
								= new DirectAccessBoundedPriorityQueue<Pair<Double, ParsingOpEmbeddingResult>>(k, cmp);
						ParsingOpEmbeddingResult gTruthResult = null;
						///////////////////////////////////////////////////////
						
						for(ParsingOpEmbeddingResult parseOpResult: parsingOpEmbeddingResult) {
							
							INDArray embedParseStep = parseOpResult.getEmbedding().transpose();
							
							if(i == gTruthIx) {
								embedGTruth = embedParseStep;
								////////////////
								gTruthResult = parseOpResult;
								////////////////
							}
							
							//take dot product and add bias
							double exponent = (current.mmul(embedParseStep)).getDouble(new int[]{0, 0}); 
							exponent = exponent + actionBias[parseOpResult.ruleIndex()];
							
							if(LOG.getLogLevel() == LogLevel.DEBUG) {
								LOG.debug("Op %s", options.get(i));
								LOG.debug("Exponent %s. Parse Op Embedding is %s", exponent, embedParseStep);
							}
							
							exponents.add(exponent);
							embedParseSteps.add(embedParseStep);
							
							if(exponent > maxExponent) {
								maxExponent = exponent;
							}
							
							i++;
						}
						
						//max exponent trick
						Iterator<Double> exponentIt = exponents.iterator();
						int jl = 0;
						for(INDArray embedParseStep: embedParseSteps) {
							
							double exponent = exponentIt.next();
							double score = Math.exp(exponent - maxExponent);
							
							scores[jl++] = score;
							expYj.addi(embedParseStep.transpose().muli(score));
							Z = Z + score;
						}
						
						assert i == jl;
						
						Iterator<ParsingOpEmbeddingResult> parsingOpEmbeddingIt = parsingOpEmbeddingResult.iterator();
						
						for(int j = 0; j < scores.length; j++) {
							scores[j] = scores[j]/Z;
							ParsingOpEmbeddingResult parsingOpEmbedding = parsingOpEmbeddingIt.next();
							///////////////////////////////////
							if(j != gTruthIx)
								partitionFunctionAppr.offer(Pair.of(scores[j], parsingOpEmbedding));
							////////////////////////////////////
						}
						
						if(i != jl || parsingOpEmbeddingIt.hasNext()) {
							throw new RuntimeException("Double check these lines. i" + i + ", jl = "+jl + 
									" parsingOp size " + parsingOpEmbeddingResult.size() + " hasNext " + parsingOpEmbeddingIt.hasNext());
						}
						
						if(LOG.getLogLevel() == LogLevel.DEBUG) {
							String scoreByte = Joiner.on(", ").join(scores);
							LOG.debug("ground truth %s, scores are %s and Z is %s ", gTruthIx, scoreByte, Z);
						}
						
						double logLikelihood = Math.log(scores[gTruthIx]);
						synchronized(stats) { 
							stats[0] = stats[0] + logLikelihood;
							stats[1] = stats[1] + scores[gTruthIx];
						}
						
						expYj.divi(Z);
						
						/* perform backpropagation through the entire computation graph
						 * 
						 * Loss is given by: negative log-likelihood + L2 regularization term + momentum term
						 *               
						 * gradient of loss with respect to A_pq will be: (y is the probability) 
						 * 				 { -pembed(y_i)_p + E_y_j[pembed(y_j)_p] } x_q
						 * with respect to b will be: -pembed(y_i) + E_y_j[pembed(y_j)]
						 * with respect to x_q: -\sum_p pembed(y_i)_p A_pq + E_y_j[\sum_p pembed(y_j)_p A_pq] */
						
						INDArray currentDerivative = Nd4j.getExecutioner()
														 .execAndReturn(new TanhDerivative(currentPreOutput.dup()));
						
						//gradients for this decision (be careful with inplace operations)
						INDArray decisionGradb = embedGTruth.mul(-1).addi(expYj.transpose()).muli(currentDerivative.transpose());
						INDArray decisionGradA = decisionGradb.mmul(x.transpose());
						INDArray decisionGradX = decisionGradb.transpose().mmul(A);
						
						synchronized(gradb) {
							gradb.addi(decisionGradb);
						}
						
						synchronized(gradA) {
							gradA.addi(decisionGradA);
						}
						
						//backprop through the action embedding
						
						/*ListIterator<ParsingOpEmbeddingResult> it = parsingOpEmbeddingResult
																				.listIterator();*/
						Iterator<Pair<Double, ParsingOpEmbeddingResult>> it = partitionFunctionAppr.iterator();
						
						List<Pair<ParsingOpEmbeddingResult, INDArray>> backpropParsingOp = new 
												LinkedList<Pair<ParsingOpEmbeddingResult, INDArray>>();
						
						while(it.hasNext()) {
							//int ix = it.nextIndex();
							Pair<Double, ParsingOpEmbeddingResult> item = it.next();
							ParsingOpEmbeddingResult parseOpResult = item.second(); //it.next();
							
							final INDArray error;
							/*if(ix == gTruthIx) {
								 error = current.mul(-1 + scores[ix]);
							} else {
								 error = current.mul(scores[ix]);
							}*/
							error = current.mul(item.first());
							backpropParsingOp.add(Pair.of(parseOpResult, error));
							
							///// Update gradActionBias /////
							synchronized(gradActionBias) {
								gradActionBias[parseOpResult.ruleIndex()] += item.first(); 
							}
							/////////////////////////////////
						}
						
						///////////////////////
						backpropParsingOp.add(Pair.of(gTruthResult, current.mul(-1 + scores[gTruthIx])));
						if(backpropParsingOp.size() >  k + 1) {
							throw new RuntimeException("Size is more than "+(k + 1) +" found " + backpropParsingOp.size());
						}
						///////////////////////
						
						//// Update gradActionBias ////
						synchronized(gradActionBias) {
							gradActionBias[gTruthResult.ruleIndex()] += -1 + scores[gTruthIx];
						}
						///////////////////////////////
						
						StreamSupport.stream(Spliterators
								.spliterator(backpropParsingOp, Spliterator.IMMUTABLE), true).unordered()
								.forEach(p->embedParsingOp.backProp(p.second(), p.first()));
						
						//update the errors for recurrent network
						if(!Double.isFinite(decisionGradX.sumNumber().doubleValue())) {
							
							LOG.info("Action History, a1 is %s", Helper.printVector(a1));
							LOG.info("Parse State, a2 is %s", Helper.printVector(a2));
							LOG.info("Word Embedding, a3 is %s", Helper.printVector(a3));
							LOG.info("Current is %s", Helper.printVector(current));
							LOG.info("A is %s", Helper.printMatrix(A));
							LOG.info("b is %s", Helper.printVector(b));
							LOG.info("Embed GTruth %s ", Helper.printVector(embedGTruth));
							LOG.info("expYj %s", Helper.printVector(expYj));
							LOG.info("gradX %s", Helper.printVector(decisionGradX));
							
							for(INDArray embedParseStep: embedParseSteps) {
								LOG.info("Embed Parse Step %s", Helper.printVector(embedParseStep) );
							}
							throw new ArithmeticException("Found NaN");
						}
						
						INDArray decisionErrorActionHistory = decisionGradX.get(NDArrayIndex.interval(0, dimAction));
						INDArray decisionErrorState = decisionGradX.get(NDArrayIndex.interval(dimAction, dimState + dimAction));
						INDArray decisionErrorWordBuffer = decisionGradX.get(NDArrayIndex
											 				.interval(dimState + dimAction, dimState + dimAction + dimBuffer));
						
						synchronized(errorActionHistory) {
							errorActionHistory[decision.getActionHistoryIx()]
													.addi(decisionErrorActionHistory);
						}
						
						synchronized(errorState) {							
							errorState[decision.getParserStateIx()].addi(decisionErrorState);
						}
						
						synchronized(errorWordBuffer) {
							errorWordBuffer[decision.getSentenceIx()].addi(decisionErrorWordBuffer);
						}
						
					});
				
				LOG.info("Iteration: %s, Sum of Likelihood %s, NLL %s", iter, stats[1], -stats[0]);
				
				sumLogLikelihood = sumLogLikelihood + stats[0];
				sumLikelihood = sumLikelihood + stats[1];
				
				/////////////// update A, b parameters ////////////
				final long start3 = System.currentTimeMillis();
				long diff = start3 - start2;
				time2 = time2 + diff;
				LOG.info("Time part 2 " + diff);
				
				//Add regularization term
				gradA.addi(A.mul(this.aRegularizer));
				gradb.addi(b.mul(this.bRegularizer));
				
				//Do gradient clipping
				double normA = gradA.normmaxNumber().doubleValue();
				double normb = gradb.normmaxNumber().doubleValue();
				double threshold = 1.0;
				
				if(normA > threshold) {
					gradA.muli(threshold/normA);
				}
				
				if(normb > threshold) {
					gradb.muli(threshold/normb);
				}
				
				LOG.debug("GradA norm %s, max number %s, min number %s",
								gradA.normmaxNumber().doubleValue(),
								gradA.maxNumber().doubleValue(), gradA.minNumber().doubleValue());
				LOG.debug("Gradb norm %s, max number %s, min number %s",
								gradb.normmaxNumber().doubleValue(),
								gradb.maxNumber().doubleValue(), gradb.minNumber().doubleValue());
				LOG.debug("Learning Rate %s", this.learningRate.getLearningRate());
				
				// update A,b using AdaGrad rule
				this.adaGradSumSquareA.addi(gradA.mul(gradA));
				this.adaGradSumSquareb.addi(gradb.mul(gradb));
				
				double initLearningRate = this.learningRate.getLearningRate();
				
				INDArray invertedLearningRateA = Nd4j.getExecutioner()
													.execAndReturn(new Sqrt(this.adaGradSumSquareA.dup()))
													.divi(initLearningRate);
				
				INDArray invertedLearningRateb = Nd4j.getExecutioner()
													 .execAndReturn(new Sqrt(this.adaGradSumSquareb.dup()))
													 .divi(initLearningRate);
				
				A.subi(gradA.div(invertedLearningRateA));
				b.subi(gradb.div(invertedLearningRateb));
				
				
				//////// Update Action Bias ///////////
//				for(int i = 0; i < actionBias.length; i++) {
//					
//					double grad = gradActionBias[i];
//					if(grad > 1.0) {
//						grad = 1.0;
//					}
//					
//					this.adaGradSumSquareActionBias[i] += grad*grad;
//					
//					double adaGradSumSquare = this.adaGradSumSquareActionBias[i];
//					double denom = Math.sqrt(adaGradSumSquare);
//					actionBias[i] = actionBias[i] - (initLearningRate * grad)/denom;
//				}
				
				////// backpropagate through the 3 RNNs ///////////////////////////
				final long start4 = System.currentTimeMillis();
				LOG.info("Time part 3 " +(start4 - start3));
				
				List<Pair<AbstractRecurrentNetworkHelper, INDArray[]>> backprop = 
									new LinkedList<Pair<AbstractRecurrentNetworkHelper, INDArray[]>>();
				backprop.add(Pair.of(embedActionHistory, errorActionHistory));
				backprop.add(Pair.of(embedParserState, errorState));
				backprop.add(Pair.of(embedWordBuffer, errorWordBuffer));
				
				StreamSupport.stream(Spliterators
							.spliterator(backprop, Spliterator.IMMUTABLE), true).unordered()
							.forEach(p-> p.first().backprop(p.second()));
				
				final long start5 = System.currentTimeMillis();
				long diff4 = (start5 - start4);
				LOG.info("Time part 4 " + diff4);
				time4 = time4 + diff4;
				
				//update the category vectors and flush the gradients (updates Recursive network)
				categEmbedding.updateParameters();
				categEmbedding.flushGradients();
				
				final long start6 = System.currentTimeMillis();
				LOG.info("Time part 5 " + (start6 - start5));
				
				//update the actions and flush the gradients
				embedParsingOp.updateParameters();
				embedParsingOp.flushGradients();
				
				//update tunable word embeddings and POS tag
				embedWordBuffer.updateParameters();
				embedWordBuffer.flushGradients();
				
				//Clear the stored results
				embedActionHistory.clearParsingOpEmbeddingResult();
				embedParserState.clearCategoryResults();
				
				final long start7 = System.currentTimeMillis();
				LOG.info("Time part 6 " + (start7 - start6));
				
				totalTime = totalTime + (start7 - start1);
			}
			
			if(iter == this.epoch) {
				this.setDisplay = true;
			}
			
			// End of iteration, print the likelihood for train and validation
			LOG.info("-------- train iteration %s  ------------", iter);
			this.calcCompositeBatchLikelihood(train);
			LOG.info("-------- train, end of iteration %s ------------", iter);
			
			LOG.info("-------- validation iteration %s  ------------", iter);
			double currentLogLikelihood = this.calcCompositeBatchLikelihood(validation);
			LOG.info("-------- validation, end of iteration %s ------------", iter);
			
			if((iter == 1 || iter%5 == 0) && iter != this.epoch) {
				try {
					this.logModel("epoch-" + iter);
				} catch (FileNotFoundException | UnsupportedEncodingException e) {
					throw new RuntimeException("Could not save model. Exception " + e);
				}
			}
			
			if(prevValidationLogLikelihood > currentLogLikelihood) { //Terminate if validation likelihood decreases
				LOG.info("Convergence reached. Maximum Log-Likelihood %s", prevValidationLogLikelihood);
				break;
			}
			
			prevValidationLogLikelihood = currentLogLikelihood;
		}
		
		LOG.info("Action History norm %s, term %s, average %s ", embedActionHistory.norm, 
				embedActionHistory.term, embedActionHistory.norm/(double)embedActionHistory.term);
		LOG.info("Parser State norm %s, term %s, average %s ", embedParserState.norm, 
				embedParserState.term, embedParserState.norm/(double)embedParserState.term);
		
		double totalSteps = (double)(this.epoch * trainSize);
		LOG.info("Total Time taken %s. Total instance %s, Average %s", totalTime, totalSteps, totalTime/totalSteps);
		LOG.info("Task 2: Time taken %s. Total instance %s, Average %s", time2, totalSteps, time2/totalSteps);
		LOG.info("Task 4: Time taken %s. Total instance %s, Average %s", time4, totalSteps, time4/totalSteps);
		
		if(this.saveModelAfterLearning) {
			try {
				this.logModel("end");
			} catch (FileNotFoundException | UnsupportedEncodingException e) {
				LOG.error("Failed to Log the model. Error: "+e);
			}
		}
		
		//Relcone the recurrent networks as updates have been made
		embedActionHistory.reclone();
		embedParserState.reclone();
		
		this.findGradientOnValidation(dataset);
	}
	
	/** In every epoch creates dataset  by parsing under the current model and using a multi-parse tree filter
	 *  which allows parser to create dataset using the current parameters. After creating the dataset, online SGD
	 *  is performed. Learning algorithm terminates when in a given epoch, the learner cannot improve the validation
	 *  accuracy of the created validation dataset. */
	public void fitCompositeDataSet(CreateCompositeDecisionDataset<SAMPLE, DI, MR> datasetCreator, 
			IJointModelImmutable<SituatedSentence<AMRMeta>, LogicalExpression, LogicalExpression> model) {
		
		EmbedActionHistory<MR> embedActionHistory = this.parser.getEmbedActionHistory();			
		EmbedWordBuffer embedWordBuffer = this.parser.getEmbedWordBuffer();
		EmbedParserState<MR> embedParserState = this.parser.getEmbedParserState();
		CategoryEmbedding<MR> categEmbedding = this.parser.getCategoryEmbedding();
		ParsingOpEmbedding<MR> embedParsingOp = this.parser.getEmbedParsingOp();
		INDArray A = this.parser.getAffineA();
		INDArray b = this.parser.getAffineb();
		Double[] actionBias = this.parser.getActionBias();
		
		final int dimAction = embedActionHistory.getDimension();
		final int dimState = embedParserState.getDimension();
		final int dimBuffer = embedWordBuffer.getDimension();
		
		long time2 = 0, time4 = 0, totalTime = 0;
		final int minEpoch = 5;
		
		int numIterations = 0;
		
		try {
			this.logModel("init");
		} catch (FileNotFoundException | UnsupportedEncodingException e) {
			throw new RuntimeException("Could not save model. Exception " + e);
		}
		
		final int k = this.partitionFunctionApproximationK - 1;
		final Comparator<Pair<Double, ParsingOpEmbeddingResult>> cmp  = 
				new Comparator<Pair<Double, ParsingOpEmbeddingResult>>() {
			public int compare(Pair<Double, ParsingOpEmbeddingResult> left, 
							   Pair<Double, ParsingOpEmbeddingResult> right) {
				return Double.compare(left.first(), right.first()); 
			}   
		};
		
		List<CompositeDataPoint<MR>> dataset = null;
		
		for(int iter = 1; iter <= this.epoch; iter++) {
			LOG.info("=========================");
			LOG.info("Create Training Data. Epoch %s", iter);
			
			//if(dataset == null) {
				dataset = datasetCreator.createDataset(model);
			//}
			 
			int dataSize = dataset.size();
			int trainSize = (int)(0.9*dataSize);
			List<CompositeDataPoint<MR>> train = dataset.subList(dataSize - trainSize, trainSize);//0, trainSize);
			List<CompositeDataPoint<MR>> validation = dataset.subList(0, dataSize - trainSize);//trainSize, dataSize);
			
			LOG.info("-------- train initialization epoch %s  ------------", iter);
			this.calcCompositeBatchLikelihood(train);
			LOG.info("-------- train, end of initialization ------------");
			
			LOG.info("-------- validation initialization epoch %s  ------------", iter);
			double prevValidationLogLikelihood = this.calcCompositeBatchLikelihood(validation);
			LOG.info("-------- validation, end of initialization ------------");
			
			//Induce dynamic lexical entry origin in parsing op --- this is a hacky way. Improve it in future.
			if(iter == 1) {
				embedParsingOp.induceDynamicOriginAndTemplate(dataset);
				categEmbedding.induceVectorsFromProcessedDataset(dataset);
			}
			
			LOG.info("Fitting Dataset of size %s, train %s, validation %s", 
											dataset.size(), trainSize, dataSize - trainSize);
			
			LOG.info("Fit Dataset Iteration: %s", iter);
			int exampleIndex = 0;
			double sumLogLikelihood = 0.0, sumLikelihood = 0.0;
			
			for(CompositeDataPoint<MR> pt: train) {
				LOG.info("=========================");
				LOG.info("Example: %s", ++exampleIndex);
				
				long start1 = System.currentTimeMillis();
				
				DerivationState<MR> dstate = pt.getState();
				List<ParsingOp<MR>> parsingOps = pt.getActionHistory();
				List<Pair<String, String>> wordBuffer = pt.getBuffer();
				List<CompositeDataPointDecision<MR>> decisions = pt.getDecisions();
				
				List<Pair<Integer, CompositeDataPointDecision<MR>>> enumeratedDecisions = new 
						ArrayList<Pair<Integer, CompositeDataPointDecision<MR>>>();
				int index = 0;
				for(CompositeDataPointDecision<MR> decision: decisions) {
					enumeratedDecisions.add(Pair.of(index++, decision));
				}
				
				LOG.debug("Sentence %s", pt.getSentence());
				
				// Compute embeddings of history, state and buffer
				List<Pair<AbstractRecurrentNetworkHelper, Object>> calcEmbedding = 
									new LinkedList<Pair<AbstractRecurrentNetworkHelper, Object>>();
				calcEmbedding.add(Pair.of(embedActionHistory, (Object)parsingOps));
				calcEmbedding.add(Pair.of(embedParserState, (Object)dstate));
				calcEmbedding.add(Pair.of(embedWordBuffer, (Object)wordBuffer));
				
				////////// Empirical Gradient Estimate //////////////
				//this.gradientChecker.gradientCheckCategory(enumeratedDecisions, calcEmbedding);
				/////////////////////////////////////////////////////
				
				List<Object> embeddings = StreamSupport.stream(Spliterators
							.spliterator(calcEmbedding, Spliterator.IMMUTABLE), true)
							.map(p->p.first().getAllTopLayerEmbedding(p.second()))
							.collect(Collectors.toList());
				
				INDArray[] topLayerA1 = (INDArray[])embeddings.get(0);
				INDArray[] topLayerA2 = (INDArray[])embeddings.get(1);
				INDArray[] topLayerA3 = (INDArray[])embeddings.get(2);
				
				LOG.info("Activation:: Action-History %s Parser-State %s Buffer %s", 
						Helper.mean(topLayerA1), Helper.mean(topLayerA2), Helper.mean(topLayerA3));
				LOG.info("Activation:: A %s b %s", Helper.meanAbs(A), Helper.meanAbs(b));
				
				INDArray gradA = Nd4j.zeros(A.shape()), gradb = Nd4j.zeros(b.shape());
				
				Double[] gradActionBias = new Double[actionBias.length];
				Arrays.fill(gradActionBias, 0.0);
				
				INDArray[] errorActionHistory = new INDArray[topLayerA1.length];
				INDArray[] errorState = new INDArray[topLayerA2.length];
				INDArray[] errorWordBuffer = new INDArray[topLayerA3.length];
				
				int max = Math.max(topLayerA1.length, Math.max(topLayerA2.length, topLayerA3.length));
				for(int i = 0; i < max; i++) {
					if(i < topLayerA1.length) { 
						errorActionHistory[i] = Nd4j.zeros(dimAction);
					} 
					if(i < topLayerA2.length) {
						errorState[i] = Nd4j.zeros(dimState);
					}
					if(i < topLayerA3.length) {
						errorWordBuffer[i] = Nd4j.zeros(dimBuffer); 
					}
				}
				
				double[] stats = new double[2]; 
				stats[0] = 0.0; stats[1] = 0.0; //0 is sum of log-likelihood, 1 is sum of likelihood
				
				long start2 = System.currentTimeMillis();
				LOG.info("Time part 1 " +(start2 - start1));
				
//				final int iter_ = iter;
				
				StreamSupport.stream(Spliterators
					.spliterator(decisions, Spliterator.IMMUTABLE), false)
					.forEach(decision -> {
						
						INDArray a1 = topLayerA1[decision.getActionHistoryIx()];
						INDArray a2 = topLayerA2[decision.getParserStateIx()];
						INDArray a3 = topLayerA3[decision.getSentenceIx()];
						
						INDArray x = Nd4j.concat(1, a1, a2, a3).transpose();

						List<ParsingOp<MR>> options = decision.getPossibleActions();
						int gTruthIx = decision.getGTruthIx();
						//LOG.info("Number of options %s", options.size());
						
						//computes g(A[a1; a2; a3]+b)
						INDArray currentPreOutput = A.mmul(x).addi(b).transposei(); 
						INDArray current = Nd4j.getExecutioner()
											   .execAndReturn(new /*RectifedLinear*/Tanh(currentPreOutput.dup()));
						
						INDArray embedGTruth = null;
						
						//some data-structures for computing gradients wrt A,b,x
						INDArray expYj = Nd4j.zeros(embedParsingOp.getDimension()); 
						
						//probability of taking each action
						Double[] scores = new Double[options.size()];
						double Z = 0; //partition function 
						int i = 0;
						
						List<ParsingOpEmbeddingResult> parsingOpEmbeddingResult = 
													   new LinkedList<ParsingOpEmbeddingResult>();
						
						List<Double> exponents = new LinkedList<Double>();
						List<INDArray> embedParseSteps = new LinkedList<INDArray>();
						Double maxExponent = Double.NEGATIVE_INFINITY;
										
						parsingOpEmbeddingResult = StreamSupport.stream(Spliterators
													.spliterator(options, Spliterator.IMMUTABLE), true)
													.map(op->embedParsingOp.getEmbedding(op))
													.collect(Collectors.toList());
						
						LOG.debug("X %s, current %s", x, current);
						
						DirectAccessBoundedPriorityQueue<Pair<Double, ParsingOpEmbeddingResult>> partitionFunctionAppr 
								= new DirectAccessBoundedPriorityQueue<Pair<Double, ParsingOpEmbeddingResult>>(k, cmp);
						ParsingOpEmbeddingResult gTruthResult = null;
						
						for(ParsingOpEmbeddingResult parseOpResult: parsingOpEmbeddingResult) {
							
							INDArray embedParseStep = parseOpResult.getEmbedding().transpose();
							
							if(i == gTruthIx) {
								embedGTruth = embedParseStep;
								gTruthResult = parseOpResult;
							}
							
							//take dot product and add bias
							double exponent = (current.mmul(embedParseStep)).getDouble(new int[]{0, 0}); 
							exponent = exponent + actionBias[parseOpResult.ruleIndex()];
							
							if(LOG.getLogLevel() == LogLevel.DEBUG) {
								LOG.debug("Op %s", options.get(i));
								LOG.debug("Exponent %s. Parse Op Embedding is %s", exponent, embedParseStep);
							}
							
							exponents.add(exponent);
							embedParseSteps.add(embedParseStep);
							
							if(exponent > maxExponent) {
								maxExponent = exponent;
							}
							
							i++;
						}
						
						//max exponent trick
						Iterator<Double> exponentIt = exponents.iterator();
						int jl = 0;
						for(INDArray embedParseStep: embedParseSteps) {
							
							double exponent = exponentIt.next();
							double score = Math.exp(exponent - maxExponent);
							
							scores[jl++] = score;
							expYj.addi(embedParseStep.transpose().muli(score));
							Z = Z + score;
						}
						
						assert i == jl;
						
						Iterator<ParsingOpEmbeddingResult> parsingOpEmbeddingIt = parsingOpEmbeddingResult.iterator();
						
						for(int j = 0; j < scores.length; j++) {
							scores[j] = scores[j]/Z;
							ParsingOpEmbeddingResult parsingOpEmbedding = parsingOpEmbeddingIt.next();
							
							if(j != gTruthIx)
								partitionFunctionAppr.offer(Pair.of(scores[j], parsingOpEmbedding));
						}
						
						if(i != jl || parsingOpEmbeddingIt.hasNext()) {
							throw new RuntimeException("Double check these lines. i" + i + ", jl = "+jl + 
									" parsingOp size " + parsingOpEmbeddingResult.size() + " hasNext " + parsingOpEmbeddingIt.hasNext());
						}
						
						if(LOG.getLogLevel() == LogLevel.DEBUG) {
							String scoreByte = Joiner.on(", ").join(scores);
							LOG.debug("ground truth %s, scores are %s and Z is %s ", gTruthIx, scoreByte, Z);
						}
						
						double logLikelihood = Math.log(scores[gTruthIx]);
						synchronized(stats) { 
							stats[0] = stats[0] + logLikelihood;
							stats[1] = stats[1] + scores[gTruthIx];
						}
						
						expYj.divi(Z);
						
						/* perform backpropagation through the entire computation graph
						 * 
						 * Loss is given by: negative log-likelihood + L2 regularization term + momentum term
						 *               
						 * gradient of loss with respect to A_pq will be: (y is the probability) 
						 * 				 { -pembed(y_i)_p + E_y_j[pembed(y_j)_p] } x_q
						 * with respect to b will be: -pembed(y_i) + E_y_j[pembed(y_j)]
						 * with respect to x_q: -\sum_p pembed(y_i)_p A_pq + E_y_j[\sum_p pembed(y_j)_p A_pq] */
						
						INDArray currentDerivative = Nd4j.getExecutioner()
														 .execAndReturn(new /*Step*/TanhDerivative(currentPreOutput.dup()));
						
						//gradients for this decision (be careful with inplace operations)
						INDArray decisionGradb = embedGTruth.mul(-1).addi(expYj.transpose()).muli(currentDerivative.transpose());
						INDArray decisionGradA = decisionGradb.mmul(x.transpose());
						INDArray decisionGradX = decisionGradb.transpose().mmul(A);
						
						synchronized(gradb) {
							gradb.addi(decisionGradb);
						}
						
						synchronized(gradA) {
							gradA.addi(decisionGradA);
						}
						
						//backprop through the action embedding
						Iterator<Pair<Double, ParsingOpEmbeddingResult>> it = partitionFunctionAppr.iterator();
						
						List<Pair<ParsingOpEmbeddingResult, INDArray>> backpropParsingOp = new 
												LinkedList<Pair<ParsingOpEmbeddingResult, INDArray>>();
						
						while(it.hasNext()) {
							Pair<Double, ParsingOpEmbeddingResult> item = it.next();
							ParsingOpEmbeddingResult parseOpResult = item.second(); //it.next();
							
							final INDArray error = current.mul(item.first());
							backpropParsingOp.add(Pair.of(parseOpResult, error));
							
							///// Update gradActionBias /////
//							synchronized(gradActionBias) {
//								gradActionBias[parseOpResult.ruleIndex()] += item.first(); 
//							}
							/////////////////////////////////
						}
						
						///////////////////////
						backpropParsingOp.add(Pair.of(gTruthResult, current.mul(-1 + scores[gTruthIx])));
						if(backpropParsingOp.size() >  k + 1) {
							throw new RuntimeException("Size is more than "+(k + 1) +" found " + backpropParsingOp.size());
						}
						///////////////////////
						
						//// Update gradActionBias ////
//						synchronized(gradActionBias) {
//							gradActionBias[gTruthResult.ruleIndex()] += -1 + scores[gTruthIx];
//						}
						///////////////////////////////
						
						//if(iter_ > 2) { //for first 2 epochs we only train via RNNs
						StreamSupport.stream(Spliterators
								.spliterator(backpropParsingOp, Spliterator.IMMUTABLE), true).unordered()
								.forEach(p->embedParsingOp.backProp(p.second(), p.first()));
						//}
						
						//update the errors for recurrent network
						if(!Double.isFinite(decisionGradX.sumNumber().doubleValue())) {
							
							LOG.info("Action History, a1 is %s", Helper.printVector(a1));
							LOG.info("Parse State, a2 is %s", Helper.printVector(a2));
							LOG.info("Word Embedding, a3 is %s", Helper.printVector(a3));
							LOG.info("Current is %s", Helper.printVector(current));
							LOG.info("A is %s", Helper.printMatrix(A));
							LOG.info("b is %s", Helper.printVector(b));
							LOG.info("Embed GTruth %s ", Helper.printVector(embedGTruth));
							LOG.info("expYj %s", Helper.printVector(expYj));
							LOG.info("gradX %s", Helper.printVector(decisionGradX));
							
							for(INDArray embedParseStep: embedParseSteps) {
								LOG.info("Embed Parse Step %s", Helper.printVector(embedParseStep) );
							}
							throw new ArithmeticException("Found NaN");
						}
						
						INDArray decisionErrorActionHistory = decisionGradX.get(NDArrayIndex.interval(0, dimAction));
						INDArray decisionErrorState = decisionGradX.get(NDArrayIndex.interval(dimAction, dimState + dimAction));
						INDArray decisionErrorWordBuffer = decisionGradX.get(NDArrayIndex
											 				.interval(dimState + dimAction, dimState + dimAction + dimBuffer));
						
						synchronized(errorActionHistory) {
							errorActionHistory[decision.getActionHistoryIx()]
													.addi(decisionErrorActionHistory);
						}
						
						synchronized(errorState) {							
							errorState[decision.getParserStateIx()].addi(decisionErrorState);
						}
						
						synchronized(errorWordBuffer) {
							errorWordBuffer[decision.getSentenceIx()].addi(decisionErrorWordBuffer);
						}
						
					});
				
				LOG.info("Iteration: %s, Sum of Likelihood %s, NLL %s", iter, stats[1], -stats[0]);
				
				sumLogLikelihood = sumLogLikelihood + stats[0];
				sumLikelihood = sumLikelihood + stats[1];
				
				/////////////// update A, b parameters ////////////
				
				final long start3 = System.currentTimeMillis();
				long diff = start3 - start2;
				time2 = time2 + diff;
				LOG.info("Time part 2 " + diff);
				
				LOG.info("Gradient:: Action-History %s Parser-State %s Buffer %s", 
						Helper.mean(errorActionHistory), Helper.mean(errorState), Helper.mean(errorWordBuffer));
				LOG.info("Gradient:: A %s b %s", Helper.meanAbs(gradA), Helper.meanAbs(gradb));
				
				//if(iter > 2) {
				//Add regularization term
				gradA.addi(A.mul(this.aRegularizer));
				gradb.addi(b.mul(this.bRegularizer));
				
				//Do gradient clipping
				double normA = gradA.normmaxNumber().doubleValue();
				double normb = gradb.normmaxNumber().doubleValue();
				double threshold = 1.0;
				
				if(normA > threshold) {
					gradA.muli(threshold/normA);
				}
				
				if(normb > threshold) {
					gradb.muli(threshold/normb);
				}
				
				// update A,b using AdaGrad rule
				this.adaGradSumSquareA.addi(gradA.mul(gradA));
				this.adaGradSumSquareb.addi(gradb.mul(gradb));
				
				double initLearningRate = this.learningRate.getLearningRate();
				
				INDArray invertedLearningRateA = Nd4j.getExecutioner()
													.execAndReturn(new Sqrt(this.adaGradSumSquareA.dup()))
													.divi(initLearningRate);
				
				INDArray invertedLearningRateb = Nd4j.getExecutioner()
													 .execAndReturn(new Sqrt(this.adaGradSumSquareb.dup()))
													 .divi(initLearningRate);
				
				A.subi(gradA.div(invertedLearningRateA));
				b.subi(gradb.div(invertedLearningRateb));
				//}
				
				//////// Update Action Bias ///////////
//				for(int i = 0; i < actionBias.length; i++) {
//					
//					double grad = gradActionBias[i];
//					if(grad > 1.0) {
//						grad = 1.0;
//					}
//					
//					this.adaGradSumSquareActionBias[i] += grad*grad;
//					
//					double adaGradSumSquare = this.adaGradSumSquareActionBias[i];
//					double denom = Math.sqrt(adaGradSumSquare);
//					actionBias[i] = actionBias[i] - (initLearningRate * grad)/denom;
//				}
				
				////// backpropagate through the 3 RNNs ///////////////////////////
				final long start4 = System.currentTimeMillis();
				LOG.info("Time part 3 " +(start4 - start3));
				
				List<Pair<AbstractRecurrentNetworkHelper, INDArray[]>> backprop = 
									new LinkedList<Pair<AbstractRecurrentNetworkHelper, INDArray[]>>();
				backprop.add(Pair.of(embedActionHistory, errorActionHistory));
				backprop.add(Pair.of(embedParserState, errorState));
				backprop.add(Pair.of(embedWordBuffer, errorWordBuffer));
				
				StreamSupport.stream(Spliterators
							.spliterator(backprop, Spliterator.IMMUTABLE), true).unordered()
							.forEach(p-> p.first().backprop(p.second()));
				
				final long start5 = System.currentTimeMillis();
				long diff4 = (start5 - start4);
				LOG.info("Time part 4 " + diff4);
				time4 = time4 + diff4;
				
				//update the category vectors and flush the gradients (updates Recursive network)
				categEmbedding.updateParameters();
				categEmbedding.flushGradients();
				
				final long start6 = System.currentTimeMillis();
				LOG.info("Time part 5 " + (start6 - start5));
				
				//update the actions and flush the gradients
				embedParsingOp.updateParameters();
				embedParsingOp.flushGradients();
				
				//update tunable word embeddings and POS tag
				embedWordBuffer.updateParameters();
				embedWordBuffer.flushGradients();
				
				//Clear the stored results
				embedActionHistory.clearParsingOpEmbeddingResult();
				embedParserState.clearCategoryResults();
				
				final long start7 = System.currentTimeMillis();
				LOG.info("Time part 6 " + (start7 - start6));
				
				totalTime = totalTime + (start7 - start1);
				LOG.info("Performed One Update %s", System.currentTimeMillis());
			}
			
			numIterations = numIterations + train.size();
			
			if(iter == this.epoch) {
				this.setDisplay = true;
			}
			
			// End of epoch, calculate the log-likelihood for train and validation
			LOG.info("-------- train iteration %s  ------------", iter);
			this.calcCompositeBatchLikelihood(train);
			LOG.info("-------- train, end of iteration %s ------------", iter);
			
			LOG.info("-------- validation iteration %s  ------------", iter);
			double currentLogLikelihood = this.calcCompositeBatchLikelihood(validation);
			LOG.info("-------- validation, end of iteration %s ------------", iter);
			
			// Log the current model //////////
			/// Save it for the first epoch and every fifth epoch except the last (which is saved separately)
			if((iter == 1 || iter%5 == 0) && iter != this.epoch) {
				try {
					this.logModel("epoch-" + iter);
				} catch (FileNotFoundException | UnsupportedEncodingException e) {
					throw new RuntimeException("Could not save model. Exception " + e);
				}
			}
			
			// Termination Condition /////////
			// Terminate if validation likelihood has not decreased in this epoch
			// and minimum number of epochs have been covered. A max epoch constraint is ensured by the for loop.
			if(prevValidationLogLikelihood > currentLogLikelihood && iter > minEpoch) { 
				LOG.info("Convergence reached. Maximum Log-Likelihood %s", prevValidationLogLikelihood);
				
				this.setDisplay = true;
				
				// End of iteration, print the likelihood for train and validation
				LOG.info("-------- train iteration %s  ------------", iter);
				this.calcCompositeBatchLikelihood(train);
				LOG.info("-------- train, end of iteration %s ------------", iter);
				
				LOG.info("-------- validation iteration %s  ------------", iter);
				this.calcCompositeBatchLikelihood(validation);
				LOG.info("-------- validation, end of iteration %s ------------", iter);
				
				this.setDisplay = false;
				
				//this.findGradientOnValidation(dataset);
				break;
			}
			
			// Relcone the recurrent networks as updates have been made
			// This needs to be done since we are doing data creation after every epoch.
			embedActionHistory.reclone();
			embedParserState.reclone();
		}
		
		LOG.info("Action History norm %s, term %s, average %s ", embedActionHistory.norm, 
				embedActionHistory.term, embedActionHistory.norm/(double)embedActionHistory.term);
		LOG.info("Parser State norm %s, term %s, average %s ", embedParserState.norm, 
				embedParserState.term, embedParserState.norm/(double)embedParserState.term);
		
		double totalSteps = (double)(numIterations);
		LOG.info("Total Time taken %s. Total instance %s, Average %s", totalTime, totalSteps, totalTime/totalSteps);
		LOG.info("Task 2: Time taken %s. Total instance %s, Average %s", time2, totalSteps, time2/totalSteps);
		LOG.info("Task 4: Time taken %s. Total instance %s, Average %s", time4, totalSteps, time4/totalSteps);
		
		if(this.saveModelAfterLearning) {
			try {
				this.logModel("end");
			} catch (FileNotFoundException | UnsupportedEncodingException e) {
				LOG.error("Failed to Log the model. Error: "+e);
			}
		}
		
		//Relcone the recurrent networks as updates have been made
		embedActionHistory.reclone();
		embedParserState.reclone();
	}
	
	public double calcCompositeBatchLikelihoodTopLayer(List<CompositeDataPoint<MR>> batch) {
		
		EmbedActionHistory<MR> embedActionHistory = this.parser.getEmbedActionHistory();			
		EmbedWordBuffer embedWordBuffer = this.parser.getEmbedWordBuffer();
		EmbedParserState<MR> embedParserState = this.parser.getEmbedParserState();
		ParsingOpEmbedding<MR> embedParsingOp = this.parser.getEmbedParsingOp();
		TopLayerMLP topLayer = this.parser.getTopLayer();
		
		final int dimAction = embedActionHistory.getDimension();
		final int dimState = embedParserState.getDimension();
		final int dimBuffer = embedWordBuffer.getDimension();
		
		int exampleIndex = 0, totalDecisions = 0;
		
		AtomicInteger correct = new AtomicInteger();
		double sumLogLikelihood = 0.0, sumLikelihood = 0.0;
		
		for(CompositeDataPoint<MR> pt: batch) {
			LOG.info("=========================");
			LOG.info("Example: %s", ++exampleIndex);
			
			DerivationState<MR> dstate = pt.getState();
			List<ParsingOp<MR>> parsingOps = pt.getActionHistory();
			List<Pair<String, String>> wordBuffer = pt.getBuffer();
			List<CompositeDataPointDecision<MR>> decisions = pt.getDecisions();
			
			LOG.debug("Sentence %s", pt.getSentence());
			
			// Compute embeddings of history, state and buffer
			List<Pair<AbstractRecurrentNetworkHelper, Object>> calcEmbedding = 
								new LinkedList<Pair<AbstractRecurrentNetworkHelper, Object>>();
			calcEmbedding.add(Pair.of(embedActionHistory, (Object)parsingOps));
			calcEmbedding.add(Pair.of(embedParserState, (Object)dstate));
			calcEmbedding.add(Pair.of(embedWordBuffer, (Object)wordBuffer));
			
			List<Object> embeddings = StreamSupport.stream(Spliterators
						.spliterator(calcEmbedding, Spliterator.IMMUTABLE), true)
						.map(p->p.first().getAllTopLayerEmbedding(p.second()))
						.collect(Collectors.toList());
			
			INDArray[] topLayerA1 = (INDArray[])embeddings.get(0);
			INDArray[] topLayerA2 = (INDArray[])embeddings.get(1);
			INDArray[] topLayerA3 = (INDArray[])embeddings.get(2);
					
			INDArray[] errorActionHistory = new INDArray[topLayerA1.length];
			INDArray[] errorState = new INDArray[topLayerA2.length];
			INDArray[] errorWordBuffer = new INDArray[topLayerA3.length];
			
			int max = Math.max(topLayerA1.length, Math.max(topLayerA2.length, topLayerA3.length));
			for(int i = 0; i < max; i++) {
				if(i < topLayerA1.length) { 
					errorActionHistory[i] = Nd4j.zeros(dimAction);
				} 
				if(i < topLayerA2.length) {
					errorState[i] = Nd4j.zeros(dimState);
				}
				if(i < topLayerA3.length) {
					errorWordBuffer[i] = Nd4j.zeros(dimBuffer); 
				}
			}
			
			double[] stats = new double[2]; 
			stats[0] = 0.0; stats[1] = 0.0; //0 is sum of log-likelihood, 1 is sum of likelihood
						
			List<Pair<Integer, CompositeDataPointDecision<MR>>> enumeratedDecisions = new 
										ArrayList<Pair<Integer, CompositeDataPointDecision<MR>>>();
			int index = 0;
			for(CompositeDataPointDecision<MR> decision: decisions) {
				enumeratedDecisions.add(Pair.of(index++, decision));
			}
			
			final int numDecisions = decisions.size();
			totalDecisions = totalDecisions + numDecisions;
			INDArray[] states = new INDArray[numDecisions];
			@SuppressWarnings("unchecked")
			List<ParsingOpEmbeddingResult>[] parsingOpEmbeddingResults = new List[numDecisions];
			
			StreamSupport.stream(Spliterators
					.spliterator(enumeratedDecisions, Spliterator.IMMUTABLE), false)
					.forEach(enumeratedDecision -> {
						
						final int ix = enumeratedDecision.first();
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
			
			//Call the top layer to give all the exponents
			List<Double>[] allDecisionExponents = topLayer.getEmbedding(parsingOpEmbeddingResults, states);
			
			StreamSupport.stream(Spliterators
				.spliterator(enumeratedDecisions, Spliterator.IMMUTABLE), false)
				.forEach(enumeratedDecision -> {
					
					final int ix = enumeratedDecision.first();
					CompositeDataPointDecision<MR> decision = enumeratedDecision.second();
					
					List<ParsingOp<MR>> options = decision.getPossibleActions();
					int gTruthIx = decision.getGTruthIx();
					
					//probability of taking each action
					Double[] scores = new Double[options.size()];
					double Z = 0; //partition function 
					int i = 0;
					
					List<ParsingOpEmbeddingResult> parsingOpEmbeddingResult = parsingOpEmbeddingResults[ix];
					Double maxExponent = Double.NEGATIVE_INFINITY;
					
					List<Double> exponents = allDecisionExponents[ix];
					
					for(ParsingOpEmbeddingResult parseOpResult: parsingOpEmbeddingResult) {
						
						INDArray embedParseStep = parseOpResult.getEmbedding().transpose();
						double exponent = exponents.get(i);
						
						if(LOG.getLogLevel() == LogLevel.DEBUG) {
							LOG.debug("Op %s", options.get(i));
							LOG.debug("Exponent %s. Parse Op Embedding is %s", exponent, embedParseStep);
						}
						
						//embedParseSteps.add(embedParseStep);
						
						if(exponent > maxExponent) {
							maxExponent = exponent;
						}
						
						i++;
					}
					
					//max exponent trick
					int jl = 0;
					for(double exponent: exponents) {
						double score = Math.exp(exponent - maxExponent);
						scores[jl++] = score;
						Z = Z + score;
					}
					
					assert i == jl;
					
					int maxScore = 0;
					for(int j=0; j < scores.length; j++) {
						scores[j] = scores[j]/Z;
						
						if(scores[j] > scores[maxScore]) {
							maxScore = j;
						}
					}
					
					double logLikelihood = Math.log(scores[gTruthIx]);
					synchronized(stats) { 
						stats[0] = stats[0] + logLikelihood;
						stats[1] = stats[1] + scores[gTruthIx];
					}
					
					if(scores[maxScore] == scores[gTruthIx]) {
						correct.incrementAndGet();
					}
					
					/////////////
					if(this.setDisplay) {
						if(scores[gTruthIx] < scores[maxScore]) { //currently printing the really bad ones
							LOG.info("Decision Index %s", ix);
							LOG.info("Right parsing action score %s -> %s ", scores[gTruthIx], options.get(gTruthIx));
							LOG.info("Highest scoring parsing action score %s -> %s", scores[maxScore], options.get(maxScore));
						}
					}
					/////////////
				});
				
			LOG.info("Example Index %s, Sum of Likelihood %s, NLL %s, sentence: %s", 
					exampleIndex, stats[1], -stats[0], pt.getSentence());
			
			sumLogLikelihood = sumLogLikelihood + stats[0];
			sumLikelihood = sumLikelihood + stats[1];
			
			//Clear the stored results
			embedActionHistory.clearParsingOpEmbeddingResult();
			embedParserState.clearCategoryResults();
		}		
					
		LOG.info("Sum Log-Likelihood %s, Sum Likelihood %s, Correct %s out of %s",
								sumLogLikelihood, sumLikelihood, correct.get(), totalDecisions);
		
		return sumLogLikelihood;
	}
			
	/** In every epoch creates dataset  by parsing under the current model and using a multi-parse tree filter
	 *  which allows parser to create dataset using the current parameters. After creating the dataset, online SGD
	 *  is performed. Learning algorithm terminates when in a given epoch, the learner cannot improve the validation
	 *  accuracy of the created validation dataset. */
	public void fitCompositeDataSetTopLayer(CreateCompositeDecisionDataset<SAMPLE, DI, MR> datasetCreator, 
			IJointModelImmutable<SituatedSentence<AMRMeta>, LogicalExpression, LogicalExpression> model) {
		
		EmbedActionHistory<MR> embedActionHistory = this.parser.getEmbedActionHistory();			
		EmbedWordBuffer embedWordBuffer = this.parser.getEmbedWordBuffer();
		EmbedParserState<MR> embedParserState = this.parser.getEmbedParserState();
		CategoryEmbedding<MR> categEmbedding = this.parser.getCategoryEmbedding();
		ParsingOpEmbedding<MR> embedParsingOp = this.parser.getEmbedParsingOp();
		TopLayerMLP topLayer = this.parser.getTopLayer();
		
		final int dimParsingOp = embedParsingOp.getDimension();
		final int dimAction = embedActionHistory.getDimension();
		final int dimState = embedParserState.getDimension();
		final int dimBuffer = embedWordBuffer.getDimension();
		
		long time2 = 0, time4 = 0, totalTime = 0;
		final int minEpoch = 5;
		
		int numIterations = 0;
		
		try {
			this.logModel("init");
		} catch (FileNotFoundException | UnsupportedEncodingException e) {
			throw new RuntimeException("Could not save model. Exception " + e);
		}
		
		final int k = this.partitionFunctionApproximationK - 1;
		final Comparator<Pair<Double, Integer>> cmp  = 
				new Comparator<Pair<Double, Integer>>() {
			public int compare(Pair<Double, Integer> left, 
							   Pair<Double, Integer> right) {
				return Double.compare(left.first(), right.first()); 
			}   
		};
		
		for(int iter = 1; iter <= this.epoch; iter++) {
			LOG.info("=========================");
			LOG.info("Create Training Data. Epoch %s", iter);
			
			List<CompositeDataPoint<MR>> dataset = datasetCreator.createDataset(model);
			
			int dataSize = dataset.size();
			int trainSize = (int)(0.9*dataSize);
			List<CompositeDataPoint<MR>> train = dataset.subList(dataSize - trainSize, trainSize);//0, trainSize);
			List<CompositeDataPoint<MR>> validation = dataset.subList(0, dataSize - trainSize);//trainSize, dataSize);
			
			LOG.info("-------- train initialization epoch %s  ------------", iter);
			this.calcCompositeBatchLikelihoodTopLayer(train);
			LOG.info("-------- train, end of initialization ------------");
			
			LOG.info("-------- validation initialization epoch %s  ------------", iter);
			double prevValidationLogLikelihood = this.calcCompositeBatchLikelihoodTopLayer(validation);
			LOG.info("-------- validation, end of initialization ------------");
			
			//Induce dynamic lexical entry origin in parsing op --- this is a hacky way. Improve it in future.
			if(iter == 1) {
				embedParsingOp.induceDynamicOriginAndTemplate(dataset);
				categEmbedding.induceVectorsFromProcessedDataset(dataset);
			}
			
			LOG.info("Fitting Dataset of size %s, train %s, validation %s", 
												dataset.size(), trainSize, dataSize - trainSize);
			
			LOG.info("Fit Dataset Iteration: %s", iter);
			int exampleIndex = 0;
			double sumLogLikelihood = 0.0, sumLikelihood = 0.0;
			
			for(CompositeDataPoint<MR> pt: train) {
				LOG.info("=========================");
				LOG.info("Example: %s", ++exampleIndex);
				
				long start1 = System.currentTimeMillis();
				
				DerivationState<MR> dstate = pt.getState();
				List<ParsingOp<MR>> parsingOps = pt.getActionHistory();
				List<Pair<String, String>> wordBuffer = pt.getBuffer();
				List<CompositeDataPointDecision<MR>> decisions = pt.getDecisions();
				
				List<Pair<Integer, CompositeDataPointDecision<MR>>> enumeratedDecisions = new 
						ArrayList<Pair<Integer, CompositeDataPointDecision<MR>>>();
				int index = 0;
				for(CompositeDataPointDecision<MR> decision: decisions) {
					enumeratedDecisions.add(Pair.of(index++, decision));
				}
				
				LOG.debug("Sentence %s", pt.getSentence());
				
				// Compute embeddings of history, state and buffer
				List<Pair<AbstractRecurrentNetworkHelper, Object>> calcEmbedding = 
									new LinkedList<Pair<AbstractRecurrentNetworkHelper, Object>>();
				calcEmbedding.add(Pair.of(embedActionHistory, (Object)parsingOps));
				calcEmbedding.add(Pair.of(embedParserState, (Object)dstate));
				calcEmbedding.add(Pair.of(embedWordBuffer, (Object)wordBuffer));
				
				////////// Empirical Gradient Estimate //////////////
//				this.gradientChecker.gradientCheckCategory(enumeratedDecisions, calcEmbedding);
				/////////////////////////////////////////////////////
				
				List<Object> embeddings = StreamSupport.stream(Spliterators
							.spliterator(calcEmbedding, Spliterator.IMMUTABLE), true)
							.map(p->p.first().getAllTopLayerEmbedding(p.second()))
							.collect(Collectors.toList());
				
				INDArray[] topLayerA1 = (INDArray[])embeddings.get(0);
				INDArray[] topLayerA2 = (INDArray[])embeddings.get(1);
				INDArray[] topLayerA3 = (INDArray[])embeddings.get(2);
						
				INDArray[] errorActionHistory = new INDArray[topLayerA1.length];
				INDArray[] errorState = new INDArray[topLayerA2.length];
				INDArray[] errorWordBuffer = new INDArray[topLayerA3.length];
				
				int max = Math.max(topLayerA1.length, Math.max(topLayerA2.length, topLayerA3.length));
				for(int i = 0; i < max; i++) {
					if(i < topLayerA1.length) { 
						errorActionHistory[i] = Nd4j.zeros(dimAction);
					} 
					if(i < topLayerA2.length) {
						errorState[i] = Nd4j.zeros(dimState);
					}
					if(i < topLayerA3.length) {
						errorWordBuffer[i] = Nd4j.zeros(dimBuffer); 
					}
				}
				
				double[] stats = new double[2]; 
				stats[0] = 0.0; stats[1] = 0.0; //0 is sum of log-likelihood, 1 is sum of likelihood
				
				long start2 = System.currentTimeMillis();
				LOG.info("Time part 1 " +(start2 - start1));
				
				//Gradient check
//				Pair<Double, Double> empiricalGrad = this.gradientChecker.gradientCheck
//												(enumeratedDecisions, topLayerA1, topLayerA2, topLayerA3);
				
				final int numDecisions = decisions.size();
				INDArray[] states = new INDArray[numDecisions];
				@SuppressWarnings("unchecked")
				List<ParsingOpEmbeddingResult>[] parsingOpEmbeddingResults = new List[numDecisions];
				
				StreamSupport.stream(Spliterators
						.spliterator(enumeratedDecisions, Spliterator.IMMUTABLE), true)
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
				
				//Call the top layer to give all the exponents
				List<Double>[] allDecisionExponents = topLayer.getEmbedding(parsingOpEmbeddingResults, states);
				
				@SuppressWarnings("unchecked")
				List<Double>[] allDecisionErrorExponents = new List[numDecisions];
				
				@SuppressWarnings("unchecked")
				DirectAccessBoundedPriorityQueue<Pair<Double, Integer>>[] allDecisionPartitionFunctionAppr = 
												new DirectAccessBoundedPriorityQueue[numDecisions];
				
				StreamSupport.stream(Spliterators
					.spliterator(enumeratedDecisions, Spliterator.IMMUTABLE), true)
					.forEach(enumeratedDecision -> {
						
						int ix = enumeratedDecision.first();
						CompositeDataPointDecision<MR> decision = enumeratedDecision.second();
						
						List<ParsingOp<MR>> options = decision.getPossibleActions();
						final int gTruthIx = decision.getGTruthIx();
						
						//probability of taking each action
						Double[] scores = new Double[options.size()];
						double Z = 0; //partition function 
						int i = 0;
						
						List<ParsingOpEmbeddingResult> parsingOpEmbeddingResult = parsingOpEmbeddingResults[ix];
						
						Double maxExponent = Double.NEGATIVE_INFINITY;
						
						///////////////////////// Debug ////////////////////////
						DirectAccessBoundedPriorityQueue<Pair<Double, Integer>> partitionFunctionAppr 
										   = new DirectAccessBoundedPriorityQueue<Pair<Double, Integer>>(k, cmp);
						///////////////////////////////////////////////////////
						
						List<Double> exponents = allDecisionExponents[ix];
						
						for(ParsingOpEmbeddingResult parseOpResult: parsingOpEmbeddingResult) {
							
							INDArray embedParseStep = parseOpResult.getEmbedding().transpose();
							double exponent = exponents.get(i);
							
							if(LOG.getLogLevel() == LogLevel.DEBUG) {
								LOG.debug("Op %s", options.get(i));
								LOG.debug("Exponent %s. Parse Op Embedding is %s", exponent, embedParseStep);
							}
							
							if(exponent > maxExponent) {
								maxExponent = exponent;
							}
							
							i++;
						}
						
						//max exponent trick
						int jl = 0;
						for(double exponent: exponents) {
							double score = Math.exp(exponent - maxExponent);
							scores[jl++] = score;
							Z = Z + score;
						}
						
						assert i == jl;
						
						for(int j = 0; j < scores.length; j++) {
							scores[j] = scores[j]/Z;
							///////////////////////////////////
							if(j != gTruthIx)
								partitionFunctionAppr.offer(Pair.of(scores[j], j));
							////////////////////////////////////
						}
						
						if(i != jl) {
							throw new RuntimeException("Double check these lines. i" + i + ", jl = "+jl + 
									" parsingOp size " + parsingOpEmbeddingResult.size());
						}
						
						if(LOG.getLogLevel() == LogLevel.DEBUG) {
							String scoreByte = Joiner.on(", ").join(scores);
							LOG.debug("ground truth %s, scores are %s and Z is %s ", gTruthIx, scoreByte, Z);
						}
						
						double logLikelihood = Math.log(scores[gTruthIx]);
						synchronized(stats) { 
							stats[0] = stats[0] + logLikelihood;
							stats[1] = stats[1] + scores[gTruthIx];
						}
						
						// compute error for every exponent in this decisions
						List<Double> errorExponent = new ArrayList<Double>(exponents.size());
						for(int j = 0; j < scores.length; j++) {
							if(j == gTruthIx) {
								errorExponent.add(-1 + scores[j]);
							} else {
								errorExponent.add(scores[j]);
							}
						}
						
						allDecisionErrorExponents[ix] = errorExponent;
						allDecisionPartitionFunctionAppr[ix] = partitionFunctionAppr;
					});
				
					//////// Backprop through the top layer //////////
				
					List<INDArray>[] allDecisionInputError = topLayer.backprop(allDecisionErrorExponents); 
					//////////////////////////////////////////////////
				
					StreamSupport.stream(Spliterators
						.spliterator(enumeratedDecisions, Spliterator.IMMUTABLE), true).unordered()
						.forEach(enumeratedDecision -> {
					
						final int ix = enumeratedDecision.first();
						CompositeDataPointDecision<MR> decision = enumeratedDecision.second();
						
						final int gTruthIx = decision.getGTruthIx();
						List<ParsingOpEmbeddingResult> parsingOpEmbeddingResult = parsingOpEmbeddingResults[ix];
						
						DirectAccessBoundedPriorityQueue<Pair<Double, Integer>> partitionFunctionAppr = 
																				allDecisionPartitionFunctionAppr[ix];
						List<INDArray> inputErrors = allDecisionInputError[ix]; //error for every loss and the state
						List<INDArray> parsingOpError = new ArrayList<INDArray>();
						INDArray decisionGradX = Nd4j.zeros(dimAction + dimState + dimBuffer);
						
						for(INDArray inputError : inputErrors) {
							parsingOpError.add(inputError.get(NDArrayIndex.interval(0, dimParsingOp)));
							decisionGradX.addi(inputError.get(NDArrayIndex.interval(dimParsingOp, inputError.size(1))));
						}
						
						
//						if(ix == 0) {
//							double estimateGradParsingOp = parsingOpError.get(0).getDouble(new int[]{0, 0});
//							LOG.info("Gradient Checks for Parsing Op. Empirical %s, Estimate %s", 
//												empiricalGrad.first(), estimateGradParsingOp);
//							double estimateGradX = decisionGradX.getDouble(new int[]{0, 0});
//							LOG.info("Gradient Checks for Concatenated State X. Empirical %s, Estimate %s", 
//									empiricalGrad.second(), estimateGradX);
//						}
						
						//backprop through the action embedding
						Iterator<Pair<Double, Integer>> it = partitionFunctionAppr.iterator();
						
						List<Pair<ParsingOpEmbeddingResult, INDArray>> backpropParsingOp = new 
												ArrayList<Pair<ParsingOpEmbeddingResult, INDArray>>();
						
						while(it.hasNext()) {
							Pair<Double, Integer> item = it.next();
							backpropParsingOp.add(Pair.of(parsingOpEmbeddingResult.get(item.second()), 
														  parsingOpError.get(item.second())));	
						}
						
						///////////////////////
						backpropParsingOp.add(Pair.of(parsingOpEmbeddingResult.get(gTruthIx),
													  parsingOpError.get(gTruthIx)));
						if(backpropParsingOp.size() >  k + 1) {
							throw new RuntimeException("Size is more than "+(k + 1) +" found " + backpropParsingOp.size());
						}
						///////////////////////
						
						StreamSupport.stream(Spliterators
								.spliterator(backpropParsingOp, Spliterator.IMMUTABLE), true).unordered()
								.forEach(p->embedParsingOp.backProp(p.second(), p.first()));
						
						//update the errors for recurrent network
						if(!Double.isFinite(decisionGradX.sumNumber().doubleValue())) {
							throw new ArithmeticException("Found NaN");
						}
						
						INDArray decisionErrorActionHistory = decisionGradX.get(NDArrayIndex.interval(0, dimAction));
						INDArray decisionErrorState = decisionGradX.get(NDArrayIndex.interval(dimAction, dimState + dimAction));
						INDArray decisionErrorWordBuffer = decisionGradX.get(NDArrayIndex
											 				.interval(dimState + dimAction, dimState + dimAction + dimBuffer));
						
						synchronized(errorActionHistory) {
							errorActionHistory[decision.getActionHistoryIx()]
													.addi(decisionErrorActionHistory);
						}
						
						synchronized(errorState) {							
							errorState[decision.getParserStateIx()].addi(decisionErrorState);
						}
						
						synchronized(errorWordBuffer) {
							errorWordBuffer[decision.getSentenceIx()].addi(decisionErrorWordBuffer);
						}
						
					});
				
				LOG.info("Iteration: %s, Sum of Likelihood %s, NLL %s", iter, stats[1], -stats[0]);
				
				sumLogLikelihood = sumLogLikelihood + stats[0];
				sumLikelihood = sumLikelihood + stats[1];
				
				/////////////// update top layer ////////////
				final long start3 = System.currentTimeMillis();
				long diff = start3 - start2;
				time2 = time2 + diff;
				LOG.info("Time part 2 " + diff);
				
				
				
				////// backpropagate through the 3 RNNs ///////////////////////////
				final long start4 = System.currentTimeMillis();
				LOG.info("Time part 3 " +(start4 - start3));
				
				List<Pair<AbstractRecurrentNetworkHelper, INDArray[]>> backprop = 
									new LinkedList<Pair<AbstractRecurrentNetworkHelper, INDArray[]>>();
				backprop.add(Pair.of(embedActionHistory, errorActionHistory));
				backprop.add(Pair.of(embedParserState, errorState));
				backprop.add(Pair.of(embedWordBuffer, errorWordBuffer));
				
				StreamSupport.stream(Spliterators
							.spliterator(backprop, Spliterator.IMMUTABLE), true).unordered()
							.forEach(p-> p.first().backprop(p.second()));
				
				final long start5 = System.currentTimeMillis();
				long diff4 = (start5 - start4);
				LOG.info("Time part 4 " + diff4);
				time4 = time4 + diff4;
				
				//update the category vectors and flush the gradients (updates Recursive network)
				categEmbedding.updateParameters();
				categEmbedding.flushGradients();
				
				final long start6 = System.currentTimeMillis();
				LOG.info("Time part 5 " + (start6 - start5));
				
				//update the actions and flush the gradients
				embedParsingOp.updateParameters();
				embedParsingOp.flushGradients();
				
				//update tunable word embeddings and POS tag
				embedWordBuffer.updateParameters();
				embedWordBuffer.flushGradients();
				
				//Clear the stored results
				embedActionHistory.clearParsingOpEmbeddingResult();
				embedParserState.clearCategoryResults();
				
				final long start7 = System.currentTimeMillis();
				LOG.info("Time part 6 " + (start7 - start6));
				
				totalTime = totalTime + (start7 - start1);
			}
			
			numIterations = numIterations + train.size();
			
			if(iter == this.epoch) {
				this.setDisplay = true;
			}
			
			// End of iteration, print the likelihood for train and validation
			LOG.info("-------- train epoch %s  ------------", iter);
			this.calcCompositeBatchLikelihoodTopLayer(train);
			LOG.info("-------- train, end of epoch %s ------------", iter);
			
			LOG.info("-------- validation epoch %s  ------------", iter);
			double currentLogLikelihood = this.calcCompositeBatchLikelihoodTopLayer(validation);
			LOG.info("-------- validation, end of epoch %s ------------", iter);
			
			// Log the current model //////////
			/// Save it for the first epoch and every fifth epoch except the last (which is saved separately)			
			if((iter == 1 || iter%5 == 0) && iter != this.epoch) {
				try {
					this.logModel("epoch-" + iter);
				} catch (FileNotFoundException | UnsupportedEncodingException e) {
					throw new RuntimeException("Could not save model. Exception " + e);
				}
			}
			
			// Termination Condition /////////
			// Terminate if validation likelihood has not decreased in this epoch
			// and minimum number of epochs have been covered. A max epoch constraint is ensured by the for loop.
			if(prevValidationLogLikelihood > currentLogLikelihood && iter > minEpoch) { 
				LOG.info("Convergence reached. Maximum Log-Likelihood %s", prevValidationLogLikelihood);
				//this.findGradientOnValidation(dataset);
				
				this.setDisplay = true;
				// End of iteration, print the likelihood for train and validation
				LOG.info("-------- train iteration %s  ------------", iter);
				this.calcCompositeBatchLikelihoodTopLayer(train);
				LOG.info("-------- train, end of iteration %s ------------", iter);
				
				LOG.info("-------- validation iteration %s  ------------", iter);
				this.calcCompositeBatchLikelihoodTopLayer(validation);
				LOG.info("-------- validation, end of iteration %s ------------", iter);
				this.setDisplay = false;
				
				break;
			}
			
			// Relcone the recurrent networks as updates have been made
			// This needs to be done since we are doing data creation after every epoch.
			embedActionHistory.reclone();
			embedParserState.reclone();
		}
		
		LOG.debug("Action History norm %s, term %s, average %s ", embedActionHistory.norm, 
				embedActionHistory.term, embedActionHistory.norm/(double)embedActionHistory.term);
		LOG.debug("Parser State norm %s, term %s, average %s ", embedParserState.norm, 
				embedParserState.term, embedParserState.norm/(double)embedParserState.term);
		
		double totalSteps = (double)(numIterations);
		LOG.info("Total Time taken %s. Total instance %s, Average %s", totalTime, totalSteps, totalTime/totalSteps);
		LOG.info("Task 2: Time taken %s. Total instance %s, Average %s", time2, totalSteps, time2/totalSteps);
		LOG.info("Task 4: Time taken %s. Total instance %s, Average %s", time4, totalSteps, time4/totalSteps);
		
		if(this.saveModelAfterLearning) {
			try {
				this.logModel("end");
			} catch (FileNotFoundException | UnsupportedEncodingException e) {
				LOG.error("Failed to Log the model. Error: "+e);
			}
		}
		
		//Relcone the recurrent networks as updates have been made
		embedActionHistory.reclone();
		embedParserState.reclone();
		topLayer.reclone();
	}
	
	
	
	/** Finds gradient on a validation set */
	private void findGradientOnValidation(List<CompositeDataPoint<MR>> dataset) {
		
		int dataSize = dataset.size();
		int trainSize = (int)(0.9*dataSize);
		List<CompositeDataPoint<MR>> validation = dataset.subList(trainSize, dataSize);
		
		LOG.info("Fitting Dataset of size %s, train %s, validation %s", 
										dataset.size(), trainSize, dataSize - trainSize);
		
		EmbedActionHistory<MR> embedActionHistory = this.parser.getEmbedActionHistory();			
		EmbedWordBuffer embedWordBuffer = this.parser.getEmbedWordBuffer();
		EmbedParserState<MR> embedParserState = this.parser.getEmbedParserState();
		CategoryEmbedding<MR> categEmbedding = this.parser.getCategoryEmbedding();
		ParsingOpEmbedding<MR> embedParsingOp = this.parser.getEmbedParsingOp();
		INDArray A = this.parser.getAffineA();
		INDArray b = this.parser.getAffineb();
		Double[] actionBias = this.parser.getActionBias();
		
		//Induce dynamic lexical entry origin in parsing op
		//embedParsingOp.induceDynamicLexicalEntryOrigin(dataset);
		
		final int dimAction = embedActionHistory.getDimension();
		final int dimState = embedParserState.getDimension();
		final int dimBuffer = embedWordBuffer.getDimension();
	
		// Maintain average of all vectors and their gradients
		INDArray parsingOpCheck = Nd4j.zeros(embedParsingOp.getDimension());
		AtomicInteger opN = new AtomicInteger();
		INDArray parsingOpGradCheck  = Nd4j.zeros(embedParsingOp.getDimension());
		INDArray embedActionCheck = Nd4j.zeros(dimAction); 
		int n = 0;
		INDArray embedActionGradCheck = Nd4j.zeros(dimAction);  
		INDArray embedParsingStateCheck = Nd4j.zeros(dimState); 
		INDArray embedParsingStateGradCheck = Nd4j.zeros(dimState);
		INDArray embedBufferCheck = Nd4j.zeros(dimBuffer); 
		INDArray embedBufferGradCheck = Nd4j.zeros(dimBuffer);
		INDArray gradACheck = Nd4j.zeros(A.shape()); 
		INDArray gradbCheck = Nd4j.zeros(b.shape());
		
		Double[] gradActionBiasCheck = new Double[actionBias.length];
		Arrays.fill(gradActionBiasCheck, 0.0);
		
		final int k = this.partitionFunctionApproximationK - 1;
		final Comparator<Pair<Double, ParsingOpEmbeddingResult>> cmp  = 
				new Comparator<Pair<Double, ParsingOpEmbeddingResult>>() {
			public int compare(Pair<Double, ParsingOpEmbeddingResult> left, 
							   Pair<Double, ParsingOpEmbeddingResult> right) {
        		return Double.compare(left.first(), right.first()); 
    		}   
		};
		
		for(CompositeDataPoint<MR> pt: validation) {
			
			long start1 = System.currentTimeMillis();
			DerivationState<MR> dstate = pt.getState();
			List<ParsingOp<MR>> parsingOps = pt.getActionHistory();
			List<Pair<String, String>> wordBuffer = pt.getBuffer();
			List<CompositeDataPointDecision<MR>> decisions = pt.getDecisions();
			
			LOG.debug("Sentence %s", pt.getSentence());
			
			// Compute embeddings of history, state and buffer
			List<Pair<AbstractRecurrentNetworkHelper, Object>> calcEmbedding = 
								new LinkedList<Pair<AbstractRecurrentNetworkHelper, Object>>();
			calcEmbedding.add(Pair.of(embedActionHistory, (Object)parsingOps));
			calcEmbedding.add(Pair.of(embedParserState, (Object)dstate));
			calcEmbedding.add(Pair.of(embedWordBuffer, (Object)wordBuffer));
			
			List<Object> embeddings = StreamSupport.stream(Spliterators
						.spliterator(calcEmbedding, Spliterator.IMMUTABLE), true)
						.map(p->p.first().getAllTopLayerEmbedding(p.second()))
						.collect(Collectors.toList());
			
			INDArray[] topLayerA1 = (INDArray[])embeddings.get(0);
			INDArray[] topLayerA2 = (INDArray[])embeddings.get(1);
			INDArray[] topLayerA3 = (INDArray[])embeddings.get(2);
			
			double[] stats = new double[2]; 
			stats[0] = 0.0; stats[1] = 0.0; //0 is sum of log-likelihood, 1 is sum of likelihood
			
			long start2 = System.currentTimeMillis();
			LOG.info("Time part 1 " +(start2 - start1));
			
			n = n + decisions.size();
			
			StreamSupport.stream(Spliterators
				.spliterator(decisions, Spliterator.IMMUTABLE), false)
				.forEach(decision -> {
					
					INDArray a1 = topLayerA1[decision.getActionHistoryIx()];
					INDArray a2 = topLayerA2[decision.getParserStateIx()];
					INDArray a3 = topLayerA3[decision.getSentenceIx()];
					
					synchronized(embedActionCheck) {
						embedActionCheck.addi(a1);
					}
					synchronized(embedParsingStateCheck) {
						embedParsingStateCheck.addi(a2);
					}
					
					synchronized(embedBufferCheck) {
						embedBufferCheck.addi(a3);
					}
					
					INDArray x = Nd4j.concat(1, a1, a2, a3).transpose();

					List<ParsingOp<MR>> options = decision.getPossibleActions();
					int gTruthIx = decision.getGTruthIx();
					
					//computes g(A[a1; a2; a3]+b)
					INDArray currentPreOutput = A.mmul(x).addi(b).transposei(); 
					INDArray current = Nd4j.getExecutioner()
										   .execAndReturn(new /*LeakyReLU*/Tanh(currentPreOutput.dup()));
					
					INDArray embedGTruth = null;
					
					//some data-structures for computing gradients wrt A,b,x
					INDArray expYj = Nd4j.zeros(embedParsingOp.getDimension()); 
					
					//probability of taking each action
					Double[] scores = new Double[options.size()];
					double Z = 0; //partition function 
					int i = 0;
					
					List<ParsingOpEmbeddingResult> parsingOpEmbeddingResult = 
												   new LinkedList<ParsingOpEmbeddingResult>();
					
					List<Double> exponents = new LinkedList<Double>();
					List<INDArray> embedParseSteps = new LinkedList<INDArray>();
					Double maxExponent = Double.NEGATIVE_INFINITY;
									
					parsingOpEmbeddingResult = StreamSupport.stream(Spliterators
												.spliterator(options, Spliterator.IMMUTABLE), true)
												.map(op->embedParsingOp.getEmbedding(op))
												.collect(Collectors.toList());
					
					LOG.debug("X %s, current %s", x, current);
					
					///////////////////////// Debug ////////////////////////
					DirectAccessBoundedPriorityQueue<Pair<Double, ParsingOpEmbeddingResult>> partitionFunctionAppr 
							= new DirectAccessBoundedPriorityQueue<Pair<Double, ParsingOpEmbeddingResult>>(k, cmp);
					ParsingOpEmbeddingResult gTruthResult = null;
					///////////////////////////////////////////////////////
					
					for(ParsingOpEmbeddingResult parseOpResult: parsingOpEmbeddingResult) {
						
						INDArray embedParseStep = parseOpResult.getEmbedding().transpose();
						
						if(i == gTruthIx) {
							embedGTruth = embedParseStep;
							////////////////
							gTruthResult = parseOpResult;
							////////////////
						}
						
						//take dot product and action bias
						double exponent = (current.mmul(embedParseStep)).getDouble(new int[]{0, 0});	
						exponent = exponent + actionBias[parseOpResult.ruleIndex()];
						
						exponents.add(exponent);
						embedParseSteps.add(embedParseStep);
						
						if(exponent > maxExponent) {
							maxExponent = exponent;
						}
						
						i++;
					}
					
					//max exponent trick
					Iterator<Double> exponentIt = exponents.iterator();
					int jl = 0;
					for(INDArray embedParseStep: embedParseSteps) {
						
						double exponent = exponentIt.next();
						double score = Math.exp(exponent - maxExponent);
						
						scores[jl++] = score;
						expYj.addi(embedParseStep.transpose().muli(score));
						Z = Z + score;
					}
					
					assert i == jl;
					
					Iterator<ParsingOpEmbeddingResult> parsingOpEmbeddingIt = parsingOpEmbeddingResult.iterator();
					
					for(int j = 0; j < scores.length; j++) {
						scores[j] = scores[j]/Z;
						ParsingOpEmbeddingResult parsingOpEmbedding = parsingOpEmbeddingIt.next();
						///////////////////////////////////
						if(j != gTruthIx)
							partitionFunctionAppr.offer(Pair.of(scores[j], parsingOpEmbedding));
						////////////////////////////////////
					}
					
					if(i != jl || parsingOpEmbeddingIt.hasNext()) {
						throw new RuntimeException("Double check these lines. i" + i + ", jl = "+jl + 
								" parsingOp size " + parsingOpEmbeddingResult.size() + " hasNext " + parsingOpEmbeddingIt.hasNext());
					}
					
					if(LOG.getLogLevel() == LogLevel.DEBUG) {
						String scoreByte = Joiner.on(", ").join(scores);
						LOG.debug("ground truth %s, scores are %s and Z is %s ", gTruthIx, scoreByte, Z);
					}
					
					double logLikelihood = Math.log(scores[gTruthIx]);
					synchronized(stats) { 
						stats[0] = stats[0] + logLikelihood;
						stats[1] = stats[1] + scores[gTruthIx];
					}
					
					expYj.divi(Z);
					
					/* perform backpropagation through the entire computation graph
					 * 
					 * Loss is given by: negative log-likelihood + L2 regularization term + momentum term
					 *               
					 * gradient of loss with respect to A_pq will be: (y is the probability) 
					 * 				 { -pembed(y_i)_p + E_y_j[pembed(y_j)_p] } x_q
					 * with respect to b will be: -pembed(y_i) + E_y_j[pembed(y_j)]
					 * with respect to x_q: -\sum_p pembed(y_i)_p A_pq + E_y_j[\sum_p pembed(y_j)_p A_pq] */
					
					INDArray currentDerivative = Nd4j.getExecutioner()
													 .execAndReturn(new /*LeakyReLUDerivative*/TanhDerivative(currentPreOutput.dup()));
					
					//gradients for this decision (be careful with inplace operations)
					INDArray decisionGradb = embedGTruth.mul(-1).addi(expYj.transpose()).muli(currentDerivative.transpose());
					INDArray decisionGradA = decisionGradb.mmul(x.transpose());
					INDArray decisionGradX = decisionGradb.transpose().mmul(A);
					
					synchronized(gradACheck) {
						gradACheck.addi(decisionGradA);
					}
					
					synchronized(gradbCheck) {
						gradbCheck.addi(decisionGradb);
					}
					
					//backprop through the action embedding
					
					Iterator<Pair<Double, ParsingOpEmbeddingResult>> it = partitionFunctionAppr.iterator();
					
					List<Pair<ParsingOpEmbeddingResult, INDArray>> backpropParsingOp = new 
											LinkedList<Pair<ParsingOpEmbeddingResult, INDArray>>();
					
					while(it.hasNext()) {
						Pair<Double, ParsingOpEmbeddingResult> item = it.next();
						ParsingOpEmbeddingResult parseOpResult = item.second(); 
						
						final INDArray error = current.mul(item.first());
						backpropParsingOp.add(Pair.of(parseOpResult, error));
						
						///// Update gradActionBias /////
						synchronized(gradActionBiasCheck) {
							gradActionBiasCheck[parseOpResult.ruleIndex()] += item.first(); 
						}
						/////////////////////////////////
					}
					
					///////////////////////
					backpropParsingOp.add(Pair.of(gTruthResult, current.mul(-1 + scores[gTruthIx])));
					if(backpropParsingOp.size() >  k + 1) {
						throw new RuntimeException("Size is more than "+(k + 1) +" found " + backpropParsingOp.size());
					}
					///////////////////////
					
					//// Update gradActionBias ////
					synchronized(gradActionBiasCheck) {
						gradActionBiasCheck[gTruthResult.ruleIndex()] += -1 + scores[gTruthIx];
					}
					///////////////////////////////
					
					
					opN.addAndGet(backpropParsingOp.size());
					
					StreamSupport.stream(Spliterators
							.spliterator(backpropParsingOp, Spliterator.IMMUTABLE), true).unordered()
							.forEach( p -> { 
										synchronized(parsingOpCheck) {
												parsingOpCheck.addi(p.first().getEmbedding());
											}
										synchronized(parsingOpGradCheck) {
											parsingOpGradCheck.addi(p.second());
										}
									});
					
					INDArray decisionErrorActionHistory = decisionGradX.get(NDArrayIndex.interval(0, dimAction));
					INDArray decisionErrorState = decisionGradX.get(NDArrayIndex.interval(dimAction, dimState + dimAction));
					INDArray decisionErrorWordBuffer = decisionGradX.get(NDArrayIndex
										 				.interval(dimState + dimAction, dimState + dimAction + dimBuffer));
					
					
					synchronized(embedActionGradCheck) {
						embedActionGradCheck.addi(decisionErrorActionHistory);
					}
					synchronized(embedParsingStateGradCheck) {
						embedParsingStateGradCheck.addi(decisionErrorState);
					}
					synchronized(embedBufferGradCheck) {
						embedBufferGradCheck.addi(decisionErrorWordBuffer);
					}
				});
			
			categEmbedding.flushGradients();
			embedParsingOp.flushGradients();
				
			//Clear the stored results
			embedActionHistory.clearParsingOpEmbeddingResult();
			embedParserState.clearCategoryResults();
		}
		
		parsingOpCheck.divi((float)opN.get());
		parsingOpGradCheck.divi((float)opN.get());
		embedActionCheck.divi((float)n);
		embedActionGradCheck.divi((float)n);  
		embedParsingStateCheck.divi((float)n); 
		embedParsingStateGradCheck.divi((float)n);
		embedBufferCheck.divi((float)n);
		embedBufferGradCheck.divi((float)n);
		gradACheck.divi((float)n);
		gradbCheck.divi((float)n);
		
		double actionBiasL2 = 0, actionBiasMax = 0, gradActionBiasL2 = 0, gradActionBiasMax = 0;
		for(int i = 0; i < gradActionBiasCheck.length; i++) {
			gradActionBiasCheck[i]  = gradActionBiasCheck[i] / (float)opN.get(); 
			
			actionBiasL2 = actionBiasL2 + actionBias[i] * actionBias[i];
			actionBiasMax = actionBiasMax + Math.max(actionBiasMax, Math.abs(actionBias[i]));
			
			gradActionBiasMax = Math.max(gradActionBiasMax, Math.abs(gradActionBiasCheck[i]));
			gradActionBiasL2 = gradActionBiasL2 + gradActionBiasCheck[i] * gradActionBiasCheck[i];
		}
		
		actionBiasL2 = Math.sqrt(actionBiasL2);
		gradActionBiasL2 = Math.sqrt(gradActionBiasL2);
		
		final int categDim = categEmbedding.getDimension();
		INDArray actionCheck = parsingOpCheck.get(NDArrayIndex.interval(0, 3));
		INDArray actionGradCheck = parsingOpGradCheck.get(NDArrayIndex.interval(0, 3));
		INDArray categCheck = parsingOpCheck.get(NDArrayIndex.interval(3, 3 + categDim));
		INDArray categGradCheck = parsingOpGradCheck.get(NDArrayIndex.interval(3, 3 + categDim));
		INDArray lexicalCheck = parsingOpCheck.get(NDArrayIndex.interval(3 + categDim, 3 + categDim + 10));
		INDArray lexicalGradCheck = parsingOpCheck.get(NDArrayIndex.interval(3 + categDim, 3 + categDim + 10));
		
		LOG.info("Details are:\n parsingOp vec %s, \n Grad %s", parsingOpCheck, parsingOpGradCheck);
		LOG.info("action vec %s, \n Grad %s", actionCheck, actionGradCheck);
		LOG.info("categ vec %s, \n Grad %s", categCheck, categGradCheck);
		LOG.info("lexical vec %s, \n Grad %s", lexicalCheck, lexicalGradCheck);
		
		LOG.info("embed Action history vec %s, \n Grad %s", embedActionCheck, embedActionGradCheck);
		LOG.info("embed parsing state vec %s, Grad %s", embedParsingStateCheck, embedParsingStateGradCheck);
		LOG.info("embed word buffer vec %s, Grad %s", embedBufferCheck, embedBufferGradCheck);
		LOG.info("Current A matrix %s, Grad %s", A, gradACheck);
		LOG.info("Current b vec %s, Grad %s", b, gradbCheck);
		
		LOG.info("Parsing Op L2 %s, Max %s, grad L2 %s Max %s", parsingOpCheck.norm2Number().doubleValue(), 
					parsingOpCheck.normmaxNumber().doubleValue(), parsingOpGradCheck.norm2Number().doubleValue(), 
					parsingOpGradCheck.normmaxNumber().doubleValue());
	
		LOG.info("Action L2 %s, Max %s, grad L2 %s Max %s", actionCheck.norm2Number().doubleValue(), 
				actionCheck.normmaxNumber().doubleValue(), actionGradCheck.norm2Number().doubleValue(), 
				actionGradCheck.normmaxNumber().doubleValue());
		LOG.info("Category L2 %s, Max %s, grad L2 %s Max %s", categCheck.norm2Number().doubleValue(), 
				categCheck.normmaxNumber().doubleValue(), categGradCheck.norm2Number().doubleValue(), 
				categGradCheck.normmaxNumber().doubleValue());
		LOG.info("Lexical L2 %s, Max %s, grad L2 %s Max %s", lexicalCheck.norm2Number().doubleValue(), 
				lexicalCheck.normmaxNumber().doubleValue(), lexicalGradCheck.norm2Number().doubleValue(), 
				lexicalGradCheck.normmaxNumber().doubleValue());
	
		LOG.info("Action History L2 %s, Max %s, grad L2 %s Max %s", embedActionCheck.norm2Number().doubleValue(), 
				embedActionCheck.normmaxNumber().doubleValue(), embedActionGradCheck.norm2Number().doubleValue(), 
				embedActionGradCheck.normmaxNumber().doubleValue());
		LOG.info("Parsing State L2 %s, Max %s, grad L2 %s Max %s", embedParsingStateCheck.norm2Number().doubleValue(), 
				embedParsingStateCheck.normmaxNumber().doubleValue(), embedParsingStateGradCheck.norm2Number().doubleValue(), 
				embedParsingStateGradCheck.normmaxNumber().doubleValue());
		LOG.info("Buffer L2 %s, Max %s, grad L2 %s Max %s", 
								embedBufferCheck.norm2Number().doubleValue(), embedBufferCheck.normmaxNumber().doubleValue(),
				embedBufferGradCheck.norm2Number().doubleValue(), embedBufferGradCheck.normmaxNumber().doubleValue());
		LOG.info("A L2 %s, Max %s, grad L2 %s Max %s", A.norm2Number().doubleValue(), A.normmaxNumber().doubleValue(), 
				gradACheck.norm2Number().doubleValue(), gradACheck.normmaxNumber().doubleValue());
		LOG.info("b L2 %s, Max %s, grad L2 %s Max %s", b.norm2Number().doubleValue(), b.normmaxNumber().doubleValue(), 
				gradbCheck.norm2Number().doubleValue(), gradbCheck.normmaxNumber().doubleValue());
		LOG.info("Action Basis L2 %s, Max %s, grad L2 %s Max %s", actionBiasL2, actionBiasMax, gradActionBiasL2, gradActionBiasMax);
	}
	
	/** Logs the model in a json file, including values of A,b; weights of the three 
	 * RNN and all category and action embeddings */
	private void logModel(String modelName) throws FileNotFoundException, UnsupportedEncodingException {
		
		String folderName = "./Log_" + modelName + "_" + System.currentTimeMillis();
		LOG.info("Logging the results in the folder %s", folderName);
		
		//create the folder
		File dir = new File(folderName);
		dir.mkdir();
		
		EmbedActionHistory<MR> embedActionHistory = this.parser.getEmbedActionHistory();			
		EmbedWordBuffer embedWordBuffer = this.parser.getEmbedWordBuffer();
		EmbedParserState<MR> embedParserState = this.parser.getEmbedParserState();
		CategoryEmbedding<MR> categEmbedding = this.parser.getCategoryEmbedding();
		ParsingOpEmbedding<MR> embedParsingOp = this.parser.getEmbedParsingOp();
		INDArray A = this.parser.getAffineA();
		INDArray b = this.parser.getAffineb();
		Double[] actionBasis = this.parser.getActionBias();
		TopLayerMLP topLayer = this.parser.getTopLayer();
		
		//Log top layer parameter A,b
		PrintWriter writer = 
					new PrintWriter(folderName+"/top_layer.json", "UTF-8");
		writer.write("{\"A\": [" + Helper.printFullMatrix(A) + "], \n");
		writer.write("\"b\":  \"" + Helper.printFullVector(b) + "\", \n");
		writer.write("\"action_basis\": \"" + Joiner.on(" ").join(actionBasis) + "\", \n");
		writer.write("}");
		writer.flush();
		writer.close();
		
		//Log the three RNN
		embedActionHistory.logNetwork(folderName);
		embedWordBuffer.logNetwork(folderName);
		embedParserState.logNetwork(folderName);
		
		//Log the top layer
		topLayer.logNetwork(folderName);
		
		//Log categories
		embedParsingOp.logEmbedding(folderName);
		
		//Log the actions
		categEmbedding.logCategoryEmbeddingAndRecursiveNetworkParam(folderName);
	}
	
	/** Bootstraps the model using the parameters in the folder 
	 * This includes bootstrapping the top layer, the three RNN and 
	 * category, action embeddings along with Recursive networks in the 
	 * category*/
	public void bootstrapModel(String folderName) {
		
		LOG.info("Bootstrapping from %s", folderName);
		EmbedActionHistory<MR> embedActionHistory = this.parser.getEmbedActionHistory();			
		EmbedWordBuffer embedWordBuffer = this.parser.getEmbedWordBuffer();
		EmbedParserState<MR> embedParserState = this.parser.getEmbedParserState();
		CategoryEmbedding<MR> categEmbedding = this.parser.getCategoryEmbedding();
		ParsingOpEmbedding<MR> embedParsingOp = this.parser.getEmbedParsingOp();
		TopLayerMLP topLayer = this.parser.getTopLayer();
		
		//Bootstrap top layer parameter A,b
		Path topLayerJsonPath = Paths.get(folderName+"/top_layer.json");
		String topLayerJsonString;
		
		try {
			topLayerJsonString = Joiner.on("\r\n").join(Files.readAllLines(topLayerJsonPath));
		} catch (IOException e) {
			throw new RuntimeException("Could not read from top_layer.json. Error: "+e);
		}
		
		JSONObject topLayerJsonObj = new JSONObject(topLayerJsonString);
		String parambString = topLayerJsonObj.getString("b");
		String paramAString = topLayerJsonObj.getJSONArray("A").join("#");
		
		INDArray A = Helper.toMatrix(paramAString);
		INDArray b = Helper.toVector(parambString).transposei();
		this.parser.setTopLayerParam(A, b);
		
//		String[] paramActionBiasString = topLayerJsonObj.getString("action_basis").split(" ");
//		Double[] actionBasis = this.parser.getActionBias();
//		for(int i = 0; i < actionBasis.length; i++) {
//			actionBasis[i] = Double.parseDouble(paramActionBiasString[i]);
//		}
		
		//Bootstrap the RNNs
		embedActionHistory.bootstrapNetworkParam(folderName);
		embedWordBuffer.bootstrapNetworkParam(folderName);
		embedParserState.bootstrapNetworkParam(folderName);
		
		//Reclone the RNNs - WordBuffer does not have clones
		embedActionHistory.reclone();
		embedParserState.reclone();
		topLayer.reclone();
		
		//Bootstrap Categories
		categEmbedding.bootstrapCategoryEmbeddingAndRecursiveNetworkParam(folderName);
		embedParsingOp.bootstrapEmbedding(folderName);
	}

	public static class Builder<SAMPLE extends IDataItem<?>, DI extends ILabeledDataItem<SAMPLE, ?>, MR> {
	
		private final IDataCollection<DI> trainingData;
		private final NeuralNetworkShiftReduceParser<Sentence, MR> parser;
		private final IValidator<DI, MR> validator;
		
		private Integer epoch = 30;
		private Double learningRate = 0.02;
		private Double learningRateDecay = 0.001;
		private Double l2 = 0.000001;
		private Integer beamSize = 10; 
		private boolean preTrain = false;
		private boolean saveModelAfterLearning = false;
		
		/** How many top samples to consider while approximating the partition function.
		 * This is used during backpropagation where we only backprop through top k samples.*/
		private Integer partitionFunctionApproximationK = 30;
		
		private CompositeImmutableLexicon<MR> compositeLexicon;
		private IParsingFilterFactory<DI, MR> parsingFilterFactory;
		private ILexiconImmutable<MR> tempLexicon;
		
		private String folderName = null;

		public Builder(IDataCollection<DI> trainingData, NeuralNetworkShiftReduceParser<Sentence, MR> parser,
					   IValidator<DI, MR> validator) {
			this.trainingData = trainingData;
			this.parser = parser;			
			this.validator = validator;
		}
		
		public RNNShiftReduceLearner<SAMPLE, DI, MR> build() {
			return new RNNShiftReduceLearner<SAMPLE, DI, MR>(trainingData, parser, validator,  
					epoch, learningRate, learningRateDecay, l2, beamSize, partitionFunctionApproximationK, 
					parsingFilterFactory, compositeLexicon, tempLexicon, preTrain, folderName, saveModelAfterLearning);
		}
		
		public Builder<SAMPLE, DI, MR> setLexiconImmutable(ILexiconImmutable<MR> tempLexicon) {
			this.tempLexicon = tempLexicon;
			return this;
		}
		
		public Builder<SAMPLE, DI, MR> setCompositeLexicon(CompositeImmutableLexicon<MR> compositeLexicon) {
			this.compositeLexicon = compositeLexicon;
			return this;
		}
		
		public Builder<SAMPLE, DI, MR> setLearningRate(Double learningRate) {
			this.learningRate = learningRate;
			return this;
		}
		
		public Builder<SAMPLE, DI, MR> setL2(Double l2) {
			this.l2 = l2;
			return this;
		}
		
		public Builder<SAMPLE, DI, MR> setLearningRateDecay(Double learningRateDecay) {
			this.learningRateDecay = learningRateDecay;
			return this;
		}
		
		public Builder<SAMPLE, DI, MR> setNumTrainingIterations(Integer iterations) {
			this.epoch = iterations;
			return this;
		}
		
		public Builder<SAMPLE, DI, MR> setBeamSize(Integer beamSize) {
			this.beamSize = beamSize;
			return this;
		}
		
		public Builder<SAMPLE, DI, MR> setPartitionFunctionApproximationK(Integer partitionFunctionApproximationK) {
			this.partitionFunctionApproximationK = partitionFunctionApproximationK;
			return this;
		}
		
		public Builder<SAMPLE, DI, MR> setParsingFilterFactory(IParsingFilterFactory<DI, MR> parsingFilterFactory) {
			this.parsingFilterFactory = parsingFilterFactory;
			return this;
		}
		
		public Builder<SAMPLE, DI, MR> pretrain(boolean preTrain) {
			this.preTrain = preTrain;
			return this;
		}
		
		public Builder<SAMPLE, DI, MR> initFromFiles(String folderName) {
			this.folderName = folderName;
			return this;
		}
		
		public Builder<SAMPLE, DI, MR> saveModelAfterLearning(boolean saveModelAfterLearning) {
			this.saveModelAfterLearning = saveModelAfterLearning;
			return this;
		}
	}
	
	public static class Creator<SAMPLE extends IDataItem<?>, DI extends ILabeledDataItem<SAMPLE, ?>, MR> 
								implements IResourceObjectCreator<RNNShiftReduceLearner<SAMPLE, DI, MR>> {

		private final String type;
		
		public Creator() {
			this("parser.rnnshiftreduce.learner");
		}

		public Creator(String type) {
			this.type = type;
		}
		
		@Override
		public RNNShiftReduceLearner<SAMPLE, DI, MR> create(Parameters params, IResourceRepository repo) {
		
			final Builder<SAMPLE, DI, MR> builder = new Builder<SAMPLE, DI, MR>(
									repo.get(params.get("train")), repo.get(params.get("baseParser")), null);
			
			if(params.contains("tempLexicon")) {
				builder.setLexiconImmutable(repo.get(params.get("tempLexicon")));
			}
			
			if(params.contains("compositeLexicon")) {
				builder.setCompositeLexicon(repo.get(params.get("compositeLexicon")));
			}
			
			if(params.contains("learningRate")) {
				builder.setLearningRate(params.getAsDouble("learningRate"));
			}
			
			if(params.contains("learningRateDecay")) {
				builder.setLearningRateDecay(params.getAsDouble("learningRateDecay"));
			}
			
			if(params.contains("l2")) {
				builder.setL2(params.getAsDouble("l2"));
			}
			
			if(params.contains("epoch")) {
				builder.setNumTrainingIterations(params.getAsInteger("epoch"));
			}
			
			if(params.contains("beamSize")) {
				builder.setBeamSize(params.getAsInteger("beamSize"));
			}
			
			if(params.contains("partitionFunctionK")) {
				builder.setPartitionFunctionApproximationK(params.getAsInteger("partitionFunctionK"));
			}
			
			if(params.contains("parsingFilterFactory")) {
				builder.setParsingFilterFactory(repo.get(params.get("parsingFilterFactory")));
			}
			
			if(params.contains("pretrain")) {
				builder.pretrain(params.getAsBoolean("preTrain"));
			}
			
			if(params.contains("folderName")) {
				builder.initFromFiles(params.get("folderName"));
			}
			
			if(params.contains("saveModelAfterLearning")) {
				builder.saveModelAfterLearning(params.getAsBoolean("saveModelAfterLearning"));
			}
			
			return builder.build();
		}

		@Override
		public String type() {
			return this.type;
		}

		@Override
		public ResourceUsage usage() {
			// TODO Auto-generated method stub
			return null;
		}		
	}
}
