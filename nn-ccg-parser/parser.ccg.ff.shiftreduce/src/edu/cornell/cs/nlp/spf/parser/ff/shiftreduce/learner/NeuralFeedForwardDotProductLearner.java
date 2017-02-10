package edu.cornell.cs.nlp.spf.parser.ff.shiftreduce.learner;

import java.io.BufferedReader;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.io.OutputStream;
import java.io.PrintWriter;
import java.io.UnsupportedEncodingException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Iterator;
import java.util.List;
import java.util.Map.Entry;
import java.util.concurrent.atomic.AtomicInteger;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

import edu.cornell.cs.nlp.spf.base.hashvector.IHashVector;
import edu.cornell.cs.nlp.spf.base.hashvector.KeyArgs;
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
import edu.cornell.cs.nlp.spf.parser.ccg.rnn.shiftreduce.categoryembeddings.LogicalExpressionEmbedding;
import edu.cornell.cs.nlp.spf.parser.ccg.rnn.shiftreduce.learning.Helper;
import edu.cornell.cs.nlp.spf.parser.ccg.rnn.shiftreduce.learning.LearningRate;
import edu.cornell.cs.nlp.spf.parser.ff.shiftreduce.dataset.CreateSparseFeatureAndStateDataset;
import edu.cornell.cs.nlp.spf.parser.ff.shiftreduce.dataset.SparseFeatureAndStateDataset;
import edu.cornell.cs.nlp.spf.parser.ff.shiftreduce.neuralparser.FeatureEmbedding;
import edu.cornell.cs.nlp.spf.parser.ff.shiftreduce.neuralparser.LocalEnsembleNeuralDotProductShiftReduceParser;
import edu.cornell.cs.nlp.spf.parser.ff.shiftreduce.neuralparser.NeuralActionEmbeddingMixer;
import edu.cornell.cs.nlp.spf.parser.ff.shiftreduce.neuralparser.NeuralDotProductShiftReduceParser;
import edu.cornell.cs.nlp.spf.parser.ff.shiftreduce.neuralparser.NeuralParsingDotProductStepScorer;
import edu.cornell.cs.nlp.spf.parser.ff.shiftreduce.neuralparser.PerceptronLayer;
import edu.cornell.cs.nlp.spf.parser.ff.shiftreduce.nonlocal.features.SemanticFeaturesEmbedding;
import edu.cornell.cs.nlp.spf.parser.ff.shiftreduce.nonlocal.features.SemanticFeaturesEmbeddingResult;
import edu.cornell.cs.nlp.spf.parser.filter.IParsingFilterFactory;
import edu.cornell.cs.nlp.spf.parser.joint.model.IJointModelImmutable;
import edu.cornell.cs.nlp.utils.composites.Pair;
import edu.cornell.cs.nlp.utils.log.ILogger;
import edu.cornell.cs.nlp.utils.log.LogLevel;
import edu.cornell.cs.nlp.utils.log.LoggerFactory;
import edu.uw.cs.lil.amr.data.AMRMeta;

public class NeuralFeedForwardDotProductLearner<SAMPLE extends IDataItem<?>, DI extends ILabeledDataItem<SAMPLE, ?>, MR> {
	
	public static final ILogger	LOG = LoggerFactory.create(NeuralFeedForwardDotProductLearner.class);

	private final NeuralDotProductShiftReduceParser<Sentence, MR> parser;
	
	private final Integer epoch;
	private final Integer minEpoch;
	private final Integer partitionFunctionApproximationK;
	private final LearningRate learningRate;
	private final double l2;
	private final Integer beamSize;
	
	private final String bootstrapFolderName;
	private final boolean saveModelAfterLearning;
	
	private final boolean doGradientCheck;
	private boolean setDisplay;
	private double empiricalGradW, empiricalGradSemanticInput;

	private final ValidationStatistics stats;
	
	public NeuralFeedForwardDotProductLearner(IDataCollection<DI> trainingData, 
			NeuralDotProductShiftReduceParser<Sentence, MR> parser, IValidator<DI,MR> validator, 
			Integer epoch, Double learningRate, Double learningRateDecay, Double l2, Integer beamSize, 
			Integer partitionFunctionApproximationK, IParsingFilterFactory<DI, MR> parsingFilterFactory, 
			CompositeImmutableLexicon<MR> compositeLexicon, ILexiconImmutable<MR> tempLexicon, 
			boolean preTrain, String folderName, boolean saveModelAfterLearning, ValidationStatistics stats) {
		this.parser = parser;
		
		this.epoch = epoch;
		this.minEpoch = epoch - 1;//2;
		this.learningRate = new LearningRate(learningRate, learningRateDecay);
		this.l2 = l2;
		
		this.partitionFunctionApproximationK = partitionFunctionApproximationK;
		
		this.beamSize = beamSize;
		LOG.setCustomLevel(LogLevel.INFO);
		
		this.bootstrapFolderName = folderName;
		this.saveModelAfterLearning = saveModelAfterLearning;
		
		this.setDisplay = false;
		this.doGradientCheck = false;
		
		this.stats = stats;
		
		LOG.info("Neural Feed Forward Shift Reduce Learner: epoch %s, learningRate %s, l2 %s, beamSize %s,",
														this.epoch, this.learningRate, l2, this.beamSize);
		LOG.info("\t... minEpoch %s, partitionFunctionK %s, bootstrapFolderName %s ", this.minEpoch, 
										  this.partitionFunctionApproximationK, this.bootstrapFolderName);
	}
	
	private double calcLossOfPoint(SparseFeatureAndStateDataset<MR> pt) {
		
		final NeuralParsingDotProductStepScorer mlpScorer = this.parser.getMLPScorer();
		final FeatureEmbedding<MR> stateFeatureEmbedding = this.parser.getStateFeatureEmbedding();
		final FeatureEmbedding<MR> actionFeatureEmbedding = this.parser.getActionFeatureEmbedding();
		final SemanticFeaturesEmbedding semanticFeaturesEmbedding = this.parser.getSemanticFeatureEmbedding();
		final INDArray W = this.parser.getAffineW();
		
		final List<IHashVector> features = pt.getPossibleActionFeatures();
		final int gTruthIx = pt.getGroundTruthIndex();
		
		final IHashVector stateFeature = pt.getStateFeature();
		
		final INDArray stateInEmbedding;
		final INDArray stateStandardFeatureEmbedding = stateFeatureEmbedding.embedFeatures(stateFeature, true).first();
		if(semanticFeaturesEmbedding != null) {
			INDArray semanticsEmbedding = semanticFeaturesEmbedding
					.getSemanticEmbedding(pt.getLastSemantics(), pt.getSndLastSemantics(), pt.getThirdLastSemantics())
					.getEmbedding();
			stateInEmbedding = Nd4j.concat(1, stateStandardFeatureEmbedding, semanticsEmbedding);
		} else {
			stateInEmbedding = stateStandardFeatureEmbedding;
		}
		
		//final INDArray stateInEmbedding = stateFeatureEmbedding.embedFeatures(stateFeature, true).first();
		final INDArray stateOutEmbedding = mlpScorer.getEmbedding(stateInEmbedding);
//		INDArray eye = Nd4j.zeros(1);
//		eye.putScalar(new int[]{0,  0}, 1.0);
//		final  INDArray stateOutEmbedding = Nd4j.concat(1, stateOutEmbedding_, eye);
		
		
		final double[] exponents = new double[features.size()];
		int i = 0;
		for(IHashVector feature: features) {
			INDArray actionEmbedding = actionFeatureEmbedding.embedFeatures(feature, true).first();
			INDArray affineActionEmbedding = W.mmul(actionEmbedding.transpose());
			exponents[i++] = stateOutEmbedding.mmul(affineActionEmbedding).getDouble(new int[]{0, 0});
		}
		
		double[] logSoftMax = mlpScorer.toLogSoftMax(exponents);
		
		//Compute negative log-liklelihood
		final double loss = -1 * logSoftMax[gTruthIx];
		
		return loss;
	}
	
	private double symmetricDifferenceGradient(INDArray vec, double epsilon, SparseFeatureAndStateDataset<MR> pt) {
		
		final double orig = vec.getDouble(new int[]{0, 0});
		
		vec.putScalar(new int[]{0,  0}, orig + epsilon);
		double loss1 = this.calcLossOfPoint(pt);

		vec.putScalar(new int[]{0,  0}, orig - epsilon);
		double loss2 = this.calcLossOfPoint(pt);

		vec.putScalar(new int[]{0,  0}, orig);
		double empiricalGrad = (loss1 - loss2)/(2.0 * epsilon);
		
		return empiricalGrad;
	}
		
	private void gradientCheck(SparseFeatureAndStateDataset<MR> pt) {

		FeatureEmbedding<MR> stateFeatureEmbedding = this.parser.getStateFeatureEmbedding();
		final SemanticFeaturesEmbedding semanticFeaturesEmbedding = this.parser.getSemanticFeatureEmbedding();
		final double epsilon = 0.00001;
		
		//Recursive network test
		{
			if(semanticFeaturesEmbedding != null) {
				//Remember to disable cache in LogicalExpressionEmbedding
				LogicalExpressionEmbedding logicalExpressionEmbedding = 
									semanticFeaturesEmbedding.getSemanticEmbeddingObject();
				final INDArray v1 = logicalExpressionEmbedding.getSemanticVector();
				final double gradV1 = this.symmetricDifferenceGradient(v1, epsilon, pt);
				logicalExpressionEmbedding.empiricalSemanticGrad = gradV1;
				
				//////////
				final double orig = v1.getDouble(new int[]{0, 0});
				
				v1.putScalar(new int[]{0,  0}, orig + epsilon);
				INDArray vec1 = logicalExpressionEmbedding.getLogicalExpressionEmbedding(pt.getLastSemantics()).getEmbedding();
				
				v1.putScalar(new int[]{0,  0}, orig - epsilon);
				INDArray vec2 = logicalExpressionEmbedding.getLogicalExpressionEmbedding(pt.getLastSemantics()).getEmbedding();
				
				LOG.info("Expression %s \n + epsilon %s \n - epsilon %s", pt.getLastSemantics(), 
											Helper.printFullVector(vec1), Helper.printFullVector(vec2));
				
				v1.putScalar(new int[]{0,  0}, orig);
				//////////
				
				final INDArray v2 = logicalExpressionEmbedding.getSemanticRecursiveW();
				final double gradV2 = this.symmetricDifferenceGradient(v2, epsilon, pt);
				logicalExpressionEmbedding.empiricalSemanticRecursiveW = gradV2;
				
				final INDArray v3 = logicalExpressionEmbedding.getSemanticRecursiveb();
				final double gradV3 = this.symmetricDifferenceGradient(v3, epsilon, pt);
				logicalExpressionEmbedding.empiricalSemanticRecursiveb = gradV3;
				
				return;
			}
		}
		
		//word embedding gradient
		{	
			IHashVector configuration = pt.getStateFeature();
			Iterator<Pair<KeyArgs, Double>> it = configuration.iterator();
			while(it.hasNext()) {
				
				KeyArgs feature = it.next().first();
				
				if(feature.getArg1().equals("HDWORD1") || feature.getArg1().equals("HDWORD2")) {
					
					INDArray vec = stateFeatureEmbedding.getGradientCheckFeature(feature);
					final double orig = vec.getDouble(new int[]{0, 0});
					
					vec.putScalar(new int[]{0,  0}, orig + epsilon);
					double loss1 = this.calcLossOfPoint(pt);

					vec.putScalar(new int[]{0,  0}, orig - epsilon);
					double loss2 = this.calcLossOfPoint(pt);

					vec.putScalar(new int[]{0,  0}, orig);
					double empiricalGrad = (loss1 - loss2)/(2.0 * epsilon);
					
					stateFeatureEmbedding.setEmpiricalGrad(empiricalGrad);
					break;
				}
			}
		}
		
		//W matrix
		{
			final INDArray W = this.parser.getAffineW();
			double orig = W.getDouble(new int[]{0, 0});
			
			W.putScalar(new int[]{0,  0}, orig + epsilon);
			double loss1 = this.calcLossOfPoint(pt);

			W.putScalar(new int[]{0,  0}, orig - epsilon);
			double loss2 = this.calcLossOfPoint(pt);

			W.putScalar(new int[]{0,  0}, orig);
			double empiricalGradW = (loss1 - loss2)/(2.0 * epsilon);
			this.empiricalGradW = empiricalGradW;
		}
	}


	/** Calculates log-likelihood and other statistics on the given batch. */
	private double calcCompositeBatchLikelihood(List<SparseFeatureAndStateDataset<MR>> processedDataset) {
		
		final NeuralParsingDotProductStepScorer mlpScorer = this.parser.getMLPScorer();
		final FeatureEmbedding<MR> stateFeatureEmbedding = this.parser.getStateFeatureEmbedding();
		final FeatureEmbedding<MR> actionFeatureEmbedding = this.parser.getActionFeatureEmbedding();
		final SemanticFeaturesEmbedding semanticFeaturesEmbedding = this.parser.getSemanticFeatureEmbedding();
		final NeuralActionEmbeddingMixer actionMixingLayer = this.parser.getActionMixingLayer();
		final INDArray W = this.parser.getAffineW();
		
		double sumLogLikelihood = 0.0, sumLikelihood = 0.0;
		AtomicInteger correct = new AtomicInteger();
		
		int exampleIndex = 0;
			
		for(SparseFeatureAndStateDataset<MR> pt: processedDataset) {
			LOG.info("=========================");
			LOG.info("Example: %s", ++exampleIndex);
			
			if(this.setDisplay) {
				LOG.info("Sentence %s", pt.getSentence());
			}
			
			final List<IHashVector> features = pt.getPossibleActionFeatures();
			final int gTruthIx = pt.getGroundTruthIndex();
			
			final IHashVector stateFeature = pt.getStateFeature();
			final INDArray stateInEmbedding;
			final INDArray stateStandardFeatureEmbedding = stateFeatureEmbedding.embedFeatures(stateFeature, true).first();
			if(semanticFeaturesEmbedding != null) {
				INDArray semanticsEmbedding = semanticFeaturesEmbedding
						.getSemanticEmbedding(pt.getLastSemantics(), pt.getSndLastSemantics(), pt.getThirdLastSemantics())
						.getEmbedding();
				stateInEmbedding = Nd4j.concat(1, stateStandardFeatureEmbedding, semanticsEmbedding);
			} else {
				stateInEmbedding = stateStandardFeatureEmbedding;
			}
			
			final INDArray stateOutEmbedding = mlpScorer.getEmbedding(stateInEmbedding);
//			INDArray eye = Nd4j.zeros(1);
//			eye.putScalar(new int[]{0,  0}, 1.0);
//			final  INDArray stateOutEmbedding = Nd4j.concat(1, stateOutEmbedding_, eye);
			
			
			final double[] exponents = new double[features.size()];
			int i = 0;
			for(IHashVector feature: features) {
				INDArray actionEmbedding = actionFeatureEmbedding.embedFeatures(feature, true).first();
				//////
				actionEmbedding = actionMixingLayer.getEmbedding(actionEmbedding);
				//////
				INDArray affineActionEmbedding = W.mmul(actionEmbedding.transpose());
				exponents[i++] = stateOutEmbedding.mmul(affineActionEmbedding).getDouble(new int[]{0, 0});
			}
			
			double[] logSoftMax = mlpScorer.toLogSoftMax(exponents);
			
			//Compute log-likelihood and likelihood
			final double logLikelihood = logSoftMax[gTruthIx];
			final double likelihood = Math.exp(logLikelihood);
			
			int maxScore = 0;
			for(int j=0; j < logSoftMax.length; j++) {
				if(logSoftMax[j] > logSoftMax[maxScore]) {
					maxScore = j;
				}
			}
			
			if(logSoftMax[maxScore] == logSoftMax[gTruthIx]) {
				correct.incrementAndGet();
			}
			
			if(this.setDisplay) {
				List<ParsingOp<MR>> possibleActions = pt.getPossibleActions();
				if(logSoftMax[gTruthIx] < logSoftMax[maxScore]) {
					LOG.info("Right parsing action score %s -> %s ", 
											logSoftMax[gTruthIx], possibleActions.get(gTruthIx));
					LOG.info("Highest scoring parsing action score %s -> %s", 
											logSoftMax[maxScore], possibleActions.get(maxScore));
				}
			}
			
			LOG.info("Example Index %s, Sum of Likelihood %s, NLL %s, sentence: %s", exampleIndex, 
								likelihood, -logLikelihood, pt.getSentence());
			
			sumLogLikelihood = sumLogLikelihood + logLikelihood;
			sumLikelihood = sumLikelihood + likelihood;
		}
		
		LOG.info("Sum Log-Likelihood %s, Sum Likelihood %s, Correct %s out of %s",
				sumLogLikelihood, sumLikelihood, correct.get(), processedDataset.size());
		
		return sumLogLikelihood;
	}
	

	/** In every epoch creates dataset  by parsing under the current model and using a multi-parse tree filter
	 *  which allows parser to create dataset using the current parameters. After creating the dataset, online SGD
	 *  is performed. Learning algorithm terminates when in a given epoch, the learner cannot improve the validation
	 *  accuracy of the created validation dataset. */
	public void fitCompositeDataSet(CreateSparseFeatureAndStateDataset/*WithExploration*/<SAMPLE, DI, MR> datasetCreator, 
			IJointModelImmutable<SituatedSentence<AMRMeta>, LogicalExpression, LogicalExpression> model) {
		
		final NeuralParsingDotProductStepScorer mlpScorer = this.parser.getMLPScorer();
		final FeatureEmbedding<MR> stateFeatureEmbedding = this.parser.getStateFeatureEmbedding();
		final FeatureEmbedding<MR> actionFeatureEmbedding = this.parser.getActionFeatureEmbedding();
		final SemanticFeaturesEmbedding semanticFeaturesEmbedding = this.parser.getSemanticFeatureEmbedding();
		final NeuralActionEmbeddingMixer actionMixingLayer = this.parser.getActionMixingLayer();
		
		if(semanticFeaturesEmbedding != null) {
			datasetCreator.induceSemanticsVector(semanticFeaturesEmbedding.getSemanticEmbeddingObject());
		}
		
		final INDArray W = this.parser.getAffineW();
		
		this.parser.disablePacking();
		
		long totalTime = 0;	
		int numIterations = 0;
		
		INDArray gradSumSquareW  = Nd4j.zeros(W.shape());
		
		// Save the initial model
		this.logModel("init");
		
		List<SparseFeatureAndStateDataset<MR>> dataset = null;
		this.parser.testing = false;
		
		final long start = System.currentTimeMillis();
		
		for(int iter = 1; iter <= this.epoch; iter++) {
			LOG.info("=========================");
			LOG.info("Create Training Data. Epoch %s", iter);
			
			if(iter %2 == 1) { //first phase
				dataset = datasetCreator.createDataset(model);
			} else if(iter %2 == 0) { //second phase -- early update
				dataset = datasetCreator.createDiscontiguousEarlyUpdateDataset(model);
			}
			
			//dataset = datasetCreator.createDatasetWithExploration(model, iter);
			stateFeatureEmbedding.registerStateFeatures(dataset);
			actionFeatureEmbedding.registerActionFeatures(dataset);
//			datasetCreator.swap();
			
			final int dataSize = dataset.size();
			final int trainSize = (int)(0.99*dataSize);
			List<SparseFeatureAndStateDataset<MR>> train = dataset.subList(dataSize - trainSize, trainSize);//0, trainSize);
			List<SparseFeatureAndStateDataset<MR>> validation = dataset.subList(0, dataSize - trainSize);//trainSize, dataSize);
			
			////shuffle train /////
			Collections.shuffle(train);
			
			LOG.info("-------- train initialization epoch %s  ------------", iter);
			this.calcCompositeBatchLikelihood(train);
			LOG.info("-------- train, end of initialization ------------");
			
			LOG.info("-------- validation initialization epoch %s  ------------", iter);
			double prevValidationLogLikelihood = this.calcCompositeBatchLikelihood(validation);
			LOG.info("-------- validation, end of initialization ------------");
			
			if(this.stats != null) {
				LOG.info("-------- validation end-to-end epoch %s  ------------", iter);
				this.stats.calcValidationMetric();
				LOG.info("-------- validation end-to-end epoch %s  ------------", iter);
			}
			
			stateFeatureEmbedding.stats();
			actionFeatureEmbedding.stats();
			
			LOG.info("Fitting Dataset of size %s, train %s, validation %s", 
											dataset.size(), trainSize, dataSize - trainSize);
			
			LOG.info("Fit Dataset Iteration: %s", iter);
			int exampleIndex = 0;
			double sumLogLikelihood = 0.0, sumLikelihood = 0.0;
			
			final long start1 = System.currentTimeMillis();
			for(SparseFeatureAndStateDataset<MR> pt: train) {
				LOG.info("=========================");
				LOG.info("Example: %s", ++exampleIndex);
				
				if(this.doGradientCheck) {
					this.gradientCheck(pt);
				}
				
				final IHashVector stateFeature = pt.getStateFeature();
				
				final INDArray stateInEmbedding;
				final Pair<INDArray, List<int[]>> stateStandardFeatureEmbedding = 
											stateFeatureEmbedding.embedFeatures(stateFeature, true);
				final SemanticFeaturesEmbeddingResult semanticEmbeddingResult;
				
				if(semanticFeaturesEmbedding != null) {
	
					semanticEmbeddingResult = semanticFeaturesEmbedding
							.getSemanticEmbedding(pt.getLastSemantics(), pt.getSndLastSemantics(), pt.getThirdLastSemantics());
					INDArray semanticsEmbedding = semanticEmbeddingResult.getEmbedding();
					stateInEmbedding = Nd4j.concat(1, stateStandardFeatureEmbedding.first(), semanticsEmbedding);
				} else {
					semanticEmbeddingResult = null;
					stateInEmbedding = stateStandardFeatureEmbedding.first();
				}
				
				//final Pair<INDArray, List<int[]>> stateInEmbedding = stateFeatureEmbedding.embedFeatures(stateFeature, true);
				//final INDArray stateOutEmbedding = mlpScorer.getEmbedding(stateInEmbedding.first());
				final INDArray stateOutEmbedding = mlpScorer.getEmbedding(stateInEmbedding);
				
//				INDArray eye = Nd4j.zeros(1);
//				eye.putScalar(new int[]{0,  0}, 1.0);
//				final  INDArray stateOutEmbedding = Nd4j.concat(1, stateOutEmbedding_, eye);
				
				final List<IHashVector> features = pt.getPossibleActionFeatures();

				double[] exponents = new double[features.size()];
				
				List<Pair<INDArray, List<int[]>>> results = new ArrayList<Pair<INDArray, List<int[]>>>();
				List<int[]> frequencies = new ArrayList<int[]>();
				int j = 0;
				
				////////////////////
				List<INDArray> batch = new ArrayList<INDArray>();
				List<List<int[]>> batchFrequencies = new ArrayList<List<int[]>>();
				for(IHashVector actionFeature: features) {
					Pair<INDArray, List<int[]>> result = actionFeatureEmbedding.embedFeatures(actionFeature, true);
					batch.add(result.first());
					batchFrequencies.add(result.second());
				}
				
				List<INDArray> actionEmbeddings = actionMixingLayer.getEmbedding(batch);
				for(INDArray actionEmbedding: actionEmbeddings) {
					INDArray affineActionEmbedding = W.mmul(actionEmbedding.transpose());
					exponents[j] = stateOutEmbedding.mmul(affineActionEmbedding).getDouble(new int[]{0, 0});
					results.add(Pair.of(actionEmbedding, batchFrequencies.get(j)));
					frequencies.add(batchFrequencies.get(j).get(0));
					j++;
				}
				////////////////////
				
				/* remove above code and uncomment this when removing mixing layer
				for(IHashVector actionFeature: features) {
					Pair<INDArray, List<int[]>> result = actionFeatureEmbedding.embedFeatures(actionFeature, true);
					INDArray affineActionEmbedding = W.mmul(result.first().transpose());
					exponents[j++] = stateOutEmbedding.mmul(affineActionEmbedding).getDouble(new int[]{0, 0});
					results.add(result);
					frequencies.add(result.second().get(0));
				}*/
				
				double[] logSoftMax = mlpScorer.toLogSoftMax(exponents);
				
				final int gTruthIx = pt.getGroundTruthIndex();
				
				//Compute log-liklelihood and likelihood
				final double logLikelihood = logSoftMax[gTruthIx];
				final double likelihood = Math.exp(logLikelihood);
				
				LOG.info("Iteration: %s, Sum of Likelihood %s, NLL %s", iter, likelihood, -logLikelihood);
				
				if(Double.isNaN(logLikelihood)) {
					continue;
				}
				
				sumLogLikelihood = sumLogLikelihood + logLikelihood;
				sumLikelihood = sumLikelihood + likelihood;
				
				//Compute gradients
				INDArray gamma = results.get(gTruthIx).first().mul(-1);
				INDArray stateMulW = stateOutEmbedding.mmul(W);
				
				List<INDArray> gradActions = new ArrayList<INDArray>();
				
				for(int i = 0; i < exponents.length; i++) {
					double prob = Math.exp(logSoftMax[i]);
					gamma.addi(results.get(i).first().mul(prob));
					
					final INDArray gradAction;
					if(i == gTruthIx) {
						gradAction = stateMulW.mul(-1 + prob);
					} else {
						gradAction = stateMulW.mul(prob);
					}
					gradActions.add(gradAction);
				}
				
				////////////////
				gradActions = actionMixingLayer.backprop(gradActions);
				////////////////
				actionFeatureEmbedding.backprop(gradActions, features, frequencies);
				
				INDArray gradStateEmbedding = gamma.mmul(W.transpose());
//				gradStateEmbedding = gradStateEmbedding.get(NDArrayIndex.point(0), 
//											NDArrayIndex.interval(0, gradStateEmbedding.size(1) - 1));
				
				//Do backpropagation through the neural network
				INDArray gradStateEmbeddingInput = mlpScorer.backprop(gradStateEmbedding);
				
				if(this.doGradientCheck) {
//					LOG.info("Gradient Check:: Batch input: Empirical %s Estimate %s", this.empiricalGrad, 
//						errorFeatureInput.get(0).getDouble(new int[]{0, 0}));
				}
				
				//Split error into standard state features and semantics
				final INDArray gradStandardStateError;
				final INDArray gradSemanticFeatureError;
				
				if(semanticFeaturesEmbedding != null) {
					final int dim = semanticFeaturesEmbedding.getDimension();
					final int size = gradStateEmbeddingInput.size(1);
					gradStandardStateError = gradStateEmbeddingInput.get(NDArrayIndex.interval(0, size - dim));
					gradSemanticFeatureError = gradStateEmbeddingInput.get(NDArrayIndex.interval(size - dim, size));
				} else {
					gradStandardStateError = gradStateEmbeddingInput;
					gradSemanticFeatureError = null;
				}
				
				//backprop the gradients to standard state feature embedding
				List<IHashVector> singletonFeature = new ArrayList<IHashVector>();
				singletonFeature.add(stateFeature);
				
				List<INDArray> singletonError = new ArrayList<INDArray>();
				//singletonError.add(gradStateEmbeddingInput);
				singletonError.add(gradStandardStateError);
				
				//stateFeatureEmbedding.backprop(singletonError, singletonFeature, stateInEmbedding.second());
				stateFeatureEmbedding.backprop(singletonError, singletonFeature, stateStandardFeatureEmbedding.second());
				
				// backprop gradients to semantics feature embedding
				// this involves backproping through the recursive networks
				if(semanticFeaturesEmbedding != null) {
					semanticFeaturesEmbedding.backprop(gradSemanticFeatureError, semanticEmbeddingResult);
				}
				
				// Update the W parameter
				INDArray gradW = stateOutEmbedding.transpose().mmul(gamma);
				Helper.updateVector(W, gradW, gradSumSquareW, this.l2, this.learningRate.getLearningRate());
				
				if(this.doGradientCheck) {
					LOG.info("Empirical Grad W %s estimated grad W %s", 
											this.empiricalGradW, gradW.getDouble(new int[]{0, 0}));
				}
				
				// Update the action and standard state feature embeddings
				stateFeatureEmbedding.update();
				actionFeatureEmbedding.update();
				
				// Update the semantic (or non-standard) feature embeddings
				if(semanticFeaturesEmbedding != null) {
					semanticFeaturesEmbedding.getSemanticEmbeddingObject().updateParameters();
				}
				
				//Flush the gradients
				stateFeatureEmbedding.flush();
				actionFeatureEmbedding.flush();
				
				if(semanticFeaturesEmbedding != null) {
					semanticFeaturesEmbedding.getSemanticEmbeddingObject().flushGradients();
					semanticFeaturesEmbedding.getSemanticEmbeddingObject().invalidateCache();
				}
				
				numIterations++;
			}
			
			LOG.info("Time taken in inner loop %s", System.currentTimeMillis() - start1);
			
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
			
			if(this.stats != null) {
				LOG.info("-------- validation end-to-end iteration %s  ------------", iter);
				this.stats.calcValidationMetric();
				LOG.info("-------- validation end-to-end iteration %s  ------------", iter);
				
			}
			
			// Log the current model //////////
			/// Save it for the first epoch and every epoch except the last (which is saved separately)
			if((iter == 1 || iter%1 == 0) && iter != this.epoch) {
				this.logModel("epoch-" + iter);
			}
			
			// Termination Condition /////////
			// Terminate if validation likelihood has not decreased in this epoch
			// and minimum number of epochs have been covered. A max epoch constraint is ensured by the for loop.
			if(prevValidationLogLikelihood > currentLogLikelihood && iter > this.minEpoch) { 
				LOG.info("Convergence reached. Maximum Log-Likelihood %s", prevValidationLogLikelihood);
				
				this.setDisplay = true;
				
				// End of iteration, print the likelihood for train and validation
				LOG.info("-------- train iteration %s  ------------", iter);
				this.calcCompositeBatchLikelihood(train);
				LOG.info("-------- train, end of iteration %s ------------", iter);
				
				LOG.info("-------- validation iteration %s  ------------", iter);
				this.calcCompositeBatchLikelihood(validation);
				LOG.info("-------- validation, end of iteration %s ------------", iter);
				
				if(this.stats != null) {
					LOG.info("-------- validation end-to-end end of iteration %s  ------------", iter);
					this.stats.calcValidationMetric();
					LOG.info("-------- validation end-to-end end of iteration %s  ------------", iter);
				}
				
				this.setDisplay = false;
				break;
			}
			
			/// Clear the dataset to release memory. This is important.
			dataset.clear();
			//LOG.info("train %s validation %s", train.size(), validation.size());
			System.gc();
			
			// Relcone the MLP as updates have been made
			// This needs to be done since we are doing data creation after every epoch.
			mlpScorer.reclone();
			/////////
			actionMixingLayer.reclone();
			/////////
		}
		
		LOG.info("Train profiling. Time taken %s", System.currentTimeMillis() - start);
		
		double totalSteps = (double)(numIterations);
		LOG.info("Total Time taken %s. Total instance %s, Average %s", totalTime, totalSteps, totalTime/totalSteps);
		
		if(this.saveModelAfterLearning) {
			this.logModel("end");
		}
		
		//Relcone the MLP as updates have been made
		mlpScorer.reclone();
		///////
		actionMixingLayer.reclone();
		///////
	
		this.parser.enablePacking();
		
		actionFeatureEmbedding.stopAddingFeatures();
		actionFeatureEmbedding.stats();
		actionFeatureEmbedding.clearSeenFeaturesStats();
		
		stateFeatureEmbedding.stopAddingFeatures();
		stateFeatureEmbedding.stats();
		stateFeatureEmbedding.clearSeenFeaturesStats();
		
		LOG.info("Setting temporary model new features");
		this.parser.modelNewFeatures  = datasetCreator.getModelNewFeatures();
		stateFeatureEmbedding.projectWordEmbeddings();
		actionFeatureEmbedding.projectWordEmbeddings();
		this.parser.testing = true;
	}
		
	/** Fits the perceptron layer on top of an already tuned dot-product parser */
	public void fitPerceptronToCompositeDataSet(CreateSparseFeatureAndStateDataset/*WithExploration*/<SAMPLE, DI, MR> datasetCreator, 
			IJointModelImmutable<SituatedSentence<AMRMeta>, LogicalExpression, LogicalExpression> model) {
		
		final FeatureEmbedding<MR> stateFeatureEmbedding = this.parser.getStateFeatureEmbedding();
		final FeatureEmbedding<MR> actionFeatureEmbedding = this.parser.getActionFeatureEmbedding();
		
		actionFeatureEmbedding.stopAddingFeatures();
		actionFeatureEmbedding.stats();
		actionFeatureEmbedding.clearSeenFeaturesStats();
		
		stateFeatureEmbedding.stopAddingFeatures();
		stateFeatureEmbedding.stats();
		stateFeatureEmbedding.clearSeenFeaturesStats();
		
		final int perceptronUpdateBeamSize = 8;
		LOG.info("Fitting perceptron. Beam size %s", perceptronUpdateBeamSize);
		
		this.parser.testing = false;
		
		for(int iter = 1; iter <= this.epoch; iter++) {
			LOG.info("=========================");
			LOG.info("Perceptron Update. Epoch %s", iter);
			
			datasetCreator.doPerceptronUpdate(model, perceptronUpdateBeamSize);
			
			// Log the current model //////////
			/// Save it for the first epoch and every epoch except the last (which is saved separately)
			if((iter <= 5 || iter%5 == 0) && iter != this.epoch) {
				this.logModel("epoch-" + iter);
			}
		}
		
		if(this.saveModelAfterLearning) {
			this.logModel("end");
		}
		
		this.parser.enablePacking();
		
		LOG.info("Setting temporary model new features");
		this.parser.modelNewFeatures  = datasetCreator.getModelNewFeatures();
		this.parser.testing = true;
	}
	
	/** Tunes unseen features by creating dataset, registering features and then dropping certain features
	 * so as to induce unseen feature scenario and then performs one epoch of parameter update but only
	 * updating the unseen features. */
	public void tuneUnseenFeatures(CreateSparseFeatureAndStateDataset<SAMPLE, DI, MR> datasetCreator, 
			IJointModelImmutable<SituatedSentence<AMRMeta>, LogicalExpression, LogicalExpression> model) {
		
		final NeuralParsingDotProductStepScorer mlpScorer = this.parser.getMLPScorer();
		final FeatureEmbedding<MR> stateFeatureEmbedding = this.parser.getStateFeatureEmbedding();
		final FeatureEmbedding<MR> actionFeatureEmbedding = this.parser.getActionFeatureEmbedding();
		final SemanticFeaturesEmbedding semanticFeaturesEmbedding = this.parser.getSemanticFeatureEmbedding();
		
		if(semanticFeaturesEmbedding != null) {
			datasetCreator.induceSemanticsVector(semanticFeaturesEmbedding.getSemanticEmbeddingObject());
		}
		
		final INDArray W = this.parser.getAffineW(); 
		
		this.parser.disablePacking();
		
		long totalTime = 0;	
		int numIterations = 0;
		
		// Save the initial model
		this.logModel("init");
		
		List<SparseFeatureAndStateDataset<MR>> dataset = null;
		this.parser.testing = false;
		
		for(int iter = 1; iter <= this.epoch; iter++) {
			LOG.info("=========================");
			LOG.info("Create Training Data. Epoch %s", iter);
			
			if(iter %2 == 1) { //first phase
				dataset = datasetCreator.createDataset(model);
			} else if(iter %2 == 0) { //second phase -- early update
				dataset = datasetCreator.createDiscontiguousEarlyUpdateDataset(model);
			}
			
			//dataset = datasetCreator.createDatasetWithExploration(model, iter);
//			stateFeatureEmbedding.registerStateFeatures(dataset);
//			actionFeatureEmbedding.registerActionFeatures(dataset);
//			datasetCreator.swap();
			 
			stateFeatureEmbedding.downsampleFeature();
			actionFeatureEmbedding.downsampleFeature();
			
			final int dataSize = dataset.size();
			final int trainSize = (int)(0.99*dataSize);
			List<SparseFeatureAndStateDataset<MR>> train = dataset.subList(dataSize - trainSize, trainSize);//0, trainSize);
			List<SparseFeatureAndStateDataset<MR>> validation = dataset.subList(0, dataSize - trainSize);//trainSize, dataSize);
			
			////shuffle train /////
			Collections.shuffle(train);
			
			LOG.info("-------- train initialization epoch %s  ------------", iter);
//			this.calcCompositeBatchLikelihood(train);
			LOG.info("-------- train, end of initialization ------------");
			
			LOG.info("-------- validation initialization epoch %s  ------------", iter);
			double prevValidationLogLikelihood = 0;//this.calcCompositeBatchLikelihood(validation);
			LOG.info("-------- validation, end of initialization ------------");
			
			if(this.stats != null) {
				LOG.info("-------- validation end-to-end epoch %s  ------------", iter);
				this.stats.calcValidationMetric();
				LOG.info("-------- validation end-to-end epoch %s  ------------", iter);
			}
			
			stateFeatureEmbedding.stats();
			actionFeatureEmbedding.stats();
			
			LOG.info("Fitting Dataset of size %s, train %s, validation %s", 
											dataset.size(), trainSize, dataSize - trainSize);
			
			LOG.info("Fit Dataset Iteration: %s", iter);
			int exampleIndex = 0;
			double sumLogLikelihood = 0.0, sumLikelihood = 0.0;
			
			for(SparseFeatureAndStateDataset<MR> pt: train) {
				LOG.info("=========================");
				LOG.info("Example: %s", ++exampleIndex);
				
				if(this.doGradientCheck) {
					this.gradientCheck(pt);
				}
				
				final IHashVector stateFeature = pt.getStateFeature();
				
				final INDArray stateInEmbedding;
				final Pair<INDArray, List<int[]>> stateStandardFeatureEmbedding = 
											stateFeatureEmbedding.embedFeatures(stateFeature, true);
				final SemanticFeaturesEmbeddingResult semanticEmbeddingResult;
				
				if(semanticFeaturesEmbedding != null) {
	
					semanticEmbeddingResult = semanticFeaturesEmbedding
							.getSemanticEmbedding(pt.getLastSemantics(), pt.getSndLastSemantics(), pt.getThirdLastSemantics());
					INDArray semanticsEmbedding = semanticEmbeddingResult.getEmbedding();
					stateInEmbedding = Nd4j.concat(1, stateStandardFeatureEmbedding.first(), semanticsEmbedding);
				} else {
					semanticEmbeddingResult = null;
					stateInEmbedding = stateStandardFeatureEmbedding.first();
				}
				
				//final Pair<INDArray, List<int[]>> stateInEmbedding = stateFeatureEmbedding.embedFeatures(stateFeature, true);
				//final INDArray stateOutEmbedding = mlpScorer.getEmbedding(stateInEmbedding.first());
				final INDArray stateOutEmbedding = mlpScorer.getEmbedding(stateInEmbedding);
				
//				INDArray eye = Nd4j.zeros(1);
//				eye.putScalar(new int[]{0,  0}, 1.0);
//				final  INDArray stateOutEmbedding = Nd4j.concat(1, stateOutEmbedding_, eye);
				
				final List<IHashVector> features = pt.getPossibleActionFeatures();

				double[] exponents = new double[features.size()];
				
				List<Pair<INDArray, List<int[]>>> results = new ArrayList<Pair<INDArray, List<int[]>>>();
				List<int[]> frequencies = new ArrayList<int[]>();
				int j = 0;
				for(IHashVector actionFeature: features) {
					Pair<INDArray, List<int[]>> result = actionFeatureEmbedding.embedFeatures(actionFeature, true);
					INDArray affineActionEmbedding = W.mmul(result.first().transpose());
					exponents[j++] = stateOutEmbedding.mmul(affineActionEmbedding).getDouble(new int[]{0, 0});
					results.add(result);
					frequencies.add(result.second().get(0));
				}
				
				double[] logSoftMax = mlpScorer.toLogSoftMax(exponents);
				
				final int gTruthIx = pt.getGroundTruthIndex();
				
				//Compute log-liklelihood and likelihood
				final double logLikelihood = logSoftMax[gTruthIx];
				final double likelihood = Math.exp(logLikelihood);
				
				LOG.info("Iteration: %s, Sum of Likelihood %s, NLL %s", iter, likelihood, -logLikelihood);
				
				if(Double.isNaN(logLikelihood)) {
					continue;
				}
				
				sumLogLikelihood = sumLogLikelihood + logLikelihood;
				sumLikelihood = sumLikelihood + likelihood;
				
				//Compute gradients
				INDArray gamma = results.get(gTruthIx).first().mul(-1);
				INDArray stateMulW = stateOutEmbedding.mmul(W);
				
				List<INDArray> gradActions = new ArrayList<INDArray>();
				
				for(int i = 0; i < exponents.length; i++) {
					double prob = Math.exp(logSoftMax[i]);
					gamma.addi(results.get(i).first().mul(prob));
					
					final INDArray gradAction;
					if(i == gTruthIx) {
						gradAction = stateMulW.mul(-1 + prob);
					} else {
						gradAction = stateMulW.mul(prob);
					}
					gradActions.add(gradAction);
				}
				
				actionFeatureEmbedding.backprop(gradActions, features, frequencies);
				
				INDArray gradStateEmbedding = gamma.mmul(W.transpose());
//				gradStateEmbedding = gradStateEmbedding.get(NDArrayIndex.point(0), 
//											NDArrayIndex.interval(0, gradStateEmbedding.size(1) - 1));
				
				//Do backpropagation through the neural network
				INDArray gradStateEmbeddingInput = mlpScorer.backprop(gradStateEmbedding);
				
				if(this.doGradientCheck) {
//					LOG.info("Gradient Check:: Batch input: Empirical %s Estimate %s", this.empiricalGrad, 
//						errorFeatureInput.get(0).getDouble(new int[]{0, 0}));
				}
				
				//Split error into standard state features and semantics
				final INDArray gradStandardStateError;
				final INDArray gradSemanticFeatureError;
				
				if(semanticFeaturesEmbedding != null) {
					final int dim = semanticFeaturesEmbedding.getDimension();
					final int size = gradStateEmbeddingInput.size(1);
					gradStandardStateError = gradStateEmbeddingInput.get(NDArrayIndex.interval(0, size - dim));
					gradSemanticFeatureError = gradStateEmbeddingInput.get(NDArrayIndex.interval(size - dim, size));
				} else {
					gradStandardStateError = gradStateEmbeddingInput;
					gradSemanticFeatureError = null;
				}
				
				//backprop the gradients to standard state feature embedding
				List<IHashVector> singletonFeature = new ArrayList<IHashVector>();
				singletonFeature.add(stateFeature);
				
				List<INDArray> singletonError = new ArrayList<INDArray>();
				//singletonError.add(gradStateEmbeddingInput);
				singletonError.add(gradStandardStateError);
				
				//stateFeatureEmbedding.backprop(singletonError, singletonFeature, stateInEmbedding.second());
				stateFeatureEmbedding.backprop(singletonError, singletonFeature, stateStandardFeatureEmbedding.second());
				
				// backprop gradients to semantics feature embedding
				// this involves backproping through the recursive networks
				if(semanticFeaturesEmbedding != null) {
					semanticFeaturesEmbedding.backprop(gradSemanticFeatureError, semanticEmbeddingResult);
				}
								
				// Update the action and standard state feature embeddings
				stateFeatureEmbedding.updateOnlyUnseenFeatures();
				actionFeatureEmbedding.updateOnlyUnseenFeatures();
				
				// Update the semantic (or non-standard) feature embeddings
				if(semanticFeaturesEmbedding != null) {
					semanticFeaturesEmbedding.getSemanticEmbeddingObject().updateParameters();
				}
				
				//Flush the gradients
				stateFeatureEmbedding.flush();
				actionFeatureEmbedding.flush();
				
				if(semanticFeaturesEmbedding != null) {
					semanticFeaturesEmbedding.getSemanticEmbeddingObject().flushGradients();
					semanticFeaturesEmbedding.getSemanticEmbeddingObject().invalidateCache();
				}
				
				numIterations++;
			}
			
			if(iter == this.epoch) {
				this.setDisplay = true;
			}
			
			// End of epoch, calculate the log-likelihood for train and validation
			LOG.info("-------- train iteration %s  ------------", iter);
//			this.calcCompositeBatchLikelihood(train);
			LOG.info("-------- train, end of iteration %s ------------", iter);
			
			LOG.info("-------- validation iteration %s  ------------", iter);
			double currentLogLikelihood = 0;//this.calcCompositeBatchLikelihood(validation);
			LOG.info("-------- validation, end of iteration %s ------------", iter);
			
			if(this.stats != null) {
				LOG.info("-------- validation end-to-end iteration %s  ------------", iter);
				this.stats.calcValidationMetric();
				LOG.info("-------- validation end-to-end iteration %s  ------------", iter);
				
			}
			
			// Log the current model //////////
			/// Save it for the first epoch and every epoch except the last (which is saved separately)
			if((iter == 1 || iter%1 == 0) && iter != this.epoch) {
				this.logModel("epoch-" + iter);
			}
			
			// Termination Condition /////////
			// Terminate if validation likelihood has not decreased in this epoch
			// and minimum number of epochs have been covered. A max epoch constraint is ensured by the for loop.
			if(prevValidationLogLikelihood > currentLogLikelihood && iter > this.minEpoch) { 
				LOG.info("Convergence reached. Maximum Log-Likelihood %s", prevValidationLogLikelihood);
				
				this.setDisplay = true;
				
				// End of iteration, print the likelihood for train and validation
				LOG.info("-------- train iteration %s  ------------", iter);
//				this.calcCompositeBatchLikelihood(train);
				LOG.info("-------- train, end of iteration %s ------------", iter);
				
				LOG.info("-------- validation iteration %s  ------------", iter);
//				this.calcCompositeBatchLikelihood(validation);
				LOG.info("-------- validation, end of iteration %s ------------", iter);
				
				if(this.stats != null) {
					LOG.info("-------- validation end-to-end end of iteration %s  ------------", iter);
					this.stats.calcValidationMetric();
					LOG.info("-------- validation end-to-end end of iteration %s  ------------", iter);
				}
				
				this.setDisplay = false;
				break;
			}
			
			/// Clear the dataset to release memory. This is important.
			dataset.clear();
			//LOG.info("train %s validation %s", train.size(), validation.size());
			System.gc();
			
			// Relcone the MLP as updates have been made
			// This needs to be done since we are doing data creation after every epoch.
			mlpScorer.reclone();
		}
		
		double totalSteps = (double)(numIterations);
		LOG.info("Total Time taken %s. Total instance %s, Average %s", totalTime, totalSteps, totalTime/totalSteps);
		
		if(this.saveModelAfterLearning) {
			this.logModel("end");
		}
		
		//Relcone the MLP as updates have been made
		mlpScorer.reclone();
	
		this.parser.enablePacking();
		
		actionFeatureEmbedding.stopAddingFeatures();
		actionFeatureEmbedding.stats();
		actionFeatureEmbedding.clearSeenFeaturesStats();
		
		stateFeatureEmbedding.stopAddingFeatures();
		stateFeatureEmbedding.stats();
		stateFeatureEmbedding.clearSeenFeaturesStats();
		
		LOG.info("Setting temporary model new features");
		this.parser.modelNewFeatures  = datasetCreator.getModelNewFeatures();
		stateFeatureEmbedding.projectWordEmbeddings();
		actionFeatureEmbedding.projectWordEmbeddings();
		this.parser.testing = true;
	}
	
	public void fixModelForTesting(CreateSparseFeatureAndStateDataset<SAMPLE, DI, MR> datasetCreator) {
		
		final FeatureEmbedding<MR> actionFeatureEmbedding = this.parser.getActionFeatureEmbedding();
		final FeatureEmbedding<MR> stateFeatureEmbedding = this.parser.getStateFeatureEmbedding();
		
		actionFeatureEmbedding.stopAddingFeatures();
		actionFeatureEmbedding.stats();
		actionFeatureEmbedding.clearSeenFeaturesStats();
		
		stateFeatureEmbedding.stopAddingFeatures();
		stateFeatureEmbedding.stats();
		stateFeatureEmbedding.clearSeenFeaturesStats();
		
		LOG.info("Setting temporary model new features");
		this.parser.modelNewFeatures  = datasetCreator.getModelNewFeatures();
		stateFeatureEmbedding.projectWordEmbeddings();
		actionFeatureEmbedding.projectWordEmbeddings();
		this.parser.testing = true;
	}
	
	public static void fixModelForTesting(NeuralDotProductShiftReduceParser<?, ?> parser,
				IJointModelImmutable<SituatedSentence<AMRMeta>, LogicalExpression, LogicalExpression> modelNewFeatures) {
		
		final FeatureEmbedding<?> actionFeatureEmbedding = parser.getActionFeatureEmbedding();
		final FeatureEmbedding<?> stateFeatureEmbedding = parser.getStateFeatureEmbedding();
		
		actionFeatureEmbedding.stopAddingFeatures();
		actionFeatureEmbedding.stats();
		actionFeatureEmbedding.clearSeenFeaturesStats();
		
		stateFeatureEmbedding.stopAddingFeatures();
		stateFeatureEmbedding.stats();
		stateFeatureEmbedding.clearSeenFeaturesStats();
		
		LOG.info("Setting temporary model new features");
		parser.modelNewFeatures  = modelNewFeatures;
		stateFeatureEmbedding.projectWordEmbeddings();
		actionFeatureEmbedding.projectWordEmbeddings();
		parser.testing = true;
	}
	
	public static void setFeatures(NeuralDotProductShiftReduceParser<?, ?> parser,
			IJointModelImmutable<SituatedSentence<AMRMeta>, LogicalExpression, LogicalExpression> modelNewFeatures) {
	
		LOG.info("Setting temporary model new features");
		parser.modelNewFeatures  = modelNewFeatures;
	}
	
	public static void setFeatures(LocalEnsembleNeuralDotProductShiftReduceParser<?, ?> parser,
			IJointModelImmutable<SituatedSentence<AMRMeta>, LogicalExpression, LogicalExpression> modelNewFeatures) {
	
		LOG.info("Setting temporary model new features for local ensemble parser");
		parser.modelNewFeatures  = modelNewFeatures;
	}

	public void logModel(String modelName) {
		
		final String folderName = modelName + "_" + System.currentTimeMillis();
		LOG.info("Logging the model %s", folderName);
		
		//create the folder
		File dir = new File(folderName);
		dir.mkdir();
		
		final NeuralParsingDotProductStepScorer mlpScorer = this.parser.getMLPScorer();
		final FeatureEmbedding<MR> stateFeatureEmbedding = this.parser.getStateFeatureEmbedding();
		final FeatureEmbedding<MR> actionFeatureEmbedding = this.parser.getActionFeatureEmbedding();
		final SemanticFeaturesEmbedding semanticFeatureEmbedding = this.parser.getSemanticFeatureEmbedding();
		final PerceptronLayer perceptronLayer = this.parser.getPerceptronLayer();
		final NeuralActionEmbeddingMixer actionMixingLayer = this.parser.getActionMixingLayer();
		
		mlpScorer.logNetwork(folderName);
		stateFeatureEmbedding.logEmbeddings(folderName, "state");
		actionFeatureEmbedding.logEmbeddings(folderName, "action");
		
		if(semanticFeatureEmbedding != null) {
			try {
				semanticFeatureEmbedding.getSemanticEmbeddingObject().logEmbeddingAndRecursiveNetworkParam(folderName);
			} catch (FileNotFoundException | UnsupportedEncodingException e) {
				throw new RuntimeException("Failed to serialized and/or save semantic feature embeddings. Error: " + e);
			}
		}
		
		if(perceptronLayer != null) {
			perceptronLayer.logPerceptronWeights(folderName);
		}
		
		if(actionMixingLayer != null) {
			actionMixingLayer.logNetwork(folderName);
		}
		
		//Save W
		try {
			OutputStream fos = Files.newOutputStream(
					Paths.get(folderName + "/W.bin"));
	        DataOutputStream dos = new DataOutputStream(fos);
		    
			Nd4j.write(this.parser.getAffineW(), dos);
			dos.flush();
		    dos.close();
		    
		} catch (IOException e) {
			throw new RuntimeException("Could not dump the matrix W " + e);
		}
	}
	
	public void logModelAsCSV(String modelName) {
		
		final String folderName = modelName + "_" + System.currentTimeMillis();
		LOG.info("Logging the model as csv %s", folderName);
		
		//create the folder
		File dir = new File(folderName);
		dir.mkdir();
		
		final NeuralParsingDotProductStepScorer mlpScorer = this.parser.getMLPScorer();
		final FeatureEmbedding<MR> stateFeatureEmbedding = this.parser.getStateFeatureEmbedding();
		final FeatureEmbedding<MR> actionFeatureEmbedding = this.parser.getActionFeatureEmbedding();
		final SemanticFeaturesEmbedding semanticFeatureEmbedding = this.parser.getSemanticFeatureEmbedding();
		final PerceptronLayer perceptronLayer = this.parser.getPerceptronLayer();
		final NeuralActionEmbeddingMixer actionMixingLayer = this.parser.getActionMixingLayer();
		
		mlpScorer.logNetworkAsCSV(folderName);
		stateFeatureEmbedding.logEmbeddingsAsCSV(folderName, "state");
		actionFeatureEmbedding.logEmbeddingsAsCSV(folderName, "action");
		
		if(semanticFeatureEmbedding != null) {
			throw new RuntimeException("Not supported");
		}
		
		if(perceptronLayer != null) {
			throw new RuntimeException("Not supported");
		}
		
		if(actionMixingLayer != null) {
			throw new RuntimeException("Not supported");
		}
		
		//Save W
		try (
				PrintWriter writer = new PrintWriter(folderName + "/W.csv", "UTF-8");
			) {
				INDArray W = this.parser.getAffineW().dup();
				W = W.reshape(new int[]{1, this.parser.getAffineW().size(0) * this.parser.getAffineW().size(1)});
				writer.print(Helper.printVectorToCSV(W));
				writer.close();
			} catch(IOException ex) {
				throw new RuntimeException("Cannot store serializable data");
			}
	}
	
	public void bootstrap(String folderName) {
		bootstrap(folderName, this.parser);
//		bootstrapAsCSV(folderName, this.parser);
//		this.logModelAsCSV("csv_" + folderName + "_");
//		System.exit(0);
	}
	
	public static void bootstrap(String folderName, NeuralDotProductShiftReduceParser<?, ?> parser) {
		
		final NeuralParsingDotProductStepScorer mlpScorer = parser.getMLPScorer();
		final FeatureEmbedding<?> stateFeatureEmbedding = parser.getStateFeatureEmbedding();
		final FeatureEmbedding<?> actionFeatureEmbedding = parser.getActionFeatureEmbedding();
		final SemanticFeaturesEmbedding semanticFeatureEmbedding = parser.getSemanticFeatureEmbedding();
		final PerceptronLayer perceptronLayer = parser.getPerceptronLayer();
		final NeuralActionEmbeddingMixer actionMixingLayer = parser.getActionMixingLayer();
		
		mlpScorer.bootstrapNetworkParam(folderName);
		mlpScorer.reclone();
		
		stateFeatureEmbedding.bootstrapEmbeddings(folderName, "state");
		actionFeatureEmbedding.bootstrapEmbeddings(folderName, "action");
		
		if(semanticFeatureEmbedding != null) {
			semanticFeatureEmbedding.getSemanticEmbeddingObject().bootstrapCategoryEmbeddingAndRecursiveNetworkParam(folderName);
		}
		
		if(perceptronLayer != null) {
			perceptronLayer.bootstrapPerceptronWeights(folderName);
		}
		
		if(actionMixingLayer != null) {
			actionMixingLayer.bootstrapNetworkParam(folderName);
			actionMixingLayer.reclone();
		}
		
		//load W
		final String paramFile = folderName+"/W.bin";
		
		try {
		
			DataInputStream dis = new DataInputStream(new FileInputStream(paramFile));
			INDArray loadedW = Nd4j.read(dis);
			Nd4j.copy(loadedW, parser.getAffineW());
			
			dis.close();
		} catch(IOException e) {
			throw new RuntimeException("Could not read the top layer param: "+e);
		}
	}
	
	public static void bootstrapAsCSV(String folderName, NeuralDotProductShiftReduceParser<?, ?> parser) {
		
		final NeuralParsingDotProductStepScorer mlpScorer = parser.getMLPScorer();
		final FeatureEmbedding<?> stateFeatureEmbedding = parser.getStateFeatureEmbedding();
		final FeatureEmbedding<?> actionFeatureEmbedding = parser.getActionFeatureEmbedding();
		final SemanticFeaturesEmbedding semanticFeatureEmbedding = parser.getSemanticFeatureEmbedding();
		final PerceptronLayer perceptronLayer = parser.getPerceptronLayer();
		final NeuralActionEmbeddingMixer actionMixingLayer = parser.getActionMixingLayer();
		
		mlpScorer.bootstrapNetworkParamFromCSV(folderName);
		mlpScorer.reclone();
		
		stateFeatureEmbedding.bootstrapEmbeddingsAsCSV(folderName, "state");
		actionFeatureEmbedding.bootstrapEmbeddingsAsCSV(folderName, "action");
		
		if(semanticFeatureEmbedding != null) {
			throw new RuntimeException("Not supported");
		}
		
		if(perceptronLayer != null) {
			throw new RuntimeException("Not supported");
		}
		
		if(actionMixingLayer != null) {
			throw new RuntimeException("Not supported");
		}
		
		//load W
		final String paramFile = folderName + "/W.csv";
		
		String line = null;
					
		try (BufferedReader br = new BufferedReader(new FileReader(paramFile))) {
			line = br.readLine();
		} catch(IOException e) {
			throw new RuntimeException("Could not read W parameters from csv. Error" + e);
		}
			
		if(line == null) {
			throw new RuntimeException("Could not read W parameters from csv. Found null.");
		}
		
		INDArray loadedW = Helper.toVector(line);
		loadedW = loadedW.reshape(parser.getAffineW().size(0), parser.getAffineW().size(1));
	
		Nd4j.copy(loadedW, parser.getAffineW());
	}

	public static class Builder<SAMPLE extends IDataItem<?>, DI extends ILabeledDataItem<SAMPLE, ?>, MR> {
	
		private final IDataCollection<DI> trainingData;
		private final NeuralDotProductShiftReduceParser<Sentence, MR> parser;
		private final IValidator<DI, MR> validator;
		
		private Integer epoch = 30;
		private Double learningRate = 0.02;
		private Double learningRateDecay = 0.001;
		private Double l2 = 0.000001;
		private Integer beamSize = 10; 
		private boolean preTrain = false;
		private boolean saveModelAfterLearning = true;
		
		/** How many top samples to consider while approximating the partition function.
		 * This is used during backpropagation where we only backprop through top k samples.*/
		private Integer partitionFunctionApproximationK = 30;
		
		private CompositeImmutableLexicon<MR> compositeLexicon;
		private IParsingFilterFactory<DI, MR> parsingFilterFactory;
		private ILexiconImmutable<MR> tempLexicon;
		
		private String folderName = null;
		
		/** Validation metric */
		private ValidationStatistics stats;

		public Builder(IDataCollection<DI> trainingData, NeuralDotProductShiftReduceParser<Sentence, MR> parser,
					   IValidator<DI, MR> validator) {
			this.trainingData = trainingData;
			this.parser = parser;			
			this.validator = validator;
		}
		
		public NeuralFeedForwardDotProductLearner<SAMPLE, DI, MR> build() {
			return new NeuralFeedForwardDotProductLearner<SAMPLE, DI, MR>(trainingData, parser, validator,  
					epoch, learningRate, learningRateDecay, l2, beamSize, partitionFunctionApproximationK, 
					parsingFilterFactory, compositeLexicon, tempLexicon, preTrain, folderName, saveModelAfterLearning, stats);
		}
		
		public Builder<SAMPLE, DI, MR> setLexiconImmutable(ILexiconImmutable<MR> tempLexicon) {
			this.tempLexicon = tempLexicon;
			return this;
		}
		
		public Builder<SAMPLE, DI, MR> setValidationStatistics(ValidationStatistics stats) {
			this.stats = stats;
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
								implements IResourceObjectCreator<NeuralFeedForwardDotProductLearner<SAMPLE, DI, MR>> {

		private final String type;
		
		public Creator() {
			this("parser.feedfoward.dotproduct.shiftreduce.learner");
		}

		public Creator(String type) {
			this.type = type;
		}
		
		@Override
		public NeuralFeedForwardDotProductLearner<SAMPLE, DI, MR> create(Parameters params, IResourceRepository repo) {
		
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
			
			if(params.contains("validationStats")) {
				builder.setValidationStatistics(repo.get("validationStats"));
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
