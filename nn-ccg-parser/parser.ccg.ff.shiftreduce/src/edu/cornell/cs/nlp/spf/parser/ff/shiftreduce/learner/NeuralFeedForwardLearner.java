package edu.cornell.cs.nlp.spf.parser.ff.shiftreduce.learner;

import java.io.File;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;

import org.nd4j.linalg.api.ndarray.INDArray;

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
import edu.cornell.cs.nlp.spf.parser.ccg.rnn.shiftreduce.learning.LearningRate;
import edu.cornell.cs.nlp.spf.parser.ff.shiftreduce.dataset.CreateSparseFeatureDataset;
import edu.cornell.cs.nlp.spf.parser.ff.shiftreduce.dataset.SparseFeatureDataset;
import edu.cornell.cs.nlp.spf.parser.ff.shiftreduce.neuralparser.FeatureEmbedding;
import edu.cornell.cs.nlp.spf.parser.ff.shiftreduce.neuralparser.NeuralParsingStepScorer;
import edu.cornell.cs.nlp.spf.parser.ff.shiftreduce.neuralparser.NeuralShiftReduceParser;
import edu.cornell.cs.nlp.spf.parser.filter.IParsingFilterFactory;
import edu.cornell.cs.nlp.spf.parser.joint.model.IJointModelImmutable;
import edu.cornell.cs.nlp.utils.composites.Pair;
import edu.cornell.cs.nlp.utils.log.ILogger;
import edu.cornell.cs.nlp.utils.log.LogLevel;
import edu.cornell.cs.nlp.utils.log.LoggerFactory;
import edu.uw.cs.lil.amr.data.AMRMeta;

public class NeuralFeedForwardLearner<SAMPLE extends IDataItem<?>, DI extends ILabeledDataItem<SAMPLE, ?>, MR> {
	
	public static final ILogger	LOG = LoggerFactory.create(NeuralFeedForwardLearner.class);

	private final NeuralShiftReduceParser<Sentence, MR> parser;
	
	private final Integer epoch;
	private final Integer minEpoch;
	private final Integer partitionFunctionApproximationK;
	private final LearningRate learningRate;
	private final Integer beamSize;
	
	private final String bootstrapFolderName;
	private final boolean saveModelAfterLearning;
	
	private final boolean doGradientCheck;
	private boolean setDisplay;
	private double empiricalGrad;

	private final ValidationStatistics stats;
	
	public NeuralFeedForwardLearner(IDataCollection<DI> trainingData, 
			NeuralShiftReduceParser<Sentence, MR> parser, IValidator<DI,MR> validator, 
			Integer epoch, Double learningRate, Double learningRateDecay, Double l2, Integer beamSize, 
			Integer partitionFunctionApproximationK, IParsingFilterFactory<DI, MR> parsingFilterFactory, 
			CompositeImmutableLexicon<MR> compositeLexicon, ILexiconImmutable<MR> tempLexicon, 
			boolean preTrain, String folderName, boolean saveModelAfterLearning, ValidationStatistics stats) {
		this.parser = parser;
		
		this.epoch = epoch;
		this.minEpoch = 2;
		this.learningRate = new LearningRate(learningRate, learningRateDecay);
		
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
	
	private double calcLoss(SparseFeatureDataset<MR> pt) {
		
		final NeuralParsingStepScorer mlpScorer = this.parser.getMLPScorer();
		final FeatureEmbedding<MR> featureEmbedding = this.parser.getFeatureEmbedding();
			
		final List<IHashVector> features = pt.getPossibleActionFeatures();
		final int gTruthIx = pt.getGroundTruthIndex();
		
		INDArray batch = featureEmbedding.embedFeatures(features).first();
		double[] exponents = mlpScorer.getEmbedding(batch);
		double[] logSoftMax = mlpScorer.toLogSoftMax(exponents);
		
		//Compute log-liklelihood and likelihood
		final double logLikelihood = logSoftMax[gTruthIx];
		final double loss = -1*logLikelihood;
		
		return loss;
	}
	
	private void gradientCheck(SparseFeatureDataset<MR> pt) {
		
		IHashVector feature = pt.getPossibleActionFeatures().get(0);
		final FeatureEmbedding<MR> featureEmbedding = this.parser.getFeatureEmbedding();
		final double epsilon = 0.00001;
		
		//Gradient check for input to MLP
		{
			final NeuralParsingStepScorer mlpScorer = this.parser.getMLPScorer();	
			final List<IHashVector> features = pt.getPossibleActionFeatures();
			final int gTruthIx = pt.getGroundTruthIndex();
			
			INDArray batch = featureEmbedding.embedFeatures(features).first();
			final double orig = batch.getDouble(new int[]{0, 0});
			
			batch.putScalar(new int[]{0, 0}, orig + epsilon);
			double[] exponents = mlpScorer.getEmbedding(batch);
			double[] logSoftMax = mlpScorer.toLogSoftMax(exponents);
			final double loss1 = -1*logSoftMax[gTruthIx];
			
			batch.putScalar(new int[]{0, 0}, orig - epsilon);
			exponents = mlpScorer.getEmbedding(batch);
			logSoftMax = mlpScorer.toLogSoftMax(exponents);
			final double loss2 = -1*logSoftMax[gTruthIx];
			
			this.empiricalGrad = (loss1 - loss2)/(2.0 * epsilon);
		}
		
		int numActiveFeature = feature.size();
		if(numActiveFeature == 0) {
			featureEmbedding.setEmpiricalGrad(null);
			return;
		}
		
		//Gradient check for individual features
		KeyArgs firstFeature = feature.iterator().next().first();
		INDArray vec = featureEmbedding.getGradientCheckFeature(firstFeature);
		
		{
			double orig = vec.getDouble(new int[]{0, 0});
			vec.putScalar(new int[]{0, 0}, orig + epsilon);
			double loss1 = this.calcLoss(pt);
			
			vec.putScalar(new int[]{0, 0}, orig - epsilon);
			double loss2 = this.calcLoss(pt);
			
			vec.putScalar(new int[]{0, 0}, orig);
			
			double empiricalGrad = (loss1 - loss2)/(2.0 * epsilon);
			featureEmbedding.setEmpiricalGrad(empiricalGrad);
		}
		
		//Gradient check for MLP
		{
			final NeuralParsingStepScorer mlpScorer = this.parser.getMLPScorer();	
			final List<IHashVector> features = pt.getPossibleActionFeatures();
			final int gTruthIx = pt.getGroundTruthIndex();
			
			INDArray batch = featureEmbedding.embedFeatures(features).first();
			Pair<double[], double[]> exponents = mlpScorer.gradientCheckGetEmbedding(batch, epsilon);
			
			double[] logSoftMax1 = mlpScorer.toLogSoftMax(exponents.first());
			final double loss1 = -1*logSoftMax1[gTruthIx];
			
			double[] logSoftMax2 = mlpScorer.toLogSoftMax(exponents.second());
			final double loss2 = -1*logSoftMax2[gTruthIx];
			
			double empiricalGrad = (loss1 - loss2)/(2.0 * epsilon);
			mlpScorer.setEmpiricalGrad(empiricalGrad);
		}
	}

	/** Calculates log-likelihood and other statistics on the given batch. */
	private double calcCompositeBatchLikelihood(List<SparseFeatureDataset<MR>> processedDataset) {
		
		final NeuralParsingStepScorer mlpScorer = this.parser.getMLPScorer();
		final FeatureEmbedding<MR> featureEmbedding = this.parser.getFeatureEmbedding();
		
		double sumLogLikelihood = 0.0, sumLikelihood = 0.0;
		AtomicInteger correct = new AtomicInteger();
		
		int exampleIndex = 0;
			
		for(SparseFeatureDataset<MR> pt: processedDataset) {
			LOG.info("=========================");
			LOG.info("Example: %s", ++exampleIndex);
			
			if(this.setDisplay) {
				LOG.info/*debug*/("Sentence %s", pt.getSentence());
			}
			
			final List<IHashVector> features = pt.getPossibleActionFeatures();
			final int gTruthIx = pt.getGroundTruthIndex();
			
			INDArray batch = featureEmbedding.embedFeatures(features).first();
			double[] exponents = mlpScorer.getEmbedding(batch);
			
			double[] logSoftMax = mlpScorer.toLogSoftMax(exponents);
			
			//Compute log-liklelihood and likelihood
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
				if(logSoftMax[gTruthIx] < logSoftMax[maxScore]) { //currently printing the really bad ones
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
	public void fitCompositeDataSet(CreateSparseFeatureDataset<SAMPLE, DI, MR> datasetCreator, 
			IJointModelImmutable<SituatedSentence<AMRMeta>, LogicalExpression, LogicalExpression> model) {
		
		final NeuralParsingStepScorer mlpScorer = this.parser.getMLPScorer();
		final FeatureEmbedding<MR> featureEmbedding = this.parser.getFeatureEmbedding();
		this.parser.disablePacking();
		
		long totalTime = 0;	
		int numIterations = 0;
		
		// Save the initial model
		this.logModel("init");
		
		List<SparseFeatureDataset<MR>> dataset = null;
		this.parser.testing = false;
		
		for(int iter = 1; iter <= this.epoch; iter++) {
			LOG.info("=========================");
			LOG.info("Create Training Data. Epoch %s", iter);
			
			//if(dataset == null) {
				dataset = datasetCreator.createDataset(model);
				//dataset = datasetCreator.createDatasetWithExploration(model, iter);
				featureEmbedding.registerFeatures(dataset);
			//}
			//System.exit(0);
			 
			final int dataSize = dataset.size();
			final int trainSize = (int)(0.9*dataSize);
			List<SparseFeatureDataset<MR>> train = dataset.subList(dataSize - trainSize, trainSize);//0, trainSize);
			List<SparseFeatureDataset<MR>> validation = dataset.subList(0, dataSize - trainSize);//trainSize, dataSize);
			
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
			
			//if(iter == 1) {
				featureEmbedding.stats();
			//}
			
			LOG.info("Fitting Dataset of size %s, train %s, validation %s", 
											dataset.size(), trainSize, dataSize - trainSize);
			
			LOG.info("Fit Dataset Iteration: %s", iter);
			int exampleIndex = 0;
			double sumLogLikelihood = 0.0, sumLikelihood = 0.0;
			
			for(SparseFeatureDataset<MR> pt: train) {
				LOG.info("=========================");
				LOG.info("Example: %s", ++exampleIndex);
				
				if(this.doGradientCheck) {
					this.gradientCheck(pt);
				}
				
				final List<IHashVector> features = pt.getPossibleActionFeatures();
				final int gTruthIx = pt.getGroundTruthIndex();
				Pair<INDArray, List<int[]>> res = featureEmbedding.embedFeatures(features);
				INDArray batch = res.first();
				double[] exponents = mlpScorer.getEmbedding(batch);
				
				double[] logSoftMax = mlpScorer.toLogSoftMax(exponents);
				
				//Compute log-liklelihood and likelihood
				final double logLikelihood = logSoftMax[gTruthIx];
				final double likelihood = Math.exp(logLikelihood);
				
				LOG.info("Iteration: %s, Sum of Likelihood %s, NLL %s", iter, likelihood, -logLikelihood);
				
				sumLogLikelihood = sumLogLikelihood + logLikelihood;
				sumLikelihood = sumLikelihood + likelihood;
				
				//Compute error
				Double[] error = new Double[exponents.length];
				for(int i = 0; i < exponents.length; i++) {
					if(i == gTruthIx) {
						error[i] = -1  + Math.exp(logSoftMax[i]);
					} else {
						error[i] = Math.exp(logSoftMax[i]);
					}
				}
				
				//Do backpropagation 
				List<INDArray> errorFeatureInput = mlpScorer.backprop(error);
				
				if(this.doGradientCheck) {
					LOG.info("Gradient Check:: Batch input: Empirical %s Estimate %s", this.empiricalGrad, 
						errorFeatureInput.get(0).getDouble(new int[]{0, 0}));
				}
				
				//Update the features
				featureEmbedding.backprop(errorFeatureInput, features, res.second());
				
				//Update the parameters
				featureEmbedding.update();
				
				//Flush the gradients
				featureEmbedding.flush();
				
				numIterations++;
			}
			
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
			/// Save it for the first epoch and every fifth epoch except the last (which is saved separately)
			if((iter == 1 || iter%5 == 0) && iter != this.epoch) {
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
		featureEmbedding.stopAddingFeatures();
		featureEmbedding.stats();
		featureEmbedding.clearSeenFeaturesStats();
		
		LOG.info("Setting temporary model new features");
		this.parser.modelNewFeatures  = datasetCreator.getModelNewFeatures();
		this.parser.testing = true;
	}
	
	public void fixModelForTesting(CreateSparseFeatureDataset<SAMPLE, DI, MR> datasetCreator) {
		
		final FeatureEmbedding<MR> featureEmbedding = this.parser.getFeatureEmbedding();
		featureEmbedding.stopAddingFeatures();
		featureEmbedding.stats();
		featureEmbedding.clearSeenFeaturesStats();
		LOG.info("Setting temporary model new features");
		this.parser.modelNewFeatures  = datasetCreator.getModelNewFeatures();
		this.parser.testing = true;
	}
	
	public void logModel(String modelName) {
		
		final String folderName = modelName + "_" + System.currentTimeMillis();
		LOG.info("Logging the model %s", folderName);
		
		//create the folder
		File dir = new File(folderName);
		dir.mkdir();
		
		final NeuralParsingStepScorer mlpScorer = this.parser.getMLPScorer();
		final FeatureEmbedding<MR> featureEmbedding = this.parser.getFeatureEmbedding();
		
		mlpScorer.logNetwork(folderName);
		featureEmbedding.logEmbeddings(folderName);
	}
	
	public void bootstrap(String folderName) {
		final NeuralParsingStepScorer mlpScorer = this.parser.getMLPScorer();
		final FeatureEmbedding<MR> featureEmbedding = this.parser.getFeatureEmbedding();
		
		mlpScorer.bootstrapNetworkParam(folderName);
		mlpScorer.reclone();
		
		featureEmbedding.bootstrapEmbeddings(folderName);
	}

	public static class Builder<SAMPLE extends IDataItem<?>, DI extends ILabeledDataItem<SAMPLE, ?>, MR> {
	
		private final IDataCollection<DI> trainingData;
		private final NeuralShiftReduceParser<Sentence, MR> parser;
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

		public Builder(IDataCollection<DI> trainingData, NeuralShiftReduceParser<Sentence, MR> parser,
					   IValidator<DI, MR> validator) {
			this.trainingData = trainingData;
			this.parser = parser;			
			this.validator = validator;
		}
		
		public NeuralFeedForwardLearner<SAMPLE, DI, MR> build() {
			return new NeuralFeedForwardLearner<SAMPLE, DI, MR>(trainingData, parser, validator,  
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
								implements IResourceObjectCreator<NeuralFeedForwardLearner<SAMPLE, DI, MR>> {

		private final String type;
		
		public Creator() {
			this("parser.feedfoward.shiftreduce.learner");
		}

		public Creator(String type) {
			this.type = type;
		}
		
		@Override
		public NeuralFeedForwardLearner<SAMPLE, DI, MR> create(Parameters params, IResourceRepository repo) {
		
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
