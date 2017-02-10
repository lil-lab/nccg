/*******************************************************************************
 * UW SPF - The University of Washington Semantic Parsing Framework
 * <p>
 * Copyright (C) 2013 Yoav Artzi
 * <p>
 * This program is free software; you can redistribute it and/or modify it under
 * the terms of the GNU General Public License as published by the Free Software
 * Foundation; either version 2 of the License, or any later version.
 * <p>
 * This program is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
 * details.
 * <p>
 * You should have received a copy of the GNU General Public License along with
 * this program; if not, write to the Free Software Foundation, Inc., 51
 * Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.
 ******************************************************************************/
package edu.uw.cs.lil.amr.neural.exp;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.PrintWriter;
import java.io.UnsupportedEncodingException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Set;

import edu.cornell.cs.nlp.spf.ccg.lexicon.ILexiconImmutable;
import edu.cornell.cs.nlp.spf.data.collection.IDataCollection;
import edu.cornell.cs.nlp.spf.data.sentence.Sentence;
import edu.cornell.cs.nlp.spf.data.singlesentence.SingleSentence;
import edu.cornell.cs.nlp.spf.data.situated.ISituatedDataItem;
import edu.cornell.cs.nlp.spf.data.situated.sentence.SituatedSentence;
import edu.cornell.cs.nlp.spf.exec.IExec;
import edu.cornell.cs.nlp.spf.explat.DistributedExperiment;
import edu.cornell.cs.nlp.spf.explat.Job;
import edu.cornell.cs.nlp.spf.explat.resources.ResourceCreatorRepository;
import edu.cornell.cs.nlp.spf.learn.ILearner;
import edu.cornell.cs.nlp.spf.mr.lambda.LogicalExpression;
import edu.cornell.cs.nlp.spf.mr.lambda.ccg.LogicalExpressionCategoryServices;
import edu.cornell.cs.nlp.spf.parser.ccg.model.IModelImmutable;
import edu.cornell.cs.nlp.spf.parser.ccg.model.IModelInit;
import edu.cornell.cs.nlp.spf.parser.ccg.model.Model;
import edu.cornell.cs.nlp.spf.parser.ccg.model.ModelLogger;
import edu.cornell.cs.nlp.spf.parser.ccg.rnn.shiftreduce.dataset.CreateCompositeDecisionDataset;
import edu.cornell.cs.nlp.spf.parser.ccg.rnn.shiftreduce.learning.RNNShiftReduceLearner;
import edu.cornell.cs.nlp.spf.parser.ccg.rnn.shiftreduce.neuralnetworkparser.NeuralNetworkShiftReduceParser;
import edu.cornell.cs.nlp.spf.parser.ff.shiftreduce.dataset.CreateSparseFeatureAndStateDataset;
import edu.cornell.cs.nlp.spf.parser.ff.shiftreduce.dataset.CreateSparseFeatureDataset;
import edu.cornell.cs.nlp.spf.parser.ff.shiftreduce.learner.NeuralFeedForwardDotProductLearner;
import edu.cornell.cs.nlp.spf.parser.ff.shiftreduce.neuralparser.LocalEnsembleNeuralDotProductShiftReduceParser;
import edu.cornell.cs.nlp.spf.parser.ff.shiftreduce.neuralparser.NeuralDotProductShiftReduceParser;
import edu.cornell.cs.nlp.spf.parser.joint.model.IJointModelImmutable;
import edu.cornell.cs.nlp.spf.parser.joint.model.IJointModelInit;
import edu.cornell.cs.nlp.spf.parser.joint.model.IJointModelProcessor;
import edu.cornell.cs.nlp.spf.parser.joint.model.JointModel;
import edu.cornell.cs.nlp.spf.reliabledist.EnslavedLocalManager;
import edu.cornell.cs.nlp.spf.reliabledist.ReliableManager;
import edu.cornell.cs.nlp.spf.test.exec.IExecTester;
import edu.cornell.cs.nlp.spf.test.stats.CompositeTestingStatistics;
import edu.cornell.cs.nlp.spf.test.stats.ITestingStatistics;
import edu.cornell.cs.nlp.utils.log.ILogger;
import edu.cornell.cs.nlp.utils.log.LoggerFactory;
import edu.uw.cs.lil.amr.Init;
import edu.uw.cs.lil.amr.data.AMRMeta;
import edu.uw.cs.lil.amr.data.LabeledAmrSentence;
import edu.uw.cs.lil.amr.data.LabeledAmrSentenceCollection;
import edu.uw.cs.lil.amr.exp.ParseJob;

public class AmrNeuralExp extends DistributedExperiment {
	public static final ILogger LOG	= LoggerFactory
											.create(AmrNeuralExp.class);

	private final LogicalExpressionCategoryServices	categoryServices;
	
	public AmrNeuralExp(File initFile) throws IOException {
		this(initFile, Collections.<String, String> emptyMap(),
				new AmrNeuralResourceRepo());
	}

	public AmrNeuralExp(File initFile, Map<String, String> envParams,
			ResourceCreatorRepository creatorRepo) throws IOException {
		super(initFile, envParams, creatorRepo);

		// //////////////////////////////////////////
		// Get parameters
		// //////////////////////////////////////////
		final File typesFile = globalParams.getAsFile("types");
		final File specmapFile = globalParams.getAsFile("specmap");

		// //////////////////////////////////////////////////
		// Init AMR.
		// //////////////////////////////////////////////////

		Init.init(typesFile, specmapFile,
				globalParams.getAsFile("stanfordModel"), false,
				globalParams.getAsFile("nerConfig"),
				globalParams.getAsFile("nerTranslation"),
				globalParams.getAsFile("propBank"),
				globalParams.getAsBoolean("underspecifyPropBank", false));

		// //////////////////////////////////////////////////
		// Category services for logical expressions.
		// //////////////////////////////////////////////////

		this.categoryServices = new LogicalExpressionCategoryServices(true);
		storeResource(CATEGORY_SERVICES_RESOURCE, categoryServices);

		// //////////////////////////////////////////////////
		// Read resources.
		// //////////////////////////////////////////////////

		readResrouces();

		// //////////////////////////////////////////////////
		// Create jobs
		// //////////////////////////////////////////////////

		for (final Parameters params : jobParams) {
			addJob(createJob(params));
		}
	}

	public AmrNeuralExp(File initFile, String[] args) throws IOException {
		this(initFile, argsToMap(args), new AmrNeuralResourceRepo());
	}

	private static Map<String, String> argsToMap(String[] args) {
		final Map<String, String> map = new HashMap<>();
		for (final String arg : args) {
			final String[] split = arg.split("=", 2);
			if (split.length == 2) {
				map.put(split[0], split[1]);
			} else {
				throw new IllegalArgumentException("Invalid argument: " + arg);
			}
		}
		return map;
	}

	private Job createJob(Parameters params) throws FileNotFoundException {
		final String type = params.get("type");
		if (type.equals("train")) {
			return createTrainJob(params);
		} else if (type.equals("test")) {
			return createTestJob(params);
		} else if (type.equals("save")) {
			return createSaveJob(params);
		} else if ("process".equals(type)) {
			return createProcessingJob(params);
		} else if ("listen".equals(type)) {
			return createListenerRegistrationJob(params);
		} else if (type.equals("log")) {
			return createModelLoggingJob(params);
		} else if ("init".equals(type)) {
			return createModelInitJob(params);
		} else if ("tinydist.master".equals(type)) {
			return createReliableManagerJob(params);
		} else if ("tinydist.worker".equals(type)) {
			return createWorkerJob(params);
		} else if ("parse".equals(type)) {
			return createParseJob(params);
		} else if ("dataset.creation".equals(type)) {
			return createDataProcessing(params);
		} else if ("train.neural.parser".equals(type)) {
			return createLearnerJob(params);
		} else if ("train.ff.neural.parser".equals(type)) {
			return createFeedForwardLearnerJob(params);
		} else if ("memoize.category.embedding".equals(type)) {
			return createMemoizeCategoryEmbeddingTask(params);
		} else if ("bootstrap.neural.model".equals(type)) {
			return bootstrapNeuralModel(params);
		} else if ("bootstrap.feed.forward.neural.model".equals(type)) {
			return bootstrapFeedForwardNeuralModel(params);
		} else if ("bootstrap.ensemble.feed.forward.neural.model".equals(type)) {
			return bootstrapEnsembleFeedForwardNeuralModel(params);
		} else if ("bootstrap.local.ensemble.feed.forward.neural.model".equals(type)) {
			return bootstrapLocalEnsembleFeedForwardNeuralModel(params);
		} else if ("do.perceptron.update".equals(type)) {
			return doPerceptronUpdate(params);
		} else if("bootstrap.dataset".equals(type)) {
			return createReadDataset(params);
		} else if ("feed.forward.catch.early.error".equals(type)) {
			return createCatchEarlyParserJob(params);
		} else {
			throw new RuntimeException("Unsupported job type: " + type);
		}
	}

	private Job createListenerRegistrationJob(Parameters params)
			throws FileNotFoundException {
		return new Job(params.get("id"), new HashSet<>(params.getSplit("dep")),
				this, createJobOutputFile(params.get("id")),
				createJobLogFile(params.get("id"))) {

			@Override
			protected void doJob() {
				LOG.info("Registering listeners to model, id=%s",
						params.get("model"));
				final Model<?, ?> model = get(params.get("model"));
				for (final String listenerId : params.getSplit("listeners")) {
					model.registerListener(get(listenerId));
				}

			}
		};
	}

	@SuppressWarnings("unchecked")
	private Job createModelInitJob(Parameters params)
			throws FileNotFoundException {

		final JointModel<SituatedSentence<AMRMeta>, LogicalExpression, LogicalExpression> model = get(
				params.get("model"));
		final List<Runnable> runnables = new LinkedList<>();
		for (final String id : params.getSplit("init")) {
			final Object init = get(id);
			if (init instanceof IModelInit) {
				runnables
						.add(() -> ((IModelInit<SituatedSentence<AMRMeta>, LogicalExpression>) init)
								.init(model));
			} else if (init instanceof IJointModelInit) {
				runnables
						.add(() -> ((IJointModelInit<SituatedSentence<AMRMeta>, LogicalExpression, LogicalExpression>) init)
								.init(model));
			} else {
				throw new RuntimeException("invalid init type");
			}
		}

		return new Job(params.get("id"), new HashSet<>(params.getSplit("dep")),
				this, createJobOutputFile(params.get("id")),
				createJobLogFile(params.get("id"))) {

			@Override
			protected void doJob() {
				for (final Runnable runnable : runnables) {
					runnable.run();
				}
			}
		};
	}

	private Job createModelLoggingJob(Parameters params)
			throws FileNotFoundException {
		final IModelImmutable<?, ?> model = get(params.get("model"));
		final ModelLogger modelLogger = get(params.get("logger"));
		return new Job(params.get("id"), new HashSet<>(params.getSplit("dep")),
				this, createJobOutputFile(params.get("id")),
				createJobLogFile(params.get("id"))) {

			@Override
			protected void doJob() {
				modelLogger.log(model, getOutputStream());
			}
		};
	}

	private Job createParseJob(Parameters params) throws FileNotFoundException {
		return new ParseJob(params.get("id"),
				new HashSet<>(params.getSplit("dep")), this,
				createJobOutputFile(params.get("id")),
				createJobLogFile(params.get("id")),
				get(params.get("sentences")), get(params.get("exec")),
				params.getAsBoolean("allowSloppy", true));
	}

	private <DI extends ISituatedDataItem<?, ?>, MR, ESTEP> Job createProcessingJob(
			Parameters params) throws FileNotFoundException {
		final JointModel<DI, MR, ESTEP> model = get(params.get("model"));
		final IJointModelProcessor<DI, MR, ESTEP> processor = get(
				params.get("processor"));
		assert model != null;
		assert processor != null;
		return new Job(params.get("id"), new HashSet<>(params.getSplit("dep")),
				this, createJobOutputFile(params.get("id")),
				createJobLogFile(params.get("id"))) {

			@Override
			protected void doJob() {
				// Process the model.
				LOG.info("Processing model...");
				final long startTime = System.currentTimeMillis();
				processor.process(model);
				LOG.info("Processing completed (%.3fsec).",
						(System.currentTimeMillis() - startTime) / 1000.0);
			}
		};
	}

	private Job createReliableManagerJob(Parameters params)
			throws FileNotFoundException {
		return new Job(params.get("id"), new HashSet<>(params.getSplit("dep")),
				this, createJobOutputFile(params.get("id")),
				createJobLogFile(params.get("id"))) {

			@Override
			protected void doJob() {
				LOG.info("Starting tinydist reliable manager");
				final ReliableManager manager = get(params.get("manager"));
				manager.start();
			}
		};
	}

	private Job createSaveJob(final Parameters params)
			throws FileNotFoundException {
		return new Job(params.get("id"), new HashSet<>(params.getSplit("dep")),
				this, createJobOutputFile(params.get("id")),
				createJobLogFile(params.get("id"))) {

			@SuppressWarnings("unchecked")
			@Override
			protected void doJob() {
				// Save the model to file.
				try {
					LOG.info("Saving model (id=%s) to: %s", params.get("model"),
							params.getAsFile("file").getAbsolutePath());
					Model.write(
							(Model<Sentence, LogicalExpression>) get(
									params.get("model")),
							params.getAsFile("file"));
					LOG.info("Model saved");
				} catch (final IOException e) {
					LOG.error("Failed to save model to: %s",
							params.get("file"));
					throw new RuntimeException(e);
				}

			}
		};
	}

	private Job createTestJob(Parameters params) throws FileNotFoundException {
		// Create test statistics.
		final List<ITestingStatistics<SituatedSentence<AMRMeta>, LogicalExpression, LabeledAmrSentence>> testingMetrics = new LinkedList<>();
		for (final String statsId : params.getSplit("stats")) {
			testingMetrics.add(get(statsId));
		}
		final ITestingStatistics<SituatedSentence<AMRMeta>, LogicalExpression, LabeledAmrSentence> testStatistics = new CompositeTestingStatistics<>(
				testingMetrics);

		// Get the executor.
		final IExec<SituatedSentence<AMRMeta>, LogicalExpression> exec = get(
				params.get("exec"));

		// Get the tester.
		final IExecTester<SituatedSentence<AMRMeta>, LogicalExpression, LabeledAmrSentence> tester = get(
				params.get("tester"));

		// Get the data.
		final IDataCollection<LabeledAmrSentence> data = get(
				params.get("data"));
		
		// Filter the data --- This is temporary. 
		// TODO in future filter using a proper filter from
		// the command line
		final List<LabeledAmrSentence> filterAmrSentences = new LinkedList<LabeledAmrSentence>();
		Iterator<LabeledAmrSentence> it = data.iterator();
		while(it.hasNext()) {
			LabeledAmrSentence labeledAmrSentence = it.next();
			filterAmrSentences.add(labeledAmrSentence);
		}
		
		LOG.info("Created filtered dataset of size %s", filterAmrSentences.size());
		final IDataCollection<LabeledAmrSentence> filterData = new LabeledAmrSentenceCollection(filterAmrSentences); 

		// Create and return the job.
		return new Job(params.get("id"), new HashSet<>(params.getSplit("dep")),
				this, createJobOutputFile(params.get("id")),
				createJobLogFile(params.get("id"))) {

			@Override
			protected void doJob() {

				// Record start time.
				final long startTime = System.currentTimeMillis();

				// Job started.
				LOG.info("============ (Job %s started)", getId());

				// Test the final model.
				tester.test(exec, filterData/*data*/, testStatistics);
				LOG.info("%s\n", testStatistics);
				getOutputStream()
						.println(testStatistics.toTabDelimitedString());

				// Output total run time..
				LOG.info("Total run time %.4f seconds",
						(System.currentTimeMillis() - startTime) / 1000.0);

				// Job completed
				LOG.info("============ (Job %s completed)", getId());
			}
		};
	}

	@SuppressWarnings("unchecked")
	private Job createTrainJob(Parameters params) throws FileNotFoundException {
		// The model to use
		final Model<Sentence, LogicalExpression> model = (Model<Sentence, LogicalExpression>) get(
				params.get("model"));

		// The learning
		final ILearner<Sentence, SingleSentence, Model<Sentence, LogicalExpression>> learner = (ILearner<Sentence, SingleSentence, Model<Sentence, LogicalExpression>>) get(
				params.get("learner"));

		return new Job(params.get("id"), new HashSet<>(params.getSplit("dep")),
				this, createJobOutputFile(params.get("id")),
				createJobLogFile(params.get("id"))) {

			@Override
			protected void doJob() {
				final long startTime = System.currentTimeMillis();

				// Start job
				LOG.info("============ (Job %s started)", getId());

				// Do the learning
				learner.train(model);

				// Output total run time
				LOG.info("Total run time %.4f seconds",
						(System.currentTimeMillis() - startTime) / 1000.0);

				// Job completed
				LOG.info("============ (Job %s completed)", getId());

			}
		};
	}

	private Job createWorkerJob(Parameters params)
			throws FileNotFoundException {
		return new Job(params.get("id"), new HashSet<>(params.getSplit("dep")),
				this, createJobOutputFile(params.get("id")),
				createJobLogFile(params.get("id"))) {

			@Override
			protected void doJob() {
				LOG.info("Starting tinydist worker");
				final EnslavedLocalManager worker = get(params.get("worker"));
				worker.run();
			}
		};
	}
	
	private Job bootstrapNeuralModel(Parameters params) throws FileNotFoundException {
		
		final String folderName = params.get("neuralModelFolder");
		
		final RNNShiftReduceLearner<Sentence, SingleSentence, LogicalExpression> learner = 
													get(params.get("learner"));
		
		final NeuralNetworkShiftReduceParser<?, LogicalExpression> parser = 
													get(params.get("parser"));
		
		final JointModel<SituatedSentence<AMRMeta>, LogicalExpression, LogicalExpression>
											model = get(params.get("model"));
		
		//model.getTheta().set("CLOSURE", -20);
		
		LOG.info("Model Theta %s", model.getTheta().get("CLOSURE"));
		
		final CreateCompositeDecisionDataset<Sentence, SingleSentence, LogicalExpression> 
							datasetCreator = get(params.get("datasetCreator"));
		
		return new Job(params.get("id"), new HashSet<>(params.getSplit("dep")),
				this, createJobOutputFile(params.get("id")),
				createJobLogFile(params.get("id"))) {
			
			@Override
			protected void doJob() {
				LOG.info("Going to bootstrap neural model");
				//induce constants for category
				datasetCreator.categoryAmrPreprocessing();
				
				//induce lexicon in parsing op embedding
				parser.getEmbedParsingOp().induceLexicalEntryEmbedding(model.getLexicon());
				
				//--- not sure if Dynamic lexical entries are being bootstrapped.
				
				//bootstrap model
				learner.bootstrapModel(folderName);
				LOG.info("Done bootstrapping");
			}
		};
	}
	
	private Job bootstrapFeedForwardNeuralModel(Parameters params) throws FileNotFoundException {
		
		final String folderName = params.get("neuralModelFolder");
		
		final NeuralFeedForwardDotProductLearner<Sentence, SingleSentence, LogicalExpression> learner = 
													get(params.get("learner"));
		
		final CreateSparseFeatureAndStateDataset<Sentence, SingleSentence, LogicalExpression> datasetCreator =
													get(params.get("datasetCreator"));

		final IJointModelImmutable<SituatedSentence<AMRMeta>, LogicalExpression, LogicalExpression> model =
													get(params.get("model"));
		
		final JointModel<SituatedSentence<AMRMeta>, LogicalExpression, LogicalExpression> modelNewFeatures =
													get(params.get("modelNewFeatures"));
		
		LOG.info("Size of model lexicon %s", modelNewFeatures.getLexicon().size());
		
		ILexiconImmutable<LogicalExpression> lexicon = model.getLexicon();
		LOG.info("Temporary Size of lexicon %s and of toCollection %s", lexicon.size(), lexicon.toCollection().size());
		modelNewFeatures.addLexEntries(lexicon.toCollection());
		
		LOG.info("Size of model lexicon now %s", modelNewFeatures.getLexicon().size());
		LOG.info("Model features. model with New Features %s",
							modelNewFeatures.getJointFeatures().size());
		LOG.info("Model parse feature set. model with New Features %s",
							modelNewFeatures.getParseFeatures().size());

		datasetCreator.modelNewFeatures = modelNewFeatures;
		
		return new Job(params.get("id"), new HashSet<>(params.getSplit("dep")),
				this, createJobOutputFile(params.get("id")),
				createJobLogFile(params.get("id"))) {
			
			@Override
			protected void doJob() {
				LOG.info("Going to bootstrap feed foward neural model");
				//bootstrap model
				learner.bootstrap(folderName);
				learner.fixModelForTesting(datasetCreator);
				LOG.info("Done bootstrapping");
			}
		};
	}
	
	private Job doPerceptronUpdate(Parameters params) throws FileNotFoundException {
		
		final String folderName = params.get("neuralModelFolder");
		
		final NeuralFeedForwardDotProductLearner<Sentence, SingleSentence, LogicalExpression> learner = 
													get(params.get("learner"));
		
		final CreateSparseFeatureAndStateDataset<Sentence, SingleSentence, LogicalExpression> datasetCreator =
													get(params.get("datasetCreator"));

		final IJointModelImmutable<SituatedSentence<AMRMeta>, LogicalExpression, LogicalExpression> model =
													get(params.get("model"));
		
		final JointModel<SituatedSentence<AMRMeta>, LogicalExpression, LogicalExpression> modelNewFeatures =
													get(params.get("modelNewFeatures"));
		
		LOG.info("Size of model lexicon %s", modelNewFeatures.getLexicon().size());
		
		ILexiconImmutable<LogicalExpression> lexicon = model.getLexicon();
		LOG.info("Temporary Size of lexicon %s and of toCollection %s", lexicon.size(), lexicon.toCollection().size());
		modelNewFeatures.addLexEntries(lexicon.toCollection());
		
		LOG.info("Size of model lexicon now %s", modelNewFeatures.getLexicon().size());
		LOG.info("Model features. model with New Features %s",
							modelNewFeatures.getJointFeatures().size());
		LOG.info("Model parse feature set. model with New Features %s",
							modelNewFeatures.getParseFeatures().size());

		datasetCreator.modelNewFeatures = modelNewFeatures;
		
		return new Job(params.get("id"), new HashSet<>(params.getSplit("dep")),
				this, createJobOutputFile(params.get("id")),
				createJobLogFile(params.get("id"))) {
			
			@Override
			protected void doJob() {
				LOG.info("Going to do perceptron update");
				//bootstrap model
				learner.bootstrap(folderName);
				learner.fitPerceptronToCompositeDataSet(datasetCreator, model);
				LOG.info("Done with perceptron update");
			}
		};
	}
	
	private Job bootstrapEnsembleFeedForwardNeuralModel(Parameters params) throws FileNotFoundException {
		
		final IJointModelImmutable<SituatedSentence<AMRMeta>, LogicalExpression, LogicalExpression> model =
													get(params.get("model"));
		
		final JointModel<SituatedSentence<AMRMeta>, LogicalExpression, LogicalExpression> modelNewFeatures =
													get(params.get("modelNewFeatures"));
		
		final List<NeuralDotProductShiftReduceParser<?, LogicalExpression>> ensemble = 
						new ArrayList<NeuralDotProductShiftReduceParser<?, LogicalExpression>>();
		
		final List<String> parserNames = params.getSplit("ensemble");
		for(String parserName: parserNames) {
			ensemble.add(get(parserName));
		}
		
		LOG.info("Size of model lexicon %s", modelNewFeatures.getLexicon().size());
		
		ILexiconImmutable<LogicalExpression> lexicon = model.getLexicon();
		modelNewFeatures.addLexEntries(lexicon.toCollection());
		
		LOG.info("Size of model lexicon now %s", modelNewFeatures.getLexicon().size());
		LOG.info("Model features. model with New Features %s",
							modelNewFeatures.getJointFeatures().size());
		LOG.info("Model parse feature set. model with New Features %s",
							modelNewFeatures.getParseFeatures().size());
		
		return new Job(params.get("id"), new HashSet<>(params.getSplit("dep")),
				this, createJobOutputFile(params.get("id")),
				createJobLogFile(params.get("id"))) {
			
			@Override
			protected void doJob() {
				LOG.info("Going to bootstrap feed foward neural model");
				
				for(NeuralDotProductShiftReduceParser<?, LogicalExpression> parser: ensemble) {
					NeuralFeedForwardDotProductLearner.setFeatures(parser, modelNewFeatures);
				}
				LOG.info("Done bootstrapping");
			}
		};
	}
	
	private Job bootstrapLocalEnsembleFeedForwardNeuralModel(Parameters params) throws FileNotFoundException {
		
		final IJointModelImmutable<SituatedSentence<AMRMeta>, LogicalExpression, LogicalExpression> model =
													get(params.get("model"));
		
		final JointModel<SituatedSentence<AMRMeta>, LogicalExpression, LogicalExpression> modelNewFeatures =
													get(params.get("modelNewFeatures"));
		
		final LocalEnsembleNeuralDotProductShiftReduceParser<?, LogicalExpression> parser =
													get(params.get("parser"));
		
		LOG.info("Size of model lexicon %s", modelNewFeatures.getLexicon().size());
		
		ILexiconImmutable<LogicalExpression> lexicon = model.getLexicon();
		modelNewFeatures.addLexEntries(lexicon.toCollection());
		
		LOG.info("Size of model lexicon now %s", modelNewFeatures.getLexicon().size());
		LOG.info("Model features. model with New Features %s",
							modelNewFeatures.getJointFeatures().size());
		LOG.info("Model parse feature set. model with New Features %s",
							modelNewFeatures.getParseFeatures().size());
		
		return new Job(params.get("id"), new HashSet<>(params.getSplit("dep")),
				this, createJobOutputFile(params.get("id")),
				createJobLogFile(params.get("id"))) {
			
			@Override
			protected void doJob() {
				LOG.info("Going to bootstrap local ensemble of feed foward neural model");
				NeuralFeedForwardDotProductLearner.setFeatures(parser, modelNewFeatures);
				LOG.info("Done bootstrapping");
			}
		};
	}
	
	private Set<String> getVocab(IDataCollection<LabeledAmrSentence> data) {
		
		Set<String> vocab = new HashSet<String>();
		for(LabeledAmrSentence amrSentence: data) {
			vocab.addAll(amrSentence.getSample().getTokens().toList());
		}
		
		return vocab;
	}
	
	@SuppressWarnings("unused")
	private void saveVocab(IDataCollection<LabeledAmrSentence> train, IDataCollection<LabeledAmrSentence> test) {
		
		Set<String> vocab = new HashSet<String>();
		vocab.addAll(this.getVocab(train));
		LOG.info("Sentences %s gave %s", train.size(), vocab.size());
		
		Set<String> vocab2 = this.getVocab(test);
		vocab.addAll(vocab2);
		LOG.info("Sentences %s gave %s", test.size(), vocab2.size());
		
		LOG.info("Total Vocabular %s", vocab.size());
		
		PrintWriter writer;
		try {
			writer = new PrintWriter("amr_vocabulary.txt", "UTF-8");
			for(String word: vocab) {
				writer.println(word);
			}
			writer.flush();
			writer.close();
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (UnsupportedEncodingException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
	
	private Job createDataProcessing(Parameters params) throws FileNotFoundException {
		
		final CreateCompositeDecisionDataset<Sentence, SingleSentence, LogicalExpression> datasetCreator =
																get(params.get("datasetCreator"));
		
		final IJointModelImmutable<SituatedSentence<AMRMeta>, LogicalExpression, LogicalExpression> model =
																get(params.get("model"));
			
		return new Job(params.get("id"), new HashSet<>(params.getSplit("dep")),
				this, createJobOutputFile(params.get("id")),
				createJobLogFile(params.get("id"))) {
			
			@Override
			protected void doJob() {
				LOG.info("Going to initialize parameters for training data");
				datasetCreator.datasetCreatorInit(model);
				
				LOG.info("Parameters Initialized. \n Going to create composite dataset");
				datasetCreator.createDataset(model);
				LOG.info("Composite dataset created.");
			}
		};
	}
	
	private Job createMemoizeCategoryEmbeddingTask(Parameters params) throws FileNotFoundException {
		
		
		final NeuralNetworkShiftReduceParser<?, LogicalExpression> parser = get(params.get("parser"));
		
		final IJointModelImmutable<SituatedSentence<AMRMeta>, LogicalExpression, LogicalExpression> model =
																			get(params.get("model"));
		
		final IDataCollection<LabeledAmrSentence> testData = get(params.get("testData"));
		
		final ILexiconImmutable<LogicalExpression> lexicon = model.getLexicon();
		
		return new Job(params.get("id"), new HashSet<>(params.getSplit("dep")),
				this, createJobOutputFile(params.get("id")),
				createJobLogFile(params.get("id"))) {
			
			@Override
			protected void doJob() {
				LOG.info("Memoizing category embedding");
				long start = System.currentTimeMillis();
				parser.getCategoryEmbedding().memoizeLexicalEntryEmbedding(testData, lexicon);
				long end = System.currentTimeMillis();
				LOG.info("Done memoizing category embedding. Time taken %s", (end - start));
			}
		};
	}
	
	private Job createReadDataset(Parameters params) throws FileNotFoundException {
		
		final CreateCompositeDecisionDataset<Sentence, SingleSentence, LogicalExpression> datasetCreator =
																get(params.get("datasetCreator"));
		final String fileName = params.get("dataFile");
		
		return new Job(params.get("id"), new HashSet<>(params.getSplit("dep")),
				this, createJobOutputFile(params.get("id")),
				createJobLogFile(params.get("id"))) {
			
			@Override
			protected void doJob() {
				LOG.info("Going to read dataset");
				datasetCreator.readFromFile(fileName);
				LOG.info("Composite dataset created. Size %s", datasetCreator.getDataset().size());
			}
		};
	}
	
	/** job for training neural parser 
	 * @throws FileNotFoundException */
	private Job createLearnerJob(Parameters params) throws FileNotFoundException {
		
		final RNNShiftReduceLearner<Sentence, SingleSentence, LogicalExpression> learner = 
																get(params.get("learner"));
		
		final CreateCompositeDecisionDataset<Sentence, SingleSentence, LogicalExpression> datasetCreator =
																get(params.get("datasetCreator"));
		
		final IJointModelImmutable<SituatedSentence<AMRMeta>, LogicalExpression, LogicalExpression> model =
																get(params.get("model"));

		return new Job(params.get("id"), new HashSet<>(params.getSplit("dep")),
				this, createJobOutputFile(params.get("id")),
				createJobLogFile(params.get("id"))) {
			
			@Override
			protected void doJob() {
				LOG.info("Starting learn neural parser job");
				//List<CompositeDataPoint<LogicalExpression>> dataset = datasetCreator.getDataset();
				//learner.fitCompositeDataSet(dataset);
				datasetCreator.datasetCreatorInit(model);
				learner.fitCompositeDataSet/*TopLayer*/(datasetCreator, model);
				LOG.info("Learning over");
			}
		};
	}
	
	/** job for training neural parser 
	 * @throws FileNotFoundException */
	private Job createCatchEarlyParserJob(Parameters params) throws FileNotFoundException {
		
		
		final CreateSparseFeatureDataset<Sentence, SingleSentence, LogicalExpression> datasetCreator =
																get(params.get("datasetCreator"));
		
		final IJointModelImmutable<SituatedSentence<AMRMeta>, LogicalExpression, LogicalExpression> model =
																get(params.get("model"));
		
		final IDataCollection<LabeledAmrSentence> data = get(params.get("data"));
		
		final JointModel<SituatedSentence<AMRMeta>, LogicalExpression, LogicalExpression> modelNewFeatures =
																get(params.get("modelNewFeatures"));
		
		LOG.info("Size of model lexicon %s", modelNewFeatures.getLexicon().size());
		
		ILexiconImmutable<LogicalExpression> lexicon = model.getLexicon();
		modelNewFeatures.addLexEntries(lexicon.toCollection());
		
		LOG.info("Size of model lexicon now %s", modelNewFeatures.getLexicon().size());
		
		datasetCreator.modelNewFeatures = modelNewFeatures;
		
		return new Job(params.get("id"), new HashSet<>(params.getSplit("dep")),
				this, createJobOutputFile(params.get("id")),
				createJobLogFile(params.get("id"))) {
			
			@Override
			protected void doJob() {
				LOG.info("Catch early error");
				datasetCreator.catchEarlyErrorParser(data, model, 150);
				LOG.info("Learning over");
			}
		};
	}
	
	/** job for training neural parser 
	 * @throws FileNotFoundException */
	private Job createFeedForwardLearnerJob(Parameters params) throws FileNotFoundException {
		
		final NeuralFeedForwardDotProductLearner<Sentence, SingleSentence, LogicalExpression> learner = 
																get(params.get("learner"));
		
		final CreateSparseFeatureAndStateDataset/*WithExploration*/<Sentence, SingleSentence, LogicalExpression> datasetCreator =
																get(params.get("datasetCreator"));
		
		final IJointModelImmutable<SituatedSentence<AMRMeta>, LogicalExpression, LogicalExpression> model =
																get(params.get("model"));
		
		final JointModel<SituatedSentence<AMRMeta>, LogicalExpression, LogicalExpression> modelNewFeatures =
																get(params.get("modelNewFeatures"));
		
		LOG.info("Size of model lexicon %s", modelNewFeatures.getLexicon().size());
		
		ILexiconImmutable<LogicalExpression> lexicon = model.getLexicon();
		modelNewFeatures.addLexEntries(lexicon.toCollection());
		
		LOG.info("Size of model lexicon now %s", modelNewFeatures.getLexicon().size());
		LOG.info("Model features. model with New Features %s",
							modelNewFeatures.getJointFeatures().size());
		LOG.info("Model parse feature set. model with New Features %s",
							modelNewFeatures.getParseFeatures().size());

		datasetCreator.modelNewFeatures = modelNewFeatures;
		
		return new Job(params.get("id"), new HashSet<>(params.getSplit("dep")),
				this, createJobOutputFile(params.get("id")),
				createJobLogFile(params.get("id"))) {
			
			@Override
			protected void doJob() {
				LOG.info("Starting learn neural feed forward parser job");
				learner.fitCompositeDataSet(datasetCreator, model);
//				learner.tuneUnseenFeatures(datasetCreator, model);
				LOG.info("Learning over");
			}
		};
	}
	
}

