package edu.cornell.cs.nlp.spf.parser.ff.shiftreduce.dataset; 

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;

import com.google.common.base.Joiner;
import java.util.Random;
import java.util.Set;
import java.util.concurrent.atomic.AtomicInteger;

import edu.cornell.cs.nlp.spf.base.hashvector.IHashVector;
import edu.cornell.cs.nlp.spf.base.token.TokenSeq;
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
import edu.cornell.cs.nlp.spf.parser.ccg.rnn.shiftreduce.dataset.CreateCompositeDecisionDataset;
import edu.cornell.cs.nlp.spf.parser.ccg.rules.RuleName;
import edu.cornell.cs.nlp.spf.parser.ccg.shiftreduce.ShiftReduceDerivation;
import edu.cornell.cs.nlp.spf.parser.ccg.shiftreduce.stacks.DerivationState;
import edu.cornell.cs.nlp.spf.parser.ccg.shiftreduce.steps.IWeightedShiftReduceStep;
import edu.cornell.cs.nlp.spf.parser.ccg.shiftreduce.steps.ShiftReduceLexicalStep;
import edu.cornell.cs.nlp.spf.parser.ff.shiftreduce.neuralparser.NeuralDotProductShiftReduceParser;
import edu.cornell.cs.nlp.spf.parser.filter.IParsingFilterFactory;
import edu.cornell.cs.nlp.spf.parser.filter.StubFilterFactory;
import edu.cornell.cs.nlp.spf.parser.joint.IJointInferenceFilterFactory;
import edu.cornell.cs.nlp.utils.log.ILogger;
import edu.cornell.cs.nlp.utils.log.LogLevel;
import edu.cornell.cs.nlp.utils.log.LoggerFactory;
import edu.uw.cs.lil.amr.data.AMRMeta;
import edu.uw.cs.lil.amr.parser.GraphAmrParser;

public class CreateSparseFeatureAndStateDatasetWithExploration<SAMPLE extends IDataItem<?>, DI extends ILabeledDataItem<SAMPLE, ?>, MR> 
					extends AbstractCreateDatasetWithExploration<SparseFeatureAndStateDataset<MR>, SAMPLE, DI, MR> {
	
	public static final ILogger	LOG = LoggerFactory.create(CreateCompositeDecisionDataset.class);
	private final Random rnd;
	private final double p;
	private final int k;
	private final AtomicInteger mismatch, numExploration;
	
	public CreateSparseFeatureAndStateDatasetWithExploration(IDataCollection<DI> trainingData, 
			NeuralDotProductShiftReduceParser<Sentence, LogicalExpression> baseNeuralAmrParser,
			IValidator<DI,MR> validator, Integer beamSize, IParsingFilterFactory<DI, MR> parsingFilterFactory, 
			CompositeImmutableLexicon<MR> compositeLexicon, ILexiconImmutable<MR> tempLexicon, 
			GraphAmrParser amrOracleParser, 
			IJointInferenceFilterFactory<DI, LogicalExpression, LogicalExpression, LogicalExpression> amrSupervisedFilterFactory, 
			double p, int k) {
		
		super(trainingData, baseNeuralAmrParser, validator, beamSize, parsingFilterFactory, compositeLexicon,
				tempLexicon, amrOracleParser, amrSupervisedFilterFactory);
		LOG.setCustomLevel(LogLevel.INFO);
		this.rnd = new Random();
		this.p = p;//0.5;
		this.k = k;//3;
		this.mismatch = new AtomicInteger();
		this.numExploration = new AtomicInteger();
		
		LOG.setCustomLevel(LogLevel.INFO);
		
		LOG.info("Create Sparse Feature And State Dataset with Exploration. p = %s, k = %s", this.p, this.k);
	}
	
	/** Select the next parse tree based on policy. */
	private int selectParseTreeStep(List<DerivationState<MR>>[] parseTrees, int[] parseTreesLen, Set<Integer> options, 
									int currentParseTree, int currentStep, int epoch) {
		
		// Current policy: If epoch <= 2 then use the current Parse tree which is initialized as viterbi
		// else find the parse tree with the highest scoring parse step involved in the transition. 
		// In case of ties, we go to parse tree with the highest score thus taking the optimal decision
		// given the current configuration. If this is same as the viterbi tree then stay with the viterbi
		// tree else with p probability
		// move to this new tree and with 1-p stay with the current tree.
		
		if(epoch <= this.k) {
			return currentParseTree;
		}
		
		if(options.size() == 1) {
			return options.iterator().next();
		}
		
		double maximumScoringStep = Double.NEGATIVE_INFINITY;
		int treeWithMaximumScoringStep = -1;
		
		for(int option: options) {
			
			final int up = parseTreesLen[option] - currentStep - 2;
			DerivationState<MR> dstate = parseTrees[option].get(up);
			double stepScore = dstate.score - dstate.getParent().score;
			
			if(stepScore > maximumScoringStep) {
				maximumScoringStep = stepScore;
				treeWithMaximumScoringStep = option;
			} else if(stepScore == maximumScoringStep) {
				if(parseTrees[option].get(0).score > parseTrees[treeWithMaximumScoringStep].get(0).score) {
					treeWithMaximumScoringStep = option;
				}
			}
		}
		
		if(treeWithMaximumScoringStep != currentParseTree) {
			this.mismatch.incrementAndGet();
		}
		
		double prob = this.rnd.nextDouble();
		if(prob > this.p) { 
			return currentParseTree;
		} else {
			
			if(treeWithMaximumScoringStep != currentParseTree) {
				this.numExploration.incrementAndGet();
			}
			
			return treeWithMaximumScoringStep;
		}
	}
	
	/** Select the next parse tree based on policy. */
	@SuppressWarnings("unused")
	private int selectParseTreeStep1(List<DerivationState<MR>>[] parseTrees, Set<Integer> options, 
									int currentParseTree, int currentStep, int epoch) {
		
		// Current policy: If epoch <= 2 then use the current Parse tree which is initialized as viterbi
		// else go to an option parse tree based on its score. For this we convert their score into a distribution.
		
		if(epoch <= 2) {
			return currentParseTree;
		}
		
		if(options.size() == 1) {
			return options.iterator().next();
		}
		
		Map<Integer, Double> prob = new HashMap<Integer, Double>();
		double sum = 0, minScore =  Double.POSITIVE_INFINITY;
		
		for(Integer option: options) {
			double score = parseTrees[option].get(0).score;
			if(score < minScore) {
				minScore = score;
			}
			sum = sum + score;
			prob.put(option, score);
		}
		
		double Z = sum - options.size() * minScore;
		if(Z == 0) {
			Z = 0.00001;
		}
		
		double prob_ = 0;
		for(Entry<Integer, Double> e: prob.entrySet()) {
			double eProb = (e.getValue() - minScore)/Z;
			prob_ = prob_ + eProb;
			prob.put(e.getKey(), eProb);
		}
		
		LOG.debug("Prob_ %s", prob_);
		
		double select = this.rnd.nextDouble();
		double cummulative = 0;
		
		for(Entry<Integer, Double> e: prob.entrySet()) {
			double eProb = e.getValue();
			cummulative = cummulative + eProb;
			
			if(select < cummulative) {
				LOG.debug("Returning %s", e.getKey());
				return e.getKey();
			}
		}
		
		LOG.debug("Unlikely case. This should happen very very rarely due to numerical issue. Select %s, cummulative %s, Z %s", 
								select, cummulative, Z);
		LOG.debug("Current returning %s", currentParseTree);
		return currentParseTree;
	}
	
	private void logAllTrees(List<DerivationState<MR>> parseTrees) {
		
		int counter = 0;
		for(DerivationState<MR> parseTree: parseTrees) {
		
			DerivationState<MR> it = parseTree;
			LOG.debug("{ parse tree %s, score %s:", ++counter, it.score);
			while(it.returnStep() != null) {
				IWeightedShiftReduceStep<MR> step = it.returnStep();
				LOG.debug("[%s-%s; %s %s %s]", step.getStart(), step.getEnd(), it.score - it.getParent().score, 
												step.getRuleName(), step.getRoot());
				it = it.getParent();
			}
			LOG.debug("}");
		}
	}
	
	@Override
	protected List<SparseFeatureAndStateDataset<MR>> preProcessDataPointsWithExploration(
			SituatedSentence<AMRMeta> situatedSentence, ShiftReduceDerivation<MR> derivations, int epoch) {
		
		final TokenSeq tk = situatedSentence.getTokens();
		final String tkString = tk.toString();
		
		List<SparseFeatureAndStateDataset<MR>> dataset = new ArrayList<SparseFeatureAndStateDataset<MR>>();
		List<DerivationState<MR>> parseTrees = derivations.getAllDerivationStates();
		final int numParseTrees = parseTrees.size();
		
		if(LOG.getLogLevel().equals(LogLevel.DEBUG)) {
			this.logAllTrees(parseTrees);
		}
		
		//Preprocessing. Creates a datastructure that stores parse trees as list of derivation state in reverse order
		//Also create a datastructure for storing length of each parse tree
		@SuppressWarnings("unchecked")
		final List<DerivationState<MR>>[] treeDecisions = new List[numParseTrees];
		final int[] parseTreesLen = new int[numParseTrees]; //number of steps in a parse tree
		
		Iterator<DerivationState<MR>> treesIt = parseTrees.iterator();
		int treeCounter = 0;
		
		double maxParseTreeScore =  Double.NEGATIVE_INFINITY;
		int maxParseTreeIx = 0;
		
		while(treesIt.hasNext()) {
			DerivationState<MR> treeIt = treesIt.next();
			
			if(treeIt.score > maxParseTreeScore) {
				maxParseTreeScore = treeIt.score;
				maxParseTreeIx = treeCounter;
			}
			
			List<DerivationState<MR>> decisions = new ArrayList<DerivationState<MR>>();
			int numStep = 0;
			
			while(treeIt != null) {
				decisions.add(treeIt);
				treeIt = treeIt.getParent();
				numStep++;
			}
			
			treeDecisions[treeCounter] = decisions;
			parseTreesLen[treeCounter] = numStep;
			
			treeCounter++;
		}
		
		Set<Integer> options = new HashSet<Integer>();
		
		for(int i = 0; i < numParseTrees; i++) {
			options.add(i);
		}
		
		////// DEBUGGING //////
		DerivationState<MR> starting = treeDecisions[0].get(parseTreesLen[0] - 1);
		for(int i = 1; i < numParseTrees; i++) {
			if(treeDecisions[i].get(parseTreesLen[i] - 1) != starting) {
				throw new RuntimeException("Last of all parse trees should be same. There is a bug somewhere.");
			}
		}
		///////////////////////
		
		//Maximum scoring parse tree
		final int viterbiParseTree = maxParseTreeIx;
		
		LOG.info("viterbiParseTree %s, score %s, best scoring tree %s", viterbiParseTree, treeDecisions[viterbiParseTree].get(0).score, 
				derivations.getScore());
		
		LOG.debug("Viterbi tree %s, options len %s, options { %s }", viterbiParseTree, 
						options.size(), Joiner.on(",").join(options));
		
		//Current parse tree represents the best achievable tree given the current configuration
		int currentParseTree = viterbiParseTree;
		int currentStep = 0;
		
		//at a given time calculates the number of words still left to be consumed
		int words = tk.size();
		
		while(options.size() != 1 || currentStep != parseTreesLen[options.iterator().next()] - 1) { //TODO check this
			
			//Pick a step using policy. This is done by first selecting
			//which parse tree to jump from the current parse tree.
			
			final int nextParseTree = this.selectParseTreeStep(treeDecisions, parseTreesLen, options, currentParseTree, currentStep, epoch);
			
			LOG.debug("Next Parse Tree %s, step %s, options { %s }", nextParseTree, currentStep, Joiner.on(",").join(options));
			
			//Find choosing which option from current dstate results in jumping to the next parse tree.
			//This step maybe sub-optimal and the set of parse trees that result can be (and generally should be) 
			//more than just the nextParseTree.
			
			//go back such that there are only n step left from the start
			final int up = parseTreesLen[currentParseTree] - currentStep - 2;
			
			//dstate calculated below represents the state of new parse tree one step ahead from the current parse tree
			//dstate and current parse tree are identical except possibly for the last step
			final DerivationState<MR> dstate = treeDecisions[currentParseTree].get(up); 
			
			LOG.debug("Up %s, next tree %s, dstate %s %s %s", up, currentParseTree, dstate, 
													dstate.hashCode(), dstate.getDebugHashCode());
			
			IWeightedShiftReduceStep<MR> step = dstate.returnStep();
			List<ParsingOp<MR>> possibleActions = dstate.possibleActions();
			final IHashVector stateFeature = dstate.stateFeature();
			List<IHashVector> possibleActionFeatures = dstate.possibleActionFeatures();
			
			LOG.debug("[%s-%s; %s %s]", step.getStart(), step.getEnd(), step.getRuleName(), step.getRoot());
			
			//Option from which this step was created
			final int optionIx = this.findGroundTruthIx(possibleActions, step.getUnderlyingParseStep());
			
			if(possibleActions.size() == 0) {
				throw new IllegalStateException("Found a pre-processed point with 0 possible actions.");
			}
			
			SparseFeatureAndStateDataset<MR> pt = new SparseFeatureAndStateDataset<MR>(stateFeature, possibleActionFeatures,
																						optionIx, tkString, possibleActions);
			LOG.debug("Created Point %s", pt);
			
			//Add to the dataset if its a non-trivial decision
			if(possibleActions.size() > 1) { 
				dataset.add(pt);
			}
			
			//Find next state to jump to
			final int newDStateUp =  parseTreesLen[nextParseTree] - currentStep - 2;
			final DerivationState<MR> newDState = treeDecisions[nextParseTree].get(newDStateUp);
			
			//consume words --- debugging purpose only
			final IWeightedShiftReduceStep<MR> newStep = newDState.returnStep();
			RuleName ruleName = newStep.getRuleName();
			
			if(ruleName.equals(ShiftReduceLexicalStep.LEXICAL_DERIVATION_STEP_RULENAME)) {
				int numWordsConsumed = newStep.getEnd() - newStep.getStart();
				words = words - numWordsConsumed;
			}
			
			//Update options
			Set<Integer> newOptions = new HashSet<Integer>();
			for(int option: options) {
				int optionUp = parseTreesLen[option] - currentStep - 2;
				
				if(treeDecisions[option].size() <= optionUp) {
					continue;
				}
				
				DerivationState<MR> optionDState = treeDecisions[option].get(optionUp); 
				
				if(optionDState == newDState) {
					newOptions.add(option);
				}
			}
			options = newOptions;
			if(newOptions.size() == 0) {
				throw new RuntimeException("There should be atleast one option.");
			}
			
			//Update
			currentParseTree = nextParseTree;
			currentStep++;
		}
		
		if(words != 0) {
			throw new RuntimeException("All words not consumed. This looks like a bug!" + words);
		}
		
		LOG.info("Exploration step %s, mismatch %s", this.numExploration.get(), this.mismatch.get());
		
		return dataset;
	}
		
	public List<SparseFeatureAndStateDataset<MR>> getDataset() {
		return super.getDataset();
	}
	
	public static class Creator<SAMPLE extends IDataItem<?>, DI extends ILabeledDataItem<SAMPLE, ?>, MR> 
				implements IResourceObjectCreator<CreateSparseFeatureAndStateDatasetWithExploration<SAMPLE, DI, MR>> {

		private final String type;

		public Creator() {
			this("data.neural.sparsefeatureandstate.exploration");
		}

		public Creator(String type) {
			this.type = type;
		}

		@Override
		public CreateSparseFeatureAndStateDatasetWithExploration<SAMPLE, DI, MR> create(Parameters params, IResourceRepository repo) {

			IDataCollection<DI> trainingData = repo.get(params.get("trainingData"));
			
			NeuralDotProductShiftReduceParser<Sentence, LogicalExpression> parser = repo.get(params.get("baseParser"));
			IValidator<DI,MR> validator = null;
			Integer beamSize = params.getAsInteger("beamSize");
			
			final IParsingFilterFactory<DI, MR> parsingFilterFactory;
			if(params.contains("parsingFilterFactory")) {
				parsingFilterFactory = repo.get(params.get("parsingFilterFactory")); 
			} else {
				parsingFilterFactory = new StubFilterFactory<>();	//TODO fix this in future
			}
			
			final CompositeImmutableLexicon<MR> compositeLexicon;
			if(params.contains("compositeLexicon")) {
				compositeLexicon = repo.get(params.get("compositeLexicon")); 
			} else {
				compositeLexicon = null;	
			}
			
			final ILexiconImmutable<MR> tempLexicon;
			if(params.contains("tempLexicon")) {
				tempLexicon = repo.get(params.get("tempLexicon"));
			} else {
				tempLexicon = null;
			}
			
			final double p = params.getAsDouble("p");
			final int k = params.getAsInteger("k");
			
			final IJointInferenceFilterFactory<DI, LogicalExpression, LogicalExpression, LogicalExpression> 
			amrSupervisedFilterFactory = repo.get(params.get("filterFactory")); 
			
			GraphAmrParser graphAmrParser = repo.get(params.get("graphAmrParser"));
			
			return new CreateSparseFeatureAndStateDatasetWithExploration<SAMPLE, DI, MR>(trainingData, parser, validator, beamSize, 
						parsingFilterFactory, compositeLexicon, tempLexicon, graphAmrParser, amrSupervisedFilterFactory, p, k);
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