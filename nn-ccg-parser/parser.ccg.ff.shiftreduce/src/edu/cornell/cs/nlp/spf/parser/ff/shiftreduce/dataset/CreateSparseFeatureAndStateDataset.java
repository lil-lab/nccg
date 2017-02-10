package edu.cornell.cs.nlp.spf.parser.ff.shiftreduce.dataset;

import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;

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
import edu.cornell.cs.nlp.spf.parser.ccg.IParseStep;
import edu.cornell.cs.nlp.spf.parser.ccg.cky.single.CKYParser;
import edu.cornell.cs.nlp.spf.parser.ccg.rnn.shiftreduce.dataset.AbstractCreateDataset;
import edu.cornell.cs.nlp.spf.parser.ccg.rnn.shiftreduce.neuralnetworkparser.NeuralNetworkShiftReduceParser;
import edu.cornell.cs.nlp.spf.parser.ccg.rules.RuleName;
import edu.cornell.cs.nlp.spf.parser.ccg.shiftreduce.stacks.DerivationState;
import edu.cornell.cs.nlp.spf.parser.ccg.shiftreduce.stacks.DerivationStateVerticalIterator;
import edu.cornell.cs.nlp.spf.parser.ccg.shiftreduce.steps.IWeightedShiftReduceStep;
import edu.cornell.cs.nlp.spf.parser.ccg.shiftreduce.steps.ShiftReduceLexicalStep;
import edu.cornell.cs.nlp.spf.parser.ccg.shiftreduce.steps.ShiftReduceParseStep;
import edu.cornell.cs.nlp.spf.parser.ff.shiftreduce.neuralparser.NeuralDotProductShiftReduceParser;
import edu.cornell.cs.nlp.spf.parser.filter.IParsingFilterFactory;
import edu.cornell.cs.nlp.spf.parser.filter.StubFilterFactory;
import edu.cornell.cs.nlp.spf.parser.joint.IJointInferenceFilterFactory;
import edu.cornell.cs.nlp.utils.log.ILogger;
import edu.cornell.cs.nlp.utils.log.LogLevel;
import edu.cornell.cs.nlp.utils.log.LoggerFactory;
import edu.uw.cs.lil.amr.data.AMRMeta;
import edu.uw.cs.lil.amr.parser.GraphAmrParser;

public class CreateSparseFeatureAndStateDataset<SAMPLE extends IDataItem<?>, DI extends ILabeledDataItem<SAMPLE, ?>, MR> 
					extends AbstractCreateDataset<SparseFeatureAndStateDataset<MR>, SAMPLE, DI, MR> {
	
	public static final ILogger	LOG = LoggerFactory.create(CreateSparseFeatureAndStateDataset.class);
	
	private final boolean storeParseTreeLog;
	private Map<SituatedSentence<AMRMeta>, DerivationState<MR>> oldOne;
	private Map<SituatedSentence<AMRMeta>, DerivationState<MR>> newOne;
	
	public CreateSparseFeatureAndStateDataset(IDataCollection<DI> trainingData, 
			NeuralNetworkShiftReduceParser<Sentence, MR> parser, IValidator<DI,MR> validator, 
			Integer beamSize, IParsingFilterFactory<DI, MR> parsingFilterFactory, 
			CompositeImmutableLexicon<MR> compositeLexicon, ILexiconImmutable<MR> tempLexicon, 
			CKYParser<Sentence, MR> ckyParser) {
		
		super(trainingData, parser, validator, beamSize, parsingFilterFactory, compositeLexicon, tempLexicon,
				ckyParser);
		
		this.storeParseTreeLog = false;
		LOG.setCustomLevel(LogLevel.INFO);
	}
	
	public CreateSparseFeatureAndStateDataset(IDataCollection<DI> trainingData, 
			NeuralDotProductShiftReduceParser<Sentence, LogicalExpression> baseNeuralAmrParser,
			IValidator<DI,MR> validator, Integer beamSize, IParsingFilterFactory<DI, MR> parsingFilterFactory, 
			CompositeImmutableLexicon<MR> compositeLexicon, ILexiconImmutable<MR> tempLexicon, 
			GraphAmrParser amrOracleParser, 
			IJointInferenceFilterFactory<DI, LogicalExpression, LogicalExpression, LogicalExpression> amrSupervisedFilterFactory) {
		
		super(trainingData, baseNeuralAmrParser, validator, beamSize, parsingFilterFactory, compositeLexicon,
				tempLexicon, amrOracleParser, amrSupervisedFilterFactory);
	
		this.storeParseTreeLog = false;
		LOG.setCustomLevel(LogLevel.INFO);
	}
	
	/** creates pre-processed datapoints from the given sentence and parseTree*/
	@Override
	protected List<SparseFeatureAndStateDataset<MR>> preProcessDataPoints(Sentence dataItemSample, 
															DerivationState<MR> parseTree) {
		throw new RuntimeException("Operated Not Supported.");
	}
	
	/** creates pre-processed datapoints from the given sentence and parseTree*/
	@Override
	protected List<SparseFeatureAndStateDataset<MR>> preProcessDataPoints(SituatedSentence<AMRMeta> dataItemSample, 
															DerivationState<MR> parseTree) {
		
		final TokenSeq tk = dataItemSample.getTokens();
		List<SparseFeatureAndStateDataset<MR>> sparseFeatureAndStateDataset = 
										new LinkedList<SparseFeatureAndStateDataset<MR>>(); 
	
		if(this.storeParseTreeLog) {
			if(this.newOne == null) {
				this.oldOne = new HashMap<SituatedSentence<AMRMeta>, DerivationState<MR>>();
				this.newOne = new HashMap<SituatedSentence<AMRMeta>, DerivationState<MR>>();
			}
			
			this.newOne.put(dataItemSample, parseTree);
		}
		
		final int n = tk.size();
		
		final List<ParsingOp<MR>> allActionsReversed = parseTree.returnParsingOps();
		for(ParsingOp<MR> op: allActionsReversed) {
			LOG.info("> %s", op);
		}
				
		int words = n; //words left to be consumed
		DerivationState<MR> dstate = parseTree;
		
		while(dstate.getParent() != null) { //at the end of this loop, it is init (null parent, left and right category)
			IWeightedShiftReduceStep<MR> step = dstate.returnStep();
			RuleName ruleName = step.getRuleName();
			
			if(ruleName.equals(ShiftReduceLexicalStep.LEXICAL_DERIVATION_STEP_RULENAME)) {
				int numWordsConsumed = step.getEnd() - step.getStart();
				words = words - numWordsConsumed;
			}
			
			//list of possible actions which are cached in step 1 of learner
			List<ParsingOp<MR>> possibleActions = dstate.possibleActions();
			List<IHashVector> possibleActionFeatures = dstate.possibleActionFeatures();
			
			if(possibleActions.size() == 0) {
				throw new IllegalStateException("Found a pre-processed point with 0 possible actions.");
			}
			
			//ground truth action
			final int gTruthIx = this.findGroundTruthIx(possibleActions, step.getUnderlyingParseStep());

			final IHashVector stateFeature = dstate.stateFeature();
			if(stateFeature == null) {
				throw new RuntimeException("State feature is null. Parent is " + dstate.getParent());
			}
			
			//parent of the dstate is the state from which dstate is created
			dstate = dstate.getParent();//it.next();
			
			SparseFeatureAndStateDataset<MR> pt = new SparseFeatureAndStateDataset<MR>(stateFeature, 
									possibleActionFeatures, gTruthIx, tk.toString(), possibleActions);
			pt.setSemantics((DerivationState<LogicalExpression>) dstate);
			LOG.debug("Pt is %s", pt);
			
			if(possibleActions.size() > 1) {
				sparseFeatureAndStateDataset.add(pt);
			}
		}
		
		assert words == 0 : "processing the entire parse tree did not exhaust the words. Its a bug!! Found "+words;
		
		return sparseFeatureAndStateDataset;
	}
	
	public List<SparseFeatureAndStateDataset<MR>> getDataset() {
		return super.getDataset();
	}
	
	private boolean checkEqual(DerivationState<MR> dstate1, DerivationState<MR> dstate2) {
		
		DerivationStateVerticalIterator<MR> vit1 = dstate1.verticalIterator();
		DerivationStateVerticalIterator<MR> vit2 = dstate2.verticalIterator();
		
		while(vit1.hasNext()) {
			if(!vit2.hasNext()) {
				return false;
			}
			
			DerivationState<MR> d1 = vit1.next();
			DerivationState<MR> d2 = vit2.next();
			
			if(d1.returnStep() == null && d2.returnStep() != null) {
				return false;
			}
			
			if(d1.returnStep() != null && d2.returnStep() == null) {
				return false;
			}
			
			if(d1.returnStep() == null && d2.returnStep() == null) {
				continue;
			}
			
			//We are interested in the step. Since if steps are same then they are same
			IParseStep<MR> step1 = d1.returnStep().getUnderlyingParseStep();
			IParseStep<MR> step2 = d2.returnStep().getUnderlyingParseStep();
			
			if(step1 instanceof ShiftReduceLexicalStep) {
				if(step2 instanceof ShiftReduceLexicalStep) {
					
					if(!((ShiftReduceLexicalStep<MR>)step1).equals((ShiftReduceLexicalStep<MR>)step2)) {
						return false;
					}
					
				} else {
					return false;
				}
			} else if(step1 instanceof ShiftReduceParseStep) {
				if(step2 instanceof ShiftReduceParseStep) {
					
					if(!((ShiftReduceParseStep<MR>)step1).equals((ShiftReduceParseStep<MR>)step2)) {
						return false;
					}
					
				} else {
					return false;
				}
			} else {
				throw new RuntimeException("None of shift reduce lexical step");
			}
		}
		
		if(vit2.hasNext()) {
			return false;
		}
		
		return true;
	}
	
	// Called at the end of an epoch, puts trees for a data item into the old map
	// while also reporting how many sentences have same trees as they had in last epoch
	// this is useful in measuring the convergence of parse trees.
	public void swap() {
		
		int matched =  0;
		if(this.oldOne.size() > 0) { //initially its empty
			
			if(this.oldOne.size() != this.newOne.size()) {
				LOG.warn("Sizes dont match. This is a bug. Exiting."); 
				System.exit(0);
			}
			
			for(Entry<SituatedSentence<AMRMeta>, DerivationState<MR>> e: this.newOne.entrySet()) {
				DerivationState<MR> oldTree = this.oldOne.get(e.getKey());
				
				if(this.checkEqual(oldTree, e.getValue())) {
					matched++;
				} else {
					LOG.info("Sentence gets a new tree %s", e.getKey().getSample().getString());
				}
			}
			
			LOG.info("Number of datapoints with same tree %s out of %s", matched, this.oldOne.size());
		}
		
		this.oldOne.clear();
		this.oldOne.putAll(this.newOne);
		this.newOne.clear();
	}
	
	public static class Creator<SAMPLE extends IDataItem<?>, DI extends ILabeledDataItem<SAMPLE, ?>, MR> 
				implements IResourceObjectCreator<CreateSparseFeatureAndStateDataset<SAMPLE, DI, MR>> {

		private final String type;

		public Creator() {
			this("data.neural.sparsefeatureandstate");
		}

		public Creator(String type) {
			this.type = type;
		}

		@Override
		public CreateSparseFeatureAndStateDataset<SAMPLE, DI, MR> create(Parameters params, IResourceRepository repo) {

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
			
			final IJointInferenceFilterFactory<DI, LogicalExpression, LogicalExpression, LogicalExpression> 
			amrSupervisedFilterFactory = repo.get(params.get("filterFactory")); 
			
			GraphAmrParser graphAmrParser = repo.get(params.get("graphAmrParser"));
			
			return new CreateSparseFeatureAndStateDataset<SAMPLE, DI, MR>(trainingData, parser, validator, beamSize, parsingFilterFactory, 
											compositeLexicon, tempLexicon, graphAmrParser, amrSupervisedFilterFactory);
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
