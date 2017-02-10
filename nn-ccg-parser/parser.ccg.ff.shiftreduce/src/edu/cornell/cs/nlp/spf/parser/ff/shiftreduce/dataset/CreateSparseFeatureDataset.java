package edu.cornell.cs.nlp.spf.parser.ff.shiftreduce.dataset;

import java.util.LinkedList;
import java.util.List;

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
import edu.cornell.cs.nlp.spf.parser.ccg.cky.single.CKYParser;
import edu.cornell.cs.nlp.spf.parser.ccg.rnn.shiftreduce.dataset.AbstractCreateDataset;
import edu.cornell.cs.nlp.spf.parser.ccg.rnn.shiftreduce.neuralnetworkparser.NeuralNetworkShiftReduceParser;
import edu.cornell.cs.nlp.spf.parser.ccg.rules.RuleName;
import edu.cornell.cs.nlp.spf.parser.ccg.shiftreduce.stacks.DerivationState;
import edu.cornell.cs.nlp.spf.parser.ccg.shiftreduce.stacks.DerivationStateVerticalIterator;
import edu.cornell.cs.nlp.spf.parser.ccg.shiftreduce.steps.IWeightedShiftReduceStep;
import edu.cornell.cs.nlp.spf.parser.ccg.shiftreduce.steps.ShiftReduceLexicalStep;
import edu.cornell.cs.nlp.spf.parser.ff.shiftreduce.neuralparser.NeuralShiftReduceParser;
import edu.cornell.cs.nlp.spf.parser.filter.IParsingFilterFactory;
import edu.cornell.cs.nlp.spf.parser.filter.StubFilterFactory;
import edu.cornell.cs.nlp.spf.parser.joint.IJointInferenceFilterFactory;
import edu.cornell.cs.nlp.utils.log.ILogger;
import edu.cornell.cs.nlp.utils.log.LogLevel;
import edu.cornell.cs.nlp.utils.log.LoggerFactory;
import edu.uw.cs.lil.amr.data.AMRMeta;
import edu.uw.cs.lil.amr.parser.GraphAmrParser;

public class CreateSparseFeatureDataset<SAMPLE extends IDataItem<?>, DI extends ILabeledDataItem<SAMPLE, ?>, MR> 
					extends AbstractCreateDataset<SparseFeatureDataset<MR>, SAMPLE, DI, MR> {
	
	public static final ILogger	LOG = LoggerFactory.create(CreateSparseFeatureAndStateDataset.class);
	
	public CreateSparseFeatureDataset(IDataCollection<DI> trainingData, 
			NeuralNetworkShiftReduceParser<Sentence, MR> parser, IValidator<DI,MR> validator, 
			Integer beamSize, IParsingFilterFactory<DI, MR> parsingFilterFactory, 
			CompositeImmutableLexicon<MR> compositeLexicon, ILexiconImmutable<MR> tempLexicon, 
			CKYParser<Sentence, MR> ckyParser) {
		
		super(trainingData, parser, validator, beamSize, parsingFilterFactory, compositeLexicon, tempLexicon,
				ckyParser);
		LOG.setCustomLevel(LogLevel.INFO);
	}
	
	public CreateSparseFeatureDataset(IDataCollection<DI> trainingData, 
			NeuralShiftReduceParser<Sentence, LogicalExpression> baseNeuralAmrParser,
			IValidator<DI,MR> validator, Integer beamSize, IParsingFilterFactory<DI, MR> parsingFilterFactory, 
			CompositeImmutableLexicon<MR> compositeLexicon, ILexiconImmutable<MR> tempLexicon, 
			GraphAmrParser amrOracleParser, 
			IJointInferenceFilterFactory<DI, LogicalExpression, LogicalExpression, LogicalExpression> amrSupervisedFilterFactory) {
		
		super(trainingData, baseNeuralAmrParser, validator, beamSize, parsingFilterFactory, compositeLexicon,
				tempLexicon, amrOracleParser, amrSupervisedFilterFactory);
		LOG.setCustomLevel(LogLevel.INFO);
	}
	
	/** creates pre-processed datapoints from the given sentence and parseTree*/
	@Override
	protected List<SparseFeatureDataset<MR>> preProcessDataPoints(Sentence dataItemSample, 
															DerivationState<MR> parseTree) {
		throw new RuntimeException("Operated Not Supported.");
	}
	
	/** creates pre-processed datapoints from the given sentence and parseTree*/
	@Override
	protected List<SparseFeatureDataset<MR>> preProcessDataPoints(SituatedSentence<AMRMeta> dataItemSample, 
															DerivationState<MR> parseTree) {
		
		final TokenSeq tk = dataItemSample.getTokens();
		List<SparseFeatureDataset<MR>> sparseFeatureDataset = new LinkedList<SparseFeatureDataset<MR>>(); 
	
		final int n = tk.size();
		
		final List<ParsingOp<MR>> allActionsReversed = parseTree.returnParsingOps();
		for(ParsingOp<MR> op: allActionsReversed) {
			LOG.info("> %s", op);
		}
		
		DerivationStateVerticalIterator<MR> it = parseTree.verticalIterator();

		if(!it.hasNext()) {
			throw new IllegalStateException("parse tree is empty"); //parse tree is empty
		}
		
		int words = n; //words left to be consumed
		DerivationState<MR> dstate = it.next();
		
		while(it.hasNext()) { //at the end of this loop, it is init (null parent, left and right category)
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

			//parent of the dstate is the state from which dstate is created
			dstate = it.next();
			
			SparseFeatureDataset<MR> pt = new SparseFeatureDataset<MR>(possibleActionFeatures, gTruthIx, 
																	   tk.toString(), possibleActions);
			LOG.debug("Pt is %s", pt);
			
			if(possibleActions.size() > 1) {
				sparseFeatureDataset.add(pt);
			}
		}
		
		assert words == 0 : "processing the entire parse tree did not exhaust the words. Its a bug!! Found "+words;
		
		return sparseFeatureDataset;
	}
	
	public List<SparseFeatureDataset<MR>> getDataset() {
		return super.getDataset();
	}
	
	public static class Creator<SAMPLE extends IDataItem<?>, DI extends ILabeledDataItem<SAMPLE, ?>, MR> 
				implements IResourceObjectCreator<CreateSparseFeatureDataset<SAMPLE, DI, MR>> {

		private final String type;

		public Creator() {
			this("data.neural.sparsefeature");
		}

		public Creator(String type) {
			this.type = type;
		}

		@Override
		public CreateSparseFeatureDataset<SAMPLE, DI, MR> create(Parameters params, IResourceRepository repo) {

			IDataCollection<DI> trainingData = repo.get(params.get("trainingData"));
			
			NeuralShiftReduceParser<Sentence, LogicalExpression> parser = repo.get(params.get("baseParser"));
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
			
			return new CreateSparseFeatureDataset<SAMPLE, DI, MR>(trainingData, parser, validator, beamSize, parsingFilterFactory, 
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
