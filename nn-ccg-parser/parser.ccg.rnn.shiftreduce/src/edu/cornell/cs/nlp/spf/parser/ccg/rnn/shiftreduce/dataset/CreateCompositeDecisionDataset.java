package edu.cornell.cs.nlp.spf.parser.ccg.rnn.shiftreduce.dataset;

import java.util.LinkedList;
import java.util.List;

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
import edu.cornell.cs.nlp.spf.parser.ccg.rnn.shiftreduce.neuralnetworkparser.NeuralNetworkShiftReduceParser;
import edu.cornell.cs.nlp.spf.parser.ccg.rules.RuleName;
import edu.cornell.cs.nlp.spf.parser.ccg.shiftreduce.stacks.DerivationState;
import edu.cornell.cs.nlp.spf.parser.ccg.shiftreduce.stacks.DerivationStateVerticalIterator;
import edu.cornell.cs.nlp.spf.parser.ccg.shiftreduce.steps.IWeightedShiftReduceStep;
import edu.cornell.cs.nlp.spf.parser.ccg.shiftreduce.steps.ShiftReduceLexicalStep;
import edu.cornell.cs.nlp.spf.parser.filter.IParsingFilterFactory;
import edu.cornell.cs.nlp.spf.parser.filter.StubFilterFactory;
import edu.cornell.cs.nlp.spf.parser.joint.IJointInferenceFilterFactory;
import edu.cornell.cs.nlp.utils.composites.Pair;
import edu.cornell.cs.nlp.utils.log.ILogger;
import edu.cornell.cs.nlp.utils.log.LogLevel;
import edu.cornell.cs.nlp.utils.log.LoggerFactory;
import edu.uw.cs.lil.amr.data.AMRMeta;
import edu.uw.cs.lil.amr.parser.GraphAmrParser;

/** Neural Parser Training Algorithm for CCG Semantic Parsing 
 * @author Dipendra Misra (dkm@cs.cornell.edu)
 * */
public class CreateCompositeDecisionDataset<SAMPLE extends IDataItem<?>, DI extends ILabeledDataItem<SAMPLE, ?>, MR> 
			extends AbstractCreateDataset<CompositeDataPoint<MR>, SAMPLE, DI, MR> {
	
	public static final ILogger	LOG = LoggerFactory.create(CreateCompositeDecisionDataset.class);
	
	public CreateCompositeDecisionDataset(IDataCollection<DI> trainingData, 
			NeuralNetworkShiftReduceParser<Sentence, MR> parser, IValidator<DI,MR> validator, 
			Integer beamSize, IParsingFilterFactory<DI, MR> parsingFilterFactory, 
			CompositeImmutableLexicon<MR> compositeLexicon, ILexiconImmutable<MR> tempLexicon, 
			CKYParser<Sentence, MR> ckyParser) {
		
		super(trainingData, parser, validator, beamSize, parsingFilterFactory, compositeLexicon, tempLexicon,
				ckyParser);
		LOG.setCustomLevel(LogLevel.INFO);
	}
	
	public CreateCompositeDecisionDataset(IDataCollection<DI> trainingData, 
			NeuralNetworkShiftReduceParser<Sentence, LogicalExpression> baseNeuralAmrParser,
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
	protected List<CompositeDataPoint<MR>> preProcessDataPoints(Sentence dataItemSample, 
															DerivationState<MR> parseTree) {
		throw new RuntimeException("Operated Not Supported.");
	}
	
	/** creates pre-processed datapoints from the given sentence and parseTree*/
	@Override
	protected List<CompositeDataPoint<MR>> preProcessDataPoints(SituatedSentence<AMRMeta> dataItemSample, 
															DerivationState<MR> parseTree) {
		
		final List<CompositeDataPoint<MR>> compositeDataSet = new LinkedList<CompositeDataPoint<MR>>();
		
		final TokenSeq tk = dataItemSample.getSample().getTokens();
		final List<String> sentence = tk.toList(); //list of words
		final List<String> tags = dataItemSample.getState().getTags().toList(); //list of tags
		
		//buffer contains sentence in reverse along with tags
		final List<Pair<String, String>> buffer = new LinkedList<Pair<String, String>>();
		for(int i = sentence.size() - 1; i >= 0; i--) {
			buffer.add(Pair.of(sentence.get(i), tags.get(i)));
		}
		
		final List<ParsingOp<MR>> allActionsReversed = parseTree.returnParsingOps();
		final List<ParsingOp<MR>> allActions = new LinkedList<ParsingOp<MR>>();
		for(int i = allActionsReversed.size() - 1; i >= 0; i--) {
			allActions.add(allActionsReversed.get(i));
		}
		
		for(ParsingOp<MR> op: allActions) {
			LOG.info("Operation %s", op);
		}

		final int n = tk.size();
		int words = n;
		int ctr = allActions.size() - 1;
		
		DerivationStateVerticalIterator<MR> it = parseTree.verticalIterator();
		
		if(!it.hasNext())
			throw new IllegalStateException("parse tree is empty"); //parse tree is empty
		
		DerivationState<MR> dstate = it.next();
		
		CompositeDataPoint.Builder<MR> builder = new CompositeDataPoint.Builder<MR>(tk.toString(), tk, dstate, allActions, buffer);
		int stateIx = 0;
		
		while(it.hasNext()) { //at the end of this loop, it is init (null parent, left and right category)
			IWeightedShiftReduceStep<MR> step = dstate.returnStep();
			RuleName ruleName = step.getRuleName();
			
			if(ruleName.equals(ShiftReduceLexicalStep.LEXICAL_DERIVATION_STEP_RULENAME)) {
				int numWordsConsumed = step.getEnd() - step.getStart();
				words = words - numWordsConsumed;
				stateIx++;
			}
			
			//list of possible actions which are cached in step 1 of learner
			List<ParsingOp<MR>> possibleActions = dstate.possibleActions();
			
			if(possibleActions.size() == 0) {
				throw new IllegalStateException("Found a pre-processed point with 0 possible actions.");
			}
			
			//ground truth action
			final int gTruthIx = this.findGroundTruthIx(possibleActions, step.getUnderlyingParseStep());
		    
			//parent of the dstate is the state from which dstate is created
			dstate = it.next();
			
			//branch change therefore change the builder
			if(!ruleName.equals(ShiftReduceLexicalStep.LEXICAL_DERIVATION_STEP_RULENAME)) { 
				if(builder.numDecision() > 0) {
					compositeDataSet.add(builder.build(true));
				}
				
				//can change buffer and all actions in future too to make them smaller.
				builder = new CompositeDataPoint.Builder<MR>(tk.toString(), tk, dstate, allActions, buffer);
				stateIx = 0;
			}
			
			CompositeDataPointDecision<MR> decision = new CompositeDataPointDecision<MR>(possibleActions, gTruthIx, n - words, 
																							ctr, builder.numCategories() - stateIx);
			
			ctr--;
			if(possibleActions.size() > 1) { //non-trivial decision has more than one options
				builder.addDecision(decision);
			}
		}
		
		if(builder.numDecision() > 0) {
			compositeDataSet.add(builder.build(true));
		}
		
		assert words == 0 : "processing the entire parse tree did not exhaust the words. Its a bug!! Found "+words;
		assert ctr == -1 : "should have finished all the rules. Its a bug!!! Found "+ctr;
		
		return compositeDataSet;
	}
	
	public List<CompositeDataPoint<MR>> getDataset() {
		return super.getDataset();
	}
	
	public static class Creator<SAMPLE extends IDataItem<?>, DI extends ILabeledDataItem<SAMPLE, ?>, MR> 
							implements IResourceObjectCreator<CreateCompositeDecisionDataset<SAMPLE, DI, MR>> {

		private final String type;
		
		public Creator() {
			this("data.neural.composite");
		}

		public Creator(String type) {
			this.type = type;
		}

		@Override
		public CreateCompositeDecisionDataset<SAMPLE, DI, MR> create(Parameters params, IResourceRepository repo) {
			
			IDataCollection<DI> trainingData = repo.get(params.get("trainingData"));
			
			NeuralNetworkShiftReduceParser<Sentence, LogicalExpression> parser = repo.get(params.get("baseParser"));
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
			
			return new CreateCompositeDecisionDataset<SAMPLE, DI, MR>(trainingData, parser, validator, beamSize, parsingFilterFactory, 
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