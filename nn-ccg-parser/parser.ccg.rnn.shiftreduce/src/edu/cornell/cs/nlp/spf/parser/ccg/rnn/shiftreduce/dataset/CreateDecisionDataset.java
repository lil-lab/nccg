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
import edu.cornell.cs.nlp.spf.parser.ParsingOp;
import edu.cornell.cs.nlp.spf.parser.ccg.cky.single.CKYParser;
import edu.cornell.cs.nlp.spf.parser.ccg.rnn.shiftreduce.neuralnetworkparser.NeuralNetworkShiftReduceParser;
import edu.cornell.cs.nlp.spf.parser.ccg.rules.RuleName;
import edu.cornell.cs.nlp.spf.parser.ccg.shiftreduce.stacks.DerivationState;
import edu.cornell.cs.nlp.spf.parser.ccg.shiftreduce.stacks.DerivationStateVerticalIterator;
import edu.cornell.cs.nlp.spf.parser.ccg.shiftreduce.steps.IWeightedShiftReduceStep;
import edu.cornell.cs.nlp.spf.parser.ccg.shiftreduce.steps.ShiftReduceLexicalStep;
import edu.cornell.cs.nlp.spf.parser.filter.IParsingFilterFactory;
import edu.cornell.cs.nlp.utils.log.ILogger;
import edu.cornell.cs.nlp.utils.log.LogLevel;
import edu.cornell.cs.nlp.utils.log.LoggerFactory;
import edu.uw.cs.lil.amr.data.AMRMeta;

/** Neural Parser Training Algorithm for CCG Semantic Parsing 
 * @author Dipendra Misra (dkm@cs.cornell.edu)
 * */
public class CreateDecisionDataset<SAMPLE extends IDataItem<?>, DI extends ILabeledDataItem<SAMPLE, ?>, MR> 
					extends AbstractCreateDataset<ProcessedDataSet<MR>, SAMPLE, DI, MR>{
	
	public static final ILogger	LOG = LoggerFactory.create(CreateDecisionDataset.class);
	
	public CreateDecisionDataset(IDataCollection<DI> trainingData, 
			NeuralNetworkShiftReduceParser<Sentence, MR> parser, IValidator<DI,MR> validator,  
			Integer beamSize, IParsingFilterFactory<DI, MR> parsingFilterFactory, 
			CompositeImmutableLexicon<MR> compositeLexicon, ILexiconImmutable<MR> tempLexicon, 
			CKYParser<Sentence, MR> ckyParser) {
		
		super(trainingData, parser, validator, beamSize, parsingFilterFactory, compositeLexicon, tempLexicon,
				ckyParser);
		LOG.setCustomLevel(LogLevel.INFO);
	}
	
	/** creates pre-processed datapoints from the given sentence and parseTree*/
	protected List<ProcessedDataSet<MR>> preProcessDataPoints(Sentence dataItemSample, 
															DerivationState<MR> parseTree) {
		
		final TokenSeq tk = dataItemSample.getTokens();
		List<ProcessedDataSet<MR>> processedDataSet = new LinkedList<ProcessedDataSet<MR>>(); 
	
		final List<String> sentence = tk.toList(); //list of words
		final List<ParsingOp<MR>> allActionsReversed = parseTree.returnParsingOps();
		final List<ParsingOp<MR>> allActions = new LinkedList<ParsingOp<MR>>();
		for(int i = allActionsReversed.size()-1; i>=0; i--) {
			allActions.add(allActionsReversed.get(i));
		}
		
		for(ParsingOp<MR> op: allActions) {
			LOG.info("Operation %s", op);
		}
		
		final int n = tk.size();
		
		DerivationStateVerticalIterator<MR> it = parseTree.verticalIterator();

		int words = n;
		int ctr = allActions.size() - 1;
		
		if(!it.hasNext())
			throw new IllegalStateException("parse tree is empty"); //parse tree is empty
		
		DerivationState<MR> dstate = it.next();
		
		while(it.hasNext()) { //at the end of this loop, it is init (null parent, left and right category)
			IWeightedShiftReduceStep<MR> step = dstate.returnStep();
			RuleName ruleName = step.getRuleName();
			
			if(ruleName.equals(ShiftReduceLexicalStep.LEXICAL_DERIVATION_STEP_RULENAME)) {
				int numWordsConsumed = step.getEnd() - step.getStart();
				words = words - numWordsConsumed; //No true --- a lexical entry can contain many words 
			}
			
			//list of possible actions which are cached in step 1 of learner
			List<ParsingOp<MR>> possibleActions = dstate.possibleActions();
			
			if(possibleActions.size() == 0) {
				throw new IllegalStateException("Found a pre-processed point with 0 possible actions.");
			}
			
			//ground truth action
			final int gTruthIx = this.findGroundTruthIx(possibleActions, step.getUnderlyingParseStep());
		    
			List<String> bufferReverse = sentence.subList(words, n);
			List<String> buffer = new LinkedList<String>();
			for(int j = 0; j < bufferReverse.size(); j++) {
				buffer.add(bufferReverse.get(bufferReverse.size() - j - 1));
			}
			
			List<ParsingOp<MR>> history = allActions.subList(0, ctr);
			
			//parent of the dstate is the state from which dstate is created
			dstate = it.next();
			
			ProcessedDataSet<MR> pt = new ProcessedDataSet<MR>(dstate, history, buffer,  
														possibleActions, gTruthIx, sentence, tk);
			LOG.debug("Pt is %s", pt);
			
			ctr--;
			if(possibleActions.size() > 1) {
				processedDataSet.add(pt);
			}
		}
		
		assert words == 0 : "processing the entire parse tree did not exhaust the words. Its a bug!! Found "+words;
		assert ctr == -1 : "should have finished all the rules. Its a bug!!! Found "+ctr;
		
		return processedDataSet;
	}

	@Override
	protected List<ProcessedDataSet<MR>> preProcessDataPoints(SituatedSentence<AMRMeta> situatedSentence,
			DerivationState<MR> parseTree) {
		throw new RuntimeException("Operation Not Supported.");
	}
}