package edu.cornell.cs.nlp.spf.parser.ccg.rnn.shiftreduce.neuralnetworkparser;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;
import java.util.Spliterator;
import java.util.Spliterators;
import java.util.stream.StreamSupport;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import edu.cornell.cs.nlp.spf.base.token.TokenSeq;
import edu.cornell.cs.nlp.spf.ccg.categories.Category;
import edu.cornell.cs.nlp.spf.ccg.categories.ICategoryServices;
import edu.cornell.cs.nlp.spf.ccg.lexicon.CompositeImmutableLexicon;
import edu.cornell.cs.nlp.spf.ccg.lexicon.ILexiconImmutable;
import edu.cornell.cs.nlp.spf.ccg.lexicon.LexicalEntry;
import edu.cornell.cs.nlp.spf.ccg.lexicon.Lexicon;
import edu.cornell.cs.nlp.spf.data.sentence.Sentence;
import edu.cornell.cs.nlp.spf.parser.ISentenceLexiconGenerator;
import edu.cornell.cs.nlp.spf.parser.ParsingOp;
import edu.cornell.cs.nlp.spf.parser.ccg.model.IDataItemModel;
import edu.cornell.cs.nlp.spf.parser.ccg.rnn.shiftreduce.learning.ParsingOpPreTrainingDataset;
import edu.cornell.cs.nlp.spf.parser.ccg.rules.ILexicalRule;
import edu.cornell.cs.nlp.spf.parser.ccg.rules.ParseRuleResult;
import edu.cornell.cs.nlp.spf.parser.ccg.rules.RuleName;
import edu.cornell.cs.nlp.spf.parser.ccg.rules.SentenceSpan;
import edu.cornell.cs.nlp.spf.parser.ccg.shiftreduce.ShiftReduceBinaryParsingRule;
import edu.cornell.cs.nlp.spf.parser.ccg.shiftreduce.ShiftReduceUnaryParsingRule;
import edu.cornell.cs.nlp.spf.parser.ccg.shiftreduce.stacks.DerivationState;
import edu.cornell.cs.nlp.spf.parser.ccg.shiftreduce.stacks.PackedState;
import edu.cornell.cs.nlp.spf.parser.ccg.shiftreduce.stacks.ShiftReduceRuleNameSet;
import edu.cornell.cs.nlp.spf.parser.ccg.shiftreduce.steps.ShiftReduceLexicalStep;
import edu.cornell.cs.nlp.utils.collections.queue.DirectAccessBoundedPriorityQueue;
import edu.cornell.cs.nlp.utils.filter.IFilter;

public class CreatePreTrainingData<DI extends Sentence, MR> extends NeuralNetworkShiftReduceParser<DI, MR> {
	
	private static final long serialVersionUID = 7597466419507036090L;
	private final int numRules;
	private final INDArray[] binaryRulesVectors;
	
	public CreatePreTrainingData(int beamSize, ShiftReduceBinaryParsingRule<MR>[] binaryRules, ILexicalRule<MR> lexicalRule, 
			List<ISentenceLexiconGenerator<DI, MR>> sentenceLexiconGenerators,
			List<ISentenceLexiconGenerator<DI, MR>> sloppyLexicalGenerators, ICategoryServices<MR> categoryServices,
			IFilter<Category<MR>> completeParseFilter, ShiftReduceUnaryParsingRule<MR>[] unaryRules) {
		super(beamSize, binaryRules, lexicalRule, sentenceLexiconGenerators, sloppyLexicalGenerators, categoryServices, completeParseFilter,
				unaryRules, 0.1, 0.1, 0.000001, 5.0, 1234);
		
		int pad = 1 + this.unaryRules.length; 
		this.numRules = 1 + this.unaryRules.length + this.binaryRules.length;
		this.binaryRulesVectors = new INDArray[this.binaryRules.length];
		
		for(int i =0; i< this.binaryRules.length; i++) {
			INDArray oneHot = Nd4j.zeros(this.numRules);
			oneHot.putScalar(pad + i, 1.0);
			this.binaryRulesVectors[i] = oneHot; 
		}
	}

	/** Parses a sentence using Neural Network model */
	public List<ParsingOpPreTrainingDataset<MR>> createPreTrainingData(DI dataItem, 
			IFilter<ParsingOp<MR>> pruningFilter, IDataItemModel<MR> model, 
			boolean allowWordSkipping, ILexiconImmutable<MR> tempLexicon, Integer beamSize_) {
		
		List<ParsingOpPreTrainingDataset<MR>> dataset = new 
												LinkedList<ParsingOpPreTrainingDataset<MR>>();
		
		if(beamSize_ == null)
			beamSize_ = 100;
		
		final Integer beamSize = beamSize_; //declare final for the loop
		
		Comparator<PackedState<MR>> dStateCmp  = new Comparator<PackedState<MR>>() {
			public int compare(PackedState<MR> left, PackedState<MR> right) {
        		return Double.compare(left.getBestScore(), right.getBestScore()); 
    		}   
		};
		
		TokenSeq tk = dataItem.getTokens();
		int n = tk.size(); //number of tokens
		
		List<DirectAccessBoundedPriorityQueue<PackedState<MR>>> beam = new 
						LinkedList<DirectAccessBoundedPriorityQueue<PackedState<MR>>>();
		List<DirectAccessBoundedPriorityQueue<PackedState<MR>>> newBeam = new 
						LinkedList<DirectAccessBoundedPriorityQueue<PackedState<MR>>>();
		
		for(int i=0; i<=n; i++) { //a beam for different number of words consumed 
			beam.add(new DirectAccessBoundedPriorityQueue<PackedState<MR>>(beamSize, dStateCmp));
			newBeam.add(new DirectAccessBoundedPriorityQueue<PackedState<MR>>(beamSize, dStateCmp));
		}
		
		PackedState<MR> init_ =  new PackedState<MR>(new DerivationState<MR>());
		beam.get(0).offer(init_);
		
		// Create the list of active lexicons
		final List<ILexiconImmutable<MR>> lexicons = new ArrayList<ILexiconImmutable<MR>>();

		// Lexicon for sloppy inference.
		if (allowWordSkipping) {
			boolean createdSloppyEntries = false;
			for (final ISentenceLexiconGenerator<DI, MR> generator : sloppyLexicalGenerators) {
				final Lexicon<MR> sloppyLexicon = new Lexicon<MR>(
						generator.generateLexicon(dataItem));
				if (sloppyLexicon.size() != 0) {
					createdSloppyEntries = true;
				}
				lexicons.add(sloppyLexicon);
			}
			if (!createdSloppyEntries) {
				LOG.warn("Sloppy inference but no sloppy entries created -- verify the parser is setup to allow sloppy inference");
			}
		}

		// Lexicon with heuristically generated lexical entries. The entries are
		// generated given the string of the sentence.
		for (final ISentenceLexiconGenerator<DI, MR> generator : sentenceLexiconGenerators) {
			lexicons.add(new Lexicon<MR>(generator.generateLexicon(dataItem)));
		}

		// The model lexicon
		lexicons.add(model.getLexicon());

		// If there's a temporary lexicon, add it too
		if (tempLexicon != null) {
			lexicons.add(tempLexicon);
		}
		
		final CompositeImmutableLexicon<MR> compositeLexicon = new CompositeImmutableLexicon<MR>(lexicons);
		boolean isEmpty = false;
		int cycle = 0;
		
		while(!isEmpty) {
			LOG.debug("=========== CYCLE %s =============", ++cycle);
			Iterator<DirectAccessBoundedPriorityQueue<PackedState<MR>>> iterBeam = beam.iterator();
			int ibj = 0;
					
			while(iterBeam.hasNext()) {
				LOG.debug("### Working on the beam %s ###", ++ibj);
				final DirectAccessBoundedPriorityQueue<PackedState<MR>> pstates = iterBeam.next();
					
				StreamSupport.stream(Spliterators.spliterator(pstates, Spliterator.IMMUTABLE), 
									true/*LOG.getLogLevel() == LogLevel.DEBUG ? false : true*/)
					    .forEach(pstate -> { 
					/* perform valid shift and reduce operations for this packed states but 
					 * computes no score. The aim is to simply create training data for pre-training */
					
			    	DerivationState<MR> dstate = pstate.getBestState();
					int wordsConsumed = dstate.wordsConsumed;
					
					//list of new potential states
					List<DerivationState<MR>> options = new LinkedList<DerivationState<MR>>();
					
					//Operation 1: Shift operation: shift a token and its lexical entry to this stack
					if(wordsConsumed < n) {
						
						for(int words = 1; words <= n - dstate.wordsConsumed; words++) {
							Iterator<? extends LexicalEntry<MR>> lexicalEntries = 
									compositeLexicon.get(tk.sub(dstate.wordsConsumed, dstate.wordsConsumed+words));
							
							while(lexicalEntries.hasNext()) {
								LexicalEntry<MR> lexicalEntry = lexicalEntries.next();
								
								SentenceSpan span = new SentenceSpan(wordsConsumed, wordsConsumed + words, n);
								
								ParsingOp<MR> op = new ParsingOp<MR>(lexicalEntry.getCategory(), span, ShiftReduceLexicalStep.LEXICAL_DERIVATION_STEP_RULENAME);
								
								if(pruningFilter != null && !pruningFilter.test(op)) { 
									continue;
								}
								
								DerivationState<MR> dNew = dstate.shift(lexicalEntry, words, span);
								options.add(dNew);
							}
						}
					}
					
					
					//Operation 2: Unary Reduce operation
					final ShiftReduceRuleNameSet<MR> lastNonTerminal = dstate.returnLastNonTerminal();
					final SentenceSpan lastSpan = dstate.returnLastSentenceSpan();
					
					if(lastNonTerminal != null) {
						if(!dstate.isUnary()) { //cannot apply two consecutive unary rules 
							for(int uj = 0; uj < this.unaryRules.length; uj++) {
								RuleName name = this.unaryRules[uj].getName();
								ParseRuleResult<MR> logical  = this.applyUnaryRule(uj, lastNonTerminal, lastSpan);
								
								if(logical != null) {
									
									SentenceSpan lastSpan_ = new SentenceSpan(lastSpan.getStart(), lastSpan.getEnd(), n);
									ParsingOp<MR> op = new ParsingOp<MR>(logical.getResultCategory(), lastSpan_, name);
									
									if(pruningFilter != null && !pruningFilter.test(op)) {
										continue;
									}
									
									DerivationState<MR> dNew =	dstate.reduceUnaryRule(name, logical.getResultCategory());
									options.add(dNew);									
								}
							}
						}
						
						ShiftReduceRuleNameSet<MR> last2ndLastNonTerminal = dstate.return2ndLastNonTerminal();
						SentenceSpan sndLastSpan = dstate.return2ndLastSentenceSpan();
						
						if(last2ndLastNonTerminal != null) {
							//Operation 3: Binary Reduce operation
							SentenceSpan joined = new SentenceSpan(sndLastSpan.getStart(), lastSpan.getEnd(), n);
							for(int bj = 0; bj < this.binaryRules.length; bj++) {
								RuleName name = this.binaryRules[bj].getName();
								
								ParseRuleResult<MR> logical  = this.applyBinaryRule(bj, last2ndLastNonTerminal, 
																lastNonTerminal, joined);
								boolean label = false;
								if(logical!= null) {
									
									SentenceSpan joined_ = new SentenceSpan(sndLastSpan.getStart(), lastSpan.getEnd(), n);
									ParsingOp<MR> op = new ParsingOp<MR>(logical.getResultCategory(), joined_, name);
									
									if(pruningFilter != null && !pruningFilter.test(op)) {
										continue;
									}
									
									DerivationState<MR> dNew = dstate.reduceBinaryRule(name, 
															logical.getResultCategory(), joined);
									options.add(dNew);
									label = true;
								}
								
								ParsingOpPreTrainingDataset<MR> point = new
										ParsingOpPreTrainingDataset<MR>(last2ndLastNonTerminal.getCategory(),
										lastNonTerminal.getCategory(), this.binaryRulesVectors[bj], label);
								dataset.add(point);
							}
						}
					}
					
					//normalize the probabilities and add them to the list
					Iterator<DerivationState<MR>> it = options.iterator();
					
					while(it.hasNext()) {
						DerivationState<MR> dNew = it.next();
						dNew.score = 0;
						
						boolean full = dNew.lenRoot() == 1 && n == dNew.wordsConsumed;
						
						if(!full) {
							PackedState<MR> pstateNew = new PackedState<MR>(dNew);
							synchronized(newBeam) {
								this.push(newBeam.get(0), pstateNew, dNew, beamSize);
							}
						}
					}
				}); 
			}
			
			
			Iterator<DirectAccessBoundedPriorityQueue<PackedState<MR>>> beamIter = beam.iterator();
			Iterator<DirectAccessBoundedPriorityQueue<PackedState<MR>>> nBeamIter = newBeam.iterator();
			
			isEmpty = true;
			
			while(nBeamIter.hasNext()) {
				DirectAccessBoundedPriorityQueue<PackedState<MR>> nBeam_ = nBeamIter.next();
				assert beamIter.hasNext();
				DirectAccessBoundedPriorityQueue<PackedState<MR>> beam_ = beamIter.next();
				beam_.clear(); //clear the current stack
				
				Iterator<PackedState<MR>> iter = nBeam_.iterator();
			
				while(iter.hasNext()) {
					PackedState<MR> ds_ = iter.next();  
					beam_.offer(ds_);
					isEmpty = false;
				}
				nBeam_.clear(); //clear the new stack
			}
		}
		
		return dataset;
	}
	
}
