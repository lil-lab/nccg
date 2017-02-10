/*******************************************************************************
 * Copyright (C) 2011 - 2015 Yoav Artzi, All rights reserved.
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
 *******************************************************************************/
package edu.cornell.cs.nlp.spf.parser.ccg.shiftreduce;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;
import java.util.ListIterator;
import java.util.Spliterator;
import java.util.Spliterators;
import java.util.concurrent.atomic.AtomicLong;
import java.util.function.Predicate;
import java.util.stream.StreamSupport;

import com.google.common.base.Function;

import edu.cornell.cs.nlp.spf.base.token.TokenSeq;
import edu.cornell.cs.nlp.spf.ccg.categories.Category;
import edu.cornell.cs.nlp.spf.ccg.categories.ICategoryServices;
import edu.cornell.cs.nlp.spf.ccg.lexicon.CompositeImmutableLexicon;
import edu.cornell.cs.nlp.spf.ccg.lexicon.ILexiconImmutable;
import edu.cornell.cs.nlp.spf.ccg.lexicon.LexicalEntry;
import edu.cornell.cs.nlp.spf.ccg.lexicon.Lexicon;
import edu.cornell.cs.nlp.spf.data.sentence.Sentence;
import edu.cornell.cs.nlp.spf.parser.IParser;
import edu.cornell.cs.nlp.spf.parser.IParserOutput;
import edu.cornell.cs.nlp.spf.parser.ISentenceLexiconGenerator;
import edu.cornell.cs.nlp.spf.parser.ParsingOp;
import edu.cornell.cs.nlp.spf.parser.ccg.model.IDataItemModel;
import edu.cornell.cs.nlp.spf.parser.ccg.rules.ILexicalRule;
import edu.cornell.cs.nlp.spf.parser.ccg.rules.ParseRuleResult;
import edu.cornell.cs.nlp.spf.parser.ccg.rules.RuleName;
import edu.cornell.cs.nlp.spf.parser.ccg.rules.SentenceSpan;
import edu.cornell.cs.nlp.spf.parser.ccg.shiftreduce.stacks.DerivationState;
import edu.cornell.cs.nlp.spf.parser.ccg.shiftreduce.stacks.PackedState;
import edu.cornell.cs.nlp.spf.parser.ccg.shiftreduce.stacks.ShiftReduceRuleNameSet;
import edu.cornell.cs.nlp.spf.parser.ccg.shiftreduce.steps.ShiftReduceLexicalStep;
import edu.cornell.cs.nlp.spf.parser.ccg.shiftreduce.steps.ShiftReduceParseStep;
import edu.cornell.cs.nlp.spf.parser.ccg.shiftreduce.steps.WeightedShiftReduceLexicalStep;
import edu.cornell.cs.nlp.spf.parser.ccg.shiftreduce.steps.WeightedShiftReduceParseStep;
import edu.cornell.cs.nlp.utils.collections.queue.DirectAccessBoundedPriorityQueue;
import edu.cornell.cs.nlp.utils.composites.Pair;
import edu.cornell.cs.nlp.utils.filter.IFilter;
import edu.cornell.cs.nlp.utils.log.ILogger;
import edu.cornell.cs.nlp.utils.log.LogLevel;
import edu.cornell.cs.nlp.utils.log.LoggerFactory;

/**
* Shift Reduce Parser: Implements the shift reduce parser
*
* @author Dipendra K. Misra
* @author Yoav Artzi
*/
public abstract class AbstractShiftReduceParser<DI extends Sentence, MR> implements
		IParser<DI, MR> {
	public static final ILogger								LOG					= LoggerFactory
																						.create(AbstractShiftReduceParser.class);

	private static final long								serialVersionUID	= -1141905985877531704L;

	/**
	 * The maximum number of cells to hold for each span.
	 */
	private final int										beamSize;

	/**
	 * Binary CCG parsing rules.
	 */
	public final ShiftReduceBinaryParsingRule<MR>[]			binaryRules;

	private final IFilter<Category<MR>>						completeParseFilter;

	/**
	 * List of lexical generators that use the sentence itself to generate
	 * lexical entries.
	 */
	protected final List<ISentenceLexiconGenerator<DI, MR>>	sentenceLexiconGenerators; //it was private earlier

	/**
	 * Lexical generators to create lexical entries for sloppy inference.
	 */
	protected final List<ISentenceLexiconGenerator<DI, MR>>	sloppyLexicalGenerators; //it was private earlier

	public final ShiftReduceUnaryParsingRule<MR>[]			unaryRules;

	protected final ICategoryServices<MR>					categoryServices;

	protected AbstractShiftReduceParser(int beamSize,
			ShiftReduceBinaryParsingRule<MR>[] binaryRules,
			List<ISentenceLexiconGenerator<DI, MR>> sentenceLexiconGenerators,
			List<ISentenceLexiconGenerator<DI, MR>> sloppyLexicalGenerators,
			ICategoryServices<MR> categoryServices, boolean pruneLexicalCells,
			IFilter<Category<MR>> completeParseFilter,
			ShiftReduceUnaryParsingRule<MR>[] unaryRules,
			Function<Category<MR>, Category<MR>> categoryTransformation,
			ILexicalRule<MR> lexicalRule, boolean breakTies) {
		this.beamSize = beamSize;
		this.binaryRules = binaryRules;
		this.sentenceLexiconGenerators = sentenceLexiconGenerators;
		this.sloppyLexicalGenerators = sloppyLexicalGenerators;
		this.categoryServices = categoryServices;
		this.completeParseFilter = completeParseFilter;
		this.unaryRules = unaryRules;
		LOG.info("Init :: %s: pruneLexicalCells=%s beamSize=%d ...",
				getClass(), pruneLexicalCells, beamSize);
		LOG.info("Init :: %s: ... sloppyLexicalGenerator=%s ...", getClass(),
				sloppyLexicalGenerators);
		LOG.info("Init :: %s: ... binary rules=%s ...", getClass(),
				Arrays.toString(binaryRules));
		LOG.info("Init :: %s: ... unary rules=%s ...", getClass(),
				Arrays.toString(unaryRules));
		LOG.info("Init :: %s: ... lexical rule=%s ...", getClass(), lexicalRule);
		LOG.info("Init :: %s: ... breakTies=%s", getClass(), breakTies);
	}
	
	protected ParseRuleResult<MR> applyUnaryRule(int ruleindex, ShiftReduceRuleNameSet<MR> ruleNameSet,
												SentenceSpan span) {
		/** applies the rule on a logical form and returns the resultant logical form **/
		
		return this.unaryRules[ruleindex].apply(ruleNameSet, span);
	}
	
	protected ParseRuleResult<MR> applyBinaryRule(int ruleindex, ShiftReduceRuleNameSet<MR> left,
									ShiftReduceRuleNameSet<MR> right, SentenceSpan span) {
		/** applies the rule on a logical form and returns the resultant logical form **/
		return this.binaryRules[ruleindex].apply(left, right, span);
	}
	
	/** WeighedLexicalStep computation is very expensive, therefore the algorithm 
	 * precomputes these score one time, prior to the actual parsing processes.
	 * This can be done, since they only depending upon the lexical entry i.e. they
	 * are local in nature */
	private ArrayList<List<Pair<WeightedShiftReduceLexicalStep<MR>, LexicalEntry<MR>>>> cmpLexicalStepScore(
			TokenSeq tk, IDataItemModel<MR> model, CompositeImmutableLexicon<MR> compositeLexicon) {
		
		int n = tk.size(); //number of tokens
		
		ArrayList<List<Pair<WeightedShiftReduceLexicalStep<MR>, LexicalEntry<MR>>>> lexSteps = new ArrayList<List<Pair<WeightedShiftReduceLexicalStep<MR>, LexicalEntry<MR>>>>(n);
		
		for(int i=0; i < n; i++) {
			List<Pair<WeightedShiftReduceLexicalStep<MR>, LexicalEntry<MR>>> lexStep = new  LinkedList<Pair<WeightedShiftReduceLexicalStep<MR>, LexicalEntry<MR>>>();
			
			for(int words = 1; words <= n-i; words++) {
				Iterator<? extends LexicalEntry<MR>> lexicalEntries = compositeLexicon.get(tk.sub(i, i+words));
				while(lexicalEntries.hasNext()) {
					LexicalEntry<MR> lexicalEntry = lexicalEntries.next();
					ShiftReduceLexicalStep<MR> lexicalStep = new ShiftReduceLexicalStep<MR>(lexicalEntry.getCategory(),
							lexicalEntry, (n == words), i, i + words);
					
					WeightedShiftReduceLexicalStep<MR> wLexicalStep = new WeightedShiftReduceLexicalStep<MR>(lexicalStep, model);
					lexStep.add(Pair.of(wLexicalStep, lexicalEntry));
				}
			}
			lexSteps.add(lexStep);
		}
		
		return lexSteps;
	}
	
	public boolean push(DirectAccessBoundedPriorityQueue<PackedState<MR>> queue,
						 DerivationState<MR> elem, int capacity) {
		PackedState<MR> pstate = new PackedState<MR>(elem);
		PackedState<MR> prior = queue.get(pstate);
		
		if(prior ==  null) { //safe to push
			/* Introduces randomization, due to threading and ties for the lowest score 
			 * state. In order to prevent this, disable threads. To introduce determinism
			 * later: when removing a state, remove all the states with the same score. */
			queue.offer(pstate); 
			return false; 
		}
		else {
			queue.remove(prior);
			prior.add(elem); 
			queue.add(prior);
			return true; //true indicates that it was packed
		}
	}
	
	
	/** Shift Reduce Parsing Algorithm: 
	 * The algorithm maintains a Stacks<MR> which contains derivation state
	 * and the number of words consumed by that stack. The algorithm performs
	 * iteration until there is a stack which can accept more words. At each such
	 * iteration, the algorithm generates a list of possible derivation states
	 * using shift-reduce rules. These new derivation states are then ranked and the 
	 * top k choices define our new stack.
	 */
	public IParserOutput<MR> parse(DI dataItem, Predicate<ParsingOp<MR>> pruningFilter,
			IDataItemModel<MR> model, boolean allowWordSkipping,
			ILexiconImmutable<MR> tempLexicon, Integer beamSize_) {
		
		long start = System.currentTimeMillis();
		
		if(beamSize_ == null)
			beamSize_ = this.beamSize;
		
		final Integer beamSize = beamSize_; //declare final for the loop
		
		Comparator<PackedState<MR>> dStateCmp  = new Comparator<PackedState<MR>>() {
			public int compare(PackedState<MR> left, PackedState<MR> right) {
        		return Double.compare(left.getBestScore(), right.getBestScore()); 
    		}   
		};
		
		final AtomicLong packedParse = new AtomicLong();
		
		LOG.debug("Utterance: %s", dataItem);
		
		TokenSeq tk = dataItem.getTokens();
		int n = tk.size(); //number of tokens
		
		List<DerivationState<MR>> completeParseTrees = new LinkedList<DerivationState<MR>>();
		
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
		
		final ArrayList<List<Pair<WeightedShiftReduceLexicalStep<MR>, LexicalEntry<MR>>>> lexicalSteps = this.cmpLexicalStepScore(tk, model, 
																			compositeLexicon);
		
		while(!isEmpty) {
			LOG.debug("=========== CYCLE %s =============", ++cycle);
			Iterator<DirectAccessBoundedPriorityQueue<PackedState<MR>>> iterBeam = beam.iterator();
			int ibj = 0;
			
			while(iterBeam.hasNext()) {
				LOG.debug("### Working on the beam %s ###", ++ibj);
				final DirectAccessBoundedPriorityQueue<PackedState<MR>> pstates = iterBeam.next();
				
				StreamSupport.stream(Spliterators.spliterator(pstates, Spliterator.IMMUTABLE), 
						LOG.getLogLevel() == LogLevel.DEBUG ? false : true)
					    .forEach(pstate -> { 
					/* perform valid shift and reduce operations for this packed states, if the packed state is
					 * finished and already represents a complete parse tree then save it separately. */
					
					/* Important: Since features are currently local that is only look at the root categories or 
					 * at the shifted lexical entries. Hence, operations can be performed on the best state
					 * currently in the packed state. This will NO LONGER HOLD if features start looking 
					 * at the complete tree segments in the state.*/
					    	
					DerivationState<MR> dstate = pstate.getBestState();
					int wordsConsumed = dstate.wordsConsumed;
					
					//Operation 1: Shift operation: shift a token and its lexical entry to this stack
					if(wordsConsumed < n) {
						 //returns all lexical entries for sentence span starting with wordsConsumed+1
						Iterator<Pair<WeightedShiftReduceLexicalStep<MR>, LexicalEntry<MR>>> iterLexSteps = 
											lexicalSteps.get(wordsConsumed).iterator();
						
						while(iterLexSteps.hasNext()) {
							Pair<WeightedShiftReduceLexicalStep<MR>, LexicalEntry<MR>> step_ = iterLexSteps.next();
							LexicalEntry<MR> lexicalEntry = step_.second();
							int words = lexicalEntry.getTokens().size();
							SentenceSpan span = new SentenceSpan(wordsConsumed, wordsConsumed + words, n);
							
							if(pruningFilter != null) { 
								SentenceSpan span_ =  new SentenceSpan(span.getStart(), span.getEnd(), 
																		span.getEnd() - span.getStart());
								ParsingOp<MR> op = new ParsingOp<MR>(lexicalEntry.getCategory(), span_, ShiftReduceLexicalStep.LEXICAL_DERIVATION_STEP_RULENAME);
								if(!pruningFilter.test(op))
									continue;
							}
							
							DerivationState<MR> dNew = dstate.shift(lexicalEntry, words, span);
							
							WeightedShiftReduceLexicalStep<MR> wLexicalStep = step_.first();
 							dNew.score = dstate.score + wLexicalStep.getStepScore();
							dNew.defineStep(wLexicalStep);
							dNew.calcDebugHashCode();
							
							boolean full = n == words && dNew.lenRoot() == 1 && this.completeParseFilter.test(
																dNew.returnLastNonTerminal().getCategory()); //double check if this line has all condition, added dNew.lenRoot() == 1
							
							if(full) { //full parse tree 
								synchronized(completeParseTrees) {
									completeParseTrees.add(dNew);
								}
							}
//							else  {
								synchronized(newBeam) {
									boolean exist = this.push(newBeam.get(dNew.lenRoot()), dNew, beamSize);
									if(exist)
										packedParse.incrementAndGet();
								}
//							}
							
							LOG.debug("Generated %s; Shift %s on %s; total Score: %s ", dNew.getDebugHashCode(),
									lexicalEntry, dstate.getDebugHashCode(), dNew.score);
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
									if(pruningFilter != null) {
										SentenceSpan lastSpan_ = new SentenceSpan(lastSpan.getStart(), lastSpan.getEnd(), lastSpan.getEnd()-lastSpan.getStart());
										ParsingOp<MR> op = new ParsingOp<MR>(logical.getResultCategory(), lastSpan_, name);
										if(!pruningFilter.test(op))
											continue;
									}
									
									DerivationState<MR> dNew =	dstate.reduceUnaryRule(name, logical.getResultCategory());
									
									boolean full = dNew.lenRoot() == 1 && n == dNew.wordsConsumed && 
											       dNew.returnLastNonTerminal().getCategory().getSemantics()!=null &&
											       this.completeParseFilter.test(dNew.returnLastNonTerminal().getCategory());
									
									List<Category<MR>> children = new LinkedList<Category<MR>>();
									children.add(lastNonTerminal.getCategory());
									
									ShiftReduceParseStep<MR> step = new ShiftReduceParseStep<MR>(dNew.returnLastNonTerminal().getCategory(),
													children, full, true, name, lastSpan.getStart(), lastSpan.getEnd());
									WeightedShiftReduceParseStep<MR> wstep = new WeightedShiftReduceParseStep<MR>(step, model);
								
									dNew.score = dstate.score + wstep.getStepScore();
									dNew.defineStep(wstep);
									dNew.calcDebugHashCode();
									
									if(full) {
										synchronized(completeParseTrees) {
											completeParseTrees.add(dNew);
										}
									}
//									else  {
										synchronized(newBeam) {
											boolean exist = this.push(newBeam.get(dNew.lenRoot()), dNew, beamSize);
											if(exist)
												packedParse.incrementAndGet();
										}
//									}
									
									LOG.debug("Generated %s; Unary-Reduce %s %s; score: %s ", dNew.getDebugHashCode(), name, 
											dstate.getDebugHashCode(), dNew.score);
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
								if(logical!= null) {
									DerivationState<MR> dNew = dstate.reduceBinaryRule(name, logical.getResultCategory(), joined);
									
									if(pruningFilter != null) {
										SentenceSpan joined_ = new SentenceSpan(sndLastSpan.getStart(), lastSpan.getEnd(), lastSpan.getEnd()-sndLastSpan.getStart());
										ParsingOp<MR> op = new ParsingOp<MR>(logical.getResultCategory(), joined_, name);
										if(!pruningFilter.test(op))
											continue;
									}
									
									boolean full = dNew.lenRoot() == 1 && n == dNew.wordsConsumed && 
											   dNew.returnLastNonTerminal().getCategory().getSemantics()!=null &&
											   this.completeParseFilter.test(dNew.returnLastNonTerminal().getCategory());
									
									ShiftReduceParseStep<MR> step = new ShiftReduceParseStep<MR>(dNew.returnLastNonTerminal().getCategory(),
											dstate.returnBothCategories(), full, false, name, joined.getStart(), joined.getEnd());
									WeightedShiftReduceParseStep<MR> wstep = new WeightedShiftReduceParseStep<MR>(step, model);
									
									dNew.score = dstate.score + wstep.getStepScore();
									dNew.defineStep(wstep);
									dNew.calcDebugHashCode();
									
									if(full) {
										synchronized(completeParseTrees) {
											completeParseTrees.add(dNew);
										}
									}
//									else {
										synchronized(newBeam) {
											boolean exist = this.push(newBeam.get(dNew.lenRoot()), dNew, beamSize);
											if(exist)
												packedParse.incrementAndGet();
										}
//									}
									
									LOG.debug("Generated %s; Binary-Reduce %s %s %s; score: %s ", dNew.getDebugHashCode(), 
											logical, name, dstate.getDebugHashCode(), dNew.score);
								}
							}
						}
					}
				});
			}
			
			
			Iterator<DirectAccessBoundedPriorityQueue<PackedState<MR>>> beamIter = beam.iterator();
			Iterator<DirectAccessBoundedPriorityQueue<PackedState<MR>>> nBeamIter = newBeam.iterator();
			
			int k=0;
			isEmpty = true;
			
			while(nBeamIter.hasNext()) {
				DirectAccessBoundedPriorityQueue<PackedState<MR>> nBeam_ = nBeamIter.next();
				assert beamIter.hasNext();
				DirectAccessBoundedPriorityQueue<PackedState<MR>> beam_ = beamIter.next();
				beam_.clear(); //clear the current stack
				
				double minScore = Double.MIN_VALUE;
				if(nBeam_.size() > 0)
					minScore = nBeam_.peek().getBestScore();
				LOG.debug("Number of states in %s th beam are %s. Beam Min Score: %s", k++, 
																nBeam_.size(), minScore);
			
				Iterator<PackedState<MR>> iter = nBeam_.iterator();
			
				while(iter.hasNext()) {
					PackedState<MR> ds_ = iter.next();  
					beam_.offer(ds_);
					isEmpty = false;
				}
				nBeam_.clear(); //clear the new stack
			}
		}
		
		final ShiftReduceParserOutput<MR> output = new ShiftReduceParserOutput<MR>(
									completeParseTrees, System.currentTimeMillis() - start);
		
		LOG.debug("Parses Packing %s ", packedParse.get());
		
		if(LOG.getLogLevel() == LogLevel.DEBUG) {
			LOG.debug("Number of complete parse trees found: %s", completeParseTrees.size());
			
			ListIterator<DerivationState<MR>> it = completeParseTrees.listIterator();  
			while(it.hasNext()) {
				LOG.debug("Parse Tree %s %s", it.nextIndex(), it.next().getDebugHashCode());
			}
		}
		
		return output;
	}
	
}
