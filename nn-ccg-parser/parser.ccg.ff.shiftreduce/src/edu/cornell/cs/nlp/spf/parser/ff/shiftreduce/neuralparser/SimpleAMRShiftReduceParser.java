package edu.cornell.cs.nlp.spf.parser.ff.shiftreduce.neuralparser;

import java.io.File;
import java.lang.reflect.Array;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashSet;
import java.util.IdentityHashMap;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;
import java.util.Set;
import java.util.Spliterator;
import java.util.Spliterators;
import java.util.concurrent.atomic.AtomicLong;
import java.util.function.Predicate;
import java.util.stream.Collectors;
import java.util.stream.StreamSupport;

import edu.cornell.cs.nlp.spf.base.hashvector.IHashVector;
import edu.cornell.cs.nlp.spf.base.token.TokenSeq;
import edu.cornell.cs.nlp.spf.ccg.categories.Category;
import edu.cornell.cs.nlp.spf.ccg.categories.ICategoryServices;
import edu.cornell.cs.nlp.spf.ccg.categories.syntax.Syntax.SimpleSyntax;
import edu.cornell.cs.nlp.spf.ccg.lexicon.CompositeImmutableLexicon;
import edu.cornell.cs.nlp.spf.ccg.lexicon.ILexiconImmutable;
import edu.cornell.cs.nlp.spf.ccg.lexicon.LexicalEntry;
import edu.cornell.cs.nlp.spf.ccg.lexicon.Lexicon;
import edu.cornell.cs.nlp.spf.data.sentence.Sentence;
import edu.cornell.cs.nlp.spf.explat.IResourceRepository;
import edu.cornell.cs.nlp.spf.explat.ParameterizedExperiment;
import edu.cornell.cs.nlp.spf.explat.ParameterizedExperiment.Parameters;
import edu.cornell.cs.nlp.spf.explat.resources.IResourceObjectCreator;
import edu.cornell.cs.nlp.spf.explat.resources.usage.ResourceUsage;
import edu.cornell.cs.nlp.spf.parser.ISentenceLexiconGenerator;
import edu.cornell.cs.nlp.spf.parser.ParsingOp;
import edu.cornell.cs.nlp.spf.parser.ccg.model.IDataItemModel;
import edu.cornell.cs.nlp.spf.parser.ccg.normalform.NormalFormValidator;
import edu.cornell.cs.nlp.spf.parser.ccg.normalform.unaryconstraint.UnaryConstraint;
import edu.cornell.cs.nlp.spf.parser.ccg.rules.BinaryRuleSet;
import edu.cornell.cs.nlp.spf.parser.ccg.rules.IBinaryParseRule;
import edu.cornell.cs.nlp.spf.parser.ccg.rules.ILexicalRule;
import edu.cornell.cs.nlp.spf.parser.ccg.rules.IUnaryParseRule;
import edu.cornell.cs.nlp.spf.parser.ccg.rules.LexicalResult;
import edu.cornell.cs.nlp.spf.parser.ccg.rules.LexicalRule;
import edu.cornell.cs.nlp.spf.parser.ccg.rules.ParseRuleResult;
import edu.cornell.cs.nlp.spf.parser.ccg.rules.RuleName;
import edu.cornell.cs.nlp.spf.parser.ccg.rules.SentenceSpan;
import edu.cornell.cs.nlp.spf.parser.ccg.rules.UnaryRuleSet;
import edu.cornell.cs.nlp.spf.parser.ccg.shiftreduce.ShiftReduceBinaryParsingRule;
import edu.cornell.cs.nlp.spf.parser.ccg.shiftreduce.ShiftReduceParserOutput;
import edu.cornell.cs.nlp.spf.parser.ccg.shiftreduce.ShiftReduceUnaryParsingRule;
import edu.cornell.cs.nlp.spf.parser.ccg.shiftreduce.logger.ShiftReduceParseTreeLogger;
import edu.cornell.cs.nlp.spf.parser.ccg.shiftreduce.sloppy.AggressiveWordSkippingLexicalGenerator;
import edu.cornell.cs.nlp.spf.parser.ccg.shiftreduce.sloppy.BackwardSkippingRule;
import edu.cornell.cs.nlp.spf.parser.ccg.shiftreduce.sloppy.ForwardSkippingRule;
import edu.cornell.cs.nlp.spf.parser.ccg.shiftreduce.sloppy.SimpleWordSkippingLexicalGenerator;
import edu.cornell.cs.nlp.spf.parser.ccg.shiftreduce.stacks.DerivationState;
import edu.cornell.cs.nlp.spf.parser.ccg.shiftreduce.stacks.LexicalParsingOp;
import edu.cornell.cs.nlp.spf.parser.ccg.shiftreduce.stacks.PackedState;
import edu.cornell.cs.nlp.spf.parser.ccg.shiftreduce.stacks.ShiftReduceRuleNameSet;
import edu.cornell.cs.nlp.spf.parser.ccg.shiftreduce.steps.AbstractShiftReduceStep;
import edu.cornell.cs.nlp.spf.parser.ccg.shiftreduce.steps.IWeightedShiftReduceStep;
import edu.cornell.cs.nlp.spf.parser.ccg.shiftreduce.steps.ShiftReduceLexicalStep;
import edu.cornell.cs.nlp.spf.parser.ccg.shiftreduce.steps.ShiftReduceParseStep;
import edu.cornell.cs.nlp.spf.parser.ccg.shiftreduce.steps.WeightedShiftReduceLexicalStep;
import edu.cornell.cs.nlp.spf.parser.ccg.shiftreduce.steps.WeightedShiftReduceParseStep;
import edu.cornell.cs.nlp.spf.parser.graph.IGraphParser;
import edu.cornell.cs.nlp.spf.parser.graph.IGraphParserOutput;
import edu.cornell.cs.nlp.utils.collections.SetUtils;
import edu.cornell.cs.nlp.utils.collections.queue.DirectAccessBoundedPriorityQueue;
import edu.cornell.cs.nlp.utils.composites.Pair;
import edu.cornell.cs.nlp.utils.filter.FilterUtils;
import edu.cornell.cs.nlp.utils.filter.IFilter;
import edu.cornell.cs.nlp.utils.log.ILogger;
import edu.cornell.cs.nlp.utils.log.LogLevel;
import edu.cornell.cs.nlp.utils.log.LoggerFactory;
import edu.cornell.cs.nlp.utils.math.LogSumExp;

	/** 
	 * Parses utterance using a simple shift reduce parser. 
	 * @author Dipendra Misra
	 */
	public class SimpleAMRShiftReduceParser<DI extends Sentence, MR> 
					implements IGraphParser<DI, MR> {
		
		private static final long serialVersionUID = -6298416191854139634L;

		public static final ILogger								LOG
							= LoggerFactory.create(SimpleAMRShiftReduceParser.class);
		
		private static final int 								numCores = 32;
		
		/** Beamsize of the parser */
		private final Integer									beamSize;
		
		/** Binary CCG parsing rules. */
		public final ShiftReduceBinaryParsingRule<MR>[]			binaryRules;

		private final IFilter<Category<MR>>						completeParseFilter;

		private final transient Predicate<ParsingOp<MR>> 		pruningFilter;
		
		private final ILexicalRule<MR> 							lexicalRule;
		
		/////TEMPORARY
		public boolean											testing;
		
		/**
		 * List of lexical generators that use the sentence itself to generate
		 * lexical entries.
		 */
		protected final List<ISentenceLexiconGenerator<DI, MR>>	sentenceLexiconGenerators;

		/**
		 * Lexical generators to create lexical entries for sloppy inference.
		 */
		protected final List<ISentenceLexiconGenerator<DI, MR>>	sloppyLexicalGenerators; 

		public final ShiftReduceUnaryParsingRule<MR>[]			unaryRules;

		protected final ICategoryServices<MR>					categoryServices;
		
		private final ShiftReduceParseTreeLogger<DI, MR>		logger;
		
		/** penalty for SKIP operations */
		private final double 									gamma;
		
		private final PostProcessing<MR>						postProcessing;
		
//		static {
//	        Nd4j.dtype = DataBuffer.Type.DOUBLE;
//	        NDArrayFactory factory = Nd4j.factory();
//	        factory.setDType(DataBuffer.Type.DOUBLE);
//	    }
		
		private boolean disablePacking;
		
		/** TODO -- separate learning parts from other components */
		public SimpleAMRShiftReduceParser(int beamSize,
				ShiftReduceBinaryParsingRule<MR>[] binaryRules, ILexicalRule<MR> lexicalRule, 
				List<ISentenceLexiconGenerator<DI, MR>> sentenceLexiconGenerators,
				List<ISentenceLexiconGenerator<DI, MR>> sloppyLexicalGenerators,
				ICategoryServices<MR> categoryServices, 
				IFilter<Category<MR>> completeParseFilter,
				ShiftReduceUnaryParsingRule<MR>[] unaryRules, double learningRate, 
				double learningRateDecay, double l2, double gamma, int seed, File outputDir, double nullClosurePenalty) {
		
			final List<RuleName> ruleNames = new LinkedList<RuleName>();
			ruleNames.add(ShiftReduceLexicalStep.LEXICAL_DERIVATION_STEP_RULENAME);
			for(int u = 0; u< unaryRules.length; u++)
				ruleNames.add(unaryRules[u].getName());
			for(int b = 0; b< binaryRules.length; b++)
				ruleNames.add(binaryRules[b].getName());
			
			this.disablePacking = false;
			
			this.beamSize = beamSize;
			this.lexicalRule = lexicalRule;
			this.binaryRules = binaryRules;
			this.sentenceLexiconGenerators = sentenceLexiconGenerators;
			this.sloppyLexicalGenerators = sloppyLexicalGenerators;
			this.categoryServices = categoryServices;
			this.completeParseFilter = completeParseFilter;
			this.unaryRules = unaryRules;
			
			for(ShiftReduceUnaryParsingRule<MR> unaryRule: this.unaryRules) {
				LOG.info("Neural Feed foward: unary Rule %s", unaryRule);
			}
			LOG.info("Number of unary rules %s, binary rules %s", 
								unaryRules.length, binaryRules.length);
			
			this.gamma = gamma;
			this.testing = true; //false;
			this.postProcessing = new PostProcessing<MR>(nullClosurePenalty);
			
			this.logger = new ShiftReduceParseTreeLogger<DI, MR>(outputDir);
			
			this.pruningFilter = new Predicate<ParsingOp<MR>>() {
				public boolean test(ParsingOp<MR> e) {
					return true;
				}
			};
			
			LOG.info("Parser Init :: %s: ... sloppyLexicalGenerator=%s ...", getClass(),
					sloppyLexicalGenerators);
			LOG.info("Parser Init :: %s: ... binary rules=%s ...", getClass(),
					Arrays.toString(binaryRules));
			LOG.info("Parser Init :: %s: ... unary rules=%s ...", getClass(),
					Arrays.toString(unaryRules));
			LOG.info("Simple AMR Shift Reduce Parser. Gamma %s, outputDir %s, null closure penalty %s", 
					gamma, outputDir.getAbsolutePath(), nullClosurePenalty);
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
		
		public int numRules() {
			//number of parsing rules = One lexical rule, all the unary rules and the binary rules
			return 1 + this.unaryRules.length + this.binaryRules.length;
		}
	
		protected boolean push(DirectAccessBoundedPriorityQueue<PackedState<MR>> queue, PackedState<MR> pstate, 
				 DerivationState<MR> elem, int capacity) {
			
			PackedState<MR> prior = queue.get(pstate);
			
			if(prior ==  null || this.disablePacking) { //safe to push
			/* Introduces randomization, due to threading and ties for the lowest score 
			* state. In order to prevent this, disable threads. To introduce determinism
			* later: when removing a state, remove all the states with the same score. */
				queue.offer(pstate); 
				return false; 
			} else {
				queue.remove(prior);
				prior.add(elem); 
				queue.add(prior);
				return true; //true indicates that it was packed
			}
		}
		
		private boolean test(Predicate<ParsingOp<MR>> pruningFilter, ParsingOp<MR> op, 
											DerivationState<MR> dstate) {
			
			if(pruningFilter == null) {
				return true; //if there is no filter, then everything passes through
			}
			
			return pruningFilter.test(op);
		}
		
		private List<LexicalResult<MR>>[][] preprocessLexicalResults(TokenSeq tk, CompositeImmutableLexicon<MR> compositeLexicon) {
			
			final int n = tk.size();
			
			@SuppressWarnings("unchecked")
			List<LexicalResult<MR>>[][] allLexicalResults = new List[n][n];
			
			for(int start = 0; start < n; start++) {
				for(int end = 0; end < start; end ++) {
					allLexicalResults[start][end] = Collections.emptyList();
				}
				
				for(int end = start + 1; end <= n; end++) {
					
					List<LexicalResult<MR>> lexicalResults = new ArrayList<LexicalResult<MR>>();
					
					Iterator<LexicalResult<MR>> it = this.lexicalRule.apply(tk.sub(start, end),
								new SentenceSpan(start, end - 1, n), compositeLexicon);
					
					while(it.hasNext()) {
						LexicalResult<MR> result = it.next();
						lexicalResults.add(result);
					}
					
					allLexicalResults[start][end - 1] = lexicalResults;
				}
			}
			
			return allLexicalResults;
		}
		
		public void enablePacking() {
			this.disablePacking = false;
		}
		
		public void disablePacking() {
			this.disablePacking = true;
		}
		
		@Override
		public IGraphParserOutput<MR> parse(DI dataItem,
				IDataItemModel<MR> model) {
			return this.parse(dataItem, this.pruningFilter, model,
			     false, null, null);
		}
		
		@Override
		public IGraphParserOutput<MR> parse(DI dataItem,
				IDataItemModel<MR> model, boolean allowWordSkipping) {
		
			return this.parse(dataItem, this.pruningFilter, model,
					allowWordSkipping, null, null);
		}
		
		@Override
		public IGraphParserOutput<MR> parse(DI dataItem, IDataItemModel<MR> model,
				boolean allowWordSkipping, ILexiconImmutable<MR> tempLexicon) {
			return this.parse(dataItem, this.pruningFilter, model,
					allowWordSkipping, tempLexicon, null);
		}

		@Override
		public IGraphParserOutput<MR> parse(DI dataItem, IDataItemModel<MR> model,
				boolean allowWordSkipping, ILexiconImmutable<MR> tempLexicon,
				Integer beamSize) {
			return this.parse(dataItem, this.pruningFilter, model,
					allowWordSkipping, tempLexicon, beamSize);
		}

		@Override
		public IGraphParserOutput<MR> parse(DI dataItem, Predicate<ParsingOp<MR>> filter,
				IDataItemModel<MR> model) {
			return this.parse(dataItem, filter, model, false, null, null);
		}

		@Override
		public IGraphParserOutput<MR> parse(DI dataItem, Predicate<ParsingOp<MR>> filter,
				IDataItemModel<MR> model, boolean allowWordSkipping) {
			return this.parse(dataItem, filter, model,
					allowWordSkipping, null, null);
		}

		@Override
		public IGraphParserOutput<MR> parse(DI dataItem, Predicate<ParsingOp<MR>> filter,
				IDataItemModel<MR> model, boolean allowWordSkipping,
				ILexiconImmutable<MR> tempLexicon) {
			return this.parse(dataItem, filter, model,
					allowWordSkipping, tempLexicon, null);
		}
			
		/** Parses a sentence using Neural Network model */
		@Override
		public IGraphParserOutput<MR> parse(DI dataItem, Predicate<ParsingOp<MR>> pruningFilter, IDataItemModel<MR> model,
				boolean allowWordSkipping, ILexiconImmutable<MR> tempLexicon, Integer beamSize_) {
			
			//skip this sentence due to metric issue
			if(dataItem.getSample().getTokens().toString()
					.startsWith("The government insists the reserves will support the armed forces and not act as")) {
				LOG.info("Skipping this sentence due to metric issue");
				return new ShiftReduceParserOutput<MR>(new ArrayList<DerivationState<MR>>(), 1);
			}
			
			final long start = System.currentTimeMillis();
			
			LOG.info("Simple Shift Reduce: Testing %s", this.testing);
			
			final Integer beamSize;
			if(beamSize_ == null) {
				beamSize = this.beamSize;
			} else {
				beamSize = beamSize_;
			}
						
			final Set<DerivationState<MR>> identityState = Collections.newSetFromMap(new IdentityHashMap<DerivationState<MR>, Boolean>());
			
			LOG.info("Beamsize %s. Packing Disabled %s", beamSize, this.disablePacking);
			
			final Comparator<PackedState<MR>> dStateCmp  = new Comparator<PackedState<MR>>() {
				public int compare(PackedState<MR> left, PackedState<MR> right) {
	        		return Double.compare(left.getBestScore(), right.getBestScore()); 
	    		}   
			};
			
			final AtomicLong packedParse = new AtomicLong();
			
			LOG.debug("Utterance: %s", dataItem);
			
			TokenSeq tk = dataItem.getTokens();
			int n = tk.size(); //number of tokens
			
			List<DerivationState<MR>> completeParseTrees = new ArrayList<DerivationState<MR>>();
			
			List<DirectAccessBoundedPriorityQueue<PackedState<MR>>> beam = new 
							ArrayList<DirectAccessBoundedPriorityQueue<PackedState<MR>>>();
			List<DirectAccessBoundedPriorityQueue<PackedState<MR>>> newBeam = new 
							ArrayList<DirectAccessBoundedPriorityQueue<PackedState<MR>>>();
			
			for(int i = 0; i <= n; i++) { //a beam for different number of words consumed 
				beam.add(new DirectAccessBoundedPriorityQueue<PackedState<MR>>(beamSize, dStateCmp));
				newBeam.add(new DirectAccessBoundedPriorityQueue<PackedState<MR>>(beamSize, dStateCmp));
			}
			
			PackedState<MR> initState =  new PackedState<MR>(new DerivationState<MR>());
			beam.get(0).offer(initState);
			
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
			
			List<LexicalResult<MR>>[][] allLexicalResults = this.preprocessLexicalResults(tk, compositeLexicon);
			
			while(!isEmpty) {
				LOG.debug("=========== CYCLE %s =============", ++cycle);
				Iterator<DirectAccessBoundedPriorityQueue<PackedState<MR>>> iterBeam = beam.iterator();
				int ibj = 0;
				
				final Set<DerivationState<MR>> cycleIdentityState = 
						Collections.newSetFromMap(new IdentityHashMap<DerivationState<MR>, Boolean>());
						
				while(iterBeam.hasNext()) {
					LOG.debug("### Working on the beam %s ###", ++ibj);
					final DirectAccessBoundedPriorityQueue<PackedState<MR>> pstates = iterBeam.next();
					
					StreamSupport.stream(Spliterators.spliterator(pstates, Spliterator.IMMUTABLE), 
										LOG.getLogLevel() == LogLevel.DEBUG ? false : true)
						    .forEach(pstate -> { 
						/* perform valid shift and reduce operations for this packed states, if the packed state is
						 * finished and already represents a complete parse tree then save it separately. 
						 * 
						 * probability/score of a parsing action is given by: 
			             * exp{w^T phi(action) } / \sum_action' exp { w^T phi(action') } */
						
						/* Important: Since features are currently local that is only look at the root categories or 
						 * at the shifted lexical entries. Hence, operations can be performed on the best state
						 * currently in the packed state. This will NO LONGER HOLD if features start looking 
						 * at the complete tree segments in the state. It holds here since feature for action are local */
						    	
				    	DerivationState<MR> dstate = pstate.getBestState();
						final int wordsConsumed = dstate.wordsConsumed;
						int childIndex = 0;
						
						//list of new potential states and the step that created them.
						List<Pair<DerivationState<MR>, AbstractShiftReduceStep<MR>>> newStateAndStep = 
										new ArrayList<Pair<DerivationState<MR>, AbstractShiftReduceStep<MR>>>();
						
						//Operation 1: Shift operation: shift a token and its lexical entry to this stack
						if(wordsConsumed < n) {
							
							for(int words = 1; words <= n - wordsConsumed; words++) {
								
								List<LexicalResult<MR>> lexicalResults =
												allLexicalResults[wordsConsumed][wordsConsumed + words - 1];
								
								for(LexicalResult<MR> lexicalResult: lexicalResults) {
									
									LexicalEntry<MR> lexicalEntry = lexicalResult.getEntry();
									
									SentenceSpan span = new SentenceSpan(wordsConsumed, wordsConsumed + words, n);
									ParsingOp<MR> op = new LexicalParsingOp<MR>(lexicalResult.getResultCategory(), span, 
																		ShiftReduceLexicalStep.LEXICAL_DERIVATION_STEP_RULENAME, lexicalEntry);
									
									if(pruningFilter != null && !this.test(pruningFilter, op, dstate)) {
										continue; 
									}
									
									//check this full line below
									boolean full = (n == words) && this.completeParseFilter.test(lexicalResult.getResultCategory());
									
									ShiftReduceLexicalStep<MR> lexicalStep1 = new ShiftReduceLexicalStep<MR>(lexicalResult.getResultCategory(),
											lexicalEntry, full, dstate.wordsConsumed, dstate.wordsConsumed + words - 1);
									
									final int myChildIndex = childIndex++;
									
									//////// Cannot skip more than one word /////////////////
									if(words > 1 && op.getCategory().getSyntax().equals(SimpleSyntax.EMPTY)) {
										continue;
									}
									
									DerivationState<MR> dNew = dstate.shift(lexicalResult, words, span);
									dNew.childIndex = myChildIndex;
									
									newStateAndStep.add(Pair.of(dNew, lexicalStep1));
									dNew.calcDebugHashCode();
									
									LOG.debug("Generated %s; Shift %s [unmodified] on %s", dNew.getDebugHashCode(),
																		lexicalEntry, dstate.getDebugHashCode());
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
									ParseRuleResult<MR> logical  = this.applyUnaryRule(uj, lastNonTerminal, new SentenceSpan(lastSpan.getStart(), 
																						lastSpan.getEnd() - 1, lastSpan.getSentenceLength()));
									
									if(logical != null) {
										SentenceSpan lastSpan_ = new SentenceSpan(lastSpan.getStart(), lastSpan.getEnd(), n);
										ParsingOp<MR> op = new ParsingOp<MR>(logical.getResultCategory(), lastSpan_, name);
										
										if(pruningFilter != null && !this.test(pruningFilter, op, dstate)) {
											continue;
										}
										
										DerivationState<MR> dNew =	dstate.reduceUnaryRule(name, logical.getResultCategory());
										
										boolean full = (dNew.lenRoot() == 1) && (n == dNew.wordsConsumed) && 
												       (dNew.returnLastNonTerminal().getCategory().getSemantics() != null) &&
												       (this.completeParseFilter.test(dNew.returnLastNonTerminal().getCategory()));
										
										List<Category<MR>> children = new LinkedList<Category<MR>>();
										children.add(lastNonTerminal.getCategory());
										
										ShiftReduceParseStep<MR> step1 = new ShiftReduceParseStep<MR>(dNew.returnLastNonTerminal().getCategory(),
												children, full, true, name, lastSpan.getStart(), lastSpan.getEnd() - 1);
								
										dNew.childIndex = childIndex++;
										newStateAndStep.add(Pair.of(dNew, step1));
										dNew.calcDebugHashCode();
										
										LOG.debug("Generated %s; Unary-Reduce %s %s; ", dNew.getDebugHashCode(), name, 
																									dstate.getDebugHashCode());
									}
								}
							}
							
							ShiftReduceRuleNameSet<MR> last2ndLastNonTerminal = dstate.return2ndLastNonTerminal();
							SentenceSpan sndLastSpan = dstate.return2ndLastSentenceSpan();
							
							if(last2ndLastNonTerminal != null) {
								//Operation 3: Binary Reduce operation
								SentenceSpan joined = new SentenceSpan(sndLastSpan.getStart(), lastSpan.getEnd() - 1, n);
								for(int bj = 0; bj < this.binaryRules.length; bj++) {
									
									RuleName name = this.binaryRules[bj].getName();
									ParseRuleResult<MR> logical  = this.applyBinaryRule(bj, last2ndLastNonTerminal, 
																	lastNonTerminal, joined);
									if(logical != null) {
										
										final SentenceSpan joined_ = new SentenceSpan(sndLastSpan.getStart(), lastSpan.getEnd(), n);
										ParsingOp<MR> op = new ParsingOp<MR>(logical.getResultCategory(), joined_, name);
										
										if(pruningFilter != null && !this.test(pruningFilter, op, dstate)) {
											continue;
										}
										
										DerivationState<MR> dNew = dstate.reduceBinaryRule(name, logical.getResultCategory(), joined_);
										
										boolean full = (dNew.lenRoot() == 1) && (n == dNew.wordsConsumed) && 
												   (dNew.returnLastNonTerminal().getCategory().getSemantics() != null) &&
												   (this.completeParseFilter.test(dNew.returnLastNonTerminal().getCategory()));
										
										ShiftReduceParseStep<MR> step1 = new ShiftReduceParseStep<MR>(dNew.returnLastNonTerminal().getCategory(),
												dstate.returnBothCategories(), full, false, name, joined_.getStart(), joined_.getEnd() - 1);
										
										dNew.childIndex = childIndex++;
										newStateAndStep.add(Pair.of(dNew, step1));
										dNew.calcDebugHashCode();
										
										LOG.debug("Generated %s; Binary-Reduce %s %s %s; ", dNew.getDebugHashCode(), 
																					logical, name, dstate.getDebugHashCode());
									}
								}
							}
						}
						
						if(newStateAndStep.size() == 0) { //terminal state
							
							if(this.testing && dstate.wordsConsumed == n) {
								synchronized(cycleIdentityState) {
									cycleIdentityState.add(dstate);
								}
							}
							return;
						}
						
						// Compute exponents
						List<Double> exponents = newStateAndStep.stream()
												.map(stateAndStep -> { 
													 
														AbstractShiftReduceStep<MR> step = stateAndStep.second();
														IHashVector stepFeatures = model.computeFeatures(step);
														return model.score(stepFeatures);
													
													})
												.collect(Collectors.toList());
						
						//normalize the probabilities and add to list
						final double logZ = LogSumExp.of(exponents);
						Iterator<Double> it = exponents.iterator();
						
						for(Pair<DerivationState<MR>, AbstractShiftReduceStep<MR>> next: newStateAndStep) {
							final DerivationState<MR> dNew = next.first();
							final AbstractShiftReduceStep<MR> step = next.second(); 
							final double logLikelihood = it.next() - logZ;
							
							final IWeightedShiftReduceStep<MR> weightedStep;
							
							//Since we compute the features, we can probably store the features in Weighted step too
							if(step instanceof ShiftReduceLexicalStep<?>) {
								weightedStep = new WeightedShiftReduceLexicalStep<MR>(
															(ShiftReduceLexicalStep<MR>)step, logLikelihood);
							} else {
								weightedStep = new WeightedShiftReduceParseStep<MR>(
															(ShiftReduceParseStep<MR>)step, logLikelihood);
							}
							
							double stepScore = weightedStep.getStepScore();
							
							if(stepScore > 0.0) throw new RuntimeException("Log likelihood can never be positive");
							dNew.score = dstate.score + stepScore;
							
							LOG.debug("Score %s; state %s; step %s", dNew.getDebugHashCode(), dNew.score, stepScore); 
							
							//Using inclusive spans in weighted step
							dNew.defineStep(weightedStep);
							
							boolean full = dNew.lenRoot() == 1 && n == dNew.wordsConsumed && 
									   dNew.returnLastNonTerminal().getCategory().getSemantics() != null &&
									   this.completeParseFilter.test(dNew.returnLastNonTerminal().getCategory());
							
							if(full) {
								synchronized(completeParseTrees) {
									completeParseTrees.add(dNew);
								}
							}
							
							PackedState<MR> pstateNew = new PackedState<MR>(dNew);
							synchronized(newBeam) {
								boolean exist = this.push(newBeam.get(0/*dNew.lenRoot()*/), pstateNew, dNew, beamSize);
								if(exist) {
									packedParse.incrementAndGet();
								}
							}
						}
					});
				}
				
				
				Iterator<DirectAccessBoundedPriorityQueue<PackedState<MR>>> beamIter = beam.iterator();
				Iterator<DirectAccessBoundedPriorityQueue<PackedState<MR>>> nBeamIter = newBeam.iterator();
				
				int k = 0;
				isEmpty = true;
				
				while(nBeamIter.hasNext()) {
					DirectAccessBoundedPriorityQueue<PackedState<MR>> nBeam_ = nBeamIter.next();
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
						LOG.debug("\t State in beam 0: %s parent %s score: %s", ds_.hashCode(), 
											ds_.getBestState().getParent().hashCode(), ds_.getBestScore());
						beam_.offer(ds_);
						isEmpty = false;
					}
					nBeam_.clear(); //clear the new stack
				}
				
				identityState.addAll(cycleIdentityState);
			}
			
			final long parsingTime = System.currentTimeMillis() - start;
			final ShiftReduceParserOutput<MR> output = new ShiftReduceParserOutput<MR>(
											completeParseTrees, parsingTime);
			
			LOG.info("Neural Shift Reduce: Number of distinct derivations %s", output.getAllDerivations().size());
			
			if(this.testing) { //Ugly hack for knowing that we are testing
				this.logger.log(output, dataItem, allowWordSkipping);
			}
						
			if(this.testing && allowWordSkipping && output.getAllDerivations().size() == 0) {
				LOG.info("Found no derivation. Stitching heuristically to produce parse trees");
				return this.postProcessing.stitch6(identityState, pruningFilter, parsingTime);
			}
			
			return output;
		}
		
		public static class Builder<DI extends Sentence, MR> {
			
			private Integer									        beamSize;
			
			private double											learningRate = 0.1;
			
			private double											learningRateDecay = 0.01;
			
			private double											l2 = 0.000001;
			
			private double											gamma = 5;
			
			private int 											seed = 1234;
			
			private double											nullClosurePenalty = 0.0;
			
			private File											outputDir = null;

			private final Set<ShiftReduceBinaryParsingRule<MR>>     binaryRules	
												= new HashSet<ShiftReduceBinaryParsingRule<MR>>();
			
			/** lexical rule is defined with default lexical rule that returns the lexical entry as it is */
			private ILexicalRule<MR>								lexicalRule
												= new LexicalRule<MR>();

			private IFilter<Category<MR>>						    completeParseFilter 
												= FilterUtils.stubTrue();

			private final List<ISentenceLexiconGenerator<DI, MR>>	sentenceLexicalGenerators	
											 	= new ArrayList<ISentenceLexiconGenerator<DI, MR>>();

			private final List<ISentenceLexiconGenerator<DI, MR>>	sloppyLexicalGenerators		
												= new ArrayList<ISentenceLexiconGenerator<DI, MR>>();

			private final Set<ShiftReduceUnaryParsingRule<MR>>	    unaryRules
												= new HashSet<ShiftReduceUnaryParsingRule<MR>>();

			private final ICategoryServices<MR>				        categoryServices; 
			
			public Builder(ICategoryServices<MR> categoryServices) {
				this.categoryServices = categoryServices;
			}

			public Builder<DI, MR> addParseRule(ShiftReduceBinaryParsingRule<MR> rule) {
				this.binaryRules.add(rule);
				return this;
			}

			public Builder<DI, MR> addParseRule(ShiftReduceUnaryParsingRule<MR> rule) {
				unaryRules.add(rule);
				return this;
			}
			
			public Builder<DI, MR> setLexicalRule(ILexicalRule<MR> lexicalRule) {
				this.lexicalRule = lexicalRule;
				return this;
			}

			public Builder<DI, MR> addSentenceLexicalGenerator(
					ISentenceLexiconGenerator<DI, MR> generator) {
				sentenceLexicalGenerators.add(generator);
				return this;
			}

			public Builder<DI, MR> addSloppyLexicalGenerator(
					ISentenceLexiconGenerator<DI, MR> sloppyGenerator) {
				sloppyLexicalGenerators.add(sloppyGenerator);
				return this;
			}
			
			public Builder<DI, MR> setOutputDir(File outputDir) {
				this.outputDir = outputDir;
				return this;
			}
			
			@SuppressWarnings("unchecked")
			public SimpleAMRShiftReduceParser<DI, MR> build() {
				return new SimpleAMRShiftReduceParser<DI, MR>(this.beamSize,
						binaryRules.toArray((ShiftReduceBinaryParsingRule<MR>[]) Array
								.newInstance(ShiftReduceBinaryParsingRule.class,
										binaryRules.size())), lexicalRule, 
						sentenceLexicalGenerators, sloppyLexicalGenerators,
						categoryServices, completeParseFilter,
						unaryRules.toArray((ShiftReduceUnaryParsingRule<MR>[]) Array
								.newInstance(ShiftReduceUnaryParsingRule.class,
										unaryRules.size())), 
						learningRate, learningRateDecay, l2, gamma, seed, outputDir, nullClosurePenalty);
			}
			
			public Builder<DI, MR> setSeed(int seed) {
				this.seed = seed;
				return this;
			}
			
			public Builder<DI, MR> setLearningRate(double learningRate) {
				this.learningRate = learningRate;
				return this;
			}
			
			public Builder<DI, MR> setNullClosurePenalty(double nullClosurePenalty) {
				this.nullClosurePenalty = nullClosurePenalty;
				return this;
			}
			
			public Builder<DI, MR> setSkipPenalty(double gamma) {
				this.gamma = gamma;
				return this;
			}
			
			public Builder<DI, MR> setLearningDecay(double learningRateDecay) {
				this.learningRateDecay = learningRateDecay;
				return this;
			}
			
			public Builder<DI, MR> setL2(double l2) {
				this.l2 = l2;
				return this;
			}

			public Builder<DI, MR> setCompleteParseFilter(
					IFilter<Category<MR>> completeParseFilter) {
				this.completeParseFilter = completeParseFilter;
				return this;
			}
			
			public Builder<DI, MR> setBeamSize(Integer beamSize) {
				this.beamSize = beamSize;
				return this;
			}
		}
		
		public static class Creator<DI extends Sentence, MR>
				implements IResourceObjectCreator<SimpleAMRShiftReduceParser<DI, MR>> {

			private final String type;
			
			public Creator() {
				this("parser.simple.amr.shiftreduce");
			}

			public Creator(String type) {
				this.type = type;
			}
			
			@SuppressWarnings("unchecked")
			@Override
			public SimpleAMRShiftReduceParser<DI, MR> create(Parameters params, IResourceRepository repo) {
				
				final Builder<DI, MR> builder = new Builder<DI, MR>( (ICategoryServices<MR>) repo.get(
																	 ParameterizedExperiment.CATEGORY_SERVICES_RESOURCE));
				
				if (params.contains("parseFilter")) {
					builder.setCompleteParseFilter((IFilter<Category<MR>>) repo
							.get(params.get("parseFilter")));
				}

				if (params.contains("beamSize")) {
					builder.setBeamSize(params.getAsInteger("beamSize"));
				}

				for (final String id : params.getSplit("generators")) {
					builder.addSentenceLexicalGenerator(
							(ISentenceLexiconGenerator<DI, MR>) repo.get(id));
				}

				for (final String id : params.getSplit("sloppyGenerators")) {
					builder.addSloppyLexicalGenerator(
							(ISentenceLexiconGenerator<DI, MR>) repo.get(id));
				}
				
				NormalFormValidator nfValidator;
				if (params.contains("nfValidator")) {
					nfValidator = repo.get(params.get("nfValidator"));
				} else {
					nfValidator = null;
				}
				
				if(params.contains("lex")) {
					builder.setLexicalRule(repo.get(params.get("lex")));
				}

				final String wordSkippingType = params.get("wordSkipping", "none");
				if (wordSkippingType.equals("simple")) {
					// Skipping lexical generator.
					builder.addSloppyLexicalGenerator(
							new SimpleWordSkippingLexicalGenerator<DI, MR>(
									(ICategoryServices<MR>) repo.get(
											ParameterizedExperiment.CATEGORY_SERVICES_RESOURCE)));

					// Skipping rules.
					final ForwardSkippingRule<MR> forwardSkip = new ForwardSkippingRule<MR>(
							(ICategoryServices<MR>) repo.get(
									ParameterizedExperiment.CATEGORY_SERVICES_RESOURCE));
					final BackwardSkippingRule<MR> backSkip = new BackwardSkippingRule<MR>(
							(ICategoryServices<MR>) repo
									.get(ParameterizedExperiment.CATEGORY_SERVICES_RESOURCE),
							false);

					// Add a normal form constraint to disallow unary steps after
					// skipping.
					final NormalFormValidator.Builder nfBuilder = new NormalFormValidator.Builder();
					if (nfValidator != null) {
						nfBuilder.addConstraints(nfValidator);
					}
					nfBuilder.addConstraint(new UnaryConstraint(SetUtils
							.createSet(forwardSkip.getName(), backSkip.getName())));
					nfValidator = nfBuilder.build();

					// Add the rules.
					addRule(builder, backSkip, nfValidator);
					addRule(builder, forwardSkip, nfValidator);
				} else if (wordSkippingType.equals("aggressive")) {
					// Skipping lexical generator.
					builder.addSloppyLexicalGenerator(
							new AggressiveWordSkippingLexicalGenerator<DI, MR>(
									(ICategoryServices<MR>) repo.get(
											ParameterizedExperiment.CATEGORY_SERVICES_RESOURCE)));
					// Skipping rules.
					final ForwardSkippingRule<MR> forwardSkip = new ForwardSkippingRule<MR>(
							(ICategoryServices<MR>) repo.get(
									ParameterizedExperiment.CATEGORY_SERVICES_RESOURCE));
					final BackwardSkippingRule<MR> backSkip = new BackwardSkippingRule<MR>(
							(ICategoryServices<MR>) repo
									.get(ParameterizedExperiment.CATEGORY_SERVICES_RESOURCE),
							true);

					// Add a normal form constraint to disallow unary steps after
					// skipping.
					final NormalFormValidator.Builder nfBuilder = new NormalFormValidator.Builder();
					if (nfValidator != null) {
						nfBuilder.addConstraints(nfValidator);
					}
					nfBuilder.addConstraint(new UnaryConstraint(SetUtils
							.createSet(forwardSkip.getName(), backSkip.getName())));
					nfValidator = nfBuilder.build();

					// Add the rules.
					addRule(builder, backSkip, nfValidator);
					addRule(builder, forwardSkip, nfValidator);
				}

				for (final String id : params.getSplit("rules")) {
					final Object rule = repo.get(id);
					if (rule instanceof BinaryRuleSet) {
						for (final IBinaryParseRule<MR> singleRule : (BinaryRuleSet<MR>) rule) {
							addRule(builder, singleRule, nfValidator);
						}
					} else if (rule instanceof UnaryRuleSet) {
						for (final IUnaryParseRule<MR> singleRule : (UnaryRuleSet<MR>) rule) {
							addRule(builder, singleRule, nfValidator);
						}
					} else {
						addRule(builder, rule, nfValidator);
					}
				}
				
				if(params.contains("outputDir")) {
					builder.setOutputDir(
								params.getAsFile("outputDir"));
				}


				if (params.contains("learningRate")) {
					builder.setLearningRate(params.getAsDouble("learningRate"));
				}
				
				if (params.contains("learningRateDecay")) {
					builder.setLearningDecay(params.getAsDouble("learningRateDecay"));
				}
				
				if (params.contains("l2")) {
					builder.setL2(params.getAsDouble("l2"));
				}
				
				if (params.contains("gamma")) {
					builder.setSkipPenalty(params.getAsDouble("gamma"));
				}
				
				if (params.contains("nullClosurePenalty")) {
					builder.setNullClosurePenalty(params.getAsDouble("nullClosurePenalty"));
				}
				
				if (params.contains("seed")) {
					builder.setSeed(params.getAsInteger("seed"));
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
			
			@SuppressWarnings("unchecked")
			private void addRule(Builder<DI, MR> builder, Object rule,
					NormalFormValidator nfValidator) {
				if (rule instanceof IBinaryParseRule) {
					builder.addParseRule(new ShiftReduceBinaryParsingRule<MR>(
							(IBinaryParseRule<MR>) rule, nfValidator));
				} else if (rule instanceof IUnaryParseRule) {
					builder.addParseRule(new ShiftReduceUnaryParsingRule<MR>(
							(IUnaryParseRule<MR>) rule, nfValidator));
				} else if (rule instanceof ShiftReduceBinaryParsingRule) {
					builder.addParseRule((ShiftReduceBinaryParsingRule<MR>) rule);
				} else if (rule instanceof ShiftReduceUnaryParsingRule) {
					builder.addParseRule((ShiftReduceUnaryParsingRule<MR>) rule);
				} else {
					throw new IllegalArgumentException("Invalid rule class: " + rule);
				}
			}
		}
		
	}

