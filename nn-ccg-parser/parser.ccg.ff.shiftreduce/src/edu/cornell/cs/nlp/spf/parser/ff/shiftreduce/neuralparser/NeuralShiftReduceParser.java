package edu.cornell.cs.nlp.spf.parser.ff.shiftreduce.neuralparser;

import java.io.File;
import java.lang.reflect.Array;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.Spliterator;
import java.util.Spliterators;
import java.util.concurrent.atomic.AtomicLong;
import java.util.function.Predicate;
import java.util.stream.StreamSupport;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import edu.cornell.cs.nlp.spf.base.hashvector.IHashVector;
import edu.cornell.cs.nlp.spf.base.hashvector.KeyArgs;
import edu.cornell.cs.nlp.spf.base.token.TokenSeq;
import edu.cornell.cs.nlp.spf.ccg.categories.Category;
import edu.cornell.cs.nlp.spf.ccg.categories.ICategoryServices;
import edu.cornell.cs.nlp.spf.ccg.categories.syntax.Syntax.SimpleSyntax;
import edu.cornell.cs.nlp.spf.ccg.lexicon.CompositeImmutableLexicon;
import edu.cornell.cs.nlp.spf.ccg.lexicon.ILexiconImmutable;
import edu.cornell.cs.nlp.spf.ccg.lexicon.LexicalEntry;
import edu.cornell.cs.nlp.spf.ccg.lexicon.Lexicon;
import edu.cornell.cs.nlp.spf.data.sentence.Sentence;
import edu.cornell.cs.nlp.spf.data.situated.sentence.SituatedSentence;
import edu.cornell.cs.nlp.spf.explat.IResourceRepository;
import edu.cornell.cs.nlp.spf.explat.ParameterizedExperiment;
import edu.cornell.cs.nlp.spf.explat.ParameterizedExperiment.Parameters;
import edu.cornell.cs.nlp.spf.explat.resources.IResourceObjectCreator;
import edu.cornell.cs.nlp.spf.explat.resources.usage.ResourceUsage;
import edu.cornell.cs.nlp.spf.mr.lambda.LogicalExpression;
import edu.cornell.cs.nlp.spf.parser.ISentenceLexiconGenerator;
import edu.cornell.cs.nlp.spf.parser.ParsingOp;
import edu.cornell.cs.nlp.spf.parser.ccg.IParseStep;
import edu.cornell.cs.nlp.spf.parser.ccg.model.IDataItemModel;
import edu.cornell.cs.nlp.spf.parser.ccg.normalform.NormalFormValidator;
import edu.cornell.cs.nlp.spf.parser.ccg.normalform.unaryconstraint.UnaryConstraint;
import edu.cornell.cs.nlp.spf.parser.ccg.rnn.shiftreduce.dataset.CKYMultiParseTreeParsingFilter;
import edu.cornell.cs.nlp.spf.parser.ccg.rnn.shiftreduce.dataset.CKYSingleParseTreeParsingFilter;
import edu.cornell.cs.nlp.spf.parser.ccg.rnn.shiftreduce.neuralnetworkparser.AbstractNeuralShiftReduceParser;
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
import edu.cornell.cs.nlp.spf.parser.ff.shiftreduce.nonlocal.features.AbstractNonLocalFeature;
import edu.cornell.cs.nlp.spf.parser.ff.shiftreduce.nonlocal.features.StackSyntaxFeatures;
import edu.cornell.cs.nlp.spf.parser.graph.IGraphParserOutput;
import edu.cornell.cs.nlp.spf.parser.joint.model.IJointModelImmutable;
import edu.cornell.cs.nlp.utils.collections.SetUtils;
import edu.cornell.cs.nlp.utils.collections.queue.DirectAccessBoundedPriorityQueue;
import edu.cornell.cs.nlp.utils.composites.Pair;
import edu.cornell.cs.nlp.utils.filter.FilterUtils;
import edu.cornell.cs.nlp.utils.filter.IFilter;
import edu.cornell.cs.nlp.utils.log.ILogger;
import edu.cornell.cs.nlp.utils.log.LogLevel;
import edu.cornell.cs.nlp.utils.log.LoggerFactory;
import edu.uw.cs.lil.amr.data.AMRMeta;

/** 
 * Parses utterance using neural network shift reduce parser
 * @author Dipendra Misra
 */
public class NeuralShiftReduceParser<DI extends Sentence, MR> implements AbstractNeuralShiftReduceParser<DI, MR> {
	
	private static final long serialVersionUID = -6847536920456870074L;

	public static final ILogger								LOG
								= LoggerFactory.create(NeuralShiftReduceParser.class);
	
	private static final int 								numCores = 32;
	
	/** Feed-forward neural network that takes dense features representing state+parsing step
	 * and returns the score of the operation */
	private final NeuralParsingStepScorer mlpScorer;
	
	/** Converts sparse features to dense features */
	private final FeatureEmbedding<MR> featureEmbedding;
		
	/** Beamsize of the parser */
	private final Integer									beamSize;
	
	/** Binary CCG parsing rules. */
	public final ShiftReduceBinaryParsingRule<MR>[]			binaryRules;

	private final IFilter<Category<MR>>						completeParseFilter;

	private final Predicate<ParsingOp<MR>> 					pruningFilter;
	
	private final ILexicalRule<MR> 							lexicalRule;
	
	/////TEMPORARY
	public  IJointModelImmutable<SituatedSentence<AMRMeta>, LogicalExpression, LogicalExpression> modelNewFeatures;
	
	/////TEMPORARY
	public boolean											testing;
	
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
	
//	private final int 										numRules;
	
	private final ShiftReduceParseTreeLogger<DI, MR>		logger;
	
	/** penalty for SKIP operations */
//	private final double 									gamma;
	
	private Predicate<ParsingOp<MR>> 						datasetCreatorFilter;
	
	private final List<AbstractNonLocalFeature<MR>> 		nonLocaFeatures;
	
//	static {
//        Nd4j.dtype = DataBuffer.Type.DOUBLE;
//        NDArrayFactory factory = Nd4j.factory();
//        factory.setDType(DataBuffer.Type.DOUBLE);
//    }
	
	private boolean disablePacking;
	
	/** TODO -- separate learning parts from other components */
	public NeuralShiftReduceParser(int beamSize,
			ShiftReduceBinaryParsingRule<MR>[] binaryRules, ILexicalRule<MR> lexicalRule, 
			List<ISentenceLexiconGenerator<DI, MR>> sentenceLexiconGenerators,
			List<ISentenceLexiconGenerator<DI, MR>> sloppyLexicalGenerators,
			ICategoryServices<MR> categoryServices, 
			IFilter<Category<MR>> completeParseFilter,
			ShiftReduceUnaryParsingRule<MR>[] unaryRules, double learningRate, 
			double learningRateDecay, double l2, double gamma, int seed, File outputDir) {
		
		LOG.setCustomLevel(LogLevel.INFO);
		
		Nd4j.getRandom().setSeed(seed);
		
		//Non-local features
		this.nonLocaFeatures = new ArrayList<AbstractNonLocalFeature<MR>>();
		this.nonLocaFeatures.add(new StackSyntaxFeatures<MR>());
//		this.nonLocaFeatures.add(new PreviousRuleFeature<MR>());
//		this.nonLocaFeatures.add(new AdjacentTemplateFeature.Builder<MR>().build());
//		this.nonLocaFeatures.add(new PreviousTreeRootAttribute<MR>());
		
		final Map<String, Integer> tagsAndDimension = new HashMap<String, Integer>();
		tagsAndDimension.put("ATTACH", 32);
		tagsAndDimension.put("CROSS", 16);//2);
		tagsAndDimension.put("DYN", 8);
		tagsAndDimension.put("DYNSKIP", 8);//2);
		tagsAndDimension.put("LOGEXP", 8);//2);
		tagsAndDimension.put("SLOPPYLEX", 16);//2);
		tagsAndDimension.put("SHIFT", 16);
		
		//joint features
		tagsAndDimension.put("SHIFTSEM", 32);
		tagsAndDimension.put("ATTRIBPOS", 32);
		tagsAndDimension.put("AMRLEX", 48);
		
		//disjoint features
		tagsAndDimension.put("POS", 12); //16 earlier
//		tagsAndDimension.put("SEMHEAD", 12);
//		tagsAndDimension.put("FACLEX", 32);
//		tagsAndDimension.put("ATTRIB", 16);
		
//		tagsAndDimension.put("LOOKAHEADPOS", 32);
		tagsAndDimension.put("STEPRULE", 16);
		tagsAndDimension.put("TEMPLATELEFTPOS", 32);
		tagsAndDimension.put("TEMPLATERIGHTPOS", 32);
		
		tagsAndDimension.put("NEXT1POS", 12);
		tagsAndDimension.put("NEXT2POS", 12);
//		tagsAndDimension.put("NEXT3POS", 12);
		tagsAndDimension.put("PREV1POS", 12);
		tagsAndDimension.put("PREV2POS", 12);
		
//		tagsAndDimension.put("PREVTEMPLATE", 32);
//		tagsAndDimension.put("PREVRULE", 12);
		
//		tagsAndDimension.put("PREVROOTATTRIB", 12);
//		tagsAndDimension.put("SNDPREVROOTATTRIB", 12);
		
//		tagsAndDimension.put("DYNORIGIN", 6);
		
//		tagsAndDimension.put("STACKSYNTAX1", 32);
//		tagsAndDimension.put("STACKSYNTAX2", 32);
		tagsAndDimension.put("STACKSYNTAX1", 24);
		tagsAndDimension.put("STACKSYNTAX2", 24);
		tagsAndDimension.put("STACKSYNTAX3", 24);
		tagsAndDimension.put("STACKATTRIB1", 12);
		tagsAndDimension.put("STACKATTRIB2", 12);
		tagsAndDimension.put("STACKATTRIB3", 12);
		
//		tagsAndDimension.put("LEXICALWORD", 48);
//		tagsAndDimension.put("ADJACENTWORD1", 48);
//		tagsAndDimension.put("ADJACENTWORD2", 48);
		
		int nIn = 0;
		
		for(int dim: tagsAndDimension.values()) {
			nIn = nIn + dim;
		}
		
		this.mlpScorer = new NeuralParsingStepScorer(nIn, learningRate, l2, seed);
		this.featureEmbedding = new FeatureEmbedding<MR>(learningRate, l2, tagsAndDimension, outputDir);
		
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
		
		this.testing = false;
		
		this.logger = new ShiftReduceParseTreeLogger<DI, MR>(outputDir);
		
		/////TEMPORARY
		this.modelNewFeatures = null;
		
		this.pruningFilter = new Predicate<ParsingOp<MR>>() {
			public boolean test(ParsingOp<MR> e) {
				return true;
			}
		};
		
//		this.numRules = 1 + this.unaryRules.length + this.binaryRules.length;
		
	
		LOG.info("Parser Init :: %s: ... sloppyLexicalGenerator=%s ...", getClass(),
				sloppyLexicalGenerators);
		LOG.info("Parser Init :: %s: ... binary rules=%s ...", getClass(),
				Arrays.toString(binaryRules));
		LOG.info("Parser Init :: %s: ... unary rules=%s ...", getClass(),
				Arrays.toString(unaryRules));
		LOG.info("Neural Feed Forward Parser. Number of tags %s, nIn %s, Gamma %s", 
								tagsAndDimension.size(), nIn, gamma);
		LOG.info(".. outputDir %s", outputDir.getAbsolutePath());
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
	
	public NeuralParsingStepScorer getMLPScorer() {
		return this.mlpScorer;
	}
	
	public FeatureEmbedding<MR> getFeatureEmbedding() {
		return this.featureEmbedding;
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
		
		if(pruningFilter instanceof CKYSingleParseTreeParsingFilter) {
			return ((CKYSingleParseTreeParsingFilter<MR>)pruningFilter).test(op);
		} else if (pruningFilter instanceof CKYMultiParseTreeParsingFilter) {
			return ((CKYMultiParseTreeParsingFilter<MR>)pruningFilter).test(op, dstate);
		} else {
			return pruningFilter.test(op);
		}
	}
	
	private void register(Predicate<ParsingOp<MR>> pruningFilter, ParsingOp<MR> op, 
										DerivationState<MR> dstate, DerivationState<MR> dNew) {
		
		if (pruningFilter instanceof CKYMultiParseTreeParsingFilter) {
			((CKYMultiParseTreeParsingFilter<MR>)pruningFilter).register(op, dstate, dNew);
		}
	}
	
	private void computeNonLocalFeatures(DerivationState<MR> state, IParseStep<MR> parseStep, 
									IHashVector features, String[] buffer, int bufferIndex, String[] tags) {
		
		for(AbstractNonLocalFeature<MR> nonLocalFeature: this.nonLocaFeatures) {
			nonLocalFeature.add(state, parseStep, features, buffer, bufferIndex, tags);
		}
	}
	
	@Override
	public void enablePacking() {
		this.disablePacking = false;
	}
	
	@Override
	public void disablePacking() {
		this.disablePacking = true;
	}
	
	@Override
	public void setDatasetCreatorFilter(Predicate<ParsingOp<MR>> datasetCreatorFilter) {
		this.datasetCreatorFilter = datasetCreatorFilter;
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
	public IGraphParserOutput<MR> parse(DI dataItem, Predicate<ParsingOp<MR>> pruningFilter, IDataItemModel<MR> model_,
			boolean allowWordSkipping, ILexiconImmutable<MR> tempLexicon, Integer beamSize_) {
		
		final long start = System.currentTimeMillis();
		
//		LOG.setCustomLevel(LogLevel.DEBUG);
//		DerivationState.LOG.setCustomLevel(LogLevel.DEBUG);
		
//		this.featureEmbedding.profile();
//		System.exit(0);
		
		AtomicLong s1 = new AtomicLong();
		AtomicLong s2 = new AtomicLong();
		AtomicLong s3 = new AtomicLong();
		
		if(this.modelNewFeatures != null) {
			model_ = (IDataItemModel<MR>) this.modelNewFeatures.createDataItemModel((SituatedSentence<AMRMeta>) dataItem);
			LOG.info("Created model");
		}
		
		final IDataItemModel<MR> model = model_;
	
		final Integer beamSize;
		if(beamSize_ == null) {
			beamSize = this.beamSize;
		} else {
			beamSize = beamSize_;
		}
		
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
		final String[] buffer = tk.subArray(0, n);
		final String[] tags = ((SituatedSentence<AMRMeta>) dataItem).getState().getTags().subArray(0, n);
		
		List<DerivationState<MR>> completeParseTrees = new LinkedList<DerivationState<MR>>();
		
		List<DirectAccessBoundedPriorityQueue<PackedState<MR>>> beam = new 
						LinkedList<DirectAccessBoundedPriorityQueue<PackedState<MR>>>();
		List<DirectAccessBoundedPriorityQueue<PackedState<MR>>> newBeam = new 
						LinkedList<DirectAccessBoundedPriorityQueue<PackedState<MR>>>();
		
		for(int i=0; i<=n; i++) { //a beam for different number of words consumed 
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
				
		while(!isEmpty) {
			LOG.debug("=========== CYCLE %s =============", ++cycle);
			Iterator<DirectAccessBoundedPriorityQueue<PackedState<MR>>> iterBeam = beam.iterator();
			int ibj = 0;
					
			while(iterBeam.hasNext()) {
				LOG.debug("### Working on the beam %s ###", ++ibj);
				final DirectAccessBoundedPriorityQueue<PackedState<MR>> pstates = iterBeam.next();
				
				List<PackedState<MR>> allPStates = new LinkedList<PackedState<MR>>(pstates);
				
				final int size = pstates.size();
				final int numBatches = (int) Math.ceil(size/(double)numCores);
				
				for(int i = 0; i < numBatches; i++) {
					
				final int startIndex = numCores * i;
				final int endIndex = Math.min(startIndex + numCores, size);
				List<PackedState<MR>> thisBatchPStates = allPStates.subList(startIndex, endIndex);
					
				StreamSupport.stream(Spliterators.spliterator(thisBatchPStates/*pstates*/, Spliterator.IMMUTABLE), 
									LOG.getLogLevel() == LogLevel.DEBUG ? false : true)
					    .forEach(pstate -> { 
					/* perform valid shift and reduce operations for this packed states, if the packed state is
					 * finished and already represents a complete parse tree then save it separately. 
					 * 
					 * probability/score of a feature is given by: 
		             * exp{ F(x, y) } / \sum_y' exp { F(x, y') } where F(.) is a MLP */
					
					/* Important: Since features are currently local that is only look at the root categories or 
					 * at the shifted lexical entries. Hence, operations can be performed on the best state
					 * currently in the packed state. This will NO LONGER HOLD if features start looking 
					 * at the complete tree segments in the state. --- and yes it does not hold here */
					    	
			    	DerivationState<MR> dstate = pstate.getBestState();
					final int wordsConsumed = dstate.wordsConsumed;
					int childIndex = 0;
					
					List<ParsingOp<MR>> possibleActions = new LinkedList<ParsingOp<MR>>();
					List<IHashVector> possibleActionFeatures = new LinkedList<IHashVector>();
					
					//list of new potential states and the step that created them.
					List<Pair<DerivationState<MR>, AbstractShiftReduceStep<MR>>> newStateAndStep = 
									new ArrayList<Pair<DerivationState<MR>, AbstractShiftReduceStep<MR>>>();
					List<IHashVector> features = new ArrayList<IHashVector>();
					
					//Operation 1: Shift operation: shift a token and its lexical entry to this stack
					if(wordsConsumed < n) {
						
						for(int words = 1; words <= n - dstate.wordsConsumed; words++) {
							
							Iterator<LexicalResult<MR>> lexicalResults = this.lexicalRule.apply(tk.sub(dstate.wordsConsumed, 
									dstate.wordsConsumed + words), new SentenceSpan(wordsConsumed, wordsConsumed + words - 1, n),
									compositeLexicon);
							
							while(lexicalResults.hasNext()/*lexicalEntries.hasNext()*/) {
								LexicalResult<MR> lexicalResult = lexicalResults.next();
								LexicalEntry<MR> lexicalEntry = lexicalResult.getEntry();//lexicalEntries.next();
								
								SentenceSpan span = new SentenceSpan(wordsConsumed, wordsConsumed + words, n);
								ParsingOp<MR> op = new LexicalParsingOp<MR>(lexicalResult.getResultCategory(), span, 
																	ShiftReduceLexicalStep.LEXICAL_DERIVATION_STEP_RULENAME, lexicalEntry);
								
								if(pruningFilter != null && !this.test(pruningFilter, op, dstate)) {
									continue; 
								}
								
								DerivationState<MR> dNew = dstate.shift(lexicalResult/*lexicalEntry*/, words, span);
								
								//check this full line below
								boolean full = (n == words) && this.completeParseFilter.test(
																			dNew.returnLastNonTerminal().getCategory());
								
								ShiftReduceLexicalStep<MR> lexicalStep1 = new ShiftReduceLexicalStep<MR>(lexicalResult.getResultCategory()
							/*lexicalEntry.getCategory()*/, lexicalEntry, full, dstate.wordsConsumed, dstate.wordsConsumed + words - 1);
								
								ShiftReduceLexicalStep<MR> lexicalStep = new ShiftReduceLexicalStep<MR>(lexicalResult.getResultCategory()
							/*lexicalEntry.getCategory()*/, lexicalEntry, full, dstate.wordsConsumed, dstate.wordsConsumed + words);
								
								dNew.childIndex = childIndex++;
								possibleActions.add(op);	
								IHashVector feature = model.computeFeatures(lexicalStep1);
								this.computeNonLocalFeatures(dstate, lexicalStep1, feature, buffer, dstate.wordsConsumed, tags);
								
								possibleActionFeatures.add(feature);
								
								ParsingOp<MR> passedOp = null;
								
								if(!this.testing && this.datasetCreatorFilter != null && 
										!this.test(this.datasetCreatorFilter, op, dstate)) {
									
									//////// Temporary ///////////////////
									// Due to a bug in the code, the intermediate categories do not have word overloading.
									// So if there is a lex+unary overloading then the lex intermediate will be without words
									// and hence match to op will fail. However we can match against lexicalEntry category.
									// for now. But this is really an ugly hack and should be fixed in the future.
									
									ParsingOp<MR> op1 = new LexicalParsingOp<MR>(lexicalEntry.getCategory(), span, 
											ShiftReduceLexicalStep.LEXICAL_DERIVATION_STEP_RULENAME, lexicalEntry);
									
									if(!this.test(this.datasetCreatorFilter, op1, dstate)) {
										continue;
									} else passedOp = op1;
								} else passedOp = op;
								
								//////// Cannot skip more than one word /////////////////
								if(words > 1 && op.getCategory().getSyntax().equals(SimpleSyntax.EMPTY)) {
									continue;
								}
								
								newStateAndStep.add(Pair.of(dNew, lexicalStep));
								features.add(feature);
								
								dNew.calcDebugHashCode();
									
								this.register(this.datasetCreatorFilter, passedOp, dstate, dNew);
								
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
							
									ShiftReduceParseStep<MR> step = new ShiftReduceParseStep<MR>(dNew.returnLastNonTerminal().getCategory(),
											children, full, true, name, lastSpan.getStart(), lastSpan.getEnd());
									
									dNew.childIndex = childIndex++;
									possibleActions.add(op);
									IHashVector feature = model.computeFeatures(step1);
									this.computeNonLocalFeatures(dstate, step1, feature, buffer, dstate.wordsConsumed, tags);
									possibleActionFeatures.add(feature);
									
									if(!this.testing && this.datasetCreatorFilter != null &&
											!this.test(this.datasetCreatorFilter, op, dstate)) {
										continue;
									}
									
									newStateAndStep.add(Pair.of(dNew, step));
									features.add(feature);
									
									dNew.calcDebugHashCode();
									this.register(this.datasetCreatorFilter, op, dstate, dNew);
									
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
									
									ShiftReduceParseStep<MR> step = new ShiftReduceParseStep<MR>(dNew.returnLastNonTerminal().getCategory(),
											dstate.returnBothCategories(), full, false, name, joined_.getStart(), joined_.getEnd());
									
									dNew.childIndex = childIndex++;
									possibleActions.add(op);
									IHashVector feature = model.computeFeatures(step1);
									this.computeNonLocalFeatures(dstate, step1, feature, buffer, dstate.wordsConsumed, tags);
									possibleActionFeatures.add(feature);
									
									if(!this.testing && this.datasetCreatorFilter != null && 
											!this.test(this.datasetCreatorFilter, op, dstate)) {
										continue;
									}
									
									newStateAndStep.add(Pair.of(dNew, step));
									features.add(feature);
									
									dNew.calcDebugHashCode();
									this.register(this.datasetCreatorFilter, op, dstate, dNew);
									
									LOG.debug("Generated %s; Binary-Reduce %s %s %s; ", dNew.getDebugHashCode(), 
																				logical, name, dstate.getDebugHashCode());
								}
							}
						}
					}
					
					if(features.size() == 0) {
						return; //do check if return really works with streams
					}
					
					//Create a batch
					final long start1 = System.currentTimeMillis();
					INDArray batch = this.featureEmbedding.embedFeatures(features).first();
					
					//Pass it through the MLP
					final long start2 = System.currentTimeMillis();
					double[] exponents = this.mlpScorer.getEmbeddingParallel(batch);
					
					//Calculate log-softmax on the MLP outputs		
					final long start3 = System.currentTimeMillis();
					double[] logSoftmax = this.mlpScorer.toLogSoftMax(exponents);
					final long start4 = System.currentTimeMillis();
					
					s1.addAndGet(start2 - start1);
					s2.addAndGet(start3 - start2);
					s3.addAndGet(start4 - start3);
					
					//normalize the probabilities and add them to the list
					Iterator<Pair<DerivationState<MR>, AbstractShiftReduceStep<MR>>> it = newStateAndStep.iterator();
					
					int ix = 0;
					while(it.hasNext()) {
						Pair<DerivationState<MR>, AbstractShiftReduceStep<MR>> next_ = it.next();
						final DerivationState<MR> dNew = next_.first();
						final AbstractShiftReduceStep<MR> step = next_.second(); 
						
						final double stepScore = logSoftmax[ix++];
						dNew.score = dstate.score + stepScore; //log-likelihood
						
						final IWeightedShiftReduceStep<MR> weightedStep;
						
						//Since we compute the features, we can probably store the features in Weighted step too
						if(step instanceof ShiftReduceLexicalStep<?>) {
							weightedStep = new WeightedShiftReduceLexicalStep<MR>(
														(ShiftReduceLexicalStep<MR>)step, stepScore/*dNew.score*/);
						} else {
							weightedStep = new WeightedShiftReduceParseStep<MR>(
														(ShiftReduceParseStep<MR>)step, stepScore/*dNew.score*/);
						}
						
						LOG.debug("Score %s; state %s; step %s", dNew.getDebugHashCode(), dNew.score, stepScore); 
						
						dNew.defineStep(weightedStep);
						dNew.setPossibleActions(possibleActions);
						dNew.setPossibleActionsFeatures(possibleActionFeatures);
						
						if(possibleActions.size() != possibleActionFeatures.size()) {
							throw new RuntimeException("Possible action is not same as possible action features");
						}
						
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
				}); }
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
			
			/*{
				Iterator<PackedState<MR>> it = beam.get(0).iterator();
				List<PackedState<MR>> states = new ArrayList<PackedState<MR>>();
				while(it.hasNext()) {
					states.add(it.next());
				}
				
				final Comparator<PackedState<MR>> dStateCmpR  = new Comparator<PackedState<MR>>() {
					public int compare(PackedState<MR> left, PackedState<MR> right) {
		        		return Double.compare(right.getBestScore(), left.getBestScore()); 
		    		}   
				};
				
				Collections.sort(states, dStateCmpR);
				
				int index = 0;
				LOG.info("=========== CYCLE %s =============", cycle);
	
				for(PackedState<MR> pstate: states) {
					DerivationState<MR> state = pstate.getBestState();
					String hashes = "Generated " + state.getDebugHashCode() + " from " + state.getParent().getDebugHashCode();
					String scores = " score " + state.score + "; step " + (state.score - state.getParent().score) + "; ";
					String scoreAndHash = scores + ", " + hashes;
					LOG.info("%s, %s; %s, %s; %s", ++index, scoreAndHash, state.possibleActions().size(), 
								state.possibleActions().get(state.childIndex), 
								state.possibleActionFeatures().get(state.childIndex));
				}
				
				for(int i = 0; i < states.size(); i++) {
					for(int j = i + 1; j < states.size(); j++) {
						DerivationState<MR> d1 = states.get(i).getBestState();
						DerivationState<MR> d2 = states.get(j).getBestState();
						
						double gap = Math.abs(d1.score - d2.score);
						ParsingOp<MR> op1 = d1.possibleActions().get(d1.childIndex);
						ParsingOp<MR> op2 = d2.possibleActions().get(d2.childIndex);
						if(op1.equals(op2) && gap > 0.01) {
							if(d1.getParent().equals(d2.getParent())) {
								LOG.info("Action 1 %s op %s", d1.getDebugHashCode(), op1);
								LOG.info("Action 2 %s op %s", d2.getDebugHashCode(), op2);
								throw new RuntimeException("Same actions dont have same score even though features are local");
							}
						}
					}
				}
			}*/
			
			//Update the cursor of the filter if it is a CKY Single or Multi Parse Tree filter
			if(this.datasetCreatorFilter instanceof CKYSingleParseTreeParsingFilter) {
				((CKYSingleParseTreeParsingFilter<MR>)this.datasetCreatorFilter).incrementCursor();
				LOG.debug("Updated CKY Single Parse Tree cursor");
			} else if (this.datasetCreatorFilter instanceof CKYMultiParseTreeParsingFilter) {
				((CKYMultiParseTreeParsingFilter<MR>)this.datasetCreatorFilter).incrementCursor();
				LOG.debug("Updated CKY Multi Parse Tree cursor");
			}
		}
		
		final ShiftReduceParserOutput<MR> output = new ShiftReduceParserOutput<MR>(
										completeParseTrees, System.currentTimeMillis() - start);
		
		LOG.info("Neural Shift Reduce: Number of distinct derivations %s", output.getAllDerivations().size());
		this.featureEmbedding.stats(); //tells about number of unseen features etc.
		LOG.info("S1 %s, S2 %s, S3 %s", s1.get(), s2.get(), s3.get());
		
//		if(tk.toString().toLowerCase().contains("principal cause is the use of")) { //Temporary
//			LOG.info("Going to store feature embeddings");
//			this.featureEmbedding.store();
//		}
		
		if(this.testing) { //Ugly hack for knowing that we are testing
			this.logger.log(output, dataItem, allowWordSkipping);
		}
		
		return output;
	}
	
	/** Same as the parser excepts finds places where the parser makes mistake. The parser is allowed to proceed,
	 * however instead of using a gold pruning filter that allows it to create valid parse trees; we let parser takes decision
	 * and throw an error whenever the parser takes a decision that does not follow any parse tree. For this to work, make sure
	 * to use a smaller beam than normally used for training. */
	public IGraphParserOutput<MR> parserCatchEarlyErrors(DI dataItem, Predicate<ParsingOp<MR>> validAmrParsingFilter, IDataItemModel<MR> model_,
			boolean allowWordSkipping, ILexiconImmutable<MR> tempLexicon, Integer beamSize_) {
		
		if(this.modelNewFeatures != null) {
			model_ = (IDataItemModel<MR>) this.modelNewFeatures.createDataItemModel((SituatedSentence<AMRMeta>) dataItem);
			LOG.info("Created model");
		}
		
		final IDataItemModel<MR> model = model_;
		
		final long start = System.currentTimeMillis();
		
		final Integer beamSize;
		if(beamSize_ == null) {
			beamSize = this.beamSize;
		} else {
			beamSize = beamSize_;
		}
		
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
		
		List<DerivationState<MR>> completeParseTrees = new LinkedList<DerivationState<MR>>();
		
		List<DirectAccessBoundedPriorityQueue<PackedState<MR>>> beam = new 
						LinkedList<DirectAccessBoundedPriorityQueue<PackedState<MR>>>();
		List<DirectAccessBoundedPriorityQueue<PackedState<MR>>> newBeam = new 
						LinkedList<DirectAccessBoundedPriorityQueue<PackedState<MR>>>();
		
		for(int i=0; i<=n; i++) { //a beam for different number of words consumed 
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
				
		while(!isEmpty) {
			LOG.debug("=========== CYCLE %s =============", ++cycle);
			Iterator<DirectAccessBoundedPriorityQueue<PackedState<MR>>> iterBeam = beam.iterator();
			int ibj = 0;
			
			 //list of non-tained dstates
			List<DerivationState<MR>> goodDStates = Collections.synchronizedList(new LinkedList<DerivationState<MR>>());
					
			while(iterBeam.hasNext()) {
				LOG.debug("### Working on the beam %s ###", ++ibj);
				final DirectAccessBoundedPriorityQueue<PackedState<MR>> pstates = iterBeam.next();
				
				List<PackedState<MR>> allPStates = new LinkedList<PackedState<MR>>(pstates);
				
				final int size = pstates.size();
				final int numBatches = (int) Math.ceil(size/(double)numCores);
				
				for(int i = 0; i < numBatches; i++) {
					
				final int startIndex = numCores * i;
				final int endIndex = Math.min(startIndex + numCores, size);
				List<PackedState<MR>> thisBatchPStates = allPStates.subList(startIndex, endIndex);
					
				StreamSupport.stream(Spliterators.spliterator(thisBatchPStates/*pstates*/, Spliterator.IMMUTABLE), 
									LOG.getLogLevel() == LogLevel.DEBUG ? false : true)
					    .forEach(pstate -> { 
					/* perform valid shift and reduce operations for this packed states, if the packed state is
					 * finished and already represents a complete parse tree then save it separately. 
					 * 
					 * probability/score of a feature is given by: 
		             * exp{ F(x, y) } / \sum_y' exp { F(x, y') } where F(.) is a MLP */
					
					/* Important: Since features are currently local that is only look at the root categories or 
					 * at the shifted lexical entries. Hence, operations can be performed on the best state
					 * currently in the packed state. This will NO LONGER HOLD if features start looking 
					 * at the complete tree segments in the state. --- and yes it does not hold here */
					    	
			    	DerivationState<MR> dstate = pstate.getBestState();
					final int wordsConsumed = dstate.wordsConsumed;
					
					List<ParsingOp<MR>> possibleActions = new LinkedList<ParsingOp<MR>>();
					List<IHashVector> possibleActionFeatures = new LinkedList<IHashVector>();
					
					//list of new potential states and the step that created them.
					List<Pair<DerivationState<MR>, AbstractShiftReduceStep<MR>>> newStateAndStep = 
									new ArrayList<Pair<DerivationState<MR>, AbstractShiftReduceStep<MR>>>();
					List<IHashVector> features = new ArrayList<IHashVector>();
					
					int childIndex = 0;
					
					//Operation 1: Shift operation: shift a token and its lexical entry to this stack
					if(wordsConsumed < n) {
						
						for(int words = 1; words <= n - dstate.wordsConsumed; words++) {
							
							Iterator<LexicalResult<MR>> lexicalResults = this.lexicalRule.apply(tk.sub(dstate.wordsConsumed, 
									dstate.wordsConsumed + words), new SentenceSpan(wordsConsumed, wordsConsumed + words - 1, n),
									compositeLexicon);
							
							while(lexicalResults.hasNext()/*lexicalEntries.hasNext()*/) {
								LexicalResult<MR> lexicalResult = lexicalResults.next();
								LexicalEntry<MR> lexicalEntry = lexicalResult.getEntry();//lexicalEntries.next();
								
								SentenceSpan span = new SentenceSpan(wordsConsumed, wordsConsumed + words, n);
								ParsingOp<MR> op = new LexicalParsingOp<MR>(lexicalResult.getResultCategory(), span, 
																	ShiftReduceLexicalStep.LEXICAL_DERIVATION_STEP_RULENAME, lexicalEntry);
								
								if(pruningFilter != null && !this.test(pruningFilter, op, dstate)) {
									continue; 
								}
								
								DerivationState<MR> dNew = dstate.shift(lexicalResult/*lexicalEntry*/, words, span);
								
								//check this full line below
								boolean full = (n == words) && this.completeParseFilter.test(
																			dNew.returnLastNonTerminal().getCategory());
								
								ShiftReduceLexicalStep<MR> lexicalStep1 = new ShiftReduceLexicalStep<MR>(lexicalResult.getResultCategory()
							/*lexicalEntry.getCategory()*/, lexicalEntry, full, dstate.wordsConsumed, dstate.wordsConsumed + words - 1);
								
								ShiftReduceLexicalStep<MR> lexicalStep = new ShiftReduceLexicalStep<MR>(lexicalResult.getResultCategory()
							/*lexicalEntry.getCategory()*/, lexicalEntry, full, dstate.wordsConsumed, dstate.wordsConsumed + words);
								
								possibleActions.add(op);	
								IHashVector feature = model.computeFeatures(lexicalStep1);
								possibleActionFeatures.add(feature);
								dNew.childIndex = childIndex++;
								
								ParsingOp<MR> passedOp = null;
								
								boolean isTainted = false;
								if(!this.testing && this.datasetCreatorFilter != null && 
										!this.test(this.datasetCreatorFilter, op, dstate)) {
									
									//////// Temporary ///////////////////
									// Due to a bug in the code, the intermediate categories do not have word overloading.
									// So if there is a lex+unary overloading then the lex intermediate will be without words
									// and hence match to op will fail. However we can match against lexicalEntry category.
									// for now. But this is really an ugly hack and should be fixed in the future.
									
									ParsingOp<MR> op1 = new LexicalParsingOp<MR>(lexicalEntry.getCategory(), span, 
											ShiftReduceLexicalStep.LEXICAL_DERIVATION_STEP_RULENAME, lexicalEntry);
									
									if(!this.test(this.datasetCreatorFilter, op1, dstate)) {
										isTainted = true;
									} else passedOp = op1;
								} else passedOp = op;
								
								//////// Cannot skip more than one word /////////////////
								if(words > 1 && op.getCategory().getSyntax().equals(SimpleSyntax.EMPTY)) {
									continue;
								}
								
								newStateAndStep.add(Pair.of(dNew, lexicalStep));
								features.add(feature);
								
								dNew.calcDebugHashCode();
									
								if(!isTainted) {
									goodDStates.add(dNew);
									this.register(this.datasetCreatorFilter, passedOp, dstate, dNew);
								} else {
									dNew.tainted = true;
								}
								
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
							
									ShiftReduceParseStep<MR> step = new ShiftReduceParseStep<MR>(dNew.returnLastNonTerminal().getCategory(),
											children, full, true, name, lastSpan.getStart(), lastSpan.getEnd());
									
									possibleActions.add(op);
									IHashVector feature = model.computeFeatures(step1);
									possibleActionFeatures.add(feature);
									dNew.childIndex = childIndex++;
									
									boolean isTainted = false;
									if(!this.testing && this.datasetCreatorFilter != null &&
											!this.test(this.datasetCreatorFilter, op, dstate)) {
										isTainted = true;
									}
									
									newStateAndStep.add(Pair.of(dNew, step));
									features.add(feature);
									
									dNew.calcDebugHashCode();
									if(!isTainted) {
										goodDStates.add(dNew);
										this.register(this.datasetCreatorFilter, op, dstate, dNew);
									} else {
										dNew.tainted = true;
									}
									
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
									
									ShiftReduceParseStep<MR> step = new ShiftReduceParseStep<MR>(dNew.returnLastNonTerminal().getCategory(),
											dstate.returnBothCategories(), full, false, name, joined_.getStart(), joined_.getEnd());
									
									possibleActions.add(op);
									IHashVector feature = model.computeFeatures(step1);
									possibleActionFeatures.add(feature);
									dNew.childIndex = childIndex++;
									
									boolean isTainted = false;
									if(!this.testing && this.datasetCreatorFilter != null && 
											!this.test(this.datasetCreatorFilter, op, dstate)) {
										isTainted = true;
									}
									
									newStateAndStep.add(Pair.of(dNew, step));
									features.add(feature);
									
									dNew.calcDebugHashCode();
									if(!isTainted) {
										goodDStates.add(dNew);
										this.register(this.datasetCreatorFilter, op, dstate, dNew);
									} else {
										dNew.tainted = true;
									}
									
									LOG.debug("Generated %s; Binary-Reduce %s %s %s; ", dNew.getDebugHashCode(), 
																				logical, name, dstate.getDebugHashCode());
								}
							}
						}
					}
					
					if(features.size() == 0) {
						return; //do check if return really works with streams
					}
					
					//Create a batch
					INDArray batch = this.featureEmbedding.embedFeatures(features).first();
					
					//Pass it through the MLP
					double[] exponents = this.mlpScorer.getEmbeddingParallel(batch);
					
					//Calculate log-softmax on the MLP outputs					
					double[] logSoftmax = this.mlpScorer.toLogSoftMax(exponents);
					
					//normalize the probabilities and add them to the list
					Iterator<Pair<DerivationState<MR>, AbstractShiftReduceStep<MR>>> it = newStateAndStep.iterator();
					
					int ix = 0;
					while(it.hasNext()) {
						Pair<DerivationState<MR>, AbstractShiftReduceStep<MR>> next_ = it.next();
						final DerivationState<MR> dNew = next_.first();
						final AbstractShiftReduceStep<MR> step = next_.second(); 
						
						dNew.score = dstate.score + logSoftmax[ix++]; //log-likelihood
						
						final IWeightedShiftReduceStep<MR> weightedStep;
						
						//Since we compute the features, we can probably store the features in Weighted step too
						if(step instanceof ShiftReduceLexicalStep<?>) {
							weightedStep = new WeightedShiftReduceLexicalStep<MR>(
														(ShiftReduceLexicalStep<MR>)step, dNew.score);
						} else {
							weightedStep = new WeightedShiftReduceParseStep<MR>(
														(ShiftReduceParseStep<MR>)step, dNew.score);
						}
						
						dNew.defineStep(weightedStep);
						dNew.setPossibleActions(possibleActions);
						dNew.setPossibleActionsFeatures(possibleActionFeatures);
						
						if(possibleActions.size() != possibleActionFeatures.size()) {
							throw new RuntimeException("Possible action is not same as possible action features");
						}
						
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
				}); }
			}
			
			
			Iterator<DirectAccessBoundedPriorityQueue<PackedState<MR>>> beamIter = beam.iterator();
			Iterator<DirectAccessBoundedPriorityQueue<PackedState<MR>>> nBeamIter = newBeam.iterator();
			
			int k = 0;
			isEmpty = true;
			boolean allAreTainted = true;
			
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
					//beam_.offer(ds_);
					//isEmpty = false;
					
					if(!ds_.getBestState().tainted) { //replace by checking all states
						allAreTainted = false;
						beam_.offer(ds_);
						isEmpty = false;
					}
				}
				nBeam_.clear(); //clear the new stack
			}
			
			
//			for(DerivationState<MR> goodDState: goodDStates) {
//				LOG.info("Parsing op %s", goodDState.returnLastNonTerminal().getCategory());
//			}
//			LOG.info("======");
			
			if(allAreTainted && !isEmpty) { //there is no derivation state that is not tainted. This is an early error.
				//none of the new derivation states that we consider, are now continuing oracle parse trees
				//there would have been derivation states earlier in the parsing history that were continuing
				//the oracle parse trees but fell of the beam. However, the states that we lost this time were
				//able to continue for the longest duration in the beam, hence these states are more useful for debugging.
				//as they probably require fewer corrections than other.
				
				for(DerivationState<MR> goodDState: goodDStates) {
					//print score, buffer when it was created, the features that were  active for this state
					final DerivationState<MR> parent = goodDState.getParent();
					final String buffer = tk.sub(parent.wordsConsumed, tk.size()).toString();
					final int childIndex = goodDState.childIndex;
					final IHashVector activeFeature = goodDState.possibleActionFeatures().get(childIndex);
					final ParsingOp<MR> op = goodDState.possibleActions().get(childIndex);
					LOG.info("Good DState: Score %s, buffer %s", goodDState.score,  buffer);
					LOG.info("\t possible action %s", op);
					Iterator<Pair<KeyArgs, Double>> it = activeFeature.iterator();
					while(it.hasNext()) {
						LOG.info("\t %s", it.next().first());
					}
				}
				
				return null;
			}
			
			//Update the cursor of the filter if it is a CKY Single or Multi Parse Tree filter
			if(this.datasetCreatorFilter instanceof CKYSingleParseTreeParsingFilter) {
				((CKYSingleParseTreeParsingFilter<MR>)this.datasetCreatorFilter).incrementCursor();
				LOG.debug("Updated CKY Single Parse Tree cursor");
			} else if (this.datasetCreatorFilter instanceof CKYMultiParseTreeParsingFilter) {
				((CKYMultiParseTreeParsingFilter<MR>)this.datasetCreatorFilter).incrementCursor();
				LOG.debug("Updated CKY Multi Parse Tree cursor");
			}
		}
		
		final ShiftReduceParserOutput<MR> output = new ShiftReduceParserOutput<MR>(
										completeParseTrees, System.currentTimeMillis() - start);
		
		LOG.info("Neural Shift Reduce: Number of distinct derivations %s", output.getAllDerivations().size());
		this.featureEmbedding.stats(); //tells about number of unseen features etc.
		
//		if(tk.toString().toLowerCase().contains("principal cause is the use of")) { //Temporary
//			LOG.info("Going to store feature embeddings");
//			this.featureEmbedding.store();
//		}
		
		if(this.testing) { //Ugly hack for knowing that we are testing
			this.logger.log(output, dataItem, allowWordSkipping);
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
		public NeuralShiftReduceParser<DI, MR> build() {
			return new NeuralShiftReduceParser<DI, MR>(this.beamSize,
					binaryRules.toArray((ShiftReduceBinaryParsingRule<MR>[]) Array
							.newInstance(ShiftReduceBinaryParsingRule.class,
									binaryRules.size())), lexicalRule, 
					sentenceLexicalGenerators, sloppyLexicalGenerators,
					categoryServices, completeParseFilter,
					unaryRules.toArray((ShiftReduceUnaryParsingRule<MR>[]) Array
							.newInstance(ShiftReduceUnaryParsingRule.class,
									unaryRules.size())), 
					learningRate, learningRateDecay, l2, gamma, seed, outputDir);
		}
		
		public Builder<DI, MR> setSeed(int seed) {
			this.seed = seed;
			return this;
		}
		
		public Builder<DI, MR> setLearningRate(double learningRate) {
			this.learningRate = learningRate;
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
	
	public static class Creator<DI extends Sentence, MR> implements IResourceObjectCreator<NeuralShiftReduceParser<DI, MR>> {

		private final String type;
		
		public Creator() {
			this("parser.feedforward.neural.shiftreduce");
		}

		public Creator(String type) {
			this.type = type;
		}
		
		@SuppressWarnings("unchecked")
		@Override
		public NeuralShiftReduceParser<DI, MR> create(Parameters params, IResourceRepository repo) {
			
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
