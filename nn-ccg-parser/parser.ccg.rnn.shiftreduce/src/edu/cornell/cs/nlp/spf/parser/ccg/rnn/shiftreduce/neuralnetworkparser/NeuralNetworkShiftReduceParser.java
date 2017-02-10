package edu.cornell.cs.nlp.spf.parser.ccg.rnn.shiftreduce.neuralnetworkparser;

import java.io.IOException;
import java.lang.reflect.Array;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.HashSet;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;
import java.util.Random;
import java.util.Set;
import java.util.Spliterator;
import java.util.Spliterators;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;
import java.util.function.Predicate;
import java.util.stream.StreamSupport;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.Tanh;
//import org.nd4j.linalg.factory.NDArrayFactory;
import org.nd4j.linalg.factory.Nd4j;
//import org.nd4j.linalg.api.buffer.DataBuffer;

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
import edu.cornell.cs.nlp.spf.parser.ISentenceLexiconGenerator;
import edu.cornell.cs.nlp.spf.parser.ParsingOp;
import edu.cornell.cs.nlp.spf.parser.ccg.model.IDataItemModel;
import edu.cornell.cs.nlp.spf.parser.ccg.normalform.NormalFormValidator;
import edu.cornell.cs.nlp.spf.parser.ccg.normalform.unaryconstraint.UnaryConstraint;
import edu.cornell.cs.nlp.spf.parser.ccg.rnn.shiftreduce.categoryembeddings.CategoryEmbedding;
import edu.cornell.cs.nlp.spf.parser.ccg.rnn.shiftreduce.dataset.CKYMultiParseTreeParsingFilter;
import edu.cornell.cs.nlp.spf.parser.ccg.rnn.shiftreduce.dataset.CKYSingleParseTreeParsingFilter;
import edu.cornell.cs.nlp.spf.parser.ccg.rnn.shiftreduce.embeddings.EmbedActionHistory;
import edu.cornell.cs.nlp.spf.parser.ccg.rnn.shiftreduce.embeddings.EmbedParserState;
import edu.cornell.cs.nlp.spf.parser.ccg.rnn.shiftreduce.embeddings.EmbedWordBuffer;
import edu.cornell.cs.nlp.spf.parser.ccg.rnn.shiftreduce.embeddings.RecurrentTimeStepOutput;
import edu.cornell.cs.nlp.spf.parser.ccg.rnn.shiftreduce.embeddings.TopLayerMLP;
import edu.cornell.cs.nlp.spf.parser.ccg.rnn.shiftreduce.learning.ParsingOpPreTrainingDataset;
import edu.cornell.cs.nlp.spf.parser.ccg.rnn.shiftreduce.parsingopembeddings.ParsingOpEmbedding;
import edu.cornell.cs.nlp.spf.parser.ccg.rnn.shiftreduce.wordembeddings.Word2Vec;
import edu.cornell.cs.nlp.spf.parser.ccg.rnn.shiftreduce.wordembeddings.WordEmbedding;
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
import edu.cornell.cs.nlp.spf.parser.ccg.shiftreduce.sloppy.AggressiveWordSkippingLexicalGenerator;
import edu.cornell.cs.nlp.spf.parser.ccg.shiftreduce.sloppy.BackwardSkippingRule;
import edu.cornell.cs.nlp.spf.parser.ccg.shiftreduce.sloppy.ForwardSkippingRule;
import edu.cornell.cs.nlp.spf.parser.ccg.shiftreduce.sloppy.SimpleWordSkippingLexicalGenerator;
import edu.cornell.cs.nlp.spf.parser.ccg.shiftreduce.stacks.DerivationState;
import edu.cornell.cs.nlp.spf.parser.ccg.shiftreduce.stacks.DerivationStateHorizontalIterator;
import edu.cornell.cs.nlp.spf.parser.ccg.shiftreduce.stacks.LexicalParsingOp;
import edu.cornell.cs.nlp.spf.parser.ccg.shiftreduce.stacks.PackedState;
import edu.cornell.cs.nlp.spf.parser.ccg.shiftreduce.stacks.PersistentEmbeddings;
import edu.cornell.cs.nlp.spf.parser.ccg.shiftreduce.stacks.ShiftReduceRuleNameSet;
import edu.cornell.cs.nlp.spf.parser.ccg.shiftreduce.steps.AbstractShiftReduceStep;
import edu.cornell.cs.nlp.spf.parser.ccg.shiftreduce.steps.IWeightedShiftReduceStep;
import edu.cornell.cs.nlp.spf.parser.ccg.shiftreduce.steps.ShiftReduceLexicalStep;
import edu.cornell.cs.nlp.spf.parser.ccg.shiftreduce.steps.ShiftReduceParseStep;
import edu.cornell.cs.nlp.spf.parser.ccg.shiftreduce.steps.WeightedShiftReduceLexicalStep;
import edu.cornell.cs.nlp.spf.parser.ccg.shiftreduce.steps.WeightedShiftReduceParseStep;
import edu.cornell.cs.nlp.spf.parser.graph.IGraphParserOutput;
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
public class NeuralNetworkShiftReduceParser<DI extends Sentence, MR> implements AbstractNeuralShiftReduceParser<DI, MR> {

	public static final ILogger								LOG
										= LoggerFactory.create(NeuralNetworkShiftReduceParser.class);

	private static final long 								serialVersionUID = -3773578975152698556L;
	
	/** Embeds action history for a derivation state using RNN */
	private final EmbedActionHistory<MR> 					embedActionHistory;
	
	/** Embeds words on the buffer using RNN */
	private final EmbedWordBuffer 							embedWordBuffer;
	
	/** Embeds category of the state using RNN */
	private final EmbedParserState<MR> 						embedParserState;
	
	/** Embeds a parsing operation */
	private final ParsingOpEmbedding<MR> 					embedParsingOp;
	
	/** Embeds category using Recursive Neural Network */
	private final CategoryEmbedding<MR> 					embedCategory;
	
	/** Embeds a token */
	private final WordEmbedding 							embedWord;
	
	/** Next three are top layer parameters used for finding probability of taking an action. */
	private final INDArray 									A;
	private final INDArray 									b;
	private final Double[]									actionBias;
	
	/** Top layer that takes input from parsing op embedding and RNNs and outputs a scalar score */
	private final TopLayerMLP								topLayer;
	
	/** Beamsize of the parser */
	private final Integer									beamSize;
	
	/** Gamma is the additional score given to a SKIP function */
	private final Double									gamma;

	/** Binary CCG parsing rules. */
	public final ShiftReduceBinaryParsingRule<MR>[]			binaryRules;

	private final IFilter<Category<MR>>						completeParseFilter;

	private final Predicate<ParsingOp<MR>> 					pruningFilter;
	
	private final ILexicalRule<MR> 							lexicalRule;

	private final List<RuleName>							ruleNames;
	
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
	
	private final int 										numRules;
	private final INDArray[] 								binaryRulesVectors;
	
//	static {
//        Nd4j.dtype = DataBuffer.Type.DOUBLE;
//        NDArrayFactory factory = Nd4j.factory();
//        factory.setDType(DataBuffer.Type.DOUBLE);
//    }
	
	private boolean disablePacking; //TODO --- temporary for one night
	
	/* ToDo -- separate learning parts from other components */
	public NeuralNetworkShiftReduceParser(int beamSize,
			ShiftReduceBinaryParsingRule<MR>[] binaryRules, ILexicalRule<MR> lexicalRule, 
			List<ISentenceLexiconGenerator<DI, MR>> sentenceLexiconGenerators,
			List<ISentenceLexiconGenerator<DI, MR>> sloppyLexicalGenerators,
			ICategoryServices<MR> categoryServices, 
			IFilter<Category<MR>> completeParseFilter,
			ShiftReduceUnaryParsingRule<MR>[] unaryRules, double learningRate, 
			double learningRateDecay, double l2, double gamma, int seed) {
		
		LOG.setCustomLevel(LogLevel.INFO);
		
		Nd4j.getRandom().setSeed(seed);
		
		Random rnd = new Random(seed);
		int categorySeed = Math.abs(rnd.nextInt(2 * seed)) + 1000;
		
		try {
			this.embedWord = new /*Word2Vec("./dataset/GoogleNews_amr2.txt");*/Word2Vec("./dataset/vocab_word2vec.txt");
		} catch (IOException e1) {
			throw new RuntimeException("Word2vec file not found");
		}
		
		this.embedCategory = new CategoryEmbedding<MR>(learningRate, learningRateDecay,
													   l2, categorySeed);
		
		final List<RuleName> ruleNames = new LinkedList<RuleName>();
		ruleNames.add(ShiftReduceLexicalStep.LEXICAL_DERIVATION_STEP_RULENAME);
		for(int u = 0; u< unaryRules.length; u++)
			ruleNames.add(unaryRules[u].getName());
		for(int b = 0; b< binaryRules.length; b++)
			ruleNames.add(binaryRules[b].getName());
		
		int seedWordBuffer = Math.abs(rnd.nextInt(2 * seed)) + 1000;
		this.embedWordBuffer = new EmbedWordBuffer(this.embedWord, 
									learningRate, l2, seedWordBuffer);
		
		this.embedParsingOp = new ParsingOpEmbedding<MR>(this.embedCategory,
										this.embedWordBuffer, ruleNames, learningRate, l2);
		
		int seedActionHistory = Math.abs(rnd.nextInt(2 * seed)) + 1000;
		this.embedActionHistory = new EmbedActionHistory<MR>(this.embedParsingOp,
									learningRate, l2, seedActionHistory);
		
		
		int seedParserState = Math.abs(rnd.nextInt(2 * seed)) + 1000;
		this.embedParserState = new EmbedParserState<MR>(this.embedCategory, 
									learningRate, l2, seedParserState);
		
		final int parsingOpDim = this.embedParsingOp.getDimension();
		final int col = this.embedWordBuffer.getDimension() + 
						this.embedParserState.getDimension() + 
						this.embedActionHistory.getDimension();
		
		/* Initialized uniformly in [-sqrt{6/(r+c)}, sqrt{6/(r+c)}] (Glorot and Bengio 10)*/
		
		double epsilonA = 2*Math.sqrt(6.0/(double)(parsingOpDim + col));
		int seedA = Math.abs(rnd.nextInt(2 * seed)) + 1000;
		this.A = Nd4j.rand(new int[]{parsingOpDim, col}, seedA);
		this.A.subi(0.5).muli(epsilonA);
		
		double epsilonb = 2*Math.sqrt(6.0/(double)(parsingOpDim + 1));
		int seedb = Math.abs(rnd.nextInt(2 * seed)) + 1000;
		this.b = Nd4j.rand(new int[]{parsingOpDim, 1}, seedb);
		this.b.subi(0.5).muli(epsilonb);
		
		this.topLayer = new TopLayerMLP(parsingOpDim + col, learningRate, l2, seed);
		
		this.disablePacking = false;
		
		this.beamSize = beamSize;
		this.lexicalRule = lexicalRule;
		this.binaryRules = binaryRules;
		this.sentenceLexiconGenerators = sentenceLexiconGenerators;
		this.sloppyLexicalGenerators = sloppyLexicalGenerators;
		this.categoryServices = categoryServices;
		this.completeParseFilter = completeParseFilter;
		this.unaryRules = unaryRules;
		
		this.pruningFilter = new Predicate<ParsingOp<MR>>() {
			public boolean test(ParsingOp<MR> e) {
				return true;
			}
		};
		
		int pad = 1 + this.unaryRules.length; 
		this.numRules = 1 + this.unaryRules.length + this.binaryRules.length;
		this.binaryRulesVectors = new INDArray[this.binaryRules.length];
		
		for(int i = 0; i< this.binaryRules.length; i++) {
			INDArray oneHot = Nd4j.zeros(this.numRules);
			oneHot.putScalar(pad + i, 1.0);
			this.binaryRulesVectors[i] = oneHot; 
		}
		
		this.ruleNames = ruleNames;
		
		this.actionBias = new Double[this.numRules];
		Arrays.fill(this.actionBias, 0.0);
		
		this.gamma  = gamma;
		
		LOG.info("Parser Init :: %s: ... sloppyLexicalGenerator=%s ...", getClass(),
				sloppyLexicalGenerators);
		LOG.info("Parser Init :: %s: ... binary rules=%s ...", getClass(),
				Arrays.toString(binaryRules));
		LOG.info("Parser Init :: %s: ... unary rules=%s ...", getClass(),
				Arrays.toString(unaryRules));
		LOG.info("Parse : Gamma :: %s", this.gamma);
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
	
	public EmbedActionHistory<MR> getEmbedActionHistory() {
		return this.embedActionHistory;
	}
	
	public EmbedWordBuffer getEmbedWordBuffer() {
		return this.embedWordBuffer;
	}
	
	public EmbedParserState<MR> getEmbedParserState() {
		return this.embedParserState;
	}
	
	public ParsingOpEmbedding<MR> getEmbedParsingOp() {
		return this.embedParsingOp;
	}
	
	public CategoryEmbedding<MR> getCategoryEmbedding() {
		return this.embedCategory;
	}
	
	public INDArray getAffineA() {
		return this.A;
	}
	
	public INDArray getAffineb() {
		return this.b;
	}
	
	public Double[] getActionBias() {
		return this.actionBias;
	}
	
	public TopLayerMLP getTopLayer() {
		return this.topLayer;
	}
	
	public void setTopLayerParam(INDArray A, INDArray b) {
		/* use assign function */
		if(A.shape().length != this.A.shape().length) {
			throw new IllegalStateException("A param must have same shape length as this.A");
		}
		
		if(this.A.size(0) != A.size(0) || this.A.size(1) != A.size(1)) {
			throw new IllegalStateException("A's dimensions dont match the current");
		}
		
		for(int i = 0; i < this.A.size(0); i++) {
			
			for(int j = 0; j < this.A.size(1); j++) {
				this.A.putScalar(new int[]{i, j}, A.getDouble(new int[]{i, j}));
			}
		}
		
		if(b.shape().length != this.b.shape().length) {
			throw new IllegalStateException("b param must have same shape length as this.b");
		}
		
		if(this.b.size(0) != b.size(0) || this.b.size(1) != b.size(1)) {
			throw new IllegalStateException("b's dimensions dont match the current");
		}
		
		for(int i = 0; i < this.b.size(0); i++) {
			
			for(int j = 0; j < this.b.size(1); j++) {
				this.b.putScalar(new int[]{i, j}, b.getDouble(new int[]{i, j}));
			}
		}
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
	
	/** encodes the derivation state, history of actions and word buffer*/
	public INDArray encodeState(DerivationState<MR> dstate, INDArray[] sentenceEmbedding, int n) {
		
		final INDArray a1, a2, a3;
    	
	    DerivationState<MR> parent = dstate.getParent();
	    
	    if(parent == null) { //Handle the degenerate case where the parent is null
	    	//there is no category embedding or action embedding. We take them as 0.
	    	
	    	a1 = Nd4j.zeros(this.embedActionHistory.getDimension());
		    
		    PersistentEmbeddings newParsingOpEmbedding = new PersistentEmbeddings(null, null); 
			dstate.setParsingOpPersistentEmbedding(newParsingOpEmbedding);
			
			a2 = Nd4j.zeros(this.embedParserState.getDimension());
			
			PersistentEmbeddings newStatePersistentEmbedding = new PersistentEmbeddings(null, null); 
			dstate.setStatePersistentEmbedding(newStatePersistentEmbedding);			
	    } else { 
		    
	    	IWeightedShiftReduceStep<MR> dstateStep = dstate.returnStep();
			/*SentenceSpan parsingSpan = new SentenceSpan(dstateStep.getStart(), dstateStep.getEnd(), n);
	    	ParsingOp<MR> parsingOp = new ParsingOp<MR>(dstateStep.getRoot(),
	    	 								parsingSpan, dstateStep.getRuleName());*/
	    	ParsingOp<MR> parsingOp = dstate.returnParsingOp();
	    	
		    /* Embedding the history of parsing operations.
		     * For the parsing operations the previous rnn state is that of its parent since 
		     * actions are added on top the parent's action list. */
		    PersistentEmbeddings parsingOpPersistentEmbedding = parent
													.getParsingOpPersistentEmbedding();
		    
		    RecurrentTimeStepOutput recurrentStepOutputParsingOp = this.embedActionHistory
			    						.getEmbedding(parsingOp, parsingOpPersistentEmbedding.getRNNState());
			    
			a1 = recurrentStepOutputParsingOp.getEmbedding();
		    
		    PersistentEmbeddings newParsingOpEmbedding = new PersistentEmbeddings(
		    					recurrentStepOutputParsingOp.getRNNState(), parsingOpPersistentEmbedding); 
			dstate.setParsingOpPersistentEmbedding(newParsingOpEmbedding);
		    
			/* Embedding the derivation state. 
		     * For the derivation state the persistent embedding to work are trickier. For shift and unary
		     * reduce its given by the previous parent. For binary reduce its given by its grandparent. */
			
			final DerivationState<MR> rnnPrevStateParent;
			final PersistentEmbeddings statePersistentEmbedding;
			final RecurrentTimeStepOutput recurrentStepOutputDState;
			
			if(dstateStep instanceof WeightedShiftReduceParseStep<?>) { // check for reduce step 
				
				DerivationStateHorizontalIterator<MR> hit = dstate.horizontalIterator();
				
				hit.next(); //same as itself
				DerivationState<MR> nextLeft; //next left dstate tree
				if(hit.hasNext()) {
					nextLeft = hit.next();
				} else {
					nextLeft = null;
				}
				
				/* Why a special case is needed. */
				if(nextLeft == null && dstate.getRightCategory() != null) { //special case 
					
					rnnPrevStateParent = null;
					Pair<RecurrentTimeStepOutput, PersistentEmbeddings> pairedResult = this.embedParserState
    						.getEmbedding(dstate.getLeftCategory(), dstate.getRightCategory()); 
					
					statePersistentEmbedding = pairedResult.second();
					recurrentStepOutputDState = pairedResult.first();
					
					a2 = recurrentStepOutputDState.getEmbedding();
					
				} else {
					rnnPrevStateParent = nextLeft;
					
					if(rnnPrevStateParent == null) {
						statePersistentEmbedding = new PersistentEmbeddings(null, null);
						
					} else {
						statePersistentEmbedding = rnnPrevStateParent.getStatePersistentEmbedding();
					}
					
					recurrentStepOutputDState = this.embedParserState
    						.getEmbedding(dstate, statePersistentEmbedding.getRNNState());
					a2 = recurrentStepOutputDState.getEmbedding();
				}
				
			} else { //lexical step
				
				rnnPrevStateParent = parent;
				statePersistentEmbedding = rnnPrevStateParent.getStatePersistentEmbedding();
				recurrentStepOutputDState = this.embedParserState
										.getEmbedding(dstate, statePersistentEmbedding.getRNNState());
				a2 = recurrentStepOutputDState.getEmbedding();
			}
		    
		    PersistentEmbeddings newStatePersistentEmbedding = new PersistentEmbeddings(
		    		recurrentStepOutputDState.getRNNState(), statePersistentEmbedding); 
		    dstate.setStatePersistentEmbedding(newStatePersistentEmbedding);
	    }
		
	    /* Embedding the word buffer. Simply look up the table of word embeddings */
		a3 = sentenceEmbedding[dstate.wordsConsumed];//this.embedWordBuffer.getEmbedding(tk.sub(dstate.wordsConsumed, n).toList());
	
		/* perform affine transformation on a1,a2,a3 to give weights to these encodings. 
		 * Return g(A[a1; a2; a3] + b)*/
		
		INDArray concat = Nd4j.concat(1, Nd4j.concat(1, a1, a2), a3).transpose();
		INDArray currentPreOutput = this.A.mmul(concat).add(this.b).transpose();
		
		INDArray current = Nd4j.getExecutioner()
				   			   .execAndReturn(new /*RectifedLinear*/Tanh(currentPreOutput.dup()));
		
		return current;
		//return concat;
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
		throw new RuntimeException("Operation not supported currently");		
	}
	
	@Override
	public IGraphParserOutput<MR> parserCatchEarlyErrors(DI dataItem, Predicate<ParsingOp<MR>> validAmrParsingFilter,
			IDataItemModel<MR> model_, boolean allowWordSkipping, ILexiconImmutable<MR> tempLexicon, Integer beamSize_) {
		throw new RuntimeException("Operation not supported currently");
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
		
		long start = System.currentTimeMillis();
		
		final Integer beamSize;
		if(beamSize_ == null) {
			beamSize = this.beamSize;
		} else {
			beamSize = beamSize_;
		}
		
		LOG.info("Beamsize %s. Packing Disabled %s", beamSize, this.disablePacking);
		
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
		final int numCores = 32;
		
		// If DI is SituatedSentence then find tag
		List<String> tags;
		try {
		    tags = ((SituatedSentence<AMRMeta>) dataItem).getState().getTags().toList();
		} catch (ClassCastException e) {
			tags = null;
		}
		
		//Bootstrap sentence embeddings
		INDArray[] sentenceEmbedding = this.embedWordBuffer.getAllSuffixEmbeddings(tk.toList(), tags);	
		
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
					
				int startIndex = numCores*i;
				int endIndex = Math.min(startIndex + numCores, size);
				List<PackedState<MR>> thisBatchPStates = allPStates.subList(startIndex, endIndex);
					
				StreamSupport.stream(Spliterators.spliterator(thisBatchPStates/*pstates*/, Spliterator.IMMUTABLE), 
									LOG.getLogLevel() == LogLevel.DEBUG ? false : true)
					    .forEach(pstate -> { 
					/* perform valid shift and reduce operations for this packed states, if the packed state is
					 * finished and already represents a complete parse tree then save it separately. 
					 * 
					 * probability/score of a feature is given by: 
		             * exp{ g([A[a1,a2,a3] + b]) . embed(y) } / \sum_y' exp { g([A[a1,a2,a3] + b]) . embed(y')} */
					
					/* Important: Since features are currently local that is only look at the root categories or 
					 * at the shifted lexical entries. Hence, operations can be performed on the best state
					 * currently in the packed state. This will NO LONGER HOLD if features start looking 
					 * at the complete tree segments in the state. --- and yes it does not hold here */
					    	
			    	DerivationState<MR> dstate = pstate.getBestState();
					int wordsConsumed = dstate.wordsConsumed;
					
					List<ParsingOp<MR>> possibleActions = new LinkedList<ParsingOp<MR>>();
					
					// Compute encoding of the given state
					INDArray current = this.encodeState(dstate, sentenceEmbedding, n);
					LOG.debug("Current %s", current);
					assert current != null;
					
					//list of new potential states and their un-normalized probabilities.
					List<UnNormalizedDerivation<MR>> unNormProb = new LinkedList<UnNormalizedDerivation<MR>>();
					double Z = 0; //partition function
					
					//Operation 1: Shift operation: shift a token and its lexical entry to this stack
					if(wordsConsumed < n) {
						
						for(int words = 1; words <= n - dstate.wordsConsumed; words++) {
							/*Iterator<? extends LexicalEntry<MR>> lexicalEntries = 
									compositeLexicon.get(tk.sub(dstate.wordsConsumed, dstate.wordsConsumed+words));*/
							
							Iterator<LexicalResult<MR>> lexicalResults = this.lexicalRule.apply(tk.sub(dstate.wordsConsumed, dstate.wordsConsumed+words), 
													new SentenceSpan(wordsConsumed, wordsConsumed + words - 1, n), compositeLexicon);
							
							while(lexicalResults.hasNext()/*lexicalEntries.hasNext()*/) {
								LexicalResult<MR> lexicalResult = lexicalResults.next();
								LexicalEntry<MR> lexicalEntry = lexicalResult.getEntry();//lexicalEntries.next();
								
								SentenceSpan span = new SentenceSpan(wordsConsumed, wordsConsumed + words, n);
								
								ParsingOp<MR> op = new LexicalParsingOp<MR>(lexicalResult.getResultCategory(), span, 
																	ShiftReduceLexicalStep.LEXICAL_DERIVATION_STEP_RULENAME, lexicalEntry);
								possibleActions.add(op);	
								
								ParsingOp<MR> passedOp = null;
								
								if(pruningFilter != null && !this.test(pruningFilter, op, dstate)/*pruningFilter.test(op)*/) {
									
									//////// Temporary ///////////////////
									// Due to a bug in the code, the intermediate categories do not have word overloading.
									// So if there is a lex+unary overloading then the lex intermediate will be without words
									// and hence match to op will fail. However we can match against lexicalEntry category.
									// for now. But this is really an ugly hack and should be fixed in the future.
									
									ParsingOp<MR> op1 = new LexicalParsingOp<MR>(lexicalEntry.getCategory(), span, 
											ShiftReduceLexicalStep.LEXICAL_DERIVATION_STEP_RULENAME, lexicalEntry);
									
									if(!this.test(pruningFilter, op1, dstate)/*pruningFilter.test(op1)*/) {
										continue;
									} else passedOp = op1;
								} else passedOp = op;
								
								//////// Cannot skip more than one word /////////////////
								if(op.getCategory().getSyntax().equals(SimpleSyntax.EMPTY)) {
									if(words > 1) {
										continue;
									}
								}
								
								DerivationState<MR> dNew = dstate.shift(lexicalResult/*lexicalEntry*/, words, span);
							
								boolean full = n == words && this.completeParseFilter.test(
										dNew.returnLastNonTerminal().getCategory());
								ShiftReduceLexicalStep<MR> lexicalStep = new ShiftReduceLexicalStep<MR>(lexicalResult.getResultCategory()
										/*lexicalEntry.getCategory()*/, lexicalEntry, full, dstate.wordsConsumed, dstate.wordsConsumed + words);
								
								// compute score here
								INDArray embedLexicalStep = this.embedParsingOp.getEmbedding(op).getEmbedding().transpose();
								
								//dot product
								double penalty = this.actionBias[0];
								if(lexicalResult.getResultCategory().getSyntax().equals(SimpleSyntax.EMPTY)) { //SKIP rule
									penalty = penalty + this.gamma;
								}
								
								final double exponent = //this.topLayer.getEmbeddingParallel(embedLexicalStep, current);
											(current.mmul(embedLexicalStep)).getDouble(new int[]{0, 0}) + penalty;
								final double score = Math.exp(exponent);
								UnNormalizedDerivation<MR> newUnNormProb = 
															new UnNormalizedDerivation<MR>(dNew, score, lexicalStep);
								unNormProb.add(newUnNormProb);
								Z = Z + score;
								
								dNew.calcDebugHashCode();
								
								this.register(pruningFilter, passedOp, dstate, dNew);
								
								LOG.debug("Generated %s; Shift %s [unmodified] on %s; Unnormalized score: %s ", dNew.getDebugHashCode(),
										lexicalEntry, dstate.getDebugHashCode(), score);
								LOG.debug("Embedding %s", embedLexicalStep);
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
									possibleActions.add(op);
									
									if(pruningFilter != null && !this.test(pruningFilter, op, dstate)) {
										continue;
									}
									
									DerivationState<MR> dNew =	dstate.reduceUnaryRule(name, logical.getResultCategory());
									
									boolean full = dNew.lenRoot() == 1 && n == dNew.wordsConsumed && 
											       dNew.returnLastNonTerminal().getCategory().getSemantics() != null &&
											       this.completeParseFilter.test(dNew.returnLastNonTerminal().getCategory());
									
									List<Category<MR>> children = new LinkedList<Category<MR>>();
									children.add(lastNonTerminal.getCategory());
									
									ShiftReduceParseStep<MR> step = new ShiftReduceParseStep<MR>(dNew.returnLastNonTerminal().getCategory(),
													children, full, true, name, lastSpan.getStart(), lastSpan.getEnd());
									
									// compute score here
									INDArray embedParseStep = this.embedParsingOp.getEmbedding(op).getEmbedding().transpose();
									//dot product
									double penalty = this.actionBias[1 + uj];
									if(logical.getResultCategory().getSyntax().equals(SimpleSyntax.EMPTY)) { //SKIP rule
										penalty = penalty + this.gamma;
									}
									
									final double exponent = //this.topLayer.getEmbeddingParallel(embedParseStep, current);
											(current.mmul(embedParseStep)).getDouble(new int[]{0, 0}) + penalty;
									final double score = Math.exp(exponent);
									UnNormalizedDerivation<MR> newUnNormProb = 
											new UnNormalizedDerivation<MR>(dNew, score, step);
									unNormProb.add(newUnNormProb);
									Z = Z + score;
									
									dNew.calcDebugHashCode();
									this.register(pruningFilter, op, dstate, dNew);
									
									LOG.debug("Generated %s; Unary-Reduce %s %s; Unnormalized score: %s ", dNew.getDebugHashCode(), name, 
											dstate.getDebugHashCode(), score);
									LOG.debug("Embedding %s", embedParseStep);
								}
							}
						}
						
						ShiftReduceRuleNameSet<MR> last2ndLastNonTerminal = dstate.return2ndLastNonTerminal();
						SentenceSpan sndLastSpan = dstate.return2ndLastSentenceSpan();
						
						if(last2ndLastNonTerminal != null) {
							//Operation 3: Binary Reduce operation
							SentenceSpan joined = new SentenceSpan(sndLastSpan.getStart(), lastSpan.getEnd(), n);
							for(int bj = 0; bj < this.binaryRules.length; bj++) {
								//LOG.debug("Applying binary rule on "+dstate.hashCode());
								RuleName name = this.binaryRules[bj].getName();
								
								ParseRuleResult<MR> logical  = this.applyBinaryRule(bj, last2ndLastNonTerminal, 
																lastNonTerminal, joined);
								if(logical!= null) {
									
									SentenceSpan joined_ = new SentenceSpan(sndLastSpan.getStart(), lastSpan.getEnd(), n);
									ParsingOp<MR> op = new ParsingOp<MR>(logical.getResultCategory(), joined_, name);
									possibleActions.add(op);
									
									if(pruningFilter != null && !this.test(pruningFilter, op, dstate)/*pruningFilter.test(op)*/) {
										continue;
									}
									
									DerivationState<MR> dNew = dstate.reduceBinaryRule(name, logical.getResultCategory(), joined);
									
									boolean full = dNew.lenRoot() == 1 && n == dNew.wordsConsumed && 
											   dNew.returnLastNonTerminal().getCategory().getSemantics()!=null &&
											   this.completeParseFilter.test(dNew.returnLastNonTerminal().getCategory());
									
									ShiftReduceParseStep<MR> step = new ShiftReduceParseStep<MR>(dNew.returnLastNonTerminal().getCategory(),
											dstate.returnBothCategories(), full, false, name, joined_.getStart(), joined_.getEnd());
									
									// compute score here
									INDArray embedParseStep = this.embedParsingOp.getEmbedding(op).getEmbedding().transpose();
									//dot product
									double penalty = this.actionBias[1 + this.unaryRules.length + bj];
									if(logical.getResultCategory().getSyntax().equals(SimpleSyntax.EMPTY)) { //SKIP rule
										penalty = penalty + this.gamma;
									}
									
									final double exponent = //this.topLayer.getEmbeddingParallel(embedParseStep, current);
													(current.mmul(embedParseStep)).getDouble(new int[]{0, 0}) + penalty;
									final double score = Math.exp(exponent);
									UnNormalizedDerivation<MR> newUnNormProb = 
																	new UnNormalizedDerivation<MR>(dNew, score, step);
									unNormProb.add(newUnNormProb);
									Z = Z + score;
									
									dNew.calcDebugHashCode();
									this.register(pruningFilter, op, dstate, dNew);
									
									LOG.debug("Generated %s; Binary-Reduce %s %s %s; Unnormalized score: %s ", dNew.getDebugHashCode(), 
											logical, name, dstate.getDebugHashCode(), score);
									LOG.debug("Embedding %s", embedParseStep);
								}
							}
						}
					}
					
					//normalize the probabilities and add them to the list
					Iterator<UnNormalizedDerivation<MR>> it = unNormProb.iterator();
					
					while(it.hasNext()) {
						UnNormalizedDerivation<MR> next_ = it.next();
						DerivationState<MR> dNew = next_.getDState();
						final double prob = next_.getUnNormalizedProb()/Z; //normalize the probability
						final double stepScore = Math.log(prob);
						dNew.score = dstate.score + stepScore; //log-likelihood
						
						final AbstractShiftReduceStep<MR> step = next_.getStep(); 
						final IWeightedShiftReduceStep<MR> weightedStep;
						
						//LOG.debug("Score %s, category %s", dNew.score, step.getRoot());
						
						if(step instanceof ShiftReduceLexicalStep<?>) {
							weightedStep = new WeightedShiftReduceLexicalStep<MR>(
														(ShiftReduceLexicalStep<MR>)step, stepScore/*dNew.score*/);
						} else {
							weightedStep = new WeightedShiftReduceParseStep<MR>(
														(ShiftReduceParseStep<MR>)step, stepScore/*dNew.score*/);
						}
						
						dNew.defineStep(weightedStep);
						dNew.setPossibleActions(possibleActions);
						
						boolean full = dNew.lenRoot() == 1 && n == dNew.wordsConsumed && 
								   dNew.returnLastNonTerminal().getCategory().getSemantics() != null &&
								   this.completeParseFilter.test(dNew.returnLastNonTerminal().getCategory()) /*&& 
								   dNew.returnLastNonTerminal().getCategory().getSyntax().equals(AMRServices.AMR)*/;
						
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
					LOG.debug("\t State in beam 0: %s parent %s score: %s", ds_.hashCode(), ds_.getBestState().getParent().hashCode(), 
																		   ds_.getBestScore());
					beam_.offer(ds_);
					isEmpty = false;
				}
				nBeam_.clear(); //clear the new stack
			}
			
			//Update the cursor of the filter if it is a CKY Single or Multi Parse Tree filter
			if(pruningFilter instanceof CKYSingleParseTreeParsingFilter) {
				((CKYSingleParseTreeParsingFilter<MR>)pruningFilter).incrementCursor();
				LOG.debug("Updated CKY Single Parse Tree cursor");
			} else if (pruningFilter instanceof CKYMultiParseTreeParsingFilter) {
				((CKYMultiParseTreeParsingFilter<MR>)pruningFilter).incrementCursor();
				LOG.debug("Updated CKY Multi Parse Tree cursor");
			}
		}
		
		final ShiftReduceParserOutput<MR> output = new ShiftReduceParserOutput<MR>(
									completeParseTrees, System.currentTimeMillis() - start);
		
		LOG.info("Neural Shift Reduce: Number of distinct derivations %s", output.getAllDerivations().size());
		
		this.embedCategory.logCachePerformance();
		/*LOG.info("Parsing History Print Time Stamp");
		this.embedActionHistory.printTimeStamps();
		LOG.info("Parser State Print Time Stamp");
		this.embedParserState.printTimeStamps();
		
		LOG.debug("Parses Packing %s ", packedParse.get());*/
		
//		if(dataItem.getString().contains("Cyber space essentially has no borders") || 
//			dataItem.getString().contains("Dmitry Medvedev promised to raise officers' salaries") ||
//			dataItem.getString().contains("Katrin pargmae did not comment on the Georgian attacks") ||
//			dataItem.getString().contains("Human Rights Watch helped produce the report") ||
//			dataItem.getString().contains("The men were mostly of them lawyers and professors")
//			/*LOG.getLogLevel() == LogLevel.DEBUG*/) {
//			
//			if(output.getBestDerivations().size() > 0) {
//			
//				ListIterator<DerivationState<MR>> it = completeParseTrees.listIterator();
//				double maxScore = output.getBestDerivations().get(0).getScore();
//				
//				while(it.hasNext()) {
//					int ix = it.nextIndex();
//					DerivationState<MR> dstate = it.next();
//					if(dstate.score == maxScore) {
//						LOG.info("{ Parse Tree* %s %s %s", ix + 1, dstate.getDebugHashCode(), dstate.score);
//					} else {
//						LOG.info("{ Parse Tree %s %s %s", ix + 1, dstate.getDebugHashCode(), dstate.score);
//					}
//					LOG.info("\t Category %s", dstate.returnLastNonTerminal().getCategory());
//					List<ParsingOp<MR>> steps = dstate.returnParsingOps();
//					for(ParsingOp<MR> step: steps) {
//						LOG.info("\t %s", step);
//					}
//					LOG.info("}, ");
//				}
//			}
//		}
		
		return output;
	}
	
	public List<ParsingOpPreTrainingDataset<MR>> createPreTrainingData(DI dataItem, 
			Predicate<ParsingOp<MR>> pruningFilter, IDataItemModel<MR> model, 
			boolean allowWordSkipping, ILexiconImmutable<MR> tempLexicon, Integer beamSize_) {
		
		List<ParsingOpPreTrainingDataset<MR>> dataset = new 
												LinkedList<ParsingOpPreTrainingDataset<MR>>();
		AtomicInteger pos = new AtomicInteger();
		
		if(beamSize_ == null)
			beamSize_ = this.beamSize;
		
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
								ParsingOp<MR> op = new ParsingOp<MR>(lexicalEntry.getCategory(), span, 
											ShiftReduceLexicalStep.LEXICAL_DERIVATION_STEP_RULENAME);
								
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
									pos.getAndIncrement();
								}
								
								ParsingOpPreTrainingDataset<MR> point = new
										ParsingOpPreTrainingDataset<MR>(last2ndLastNonTerminal.getCategory(),
										lastNonTerminal.getCategory(), this.binaryRulesVectors[bj], label);
								synchronized(dataset) {
									dataset.add(point);
								}
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
			
			if(pos.get() >= 2) { //we want very few examples from every training example
				break;
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
	
	public static class Builder<DI extends Sentence, MR> {
		
		private Integer									        beamSize;
		
		private double											learningRate = 0.1;
		
		private double											learningRateDecay = 0.01;
		
		private double											l2 = 0.000001;
		
		private double											gamma = 5;
		
		private int 											seed = 1234;

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
		
		@SuppressWarnings("unchecked")
		public NeuralNetworkShiftReduceParser<DI, MR> build() {
			return new NeuralNetworkShiftReduceParser<DI, MR>(this.beamSize,
					binaryRules.toArray((ShiftReduceBinaryParsingRule<MR>[]) Array
							.newInstance(ShiftReduceBinaryParsingRule.class,
									binaryRules.size())), lexicalRule, 
					sentenceLexicalGenerators, sloppyLexicalGenerators,
					categoryServices, completeParseFilter,
					unaryRules.toArray((ShiftReduceUnaryParsingRule<MR>[]) Array
							.newInstance(ShiftReduceUnaryParsingRule.class,
									unaryRules.size())), 
					learningRate, learningRateDecay, l2, gamma, seed);
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
	
	public static class Creator<DI extends Sentence, MR> implements IResourceObjectCreator<NeuralNetworkShiftReduceParser<DI, MR>> {

		private final String type;
		
		public Creator() {
			this("parser.neuralNetworkShiftReduce");
		}

		public Creator(String type) {
			this.type = type;
		}
		
		@SuppressWarnings("unchecked")
		@Override
		public NeuralNetworkShiftReduceParser<DI, MR> create(Parameters params, IResourceRepository repo) {
			
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
