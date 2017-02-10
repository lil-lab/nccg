package edu.cornell.cs.nlp.spf.parser.ff.shiftreduce.neuralparser;

import java.io.DataInputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.lang.reflect.Array;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.IdentityHashMap;
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
//import org.nd4j.linalg.factory.NDArrayFactory;
import org.nd4j.linalg.factory.Nd4j;
//import org.nd4j.linalg.api.buffer.DataBuffer;

import com.google.common.base.Joiner;

import edu.cornell.cs.nlp.spf.base.hashvector.HashVectorFactory;
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
import edu.cornell.cs.nlp.spf.data.situated.sentence.SituatedSentence;
import edu.cornell.cs.nlp.spf.explat.IResourceRepository;
import edu.cornell.cs.nlp.spf.explat.ParameterizedExperiment;
import edu.cornell.cs.nlp.spf.explat.ParameterizedExperiment.Parameters;
import edu.cornell.cs.nlp.spf.explat.resources.IResourceObjectCreator;
import edu.cornell.cs.nlp.spf.explat.resources.usage.ResourceUsage;
import edu.cornell.cs.nlp.spf.mr.lambda.LogicalConstant;
import edu.cornell.cs.nlp.spf.mr.lambda.LogicalExpression;
import edu.cornell.cs.nlp.spf.parser.ISentenceLexiconGenerator;
import edu.cornell.cs.nlp.spf.parser.ParsingOp;
import edu.cornell.cs.nlp.spf.parser.ccg.IParseStep;
import edu.cornell.cs.nlp.spf.parser.ccg.model.IDataItemModel;
import edu.cornell.cs.nlp.spf.parser.ccg.normalform.NormalFormValidator;
import edu.cornell.cs.nlp.spf.parser.ccg.normalform.unaryconstraint.UnaryConstraint;
import edu.cornell.cs.nlp.spf.parser.ccg.rnn.shiftreduce.dataset.CKYMultiParseTreeParsingFilter;
import edu.cornell.cs.nlp.spf.parser.ccg.rnn.shiftreduce.dataset.CKYSingleParseTreeParsingFilter;
import edu.cornell.cs.nlp.spf.parser.ccg.rnn.shiftreduce.learning.Helper;
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
import edu.cornell.cs.nlp.spf.parser.ff.shiftreduce.nonlocal.features.SemanticFeaturesEmbedding;
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
	 * Parses utterance using a feed forward neural network shift reduce parser 
	 * with a dot product objective making it faster.
	 * @author Dipendra Misra
	 */
	public class LocalEnsembleNeuralDotProductShiftReduceParser<DI extends Sentence, MR> 
					implements AbstractNeuralShiftReduceParser<DI, MR> {
		
		private static final long 								serialVersionUID = 710325202611085597L;

		public static final ILogger								LOG
							= LoggerFactory.create(LocalEnsembleNeuralDotProductShiftReduceParser.class);
		
		private static final int 								numCores = 32;
		
		private final int										numEnsemble;
		
		private final List<Double>								mixingProbability;
		
		private final List<String>								bootstrapFolders;
		
		/** This variable and next 3 are different for different parsers in the ensemble. 
		 * Feed-forward neural network that takes dense features representing state+parsing step
		 * and returns the score of the operation */
		private final NeuralParsingDotProductStepScorer[] 		mlpScorer;
		
		/** Converts sparse action features to dense features */
		private final FeatureEmbedding<MR>[] 					actionFeatureEmbedding;
		
		/** Converts sparse state features to dense features */
		private final FeatureEmbedding<MR>[] 					stateFeatureEmbedding;
		
		/** Projective transformation for action embedding */
		private final INDArray W[];
		
		/** Feature for specially handling the semantics
		TODO incorporate this in integrated feature interface */
		private final SemanticFeaturesEmbedding[]				semanticFeatureEmbedding;
		
		/** Beamsize of the parser */
		private final Integer									beamSize;
		
		/** Binary CCG parsing rules. */
		public final ShiftReduceBinaryParsingRule<MR>[]			binaryRules;

		private final IFilter<Category<MR>>						completeParseFilter;

		private final transient Predicate<ParsingOp<MR>> 		pruningFilter;
		
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
		
		private final ShiftReduceParseTreeLogger<DI, MR>		logger;
		
		/** penalty for SKIP operations */
		private final double 									gamma;
		
		private transient Predicate<ParsingOp<MR>> 				datasetCreatorFilter;
		
		private final List<AbstractNonLocalFeature<MR>> 		nonLocalActionFeatures;
		private final List<AbstractNonLocalFeature<MR>> 		nonLocalStateFeatures;
		
		private final PostProcessing<MR>						postProcessing;
		
		public Set<LogicalConstant>								conjunctionConstants;
	
//		static {
//	        Nd4j.dtype = DataBuffer.Type.DOUBLE;
//	        NDArrayFactory factory = Nd4j.factory();
//	        factory.setDType(DataBuffer.Type.DOUBLE);
//	    }
		
		private boolean disablePacking;
		
		/** TODO -- separate learning parts from other components */
		public LocalEnsembleNeuralDotProductShiftReduceParser(int numEnsemble, List<Double> mixingProbability, 
				List<String> bootstrapFolders, int beamSize,
				ShiftReduceBinaryParsingRule<MR>[] binaryRules, ILexicalRule<MR> lexicalRule, 
				List<ISentenceLexiconGenerator<DI, MR>> sentenceLexiconGenerators,
				List<ISentenceLexiconGenerator<DI, MR>> sloppyLexicalGenerators,
				ICategoryServices<MR> categoryServices, 
				IFilter<Category<MR>> completeParseFilter,
				ShiftReduceUnaryParsingRule<MR>[] unaryRules, double learningRate, 
				double learningRateDecay, double l2, double gamma, int seed, File outputDir) {
		
			LOG.setCustomLevel(LogLevel.INFO);
			Nd4j.getRandom().setSeed(seed);
			
			this.numEnsemble = numEnsemble;
			this.mixingProbability = mixingProbability;
			this.bootstrapFolders = bootstrapFolders;
			
			if(this.mixingProbability.size() != this.numEnsemble || this.bootstrapFolders.size() != this.numEnsemble) {
				throw new RuntimeException("Boostrap folder not same as mixing probability. Found " +  
						this.mixingProbability.size() + " " + this.bootstrapFolders.size() + " " + this.numEnsemble);
			}
			
			double sum = 0;
			for(double prob: this.mixingProbability) {
				if(prob < 0 || prob > 1) {
					throw new RuntimeException("Invalid probability " + prob);
				}
				sum = sum + prob;
			}
			
			if(sum != 1.0) {
				throw new RuntimeException("Sum of mixing probability does not sum to 1.0. Found " + sum);
			}
			
			//Non-local features
			this.nonLocalActionFeatures = new ArrayList<AbstractNonLocalFeature<MR>>();
			
			//Action features
			final Map<String, Integer> tagsAndDimensionAction = new HashMap<String, Integer>();
			tagsAndDimensionAction.put("ATTACH", 32);
			tagsAndDimensionAction.put("CROSS", 16);
			tagsAndDimensionAction.put("DYN", 8);
			tagsAndDimensionAction.put("DYNSKIP", 8);
			tagsAndDimensionAction.put("LOGEXP", 8);
			tagsAndDimensionAction.put("SLOPPYLEX", 16);
			tagsAndDimensionAction.put("SHIFT", 16);
			
			//joint features
			tagsAndDimensionAction.put("SHIFTSEM", 32);
			tagsAndDimensionAction.put("ATTRIBPOS", 32);
			tagsAndDimensionAction.put("AMRLEX", 48);
			
			//disjoint features
			tagsAndDimensionAction.put("POS", 12);
			tagsAndDimensionAction.put("STEPRULE", 16);
			tagsAndDimensionAction.put("TEMPLATELEFTPOS", 32);
			tagsAndDimensionAction.put("TEMPLATERIGHTPOS", 32);
			
			tagsAndDimensionAction.put("NEXT1POS", 12);
			tagsAndDimensionAction.put("NEXT2POS", 12);
			tagsAndDimensionAction.put("PREV1POS", 12);
			tagsAndDimensionAction.put("PREV2POS", 12);
			
			int nInAction = 0;
			
			for(int dim: tagsAndDimensionAction.values()) {
				nInAction = nInAction + dim;
			}
			
			this.actionFeatureEmbedding = new FeatureEmbedding[numEnsemble];
			for(int i = 0; i < this.numEnsemble; i++) {
				this.actionFeatureEmbedding[i] = 
						new FeatureEmbedding<MR>(learningRate, l2, tagsAndDimensionAction, outputDir);
			}
			
			// State Features
			final Map<String, Integer> tagsAndDimensionState = new HashMap<String, Integer>();
			tagsAndDimensionState.put("STACKSYNTAX1", 24);
			tagsAndDimensionState.put("STACKSYNTAX2", 24);
			tagsAndDimensionState.put("STACKSYNTAX3", 24);
			tagsAndDimensionState.put("STACKATTRIB1", 12);
			tagsAndDimensionState.put("STACKATTRIB2", 12);
			tagsAndDimensionState.put("STACKATTRIB3", 12);
			
			int nInState = 0;
			
			for(int dim: tagsAndDimensionState.values()) {
				nInState = nInState + dim;
			}
			
			//Semantic features
			this.semanticFeatureEmbedding = new SemanticFeaturesEmbedding[numEnsemble];
			final int semanticDim = 35; 
			for(int i = 0; i < this.numEnsemble; i++) {
				this.semanticFeatureEmbedding[i] = null; //new SemanticFeaturesEmbedding(semanticDim, learningRate, l2);
			}
//			nInState = nInState + this.semanticFeatureEmbedding[0].getDimension();
			
			final int nOut = 35;
			
			this.stateFeatureEmbedding = new FeatureEmbedding[numEnsemble];
			this.mlpScorer = new NeuralParsingDotProductStepScorer[numEnsemble];
			this.W = new INDArray[numEnsemble];
			for(int i = 0; i < this.numEnsemble; i++) {
				this.mlpScorer[i] = new NeuralParsingDotProductStepScorer(nInState, nOut, learningRate, l2, seed);
				this.stateFeatureEmbedding[i] = new FeatureEmbedding<MR>(learningRate, l2, tagsAndDimensionState, outputDir);
				this.W[i] = Helper.getXavierInitiliazation(nOut/* + 1*/, nInAction);
			}
			
			this.nonLocalStateFeatures = new ArrayList<AbstractNonLocalFeature<MR>>();
			this.nonLocalStateFeatures.add(new StackSyntaxFeatures<MR>());
			
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
			this.testing = false;
			
			this.logger = new ShiftReduceParseTreeLogger<DI, MR>(outputDir);
			this.postProcessing = new PostProcessing<MR>(0.0);
			
			/////TEMPORARY
			this.modelNewFeatures = null;
			
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
			LOG.info("Ensemble of %s Neural Dot Product Feed Forward Parser. # action tags %s, nIn %s, Gamma %s", 
									this.numEnsemble, tagsAndDimensionAction.size(), nInState, gamma);
			LOG.info("Mixing probability %s, Bootstrap folders %s",
					Joiner.on(", ").join(this.mixingProbability), Joiner.on(", ").join(this.bootstrapFolders));
			LOG.info(".. # action non local features %s, # state tags %s, # state non local features %s", 
						this.nonLocalActionFeatures.size(), tagsAndDimensionState.size(), this.nonLocalStateFeatures.size());
			LOG.info(".. outputDir %s", outputDir.getAbsolutePath());
		}
		
		private void computeNonLocalFeatures(DerivationState<MR> state, IParseStep<MR> parseStep, 
									IHashVector features, String[] buffer, int bufferIndex, String[] tags) {
			
			for(AbstractNonLocalFeature<MR> nonLocalFeature: this.nonLocalActionFeatures) {
				nonLocalFeature.add(state, parseStep, features, buffer, bufferIndex, tags);
			}
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
		
		private Pair<IHashVector,INDArray> calculateStateEmbedding(int ensembleIx, DerivationState<MR> dstate,
														String[] buffer, int wordsConsumed, String[] tags) {
			
			//currently only non-local features
			IHashVector feature = HashVectorFactory.create();
			for(AbstractNonLocalFeature<MR> nonLocalFeature: this.nonLocalStateFeatures) {
				nonLocalFeature.add(dstate, null, feature, buffer, wordsConsumed, tags);
			}
			
			final INDArray embedding;
			final INDArray nonLocalFeatureEmbedding = this.stateFeatureEmbedding[ensembleIx].embedFeatures(feature).first();
			
			if(this.semanticFeatureEmbedding[ensembleIx] != null) {
				@SuppressWarnings("unchecked")
				INDArray semanticEmbedding = this.semanticFeatureEmbedding[ensembleIx]
										.getSemanticEmbedding((DerivationState<LogicalExpression>) dstate).getEmbedding();
				embedding = Nd4j.concat(1, nonLocalFeatureEmbedding, semanticEmbedding);
			} else {
				embedding = nonLocalFeatureEmbedding;
			}
			
			return Pair.of(feature, embedding);
		}
		
		private void readObject(java.io.ObjectInputStream in) throws IOException, ClassNotFoundException {
			
			in.defaultReadObject();
			
			for(int i = 0; i < this.numEnsemble; i++) {
				
				//bootstrap the parameters
				final String folderName = this.bootstrapFolders.get(i);
				LOG.info("Going to bootstrap from folder %s", folderName);
				
				this.mlpScorer[i].bootstrapNetworkParam(folderName);
				this.mlpScorer[i].reclone();
				
				this.stateFeatureEmbedding[i].bootstrapEmbeddings(folderName, "state");
				this.actionFeatureEmbedding[i].bootstrapEmbeddings(folderName, "action");
				
				if(this.semanticFeatureEmbedding[i] != null) {
					this.semanticFeatureEmbedding[i].getSemanticEmbeddingObject()
													.bootstrapCategoryEmbeddingAndRecursiveNetworkParam(folderName);
				}
				
				//load W
				final String paramFile = folderName+"/W.bin";
				
				try {
				
					DataInputStream dis = new DataInputStream(new FileInputStream(paramFile));
					INDArray loadedW = Nd4j.read(dis);
					Nd4j.copy(loadedW, this.W[i]);
					
					dis.close();
				} catch(IOException e) {
					throw new RuntimeException("Could not read the top layer param: "+e);
				}
				
				//Prepare parser for testing
				this.actionFeatureEmbedding[i].stopAddingFeatures();
				this.actionFeatureEmbedding[i].stats();
				this.actionFeatureEmbedding[i].clearSeenFeaturesStats();
				
				this.stateFeatureEmbedding[i].stopAddingFeatures();
				this.stateFeatureEmbedding[i].stats();
				this.stateFeatureEmbedding[i].clearSeenFeaturesStats();
				
				this.testing = true;
			}
			
			LOG.info("Boostrapped the worker parser");
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
		
		@Override
		public ILogger getLOG() {
			return LOG;
		}
	
		/** Parses a sentence using Neural Network model */
		@Override
		public IGraphParserOutput<MR> parse(DI dataItem, Predicate<ParsingOp<MR>> pruningFilter, IDataItemModel<MR> model_,
				boolean allowWordSkipping, ILexiconImmutable<MR> tempLexicon, Integer beamSize_) {
			
			//skip this sentence due to metric issue
//			if(dataItem.getSample().getTokens().toString()
//					.startsWith("The government insists the reserves will support the armed forces and not act as")) {
//				LOG.info("Skipping this sentence due to metric issue");
//				return new ShiftReduceParserOutput<MR>(new ArrayList<DerivationState<MR>>(), 1);
//			}
			
			final long start = System.currentTimeMillis();
			
			StringBuilder stateFeature = new StringBuilder();
			StringBuilder actionFeature = new StringBuilder();
			
			for(int ensembleIx = 0; ensembleIx < this.numEnsemble; ensembleIx++) {
				stateFeature.append(this.stateFeatureEmbedding[ensembleIx].isAddingFeatures() + ", ");
				actionFeature.append(this.actionFeatureEmbedding[ensembleIx].isAddingFeatures() + ", ");
			}
			
			DerivationState.LOG.setCustomLevel(LogLevel.DEBUG);
			LOG.info("Testing %s state %s action %s", this.testing, stateFeature.toString(), actionFeature.toString());
			
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
			
			//////////
//			this.disablePacking();
			//////////
			
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
			final String[] buffer = tk.subArray(0, n);
			final String[] tags = ((SituatedSentence<AMRMeta>) dataItem).getState().getTags().subArray(0, n);
			
			List<DerivationState<MR>> completeParseTrees = new ArrayList<DerivationState<MR>>();
			
			List<DirectAccessBoundedPriorityQueue<PackedState<MR>>> beam = new 
							ArrayList<DirectAccessBoundedPriorityQueue<PackedState<MR>>>();
			List<DirectAccessBoundedPriorityQueue<PackedState<MR>>> newBeam = new 
							ArrayList<DirectAccessBoundedPriorityQueue<PackedState<MR>>>();
			
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
			             * exp{ Phi1(action) W MLP (Phi2(x))) } / \sum_action' exp { Phi1(action') W MLP (Phi2(x))) } */
						
						/* Important: Since features are currently local that is only look at the root categories or 
						 * at the shifted lexical entries. Hence, operations can be performed on the best state
						 * currently in the packed state. This will NO LONGER HOLD if features start looking 
						 * at the complete tree segments in the state. It holds here if stack features follow root equivalence */
						    	
						DerivationState<MR> dstate = pstate.getBestState();
						final int wordsConsumed = dstate.wordsConsumed;
						int childIndex = 0;
						
						final INDArray[] stateFeatureEmbedding = new INDArray[this.numEnsemble];
						
						for(int ensembleIx = 0; ensembleIx < this.numEnsemble; ensembleIx++) {
							Pair<IHashVector, INDArray> stateResult = 
									this.calculateStateEmbedding(ensembleIx, dstate, buffer, wordsConsumed, tags);
							final INDArray stateFeatureInEmbedding = stateResult.second();
							stateFeatureEmbedding[ensembleIx] = this.mlpScorer[ensembleIx]
											.getEmbeddingParallel(stateFeatureInEmbedding).mmul(this.W[ensembleIx]);
//							INDArray eye = Nd4j.zeros(1);
//							eye.putScalar(new int[]{0,  0}, 1.0);
//							final INDArray stateFeatureEmbedding = Nd4j.concat(1, stateFeatureEmbedding_, eye).mmul(this.W);
						}
						
						List<ParsingOp<MR>> possibleActions = new ArrayList<ParsingOp<MR>>();
						List<IHashVector> possibleActionFeatures = new ArrayList<IHashVector>();
						
						//list of new potential states and the step that created them.
						List<Pair<DerivationState<MR>, AbstractShiftReduceStep<MR>>> newStateAndStep = 
										new ArrayList<Pair<DerivationState<MR>, AbstractShiftReduceStep<MR>>>();
						List<INDArray[]> actionEmbeddings = new ArrayList<INDArray[]>();
						
						//Operation 1: Shift operation: shift a token and its lexical entry to this stack
						if(wordsConsumed < n) {
							
							for(int words = 1; words <= n - wordsConsumed; words++) {
								
								List<LexicalResult<MR>> lexicalResults =
												allLexicalResults[wordsConsumed][wordsConsumed + words - 1];
//								Iterator<LexicalResult<MR>> lexicalResults = this.lexicalRule.apply(tk.sub(dstate.wordsConsumed, 
//										dstate.wordsConsumed + words), new SentenceSpan(wordsConsumed, wordsConsumed + words - 1, n),
//										compositeLexicon);
								
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
									
									DerivationState<MR> dNew = dstate.shift(lexicalResult, words, span);
									dNew.childIndex = myChildIndex;
									
									ShiftReduceLexicalStep<MR> lexicalStep = new ShiftReduceLexicalStep<MR>(lexicalResult.getResultCategory(),
											lexicalEntry, full, dstate.wordsConsumed, dstate.wordsConsumed + words);
													
									newStateAndStep.add(Pair.of(dNew, lexicalStep));
									INDArray[] actionEmbedding = new INDArray[this.numEnsemble];
									
									for(int ensembleIx = 0; ensembleIx < this.numEnsemble; ensembleIx++) {
										actionEmbedding[ensembleIx] = this.actionFeatureEmbedding[ensembleIx].embedFeatures(feature).first();
									}
									actionEmbeddings.add(actionEmbedding);
									
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
										
										INDArray[] actionEmbedding = new INDArray[this.numEnsemble];
										
										for(int ensembleIx = 0; ensembleIx < this.numEnsemble; ensembleIx++) {
											actionEmbedding[ensembleIx] = this.actionFeatureEmbedding[ensembleIx].embedFeatures(feature).first();
										}
										actionEmbeddings.add(actionEmbedding);
										
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
										INDArray[] actionEmbedding = new INDArray[this.numEnsemble];
										
										for(int ensembleIx = 0; ensembleIx < this.numEnsemble; ensembleIx++) {
											actionEmbedding[ensembleIx] = this.actionFeatureEmbedding[ensembleIx].embedFeatures(feature).first();
										}
										actionEmbeddings.add(actionEmbedding);
										
										dNew.calcDebugHashCode();
										this.register(this.datasetCreatorFilter, op, dstate, dNew);
										
										LOG.debug("Generated %s; Binary-Reduce %s %s %s; ", dNew.getDebugHashCode(), 
																					logical, name, dstate.getDebugHashCode());
									}
								}
							}
						}
						
						final int numAction = actionEmbeddings.size();
						
						if(numAction == 0) {
							return;
						}
						
						double[] logSoftmax = new double[numAction];
						Arrays.fill(logSoftmax, 0.0);
						
						for(int ensembleIx = 0; ensembleIx < this.numEnsemble; ensembleIx++) {
						
							double[] exponents = new double[numAction];
							Iterator<INDArray[]> actionIt = actionEmbeddings.iterator();
							Iterator<ParsingOp<MR>> possibleActionsIt = possibleActions.iterator();
							
							for(int j = 0; j < exponents.length; j++) {
								// F(\phi(x)). (W F(\phi(a) + b))
								INDArray affineActionEmbedding = actionIt.next()[ensembleIx].transpose();
								exponents[j] = stateFeatureEmbedding[ensembleIx].mmul(affineActionEmbedding).getDouble(new int[]{0, 0});
								
								///////////////////
								if(possibleActionsIt.next().getCategory().getSyntax().equals(SimpleSyntax.EMPTY)) { //SKIP rule
										exponents[j] += this.gamma;
								}
								//////////////////
							}
							
							double[] ensembleLogSoftmax = this.mlpScorer[ensembleIx].toLogSoftMax(exponents);
							
							for(int j = 0; j < ensembleLogSoftmax.length; j++) {
								logSoftmax[j] = logSoftmax[j] + Math.exp(ensembleLogSoftmax[j]) * this.mixingProbability.get(ensembleIx);
							}
						}
						
						//currently logSoftmax contains probability so we take log of  it
						for(int j = 0; j < numAction; j++) {
							logSoftmax[j] = Math.log(logSoftmax[j]);
						}
						
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
							
							//////////////////////
							if(this.testing && dNew.wordsConsumed == n) {
								synchronized(cycleIdentityState) {
									cycleIdentityState.add(dNew);
								}
							}
							/////////////////////
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
				
				//////////// remove states that are passing to next cycle
				if(this.testing) {
					List<DerivationState<MR>> toRemove = Collections.synchronizedList(new LinkedList<DerivationState<MR>>());
					
					StreamSupport.stream(Spliterators.spliterator(beam.get(0), Spliterator.IMMUTABLE), true)
						.forEach(pstate -> { 
		
							for(DerivationState<MR> iDState: cycleIdentityState) {
								if(pstate.containsState(iDState)) {
									toRemove.add(iDState);
								}
							}
						});
					cycleIdentityState.removeAll(toRemove);
				}
				identityState.addAll(cycleIdentityState);
//				/////////////////////
				
				if(LOG.getLogLevel() == LogLevel.DEBUG) {
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
					LOG.debug("=========== CYCLE %s =============", cycle);
		
					for(PackedState<MR> pstate: states) {
						DerivationState<MR> state = pstate.getBestState();
						String hashes = "Generated " + state.getDebugHashCode() + " from " + state.getParent().getDebugHashCode();
						String scores = " score " + state.score + "; step " + (state.score - state.getParent().score) + "; ";
						String scoreAndHash = scores + ", " + hashes;
						LOG.debug("%s, %s; %s, %s; %s", ++index, scoreAndHash, state.possibleActions().size(), 
									state.possibleActions().get(state.childIndex), 
									state.possibleActionFeatures().get(state.childIndex));
					}
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
			
			final long parsingTime = System.currentTimeMillis() - start;
			final ShiftReduceParserOutput<MR> output = new ShiftReduceParserOutput<MR>(
											completeParseTrees, parsingTime);
			
			LOG.info("Neural Shift Reduce: Number of distinct derivations %s", output.getAllDerivations().size());
			
			if(this.testing) { //Ugly hack for knowing that we are testing
				this.logger.log(output, dataItem, allowWordSkipping);
			}
			
			//////////////////////
			if(this.testing && allowWordSkipping && output.getAllDerivations().size() == 0) {
				LOG.info("Found no derivation. Using best-tree-on-the-stach-heuristic");
				return this.postProcessing.stitch6(identityState, pruningFilter, parsingTime);
			}
			////////////////////
			
			return output;
		}
		
		@Override
		public IGraphParserOutput<MR> parserCatchEarlyErrors(DI dataItem,
				Predicate<ParsingOp<MR>> validAmrParsingFilter, IDataItemModel<MR> model_, boolean allowWordSkipping,
				ILexiconImmutable<MR> tempLexicon, Integer beamSize_) {
			throw new RuntimeException("Operation not supported");
		}
		
		public static class Builder<DI extends Sentence, MR> {
			
			private Integer									        beamSize;
			
			private double											learningRate = 0.1;
			
			private double											learningRateDecay = 0.01;
			
			private double											l2 = 0.000001;
			
			private double											gamma = 5;
			
			private int 											seed = 1234;
			
			private int												numEnsemble;
			
			private List<Double>									mixingProbability;
			
			private List<String>									bootstrapFolders;
			
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
				this.numEnsemble = 0;
				this.mixingProbability = new ArrayList<Double>();
				this.bootstrapFolders = new ArrayList<String>();
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
			
			public Builder<DI, MR> addMixingProbability(double mixingProbability) {
				this.mixingProbability.add(mixingProbability);
				this.numEnsemble++;
				return this;
			}
			
			public Builder<DI, MR> addBootstrapFolder(String bootstrapFolder) {
				this.bootstrapFolders.add(bootstrapFolder);
				return this;
			}
			
			@SuppressWarnings("unchecked")
			public LocalEnsembleNeuralDotProductShiftReduceParser<DI, MR> build() {
				return new LocalEnsembleNeuralDotProductShiftReduceParser<DI, MR>(this.numEnsemble,
						this.mixingProbability, this.bootstrapFolders, this.beamSize,
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
		
		public static class Creator<DI extends Sentence, MR>
				implements IResourceObjectCreator<LocalEnsembleNeuralDotProductShiftReduceParser<DI, MR>> {

			private final String type;
			
			public Creator() {
				this("parser.local.ensemble.feedforward.dotproduct.neural.shiftreduce");
			}

			public Creator(String type) {
				this.type = type;
			}
			
			@SuppressWarnings("unchecked")
			@Override
			public LocalEnsembleNeuralDotProductShiftReduceParser<DI, MR> create(Parameters params, IResourceRepository repo) {
				
				final Builder<DI, MR> builder = new Builder<DI, MR>( (ICategoryServices<MR>) repo.get(
																	 ParameterizedExperiment.CATEGORY_SERVICES_RESOURCE));
				
				if (params.contains("parseFilter")) {
					builder.setCompleteParseFilter((IFilter<Category<MR>>) repo
							.get(params.get("parseFilter")));
				}
				
				for (final String bootstrapFolder : params.getSplit("bootstrapFolders")) {
					builder.addBootstrapFolder(bootstrapFolder);
				}
				
				for (final String prob : params.getSplit("mixingProbability")) {
					builder.addMixingProbability(Double.parseDouble(prob));
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
