package edu.cornell.cs.nlp.spf.parser.ccg.rnn.shiftreduce.categoryembeddings;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.PrintWriter;
import java.io.UnsupportedEncodingException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.Map.Entry;
import java.util.Random;
import java.util.Set;
import java.util.List;
import java.util.Map;
import java.util.Spliterator;
import java.util.Spliterators;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.StreamSupport;

import org.json.JSONObject;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.Sqrt;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

import com.google.common.base.Joiner;

import edu.cornell.cs.nlp.spf.base.token.TokenSeq;
import edu.cornell.cs.nlp.spf.ccg.categories.Category;
import edu.cornell.cs.nlp.spf.ccg.categories.syntax.Slash;
import edu.cornell.cs.nlp.spf.ccg.categories.syntax.Syntax;
import edu.cornell.cs.nlp.spf.ccg.categories.syntax.Syntax.SimpleSyntax;
import edu.cornell.cs.nlp.spf.ccg.lexicon.ILexiconImmutable;
import edu.cornell.cs.nlp.spf.ccg.lexicon.LexicalEntry;
import edu.cornell.cs.nlp.spf.data.collection.IDataCollection;
import edu.cornell.cs.nlp.spf.data.singlesentence.SingleSentence;
import edu.cornell.cs.nlp.spf.mr.lambda.LogicalExpression;
import edu.cornell.cs.nlp.spf.mr.language.type.Type;
import edu.cornell.cs.nlp.spf.parser.ParsingOp;
import edu.cornell.cs.nlp.spf.parser.ccg.rnn.shiftreduce.dataset.CompositeDataPoint;
import edu.cornell.cs.nlp.spf.parser.ccg.rnn.shiftreduce.dataset.CompositeDataPointDecision;
import edu.cornell.cs.nlp.spf.parser.ccg.rnn.shiftreduce.embeddings.AbstractEmbedding;
import edu.cornell.cs.nlp.spf.parser.ccg.rnn.shiftreduce.learning.Helper;
import edu.cornell.cs.nlp.spf.parser.ccg.rnn.shiftreduce.learning.LearningRate;
import edu.cornell.cs.nlp.spf.parser.ccg.rnn.shiftreduce.learning.LearningRateStats;
import edu.cornell.cs.nlp.spf.parser.ccg.rnn.shiftreduce.recursive.AveragingNetwork;
import edu.cornell.cs.nlp.spf.parser.ccg.rnn.shiftreduce.recursive.GradientWrapper;
import edu.cornell.cs.nlp.spf.parser.ccg.rnn.shiftreduce.recursive.RecursiveTreeNetwork;
import edu.cornell.cs.nlp.spf.parser.ccg.rnn.shiftreduce.recursive.Tree;
import edu.uw.cs.lil.amr.data.LabeledAmrSentence;
import edu.uw.cs.lil.amr.lambda.AMRServices;
import edu.cornell.cs.nlp.utils.log.ILogger;
import edu.cornell.cs.nlp.utils.log.LogLevel;
import edu.cornell.cs.nlp.utils.log.LoggerFactory;

/**
 * Creates embedding of a category
 * @author Dipendra Misra 
 * */
public class CategoryEmbedding<MR> implements AbstractEmbedding {
	
	public static final ILogger LOG = LoggerFactory.create(CategoryEmbedding.class);

	/** recursive network for embedding syntax */
	private final int syntaxNOut;
	private final RecursiveTreeNetwork syntaxsRecursiveNetwork;
	
	/** Dimensionality of simple syntax labels. Dimensionality of attributes is given by
	 * syntaxNOut - simpleSyntaxNOut. This is since simple syntax labels are concatenated
	 * with attribute vectors.*/
	private final int simpleSyntaxNOut;
	private final HashMap<SimpleSyntax, INDArray> simpleSyntaxVectors;
	private final HashMap<SimpleSyntax, GradientWrapper> simpleSyntaxVectorsGrad;
	private final HashMap<SimpleSyntax, INDArray> adaGradSumSquareSimpleSyntax;
	
	private final HashMap<String, INDArray> attributeVectors;
	private final HashMap<String, GradientWrapper> attributeVectorsGrad;
	private final HashMap<String, INDArray> adaGradSumSquareAttribute;
	
	private final HashMap<Slash, INDArray> slashVectors;
	private final HashMap<Slash, GradientWrapper> slashVectorsGrad;
	private final HashMap<Slash, INDArray> adaGradSumSquareSlash;
	
	/** recursive network for embedding semantics */
	private final int semanticsNOut;
	private final RecursiveTreeNetwork semanticsRecursiveNetwork;
	private final HashMap<Type, INDArray> typeVectors;
	private final HashMap<Type, GradientWrapper> typeVectorsGrad;
	private final HashMap<Type, INDArray> adaGradSumSquareType;
	
	private final HashMap<String, INDArray> baseConstantVectors;
	private final HashMap<String, GradientWrapper> baseConstantVectorsGrad;
	private final HashMap<String, INDArray> adaGradSumSquareBaseConstant;
	
	/** Store the parameters that have been used, so that only these parameters are updated in 
	 * that iteration. Make sure these collection are thread-safe and are cleared after update. */
	private final Set<SimpleSyntax> updatedSimpleSyntax;
	private final Set<String> updatedAttribute;
	private final Set<Slash> updatedSlash;
	private final Set<Type> updatedTypes;
	private final Set<String> updatedBaseConstant;
	private final AtomicBoolean updatedNullLogic;
	
	/** embedding of null logical expression */
	private INDArray nullLogic;
	private final GradientWrapper nullLogicGrad;
	private final INDArray adaGradSumSquareNullLogic;
	
	/** learning rate for updating the syntax leaf vectors (simple syntax and slash) and the semantic
	 * leaf vectors (based constant and type) */
	private final LearningRate learningRate;
	
	/** Statistics on learning rate for different embeddings. */
	private final LearningRateStats learningRateStatsSimpleSyntax;
	private final LearningRateStats learningRateStatsAttribute;
	private final LearningRateStats learningRateStatsSlash;
	private final LearningRateStats learningRateStatsTypes;
	private final LearningRateStats learningRateStatsBaseConstant;
	private final LearningRateStats learningRateStatsNullLogic;
	
	/** Store the mean of activations and gradient for debugging */
	private final Map<String, Double> meanActivations;
	private final Map<String, Double> meanGradients;
	
	
	/** Random number generator for generating random category embeddings */
	private final Random rnd;
	private final int seed;
	
	/** l2 Regularization factor */
	private final Double regularizer;
	
	/** If true then use recurisve else averaging */
	private boolean useRecursive;
	
	/** Cache category embeddings. Computing embeddings of category is costly due to feed forwarding
	 * through two deep recursive neural network. Therefore, we cache the category embeddings. These
	 * cached embeddings are invalidated when the model changes (parameters of recursive networks or 
	 * the leaf vectors of recursive networks). Cache syntax and semantic embedding separately 
	 * for better generalization */
	private final ConcurrentHashMap<Syntax, Tree> cacheSyntaxEmbedding;
	private AtomicInteger cacheSyntacticHit, cacheSyntacticMiss;
	private final ConcurrentHashMap<LogicalExpression, Tree> cacheSemanticEmbedding;
	private final int maxCacheSize;
	private AtomicInteger cacheSemanticHit, cacheSemanticMiss;
	
	//For GradientCheck: Empirical Gradients
	public double empiricalSyntaxGrad, empiricalSemanticGrad;
	public double empiricalSyntaxRecursiveW, empiricalSyntaxRecursiveb, 
				  empiricalSemanticRecursiveW, empiricalSemanticRecursiveb;
	private final boolean doGradientchecks;
	
	public CategoryEmbedding(double learningRate, double learningRateDecay, double l2, int seed) {
		
		this.learningRate = new LearningRate(learningRate, learningRateDecay);
		this.regularizer = l2;
		this.seed = seed;
		this.rnd = new Random(seed);
		this.useRecursive = true;
		
		this.learningRateStatsSimpleSyntax = new LearningRateStats();
		this.learningRateStatsAttribute = new LearningRateStats();
		this.learningRateStatsSlash = new LearningRateStats();
		this.learningRateStatsTypes = new LearningRateStats();
		this.learningRateStatsBaseConstant = new LearningRateStats();
		this.learningRateStatsNullLogic = new LearningRateStats();
		
		this.meanActivations = new HashMap<String, Double>();
		this.meanGradients = new HashMap<String, Double>();
		
		this.syntaxNOut = 15;//20;//15;
		this.syntaxsRecursiveNetwork = new RecursiveTreeNetwork(this.syntaxNOut, this.learningRate,
															  this.regularizer, this.rnd, this.seed);
		
		this.simpleSyntaxNOut = 12;
		this.simpleSyntaxVectors = new HashMap<SimpleSyntax, INDArray>();
		this.simpleSyntaxVectorsGrad = new HashMap<SimpleSyntax, GradientWrapper>();
		this.adaGradSumSquareSimpleSyntax = new HashMap<SimpleSyntax, INDArray>();
		
		this.slashVectors = new HashMap<Slash, INDArray>();
		this.slashVectorsGrad = new HashMap<Slash, GradientWrapper>();
		this.adaGradSumSquareSlash = new HashMap<Slash, INDArray>();
		
		this.attributeVectors = new HashMap<String, INDArray>();
		this.attributeVectorsGrad = new HashMap<String, GradientWrapper>();
		this.adaGradSumSquareAttribute = new HashMap<String, INDArray>();
		
		//initialize the syntax vector
		this.initializeSyntaxVector();
		
		this.semanticsNOut = 35; //30;//40;//30;//35;
		this.semanticsRecursiveNetwork = new RecursiveTreeNetwork(this.semanticsNOut, 
											this.learningRate, this.regularizer, this.rnd, this.seed);
		this.baseConstantVectors = new HashMap<String, INDArray>();
		this.baseConstantVectorsGrad = new HashMap<String, GradientWrapper>();
		this.adaGradSumSquareBaseConstant = new HashMap<String, INDArray>();
		
		this.typeVectors = new HashMap<Type, INDArray>();
		this.typeVectorsGrad = new HashMap<Type, GradientWrapper>();
		this.adaGradSumSquareType = new HashMap<Type, INDArray>();
		
		//Null logic initialized using Glorot and Bengio scheme
		double epsilon = 2*Math.sqrt(6.0/(double)(this.semanticsNOut + 1));
		this.nullLogic = this.initUniformlyRandom(this.semanticsNOut, epsilon);
		this.nullLogicGrad = new GradientWrapper(this.semanticsNOut);
		this.adaGradSumSquareNullLogic = Nd4j.zeros(this.semanticsNOut);
		//semantics will be initialized from training data
		
		this.maxCacheSize = 400000;
		this.cacheSyntaxEmbedding = new ConcurrentHashMap<Syntax, Tree>();
		this.cacheSemanticEmbedding = new ConcurrentHashMap<LogicalExpression, Tree>();
		
		this.cacheSyntacticHit = new AtomicInteger();
		this.cacheSyntacticMiss = new AtomicInteger();
		this.cacheSemanticHit = new AtomicInteger();
		this.cacheSemanticMiss = new AtomicInteger();
		
		this.updatedSimpleSyntax = Collections.synchronizedSet(new HashSet<SimpleSyntax>());
		this.updatedAttribute = Collections.synchronizedSet(new HashSet<String>());
		this.updatedSlash = Collections.synchronizedSet(new HashSet<Slash>());
		this.updatedTypes = Collections.synchronizedSet(new HashSet<Type>());
		this.updatedBaseConstant = Collections.synchronizedSet(new HashSet<String>());
		this.updatedNullLogic = new AtomicBoolean(false);
		
		LOG.info("Category Embedding with syntax dimensionality %s, semantic dimensionality %s. Recursive %s", 
											this.syntaxNOut, this.semanticsNOut, this.useRecursive);
		
		this.doGradientchecks = false;
		LOG.setCustomLevel(LogLevel.INFO);
		
		/*GPUServices gpuService = new GPUServices(this.semanticsNOut);
		gpuService.test();
		System.exit(0);*/
	}
	
	
	/** Returns a vector of size 1xdim, with each value uniformly
	 * randomly in [-epsilon, epsilon] */
	public INDArray initUniformlyRandom(int dim, double epsilon) {
		
		int localSeed = Math.abs(this.rnd.nextInt(2 * this.seed)) + 1000;
		INDArray vec = Nd4j.rand(new int[]{1, dim}, localSeed);
		vec.subi(0.5).muli(epsilon);
		
		return vec;
	}
	
	/** Initialize vectors for syntactic components namely slashes (e.g., /,\) and
	 *  simpleSyntax (e.g., NP, S). Initialization is done as uniformly random
	 *  in [-\sqrt{6/(r+c)}, \sqrt{6/(r+c)}] (Glorot and Bengio, 10)  
	 */
	private void initializeSyntaxVector() {
		
		double epsilonSimpleSyntax = 2*Math.sqrt(6.0/(double)(this.simpleSyntaxNOut + 1));
		double epsilon = 2*Math.sqrt(6.0/(double)(this.syntaxNOut + 1));
		double epsilonAttribute = 2*Math.sqrt(6.0/(double)(this.syntaxNOut - this.simpleSyntaxNOut + 1));

		for(SimpleSyntax simpleSyntax: Syntax.getAllSimpleSyntax()) {
			
			INDArray simpleSyntaxVector = this.initUniformlyRandom(this.simpleSyntaxNOut, epsilonSimpleSyntax);
			this.simpleSyntaxVectors.put(simpleSyntax, simpleSyntaxVector);
			this.simpleSyntaxVectorsGrad.put(simpleSyntax, new GradientWrapper(this.simpleSyntaxNOut));
			this.adaGradSumSquareSimpleSyntax.put(simpleSyntax, Nd4j.zeros(this.simpleSyntaxNOut));
		}
		
		//initialize slash vectors
		INDArray forwardSlashVector = this.initUniformlyRandom(this.syntaxNOut, epsilon);
		this.slashVectors.put(Slash.FORWARD, forwardSlashVector);
		this.slashVectorsGrad.put(Slash.FORWARD, new GradientWrapper(this.syntaxNOut));
		this.adaGradSumSquareSlash.put(Slash.FORWARD, Nd4j.zeros(this.syntaxNOut));
		
		INDArray backwardSlashVector = this.initUniformlyRandom(this.syntaxNOut, epsilon);
		this.slashVectors.put(Slash.BACKWARD, backwardSlashVector);
		this.slashVectorsGrad.put(Slash.BACKWARD, new GradientWrapper(this.syntaxNOut));
		this.adaGradSumSquareSlash.put(Slash.BACKWARD, Nd4j.zeros(this.syntaxNOut));
		
		INDArray verticalSlashVector = this.initUniformlyRandom(this.syntaxNOut, epsilon);
		this.slashVectors.put(Slash.VERTICAL, verticalSlashVector);
		this.slashVectorsGrad.put(Slash.VERTICAL, new GradientWrapper(this.syntaxNOut));
		this.adaGradSumSquareSlash.put(Slash.VERTICAL, Nd4j.zeros(this.syntaxNOut));
		
		//initialize attribute vector
		List<String> attributes = Arrays.asList(Syntax.VARIABLE_ATTRIBUTE, null, "nb", "sg", "pl");
		for(String attribute: attributes) {
			INDArray attributeVector = this.initUniformlyRandom(this.syntaxNOut - this.simpleSyntaxNOut, epsilonAttribute);
			this.attributeVectors.put(attribute, attributeVector);
			this.attributeVectorsGrad.put(attribute, new GradientWrapper(this.syntaxNOut - this.simpleSyntaxNOut));
			this.adaGradSumSquareAttribute.put(attribute, Nd4j.zeros(this.syntaxNOut - this.simpleSyntaxNOut));
		}
	}
	
	///////////////// TEMPORARY HACK
	public INDArray getSyntaxVector() {
		return this.simpleSyntaxVectors.get(Syntax.NP);
	}
	
	public INDArray getSemanticVector() {
		for(Entry<String, INDArray> e: this.baseConstantVectors.entrySet()) {
			if(e.getKey().compareToIgnoreCase("and") == 0) {
				return e.getValue();
			}
		}
		return null;
	}
	
	public INDArray getSemanticGradVector() {
		for(Entry<String, GradientWrapper> e: this.baseConstantVectorsGrad.entrySet()) {
			if(e.getKey().compareToIgnoreCase("and") == 0) {
				return e.getValue().getGradient();
			}
		}
		return null;
	}
	
	public INDArray getSyntaxRecursiveW() {
		return this.syntaxsRecursiveNetwork.getW();
	}
	
	public INDArray getSyntaxRecursiveb() {
		return this.syntaxsRecursiveNetwork.getb();
	}
	
	public INDArray getSemanticRecursiveW() {
		return this.semanticsRecursiveNetwork.getW();
	}
	
	public INDArray getSemanticRecursiveb() {
		return this.semanticsRecursiveNetwork.getb();
	}
	/////////////////
	
	public void induceCategoricalVectors(IDataCollection<SingleSentence> trainingData) {
		
		//unseen type and logical constant can be seen at test time hence add UNK
		INDArray unkTypeVector = Nd4j.zeros(this.semanticsNOut);
		this.typeVectors.put(null, unkTypeVector); //for type, null does the job of $UNK$
		this.typeVectorsGrad.put(null, new GradientWrapper(this.semanticsNOut));
		
		INDArray unkBaseConstantVector = Nd4j.zeros(this.semanticsNOut);
		this.baseConstantVectors.put("$UNK$", unkBaseConstantVector);
		this.baseConstantVectorsGrad.put("$UNK$", new GradientWrapper(this.semanticsNOut));
		
		for(SingleSentence dataItem: trainingData) {
			
			final LogicalExpression exp = (LogicalExpression)(dataItem.getLabel());
			AddSemanticConstantsVisitor.addSemanticConstant(exp, this.semanticsNOut, 
					this.baseConstantVectors, this.baseConstantVectorsGrad, this.typeVectors,
					this.typeVectorsGrad, this.rnd, this.seed);		
		}
		
		LOG.debug("Type Vectors");
		for(Entry<Type, INDArray> e: this.typeVectors.entrySet()) {
			this.adaGradSumSquareType.put(e.getKey(), Nd4j.zeros(this.semanticsNOut));
			LOG.debug(e.getKey());
		}
		
		LOG.debug("Base Names of Logical Constants");
		for( Entry<String, INDArray> e: this.baseConstantVectors.entrySet()) {
			this.adaGradSumSquareBaseConstant.put(e.getKey(), Nd4j.zeros(this.semanticsNOut));
			LOG.debug(e.getKey()+" ");
		}
		
		LOG.info("Number of type vectors %s", this.typeVectors.size());
		LOG.info("Number of base constant vectors %s", this.baseConstantVectors.size());
	}
	
	public void initializeAmrSpecificSyntaxVectors() {
	
		double epsilon = 2*Math.sqrt(6.0/(double)(this.simpleSyntaxNOut + 1));
		
		List<SimpleSyntax> amrSpecificSimpleSyntax = new LinkedList<SimpleSyntax>();
		amrSpecificSimpleSyntax.add(AMRServices.AMR);
		amrSpecificSimpleSyntax.add(AMRServices.I);
		amrSpecificSimpleSyntax.add(AMRServices.ID);
		amrSpecificSimpleSyntax.add(AMRServices.KEY);
		amrSpecificSimpleSyntax.add(AMRServices.TXT);

		for(SimpleSyntax simpleSyntax: amrSpecificSimpleSyntax) {
			
			INDArray simpleSyntaxVector = this.initUniformlyRandom(this.simpleSyntaxNOut, epsilon);
			this.simpleSyntaxVectors.put(simpleSyntax, simpleSyntaxVector);
			this.simpleSyntaxVectorsGrad.put(simpleSyntax, new GradientWrapper(this.simpleSyntaxNOut));
			this.adaGradSumSquareSimpleSyntax.put(simpleSyntax, Nd4j.zeros(this.simpleSyntaxNOut));
		}
		
		/* There are some patterns as well for which we currently use null. This solution
		 * is temporary since simple syntax should be constant of the problem. */
	}
	
	public void induceCategoricalVectorsFromAmr(IDataCollection<LabeledAmrSentence> trainingData) {
		
		//unseen type and logical constant can be seen at test time hence add UNK
		INDArray unkTypeVector = Nd4j.zeros(this.semanticsNOut);
		this.typeVectors.put(null, unkTypeVector); //for type, null does the job of $UNK$
		this.typeVectorsGrad.put(null, new GradientWrapper(this.semanticsNOut));
		
		INDArray unkBaseConstantVector = Nd4j.zeros(this.semanticsNOut);
		this.baseConstantVectors.put("$UNK$", unkBaseConstantVector);
		this.baseConstantVectorsGrad.put("$UNK$", new GradientWrapper(this.semanticsNOut));
		
		for(LabeledAmrSentence dataItem: trainingData) {
			
			final LogicalExpression exp = (LogicalExpression)(dataItem.getLabel());
			
			//Convert Logical Expression to underspecified form 
			final LogicalExpression underspecified = AMRServices.underspecifyAndStrip(exp);
				
			AddSemanticConstantsVisitor.addSemanticConstant(underspecified, this.semanticsNOut, 
					this.baseConstantVectors, this.baseConstantVectorsGrad, this.typeVectors,
					this.typeVectorsGrad, this.rnd, this.seed);		
		}
		
		LOG.debug("Type Vectors");
		for(Entry<Type, INDArray> e: this.typeVectors.entrySet()) {
			this.adaGradSumSquareType.put(e.getKey(), Nd4j.zeros(this.semanticsNOut));
			LOG.debug(e.getKey());
		}
		
		LOG.debug("Base Names of Logical Constants");
		for( Entry<String, INDArray> e: this.baseConstantVectors.entrySet()) {
			this.adaGradSumSquareBaseConstant.put(e.getKey(), Nd4j.zeros(this.semanticsNOut));
			LOG.debug(e.getKey()+" ");
		}
		
		LOG.info("Number of type vectors from raw data %s", this.typeVectors.size());
		LOG.info("Number of base constant vectors from raw data %s", this.baseConstantVectors.size());
	}
	
	/** Sometimes training data does not cover all category vectors. We therefore, all bootstrap from 
	 * processed dataset i.e. which is used for training neural parser. Be aware that this extra
	 * bootstrapping while increasing coverage, can be time consuming. */
	public void induceVectorsFromProcessedDataset(List<CompositeDataPoint<MR>> dataPoint) {
		
		for(CompositeDataPoint<MR> point: dataPoint) {
			List<CompositeDataPointDecision<MR>> decisions = point.getDecisions();
			for(CompositeDataPointDecision<MR> decision: decisions) {
				List<ParsingOp<MR>> actions = decision.getPossibleActions();
				for(ParsingOp<MR> action: actions) {
					
					final Category<MR> category = action.getCategory();
					final LogicalExpression exp = (LogicalExpression)(category.getSemantics());
					
					if(exp == null) {
						continue;
					}
					
					//Convert Logical Expression to underspecified form 
					final LogicalExpression underspecified = AMRServices.underspecifyAndStrip(exp);
						
					AddSemanticConstantsVisitor.addSemanticConstant(underspecified, this.semanticsNOut, 
							this.baseConstantVectors, this.baseConstantVectorsGrad, this.typeVectors,
							this.typeVectorsGrad, this.rnd, this.seed);		
				}
			}
		}
		

		for(Entry<Type, INDArray> e: this.typeVectors.entrySet()) {
			this.adaGradSumSquareType.put(e.getKey(), Nd4j.zeros(this.semanticsNOut));
		}
		
		for( Entry<String, INDArray> e: this.baseConstantVectors.entrySet()) {
			this.adaGradSumSquareBaseConstant.put(e.getKey(), Nd4j.zeros(this.semanticsNOut));
		}
		
		LOG.info("Number of type vectors from processed dataset %s", this.typeVectors.size());
		LOG.info("Number of base constant vectors from processed dataset %s", this.baseConstantVectors.size());
	}
	
	/** This function stores embedding of categories of all lexical entries in the lexicon
	 * that can be used in the test data. If the test data is null the this function computes
	 * embedding of all the lexical entries in the lexicon. This is highly unrecommended since
	 * lexical can be very large for AMR.*/
	@SuppressWarnings("unchecked")
	public void memoizeLexicalEntryEmbedding(IDataCollection<LabeledAmrSentence> testData, 
												ILexiconImmutable<LogicalExpression> lexicon) {
		
		Collection<Category<LogicalExpression>> categorySpace = 
										new HashSet<Category<LogicalExpression>>();
		
		if(testData == null) {
			Collection<LexicalEntry<LogicalExpression>> lexicalEntrySpace = lexicon.toCollection();
			for(LexicalEntry<LogicalExpression> lexicalEntry: lexicalEntrySpace) {
				categorySpace.add(lexicalEntry.getCategory());
			}
		} else {
			categorySpace = new HashSet<Category<LogicalExpression>>();
			
			for(LabeledAmrSentence labeledAmrSentence: testData) {
				TokenSeq tk = labeledAmrSentence.getSample().getTokens();
				int n = tk.size();
				
				for(int i = 0; i < n; i++) {
					for(int j = i; j < n; j++) {
						//sequence [i,j]
						Iterator<? extends LexicalEntry<LogicalExpression>> lexicalEntries = 
														lexicon.get(tk.sub(i, j + 1));
						while(lexicalEntries.hasNext()) {
							categorySpace.add(lexicalEntries.next().getCategory());
						}
					}
				}
			}
		}
		
		LOG.info("Going to embed %s lexical entry categories", categorySpace.size());
		
		StreamSupport.stream(Spliterators
				.spliterator(categorySpace, Spliterator.IMMUTABLE), true).unordered()
				.forEach(category -> {
					
					if(category.getSemantics() == null) {
						LOG.info("Logical expression was null");
					}
					this.getCategoryEmbedding((Category<MR>)category).getEmbedding();
				});
		
		this.logCachePerformance();
		LOG.info("Size of cache is %s", this.cacheSemanticEmbedding.size());
		
		this.cacheSemanticHit.set(0);
		this.cacheSemanticMiss.set(0);
		this.cacheSyntacticHit.set(0);
		this.cacheSyntacticMiss.set(0);
	}
	
	/** returns embedding of a category under the given model. The embedding of a category is
	 * given by the concatenation of the embedding of syntax and semantics. This is a costly
	 * operation and therefore a cache is used to optimize. The cache is emptied after every
	 * update step or after bootstrapping a model. Similarly, the updated parameters must be cleared
	 * after every iteration or after bootstrapping a model. */
	public CategoryEmbeddingResult getCategoryEmbedding(Category<MR> categ) {
		
		//find embedding of syntax
		Tree syntacticTree = this.findSyntaxInCache(categ.getSyntax());
		if(syntacticTree == null) {
			
			this.cacheSyntacticMiss.incrementAndGet();
			
			syntacticTree = SyntaxsVisitor.embedSyntaxs(categ.getSyntax(), this.syntaxsRecursiveNetwork,
									this.simpleSyntaxVectors, this.simpleSyntaxVectorsGrad, this.attributeVectors, 
									this.attributeVectorsGrad, this.slashVectors, this.slashVectorsGrad, this.updatedAttribute, 
									this.updatedSimpleSyntax, this.updatedSlash, this.useRecursive);
			this.updateSyntaxCache(categ.getSyntax(), syntacticTree);
		} else { 
			this.cacheSyntacticHit.incrementAndGet();
		}
			
		INDArray syntacticEnc = syntacticTree.getVector();
//		INDArray averageSyntax = AveragingNetwork.average(syntacticTree);
//		INDArray syntacticEnc = Nd4j.concat(1, syntacticEnc1, averageSyntax);
		
		//find embedding of semantics
		MR semantic = categ.getSemantics();
		/*if(!(semantic instanceof LogicalExpression)) // --- not sure what to do
			throw new RuntimeException("MR should be a logical expression. Cannot encode other MRs");*/
		
		final LogicalExpression exp = (LogicalExpression)semantic;
		Tree semanticTree = this.findSemanticInCache(exp);
		if(semanticTree == null) {
			
			this.cacheSemanticMiss.incrementAndGet();
			
			semanticTree = SemanticsVisitor.embedSemantics(exp, 
				this.semanticsRecursiveNetwork, this.baseConstantVectors, this.baseConstantVectorsGrad,
				this.typeVectors, this.typeVectorsGrad, this.nullLogic, this.nullLogicGrad, this.updatedTypes, 
				this.updatedBaseConstant, this.updatedNullLogic, this.useRecursive);
			this.updateSemanticCache(exp, semanticTree);
		} else {
			this.cacheSemanticHit.incrementAndGet();
		}
		
		INDArray semanticEnc = semanticTree.getVector();
//		INDArray averageSemantic = AveragingNetwork.average(semanticTree);
//		INDArray semanticEnc = Nd4j.concat(1, semanticEnc1, averageSemantic);
		
		//return the concatenated vectors
		INDArray embedding = Nd4j.concat(1, syntacticEnc, semanticEnc);
		CategoryEmbeddingResult result = new CategoryEmbeddingResult(embedding, syntacticTree, semanticTree);
		
		return result;
	}
	
	public void backprop(Tree syntacticTree, Tree semanticsTree, INDArray error) {
		
		double errorNorm2 = error.normmaxNumber().doubleValue();
		
		LOG.debug("Category Embedding with error %s, ", error);
		
		if(Double.isNaN(errorNorm2)) {
			LOG.info("Problem with backprop in Category Embedding. Got NaN.");
		}
		
		if(Double.isInfinite(errorNorm2)) {
			LOG.info("Problem with backprop in Category Embedding. Got infinite.");
		}		
		
		//split the error
		INDArray syntaxError = error.get(NDArrayIndex.interval(0, this.syntaxNOut));
//		INDArray averageSyntaxError = error.get(NDArrayIndex.interval(this.syntaxNOut, 2*this.syntaxNOut));
		INDArray semanticsError = error.get(NDArrayIndex.interval(this.syntaxNOut,  
															this.syntaxNOut + this.semanticsNOut));
//		INDArray averageSemanticError = error.get(NDArrayIndex.interval(2*this.syntaxNOut + this.semanticsNOut,   
//															2*this.syntaxNOut + 2*this.semanticsNOut));
//		
//		AveragingNetwork.backprop(syntacticTree, averageSyntaxError);
//		AveragingNetwork.backprop(semanticsTree, averageSemanticError);
		
		LOG.debug("{--- syntax ---");
		if(this.useRecursive) {
			this.syntaxsRecursiveNetwork.backProp(syntacticTree, syntaxError);
		} else {
			AveragingNetwork.backprop(syntacticTree, syntaxError);
		}
		LOG.debug("}\n{--- semantic ---");
		if(this.useRecursive) {
			this.semanticsRecursiveNetwork.backProp(semanticsTree, semanticsError);
		} else {
			AveragingNetwork.backprop(semanticsTree, semanticsError);
		}
		LOG.debug("---}\n");
	}

	@Override
	public int getDimension() {
		return this.syntaxNOut + this.semanticsNOut;
	}

	@Override
	public Object getEmbedding(Object obj) {
		throw new RuntimeException("Operation not supported");
	}
	
	private void updateSyntaxCache(Syntax syntax, Tree syntacticTree) {
		
		 //use smarter scheme such as LRU here in future
		if(this.cacheSyntaxEmbedding.size() > this.maxCacheSize)
			return;
			
		this.cacheSyntaxEmbedding.put(syntax, syntacticTree);
	}
	
	private void updateSemanticCache(LogicalExpression exp, Tree semanticTree) {
		
		//use smarter scheme such as LRU here in future
		if(exp == null || this.cacheSemanticEmbedding.size() > this.maxCacheSize)
			return;
			
		this.cacheSemanticEmbedding.put(exp, semanticTree);
	}
	
	private Tree findSyntaxInCache(Syntax syntax) {		
		return this.cacheSyntaxEmbedding.get(syntax);
	}
	
	private Tree findSemanticInCache(LogicalExpression exp) {		
		if(exp == null) {
			return null;
		}
		
		return this.cacheSemanticEmbedding.get(exp);
	}
	
	public void invalidateCache() {
		//cache is used by multiple threads so synchronize access to it
		synchronized(this.cacheSyntaxEmbedding) {
			this.cacheSyntaxEmbedding.clear();
		}
		
		synchronized(this.cacheSemanticEmbedding) {
			this.cacheSemanticEmbedding.clear();
		}
	}
	
	public void logCachePerformance() {
		
		assert this.cacheSyntacticHit.get()+this.cacheSyntacticMiss.get() == 
			this.cacheSemanticHit.get() + this.cacheSemanticMiss.get() : "Cache is buggy";
			
		LOG.info("Total Calls to cache " + (this.cacheSyntacticHit.get()+this.cacheSyntacticMiss.get()));
		
		double syntacticCacheHit = this.cacheSyntacticHit.get()
				/(double)Math.max(this.cacheSyntacticHit.get() + this.cacheSyntacticMiss.get(), 1);
		LOG.info("Total Syntactic Cache Miss %s", this.cacheSyntacticMiss.get());
		LOG.info("Syntactic Cache Hit %s", syntacticCacheHit);
		
		double semanticCacheHit = this.cacheSemanticHit.get()
				/(double)Math.max(this.cacheSemanticHit.get() + this.cacheSemanticMiss.get(), 1);
		LOG.info("Total Semantic Cache Miss %s", this.cacheSemanticMiss.get());
		LOG.info("Semantic Cache Hit %s", semanticCacheHit);
		
		LOG.info("Size of the syntactic cache "+this.cacheSyntaxEmbedding.size());
		LOG.info("Size of the semantic cache "+this.cacheSemanticEmbedding.size());
	}
	
	/** performs gradient descent on the category vectors using rawGradient representing
	 *  del loss /del vector. One can add more terms to loss such as L2 norm with respect to vector.
	 *  Sum of square of gradient is passed for AdaGrad  learning rate. 
	 *  Warning: rawGradient gets affected by this function.
	 *  */
	private INDArray updateVectors(INDArray vector, GradientWrapper rawGradient, 
								  INDArray sumSquareGradient, LearningRateStats learningRateStats, String label) {
		
		int numTerms = rawGradient.numTerms();
		if(numTerms == 0) {	//no terms to add
			return vector;
		}
		
		INDArray sumGradient = rawGradient.getGradient();
		
		if(sumGradient.normmaxNumber().doubleValue() == 0) {
			LOG.warn("Category: 0 gradient found");
		}
		
		//// Code below is for debugging
		final double meanActivation = Helper.meanAbs(vector);
		final double meanGradient = Helper.meanAbs(sumGradient);
		
		synchronized(this.meanActivations) {
			if(this.meanActivations.containsKey(label)) {
				double oldVal = this.meanActivations.get(label);
				this.meanActivations.put(label, meanActivation + oldVal);
			} else {
				this.meanActivations.put(label, meanActivation);
			}
		}
		
		synchronized(this.meanGradients) {
			if(this.meanGradients.containsKey(label)) {
				double oldVal = this.meanGradients.get(label);
				this.meanGradients.put(label, meanGradient + oldVal);
			} else {
				this.meanGradients.put(label, meanGradient);
			}
		}
		///////
		
		//Add regularization
		sumGradient.addi(vector.mul(this.regularizer));
	
		//Clip the gradient
		double norm = sumGradient.normmaxNumber().doubleValue();
		double threshold = 1.0;
		
		if(norm > threshold) {
			sumGradient.muli(threshold/norm);
		}
		
		if(norm > 0.0) {
			LOG.debug("Category Vector, Learning Rate %s, Gradient norm %s, max %s, min %s ",
					this.learningRate.getLearningRate(), sumGradient.normmaxNumber().doubleValue(), 
					sumGradient.maxNumber().doubleValue(), sumGradient.minNumber().doubleValue());
		}
		
		//Perform AdaGrad based SGD step
		sumSquareGradient.addi(sumGradient.mul(sumGradient));
		
		if(sumSquareGradient.normmaxNumber().doubleValue() == 0) {
			sumSquareGradient.addi(0.00001);
		}
		
		double initLearningRate = this.learningRate.getLearningRate();
		
		INDArray invertedLearningRate = Nd4j.getExecutioner()
											.execAndReturn(new Sqrt(sumSquareGradient.dup()))
											.divi(initLearningRate);
		
		double minLearningRate = /*sumGradient.minNumber().doubleValue();*/1.0/(invertedLearningRate.maxNumber().doubleValue());
		double maxLearningRate = /*sumGradient.maxNumber().doubleValue();*/1.0/(invertedLearningRate.minNumber().doubleValue());
		
		synchronized(learningRateStats) {
			learningRateStats.min(minLearningRate);
			learningRateStats.max(maxLearningRate);
		}
		
		//sumGradient.muli(this.learningRate.getLearningRate());
		sumGradient.divi(invertedLearningRate);
		
		vector.subi(sumGradient);
		
		return vector;
	}
	
	public void updateParameters() {
		//Use Stream
		//the embeddings in cache are no longer true therefore invalidate them
		this.invalidateCache();
		
		LOG.info("Activation:: W-recursive-syntax %s b-recursive-syntax %s", Helper.meanAbs(this.syntaxsRecursiveNetwork.getW()), 
																			Helper.meanAbs(this.syntaxsRecursiveNetwork.getb()));
		LOG.info("Activation:: W-recursive-semantic %s b-recursive-semantic %s", Helper.meanAbs(this.semanticsRecursiveNetwork.getW()), 
																				Helper.meanAbs(this.semanticsRecursiveNetwork.getb()));
		LOG.info("Gradient:: W-recursive-syntax %s b-recursive-syntax %s", Helper.meanAbs(this.syntaxsRecursiveNetwork.getGradW()), 
									  Helper.meanAbs(this.syntaxsRecursiveNetwork.getGradb()));
		LOG.info("Gradient:: W-recursive-semantic %s  b-recursive-semantic %s", Helper.meanAbs(this.semanticsRecursiveNetwork.getGradW()), 
									  Helper.meanAbs(this.semanticsRecursiveNetwork.getGradb()));
		
		this.learningRateStatsSimpleSyntax.unset();
		this.learningRateStatsAttribute.unset();
		this.learningRateStatsSlash.unset();
		this.learningRateStatsBaseConstant.unset();
		this.learningRateStatsTypes.unset();
		this.learningRateStatsNullLogic.unset();
		
		if(this.doGradientchecks) {
			LOG.info("Gradient Check. Syntax. Empirical %s. Estimate %s", this.empiricalSyntaxGrad, 
							this.simpleSyntaxVectorsGrad.get(SimpleSyntax.NP).getGradient().getDouble(new int[]{0, 0}));
			LOG.info("Gradient Check. Semantic. Empirical %s. Estimate %s", this.empiricalSemanticGrad, 
							this.getSemanticGradVector().getDouble(new int[]{0, 0}));
			LOG.info("Gradient Check. Syntax Recursive W. Empirical %s. Estimate %s", this.empiricalSyntaxRecursiveW, 
					this.syntaxsRecursiveNetwork.getGradW().getDouble(new int[]{0, 0}));
			LOG.info("Gradient Check. Syntax Recursive b. Empirical %s. Estimate %s", this.empiricalSyntaxRecursiveb, 
					this.syntaxsRecursiveNetwork.getGradb().getDouble(new int[]{0, 0}));
			LOG.info("Gradient Check. Semantic Recursive W. Empirical %s. Estimate %s", this.empiricalSemanticRecursiveW, 
					this.semanticsRecursiveNetwork.getGradW().getDouble(new int[]{0, 0}));
			LOG.info("Gradient Check. Semantic Recursive b. Empirical %s. Estimate %s", this.empiricalSemanticRecursiveb, 
					this.semanticsRecursiveNetwork.getGradb().getDouble(new int[]{0, 0}));
		}
		
		
		
//		StreamSupport.stream(Spliterators
//				.spliterator(this.simpleSyntaxVectors.entrySet(), Spliterator.IMMUTABLE), true)
//				.unordered()
//				.forEach(p-> this.updateVectors(p.getValue(), this.simpleSyntaxVectorsGrad.get(p.getKey()), 
//						 this.adaGradSumSquareSimpleSyntax.get(p.getKey())));
//		
//		StreamSupport.stream(Spliterators
//				.spliterator(this.slashVectors.entrySet(), Spliterator.IMMUTABLE), true)
//				.unordered()
//				.forEach(p-> this.updateVectors(p.getValue(), this.slashVectorsGrad.get(p.getKey()), 
//						 this.adaGradSumSquareSlash.get(p.getKey())));
//		
//		StreamSupport.stream(Spliterators
//				.spliterator(this.attributeVectors.entrySet(), Spliterator.IMMUTABLE), true)
//				.unordered()
//				.forEach(p-> this.updateVectors(p.getValue(), this.attributeVectorsGrad.get(p.getKey()), 
//						 this.adaGradSumSquareAttribute.get(p.getKey())));
		
		StreamSupport.stream(Spliterators
				.spliterator(this.updatedSimpleSyntax, Spliterator.IMMUTABLE), true)
				.unordered()
				.forEach(p-> this.updateVectors(this.simpleSyntaxVectors.get(p), this.simpleSyntaxVectorsGrad.get(p), 
						 this.adaGradSumSquareSimpleSyntax.get(p), this.learningRateStatsSimpleSyntax, "Simple-Syntax"));
		
		StreamSupport.stream(Spliterators
				.spliterator(this.updatedSlash, Spliterator.IMMUTABLE), true)
				.unordered()
				.forEach(p-> this.updateVectors(this.slashVectors.get(p), this.slashVectorsGrad.get(p), 
						 this.adaGradSumSquareSlash.get(p), this.learningRateStatsSlash, "Slash"));
		
		StreamSupport.stream(Spliterators
				.spliterator(this.updatedAttribute, Spliterator.IMMUTABLE), true)
				.unordered()
				.forEach(p-> this.updateVectors(this.attributeVectors.get(p), this.attributeVectorsGrad.get(p), 
						 this.adaGradSumSquareAttribute.get(p), this.learningRateStatsAttribute, "Attribute"));
		
		this.syntaxsRecursiveNetwork.updateParameters();
		
		if(this.updatedNullLogic.get()) { 
			this.updateVectors(this.nullLogic, this.nullLogicGrad, 
								this.adaGradSumSquareNullLogic, this.learningRateStatsNullLogic, "Null-Logic");
		}
		
//		StreamSupport.stream(Spliterators
//				.spliterator(this.baseConstantVectors.entrySet(), Spliterator.IMMUTABLE), true)
//				.unordered()
//				.forEach(p-> this.updateVectors(p.getValue(), this.baseConstantVectorsGrad.get(p.getKey()), 
//						 this.adaGradSumSquareBaseConstant.get(p.getKey())));
//		
//		StreamSupport.stream(Spliterators
//				.spliterator(this.typeVectors.entrySet(), Spliterator.IMMUTABLE), true)
//				.unordered()
//				.forEach(p-> this.updateVectors(p.getValue(), this.typeVectorsGrad.get(p.getKey()), 
//						 this.adaGradSumSquareType.get(p.getKey())));
		
		StreamSupport.stream(Spliterators
				.spliterator(this.updatedBaseConstant, Spliterator.IMMUTABLE), true)
				.unordered()
				.forEach(p-> this.updateVectors(this.baseConstantVectors.get(p), this.baseConstantVectorsGrad.get(p), 
						 this.adaGradSumSquareBaseConstant.get(p), this.learningRateStatsBaseConstant, "Base-Constant"));
		
		StreamSupport.stream(Spliterators
				.spliterator(this.updatedTypes, Spliterator.IMMUTABLE), true)
				.unordered()
				.forEach(p-> this.updateVectors(this.typeVectors.get(p), this.typeVectorsGrad.get(p), 
						 this.adaGradSumSquareType.get(p), this.learningRateStatsTypes, "Type"));
		
		this.semanticsRecursiveNetwork.updateParameters();
		
//		LOG.info("Category Embedding. Learning Rate Simple Syntax Stats %s", this.learningRateStatsSimpleSyntax);
//		LOG.info("Category Embedding. Learning Rate Attribute Stats %s", this.learningRateStatsAttribute);
//		LOG.info("Category Embedding. Learning Rate Slash Stats %s", this.learningRateStatsSlash);
//		LOG.info("Category Embedding. Learning Rate Base Constant Stats %s", this.learningRateStatsBaseConstant);
//		LOG.info("Category Embedding. Learning Rate Type Stats %s", this.learningRateStatsTypes);
//		LOG.info("Category Embedding. Learning Rate Null logic Stats %s", this.learningRateStatsNullLogic);
		
		/////////////
		final Map<String, Integer> counts = new HashMap<String, Integer>();
		counts.put("Simple-Syntax", this.updatedSimpleSyntax.size());
		counts.put("Slash", this.updatedSlash.size());
		counts.put("Attribute", this.updatedAttribute.size());
		if(this.updatedNullLogic.get()) {
			counts.put("Null-Logic", 1);
		} else {
			counts.put("Null-Logic", 0);
		}
		counts.put("Type", this.updatedTypes.size());
		counts.put("Base-Constant", this.updatedBaseConstant.size());
		
		for(Entry<String, Double> e: this.meanActivations.entrySet()) {
			Integer i = counts.get(e.getKey());
			if(i == null || i == 0) {
				LOG.info("Activation:: %s  NA", e.getKey());
			} else {
				LOG.info("Activation:: %s  %s", e.getKey(), e.getValue()/(double)i);
			}
		}
		
		this.meanActivations.clear();
		
		for(Entry<String, Double> e: this.meanGradients.entrySet()) {
			Integer i = counts.get(e.getKey());
			if(i == null || i == 0) {
				LOG.info("Gradient:: %s  NA", e.getKey());
			} else {
				LOG.info("Gradient:: %s  %s", e.getKey(), e.getValue()/(double)i);
			}
		}
		
		this.meanGradients.clear();
		/////////////
	}
	
	public void flushGradients() {
			
//		StreamSupport.stream(Spliterators
//				.spliterator(this.simpleSyntaxVectorsGrad.entrySet(), Spliterator.IMMUTABLE), true)
//				.unordered()
//				.forEach(p-> p.getValue().flush());
//		
//		StreamSupport.stream(Spliterators
//				.spliterator(this.slashVectorsGrad.entrySet(), Spliterator.IMMUTABLE), true)
//				.unordered()
//				.forEach(p-> p.getValue().flush());
//		
//		StreamSupport.stream(Spliterators
//				.spliterator(this.attributeVectorsGrad.entrySet(), Spliterator.IMMUTABLE), true)
//				.unordered()
//				.forEach(p-> p.getValue().flush());
		
		StreamSupport.stream(Spliterators
		.spliterator(this.updatedSimpleSyntax, Spliterator.IMMUTABLE), true)
		.unordered()
		.forEach(p-> this.simpleSyntaxVectorsGrad.get(p).flush());

		StreamSupport.stream(Spliterators
		.spliterator(this.updatedSlash, Spliterator.IMMUTABLE), true)
		.unordered()
		.forEach(p-> this.slashVectorsGrad.get(p).flush());

		StreamSupport.stream(Spliterators
		.spliterator(this.updatedAttribute, Spliterator.IMMUTABLE), true)
		.unordered()
		.forEach(p-> this.attributeVectorsGrad.get(p).flush());
		
		this.updatedSimpleSyntax.clear();
		this.updatedSlash.clear();
		this.updatedAttribute.clear();
		
		this.syntaxsRecursiveNetwork.flushGradients();
		
		if(this.updatedNullLogic.get()) {
			this.nullLogicGrad.flush();
			this.updatedNullLogic.set(false);
		}
		
//		StreamSupport.stream(Spliterators
//				.spliterator(this.baseConstantVectorsGrad.entrySet(), Spliterator.IMMUTABLE), true)
//				.unordered()
//				.forEach(p-> p.getValue().flush());
//		
//		StreamSupport.stream(Spliterators
//				.spliterator(this.typeVectorsGrad.entrySet(), Spliterator.IMMUTABLE), true)
//				.unordered()
//				.forEach(p-> p.getValue().flush());
		
		StreamSupport.stream(Spliterators
				.spliterator(this.updatedBaseConstant, Spliterator.IMMUTABLE), true)
				.unordered()
				.forEach(p-> this.baseConstantVectorsGrad.get(p).flush());
		
		StreamSupport.stream(Spliterators
				.spliterator(this.updatedTypes, Spliterator.IMMUTABLE), true)
				.unordered()
				.forEach(p-> this.typeVectorsGrad.get(p).flush());
		
		this.updatedBaseConstant.clear();
		this.updatedTypes.clear();
		
		this.semanticsRecursiveNetwork.flushGradients();
	}
	
	public void flushAdaGradHistory() {
		
		for(Entry<SimpleSyntax, INDArray> e: this.adaGradSumSquareSimpleSyntax.entrySet()) {
			e.getValue().muli(0);
		}
		
		for(Entry<Slash, INDArray> e: this.adaGradSumSquareSlash.entrySet()) {
			e.getValue().muli(0);
		}
		
		for(Entry<String, INDArray> e: this.adaGradSumSquareAttribute.entrySet()) {
			e.getValue().muli(0);
		}
		
		this.adaGradSumSquareNullLogic.muli(0);
		
		for(Entry<String, INDArray> e: this.adaGradSumSquareBaseConstant.entrySet()) {
			e.getValue().muli(0);
		}
		
		for(Entry<Type, INDArray> e: this.adaGradSumSquareType.entrySet()) {
			e.getValue().muli(0);
		}
	}
	
	/** Decay the learning rate for category embedding */
	public void decay() {
		this.learningRate.decay();
	}
	
	public void printCategoryEmbeddings() {
		
		LOG.info("Simple Syntax Vectors");
		for(Entry<SimpleSyntax, INDArray> e: this.simpleSyntaxVectors.entrySet()) {
			LOG.info("%s  => %s", e.getKey(), Helper.printVector(e.getValue()));
		}
		
		LOG.info("Slash Vectors");
		for(Entry<Slash, INDArray> e: this.slashVectors.entrySet()) {
			LOG.info("%s  => %s", e.getKey(), Helper.printVector(e.getValue()));
		}
		
		LOG.info("Attribute Vectors");
		for(Entry<String, INDArray> e: this.attributeVectors.entrySet()) {
			LOG.info("%s  => %s", e.getKey(), Helper.printVector(e.getValue()));
		}
		
		LOG.info("Base Constant Vectors");
		for(Entry<String, INDArray> e: this.baseConstantVectors.entrySet()) {
			LOG.info("%s  => %s", e.getKey(), Helper.printVector(e.getValue()));
		}
		
		LOG.info("Type Vectors");
		for(Entry<Type, INDArray> e: this.typeVectors.entrySet()) {
			LOG.info("%s  => %s", e.getKey(), Helper.printVector(e.getValue()));
		}
		
	}
	
	/** Returns Json string containing the embedding of the categories and the recursive network 
	 * Fix problems in creating Json
	 * @throws UnsupportedEncodingException 
	 * @throws FileNotFoundException */
	public void logCategoryEmbeddingAndRecursiveNetworkParam(String folderName) 
											throws FileNotFoundException, UnsupportedEncodingException {
		
		PrintWriter writerSyntax = new PrintWriter(folderName+"/syntax.json", "UTF-8");
		
		writerSyntax.write("{\"Simple_Syntax_Vectors\": { \n");
		
		for(Entry<SimpleSyntax, INDArray> e: this.simpleSyntaxVectors.entrySet()) {
			writerSyntax.write("\""+e.getKey()+"\" : \"" + Helper.printFullVector(e.getValue())+"\",\n");
		}
		
		writerSyntax.write("}, \n \"Slash_Vectors\": {");
		
		for(Entry<Slash, INDArray> e: this.slashVectors.entrySet()) {
			writerSyntax.write("\""+e.getKey()+"\" : \"" + Helper.printFullVector(e.getValue())+"\",\n");
		}
		
		writerSyntax.write("}, \n \"Attribute_Vectors\": {");
		
		for(Entry<String, INDArray> e: this.attributeVectors.entrySet()) {
			writerSyntax.write("\""+e.getKey()+"\" : \"" + Helper.printFullVector(e.getValue())+"\",\n");
		}
		
		writerSyntax.write("}, \n \"Simple_Recursive_Network\": { ");
		
		writerSyntax.write("\"W\":[" 
									+ Helper.printFullMatrix(this.syntaxsRecursiveNetwork.getW()) + "],\n");
		writerSyntax.write("\"b\":\""
									+ Helper.printFullVector(this.syntaxsRecursiveNetwork.getb()) + "\"");
		
		writerSyntax.write("}\n}");
		
		writerSyntax.flush();
		writerSyntax.close();
		
		PrintWriter writerSemantic = new PrintWriter(folderName+"/semantic.json", "UTF-8");
		
		writerSemantic.write("{\"Base_Constant_Vectors\": { \n");
		
		for(Entry<String, INDArray> e: this.baseConstantVectors.entrySet()) {
			writerSemantic.write("\""+e.getKey()+"\" : \"" + Helper.printFullVector(e.getValue())+"\",\n");
		}
		
		writerSemantic.write("}, \n \"Type_Vectors\": {");
		
		for(Entry<Type, INDArray> e: this.typeVectors.entrySet()) {
			writerSemantic.write("\""+e.getKey()+"\" : \"" + Helper.printFullVector(e.getValue())+"\",\n");
		}
		
		writerSemantic.write("}, \n \"null_logic\":  \"" + Helper.printFullVector(this.nullLogic) + "\",\n");
		writerSemantic.write("\"Simple_Recursive_Network\": { ");
		
		writerSemantic.write("\"W\":[" + 
								Helper.printFullMatrix(this.semanticsRecursiveNetwork.getW()) + "],\n");
		writerSemantic.write("\"b\":\"" + 
								Helper.printFullVector(this.semanticsRecursiveNetwork.getb()) + "\"");
		
		writerSemantic.write("}\n}");
		
		writerSemantic.flush();
		writerSemantic.close();
	}
	
	/** Reads category and recursive network parameters from Json file 
	 * Todo: use standard library for making things safe */
	public void bootstrapCategoryEmbeddingAndRecursiveNetworkParam(String folderName) {
	
		this.invalidateCache();
		
		Path syntaxJsonPath = Paths.get(folderName+"/syntax.json");
		String syntaxJsonString;
		
		try {
			syntaxJsonString = Joiner.on("\r\n").join(Files.readAllLines(syntaxJsonPath));
		} catch (IOException e) {
			throw new RuntimeException("Could not read from syntax.json. Error: "+e);
		}
		
		JSONObject objSyntax = new JSONObject(syntaxJsonString);
		
		//Bootstrap the syntax embeddings
		for(Entry<Slash, INDArray> e: this.slashVectors.entrySet()) {
			String safeKey = e.getKey().toString().replace("\"", "\\\""); 
			String indarrayString = objSyntax.getJSONObject("Slash_Vectors").getString(safeKey);
			this.slashVectors.put(e.getKey(), Helper.toVector(indarrayString));
		}
		
		for(Entry<SimpleSyntax, INDArray> e: this.simpleSyntaxVectors.entrySet()) {
			String safeKey = e.getKey().toString().replace("\"", "\\\""); 
			String indarrayString = objSyntax.getJSONObject("Simple_Syntax_Vectors").getString(safeKey);
			this.simpleSyntaxVectors.put(e.getKey(), Helper.toVector(indarrayString));
		}
		
		for(Entry<String, INDArray> e: this.attributeVectors.entrySet()) {
			final String safeKey;
			if(e.getKey() == null) {
				safeKey = "null";
			} else {
				safeKey = e.getKey().toString().replace("\"", "\\\""); 
			}
			String indarrayString = objSyntax.getJSONObject("Attribute_Vectors").getString(safeKey);
			this.attributeVectors.put(e.getKey(), Helper.toVector(indarrayString));
		}
		
		//Bootstrap Recursive network for syntax
		String syntaxRecursiveBiasString = objSyntax.getJSONObject("Simple_Recursive_Network").getString("b");
		String syntaxRecursiveWeightString = objSyntax.getJSONObject("Simple_Recursive_Network").getJSONArray("W").join("#");
		
		INDArray syntaxb = Helper.toVector(syntaxRecursiveBiasString).transposei();
		INDArray syntaxW = Helper.toMatrix(syntaxRecursiveWeightString);
		this.syntaxsRecursiveNetwork.setParam(syntaxW, syntaxb);
		
		
		Path semanticJsonPath = Paths.get(folderName+"/semantic.json");
		String semanticsJsonString;
		
		try {
			semanticsJsonString = Joiner.on("\r\n").join(Files.readAllLines(semanticJsonPath));
		} catch (IOException e) {
			throw new RuntimeException("Could not read from semantic.json. Error: "+e);
		}
		
		JSONObject objSemantic = new JSONObject(semanticsJsonString);
		
		//Bootstrap the semantic embeddings
		String indarrayNullString = objSemantic.getString("null_logic");
		this.nullLogic = Helper.toVector(indarrayNullString);
		
		for(Entry<String, INDArray> e: this.baseConstantVectors.entrySet()) {
			String safeKey = e.getKey().replace("\"", "\\\""); 
			String indarrayString = objSemantic.getJSONObject("Base_Constant_Vectors").getString(safeKey);
			this.baseConstantVectors.put(e.getKey(), Helper.toVector(indarrayString));
		}
		
		for(Entry<Type, INDArray> e: this.typeVectors.entrySet()) {
			
			final String key;
			if(e.getKey() == null) {
				key = "null";
			} else {
				key = e.getKey().toString();
			}
			
			String safeKey = key.replace("\"", "\\\""); 
			String indarrayString = objSemantic.getJSONObject("Type_Vectors").getString(safeKey);
			this.typeVectors.put(e.getKey(), Helper.toVector(indarrayString));
		}
		
		//Bootstrap Recursive network for semantic
		String semanticRecursiveBiasString = objSemantic.getJSONObject("Simple_Recursive_Network").getString("b");
		String semanticRecursiveWeightString = objSemantic.getJSONObject("Simple_Recursive_Network").getJSONArray("W").join("#");
		
		INDArray semanticb  = Helper.toVector(semanticRecursiveBiasString).transposei();
		INDArray semanticW = Helper.toMatrix(semanticRecursiveWeightString);
		this.semanticsRecursiveNetwork.setParam(semanticW, semanticb);
	}
}
