package edu.cornell.cs.nlp.spf.parser.ccg.rnn.shiftreduce.categoryembeddings;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.ObjectInput;
import java.io.ObjectInputStream;
import java.io.ObjectOutput;
import java.io.ObjectOutputStream;
import java.io.OutputStream;
import java.io.Serializable;
import java.io.UnsupportedEncodingException;
import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Map.Entry;
import java.util.Set;
import java.util.Spliterator;
import java.util.Spliterators;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.StreamSupport;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.Sqrt;
import org.nd4j.linalg.factory.Nd4j;

import edu.cornell.cs.nlp.spf.base.token.TokenSeq;
import edu.cornell.cs.nlp.spf.ccg.categories.Category;
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
import edu.cornell.cs.nlp.spf.parser.ccg.rnn.shiftreduce.learning.LearningRate;
import edu.cornell.cs.nlp.spf.parser.ccg.rnn.shiftreduce.recursive.AveragingNetwork;
import edu.cornell.cs.nlp.spf.parser.ccg.rnn.shiftreduce.recursive.GradientWrapper;
import edu.cornell.cs.nlp.spf.parser.ccg.rnn.shiftreduce.recursive.RecursiveTreeNetwork;
import edu.cornell.cs.nlp.spf.parser.ccg.rnn.shiftreduce.recursive.Tree;
import edu.cornell.cs.nlp.utils.log.ILogger;
import edu.cornell.cs.nlp.utils.log.LogLevel;
import edu.cornell.cs.nlp.utils.log.LoggerFactory;
import edu.uw.cs.lil.amr.data.LabeledAmrSentence;
import edu.uw.cs.lil.amr.lambda.AMRServices;

/**
 * Creates embedding of a logical expression
 * TODO share functions between this class and Category Embedding
 * @author Dipendra Misra 
 * */
public class LogicalExpressionEmbedding implements AbstractEmbedding, Serializable {
	
	private static final long serialVersionUID = 8528966389515502272L;

	public static final ILogger LOG = LoggerFactory.create(LogicalExpressionEmbedding.class);

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
	
	/** l2 Regularization factor */
	private final Double regularizer;
	
	/** If true then use recursive else averaging */
	private boolean useRecursive;
	
	/** Cache category embeddings. Computing embeddings of category is costly due to feed forwarding
	 * through two deep recursive neural network. Therefore, we cache the category embeddings. These
	 * cached embeddings are invalidated when the model changes (parameters of recursive networks or 
	 * the leaf vectors of recursive networks). Cache syntax and semantic embedding separately 
	 * for better generalization */
	private final ConcurrentHashMap<LogicalExpression, Tree> cacheSemanticEmbedding;
	private final int maxCacheSize;
	private AtomicInteger cacheSemanticHit, cacheSemanticMiss;
	
	//For GradientCheck: Empirical Gradients
	public double empiricalSemanticGrad;
	public double empiricalSemanticRecursiveW, empiricalSemanticRecursiveb;
	private final boolean doGradientchecks;
	
	public LogicalExpressionEmbedding(int dim, double learningRate, double l2) {
		
		this.learningRate = new LearningRate(learningRate, 0.0);
		this.regularizer = l2;
		this.useRecursive = true;
				
		this.semanticsNOut = dim; //35; //30;//40;//30;//35;
		this.semanticsRecursiveNetwork = new RecursiveTreeNetwork(this.semanticsNOut, 
											this.learningRate, this.regularizer, null, 0);
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
		
		this.maxCacheSize = 0;//400000;
		this.cacheSemanticEmbedding = new ConcurrentHashMap<LogicalExpression, Tree>();
		
		this.cacheSemanticHit = new AtomicInteger();
		this.cacheSemanticMiss = new AtomicInteger();
		
		this.updatedTypes = Collections.synchronizedSet(new HashSet<Type>());
		this.updatedBaseConstant = Collections.synchronizedSet(new HashSet<String>());
		this.updatedNullLogic = new AtomicBoolean(false);
		
		LOG.info("Logical Expression dimensionality %s. Recursive %s", 
											this.semanticsNOut, this.useRecursive);
		
		this.doGradientchecks = false;
		LOG.setCustomLevel(LogLevel.INFO);
	}
	
	
	/** Returns a vector of size 1xdim, with each value uniformly
	 * randomly in [-epsilon, epsilon] */
	public INDArray initUniformlyRandom(int dim, double epsilon) {
		
		INDArray vec = Nd4j.rand(new int[]{1, dim});
		vec.subi(0.5).muli(epsilon);
		
		return vec;
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
		
	public INDArray getSemanticRecursiveW() {
		return this.semanticsRecursiveNetwork.getW();
	}
	
	public INDArray getSemanticRecursiveb() {
		return this.semanticsRecursiveNetwork.getb();
	}
	
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
					this.typeVectorsGrad, null, 0);		
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
					this.typeVectorsGrad, null, 0);		
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
	public void induceVectorsFromProcessedDataset(List<CompositeDataPoint<LogicalExpression>> dataPoint) {
		
		for(CompositeDataPoint<LogicalExpression> point: dataPoint) {
			List<CompositeDataPointDecision<LogicalExpression>> decisions = point.getDecisions();
			for(CompositeDataPointDecision<LogicalExpression> decision: decisions) {
				List<ParsingOp<LogicalExpression>> actions = decision.getPossibleActions();
				for(ParsingOp<LogicalExpression> action: actions) {
					
					final Category<LogicalExpression> category = action.getCategory();
					final LogicalExpression exp = category.getSemantics();
					
					if(exp == null) {
						continue;
					}
					
					//Convert Logical Expression to underspecified form 
					final LogicalExpression underspecified = AMRServices.underspecifyAndStrip(exp);
						
					AddSemanticConstantsVisitor.addSemanticConstant(underspecified, this.semanticsNOut, 
							this.baseConstantVectors, this.baseConstantVectorsGrad, this.typeVectors,
							this.typeVectorsGrad, null, 0);		
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
					this.getLogicalExpressionEmbedding((LogicalExpression)category.getSemantics()).getEmbedding();
				});
		
		this.logCachePerformance();
		LOG.info("Size of cache is %s", this.cacheSemanticEmbedding.size());
		
		this.cacheSemanticHit.set(0);
		this.cacheSemanticMiss.set(0);
	}
	
	/** returns embedding of a category under the given model. The embedding of a category is
	 * given by the concatenation of the embedding of syntax and semantics. This is a costly
	 * operation and therefore a cache is used to optimize. The cache is emptied after every
	 * update step or after bootstrapping a model. Similarly, the updated parameters must be cleared
	 * after every iteration or after bootstrapping a model. */
	public CategoryEmbeddingResult getLogicalExpressionEmbedding(LogicalExpression exp) {
		
		Tree semanticTree = null;//this.findSemanticInCache(exp);
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
		
		CategoryEmbeddingResult result = new CategoryEmbeddingResult(semanticEnc, null, semanticTree);
		
		return result;
	}
	
	@SuppressWarnings("unused")
	private void printTree(LogicalExpression exp, Tree t) {
		
		LOG.info("Exp %s", exp);
		this.printTree(t);
	}
	
	private void printTree(Tree t) {
		
		LOG.info("(%s ", t.getLabel());
		for(int i = t.numChildren() - 1; i >= 0; i--) {
			this.printTree(t.getChild(i));
		}
		LOG.info(")");
	}
	
	public void backprop(Tree semanticsTree, INDArray error) {
		
		if(LOG.getLogLevel() == LogLevel.DEBUG) {
			
			double errorNorm2 = error.normmaxNumber().doubleValue();
			
			LOG.debug("Category Embedding with error %s, ", error);
			
			if(Double.isNaN(errorNorm2)) {
				LOG.info("Problem with backprop in Category Embedding. Got NaN.");
			}
			
			if(Double.isInfinite(errorNorm2)) {
				LOG.info("Problem with backprop in Category Embedding. Got infinite.");
			}		
		}
		
//		INDArray averageSemanticError = error.get(NDArrayIndex.interval(2*this.syntaxNOut + this.semanticsNOut,   
//															2*this.syntaxNOut + 2*this.semanticsNOut));
//		
//		AveragingNetwork.backprop(syntacticTree, averageSyntaxError);
//		AveragingNetwork.backprop(semanticsTree, averageSemanticError);
		
		
		if(this.useRecursive) {
			this.semanticsRecursiveNetwork.backProp(semanticsTree, error);
		} else {
			AveragingNetwork.backprop(semanticsTree, error);
		}
	}

	@Override
	public int getDimension() {
		return this.semanticsNOut;
	}

	@Override
	public Object getEmbedding(Object obj) {
		throw new RuntimeException("Operation not supported");
	}
	
	private void updateSemanticCache(LogicalExpression exp, Tree semanticTree) {
		
		//use smarter scheme such as LRU here in future
		if(exp == null || this.cacheSemanticEmbedding.size() > this.maxCacheSize)
			return;
			
		this.cacheSemanticEmbedding.put(exp, semanticTree);
	}
	
	private Tree findSemanticInCache(LogicalExpression exp) {		
		if(exp == null) {
			return null;
		}
		
		return this.cacheSemanticEmbedding.get(exp);
	}
	
	public void invalidateCache() {
		//cache is used by multiple threads so synchronize access to it
		
		synchronized(this.cacheSemanticEmbedding) {
			this.cacheSemanticEmbedding.clear();
		}
	}
	
	public void logCachePerformance() {
		
		double semanticCacheHit = this.cacheSemanticHit.get()
				/(double)Math.max(this.cacheSemanticHit.get() + this.cacheSemanticMiss.get(), 1);
		LOG.info("Total Semantic Cache Miss %s", this.cacheSemanticMiss.get());
		LOG.info("Semantic Cache Hit %s", semanticCacheHit);
		
		LOG.info("Size of the semantic cache "+this.cacheSemanticEmbedding.size());
	}
	
	/** performs gradient descent on the category vectors using rawGradient representing
	 *  del loss /del vector. One can add more terms to loss such as L2 norm with respect to vector.
	 *  Sum of square of gradient is passed for AdaGrad  learning rate. 
	 *  Warning: rawGradient gets affected by this function.
	 *  */
	private INDArray updateVectors(INDArray vector, GradientWrapper rawGradient, 
								  INDArray sumSquareGradient, double learningRate, String label) {
		
		int numTerms = rawGradient.numTerms();
		if(numTerms == 0) {	//no terms to add
			return vector;
		}
		
		INDArray sumGradient = rawGradient.getGradient();
		
		if(sumGradient.normmaxNumber().doubleValue() == 0) {
			LOG.warn("Category: 0 gradient found");
		}
		
		//Add regularization
		sumGradient.addi(vector.mul(this.regularizer));
	
		//Clip the gradient
		double norm = sumGradient.normmaxNumber().doubleValue();
		double threshold = 5.0;
		
		if(norm > threshold) {
			sumGradient.muli(threshold/norm);
		}
		
		//Perform AdaGrad based SGD step
		sumSquareGradient.addi(sumGradient.mul(sumGradient));
		
		INDArray invertedLearningRate = Nd4j.getExecutioner()
											.execAndReturn(new Sqrt(sumSquareGradient.dup()))
											.divi(learningRate);
		
		sumGradient.divi(invertedLearningRate);
		vector.subi(sumGradient);
		
		return vector;
	}
	
	public void updateParameters() {
		//the embeddings in cache are no longer true therefore invalidate them
		this.invalidateCache();
		
		if(this.doGradientchecks) {
			LOG.info("Gradient Check. Semantic. Empirical %s. Estimate %s", this.empiricalSemanticGrad, 
							this.getSemanticGradVector().getDouble(new int[]{0, 0}));
			LOG.info("Gradient Check. Semantic Recursive W. Empirical %s. Estimate %s", this.empiricalSemanticRecursiveW, 
					this.semanticsRecursiveNetwork.getGradW().getDouble(new int[]{0, 0}));
			LOG.info("Gradient Check. Semantic Recursive b. Empirical %s. Estimate %s", this.empiricalSemanticRecursiveb, 
					this.semanticsRecursiveNetwork.getGradb().getDouble(new int[]{0, 0}));
		}
		
		final double learningRate = this.learningRate.getLearningRate();
		
		if(this.updatedNullLogic.get()) { 
			this.updateVectors(this.nullLogic, this.nullLogicGrad, 
								this.adaGradSumSquareNullLogic, learningRate, "Null-Logic");
		}
				
		StreamSupport.stream(Spliterators
				.spliterator(this.updatedBaseConstant, Spliterator.IMMUTABLE), true)
				.unordered()
				.forEach(p-> this.updateVectors(this.baseConstantVectors.get(p), this.baseConstantVectorsGrad.get(p), 
						 this.adaGradSumSquareBaseConstant.get(p), learningRate, "Base-Constant"));
		
		StreamSupport.stream(Spliterators
				.spliterator(this.updatedTypes, Spliterator.IMMUTABLE), true)
				.unordered()
				.forEach(p-> this.updateVectors(this.typeVectors.get(p), this.typeVectorsGrad.get(p), 
						 this.adaGradSumSquareType.get(p), learningRate, "Type"));
		
		this.semanticsRecursiveNetwork.updateParameters();
	}
	
	public void flushGradients() {
					
		if(this.updatedNullLogic.get()) {
			this.nullLogicGrad.flush();
			this.updatedNullLogic.set(false);
		}
		
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
		
		this.adaGradSumSquareNullLogic.muli(0);
		
		for(Entry<String, INDArray> e: this.adaGradSumSquareBaseConstant.entrySet()) {
			e.getValue().muli(0);
		}
		
		for(Entry<Type, INDArray> e: this.adaGradSumSquareType.entrySet()) {
			e.getValue().muli(0);
		}
	}
	
	/** Returns Json string containing the embedding of the categories and the recursive network 
	 * Fix problems in creating Json
	 * @throws UnsupportedEncodingException 
	 * @throws FileNotFoundException */
	public void logEmbeddingAndRecursiveNetworkParam(String folderName) 
											throws FileNotFoundException, UnsupportedEncodingException {
		
		try (
				OutputStream file = new FileOutputStream(folderName + "/type_vectors.ser");
				OutputStream buffer = new BufferedOutputStream(file);
				ObjectOutput output = new ObjectOutputStream(buffer);
			) {
				output.writeObject(this.typeVectors);
			} catch(IOException ex) {
				throw new RuntimeException("Cannot store serializable data");
			}
		
		try (
				OutputStream file = new FileOutputStream(folderName + "/base_constant_vectors.ser");
				OutputStream buffer = new BufferedOutputStream(file);
				ObjectOutput output = new ObjectOutputStream(buffer);
			) {
				output.writeObject(this.baseConstantVectors);
			} catch(IOException ex) {
				throw new RuntimeException("Cannot store serializable data");
			}
		
		try (
				OutputStream file = new FileOutputStream(folderName + "/null_logic.ser");
				OutputStream buffer = new BufferedOutputStream(file);
				ObjectOutput output = new ObjectOutputStream(buffer);
			) {
				output.writeObject(this.nullLogic);
			} catch(IOException ex) {
				throw new RuntimeException("Cannot store serializable data");
			}
		
		try (
				OutputStream file = new FileOutputStream(folderName + "/semantic_recursive_W.ser");
				OutputStream buffer = new BufferedOutputStream(file);
				ObjectOutput output = new ObjectOutputStream(buffer);
			) {
				output.writeObject(this.semanticsRecursiveNetwork.getW());
			} catch(IOException ex) {
				throw new RuntimeException("Cannot store serializable data");
			}
		
		try (
				OutputStream file = new FileOutputStream(folderName + "/semantic_recursive_b.ser");
				OutputStream buffer = new BufferedOutputStream(file);
				ObjectOutput output = new ObjectOutputStream(buffer);
			) {
				output.writeObject(this.semanticsRecursiveNetwork.getb());
			} catch(IOException ex) {
				throw new RuntimeException("Cannot store serializable data");
			}
	}
	
	/** Reads category and recursive network parameters from Json file 
	 * Todo: use standard library for making things safe */
	public void bootstrapCategoryEmbeddingAndRecursiveNetworkParam(String folderName) {
	
		this.invalidateCache();
		
		try (
			     InputStream file = new FileInputStream(folderName + "/type_vectors.ser");
			     InputStream buffer = new BufferedInputStream(file);
			     ObjectInput input = new ObjectInputStream (buffer);
			) {
				 @SuppressWarnings("unchecked")
				 HashMap<Type, INDArray> typeVectors = (HashMap<Type, INDArray>) input.readObject();
				 this.typeVectors.clear();
				 this.typeVectors.putAll(typeVectors);
				 
				 LOG.info("Bootstrapped type vectors embeddings for %s features", this.typeVectors.size());
			} catch(Exception e) {
				throw new RuntimeException("Could not deserialize type_vectors.ser. Error: " + e);
		    }
		
		try (
			     InputStream file = new FileInputStream(folderName + "/base_constant_vectors.ser");
			     InputStream buffer = new BufferedInputStream(file);
			     ObjectInput input = new ObjectInputStream (buffer);
			) {
				 @SuppressWarnings("unchecked")
				 HashMap<String, INDArray> baseConstantVectors = (HashMap<String, INDArray>) input.readObject();
				 this.baseConstantVectors.clear();
				 this.baseConstantVectors.putAll(baseConstantVectors);
				 
				 LOG.info("Bootstrapped base constant vectors embeddings for %s features",
						 		this.baseConstantVectors.size());
			} catch(Exception e) {
				throw new RuntimeException("Could not deserialize base_constant_vectors.ser. Error: " + e);
		    }
		
		try (
			     InputStream file = new FileInputStream(folderName + "/null_logic.ser");
			     InputStream buffer = new BufferedInputStream(file);
			     ObjectInput input = new ObjectInputStream (buffer);
			) {
				 this.nullLogic = (INDArray) input.readObject();
				 LOG.info("Bootstrapped null logic");
			} catch(Exception e) {
				throw new RuntimeException("Could not deserialize null_logic.ser. Error: " + e);
		    }
		
		final INDArray semanticW, semanticb;
		
		try (
			     InputStream file = new FileInputStream(folderName + "/semantic_recursive_W.ser");
			     InputStream buffer = new BufferedInputStream(file);
			     ObjectInput input = new ObjectInputStream (buffer);
			) {
				 semanticW = (INDArray) input.readObject();
				 LOG.info("Bootstrapped W for recursive semantic embeddings");
			} catch(Exception e) {
				throw new RuntimeException("Could not deserialize semantic_recursive_W.ser. Error: " + e);
		    }
		
		try (
			     InputStream file = new FileInputStream(folderName + "/semantic_recursive_b.ser");
			     InputStream buffer = new BufferedInputStream(file);
			     ObjectInput input = new ObjectInputStream (buffer);
			) {
			    semanticb = (INDArray) input.readObject();
			    LOG.info("Bootstrapped b for recursive semantic embeddings"); 
			} catch(Exception e) {
				throw new RuntimeException("Could not deserialize semantic_recursive_b.ser. Error: " + e);
		    }
		
		
		this.semanticsRecursiveNetwork.setParam(semanticW, semanticb);
	}
}
