package edu.cornell.cs.nlp.spf.parser.ff.shiftreduce.neuralparser;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.ObjectInput;
import java.io.ObjectInputStream;
import java.io.ObjectOutput;
import java.io.ObjectOutputStream;
import java.io.OutputStream;
import java.io.PrintWriter;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Random;
import java.util.Set;
import java.util.Spliterator;
import java.util.Spliterators;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;
import java.util.stream.Collectors;
import java.util.stream.StreamSupport;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.Sqrt;
import org.nd4j.linalg.factory.Nd4j;

import com.google.common.base.Charsets;
import com.google.common.base.Joiner;
import com.google.common.io.Files;
import edu.cornell.cs.nlp.spf.base.hashvector.HashVectorFactory;
import edu.cornell.cs.nlp.spf.base.hashvector.IHashVector;
import edu.cornell.cs.nlp.spf.base.hashvector.KeyArgs;
import edu.cornell.cs.nlp.spf.parser.ccg.rnn.shiftreduce.learning.Helper;
import edu.cornell.cs.nlp.spf.parser.ff.shiftreduce.dataset.SparseFeatureAndStateDataset;
import edu.cornell.cs.nlp.spf.parser.ff.shiftreduce.dataset.SparseFeatureDataset;
import edu.cornell.cs.nlp.utils.composites.Pair;
import edu.cornell.cs.nlp.utils.composites.Triplet;
import edu.cornell.cs.nlp.utils.log.ILogger;
import edu.cornell.cs.nlp.utils.log.LogLevel;
import edu.cornell.cs.nlp.utils.log.LoggerFactory;

/** Given a list of features, produces embedding for a MLP. Embeddings are produced
 * by converting each feature into its dense representation and then averaging the
 * values of features by tag. The embedding for every tag is then concatenated into one. */
public class FeatureEmbedding<MR> implements Serializable {
	
	private static final long serialVersionUID = 4979186256288550636L;

	public static final ILogger	LOG = LoggerFactory.create(FeatureEmbedding.class);

	private final int nIn;
	private final double learningRate;
	private final double l2;
	
	/** Learnable embeddings of single feature and their gradients
	 * and sum of square of gradients for adaptive learning rates. */
	private Map<KeyArgs, INDArray> featureEmbedding;
	private final Map<KeyArgs, INDArray> gradFeatureEmbedding;
	private final Map<KeyArgs, INDArray> sumSquareGradfeatureEmbedding;
	
	private final Map<KeyArgs, INDArray> originalFeatureEmbedding;
	
	/** Embeddings used for inactive tag */
	private final List<INDArray> inactiveTagEmbedding;
	private final List<INDArray> gradInactiveTagEmbedding;
	private final List<INDArray> sumSquareGradInactiveTagEmbedding;
	private final Set<Integer> updatedInactiveTag;
	
	private final Set<KeyArgs> updatedFeatures;
	
	/** All possible tags and their mapping to indices */
	private final Map<String, Integer> tags; 
	
	/** Dimensions of all tags by their indices */
	private final int[] tagDimensions;
	
	/** If paddedIndex[i] = j then i'th features start from index j (including). 
	 * As requirement, paddedIndex[0] = 0 always.*/
	private final int[] paddedIndex; 
		
	/** Stop adding */
	private boolean addFeautres;
	
	/** Number of features that are unseen */
	private final Set<KeyArgs> unseenFeatures;
	
	/** Number of features that are seen */
	private final Set<KeyArgs> seenFeatures;
	
	/** Number of examples where no feature is active */
	private final AtomicInteger exampleWithNoActiveFeatures;
	
	private final double[] epsilon;
	
	/** Unseen feature embedding by tag*/
	///////////////////////////////
	private final boolean useUnseenFeatureEmbedding;
	private final List<INDArray> unseenFeatureEmbeddingByTag;
	private final List<INDArray> gradUnseenFeatureEmbeddingByTag;
	private final List<INDArray> sumSquareGradUnseenFeatureEmbeddingByTag;
	private final Set<Integer> updatedUnseenFeatureTag;
	private final Set<Integer> toUpdate;
	//////////////////////////////
	
	/** Features by tag*/
	private final Map<String, Set<KeyArgs>> seenFeaturesByTag;
	private final Map<String, Set<KeyArgs>> unseenFeaturesByTag;
	
	/** Features in map always obey dimensionality constrained except for word embeddings.
	 * These are generally large, therefore we learn a projection to a smaller space.
	 * These word embeddings are not updated since they are trained from a 
	 * larger dataset. We however do learn a projection. */
	private int wordDim;
	private Set<String> wordTags;
	private INDArray WProjection;
	private INDArray gradWProjection;
	private INDArray sumSquareGradWProjection;
	private AtomicBoolean updatedWProjection;
	private boolean optimized;
	
	/** Output directory where logs are saved*/
	private final File outputDir;
	
	private KeyArgs gradientCheckFeature;
	private Double empiricalGrad;
	
	public final AtomicInteger nonWord2VecInitVectors;
	
	public boolean collectStats;
	
	public FeatureEmbedding(double learningRate, double l2,  
							Map<String, Integer> tagsAndDimension, File outputDir) {
		
		this.learningRate = learningRate;
		this.l2 = l2;
		
		//embeddings of different features
		this.featureEmbedding = new HashMap<KeyArgs, INDArray>();
		this.gradFeatureEmbedding = new HashMap<KeyArgs, INDArray>();
		this.sumSquareGradfeatureEmbedding = new HashMap<KeyArgs, INDArray>();
		this.updatedFeatures = Collections.synchronizedSet(new HashSet<KeyArgs>());
		
		this.originalFeatureEmbedding = new HashMap<KeyArgs, INDArray>();
		
		//when a given feature tag is inactive
		this.inactiveTagEmbedding = new ArrayList<INDArray>();
		this.gradInactiveTagEmbedding = new ArrayList<INDArray>();
		this.sumSquareGradInactiveTagEmbedding = new ArrayList<INDArray>();
		this.updatedInactiveTag = Collections.synchronizedSet(new HashSet<Integer>());
		
		//for unseen features
		this.useUnseenFeatureEmbedding = false;
		this.unseenFeatureEmbeddingByTag = new ArrayList<INDArray>();
		this.gradUnseenFeatureEmbeddingByTag = new ArrayList<INDArray>();
		this.sumSquareGradUnseenFeatureEmbeddingByTag = new ArrayList<INDArray>();
		this.updatedUnseenFeatureTag = Collections.synchronizedSet(new HashSet<Integer>());
		Set<Integer> toUpdate = new HashSet<Integer>();
		
		this.tags = new HashMap<String, Integer>();
		
		this.addFeautres = true;
		this.collectStats = false;
				
		this.gradientCheckFeature = null;
		this.empiricalGrad = null;

		if(this.collectStats) {
			this.seenFeaturesByTag = Collections.synchronizedMap(new HashMap<String, Set<KeyArgs>>());
			this.unseenFeaturesByTag = Collections.synchronizedMap(new HashMap<String, Set<KeyArgs>>());
			this.exampleWithNoActiveFeatures = new AtomicInteger();
			this.seenFeatures = Collections.synchronizedSet(new HashSet<KeyArgs>());
			this.unseenFeatures = Collections.synchronizedSet(new HashSet<KeyArgs>());
		} else {
			this.seenFeaturesByTag = null;
			this.unseenFeaturesByTag = null;
			this.exampleWithNoActiveFeatures = null;
			this.seenFeatures = null;
			this.unseenFeatures = null;
		}
		
		this.tagDimensions = new int[tagsAndDimension.size()];
		this.paddedIndex = new int[tagsAndDimension.size()];
		this.epsilon = new double[tagsAndDimension.size()];
		
		int tagIx = 0;
		int padding = 0;
		StringBuilder s = new StringBuilder();
		
		for(Entry<String, Integer> tagAndDimension: tagsAndDimension.entrySet()) {
			final String tag = tagAndDimension.getKey();
			final int dim = tagAndDimension.getValue();
			s.append(tag + ": " + dim + ", ");
			
			this.tags.put(tag, tagIx);
			this.tagDimensions[tagIx] = dim;
			this.paddedIndex[tagIx] = padding;
			padding = padding + dim;
			
			final double epsilon = 2 * Math.sqrt(6.0/(double)(dim + 1.0));
			this.epsilon[tagIx] = epsilon;
			
			INDArray vec = this.getXavierInitializedVector(dim, epsilon);
			this.inactiveTagEmbedding.add(vec);
			this.gradInactiveTagEmbedding.add(Nd4j.zeros(dim));
			this.sumSquareGradInactiveTagEmbedding.add(Nd4j.zeros(dim));
			
			if(this.useUnseenFeatureEmbedding && (tag.compareTo("ATTACH") == 0 || tag.compareTo("AMRLEX") == 0)) {
				this.unseenFeatureEmbeddingByTag.add(this.getXavierInitializedVector(dim, epsilon));
				toUpdate.add(tagIx);
			} else {
				this.unseenFeatureEmbeddingByTag.add(Nd4j.zeros(dim));
			}
			this.gradUnseenFeatureEmbeddingByTag.add(Nd4j.zeros(dim));
			this.sumSquareGradUnseenFeatureEmbeddingByTag.add(Nd4j.zeros(dim));
			
			if(this.collectStats) {
				this.seenFeaturesByTag.put(tag, Collections.synchronizedSet(new HashSet<KeyArgs>()));
				this.unseenFeaturesByTag.put(tag, Collections.synchronizedSet(new HashSet<KeyArgs>()));
			}
			
			tagIx++;
		}
		this.toUpdate = Collections.unmodifiableSet(toUpdate);
		LOG.info("To Update %s", Joiner.on(",").join(this.toUpdate));
		
		this.nIn = padding;
		this.outputDir = outputDir;
		
		this.WProjection = null;
		this.gradWProjection = null;
		this.sumSquareGradWProjection = null;
		this.optimized = false;
		this.updatedWProjection = new AtomicBoolean(false);
		
		this.nonWord2VecInitVectors = new AtomicInteger(0);
		LOG.info("Feature Embedding. Learning rate %s, l2 %s, Stats %s, Number of tags %s. nIn %s,", 
					this.learningRate, this.l2, this.collectStats, this.tags.size(), this.nIn);
		LOG.info("...  tagDim { %s }, use unseen feature embedding %s.", s.toString(), this.useUnseenFeatureEmbedding);
	}
	
	public int numFeatures() {
		return this.featureEmbedding.size();
	}
 	
	public void registerWord2Vec(String fileName, String[] wordTags) {
		
		if(wordTags.length == 0) { 
			return;
		}
		
		this.wordTags = new HashSet<String>();
		final int tagDim = this.tagDimensions[this.tags.get(wordTags[0])];
		
		for(int i = 0; i < wordTags.length; i++) {
			
			this.wordTags.add(wordTags[i]);
			
			if(this.tagDimensions[this.tags.get(wordTags[i])] != tagDim) {
				throw new RuntimeException("Tag dimension must be same for all word features so that "
											+ "same projection matrix can be used");
			}
		}
		
		Integer wordDim = null;
		
		try {
			File file = new File(fileName);
			List<String> lines = Files.readLines(file, Charsets.UTF_8);
			
			for(String line: lines) {
				
				final String bytes[] = line.split(":");
				
				if(bytes.length != 2) {
					throw new RuntimeException("Expecting word2vec file in format word:val1,val2...valk\n");
				}
				
				final String word = bytes[0];
				final String rem = bytes[1];
				
				for(int i = 0; i < wordTags.length; i++) {
					
					INDArray vec = Helper.toVector(rem);
					
					if(wordDim == null) {
						wordDim = vec.size(1);
					} else {
						if(wordDim != vec.size(1)) {
							throw new RuntimeException("Word dimension must remain constant");
						}
					}
					
					KeyArgs feature = new KeyArgs(wordTags[i], word);
					
					this.featureEmbedding.put(feature, vec);
					this.gradFeatureEmbedding.put(feature, Nd4j.zeros(wordDim));
					this.sumSquareGradfeatureEmbedding.put(feature, Nd4j.zeros(wordDim));
				}
			}
			
			this.wordDim = wordDim;
			this.WProjection = Helper.getXavierInitiliazation(tagDim, wordDim);
			this.gradWProjection = Nd4j.zeros(tagDim, wordDim);
			this.sumSquareGradWProjection = Nd4j.zeros(tagDim, wordDim);
			
			LOG.info("Loaded word2vec features %s for %s many tags. Word dimension %s", lines.size(), wordTags.length, wordDim);
			LOG.info("... word projection matrix %s x %s", this.WProjection.size(0), this.WProjection.size(1));
			
		} catch(Exception e) {
			throw new RuntimeException("Cannot register word2vec features. Error " + e);
		}
	}
	
	public void registerFeatures(List<SparseFeatureDataset<MR>> dataset) {
		
		final long start = System.currentTimeMillis();
		
		for(SparseFeatureDataset<MR> p: dataset) {
			List<IHashVector> features = p.getPossibleActionFeatures();
			for(IHashVector feature: features) {
				Iterator<Pair<KeyArgs, Double>> it = feature.iterator();
				while(it.hasNext()) {
					KeyArgs fName = it.next().first();
					this.registerNewFeature(fName);
				}
			}
		}
		
		LOG.info("Total features %s. Time taken %s", this.featureEmbedding.size(), 
												System.currentTimeMillis() - start);
	}
	
	public void registerStateFeatures(List<SparseFeatureAndStateDataset<MR>> dataset) {
		
		final long start = System.currentTimeMillis();
		
		for(SparseFeatureAndStateDataset<MR> p: dataset) {
			IHashVector feature = p.getStateFeature();
			Iterator<Pair<KeyArgs, Double>> it = feature.iterator();
			while(it.hasNext()) {
				KeyArgs fName = it.next().first();
				this.registerNewFeature(fName);
			}
		}
		
		LOG.info("Total state features %s. Non-Word2Vec Vectors %s, Time taken %s", this.featureEmbedding.size(), 
										this.nonWord2VecInitVectors.get(), System.currentTimeMillis() - start);
	}

	public void registerActionFeatures(List<SparseFeatureAndStateDataset<MR>> dataset) {
		
		final long start = System.currentTimeMillis();
		
		for(SparseFeatureAndStateDataset<MR> p: dataset) {
			List<IHashVector> features = p.getPossibleActionFeatures();
			for(IHashVector feature: features) {
				Iterator<Pair<KeyArgs, Double>> it = feature.iterator();
				while(it.hasNext()) {
					KeyArgs fName = it.next().first();
					this.registerNewFeature(fName);
				}
			}
		}
		
		LOG.info("Total action features %s. Time taken %s", this.featureEmbedding.size(), 
												System.currentTimeMillis() - start);
	}
	
	public INDArray getGradientCheckFeature(KeyArgs ftr) {
		
		INDArray vec = this.featureEmbedding.get(ftr);
		if(vec == null) {
			throw new RuntimeException("Feature not present. Use another feature");
		}
		
		this.gradientCheckFeature = ftr;
		return vec;
	}
	
	public void setEmpiricalGrad(Double empiricalGrad) {
		this.empiricalGrad = empiricalGrad;
	}
	
	public void stopAddingFeatures() {
		this.addFeautres = false;
	}
	
	public boolean isAddingFeatures() {
		return this.addFeautres;
	}
	
	private INDArray getXavierInitializedVector(int dim, double epsilon) {
		
		INDArray vec = Nd4j.rand(new int[]{1, dim});
		vec.subi(0.5).muli(epsilon);
		
		return vec;
	}
	
	public void downsampleFeature() {
		
		int i = 0;
		final int totalSize = this.featureEmbedding.size();
		this.originalFeatureEmbedding.putAll(this.featureEmbedding);
		
		Set<KeyArgs> allFeatures = new HashSet<KeyArgs>();
		allFeatures.addAll(this.featureEmbedding.keySet());
		
		Set<String> cluster = new HashSet<String>();
		
		for(KeyArgs feature: allFeatures) {
			
			double p = Math.random();
			if(p < 0.05) {
				//remove it
				i++;
				cluster.add(this.getFeatureCluster(feature));
				this.featureEmbedding.remove(feature);
				this.gradFeatureEmbedding.remove(feature);
				this.sumSquareGradfeatureEmbedding.remove(feature);
			}
		}	
		LOG.info("Dropped %s, namely %s type, out of %s embeddings", i, Joiner.on(", ").join(cluster), totalSize);
	}
	
	/** Features are averaged by cluster. This function returns the cluster name 
	 * for a given feature. */
	private String getFeatureCluster(KeyArgs feature) {
		
		final String arg1 = feature.getArg1();
//		if(arg1.compareTo("AMRLEX") == 0) { ////--- this is ugly HACK :/
//			return arg1 + "-" + feature.getArg2();
//		} else {
//			return arg1;
//		}
		return arg1;
	}
	
	/** Returns embedding of feature that is known to exist */
	private INDArray getFeatureEmbedding(KeyArgs feature) {
		
		if(this.optimized) {
			return this.featureEmbedding.get(feature);
		}
		
		final String cluster = this.getFeatureCluster(feature);
		
		if(this.wordTags != null && this.wordTags.contains(cluster)) {
			return this.featureEmbedding.get(feature).mmul(this.WProjection.transpose());
		} else {
			return this.featureEmbedding.get(feature);
		}
	}
	
	public void projectWordEmbeddings() {
		
		if(this.wordTags == null) {
			return;
		}
		
		for(String wordTag: this.wordTags) {
			for(Entry<KeyArgs, INDArray> e: this.featureEmbedding.entrySet()) {
				if(e.getKey().getArg1().equals(wordTag)) {
					INDArray val = e.getValue();
					val = val.mmul(this.WProjection.transpose());
				}
			}
		}
		
		this.optimized = true;
	}
	
	private void registerNewFeature(KeyArgs feature) {
		
		if(this.featureEmbedding.containsKey(feature)) {
			return;//return this.featureEmbedding.get(feature);
		}
		
		//do not register word embeddings here. They must be registered separately
		final String cluster = this.getFeatureCluster(feature);
		if(this.wordTags != null && this.wordTags.contains(cluster)) {
			
			INDArray vec = Helper.getXavierInitiliazation(1, this.wordDim);
			this.featureEmbedding.put(feature, vec);
			this.gradFeatureEmbedding.put(feature, Nd4j.zeros(this.wordDim));
			this.sumSquareGradfeatureEmbedding.put(feature, Nd4j.zeros(this.wordDim));
			this.nonWord2VecInitVectors.incrementAndGet();
			
			return;
		}
		
		final int tagIx = this.tags.get(this.getFeatureCluster(feature)/*feature.getArg1()*/);
		final int tagDim = this.tagDimensions[tagIx];
		
		INDArray vec = this.getXavierInitializedVector(tagDim, this.epsilon[tagIx]);
		
		this.featureEmbedding.put(feature, vec);
		this.gradFeatureEmbedding.put(feature, Nd4j.zeros(tagDim));
		this.sumSquareGradfeatureEmbedding.put(feature, Nd4j.zeros(tagDim));
	
//		synchronized(this) {
//			
//			if(this.featureEmbedding.containsKey(feature)) {
//				return this.featureEmbedding.get(feature);
//			}
//			
//			this.featureEmbedding.put(feature, vec);
//			this.gradFeatureEmbedding.put(feature, Nd4j.zeros(tagDim));
//			this.sumSquareGradfeatureEmbedding.put(feature, Nd4j.zeros(tagDim));
//		}
		
		//return vec;
	}
	
	private int[] lambdaEmbedFeature(Pair<Integer, IHashVector> p, INDArray batch) {
	
		final IHashVector vector = p.second();
		final int ix = p.first();

		int[] tagsFreq = new int[this.tags.size()];
		Arrays.fill(tagsFreq, 0);
		
		Iterator<Pair<KeyArgs, Double>> it = vector.iterator();
	
		final long start1 = System.currentTimeMillis();
		while(it.hasNext()) {
			
			Pair<KeyArgs, Double> feature = it.next();
			if(feature.second() == 0.0) {
				//inactive feature, though inactive features should not be here in the first place
				continue; 
			}
			
			final String tag = this.getFeatureCluster(feature.first());//feature.first().getArg1();
			final int tagIx;
			if(this.tags.containsKey(tag)) {
				tagIx = this.tags.get(tag);
			} else {
				throw new RuntimeException("Unknown tag " + tag);
			}
			tagsFreq[tagIx]++;
			
			final int tagDim = this.tagDimensions[tagIx];
			final INDArray featureEmbedding;
			if(this.featureEmbedding.containsKey(feature.first())) {
				featureEmbedding = this.getFeatureEmbedding(feature.first());//this.featureEmbedding.get(feature.first());
				if(this.collectStats) {
					this.seenFeatures.add(feature.first());
					this.seenFeaturesByTag.get(tag/*feature.first().getArg1()*/).add(feature.first());
				}
			} else if(this.addFeautres) {
				//return but don't register. Reason we are returning a random vector
				//is since we don't want to have symmetry issues.
				featureEmbedding = this.getXavierInitializedVector(tagDim, this.epsilon[tagIx]);
			} else {
				
				if(this.useUnseenFeatureEmbedding) {
					featureEmbedding = this.unseenFeatureEmbeddingByTag.get(tagIx);
				} else {
					featureEmbedding = Nd4j.zeros(tagDim);
				}
				if(this.collectStats) {
					this.unseenFeatures.add(feature.first());
					this.unseenFeaturesByTag.get(tag/*feature.first().getArg1()*/).add(feature.first());
				}
			}
			
			final int padding = this.paddedIndex[tagIx];
			for(int i = 0; i < tagDim; i++) {
				double oldVal = batch.getDouble(ix, padding + i);
				double featureVal = featureEmbedding.getDouble(0, i);
				batch.putScalar(new int[]{ix, padding + i}, oldVal + featureVal);
			}
			
//			INDArray v = batch.get(NDArrayIndex.point(ix), NDArrayIndex.interval(padding, padding + tagDim));
//			if(v.size(0) == featureEmbedding.size(0) && v.size(1) == featureEmbedding.size(1)) {
//				v.addi(featureEmbedding);
//			} else {
//				v.transposei().addi(featureEmbedding);
//			}
		}
		final long start2 = System.currentTimeMillis();
		
		//divide by tagsFreq
		boolean emptyFeatures = true;
		for(int i = 0; i < tagsFreq.length; i++) {
			final int numTag = tagsFreq[i];
			final int tagDim = this.tagDimensions[i];
			final int padding = this.paddedIndex[i];
			
			if(numTag == 0) {
				for(int j = 0; j < tagDim; j++) {	
					double val = this.inactiveTagEmbedding.get(i).getDouble(0, j);
					batch.putScalar(new int[]{ix, padding + j}, val);
				}
				
//				INDArray w = this.inactiveTagEmbedding.get(i);
//				INDArray v = batch.get(NDArrayIndex.point(ix), NDArrayIndex.interval(padding, padding + tagDim));
//				if(v.size(0) == w.size(0) && v.size(1) == w.size(1)) {
//					v.addi(w);
//				} else {
//					v.transposei().addi(w);
//				}
				
				continue;
			}
			
			emptyFeatures = false;
			
			for(int j = 0; j < tagDim; j++) {
				double oldVal = batch.getDouble(ix, padding + j);
				batch.putScalar(new int[]{ix, padding + j}, oldVal/(double)numTag);
			}
			
			//batch.get(NDArrayIndex.point(ix), NDArrayIndex.interval(padding, padding + tagDim)).divi((double)numTag);
		}
		
		if(this.collectStats && emptyFeatures) {
			this.exampleWithNoActiveFeatures.incrementAndGet();
		}
		final long start3 = System.currentTimeMillis();
		
		return tagsFreq;
	}
	
	/** Compute a batch of dense embeddings given  list of sparse features.
	 * The ordering in batch follows the ordering of features i.e. ith embedding
	 * in the batch corresponds to the ith feature */
	public Pair<INDArray, List<int[]>> embedFeatures(IHashVector feature) {
		List<IHashVector> features = new ArrayList<IHashVector>();
		features.add(feature);
		
		return this.embedFeatures(features, false);
	}
	
	/** Compute a batch of dense embeddings given  list of sparse features.
	 * The ordering in batch follows the ordering of features i.e. ith embedding
	 * in the batch corresponds to the ith feature. Additional takes flag to do this
	 * computation in parallel. */
	public Pair<INDArray, List<int[]>> embedFeatures(IHashVector feature, boolean parallel) {
		List<IHashVector> features = new ArrayList<IHashVector>();
		features.add(feature);
		
		return this.embedFeatures(features, parallel);
	}
	
	/** Compute a batch of dense embeddings given  list of sparse features.
	 * The ordering in batch follows the ordering of features i.e. ith embedding
	 * in the batch corresponds to the ith feature */
	public Pair<INDArray, List<int[]>> embedFeatures(List<IHashVector> features) {
		return this.embedFeatures(features, false);
	}
	
	/** Version of embedFeatures that allows embedding features in parallel. This can be done
	 * at learning time to fasten it, however do note that overdoing it may create too many threads
	 * and cause overhead. */
	public Pair<INDArray, List<int[]>> embedFeatures(List<IHashVector> features, boolean parallel) {
		
		INDArray batch = Nd4j.zeros(features.size(), this.nIn);
		
		int e = 0;
		List<Pair<Integer, IHashVector>> enumeratedFeatures = new ArrayList<Pair<Integer, IHashVector>>();
		for(IHashVector vector: features) {
			enumeratedFeatures.add(Pair.of(e++, vector));
		}
		
		List<int[]> tagFrequencies = StreamSupport.stream(Spliterators
				.spliterator(enumeratedFeatures, Spliterator.IMMUTABLE), parallel)
				.map(p -> this.lambdaEmbedFeature(p, batch))
				.collect(Collectors.toList());
		
		if(LOG.getLogLevel() == LogLevel.DEBUG) {
			
			LOG.debug("Batch %s", batch);
			String s = "";
			for(int[] tagFreq: tagFrequencies) {
				for(int i = 0; i < tagFreq.length; i++) {
					s = s + ", " + tagFreq[i];
				}
				s = s + "\n";
			}
			LOG.debug("Tag Frequencies %s", s);
		}
		
		return Pair.of(batch, tagFrequencies);
	}
	
	/** Update the gradients */
	public void backprop(List<INDArray> error, List<IHashVector> features, List<int[]> tagFrequencies) {
		
		
		Iterator<INDArray> it = error.iterator();
		Iterator<int[]> freqIt = tagFrequencies.iterator();
		
		List<Triplet<INDArray, IHashVector, int[]>> enumerated = 
								new ArrayList<Triplet<INDArray, IHashVector, int[]>>();
		for(IHashVector ptFeatures: features) {
			
			if(!it.hasNext()) {
				throw new RuntimeException("Error and features should be of same size. Found "
								+ error.size() + " and " + features.size() + " resp.");		
			}
			
			enumerated.add(Triplet.of(it.next(), ptFeatures, freqIt.next()));
		}
		
		StreamSupport.stream(Spliterators
				.spliterator(enumerated, Spliterator.IMMUTABLE), true)
				.unordered()
				.forEach(p-> {
		
					INDArray ptError = p.first();
					IHashVector ptFeatures = p.second();
					int[] tagFreq = p.third();
					
					Iterator<Pair<KeyArgs, Double>> featureIt = ptFeatures.iterator();
					
					while(featureIt.hasNext()) {
						
						Pair<KeyArgs, Double> feature = featureIt.next();
						if(feature.second() == 0.0) {
							//inactive feature, though inactive features should not be here in the first place
							continue; 
						}
						
						final int tagIx;
						final String tag = this.getFeatureCluster(feature.first());//feature.first().getArg1();
						if(this.tags.containsKey(tag)) {
							tagIx = this.tags.get(tag);
						} else {
							throw new RuntimeException("Unknown tag " + tag + " that was known during embedding.");
						}
						
						if(tagFreq[tagIx] == 0) {
							throw new RuntimeException("Freq is 0 when it should have not been");
						}
						
						final int tagDim = this.tagDimensions[tagIx];
						final int padding = this.paddedIndex[tagIx];
						
						INDArray newGrad = Nd4j.zeros(tagDim);
						for(int i = 0; i < tagDim; i++) {
							double sumVal = ptError.getDouble(0, padding + i);
							double avgVal = sumVal/(double)tagFreq[tagIx];
							newGrad.putScalar(new int[]{0,  i}, avgVal);
						}
						
						////// word embedding /////
						final String cluster = this.getFeatureCluster(feature.first());
						if(this.wordTags != null && this.wordTags.contains(cluster)) {
							
							final INDArray featureEmbedding = this.featureEmbedding.get(feature.first());
							
							//gradient wrt W
							INDArray gradW = newGrad.transpose().mmul(featureEmbedding); 
							synchronized(this.gradWProjection) {
								this.gradWProjection.addi(gradW);
							}
							
							this.updatedWProjection.set(true);
							
							//new gradient
							newGrad = newGrad.mmul(this.WProjection);
						}
						///////////////////////////
						
						if(this.gradFeatureEmbedding.containsKey(feature.first())) {
							INDArray grad = this.gradFeatureEmbedding.get(feature.first());
							this.updatedFeatures.add(feature.first());
							
							synchronized(grad) {
								grad.addi(newGrad);
							}
						} else {
							if(this.useUnseenFeatureEmbedding) {
								INDArray grad = this.gradUnseenFeatureEmbeddingByTag.get(tagIx);
								this.updatedUnseenFeatureTag.add(tagIx);
								
								synchronized(grad) {
									grad.addi(newGrad);
								}
							} else {
								throw new RuntimeException("Feature should have been registered when during feed forwarding.");
							}
						}
					}
					
					for(int i = 0; i < tagFreq.length; i++) {
						
						if(tagFreq[i] == 0) {
							final int tagDim = this.tagDimensions[i];
							final int padding = this.paddedIndex[i];
							
							INDArray newGrad = Nd4j.zeros(tagDim);
							for(int j = 0; j < tagDim; j++) {
								double val = ptError.getDouble(0, padding + j);
								newGrad.putScalar(new int[]{0,  j}, val);
							}
							
							INDArray grad = this.gradInactiveTagEmbedding.get(i);
							synchronized(grad) {
								grad.addi(newGrad);
							}
							this.updatedInactiveTag.add(i);
						}
					}
				});
	}
	
	/** Update this vector using AdaGrad */
	private void updateVector(INDArray vec, INDArray grad, INDArray sumSquareGrad) {
		
		this.updateVector(vec, grad, sumSquareGrad, null);
//		//Add regularizer
//		grad.addi(vec.mul(this.l2));
//		
//		//not performing clipping
//		
//		INDArray squaredGrad = grad.mul(grad);
//		
//		//update AdaGrad history
//		sumSquareGrad.addi(squaredGrad);
//		
//		//Update the vectors
//		INDArray invertedLearningRate = Nd4j.getExecutioner()
//											.execAndReturn(new Sqrt(sumSquareGrad.dup()))
//											.divi(this.learningRate);
//	
//		vec.subi(grad.div(invertedLearningRate));	
	}
	
	/** Update this vector using AdaGrad */
	private void updateVector(INDArray vec, INDArray grad, INDArray sumSquareGrad, Double threshold) {
		
		//Add regularizer
		grad.addi(vec.mul(this.l2));
		
		//not performing clipping
		if(threshold != null) {
			double norm = grad.normmaxNumber().doubleValue();
			if(norm > threshold) {
				grad.muli(threshold/norm);
			}
		}
		
		INDArray squaredGrad = grad.mul(grad);
		
		//update AdaGrad history
		sumSquareGrad.addi(squaredGrad);
		
		//Update the vectors
		INDArray invertedLearningRate = Nd4j.getExecutioner()
											.execAndReturn(new Sqrt(sumSquareGrad.dup()))
											.divi(this.learningRate);
	
		vec.subi(grad.div(invertedLearningRate));	
	}
	
	/** Update the feature embeddings */
	public void update() {
		
		if(this.empiricalGrad != null) {
			LOG.info("Gradient Check. Empirical Grad %s. Estimated Grad %s", this.empiricalGrad, 
				this.gradFeatureEmbedding.get(this.gradientCheckFeature).getDouble(0, 0));
		}
		
		StreamSupport.stream(Spliterators
				.spliterator(this.updatedFeatures, Spliterator.IMMUTABLE), true)
				.unordered()
				.forEach(ka-> {
		
						INDArray vec = this.featureEmbedding.get(ka);
						INDArray grad = this.gradFeatureEmbedding.get(ka);
						INDArray sumSquareGrad = this.sumSquareGradfeatureEmbedding.get(ka);
							
						this.updateVector(vec, grad, sumSquareGrad);
				});
		
		StreamSupport.stream(Spliterators
				.spliterator(this.updatedInactiveTag, Spliterator.IMMUTABLE), true)
				.unordered()
				.forEach(i-> {
		
					INDArray vec = this.inactiveTagEmbedding.get(i);
					INDArray grad = this.gradInactiveTagEmbedding.get(i);
					INDArray sumSquareGrad = this.sumSquareGradInactiveTagEmbedding.get(i);
					
					this.updateVector(vec, grad, sumSquareGrad);	
				});
		
		if(this.updatedWProjection.get()) {
			this.updateVector(this.WProjection, this.gradWProjection, this.sumSquareGradWProjection);
		}
	}
	
	public void updateOnlyUnseenFeatures() {
		
		StreamSupport.stream(Spliterators
				.spliterator(this.updatedUnseenFeatureTag, Spliterator.IMMUTABLE), true)
				.unordered()
				.forEach(i-> {
					
					if(this.toUpdate.contains(i)) {
						INDArray vec = this.unseenFeatureEmbeddingByTag.get(i);
						INDArray grad = this.gradUnseenFeatureEmbeddingByTag.get(i);
						INDArray sumSquareGrad = this.sumSquareGradUnseenFeatureEmbeddingByTag.get(i);
						
						this.updateVector(vec, grad, sumSquareGrad, 5.0);	
					}
				});
	}
	
	/** Flush the gradients*/
	public void flush() {
		
		for(KeyArgs ka: this.updatedFeatures) {
			INDArray grad = this.gradFeatureEmbedding.get(ka);
			grad.muli(0);
		}
		
		for(Integer i: this.updatedInactiveTag) {
			INDArray grad = this.gradInactiveTagEmbedding.get(i);
			grad.muli(0);
		}
		
		if(this.gradWProjection != null) {
			this.gradWProjection.muli(0);
		}
		
		if(this.useUnseenFeatureEmbedding) {
			for(Integer i: this.updatedUnseenFeatureTag) {
				INDArray grad = this.gradUnseenFeatureEmbeddingByTag.get(i);
				grad.muli(0);
			}
			this.updatedUnseenFeatureTag.clear();
		}
		
		this.updatedFeatures.clear();
		this.updatedInactiveTag.clear();
		this.updatedWProjection.set(false);
	}
	
	public void logEmbeddings(String folderName, String label) {
		
		if(label.length() != 0) {
			label = label + "_";
		}
		
		try (
				OutputStream file = new FileOutputStream(folderName + "/" + label + "feature_embedding.ser");
				OutputStream buffer = new BufferedOutputStream(file);
				ObjectOutput output = new ObjectOutputStream(buffer);
			) {
				output.writeObject(this.featureEmbedding);
			} catch(IOException ex) {
				throw new RuntimeException("Cannot store serializable data");
			}
		
		try (
				OutputStream file = new FileOutputStream(folderName + "/" + label + "inactive_tag_embedding.ser");
				OutputStream buffer = new BufferedOutputStream(file);
				ObjectOutput output = new ObjectOutputStream(buffer);
			) {
				output.writeObject(this.inactiveTagEmbedding);
			} catch(IOException ex) {
				throw new RuntimeException("Cannot store serializable data");
			}
		
		if(this.WProjection != null) {
			
			try (
					OutputStream file = new FileOutputStream(folderName + "/" + label + "w_projection.ser");
					OutputStream buffer = new BufferedOutputStream(file);
					ObjectOutput output = new ObjectOutputStream(buffer);
				) {
					output.writeObject(this.WProjection);
				} catch(IOException ex) {
					throw new RuntimeException("Cannot store serializable data");
				}
		}
		
		try (
				OutputStream file = new FileOutputStream(folderName + "/" + label + "unseen_feature_embedding.ser");
				OutputStream buffer = new BufferedOutputStream(file);
				ObjectOutput output = new ObjectOutputStream(buffer);
			) {
				output.writeObject(this.unseenFeatureEmbeddingByTag);
			} catch(IOException ex) {
				throw new RuntimeException("Cannot store serializable data");
			}
		
		for(int i = 0; i < this.unseenFeatureEmbeddingByTag.size(); i++) {
			INDArray vec = this.unseenFeatureEmbeddingByTag.get(i);
			LOG.info("%s -> %s", i, Helper.printFullVector(vec));
		}
	}
	
	public void logEmbeddingsAsCSV(String folderName, String label) {
		
		if(label.length() != 0) {
			label = label + "_";
		}
		
		final List<KeyArgs> keys = new ArrayList<KeyArgs>();
		
		try (
				PrintWriter writer = new PrintWriter(folderName + "/" + label +  "feature_embedding_values.csv", "UTF-8");
			) {
			
				for(Entry<KeyArgs, INDArray> e: this.featureEmbedding.entrySet()) {
					keys.add(e.getKey());
					writer.println(Helper.printVectorToCSV(e.getValue()));
				}
				writer.close();
			} catch(IOException ex) {
				throw new RuntimeException("Cannot store serializable data");
			}
		
		try (
				OutputStream file = new FileOutputStream(folderName + "/" + label + "feature_embedding_keys.ser");
				OutputStream buffer = new BufferedOutputStream(file);
				ObjectOutput output = new ObjectOutputStream(buffer);
			) {
				output.writeObject(keys);
			} catch(IOException ex) {
				throw new RuntimeException("Cannot store keys");
			}
		
		try (
				PrintWriter writer = new PrintWriter(folderName + "/" + label +  "inactive_tag_embedding.csv", "UTF-8");
			) {
			
				for(INDArray indarray: this.inactiveTagEmbedding) {
					writer.println(Helper.printVectorToCSV(indarray));
				}
				writer.close();
			} catch(IOException ex) {
				throw new RuntimeException("Cannot store serializable data");
			}
		
		if(this.WProjection != null) {
			throw new RuntimeException("Not supported");
		}
		
		try (
				PrintWriter writer = new PrintWriter(folderName + "/" + label + "unseen_feature_embedding.csv", "UTF-8");
			) {
				for(INDArray indarray: this.unseenFeatureEmbeddingByTag) {
					writer.println(Helper.printVectorToCSV(indarray));
				}
				writer.close();
			} catch(IOException ex) {
				throw new RuntimeException("Cannot store serializable data");
			}
	}
	
	public void logEmbeddings(String folderName) {
		this.logEmbeddings(folderName, "");
	}
	
	@SuppressWarnings("unchecked")
	public void bootstrapEmbeddings(String folderName, String label) {
		
		if(label.length() != 0) {
			label = label + "_";
		}
		
		try (
			     InputStream file = new FileInputStream(folderName + "/" + label + "feature_embedding.ser");
			     InputStream buffer = new BufferedInputStream(file);
			     ObjectInput input = new ObjectInputStream (buffer);
			) {
				 Map<KeyArgs, INDArray> featureEmbedding = (Map<KeyArgs, INDArray>) input.readObject();
				 this.featureEmbedding.clear();
//				 this.gradFeatureEmbedding.clear();
//				 this.sumSquareGradfeatureEmbedding.clear();
				 for(Entry<KeyArgs, INDArray> e: featureEmbedding.entrySet()) {
					 this.featureEmbedding.put(e.getKey(), e.getValue());
//					 this.gradFeatureEmbedding.put(e.getKey(), Nd4j.zeros(e.getValue().size(1)));
//					 this.sumSquareGradfeatureEmbedding.put(e.getKey(), Nd4j.zeros(e.getValue().size(1)).addi(0.000001));
				 }
				 
				 LOG.info("Bootstrapped embeddings for %s features", this.featureEmbedding.size());
			} catch(Exception e) {
				throw new RuntimeException("Could not deserialize feature embedding. Error: " + e);
		    }
		
		try (
			     InputStream file = new FileInputStream(folderName + "/" + label + "inactive_tag_embedding.ser");
			     InputStream buffer = new BufferedInputStream(file);
			     ObjectInput input = new ObjectInputStream (buffer);
			) {
				 List<INDArray> inactiveTagEmbedding = (List<INDArray>) input.readObject();
				 this.inactiveTagEmbedding.clear();
//				 this.gradInactiveTagEmbedding.clear();
//				 this.sumSquareGradInactiveTagEmbedding.clear();
				 for(INDArray e: inactiveTagEmbedding) {
					 this.inactiveTagEmbedding.add(e);
//					 this.gradInactiveTagEmbedding.add(Nd4j.zeros(e.size(1)));
//					 this.sumSquareGradInactiveTagEmbedding.add(Nd4j.zeros(e.size(1)).addi(0.000001));
				 }
				 
				 LOG.info("Bootstrapped embeddings for %s tags", this.inactiveTagEmbedding.size());
			} catch(Exception e) {
				throw new RuntimeException("Could not deserialize inactive tag embedding. Error: " + e);
		    }
		
		try (
			     InputStream file = new FileInputStream(folderName + "/" + label + "w_projection.ser");
			     InputStream buffer = new BufferedInputStream(file);
			     ObjectInput input = new ObjectInputStream (buffer);
			) {
				this.WProjection = (INDArray) input.readObject();
				this.gradWProjection = Nd4j.zeros(this.WProjection.shape());
				this.sumSquareGradWProjection = Nd4j.zeros(this.WProjection.shape());
				
				 LOG.info("Bootstrapped W projection %s - %s", this.WProjection.size(0), this.WProjection.size(1));
			} catch(Exception e) {
				LOG.warn("Could not deserialize W projection. Error: " + e);
		    }
		
		if(this.useUnseenFeatureEmbedding) {
			
			try (
				     InputStream file = new FileInputStream(folderName + "/" + label + "unseen_feature_embedding.ser");
				     InputStream buffer = new BufferedInputStream(file);
				     ObjectInput input = new ObjectInputStream (buffer);
				) {
					 List<INDArray> unseenFeatureEmbeddingByTag = (List<INDArray>) input.readObject();
					 this.unseenFeatureEmbeddingByTag.clear();
					 for(INDArray e: unseenFeatureEmbeddingByTag) {
						 this.unseenFeatureEmbeddingByTag.add(e);
					 }
					 
					 LOG.info("Bootstrapped unseen feature embeddings for %s tags", this.unseenFeatureEmbeddingByTag.size());
				} catch(Exception e) {
					LOG.warn("Could not deserialize unseen feature embedding. Error: " + e);
			    }
		}
	}
	
	@SuppressWarnings("unchecked")
	public void bootstrapEmbeddingsAsCSV(String folderName, String label) {
		
		if(label.length() != 0) {
			label = label + "_";
		}
		
		final List<KeyArgs> keys;
		try (
			     InputStream file = new FileInputStream(folderName + "/" + label + "feature_embedding_keys.ser");
			     InputStream buffer = new BufferedInputStream(file);
			     ObjectInput input = new ObjectInputStream (buffer);
			) {
				 keys = (List<KeyArgs>) input.readObject();
			} catch(Exception e) {
				throw new RuntimeException("Could not deserialize feature embedding keys. Error: " + e);
		    }
		
		final List<INDArray> values = new ArrayList<INDArray>();
		final String keysParamFile = folderName + "/" + label + "feature_embedding_values.csv";
		 
		try (BufferedReader br = new BufferedReader(new FileReader(keysParamFile))) {
			
			String line;
		    while ((line = br.readLine()) != null) {
				 values.add(Helper.toVector(line));
		    }
			
		} catch(IOException e) {
			LOG.warn("Could not deserialize feature embedding values. Error: " + e);
		}
		
		if(keys.size() != values.size()) {
			throw new RuntimeException("Different key and value sizes. Found keys: "
										+ keys.size() + " and values: " + values.size());
		}
		
		this.featureEmbedding.clear();
		
		Iterator<KeyArgs> it = keys.iterator();
		for(INDArray value: values) {
			this.featureEmbedding.put(it.next(), value);
		}
		
		LOG.info("Bootstrapped embeddings for %s features", this.featureEmbedding.size());		
		
		final String inactiveTagParamFile = folderName + "/" + label + "inactive_tag_embedding.csv";
		this.inactiveTagEmbedding.clear();
		 
		try (BufferedReader br = new BufferedReader(new FileReader(inactiveTagParamFile))) {
			
			String line;
		    while ((line = br.readLine()) != null) {
				 this.inactiveTagEmbedding.add(Helper.toVector(line));
		    }
			
		} catch(IOException e) {
			throw new RuntimeException("Could not read inactive tag embedding. Error: " + e);
		}
		
		LOG.info("Bootstrapped embeddings for %s tags", this.inactiveTagEmbedding.size());
		
		//not supporting WProject for now
		this.WProjection = null;
		
		if(this.useUnseenFeatureEmbedding) {
			
			final String paramFile = folderName + "/" + label + "unseen_feature_embedding.csv";
			this.unseenFeatureEmbeddingByTag.clear();
			 
			try (BufferedReader br = new BufferedReader(new FileReader(paramFile))) {
				
				String line;
			    while ((line = br.readLine()) != null) {
					 this.unseenFeatureEmbeddingByTag.add(Helper.toVector(line));
			    }
				
			} catch(IOException e) {
				LOG.warn("Could not read unseen feature embedding. Error: " + e);
			}
			
			 LOG.info("Bootstrapped unseen feature embeddings for %s tags", this.unseenFeatureEmbeddingByTag.size());
		}
	}
	
	public void bootstrapEmbeddings(String folderName) {
		this.bootstrapEmbeddings(folderName, "");
	}
	
	public void clearSeenFeaturesStats() {
		
		if(!this.collectStats) {
			return;
		}
		
		for(String key: this.tags.keySet()) {
			this.seenFeaturesByTag.get(key).clear();
			this.unseenFeaturesByTag.get(key).clear();
		}
		this.seenFeatures.clear();
		this.unseenFeatures.clear();
		this.exampleWithNoActiveFeatures.set(0);
	}
	
	public void store() {
		
		try {
			
			final String allFeaturesFileName = this.outputDir.getAbsolutePath() + "/all_features.txt";
			final PrintWriter allFeaturesWriter = new PrintWriter(allFeaturesFileName);
			
			for(String key: this.tags.keySet()) {
				
				allFeaturesWriter.write("Feature: " + key + "\n");
				for(Entry<KeyArgs, INDArray> e: this.featureEmbedding.entrySet()) {
					
					if(e.getKey() == null) {
						continue;
					}
					
					final String myKey = this.getFeatureCluster(e.getKey());//e.getKey().getArg1();
					if(myKey.compareTo(key) != 0) {
						continue;
					}
					allFeaturesWriter.write(e.getKey().toString() + "\n");
				}
			}
			
			allFeaturesWriter.flush();
			allFeaturesWriter.close();
			
			final String seenFeaturesFileName = this.outputDir.getAbsolutePath() + "/seen_features.txt";
			final String unseenFeaturesFileName = this.outputDir.getAbsolutePath() + "/unseen_features.txt";
			
			final PrintWriter seenFeaturesWriter = new PrintWriter(seenFeaturesFileName);
			final PrintWriter unseenFeaturesWriter = new PrintWriter(unseenFeaturesFileName);
			
			for(String key: this.tags.keySet()) {
				
				Set<KeyArgs> seenFeatureThisTag = this.seenFeaturesByTag.get(key);
				Set<KeyArgs> unseenFeatureThisTag = this.unseenFeaturesByTag.get(key);
				
				seenFeaturesWriter.write("Feature: " + key + "\n");
				unseenFeaturesWriter.write("Feature: " + key + "\n");
				
				for(KeyArgs seenFeature: seenFeatureThisTag) {
					seenFeaturesWriter.write(seenFeature.toString() + "\n");
				}
				
				for(KeyArgs unseenFeature: unseenFeatureThisTag) {
					unseenFeaturesWriter.write(unseenFeature.toString() + "\n");
				}
			}
			
			seenFeaturesWriter.flush();
			seenFeaturesWriter.close();
			unseenFeaturesWriter.flush();
			unseenFeaturesWriter.close();
		} catch(Exception e) {
			LOG.warn("Cannot save shift feature embedding stats. Exception " + e);
		}
		
	}
	
	public void stats() {
		
		if(!this.collectStats) {
			return;
		}
		
		for(String key: this.tags.keySet()) {
			LOG.info("Number of features of tag %s -> %s, %s", key,
					this.seenFeaturesByTag.get(key).size(), this.unseenFeaturesByTag.get(key).size());
		}
		
		LOG.info("Number of features %s", this.featureEmbedding.size());
		LOG.info("Number of Seen Features %s", this.seenFeatures.size());
		LOG.info("Number of Unseen features %s", this.unseenFeatures.size());
		LOG.info("Number of examples with 0 active features %s", this.exampleWithNoActiveFeatures.get());
	}
	
	public void profile() {
		
		final long start = System.currentTimeMillis();
		
		Random rnd = new Random();
		
		List<KeyArgs> keys = new ArrayList<KeyArgs>(this.featureEmbedding.keySet());
		int maxStep = 100;
		for(int t = 1; t <= maxStep; t++) {

			List<IHashVector> features = new ArrayList<IHashVector>();
			for(int i = 0; i< 10; i++) {
				
				IHashVector feature = HashVectorFactory.create();
				for(int j = 0; j < 3; j++) {
					
					int ix = rnd.nextInt(keys.size());
					KeyArgs key = keys.get(ix);
					if(key != null)
						feature.add(key, 1.0);
				}
	
				features.add(feature);
			}
			
			this.embedFeatures(features);
		}
		
		final long end = System.currentTimeMillis();
		final long gap = end - start;
		
		LOG.info("Time taken %s, average %s", gap, gap/(double)maxStep);
		
	}
	
//	public static void main(String[] args) throws Exception {
//		
//		INDArray batch = Nd4j.rand(new int[]{20, 300});
//		
//		for(int i = 1; i <= 100; i++) {
//			int shape[] = batch.get(NDArrayIndex.point(3), NDArrayIndex.interval(100, 120)).shape();
//			System.out.println("Shape is " + shape[0] + " x " + shape[1]);
//		}
//				
//	}
}
