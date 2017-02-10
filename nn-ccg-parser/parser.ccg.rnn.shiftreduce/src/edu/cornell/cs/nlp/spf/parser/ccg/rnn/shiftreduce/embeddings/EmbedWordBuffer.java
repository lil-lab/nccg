package edu.cornell.cs.nlp.spf.parser.ccg.rnn.shiftreduce.embeddings;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.ObjectInput;
import java.io.ObjectInputStream;
import java.io.ObjectOutput;
import java.io.ObjectOutputStream;
import java.io.OutputStream;
import java.io.PrintWriter;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.ListIterator;
import java.util.Map;
import java.util.Map.Entry;
import java.util.stream.StreamSupport;
import java.util.Set;
import java.util.Spliterator;
import java.util.Spliterators;

import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.GravesLSTM;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.Sqrt;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

import edu.cornell.cs.nlp.spf.data.collection.IDataCollection;
import edu.cornell.cs.nlp.spf.parser.ccg.rnn.shiftreduce.learning.Helper;
import edu.cornell.cs.nlp.spf.parser.ccg.rnn.shiftreduce.wordembeddings.WordEmbedding;
import edu.uw.cs.lil.amr.data.LabeledAmrSentence;
import edu.cornell.cs.nlp.utils.composites.Pair;
import edu.cornell.cs.nlp.utils.log.ILogger;
import edu.cornell.cs.nlp.utils.log.LoggerFactory;

/** Embeds words buffer */
public class EmbedWordBuffer implements AbstractEmbedding, AbstractRecurrentNetworkHelper {
	
	public static final ILogger LOG = LoggerFactory.create(EmbedWordBuffer.class);
	
	private final WordEmbedding word2VecEmbedding;
	private final MultiLayerNetwork net;
	private final RecurrentSequenceEmbedding wordSeqEmbedding;
	private final int nIn, nOut;
	private final double learningRate;
	private final double l2;
	private final int seed;
	
	/** use only google word2vec or also use tunable word vectors with pos */
	private final boolean useOnlyWord2Vec;
	
	/** Tunable word embedding */
	private int tunableWordEmbeddingDim;
	private final Map<String, INDArray> tunableWordEmbedding;
	private final Map<String, INDArray> gradTunableWordEmbedding;
	private final Map<String, INDArray> adaGradSumSquareTunableWordEmbedding;
	
	/** POS Tag embeddings */
	private boolean usePOSTagging;
	private int posDim;
	private final Map<String, INDArray> posEmbedding;
	private final Map<String, INDArray> gradPOSEmbedding;
	private final Map<String, INDArray> adaGradSumSquarePOSEmbedding;	
	
	/** The current sentence that was forwarded through the Word Buffer RNN */
	private List<Pair<String, String>> currentSentence;
	
	/** Number of words in the sentence that has been just fed*/
	private int wordSeqLength;
	
	/** Mean of activations and gradients for debugging */
	private final Map<String, Double> meanActivations;
	private final Map<String, Double> meanGradients;
	private double meanInputActivations;
	
	public EmbedWordBuffer(WordEmbedding wordEmbedding, double learningRate, double l2, 
			int seed) {
		//this.nIn = wordEmbedding.getDimension();
		this.nOut = 32;//50//32;
		this.learningRate = learningRate;
		this.l2 = l2;
		this.seed = seed;
		
		this.useOnlyWord2Vec = false;
		
		this.meanActivations = new HashMap<String, Double>();
		this.meanGradients = new HashMap<String, Double>();
		this.meanInputActivations = 0.0;
		
		//tunable word embedding
		this.tunableWordEmbeddingDim = 85;
		this.tunableWordEmbedding = new HashMap<String, INDArray>();
		this.gradTunableWordEmbedding = new HashMap<String, INDArray>();
		this.adaGradSumSquareTunableWordEmbedding = new HashMap<String, INDArray>();
		
		//pos tag embedding
		this.posDim = 15;
		this.usePOSTagging = true;
		this.posEmbedding = new HashMap<String, INDArray>();
		this.gradPOSEmbedding = new HashMap<String, INDArray>();
		this.adaGradSumSquarePOSEmbedding = new HashMap<String, INDArray>();
		
		this.currentSentence = null;
		
		//Initialize with null values representing UNKNOWN
		final double epsilonWord = 2*Math.sqrt(6/(double)(this.tunableWordEmbeddingDim + 1));
		INDArray vecWord = this.initialize(this.tunableWordEmbeddingDim, epsilonWord);
		INDArray gradWord = Nd4j.zeros(this.tunableWordEmbeddingDim);
		INDArray adaGradSumSquareWord = Nd4j.zeros(this.tunableWordEmbeddingDim).addi(0.00001);
		
		this.tunableWordEmbedding.put(null, vecWord);
		this.gradTunableWordEmbedding.put(null, gradWord);
		this.adaGradSumSquareTunableWordEmbedding.put(null, adaGradSumSquareWord);
		
		final double epsilonPOS = 2*Math.sqrt(6/(double)(this.posDim + 1));;
		INDArray vecPOS = this.initialize(this.posDim, epsilonPOS);
		INDArray gradPOS = Nd4j.zeros(this.posDim);
		INDArray adaGradSumSquarePOS = Nd4j.zeros(this.posDim).addi(0.00001);
		
		this.posEmbedding.put(null, vecPOS);
		this.gradPOSEmbedding.put(null, gradPOS);
		this.adaGradSumSquarePOSEmbedding.put(null, adaGradSumSquarePOS);
		
		if(this.usePOSTagging) {
			this.nIn = this.tunableWordEmbeddingDim + this.posDim + wordEmbedding.getDimension();	
		} else {
			this.nIn = this.tunableWordEmbeddingDim + wordEmbedding.getDimension();
		}
		
		this.net = this.buildRecurrentNetwork(this.nIn, this.nOut);
		
		this.wordSeqEmbedding = new RecurrentSequenceEmbedding(this.net, this.nIn, this.nOut);
		this.word2VecEmbedding = wordEmbedding;
		this.wordSeqLength = -1;
		LOG.info("Embed Word  Buffer nIn %s nOut %s. Use Only Word2Vec %s. Use POS Tagging %s",
				this.nIn, this.nOut, this.useOnlyWord2Vec, this.usePOSTagging);
	}
	
	@Override
	public MultiLayerNetwork buildRecurrentNetwork(int nIn, int nOut) {
		
		int lstmLayerSize = 100;
		//Set up network configuration:
		MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
					.updater(org.deeplearning4j.nn.conf.Updater.ADAM)
					.optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).iterations(1)
					.learningRate(this.learningRate*2.0)
					.momentum(0.0)
					.seed(this.seed)
					.regularization(true)
					.l2(this.l2)
					.list(2)
					.layer(0, new GravesLSTM.Builder().nIn(nIn).nOut(lstmLayerSize)
							.updater(Updater.ADAM).dropOut(0.0)
							.activation("hardtanh").weightInit(WeightInit.XAVIER).build())
					.layer(1, new GravesLSTM.Builder().nIn(lstmLayerSize).nOut(nOut)
							.updater(Updater.ADAM).dropOut(0.0)
							.activation("hardtanh").weightInit(WeightInit.XAVIER).build())
					.pretrain(false).backprop(true)
					.build();
		
		MultiLayerNetwork net_ = new MultiLayerNetwork(conf);
		net_.init();
		return net_;
	}
	
	private INDArray initialize(int dim, double epsilon) {
		
		INDArray vec = Nd4j.rand(new int[]{1, dim});
		vec.subi(0.5).muli(epsilon);
		
		return vec;
	}
	
	/** Initialize the tunable word and pos embeddings. */
	public void initializeWordAndPosEmbeddings(IDataCollection<LabeledAmrSentence> data) {
		
		Set<String> wordVocab = new HashSet<String>();
		Set<String> posVocab = new HashSet<String>();
		
		for(LabeledAmrSentence labeledSentence: data) {
			wordVocab.addAll(labeledSentence.getSample().getTokens().toList());
			posVocab.addAll(labeledSentence.getSample().getState().getTags().toList());
		}
		
		final double epsilonWord = 2*Math.sqrt(6/(double)(this.tunableWordEmbeddingDim + 1));
		for(String word: wordVocab) {
			INDArray vec = this.initialize(this.tunableWordEmbeddingDim, epsilonWord);
			INDArray grad = Nd4j.zeros(this.tunableWordEmbeddingDim);
			INDArray adaGradSumSquare = Nd4j.zeros(this.tunableWordEmbeddingDim).addi(0.00001);
			
			this.tunableWordEmbedding.put(word, vec);
			this.gradTunableWordEmbedding.put(word, grad);
			this.adaGradSumSquareTunableWordEmbedding.put(word, adaGradSumSquare);
		}
		
		final double epsilonPOS = 2*Math.sqrt(6/(double)(this.posDim + 1));
		for(String pos: posVocab) {
			INDArray vec = this.initialize(this.posDim, epsilonPOS);
			INDArray grad = Nd4j.zeros(this.posDim);
			INDArray adaGradSumSquare = Nd4j.zeros(this.posDim).addi(0.00001);
			
			this.posEmbedding.put(pos, vec);
			this.gradPOSEmbedding.put(pos, grad);
			this.adaGradSumSquarePOSEmbedding.put(pos, adaGradSumSquare);
		}
		
		LOG.info("Loaded: Words %s and POS tags %s.", this.tunableWordEmbedding.size(), this.posEmbedding.size());
	}
	
	/** Dumps the configuration and parameters of the network in a file inside the given folder */
	public void logNetwork(String folderName) {
		
		try {
			
			PrintWriter writer = new PrintWriter(folderName + "/word_buffer_conf.json", "UTF-8");
			
			writer.write(this.net.getLayerWiseConfigurations().toJson());
			writer.flush();
			writer.close();
			
		}  catch (IOException e) {
			throw new RuntimeException("Could not dump the word buffer conf: "+e);
		}
		
		try {
			OutputStream fos = Files.newOutputStream(
					Paths.get(folderName + "/word_buffer_param.bin"));
	        DataOutputStream dos = new DataOutputStream(fos);
		    
			Nd4j.write(this.net.params(), dos);
			dos.flush();
		    dos.close();
		    
		} catch (IOException e) {
			throw new RuntimeException("Could not dump the word buffer params: "+e);
		}
		
		//save the tunable word embedding
		try (
			OutputStream file = new FileOutputStream(folderName + "/tunable_word_embedding.ser");
			OutputStream buffer = new BufferedOutputStream(file);
			ObjectOutput output = new ObjectOutputStream(buffer);
		) {
			output.writeObject(this.tunableWordEmbedding);
		} catch(IOException ex) {
			throw new RuntimeException("Cannot serialize and store tunable embedding");
		}
		
		//save the pos embedding
		try (
			OutputStream file = new FileOutputStream(folderName + "/pos_embedding.ser");
			OutputStream buffer = new BufferedOutputStream(file);
			ObjectOutput output = new ObjectOutputStream(buffer);
		) {
			output.writeObject(this.posEmbedding);
		} catch(IOException ex) {
			throw new RuntimeException("Cannot serialize and store pos embedding");
		}
	}
	
	/** Bootstraps the network with the parameters in the word_buffer_param.bin
	 *  inside the given folder */
	@SuppressWarnings("unchecked")
	public void bootstrapNetworkParam(String folderName) {
		
		final String paramFile = folderName+"/word_buffer_param.bin";
		
		try {
		
			DataInputStream dis = new DataInputStream(new FileInputStream(paramFile));
			INDArray newParams = Nd4j.read(dis);
	
			dis.close();
			this.net.init();
			this.net.setParameters(newParams);
			
		} catch(IOException e) {
			throw new RuntimeException("Could not read the word buffer param: "+e);
		}
		
		try(
			 InputStream file = new FileInputStream(folderName + "/tunable_word_embedding.ser");
			 InputStream buffer = new BufferedInputStream(file);
			 ObjectInput input = new ObjectInputStream (buffer);
		) {
			Map<String, INDArray> loadedTunableWordEmbedding = (HashMap<String, INDArray>)input.readObject();
			this.tunableWordEmbedding.clear();
			this.tunableWordEmbedding.putAll(loadedTunableWordEmbedding);
		} catch(Exception e) {
			throw new RuntimeException("Could not deserialize tunalbe word embeddings. Error: " + e);
		}
		
		try(
			InputStream file = new FileInputStream(folderName + "/pos_embedding.ser");
			InputStream buffer = new BufferedInputStream(file);
			ObjectInput input = new ObjectInputStream (buffer);
		) {
			Map<String, INDArray> loadedPOSEmbedding = (HashMap<String, INDArray>)input.readObject();
			this.posEmbedding.clear();
			this.posEmbedding.putAll(loadedPOSEmbedding);
		} catch(Exception e) {
			throw new RuntimeException("Could not deserialize pos embeddings. Error: " + e);
		}
	}
	
	@Override
	public int getDimension() {
		return this.nOut;
	}
	
	@Override
	public Object getEmbedding(Object obj) {
	
		if(!(obj instanceof List))
			throw new RuntimeException("Object type should be: List<String>");
		
		@SuppressWarnings("unchecked")
		final List<String> buffer = (List<String>)obj;
		
		return this.getEmbedding(buffer);
	}
	
	@Override
	public Object getAllTopLayerEmbedding(Object obj) {
	
		if(!(obj instanceof List))
			throw new RuntimeException("Object type should be: List<Pair<String, String>>");
		
		@SuppressWarnings("unchecked")
		final List<Pair<String, String>> buffer = (List<Pair<String, String>>)obj;
		
		return this.getAllTopLayerEmbedding(buffer);
	}
	
	/** returns embedding of all suffixes of the word buffer. Size of the returned array is n+1
	 * where n is the size of the buffer. ith entry (i in {0,...n}) in the buffer refers to encoding
	 * the suffix with first i words removed. This includes embedding of the empty sequence. 
	 * Assumes the sentence is in normal left to right order. */
	public INDArray[] getAllSuffixEmbeddings(List<String> buffer, List<String> tags) {
		
		if(!this.useOnlyWord2Vec && tags == null) {
			throw new RuntimeException("POS tags are null. Sentence must be a situated sentence<AMR Meta>");
		}
		
		//this.currentSentence = buffer;
		final int n = buffer.size();
		INDArray[] suffixEmbedding = new INDArray[n + 1];
		
		suffixEmbedding[n] = Nd4j.zeros(this.nOut); //encoding of empty sequence
		
		if(buffer.size() == 0) { //edge case
			return suffixEmbedding;
		}
		
		INDArray input = Nd4j.zeros(new int[]{1, this.nIn, n});
		int ix = 0;
		
		Iterator<String> tagIt = tags.iterator();
		
		for(String word: buffer) {
			
			/////////
			final int padding;
			if(this.useOnlyWord2Vec) {
				padding = 0;
			} else {
				
				if(this.usePOSTagging) {
					padding = this.tunableWordEmbeddingDim + this.posDim;
				} else {
					padding = this.tunableWordEmbeddingDim;
				}
			
				String tag = tagIt.next();
				
				INDArray wordEmbedding = this.tunableWordEmbedding.get(word);
				if(wordEmbedding == null) {
					wordEmbedding = this.tunableWordEmbedding.get(null);
				}
				
				final INDArray tunableWordAndTagEmbedding;
				if(this.usePOSTagging) {
					INDArray tagEmbedding = this.posEmbedding.get(tag);
					if(tagEmbedding == null) {
						tagEmbedding = this.posEmbedding.get(null);
						LOG.warn("Unknown pos tag %s", tag);
					}
					tunableWordAndTagEmbedding = Nd4j.concat(1, wordEmbedding, tagEmbedding);
				} else {
					tunableWordAndTagEmbedding = wordEmbedding;
				}
				
				for(int i = 0; i < tunableWordAndTagEmbedding.size(1); i++) {
					input.putScalar(new int[]{0, i, n - ix -1}, tunableWordAndTagEmbedding.getDouble(new int[]{0, i}));
				}
				
				if(tunableWordAndTagEmbedding.size(1) != padding) {
					throw new RuntimeException("Padding dimension not same as tunable + tag embedding");
				}
			}
			/////////

			HashMap<Integer, Double> embedding_ = this.word2VecEmbedding.getWordEmbedding(word);
			Iterator<Entry<Integer, Double>> it = embedding_.entrySet().iterator();
		    while (it.hasNext()) {
		    	Map.Entry<Integer, Double> pair = (Map.Entry<Integer, Double>)it.next();
		    	input.putScalar(new int[]{0, padding + pair.getKey(), n - ix -1}, pair.getValue());
		    }
			
			ix++;
		}
		
		List<INDArray> result = this.wordSeqEmbedding.getIntermediateEmbeddings(input);
		INDArray topLayer = result.get(result.size() - 1);

		for(int i = 0; i < n; i++) {
			INDArray embed = Nd4j.zeros(this.nOut);
			for(int j = 0; j < this.nOut; j++)
				embed.putScalar(j, topLayer.getDouble(new int[]{0, j, n - i - 1}));
			
			suffixEmbedding[i] = embed;
		}
		return suffixEmbedding;
	}
	
	/** returns embedding of the current buffer. 
	 * Assumes the sentence is in reverse order that is right to left. */
	public INDArray getEmbedding(List<String> buffer) {
		
		if(!this.useOnlyWord2Vec) {
			throw new RuntimeException("Operation supported only with Word2Vec");
		}
		
		this.wordSeqLength = buffer.size();
		
		if(buffer.size() == 0) //edge case
			return Nd4j.zeros(this.getDimension());
		
		INDArray input = Nd4j.zeros(new int[]{1, this.nIn, buffer.size()});
		int pad = 0;
		
		for(String word: buffer) {

			/////////
			final int padding;
			if(this.useOnlyWord2Vec) {
				padding = 0;
			} else {
				padding = this.tunableWordEmbeddingDim;
				INDArray wordEmbedding = this.tunableWordEmbedding.get(word);
				if(wordEmbedding == null) {
					wordEmbedding = this.tunableWordEmbedding.get(null);
				}
				
				for(int i = 0; i < wordEmbedding.size(1); i++) {
					input.putScalar(new int[]{0, i, pad}, wordEmbedding.getDouble(new int[]{0,i}));
				}
				
				if(wordEmbedding.size(1) != padding) {
					throw new RuntimeException("Padding not same as word embedding");
				}
			}
			/////////

			HashMap<Integer, Double> embedding_ = this.word2VecEmbedding.getWordEmbedding(word);
			Iterator<Entry<Integer, Double>> it = embedding_.entrySet().iterator();
		    while (it.hasNext()) {
		    	Map.Entry<Integer, Double> pair = (Map.Entry<Integer, Double>)it.next();
		    	input.putScalar(new int[]{0, padding + pair.getKey(), pad}, pair.getValue());
		    }
			
			pad++;
		}
		
		return this.wordSeqEmbedding.getEmbedding(input);
	}
	
	/** returns embedding of all suffixes of the current buffer. 
	 * Assumes the sentence is in reverse order that is right to left. This
	 * function is almost identical to get all suffix embedding except for assuming
	 * a different ordering. */
	public INDArray[] getAllTopLayerEmbedding(List<Pair<String, String>> buffer) {
		
		this.currentSentence = buffer;
		this.wordSeqLength = buffer.size();
		
		INDArray input = Nd4j.zeros(new int[]{1, this.nIn, buffer.size()});
		int ix = 0;
		
		for(Pair<String, String> wordAndTag : buffer) {
			
			final String word = wordAndTag.first();
			final String tag = wordAndTag.second();
			
			/////////
			final int padding;
			if(this.useOnlyWord2Vec) {
				padding = 0;
			} else {
				
				if(this.usePOSTagging) {
					padding = this.tunableWordEmbeddingDim + this.posDim;
				} else {
					padding = this.tunableWordEmbeddingDim;
				}
				
				INDArray wordEmbedding = this.tunableWordEmbedding.get(word);
				if(wordEmbedding == null) {
					wordEmbedding = this.tunableWordEmbedding.get(null);
				}
				
				INDArray tunableWordAndTagEmbedding;
				if(this.usePOSTagging) {
					INDArray tagEmbedding = this.posEmbedding.get(tag);
					if(tagEmbedding == null) {
						tagEmbedding = this.posEmbedding.get(null);
						LOG.warn("Unknown pos tag %s", tag);
					}
					
					tunableWordAndTagEmbedding = Nd4j.concat(1, wordEmbedding, tagEmbedding);
				} else {
					tunableWordAndTagEmbedding = wordEmbedding;
				}
				
				for(int i = 0; i < tunableWordAndTagEmbedding.size(1); i++) {
					input.putScalar(new int[]{0, i, ix}, tunableWordAndTagEmbedding.getDouble(new int[]{0, i}));
				}
				
				if(tunableWordAndTagEmbedding.size(1) != padding) {
					throw new RuntimeException("Padding dimension not same as tunable + tag embedding");
				}
			}
			/////////
			
			HashMap<Integer, Double> word2VecEmbedding = this.word2VecEmbedding.getWordEmbedding(word);

			Iterator<Entry<Integer, Double>> it = word2VecEmbedding.entrySet().iterator();
		    while (it.hasNext()) {
		    	Map.Entry<Integer, Double> pair = (Map.Entry<Integer, Double>)it.next();
		    	input.putScalar(new int[]{0, padding + pair.getKey(), ix}, pair.getValue());
		    }
			
			ix++;
		}
		
		this.meanInputActivations = Helper.meanAbs(input);
		
		return this.wordSeqEmbedding.getAllTopLayerEmbeddings(input);
	}
	
	@Override
	public void backprop(INDArray error) {
			
		if(this.wordSeqLength == 0) { //nothing to do
			return;
		}
		
		// create a error feedback for the sequence
		final int dim = this.getDimension();
		assert error.size(1) == dim;
		
		INDArray paddedError = Nd4j.zeros(new int[]{1, dim, this.wordSeqLength});
		
		for(int i = 0; i < dim; i++) {
			paddedError.putScalar(new int[]{0,  i, this.wordSeqLength -1 }, error.getDouble(i));
		}
		
		/*List<INDArray> errorTimeSeries = */this.wordSeqEmbedding.backprop(paddedError);
		
		if(!this.useOnlyWord2Vec) {
			throw new RuntimeException("Supported only with word2vec static");
//			ListIterator<INDArray> it = errorTimeSeries.listIterator();
//			
//			for(String word: this.currentSentence) {
//				INDArray grad = this.gradTunableWordEmbedding.get(word);
//				if(grad == null) {
//					grad = this.gradTunableWordEmbedding.get(null);
//				}
//				INDArray errorWord = it.next().get(NDArrayIndex.interval(0, this.tunableWordEmbeddingDim));
//				grad.addi(errorWord);
//			}
//			
//			if(it.hasNext()) {
//				throw new RuntimeException("Time series length" + errorTimeSeries.size() + 
//										    " while sentence length " + this.currentSentence.size());
//			}
		}
	}
	
	@Override
	public void backprop(INDArray[] error) {
		
		assert error.length == this.wordSeqLength + 1;
		
		if(this.wordSeqLength == 0) { //nothing to do
			return;
		}
		
		// create a error feedback for the sequence
		final int dim = this.getDimension();
		
		INDArray paddedError = Nd4j.zeros(new int[]{1, dim, this.wordSeqLength});
		//INDArray subMatrix = paddedError.tensorAlongDimension(0, 2, 1);
		
		for(int j = 1; j < error.length; j++) {
			for(int i = 0; i < dim; i++) {
				paddedError.putScalar(new int[]{0,  i, j - 1 }, error[j].getDouble(i));
			}
			//subMatrix.putColumn(j - 1, error[j]);
		}
		
		List<INDArray> errorTimeSeries = this.wordSeqEmbedding.backprop(paddedError);
		
		LOG.info("Activation:: Word-Buffer-Input %s Time %s", this.meanInputActivations, System.currentTimeMillis());
		LOG.info("Gradient:: Word-Buffer-Recurrent-Param %s Time %s", this.wordSeqEmbedding.getMeanGradients(), System.currentTimeMillis());
		LOG.info("Gradient:: Word-Buffer-Input %s Time %s", this.wordSeqEmbedding.getMeanInputGradients(), System.currentTimeMillis());
		
		if(!this.useOnlyWord2Vec) {
			ListIterator<INDArray> it = errorTimeSeries.listIterator();
			
			for(Pair<String,String> wordAndTag: this.currentSentence) {
				
				final String word = wordAndTag.first();
				final String tag = wordAndTag.second();
				
				INDArray errorWordAndTag = it.next();
				
				INDArray gradWord = this.gradTunableWordEmbedding.get(word);
				if(gradWord == null) {
					gradWord = this.gradTunableWordEmbedding.get(null);
				}
				INDArray errorWord = errorWordAndTag.get(NDArrayIndex.interval(0, this.tunableWordEmbeddingDim));
				synchronized(gradWord) {
					gradWord.addi(errorWord);	
				}
				
				if(this.usePOSTagging) {
				
					INDArray gradTag = this.gradPOSEmbedding.get(tag);
					if(gradTag == null) {
						gradTag = this.gradPOSEmbedding.get(null);
					}
					INDArray errorTag = errorWordAndTag.get(NDArrayIndex.
							interval(this.tunableWordEmbeddingDim, this.tunableWordEmbeddingDim + this.posDim));
					gradTag.addi(errorTag);
				}	
			}
			
			if(it.hasNext()) {
				throw new RuntimeException("Time series length" + errorTimeSeries.size() + 
										    " while sentence length " + this.currentSentence.size());
			}
		}
	}
	
	public int tunableWordDim() {
		return this.tunableWordEmbeddingDim;
	}
	
	public INDArray getTunableWordEmbedding(String token) {
		
		INDArray vec = this.tunableWordEmbedding.get(token);
		if(vec == null) {
			vec = this.tunableWordEmbedding.get(null);
		}
		
		return vec;
	}
	
	public void addTunableWordEmbeddingGrad(String token, INDArray error) {
		
		INDArray grad = this.gradTunableWordEmbedding.get(token);
		if(grad == null) {
			grad = this.gradTunableWordEmbedding.get(null);
		}
		
		synchronized(grad) {
			grad.addi(error);
		}
	}
	
	public int posDim() {
		return this.posDim;
	}
	
	public INDArray getPOSEmbedding(String tag) {
		
		INDArray vec = this.posEmbedding.get(tag);
		if(vec == null) {
			vec = this.posEmbedding.get(null);
		}
		
		return vec;
	}
	
	public void addPOSEmbeddingGrad(String tag, INDArray error) {
		
		INDArray grad = this.gradPOSEmbedding.get(tag);
		if(grad == null) {
			grad = this.gradPOSEmbedding.get(null);
		}
		
		synchronized(grad) {
			grad.addi(error);
		}
	}
	
	/** Update the vector given the gradient and adagrad history. Given gradient
	 * will update the AdaGrad history. All the parameters must not be used by other threads.
	 * Warning: This function modifies the gradient value and adagrad history. */
	private void update(INDArray vec, INDArray grad, INDArray sumSquareGrad, String label) {
		
		//// Code below is for debugging
		final double meanActivation = Helper.meanAbs(vec);
		final double meanGradient = Helper.meanAbs(grad);
		
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
		
		
		//Add regularizer
		grad.addi(vec.mul(this.l2));
		
		//not performing clipping
		
		//update AdaGrad history
		sumSquareGrad.addi(grad.mul(grad));
		
		//Update the vectors
		INDArray invertedLearningRate = Nd4j.getExecutioner()
											.execAndReturn(new Sqrt(sumSquareGrad.dup()))
											.divi(this.learningRate);
	
		vec.subi(grad.div(invertedLearningRate));
	}
	
	public void updateParameters() {
		
		if(this.useOnlyWord2Vec) {
			return;
		}
		
		StreamSupport.stream(Spliterators
				.spliterator(this.currentSentence, Spliterator.IMMUTABLE), true).unordered()
				.forEach(wordAndTag-> {
					
					String word = wordAndTag.first();
					String tag = wordAndTag.second();
					
					INDArray vecWord = this.tunableWordEmbedding.get(word);
					INDArray gradWord = this.gradTunableWordEmbedding.get(word);
					INDArray sumSquareGradWord = this.adaGradSumSquareTunableWordEmbedding.get(word);
				
					this.update(vecWord, gradWord, sumSquareGradWord, "tunable-word");
					
					if(this.usePOSTagging) {
						INDArray vecTag = this.posEmbedding.get(tag);
						INDArray gradTag = this.gradPOSEmbedding.get(tag);
						INDArray sumSquareGradTag = this.adaGradSumSquarePOSEmbedding.get(tag);
					
						this.update(vecTag, gradTag, sumSquareGradTag, "pos");	
					}
				});
		
		/////////////
		for(Entry<String, Double> e: this.meanActivations.entrySet()) {
			LOG.info("Activation:: %s  %s", e.getKey(), e.getValue()/(double)this.currentSentence.size());
		}
		
		this.meanActivations.clear();
		
		for(Entry<String, Double> e: this.meanGradients.entrySet()) {
			LOG.info("Gradient:: %s  %s", e.getKey(), e.getValue()/(double)this.currentSentence.size());
		}
		
		this.meanGradients.clear();
		/////////////
	}
	
	public void flushGradients() {
		
		if(this.useOnlyWord2Vec) {
			return;
		}
		
		StreamSupport.stream(Spliterators
				.spliterator(this.currentSentence, Spliterator.IMMUTABLE), true).unordered()
				.forEach(wordAndTag -> {
					
					String word = wordAndTag.first();
					String tag = wordAndTag.second();
					
					INDArray gradWord = this.gradTunableWordEmbedding.get(word);
					gradWord.muli(0);
					
					if(this.usePOSTagging) {
						INDArray gradTag = this.gradPOSEmbedding.get(tag);
						gradTag.muli(0);
					}
				});
	}
	
}
