package edu.cornell.cs.nlp.spf.parser.ff.shiftreduce.neuralparser;

import java.io.BufferedReader;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileReader;
import java.io.IOException;
import java.io.OutputStream;
import java.io.PrintWriter;
import java.io.Serializable;
import java.lang.reflect.Method;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.Spliterator;
import java.util.Spliterators;
import java.util.Map.Entry;
import java.util.stream.Collectors;
import java.util.stream.StreamSupport;

import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.api.Updater;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.updater.UpdaterCreator;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.api.StepFunction;
import org.deeplearning4j.optimize.stepfunctions.NegativeGradientStepFunction;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import edu.cornell.cs.nlp.spf.base.hashvector.KeyArgs;
import edu.cornell.cs.nlp.spf.parser.ccg.rnn.shiftreduce.embeddings.AbstractEmbedding;
import edu.cornell.cs.nlp.spf.parser.ccg.rnn.shiftreduce.learning.Helper;
import edu.cornell.cs.nlp.utils.log.ILogger;
import edu.cornell.cs.nlp.utils.log.LoggerFactory;

/** Defines a neural architecture that is used for scoring parsing steps
 * @author Dipendra Misra
 *  */
public class NeuralParsingDotProductStepScorer implements AbstractEmbedding, Serializable {
	
	private static final long serialVersionUID = -416119476060333760L;

	public static final ILogger	LOG = LoggerFactory.create(NeuralParsingDotProductStepScorer.class);

	private /*final*/ MultiLayerNetwork net;
	private final int nIn, nOut;
	private final MultiLayerNetwork[] netClones;
	private final boolean[] free; 
	/** Number of clones of MLP for parallel embedding. Should be enough to accomodate
	 * the maximum threads that can be active at a given time. */
	private final static int numClones = 64;
	private final Updater updaters[];
	private final double learningRate;
	private final double l2;
	private final int seed;
	
	private Double empiricalGrad;
	
	public NeuralParsingDotProductStepScorer(int nIn, int nOut, double learningRate, double l2, int seed) {
		this.nIn = nIn;
		this.nOut = nOut;
		this.learningRate = learningRate;
		this.l2 = l2;
		this.seed = seed;
		
		this.net = this.buildMLP();
		int  nLayers = net.getnLayers();
		this.updaters = new Updater[nLayers];
		
		for(int l = 0; l < nLayers;l ++) {
			this.updaters[l] = UpdaterCreator.getUpdater(this.net.getLayer(l));
 		}
		
		this.netClones = new MultiLayerNetwork[numClones];
		this.free = new boolean[numClones];
		
		for(int i = 0; i< numClones; i++) {
			this.netClones[i] = this.clone();
			this.free[i] = true;
		}
		
		this.empiricalGrad = null;
		
		//////////
//		try {
//			Method method = MultiLayerNetwork.class.getDeclaredMethod("initGradientsView");
//			method.setAccessible(true);
//			method.invoke(this.net);
//		} catch(Exception e) {
//			throw new RuntimeException("Reflection failed. Error " + e);
//		}
		//////////
		
		LOG.info("Neural MLP Parsing Step Scorer: nIn %s, learning rate %s, l2 %s, seed %s", 
										this.nIn, this.learningRate, this.l2, this.seed);	
	}
	
	public void reclone() {
		
		for(int i = 0; i < numClones; i++) {
			this.netClones[i] = this.clone();
			this.free[i] = true;
		}
	}
	
	public MultiLayerNetwork clone() {
		
		MultiLayerNetwork copy = new MultiLayerNetwork(this.net.getLayerWiseConfigurations());
		copy.init();
		copy.setParams(this.net.params().dup());
		return copy;
	}
	
	public void setEmpiricalGrad(double empiricalGrad) {
		this.empiricalGrad = empiricalGrad;
	}
	
	public MultiLayerNetwork buildMLP() {
		
		int hiddenUnits = 65;//50; //80; //65; //80; //65; //50; //60;//75;//80;//200;//100;
		
		//Set up network configuration:
		MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
					.updater(org.deeplearning4j.nn.conf.Updater.ADAGRAD)
					.optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).iterations(1)
					.learningRate(this.learningRate)
					.momentum(0.0)
					.seed(this.seed)
					.regularization(true)
					.l2(this.l2)
					.list(3)
					.layer(0, new DenseLayer.Builder().nIn(this.nIn).nOut(hiddenUnits)
							.updater(org.deeplearning4j.nn.conf.Updater.ADAGRAD)
							.activation("relu").weightInit(WeightInit.XAVIER).biasInit(0.1).build())
					.layer(1, new DenseLayer.Builder().nIn(hiddenUnits).nOut(hiddenUnits)
							.updater(org.deeplearning4j.nn.conf.Updater.ADAGRAD)
							.activation("relu").weightInit(WeightInit.XAVIER).biasInit(0.1).build())
					.layer(2, new DenseLayer.Builder().nIn(hiddenUnits).nOut(this.nOut)
							.updater(org.deeplearning4j.nn.conf.Updater.ADAGRAD)
							.activation("identity").weightInit(WeightInit.XAVIER).biasInit(0.0).build())
					.pretrain(false).backprop(true)
					.build();
				
		MultiLayerNetwork net = new MultiLayerNetwork(conf);
		net.init();
		
		return net;
	}
	
	
	/** This is a thread-safe implementation of getEmbedding*/
	public INDArray getEmbeddingParallel(INDArray vec) {

		int myNetworkId = -1;
		//find a free network
		boolean found = false;
		while(!found) {
			synchronized(this.free) {
				for(int i=0; i<numClones; i++) {
					if(this.free[i]) {
						found = true;
						myNetworkId = i;
						this.free[i] = false;
						break;
					}
				}
			}
		}
		
		MultiLayerNetwork mynet = this.netClones[myNetworkId];
		
		List<INDArray> layerWiseActivations = mynet.feedForward(vec);
		
		synchronized(this.free) {
			this.free[myNetworkId] = true;
		}
		
		INDArray topLayer = layerWiseActivations.get(layerWiseActivations.size() - 1);
		return topLayer;
	}
	
	/** This is a thread-safe implementation of getEmbedding*/
	public INDArray getEmbedding(INDArray vec) {

		List<INDArray> layerWiseActivations = this.net.feedForward(vec);
		INDArray topLayer = layerWiseActivations.get(layerWiseActivations.size() - 1);
		
		return topLayer;
	}
	
	/** Given exponents, performs softmax followed by log operation. Max exponent trick
	 * is used to handle overflow issues. */
	public double[] toLogSoftMax(double[] exponents) {
		
		double max = Double.MIN_VALUE;
		for(int i = 0; i < exponents.length; i++) {
			if(exponents[i] > max) {
				max = exponents[i];
			}
		}
		
		double Z = 0.0;
		for(int i = 0; i < exponents.length; i++) {
			Z = Z + Math.exp(exponents[i] - max);
		}
		
		final double logZ = Math.log(Z);
		
		double[] logSoftMax = new double[exponents.length];
		
		for(int i = 0; i < exponents.length; i++) {
			logSoftMax[i] = exponents[i] - max - logZ;
		}
		
		return logSoftMax;
	}
	
	/** Gradient check for MLP. Perturbs a parameter in first layer and produces two output. */
	public edu.cornell.cs.nlp.utils.composites.Pair<double[], double[]> gradientCheckGetEmbedding(INDArray batch, double epsilon) {
		
		final Layer layer = this.net.getLayer(0);
		INDArray params = layer.params();
		final double orig = params.getDouble(new int[]{0, 0});
		final int batchSize = batch.size(0);
		
		final double[] exponents1 = new double[batchSize];
		params.putScalar(new int[]{0,  0}, orig + epsilon);
		layer.setParams(params);
		
		{   
			List<INDArray> layerWiseActivations = this.net.feedForward(batch);
			INDArray topLayer = layerWiseActivations.get(layerWiseActivations.size() - 1);
			
			for(int i = 0; i < batchSize; i++) {		
				exponents1[i] = topLayer.getDouble(new int[]{i, 0});
			}
		}
		
		final double[] exponents2 = new double[batchSize];
		params.putScalar(new int[]{0,  0}, orig - epsilon);
		layer.setParams(params);
		
		{   
			List<INDArray> layerWiseActivations = this.net.feedForward(batch);
			INDArray topLayer = layerWiseActivations.get(layerWiseActivations.size() - 1);
			
			for(int i = 0; i < batchSize; i++) {		
				exponents2[i] = topLayer.getDouble(new int[]{i, 0});
			}
		}
		
		params.putScalar(new int[]{0,  0}, orig);
		layer.setParams(params);
		
		return edu.cornell.cs.nlp.utils.composites.Pair.of(exponents1, exponents2);
	}
	
	
	@Override
	public int getDimension() {
		return this.nOut;
	}

	@Override
	public Object getEmbedding(Object obj) {
		throw new RuntimeException("Operation not supported");
	}
	
	/** backprops the loss using standard backpropagation algo. to the leaves and
	 *  returns the loss at each leaf. Assumes that forward pass has been made
	 *  on the batch for which backprop is being done. */
	public INDArray backprop(INDArray error) {
		
		return this.net.backpropGradient(error).getSecond();
		
//		int numLayers = this.net.getnLayers();
//		INDArray it = error;
//		
//		for(int i = numLayers - 1; i >= 0; i--) {
//			Layer layer = this.net.getLayer(i);
//			Pair<Gradient, INDArray> e = layer.backpropGradient(it);
//			
//			/* update the parameters of this layer 
//			 * add l2 norm, momentum etc. for the first error in future */
//			Gradient g = e.getFirst();
//			
//			LOG.info("Layer %s, norm of gradient %s", i, g.gradient().norm2Number().doubleValue());
//			
//			if(empiricalGrad != null && i == 0) {
//				final INDArray gradient = g.gradient();
//				LOG.info("Shape of gradient %s x %s", gradient.size(0), gradient.size(1));
//				double estimateGradient = gradient.getDouble(new int[]{0, 0});
//				LOG.info("Found %s vs %s", this.empiricalGrad, estimateGradient);
//			}
//			
//			Updater updater = this.updaters[i];
//			updater.update(layer, g, 0, 1);
//			
//			//update the parameters of the layer
//			StepFunction stepFunction = new NegativeGradientStepFunction();
//			INDArray params = layer.params(); 
//			stepFunction.step(params, g.gradient());
//			layer.setParams(params);
//			
//			it = e.getSecond();
//		}
//		
//		LOG.info("Layer Input, norm of gradient %s", it.norm2Number().doubleValue());
//		
//		return it;
	}
	
	/** Dumps the configuration and parameters of the network in a file inside the given folder */
	public void logNetwork(String folderName) {
		
//		try {
//	        File tempFile = new File(folderName + "/configuration_mlp");
//	        ModelSerializer.writeModel(this.net, tempFile, true);
//		} catch(Exception e) {
//			throw new RuntimeException("Could not dump the mlp configuration " + e);
//		}
		
		try {
			
			PrintWriter writer = new PrintWriter(folderName + "/mlp_conf.json", "UTF-8");
			
			writer.write(this.net.getLayerWiseConfigurations().toJson());
			writer.flush();
			writer.close();
			
		}  catch (IOException e) {
			throw new RuntimeException("Could not dump the mlp conf: "+e);
		}
		
		try {
			OutputStream fos = Files.newOutputStream(
					Paths.get(folderName + "/mlp_param.bin"));
	        DataOutputStream dos = new DataOutputStream(fos);
		    
			Nd4j.write(this.net.params(), dos);
			dos.flush();
		    dos.close();
		    
		} catch (IOException e) {
			throw new RuntimeException("Could not dump the mlp params: "+e);
		}
	}
	
	/** Dumps the configuration and parameters of the network in a file inside the given folder */
	public void logNetworkAsCSV(String folderName) {
		
//		try {
//	        File tempFile = new File(folderName + "/configuration_mlp");
//	        ModelSerializer.writeModel(this.net, tempFile, true);
//		} catch(Exception e) {
//			throw new RuntimeException("Could not dump the mlp configuration " + e);
//		}
		
		try {
			
			PrintWriter writer = new PrintWriter(folderName + "/mlp_conf.json", "UTF-8");
			
			writer.write(this.net.getLayerWiseConfigurations().toJson());
			writer.flush();
			writer.close();
			
		}  catch (IOException e) {
			throw new RuntimeException("Could not dump the mlp conf: "+e);
		}
		
		try (
				PrintWriter writer = new PrintWriter(folderName + "/mlp_param.csv", "UTF-8");
			) {
				writer.println(Helper.printVectorToCSV(this.net.params()));
				writer.close();
			} catch(IOException ex) {
				throw new RuntimeException("Cannot store serializable data");
			}
	}
	
	/** Bootstraps the network with the parameters in the action_history_param.bin
	 *  inside the given folder */
	public void bootstrapNetworkParam(String folderName) {
		
		final String paramFile = folderName+"/mlp_param.bin";
//		final String paramFile = folderName+"/configuration_mlp";
		
		try {
			
//			this.net = ModelSerializer.restoreMultiLayerNetwork(paramFile);
		
			DataInputStream dis = new DataInputStream(new FileInputStream(paramFile));
			INDArray newParams = Nd4j.read(dis);
			dis.close();
			this.net.init();
			this.net.setParameters(newParams);
			
		} catch(IOException e) {
			throw new RuntimeException("Could not read the top layer param: "+e);
		}
	}
	
	/** Bootstraps the network with the parameters in csv format */
	public void bootstrapNetworkParamFromCSV(String folderName) {
		
		final String paramFile = folderName + "/mlp_param.csv";
		
		String line = null;
				
		try (BufferedReader br = new BufferedReader(new FileReader(paramFile))) {
		    line = br.readLine();
		} catch(IOException e) {
			throw new RuntimeException("Could not read the top layer param: "+e);
		}
		
		if(line == null) {
			throw new RuntimeException("Could not read mlp parameters from csv. Found null.");
		}
		
		INDArray newParams = Helper.toVector(line);
		this.net.init();
		this.net.setParameters(newParams);
	}
	
	public static void main(String[] args) throws Exception {
		
		NeuralParsingDotProductStepScorer actionMixer = new NeuralParsingDotProductStepScorer(356, 300, 0.05, 0.000001, 1234);
		Nd4j.getRandom().setSeed(1234);
		
		List<INDArray> randomVectors = new ArrayList<INDArray>();
		for(int i = 0; i < 30; i++) {
			INDArray vec = Nd4j.rand(new int[]{1, 356});
			randomVectors.add(vec);
			System.out.println(i + " -> " + vec);
		}
		
		List<INDArray> newVec1 = StreamSupport.stream(Spliterators.spliterator(randomVectors, Spliterator.IMMUTABLE), true)
				.map(randomVector -> 
						actionMixer.getEmbeddingParallel(randomVector))
				.collect(Collectors.toList());
		
		for(INDArray vec: newVec1) {
			System.out.println(vec);
		}
		
		System.out.println("Again");
		
		List<INDArray> newVec2 = StreamSupport.stream(Spliterators.spliterator(randomVectors, Spliterator.IMMUTABLE), true)
				.map(randomVector -> 
						actionMixer.getEmbeddingParallel(randomVector))
				.collect(Collectors.toList());
		
		for(INDArray vec: newVec2) {
			System.out.println(vec);
		}
		
	}
}
