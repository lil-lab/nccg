package edu.cornell.cs.nlp.spf.parser.ff.shiftreduce.neuralparser;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.File;
import java.io.FileInputStream;
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

import edu.cornell.cs.nlp.spf.parser.ccg.rnn.shiftreduce.embeddings.AbstractEmbedding;
import edu.cornell.cs.nlp.utils.log.ILogger;
import edu.cornell.cs.nlp.utils.log.LoggerFactory;

/** Defines a neural architecture that is used for mixing actions
 * @author Dipendra Misra
 *  */
public class NeuralActionEmbeddingMixer implements AbstractEmbedding, Serializable {
	
	private static final long serialVersionUID = -416119476060333760L;

	public static final ILogger	LOG = LoggerFactory.create(NeuralActionEmbeddingMixer.class);

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
	
	public NeuralActionEmbeddingMixer(int nIn, int nOut, double learningRate, double l2, int seed) {
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
		
		LOG.info("Neural Action Mixer: nIn %s, learning rate %s, l2 %s, seed %s", 
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
		
		//Set up network configuration:
		MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
					.updater(org.deeplearning4j.nn.conf.Updater.ADAGRAD)
					.optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).iterations(1)
					.learningRate(this.learningRate)
					.momentum(0.0)
					.seed(this.seed)
					.regularization(true)
					.l2(this.l2)
					.list(1)
					.layer(0, new DenseLayer.Builder().nIn(this.nIn).nOut(this.nOut)
							.updater(org.deeplearning4j.nn.conf.Updater.ADAGRAD)
							.activation("identity").weightInit(WeightInit.XAVIER).biasInit(0.0).build())
					.pretrain(false).backprop(true)
					.build();
				
		MultiLayerNetwork net = new MultiLayerNetwork(conf);
		net.init();
		
		return net;
	}
	
	
	/** This is a thread-safe implementation of getEmbedding*/
	public List<INDArray> getEmbeddingParallel(List<INDArray> batch) {
		
		// Prepare the batch
		final int batchSize = batch.size();
		INDArray vec = Nd4j.zeros(new int[]{batchSize, this.nIn});
		int ix = 0;
		for(INDArray single: batch) {
			for(int i = 0; i < this.nIn; i++) {
				vec.putScalar(new int[]{ix, i}, single.getDouble(new int[]{0, i}));
			}
			ix++;
		}

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
		
		//Split results back into individual vectors
		List<INDArray> results = new ArrayList<INDArray>();
		for(ix = 0; ix < batchSize; ix++) {
			
			INDArray result = Nd4j.zeros(new int[]{1, this.nOut});
			for(int i = 0; i < this.nOut; i++) {
				result.putScalar(new int[]{0, i}, topLayer.getDouble(new int[]{ix, i}));
			}
			results.add(result);
		}
		
		return results;
	}
	
	/** Embeds object using getEmbedding*/
	public INDArray getEmbedding(INDArray vec) {

		List<INDArray> layerWiseActivations = this.net.feedForward(vec);
		INDArray topLayer = layerWiseActivations.get(layerWiseActivations.size() - 1);
		
		return topLayer;
	}
	
	/** This is NOT a thread-safe implementation of getEmbedding and uses the main net for 
	 * embedding. */
	public List<INDArray> getEmbedding(List<INDArray> batch) {

		// Prepare the batch
		final int batchSize = batch.size();
		INDArray vec = Nd4j.zeros(new int[]{batchSize, this.nIn});
		int ix = 0;
		for(INDArray single: batch) {
			for(int i = 0; i < this.nIn; i++) {
				vec.putScalar(new int[]{ix, i}, single.getDouble(new int[]{0, i}));
			}
			ix++;
		}

		List<INDArray> layerWiseActivations = this.net.feedForward(vec);
		INDArray topLayer = layerWiseActivations.get(layerWiseActivations.size() - 1);
		
		//Split results back into individual vectors
		List<INDArray> results = new ArrayList<INDArray>();
		for(ix = 0; ix < batchSize; ix++) {
			
			INDArray result = Nd4j.zeros(new int[]{1, this.nOut});
			for(int i = 0; i < this.nOut; i++) {
				result.putScalar(new int[]{0, i}, topLayer.getDouble(new int[]{ix, i}));
			}
			results.add(result);
		}
		
		return results;
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
	public List<INDArray> backprop(List<INDArray> batchError) {
		
		// Prepare the batch
		final int batchSize = batchError.size();
		INDArray error = Nd4j.zeros(new int[]{batchSize, this.nOut});
		int ix = 0;
		for(INDArray single: batchError) {
			for(int i = 0; i < this.nOut; i++) {
				error.putScalar(new int[]{ix, i}, single.getDouble(new int[]{0, i}));
			}
			ix++;
		}

		// Backpropagate the error
		INDArray it = this.net.backpropGradient(error).getSecond();
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
//			LOG.info("Action mix Layer %s, norm of gradient %s", i, g.gradient().norm2Number().doubleValue());
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
		
		LOG.info("Action mix Layer Input, norm of gradient %s", it.norm2Number().doubleValue());
		
		//Split results back into individual vectors
		List<INDArray> results = new ArrayList<INDArray>();
		for(ix = 0; ix < batchSize; ix++) {
			
			INDArray result = Nd4j.zeros(new int[]{1, this.nIn});
			for(int i = 0; i < this.nIn; i++) {
				result.putScalar(new int[]{0, i}, it.getDouble(new int[]{ix, i}));
			}
			results.add(result);
		}
		
		return results;
	}
	
	/** Dumps the configuration and parameters of the network in a file inside the given folder */
	public void logNetwork(String folderName) {
		
//		try {
//	        File tempFile = new File(folderName + "/action_mix_mlp");
//	        ModelSerializer.writeModel(this.net, tempFile, true);
//		} catch(Exception e) {
//			throw new RuntimeException("Could not dump the action mix mlp " + e);
//		}
		
		try {
			
			PrintWriter writer = new PrintWriter(folderName + "/action_mix_mlp_conf.json", "UTF-8");
			
			writer.write(this.net.getLayerWiseConfigurations().toJson());
			writer.flush();
			writer.close();
			
		}  catch (IOException e) {
			throw new RuntimeException("Could not dump the mlp conf: "+e);
		}
		
		try {
			OutputStream fos = Files.newOutputStream(
					Paths.get(folderName + "/action_mix_mlp_param.bin"));
	        DataOutputStream dos = new DataOutputStream(fos);
		    
			Nd4j.write(this.net.params(), dos);
			dos.flush();
		    dos.close();
		    
		} catch (IOException e) {
			throw new RuntimeException("Could not dump the mlp params: "+e);
		}
	}
	
	/** Bootstraps the network with the parameters in the action_history_param.bin
	 *  inside the given folder */
	public void bootstrapNetworkParam(String folderName) {
		
		final String paramFile = folderName+"/action_mix_mlp_param.bin";
//		final String paramFile = folderName+"/action_mix_mlp";

		try {
		
//			this.net = ModelSerializer.restoreMultiLayerNetwork(paramFile);
			
			DataInputStream dis = new DataInputStream(new FileInputStream(paramFile));
			INDArray newParams = Nd4j.read(dis);
			dis.close();
			
			this.net.init();
			this.net.setParameters(newParams);
			
		} catch(IOException e) {
			throw new RuntimeException("Could not bootstrap neural action mixer " + e);
		}
	}
	
	public static void main(String[] args) throws Exception {
		
		
		NeuralActionEmbeddingMixer actionMixer = new NeuralActionEmbeddingMixer(356, 300, 0.05, 0.000001, 1234);
		Nd4j.getRandom().setSeed(1234);
		
		List<List<INDArray>> randomVectors = new ArrayList<List<INDArray>>();
		for(int i = 0; i < 40; i++) {
			List<INDArray> vec = new ArrayList<INDArray>();
			for(int j = 0; j < 100; j++) {
				vec.add(Nd4j.rand(new int[]{1, 356}));
			}
			randomVectors.add(vec);
			System.out.println(i + " -> " + vec.get(0));
		}
		
		List<List<INDArray>> vec1 = StreamSupport.stream(Spliterators.spliterator(randomVectors, Spliterator.IMMUTABLE), true)
				.parallel().map(randomVector -> 
						actionMixer.getEmbeddingParallel(randomVector))
				.collect(Collectors.toList());
		
		System.out.println("Again");
		
		List<List<INDArray>> vec2 = StreamSupport.stream(Spliterators.spliterator(randomVectors, Spliterator.IMMUTABLE), true)
				.parallel().map(randomVector -> 
						actionMixer.getEmbeddingParallel(randomVector))
				.collect(Collectors.toList());
		
		int i = 0;
		double maxAbs = 0;
		for(List<INDArray> vs: vec2) {
			int j = 0;
			for(INDArray v: vs) {
				INDArray v2 = vec1.get(i).get(j++);
				maxAbs = Math.max(maxAbs, Math.abs(v.subi(v2).minNumber().doubleValue()));
				maxAbs = Math.max(maxAbs, Math.abs(v.subi(v2).maxNumber().doubleValue()));
				System.out.println("======== \n " + v + "\n " + v2);
			}
			i++;
		}
		if(maxAbs > 0.01) {
			System.out.println("Error exists");
		} else {
			System.out.println("Test 1 passed with max absolute difference as " + maxAbs);
		}
		
		List<List<INDArray>> vec3 = new ArrayList<List<INDArray>>();
		
		for(List<INDArray> vs: randomVectors) {
			List<INDArray> outputs = new ArrayList<INDArray>();
			for(INDArray randomVector: vs) {
				INDArray output = actionMixer.getEmbedding(randomVector);
				outputs.add(output);
			}
			vec3.add(outputs);
		}
		
		i = 0;
		maxAbs = 0;
		for(List<INDArray> vs: vec3) {
			int j = 0;
			for(INDArray v: vs) {
				INDArray v2 = vec1.get(i).get(j++);
				maxAbs = Math.max(maxAbs, Math.abs(v.subi(v2).minNumber().doubleValue()));
				maxAbs = Math.max(maxAbs, Math.abs(v.subi(v2).maxNumber().doubleValue()));
				System.out.println("======== \n " + v + "\n " + v2);
			}
			i++;
		}
		if(maxAbs > 0.01) {
			System.out.println("Error exists");
		} else {
			System.out.println("Test 2 passed with max absolute difference as " + maxAbs);
		}
	}
}
