package edu.cornell.cs.nlp.spf.parser.ccg.rnn.shiftreduce.embeddings;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.io.PrintWriter;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;

import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.api.Updater;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.updater.UpdaterCreator;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.api.StepFunction;
import org.deeplearning4j.optimize.stepfunctions.NegativeGradientStepFunction;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import edu.cornell.cs.nlp.spf.parser.ccg.rnn.shiftreduce.parsingopembeddings.ParsingOpEmbeddingResult;

/*** This class takes the parsing op and embedding from the 3 RNNs and outputs a single scalar
 * which represents the exponent, proportional to the score you achieve. */
public class TopLayerMLP implements AbstractEmbedding {

	private final MultiLayerNetwork net;
	private final int nIn;
	private final MultiLayerNetwork[] netClones;
	private final boolean[] free; 
	private final int numMaxThreads;
	private final Updater updaters[];
	private final double learningRate;
	private final double l2;
	private final int seed;
	
	public TopLayerMLP(int nIn, double learningRate, double l2, int seed) {
		this.nIn = nIn;
		this.learningRate = learningRate;
		this.l2 = l2;
		this.seed = seed;
		
		this.net = this.buildMLP(nIn);
		int  nLayers = net.getnLayers();
		this.updaters = new Updater[nLayers];
		
		for(int l = 0; l < nLayers;l ++) {
			this.updaters[l] = UpdaterCreator.getUpdater(this.net.getLayer(l));
 		}
		
		this.numMaxThreads = 32;
		this.netClones = new MultiLayerNetwork[32];
		this.free = new boolean[32];
		
		for(int i = 0; i<32; i++) {
			this.netClones[i] = this.clone();
			this.free[i] = true;
		}
	}
	
	public void reclone() {
		
		for(int i = 0; i<32; i++) {
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
	
	public MultiLayerNetwork buildMLP(int nIn) {
		
		int hiddenUnits = 100;
		
		//Set up network configuration:
		MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
					.updater(org.deeplearning4j.nn.conf.Updater.ADAGRAD)
					.optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).iterations(1)
					.learningRate(this.learningRate)
					.momentum(0.0)
					.seed(this.seed)
					.regularization(true)
					.l2(this.l2)
					.list(2)
					.layer(0, new DenseLayer.Builder().nIn(nIn).nOut(hiddenUnits)
							.updater(org.deeplearning4j.nn.conf.Updater.ADAGRAD)
							.activation("leakyrelu").weightInit(WeightInit.NORMALIZED).build())
					.layer(1, new DenseLayer.Builder().nIn(hiddenUnits).nOut(1)
							.updater(org.deeplearning4j.nn.conf.Updater.ADAGRAD)
							.activation("leakyrelu").weightInit(WeightInit.NORMALIZED).build())
					.pretrain(false).backprop(true)
					.build();
				
		MultiLayerNetwork net = new MultiLayerNetwork(conf);
		net.init();
		
		return net;
	}
	
	/** This is a thread-safe implementation of getEmbedding*/
	public double getEmbeddingParallel(INDArray parsingOpEmbedding, INDArray currentStateEmbedding) {

		INDArray batch = Nd4j.concat(1, parsingOpEmbedding, currentStateEmbedding);
		
		int myNetworkId = -1;
		//find a free network
		boolean found = false;
		while(!found) {
			synchronized(this.free) {
				for(int i=0; i<this.numMaxThreads; i++) {
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
		
		List<INDArray> layerWiseActivations = mynet.feedForward(batch);
		
		synchronized(this.free) {
			this.free[myNetworkId] = true;
		}
		
		INDArray topLayer = layerWiseActivations.get(layerWiseActivations.size() - 1);
		
		double exponent = topLayer.getDouble(new int[]{0, 0});
		return exponent;
	}
	
	/** Given a parsing operation and current state it embeds them and a single scalar which is used as exponent. */
	public double getEmbedding(INDArray parsingOpEmbedding, INDArray currentStateEmbedding) {
		
		INDArray batch = Nd4j.concat(1, parsingOpEmbedding, currentStateEmbedding);
		
		List<INDArray> layerWiseActivations = this.net.feedForward(batch);
		INDArray topLayer = layerWiseActivations.get(layerWiseActivations.size() - 1);
		
		double exponent = topLayer.getDouble(new int[]{0, 0});
		return exponent;
	}
	
	/** Given a current state, this function allows you to embed a list of parsing operaiton embeddings.
	 * This is used by the learner where the list of parsing op embeddings correspond to all the 
	 * different options (or a subset) that can be tried . */
	public double[] getEmbedding(List<INDArray> parsingOpEmbeddings, INDArray currentStateEmbedding) {
		
		int size = parsingOpEmbeddings.size();
		if(size == 0) {
			throw new RuntimeException("No parsing operations to embed.");
		}
		
		int nIn = parsingOpEmbeddings.get(0).size(1) + currentStateEmbedding.size(1);
		
		INDArray batch = Nd4j.zeros(new int[]{parsingOpEmbeddings.size(), nIn});
		
		int ix = 0;
		for(INDArray parsingOpEmbedding: parsingOpEmbeddings) {
			for(int i = 0; i < nIn; i++) {
				if(i < parsingOpEmbedding.size(1)) { 
					batch.putScalar(new int[]{ix, i}, parsingOpEmbedding.getDouble(new int[]{0, i}));
				} else {
					batch.putScalar(new int[]{ix, i}, 
								currentStateEmbedding.getDouble(new int[]{0, i - parsingOpEmbedding.size(1)}));
				}
			} 
			ix++;
		}
		
		List<INDArray> layerWiseActivations = this.net.feedForward(batch);
		INDArray topLayer = layerWiseActivations.get(layerWiseActivations.size() - 1);
		
		double[] exponents = new double[size];
		for(int i = 0; i < exponents.length; i++ ) {
			exponents[i] = topLayer.getDouble(new int[]{i, 0});
		}
		
		return exponents;
	}
	
	/** Given a current state, this function allows you to embed a list of parsing operaiton embeddings.
	 * This is used by the learner where the list of parsing op embeddings correspond to all the 
	 * different options (or a subset) that can be tried . */
	public List<Double>[] getEmbedding(List<ParsingOpEmbeddingResult>[] parsingOpEmbeddingResults, INDArray[] stateEmbedding) {
		
		if(parsingOpEmbeddingResults.length != stateEmbedding.length) {
			throw new RuntimeException("Decision length dont match.");
		}
		
		int batchSize = 0;
		
		for(int i = 0; i < stateEmbedding.length; i++) {
			
			List<ParsingOpEmbeddingResult> parsingOpEmbeddingResult = parsingOpEmbeddingResults[i];
			
			int size = parsingOpEmbeddingResult.size();
			if(size == 0) {
				throw new RuntimeException("No parsing operations to embed.");
			}
			
			batchSize = batchSize + parsingOpEmbeddingResult.size();
		}
		
	
		INDArray batch = Nd4j.zeros(new int[]{batchSize, this.nIn});
		
		int ix = 0;
		for(int i = 0; i < stateEmbedding.length; i++) {
			
			List<ParsingOpEmbeddingResult> parsingOpEmbeddingResult = parsingOpEmbeddingResults[i];
			INDArray currentStateEmbedding = stateEmbedding[i];
			
			for(ParsingOpEmbeddingResult parsingOpEmbeddingResult_: parsingOpEmbeddingResult) {
				INDArray parsingOpEmbedding = parsingOpEmbeddingResult_.getEmbedding();
				
				for(int j = 0; j < this.nIn; j++) {
					if(j < parsingOpEmbedding.size(1)) { 
						batch.putScalar(new int[]{ix, j}, parsingOpEmbedding.getDouble(new int[]{0, j}));
					} else {
						batch.putScalar(new int[]{ix, j}, 
									currentStateEmbedding.getDouble(new int[]{0, j - parsingOpEmbedding.size(1)}));
					}
				} 
				ix++;
			}
		}
		
		List<INDArray> layerWiseActivations = this.net.feedForward(batch);
		INDArray topLayer = layerWiseActivations.get(layerWiseActivations.size() - 1);
		
		@SuppressWarnings("unchecked")
		List<Double>[] allExponents = new List[stateEmbedding.length];
		
		int eix = 0;
		for(int i = 0; i < stateEmbedding.length; i++) {
		
			int numParsingOps = parsingOpEmbeddingResults[i].size();
			List<Double> exponents = new ArrayList<Double>(numParsingOps);
			
			for(int j = 0; j < numParsingOps; j++ ) {
				exponents.add(topLayer.getDouble(new int[]{eix, 0}));
				eix++;
			}
			
			allExponents[i] = exponents;
		}
		
		return allExponents;
	}
	
	
	/** backprops the loss using standard backpropagation algo. to the leaves and
	 *  returns the loss at each leaf. Assumes that forward pass has been made
	 *  on the batch for which backprop is being done. */
	public List<INDArray> backprop(Double[] errorExponent) {

		INDArray error = Nd4j.zeros(new int[]{errorExponent.length, 1});
		for(int i = 0; i < errorExponent.length; i++) {
			error.putScalar(new int[]{i, 0}, errorExponent[i]);
		}
		
		int numLayers = this.net.getnLayers();
		INDArray it = error;
		
		for(int i = numLayers - 1; i >= 0; i--) {
			Layer layer = this.net.getLayer(i);
			Pair<Gradient, INDArray> e = layer.backpropGradient(it);
			
			/* update the parameters of this layer 
			 * add l2 norm, momentum etc. for the first error in future */
			Gradient g = e.getFirst();
			
			Updater updater = this.updaters[i];
			updater.update(layer, g, 0, 1);
			
			//update the parameters of the layer
			StepFunction stepFunction = new NegativeGradientStepFunction();
			INDArray params = layer.params(); 
			stepFunction.step(params, g.gradient());
			layer.setParams(params);
			
			it = e.getSecond();
		}
		
		int[] shape = it.shape();
		
		List<INDArray> errorBatch = new LinkedList<INDArray>();
		int batchSize = shape[0]; int dim = shape[1];
		
		for(int i = 0; i < batchSize; i++) {
			
			INDArray inputError = Nd4j.zeros(new int[]{1, dim});
			for(int j = 0; j < dim; j++) {
				inputError.putScalar(new int[]{0, j}, it.getDouble(new int[]{i, j}));
			}
			errorBatch.add(inputError);
		}
		
		return errorBatch;
	}

	/** backprops the loss using standard backpropagation algo. to the leaves and
	 *  returns the loss at each leaf. Assumes that forward pass has been made
	 *  on the batch for which backprop is being done. */
	public List<INDArray>[] backprop(List<Double>[] errorExponent) {

		int batchSize = 0;
		for(int i = 0; i < errorExponent.length; i++) {
			batchSize = batchSize + errorExponent[i].size();
		}
		
		INDArray error = Nd4j.zeros(new int[]{batchSize, 1});
		
		int ix = 0;
		for(int i = 0; i < errorExponent.length; i++) {
			
			Iterator<Double> it = errorExponent[i].iterator();
			while(it.hasNext()) {
				error.putScalar(new int[]{ix, 0}, it.next() * batchSize);
				ix++;
			}
		}
		
		int numLayers = this.net.getnLayers();
		INDArray it = error;
		
		for(int i = numLayers - 1; i >= 0; i--) {
			Layer layer = this.net.getLayer(i);
			Pair<Gradient, INDArray> e = layer.backpropGradient(it);
			
			/* update the parameters of this layer 
			 * add l2 norm, momentum etc. for the first error in future */
			Gradient g = e.getFirst();
			
			Updater updater = this.updaters[i];
			updater.update(layer, g, 0, 1);
			
			//update the parameters of the layer
			StepFunction stepFunction = new NegativeGradientStepFunction();
			INDArray params = layer.params(); 
			stepFunction.step(params, g.gradient());
			layer.setParams(params);
			
			it = e.getSecond();
		}
			
		@SuppressWarnings("unchecked")
		List<INDArray>[] errorBatch = new List[errorExponent.length];
		ix = 0;
		for(int i = 0; i < errorExponent.length; i++) {
			
			errorBatch[i] = new ArrayList<INDArray>(errorExponent[i].size());
			Iterator<Double> expIt = errorExponent[i].iterator();
			while(expIt.hasNext()) {
				
				expIt.next();
				
				INDArray inputError = Nd4j.zeros(new int[]{1, this.nIn});
				for(int j = 0; j < this.nIn; j++) {
					inputError.putScalar(new int[]{0, j}, it.getDouble(new int[]{ix, j})/batchSize);
				}
				errorBatch[i].add(inputError);
				ix++;
			}
		}
		
		return errorBatch;
	}

	@Override
	public int getDimension() {
		return 1;
	}

	@Override
	public Object getEmbedding(Object obj) {
		throw new RuntimeException("Operation Not supported");
	}
	
	/** Dumps the configuration and parameters of the network in a file inside the given folder */
	public void logNetwork(String folderName) {
		
		try {
			
			PrintWriter writer = new PrintWriter(folderName + "/top_layer_mlp_conf.json", "UTF-8");
			
			writer.write(this.net.getLayerWiseConfigurations().toJson());
			writer.flush();
			writer.close();
			
		}  catch (IOException e) {
			throw new RuntimeException("Could not dump the top layer mlp conf: "+e);
		}
		
		try {
			OutputStream fos = Files.newOutputStream(
					Paths.get(folderName + "/top_layer_mlp_param.bin"));
	        DataOutputStream dos = new DataOutputStream(fos);
		    
			Nd4j.write(this.net.params(), dos);
			dos.flush();
		    dos.close();
		    
		} catch (IOException e) {
			throw new RuntimeException("Could not dump the top layer mlp params: "+e);
		}
	}
	
	/** Bootstraps the network with the parameters in the action_history_param.bin
	 *  inside the given folder */
	public void bootstrapNetworkParam(String folderName) {
		
		final String paramFile = folderName+"/top_layer_mlp_param.bin";
		
		try {
		
			DataInputStream dis = new DataInputStream(new FileInputStream(paramFile));
			INDArray newParams = Nd4j.read(dis);
	
			dis.close();
			//MultiLayerNetwork savedNetwork = new MultiLayerNetwork(confFromJson);
			this.net.init();
			this.net.setParameters(newParams);
			
		} catch(IOException e) {
			throw new RuntimeException("Could not read the top layer param: "+e);
		}
	}
	
	//For testing
	public static void main(String[] args) throws Exception {
		
		TopLayerMLP mlp = new TopLayerMLP(10, 0.01, 0.0, 1234);
		
		INDArray parsingOpEmbedding = Nd4j.rand(new int[]{1, 5}, 1111);
		INDArray state = Nd4j.rand(new int[]{1, 5}, 1212);
		
		//Check getEmbedding
		System.out.println("Val " + mlp.getEmbedding(parsingOpEmbedding, state));
		
		//Check getEmbedidng with multiple operations
		INDArray parsingOpEmbedding2 = Nd4j.rand(new int[]{1, 5}, 1331);
		INDArray parsingOpEmbedding3 = Nd4j.rand(new int[]{1, 5}, 1441);
		INDArray parsingOpEmbedding4 = Nd4j.rand(new int[]{1, 5}, 1331);
		
		List<INDArray> parsingOpEmbeddings = new LinkedList<INDArray>();
		parsingOpEmbeddings.add(parsingOpEmbedding);
		parsingOpEmbeddings.add(parsingOpEmbedding2);
		parsingOpEmbeddings.add(parsingOpEmbedding3);
		parsingOpEmbeddings.add(parsingOpEmbedding4); //same as parsingOpEmbeddings2

		double[] exponents = mlp.getEmbedding(parsingOpEmbeddings, state);
		for(int i = 0; i < exponents.length; i++) {
			System.out.println("exponent index " + i + " value " + exponents[i]);
		}

		//Check learning
		double[] labels = new double[exponents.length];
		labels[0] = 0.0; labels[1]  = 0.5; labels[2] = 1.0; labels[3] = 0.5;
		int step = 200;
		Double[] dl = new Double[exponents.length];
		
		double empirical = 0;
		for(int i = 0; i < step; i++) {
			
			{
				double epsilon = 0.001;
				
				INDArray vec = parsingOpEmbeddings.get(0);
				double param = vec.getDouble(new int[]{0, 0});
				
				vec.putScalar(new int[]{0, 0}, param + epsilon);
				double[] exponents_ = mlp.getEmbedding(parsingOpEmbeddings, state);
				
				double loss1 = 0;
				for(int j = 0; j < exponents_.length; j++) {
					dl[j] = 2*(exponents_[j] - labels[j]);
					loss1 = loss1 + (exponents_[j] - labels[j])*(exponents_[j] - labels[j]);
				}
				
				vec.putScalar(new int[]{0, 0}, param - epsilon);
				double[] exponents2 = mlp.getEmbedding(parsingOpEmbeddings, state);
				
				double loss2 = 0;
				for(int j = 0; j < exponents2.length; j++) {
					dl[j] = 2*(exponents2[j] - labels[j]);
					loss2 = loss2 + (exponents2[j] - labels[j])*(exponents2[j] - labels[j]);
				}
				
				empirical = (loss1 - loss2)/(2*epsilon);
				
				vec.putScalar(new int[]{0, 0}, param);
			}
			
			
			double[] exponents_ = mlp.getEmbedding(parsingOpEmbeddings, state);
			
			double loss = 0;
			for(int j = 0; j < exponents_.length; j++) {
				dl[j] = 2*(exponents_[j] - labels[j]);
				loss = loss + (exponents_[j] - labels[j])*(exponents_[j] - labels[j]);
			}
			
			List<INDArray> error = mlp.backprop(dl);
			
			double estimate = error.get(0).getDouble(new int[]{0, 0});
			
			System.out.println("Gradient estimate" + estimate  + " empirical " + empirical);
			
			System.out.println("Loss at step " + i + " = " + loss);
		}
		
		System.out.println("All checks passed.");
	}
}
