package edu.cornell.cs.nlp.spf.parser.ccg.rnn.shiftreduce.embeddings;

import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.concurrent.atomic.AtomicLong;

import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.Updater;
import org.deeplearning4j.nn.updater.UpdaterCreator;
import org.deeplearning4j.optimize.api.StepFunction;
import org.deeplearning4j.optimize.stepfunctions.NegativeGradientStepFunction;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import edu.cornell.cs.nlp.spf.parser.ccg.rnn.shiftreduce.learning.Helper;
import edu.cornell.cs.nlp.utils.log.ILogger;
import edu.cornell.cs.nlp.utils.log.LogLevel;
import edu.cornell.cs.nlp.utils.log.LoggerFactory;

/** 
 * This class defines function for embedding a sequence using RNN. To do so,
 * embeddings of the elements in the input space is given. 
 * 
 * @author Dipendra Misra
 */
public class RecurrentSequenceEmbedding implements AbstractEmbedding {
	
	public static final ILogger LOG = LoggerFactory.create(RecurrentSequenceEmbedding.class);
	
	/** recurrent neural network */
	private final MultiLayerNetwork net;
	private final int nIn;
	private final int nOut;
	private final int numThreads;
	private final MultiLayerNetwork[] netClones;
	private final Updater updaters[];
	private final boolean[] free; 
	private AtomicLong pre, mid, post;
	private AtomicLong count, exampleSize;
	
	/** Mean gradients for debugging */
	private final double[] 	meanGradients;
	private Double meanInputGradients;
	
	public RecurrentSequenceEmbedding(MultiLayerNetwork net, int nIn, int nOut) {
		this.net = net;
		this.nIn = nIn;
		this.nOut = nOut; //nIn, nOut can also be inferred from the net
		this.numThreads = 32;
		this.netClones = new MultiLayerNetwork[32];
		this.free = new boolean[32];
		
		for(int i = 0; i<32; i++) {
			this.netClones[i] = this.clone();
			this.free[i] = true;
		}
		
		int  nLayers = net.getnLayers();
		this.updaters = new Updater[nLayers];
		
		for(int l = 0; l < nLayers;l ++) {
			this.updaters[l] = UpdaterCreator.getUpdater(this.net.getLayer(l));
 		}
		
		this.pre = new AtomicLong();
		this.mid = new AtomicLong();
		this.post = new AtomicLong();
		this.count = new AtomicLong();
		this.exampleSize = new AtomicLong();
		//this.packedInput();
		//this.packedInput();
		
		this.meanGradients = new double[this.net.getnLayers()];
		this.meanInputGradients = 0.0;
	}
	
	public void reclone() {
		
		for(int i = 0; i<32; i++) {
			this.netClones[i] = this.clone();
			this.free[i] = true;
		}
	}
	
	public void packedInput() {
		
		final int numSamples = 23;
		INDArray input = Nd4j.zeros(new int[]{numSamples, this.nIn, 1});
		
		for(int j = 0; j < numSamples; j++) {
			for(int i = 0; i < this.nIn; i++) {
				double val = Math.random();
				input.putScalar(new int[]{j, i, 0}, val);
			}
		}
		
		//this.getEmbedding(input, null);
		this.getAllEmbedding(input, null);
	}
	
	public void testRecurrent() {
		
		for(int k = 0; k<25; k++) {
			INDArray input = Nd4j.zeros(new int[]{1, this.nIn, 1});
			for(int i=0; i<this.nIn; i++) {
				double val = Math.random();
				input.putScalar(new int[]{0,i,0}, val);
			}
			
			long start = System.currentTimeMillis();
			this.net.feedForward(input);
			long end = System.currentTimeMillis();
			LOG.info("Time taken for one example is %s", (end - start));
			
			final int numSamples = 10;
			INDArray input2 = Nd4j.zeros(new int[]{numSamples, this.nIn, 1});
			for(int j = 0; j< numSamples; j++) {
				for(int i=0; i<this.nIn; i++) {
					double val = Math.random();
					input2.putScalar(new int[]{j, i, 0}, val);
				}
			}
			
			this.net.rnnClearPreviousState();
			
			start = System.currentTimeMillis();
			this.net.feedForward(input2);
			end = System.currentTimeMillis();
			LOG.info("Time taken is for %s dimensional input %s", numSamples, (end - start));
		}
		System.exit(0);
	}
	
	public MultiLayerNetwork clone() {
		
		MultiLayerNetwork copy = new MultiLayerNetwork(this.net.getLayerWiseConfigurations());
		copy.init();
		copy.setParams(this.net.params().dup());
		return copy;
	}
	
	public void testClone() {
		
		MultiLayerNetwork copy = this.clone();
		
		for(int s = 1; s <= 5; s++) {
			/*code for checking if they are really clone */
			INDArray sample = Nd4j.zeros(new int[]{1, this.nIn, 1});
			for(int i=0; i<this.nIn; i++) {
				double val = Math.random();
				sample.putScalar(new int[]{0, i, 0}, val);
			}
			
			List<INDArray> v1 = this.net.feedForward(sample);
			List<INDArray> v2 = copy.feedForward(sample);
			
			if(v1.size() != v2.size())
				throw new IllegalStateException("Clone does not work!!");
			
			for(int i=0; i<v1.size(); i++) {
				INDArray w1 = v1.get(i);
				INDArray w2 = v2.get(i);
				
				if(!w1.equals(w2))
					throw new IllegalStateException("Clone does not work!!");
			}
		}
		
		LOG.info("They matched");
	}
	
	/** Returns the mean of gradients of different layer from the last backprop pass */
	public String getMeanGradients() {
		StringBuilder s = new StringBuilder();
		s.append("{");
		for(int i = 0; i < this.meanGradients.length; i++) {
			s.append("layer-" + i + ": " + this.meanGradients[i] + ", ");
		}
		s.append("}");
		return s.toString();
	}
	
	public double getMeanInputGradients() {
		return this.meanInputGradients;
	}
	
	@Override
	public int getDimension() {
		return this.nOut;
	}
	
	@Override
	public Object getEmbedding(Object obj) {
	
		//INDArray example = null; 
		if(obj.getClass() != INDArray.class)//example.getClass())
			throw new RuntimeException("Object Class should be of type: INDArray");
		
		final INDArray input = (INDArray)obj;
		
		return this.getEmbedding(input);
	}
	
	public void printTimeStamps() {
		LOG.info("Pre is %s", this.pre.get());
		LOG.info("Mid is %s", this.mid.get());
		LOG.info("Post is %s", this.post.get());
		LOG.info("Calls to recurrent network %s", this.count.get());
		LOG.info("Number of examples %s", this.exampleSize.get());
		LOG.info("Size of every example %s", this.exampleSize.get()/(double)this.count.get());
	}
	
	/** makes prediction on a time-series data using the recurrent-network and returns 
	 * all the intermediate embeddings including the final embedding */
	public List<INDArray> getIntermediateEmbeddings(INDArray input) {
		
		int[] inputShape = input.shape();
		assert inputShape.length == 3;
		
		if(inputShape[2] == 0) {
			List<INDArray> zero = new LinkedList<INDArray>();
			for(int i=0; i<this.net.getnLayers(); i++) {
				//Recommendations: put a 0 vector for each layer of the size of that layer
				zero.add(Nd4j.zeros(this.nOut)); 
			}
			return zero;
		}
		
		synchronized(this.net) {
			return this.net.feedForward(input);
		}
	}
	
	/** makes prediction on a time-series data using the recurrent-network given a set of input with 
	 * the same recurrent input and returns all the intermediate embeddings including the final embedding */
	public List<RecurrentTimeStepOutput> getAllEmbedding(INDArray input, 
														final Map<String, INDArray>[] rnnState) {
		/* CHECK The Code below for whether vectors are being copied or being referenced. If referenced
		 * then there might be bugs. */		
		final int numExamples = input.size(0);
		final int numLayers = this.net.getnLayers();
		this.exampleSize.addAndGet(numExamples);
		
		INDArray result;
		@SuppressWarnings("unchecked")
		Map<String, INDArray>[] newRNNState = new HashMap[numLayers];

		this.count.incrementAndGet();
		int myNetworkId = -1;
		//find a free network
		boolean found = false;
		while(!found) {
			synchronized(this.free) {
				for(int i=0; i<this.numThreads; i++) {
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
		
		long start1 = System.currentTimeMillis();
		//set the rnn activations
		if(rnnState != null) {
			for(int layer = 0; layer < rnnState.length; layer++) {
				mynet.rnnSetPreviousState(layer, rnnState[layer]);
			}
		}
		else  {
				mynet.rnnClearPreviousState();
		}
		
		long start2 = System.currentTimeMillis(); 
		//do feedforwarding
		result = mynet.rnnTimeStep(input);
		long start3 = System.currentTimeMillis();
		
		//get the new activations
		for(int i = 0; i < numLayers; i++) {
			newRNNState[i] = mynet.rnnGetPreviousState(i);
		}
		long start4 = System.currentTimeMillis();
		
		this.pre.addAndGet(start2 - start1);
		this.mid.addAndGet(start3 - start2);
		this.post.addAndGet(start4 - start3);
		
		List<RecurrentTimeStepOutput> outputs = new LinkedList<RecurrentTimeStepOutput>();
		
		for(int ex = 0; ex < numExamples; ex++) {
			
			INDArray embedding = Nd4j.zeros(this.nOut);
			for(int i = 0; i < this.nOut; i++) {
				embedding.putScalar(i, result.getDouble(new int[]{ex, i, 0}));
			}
			
			@SuppressWarnings("unchecked")
			Map<String, INDArray>[] exampleRNNState = new HashMap[numLayers];
			for(int layer = 0; layer < numLayers; layer++) {
				
				Map<String, INDArray> state = new HashMap<String, INDArray>();
				for(Entry<String, INDArray> e: newRNNState[layer].entrySet()) {
					String key = e.getKey();
					INDArray batchStateActivations = e.getValue();
					int numLayerSize = batchStateActivations.shape()[1];  
					INDArray stateActivation = Nd4j.zeros(numLayerSize);
					
					for(int j = 0; j < numLayerSize; j++) {
						stateActivation.putScalar(j, batchStateActivations.getDouble(new int[]{ex, j}));
					}
					
					state.put(key, stateActivation);
				}
				
				exampleRNNState[layer] = state;
			}
			
			RecurrentTimeStepOutput output = new RecurrentTimeStepOutput(embedding, exampleRNNState);
			outputs.add(output);
		}
		
		synchronized(this.free) {
			this.free[myNetworkId] = true;
		}
		
		return outputs;
	}
	
	/** makes prediction on a time-series data using the recurrent-network given input and recurrent
	 * input and returns all the intermediate embeddings including the final embedding */
	public RecurrentTimeStepOutput getEmbedding(INDArray input, final Map<String, INDArray>[] rnnState) {
		
		/* CHECK The Code below for whether vectors are being copied or being referenced. If referenced
		 * then there might be bugs. */
		final int numLayers = this.net.getnLayers();
		INDArray result;
		this.exampleSize.addAndGet(input.size(0));
		
		@SuppressWarnings("unchecked")
		Map<String, INDArray>[] newRNNState = new HashMap[numLayers];

		this.count.incrementAndGet();
		int myNetworkId = -1;
		//find a free network
		boolean found = false;
		while(!found) {
			synchronized(this.free) {
				for(int i=0; i<this.numThreads; i++) {
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
		
		long start1 = System.currentTimeMillis();
		//set the rnn activations
		mynet.rnnClearPreviousState();
		if(rnnState != null) {
			for(int layer = 0; layer < rnnState.length; layer++) {
				mynet.rnnSetPreviousState(layer, rnnState[layer]);
			}
		} else {
			mynet.rnnClearPreviousState();
		}
		
		long start2 = System.currentTimeMillis(); 
		//do feedforwarding
		result = mynet.rnnTimeStep(input);
		
		long start3 = System.currentTimeMillis();
		//get the new activations
		for(int i = 0; i < numLayers; i++) {
			newRNNState[i] = mynet.rnnGetPreviousState(i);
		}
		long start4 = System.currentTimeMillis();
		
		this.pre.addAndGet(start2 - start1);
		this.mid.addAndGet(start3 - start2);
		this.post.addAndGet(start4 - start3);
		
		INDArray embedding = Nd4j.zeros(this.nOut);
		for(int i = 0; i < this.nOut; i++)
			embedding.putScalar(i, result.getDouble(new int[]{0, i, 0}));
		
		synchronized(this.free) {
			this.free[myNetworkId] = true;
		}
		
		RecurrentTimeStepOutput output = new RecurrentTimeStepOutput(embedding, newRNNState);
		return output;
	}
	
	/** makes prediction on a time-series data using the recurrent-network and passes the 
	 * embedding in the top layer. It returns output of size one more than the input time-series
	 * with one extra representing no-sequence. Thus if time series is {a,b,c} then output array
	 * corresponds to embedding of [ {}, {a}, {a,b}, {a,b,c}]. */
	public INDArray[] getAllTopLayerEmbeddings(INDArray input) {
		
		/* CHECK The Code below for whether vectors are being copied or being referenced. If referenced
		 * then there might be bugs. */
		int[] inputShape = input.shape();
		assert inputShape.length == 3;
		
		INDArray[] prefixEmbedding = new INDArray[inputShape[2] + 1];
		prefixEmbedding[0] = Nd4j.zeros(this.nOut); //representing embedding of empty sequence
		
		if(inputShape[2] == 0)
			return prefixEmbedding;
		
		List<INDArray> result;
		
		synchronized(this.net) {
			result = this.net.feedForward(input);
		}
		
		INDArray topLayer = result.get(result.size()-1);
		//INDArray subMatrix = topLayer.tensorAlongDimension(0, 2, 1);
		
		for(int i = 0; i < inputShape[2]; i++) {
			
			INDArray vec = Nd4j.zeros(this.nOut);
			for(int j = 0; j < this.nOut; j++) {
				vec.putScalar(j, topLayer.getDouble(new int[]{0, j, i}));
			}
			
			//INDArray vec = subMatrix.getColumn(i).transposei();
			prefixEmbedding[i + 1] = vec;
		}
		
		return prefixEmbedding;
	}
	
	
	/** makes prediction on a time-series data using the recurrent-network and passes the 
	 * value of the last prediction. */
	public INDArray getEmbedding(INDArray input) {
		
		/* CHECK The Code below for whether vectors are being copied or being referenced. If referenced
		 * then there might be bugs. */
		int[] inputShape = input.shape();
		assert inputShape.length == 3;
		
		if(inputShape[2] == 0)
			return Nd4j.zeros(this.nOut);
		
		List<INDArray> result;
		
		synchronized(this.net) {
			result = this.net.feedForward(input);
		}
		
		//use the embedding of last time series in the final layer
		INDArray topLayer = result.get(result.size()-1);
		int[] shape = topLayer.shape();
		
		INDArray embedding = Nd4j.zeros(this.nOut);
		for(int i = 0; i < this.nOut; i++) {
			embedding.putScalar(i, topLayer.getDouble(new int[]{0, i, shape[2]-1}));
		}
		
		/** if embedding contains NaN or infinity then print it*/
		if(LOG.getLogLevel() == LogLevel.DEBUG) {
			for(int i = 0; i < this.nOut; i++) {
				double val = embedding.getDouble(i);
				if(!Double.isFinite(val)) {
					LOG.info("Found bug size %s, in %s", this.nOut, Helper.printVector(embedding));
					break;
				}
			}
		}
		
		return embedding; //topLayer.getColumns(new int[]{0,shape[1]-1}); //check this line
	}
	
	/** backprops the loss using backpropagation-through-time to the leaves and
	 *  returns the loss at each leaf. Assumes that forward pass has been made
	 *  on the time series for which backprop is being done. The loss is returned
	 *  only for the output corresponding to the last timestep i.e. no intermediate
	 *  time-steps are assumed.  */
	public List<INDArray> backprop(INDArray error) {

		int numLayers = this.net.getnLayers();
		INDArray it = error;
		
		for(int i = numLayers - 1; i >= 0; i--) {
			Layer layer = this.net.getLayer(i);
			Pair<Gradient, INDArray> e = layer.backpropGradient(it);
			
			/* update the parameters of this layer 
			 * add l2 norm, momentum etc. for the first error in future */
			Gradient g = e.getFirst();
			
			this.meanGradients[i] = Helper.meanAbs(g.gradient()); 
			
			Updater updater = this.updaters[i];
			updater.update(layer, g, 0, 1);
			
			//update the parameters of the layer
			StepFunction stepFunction = new NegativeGradientStepFunction();
			INDArray params = layer.params(); 
			stepFunction.step(params, g.gradient());
			layer.setParams(params);
			
			it = e.getSecond();
		}
		
		this.meanInputGradients = Helper.meanAbs(it);
		
		int[] shape = it.shape();
		
		List<INDArray> timeSeries = new LinkedList<INDArray>();
		int dim = shape[1], timeSeriesLength = shape[2];
		
		for(int i = 0; i < timeSeriesLength; i++) {
			
			INDArray inputError = Nd4j.zeros(shape[1]);
			for(int j = 0; j < dim; j++) {
				inputError.putScalar(j, it.getDouble(new int[]{0, j, i}));
			}
			timeSeries.add(inputError);
		}
		
		return timeSeries;
	}

}
