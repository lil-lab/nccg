package edu.cornell.cs.nlp.spf.parser.ccg.rnn.shiftreduce.testunit;

import java.util.ArrayList;
import java.util.List;
import java.util.stream.IntStream;
import java.util.stream.Stream;

import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.distribution.UniformDistribution;
import org.deeplearning4j.nn.conf.layers.GravesLSTM;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import edu.cornell.cs.nlp.spf.parser.ccg.rnn.shiftreduce.embeddings.AbstractRecurrentNetworkHelper;
import edu.cornell.cs.nlp.spf.parser.ccg.rnn.shiftreduce.embeddings.RecurrentSequenceEmbedding;

/** Class for testing the recurrent neural network. Test involves creating a synthetic
 *  sequence and a label, along with a convex loss and tracking the loss function with 
 *  every epoch. We consider both the scenario when we update the sequence and when we 
 *  don't update the sequence. 
 *  
 *  @author Dipendra Misra
 * */
public class RecurrentNetworkTest implements AbstractRecurrentNetworkHelper{
	
	private final RecurrentSequenceEmbedding recurrentEmbedding;
	private final MultiLayerNetwork net;
	private final int nIn, nOut;
	private INDArray input;
	private boolean updateLeafVectors;
	private double learningRate;
	
	public RecurrentNetworkTest(int nIn, int nOut, boolean updateLeafVectors) {
		this.learningRate = 1.5;
		this.net = this.buildRecurrentNetwork(nIn, nOut);
		this.nIn = nIn;
		this.nOut = nOut;
		this.input = null;
		this.updateLeafVectors = updateLeafVectors;
		this.recurrentEmbedding = new RecurrentSequenceEmbedding(this.net, nIn,  nOut);
	}

	@Override
	public MultiLayerNetwork buildRecurrentNetwork(int nIn, int nOut) {
		int lstmLayerSize = 100;
		//Set up network configuration:
		MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
					.optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).iterations(1)
					.learningRate(this.learningRate)
					.momentum(0.5)
					.rmsDecay(0.95)
					.seed(12345)
					.regularization(true)
					.l2(0.1)
					.list(2)
					.layer(0, new GravesLSTM.Builder().nIn(nIn).nOut(lstmLayerSize)
							.updater(Updater.SGD)
							.activation("hardtanh").weightInit(WeightInit.DISTRIBUTION)
							.dist(new UniformDistribution(-0.08, 0.08)).build())
					.layer(1, new GravesLSTM.Builder().nIn(lstmLayerSize).nOut(nOut)
							.updater(Updater.SGD)
							.activation("hardtanh").weightInit(WeightInit.DISTRIBUTION)
							.dist(new UniformDistribution(-0.08, 0.08)).build())
					/*.layer(2, new RnnOutputLayer.Builder(LossFunction.MCXENT).activation("softmax") 
							.updater(Updater.RMSPROP)
							.nIn(25).nOut(nOut).weightInit(WeightInit.DISTRIBUTION)
							.dist(new UniformDistribution(-0.08, 0.08)).build())*/
					.pretrain(false).backprop(true)
					.build();
		
		MultiLayerNetwork net = new MultiLayerNetwork(conf);
		net.init();
		
		return net;
	}
	
	@Override
	public Object getAllTopLayerEmbedding(Object obj) {
		throw new IllegalStateException("Operation Not Supported");
	}

	@Override
	public void backprop(INDArray error) {
		
		int timeSeriesLength = this.input.size(2);
		INDArray paddedError = Nd4j.zeros(new int[]{1, this.nOut, timeSeriesLength});
		for(int j = 0; j < this.nOut; j++) {
			paddedError.putScalar(new int[]{0, j, timeSeriesLength - 1}, error.getDouble(j));
		}
		
		List<INDArray> result = this.recurrentEmbedding.backprop(paddedError);
		
		if(this.updateLeafVectors) {
			int i = 0;
			for(INDArray updateLeaf: result) {
				for(int j = 0; j < this.nIn; j++) {
					double oldVal = this.input.getDouble(new int[]{0, j, i});
					double newVal = oldVal - this.learningRate*updateLeaf.getDouble(j);
					this.input.putScalar(new int[]{0, j, i}, newVal);
				}
				i++;
			}
		}
	}
	
	@Override
	public void backprop(INDArray[] error) {
		
		int timeSeriesLength = this.input.size(2);
		
		if(error.length != timeSeriesLength + 1) {
			throw new RuntimeException("error array must be of length one more than  timeSeriesLength. Expected "
										+ (timeSeriesLength +1 ) + " found " + error.length);
		}
		
		INDArray paddedError = Nd4j.zeros(new int[]{1, this.nOut, timeSeriesLength});
		for(int i = 1; i < error.length; i++) {
			for(int j = 0; j < this.nOut; j++) {
				paddedError.putScalar(new int[]{0, j, i - 1}, error[i].getDouble(j));
			}
		}
		
		List<INDArray> result = this.recurrentEmbedding.backprop(paddedError);
		
		if(this.updateLeafVectors) {
			int i = 0;
			for(INDArray updateLeaf: result) {
				for(int j = 0; j < this.nIn; j++) {
					double oldVal = this.input.getDouble(new int[]{0, j, i});
					double newVal = oldVal - this.learningRate*updateLeaf.getDouble(j);
					this.input.putScalar(new int[]{0, j, i}, newVal);
				}
				i++;
			}
		}
	}
	
	private void learn(INDArray input, INDArray label, int numEpoch) {
		
		this.input = input; 
		
		for(int i = 1; i <= numEpoch; i++) {
			
			INDArray output = this.recurrentEmbedding.getEmbedding(input);
						
			/* compute error: loss = \sum_i (prediction_i - label_i)^2
			 * error: dl/dprediction_i = 2(prediction_i - label_i) */

			double loss = 0;
			for(int k = 0; k < this.nOut; k++) {
				double diff = output.getDouble(new int[]{0, k}) - 
											   label.getDouble(new int[]{0, k});
				loss = loss + diff * diff;
			}
			
			System.out.println("Iteration " + i + ", loss "+loss);
			
			INDArray error = output.dup().subi(label).mul(2); 
			this.backprop(error);
		}
	}
	
	public static void main(String[] args) throws Exception {
		
		int nIn = 200, nOut = 10;
		boolean updateLeafVectors = true;
		RecurrentNetworkTest testUnit = new RecurrentNetworkTest(nIn, nOut, updateLeafVectors); 
		
		/* Create a synthetic sequence */
		int timeSeriesLength = 30;
		INDArray input = Nd4j.rand(new int[]{1, nIn, timeSeriesLength});
		INDArray label = Nd4j.rand(new int[]{1, nOut});
		
		/* update the sequence */
		int numEpoch = 50;
		testUnit.learn(input, label, numEpoch);
	}
}
