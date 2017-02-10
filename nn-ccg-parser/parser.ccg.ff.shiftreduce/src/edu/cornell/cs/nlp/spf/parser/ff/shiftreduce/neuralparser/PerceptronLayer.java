package edu.cornell.cs.nlp.spf.parser.ff.shiftreduce.neuralparser;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.io.Serializable;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Arrays;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import edu.cornell.cs.nlp.utils.log.ILogger;
import edu.cornell.cs.nlp.utils.log.LoggerFactory;

/** We add a linear perceptron layer on top of dot-product parser */
public class PerceptronLayer implements Serializable {
	
	private static final long serialVersionUID = -5376363308730123752L;

	public static final ILogger	LOG = LoggerFactory.create(PerceptronLayer.class);
	
	private final INDArray weights;
	private final int dim;
	private final double l2;
	private final double learningRate;
	
	public PerceptronLayer(int dim) {
		
		this.dim = dim;
		this.l2 = 0.0;
		this.learningRate = 1.0;
		
		//Weight initialization scheme, currently initializes by 0.0
		this.weights = Nd4j.zeros(this.dim);
		
		//Initialize the last weight to 1 as it represents the score without perceptron
		this.weights.putScalar(new int[]{0,  this.dim - 1}, 10.0);
		
		LOG.info("Created perceptron layer of dim %s, learning rate %s, l2 %s", this.dim, this.learningRate, this.l2);
	}
	
	/** computes linear score by performing dot product */
	public double getLinearScore(INDArray feature) {
		return this.weights.mmul(feature.transpose()).getDouble(new int[]{0, 0});
	}
	
	/** converts linear score to probabilites by exponentiating followed by normalizing */
	public static double[] normalizeToLogProbability(double[] linearScore) {
		
		double max = Double.NEGATIVE_INFINITY;
		for(int i = 0; i < linearScore.length; i++) {
			if(linearScore[i] > max) {
				max = linearScore[i];
			}
		}
		
		double[] logProbability = new double[linearScore.length];
		Arrays.fill(logProbability, 0.0);
		double sum = 0.0;
		
		for(int i = 0; i < linearScore.length; i++) {
			sum = sum + Math.exp(linearScore[i] - max);
			logProbability[i] = linearScore[i] - max;
		}
		
		sum = Math.log(sum);
		
		for(int i = 0; i < linearScore.length; i++) {
			logProbability[i] = logProbability[i] - sum;
		}
		
		return logProbability;
	}
	
	public INDArray getWeights() {
		return this.weights;
	}
	
	/** Perform one update i.e. we  do
	 * w^{t+1} = w^{t} + goldFeature - argmaxFeature
	 */
	public void update(INDArray goldFeature, INDArray argmaxFeature) {
		INDArray l2Part = this.weights.mul(this.l2);
		INDArray update = goldFeature.sub(argmaxFeature).subi(l2Part).muli(this.learningRate);
		//this.weights.addi(goldFeature).subi(argmaxFeature).subi(l2Part);
		this.weights.addi(update);
		LOG.info("Feature difference %s", goldFeature.sub(argmaxFeature));
		LOG.info("Weights %s, norm2 %s, l2 %s", this.weights, this.weights.norm2Number().doubleValue(), l2Part.norm2Number().doubleValue());
	}
	
	public void logPerceptronWeights(String folderName) {
		
		//Save W
		try {
			OutputStream fos = Files.newOutputStream(
					Paths.get(folderName + "/perceptron_weights.bin"));
	        DataOutputStream dos = new DataOutputStream(fos);
		    
			Nd4j.write(this.weights, dos);
			dos.flush();
		    dos.close();
		    
		} catch (IOException e) {
			throw new RuntimeException("Could not dump the perceptron weight " + e);
		}
	}
	
	public void bootstrapPerceptronWeights(String folderName) {
	
		//load W
		final String paramFile = folderName+"/perceptron_weights.bin";
		
		try {
		
			DataInputStream dis = new DataInputStream(new FileInputStream(paramFile));
			INDArray loadedW = Nd4j.read(dis);
			Nd4j.copy(loadedW, this.weights);
			
			dis.close();
			LOG.info("Bootstrapped perceptron weight: %s", this.weights);
		} catch(IOException e) {
			LOG.info("Could not bootstrap perceptron weight");
//			throw new RuntimeException("could not read perceptron weight "+e);
		}
	}
	
	public static void main(String[] args) throws Exception {
		
		double[] v = new double[3];
		v[0] = 1.0;
		v[1] = 2.0;
		v[2] = 3.0;
		
		double[] w = PerceptronLayer.normalizeToLogProbability(v);
		double sum = 0;
		for(int i = 0; i < w.length; i++) {
			System.out.println("w[" + i + "] = " + w[i]);
			sum = sum + w[i];
		}
		
		System.out.println("w sum is " + sum);
	}
}
