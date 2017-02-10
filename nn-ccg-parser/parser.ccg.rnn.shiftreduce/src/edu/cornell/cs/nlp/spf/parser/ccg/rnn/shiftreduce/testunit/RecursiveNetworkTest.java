package edu.cornell.cs.nlp.spf.parser.ccg.rnn.shiftreduce.testunit;

import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;
import java.util.Random;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import edu.cornell.cs.nlp.spf.parser.ccg.rnn.shiftreduce.learning.LearningRate;
import edu.cornell.cs.nlp.spf.parser.ccg.rnn.shiftreduce.recursive.GradientWrapper;
import edu.cornell.cs.nlp.spf.parser.ccg.rnn.shiftreduce.recursive.RecursiveTreeNetwork;
import edu.cornell.cs.nlp.spf.parser.ccg.rnn.shiftreduce.recursive.Tree;
import edu.cornell.cs.nlp.utils.composites.Pair;

/** Class for testing recursive neural network. Creates a synthetic tree and 
 * label and minimizes the squared loss of the synthetic tree 
 * 
 * @author Dipendra Misra
 * */
public class RecursiveNetworkTest {
	
	private final RecursiveTreeNetwork net;
	private final int dim;
	private final List<Pair<INDArray, GradientWrapper>> vectorAndGradients;
	private final double learningRate;
	private final double l2;
	
	public RecursiveNetworkTest(int dim, double learningRate, double regularizer) {
		this.dim = dim;
		this.learningRate = learningRate;
		this.l2 = regularizer;
		LearningRate lr = new LearningRate(learningRate, 0);
		this.net = new RecursiveTreeNetwork(dim, lr, regularizer, new Random(), 12345);
		this.vectorAndGradients = new LinkedList<Pair<INDArray, GradientWrapper>>();
	}
	
	public void addVectorAndGradient(INDArray vector, GradientWrapper gradient) {
		this.vectorAndGradients.add(Pair.of(vector, gradient));
	}
	
	private INDArray estimateGradient(INDArray vec, Tree t, INDArray label) {
		
		final double epsilon = 0.001;
		final INDArray vecEmpiricalGrad = Nd4j.zeros(vec.shape());
		
		for(int i = 0; i < vec.size(0); i++) {
			for(int j = 0; j < vec.size(1); j++) {
			
				double orig = vec.getDouble(new int[]{i, j});
				
				vec.putScalar(new int[]{i,  j}, orig + epsilon);
				INDArray predict1 = this.net.feedForward(t);
				
				// compute the loss (squared loss) and gradient.
				double loss1 = 0;
				for(int k = 0; k < this.dim; k++) {
					double diff = predict1.getDouble(k) - label.getDouble(k);
					loss1 = loss1 + diff * diff;
				}
				
				vec.putScalar(new int[]{i,  j}, orig - epsilon);
				INDArray predict2 = this.net.feedForward(t);
				
				// compute the loss (squared loss) and gradient.
				double loss2 = 0;
				for(int k = 0; k < this.dim; k++) {
					double diff = predict2.getDouble(k) - label.getDouble(k);
					loss2 = loss2 + diff * diff;
				}
				
				double paramGrad = (loss1 - loss2)/(2*epsilon);
				vecEmpiricalGrad.putScalar(new int[]{i,  j}, paramGrad);
				vec.putScalar(new int[]{i,  j}, orig);
			}
		}
		
		return vecEmpiricalGrad;
	}
	
	public void gradientCheck(Tree t, INDArray label, int numEpochs) {
		
		final INDArray W = this.net.getW();
		final INDArray b = this.net.getb();
		final INDArray gradW = this.net.getGradW();
		final INDArray gradb = this.net.getGradb();
		final int[] shapeW = W.shape();
		
		for(int n = 0; n <  numEpochs; n++) {
			
			final List<INDArray> empirical = new ArrayList<INDArray>();
			final INDArray empiricalGradW;
			final INDArray empiricalGradb;
			
			{
				for(Pair<INDArray, GradientWrapper> e: this.vectorAndGradients) {
					
					INDArray vec = e.first();
					INDArray vecEmpiricalGrad = this.estimateGradient(vec, t, label);
					empirical.add(vecEmpiricalGrad);
				}
				
				empiricalGradW = this.estimateGradient(W, t, label);
				empiricalGradb = this.estimateGradient(b, t, label);
			}
			
			INDArray predict = this.net.feedForward(t);
			
			/* compute the loss (squared loss) and gradient. */
			double loss = 0;
			for(int j = 0; j < this.dim; j++) {
				double diff = predict.getDouble(j) - label.getDouble(j);
				loss = loss + diff * diff;
			}
			System.out.println("Loss is "+loss);
			INDArray error = predict.dup().subi(label).mul(2);
			
			this.net.backProp(t, error);
			
			//Check for gradW, gradb
			for(int i = 0; i < shapeW[0]; i++) {
				for(int j = 0; j < shapeW[1]; j++) {
					System.out.println("W: Empirical Grad: " + empiricalGradW.getDouble(new int[]{i, j}) +
							";  Estimate Grad " + gradW.getDouble(new int[]{i, j}));
				}
				System.out.println("b: Empirical Grad: " + empiricalGradb.getDouble(new int[]{i, 0}) +
						";  Estimate Grad " + gradb.getDouble(new int[]{i, 0}));
			}
			
			this.net.updateParameters();
			this.net.flushGradients();
				
			//update the leaf vectors
			int j = 0;
			for(Pair<INDArray, GradientWrapper> vG: this.vectorAndGradients) {
				INDArray vector = vG.first();
				GradientWrapper gradient = vG.second();
				int numTerms = gradient.numTerms();
				
				INDArray empiricalGrad = empirical.get(j++);
				for(int i = 0; i < empiricalGrad.size(1); i++) {
					System.out.println("Leaf Vectors: Empirical Grad: " + empiricalGrad.getDouble(new int[]{0, i}) +
										";  Estimate Grad " + gradient.getGradient().getDouble(new int[]{0, i}));
				}
				
				if(numTerms == 0) {
					continue;
				}
				
				INDArray realGradient = gradient.getGradient().div((double)numTerms);
				realGradient.addi(vector.mul(this.l2));
				vector.subi(realGradient.mul(this.learningRate));
				gradient.flush();
			}
		}
	}
	
	public void learn(Tree t, INDArray label, int numEpochs) {
		
		for(int i = 1; i <= numEpochs; i++) {
			
			INDArray predict = this.net.feedForward(t);
			
			/* compute the loss (squared loss) and gradient. */
			double loss = 0;
			for(int j = 0; j < this.dim; j++) {
				double diff = predict.getDouble(j) - label.getDouble(j);
				loss = loss + diff * diff;
			}
			System.out.println("Loss is "+loss);
			INDArray error = predict.dup().subi(label).mul(2);
			
			this.net.backProp(t, error);
			this.net.updateParameters();
			this.net.flushGradients();
			
			//update the leaf vectors
			for(Pair<INDArray, GradientWrapper> vG: this.vectorAndGradients) {
				INDArray vector = vG.first();
				GradientWrapper gradient = vG.second();
				int numTerms = gradient.numTerms();
				if(numTerms == 0) {
					continue;
				}
				
				INDArray realGradient = gradient.getGradient().div((double)numTerms);
				realGradient.addi(vector.mul(this.l2));
				vector.subi(realGradient.mul(this.learningRate));
				gradient.flush();
			}
		}
	}

	public static void main(String args[]) throws Exception {
		
		// create certain leaf vectors and their gradient 
		int dim = 30;
		INDArray a = Nd4j.rand(new int[]{1, dim});
		INDArray c = Nd4j.rand(new int[]{1, dim});
		INDArray b = Nd4j.rand(new int[]{1, dim});
		
		GradientWrapper gradA = new GradientWrapper(dim); 
		GradientWrapper gradB = new GradientWrapper(dim);
		GradientWrapper gradC = new GradientWrapper(dim);
		
		// create tree (label1 (a (label2 (b c))))
		Tree aLeaf  = new Tree("a", new LinkedList<Tree>());
		aLeaf.setVector(a); aLeaf.setGradient(gradA);
		
		Tree bLeaf  = new Tree("b", new LinkedList<Tree>());
		bLeaf.setVector(b); bLeaf.setGradient(gradB);
		
		Tree cLeaf  = new Tree("c", new LinkedList<Tree>());
		cLeaf.setVector(c); cLeaf.setGradient(gradC);
		
		List<Tree> n2Children = new LinkedList<Tree>();
		n2Children.add(bLeaf); n2Children.add(cLeaf);
		Tree n2 = new Tree("label2", n2Children);
		
		List<Tree> n1Children = new LinkedList<Tree>();
		n1Children.add(aLeaf); n1Children.add(n2);
		Tree root = new Tree("label1", n1Children);
		
		INDArray label = Nd4j.rand(new int[]{1, dim});
		
		// train a recursive neural network
		double learningRate = 0.01;
		double l2 = 0.01;
		RecursiveNetworkTest test = new RecursiveNetworkTest(dim, learningRate, l2);
		test.addVectorAndGradient(a, gradA);
		test.addVectorAndGradient(b, gradB);
		test.addVectorAndGradient(c, gradC);
		
		// test for gradients
		final int numEpochs = 20;
		System.out.println("Recursive Network: Gradient Test. See if Estimated and Empirical Gradients are close.");
		test.gradientCheck(root, label, numEpochs);
		
		// test for optimization
		System.out.println("Recursive Network: Optimization test. See if loss decreases to near 0");
		test.learn(root, label, numEpochs);
	}
}
