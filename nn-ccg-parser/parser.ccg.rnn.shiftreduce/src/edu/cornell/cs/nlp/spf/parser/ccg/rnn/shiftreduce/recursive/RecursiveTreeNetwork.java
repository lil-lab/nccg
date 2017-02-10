package edu.cornell.cs.nlp.spf.parser.ccg.rnn.shiftreduce.recursive;

import java.io.Serializable;
import java.util.Random;
import java.util.stream.IntStream;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.Sqrt;
import org.nd4j.linalg.api.ops.impl.transforms.Tanh;
import org.nd4j.linalg.api.ops.impl.transforms.TanhDerivative;
import org.nd4j.linalg.factory.Nd4j;

import edu.cornell.cs.nlp.spf.parser.ccg.rnn.shiftreduce.embeddings.AbstractEmbedding;
import edu.cornell.cs.nlp.spf.parser.ccg.rnn.shiftreduce.learning.Helper;
import edu.cornell.cs.nlp.spf.parser.ccg.rnn.shiftreduce.learning.LearningRate;
import edu.cornell.cs.nlp.utils.log.ILogger;
import edu.cornell.cs.nlp.utils.log.LogLevel;
import edu.cornell.cs.nlp.utils.log.LoggerFactory;

public class RecursiveTreeNetwork implements AbstractRecursiveTreeNetwork, AbstractEmbedding, Serializable {

	private static final long serialVersionUID = 4962509636176449621L;

	public static final ILogger LOG = LoggerFactory.create(RecursiveTreeNetwork.class);
	
	private final INDArray W, b;
	private final int n;
	private final INDArray gradW, gradb;
	private final double regularizer;
	private final LearningRate learningRate;
	
	/** Running sum of square of gradients for AdaGrad learning rate */
	private final INDArray adaGradSumSquareGradW, adaGradSumSquareGradb; 
	
	/** Optimization variables that are needed frequently*/
	private final int[] firstHalf;
	private final int[] secondHalf;
	
	/** for a binarized tree, W is in Rnx2n and b is in Rn*/
	public RecursiveTreeNetwork(int n, LearningRate learningRate, Double regularizer, 
								Random rnd, int seed) {
		this.n = n;
		
		//initialize parameters uniformly in [-sqrt{6/(r+c)}, -sqrt{6/(r+c)}]
		double epsilonW = 2*Math.sqrt(6.0/(double)(n + 2*n));
		if(rnd == null) {
			this.W = Nd4j.rand(new int[]{n, 2*n}); 
		} else {
			int seedW = Math.abs(rnd.nextInt(2 * seed)) + 1000; 
			this.W = Nd4j.rand(new int[]{n, 2*n}, seedW); 
		}
		
		this.W.subi(0.5).muli(epsilonW);
		
		double epsilonb = 2*Math.sqrt(6.0/(double)(n + 1));
		if(rnd == null) {
			this.b = Nd4j.rand(new int[]{n, 1}); 
		} else {
			int seedb = Math.abs(rnd.nextInt(2 * seed)) + 1000;
			this.b = Nd4j.rand(new int[]{n, 1}, seedb); 
		}
		
		this.b.subi(0.5).muli(epsilonb);
		
		this.gradW = Nd4j.zeros(n, 2*n);
		this.gradb = Nd4j.zeros(n, 1);
		
		double epsilon = 0.00001;
		this.adaGradSumSquareGradW = Nd4j.zeros(n, 2*n).addi(epsilon); 
		this.adaGradSumSquareGradb = Nd4j.zeros(n, 1).addi(epsilon);
		
		this.learningRate = learningRate;
		this.regularizer = regularizer;
		
		this.firstHalf = new int[n];
		this.secondHalf = new int[n];
		
		for(int i = 0; i < 2*n; i++) {
			if(i < n) {
				this.firstHalf[i] = i;
			}
			else {
				this.secondHalf[i - n] = i;
			}
		}
		
		LOG.setCustomLevel(LogLevel.INFO);
	}
	
	@Override
	public INDArray feedForward(Tree t) {
		
		if(t.numChildren() == 0) { //leaf vector, leaf vectors are already initialized
			assert t.getVector() != null : "Label of t is " + t.getLabel();
			int[] shape = t.getVector().shape();
			if(shape[0] !=1 || shape[1] != this.getDimension())
				throw new RuntimeException("Feed forward null. Label " + t.getLabel() + 
											". Shape is " + shape[0] + " and " + shape[1]);
			
			return t.getVector();
		}
		else if(t.numChildren() == 2) { //binary
			
			//do these recursive calls in parallel in future
			INDArray left  = this.feedForward(t.getChild(0)); 
			INDArray right = this.feedForward(t.getChild(1));
			
			//perform composition
			INDArray concat = Nd4j.concat(1, left, right).transpose();
			INDArray transformed = this.W.mmul(concat).add(this.b).transpose();			
			t.setPreOutput(transformed);
			
			//next operation does not create new copy of transformed
			INDArray nonLinear = Nd4j.getExecutioner().execAndReturn(new Tanh(transformed.dup()));
			t.setVector(nonLinear);
			
			return nonLinear;
		}
	
		throw new IllegalStateException("Binarize the tree");
	}
	
	@Override
	public void backProp(Tree t, INDArray error) {
		//error term gives us del loss/ del y (y is the output of this node)
		
		if(error.normmaxNumber().doubleValue() == 0) {
			return;
		}
		
		if(LOG.getLogLevel() == LogLevel.DEBUG) {
			
			int dim = error.size(1); 
			for(int i = 0; i < dim; i++) {
				double u = error.getDouble(i);
				if(Double.isNaN(u)) {
					LOG.info("NaN in the backprop error heading itself");
				}
				
				if(Double.isInfinite(u)) {
					LOG.info("Infinity in the backprop error heading itself");
				}
			}
			
			LOG.debug("Recusive Tree Network Error: %s", Helper.printFullVector(error));
			LOG.debug("Maximum Number (gradW): %s, (gradb): %s ", 
					 this.gradW.maxNumber().doubleValue(), this.gradb.maxNumber().doubleValue());
		}
		
		if(t.numChildren() == 0) { 
			//fine tune leaf vector embeddings
			LOG.debug("Leaf with gradient %s", Helper.printVector(error));
			//GradientWrapper gw = t.getGradient();
			//synchronized(gw) {
			t.addGradient(error); //from now on, the synchronization is done within the tree
			//}
		}
		else if(t.numChildren() == 2) {
			
			/* add to the gradient the loss due to this node given by
			 * del+ loss /del theta = error * del+ y / del theta  */
			
			//nonLinear Derivative = [g'(Wx+b)]
			INDArray nonLinearDerivative = Nd4j.getExecutioner()
					.execAndReturn(new TanhDerivative(t.getPreOutput().dup()));
			INDArray nonLinearDerivativeTranspose = nonLinearDerivative.transpose();
			
			/* del loss / del W 
			 * complex tensor. Handle each row of W at a time. */
			
			final int[] shape = this.W.shape();
			assert shape.length == 2;
			
			Tree left = t.getChild(0);
			Tree right = t.getChild(1);
			
			INDArray leftVector = left.getVector();
			INDArray rightVector = right.getVector();
			INDArray x  = Nd4j.concat(1, leftVector, rightVector); //can be cached in future
			
			double v = x.normmaxNumber().doubleValue();
			if(Double.isNaN(v)) {
				LOG.info("category sent value NaN");
			}
			
			if(Double.isInfinite(v)) {
				LOG.info("category sent value Infinite");
			}
			
			INDArray errorTimesNonLinearDerivative = error.mul(nonLinearDerivative).transpose();
			
			synchronized(this.gradW) {
				this.gradW.addi(errorTimesNonLinearDerivative.mmul(x));
			}

			//del loss / del b = error * del+ y / del b
			synchronized(this.gradb) {
				this.gradb.addi(errorTimesNonLinearDerivative);
			}
			
			//compute loss for every child
			/* backprop through structure: (y is this node's activation)
			 * calculate del loss / del y * del y / del child-output for 
			 * both branches. 
			 * childrenLoss = del y/del x = [ del y_i / del x_j ] = [g'(Wx+b)_i W_ij] */
			
			INDArray childrenLoss = Nd4j.zeros(shape[0], shape[1]);
			/*for(int col = 0; col < shape[1]; col++) {
				childrenLoss.putColumn(col, nonLinearDerivativeTranspose.mul(this.W.getColumn(col)));//Hadamhard product
				//childrenLoss = childrenLoss.addColumnVector(nonLinearDerivative).mul(this.W); 
			}*/
			
			IntStream.range(0, shape[1]).parallel().unordered()
					.forEach(col -> childrenLoss.putColumn(col, nonLinearDerivativeTranspose.mul(this.W.getColumn(col))));
			
			//del loss / del leftvector
			INDArray leftLoss = error.mmul(childrenLoss.getColumns(this.firstHalf)); 
			 //del loss / del rightvector
			INDArray rightLoss = error.mmul(childrenLoss.getColumns(this.secondHalf));
			
			LOG.debug("x is %s \n leftLoss %s \n rightLoss %s", x, leftLoss, rightLoss);
			
			//Gradient Clipping to prevent exploding gradients
			double leftNorm = leftLoss.norm2(1).getDouble(0);
			double rightNorm = rightLoss.norm2(1).getDouble(0);
			
			if(leftNorm > 100 || rightNorm > 100) {
				LOG.debug("How come the norm be so large Left Norm: %s Right Norm: %s", leftNorm, rightNorm);
			}
			
			if(Double.isInfinite(leftNorm) || Double.isInfinite(rightNorm)) {
				LOG.info("Left or right norm is infinite");
			}
			
			if(Double.isNaN(leftNorm) || Double.isNaN(rightNorm)) {
				LOG.info("x values %s", v);
				LOG.info("Left or right norm is NaN");
			}
			
			//Gradient clipping to prevent gradient explosion
			double threshold = 5.0;
			if(leftNorm > threshold) {
				leftLoss.divi(leftNorm).muli(threshold);
			}
			
			if(rightNorm > threshold) {
				rightLoss.divi(rightNorm).muli(threshold);
			}
			
			LOG.debug("{");
			this.backProp(left, leftLoss);
			LOG.debug("} {");
			this.backProp(right, rightLoss);
			LOG.debug("}");
		}
		else new IllegalStateException("Binarize the tree");
		
	}
	
	/** update parameters */
	public void updateParameters() {
		
		//Add regularization term
		this.gradW.addi(this.W.mul(this.regularizer));
		this.gradb.addi(this.b.mul(this.regularizer));
		
		//Clip the gradient
		double normGradW = this.gradW.normmaxNumber().doubleValue();
		double threshold = 1.0;
		if(normGradW > threshold) {
			this.gradW.muli(threshold/normGradW);
		}
		
		double normGradb = this.gradb.normmaxNumber().doubleValue();
		if(normGradb > threshold) {
			this.gradb.muli(threshold/normGradb);
		}
		
		LOG.debug("Recursive gradW norm %s, max %s, min %s", this.gradW.normmaxNumber().doubleValue(), 
				this.gradW.maxNumber().doubleValue(), this.gradW.minNumber().doubleValue());
		
		LOG.debug("Recursive gradb norm %s, max %s, min %s", this.gradb.normmaxNumber().doubleValue(), 
				this.gradb.maxNumber().doubleValue(), this.gradb.minNumber().doubleValue());
		LOG.debug("Recursive Learning Rate %s", this.learningRate.getLearningRate());
		
		//update AdaGrad running sum of square
		this.adaGradSumSquareGradW.addi(this.gradW.mul(this.gradW));
		this.adaGradSumSquareGradb.addi(this.gradb.mul(this.gradb));
		
		double initLearningRate = this.learningRate.getLearningRate();
		
		INDArray invertedLearningRateGradW = Nd4j.getExecutioner()
											.execAndReturn(new Sqrt(this.adaGradSumSquareGradW.dup()))
											.divi(initLearningRate);
		
		INDArray invertedLearningRateGradb = Nd4j.getExecutioner()
											 .execAndReturn(new Sqrt(this.adaGradSumSquareGradb.dup()))
											 .divi(initLearningRate);
		
		/*this.W.subi(this.gradW.mul(this.learningRate.getLearningRate()));
		this.b.subi(this.gradb.mul(this.learningRate.getLearningRate()));*/
		
		this.W.subi(this.gradW.div(invertedLearningRateGradW));
		this.b.subi(this.gradb.div(invertedLearningRateGradb));
	}
	
	public INDArray getW() {
		return this.W;
	}
	
	public INDArray getb() {
		return this.b;
	}
	

	public INDArray getGradW() {
		return this.gradW;
	}
	
	public INDArray getGradb() {
		return this.gradb;
	}
	
	public void setParam(INDArray W, INDArray b) {		
		
		if(W.shape().length != this.W.shape().length) {
			throw new IllegalStateException("W param must have same shape length as this.W");
		}
		
		if(this.W.size(0) != W.size(0) || this.W.size(1) != W.size(1)) {
			throw new IllegalStateException("W's dimensions dont match the current");
		}
		
		for(int i = 0; i < this.W.size(0); i++) {
			
			for(int j = 0; j < this.W.size(1); j++) {
				this.W.putScalar(new int[]{i, j}, W.getDouble(new int[]{i, j}));
			}
		}
		
		if(b.shape().length != this.b.shape().length) {
			throw new IllegalStateException("b param must have same shape length as this.b");
		}
		
		if(this.b.size(0) != b.size(0) || this.b.size(1) != b.size(1)) {
			throw new IllegalStateException("b's dimensions dont match the current");
		}
		
		for(int i = 0; i < this.b.size(0); i++) {
			
			for(int j = 0; j < this.b.size(1); j++) {
				this.b.putScalar(new int[]{i, j}, b.getDouble(new int[]{i, j}));
			}
		}
	}
	
	/** clears the gradients */
	public void flushGradients() {
		this.gradW.muli(0);
		this.gradb.muli(0);
	}

	@Override
	public int getDimension() {
		return this.n;
	}

	@Override
	public Object getEmbedding(Object obj) {
		//check if obj can be casted as a tree
		return this.getEmbedding((Tree)obj);
	}
}
