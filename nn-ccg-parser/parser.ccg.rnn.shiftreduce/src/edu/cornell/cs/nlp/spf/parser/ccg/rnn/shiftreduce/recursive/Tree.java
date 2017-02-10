package edu.cornell.cs.nlp.spf.parser.ccg.rnn.shiftreduce.recursive;

import java.io.Serializable;
import java.util.Iterator;
import java.util.List;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.indexing.NDArrayIndex;

import edu.cornell.cs.nlp.spf.parser.ccg.rnn.shiftreduce.learning.Helper;

public class Tree implements Serializable {

	private static final long serialVersionUID = -7653835747262100388L;
	
	/** Vector representing the embedding of the subtree rooted at this node.
	 * Must be a row-vector. */
	private INDArray vector;
	private INDArray preOutput; //before non-linearity is applied
	
	/** Gradient is null except for leaves. For leaves, one can supply a concatenation of 
	 * vectors in which case the vector is set to be concatenation of the individual vectors
	 * while one supplies an array of gradient. While performing updates, the gradient will 
	 * be split according to dimensions in the gradient and individual gradients will get the 
	 * error. Gradient array must be in left-to-right ordering representing the concatenation 
	 * of vectors. */
	private GradientWrapper gradient[]; //packs sum of gradient and number of terms
	private final String label;
	private final List<Tree> children;
	private final int numChild;
	
	public Tree(String label, List<Tree> children) {
		this.label = label;
		this.children = children;
		this.numChild = children.size();
	}
	
	public int numChildren() {
		return this.numChild;
	}
	
	public int numLeaves() {
		
		if(this.numChild == 0) {
			return 1;
		} else {
			int numLeaves = 0;
			for(Tree child: children) {
				numLeaves = numLeaves + child.numLeaves();
			}
			
			return numLeaves;
		}
	}
	
	public Tree getChild(int i) { 
		/* use iterator for trees with large degree */
		return this.children.get(i);
	}
	
	public Iterator<Tree> getChildren() {
		return this.children.iterator();
	}
	
	public INDArray getVector() {
		return this.vector;
	}
	
	public void setVector(INDArray vector) {
		this.vector = vector;
	}
	
	public void setGradient(GradientWrapper gradient) {
		this.gradient = new GradientWrapper[1];
		this.gradient[0] = gradient;
	}
	
	public void setGradient(GradientWrapper gradient1, GradientWrapper gradient2) {
		this.gradient = new GradientWrapper[2];
		this.gradient[0] = gradient1;
		this.gradient[1] = gradient2;
	}
	
	public void addGradient(INDArray gradient) {
		int pad = 0;
		for(int i = 0; i < this.gradient.length; i++) {
			GradientWrapper gw = this.gradient[i];
			INDArray gwError = gradient.get(NDArrayIndex.interval(pad, pad + gw.getDimension()));
			synchronized(gw) {
				gw.addGradient(gwError);
			}
			pad = pad + gw.getDimension();
		}
	}
	
	public String getLabel() {
		return this.label;
	}
	
	public INDArray getPreOutput() {
		return this.preOutput;
	}
	
	public void setPreOutput(INDArray preOutput) {
		this.preOutput = preOutput;
	}
	
	@Override
	public String toString() {
		StringBuilder sbuilder = new StringBuilder();
		this.stringIt(sbuilder);
		return sbuilder.toString();
	}
	
	public void stringIt(StringBuilder builder) {
		builder.append("Label "+this.label+", Num Children "+this.numChild+"\n");
		builder.append("Vector " + Helper.printVector(this.vector)+"\n");
		
		if(this.preOutput != null) {
			builder.append("Preoutput: "+Helper.printVector(this.preOutput)+"\n");
		} else {
			builder.append("preOutput is null\n");
		}
		Iterator<Tree> it = this.children.iterator();
		int child = 0;
		while(it.hasNext()) {
			builder.append("( child: "+(++child)+"\n");
			it.next().stringIt(builder);
			builder.append(")");
		}
	}
 }
