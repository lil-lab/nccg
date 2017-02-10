package edu.cornell.cs.nlp.spf.parser.ccg.shiftreduce.stacks;

import java.io.Serializable;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;

import org.nd4j.linalg.api.ndarray.INDArray;

/** This class defines a persistent tree data-structure used by RNNs
 * for embedding the derivation state. In brief, for a given dstate, you can 
 * use its PersistentDerivationStateEmbeddings to travel and find the embeddings
 * of the derivation states from which it is derived. */
public class PersistentEmbeddings implements Serializable {

	private static final long serialVersionUID = -399928719598823562L;
	
	/** contains activations for every layer. Each element in the list corresponds to a layer. */
	private final Map<String, INDArray>[] rnnState; 
	private final PersistentEmbeddings parent;
	/** children can be keep on expanding hence not final */
	private List<PersistentEmbeddings> children;

	public PersistentEmbeddings(Map<String, INDArray>[] rnnState, PersistentEmbeddings parent) {
		this.rnnState = rnnState;
		this.parent = parent;
		this.children = new LinkedList<PersistentEmbeddings>();
	}
	
	public Map<String, INDArray>[] getRNNState() {
		return this.rnnState; 
	}
	
	public void addChild(PersistentEmbeddings child) {
		this.children.add(child);
	}
	
	public PersistentEmbeddings getParent() {
		return this.parent;
	}
}
