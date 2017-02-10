package edu.cornell.cs.nlp.spf.parser.ccg.rnn.shiftreduce.learning;

import org.nd4j.linalg.api.ndarray.INDArray;

import edu.cornell.cs.nlp.spf.ccg.categories.Category;

public class ParsingOpPreTrainingDataset<MR> {

	private final Category<MR> categ1;
	private final Category<MR> categ2;
	/** one hot vector encoding the rule*/
	private final INDArray rule;
	/** label which is true when the rule can be applied on the categories
	 *  and false when it cannot be. */
	private final boolean label;
	
	public ParsingOpPreTrainingDataset(Category<MR> categ1, Category<MR> categ2,  
									   INDArray rule, boolean label) {
		this.categ1 = categ1;
		this.categ2 = categ2;
		this.rule = rule;
		this.label = label;
	}
	
	public Category<MR> getCategory1() {
		return this.categ1;
	}
	
	public Category<MR> getCategory2() {
		return this.categ2;
	}
	
	public INDArray getRule() {
		return this.rule;
	}
	
	public boolean getLabel() {
		return this.label;
	}
	
}
