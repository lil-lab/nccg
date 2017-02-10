package edu.cornell.cs.nlp.spf.parser.ccg.shiftreduce.stacks;

import java.io.Serializable;

import edu.cornell.cs.nlp.spf.ccg.categories.Category;
import edu.cornell.cs.nlp.spf.parser.ccg.rules.IArrayRuleNameSet;
import edu.cornell.cs.nlp.spf.parser.ccg.rules.RuleName;

/**
 * Pack states with the same number of words consumed, same spans for each leave and same
 * number of tree segments and same root categories for all the tree segments
 * 
 * @author Dipendra K. Misra
 */
public class ShiftReduceRuleNameSet<MR> implements IArrayRuleNameSet, Serializable {

	private static final long serialVersionUID = 1205312353717117303L;
	
	private final RuleName ruleName;
	private final Category<MR> category;
	
	public ShiftReduceRuleNameSet(RuleName ruleName, Category<MR> category)
	{
		this.ruleName = ruleName;
		this.category = category;
		assert this.ruleName != null : "Category was " + category;
	}
	
	public Category<MR> getCategory() {
		return this.category;
	}
	
	@Override
	public RuleName getRuleName(int index) {
		assert index == 0 && this.ruleName != null;
		return this.ruleName;
	}

	@Override
	public int numRuleNames() {
		assert this.ruleName != null;
		return 1;
	}

}
