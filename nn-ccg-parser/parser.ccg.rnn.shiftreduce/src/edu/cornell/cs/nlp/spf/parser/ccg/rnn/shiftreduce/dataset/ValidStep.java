package edu.cornell.cs.nlp.spf.parser.ccg.rnn.shiftreduce.dataset;

import java.io.Serializable;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;

import edu.cornell.cs.nlp.spf.ccg.categories.Category;
import edu.cornell.cs.nlp.spf.parser.ccg.rules.RuleName;
import edu.cornell.cs.nlp.utils.composites.Pair;
import edu.cornell.cs.nlp.utils.log.ILogger;
import edu.cornell.cs.nlp.utils.log.LoggerFactory;

public class ValidStep<MR> implements Serializable {
	
	public static final ILogger	LOG = LoggerFactory.create(ValidStep.class);
	
	private static final long serialVersionUID = 1580699213144546817L;

	/** for a given sentence span represents what are the valid rules that can
	 *  be applied and what categories do they derive. */
	private final List<Pair<RuleName, Category<MR>>> validRuleAndCategory;
	
	private boolean added;
	
	public ValidStep() {
		this.validRuleAndCategory = new LinkedList<Pair<RuleName, Category<MR>>>();
		this.added = false;
	}
	
	public void add(RuleName ruleName, Category<MR> categ) {
		//for a given span, we don't want to add more than one viterbi step that generated it.
		if(this.added) {
			LOG.warn("Refusing to add another step");
			return;
		}
		this.validRuleAndCategory.add(Pair.of(ruleName, categ));
		this.added = true;
	}
	
	public void add(RuleName baseRuleName, Category<MR> baseCateg, 
					RuleName unaryRuleName, Category<MR> unaryCateg) {
		//for a given span, we don't want to add more than one viterbi step that generated it.
		if(this.added) {
			LOG.warn("Refusing to add another step");
			return;
		}
		this.validRuleAndCategory.add(Pair.of(baseRuleName, baseCateg));
		this.validRuleAndCategory.add(Pair.of(unaryRuleName, unaryCateg));
		this.added = true;
	}
	
	public Iterator<Pair<RuleName, Category<MR>>> iterator() {
		return this.validRuleAndCategory.iterator();
	}
	
	public void check() {
		if(this.validRuleAndCategory.size() > 2) {
			throw new RuntimeException("Cannot contain more than 2 rules.");
		}
	}
	
	public String toString() {
		StringBuilder s = new StringBuilder();
		for(Pair<RuleName, Category<MR>> validRuleAndCategory: this.validRuleAndCategory) {
				s.append(validRuleAndCategory.first() + ": " + validRuleAndCategory.second() +", ");
		}
		return s.toString();
	}
}