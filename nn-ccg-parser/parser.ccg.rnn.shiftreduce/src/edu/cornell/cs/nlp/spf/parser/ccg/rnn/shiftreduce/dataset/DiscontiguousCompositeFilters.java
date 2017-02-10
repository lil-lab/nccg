package edu.cornell.cs.nlp.spf.parser.ccg.rnn.shiftreduce.dataset;

import java.io.Serializable;
import java.util.Collections;
import java.util.List;
import java.util.function.Predicate;

import edu.cornell.cs.nlp.spf.mr.lambda.LogicalExpression;
import edu.cornell.cs.nlp.spf.parser.ParsingOp;

public class DiscontiguousCompositeFilters implements Serializable {
	
	private static final long serialVersionUID = -3602173286742885922L;
	private List<Predicate<ParsingOp<LogicalExpression>>> filters;
	
	public DiscontiguousCompositeFilters(List<Predicate<ParsingOp<LogicalExpression>>> filters) {
		this.filters = Collections.unmodifiableList(filters);
	}
	
	public List<Predicate<ParsingOp<LogicalExpression>>> getFilters() {
		return this.filters;
	}
}
