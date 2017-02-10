package edu.cornell.cs.nlp.spf.parser.ccg.rnn.shiftreduce.dataset;

import java.util.function.Predicate;

import edu.cornell.cs.nlp.spf.mr.lambda.LogicalExpression;
import edu.cornell.cs.nlp.spf.parser.ParsingOp;
import edu.cornell.cs.nlp.spf.parser.ccg.cky.chart.Chart;
import edu.uw.cs.lil.amr.data.LabeledAmrSentence;

public interface AbstractAMREarlyUpdateFilterFactory {
	
	public Predicate<ParsingOp<LogicalExpression>> createFilter(Chart<LogicalExpression> chart,
																LabeledAmrSentence dataItem);

}
