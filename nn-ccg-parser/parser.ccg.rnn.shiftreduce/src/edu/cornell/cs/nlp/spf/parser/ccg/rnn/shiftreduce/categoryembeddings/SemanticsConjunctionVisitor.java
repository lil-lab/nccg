package edu.cornell.cs.nlp.spf.parser.ccg.rnn.shiftreduce.categoryembeddings;

import java.util.HashSet;
import java.util.Set;

import edu.cornell.cs.nlp.spf.mr.lambda.Lambda;
import edu.cornell.cs.nlp.spf.mr.lambda.Literal;
import edu.cornell.cs.nlp.spf.mr.lambda.LogicalConstant;
import edu.cornell.cs.nlp.spf.mr.lambda.LogicalExpression;
import edu.cornell.cs.nlp.spf.mr.lambda.Variable;
import edu.cornell.cs.nlp.spf.mr.lambda.visitor.ILogicalExpressionVisitor;
import edu.uw.cs.lil.amr.lambda.OverloadedLogicalConstant;

/** Visitor for traversing a Logical Expression */
public class SemanticsConjunctionVisitor implements ILogicalExpressionVisitor {
	
	private final Set<LogicalConstant> updateBaseConstant;
	
	public SemanticsConjunctionVisitor() {
		this.updateBaseConstant = new HashSet<LogicalConstant>();
	}
	
	public static Set<LogicalConstant> embedSemantics(LogicalExpression exp) {
		
		if(exp == null) { //exp can be null such as in SKIP and PUNCT entries
			return new HashSet<LogicalConstant>();
		}
		
		SemanticsConjunctionVisitor sv = new SemanticsConjunctionVisitor();
		exp.accept(sv);
		
		return sv.updateBaseConstant;
	}
	
	public void visit(Lambda lambda) {
		/* Degree of freedom: argument and body. Argument is single variable
		 * Return: autoencode(embed[argument], embed[body]) */
		
		LogicalExpression body = lambda.getBody();
		//visit body and add its vector to the result
		body.accept(this); 
		
		Variable var = lambda.getArgument();
		//add embedding of the variable
		var.accept(this);
	}

	public void visit(Literal literal) {
		/* Degree of freedom: predicate, list of arguments */
		
		LogicalExpression pred = literal.getPredicate();
		pred.accept(this);
		int numArgs = literal.numArgs();
		
		if(numArgs == 0) { 
			return; 
		}
		
		for(int i=0; i<numArgs; i++) {
			LogicalExpression exp = literal.getArg(i);
			exp.accept(this);
		}
	}

	public void visit(LogicalConstant logicalConstant) {
		
		// In AMR, logical constants can be overloaded so we first strip them.
		logicalConstant = OverloadedLogicalConstant.getWrapped(logicalConstant);
		
		String base = logicalConstant.getBaseName();
		
		if(base.compareTo("and") == 0) {
			this.updateBaseConstant.add(logicalConstant);
		}
		
		if(base.startsWith("cop")) {
			String possibleDigit = base.substring(3, base.length());
			if(possibleDigit.matches("^-?\\d+$")) {
				this.updateBaseConstant.add(logicalConstant);
			}
		}
	}

	public void visit(Variable variable) {
	}
	
}