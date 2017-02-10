package edu.cornell.cs.nlp.spf.parser.ccg.rnn.shiftreduce.categoryembeddings;

import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Set;
import java.util.Stack;
import java.util.concurrent.atomic.AtomicBoolean;

import org.nd4j.linalg.api.ndarray.INDArray;

import edu.cornell.cs.nlp.spf.mr.lambda.Lambda;
import edu.cornell.cs.nlp.spf.mr.lambda.Literal;
import edu.cornell.cs.nlp.spf.mr.lambda.LogicalConstant;
import edu.cornell.cs.nlp.spf.mr.lambda.LogicalExpression;
import edu.cornell.cs.nlp.spf.mr.lambda.Variable;
import edu.cornell.cs.nlp.spf.mr.lambda.visitor.ILogicalExpressionVisitor;
import edu.cornell.cs.nlp.spf.mr.language.type.Type;
import edu.cornell.cs.nlp.spf.parser.ccg.rnn.shiftreduce.recursive.AveragingNetwork;
import edu.cornell.cs.nlp.spf.parser.ccg.rnn.shiftreduce.recursive.GradientWrapper;
import edu.cornell.cs.nlp.spf.parser.ccg.rnn.shiftreduce.recursive.RecursiveTreeNetwork;
import edu.cornell.cs.nlp.spf.parser.ccg.rnn.shiftreduce.recursive.Tree;
import edu.uw.cs.lil.amr.lambda.OverloadedLogicalConstant;

/** Visitor for traversing a Logical Expression */
public class SemanticsVisitor implements ILogicalExpressionVisitor {
	
	private Stack<Tree> result;
	private final RecursiveTreeNetwork rte;
	private HashMap<Type, INDArray> typeVectors;
	private HashMap<Type, GradientWrapper> typeVectorsGrad;
	private HashMap<String, INDArray> baseConstantVectors;
	private HashMap<String, GradientWrapper> baseConstantVectorsGrad;
	
	private final Set<Type> updatedType;
	private final Set<String> updateBaseConstant;
	
	public SemanticsVisitor(RecursiveTreeNetwork rte, 
									HashMap<String, INDArray> baseConstantVectors,
									HashMap<String, GradientWrapper> baseConstantVectorsGrad,
									HashMap<Type, INDArray> typeVectors, 
									HashMap<Type, GradientWrapper> typeVectorsGrad, 
									Set<Type> updatedType, Set<String> updatedBaseConstant) {
		this.rte = rte;
		this.typeVectors = typeVectors;
		this.typeVectorsGrad = typeVectorsGrad;
		this.baseConstantVectors = baseConstantVectors;
		this.baseConstantVectorsGrad = baseConstantVectorsGrad;
		
		this.updatedType = updatedType;
		this.updateBaseConstant = updatedBaseConstant;
		
		this.result = new Stack<Tree>();
	}
	
	public static Tree embedSemantics(LogicalExpression exp, RecursiveTreeNetwork rte,
									HashMap<String, INDArray> baseConstantVectors,
									HashMap<String, GradientWrapper> baseConstantVectorsGrad,
									HashMap<Type, INDArray> typeVectors, 
									HashMap<Type, GradientWrapper> typeVectorsGrad, INDArray nullLogic, 
									GradientWrapper nullLogicGrad, Set<Type> updatedType, Set<String> updatedBaseConstant, 
									AtomicBoolean updatedNullLogic, boolean useRecursive) {
		
		if(exp == null) { //exp can be null such as in SKIP and PUNCT entries
			Tree t = new Tree("NULL", new LinkedList<Tree>());
			t.setVector(nullLogic);
			t.setGradient(nullLogicGrad); //probably can do something about it. Like not set it.
			updatedNullLogic.set(true);
			return t;
		}
		
		SemanticsVisitor sv = new SemanticsVisitor(rte, baseConstantVectors, baseConstantVectorsGrad, 
				 								   typeVectors, typeVectorsGrad, updatedType, updatedBaseConstant);
		exp.accept(sv);
		assert sv.result.size() == 1;
		if(useRecursive) {
			sv.rte.feedForward(sv.result.peek());
		} else {
			AveragingNetwork.averageAndSet(sv.result.peek());
		}
		return sv.result.peek();
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
		
		//one has embedding of body and variable in the list
		assert this.result.size() >= 2;

		List<Tree> subtrees = new LinkedList<Tree>();
		subtrees.add(result.pop()); //tree of var
		subtrees.add(result.pop()); //tree of body
		Tree t = new Tree("Lambda", subtrees);
		
		this.result.push(t);
	}

	public void visit(Literal literal) {
		/* Degree of freedom: predicate, list of arguments */
		
		LogicalExpression pred = literal.getPredicate();
		pred.accept(this);
		int numArgs = literal.numArgs();
		
		if(numArgs == 0) { /*special case*/
			return; //we do not introduce internal nodes with one child so we make child as the parent
		}
		
		for(int i=0; i<numArgs; i++) {
			LogicalExpression exp = literal.getArg(i);
			exp.accept(this);
		}
		
		/* there must be numArgs+1 embedding in result right now
		 * encode them in right to left order. result contains:
		 * pred, arg1, arg2, ... argk which will be passed as, 
		 * argk, argk-1, ... arg1, pred*/
		
		assert this.result.size() >= numArgs + 1;
		
		List<Tree> children = new LinkedList<Tree>();
		Tree last = this.result.pop();
		Tree sLast = this.result.pop();
		int size = this.result.size();
		children.add(sLast);
		children.add(last); //right to left tree
		
		Tree current = new Tree("Literal", children);
		
		for(int i = size - 3; i >= size - numArgs - 1 ; i--) { //TO-Do optimize this
			children = new LinkedList<Tree>();
			children.add(this.result.pop());
			children.add(current);
			Tree t = new Tree("Literal", children); 
			current  = t;
		}
		
		this.result.push(current);

	}

	public void visit(LogicalConstant logicalConstant) {
		/* Degree of freedom: constant itself 
		 * Returns the embedding of this constant which is concatenation of
		 * embedding of the constant's base name and of its type */
		
		// In AMR, logical constants can be overloaded so we first strip them.
		logicalConstant = OverloadedLogicalConstant.getWrapped(logicalConstant);
		
		String base = logicalConstant.getBaseName();
		Type type = logicalConstant.getType();
		INDArray baseAr, typeAr;
		GradientWrapper gradBase, gradType;
		
		if(this.baseConstantVectors.containsKey(base)) {
			baseAr = this.baseConstantVectors.get(base);
			gradBase = this.baseConstantVectorsGrad.get(base);
			this.updateBaseConstant.add(base);
		} else { 
			baseAr = this.baseConstantVectors.get("$UNK$");
			gradBase = this.baseConstantVectorsGrad.get("$UNK$");
			this.updateBaseConstant.add("$UNK$");
		}
		
		if(this.typeVectors.containsKey(type)) {
			typeAr = this.typeVectors.get(type);
			gradType = this.typeVectorsGrad.get(type);
			this.updatedType.add(type);
		} else {
			typeAr = this.typeVectors.get(null);
			gradType = this.typeVectorsGrad.get(null);
			this.updatedType.add(null);
		}
		
		/*
		Tree t = new Tree("LogicalConstant", new LinkedList<Tree>());
		t.setVector(baseAr);//t.setVector(Nd4j.concat(1, baseAr, typeAr));
		
		t.setGradient(gradBase); 
		this.result.push(t);*/
		
		Tree typeSubTree = new Tree("Type", new LinkedList<Tree>());
		typeSubTree.setVector(typeAr);
		typeSubTree.setGradient(gradType); 
		
		Tree baseSubTree = new Tree("LogicalConstantBase", new LinkedList<Tree>());
		baseSubTree.setVector(baseAr);
		baseSubTree.setGradient(gradBase);
		
		List<Tree> subTrees = new LinkedList<Tree>();
		subTrees.add(baseSubTree);
		subTrees.add(typeSubTree);
		
		Tree t = new Tree("LogicalConstant", subTrees);
	
		this.result.push(t);	
	}

	public void visit(Variable variable) {
		/* Degree of freedom: variable itself 
		 * Returns the embedding of the type of the variable */
		
		Type type = variable.getType();
		
		Tree t = new Tree("Variable", new LinkedList<Tree>());
		if(this.typeVectors.containsKey(type)) {
			t.setVector(this.typeVectors.get(type));
			t.setGradient(this.typeVectorsGrad.get(type));
			this.updatedType.add(type);
		} else {
			t.setVector(this.typeVectors.get(null));
			t.setGradient(this.typeVectorsGrad.get(null));
			this.updatedType.add(null);
		}
		
		this.result.push(t);
	}
	
}