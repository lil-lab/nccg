package edu.cornell.cs.nlp.spf.parser.ccg.rnn.shiftreduce.categoryembeddings;

import java.io.Serializable;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Set;
import java.util.Stack;
import java.util.concurrent.atomic.AtomicBoolean;

import org.nd4j.linalg.api.ndarray.INDArray;

import edu.cornell.cs.nlp.spf.mr.lambda.Lambda;
import edu.cornell.cs.nlp.spf.mr.lambda.Literal;
import edu.cornell.cs.nlp.spf.mr.lambda.LogicLanguageServices;
import edu.cornell.cs.nlp.spf.mr.lambda.LogicalConstant;
import edu.cornell.cs.nlp.spf.mr.lambda.LogicalExpression;
import edu.cornell.cs.nlp.spf.mr.lambda.Variable;
import edu.cornell.cs.nlp.spf.mr.lambda.visitor.ILogicalExpressionVisitor;
import edu.cornell.cs.nlp.spf.mr.language.type.Type;
import edu.cornell.cs.nlp.spf.parser.ccg.rnn.shiftreduce.recursive.AveragingNetwork;
import edu.cornell.cs.nlp.spf.parser.ccg.rnn.shiftreduce.recursive.GradientWrapper;
import edu.cornell.cs.nlp.spf.parser.ccg.rnn.shiftreduce.recursive.RecursiveTreeNetwork;
import edu.cornell.cs.nlp.spf.parser.ccg.rnn.shiftreduce.recursive.Tree;
import edu.cornell.cs.nlp.utils.log.ILogger;
import edu.cornell.cs.nlp.utils.log.LoggerFactory;
import edu.uw.cs.lil.amr.lambda.AMRServices;
import edu.uw.cs.lil.amr.lambda.OverloadedLogicalConstant;

/** Visitor for traversing a Logical Expression that considers only
 * relevant information */
public class ShallowSemanticsVisitor implements ILogicalExpressionVisitor, Serializable {
	
	private static final long serialVersionUID = -1516231301223361877L;
	public static final ILogger	LOG = LoggerFactory.create(ShallowSemanticsVisitor.class);
	
	private Stack<Tree> result;
	private final RecursiveTreeNetwork rte;
	private HashMap<Type, INDArray> typeVectors;
	private HashMap<Type, GradientWrapper> typeVectorsGrad;
	private HashMap<String, INDArray> baseConstantVectors;
	private HashMap<String, GradientWrapper> baseConstantVectorsGrad;
	
	private final Set<Type> updatedType;
	private final Set<String> updateBaseConstant;
	
	private final INDArray nullLogic;
	private final GradientWrapper nullLogicGrad;
	
	public ShallowSemanticsVisitor(RecursiveTreeNetwork rte, 
									HashMap<String, INDArray> baseConstantVectors,
									HashMap<String, GradientWrapper> baseConstantVectorsGrad,
									HashMap<Type, INDArray> typeVectors, 
									HashMap<Type, GradientWrapper> typeVectorsGrad, 
									Set<Type> updatedType, Set<String> updatedBaseConstant, 
									INDArray nullLogic, GradientWrapper nullLogicGrad) {
		this.rte = rte;
		this.typeVectors = typeVectors;
		this.typeVectorsGrad = typeVectorsGrad;
		this.baseConstantVectors = baseConstantVectors;
		this.baseConstantVectorsGrad = baseConstantVectorsGrad;
		
		this.updatedType = updatedType;
		this.updateBaseConstant = updatedBaseConstant;
		
		this.nullLogic = nullLogic;
		this.nullLogicGrad = nullLogicGrad;
		
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
		
		ShallowSemanticsVisitor sv = new ShallowSemanticsVisitor(rte, baseConstantVectors, baseConstantVectorsGrad, 
				 								   typeVectors, typeVectorsGrad, updatedType, updatedBaseConstant, nullLogic, nullLogicGrad);
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
		 * Return: embed[body] */
		
		LogicalExpression body = lambda.getBody();
		
		if(body instanceof Variable) {
			Tree nullTree = new Tree("null", new LinkedList<Tree>());
			nullTree.setVector(this.nullLogic);
			nullTree.setGradient(this.nullLogicGrad);

			this.result.push(nullTree);
			return;
		}
		
		body.accept(this); 
	}
	
	/** In shallow semantics, we ignore several structures
	 * importantly we ignore skolem constant, variables and conjunction */
	private boolean isSafe(LogicalExpression exp) {
		
		if(AMRServices.isSkolemPredicate(exp) || exp.equals(LogicLanguageServices.getConjunctionPredicate())) {
			return false;
		}
		
		if(exp instanceof Variable) {
			return false;
		}
		
		return true;
	}

	public void visit(Literal literal) {
		/* Degree of freedom: predicate, list of arguments
		 * we create a list of arguments that do not contain variables, 
		 * skolem predicate or skolem id.*/
		
		LogicalExpression pred = literal.getPredicate();
		int numArgs = 0;
		
		if(this.isSafe(pred)) {
			pred.accept(this);
			numArgs++;
		}
		
		final int literalArg = literal.numArgs();
		
		for(int i = 0; i < literalArg; i++) {
			if(i == 0 && AMRServices.isSkolemPredicate(pred)) { //skolem id
				continue;
			}
			
			LogicalExpression exp = literal.getArg(i);
			if(this.isSafe(exp)) {
				exp.accept(this);
				numArgs++;
			}
		}
		
		//if number of items pushed are 1 or less then return
		if(numArgs == 0) {
			Tree nullTree = new Tree("null", new LinkedList<Tree>());
			nullTree.setVector(this.nullLogic);
			nullTree.setGradient(this.nullLogicGrad);

			this.result.push(nullTree);
			return;
		}
		
		if(numArgs == 1) {
			return;
		}
		
		/* there must be numArgs embedding in result right now
		 * encode them in right to left order. result contains:
		 * pred, arg1, arg2, ... argk which will be passed as, 
		 * argk, argk-1, ... arg1*/
		
		assert this.result.size() >= numArgs;
		
		List<Tree> children = new LinkedList<Tree>();
		Tree last = this.result.pop();
		Tree sLast = this.result.pop();
		int size = this.result.size();
		children.add(sLast);
		children.add(last); //right to left tree
		
		Tree current = new Tree("Literal", children);
		
		for(int i = size - 3; i >= size - numArgs; i--) { //TO-Do optimize this
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
		
		throw new RuntimeException("Variable's are not being embedded. Should not have reached here but handled by its parent");
	}
	
}