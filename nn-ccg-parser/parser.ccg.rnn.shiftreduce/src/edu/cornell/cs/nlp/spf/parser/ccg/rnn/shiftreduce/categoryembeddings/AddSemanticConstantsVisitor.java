package edu.cornell.cs.nlp.spf.parser.ccg.rnn.shiftreduce.categoryembeddings;

import java.util.HashMap;
import java.util.Random;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import edu.cornell.cs.nlp.spf.mr.lambda.Lambda;
import edu.cornell.cs.nlp.spf.mr.lambda.Literal;
import edu.cornell.cs.nlp.spf.mr.lambda.LogicalConstant;
import edu.cornell.cs.nlp.spf.mr.lambda.LogicalExpression;
import edu.cornell.cs.nlp.spf.mr.lambda.Variable;
import edu.cornell.cs.nlp.spf.mr.lambda.visitor.ILogicalExpressionVisitor;
import edu.cornell.cs.nlp.spf.mr.language.type.Type;
import edu.cornell.cs.nlp.spf.parser.ccg.rnn.shiftreduce.recursive.GradientWrapper;
import edu.uw.cs.lil.amr.lambda.OverloadedLogicalConstant;

/** Visitor for traversing a Logical Expression */
public class AddSemanticConstantsVisitor implements ILogicalExpressionVisitor {
	
	private final int dim;
	private HashMap<Type, INDArray> typeVectors;
	private HashMap<Type, GradientWrapper> typeVectorsGrad;
	private HashMap<String, INDArray> baseConstantVectors;
	private HashMap<String, GradientWrapper> baseConstantVectorsGrad;
	private final Random rnd;
	private final int seed;
	
	/** Semantic embeddings are uniformly randomized in [-epsilon, epsilon]*/
	private final double epsilon; 
	
	public AddSemanticConstantsVisitor(int dim, HashMap<String, INDArray> baseConstantVectors,
									HashMap<String, GradientWrapper> baseConstantVectorsGrad,
									HashMap<Type, INDArray> typeVectors, 
									HashMap<Type, GradientWrapper> typeVectorsGrad, 
									Random rnd, int seed) {
		this.dim = dim;
		this.typeVectors = typeVectors;
		this.typeVectorsGrad = typeVectorsGrad;
		this.baseConstantVectors = baseConstantVectors;
		this.baseConstantVectorsGrad = baseConstantVectorsGrad;
		this.rnd = rnd;
		this.seed = seed;
		this.epsilon = 2*Math.sqrt(6.0/(double)(dim));
	}
	
	public static void addSemanticConstant(LogicalExpression exp, int dim, 
									HashMap<String, INDArray> baseConstantVectors,
									HashMap<String, GradientWrapper> baseConstantVectorsGrad,
									HashMap<Type, INDArray> typeVectors, 
									HashMap<Type, GradientWrapper> typeVectorsGrad, 
									Random rnd, int seed) {
		AddSemanticConstantsVisitor sv = new AddSemanticConstantsVisitor(dim, 
												baseConstantVectors, baseConstantVectorsGrad, 
				 								typeVectors, typeVectorsGrad, rnd, seed);
		if(exp == null) {
			return ;
		}
		exp.accept(sv);
		Type type = exp.getType();
		sv.addType(type);
	}
	
	public INDArray initUniformInitialization() {
		
		final INDArray vec;
		if(this.rnd == null) {
			vec = Nd4j.rand(new int[]{1, this.dim});	 
		} else {
			int localSeed = Math.abs(this.rnd.nextInt(2 * this.seed)) + 1;
			vec = Nd4j.rand(new int[]{1, this.dim}, localSeed);	
		}
		vec.subi(0.5).muli(this.epsilon);
		
		return vec;
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
		
		Type type = lambda.getType();
		this.addType(type);
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
		
		Type type = literal.getType();
		this.addType(type);
	}

	public void visit(LogicalConstant logicalConstant) {
		/* Degree of freedom: constant itself 
		 * Returns the embedding of this constant which is concatenation of
		 * embedding of the constant's base name and of its type */
		
		// In AMR, logical constants can be overloaded so we first strip them.
		logicalConstant = OverloadedLogicalConstant.getWrapped(logicalConstant);		
		
		String base = logicalConstant.getBaseName();
		Type type = logicalConstant.getType();
		
		this.addType(type);
		this.addBaseLogicalConstant(base);
	}

	public void visit(Variable variable) {
		Type type = variable.getType();
		this.addType(type);
	}
	
	public void addType(Type type) {
		
		if(type == null)
			return;
		if(this.typeVectors.containsKey(type))
			return;
		
		INDArray typeVector = initUniformInitialization();
		this.typeVectors.put(type, typeVector);
		this.typeVectorsGrad.put(type, new GradientWrapper(this.dim));
	}
	
	public void addBaseLogicalConstant(String base) {
		
		if(base == null)
			return;
		if(this.baseConstantVectors.containsKey(base))
			return;

		INDArray baseVector = initUniformInitialization();
		this.baseConstantVectors.put(base, baseVector);
		this.baseConstantVectorsGrad.put(base, new GradientWrapper(this.dim));
	}
	
}