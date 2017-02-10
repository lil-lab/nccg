package edu.cornell.cs.nlp.spf.parser.ccg.rnn.shiftreduce.categoryembeddings;

import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map.Entry;
import java.util.Set;
import java.util.Stack;

import org.nd4j.linalg.api.ndarray.INDArray;
//import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4j;

import edu.cornell.cs.nlp.spf.ccg.categories.syntax.ComplexSyntax;
import edu.cornell.cs.nlp.spf.ccg.categories.syntax.Slash;
import edu.cornell.cs.nlp.spf.ccg.categories.syntax.Syntax;
import edu.cornell.cs.nlp.spf.ccg.categories.syntax.Syntax.SimpleSyntax;
import edu.cornell.cs.nlp.spf.parser.ccg.rnn.shiftreduce.recursive.AveragingNetwork;
import edu.cornell.cs.nlp.spf.parser.ccg.rnn.shiftreduce.recursive.GradientWrapper;
//import edu.cornell.cs.nlp.spf.parser.ccg.rnn.shiftreduce.embeddings.RecursiveTreeEmbedding;
import edu.cornell.cs.nlp.spf.parser.ccg.rnn.shiftreduce.recursive.RecursiveTreeNetwork;
import edu.cornell.cs.nlp.spf.parser.ccg.rnn.shiftreduce.recursive.Tree;
import edu.uw.cs.lil.amr.parser.rules.coordination.CoordinationSyntax;

/** Visitor for traversing syntactic category. Produces embedding of syntax. 
 * @author Dipendra Misra 
 * */ 
public class SyntaxsVisitor {
	
	private Stack<Tree> result;
	private final RecursiveTreeNetwork rte;
	private final HashMap<SimpleSyntax, INDArray> simpleSyntaxVectors;
	private final HashMap<SimpleSyntax, GradientWrapper> simpleSyntaxVectorsGrad;
	private final HashMap<String, INDArray> attributeVectors;
	private final HashMap<String, GradientWrapper> attributeVectorsGrad;
	private HashMap<Slash, INDArray> slashVectors;
	private HashMap<Slash, GradientWrapper> slashVectorsGrad;
	
	private final Set<String> updatedAttribute;
	private final Set<SimpleSyntax> updatedSimpleSyntax;
	private final Set<Slash> updatedSlash;
	
	public SyntaxsVisitor(RecursiveTreeNetwork rte, 
			HashMap<SimpleSyntax, INDArray> simpleSyntaxVectors, 
			HashMap<SimpleSyntax, GradientWrapper> simpleSyntaxVectorsGrad, 
			HashMap<String, INDArray> attributeVectors,
			HashMap<String, GradientWrapper> attributeVectorsGrad,
			HashMap<Slash, INDArray> slashVectors,
			HashMap<Slash, GradientWrapper> slashVectorsGrad, 
			Set<String> updatedAttribute, Set<SimpleSyntax> updatedSimpleSyntax,
			Set<Slash> updatedSlash) {
		this.rte = rte;
		this.simpleSyntaxVectors = simpleSyntaxVectors;
		this.simpleSyntaxVectorsGrad = simpleSyntaxVectorsGrad;
		this.attributeVectors = attributeVectors;
		this.attributeVectorsGrad = attributeVectorsGrad;
		this.slashVectors = slashVectors;
		this.slashVectorsGrad = slashVectorsGrad;

		this.updatedAttribute = updatedAttribute;
		this.updatedSimpleSyntax = updatedSimpleSyntax;
		this.updatedSlash = updatedSlash;
		
		this.result = new Stack<Tree>();
	}
	
	public static Tree embedSyntaxs(Syntax syntax, RecursiveTreeNetwork rte,
										HashMap<SimpleSyntax, INDArray> simpleSyntaxVectors, 
										HashMap<SimpleSyntax, GradientWrapper> simpleSyntaxVectorsGrad, 
										HashMap<String, INDArray> attributeVectors,
										HashMap<String, GradientWrapper> attributeVectorsGrad, 
										HashMap<Slash, INDArray> slashVectors, 
										HashMap<Slash, GradientWrapper> slashVectorsGrad, 
										Set<String> updatedAttribute, Set<SimpleSyntax> updatedSimpleSyntax,
										Set<Slash> updatedSlash, boolean useRecursive) {
		SyntaxsVisitor sv = new SyntaxsVisitor(rte, simpleSyntaxVectors, simpleSyntaxVectorsGrad,
						attributeVectors, attributeVectorsGrad, slashVectors, slashVectorsGrad, updatedAttribute,
						updatedSimpleSyntax, updatedSlash);
		
		if(syntax.numSlashes() > 0) //syntax instanceof ComplexSyntax)
			sv.visit((ComplexSyntax)syntax);
		else if(syntax.numSlashes() == 0) //syntax instanceof ComplexSyntax) 
			sv.visit((SimpleSyntax)syntax);
		else throw new RuntimeException("Syntax is neither simple nor complex ");
		
		assert sv.result.size() == 1;
		
		if(useRecursive) {
			sv.rte.feedForward(sv.result.peek());
		} else {
			AveragingNetwork.averageAndSet(sv.result.peek());
		}
		
		return sv.result.peek();
	}
	
	public void visit(ComplexSyntax syntax) {
		
		if(syntax.getLeft().numSlashes() > 0)// instanceof ComplexSyntax)
			this.visit((ComplexSyntax)syntax.getLeft());
		else if(syntax.getLeft().numSlashes() == 0)// instanceof ComplexSyntax) 
			this.visit((SimpleSyntax)syntax.getLeft());
		else throw new RuntimeException("Syntax is neither simple nor complex ");
		
		if(syntax.getRight().numSlashes() > 0)// instanceof ComplexSyntax)
			this.visit((ComplexSyntax)syntax.getRight());
		else if(syntax.getRight().numSlashes() == 0)// instanceof ComplexSyntax) 
			this.visit((SimpleSyntax)syntax.getRight());
		else throw new RuntimeException("Syntax is neither simple nor complex ");
		
		//embedding of the slash
		this.visit(syntax.getSlash());
		
		assert this.result.size() >= 3;
		
		List<Tree> children = new LinkedList<Tree>();
		children.add(this.result.pop()); //tree of right half
		children.add(this.result.pop()); //tree for slash
		Tree slashAndRight = new Tree("ComplexSyntax", children);
		
		children = new LinkedList<Tree>();
		children.add(this.result.pop());
		children.add(slashAndRight);
		Tree top = new Tree("CompleteSyntax", children);
		
		this.result.add(top);
	}
	
	/** returns the vector for a given simple syntactic category */
	public void visit(SimpleSyntax ssyntax) {
		
		if(ssyntax instanceof CoordinationSyntax) {
			this.visit((CoordinationSyntax)ssyntax);
			return;
		}
		
		Tree t = new Tree("SimpleSyntax "+ ssyntax.toString(), new LinkedList<Tree>());
		
		boolean found = false; 
		String label = ssyntax.getLabel();
		
		INDArray labelVector = null;
		GradientWrapper labelGrad = null;
		
		for(Entry<SimpleSyntax, INDArray> e: this.simpleSyntaxVectors.entrySet()) {
			if(e.getKey().getLabel().equals(label)) {
				for(Entry<SimpleSyntax, GradientWrapper> egrad: this.simpleSyntaxVectorsGrad.entrySet()) {
					if(egrad.getKey().getLabel().equals(label)) {
						found = true;
						labelVector = e.getValue();	
						labelGrad = egrad.getValue();
						this.updatedSimpleSyntax.add(egrad.getKey());
						break;
					}
				}
				if(found) {
					break;
				}
			}
		}
		
		if(labelVector == null || labelGrad == null || !found) {
			throw new RuntimeException("Simple syntax that is not present/null "+ssyntax.toString() + " found " + found);
		}
		
		INDArray attributeVector = this.attributeVectors.get(ssyntax.getAttribute());
		GradientWrapper attributeGrad = this.attributeVectorsGrad.get(ssyntax.getAttribute());
		
		if(attributeVector == null || attributeGrad == null) {
			throw new RuntimeException("Attribute not found/null. Attribute is: "+ssyntax.getAttribute());
		}
		
		this.updatedAttribute.add(ssyntax.getAttribute());
		
		INDArray concat = Nd4j.concat(1, labelVector, attributeVector);
		t.setVector(concat);
		t.setGradient(labelGrad, attributeGrad);
		
		this.result.push(t);
	}
	
	public void visit(CoordinationSyntax coordinationSyntax) {
		
		Syntax coordinatedSyntax = coordinationSyntax.getCoordinatedSyntax();
		
		this.visit(Syntax.C);
		Tree coordinationTree = this.result.pop();
		
		if(coordinatedSyntax.numSlashes() > 0) 
			this.visit((ComplexSyntax)coordinatedSyntax);
		else if(coordinatedSyntax.numSlashes() == 0) 
			this.visit((SimpleSyntax)coordinatedSyntax);
		else throw new RuntimeException("Syntax is neither simple nor complex ");
		
		Tree coordinatedSyntaxTree = this.result.pop();
		
		List<Tree> children = new LinkedList<Tree>();
		children.add(coordinatedSyntaxTree);
		children.add(coordinationTree);
		
		Tree t = new Tree("CoordinationSyntax", children);
		
		this.result.push(t);
	}
	
	/** returns the vector for a given slash */
	public void visit(Slash slash) {
		
		Tree t = new Tree("Slash", new LinkedList<Tree>());
		
		if(this.slashVectors.containsKey(slash)) {
			t.setVector(this.slashVectors.get(slash));
			t.setGradient(this.slashVectorsGrad.get(slash));
		} else {
			throw new RuntimeException("unknown slash");
		}
		
		this.updatedSlash.add(slash);
		this.result.push(t);
	}
}