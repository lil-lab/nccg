package edu.cornell.cs.nlp.spf.parser.ff.shiftreduce.nonlocal.features;

import java.util.stream.Collectors;

import edu.cornell.cs.nlp.spf.base.hashvector.IHashVector;
import edu.cornell.cs.nlp.spf.ccg.categories.Category;
import edu.cornell.cs.nlp.spf.ccg.categories.syntax.Syntax;
import edu.cornell.cs.nlp.spf.parser.ccg.IParseStep;
import edu.cornell.cs.nlp.spf.parser.ccg.shiftreduce.stacks.DerivationState;
import edu.cornell.cs.nlp.spf.parser.ccg.shiftreduce.steps.IWeightedShiftReduceStep;
import edu.cornell.cs.nlp.spf.parser.ccg.shiftreduce.steps.ShiftReduceLexicalStep;
import edu.cornell.cs.nlp.utils.composites.Pair;
import edu.cornell.cs.nlp.utils.log.ILogger;
import edu.cornell.cs.nlp.utils.log.LoggerFactory;

/** Triggers on the children of the root of last tree on the stack */
public class SyntaxChildrenFeature<MR> implements AbstractNonLocalFeature<MR> {

	public static final ILogger								LOG
								= LoggerFactory.create(SyntaxChildrenFeature.class);

	private static final long serialVersionUID = -6611412639911325842L;
	private static final String TAG_LEFT_SYNTAX = "SYNCHILDL";
	private static final String TAG_RIGHT_SYNTAX = "SYNCHILDR";
	private static final String TAG_UNARY_SYNTAX = "SYNCHILDU";
	private static final String TAG_LEFT_ATTRIB = "SYNATTRIBCHILDL";
	private static final String TAG_RIGHT_ATTRIB = "SYNATTRIBCHILDR";
	private static final String TAG_UNARY_ATTRIB = "SYNATTRIBCHILDU";
	private static final String None = "None";
	
	
	private Pair<String, String> getStrippedSyntaxAndAttrib(Syntax syntax) {
		
		final String strippedSyntax = syntax.stripAttributes().toString();
		final String attribute = syntax.getAttributes().stream().sorted()
									.collect(Collectors.joining("+"));
		
		return Pair.of(strippedSyntax, attribute);
	}

	/** Triggers on the syntax of left and right children of the root of first tree on the stack. */
	@Override
	public void add(DerivationState<MR> state, IParseStep<MR> parseStep, IHashVector features, String[] buffer,
			int bufferIndex, String[] tags) {
		
		Category<MR> left = state.getLeftCategory();
		Category<MR> right = state.getRightCategory();
		
		final DerivationState<MR> parent = state.getParent();
	
		IWeightedShiftReduceStep<MR> weightedStep = state.returnStep();
		
		if(weightedStep == null) {
			
			features.set(TAG_LEFT_SYNTAX, None, 1.0);
			features.set(TAG_RIGHT_SYNTAX, None, 1.0);
			features.set(TAG_UNARY_SYNTAX, None, 1.0);
			features.set(TAG_LEFT_ATTRIB, None, 1.0);
			features.set(TAG_RIGHT_ATTRIB, None, 1.0);
			features.set(TAG_UNARY_ATTRIB, None, 1.0);
		
			return;
		}
		
		IParseStep<MR> step = weightedStep.getUnderlyingParseStep();
		
		if(left != null) {
			if(step.getRuleName().equals(ShiftReduceLexicalStep.LEXICAL_DERIVATION_STEP_RULENAME)) {
				
				features.set(TAG_LEFT_SYNTAX, None, 1.0);
				features.set(TAG_RIGHT_SYNTAX, None, 1.0);
				features.set(TAG_UNARY_SYNTAX, None, 1.0);
				features.set(TAG_LEFT_ATTRIB, None, 1.0);
				features.set(TAG_RIGHT_ATTRIB, None, 1.0);
				features.set(TAG_UNARY_ATTRIB, None, 1.0);
				
			} else if(step.numChildren() == 1) { //unary rule
				
				final Category<MR> unaryCategory;
				if(right != null) {
					unaryCategory = parent.getRightCategory();
				} else {
					unaryCategory = parent.getLeftCategory();
				}
				
				Pair<String, String> result = this.getStrippedSyntaxAndAttrib(unaryCategory.getSyntax());
				
				features.set(TAG_LEFT_SYNTAX, None, 1.0);
				features.set(TAG_RIGHT_SYNTAX, None, 1.0);
				features.set(TAG_UNARY_SYNTAX, result.first(), 1.0);
				features.set(TAG_LEFT_ATTRIB, None, 1.0);
				features.set(TAG_RIGHT_ATTRIB, None, 1.0);
				features.set(TAG_UNARY_ATTRIB, result.second(), 1.0);
				
			} else { //binary rule
				Category<MR> leftChild = parent.getLeftCategory();
				Category<MR> rightChild = parent.getRightCategory();
				Pair<String, String> leftResult = this.getStrippedSyntaxAndAttrib(leftChild.getSyntax());
				Pair<String, String> rightResult = this.getStrippedSyntaxAndAttrib(rightChild.getSyntax());
				
				features.set(TAG_LEFT_SYNTAX, leftResult.first(), 1.0);
				features.set(TAG_RIGHT_SYNTAX, rightResult.first(), 1.0);
				features.set(TAG_UNARY_SYNTAX, None, 1.0);
				features.set(TAG_LEFT_ATTRIB, leftResult.second(), 1.0);
				features.set(TAG_RIGHT_ATTRIB, rightResult.second(), 1.0);
				features.set(TAG_UNARY_ATTRIB, None, 1.0);
			}
		} else {
			
			features.set(TAG_LEFT_SYNTAX, None, 1.0);
			features.set(TAG_RIGHT_SYNTAX, None, 1.0);
			features.set(TAG_UNARY_SYNTAX, None, 1.0);
			features.set(TAG_LEFT_ATTRIB, None, 1.0);
			features.set(TAG_RIGHT_ATTRIB, None, 1.0);
			features.set(TAG_UNARY_ATTRIB, None, 1.0);
		}
	}
}
