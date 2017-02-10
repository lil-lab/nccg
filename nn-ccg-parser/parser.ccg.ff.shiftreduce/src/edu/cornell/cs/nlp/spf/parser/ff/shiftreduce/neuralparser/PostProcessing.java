package edu.cornell.cs.nlp.spf.parser.ff.shiftreduce.neuralparser;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;
import java.util.function.Predicate;

import edu.cornell.cs.nlp.spf.ccg.categories.Category;
import edu.cornell.cs.nlp.spf.ccg.categories.SimpleCategory;
import edu.cornell.cs.nlp.spf.ccg.categories.syntax.Syntax.SimpleSyntax;
import edu.cornell.cs.nlp.spf.mr.lambda.Lambda;
import edu.cornell.cs.nlp.spf.mr.lambda.Literal;
import edu.cornell.cs.nlp.spf.mr.lambda.LogicLanguageServices;
import edu.cornell.cs.nlp.spf.mr.lambda.LogicalConstant;
import edu.cornell.cs.nlp.spf.mr.lambda.LogicalExpression;
import edu.cornell.cs.nlp.spf.mr.lambda.Variable;
import edu.cornell.cs.nlp.spf.mr.language.type.ComplexType;
import edu.cornell.cs.nlp.spf.mr.language.type.Type;
import edu.cornell.cs.nlp.spf.parser.ParsingOp;
import edu.cornell.cs.nlp.spf.parser.ccg.rules.SentenceSpan;
import edu.cornell.cs.nlp.spf.parser.ccg.shiftreduce.ShiftReduceDerivation;
import edu.cornell.cs.nlp.spf.parser.ccg.shiftreduce.ShiftReduceParserOutput;
import edu.cornell.cs.nlp.spf.parser.ccg.shiftreduce.stacks.DerivationState;
import edu.cornell.cs.nlp.spf.parser.ccg.shiftreduce.stacks.DerivationStateHorizontalIterator;
import edu.cornell.cs.nlp.spf.parser.ccg.shiftreduce.stacks.ShiftReduceRuleNameSet;
import edu.cornell.cs.nlp.utils.composites.Pair;
import edu.cornell.cs.nlp.utils.log.ILogger;
import edu.cornell.cs.nlp.utils.log.LoggerFactory;
import edu.uw.cs.lil.amr.lambda.AMRServices;
import edu.uw.cs.lil.amr.parser.SloppyAmrClosure;
import edu.uw.cs.lil.amr.parser.rules.coordination.CoordinationServices;
import jersey.repackaged.com.google.common.collect.Lists;

public class PostProcessing<MR> implements Serializable {
	
	private static final long serialVersionUID = 8973743855004565461L;
	public static final ILogger	LOG = LoggerFactory.create(PostProcessing.class);
	
	/** penalty while stitching if closure returns null */
	private final double nullClosurePenalty;
	
	public PostProcessing(double nullClosurePenalty) {
		this.nullClosurePenalty = nullClosurePenalty;
	}
	
	public List<Pair<ShiftReduceRuleNameSet<MR>, SentenceSpan>> getStack(DerivationState<MR> dstate) {
		
		List<Pair<ShiftReduceRuleNameSet<MR>, SentenceSpan>> stack = 
				new ArrayList<Pair<ShiftReduceRuleNameSet<MR>, SentenceSpan>>();
		
		DerivationStateHorizontalIterator<MR> hit = dstate.horizontalIterator();
		boolean first = true;
		
		while(hit.hasNext()) {
			
			DerivationState<MR> state = hit.next();
			
			if(first) {
				first = false;
				if(state.getRightCategory() != null) {
					stack.add(Pair.of(state.getRightSet(), state.getRightSpan()));
				}	
			}
			
			if(state.getLeftCategory() != null) {
				stack.add(Pair.of(state.getLeftSet(), state.getLeftSpan()));
			}
		}
		
		////Debug
//		for(Pair<ShiftReduceRuleNameSet<MR>, SentenceSpan> l: stack) {
//			LOG.info("BestTree %s - %s", l.second().getStart(), l.second().getEnd());
//		}
		
		//return the reversed stack
		return Lists.reverse(stack);
	}
	
	private LogicalExpression makeLiteral(LogicalExpression[] arguments) {
		
		final LogicalConstant and = LogicLanguageServices.getConjunctionPredicate();			
		LogicalExpression[] newArguments = new LogicalExpression[arguments.length + 1];
		
		//Head predicate
		//Type trueType = and.getType().getRange(); //define it to true
		Type etType = LogicLanguageServices.getTypeRepository().getType("<e,t>");
		Type eType = LogicLanguageServices.getTypeRepository().getType("e");
		
		LogicalConstant headPredicate = LogicalConstant.create("and", etType, true); //changed head to and
		Variable var = new Variable(eType);
		LogicalExpression[] headArgs = new LogicalExpression[1];
		headArgs[0] = var;
		newArguments[0] = new Literal(headPredicate, headArgs);
		LOG.debug("Head predicate %s", newArguments[0]);
		
		//Children			
		Type argType = arguments[0].getType();
		Type copType = ComplexType.create("<" + eType + "," + argType + ">", eType, argType, null);
		LOG.debug("argType %s, copType %s", argType, copType);
		
		for(int i = 0; i < arguments.length; i++) {
			//convert arg to dummy(x, arg) where dummy is a fn of type 
			
			LogicalExpression cop = CoordinationServices.createpCOpPredicate(i + 1, argType);
			LogicalExpression[] args = new LogicalExpression[2];
			args[0] = var;
			args[1] = arguments[i];
				LogicalExpression exp = new Literal(cop, args);
				
				newArguments[i + 1] = exp;
			}
		
		LogicalExpression exp = new Literal(and, newArguments);
		Lambda lambda = new Lambda(var, exp);			
		LogicalExpression skolemize = AMRServices.skolemize(lambda);
		
		return skolemize;
	}
	
	public ShiftReduceParserOutput<MR> stitch6(Set<DerivationState<MR>> states, 
									Predicate<ParsingOp<MR>> pruningFilter, long parsingTime) {
		
		LOG.info("Stitch6: Best Tree on Stack: Considering %s states", states.size());
		
		//Find states with minimum number of trees
		int minTree = Integer.MAX_VALUE;
		List<DerivationState<MR>> minTreeStates = new ArrayList<DerivationState<MR>>();
		for(DerivationState<MR> state: states) {
			
			if(state.numTree < minTree) {
				minTreeStates.clear();
				minTree = state.numTree;
				minTreeStates.add(state);
			} else if(state.numTree == minTree) {
				minTreeStates.add(state);
			}
		} //can also think of it as A* score + heuristic i.e. state score + number of tree score
		
		LOG.info("Minimum number of tree %s, found %s states with these many trees ", minTree, minTreeStates.size());
		
		//Consider the top 10 highest ranking states with minimum number of trees
		List<DerivationState<MR>> allStates = new ArrayList<DerivationState<MR>>(minTreeStates);
		allStates.sort((s1, s2) -> Double.compare(s2.score, s1.score));
		allStates = allStates.subList(0, Math.min(100, allStates.size()));
		
		final List<ShiftReduceDerivation<MR>> derivations = new ArrayList<ShiftReduceDerivation<MR>>();
		
		for(DerivationState<MR> dstate: allStates) {
		
			final List<Pair<ShiftReduceRuleNameSet<MR>, SentenceSpan>> initStack = this.getStack(dstate);
			
			//Apply CLOSURE to every category
			final List<Pair<ShiftReduceRuleNameSet<MR>, SentenceSpan>> closedStack = new 
									ArrayList<Pair<ShiftReduceRuleNameSet<MR>, SentenceSpan>>();
			
			//Number of trees whose root cannot be closed
			int numNullCategory = 0;
			
			for(Pair<ShiftReduceRuleNameSet<MR>, SentenceSpan> tree: initStack) {
				
				Category<MR> categ = tree.first().getCategory();
				LOG.debug("Tree %s - %s category %s", tree.second().getStart(), 
									tree.second().getEnd(), tree.first().getCategory());
				
				if(!categ.getSyntax().equals(AMRServices.getCompleteSentenceSyntax())
						&& categ.getSemantics() != null) {
					LogicalExpression closedSemantics = 
								SloppyAmrClosure.of((LogicalExpression) categ.getSemantics());
										
					@SuppressWarnings("unchecked")
					Category<MR> closedCategory = (Category<MR>) 
							new SimpleCategory<LogicalExpression>(
									(SimpleSyntax) AMRServices.getCompleteSentenceSyntax(), closedSemantics);
					
					ShiftReduceRuleNameSet<MR> closedRuleNameSet =
							new ShiftReduceRuleNameSet<MR>(tree.first().getRuleName(0), closedCategory);
					closedStack.add(Pair.of(closedRuleNameSet, tree.second()));
				} else {
					closedStack.add(tree);
				}
			}
			
			for(Pair<ShiftReduceRuleNameSet<MR>, SentenceSpan> tree: closedStack) {
				LOG.debug("Closed Tree %s - %s category %s", tree.second().getStart(),
											tree.second().getEnd(), tree.first().getCategory());
			}
			
			List<LogicalExpression> arguments = new ArrayList<LogicalExpression>();
			for(Pair<ShiftReduceRuleNameSet<MR>, SentenceSpan> tree: closedStack) {
				if(tree.first().getCategory().getSemantics() != null) {
					
					LogicalExpression exp = (LogicalExpression) tree.first().getCategory().getSemantics();
					arguments.add(exp);
				} else {
					numNullCategory++;
				}
			}
			
			dstate.score = dstate.score - this.nullClosurePenalty * numNullCategory;
			
			LogicalExpression finalSemantics = this.makeLiteral(arguments.toArray(new LogicalExpression[arguments.size()]));
			
			@SuppressWarnings("unchecked")
			Category<MR> finalCategory = (Category<MR>)
					new SimpleCategory<LogicalExpression>((SimpleSyntax) AMRServices.getCompleteSentenceSyntax(), finalSemantics);
			
			ShiftReduceDerivation<MR> derivation = new ShiftReduceDerivation<MR>(dstate, finalCategory);
			derivations.add(derivation);
		}
		
		return new ShiftReduceParserOutput<MR>(derivations, parsingTime, true);
	}
}
