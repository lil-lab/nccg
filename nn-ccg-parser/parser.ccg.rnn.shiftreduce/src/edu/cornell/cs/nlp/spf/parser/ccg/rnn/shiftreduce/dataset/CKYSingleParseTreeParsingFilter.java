package edu.cornell.cs.nlp.spf.parser.ccg.rnn.shiftreduce.dataset;

import java.io.Serializable;
import java.util.Iterator;
import java.util.LinkedHashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.function.Predicate;

import edu.cornell.cs.nlp.spf.ccg.categories.Category;
import edu.cornell.cs.nlp.spf.parser.ParsingOp;
import edu.cornell.cs.nlp.spf.parser.ccg.IOverloadedParseStep;
import edu.cornell.cs.nlp.spf.parser.ccg.IParseStep;
import edu.cornell.cs.nlp.spf.parser.ccg.cky.CKYDerivation;
import edu.cornell.cs.nlp.spf.parser.ccg.cky.chart.Cell;
import edu.cornell.cs.nlp.spf.parser.ccg.cky.steps.IWeightedCKYStep;
import edu.cornell.cs.nlp.spf.parser.ccg.cky.steps.WeightedCKYLexicalStep;
import edu.cornell.cs.nlp.spf.parser.ccg.cky.steps.WeightedCKYParseStep;
import edu.cornell.cs.nlp.spf.parser.ccg.rnn.shiftreduce.dataset.ValidStep;
import edu.cornell.cs.nlp.spf.parser.ccg.rules.OverloadedRuleName;
import edu.cornell.cs.nlp.spf.parser.ccg.rules.RuleName;
import edu.cornell.cs.nlp.spf.parser.ccg.rules.SentenceSpan;
import edu.cornell.cs.nlp.spf.parser.ccg.rules.UnaryRuleName;
import edu.cornell.cs.nlp.utils.composites.Pair;
import edu.cornell.cs.nlp.utils.log.ILogger;
import edu.cornell.cs.nlp.utils.log.LoggerFactory;

public class CKYSingleParseTreeParsingFilter<MR> implements Predicate<ParsingOp<MR>>, Serializable {
	
	public static final ILogger	LOG = LoggerFactory.create(CKYSingleParseTreeParsingFilter.class);
	
	private static final long serialVersionUID = -8733388090384651246L;

	/** Derivation category */
	private final Category<MR> category;
	
	/** supervised pruning filter that removes parsing operations that won't lead to 
	 * the gold labeled logical form*/
	//private final Predicate<ParsingOp<MR>> supervisedPruningFilter;
	
	/** Valid step for a given span */
	private final ValidStep<MR>[][] validStep;
	
	/** left to right ordering of parsing steps */
	private final List<ParsingOp<MR>> orderedParsingOp;
	
	/** cursor which tells which ordered parsing op has been reached */
	private int cursor;
	
	/** number of tokens in the sentence */
	private final int n;
	
	@SuppressWarnings("unchecked")
	public CKYSingleParseTreeParsingFilter(CKYDerivation<MR> bestDerivation,
							Predicate<ParsingOp<MR>> supervisedPruningFilter, int n) {
		if(bestDerivation == null)  {
			this.category = null;
		} else {
			this.category = bestDerivation.getCategory();
		}
		
		this.n = n;
		
		//find valid rules
		this.validStep = new ValidStep[n][n];
		
		for(int start = 0; start < n; start++) {
			for(int end = start; end < n; end++) {
				this.validStep[start][end] = new ValidStep<MR>();
			}
		}
		
		//check if the shifted category exists in the chart
		Cell<MR> bestDerivationCell = bestDerivation.getCell();
		LOG.info("Number of packed parse trees %s", this.numPackedParseTrees(bestDerivationCell));
		this.findParseTreeRecursively/*findValidStep*/(bestDerivationCell);
		this.printValidStep();
		
		for(int start = 0; start < n; start++) {
			for(int end = start; end < n; end++) {
				this.validStep[start][end].check();
			}
		}
		
		//order valid rules from left to right
		this.orderedParsingOp = new LinkedList<ParsingOp<MR>>();
		this.cursor = 0;
		
		for(int i = 0; i < n; i++) {
			for(int j = i; j >= 0; j--) {
				SentenceSpan span = new SentenceSpan(j, i + 1, n); //span in SR notation
				
				Iterator<Pair<RuleName, Category<MR>>> it = this.validStep[j][i].iterator();
				//there can be atmost two rules in one span of which one has to be unary
				Pair<RuleName, Category<MR>> rule1 = null, rule2 = null; 
				
				if(it.hasNext()) {
					rule1 = it.next();
					if(it.hasNext()) {
						rule2 = it.next();
					}
				}
				
				if(rule1 == null) {
					continue;
				} else if(rule2 == null) {
					this.orderedParsingOp.add(new ParsingOp<MR>(rule1.second(), span, rule1.first()));
				} else { //both rules are not null. Add unary at the end
					//rule1 is base and rule2 is unary that is overloads base rule
					this.orderedParsingOp.add(new ParsingOp<MR>(rule1.second(), span, rule1.first()));
					this.orderedParsingOp.add(new ParsingOp<MR>(rule2.second(), span, rule2.first()));
				}
			}
		}
		
		LOG.info("Num rules %s", this.orderedParsingOp.size());
		this.printLeftToRightParsingOps();
	}
	
	public Category<MR> getCategory() {
		return this.category;
	}

	/** This function only adds steps corresponding to one single parse tree
	 * to the valid step array*/
	private void findParseTreeRecursively(Cell<MR> finalCell) {
		
		if(finalCell.getCategory() == null)
			throw new RuntimeException("Null category");
		
		//take on step
		List<Cell<MR>> buffer = new LinkedList<Cell<MR>>();
		buffer.add(finalCell);
		
		while(!buffer.isEmpty()) {
			Cell<MR> cell = buffer.remove(0);
			List<IWeightedCKYStep<MR>> viterbiSteps = cell.getViterbiSteps();
			
			//arbitrarily take one parsing step and add it
			IWeightedCKYStep<MR> step = viterbiSteps.get(0);
			this.addStep(step);
			
			//get the child cell of this step
			for(int i = 0; i < step.numChildren(); i++) {
				buffer.add(step.getChildCell(i));
			}
		}
	}
	
	/** Finds number of parse trees that are packed in a cell */
	private int numPackedParseTrees(Cell<MR> cell) {
		
		if(cell.getCategory() == null)
			throw new RuntimeException("Null category");
		
		int numParseTrees =  0;
		List<IWeightedCKYStep<MR>> viterbiSteps = cell.getViterbiSteps();
		
		for(IWeightedCKYStep<MR> step: viterbiSteps) {
			
			int numTrees = 1;
			for(int i = 0; i < step.numChildren(); i++) {
				numTrees = numTrees * this.numPackedParseTrees(step.getChildCell(i));
			}
			
			numParseTrees = numParseTrees + numTrees;
		}
		
		return numParseTrees;
	}
	 
	/** This function adds all the max scoring steps in the cell. These represents
	 * many parse trees.*/
	@SuppressWarnings("unused")
	private void findValidStep(Cell<MR> cell) {
		
		if(cell.getCategory() == null)
			throw new RuntimeException("Null category");			
		
		LinkedHashSet<IWeightedCKYStep<MR>> viterbiSteps = cell.getMaxSteps();
		
		for(IWeightedCKYStep<MR> viterbiStep: viterbiSteps) {
			this.addStep(viterbiStep);
		}
	}
	
	private void addStep(IWeightedCKYStep<MR> step) {
		
		RuleName ruleName = step.getRuleName();
		if(step.getRuleName() == null)
			throw new RuntimeException("Null rule name");
		
		if (ruleName instanceof OverloadedRuleName) {
			//Special case: when unary rules are overloaded with lexical or binary rules
			
			IWeightedCKYStep<MR> weightedParseStep = (IWeightedCKYStep<MR>)step;
			final IParseStep<MR> parseStep;
			
			if(weightedParseStep instanceof WeightedCKYLexicalStep) {
				parseStep = ((WeightedCKYLexicalStep<MR>)weightedParseStep).getStep();
			} else if(weightedParseStep instanceof WeightedCKYParseStep) {
				parseStep = ((WeightedCKYParseStep<MR>)weightedParseStep).getStep();
			} else {
				throw new RuntimeException("CKY step that is neither lexical step nor parse step.");
			}
			
			IOverloadedParseStep<MR> overloadedStep = (IOverloadedParseStep<MR>)parseStep;
			
			UnaryRuleName unaryRule = ((OverloadedRuleName) ruleName).getUnaryRule();
			RuleName baseRule = ((OverloadedRuleName) ruleName).getOverloadedRuleName();
			
			LOG.info("Found overloading. Rules are %s and %s", baseRule, unaryRule);
			
			int start = step.getStart();
			int end = step.getEnd();
		
			//Add the base rule before, this information is important. 
			//To prevent users from making mistake, use functions next time
			this.validStep[start][end].add(baseRule, overloadedStep.getIntermediate(),  
										   unaryRule, step.getRoot());
		} else {
		
			int start = step.getStart();
			int end = step.getEnd();
		
			this.validStep[start][end].add(step.getRuleName(), step.getRoot());
		}
	}
	
	private void printValidStep() {
		
		for(int start = 0; start < this.n; start ++ ) {
			for(int end = start; end < this.n; end++) {
				Iterator<Pair<RuleName, Category<MR>>> it = this.validStep[start][end].iterator();
				while(it.hasNext()) {
					Pair<RuleName, Category<MR>> step = it.next();
					LOG.info("[%s - %s RuleName: %s, Category: %s]", start, end, step.first(), step.second());
				}
			}
		}
	}
	
	private void printLeftToRightParsingOps() {
		
		Iterator<ParsingOp<MR>> it = this.orderedParsingOp.iterator();
		int count = 1;
		while(it.hasNext()) {
			LOG.info("Rule %s,  %s", count++, it.next());
		}
	}
	
	@Override
	public boolean test(ParsingOp<MR> op) {
		
		if(this.cursor >= this.orderedParsingOp.size()) {
			return false;
		}
	
		ParsingOp<MR> gTruth = this.orderedParsingOp.get(this.cursor);
		
		if(gTruth.getRule().equals(op.getRule()) 
			&& gTruth.getCategory().equals(op.getCategory())
			&& gTruth.getSpan().equals(op.getSpan())) {
			return true;
		}
		
		return false;
	}
	
	public void incrementCursor() {
		this.cursor++;
	}
	
	public void clearCursor() {
		this.cursor = 0;
	}
	
	//@Override
	public boolean test1(ParsingOp<MR> op) {
		
		/* if it fails using supervised filter then reject it.
		 * out of the supervised pruning filter and cky chart filter. 
		 * the cheaper to compute filter should come before the other. */
//		if(!supervisedPruningFilter.test(op))
//			return false;
		
		RuleName ruleName = op.getRule();
		Category<MR> opCategory = op.getCategory();
		int start = op.getSpan().getStart();
		int end = op.getSpan().getEnd() - 1;

		/* WARNING: CKY and SR use different notion of span. One does [start, end) and other does 
		 * [start, end]. Please verify this thing. Best way is to check what corresponds to a lexical
		 * step with one word [start, start] or [start, start+1). */
		
		/*if(ruleName.equals(ShiftReduceLexicalStep.LEXICAL_DERIVATION_STEP_RULENAME)) { //shift operation
			if(end != start)  //actually its wrong. should be based on words
				throw new RuntimeException("Its a lexical rule. Found "+start+" and "+end);
		}*/
		
		Iterator<Pair<RuleName, Category<MR>>> it = this.validStep[start][end].iterator();
		while(it.hasNext()) {
			
			Pair<RuleName, Category<MR>> step = it.next();
			RuleName gTruthRuleName = step.first();
			Category<MR> gTruthCategory = step.second();
			
			if(gTruthRuleName.equals(ruleName) && gTruthCategory.equals(opCategory))
				return true;
			
			if (gTruthRuleName instanceof OverloadedRuleName) {
				throw new RuntimeException("Should not reach here now");
			}
		}
		return false;
	}
	
	@Override
	public String toString() {
		StringBuilder s = new StringBuilder();
		
		s.append("Category: " + this.category + ",\n");
		s.append("Cursor: " + this.cursor + "; n = " + this.n + ",\n Ordered Parsing Op: {\n");
		for(ParsingOp<MR> parsingOp: this.orderedParsingOp) {
			s.append(parsingOp +", \n");
		}
		s.append("}, \n Valid Step [");
		for(int i = 0; i < this.validStep.length; i ++ ) {
			s.append("[");
			for(int j = i; j < this.validStep[i].length; j++) {
				s.append("(" + i + ", " + j + ") " + this.validStep[i][j] + "\n");
			}
			s.append("],\n");
		}
		s.append("]");
		
		return s.toString();
	}
}
