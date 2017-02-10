package edu.cornell.cs.nlp.spf.parser.ccg.rnn.shiftreduce.dataset;

import java.util.Iterator;
import java.util.LinkedHashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.function.Predicate;

import edu.cornell.cs.nlp.spf.base.token.TokenSeq;
import edu.cornell.cs.nlp.spf.ccg.categories.Category;
import edu.cornell.cs.nlp.spf.data.IDataItem;
import edu.cornell.cs.nlp.spf.data.ILabeledDataItem;
import edu.cornell.cs.nlp.spf.data.sentence.Sentence;
import edu.cornell.cs.nlp.spf.parser.ParsingOp;
import edu.cornell.cs.nlp.spf.parser.ccg.cky.CKYDerivation;
import edu.cornell.cs.nlp.spf.parser.ccg.cky.CKYParserOutput;
import edu.cornell.cs.nlp.spf.parser.ccg.cky.chart.Cell;
import edu.cornell.cs.nlp.spf.parser.ccg.cky.single.CKYParser;
import edu.cornell.cs.nlp.spf.parser.ccg.cky.steps.IWeightedCKYStep;
import edu.cornell.cs.nlp.spf.parser.ccg.model.IDataItemModel;
import edu.cornell.cs.nlp.spf.parser.ccg.rules.OverloadedRuleName;
import edu.cornell.cs.nlp.spf.parser.ccg.rules.RuleName;
import edu.cornell.cs.nlp.spf.parser.ccg.rules.UnaryRuleName;
import edu.cornell.cs.nlp.spf.parser.filter.IParsingFilterFactory;
import edu.cornell.cs.nlp.utils.composites.Pair;
import edu.cornell.cs.nlp.utils.log.ILogger;
import edu.cornell.cs.nlp.utils.log.LoggerFactory;

public class CKYParsingFilterFactory<SAMPLE extends IDataItem<?>, DI extends ILabeledDataItem<SAMPLE, ?>, MR> {

	public static final ILogger	LOG = LoggerFactory.create(CKYParsingFilterFactory.class);
	private final CKYParser<Sentence, MR> ckyParser;
	private final IParsingFilterFactory<DI, MR> parsingFilterFactory;
	
	public CKYParsingFilterFactory(CKYParser<Sentence, MR> ckyParser, 
							IParsingFilterFactory<DI, MR> parsingFilterFactory) {
		this.ckyParser = ckyParser;
		this.parsingFilterFactory = parsingFilterFactory;
	}
	
	/** Create CKY filter for a given sentence. This filter will guide Neural Shift Reduce parser
	 *  to take steps that will make Shift-Reduce parse get the same parse tree as CKY */
	public Predicate<ParsingOp<MR>> create(DI dataItem, IDataItemModel<MR> dataItemModel) {
		
		final SAMPLE dataItemSample = dataItem.getSample();
		
		TokenSeq tk = ((Sentence) dataItemSample).getTokens();
		int n = tk.size(); //number of tokens
		
		CKYParserOutput<MR> output = this.ckyParser.parse((Sentence)dataItemSample, dataItemModel);
		
		LOG.info("Number of dervations are %s", output.getAllDerivations().size());
		
		List<CKYDerivation<MR>> it = output.getAllDerivations();
		
		CKYDerivation<MR> bestDerivation = null;
		double bestDerivationScore = Double.NEGATIVE_INFINITY;
		
		for(CKYDerivation<MR> derivation: it) {
			
			//check for S in future
			if(derivation.getCategory().getSemantics().equals(dataItem.getLabel())) {
				
				if(derivation.getScore() > bestDerivationScore) {
					bestDerivation = derivation;
					bestDerivationScore = derivation.getScore();
				}
			}
		}
		
		if(bestDerivation != null) {
			LOG.info("A CKY parse tree has the ground truth label. Score %s", bestDerivationScore);
			LOG.info("CKY parse tree Category %s", bestDerivation.getCategory());
		}
		else {
			LOG.info("CKY failed to parse the sentence.");
			return null;
		}
		
		Predicate<ParsingOp<MR>> supervisedPruningFilter = this.parsingFilterFactory.create(dataItem);
		
		return new CKYParsingFilter<MR>(bestDerivation/*chart*/, supervisedPruningFilter, n);
	}
	
	public static class CKYParsingFilter<MR> implements Predicate<ParsingOp<MR>> {
	
		/** chart created using CKY parsing algorithm. This is used as an oracle to guide
		 * other parsers hence must be accurate. */
		
		/** supervised pruning filter that removes parsing operations that won't lead to 
		 * the gold labeled logical form*/
		private final Predicate<ParsingOp<MR>> supervisedPruningFilter;
		
		private final ValidStep<MR>[][] validStep;
		
		/** number of tokens in the sentence */
		private final int n;
		
		@SuppressWarnings("unchecked")
		public CKYParsingFilter(CKYDerivation<MR> bestDerivation,
				Predicate<ParsingOp<MR>> supervisedPruningFilter, int n) {
			this.supervisedPruningFilter = supervisedPruningFilter;
			this.n = n;
			
			this.validStep = new ValidStep[n][n];
			
			for(int start = 0; start < n; start++) {
				for(int end = start; end < n; end++) {
					this.validStep[start][end] = new ValidStep<MR>();
				}
			}
			
			//check if the shifted category exists in the chart
			Cell<MR> bestDerivationCell = bestDerivation.getCell();
			this.recursivelyFindValidStep(bestDerivationCell);
			this.printValidStep();
		}
		
		public void recursivelyFindValidStep(Cell<MR> cell) {
			
			LinkedHashSet<IWeightedCKYStep<MR>> viterbiSteps = cell.getMaxSteps();
			
			for(IWeightedCKYStep<MR> viterbiStep: viterbiSteps) {
				
				int start = viterbiStep.getStart();
				int end = viterbiStep.getEnd();
				
				if(viterbiStep.getRuleName() == null || cell.getCategory() == null)
					throw new RuntimeException("Null rule name or category");
				
				this.validStep[start][end].add(viterbiStep.getRuleName(), viterbiStep.getRoot());
			}
		}
		
		public void printValidStep() {
			
			for(int start = 0; start < this.n; start ++ ) {
				for(int end = start; end < this.n; end++) {
					Iterator<Pair<RuleName, Category<MR>>> it = this.validStep[start][end].iterator();
					while(it.hasNext()) {
						Pair<RuleName, Category<MR>> step = it.next();
						LOG.info("[%s - %s RuleName: %s, Category: %s", start, end, step.first(), step.second());
					}
				}
			}
		}
		
		@Override
		public boolean test(ParsingOp<MR> op) {
			
			/* if it fails using supervised filter then reject it.
			 * out of the supervised pruning filter and cky chart filter. 
			 * the cheaper to compute filter should come before the other. */
			if(!supervisedPruningFilter.test(op))
				return false;
			
			RuleName ruleName = op.getRule();
			Category<MR> opCategory = op.getCategory();
			int start = op.getSpan().getStart();
			int end = op.getSpan().getEnd() - 1;

			/* WARNING: CKY and SR use different notion of span. One does [start, end) and other does 
			 * [start, end]. Please verify this thing. Best way is to check what corresponds to a lexical
			 * step [start, start] or [start, start+1). */
			
			/*if(ruleName.equals(ShiftReduceLexicalStep.LEXICAL_DERIVATION_STEP_RULENAME)) { //shift operation
				if(end != start) 
					throw new RuntimeException("Its a lexical rule. Found "+start+" and "+end);
			}*/
			
			Iterator<Pair<RuleName, Category<MR>>> it = this.validStep[start][end].iterator();
			while(it.hasNext()) {
				
				Pair<RuleName, Category<MR>> step = it.next();
				RuleName gTruthRuleName = step.first();
				Category<MR> gTruthCategory = step.second();
				
				if(gTruthRuleName.equals(ruleName) && gTruthCategory.equals(opCategory))
					return true;
				
				///// Special case, handling compounding
				if (gTruthRuleName instanceof OverloadedRuleName) {
				
					// Case the rule is a binary+unary rule. This is how unary rules are used in CKY. 
					UnaryRuleName unary = ((OverloadedRuleName) gTruthRuleName).getUnaryRule();
					RuleName binary = ((OverloadedRuleName) gTruthRuleName).getOverloadedRuleName();
					
					/* atleast one of the following two cases must be satisfied for 
					 * it to possible valid. However, note that these two cases are not strict.
					 * Currently not checking category for binary rules and not checking if unary
					 * is applied multiple times. This is due to insufficient information at the moment.
					 */
					
					/* matches the unary rule and the category
					 * In future, check if unary rule is applied over another unary rule
					 */
					if(unary.equals(ruleName) && gTruthCategory.equals(opCategory)) { 
						return true;
					}
					
					//matches the binary rule. In future, also check category
					if(binary.equals(ruleName)) {
						return true;
					}
				}
			}
			
			return false;
		}
	}
	
	public static class ValidStep<MR> {
		/** for a given sentence span represents what are the valid rules that can
		 *  be applied and what categories do they derive. */
		private final List<Pair<RuleName, Category<MR>>> validRuleAndCategory;
		
		public ValidStep() {
			this.validRuleAndCategory = new LinkedList<Pair<RuleName, Category<MR>>>();
		}
		
		public void add(RuleName ruleName, Category<MR> categ) {
			this.validRuleAndCategory.add(Pair.of(ruleName, categ));
		}
		
		public Iterator<Pair<RuleName, Category<MR>>> iterator() {
			return this.validRuleAndCategory.iterator();
		}
	}
}
