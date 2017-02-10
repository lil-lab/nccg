package edu.cornell.cs.nlp.spf.parser.ccg.rnn.shiftreduce.dataset;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.function.Predicate;
import java.util.stream.Collectors;

import edu.cornell.cs.nlp.spf.base.token.TokenSeq;
import edu.cornell.cs.nlp.spf.ccg.categories.Category;
import edu.cornell.cs.nlp.spf.ccg.categories.syntax.Syntax;
import edu.cornell.cs.nlp.spf.mr.lambda.LogicalExpression;
import edu.cornell.cs.nlp.spf.parser.ParsingOp;
import edu.cornell.cs.nlp.spf.parser.ccg.cky.chart.Cell;
import edu.cornell.cs.nlp.spf.parser.ccg.cky.chart.Chart;
import edu.cornell.cs.nlp.utils.log.ILogger;
import edu.cornell.cs.nlp.utils.log.LoggerFactory;
import edu.uw.cs.lil.amr.data.LabeledAmrSentence;
import edu.uw.cs.lil.amr.lambda.AMRServices;
import edu.uw.cs.lil.amr.lambda.GetAmrSubExpressions;
import edu.uw.cs.lil.amr.lambda.StripOverload;

/** Conservative parsing filter that considers only contiguous cell pattern
 * from the list of cells considered by CKY parser for its early update. */ 
public class AMRConservativeEarlyUpdateParsingFilterFactory implements AbstractAMREarlyUpdateFilterFactory {

	public static final ILogger	LOG = LoggerFactory.create(AMRConservativeEarlyUpdateParsingFilterFactory.class);
	
	/** Marking it true gives us the filter as exactly used by CKY. */
	private final boolean useViterbiCellsPerSemantics;
	
	public AMRConservativeEarlyUpdateParsingFilterFactory() {
		this.useViterbiCellsPerSemantics = false;
	}
	
	/** Returns a list of cells that are contiguous (non-overlapping continuous) */
	private List<Cell<LogicalExpression>> optimalContiguousSpanningCell(Set<Cell<LogicalExpression>>[][] cellSpace, 
									int senLen) {
		
		//bestSolutions[i] represents best series of contiguous cell starting with index i
		Solution[] bestSolutions = new Solution[senLen + 1];
		
		//Base case
		bestSolutions[senLen] = new Solution(new ArrayList<Cell<LogicalExpression>>(), 0.0);
		
		for(int i = senLen - 1; i >= 0; i--) {
			
			//Here we compute the best solution in the span [i,n]
			//Our final answer will be given by some span of type [0,k] 

			double bestScoreSoFar = Double.NEGATIVE_INFINITY;
			Cell<LogicalExpression> bestScoreNewCell = null;
			
			for(int j = i + 1; j <= senLen; j++) {
				
				if(cellSpace[i][j - 1] == null) { //no relevant cell
					continue;
				}
				
				Iterator<Cell<LogicalExpression>> it = cellSpace[i][j - 1].iterator();
				Solution solution = bestSolutions[j];
				
				while(it.hasNext()) {
					
					Cell<LogicalExpression> cellIJ = it.next();
					
					//Good place to add hard constraints and remove certain
					//type of cells that are not needed.
					
					double cellScore = cellIJ.getViterbiScore();
					double score = cellScore + solution.getScore();
					
					//Update if score is better than what we have seen so far. Arbitrarily breaking ties
					if(score > bestScoreSoFar) {
						bestScoreSoFar = score;
						bestScoreNewCell = cellIJ;
					} 
				}				
			}
			
			// If one cannot derive a solution then we simply put empty solution with 0 cost
			// since even first few contiguous cells can be used for learning. This cost of 0 is similar to
			// skipping cost. In future, we can add a skipping penalty. 
			if(bestScoreNewCell == null) {				
				bestSolutions[i] = new Solution(new ArrayList<Cell<LogicalExpression>>(), 0.0);
			} else {
				
				//Add the new solution
				List<Cell<LogicalExpression>> cells = new ArrayList<Cell<LogicalExpression>>();
				cells.add(bestScoreNewCell);
				
				Solution nextSolution = bestSolutions[bestScoreNewCell.getEnd() + 1];
				cells.addAll(nextSolution.getCells());
				
				bestSolutions[i] = new Solution(cells, bestScoreSoFar); 	
			}
		}
		
		return bestSolutions[0].getCells();
	}
	
	@Override
	public Predicate<ParsingOp<LogicalExpression>> createFilter(Chart<LogicalExpression> chart,
			LabeledAmrSentence dataItem) {
		
		// Get all AMR sub-expression from the stripped and underspecified
		// labeled LF. Remove solitary references, as they lead to noisy
		// updates.
		final Set<LogicalExpression> subExpressions = GetAmrSubExpressions
				.of(AMRServices.underspecifyAndStrip(dataItem.getLabel()))
				.stream().collect(Collectors.toSet());

		// Identify the max-scoring spans that hold sub-expressions of the
		// labeled LF. If multiple spans contain the same sub-expression, prefer
		// the one with the highest score.
		final Map<LogicalExpression, List<Cell<LogicalExpression>>> subExpCells = new HashMap<>();
		final Map<LogicalExpression, Double> subExpCellViterbiScores = new HashMap<>();
		for (final Cell<LogicalExpression> cell : chart) {
			final Category<LogicalExpression> category = cell.getCategory();
			if (category.getSemantics() == null) {
				continue;
			}

			// Verify that the cell is a valid span according to CCGBank
			// constraints, if available.
			final TokenSeq tokens = dataItem.getSample().getTokens()
					.sub(cell.getStart(), cell.getEnd() + 1);
			final Set<Syntax> ccgBankCategories = dataItem
					.getCCGBankCategories(tokens);
			if (ccgBankCategories != null) {
				if (ccgBankCategories.isEmpty()) {
					continue;
				}
				boolean found = false;
				for (final Syntax syntax : ccgBankCategories) {
					if (cell.getCategory().getSyntax().stripAttributes()
							.equals(syntax.stripAttributes())) {
						found = true;
						break;
					}
				}
				if (!found) {
					continue;
				}
			}

			final LogicalExpression semantics = StripOverload.of(
					AMRServices.underspecifyAndStrip(category.getSemantics()));
			if (subExpressions.contains(semantics)) {
				
				if(this.useViterbiCellsPerSemantics) {
				
					if (!subExpCellViterbiScores.containsKey(semantics)
							|| subExpCellViterbiScores.get(semantics) < cell
									.getViterbiScore()) {
						if (!subExpCells.containsKey(semantics)) {
							subExpCells.put(semantics, new LinkedList<>());
						} else {
							subExpCells.get(semantics).clear();
						}
						subExpCells.get(semantics).add(cell);
						subExpCellViterbiScores.put(semantics,
								cell.getViterbiScore());
					} else if (subExpCellViterbiScores.containsKey(semantics)
							&& subExpCellViterbiScores.get(semantics) == cell
									.getViterbiScore()) {
						subExpCells.get(semantics).add(cell);
					}
				} else {
					if (!subExpCells.containsKey(semantics)) {
						subExpCells.put(semantics, new LinkedList<>());
					}
					subExpCells.get(semantics).add(cell);
				}
			}
		}
		
		//Process cells into span
		final int senLen = dataItem.getSample().getTokens().size();
		int count = 0;
		
		@SuppressWarnings("unchecked")
		final Set<Cell<LogicalExpression>>[][] cellSpace = new HashSet[senLen][senLen];
		for(List<Cell<LogicalExpression>> subExpCell: subExpCells.values()) {
			for(Cell<LogicalExpression> cell: subExpCell) {
			
				if(cellSpace[cell.getStart()][cell.getEnd()] == null) {
					cellSpace[cell.getStart()][cell.getEnd()] = new HashSet<Cell<LogicalExpression>>();
				}
				
				cellSpace[cell.getStart()][cell.getEnd()].add(cell);
				count++;
				LOG.debug("Cell considered %s - %s", cell.getStart(), cell.getEnd());
			}
		}
		
		LOG.debug("Number of cells considered %s", count);

		final List<Cell<LogicalExpression>> optimalCell = this.optimalContiguousSpanningCell(cellSpace, senLen);
		
		for(Cell<LogicalExpression> cell: optimalCell) {
			LOG.debug("Cell %s - %s score %s", cell.getStart(), cell.getEnd(), cell.getViterbiScore());
		}
		
		if(optimalCell.size() == 0) { //trivial solution, return null
			LOG.debug("Conservative Filter could not find a partial parse tree");
			return null;
		}
		
		return new CKYMultiParseTreeParsingFilter<LogicalExpression>(optimalCell, senLen);
	}
	
	protected class Solution {
		
		// List of contiguous cells i.e. cell_{i+1}'s start = cell_{i}'s end + 1
		// and last cell's end is same as sentence length - 1
		private final List<Cell<LogicalExpression>> cells;
		
		// Score of this solution
		private final double score;
		
		protected Solution(List<Cell<LogicalExpression>> cells, double score) {
			
			this.cells = cells;
			this.score = score;
		}
		
		public double getScore() {
			return this.score;
		}
		
		public List<Cell<LogicalExpression>> getCells() {
			return this.cells;
		}
	}

}
