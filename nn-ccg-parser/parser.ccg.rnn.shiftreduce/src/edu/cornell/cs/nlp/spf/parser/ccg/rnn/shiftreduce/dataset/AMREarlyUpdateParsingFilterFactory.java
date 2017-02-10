package edu.cornell.cs.nlp.spf.parser.ccg.rnn.shiftreduce.dataset;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.Set;
import java.util.function.Predicate;

import edu.cornell.cs.nlp.spf.mr.lambda.LogicalExpression;
import edu.cornell.cs.nlp.spf.parser.ParsingOp;
import edu.cornell.cs.nlp.spf.parser.ccg.cky.chart.Cell;
import edu.cornell.cs.nlp.spf.parser.ccg.cky.chart.Chart;
import edu.cornell.cs.nlp.utils.log.ILogger;
import edu.cornell.cs.nlp.utils.log.LoggerFactory;
import edu.uw.cs.lil.amr.data.LabeledAmrSentence;
import edu.uw.cs.lil.amr.lambda.AMRServices;
import edu.uw.cs.lil.amr.lambda.GetAmrSubExpressions;

/** Class creates sparse feature and state dataset from chart using early updates */
public class AMREarlyUpdateParsingFilterFactory implements AbstractAMREarlyUpdateFilterFactory {
	
	public static final ILogger	LOG = LoggerFactory.create(AMREarlyUpdateParsingFilterFactory.class);
	
	private final double lambda;
	
	public AMREarlyUpdateParsingFilterFactory(double lambda) {
		this.lambda = lambda;
		LOG.info("Early update parsing filter factory created. Lambda = %s", this.lambda);
	}
	
	/** Computes matching score between cell and semantics. The score is given by: 
	 * score = match(label, cell label) - span_penalty where 
	 * 
	 * match(label, cell label) = -infinity if cell label is not a sub-expression of label
	 *                          else 0
	 * 
	 * span_penalty = ([end - start + 1]/n)^2 */
	private double getScore(Cell<LogicalExpression> cell, Set<LogicalExpression> subExpressions, int senLen) {
		
		final LogicalExpression cellExp = cell.getCategory().getSemantics();
		
		if(cellExp == null) {
			return Double.NEGATIVE_INFINITY;
		}
		
		// Check if cell label is a sub-expression
		//GetSubAmrExpressions
		boolean isSubExpression = false;
		
		for(LogicalExpression exp: subExpressions) {
			if(cellExp.equals(exp)) {
				isSubExpression = true;
				break;
			}
		}
		
		final double score;
		
		if(!isSubExpression) {
			score = Double.NEGATIVE_INFINITY;
		} else {
		
			// Penalty based on cell span
			int spanDiff = cell.getEnd() - cell.getStart() + 1;
			double t = (spanDiff /(double) senLen); 
			score = t * t;
			
			assert score <= 1 : " This scoring function should always be less than equal to 1 " + cell.getStart() 
								 + "-" + cell.getEnd() + " " + senLen;
		}
		
		return score;
	}
	
	@Override
	public Predicate<ParsingOp<LogicalExpression>> createFilter(Chart<LogicalExpression> chart,
			LabeledAmrSentence dataItem) {
		
		LogicalExpression underspecified = AMRServices.underspecifyAndStrip(dataItem.getLabel());
		Set<LogicalExpression> subExpressions = GetAmrSubExpressions.of(underspecified);
		
		final int senLen = chart.getSentenceLength();
		
		//Stored best solutions for [i, senLen - 1]
		//This is filled via dynamic programming from right to left
		Solution bestSolutions[] = new Solution[senLen + 1];
		
		//Base case
		bestSolutions[senLen] = new Solution(new ArrayList<Cell<LogicalExpression>>(), 0.0);
		
		for(int i = senLen - 1; i >= 0; i--) {
			
			//Here we compute the best solution in the span [i,n]
			//Our final answer will be given by some span of type [0,k] 

			double bestScoreSoFar = Double.NEGATIVE_INFINITY;
			Cell<LogicalExpression> bestScoreNewCell = null;
			
			for(int j = i + 1; j <= senLen; j++) {
				
				Iterator<Cell<LogicalExpression>> it = chart.getSpanIterable(i, j - 1).iterator();
				Solution solution = bestSolutions[j];
				
				while(it.hasNext()) {
					
					Cell<LogicalExpression> cellIJ = it.next();
					
					//Good place to add hard constraints and remove certain
					//type of cells that are not needed.
					
					double cellScore = this.getScore(cellIJ, subExpressions, senLen);
					double score = cellScore + solution.getScore();
					
					if(score > bestScoreSoFar) {
						bestScoreSoFar = score;
						bestScoreNewCell = cellIJ;
					} else if(score == bestScoreSoFar && bestScoreNewCell != null) {
						//for tie breaking we use the cell viterbi score given by CKY parse
						
						if(cellIJ.getViterbiScore() > bestScoreNewCell.getViterbiScore()) {
							bestScoreSoFar = score;
							bestScoreNewCell = cellIJ;
						}
					}
				}				
			}
			
			// If one cannot derive a solution then we simply put empty solution with 0 cost
			// since even first few contiguous cells can be used for learning. This cost of 0 is similar
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
		
		List<Cell<LogicalExpression>> finalCell = bestSolutions[0].getCells();
		
		//if number of cells are 0 then we cannot do anything.
		//in this case we simply return null
		if(finalCell.size() == 0) {
			return null;
		}
		
		return new CKYMultiParseTreeParsingFilter<LogicalExpression>(finalCell, senLen);
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
