/*******************************************************************************
 * Copyright (C) 2011 - 2015 Yoav Artzi, All rights reserved.
 * <p>
 * This program is free software; you can redistribute it and/or modify it under
 * the terms of the GNU General Public License as published by the Free Software
 * Foundation; either version 2 of the License, or any later version.
 * <p>
 * This program is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
 * details.
 * <p>
 * You should have received a copy of the GNU General Public License along with
 * this program; if not, write to the Free Software Foundation, Inc., 51
 * Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.
 *******************************************************************************/
package edu.cornell.cs.nlp.spf.parser.ccg.shiftreduce;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;

import edu.cornell.cs.nlp.spf.base.hashvector.IHashVector;
import edu.cornell.cs.nlp.spf.ccg.categories.Category;
import edu.cornell.cs.nlp.spf.parser.IDerivation;
import edu.cornell.cs.nlp.spf.parser.ccg.shiftreduce.stacks.DerivationState;
import edu.cornell.cs.nlp.spf.parser.graph.IGraphParserOutput;
import edu.cornell.cs.nlp.utils.collections.CollectionUtils;
import edu.cornell.cs.nlp.utils.collections.IScorer;
import edu.cornell.cs.nlp.utils.filter.IFilter;
import edu.cornell.cs.nlp.utils.log.ILogger;
import edu.cornell.cs.nlp.utils.log.LoggerFactory;
import edu.cornell.cs.nlp.utils.math.LogSumExp;

/**
 * Parser output of the CKY parser, including the chart and all possible parses.
 *
 * @param <MR>
 *            Type of meaning representation
 * @author Dipendra K. Misra
 * @author Yoav Artzi
 */
public class ShiftReduceParserOutput<MR> implements IGraphParserOutput<MR> {
	
	public static final ILogger								LOG
								= LoggerFactory.create(ShiftReduceParserOutput.class);

	/** All complete parses */
	private final List<ShiftReduceDerivation<MR>>	allParses;

	/** Max scoring complete parses */
	private final List<ShiftReduceDerivation<MR>>	bestParses;

	/** Total parsing time */
	private final long								parsingTime;
	
	private final boolean							isStiched;

	public ShiftReduceParserOutput(List<DerivationState<MR>> dstates, long parsingTime) {
		//assumes each derivation state is a complete parse tree
		
		List<ShiftReduceDerivation<MR>> derivations = new LinkedList<ShiftReduceDerivation<MR>>();
		
		//cluster dstates by category and create a derivation for each category, O(nk) slow computation
		List<List<DerivationState<MR>>> clusters = new LinkedList<List<DerivationState<MR>>>();
		
		Iterator<DerivationState<MR>> iter = dstates.iterator();
		while(iter.hasNext()) {
			boolean found = false;
			DerivationState<MR> dstates_ = iter.next();
			dstates_.computeFeatures(); //compute the features for this parse tree, it has probably not been divided
			
			Iterator<List<DerivationState<MR>>> iterCl = clusters.iterator();
			while(iterCl.hasNext()) {
				List<DerivationState<MR>> clusters_ = iterCl.next();
				assert clusters_.size() > 0;
				if(clusters_.get(0).returnLastNonTerminal().getCategory().
						equals(dstates_.returnLastNonTerminal().getCategory())) {
					found = true;
					clusters_.add(dstates_);
					break;
				}
			}
			
			if(!found) {
				List<DerivationState<MR>> init = new LinkedList<DerivationState<MR>>();
				init.add(dstates_);
				clusters.add(init);
			}
		}
		
		Iterator<List<DerivationState<MR>>> iterCl = clusters.iterator();
		while(iterCl.hasNext())
				derivations.add(new ShiftReduceDerivation<MR>(iterCl.next()));

		this.parsingTime = parsingTime;
		this.allParses = Collections.unmodifiableList(derivations);
		this.bestParses = Collections.unmodifiableList(findBestParses(this.allParses));
		this.isStiched = false;
	}
	
	public ShiftReduceParserOutput(ShiftReduceDerivation<MR> singleton, long parsingTime) {
		
		List<ShiftReduceDerivation<MR>> allParses = new LinkedList<ShiftReduceDerivation<MR>>();
		List<ShiftReduceDerivation<MR>> bestParses = new LinkedList<ShiftReduceDerivation<MR>>();
		
		if(singleton != null) {
			allParses.add(singleton);
			bestParses.add(singleton);	
		}
		
		this.allParses = Collections.unmodifiableList(allParses);
		this.bestParses = Collections.unmodifiableList(bestParses);
		this.parsingTime = parsingTime;
		this.isStiched = true;
	}
	
	public ShiftReduceParserOutput(List<ShiftReduceDerivation<MR>> derivations, long parsingTime, boolean marker) {
		
		this.allParses = Collections.unmodifiableList(derivations);
		this.bestParses = Collections.unmodifiableList(findBestParses(this.allParses));
		this.parsingTime = parsingTime;
		this.isStiched = true;
	}	
	
	public ShiftReduceParserOutput(List<DerivationState<MR>> states, List<Category<MR>> categories, long parsingTime) {
		
		final List<List<DerivationState<MR>>> clusters = new ArrayList<List<DerivationState<MR>>>();
		final List<Category<MR>> clusterCategory = new ArrayList<Category<MR>>();
		
		Iterator<DerivationState<MR>> dit = states.iterator();
		Iterator<Category<MR>> cit = categories.iterator();
		
		while(dit.hasNext()) {
			
			final DerivationState<MR> state = dit.next();
			final Category<MR> stateCategory = cit.next();
			
			//check if state is part of a cluster
			int i = 0;
			boolean found = false;
			for(Category<MR> c: clusterCategory) {
				
				if(c.equals(stateCategory)) {
					found = true;
					break;
				}
				i++;
			}
			
			if(found) {
				//add to cluster
				clusters.get(i).add(state); 
			} else {
				List<DerivationState<MR>> newCluster = new ArrayList<DerivationState<MR>>();
				newCluster.add(state);
				clusters.add(newCluster);
				clusterCategory.add(stateCategory);
			}
		}
		
		//Create derivations
		Iterator<Category<MR>> clusterCategoryIt = clusterCategory.iterator();
		List<ShiftReduceDerivation<MR>> derivations = new ArrayList<ShiftReduceDerivation<MR>>();
		for(List<DerivationState<MR>> cluster: clusters) {
			ShiftReduceDerivation<MR> derivation = new ShiftReduceDerivation<MR>(cluster, clusterCategoryIt.next());
			derivations.add(derivation);
		}
		
		this.allParses = Collections.unmodifiableList(derivations);
		this.bestParses = Collections.unmodifiableList(findBestParses(this.allParses));
		this.parsingTime = parsingTime;
		this.isStiched = true;
	}
	

	private static <MR> List<ShiftReduceDerivation<MR>> findBestParses(
			List<ShiftReduceDerivation<MR>> all) {
		return findBestParses(all, null);
	}

	private static <MR> List<ShiftReduceDerivation<MR>> findBestParses(
			List<ShiftReduceDerivation<MR>> all, IFilter<Category<MR>> filter) {
		final List<ShiftReduceDerivation<MR>> best = new LinkedList<ShiftReduceDerivation<MR>>();
		double bestScore = -Double.MAX_VALUE;
		for (final ShiftReduceDerivation<MR> p : all) {
			if (filter == null || filter.test(p.getCategory())) {
				if (p.getScore() == bestScore) {
					best.add(p);
				}
				if (p.getScore() > bestScore) {
					bestScore = p.getScore();
					best.clear();
					best.add(p);
				}
			}
		}
		return best;
	}
	
	public boolean isStiched() {
		return this.isStiched;
	}

	@Override
	public List<ShiftReduceDerivation<MR>> getAllDerivations() {
		return allParses;
	}

	@Override
	public List<ShiftReduceDerivation<MR>> getBestDerivations() {
		return bestParses;
	}

	@Override
	public List<ShiftReduceDerivation<MR>> getDerivations(
			final IFilter<Category<MR>> filter) {
		final List<ShiftReduceDerivation<MR>> parses = new ArrayList<ShiftReduceDerivation<MR>>(
				allParses);
		CollectionUtils.filterInPlace(parses,
				new IFilter<IDerivation<MR>>() {

					@Override
					public boolean test(IDerivation<MR> e) {
						return filter.test(e.getCategory());
					}
				});
		return parses;
	}

	@Override
	public List<ShiftReduceDerivation<MR>> getMaxDerivations(
			IFilter<Category<MR>> filter) {
		return findBestParses(allParses, filter);
	}

	@Override
	public long getParsingTime() {
		return parsingTime;
	}

	@Override
	public boolean isExact() {
		return true; //can change in future -dipendra
	}

	@Override
	public IHashVector logExpectedFeatures() {
		throw new RuntimeException("Not supporting this function. Probably does not make sense.");
	}

	@Override
	public IHashVector logExpectedFeatures(IFilter<Category<MR>> filter) {
		throw new RuntimeException("Not supporting this function. Probably does not make sense.");
	}

	@Override
	public IHashVector logExpectedFeatures(IScorer<Category<MR>> initialScorer) {
		throw new RuntimeException("Not supporting this function. Probably does not make sense.");
	}

	@Override
	public double logNorm() {
		
		List<Double> logInsideScoreList = new ArrayList<Double>();
		
		for(ShiftReduceDerivation<MR> derivation: this.allParses) {
			logInsideScoreList.add(derivation.getLogInsideScore());
		}
		
		final double logNorm = LogSumExp.of(logInsideScoreList);
		
		return logNorm;
		//return 0;
	}

	@Override
	public double logNorm(IFilter<Category<MR>> filter) {
		//throw new RuntimeException("Operation not supported");
		LOG.warn("logNorm(IFilter<Category<MR>> filter) not supported but used!!!!!!");
		return 0;
	}

}
