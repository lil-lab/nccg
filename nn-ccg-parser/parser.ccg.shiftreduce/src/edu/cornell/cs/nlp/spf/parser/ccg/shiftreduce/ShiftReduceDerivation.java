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
import java.util.Comparator;
import java.util.HashSet;
import java.util.Iterator;
import java.util.LinkedHashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Set;

import edu.cornell.cs.nlp.spf.base.hashvector.HashVectorFactory;
import edu.cornell.cs.nlp.spf.base.hashvector.IHashVector;
import edu.cornell.cs.nlp.spf.base.hashvector.IHashVectorImmutable;
import edu.cornell.cs.nlp.spf.ccg.categories.Category;
import edu.cornell.cs.nlp.spf.ccg.lexicon.LexicalEntry;
import edu.cornell.cs.nlp.spf.parser.RuleUsageTriplet;
import edu.cornell.cs.nlp.spf.parser.ccg.shiftreduce.stacks.DerivationState;
import edu.cornell.cs.nlp.spf.parser.ccg.shiftreduce.steps.IWeightedShiftReduceStep;
import edu.cornell.cs.nlp.spf.parser.graph.IGraphDerivation;
import edu.cornell.cs.nlp.utils.math.LogSumExp;

/**
 * Pack of Parse Trees corresponding to a single category in the final derivation
 *
 * @author Dipendra K. Misra
 * @author Yoav Artzi
 * @param <MR>
 *            Meaning representation.
 */
public class ShiftReduceDerivation<MR> implements IGraphDerivation<MR> {

	final List<DerivationState<MR>> dstates;    //has to be a set of complete parse tree
	final List<DerivationState<MR>> maxDStates; //parse trees with maximum score in this derivation
	final Category<MR> category; 
	
	public ShiftReduceDerivation(List<DerivationState<MR>> dstates) {
		assert dstates.size() > 0;
		this.dstates = dstates;
		this.category = dstates.get(0).returnLastNonTerminal().getCategory(); //all have the same category
		
		double maxScore = Collections.max(dstates, new Comparator<DerivationState<MR>>() {
			public int compare(DerivationState<MR> left, DerivationState<MR> right) {
        		return Double.compare(left.score, right.score); 
    		}   
		}).score;
		
		Iterator<DerivationState<MR>> iter = this.dstates.iterator();
		this.maxDStates = new LinkedList<DerivationState<MR>>();
		
		while(iter.hasNext()) {
			DerivationState<MR> dstate_ = iter.next();
			if(dstate_.score == maxScore)
				maxDStates.add(dstate_);
		}
	}
	
	public ShiftReduceDerivation(DerivationState<MR> dstate, Category<MR> category) {
		
		this.dstates = new LinkedList<DerivationState<MR>>();
		this.maxDStates = new LinkedList<DerivationState<MR>>();
		
		dstates.add(dstate);
		maxDStates.add(dstate);
	
	    this.category = category;
	}
	
	public ShiftReduceDerivation(List<DerivationState<MR>> dstates, Category<MR> category) {
		assert dstates.size() > 0;
		this.dstates = dstates;
		this.category = category;
		
		double maxScore = Collections.max(dstates, new Comparator<DerivationState<MR>>() {
			public int compare(DerivationState<MR> left, DerivationState<MR> right) {
        		return Double.compare(left.score, right.score); 
    		}   
		}).score;
		
		Iterator<DerivationState<MR>> iter = this.dstates.iterator();
		this.maxDStates = new LinkedList<DerivationState<MR>>();
		
		while(iter.hasNext()) {
			DerivationState<MR> dstate_ = iter.next();
			if(dstate_.score == maxScore)
				maxDStates.add(dstate_);
		}
	}
	
	public List<DerivationState<MR>> getMaxScoringDerivationStates() {
		return this.maxDStates;
	}
	
	public List<DerivationState<MR>> getAllDerivationStates() {
		return this.dstates;
	}

	@Override
	public boolean equals(Object obj) {
		if (this == obj) {
			return true;
		}
		if (obj == null) {
			return false;
		}
		if (getClass() != obj.getClass()) {
			return false;
		}
		@SuppressWarnings("rawtypes")
		
		final ShiftReduceDerivation other = (ShiftReduceDerivation) obj;
		/* two derivations are equal if they have the same category. Even if they 
		 * don't have the same number of packed states, they are equivalent and should
		 * be coalesced since a derivation is created for unique category.  */
		return this.category.equals(other.category);
		
		//return true;
	}

	@Override
	public LinkedHashSet<LexicalEntry<MR>> getAllLexicalEntries() {
		LinkedHashSet<LexicalEntry<MR>> result = new LinkedHashSet<LexicalEntry<MR>>();
		Iterator<DerivationState<MR>> iter = this.dstates.iterator();
		
		while(iter.hasNext()) {
			result.addAll(iter.next().returnLexicalEntries());
		}
		
		return result;
	}

	@Override
	public Set<IWeightedShiftReduceStep<MR>> getAllSteps() {
		
		HashSet<IWeightedShiftReduceStep<MR>> result = new HashSet<IWeightedShiftReduceStep<MR>>();
		Iterator<DerivationState<MR>> iter = this.dstates.iterator();
		
		while(iter.hasNext()) {
			result.addAll(iter.next().returnSteps());
		}
		
		return result;
	}

	@Override
	public IHashVectorImmutable getAverageMaxFeatureVector() {
		
		Iterator<DerivationState<MR>> iter = this.maxDStates.iterator();
		IHashVector feature = HashVectorFactory.create(); 
		
		while(iter.hasNext()) {
			feature = feature.addTimes(1, iter.next().getFeatures());
		}
		
		feature.divideBy(this.maxDStates.size());
		
		return feature; 
	}

	@Override
	public Category<MR> getCategory() {
		return this.category; 
	}

	@Override
	public LinkedHashSet<LexicalEntry<MR>> getMaxLexicalEntries() {
		LinkedHashSet<LexicalEntry<MR>> result = new LinkedHashSet<LexicalEntry<MR>>();
		Iterator<DerivationState<MR>> iter = this.maxDStates.iterator();
		
		while(iter.hasNext()) {
			result.addAll(iter.next().returnLexicalEntries());
		}
		
		return result;
	}

	@Override
	public LinkedHashSet<RuleUsageTriplet> getMaxRulesUsed() {
		List<RuleUsageTriplet> result = new LinkedList<RuleUsageTriplet>();
		Iterator<DerivationState<MR>> iter = this.maxDStates.iterator();
		
		while(iter.hasNext()) {
			result.addAll(iter.next().returnAllRuleUsageTriplet());
		}
		
		return new LinkedHashSet<RuleUsageTriplet>(result);
	}

	@Override
	public LinkedHashSet<IWeightedShiftReduceStep<MR>> getMaxSteps() {
		List<IWeightedShiftReduceStep<MR>> result = new LinkedList<IWeightedShiftReduceStep<MR>>();
		Iterator<DerivationState<MR>> iter = this.maxDStates.iterator();
		
		while(iter.hasNext()) {
			result.addAll(iter.next().returnSteps());
		}
		
		return new LinkedHashSet<IWeightedShiftReduceStep<MR>>(result);
	}

	@Override
	public double getScore() {
		return this.maxDStates.get(0).score; //all maxDStates have same max score
	}

	@Override
	public MR getSemantics() {
		return this.category.getSemantics();
	}

	@Override
	public int hashCode() {
		return this.category.hashCode();
	}

	/**
	 * The number of parses packed into this derivation.
	 */
	@Override
	public long numParses() {
		return this.dstates.size();
	}

	@Override
	public String toString() {
		return this.category.toString(); //a proper string output is needed
	}

	@Override
	public double getLogInsideScore() {
		
		List<Double> logScoreList = new ArrayList<Double>();
		
		for(DerivationState<MR> dstate: this.dstates) {
			logScoreList.add(dstate.score);
		}
		return LogSumExp.of(logScoreList);
		
		//return this.maxDStates.get(0).score; //return viterbi score
	}

}
