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
package edu.cornell.cs.nlp.spf.parser.ccg.shiftreduce.steps;

import java.io.Serializable;

import edu.cornell.cs.nlp.spf.base.hashvector.HashVectorFactory;
import edu.cornell.cs.nlp.spf.base.hashvector.IHashVector;
import edu.cornell.cs.nlp.spf.base.hashvector.IHashVectorImmutable;
import edu.cornell.cs.nlp.spf.ccg.categories.Category;
import edu.cornell.cs.nlp.spf.ccg.lexicon.LexicalEntry;
import edu.cornell.cs.nlp.spf.parser.ccg.ILexicalParseStep;
import edu.cornell.cs.nlp.spf.parser.ccg.model.IDataItemModel;
import edu.cornell.cs.nlp.spf.parser.ccg.rules.RuleName;

/**
 * A {@link ShiftReduceLexicalStep} with a features and weight.
 *
 * @author Yoav Artzi
 * @param <MR>
 *            Meaning representation.
 */
public class WeightedShiftReduceLexicalStep<MR> implements ILexicalParseStep<MR>,
		IWeightedShiftReduceStep<MR>, Serializable {

	private static final long serialVersionUID = 747878036349121383L;

	private final int							hashCode;

	private final ShiftReduceLexicalStep<MR>	step;
	private final IHashVector					stepFeatures;
	private final Double						stepScore;

	public WeightedShiftReduceLexicalStep(ShiftReduceLexicalStep<MR> step,
			IDataItemModel<MR> model) {
		assert step != null;
		this.step = step;
		this.stepFeatures = model.computeFeatures(step);
		assert stepFeatures != null;
		this.stepScore = model.score(stepFeatures);
		assert !Double.isNaN(stepScore) && !Double.isInfinite(stepScore) : "Invalid step score";
		this.hashCode = calcHashCode();
	}
	
	public WeightedShiftReduceLexicalStep(ShiftReduceLexicalStep<MR> step, Double score) {
		this.step = step;
		this.stepFeatures = null;//HashVectorFactory.create();
		this.stepScore = score;
		this.hashCode = calcHashCode();
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

		@SuppressWarnings("unchecked")
		final WeightedShiftReduceLexicalStep<MR> other = (WeightedShiftReduceLexicalStep<MR>) obj;

		// Short circuit with hash code comparison.
		if (hashCode != other.hashCode) {
			return false;
		}
		
		////////////////
		if(stepFeatures == null && other.stepFeatures != null || 
		   stepFeatures != null && other.stepFeatures == null) {
			return false;
		}
	    ////////////////

		if (!step.equals(other.step)) {
			return false;
		}
		
		if (stepFeatures != null && !stepFeatures.equals(other.stepFeatures)) {
			return false;
		}
		return true;
	}

	@Override
	public Category<MR> getChild(int i) {
		return step.getChild(i);
	}

	@Override
	public int getEnd() {
		return step.getEnd();
	}

	@Override
	public LexicalEntry<MR> getLexicalEntry() {
		return step.getLexicalEntry();
	}

	@Override
	public Category<MR> getRoot() {
		return step.getRoot();
	}

	@Override
	public RuleName getRuleName() {
		return step.getRuleName();
	}

	@Override
	public int getStart() {
		return step.getStart();
	}

	@Override
	public IHashVector getStepFeatures() {
		return stepFeatures;
	}

	/**
	 * The linear score of this step.
	 */
	@Override
	public double getStepScore() {
		return stepScore;
	}

	@Override
	public int hashCode() {
		return hashCode;
	}

	@Override
	public boolean isFullParse() {
		return step.isFullParse();
	}

	@Override
	public int numChildren() {
		return step.numChildren();
	}

	@Override
	public String toString() {
		return toString(true, true, null);
	}

	@Override
	public String toString(boolean verbose, boolean recursive) {
		return toString(verbose, recursive, null);
	}

	@Override
	public String toString(boolean verbose, boolean recursive,
			IHashVectorImmutable theta) {
		final StringBuilder sb = new StringBuilder(step.toString(verbose,
				recursive));
		if(stepFeatures != null) {
			sb.append("{").append(
					theta == null ? stepFeatures : theta.printValues(stepFeatures));
		}
		sb.append(" -> ").append(stepScore).append("}");
		return sb.toString();
	}

	private int calcHashCode() {
		final int prime = 31;
		int result = 1;
		if(step != null) {
			result = prime * result + step.hashCode();
		}
		
		if(stepFeatures != null) {
			result = prime * result + stepFeatures.hashCode();
		}
		return result;
	}

	@Override
	public ShiftReduceLexicalStep<MR> getUnderlyingParseStep() {
		return this.step;
	}

}
