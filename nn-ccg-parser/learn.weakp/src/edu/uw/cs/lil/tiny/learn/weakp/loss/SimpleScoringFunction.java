/*******************************************************************************
 * UW SPF - The University of Washington Semantic Parsing Framework
 * <p>
 * Copyright (C) 2013 Yoav Artzi
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
 ******************************************************************************/
package edu.uw.cs.lil.tiny.learn.weakp.loss;

import java.util.List;

import edu.uw.cs.lil.tiny.learn.weakp.loss.parser.IScoreFunction;

/**
 * Supports only score function that don't require the original data item.
 * 
 * @author Yoav Artzi
 * @param <MR>
 */
public class SimpleScoringFunction<MR> implements IScoreFunction<MR> {
	
	private final List<ILossFunction<MR>>	lossFunctions;
	
	public SimpleScoringFunction(List<ILossFunction<MR>> lossFunctions) {
		this.lossFunctions = lossFunctions;
	}
	
	@Override
	public double score(MR label) {
		double loss = 0.0;
		for (final ILossFunction<MR> lossFunction : lossFunctions) {
			loss += lossFunction.calculateLoss(null, label);
		}
		return -loss;
	}
	
	@Override
	public String toString() {
		return new StringBuilder(SimpleScoringFunction.class.getName())
				.append(" :: ").append(lossFunctions).toString();
	}
	
}
