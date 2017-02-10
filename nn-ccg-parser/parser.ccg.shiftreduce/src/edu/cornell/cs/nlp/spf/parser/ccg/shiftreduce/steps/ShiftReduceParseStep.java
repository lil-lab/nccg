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
import java.util.List;

import edu.cornell.cs.nlp.spf.ccg.categories.Category;
import edu.cornell.cs.nlp.spf.parser.ccg.rules.RuleName;

/**
 * A single CKY parse step.
 *
 * @author Yoav Artzi
 * @param <MR>
 */
public class ShiftReduceParseStep<MR> extends AbstractShiftReduceStep<MR> implements Serializable {

	private static final long serialVersionUID = 7695058364106992558L;
	
	boolean isUnary = false;
	
	public ShiftReduceParseStep(Category<MR> root, List<Category<MR>> children,
			boolean isFullParse, boolean isUnary, RuleName ruleName, int start, int end) {
		super(root, children, isFullParse, isUnary, ruleName, start, end);
		this.isUnary = isUnary;
	}
	
}
