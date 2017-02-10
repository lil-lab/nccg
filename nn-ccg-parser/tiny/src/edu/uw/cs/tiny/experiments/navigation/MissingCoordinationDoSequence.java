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
package edu.uw.cs.tiny.experiments.navigation;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.LinkedList;
import java.util.List;

import edu.uw.cs.lil.tiny.ccg.categories.Category;
import edu.uw.cs.lil.tiny.ccg.categories.ComplexCategory;
import edu.uw.cs.lil.tiny.ccg.categories.ICategoryServices;
import edu.uw.cs.lil.tiny.mr.lambda.LogicalExpression;
import edu.uw.cs.lil.tiny.parser.ccg.rules.IUnaryParseRule;
import edu.uw.cs.lil.tiny.parser.ccg.rules.ParseRuleResult;
import edu.uw.cs.utils.composites.Pair;

/**
 * Unary parse rules that compensate for missing coordination tokens.
 * 
 * @author Yoav Artzi
 */
public class MissingCoordinationDoSequence implements
		IUnaryParseRule<LogicalExpression> {
	
	/**
	 * Missing coordination categories.
	 */
	private final List<Pair<String, ComplexCategory<LogicalExpression>>>	categories;
	private final ICategoryServices<LogicalExpression>						categoryServices;
	
	public MissingCoordinationDoSequence(
			ICategoryServices<LogicalExpression> categoryServices) {
		this.categoryServices = categoryServices;
		
		// Init the list of possible categories to apply
		final List<Pair<String, ComplexCategory<LogicalExpression>>> modifiableCategories = new ArrayList<Pair<String, ComplexCategory<LogicalExpression>>>();
		categories = Collections.unmodifiableList(modifiableCategories);
		
		// For the navigation domain:
		modifiableCategories
				.add(Pair
						.of("nullcord_s_s_s_do-seq",
								(ComplexCategory<LogicalExpression>) categoryServices
										.parse("S\\S/S : (lambda $0:t (lambda $1:t (do-sequentially:<t*,t> $1 $0)))")));
		
	}
	
	@Override
	public Collection<ParseRuleResult<LogicalExpression>> apply(
			Category<LogicalExpression> category) {
		final List<ParseRuleResult<LogicalExpression>> result = new LinkedList<ParseRuleResult<LogicalExpression>>();
		
		for (final Pair<String, ComplexCategory<LogicalExpression>> categoryPair : categories) {
			final Category<LogicalExpression> applyResult = categoryServices
					.apply(categoryPair.second(), category);
			if (applyResult != null && applyResult.getSem() != null) {
				result.add(new ParseRuleResult<LogicalExpression>(categoryPair
						.first(), applyResult));
			}
		}
		
		return result;
	}
	
	public List<String> getRuleNames() {
		final List<String> names = new ArrayList<String>(categories.size());
		for (final Pair<String, ComplexCategory<LogicalExpression>> categoryPair : categories) {
			names.add(categoryPair.first());
		}
		return names;
	}
}
