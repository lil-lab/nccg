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
package edu.cornell.cs.nlp.spf.atis.parser.ccg.rules.typeshifting;

import edu.cornell.cs.nlp.spf.ccg.categories.Category;
import edu.cornell.cs.nlp.spf.ccg.categories.ComplexCategory;
import edu.cornell.cs.nlp.spf.ccg.categories.ICategoryServices;
import edu.cornell.cs.nlp.spf.ccg.categories.syntax.Syntax;
import edu.cornell.cs.nlp.spf.mr.lambda.LogicalExpression;
import edu.cornell.cs.nlp.spf.parser.ccg.rules.ParseRuleResult;
import edu.cornell.cs.nlp.spf.parser.ccg.rules.SentenceSpan;
import edu.cornell.cs.nlp.spf.parser.ccg.rules.RuleName.Direction;
import edu.cornell.cs.nlp.spf.parser.ccg.rules.primitivebinary.application.AbstractApplication;

/**
 * A rule doing composition of PPs
 * <ul>
 * PP PP -> PP
 * </ul>
 *
 * @author Luke Zettlemoyer
 */
public class APComposition extends AbstractApplication<LogicalExpression> {
	/**
	 * 
	 */
	private static final long	serialVersionUID	= -4954660984561726608L;

	private static String								RULE_LABEL	= "apcomp";

	private final ComplexCategory<LogicalExpression>	workerCategory;

	public APComposition(ICategoryServices<LogicalExpression> categoryServices) {
		super(RULE_LABEL, Direction.FORWARD, categoryServices);
		workerCategory = (ComplexCategory<LogicalExpression>) categoryServices
				.read("AP/AP/AP : (lambda $0:<ev,t> (lambda $1:<ev,t> (lambda $2:ev (and:<t*,t> ($0 $2) ($1 $2)))))");
	}

	@Override
	public ParseRuleResult<LogicalExpression> apply(
			Category<LogicalExpression> left, Category<LogicalExpression> right, SentenceSpan span) {

		if (!left.getSyntax().equals(Syntax.AP)
				|| !right.getSyntax().equals(Syntax.AP)) {
			return null;
		}

		final ParseRuleResult<LogicalExpression> first = doApplication(
				workerCategory, left, false);
		if (first == null) {
			return null;
		}
		return doApplication(first.getResultCategory(), right, false);
	}
}
