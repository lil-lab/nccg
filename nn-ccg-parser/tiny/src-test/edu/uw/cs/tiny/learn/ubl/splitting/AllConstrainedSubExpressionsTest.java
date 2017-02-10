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
package edu.uw.cs.tiny.learn.ubl.splitting;

import java.util.Set;

import junit.framework.Assert;

import org.junit.Test;

import edu.uw.cs.lil.tiny.genlex.ccg.unification.split.AllConstrainedSubExpressions;
import edu.uw.cs.lil.tiny.mr.lambda.LogicalExpression;
import edu.uw.cs.tiny.LogicalExpressionTestServices;

public class AllConstrainedSubExpressionsTest {
	
	@Test
	public void test() {
		final Set<LogicalExpression> subs = AllConstrainedSubExpressions
				.of(LogicalExpressionTestServices
						.getCategoryServices()
						.parseSemantics(
								"(lambda $0:e (and:<t*,t> (or:<t*,t> (boo:<t,t> goo:t) (boo:<t,t> loo:t)) (foo:<t,t> goo:t)))"));
		Assert.assertTrue(subs.contains(LogicalExpressionTestServices
				.getCategoryServices().parseSemantics("goo:t")));
		Assert.assertTrue(subs
				.contains(LogicalExpressionTestServices
						.getCategoryServices()
						.parseSemantics(
								"(lambda $0:e (and:<t*,t> (or:<t*,t> (boo:<t,t> goo:t) (boo:<t,t> loo:t)) (foo:<t,t> goo:t)))")));
		Assert.assertTrue(subs.contains(LogicalExpressionTestServices
				.getCategoryServices().parseSemantics("(foo:<t,t> goo:t)")));
		Assert.assertTrue(subs
				.contains(LogicalExpressionTestServices
						.getCategoryServices()
						.parseSemantics(
								"(and:<t*,t> (or:<t*,t> (boo:<t,t> goo:t) (boo:<t,t> loo:t)) (foo:<t,t> goo:t))")));
		Assert.assertFalse(subs.contains(LogicalExpressionTestServices
				.getCategoryServices().parseSemantics("boo:<t,t>")));
		Assert.assertFalse(subs.contains(LogicalExpressionTestServices
				.getCategoryServices().parseSemantics("loo:t")));
	}
	
}
