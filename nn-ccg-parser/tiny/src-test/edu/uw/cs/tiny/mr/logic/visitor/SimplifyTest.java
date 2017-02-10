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
package edu.uw.cs.tiny.mr.logic.visitor;

import static org.junit.Assert.assertTrue;

import org.junit.Test;

import edu.uw.cs.lil.tiny.mr.lambda.LogicalExpression;
import edu.uw.cs.lil.tiny.mr.lambda.visitor.Simplify;
import edu.uw.cs.tiny.LogicalExpressionTestServices;

public class SimplifyTest {
	
	@Test
	public void test() {
		final LogicalExpression exp = LogicalExpressionTestServices
				.getCategoryServices()
				.parseSemantics(
						"(lambda $0:e (lambda $1:e (lambda $2:e (foo:<e,<e,<e,t>>> $0 $1 $2))))");
		final LogicalExpression result = LogicalExpressionTestServices
				.getCategoryServices().parseSemantics("foo:<e,<e,<e,t>>>");
		assertTrue(String.format("Expected: %s\nGot: %s", result,
				Simplify.of(exp)), result.equals(Simplify.of(exp)));
	}
	
	@Test
	public void test2() {
		final LogicalExpression exp = LogicalExpressionTestServices
				.getCategoryServices()
				.parseSemantics(
						"(lambda $0:e (lambda $1:e (lambda $2:e (and:<t*,t> (foo:<e,<e,<e,t>>> $0 $1 $2) ((lambda $3:e true:t) b:e)))))");
		final LogicalExpression result = LogicalExpressionTestServices
				.getCategoryServices().parseSemantics("foo:<e,<e,<e,t>>>");
		assertTrue(String.format("Expected: %s\nGot: %s", result,
				Simplify.of(exp)), result.equals(Simplify.of(exp)));
	}
	
	@Test
	public void test3() {
		final LogicalExpression exp = LogicalExpressionTestServices
				.getCategoryServices().parseSemantics(
						"(lambda $0:<e,t> (right:<<e,t>,<e,t>> $0))");
		final LogicalExpression result = LogicalExpressionTestServices
				.getCategoryServices().parseSemantics("right:<<e,t>,<e,t>>");
		assertTrue(String.format("Expected: %s\nGot: %s", result,
				Simplify.of(exp)), result.equals(Simplify.of(exp)));
	}
	
	@Test
	public void test4() {
		final LogicalExpression exp = LogicalExpressionTestServices
				.getCategoryServices().parseSemantics(
						"(lambda $0:e (#0<e,<e,e>>:<e,<e,e>> #1e:e $0))");
		final LogicalExpression result = LogicalExpressionTestServices
				.getCategoryServices().parseSemantics(
						"(#0<e,<e,e>>:<e,<e,e>> #1e:e)");
		assertTrue(String.format("Expected: %s\nGot: %s", result,
				Simplify.of(exp)), result.equals(Simplify.of(exp)));
	}
	
	@Test
	public void test5() {
		final LogicalExpression exp = LogicalExpressionTestServices
				.getCategoryServices()
				.parseSemantics(
						"(lambda $0:t (do_until:<e,<t,e>> (do:<e,e> travel:m) $0))");
		final LogicalExpression result = LogicalExpressionTestServices
				.getCategoryServices().parseSemantics(
						"(do_until:<e,<t,e>> (do:<e,e> travel:m))");
		assertTrue(String.format("Expected: %s\nGot: %s", result,
				Simplify.of(exp)), result.equals(Simplify.of(exp)));
	}
	
}
