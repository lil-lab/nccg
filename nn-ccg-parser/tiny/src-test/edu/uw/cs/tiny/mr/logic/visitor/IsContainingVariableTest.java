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

import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

import org.junit.Test;

import edu.uw.cs.lil.tiny.mr.lambda.Lambda;
import edu.uw.cs.lil.tiny.mr.lambda.LogicLanguageServices;
import edu.uw.cs.lil.tiny.mr.lambda.LogicalExpression;
import edu.uw.cs.lil.tiny.mr.lambda.Variable;
import edu.uw.cs.lil.tiny.mr.lambda.visitor.IsContainingVariable;
import edu.uw.cs.lil.tiny.mr.lambda.visitor.ReplaceExpression;
import edu.uw.cs.tiny.LogicalExpressionTestServices;

public class IsContainingVariableTest {
	
	@Test
	public void test() {
		final LogicalExpression exp = LogicalExpressionTestServices
				.getCategoryServices()
				.parseSemantics(
						"(lambda $0:e (lambda $1:<e,t> (lambda $2:e (and:<t*,t> ($1 $2) (boo:<e,<e,t>> $0 $2)))))");
		assertFalse(IsContainingVariable.of(exp, new Variable(
				LogicLanguageServices.getTypeRepository().getEntityType())));
		assertTrue(IsContainingVariable.of(exp, ((Lambda) exp).getArgument()));
		assertTrue(IsContainingVariable.of(((Lambda) exp).getBody(),
				((Lambda) exp).getArgument()));
		final Variable var = new Variable(LogicLanguageServices
				.getTypeRepository().getEntityType());
		assertTrue(IsContainingVariable.of(
				ReplaceExpression.of(((Lambda) exp).getBody(),
						((Lambda) exp).getArgument(), var), var));
		
	}
	
}
