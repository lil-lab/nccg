package edu.uw.cs.lil.amr.parser.rules.amrspecials.keywords;

import edu.cornell.cs.nlp.spf.ccg.categories.Category;
import edu.cornell.cs.nlp.spf.explat.IResourceRepository;
import edu.cornell.cs.nlp.spf.explat.ParameterizedExperiment.Parameters;
import edu.cornell.cs.nlp.spf.explat.resources.IResourceObjectCreator;
import edu.cornell.cs.nlp.spf.explat.resources.usage.ResourceUsage;
import edu.cornell.cs.nlp.spf.mr.lambda.Literal;
import edu.cornell.cs.nlp.spf.mr.lambda.LogicalExpression;
import edu.cornell.cs.nlp.spf.parser.ccg.rules.IUnaryParseRule;
import edu.cornell.cs.nlp.spf.parser.ccg.rules.ParseRuleResult;
import edu.cornell.cs.nlp.spf.parser.ccg.rules.SentenceSpan;
import edu.cornell.cs.nlp.spf.parser.ccg.rules.UnaryRuleName;
import edu.uw.cs.lil.amr.lambda.AMRServices;

/**
 * Rule to create a single keyword: A:f -> KEY:skolemize(f), where A:f is a
 * valid keyword according to {@link KeywordUtil#isValidCategory(Category)} and
 * this is a single word.
 *
 * @author Yoav Artzi
 */
@Deprecated
public class ShiftToKeyword implements IUnaryParseRule<LogicalExpression> {

	private static final String	LABEL				= "keyword";

	private static final long	serialVersionUID	= 1140251107175394817L;

	private final UnaryRuleName	name				= UnaryRuleName
															.create(LABEL);

	@Override
	public ParseRuleResult<LogicalExpression> apply(
			Category<LogicalExpression> category, SentenceSpan span) {
		if (isValidArgument(category, span)) {
			final Literal skolemized = AMRServices.skolemize(category
					.getSemantics());
			if (skolemized != null) {
				return new ParseRuleResult<>(name, Category.create(
						AMRServices.KEY, skolemized));
			}
		}
		return null;
	}

	@Override
	public UnaryRuleName getName() {
		return name;
	}

	@Override
	public boolean isValidArgument(Category<LogicalExpression> category,
			SentenceSpan span) {
		return span.length() == 1 && KeywordUtil.isValidCategory(category);
	}

	public static class Creator implements
			IResourceObjectCreator<ShiftToKeyword> {

		private final String	type;

		public Creator() {
			this("rule.shift.amr.keyword.create");
		}

		public Creator(String type) {
			this.type = type;
		}

		@Override
		public ShiftToKeyword create(Parameters params, IResourceRepository repo) {
			return new ShiftToKeyword();
		}

		@Override
		public String type() {
			return type;
		}

		@Override
		public ResourceUsage usage() {
			return ResourceUsage
					.builder(type, ShiftToKeyword.class)
					.setDescription(
							"Rule to create a single keyword: A:f -> KEY:skolemize(f), where A:f is a"
									+ " valid keyword according to {@link KeywordUtil#isValidCategory(Category)} and"
									+ " this is a single word.").build();
		}

	}

}
