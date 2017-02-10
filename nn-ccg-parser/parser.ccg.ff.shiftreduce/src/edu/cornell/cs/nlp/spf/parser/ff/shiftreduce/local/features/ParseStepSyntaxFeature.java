package edu.cornell.cs.nlp.spf.parser.ff.shiftreduce.local.features;

import java.util.Collections;
import java.util.Set;
import java.util.stream.Collectors;

import edu.cornell.cs.nlp.spf.base.hashvector.IHashVector;
import edu.cornell.cs.nlp.spf.base.hashvector.KeyArgs;
import edu.cornell.cs.nlp.spf.ccg.categories.syntax.Syntax;
import edu.cornell.cs.nlp.spf.data.situated.sentence.SituatedSentence;
import edu.cornell.cs.nlp.spf.explat.IResourceRepository;
import edu.cornell.cs.nlp.spf.explat.ParameterizedExperiment.Parameters;
import edu.cornell.cs.nlp.spf.explat.resources.IResourceObjectCreator;
import edu.cornell.cs.nlp.spf.explat.resources.usage.ResourceUsage;
import edu.cornell.cs.nlp.spf.mr.lambda.LogicalExpression;
import edu.cornell.cs.nlp.spf.parser.ccg.IParseStep;
import edu.cornell.cs.nlp.spf.parser.ccg.model.parse.IParseFeatureSet;
import edu.uw.cs.lil.amr.data.AMRMeta;

/** Feature that triggers on the stripped syntax and accumulate attribute of the parse step category. */
public class ParseStepSyntaxFeature implements
				IParseFeatureSet<SituatedSentence<AMRMeta>, LogicalExpression> {
	
	private static final long serialVersionUID 		= 1268207102276391425L;
	private static final String	SYNTAXTAG			= "PARSESTEPSYNTAX";
	private static final String	ATTRIBTAG			= "PARSESTEPATTRIB";
	
	public ParseStepSyntaxFeature() {
	}
	
	/** Triggers on the rule used by the parser */
	@Override
	public void setFeatures(IParseStep<LogicalExpression> parseStep, IHashVector feats,
			SituatedSentence<AMRMeta> dataItem) {
			
		final Syntax syntax = parseStep.getRoot().getSyntax();
		
		feats.add(SYNTAXTAG, syntax.stripAttributes().toString(), 1.0);
		final String attributeSeq = syntax.getAttributes().stream().sorted()
							.collect(Collectors.joining("+"));
		feats.add(ATTRIBTAG, attributeSeq, 1.0);
	}

	@Override
	public Set<KeyArgs> getDefaultFeatures() {
		return Collections.emptySet(); //not sure how this works
	}
	
	public static class Creator implements IResourceObjectCreator<ParseStepSyntaxFeature> {

		private final String type;
		
		public Creator() {
			this("feat.parse.step.syntax");
		}
		
		public Creator(String type) {
			this.type = type;
		}
		
		@Override
		public ParseStepSyntaxFeature create(Parameters params,
				IResourceRepository repo) {
			return new ParseStepSyntaxFeature();
		}
		
		@Override
		public String type() {
			return type;
		}
		
		@Override
		public ResourceUsage usage() {
			return ResourceUsage.builder(type, StepRuleFeature.class)
					.setDescription(
							"Feature that computes stripped syntax and attribute of root of parse step")
					.build();
		}
	}

}