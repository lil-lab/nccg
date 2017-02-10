package edu.cornell.cs.nlp.spf.parser.ff.shiftreduce.local.features;

import java.util.Collections;
import java.util.Set;

import edu.cornell.cs.nlp.spf.base.hashvector.IHashVector;
import edu.cornell.cs.nlp.spf.base.hashvector.KeyArgs;
import edu.cornell.cs.nlp.spf.data.situated.sentence.SituatedSentence;
import edu.cornell.cs.nlp.spf.explat.IResourceRepository;
import edu.cornell.cs.nlp.spf.explat.ParameterizedExperiment.Parameters;
import edu.cornell.cs.nlp.spf.explat.resources.IResourceObjectCreator;
import edu.cornell.cs.nlp.spf.explat.resources.usage.ResourceUsage;
import edu.cornell.cs.nlp.spf.mr.lambda.LogicalExpression;
import edu.cornell.cs.nlp.spf.parser.ccg.IParseStep;
import edu.cornell.cs.nlp.spf.parser.ccg.model.parse.IParseFeatureSet;
import edu.uw.cs.lil.amr.data.AMRMeta;

public class StepRuleFeature implements
				IParseFeatureSet<SituatedSentence<AMRMeta>, LogicalExpression> {
	
	private static final long serialVersionUID = -8037645333455211383L;
	private static final String	DEFAULT_TAG			= "STEPRULE";
	private final String		tag;
	
	public StepRuleFeature(String tag) {
		this.tag = tag;
	}
	
	/** Triggers on the rule used by the parser */
	@Override
	public void setFeatures(IParseStep<LogicalExpression> parseStep, IHashVector feats,
			SituatedSentence<AMRMeta> dataItem) {
			
		final String ruleName = parseStep.getRuleName().toString();
		feats.add(tag, "unigram", ruleName, 1.0);		
	}

	@Override
	public Set<KeyArgs> getDefaultFeatures() {
		return Collections.emptySet(); //not sure how this works
	}
	
	public static class Creator implements IResourceObjectCreator<StepRuleFeature> {

		private final String type;
		
		public Creator() {
			this("feat.step.rule");
		}
		
		public Creator(String type) {
			this.type = type;
		}
		
		@Override
		public StepRuleFeature create(Parameters params,
				IResourceRepository repo) {
			return new StepRuleFeature(params.get("tag", DEFAULT_TAG));
		}
		
		@Override
		public String type() {
			return type;
		}
		
		@Override
		public ResourceUsage usage() {
			return ResourceUsage.builder(type, StepRuleFeature.class)
					.setDescription(
							"Feature that simply computes rule name of the step")
					.addParam("tag", String.class,
							"Feature set tag (default: " + DEFAULT_TAG + ")")
					.build();
		}
	}

}
