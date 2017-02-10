package edu.uw.cs.lil.amr.features.decomposed;

import java.util.Collections;
import java.util.Set;

import edu.cornell.cs.nlp.spf.base.hashvector.IHashVector;
import edu.cornell.cs.nlp.spf.base.hashvector.KeyArgs;
import edu.cornell.cs.nlp.spf.base.token.TokenSeq;
import edu.cornell.cs.nlp.spf.data.situated.sentence.SituatedSentence;
import edu.cornell.cs.nlp.spf.explat.IResourceRepository;
import edu.cornell.cs.nlp.spf.explat.ParameterizedExperiment.Parameters;
import edu.cornell.cs.nlp.spf.explat.resources.IResourceObjectCreator;
import edu.cornell.cs.nlp.spf.explat.resources.usage.ResourceUsage;
import edu.cornell.cs.nlp.spf.mr.lambda.LogicalExpression;
import edu.cornell.cs.nlp.spf.parser.ccg.ILexicalParseStep;
import edu.cornell.cs.nlp.spf.parser.ccg.IParseStep;
import edu.cornell.cs.nlp.spf.parser.ccg.model.parse.IParseFeatureSet;
import edu.uw.cs.lil.amr.data.AMRMeta;

/**
 * POS tag feature. Includes only the part-of-speech tag and fires only for
 * lexical steps. If the step includes multiple tokens, will create a feature
 * for the tag of each one. The feature is additive.
 *
 * @author Yoav Artzi
 *
 */
public class PosTagFeatureSet implements
		IParseFeatureSet<SituatedSentence<AMRMeta>, LogicalExpression> {

	private static final String	DEFAULT_TAG			= "POS";
	private static final long	serialVersionUID	= 1418061310817764933L;
	private final String		tag;

	public PosTagFeatureSet(String tag) {
		this.tag = tag;
	}

	@Override
	public Set<KeyArgs> getDefaultFeatures() {
		// Nothing to do.
		return Collections.emptySet();
	}

	@Override
	public void setFeatures(IParseStep<LogicalExpression> step,
			IHashVector feats, SituatedSentence<AMRMeta> dataItem) {
		if (step instanceof ILexicalParseStep) {
			final TokenSeq tags = dataItem.getState().getTags();
			if (tags != null) {
				for (final String pos : tags.subList(step.getStart(),
						step.getEnd() + 1)) { //the +1  was missing
					feats.add(tag, pos, 1.0);
				}
			}
		}
	}

	public static class Creator
			implements IResourceObjectCreator<PosTagFeatureSet> {

		private final String type;

		public Creator() {
			this("feat.decomposed.pos");
		}

		public Creator(String type) {
			this.type = type;
		}

		@Override
		public PosTagFeatureSet create(Parameters params,
				IResourceRepository repo) {
			return new PosTagFeatureSet(params.get("tag", DEFAULT_TAG));
		}

		@Override
		public String type() {
			return type;
		}

		@Override
		public ResourceUsage usage() {
			return ResourceUsage.builder(type, PosTagFeatureSet.class)
					.addParam("tag", String.class,
							"Feature set tag (default: " + DEFAULT_TAG + ")")
					.setDescription(
							"POS tag feature. Includes only the part-of-speech tag and fires only for lexical steps. If the step includes multiple tokens, will create a feature for the tag of each one. The feature is additive.")
					.build();
		}

	}

}
