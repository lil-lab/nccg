package edu.uw.cs.lil.amr.features.decomposed;

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
import edu.cornell.cs.nlp.spf.parser.ccg.ILexicalParseStep;
import edu.cornell.cs.nlp.spf.parser.ccg.IParseStep;
import edu.cornell.cs.nlp.spf.parser.ccg.model.parse.IParseFeatureSet;
import edu.uw.cs.lil.amr.data.AMRMeta;

/**
 * Syntactic attribute feature. Includes only the syntactic attribute and fires
 * only for lexical steps. If the step includes multiple attributes, will create
 * a feature for the each one. The feature is additive.
 *
 * @author Yoav Artzi
 *
 */
public class SyntacticAttributeFeatureSet implements
		IParseFeatureSet<SituatedSentence<AMRMeta>, LogicalExpression> {

	private static final String	DEFAULT_TAG			= "ATTRIB";

	private static final long	serialVersionUID	= -2744936571924653886L;

	private final String		tag;

	public SyntacticAttributeFeatureSet(String tag) {
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
			for (final String attribute : step.getRoot().getSyntax()
					.getAttributes()) {
				feats.add(tag, attribute, 1.0);
			}
		}
	}

	public static class Creator
			implements IResourceObjectCreator<SyntacticAttributeFeatureSet> {

		private final String type;

		public Creator() {
			this("feat.decomposed.syntactic.attribute");
		}

		public Creator(String type) {
			this.type = type;
		}

		@Override
		public SyntacticAttributeFeatureSet create(Parameters params,
				IResourceRepository repo) {
			return new SyntacticAttributeFeatureSet(
					params.get("tag", DEFAULT_TAG));
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
							"Syntactic attribute feature. Includes only the syntactic attribute and fires only for lexical steps. If the step includes multiple attributes, will create a feature for the each one. The feature is additive.")
					.build();
		}

	}

}
