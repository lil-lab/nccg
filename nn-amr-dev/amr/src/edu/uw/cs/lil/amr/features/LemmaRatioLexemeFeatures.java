package edu.uw.cs.lil.amr.features;

import java.util.Collections;
import java.util.Set;

import edu.cornell.cs.nlp.spf.base.hashvector.IHashVector;
import edu.cornell.cs.nlp.spf.base.hashvector.KeyArgs;
import edu.cornell.cs.nlp.spf.ccg.lexicon.LexicalEntry;
import edu.cornell.cs.nlp.spf.ccg.lexicon.factored.lambda.FactoredLexicalEntry;
import edu.cornell.cs.nlp.spf.ccg.lexicon.factored.lambda.Lexeme;
import edu.cornell.cs.nlp.spf.data.situated.sentence.SituatedSentence;
import edu.cornell.cs.nlp.spf.explat.IResourceRepository;
import edu.cornell.cs.nlp.spf.explat.ParameterizedExperiment.Parameters;
import edu.cornell.cs.nlp.spf.explat.resources.IResourceObjectCreator;
import edu.cornell.cs.nlp.spf.explat.resources.usage.ResourceUsage;
import edu.cornell.cs.nlp.spf.mr.lambda.LogicalConstant;
import edu.cornell.cs.nlp.spf.mr.lambda.LogicalExpression;
import edu.cornell.cs.nlp.spf.parser.ccg.ILexicalParseStep;
import edu.cornell.cs.nlp.spf.parser.ccg.IParseStep;
import edu.cornell.cs.nlp.spf.parser.ccg.model.lexical.ILexicalFeatureSet;
import edu.uw.cs.lil.amr.data.AMRMeta;
import edu.uw.cs.lil.amr.lambda.AMRServices;

/**
 * Features to measure the ratio of constants that can be generated from
 * lemmatization of the words. Each POS tag gets a different feature.
 *
 * @author Yoav Artzi
 */
@Deprecated
public class LemmaRatioLexemeFeatures implements
		ILexicalFeatureSet<SituatedSentence<AMRMeta>, LogicalExpression> {
	private static final String	DEFAULT_FEATURE_TAG	= "LEMRATIO";

	private static final long	serialVersionUID	= -4255081684861791404L;

	private final String		featureTag;

	private final double		scale;

	public LemmaRatioLexemeFeatures(String featureTag, double scale) {
		this.featureTag = featureTag;
		this.scale = scale;
	}

	@Override
	public boolean addEntry(LexicalEntry<LogicalExpression> entry,
			IHashVector parametersVector) {
		// Nothing to do.
		return false;
	}

	@Override
	public Set<KeyArgs> getDefaultFeatures() {
		return Collections.emptySet();
	}

	@Override
	public void setFeatures(IParseStep<LogicalExpression> parseStep,
			IHashVector feats, SituatedSentence<AMRMeta> dataItem) {
		if (parseStep instanceof ILexicalParseStep
				&& ((ILexicalParseStep<LogicalExpression>) parseStep)
						.getLexicalEntry() instanceof FactoredLexicalEntry) {
			final Lexeme lexeme = ((FactoredLexicalEntry) ((ILexicalParseStep<LogicalExpression>) parseStep)
					.getLexicalEntry()).getLexeme();
			final AMRMeta meta = dataItem.getState();
			final int start = parseStep.getStart();
			final int numTokens = lexeme.getTokens().size();
			final boolean[] tokenFlags = new boolean[numTokens];
			int found = 0;
			for (final LogicalConstant constant : lexeme.getConstants()) {
				final String constantLemma = AMRServices
						.lemmatizeConstant(constant);
				for (int i = 0; i < numTokens; ++i) {
					if (!tokenFlags[i]) {
						if (meta.getLemmas(start + i).contains(constantLemma)) {
							++found;
							tokenFlags[i] = true;
							break;
						}
					}

				}
			}
			if (found == 0) {
				return;
			} else {
				final double ratio = (double) found / numTokens;
				final String posSeq = meta.getTags()
						.sub(start, parseStep.getEnd() + 1).toString("+");
				feats.set(featureTag, posSeq, ratio * scale);
			}
		}
	}

	public static class Creator implements
			IResourceObjectCreator<LemmaRatioLexemeFeatures> {

		private final String	type;

		public Creator() {
			this("feat.lex.lemma.ratio");
		}

		public Creator(String type) {
			this.type = type;
		}

		@Override
		public LemmaRatioLexemeFeatures create(Parameters params,
				IResourceRepository repo) {
			return new LemmaRatioLexemeFeatures(params.get("tag",
					DEFAULT_FEATURE_TAG), params.getAsDouble("scale", 1.0));
		}

		@Override
		public String type() {
			return type;
		}

		@Override
		public ResourceUsage usage() {
			return ResourceUsage
					.builder(type, LemmaRatioLexemeFeatures.class)
					.setDescription(
							"Features to measure the ratio of constants that can be generated from lemmatization of the words.")
					.addParam("scale", Double.class,
							"Scaling factor (default: 1.0)")
					.addParam(
							"tag",
							String.class,
							"Feature tag (default: " + DEFAULT_FEATURE_TAG
									+ ")").build();
		}

	}

}
