package edu.uw.cs.lil.amr.features;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Iterator;
import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;

import edu.cornell.cs.nlp.spf.base.hashvector.IHashVector;
import edu.cornell.cs.nlp.spf.base.hashvector.KeyArgs;
import edu.cornell.cs.nlp.spf.ccg.lexicon.LexicalEntry;
import edu.cornell.cs.nlp.spf.ccg.lexicon.factored.lambda.FactoredLexicalEntry;
import edu.cornell.cs.nlp.spf.data.situated.sentence.SituatedSentence;
import edu.cornell.cs.nlp.spf.explat.IResourceRepository;
import edu.cornell.cs.nlp.spf.explat.ParameterizedExperiment.Parameters;
import edu.cornell.cs.nlp.spf.explat.resources.IResourceObjectCreator;
import edu.cornell.cs.nlp.spf.explat.resources.usage.ResourceUsage;
import edu.cornell.cs.nlp.spf.mr.lambda.LogicalExpression;
import edu.cornell.cs.nlp.spf.parser.ccg.ILexicalParseStep;
import edu.cornell.cs.nlp.spf.parser.ccg.IParseStep;
import edu.cornell.cs.nlp.spf.parser.ccg.model.lexical.ILexicalFeatureSet;
import edu.uw.cs.lil.amr.data.AMRMeta;
import edu.uw.cs.lil.amr.lambda.AMRServices;
import edu.uw.cs.lil.amr.learn.genlex.TemplateLemmaGenlex;

/**
 * Features to trigger on lexemes that can be generated with lemmatization.
 *
 * @author Yoav Artzi
 * @see TemplateLemmaGenlex
 */
@Deprecated
public class LemmaLexemeFeatures implements
		ILexicalFeatureSet<SituatedSentence<AMRMeta>, LogicalExpression> {
	private static final String	DEFAULT_FEATURE_TAG	= "LEMGEN";

	private static final long	serialVersionUID	= 2122761115861040245L;

	private final String		featureTag;

	public LemmaLexemeFeatures(String featureTag) {
		this.featureTag = featureTag;
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
	public void setFeatures(IParseStep<LogicalExpression> step,
			IHashVector features, SituatedSentence<AMRMeta> dataItem) {
		if (step instanceof ILexicalParseStep) {
			final LexicalEntry<LogicalExpression> entry = ((ILexicalParseStep<LogicalExpression>) step)
					.getLexicalEntry();
			if (entry instanceof FactoredLexicalEntry) {
				// Get the "lemmatized" form of all the constants used in the
				// lexeme.
				final List<String> constantLemmas = new ArrayList<>(
						((FactoredLexicalEntry) entry).getLexeme()
								.getConstants().stream()
								.map(AMRServices::lemmatizeConstant)
								.collect(Collectors.toList()));
				// Iterate over the word, for each one, try to find the constant
				// that "generated" it and remove it from the list. If a word is
				// not generated by a constant, simply return.
				for (int i = step.getStart(); i <= step.getEnd(); ++i) {
					final Set<String> lemmas = dataItem.getState().getLemmas(i);
					final Iterator<String> iterator = constantLemmas.iterator();
					boolean found = false;
					while (iterator.hasNext()) {
						if (lemmas.contains(iterator.next())) {
							iterator.remove();
							found = true;
							break;
						}
					}
					if (!found) {
						return;
					}
				}

				// All words were associated with constants (this entry can be
				// generated using lemma generators).
				features.set(featureTag, 1.0);
			}
		}
	}

	public static class Creator implements
			IResourceObjectCreator<LemmaLexemeFeatures> {

		private final String	type;

		public Creator() {
			this("feat.lex.lemma");
		}

		public Creator(String type) {
			this.type = type;
		}

		@Override
		public LemmaLexemeFeatures create(Parameters params,
				IResourceRepository repo) {
			return new LemmaLexemeFeatures(params.get("tag",
					DEFAULT_FEATURE_TAG));
		}

		@Override
		public String type() {
			return type;
		}

		@Override
		public ResourceUsage usage() {
			return ResourceUsage
					.builder(type, LemmaLexemeFeatures.class)
					.setDescription(
							"Features to trigger on lexemes that can be generated with lemmatization, crossed with the POS tags of the words")
					.addParam(
							"tag",
							String.class,
							"Feature tag (default: " + DEFAULT_FEATURE_TAG
									+ ")").build();
		}

	}

}
