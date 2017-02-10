package edu.cornell.cs.nlp.spf.parser.ff.shiftreduce.local.features;

import java.util.Collections;
import java.util.Set;
import java.util.function.Predicate;

import edu.cornell.cs.nlp.spf.base.hashvector.IHashVector;
import edu.cornell.cs.nlp.spf.base.hashvector.KeyArgs;
import edu.cornell.cs.nlp.spf.ccg.lexicon.LexicalEntry;
import edu.cornell.cs.nlp.spf.ccg.lexicon.factored.lambda.FactoredLexicalEntry;
import edu.cornell.cs.nlp.spf.ccg.lexicon.factored.lambda.Lexeme;
import edu.cornell.cs.nlp.spf.ccg.lexicon.factored.lambda.LexicalTemplate;
import edu.cornell.cs.nlp.spf.data.situated.sentence.SituatedSentence;
import edu.cornell.cs.nlp.spf.explat.IResourceRepository;
import edu.cornell.cs.nlp.spf.explat.ParameterizedExperiment.Parameters;
import edu.cornell.cs.nlp.spf.explat.resources.IResourceObjectCreator;
import edu.cornell.cs.nlp.spf.explat.resources.usage.ResourceUsage;
import edu.cornell.cs.nlp.spf.mr.lambda.LogicalExpression;
import edu.cornell.cs.nlp.spf.parser.ccg.ILexicalParseStep;
import edu.cornell.cs.nlp.spf.parser.ccg.IParseStep;
import edu.cornell.cs.nlp.spf.parser.ccg.factoredlex.features.FactoredLexicalFeatureSet;
import edu.cornell.cs.nlp.spf.parser.ccg.features.basic.scorer.UniformScorer;
import edu.cornell.cs.nlp.utils.collections.ISerializableScorer;
import edu.cornell.cs.nlp.utils.filter.IFilter;
import edu.cornell.cs.nlp.utils.function.PredicateUtils;
import edu.uw.cs.lil.amr.data.AMRMeta;
import edu.uw.cs.lil.amr.features.AmrLexicalFeatures;

public class TemplatAdjacentPOSFeature
					extends FactoredLexicalFeatureSet<SituatedSentence<AMRMeta>> {

	private static final long serialVersionUID = -5236247944269889599L;
	
	private static final String	DEFAULT_FEATURE_TAG	= "TEMPLATEADJACENTPOS";
	private static final String	DEFAULT_LEFT_TAG	= "TEMPLATELEFTPOS";
	private static final String	DEFAULT_RIGHT_TAG	= "TEMPLATERIGHTPOS";
	private static final String NoPOS				= "None";

	protected TemplatAdjacentPOSFeature(
			Predicate<LexicalEntry<LogicalExpression>> ignoreFilter,
			ISerializableScorer<LexicalEntry<LogicalExpression>> entryInitialScorer,
			double entryScale, String featureTag,
			ISerializableScorer<Lexeme> lexemeInitialScorer, double lexemeScale,
			ISerializableScorer<LexicalTemplate> templateInitialScorer,
			double templateScale, double posScale,
			boolean computeSyntaxAttributeFeatures) {
		super(ignoreFilter, entryInitialScorer, entryScale, featureTag,
				lexemeInitialScorer, lexemeScale, templateInitialScorer,
				templateScale, computeSyntaxAttributeFeatures);
	}
	
	/** Calculates two features: template and next word's POS and template and previous word's POS */
	@Override
	public void setFeatures(IParseStep<LogicalExpression> parseStep, IHashVector feats,
			SituatedSentence<AMRMeta> dataItem) {
		
		if (parseStep instanceof ILexicalParseStep) {
			final LexicalEntry<LogicalExpression> entry = ((ILexicalParseStep<LogicalExpression>) parseStep)
																.getLexicalEntry();
			if (entry instanceof FactoredLexicalEntry
					&& ignoreFilter.test(entry) && !entry.isDynamic()) {
				final Integer index = getTemplateId(
						((FactoredLexicalEntry) entry).getTemplate());
				if (index != null) {
					
					final String indexString = index.toString();
					
					final int n = dataItem.getTokens().size();
					final int end = parseStep.getEnd();
					
					final String nextWordPOS;

					if(end == n - 1) { //no words left
						nextWordPOS = NoPOS;
					} else { //one word left
						nextWordPOS = dataItem.getState().getTags().get(end + 1);
					} 
					
					feats.add(DEFAULT_RIGHT_TAG, indexString, nextWordPOS, 1.0);

					final String prevWordPOS;
					final int start = parseStep.getStart();
					
					if(start == 0) { //no words eaten so far
						prevWordPOS = NoPOS;
					} else { //one word left
						prevWordPOS = dataItem.getState().getTags().get(start - 1);
					} 
					
					feats.add(DEFAULT_LEFT_TAG, indexString, prevWordPOS, 1.0);
				}
			}
		}
	}

	@Override
	public Set<KeyArgs> getDefaultFeatures() {
		return Collections.emptySet(); //not sure how this works
	}
	

	public static class Builder {

		private boolean													computeSyntaxAttributeFeatures	= false;

		private ISerializableScorer<LexicalEntry<LogicalExpression>>	entryInitialScorer				= new UniformScorer<>(
				0.0);
		private double													entryScale						= 1.0;
		private String													featureTag						= DEFAULT_FEATURE_TAG;
		private Predicate<LexicalEntry<LogicalExpression>>				ignoreFilter					= PredicateUtils
				.alwaysTrue();
		private ISerializableScorer<Lexeme>								lexemeInitialScorer				= new UniformScorer<>(
				0.0);
		private double													lexemeScale						= 1.0;
		private double													synScale						= 1.0;
		private ISerializableScorer<LexicalTemplate>					templateInitialScorer			= new UniformScorer<>(
				0.0);
		private double													templateScale					= 1.0;

		public Builder() {
			// Nothing to do.
		}

		public TemplatAdjacentPOSFeature build() {
			return new TemplatAdjacentPOSFeature(ignoreFilter, entryInitialScorer,
					entryScale, featureTag, lexemeInitialScorer, lexemeScale,
					templateInitialScorer, templateScale, synScale,
					computeSyntaxAttributeFeatures);
		}

		public Builder setComputeSyntaxAttributeFeatures(
				boolean computeSyntaxAttributeFeatures) {
			this.computeSyntaxAttributeFeatures = computeSyntaxAttributeFeatures;
			return this;
		}

		public Builder setEntryInitialScorer(
				ISerializableScorer<LexicalEntry<LogicalExpression>> entryInitialScorer) {
			this.entryInitialScorer = entryInitialScorer;
			return this;
		}

		public Builder setEntryScale(double entryScale) {
			this.entryScale = entryScale;
			return this;
		}

		public Builder setFeatureTag(String featureTag) {
			this.featureTag = featureTag;
			return this;
		}

		public Builder setIgnoreFilter(
				Predicate<LexicalEntry<LogicalExpression>> ignoreFilter) {
			this.ignoreFilter = ignoreFilter;
			return this;
		}

		public Builder setLexemeInitialScorer(
				ISerializableScorer<Lexeme> lexemeInitialScorer) {
			this.lexemeInitialScorer = lexemeInitialScorer;
			return this;
		}

		public Builder setLexemeScale(double lexemeScale) {
			this.lexemeScale = lexemeScale;
			return this;
		}

		public Builder setSynScale(double synScale) {
			this.synScale = synScale;
			return this;
		}

		public Builder setTemplateInitialScorer(
				ISerializableScorer<LexicalTemplate> templateInitialScorer) {
			this.templateInitialScorer = templateInitialScorer;
			return this;
		}

		public Builder setTemplateScale(double templateScale) {
			this.templateScale = templateScale;
			return this;
		}

	}

	public static class Creator
			implements IResourceObjectCreator<TemplatAdjacentPOSFeature> {

		private final String type;

		public Creator() {
			this("feat.template.adjacent.pos");
		}

		public Creator(String type) {
			this.type = type;
		}

		@Override
		public TemplatAdjacentPOSFeature create(Parameters params,
				IResourceRepository repo) {

			final Builder builder = new Builder();

			if (params.contains("entryScorer")) {
				builder.setEntryInitialScorer(
						repo.get(params.get("entryScorer")));
			}

			if (params.contains("entryScale")) {
				builder.setEntryScale(params.getAsDouble("entryScale"));
			}

			if (params.contains("posScale")) {
				builder.setSynScale(params.getAsDouble("posScale"));
			}

			if (params.contains("tag")) {
				builder.setFeatureTag(params.get("tag"));
			}

			if (params.contains("ignoreFilter")) {
				builder.setIgnoreFilter(repo.get(params.get("ignoreFilter")));
			}

			if (params.contains("lexemeScorer")) {
				builder.setLexemeInitialScorer(
						repo.get(params.get("lexemeScorer")));
			}

			if (params.contains("lexemeScale")) {
				builder.setLexemeScale(params.getAsDouble("lexemeScale"));
			}

			if (params.contains("templateScorer")) {
				builder.setTemplateInitialScorer(
						repo.get(params.get("templateScorer")));
			}

			if (params.contains("templateScale")) {
				builder.setTemplateScale(params.getAsDouble("templateScale"));
			}

			if (params.contains("syntaxAttrib")) {
				builder.setComputeSyntaxAttributeFeatures(
						params.getAsBoolean("syntaxAttrib"));
			}

			return builder.build();
		}

		@Override
		public String type() {
			return type;
		}

		@Override
		public ResourceUsage usage() {
			return ResourceUsage.builder(type, AmrLexicalFeatures.class)
					.setDescription(
							"Lexical features for using with a factored lexicon")
					.addParam("syntaxAttrib", Boolean.class,
							"Compute syntax attribute features (default: false)")
					.addParam("entryScorer", ISerializableScorer.class,
							"Initial scorer for binary lexical entry binary features (also for lexeme-template pairings) (default: f(e) = 0.0)")
					.addParam("entryScale", Double.class,
							"Scaling factor for lexical entry binary features (also for lexeme-template pairings) (default: 1.0)")
					.addParam("tag", String.class,
							"Feature set primary tag (default: "
									+ DEFAULT_FEATURE_TAG + ")")
					.addParam("ignoreFilter", IFilter.class,
							"Filter to ignore lexical entries (default: f(e) = true, ignore nothing)")
					.addParam("lexemeScorer", ISerializableScorer.class,
							"Initial scorer for lexeme features (default: f(l) = 0.0)")
					.addParam("lexemeScale", Double.class,
							"Scaling factor for lexeme features, may be 0.0 to disable feature  (default: 1.0)")
					.addParam("posScale", Double.class,
							"Scaling factor for POS features (default: 1.0)")
					.addParam("templateScorer", ISerializableScorer.class,
							"Initial scorer for template features (default: f(l) = 0.0)")
					.addParam("templateScale", Double.class,
							"Scaling factor for template features, may be 0.0 to disable feature  (default: 1.0)")
					.build();
		}

	}
}

