package edu.cornell.cs.nlp.spf.parser.ff.shiftreduce.nonlocal.features;

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
import edu.cornell.cs.nlp.spf.parser.ccg.shiftreduce.stacks.DerivationState;
import edu.cornell.cs.nlp.spf.parser.ccg.shiftreduce.steps.IWeightedShiftReduceStep;
import edu.cornell.cs.nlp.spf.parser.ccg.shiftreduce.steps.ShiftReduceLexicalStep;
import edu.cornell.cs.nlp.utils.collections.ISerializableScorer;
import edu.cornell.cs.nlp.utils.filter.IFilter;
import edu.cornell.cs.nlp.utils.function.PredicateUtils;
import edu.uw.cs.lil.amr.data.AMRMeta;
import edu.uw.cs.lil.amr.features.AmrLexicalFeatures;

/** Non-local feature that is fired on the template used by the previous rule. 
 * This should potentially help us when sequence of lexical steps are performed
 * and common patterns occur in template. */
public class AdjacentTemplateFeature<MR>
					extends FactoredLexicalFeatureSet<SituatedSentence<AMRMeta>> 
					implements AbstractNonLocalFeature<MR> {
	
	private static final long serialVersionUID = -8783463952701325435L;
	private static final String	DEFAULT_FEATURE_TAG	= "PREVTEMPLATE";
	
	protected AdjacentTemplateFeature(
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
	
	@Override
	public void add(DerivationState<MR> state, IParseStep<MR> parseStep, IHashVector features, 
						String[] buffer, int bufferIndex, String[] tags) {
	
		final IWeightedShiftReduceStep<MR> step = state.returnStep();
		if(step == null || !step.getRuleName().equals(ShiftReduceLexicalStep.LEXICAL_DERIVATION_STEP_RULENAME)) {
			return;
		}
		
		IParseStep<MR> prevStep = step.getUnderlyingParseStep();
		if (prevStep instanceof ILexicalParseStep) {
			@SuppressWarnings("unchecked")
			final LexicalEntry<LogicalExpression> entry = ((ILexicalParseStep<LogicalExpression>) prevStep)
																.getLexicalEntry();
			if (entry instanceof FactoredLexicalEntry
					&& ignoreFilter.test(entry) && !entry.isDynamic()) {
				final Integer index = getTemplateId(
						((FactoredLexicalEntry) entry).getTemplate());
				if (index != null) {
					features.add(DEFAULT_FEATURE_TAG, index.toString(), 1.0);
				}
			}
		}
	}
	
	@Override
	public void setFeatures(IParseStep<LogicalExpression> parseStep, IHashVector feats,
			SituatedSentence<AMRMeta> dataItem) {
		throw new RuntimeException("Operation not supported. Use non-local feature interface.");
	}

	@Override
	public Set<KeyArgs> getDefaultFeatures() {
		return Collections.emptySet(); //not sure how this works
	}

	public static class Builder<MR> {

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

		public AdjacentTemplateFeature<MR> build() {
			return new AdjacentTemplateFeature<MR>(ignoreFilter, entryInitialScorer,
					entryScale, featureTag, lexemeInitialScorer, lexemeScale,
					templateInitialScorer, templateScale, synScale,
					computeSyntaxAttributeFeatures);
		}

		public Builder<MR> setComputeSyntaxAttributeFeatures(
				boolean computeSyntaxAttributeFeatures) {
			this.computeSyntaxAttributeFeatures = computeSyntaxAttributeFeatures;
			return this;
		}

		public Builder<MR> setEntryInitialScorer(
				ISerializableScorer<LexicalEntry<LogicalExpression>> entryInitialScorer) {
			this.entryInitialScorer = entryInitialScorer;
			return this;
		}

		public Builder<MR> setEntryScale(double entryScale) {
			this.entryScale = entryScale;
			return this;
		}

		public Builder<MR> setFeatureTag(String featureTag) {
			this.featureTag = featureTag;
			return this;
		}

		public Builder<MR> setIgnoreFilter(
				Predicate<LexicalEntry<LogicalExpression>> ignoreFilter) {
			this.ignoreFilter = ignoreFilter;
			return this;
		}

		public Builder<MR> setLexemeInitialScorer(
				ISerializableScorer<Lexeme> lexemeInitialScorer) {
			this.lexemeInitialScorer = lexemeInitialScorer;
			return this;
		}

		public Builder<MR> setLexemeScale(double lexemeScale) {
			this.lexemeScale = lexemeScale;
			return this;
		}

		public Builder<MR> setSynScale(double synScale) {
			this.synScale = synScale;
			return this;
		}

		public Builder<MR> setTemplateInitialScorer(
				ISerializableScorer<LexicalTemplate> templateInitialScorer) {
			this.templateInitialScorer = templateInitialScorer;
			return this;
		}

		public Builder<MR> setTemplateScale(double templateScale) {
			this.templateScale = templateScale;
			return this;
		}

	}

	public static class Creator<MR>
			implements IResourceObjectCreator<AdjacentTemplateFeature<MR>> {

		private final String type;

		public Creator() {
			this("feat.adjacent.template");
		}

		public Creator(String type) {
			this.type = type;
		}

		@Override
		public AdjacentTemplateFeature<MR> create(Parameters params,
				IResourceRepository repo) {

			final Builder<MR> builder = new Builder<MR>();

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
