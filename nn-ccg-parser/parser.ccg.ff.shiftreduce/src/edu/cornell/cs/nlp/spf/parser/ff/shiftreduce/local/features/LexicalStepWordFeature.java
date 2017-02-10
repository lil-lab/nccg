package edu.cornell.cs.nlp.spf.parser.ff.shiftreduce.local.features;

import java.util.Collections;
import java.util.Set;

import edu.cornell.cs.nlp.spf.base.hashvector.IHashVector;
import edu.cornell.cs.nlp.spf.base.hashvector.KeyArgs;
import edu.cornell.cs.nlp.spf.base.token.TokenSeq;
import edu.cornell.cs.nlp.spf.ccg.lexicon.LexicalEntry;
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

public class LexicalStepWordFeature implements
			IParseFeatureSet<SituatedSentence<AMRMeta>, LogicalExpression> {

	private static final long serialVersionUID 		= -2228447872453428215L;
	private static final String	DEFAULT_TAG			= "LEXICALWORD";
	private final String tag;

	public LexicalStepWordFeature(String tag) {
		this.tag = tag;
	}

	/** Triggers on words in the lexical entry in a lexical step */
	@Override
	public void setFeatures(IParseStep<LogicalExpression> parseStep, IHashVector feats,
									SituatedSentence<AMRMeta> dataItem) {
		if (parseStep instanceof ILexicalParseStep) {
			final LexicalEntry<LogicalExpression> entry = ((ILexicalParseStep<LogicalExpression>) parseStep)
																.getLexicalEntry();
			TokenSeq tk = entry.getTokens();
			final int n = tk.size();
			
			for(int i = 0; i < n; i++) {
				feats.add(this.tag, tk.get(i), 1.0);
			}
		}
	}
	
	@Override
	public Set<KeyArgs> getDefaultFeatures() {
		return Collections.emptySet(); //not sure how this works
	}

	public static class Creator implements IResourceObjectCreator<LexicalStepWordFeature> {

		private final String type;

		public Creator() {
			this("feat.lexical.word");
		}

		public Creator(String type) {
			this.type = type;
		}

		@Override
		public LexicalStepWordFeature create(Parameters params,
			IResourceRepository repo) {
			return new LexicalStepWordFeature(params.get("tag", DEFAULT_TAG));
		}

		@Override
		public String type() {
			return type;
		}

		@Override
		public ResourceUsage usage() {
			return ResourceUsage.builder(type, LexicalStepWordFeature.class)
					.setDescription(
							"Feature that simply triggers on the words in the lexical step")
					.addParam("tag", String.class,
							"Feature set tag (default: " + DEFAULT_TAG + ")")
					.build();
		}
	}
}