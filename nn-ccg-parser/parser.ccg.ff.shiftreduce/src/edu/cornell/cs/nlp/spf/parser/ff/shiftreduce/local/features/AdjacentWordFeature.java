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

public class AdjacentWordFeature implements
			IParseFeatureSet<SituatedSentence<AMRMeta>, LogicalExpression> {

	private static final long serialVersionUID 		= -8941878324137636811L;
	private static final String	DEFAULT_TAG			= "ADJACENTWORD";	
	private static final String	ADJACENT_TAG_1		= "ADJACENTWORD1";
	private static final String	ADJACENT_TAG_2		= "ADJACENTWORD2";
	
	private static final String NoWord				= "None";

	//private final String tag;

	public AdjacentWordFeature(String tag) {
		//this.tag = tag;
	}

	/** Triggers on words in the lexical entry in a lexical step */
	@Override
	public void setFeatures(IParseStep<LogicalExpression> parseStep, IHashVector feats,
									SituatedSentence<AMRMeta> dataItem) {
	
		//Triggers on the next word in the Parsing Step
		final int end = parseStep.getEnd();
		final int n = dataItem.getTokens().size();
		
//		if(end == n - 1) {
//			feats.add(this.tag, NoWord, 1.0);
//		} else {
//			final String word = dataItem.getTokens().get(end + 1);
//			feats.add(this.tag, word, 1.0);
//		}
		
		if(end == n - 1) {
			feats.add(ADJACENT_TAG_1, NoWord, 1.0);
			feats.add(ADJACENT_TAG_2, NoWord, 1.0);
		} else if(end == n - 2) {
			final String word = dataItem.getTokens().get(end + 1);
			feats.add(ADJACENT_TAG_1, word, 1.0);
			feats.add(ADJACENT_TAG_2, NoWord, 1.0);
		} else {
			final String word1 = dataItem.getTokens().get(end + 1);
			feats.add(ADJACENT_TAG_1, word1, 1.0);
			final String word2 = dataItem.getTokens().get(end + 2);
			feats.add(ADJACENT_TAG_2, word2, 1.0);
		}
	}
	
	@Override
	public Set<KeyArgs> getDefaultFeatures() {
		return Collections.emptySet(); //not sure how this works
	}

	public static class Creator implements IResourceObjectCreator<AdjacentWordFeature> {

		private final String type;

		public Creator() {
			this("feat.adjacent.word");
		}

		public Creator(String type) {
			this.type = type;
		}

		@Override
		public AdjacentWordFeature create(Parameters params,
			IResourceRepository repo) {
			return new AdjacentWordFeature(params.get("tag", DEFAULT_TAG));
		}

		@Override
		public String type() {
			return type;
		}

		@Override
		public ResourceUsage usage() {
			return ResourceUsage.builder(type, AdjacentWordFeature.class)
					.setDescription(
							"Feature that simply triggers on the words around the parse step")
					.addParam("tag", String.class,
							"Feature set tag (default: " + DEFAULT_TAG + ")")
					.build();
		}
	}
}
