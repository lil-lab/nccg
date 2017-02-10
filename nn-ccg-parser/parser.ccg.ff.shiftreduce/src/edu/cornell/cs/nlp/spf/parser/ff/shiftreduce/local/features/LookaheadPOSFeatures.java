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

public class LookaheadPOSFeatures implements
				IParseFeatureSet<SituatedSentence<AMRMeta>, LogicalExpression> {

	private static final long 	serialVersionUID 	= 7691222373237990628L;
	private static final String	DEFAULT_TAG			= "LOOKAHEADPOS";
	private final String		tag;
	private static final String NoPOS				= "None";

	public LookaheadPOSFeatures(String tag) {
		this.tag = tag;
	}
	
	/** Calculates bigram and unigram POS features based on what is on buffer */
	@Override
	public void setFeatures(IParseStep<LogicalExpression> parseStep, IHashVector feats,
			SituatedSentence<AMRMeta> dataItem) {
		
		final int n = dataItem.getTokens().size();
		final int end = parseStep.getEnd();
		
		if(end == n - 1) { //no words left
			
			final String unigramPosSeq = NoPOS;
			feats.add(tag, "unigram", unigramPosSeq, 1.0);
			
			final String bigramPosSeq = NoPOS + "+" + NoPOS;
			feats.add(tag, "bigram", bigramPosSeq, 1.0);
			
		} else if(end == n - 2) { //one word left
			
			final String unigramPosSeq = dataItem.getState().getTags()
					.sub(end + 1,  end + 2).toString("+");
			feats.add(tag, "unigram", unigramPosSeq, 1.0);
			
			final String bigramPosSeq = unigramPosSeq + "+" + NoPOS;
			feats.add(tag, "bigram", bigramPosSeq, 1.0);
			
		} else { //two or more words left
			
			final String unigramPosSeq = dataItem.getState().getTags()
					.sub(end + 1,  end + 2).toString("+");
			feats.add(tag, "unigram", unigramPosSeq, 1.0);
			
			final String bigramPosSeq = dataItem.getState().getTags()
					.sub(end + 1,  end + 3).toString("+");
			feats.add(tag, "bigram", bigramPosSeq, 1.0);
		}
	}

	@Override
	public Set<KeyArgs> getDefaultFeatures() {
		return Collections.emptySet(); //not sure how this works
	}
	
	public static class Creator implements IResourceObjectCreator<LookaheadPOSFeatures> {

		private final String type;
		
		public Creator() {
			this("feat.lookahead.pos");
		}
		
		public Creator(String type) {
			this.type = type;
		}
		
		@Override
		public LookaheadPOSFeatures create(Parameters params,
				IResourceRepository repo) {
			return new LookaheadPOSFeatures(params.get("tag", DEFAULT_TAG));
		}
		
		@Override
		public String type() {
			return type;
		}
		
		@Override
		public ResourceUsage usage() {
			return ResourceUsage.builder(type, LookaheadPOSFeatures.class)
					.setDescription(
							"Features that computes bigram of pos of next two words on the buffer")
					.addParam("tag", String.class,
							"Feature set tag (default: " + DEFAULT_TAG + ")")
					.build();
		}
	}

}
