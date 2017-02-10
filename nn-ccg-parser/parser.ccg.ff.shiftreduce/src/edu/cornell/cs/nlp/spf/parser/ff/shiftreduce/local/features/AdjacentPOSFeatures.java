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

public class AdjacentPOSFeatures implements
				IParseFeatureSet<SituatedSentence<AMRMeta>, LogicalExpression> {

	private static final long serialVersionUID 			= -1153077032675131452L;
	private static final String	TAGNEXTPOS1				= "NEXT1POS";
	private static final String	TAGNEXTPOS2				= "NEXT2POS";
//	private static final String	TAGNEXTPOS3				= "NEXT3POS";
	private static final String	TAGPREVPOS1				= "PREV1POS";
	private static final String	TAGPREVPOS2				= "PREV2POS";
	private static final String NoPOS					= "None";

	public AdjacentPOSFeatures() {
	}
	
	/** Calculates bigram and unigram POS features based on what is on buffer */
	@Override
	public void setFeatures(IParseStep<LogicalExpression> parseStep, IHashVector feats,
			SituatedSentence<AMRMeta> dataItem) {
		
		final int n = dataItem.getTokens().size();
		final int end = parseStep.getEnd();
		
		if(end == n - 1) { //no words left
			feats.add(TAGNEXTPOS1, NoPOS, 1.0);
			feats.add(TAGNEXTPOS2, NoPOS, 1.0);
			
		} else if(end == n - 2) { //one word left
			
			final String nPOS1 = dataItem.getState().getTags()
					.sub(end + 1,  end + 2).toString("+");
			feats.add(TAGNEXTPOS1, nPOS1, 1.0);

			feats.add(TAGNEXTPOS2, NoPOS, 1.0);
			
		} else { //two or more words left
			
			final String nPOS1 = dataItem.getState().getTags()
					.sub(end + 1,  end + 2).toString("+");
			feats.add(TAGNEXTPOS1, nPOS1, 1.0);
			
			final String nPOS2 = dataItem.getState().getTags()
					.sub(end + 2,  end + 3).toString("+");
			feats.add(TAGNEXTPOS2, nPOS2, 1.0);
		}
		
//		if(end >= n - 3) {
//			feats.add(TAGNEXTPOS3, NoPOS, 1.0);
//		} else {
//			final String nPOS = dataItem.getState().getTags().get(end + 3);
//			feats.add(TAGNEXTPOS3, nPOS, 1.0);
//		}
		
		final int start = parseStep.getStart();
		
		if(start == 0) { //no words left
			feats.add(TAGPREVPOS1, NoPOS, 1.0);
			feats.add(TAGPREVPOS2, NoPOS, 1.0);
			
		} else if(start == 1) { //one word left

			final String pPOS1 = dataItem.getState().getTags()
					.sub(start - 1,  start).toString("+");
			feats.add(TAGPREVPOS1, pPOS1, 1.0);

			feats.add(TAGPREVPOS2, NoPOS, 1.0);
			
		} else { //two or more words left
			final String pPOS1 = dataItem.getState().getTags()
					.sub(start - 1,  start).toString("+");
			feats.add(TAGPREVPOS1, pPOS1, 1.0);
			
			final String pPOS2 = dataItem.getState().getTags()
					.sub(start - 2,  start - 1).toString("+");
			feats.add(TAGPREVPOS2, pPOS2, 1.0);
		}
	}

	@Override
	public Set<KeyArgs> getDefaultFeatures() {
		return Collections.emptySet(); //not sure how this works
	}
	
	public static class Creator implements IResourceObjectCreator<AdjacentPOSFeatures> {

		private final String type;
		
		public Creator() {
			this("feat.adjacent.pos");
		}
		
		public Creator(String type) {
			this.type = type;
		}
		
		@Override
		public AdjacentPOSFeatures create(Parameters params,
				IResourceRepository repo) {
			return new AdjacentPOSFeatures();
		}
		
		@Override
		public String type() {
			return type;
		}
		
		@Override
		public ResourceUsage usage() {
			return ResourceUsage.builder(type, AdjacentPOSFeatures.class)
					.setDescription(
							"Features that computes adjacent POS tags")
					.build();
		}
	}

}
