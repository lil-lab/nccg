package edu.cornell.cs.nlp.spf.parser.ff.shiftreduce.local.features;

import java.util.Collections;
import java.util.Set;

import edu.cornell.cs.nlp.spf.base.hashvector.IHashVector;
import edu.cornell.cs.nlp.spf.base.hashvector.KeyArgs;
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

/** Feature that triggers on the origin of a dynamic lexical entry in a shift step. 
 * @author Dipendra Misra 
 * */
public class DynamicOriginFeature implements
					IParseFeatureSet<SituatedSentence<AMRMeta>, LogicalExpression> {

	private static final long serialVersionUID = -1186981091421754631L;
	private final static String DEFAULT_TAG = "DYNORIGIN";
	private final String tag;
	
	public DynamicOriginFeature(String tag) {
		this.tag = tag;
	}
	
	
	@Override
	public void setFeatures(IParseStep<LogicalExpression> parseStep, IHashVector feats,
			SituatedSentence<AMRMeta> dataItem) {
		
		if (parseStep instanceof ILexicalParseStep) {
			final LexicalEntry<LogicalExpression> entry = ((ILexicalParseStep<LogicalExpression>) parseStep)
																.getLexicalEntry();
			
			if(entry.isDynamic()) {
				feats.add(this.tag, entry.getOrigin(), 1.0);
			}
		}
	}
	
	@Override
	public Set<KeyArgs> getDefaultFeatures() {
		return Collections.emptySet();
	} 
	
	public static class Creator implements IResourceObjectCreator<DynamicOriginFeature> {

		private final String type;
		
		public Creator() {
			this("feat.dyn.origin");
		}
		
		public Creator(String type) {
			this.type = type;
		}
		
		@Override
		public DynamicOriginFeature create(Parameters params,
				IResourceRepository repo) {
			return new DynamicOriginFeature(params.get("tag", DEFAULT_TAG));
		}
		
		@Override
		public String type() {
			return type;
		}
		
		@Override
		public ResourceUsage usage() {
			return ResourceUsage.builder(type, DynamicOriginFeature.class)
					.setDescription(
							"Feature that is triggered on origin of dynamic lexical entries in shift step")
					.addParam("tag", String.class,
							"Feature set tag (default: " + DEFAULT_TAG + ")")
					.build();
		}
	}

}

