package edu.uw.cs.lil.amr.features.decomposed;

import java.util.Collections;
import java.util.Set;

import edu.cornell.cs.nlp.spf.base.hashvector.IHashVector;
import edu.cornell.cs.nlp.spf.base.hashvector.KeyArgs;
import edu.cornell.cs.nlp.spf.data.IDataItem;
import edu.cornell.cs.nlp.spf.explat.IResourceRepository;
import edu.cornell.cs.nlp.spf.explat.ParameterizedExperiment.Parameters;
import edu.cornell.cs.nlp.spf.explat.resources.IResourceObjectCreator;
import edu.cornell.cs.nlp.spf.explat.resources.usage.ResourceUsage;
import edu.cornell.cs.nlp.spf.mr.lambda.Lambda;
import edu.cornell.cs.nlp.spf.mr.lambda.Literal;
import edu.cornell.cs.nlp.spf.mr.lambda.LogicalConstant;
import edu.cornell.cs.nlp.spf.mr.lambda.LogicalExpression;
import edu.cornell.cs.nlp.spf.parser.ccg.IOverloadedParseStep;
import edu.cornell.cs.nlp.spf.parser.ccg.IParseStep;
import edu.cornell.cs.nlp.spf.parser.ccg.model.parse.IParseFeatureSet;
import edu.cornell.cs.nlp.spf.parser.ccg.shiftreduce.steps.AbstractShiftReduceStep;
import edu.uw.cs.lil.amr.lambda.AMRServices;

/**
 * Simple feature to trigger on the head of the AMR expression.
 *
 * @author Yoav Artzi
 *
 */
public class ShiftHeadPredicateFeatureSet<DI extends IDataItem<?>>
		implements IParseFeatureSet<DI, LogicalExpression> {

	private static final String	DEFAULT_TAG			= "SEMHEAD";
	private static final long	serialVersionUID	= 2255537024084940329L;
	private final String		tag;

	public ShiftHeadPredicateFeatureSet(String tag) {
		this.tag = tag;
	}

	@Override
	public Set<KeyArgs> getDefaultFeatures() {
		// Nothing to do.
		return Collections.emptySet();
	}

	@Override
	public void setFeatures(IParseStep<LogicalExpression> step,
			IHashVector feats, DI dataItem) {

		if (step instanceof AbstractShiftReduceStep) {
			final AbstractShiftReduceStep<LogicalExpression> shiftReduceParseStep = (AbstractShiftReduceStep<LogicalExpression>) step;
			if (shiftReduceParseStep.isUnary()) {
				final LogicalExpression semantics = shiftReduceParseStep
						.getRoot().getSemantics();
				if (semantics != null) {
					final LogicalConstant instanceType;
					if (AMRServices.isSkolemTerm(semantics)) {
						instanceType = AMRServices
								.getTypingPredicate((Literal) semantics);
					} else if (AMRServices.isSkolemTermBody(semantics)) {
						instanceType = AMRServices
								.getTypingPredicate((Lambda) semantics);
					} else {
						return;
					}
					if (instanceType != null) {
						feats.set(tag, instanceType.getBaseName(), 1.0);
					}
				}
			}
		} else { // Below code is used by CKY or parsers that handle unary rule
					// via overloading
			if (step instanceof IOverloadedParseStep) {
				final LogicalExpression semantics = ((IOverloadedParseStep<LogicalExpression>) step)
						.getIntermediate().getSemantics();
				if (semantics != null) {
					final LogicalConstant instanceType;
					if (AMRServices.isSkolemTerm(semantics)) {
						instanceType = AMRServices
								.getTypingPredicate((Literal) semantics);
					} else if (AMRServices.isSkolemTermBody(semantics)) {
						instanceType = AMRServices
								.getTypingPredicate((Lambda) semantics);
					} else {
						return;
					}
					if (instanceType != null) {
						feats.set(tag, instanceType.getBaseName(), 1.0);
					}
				}
			}
		}
	}

	public static class Creator<DI extends IDataItem<?>> implements
			IResourceObjectCreator<ShiftHeadPredicateFeatureSet<DI>> {

		private final String type;

		public Creator() {
			this("feat.decomposed.amrhead");
		}

		public Creator(String type) {
			this.type = type;
		}

		@Override
		public ShiftHeadPredicateFeatureSet<DI> create(Parameters params,
				IResourceRepository repo) {
			return new ShiftHeadPredicateFeatureSet<>(
					params.get("tag", DEFAULT_TAG));
		}

		@Override
		public String type() {
			return type;
		}

		@Override
		public ResourceUsage usage() {
			return ResourceUsage
					.builder(type, ShiftHeadPredicateFeatureSet.class)
					.addParam("tag", String.class,
							"Feature set tag (default: " + DEFAULT_TAG + ")")
					.build();
		}

	}

}
