package edu.uw.cs.lil.amr.learn.estimators;

import java.io.Serializable;
import java.util.function.IntFunction;

import edu.cornell.cs.nlp.spf.base.hashvector.IHashVector;
import edu.cornell.cs.nlp.spf.base.hashvector.IHashVectorImmutable;
import edu.cornell.cs.nlp.spf.base.hashvector.KeyArgs;
import edu.cornell.cs.nlp.spf.explat.IResourceRepository;
import edu.cornell.cs.nlp.spf.explat.ParameterizedExperiment.Parameters;
import edu.cornell.cs.nlp.spf.explat.resources.IResourceObjectCreator;
import edu.cornell.cs.nlp.spf.explat.resources.usage.ResourceUsage;
import edu.cornell.cs.nlp.utils.composites.Pair;
import edu.cornell.cs.nlp.utils.log.ILogger;
import edu.cornell.cs.nlp.utils.log.LoggerFactory;

/**
 * AdaGrad with LASSO (L1) regularization). The AdaGrad update rule from Duchi
 * et al. 2010. Also see Notes on AdaGrad by Chris Dyer. LASSO is applied with
 * the FOBOS framework (Duchi and Singer 2009) as described in Green et al.
 * 2013 and in the Stanford CoreNLP code base.
 *
 * @author Yoav Artzi
 */
@Deprecated
public class AdaGradEstimatorWithLasso extends AbstractAdaGradEstimator {

	public static final ILogger	LOG					= LoggerFactory
			.create(AdaGradEstimatorWithLasso.class);

	private static final long	serialVersionUID	= 6273874056121731755L;

	private final double		regCoef;

	public AdaGradEstimatorWithLasso(boolean initHistory,
			IntFunction<Double> rateFunction, double regCoef) {
		super(initHistory, rateFunction);
		this.regCoef = regCoef;
	}

	@Override
	public boolean applyUpdate(IHashVector gradient, IHashVector weights) {
		final IHashVectorImmutable update = computeUpdate(gradient);

		if (update == null) {
			return false;
		}

		// Apply the update to theta.
		update.addTimesInto(1.0, weights);

		// Iterate over the updated weights. If a weight dropped below the
		// adaptive threshold, set it to zero.
		final IHashVector threshold = createUpdateHistoryMultiplier();
		threshold.multiplyBy(regCoef * regCoef);
		for (final Pair<KeyArgs, Double> updateEntry : update) {
			final KeyArgs key = updateEntry.first();
			if (Math.abs(weights.get(key)) < threshold.get(key)) {
				weights.set(key, 0.0);
			}
		}
		weights.dropZeros();

		numUpdates++;
		return true;
	}

	public static class Creator
			implements IResourceObjectCreator<AdaGradEstimatorWithLasso> {

		private final String type;

		public Creator() {
			this("estimator.adagrad.lasso");
		}

		public Creator(String type) {
			this.type = type;
		}

		@SuppressWarnings("unchecked")
		@Override
		public AdaGradEstimatorWithLasso create(Parameters params,
				IResourceRepository repo) {
			final IntFunction<Double> rateFunction;
			if (params.contains("c") && params.contains("alpha0")) {
				final double c = params.getAsDouble("c");
				final double alpha0 = params.getAsDouble("alpha0");
				rateFunction = (Serializable & IntFunction<Double>) n -> (alpha0
						/ (1 + c * n));
				LOG.info(
						"Estimator with a decaying leanring rate: %f / (1 + %f * num_updates)",
						alpha0, c);
			} else {
				final double rate = params.getAsDouble("rate", 1.0);
				rateFunction = (Serializable & IntFunction<Double>) n -> rate;
				LOG.info("Estimator with a constant learning rate: %f", rate);
			}

			return new AdaGradEstimatorWithLasso(
					params.getAsBoolean("initHistory", false), rateFunction,
					params.getAsDouble("reg"));
		}

		@Override
		public String type() {
			return type;
		}

		@Override
		public ResourceUsage usage() {
			return ResourceUsage.builder(type, AdaGradEstimatorWithLasso.class)
					.setDescription(
							"The AdaGrad update rule from Duchi et al. 2010. Also see Notes on AdaGrad by Chris Dyer.")
					.addParam("reg", Double.class,
							"L1 (LASSO) Regularization coefficient")
					.addParam("initHistory", Boolean.class,
							"Initialize the history vector with 1.0s. This is a modification of the vanilla AdaGrad algorihtm (default: false)")
					.addParam("rate", Double.class,
							"Learning rate (only used of alpha0 or c are not used) (default: 1.0)")
					.addParam("c", Double.class,
							"Decaying learning rate: alpha0 / (1+c*num_updates) (only used if alpha0 is specified as well, rate is then ignored)")
					.addParam("alpha0", Double.class,
							"Decaying learning rate: alpha0 / (1+c*num_updates) (only used if c is specified as well, rate is then ignored)")
					.build();
		}

	}

}
