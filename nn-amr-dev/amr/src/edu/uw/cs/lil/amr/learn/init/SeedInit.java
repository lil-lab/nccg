package edu.uw.cs.lil.amr.learn.init;

import java.util.List;

import edu.cornell.cs.nlp.spf.base.hashvector.HashVectorFactory;
import edu.cornell.cs.nlp.spf.base.hashvector.IHashVector;
import edu.cornell.cs.nlp.spf.ccg.lexicon.LexicalEntry;
import edu.cornell.cs.nlp.spf.ccg.lexicon.Lexicon;
import edu.cornell.cs.nlp.spf.data.ILabeledDataItem;
import edu.cornell.cs.nlp.spf.data.situated.sentence.SituatedSentence;
import edu.cornell.cs.nlp.spf.explat.IResourceRepository;
import edu.cornell.cs.nlp.spf.explat.ParameterizedExperiment.Parameters;
import edu.cornell.cs.nlp.spf.explat.resources.IResourceObjectCreator;
import edu.cornell.cs.nlp.spf.explat.resources.usage.ResourceUsage;
import edu.cornell.cs.nlp.spf.mr.lambda.LogicalExpression;
import edu.cornell.cs.nlp.spf.parser.ccg.model.IDataItemModel;
import edu.cornell.cs.nlp.spf.parser.joint.IJointInferenceFilterFactory;
import edu.cornell.cs.nlp.spf.parser.joint.graph.IJointGraphDerivation;
import edu.cornell.cs.nlp.spf.parser.joint.model.IJointDataItemModel;
import edu.cornell.cs.nlp.spf.parser.joint.model.IJointModelInit;
import edu.cornell.cs.nlp.spf.parser.joint.model.JointModel;
import edu.cornell.cs.nlp.utils.collections.ListUtils;
import edu.cornell.cs.nlp.utils.log.ILogger;
import edu.cornell.cs.nlp.utils.log.LoggerFactory;
import edu.uw.cs.lil.amr.data.AMRMeta;
import edu.uw.cs.lil.amr.data.LabeledAmrSentenceLex;
import edu.uw.cs.lil.amr.data.LabeledAmrSentenceLexCollection;
import edu.uw.cs.lil.amr.parser.GraphAmrDerivation;
import edu.uw.cs.lil.amr.parser.GraphAmrParser;
import edu.uw.cs.lil.amr.parser.GraphAmrParserOutput;

/**
 * Initialize the model by parsing each seed example with its annotated lexical
 * entries. Add the max-scoring mean features of each correct derivation to the
 * model.
 *
 * @author Yoav Artzi
 */
@Deprecated
public class SeedInit implements
		IJointModelInit<SituatedSentence<AMRMeta>, LogicalExpression, LogicalExpression> {

	public static final ILogger																																			LOG	= LoggerFactory
			.create(SeedInit.class);
	private final LabeledAmrSentenceLexCollection																														data;
	private final IJointInferenceFilterFactory<ILabeledDataItem<SituatedSentence<AMRMeta>, LogicalExpression>, LogicalExpression, LogicalExpression, LogicalExpression>	filterFactory;

	private final GraphAmrParser																																		parser;
	private final double																																				scale;

	public SeedInit(LabeledAmrSentenceLexCollection data, GraphAmrParser parser,
			IJointInferenceFilterFactory<ILabeledDataItem<SituatedSentence<AMRMeta>, LogicalExpression>, LogicalExpression, LogicalExpression, LogicalExpression> filterFactory,
			double scale) {
		this.data = data;
		this.parser = parser;
		this.filterFactory = filterFactory;
		this.scale = scale;
	}

	private static void logParse(
			IJointGraphDerivation<LogicalExpression, LogicalExpression> derivation,
			IDataItemModel<LogicalExpression> dataItemModel) {
		LOG.info("[%.2f] Entries:", derivation.getViterbiScore(), derivation);
		for (final LexicalEntry<LogicalExpression> entry : derivation
				.getMaxLexicalEntries()) {
			LOG.info("\t[%f] %s %s {%s}", dataItemModel.score(entry), entry,
					dataItemModel.getTheta()
							.printValues(dataItemModel.computeFeatures(entry)),
					entry.getOrigin());
		}
		LOG.info("Rules used: %s",
				ListUtils.join(derivation.getMaxParsingRules(), ", "));
		LOG.info("Features: %s", dataItemModel.getTheta()
				.printValues(derivation.getMeanMaxFeatures()));
	}

	@Override
	public void init(
			JointModel<SituatedSentence<AMRMeta>, LogicalExpression, LogicalExpression> model) {
		final IHashVector update = HashVectorFactory.create();
		for (final LabeledAmrSentenceLex dataItem : data) {
			LOG.info("Parsing:");
			LOG.info(dataItem);
			final IJointDataItemModel<LogicalExpression, LogicalExpression> dataItemModel = model
					.createJointDataItemModel(dataItem.getSample());
			final Lexicon<LogicalExpression> lexicon = new Lexicon<>(
					dataItem.getEntries());
			final GraphAmrParserOutput output = parser.parse(
					dataItem.getSample(), dataItemModel,
					filterFactory.createJointFilter(dataItem), false, lexicon);
			final List<GraphAmrDerivation> derivations = output
					.getDerivations(dataItem.getLabel());
			LOG.info("%d correct seed derivation", derivations.size());
			for (final GraphAmrDerivation derivation : derivations) {
				logParse(derivation, dataItemModel);
				derivation.getMeanMaxFeatures().addTimesInto(1.0, update);
			}
		}
		update.divideBy(data.size());
		update.addTimesInto(scale, model.getTheta());
	}

	public static class Creator implements IResourceObjectCreator<SeedInit> {

		private final String type;

		public Creator() {
			this("init.seedlex");
		}

		public Creator(String type) {
			this.type = type;
		}

		@Override
		public SeedInit create(Parameters params, IResourceRepository repo) {
			return new SeedInit(repo.get(params.get("data")),
					repo.get(params.get("parser")),
					repo.get(params.get("filter")),
					params.getAsDouble("scale", 1.0));
		}

		@Override
		public String type() {
			return type;
		}

		@Override
		public ResourceUsage usage() {
			// TODO Auto-generated method stub
			return null;
		}

	}

}
