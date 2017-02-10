package edu.cornell.cs.nlp.spf.parser.ff.shiftreduce.learner;

import java.util.ArrayList;
import java.util.List;
import java.util.function.Predicate;

import edu.cornell.cs.nlp.spf.data.IDataItem;
import edu.cornell.cs.nlp.spf.data.ILabeledDataItem;
import edu.cornell.cs.nlp.spf.data.collection.IDataCollection;
import edu.cornell.cs.nlp.spf.data.sentence.Sentence;
import edu.cornell.cs.nlp.spf.data.situated.labeled.LabeledSituatedSentence;
import edu.cornell.cs.nlp.spf.data.situated.sentence.SituatedSentence;
import edu.cornell.cs.nlp.spf.explat.IResourceRepository;
import edu.cornell.cs.nlp.spf.explat.ParameterizedExperiment.Parameters;
import edu.cornell.cs.nlp.spf.explat.resources.IResourceObjectCreator;
import edu.cornell.cs.nlp.spf.explat.resources.usage.ResourceUsage;
import edu.cornell.cs.nlp.spf.mr.lambda.LogicalExpression;
import edu.cornell.cs.nlp.spf.parser.ParsingOp;
import edu.cornell.cs.nlp.spf.parser.ff.shiftreduce.neuralparser.NeuralShiftReduceParser;
import edu.cornell.cs.nlp.spf.parser.graph.IGraphDerivation;
import edu.cornell.cs.nlp.spf.parser.graph.IGraphParserOutput;
import edu.cornell.cs.nlp.spf.parser.joint.model.IJointDataItemModel;
import edu.cornell.cs.nlp.spf.parser.joint.model.IJointModelImmutable;
import edu.cornell.cs.nlp.utils.log.ILogger;
import edu.cornell.cs.nlp.utils.log.LoggerFactory;
import edu.uw.cs.lil.amr.data.AMRMeta;
import edu.uw.cs.lil.amr.data.LabeledAmrSentence;
import edu.uw.cs.lil.amr.parser.AmrParsingFilter;
import edu.uw.cs.lil.amr.test.SmatchStats;

public class ValidationStatistics {
	
	public static final ILogger	LOG = LoggerFactory.create(ValidationStatistics.class);

	private final NeuralShiftReduceParser<Sentence, LogicalExpression> parser;
	private final IJointModelImmutable<SituatedSentence<AMRMeta>, LogicalExpression, LogicalExpression> model;
	private final Integer beamSize;
	private final Predicate<ParsingOp<LogicalExpression>> validAmrParsingFilter;
	///metric for calculating validation
	private final SmatchStats stats;	
	private final List<LabeledAmrSentence> validation;
	private final int maxSentenceLength;
	
	public ValidationStatistics(NeuralShiftReduceParser<Sentence, LogicalExpression> parser, 
					IJointModelImmutable<SituatedSentence<AMRMeta>, LogicalExpression, LogicalExpression> model,  
					 IDataCollection<LabeledAmrSentence> dataset, 
					Integer beamSize, SmatchStats stats) {
		
		this.parser = parser;
		this.model = model;
		this.beamSize = beamSize;
		this.stats = stats;
		final int validationSize = (int)(0.1 * dataset.size());
		
		this.maxSentenceLength = 15;
		
		this.validation = new ArrayList<LabeledAmrSentence>();
		int index = 0;
		for(LabeledAmrSentence e: dataset) {
			
			final SituatedSentence<AMRMeta> situatedSentence = 
				    ((LabeledSituatedSentence<AMRMeta, LogicalExpression>) e).getSample();
			final Sentence sentence = situatedSentence.getSample(); 

			// filter the dataItem based on length
			if(this.maxSentenceLength >=0 && sentence.getTokens().size() > this.maxSentenceLength) {
				continue;
			}
			
			this.validation.add(e);//.subList(0, 4/*validationSize*/);
			index++;
			if(index == validationSize) {
				break;
			}
		}
		
		this.validAmrParsingFilter = new AmrParsingFilter();
		LOG.info("Validation Statistics: validation size %s, beam size %s", this.validation.size(), this.beamSize);
	}

	/** Log-likelihood is not a good choice for validation therefore we use 
	 * SMATCH statistics for validation. */
	public void calcValidationMetric() {
		
		for(LabeledAmrSentence pt: this.validation) {
			
			final SituatedSentence<AMRMeta> situatedSentence = 
									    ((LabeledSituatedSentence<AMRMeta, LogicalExpression>) pt).getSample();
			final Sentence sentence = situatedSentence.getSample(); 
			final IJointDataItemModel<LogicalExpression, LogicalExpression> dataItemModel = 
										this.model.createJointDataItemModel(situatedSentence);

			LabeledAmrSentence underspecified = pt;//new LabeledAmrSentence();
			
			//parse with a small beam
			IGraphParserOutput<LogicalExpression> output = this.parser
									.parse(sentence, this.validAmrParsingFilter, dataItemModel, false, null, this.beamSize); 
			
			List<? extends IGraphDerivation<LogicalExpression>> bestDerivations = output.getBestDerivations();
			
			if(bestDerivations.size() == 0) {
				this.stats.recordNoParse(underspecified);
			} else if(bestDerivations.size() == 1) {
				this.stats.recordParse(underspecified, bestDerivations.get(0).getSemantics());
			} else {
				List<LogicalExpression> candidates = new ArrayList<LogicalExpression>();
				for(IGraphDerivation<LogicalExpression> e: bestDerivations) {
					candidates.add(e.getSemantics());
				}
				this.stats.recordParses(underspecified, candidates);
			}
		}
		
		LOG.info("Final %s", this.stats.toString());
	}
	
	public static class Creator<SAMPLE extends IDataItem<?>, DI extends ILabeledDataItem<SAMPLE, ?>, MR> 
						implements IResourceObjectCreator<ValidationStatistics> {

		private final String type;
		
		private NeuralShiftReduceParser<Sentence, LogicalExpression> parser;
		private IJointModelImmutable<SituatedSentence<AMRMeta>, LogicalExpression, LogicalExpression> model;
		private Integer beamSize = 20;
		
		///metric for calculating validation
		private SmatchStats stats;	
		private IDataCollection<LabeledAmrSentence> dataset;
		
		
		public Creator() {
			this("validation.statistics");
		}
		
		public Creator(String type) {
			this.type = type;
		}

		@SuppressWarnings("unchecked")
		@Override
		public ValidationStatistics create(Parameters params, IResourceRepository repo) {
			
			this.parser = (NeuralShiftReduceParser<Sentence, LogicalExpression>)repo.get(params.get("parser"));
			this.model = (IJointModelImmutable<SituatedSentence<AMRMeta>, LogicalExpression, LogicalExpression>)
								repo.get(params.get("model"));
			
			if(params.contains("beamSize")) {
				this.beamSize = params.getAsInteger("beamSize");
			}
			
			this.stats = (SmatchStats)repo.get(params.get("stats"));
			LOG.info("----");
			this.dataset = (IDataCollection<LabeledAmrSentence>)repo.get(params.get("data"));
			
			return new ValidationStatistics(parser, model, dataset, beamSize, stats);
		}

		@Override
		public String type() {
			return this.type;
		}

		@Override
		public ResourceUsage usage() {
			throw new RuntimeException("Operation not supported");
		}

	}
	
}
