package edu.cornell.cs.nlp.spf.parser.ccg.rnn.shiftreduce.dataset;

import java.util.List;
import java.util.function.Predicate;

import edu.cornell.cs.nlp.spf.base.token.TokenSeq;
import edu.cornell.cs.nlp.spf.data.IDataItem;
import edu.cornell.cs.nlp.spf.data.ILabeledDataItem;
import edu.cornell.cs.nlp.spf.data.situated.sentence.SituatedSentence;
import edu.cornell.cs.nlp.spf.mr.lambda.LogicalExpression;
import edu.cornell.cs.nlp.spf.parser.ParsingOp;
import edu.cornell.cs.nlp.spf.parser.ccg.cky.CKYDerivation;
import edu.cornell.cs.nlp.spf.parser.ccg.cky.CKYParserOutput;
import edu.cornell.cs.nlp.spf.parser.ccg.cky.chart.Chart;
import edu.cornell.cs.nlp.spf.parser.filter.IParsingFilterFactory;
import edu.cornell.cs.nlp.spf.parser.graph.IGraphDerivation;
import edu.cornell.cs.nlp.spf.parser.joint.IJointInferenceFilter;
import edu.cornell.cs.nlp.spf.parser.joint.IJointInferenceFilterFactory;
import edu.cornell.cs.nlp.spf.parser.joint.InferencePair;
import edu.cornell.cs.nlp.spf.parser.joint.model.IJointDataItemModel;
import edu.cornell.cs.nlp.utils.log.ILogger;
import edu.cornell.cs.nlp.utils.log.LoggerFactory;
import edu.uw.cs.lil.amr.data.AMRMeta;
import edu.uw.cs.lil.amr.parser.GraphAmrDerivation;
import edu.uw.cs.lil.amr.parser.GraphAmrParser;
import edu.uw.cs.lil.amr.parser.GraphAmrParserOutput;

public class AMRParsingFilterFactory<SAMPLE extends IDataItem<?>, DI extends ILabeledDataItem<SAMPLE, ?>> {

	public static final ILogger	LOG = LoggerFactory.create(AMRParsingFilterFactory.class);
	private final GraphAmrParser amrParser;
	private final IParsingFilterFactory<DI, LogicalExpression> parsingFilterFactory;
	private final IJointInferenceFilterFactory<ILabeledDataItem<SituatedSentence<AMRMeta>, 
					LogicalExpression>, LogicalExpression, LogicalExpression, LogicalExpression> amrSupervisedFilterFactory;
	
	private Chart<LogicalExpression> chart;
	
	public AMRParsingFilterFactory(GraphAmrParser amrParser, 
									IParsingFilterFactory<DI, LogicalExpression> parsingFilterFactory, 
									IJointInferenceFilterFactory<ILabeledDataItem<SituatedSentence<AMRMeta>, 
									LogicalExpression>, LogicalExpression, LogicalExpression, LogicalExpression> amrSupervisedFilterFactory) {
		this.amrParser = amrParser;
		this.parsingFilterFactory = parsingFilterFactory;
		this.amrSupervisedFilterFactory = amrSupervisedFilterFactory;
	}
	
	/** Create an AMR parser filter for a given sentence. This filter will guide Neural Shift Reduce parser
	 *  to take steps that will make Shift-Reduce parse get the same parse tree as the AMR parser */
	public Predicate<ParsingOp<LogicalExpression>> create(DI dataItem, 
										IJointDataItemModel<LogicalExpression, LogicalExpression> dataItemModel) {
		
		@SuppressWarnings("unchecked")
		final SituatedSentence<AMRMeta> sentence = (SituatedSentence<AMRMeta>) dataItem.getSample();
		
		TokenSeq tk = sentence.getTokens();
		int n = tk.size(); //number of tokens
		
		@SuppressWarnings("unchecked")
		IJointInferenceFilter<LogicalExpression, LogicalExpression, LogicalExpression> amrSupervisedFilter = 
						this.amrSupervisedFilterFactory.createJointFilter(
								(ILabeledDataItem<SituatedSentence<AMRMeta>, LogicalExpression>) dataItem);
		
		GraphAmrParserOutput amrParserOutput = this.amrParser.parse(sentence, dataItemModel, amrSupervisedFilter); 
		this.chart = ((CKYParserOutput<LogicalExpression>)amrParserOutput.getBaseParserOutput()).getChart();
		
		List<GraphAmrDerivation> maxDerivations = amrParserOutput.getDerivations(((LogicalExpression)dataItem.getLabel()));
												//amrParserOutput.getMaxDerivations(((LogicalExpression)dataItem.getLabel()));
		LOG.info("Number of CKY Graph derivations are %s", maxDerivations.size()); 
		
		CKYDerivation<LogicalExpression> bestDerivation = null;
		double bestDerivationScore =  Double.NEGATIVE_INFINITY;
		int numTies = 0, numInferencePairs = 0;
		
		if(maxDerivations.size() > 0) {
		
			for(GraphAmrDerivation graphAmrDerivation: maxDerivations) {
				
				//TODO: use graphAmrDerivation.maxInferencePairs()
				List<InferencePair<LogicalExpression, LogicalExpression, IGraphDerivation<LogicalExpression>>> inferencePairs = 
																	graphAmrDerivation.getInferencePairs();
					
				for(InferencePair<LogicalExpression, LogicalExpression, IGraphDerivation<LogicalExpression>> inferencePair: inferencePairs) {
					
					CKYDerivation<LogicalExpression> baseDerivation = (CKYDerivation<LogicalExpression>) inferencePair.getBaseDerivation();
					if(baseDerivation.getScore() > bestDerivationScore) {
						
						bestDerivationScore = baseDerivation.getScore();
						bestDerivation = baseDerivation;
						numTies = 0;
					} else if(baseDerivation.getScore() == bestDerivationScore) {
						numTies++; //we arbitrarily stay with the first one 
					}
					
//					LOG.info("InferencePair %s -> %s", baseDerivation.getScore(), baseDerivation.getCategory());
					numInferencePairs++;
				}
			}
			
			LOG.info("A CKY parse tree has the ground truth label. Num inference pairs %s, num ties %s, Score %s", 
							numInferencePairs, numTies, bestDerivationScore);
			 
			LOG.info("CKY parse tree Category %s", bestDerivation.getCategory());
		
		} else {
			LOG.info("CKY failed to parse the sentence.");
			return null;
		}
		
		Predicate<ParsingOp<LogicalExpression>> supervisedPruningFilter = this.parsingFilterFactory.create(dataItem);
				
		return new CKYMultiParseTreeParsingFilter<LogicalExpression>(bestDerivation, supervisedPruningFilter, n);
	}
	
	public Chart<LogicalExpression> getChart() {
		return this.chart;
	}
}