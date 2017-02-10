package edu.cornell.cs.nlp.spf.parser.ff.shiftreduce.dataset;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.ObjectInput;
import java.io.ObjectInputStream;
import java.io.ObjectOutput;
import java.io.ObjectOutputStream;
import java.io.OutputStream;
import java.util.LinkedList;
import java.util.List;
import java.util.function.Predicate;

import edu.cornell.cs.nlp.spf.ccg.categories.Category;
import edu.cornell.cs.nlp.spf.ccg.lexicon.CompositeImmutableLexicon;
import edu.cornell.cs.nlp.spf.ccg.lexicon.ILexiconImmutable;
import edu.cornell.cs.nlp.spf.data.IDataItem;
import edu.cornell.cs.nlp.spf.data.ILabeledDataItem;
import edu.cornell.cs.nlp.spf.data.collection.IDataCollection;
import edu.cornell.cs.nlp.spf.data.sentence.Sentence;
import edu.cornell.cs.nlp.spf.data.situated.sentence.SituatedSentence;
import edu.cornell.cs.nlp.spf.data.utils.IValidator;
import edu.cornell.cs.nlp.spf.mr.lambda.LogicalExpression;
import edu.cornell.cs.nlp.spf.parser.ParsingOp;
import edu.cornell.cs.nlp.spf.parser.ccg.rnn.shiftreduce.dataset.AMRParsingFilterFactory;
import edu.cornell.cs.nlp.spf.parser.ccg.rnn.shiftreduce.dataset.AbstractCreateDataset;
import edu.cornell.cs.nlp.spf.parser.ccg.rnn.shiftreduce.dataset.CKYMultiParseTreeParsingFilter;
import edu.cornell.cs.nlp.spf.parser.ccg.rnn.shiftreduce.dataset.CreateDecisionDataset;
import edu.cornell.cs.nlp.spf.parser.ccg.rnn.shiftreduce.neuralnetworkparser.AbstractNeuralShiftReduceParser;
import edu.cornell.cs.nlp.spf.parser.ccg.rnn.shiftreduce.neuralnetworkparser.NeuralNetworkShiftReduceParser;
import edu.cornell.cs.nlp.spf.parser.ccg.shiftreduce.ShiftReduceDerivation;
import edu.cornell.cs.nlp.spf.parser.ccg.shiftreduce.ShiftReduceParserOutput;
import edu.cornell.cs.nlp.spf.parser.ccg.shiftreduce.stacks.DerivationState;
import edu.cornell.cs.nlp.spf.parser.filter.IParsingFilterFactory;
import edu.cornell.cs.nlp.spf.parser.joint.IJointInferenceFilterFactory;
import edu.cornell.cs.nlp.spf.parser.joint.model.IJointDataItemModel;
import edu.cornell.cs.nlp.spf.parser.joint.model.IJointModelImmutable;
import edu.cornell.cs.nlp.utils.log.ILogger;
import edu.cornell.cs.nlp.utils.log.LogLevel;
import edu.cornell.cs.nlp.utils.log.LoggerFactory;
import edu.uw.cs.lil.amr.data.AMRMeta;
import edu.uw.cs.lil.amr.parser.AmrParsingFilter;
import edu.uw.cs.lil.amr.parser.GraphAmrParser;

/** Creates dataset with exploration i.e. by taking sub-optimal decisions in order to explore more 
 * choices. */
public abstract class AbstractCreateDatasetWithExploration<Dataset, SAMPLE extends IDataItem<?>,
															DI extends ILabeledDataItem<SAMPLE, ?>, MR> 
								extends AbstractCreateDataset<Dataset, SAMPLE, DI, MR> {

	/** TODO in this file check if a lexical entry is getting added multiple times. The temp lexicon and model lexical
	 * are getting mixed while parsing. */
	public static final ILogger	LOG = LoggerFactory.create(CreateDecisionDataset.class);

	private final IDataCollection<DI> trainingData;
	private final AbstractNeuralShiftReduceParser<Sentence, LogicalExpression> baseNeuralAmrParser;
	private final Integer beamSize;
	
	private final AMRParsingFilterFactory<SAMPLE, DI> amrParsingFilterFactory;
	
	private final AmrParsingFilter validAmrParsingFilter; 
	
	/////////TEMPORARY ADDED FOR HANDLING NEW FEATURES ///////////
	public IJointModelImmutable<SituatedSentence<AMRMeta>, 
									LogicalExpression, LogicalExpression> modelNewFeatures;
	/////////////////////////////////////////////////////////////
	
	/** Length of the maximum sentence that is accepted for training. */
	private final int maxSentenceLength;
	
	/** Save the dataset. Note that saving the dataset can take huge amount of disk space. */
	private final boolean saveCreatedDataset;
	
	/** Bootstrap saved data*/
	private final boolean bootstrapDataset;
	
	private boolean isMemoized;
	private final List<Predicate<ParsingOp<LogicalExpression>>> memoizedFilters;
	
	@SuppressWarnings("unchecked")
	public AbstractCreateDatasetWithExploration(IDataCollection<DI> trainingData,
			AbstractNeuralShiftReduceParser<Sentence, LogicalExpression> baseNeuralAmrParser,
			IValidator<DI, MR> validator, Integer beamSize, IParsingFilterFactory<DI, MR> parsingFilterFactory,
			CompositeImmutableLexicon<MR> compositeLexicon, ILexiconImmutable<MR> tempLexicon,
			GraphAmrParser amrOracleParser,
			IJointInferenceFilterFactory<DI, LogicalExpression, LogicalExpression, LogicalExpression> amrSupervisedFilterFactory) {
		super(trainingData, baseNeuralAmrParser, validator, beamSize, parsingFilterFactory, compositeLexicon, tempLexicon,
				amrOracleParser, amrSupervisedFilterFactory);
		
		this.trainingData = trainingData;
		this.baseNeuralAmrParser = baseNeuralAmrParser;
		this.beamSize = 500;//beamSize;
		this.maxSentenceLength = 15;/*10*/
		
		this.saveCreatedDataset = false;
		this.bootstrapDataset = false;
		
		this.isMemoized = false;
		this.memoizedFilters = new LinkedList<Predicate<ParsingOp<LogicalExpression>>>(); 
		
		this.amrParsingFilterFactory = new AMRParsingFilterFactory<SAMPLE, DI>(
				amrOracleParser, (IParsingFilterFactory<DI, LogicalExpression>)parsingFilterFactory, 
					(IJointInferenceFilterFactory<ILabeledDataItem<SituatedSentence<AMRMeta>, LogicalExpression>, LogicalExpression, 
							LogicalExpression, LogicalExpression>) amrSupervisedFilterFactory);
		
		this.validAmrParsingFilter = new AmrParsingFilter();
		
		LOG.setCustomLevel(LogLevel.INFO);
	}
	
	/** creates pre-processed datapoints from the given sentence and parseTree*/
	protected List<Dataset> preProcessDataPoints(Sentence sentence, DerivationState<MR> parseTree) {
		throw new RuntimeException("Operation not supported. Use direct extension of AbstractCreateDataset.");
	}
	
	/** creates pre-processed datapoints from the given situated sentence and parseTree*/
	protected List<Dataset> preProcessDataPoints(SituatedSentence<AMRMeta> situatedSentence, 
																			 DerivationState<MR> parseTree) {
		throw new RuntimeException("Operation not supported. Use direct extension of AbstractCreateDataset.");
	}

	
	/** creates pre-processed datapoints from the given situated sentence and parseTree*/
	protected abstract List<Dataset> preProcessDataPointsWithExploration(SituatedSentence<AMRMeta> situatedSentence, 
																	ShiftReduceDerivation<MR> derivations, int epoch);

	/** Creates data for situated model */
	@SuppressWarnings("unchecked")
	public List<Dataset> createDatasetWithExploration(IJointModelImmutable<SituatedSentence<AMRMeta>, 
															LogicalExpression, LogicalExpression> model, int epoch) {
		
		LOG.info("Data Creator With Exploration Statistics");
		LOG.info("Size of Raw dataset %s", this.trainingData.size());
		LOG.info("Data Creator Beam Size: %s ", this.beamSize);
		LOG.info("Size of Lexicon: %s", model.getLexicon().size());
		
		this.baseNeuralAmrParser.disablePacking();
		
		long totalParsingTime = 0;
		int parsed = 0;
					
		List<Dataset> processedDataSet =  new LinkedList<Dataset>();
		
		//Saved filters
		List<Predicate<ParsingOp<LogicalExpression>>> savedFilters = 
													new LinkedList<Predicate<ParsingOp<LogicalExpression>>>();
		
		if(this.bootstrapDataset) {
			try (
				    InputStream file = new FileInputStream("./dataset_filters.ser");
				    InputStream buffer = new BufferedInputStream(file);
				    ObjectInput input = new ObjectInputStream (buffer);
				) {
					savedFilters = (LinkedList<Predicate<ParsingOp<LogicalExpression>>>)input.readObject();
					LOG.info("Loaded filters. There are %s many of them.", savedFilters.size());
				} catch(Exception e) {
					throw new RuntimeException("Could not deserialize AMR parsing filter. Error: " + e);
				}
		}
		
		//Store filters for saving
		List<Predicate<ParsingOp<LogicalExpression>>> storedFilters = 
											new LinkedList<Predicate<ParsingOp<LogicalExpression>>>();
		
		int ex = 0;
		for (final DI dataItem : this.trainingData) {
			
			final SituatedSentence<AMRMeta> situatedSentence = (SituatedSentence<AMRMeta>) dataItem.getSample();
			final Sentence sentence = situatedSentence.getSample(); 
			
			// filter the dataItem based on length
			if(this.maxSentenceLength >=0 && sentence.getTokens().size() > this.maxSentenceLength) {
				LOG.warn("Sentence exceeding maximum sentence limit");
				continue;
			}
			
			ex++;
			
//			if(ex == 30) { //TODO temporary
//				ex--;
//				break;
//			}
			
			LOG.info("=========================");
			LOG.info("Utterance: %s", sentence);
			LOG.info("Meaning Representation: %s", dataItem.getLabel());
			
			final IJointDataItemModel<LogicalExpression, LogicalExpression> dataItemModel = 
														model.createJointDataItemModel(situatedSentence);
			final IJointDataItemModel<LogicalExpression, LogicalExpression> dataItemNewFeaturesModel = 
													this.modelNewFeatures.createJointDataItemModel(situatedSentence);

			//parse with the help of oracle parse			
			final Predicate<ParsingOp<LogicalExpression>> pruningFilter;
			if(this.bootstrapDataset) {
				pruningFilter = savedFilters.get(ex - 1);
				((CKYMultiParseTreeParsingFilter<LogicalExpression>)pruningFilter).clearCursor();
			} else {
				if(this.isMemoized) {
					pruningFilter = this.memoizedFilters.get(ex - 1);
					if(pruningFilter != null) {
						((CKYMultiParseTreeParsingFilter<LogicalExpression>)pruningFilter).clearCursor();
					}
				} else {
					pruningFilter = this.amrParsingFilterFactory.create(dataItem, dataItemModel);
					this.memoizedFilters.add(pruningFilter);
				}
			}
			
			storedFilters.add(pruningFilter);
			
			if(pruningFilter == null) {
				LOG.info("null filter. skipping");
				continue;
			}
			
			this.setDatasetCreateFilter(pruningFilter);
			
			ShiftReduceParserOutput<LogicalExpression> output = (ShiftReduceParserOutput<LogicalExpression>)
																		this.baseNeuralAmrParser.parse(sentence, this.validAmrParsingFilter, 
																				 dataItemNewFeaturesModel, true, null
																					/*model.getLexicon()*/, this.beamSize); 
			
			List<ShiftReduceDerivation<LogicalExpression>> derivations = output.getAllDerivations();
			
			LOG.info("Parsing time: %s", output.getParsingTime());
			totalParsingTime = totalParsingTime + output.getParsingTime();
			
			//check if correct logical form was derived
			ShiftReduceDerivation<LogicalExpression> correct = null;
			final Category<LogicalExpression> underspecifiedAmrFilterCategory;
			
			underspecifiedAmrFilterCategory = 
							((CKYMultiParseTreeParsingFilter<LogicalExpression>)pruningFilter).getCategory();
			
			if(underspecifiedAmrFilterCategory == null) {
				correct = null;
			} else {
				
				for(ShiftReduceDerivation<LogicalExpression> derivation: derivations) {
					//LOG.info("Derived logical form %s", derivation);
					if(underspecifiedAmrFilterCategory.equals(derivation.getCategory())) {
						correct = derivation;
						break;
					}
				}
				
				int numCorrect = 0;
				for(ShiftReduceDerivation<LogicalExpression> derivation: derivations) {
					if(underspecifiedAmrFilterCategory.equals(derivation.getCategory())) {
						numCorrect++;
					}
				}
				
				LOG.info("Shift Reduce Derivations %s. Number Correct %s", derivations.size(), numCorrect);
				
				////////////////////// debug /////////////////////
				if(correct == null) {
					
					((CKYMultiParseTreeParsingFilter<LogicalExpression>)pruningFilter).clearCursor();
					DerivationState.LOG.setCustomLevel(LogLevel.DEBUG);
					NeuralNetworkShiftReduceParser.LOG.setCustomLevel(LogLevel.DEBUG);
					this.baseNeuralAmrParser.parse(sentence, this.validAmrParsingFilter, 
							 dataItemNewFeaturesModel, true, model.getLexicon(), this.beamSize);
					LOG.info("CKY can parser a sentence that SR cannot with AMR Filter. This is a bug. Exiting");
					System.exit(0);
				}
				/////////////////////////////////////////////////
			}
			
			if(correct == null) { //failed to parse
				LOG.info("Failed to parse the utterance");
				continue;
			}
			
			parsed++;
			LOG.info("successfully parsed the utterance. ");

			LOG.info("Shift Reduce %s: Number of highest scoring parse trees %s. Number of parse Trees %s / Filter Size %s", 
											ex, correct.getMaxScoringDerivationStates().size(), correct.numParses(), 
											((CKYMultiParseTreeParsingFilter<LogicalExpression>)pruningFilter).numParseTrees());
				
				List<Dataset> preProcessedDataSetSample =  
								this.preProcessDataPointsWithExploration(situatedSentence, (ShiftReduceDerivation<MR>) correct, epoch);
				LOG.info("Generated % decision points", preProcessedDataSetSample.size());
				processedDataSet.addAll(preProcessedDataSetSample);
		}
		
		if(storedFilters.size() != ex) {
			throw new RuntimeException("Dataset filters and Effective Training data are of different size. "
									 + storedFilters.size() + " and " + " " + ex + " resp.");
		}
			
		/* Save the dataset. Since the entire dataset takes lots of space, therefore we
		 * only store the AMR filters.*/
		if(this.saveCreatedDataset) {		
			try (
				      OutputStream file = new FileOutputStream("./dataset_filters.ser");
				      OutputStream buffer = new BufferedOutputStream(file);
				      ObjectOutput output = new ObjectOutputStream(buffer);
				) {
					  LOG.info("Saved filter size %s", storedFilters.size());
				      output.writeObject(storedFilters);
				} catch(IOException e) {
				      throw new RuntimeException("Dataset Filters could not be saved. Exception " + e);
				}
		}
		
		this.setDatasetCreateFilter(null);
		
		this.isMemoized = true;
		this.baseNeuralAmrParser.enablePacking();
		
		LOG.info("Dataset size %s", processedDataSet.size());
		LOG.info("Num parsed %s / %s", parsed, ex);
		LOG.info("Total Parsing Time %s", totalParsingTime);
		LOG.info("Average Parsing Time %s", totalParsingTime/Math.max((double)ex, 1));
		
		return processedDataSet;
	}
	
	public IJointModelImmutable<SituatedSentence<AMRMeta>, LogicalExpression, LogicalExpression> getModelNewFeatures() {
		return this.modelNewFeatures;
	}
}
