package edu.cornell.cs.nlp.spf.parser.ff.shiftreduce.neuralparser;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.function.Predicate;

import com.google.common.base.Joiner;

import edu.cornell.cs.nlp.spf.ccg.lexicon.ILexiconImmutable;
import edu.cornell.cs.nlp.spf.data.sentence.Sentence;
import edu.cornell.cs.nlp.spf.explat.IResourceRepository;
import edu.cornell.cs.nlp.spf.explat.ParameterizedExperiment.Parameters;
import edu.cornell.cs.nlp.spf.explat.resources.IResourceObjectCreator;
import edu.cornell.cs.nlp.spf.explat.resources.usage.ResourceUsage;
import edu.cornell.cs.nlp.spf.parser.ParsingOp;
import edu.cornell.cs.nlp.spf.parser.ccg.model.IDataItemModel;
import edu.cornell.cs.nlp.spf.parser.ccg.rnn.shiftreduce.neuralnetworkparser.AbstractNeuralShiftReduceParser;
import edu.cornell.cs.nlp.spf.parser.ccg.shiftreduce.ShiftReduceDerivation;
import edu.cornell.cs.nlp.spf.parser.ccg.shiftreduce.ShiftReduceParserOutput;
import edu.cornell.cs.nlp.spf.parser.ccg.shiftreduce.stacks.DerivationState;
import edu.cornell.cs.nlp.spf.parser.ff.shiftreduce.learner.NeuralFeedForwardDotProductLearner;
import edu.cornell.cs.nlp.spf.parser.graph.IGraphParserOutput;
import edu.cornell.cs.nlp.utils.log.ILogger;
import edu.cornell.cs.nlp.utils.log.LoggerFactory;

	/** 
	 * Parses utterance using an ensemble of feed forward neural network shift reduce parser 
	 * with a dot product objective making it faster.
	 * @author Dipendra Misra
	 */
	public class EnsembleShiftReduceParser<DI extends Sentence, MR> 
					implements AbstractNeuralShiftReduceParser<DI, MR> {
		
		private static final long serialVersionUID = -662119526885065739L;

		public static final ILogger								LOG
							= LoggerFactory.create(EnsembleShiftReduceParser.class);

		private final List<AbstractNeuralShiftReduceParser<DI, MR>> ensemble;
		private final List<Double> mixingProbability;
		private final Predicate<ParsingOp<MR>> pruningFilter;
		
		/** TODO -- separate learning parts from other components */
		public EnsembleShiftReduceParser(List<AbstractNeuralShiftReduceParser<DI, MR>> ensemble, 
										List<Double> mixingProbability) {
		
			this.ensemble = ensemble;
			this.mixingProbability = mixingProbability;
			this.pruningFilter = null;
			
			double sum = 0.0;
			for(double prob: this.mixingProbability) {
				if(prob < 0) {
					LOG.warn("Mixing probability less than 0. Found %s", prob);
					System.exit(0);
				}
				sum = sum + prob;
			}
			
			if(sum != 1.0) {
				LOG.warn("Mixing probabilities dont sum to 1. Found %s", sum);
				System.exit(0);
			}
			
			LOG.info("Ensemble of %s Shift Reduce Parser. Mixing probability %s sum to 1.0 verified", 
										this.ensemble.size(), Joiner.on(", ").join(this.mixingProbability));
		}
		
		@Override
		public void enablePacking() {
			for(AbstractNeuralShiftReduceParser<DI, MR> parser: ensemble) {
				parser.enablePacking();
			}
		}
		
		@Override
		public void disablePacking() {
			for(AbstractNeuralShiftReduceParser<DI, MR> parser: ensemble) {
				parser.disablePacking();
			}
		}
		
		@Override
		public void setDatasetCreatorFilter(Predicate<ParsingOp<MR>> datasetCreatorFilter) {
			
			for(AbstractNeuralShiftReduceParser<DI, MR> parser: ensemble) {
				parser.setDatasetCreatorFilter(datasetCreatorFilter);
			}
		}
		
		@Override
		public IGraphParserOutput<MR> parse(DI dataItem,
				IDataItemModel<MR> model) {
			return this.parse(dataItem, this.pruningFilter, model,
			     false, null, null);
		}
		
		@Override
		public IGraphParserOutput<MR> parse(DI dataItem,
				IDataItemModel<MR> model, boolean allowWordSkipping) {
		
			return this.parse(dataItem, this.pruningFilter, model,
					allowWordSkipping, null, null);
		}
		
		@Override
		public IGraphParserOutput<MR> parse(DI dataItem, IDataItemModel<MR> model,
				boolean allowWordSkipping, ILexiconImmutable<MR> tempLexicon) {
			return this.parse(dataItem, this.pruningFilter, model,
					allowWordSkipping, tempLexicon, null);
		}
	
		@Override
		public IGraphParserOutput<MR> parse(DI dataItem, IDataItemModel<MR> model,
				boolean allowWordSkipping, ILexiconImmutable<MR> tempLexicon,
				Integer beamSize) {
			return this.parse(dataItem, this.pruningFilter, model,
					allowWordSkipping, tempLexicon, beamSize);
		}
	
		@Override
		public IGraphParserOutput<MR> parse(DI dataItem, Predicate<ParsingOp<MR>> filter,
				IDataItemModel<MR> model) {
			return this.parse(dataItem, filter, model, false, null, null);
		}
	
		@Override
		public IGraphParserOutput<MR> parse(DI dataItem, Predicate<ParsingOp<MR>> filter,
				IDataItemModel<MR> model, boolean allowWordSkipping) {
			return this.parse(dataItem, filter, model,
					allowWordSkipping, null, null);
		}
	
		@Override
		public IGraphParserOutput<MR> parse(DI dataItem, Predicate<ParsingOp<MR>> filter,
				IDataItemModel<MR> model, boolean allowWordSkipping,
				ILexiconImmutable<MR> tempLexicon) {
			return this.parse(dataItem, filter, model,
					allowWordSkipping, tempLexicon, null);
		}
		
		private void bootstrapParser(NeuralDotProductShiftReduceParser<DI, MR> basicParser, String folderName) {
			
			NeuralFeedForwardDotProductLearner.bootstrap(folderName, basicParser);
			
			final FeatureEmbedding<?> actionFeatureEmbedding = basicParser.getActionFeatureEmbedding();
			final FeatureEmbedding<?> stateFeatureEmbedding = basicParser.getStateFeatureEmbedding();
			
			actionFeatureEmbedding.stopAddingFeatures();
			actionFeatureEmbedding.stats();
			actionFeatureEmbedding.clearSeenFeaturesStats();
			
			stateFeatureEmbedding.stopAddingFeatures();
			stateFeatureEmbedding.stats();
			stateFeatureEmbedding.clearSeenFeaturesStats();
			
			basicParser.testing = true;
		}
		
		private void readObject(java.io.ObjectInputStream in) throws IOException, ClassNotFoundException {
			
			in.defaultReadObject();
			
			//Hack for now
			NeuralDotProductShiftReduceParser<DI, MR> basicParser1 = (NeuralDotProductShiftReduceParser<DI, MR>) this.ensemble.get(0);
			NeuralDotProductShiftReduceParser<DI, MR> basicParser2 = (NeuralDotProductShiftReduceParser<DI, MR>) this.ensemble.get(1);
			NeuralDotProductShiftReduceParser<DI, MR> basicParser3 = (NeuralDotProductShiftReduceParser<DI, MR>) this.ensemble.get(2);
			
			
			//Load
			this.bootstrapParser(basicParser1, "epoch-3_1463521436781");
			this.bootstrapParser(basicParser2, "epoch-3_1463475672657");
			this.bootstrapParser(basicParser3, "epoch-3_1463477737630");
			
			LOG.info("Boostrapped the worker parser");
		}
	
		/** Parses a sentence using Neural Network model */
		@Override
		public IGraphParserOutput<MR> parse(DI dataItem, Predicate<ParsingOp<MR>> pruningFilter, IDataItemModel<MR> model_,
				boolean allowWordSkipping, ILexiconImmutable<MR> tempLexicon, Integer beamSize_) {
		
			final long start = System.currentTimeMillis();
			
			List<ShiftReduceParserOutput<MR>> parserOutputs = new ArrayList<ShiftReduceParserOutput<MR>>();
			
			//If it contains unstitched option then we only consider them since
			//stitched output are noisy
			boolean containsUnstitched = false;
			
			// Parse using parsers in the ensemble
			for(AbstractNeuralShiftReduceParser<DI, MR> parser: this.ensemble) {
				
				ShiftReduceParserOutput<MR> output = (ShiftReduceParserOutput<MR>) parser.parse(dataItem, pruningFilter, model_,
														allowWordSkipping, tempLexicon, beamSize_);
				
				parserOutputs.add(output);
				if(!output.isStiched()) {
					containsUnstitched = true;
				}
			}
			
			if(containsUnstitched) {
				
				List<ShiftReduceParserOutput<MR>> unStitchedParserOutputs = new ArrayList<ShiftReduceParserOutput<MR>>();
				for(ShiftReduceParserOutput<MR> output: parserOutputs) {
					if(!output.isStiched()) {
						unStitchedParserOutputs.add(output);
					}
				}
				
				final int numOutput = parserOutputs.size();
				parserOutputs.clear();
				parserOutputs.addAll(unStitchedParserOutputs);
				LOG.info("Considering only unstitched options. Total %s, unstitched output %s", numOutput, parserOutputs.size());
			}
			
			Iterator<Double> mixProbabilityIt = this.mixingProbability.iterator();
			
			if(!containsUnstitched) {
				// all derivations are created by stitching
				
				List<ShiftReduceDerivation<MR>> newDerivations = new ArrayList<ShiftReduceDerivation<MR>>();
				
				for(ShiftReduceParserOutput<MR> output: parserOutputs) {
					
					final double modelLogProb = Math.log(mixProbabilityIt.next());
					
					List<ShiftReduceDerivation<MR>> derivations = output.getAllDerivations();
					for(ShiftReduceDerivation<MR> derivation: derivations) {
						List<DerivationState<MR>> derivationDStates = derivation.getAllDerivationStates();
						
						if(derivationDStates.size() != 1) {
							throw new RuntimeException("Assumes derivation pack one tree");
						}
						
						DerivationState<MR> dstate = derivationDStates.get(0);
						dstate.score = dstate.score + modelLogProb;
						
						ShiftReduceDerivation<MR> newDerivation = new ShiftReduceDerivation<MR>(dstate, derivation.getCategory());
						newDerivations.add(newDerivation);
					}
				}
				
				final ShiftReduceParserOutput<MR> output = 
						new ShiftReduceParserOutput<MR>(newDerivations, System.currentTimeMillis() - start, true);
				return output;
			} else {
			
				// Merge these derivations
				// We are working with unstitched derivations only
				List<DerivationState<MR>> dstates = new ArrayList<DerivationState<MR>>();
		
				for(ShiftReduceParserOutput<MR> output: parserOutputs) {
					
					final double modelLogProb = Math.log(mixProbabilityIt.next());
					
					List<ShiftReduceDerivation<MR>> derivations = output.getAllDerivations();
					for(ShiftReduceDerivation<MR> derivation: derivations) {
						List<DerivationState<MR>> derivationDStates = derivation.getAllDerivationStates();
						
						for(DerivationState<MR> dstate: derivationDStates) {
							dstate.score = dstate.score + modelLogProb;
							dstates.add(dstate);
						}
					}
				}
				
				final ShiftReduceParserOutput<MR> output = 
							new ShiftReduceParserOutput<MR>(dstates, System.currentTimeMillis() - start);
				
				return output;
			}
		}
		
		@Override
		public IGraphParserOutput<MR> parserCatchEarlyErrors(DI dataItem,
				Predicate<ParsingOp<MR>> validAmrParsingFilter, IDataItemModel<MR> model_, boolean allowWordSkipping,
				ILexiconImmutable<MR> tempLexicon, Integer beamSize_) {
			throw new RuntimeException("Operation not supported");
		}
		
		public static class Builder<DI extends Sentence, MR> {
			
			private List<AbstractNeuralShiftReduceParser<DI, MR>> ensemble = 
										new ArrayList<AbstractNeuralShiftReduceParser<DI, MR>>();
			private List<Double> mixingProb = new ArrayList<Double>();
			
			public EnsembleShiftReduceParser<DI, MR> build() {
				return new EnsembleShiftReduceParser<DI, MR>(this.ensemble, this.mixingProb);
			}
			
			public Builder<DI, MR> addParser(AbstractNeuralShiftReduceParser<DI, MR> parser) {
				this.ensemble.add(parser);
				return this;
			}
			
			public Builder<DI, MR> addProbability(double prob) {
				this.mixingProb.add(prob);
				return this;
			}
		}
		
		public static class Creator<DI extends Sentence, MR>
				implements IResourceObjectCreator<EnsembleShiftReduceParser<DI, MR>> {
	
			private final String type;
			
			public Creator() {
				this("parser.ensemble.neural.shiftreduce");
			}
	
			public Creator(String type) {
				this.type = type;
			}
			
			@Override
			public EnsembleShiftReduceParser<DI, MR> create(Parameters params, IResourceRepository repo) {
				
				final Builder<DI, MR> builder = new Builder<DI, MR>();
				
				for (final String parserRef : params.getSplit("parsers")) {
					final AbstractNeuralShiftReduceParser<DI, MR> parser = repo.get(parserRef);
					builder.addParser(parser);
				}
				
				for (final String prob : params.getSplit("probs")) {
					builder.addProbability(Double.parseDouble(prob));
				}
								
				return builder.build();
			}
	
			@Override
			public String type() {
				return this.type;
			}
	
			@Override
			public ResourceUsage usage() {
				// TODO Auto-generated method stub
				return null;
			}
		}	
	}