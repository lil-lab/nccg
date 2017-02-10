package edu.cornell.cs.nlp.spf.parser.ccg.shiftreduce.logger;

import java.io.File;
import java.io.PrintWriter;
import java.io.Serializable;
import java.util.Iterator;
import java.util.List;

import edu.cornell.cs.nlp.spf.base.hashvector.IHashVector;
import edu.cornell.cs.nlp.spf.base.hashvector.KeyArgs;
import edu.cornell.cs.nlp.spf.data.sentence.Sentence;
import edu.cornell.cs.nlp.spf.parser.ccg.shiftreduce.ShiftReduceDerivation;
import edu.cornell.cs.nlp.spf.parser.ccg.shiftreduce.ShiftReduceParserOutput;
import edu.cornell.cs.nlp.spf.parser.ccg.shiftreduce.stacks.DerivationState;
import edu.cornell.cs.nlp.spf.parser.ccg.shiftreduce.steps.IWeightedShiftReduceStep;
import edu.cornell.cs.nlp.utils.composites.Pair;
import edu.cornell.cs.nlp.utils.log.ILogger;
import edu.cornell.cs.nlp.utils.log.LoggerFactory;

public class ShiftReduceParseTreeLogger<DI extends Sentence, MR> implements Serializable {
	
	private static final long serialVersionUID = -274615442315647796L;

	public static final ILogger LOG = LoggerFactory.create(ShiftReduceParseTreeLogger.class);

	private final File	outputDir;
	
	public ShiftReduceParseTreeLogger(File outputDir) {
		this.outputDir = outputDir;
	}
	
	private void logDerivationState(DerivationState<MR> dstate, PrintWriter writer, double derivationScore) {
		
		try {
			
			if(dstate.score == derivationScore) {
				writer.write("Viterbi dstate*\n");
			}
			
			DerivationState<MR> it = dstate;
			
			while(it != null) {
				//print step along with active features
				IWeightedShiftReduceStep<MR> step = it.returnStep();
				if(step != null) {
					IHashVector feature = it.possibleActionFeatures().get(it.childIndex);
					
					StringBuilder featureString = new StringBuilder();
					Iterator<Pair<KeyArgs, Double>> featureIt = feature.iterator();
					while(featureIt.hasNext()) {
						featureString.append(", "  + featureIt.next().first());
					}
					
					writer.write("[" + step.getStart() + "-" + step.getEnd() + ", " + step.getStepScore() + ", " +  
							      it.score + ", " +  step.getRuleName() + ", " + step.getRoot() + ", " + featureString.toString() + "]\n");
				}
				it = it.getParent();
			}
		} catch(Exception e) {
			LOG.warn("Cannot save shift reduce parse trees. Exception " + e);
		}
	}

	public void log(ShiftReduceParserOutput<MR> parserOutput, DI dataItem, boolean allowSkipping) {
		
		final long start = System.currentTimeMillis();
		try {
			
			final String fileName = this.outputDir.getAbsolutePath() + "/trees/" 
													+ System.currentTimeMillis() + "_" + allowSkipping;
			final PrintWriter writer = new PrintWriter(fileName);
			
			writer.write("Text: " + dataItem.toString() + "\n");
			
			List<ShiftReduceDerivation<MR>> derivations = parserOutput.getAllDerivations();
			//Best score derivations are derivations
			final double bestScore = parserOutput.getBestDerivations().get(0).getScore();
	
			int ix = 0;
			for(ShiftReduceDerivation<MR> derivation: derivations) {
				
				final double derivationScore = derivation.getScore();
				if(derivationScore == bestScore) {
					writer.write("Derivation** #" + (++ix) + ": Category " + derivation.getCategory() + "\n");
				} else {
					writer.write("Derivation #" + (++ix) + ": Category " + derivation.getCategory() + "\n");
				}
				
				List<DerivationState<MR>> dstates = derivation.getAllDerivationStates();
				for(DerivationState<MR> dstate: dstates) {
					this.logDerivationState(dstate, writer, derivationScore);
					writer.write("\n");
				}
			}
			
			writer.flush();
			writer.close();
			LOG.info("Logged parse trees in %s", fileName);
		} catch(Exception e) {
			LOG.warn("Cannot save shift reduce parse trees. Exception " + e);
		}
		
		LOG.info("Logging time %s", (System.currentTimeMillis() - start));
	}
}
