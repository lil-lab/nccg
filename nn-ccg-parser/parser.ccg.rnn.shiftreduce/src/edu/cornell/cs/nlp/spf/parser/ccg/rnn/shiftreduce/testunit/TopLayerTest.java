package edu.cornell.cs.nlp.spf.parser.ccg.rnn.shiftreduce.testunit;

import java.util.ArrayList;
import java.util.List;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import edu.cornell.cs.nlp.spf.parser.ccg.rnn.shiftreduce.embeddings.TopLayerMLP;
import edu.cornell.cs.nlp.spf.parser.ccg.rnn.shiftreduce.parsingopembeddings.ParsingOpEmbeddingResult;
import edu.cornell.cs.nlp.utils.composites.Pair;

public class TopLayerTest {
	
	public static Pair<Double, List<Double>[]> calcLoss(List<Double>[] allExponents, List<Integer> gTruths) {
		
		int numDecisions = allExponents.length;
		@SuppressWarnings("unchecked")
		List<Double>[] probs = new List[numDecisions];
		double loss = 0.0;
		for(int j = 0; j < numDecisions; j++) {
			
			String s = "";
			probs[j] = new ArrayList<Double>();
			List<Double> unNormProbs = new ArrayList<Double>();
			
			double Z = 0;
			
			for(Double exponent: allExponents[j]) {
				s = s + ", " + exponent;
				double score = Math.exp(exponent);
				unNormProbs.add(score);
				Z = Z + score;
			}
			
			for(Double score: unNormProbs) {
				probs[j].add(score/Z);
			}
			
			loss = loss - Math.log(probs[j].get(gTruths.get(j)));
		}
		
		return Pair.of(loss, probs);
	}

	public static void main(String[] args) throws Exception {
		
		Nd4j.getRandom().setSeed(1234);
		
		final boolean checkInputGradient = true; 
		int nState = 10, nParsingOp = 30;
		int nIn = nParsingOp + nState;
		TopLayerMLP topLayer = new TopLayerMLP(nIn, 0.01, 0.0000, 1211);
		
		int numDecisions = 4;
		List<Integer> possibleOptions = new ArrayList<Integer>(numDecisions);
		
		possibleOptions.add(10);
		possibleOptions.add(3);
		possibleOptions.add(5);
		possibleOptions.add(17);
		
		List<Integer> gTruths = new ArrayList<Integer>(numDecisions);
		
		gTruths.add(7);
		gTruths.add(0);
		gTruths.add(4);
		gTruths.add(14);
		
		INDArray[] stateEmbeddings = new INDArray[numDecisions];
		@SuppressWarnings("unchecked")
		List<ParsingOpEmbeddingResult>[] results = new List[numDecisions];
		
		for(int i = 0; i < numDecisions; i++) {
			stateEmbeddings[i] = Nd4j.rand(new int[]{1, nState});
			
			results[i] = new ArrayList<ParsingOpEmbeddingResult>();
			
			for(int j = 0; j < possibleOptions.get(i); j++) {
				INDArray embedding = Nd4j.rand(new int[]{1, nParsingOp});
				ParsingOpEmbeddingResult op = new ParsingOpEmbeddingResult(-1, embedding, null, null, null, null);
				results[i].add(op);
			}
		}
		
		int epoch = 100;
		double empirical = 0;
		for(int i = 1; i < epoch; i++) {
			
			if(checkInputGradient) {
				
				double epsilon = 0.01;
				
				INDArray vec = results[0].get(0).getEmbedding();
				double orig = vec.getDouble(new int[]{0, 0});
				
				vec.putScalar(new int[]{0, 0}, orig + epsilon);
				List<Double>[] allExponents = topLayer.getEmbedding(results, stateEmbeddings);
				double loss1 = TopLayerTest.calcLoss(allExponents, gTruths).first();
				
				vec.putScalar(new int[]{0, 0}, orig - epsilon);
				allExponents = topLayer.getEmbedding(results, stateEmbeddings);
				double loss2 = TopLayerTest.calcLoss(allExponents, gTruths).first();
				
				empirical = (loss1 - loss2)/(2*epsilon);
				
				vec.putScalar(new int[]{0, 0}, orig);
			}
			
			List<Double>[] allExponents = topLayer.getEmbedding(results, stateEmbeddings);
			
			//compute loss
			Pair<Double, List<Double>[]> lossAndProbs = TopLayerTest.calcLoss(allExponents, gTruths);
			
			System.out.println("Step " + i + " loss " + lossAndProbs.first());
			List<Double>[] probs = lossAndProbs.second();
			
			@SuppressWarnings("unchecked")
			List<Double>[] errorExponents = new List[numDecisions];
			
			//pass gradients
			for(int j = 0; j < numDecisions; j++) {
				
				errorExponents[j] = new ArrayList<Double>();
				
				for(int k = 0; k < probs[j].size(); k++) {
					
					final double error;
					if(k == gTruths.get(j)) {
						error = -1 + probs[j].get(k);
					} else {
						error = probs[j].get(k);
					}
					
					errorExponents[j].add(error);
				}
				
			}
			
			List<INDArray> [] inputError = topLayer.backprop(errorExponents);
			
			if(checkInputGradient) {
				//Estimated Gradient
				double estimate = inputError[0].get(0).getDouble(new int[]{0, 0});
				System.out.println("Gradient Estimate " + estimate + " Empirical " + empirical);
				//System.exit(0);
			}
		}
	}
}
