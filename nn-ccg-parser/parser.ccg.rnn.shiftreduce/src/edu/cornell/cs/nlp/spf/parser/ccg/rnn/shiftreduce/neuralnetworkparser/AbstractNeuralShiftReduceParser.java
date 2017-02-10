package edu.cornell.cs.nlp.spf.parser.ccg.rnn.shiftreduce.neuralnetworkparser;

import java.util.List;
import java.util.function.Predicate;

import edu.cornell.cs.nlp.spf.ccg.lexicon.ILexiconImmutable;
import edu.cornell.cs.nlp.spf.data.sentence.Sentence;
import edu.cornell.cs.nlp.spf.parser.ParsingOp;
import edu.cornell.cs.nlp.spf.parser.ccg.model.IDataItemModel;
import edu.cornell.cs.nlp.spf.parser.graph.IGraphParser;
import edu.cornell.cs.nlp.spf.parser.graph.IGraphParserOutput;
import edu.cornell.cs.nlp.utils.log.ILogger;

/** A parser that defines functionalities for all neural shift reduce parser */

public interface AbstractNeuralShiftReduceParser<DI extends Sentence, MR> extends IGraphParser<DI, MR> {

	public void setDatasetCreatorFilter(Predicate<ParsingOp<MR>> datasetCreatorFilter);
	
	public IGraphParserOutput<MR> parserCatchEarlyErrors(DI dataItem, Predicate<ParsingOp<MR>> validAmrParsingFilter, IDataItemModel<MR> model_,
			boolean allowWordSkipping, ILexiconImmutable<MR> tempLexicon, Integer beamSize_);
	
	public void enablePacking();
	
	public void disablePacking();
	
	public default IGraphParserOutput<MR> parseSubSpan(DI dataItem, Predicate<ParsingOp<MR>> pruningFilter, IDataItemModel<MR> model_,
			boolean allowWordSkipping, ILexiconImmutable<MR> tempLexicon, Integer beamSize_, int spanStartIndex, int spanEndIndex) {
		throw new RuntimeException("Operation not permitted");
	}
	
	public default double doEarlyUpdatePerceptron(DI dataItem, Predicate<ParsingOp<MR>> pruningFilter, IDataItemModel<MR> model_,
			boolean allowWordSkipping, ILexiconImmutable<MR> tempLexicon, Integer beamSize_, List<ParsingOp<MR>> goldTree) {
		throw new RuntimeException("Operation not permitted");
	}
	
	public default ILogger getLOG() {
		return null;
	}
	
}
