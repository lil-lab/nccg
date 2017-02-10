package edu.uw.cs.lil.amr.util.dataprep;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.PrintStream;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedList;
import java.util.List;

import edu.cornell.cs.nlp.spf.data.singlesentence.SingleSentence;
import edu.cornell.cs.nlp.spf.data.singlesentence.SingleSentenceCollection;
import edu.cornell.cs.nlp.utils.log.ILogger;
import edu.cornell.cs.nlp.utils.log.Log;
import edu.cornell.cs.nlp.utils.log.LogLevel;
import edu.cornell.cs.nlp.utils.log.Logger;
import edu.cornell.cs.nlp.utils.log.LoggerFactory;
import edu.uw.cs.lil.amr.Init;
import edu.uw.cs.lil.amr.data.Tokenizer;

/**
 * Create the CMU data splits. Currently not used.
 *
 * @author Yoav Artzi
 */
@Deprecated
public class SplitData {
	public static final ILogger	LOG	= LoggerFactory.create(SplitData.class);

	public static void main(String[] args) throws IOException {
		// //////////////////////////////////////////
		// Init logging
		// //////////////////////////////////////////

		Logger.DEFAULT_LOG = new Log(System.err);
		Logger.setSkipPrefix(true);
		LogLevel.INFO.set();

		// //////////////////////////////////////////
		// Init AMR.
		// //////////////////////////////////////////

		Init.init(/* "../resources/amr.types" */new File(args[0]), true);

		// Output dir.
		final File outputDir = new File(args[1]);

		// Read data and split to train, dev, and test.
		final List<SingleSentence> train = new LinkedList<>();
		final List<SingleSentence> dev = new LinkedList<>();
		final List<SingleSentence> test = new LinkedList<>();
		for (int i = 2; i < args.length; ++i) {
			for (final SingleSentence dataItem : SingleSentenceCollection.read(
					new File(args[i]), new Tokenizer())) {
				final String fileValue = dataItem.getProperties().get("file");
				if (fileValue != null && fileValue.contains("_ENG_2008")) {
					test.add(dataItem);
				} else if (fileValue != null && fileValue.contains("_ENG_2007")) {
					dev.add(dataItem);
				} else if (fileValue != null && fileValue.contains("_ENG_")) {
					train.add(dataItem);
				} else {
					LOG.warn("Unidentified file ID: " + fileValue);
				}
			}
		}

		// Split to train-dev folds.
		final int numFolds = 5;
		final List<List<SingleSentence>> devFolds = new ArrayList<>(numFolds);
		for (int i = 0; i < numFolds; ++i) {
			devFolds.add(new LinkedList<SingleSentence>());
		}
		final List<SingleSentence> allFolds = new LinkedList<>();
		allFolds.addAll(train);
		allFolds.addAll(dev);
		Collections.shuffle(allFolds);
		int i = 0;
		for (final SingleSentence dataItem : allFolds) {
			devFolds.get(i % numFolds).add(dataItem);
			++i;
		}

		// Write files.
		try {
			write(train, new File(outputDir, "train.lam"));
			write(dev, new File(outputDir, "dev.lam"));
			write(test, new File(outputDir, "test.lam"));
			for (i = 0; i < numFolds; ++i) {
				write(devFolds.get(i),
						new File(outputDir, String.format("fold%d.lam", i)));
			}
		} catch (final FileNotFoundException e) {
			throw new IllegalStateException(e);
		}
	}

	private static void write(List<SingleSentence> data, File outputFile)
			throws FileNotFoundException {
		try (PrintStream out = new PrintStream(outputFile)) {

			for (final SingleSentence dataItem : data) {
				out.println(dataItem);
				out.println();
			}
		}
	}

}
