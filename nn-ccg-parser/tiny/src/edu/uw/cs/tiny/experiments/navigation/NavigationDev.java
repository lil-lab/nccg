/*******************************************************************************
 * UW SPF - The University of Washington Semantic Parsing Framework
 * <p>
 * Copyright (C) 2013 Yoav Artzi
 * <p>
 * This program is free software; you can redistribute it and/or modify it under
 * the terms of the GNU General Public License as published by the Free Software
 * Foundation; either version 2 of the License, or any later version.
 * <p>
 * This program is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
 * details.
 * <p>
 * You should have received a copy of the GNU General Public License along with
 * this program; if not, write to the Free Software Foundation, Inc., 51
 * Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.
 ******************************************************************************/
/**
 * Yoav Artzi, Computer Science, University of Washington
 */
package edu.uw.cs.tiny.experiments.navigation;

import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Set;

import edu.uw.cs.lil.tiny.ccg.categories.ICategoryServices;
import edu.uw.cs.lil.tiny.ccg.categories.syntax.Syntax;
import edu.uw.cs.lil.tiny.ccg.lexicon.Lexicon;
import edu.uw.cs.lil.tiny.ccg.lexicon.factored.lambda.Lexeme;
import edu.uw.cs.lil.tiny.data.IDataItem;
import edu.uw.cs.lil.tiny.data.sentence.Sentence;
import edu.uw.cs.lil.tiny.data.singlesentence.SingleSentenceDataset;
import edu.uw.cs.lil.tiny.genlex.ccg.unification.split.IUnificationSplitter;
import edu.uw.cs.lil.tiny.genlex.ccg.unification.split.Splitter;
import edu.uw.cs.lil.tiny.learn.ubl.AbstractUBL;
import edu.uw.cs.lil.tiny.learn.ubl.LexicalSplittingCountScorer;
import edu.uw.cs.lil.tiny.learn.ubl.UBLStocGradient;
import edu.uw.cs.lil.tiny.mr.lambda.FlexibleTypeComparator;
import edu.uw.cs.lil.tiny.mr.lambda.LogicLanguageServices;
import edu.uw.cs.lil.tiny.mr.lambda.LogicalConstant;
import edu.uw.cs.lil.tiny.mr.lambda.LogicalExpression;
import edu.uw.cs.lil.tiny.mr.lambda.Ontology;
import edu.uw.cs.lil.tiny.mr.lambda.ccg.LogicalExpressionCategoryServices;
import edu.uw.cs.lil.tiny.mr.lambda.ccg.SimpleFullParseFilter;
import edu.uw.cs.lil.tiny.mr.lambda.primitivetypes.GenericRecursiveSimplifier;
import edu.uw.cs.lil.tiny.mr.language.type.TypeRepository;
import edu.uw.cs.lil.tiny.parser.ccg.cky.CKYBinaryParsingRule;
import edu.uw.cs.lil.tiny.parser.ccg.cky.single.CKYParser;
import edu.uw.cs.lil.tiny.parser.ccg.cky.single.CKYUnaryParsingRule;
import edu.uw.cs.lil.tiny.parser.ccg.factoredlex.features.scorers.LexicalEntryLexemeBasedScorer;
import edu.uw.cs.lil.tiny.parser.ccg.features.basic.LexicalFeatureSet;
import edu.uw.cs.lil.tiny.parser.ccg.features.basic.scorer.ExpLengthLexicalEntryScorer;
import edu.uw.cs.lil.tiny.parser.ccg.features.basic.scorer.ScalingScorer;
import edu.uw.cs.lil.tiny.parser.ccg.features.basic.scorer.SkippingSensitiveLexicalEntryScorer;
import edu.uw.cs.lil.tiny.parser.ccg.features.lambda.LogicalExpressionCoordinationFeatureSet;
import edu.uw.cs.lil.tiny.parser.ccg.features.lambda.LogicalExpressionTypeFeatureSet;
import edu.uw.cs.lil.tiny.parser.ccg.model.Model;
import edu.uw.cs.lil.tiny.parser.ccg.rules.primitivebinary.BackwardApplication;
import edu.uw.cs.lil.tiny.parser.ccg.rules.primitivebinary.BackwardComposition;
import edu.uw.cs.lil.tiny.parser.ccg.rules.primitivebinary.ForwardApplication;
import edu.uw.cs.lil.tiny.parser.ccg.rules.primitivebinary.ForwardComposition;
import edu.uw.cs.lil.tiny.parser.ccg.rules.skipping.BackwardSkippingRule;
import edu.uw.cs.lil.tiny.parser.ccg.rules.skipping.ForwardSkippingRule;
import edu.uw.cs.lil.tiny.test.Tester;
import edu.uw.cs.lil.tiny.test.stats.ExactMatchTestingStatistics;
import edu.uw.cs.lil.tiny.utils.string.StubStringFilter;
import edu.uw.cs.utils.collections.IScorer;
import edu.uw.cs.utils.collections.SetUtils;
import edu.uw.cs.utils.log.ILogger;
import edu.uw.cs.utils.log.Log;
import edu.uw.cs.utils.log.LogLevel;
import edu.uw.cs.utils.log.Logger;
import edu.uw.cs.utils.log.LoggerFactory;

/**
 * The UBL learning experiment setup.
 * 
 * @author Luke Zettlemoyer
 */
public class NavigationDev {
	private static final ILogger	LOG	= LoggerFactory
												.create(NavigationDev.class
														.getName());
	
	public static void main(String[] args) {
		Logger.DEFAULT_LOG = new Log(System.out);
		Logger.setSkipPrefix(true);
		LogLevel.INFO.set();
		
		// Record start time
		final long startTime = System.currentTimeMillis();
		
		// Data directories
		final String expDir = "experiments/navigation";
		final String expDataDir = expDir + "/data";
		
		// Init the logical expression type system
		LogicLanguageServices.setInstance(new LogicLanguageServices.Builder(
				new TypeRepository(new File(expDataDir + "/nav.types")))
				.setNumeralTypeName("i")
				.setTypeComparator(new FlexibleTypeComparator()).build());
		
		// CCG LogicalExpression category services for handling categories
		// with LogicalExpression as semantics
		final ICategoryServices<LogicalExpression> categoryServices = new LogicalExpressionCategoryServices(
				false, true); // passing false means no type checking
		
		// Load the ontology
		final List<File> ontologyFiles = new LinkedList<File>();
		ontologyFiles.add(new File(expDataDir + "/nav.preds.ont"));
		ontologyFiles.add(new File(expDataDir + "/nav.consts.ont"));
		try {
			// Ontology is currently not used, so we are just reading it, not
			// storing
			new Ontology(ontologyFiles);
		} catch (final IOException e) {
			throw new RuntimeException(e);
		}
		
		// Set similifier for do-seq predicate
		LogicLanguageServices
				.instance()
				.setSimplifier(
						(LogicalConstant) categoryServices
								.parseSemantics("do-sequentially:<t*,t>"),
						GenericRecursiveSimplifier.INSTANCE, true);
		
		// Read the training and testing sets
		// In our case the train and test set are identical
		System.out.println("loading train");
		final SingleSentenceDataset train = SingleSentenceDataset.read(
				new File(expDataDir + "/nav.train"), new StubStringFilter(),
				true);
		System.out.println("loading test");
		final SingleSentenceDataset test = SingleSentenceDataset.read(new File(
				expDataDir + "/nav.test.wasp"), new StubStringFilter(), true);
		LOG.info("Train Size: " + train.size());
		LOG.info("Test Size: " + test.size());
		
		// Init the lexicon
		final Lexicon<LogicalExpression> fixed = new Lexicon<LogicalExpression>();
		// fixed.addEntriesFromFile(expDir + "/np-fixedlex.nav",
		// categoryServices);
		// fixed.addEntriesFromFile(expDataDir + "/nav-seed.lex",
		// categoryServices);
		
		// Create the category splitter
		final IUnificationSplitter splitter = new Splitter(categoryServices);
		
		// Create the lexical feature set
		// final LexicalCooccuranceScorer gizaScores = LexicalCooccuranceScorer
		// .loadFromFile(expDataDir + "/nav.train.giza_probs");
		
		final IScorer<Lexeme> scorer = new ScalingScorer<Lexeme>(10.0,
				new LexicalSplittingCountScorer.Builder(splitter, train,
						categoryServices).build());
		final LexicalFeatureSet<IDataItem<Sentence>, LogicalExpression> lexPhi = new LexicalFeatureSet.Builder<IDataItem<Sentence>, LogicalExpression>()
				.setInitialFixedScorer(
						new ExpLengthLexicalEntryScorer<LogicalExpression>(
								10.0, 1.1))
				.setInitialScorer(
						new SkippingSensitiveLexicalEntryScorer<LogicalExpression>(
								categoryServices.getEmptyCategory(), -1.0,
								new LexicalEntryLexemeBasedScorer(scorer)))
				.build();
		
		// Create the entire feature collection
		final Model<IDataItem<Sentence>, LogicalExpression> model = new Model.Builder<IDataItem<Sentence>, LogicalExpression>()
				.addParseFeatureSet(
						new LogicalExpressionCoordinationFeatureSet<IDataItem<Sentence>>(
								true, true, true))
				.addParseFeatureSet(
						new LogicalExpressionTypeFeatureSet<IDataItem<Sentence>>())
				.addLexicalFeatureSet(lexPhi).build();
		
		// Initialize lexical features. This is not "natural" for every lexical
		// feature set, only for this one, so it's done here and not on all
		// lexical feature sets.
		model.addFixedLexicalEntries(fixed.toCollection());
		
		// extra functionality for synonyms, ordinals, etc
		final Map<List<String>, LogicalExpression> numbers = new HashMap<List<String>, LogicalExpression>();
		for (int i = 0; i < 20; i++) {
			final List<String> tokens = new LinkedList<String>();
			tokens.add(String.valueOf(i));
			numbers.put(tokens, categoryServices.parseSemantics(i + ":n"));
		}
		
		numbers.put(Arrays.asList(new String[] { "first" }),
				categoryServices.parseSemantics("1:n"));
		numbers.put(Arrays.asList(new String[] { "second" }),
				categoryServices.parseSemantics("2:n"));
		numbers.put(Arrays.asList(new String[] { "third" }),
				categoryServices.parseSemantics("3:n"));
		numbers.put(Arrays.asList(new String[] { "fourth" }),
				categoryServices.parseSemantics("4:n"));
		numbers.put(Arrays.asList(new String[] { "fifth" }),
				categoryServices.parseSemantics("5:n"));
		numbers.put(Arrays.asList(new String[] { "sixth" }),
				categoryServices.parseSemantics("6:n"));
		numbers.put(Arrays.asList(new String[] { "seventh" }),
				categoryServices.parseSemantics("7:n"));
		numbers.put(Arrays.asList(new String[] { "eigth" }),
				categoryServices.parseSemantics("8:n"));
		numbers.put(Arrays.asList(new String[] { "ninth" }),
				categoryServices.parseSemantics("9:n"));
		numbers.put(Arrays.asList(new String[] { "tenth" }),
				categoryServices.parseSemantics("10:n"));
		
		final Set<Set<List<String>>> synSets = new HashSet<Set<List<String>>>();
		final Set<List<String>> synSet1 = new HashSet<List<String>>();
		synSet1.add(Arrays.asList(new String[] { "intersection" }));
		synSet1.add(Arrays.asList(new String[] { "intersections" }));
		synSet1.add(Arrays.asList(new String[] { "junction" }));
		synSet1.add(Arrays.asList(new String[] { "junctions" }));
		synSets.add(synSet1);
		
		final SynonymLexicalGenerator synSub = new SynonymLexicalGenerator(
				model.getLexicon(), synSets, numbers);
		
		// Create the parser -- support empty rule
		final CKYParser<LogicalExpression> parser = new CKYParser.Builder<LogicalExpression>(
				categoryServices, new SimpleFullParseFilter<LogicalExpression>(
						SetUtils.createSingleton((Syntax) Syntax.S)))
				.addBinaryParseRule(
						new CKYBinaryParsingRule<LogicalExpression>(
								new ForwardComposition<LogicalExpression>(
										categoryServices)))
				.addBinaryParseRule(
						new CKYBinaryParsingRule<LogicalExpression>(
								new BackwardComposition<LogicalExpression>(
										categoryServices)))
				.addBinaryParseRule(
						new CKYBinaryParsingRule<LogicalExpression>(
								new ForwardApplication<LogicalExpression>(
										categoryServices)))
				.addBinaryParseRule(
						new CKYBinaryParsingRule<LogicalExpression>(
								new BackwardApplication<LogicalExpression>(
										categoryServices)))
				.addBinaryParseRule(
						new CKYBinaryParsingRule<LogicalExpression>(
								new ForwardSkippingRule<LogicalExpression>(
										categoryServices)))
				.addBinaryParseRule(
						new CKYBinaryParsingRule<LogicalExpression>(
								new BackwardSkippingRule<LogicalExpression>(
										categoryServices)))
				.addUnaryParseRule(
						new CKYUnaryParsingRule<LogicalExpression>(
								new MissingCoordinationDoSequence(
										categoryServices)))
				.setWordSkippingLexicalGenerator(synSub)
				.setMaxNumberOfCellsInSpan(500).build();
		
		// Create the tester
		final Tester<Sentence, LogicalExpression> tester = new Tester.Builder<Sentence, LogicalExpression>(
				test, parser).build();
		
		// Create the supervised learner
		final AbstractUBL t = new UBLStocGradient.Builder(train,
				categoryServices, parser, splitter).setEpochs(10).setC(0.00001)
				.setAlpha0(1.0).setTester(tester).setPruneLex(false)
				.setExpandLexicon(true).build();
		
		// Do the learning
		// // comment this line out to skip training. CYY
		t.train(model); // load model & tests
		
		// Test the final model
		final ExactMatchTestingStatistics<Sentence, LogicalExpression> stats = new ExactMatchTestingStatistics<Sentence, LogicalExpression>();
		tester.test(model, stats);
		LOG.info("%s", stats);
		
		// Output total runtime
		LOG.info("Total runtime %.4f seconds",
				(System.currentTimeMillis() - startTime) / 1000.0);
	}
}
