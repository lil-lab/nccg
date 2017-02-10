/*******************************************************************************
 * Copyright (C) 2011 - 2015 Yoav Artzi, All rights reserved.
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
 *******************************************************************************/
package edu.cornell.cs.nlp.spf.geoquery;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import edu.cornell.cs.nlp.spf.base.hashvector.HashVectorFactory;
import edu.cornell.cs.nlp.spf.base.hashvector.HashVectorFactory.Type;
import edu.cornell.cs.nlp.spf.base.hashvector.KeyArgs;
import edu.cornell.cs.nlp.spf.ccg.categories.syntax.Syntax;
import edu.cornell.cs.nlp.spf.ccg.lexicon.ILexicon;
import edu.cornell.cs.nlp.spf.ccg.lexicon.LexicalEntry;
import edu.cornell.cs.nlp.spf.ccg.lexicon.LexicalEntry.Origin;
import edu.cornell.cs.nlp.spf.ccg.lexicon.Lexicon;
import edu.cornell.cs.nlp.spf.ccg.lexicon.factored.lambda.FactoredLexicalEntry;
import edu.cornell.cs.nlp.spf.ccg.lexicon.factored.lambda.FactoredLexicon;
import edu.cornell.cs.nlp.spf.ccg.lexicon.factored.lambda.FactoringServices;
import edu.cornell.cs.nlp.spf.data.collection.CompositeDataCollection;
import edu.cornell.cs.nlp.spf.data.collection.IDataCollection;
import edu.cornell.cs.nlp.spf.data.sentence.Sentence;
import edu.cornell.cs.nlp.spf.data.sentence.SentenceLengthFilter;
import edu.cornell.cs.nlp.spf.data.singlesentence.SingleSentence;
import edu.cornell.cs.nlp.spf.data.singlesentence.SingleSentenceCollection;
import edu.cornell.cs.nlp.spf.data.utils.LabeledValidator;
import edu.cornell.cs.nlp.spf.genlex.ccg.ILexiconGenerator;
import edu.cornell.cs.nlp.spf.genlex.ccg.template.TemplateSupervisedGenlex;
import edu.cornell.cs.nlp.spf.learn.ILearner;
import edu.cornell.cs.nlp.spf.learn.validation.perceptron.ValidationPerceptron;
import edu.cornell.cs.nlp.spf.mr.lambda.FlexibleTypeComparator;
import edu.cornell.cs.nlp.spf.mr.lambda.LogicLanguageServices;
import edu.cornell.cs.nlp.spf.mr.lambda.LogicalConstant;
import edu.cornell.cs.nlp.spf.mr.lambda.LogicalExpression;
import edu.cornell.cs.nlp.spf.mr.lambda.ccg.LogicalExpressionCategoryServices;
import edu.cornell.cs.nlp.spf.mr.lambda.ccg.SimpleFullParseFilter;
import edu.cornell.cs.nlp.spf.mr.language.type.TypeRepository;
import edu.cornell.cs.nlp.spf.parser.IParser;
import edu.cornell.cs.nlp.spf.parser.ccg.factoredlex.features.FactoredLexicalFeatureSet;
import edu.cornell.cs.nlp.spf.parser.ccg.features.basic.DynamicWordSkippingFeatures;
import edu.cornell.cs.nlp.spf.parser.ccg.features.basic.LexicalFeaturesInit;
import edu.cornell.cs.nlp.spf.parser.ccg.features.basic.scorer.ExpLengthLexicalEntryScorer;
import edu.cornell.cs.nlp.spf.parser.ccg.features.lambda.LogicalExpressionCoordinationFeatureSet;
import edu.cornell.cs.nlp.spf.parser.ccg.lambda.pruning.SupervisedFilterFactory;
import edu.cornell.cs.nlp.spf.parser.ccg.model.LexiconModelInit;
import edu.cornell.cs.nlp.spf.parser.ccg.model.Model;
import edu.cornell.cs.nlp.spf.parser.ccg.model.ModelLogger;
import edu.cornell.cs.nlp.spf.parser.ccg.normalform.NormalFormValidator;
import edu.cornell.cs.nlp.spf.parser.ccg.rules.lambda.PluralExistentialTypeShifting;
import edu.cornell.cs.nlp.spf.parser.ccg.rules.lambda.ThatlessRelative;
import edu.cornell.cs.nlp.spf.parser.ccg.rules.lambda.typeraising.ForwardTypeRaisedComposition;
import edu.cornell.cs.nlp.spf.parser.ccg.rules.lambda.typeshifting.PrepositionTypeShifting;
import edu.cornell.cs.nlp.spf.parser.ccg.rules.primitivebinary.application.BackwardApplication;
import edu.cornell.cs.nlp.spf.parser.ccg.rules.primitivebinary.application.ForwardApplication;
import edu.cornell.cs.nlp.spf.parser.ccg.rules.primitivebinary.composition.BackwardComposition;
import edu.cornell.cs.nlp.spf.parser.ccg.rules.primitivebinary.composition.ForwardComposition;
import edu.cornell.cs.nlp.spf.parser.ccg.shiftreduce.ShiftReduceBinaryParsingRule;
import edu.cornell.cs.nlp.spf.parser.ccg.shiftreduce.ShiftReduceUnaryParsingRule;
import edu.cornell.cs.nlp.spf.parser.ccg.shiftreduce.single.ShiftReduce;
import edu.cornell.cs.nlp.spf.parser.ccg.shiftreduce.sloppy.BackwardSkippingRule;
import edu.cornell.cs.nlp.spf.parser.ccg.shiftreduce.sloppy.ForwardSkippingRule;
import edu.cornell.cs.nlp.spf.parser.ccg.shiftreduce.sloppy.SimpleWordSkippingLexicalGenerator;
import edu.cornell.cs.nlp.spf.test.Tester;
import edu.cornell.cs.nlp.spf.test.stats.ExactMatchTestingStatistics;
import edu.cornell.cs.nlp.utils.collections.SetUtils;
import edu.cornell.cs.nlp.utils.function.PredicateUtils;
import edu.cornell.cs.nlp.utils.log.ILogger;
import edu.cornell.cs.nlp.utils.log.Log;
import edu.cornell.cs.nlp.utils.log.LogLevel;
import edu.cornell.cs.nlp.utils.log.Logger;
import edu.cornell.cs.nlp.utils.log.LoggerFactory;

/**
 * Cross validation experiment for GeoQuery using fold0 for testing. This class
 * is intended to illustrate how an experiment is structured. For complete
 * experiments see the accompanying ExPlat files.
 *
 * @author Yoav Artzi
 * @author Dipendra K. Misra
 */

public class GeoExpShiftReduce {
	public static final ILogger	LOG	= LoggerFactory.create(GeoExpShiftReduce.class);

	private GeoExpShiftReduce() {
		// Private ctor. Service class.
	}

	public static void main(String[] args) {

		// //////////////////////////////////////////
		// Init logging
		// //////////////////////////////////////////
		
		Logger.DEFAULT_LOG = new Log(System.err);
		Logger.setSkipPrefix(true);
		LogLevel.setLogLevel(LogLevel.INFO);
		
		// //////////////////////////////////////////
		// Set some locations to use later
		// //////////////////////////////////////////

		final File resourceDir = new File("geoquery/resources/");
		final File dataDir = new File("geoquery/experiments/data");

		// //////////////////////////////////////////
		// Use tree hash vector
		// //////////////////////////////////////////

		HashVectorFactory.DEFAULT = Type.FAST_TREE;

		// //////////////////////////////////////////
		// Init lambda calculus system.
		// //////////////////////////////////////////

		final File typesFile = new File(resourceDir, "geo.types");
		final File predOntology = new File(resourceDir, "geo.preds.ont");
		final File simpleOntology = new File(resourceDir, "geo.consts.ont");

		try {
			// Init the logical expression type system
			LogicLanguageServices.setInstance(new LogicLanguageServices.Builder(
					new TypeRepository(typesFile), new FlexibleTypeComparator())
							.addConstantsToOntology(simpleOntology)
							.addConstantsToOntology(predOntology)
							.setUseOntology(true).setNumeralTypeName("i")
							.closeOntology(true).build());
		} catch (final IOException e) {
			throw new RuntimeException(e);
		}
		
		// //////////////////////////////////////////////////
		// Category services for logical expressions
		// //////////////////////////////////////////////////

		final LogicalExpressionCategoryServices categoryServices = new LogicalExpressionCategoryServices(
				true);

		// //////////////////////////////////////////////////
		// Lexical factoring services
		// //////////////////////////////////////////////////

		FactoringServices.set(new FactoringServices.Builder()
				.addConstant(LogicalConstant.read("exists:<<e,t>,t>"))
				.addConstant(LogicalConstant.read("the:<<e,t>,e>")).build());

		// //////////////////////////////////////////////////
		// Read initial lexicon
		// //////////////////////////////////////////////////

		// Create a static set of lexical entries, which are factored using
		// non-maximal factoring (each lexical entry is factored to multiple
		// entries). This static set is used to init the model with various
		// templates and lexemes.

		final File seedLexiconFile = new File(resourceDir, "seed.lex");
		final File npLexiconFile = new File(resourceDir, "np-list.lex");

		final Lexicon<LogicalExpression> readLexicon = new Lexicon<LogicalExpression>();
		readLexicon.addEntriesFromFile(seedLexiconFile, categoryServices,
				Origin.FIXED_DOMAIN);

		final Lexicon<LogicalExpression> semiFactored = new Lexicon<LogicalExpression>();
		for (final LexicalEntry<LogicalExpression> entry : readLexicon
				.toCollection()) {
			for (final FactoredLexicalEntry factoredEntry : FactoringServices
					.factor(entry, true, true, 2)) {
				semiFactored.add(FactoringServices.factor(factoredEntry));
			}
		}

		// Read NP list
		final ILexicon<LogicalExpression> npLexicon = new FactoredLexicon();
		npLexicon.addEntriesFromFile(npLexiconFile, categoryServices,
				Origin.FIXED_DOMAIN);

		// //////////////////////////////////////////////////
		// Shift Reduce parser
		// //////////////////////////////////////////////////

		// Use the Hockenmeir-Bisk normal form parsing constaints. To parse without
		// NF constraints, just set this variable to null.
		final NormalFormValidator nf = null; /*new NormalFormValidator.Builder()
				.addConstraint(
						new HBComposedConstraint(Collections.emptySet(), false))
				.build();*/

		final IParser<Sentence, LogicalExpression> parser = new ShiftReduce.Builder<Sentence, LogicalExpression>(
				categoryServices)
				.setCompleteParseFilter(
						new SimpleFullParseFilter(SetUtils
								.createSingleton((Syntax) Syntax.S)))
				.setPruneLexicalCells(true)
				.addSloppyLexicalGenerator(
						new SimpleWordSkippingLexicalGenerator<Sentence, LogicalExpression>(
								categoryServices))
				.setMaxStatesInBeam(500)
				.addParseRule(
						new ShiftReduceBinaryParsingRule<LogicalExpression>(
								new ForwardComposition<LogicalExpression>(
										categoryServices, 1, false), 
								nf))
				.addParseRule(
						new ShiftReduceBinaryParsingRule<LogicalExpression>(
								new BackwardComposition<LogicalExpression>(
										categoryServices, 1, false),
								nf))
				.addParseRule(
						new ShiftReduceBinaryParsingRule<LogicalExpression>(
								new ForwardApplication<LogicalExpression>(
										categoryServices),
								nf))
				.addParseRule(
						new ShiftReduceBinaryParsingRule<LogicalExpression>(
								new BackwardApplication<LogicalExpression>(
										categoryServices),
								nf))
				.addParseRule(
						new ShiftReduceUnaryParsingRule<LogicalExpression>(
								new PrepositionTypeShifting(
										categoryServices),
								nf))
				.addParseRule(
						new ForwardSkippingRule<LogicalExpression>(
								categoryServices))
				.addParseRule(
						new BackwardSkippingRule<LogicalExpression>(
								categoryServices, false))
				.addParseRule(
						new ShiftReduceBinaryParsingRule<LogicalExpression>(
								new ForwardTypeRaisedComposition(
										categoryServices),
								nf))
				.addParseRule(
						new ShiftReduceBinaryParsingRule<LogicalExpression>(
								new ThatlessRelative(categoryServices),
								nf))
				.addParseRule(
						new ShiftReduceBinaryParsingRule<LogicalExpression>(
								new PluralExistentialTypeShifting(
										categoryServices),
								nf))
				.build();
		
		// //////////////////////////////////////////////////
		// Model
		// //////////////////////////////////////////////////

		final Model<Sentence, LogicalExpression> model = new Model.Builder<Sentence, LogicalExpression>()
				.setLexicon(new FactoredLexicon())
				.addFeatureSet(new FactoredLexicalFeatureSet.Builder<Sentence>()
						.setTemplateScale(0.1).build())
				.addFeatureSet(new DynamicWordSkippingFeatures<>(
						categoryServices.getEmptyCategory()))
				.addFeatureSet(
						new LogicalExpressionCoordinationFeatureSet<Sentence>(
								true, true, true))
				.build();
		
		// Model logger
		final ModelLogger modelLogger = new ModelLogger(true);

		// //////////////////////////////////////////////////
		// Validation function
		// //////////////////////////////////////////////////

		final LabeledValidator<SingleSentence, LogicalExpression> validator = new LabeledValidator<SingleSentence, LogicalExpression>();

		// //////////////////////////////////////////////////
		// Genlex function
		// //////////////////////////////////////////////////

		final TemplateSupervisedGenlex<Sentence, SingleSentence> genlex = new TemplateSupervisedGenlex<Sentence, SingleSentence>(
				4, false, ILexiconGenerator.GENLEX_LEXICAL_ORIGIN);/*new TemplateSupervisedGenlex.Builder<Sentence, SingleSentence>(
				4, "genlex", false).addFromLexicon(semiFactored).build();*/

		// //////////////////////////////////////////////////
		// Load training and testing data
		// //////////////////////////////////////////////////

		final List<IDataCollection<? extends SingleSentence>> folds = new ArrayList<IDataCollection<? extends SingleSentence>>(
				10);
		for (int i = 0; i < 10; ++i) {
			folds.add(SingleSentenceCollection
					.read(new File(dataDir, String.format("fold%d.ccg", i))));
		}
		final CompositeDataCollection<SingleSentence> train = new CompositeDataCollection<SingleSentence>(
				folds.subList(1, folds.size()));
		final IDataCollection<? extends SingleSentence> test = folds.get(0);

		// //////////////////////////////////////////////////
		// Tester
		// //////////////////////////////////////////////////

		final Tester<Sentence, LogicalExpression, SingleSentence> tester = new Tester.Builder<Sentence, LogicalExpression, SingleSentence>(
				test, parser).build();

		// //////////////////////////////////////////////////
		// Learner
		// //////////////////////////////////////////////////
		
		final ILearner<Sentence, SingleSentence, Model<Sentence, LogicalExpression>> learner = new ValidationPerceptron.Builder<Sentence, SingleSentence, LogicalExpression>(
				train, parser, validator)
				.setGenlex(genlex, categoryServices)
				.setLexiconGenerationBeamSize(1000)
				.setNumTrainingIterations(1)
				.setParsingFilterFactory(
						new SupervisedFilterFactory<>(PredicateUtils.alwaysTrue())
						/*new SupervisedFilterFactory<SingleSentence>(FilterUtils
								.<LogicalConstant> stubTrue())*/)
				.setProcessingFilter(
						new SentenceLengthFilter<SingleSentence>(50))
				.setErrorDriven(true)
				.setConflateGenlexAndPrunedParses(false)
				.build();

		// //////////////////////////////////////////////////
		// Init model
		// //////////////////////////////////////////////////

		new LexiconModelInit<Sentence, LogicalExpression>(semiFactored)
				.init(model);
		new LexiconModelInit<Sentence, LogicalExpression>(npLexicon)
				.init(model);
		new LexicalFeaturesInit<Sentence, LogicalExpression>(semiFactored,
				KeyArgs.read("FACLEX#LEX"),
				new ExpLengthLexicalEntryScorer<LogicalExpression>(10.0, 1.1))
						.init(model);
		new LexicalFeaturesInit<Sentence, LogicalExpression>(npLexicon,
				KeyArgs.read("FACLEX#LEX"),
				new ExpLengthLexicalEntryScorer<LogicalExpression>(10.0, 1.1))
						.init(model);
		new LexicalFeaturesInit<Sentence, LogicalExpression>(semiFactored,
				KeyArgs.read("FACLEX#XEME"), 10.0).init(model);
		new LexicalFeaturesInit<Sentence, LogicalExpression>(npLexicon,
				KeyArgs.read("FACLEX#XEME"), 10.0).init(model);
		
		// Init the weight for the dynamic word skipping feature.
		model.getTheta().set("DYNSKIP", -1.0);
		
		// //////////////////////////////////////////////////
		// Log initial model
		// //////////////////////////////////////////////////

		LOG.info("Initial model:");
		modelLogger.log(model, System.err);

		// //////////////////////////////////////////////////
		// Training
		// //////////////////////////////////////////////////

		long startTime = System.currentTimeMillis();
		
		learner.train(model);

		// Output total run time
		LOG.info("Total training time %.4f seconds",
				(System.currentTimeMillis() - startTime) / 1000.0);
		
		// //////////////////////////////////////////////////
		// Log final model
		// //////////////////////////////////////////////////

		LOG.info("Final model:");
		modelLogger.log(model, System.err);
		
		// //////////////////////////////////////////////////
		// Testing
		// //////////////////////////////////////////////////

		startTime = System.currentTimeMillis();
		
		final ExactMatchTestingStatistics<Sentence, LogicalExpression, SingleSentence> stats = new
			  ExactMatchTestingStatistics<Sentence, LogicalExpression, SingleSentence>();
		
		tester.test(model, stats);
		LOG.info(stats.toString());

		LOG.info("Total testing time %.4f seconds",
				(System.currentTimeMillis() - startTime) / 1000.0);

		//${workspace_loc:geoquery}/..

	}
}
