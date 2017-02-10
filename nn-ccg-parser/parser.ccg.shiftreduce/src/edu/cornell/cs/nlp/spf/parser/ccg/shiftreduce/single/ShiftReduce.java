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
package edu.cornell.cs.nlp.spf.parser.ccg.shiftreduce.single;

import java.lang.reflect.Array;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.function.Predicate;

import com.google.common.base.Function;

import edu.cornell.cs.nlp.spf.ccg.categories.Category;
import edu.cornell.cs.nlp.spf.ccg.categories.ICategoryServices;
import edu.cornell.cs.nlp.spf.ccg.lexicon.ILexiconImmutable;
import edu.cornell.cs.nlp.spf.data.sentence.Sentence;
import edu.cornell.cs.nlp.spf.explat.IResourceRepository;
import edu.cornell.cs.nlp.spf.explat.ParameterizedExperiment;
import edu.cornell.cs.nlp.spf.explat.ParameterizedExperiment.Parameters;
import edu.cornell.cs.nlp.spf.explat.resources.IResourceObjectCreator;
import edu.cornell.cs.nlp.spf.explat.resources.usage.ResourceUsage;
import edu.cornell.cs.nlp.spf.parser.IParserOutput;
import edu.cornell.cs.nlp.spf.parser.ISentenceLexiconGenerator;
import edu.cornell.cs.nlp.spf.parser.ParsingOp;
import edu.cornell.cs.nlp.spf.parser.ccg.model.IDataItemModel;
import edu.cornell.cs.nlp.spf.parser.ccg.normalform.NormalFormValidator;
import edu.cornell.cs.nlp.spf.parser.ccg.normalform.unaryconstraint.UnaryConstraint;
import edu.cornell.cs.nlp.spf.parser.ccg.rules.BinaryRuleSet;
import edu.cornell.cs.nlp.spf.parser.ccg.rules.IBinaryParseRule;
import edu.cornell.cs.nlp.spf.parser.ccg.rules.ILexicalRule;
import edu.cornell.cs.nlp.spf.parser.ccg.rules.IUnaryParseRule;
import edu.cornell.cs.nlp.spf.parser.ccg.rules.LexicalRule;
import edu.cornell.cs.nlp.spf.parser.ccg.rules.UnaryRuleSet;
import edu.cornell.cs.nlp.spf.parser.ccg.shiftreduce.AbstractShiftReduceParser;
import edu.cornell.cs.nlp.spf.parser.ccg.shiftreduce.ShiftReduceBinaryParsingRule;
import edu.cornell.cs.nlp.spf.parser.ccg.shiftreduce.ShiftReduceUnaryParsingRule;
import edu.cornell.cs.nlp.spf.parser.ccg.shiftreduce.sloppy.AggressiveWordSkippingLexicalGenerator;
import edu.cornell.cs.nlp.spf.parser.ccg.shiftreduce.sloppy.BackwardSkippingRule;
import edu.cornell.cs.nlp.spf.parser.ccg.shiftreduce.sloppy.ForwardSkippingRule;
import edu.cornell.cs.nlp.spf.parser.ccg.shiftreduce.sloppy.SimpleWordSkippingLexicalGenerator;
import edu.cornell.cs.nlp.utils.collections.SetUtils;
import edu.cornell.cs.nlp.utils.filter.FilterUtils;
import edu.cornell.cs.nlp.utils.filter.IFilter;
import edu.cornell.cs.nlp.utils.log.ILogger;
import edu.cornell.cs.nlp.utils.log.LoggerFactory;
//import it.unimi.dsi.fastutil.Arrays;

/** Shift Reduce: Implements different forms of the shift reduce parser.
 *  The actual parser is implemented in AbstractShiftReduceParser.java
 *  
 * @author Dipendra K. Misra
 * @author Yoav Artzi
 * @author Luke Zettlemoyer
 * @param <DI>
 *            Data item.
 * @param <MR>
 *            Meaning representation.
 */
public class ShiftReduce<DI extends Sentence, MR> extends
		AbstractShiftReduceParser<DI, MR> {

	private static final long serialVersionUID = -4128451708385357708L;

	public static final ILogger	LOG					= LoggerFactory
															.create(ShiftReduce.class);
	
	private Predicate<ParsingOp<MR>> pruningFilter;
	
	private ShiftReduce(int beamSize,
			ShiftReduceBinaryParsingRule<MR>[] binaryRules,
			List<ISentenceLexiconGenerator<DI, MR>> sentenceLexiconGenerators,
			List<ISentenceLexiconGenerator<DI, MR>> sloppyLexicalGenerators,
			ICategoryServices<MR> categoryServices, boolean pruneLexicalCells,
			IFilter<Category<MR>> completeParseFilter,
			ShiftReduceUnaryParsingRule<MR>[] unaryRules,
			Function<Category<MR>, Category<MR>> categoryTransformation,
			ILexicalRule<MR> lexicalRule, boolean breakTies) {
		super(beamSize, binaryRules, sentenceLexiconGenerators, sloppyLexicalGenerators,
				categoryServices, pruneLexicalCells, completeParseFilter, unaryRules,
				categoryTransformation, lexicalRule, breakTies);
		
		this.pruningFilter = new Predicate<ParsingOp<MR>>() {
			@Override
			public boolean test(ParsingOp<MR> e) {
				return true;
			}
		};
	}

	@Override
	public IParserOutput<MR> parse(DI dataItem,
			IDataItemModel<MR> model) {
	
		return super.parse(dataItem, this.pruningFilter, model,
				false, null, null);
	}
	
	@Override
	public IParserOutput<MR> parse(DI dataItem,
			IDataItemModel<MR> model, boolean allowWordSkipping) {
	
		return super.parse(dataItem, this.pruningFilter, model,
				allowWordSkipping, null, null);
	}
	
	@Override
	public IParserOutput<MR> parse(DI dataItem, IDataItemModel<MR> model,
			boolean allowWordSkipping, ILexiconImmutable<MR> tempLexicon) {
		return super.parse(dataItem, this.pruningFilter, model,
				allowWordSkipping, tempLexicon, null);
	}

	@Override
	public IParserOutput<MR> parse(DI dataItem, IDataItemModel<MR> model,
			boolean allowWordSkipping, ILexiconImmutable<MR> tempLexicon,
			Integer beamSize) {
		return super.parse(dataItem, this.pruningFilter, model,
				allowWordSkipping, tempLexicon, beamSize);
	}

	@Override
	public IParserOutput<MR> parse(DI dataItem, Predicate<ParsingOp<MR>> filter,
			IDataItemModel<MR> model) {
		return super.parse(dataItem, filter, model, false, null, null);
	}

	@Override
	public IParserOutput<MR> parse(DI dataItem, Predicate<ParsingOp<MR>> filter,
			IDataItemModel<MR> model, boolean allowWordSkipping) {
		return super.parse(dataItem, filter, model,
				allowWordSkipping, null, null);
	}

	@Override
	public IParserOutput<MR> parse(DI dataItem, Predicate<ParsingOp<MR>> filter,
			IDataItemModel<MR> model, boolean allowWordSkipping,
			ILexiconImmutable<MR> tempLexicon) {
		return super.parse(dataItem, filter, model,
				allowWordSkipping, tempLexicon, null);
	}
		
	@Override
	public IParserOutput<MR> parse(DI dataItem, Predicate<ParsingOp<MR>> pruningFilter,
			IDataItemModel<MR> model, boolean allowWordSkipping,
			ILexiconImmutable<MR> tempLexicon, Integer beamSize) {
		return super.parse(dataItem, pruningFilter, model, allowWordSkipping, tempLexicon, beamSize);
	}

	/**
	 * Builder for {@link ShiftReduce}.
	 *
	 * @author Yoav Artzi
	 */
	public static class Builder<DI extends Sentence, MR> {

		private final Set<ShiftReduceBinaryParsingRule<MR>> 	binaryRules					= new HashSet<ShiftReduceBinaryParsingRule<MR>>();

		private boolean											breakTies					= false;

		private final ICategoryServices<MR>						categoryServices;
		private Function<Category<MR>, Category<MR>>			categoryTransformation		= null;

		private IFilter<Category<MR>>							completeParseFilter			= FilterUtils
																									.stubTrue();

		private ILexicalRule<MR>								lexicalRule					= new LexicalRule<MR>();

		/** The maximum number of cells allowed in each span */
		private int												maxStatesInBeam				= 5000;

		private boolean											pruneLexicalCells			= false;

		private final List<ISentenceLexiconGenerator<DI, MR>>	sentenceLexicalGenerators	= new ArrayList<ISentenceLexiconGenerator<DI, MR>>();

		private final List<ISentenceLexiconGenerator<DI, MR>>	sloppyLexicalGenerators		= new ArrayList<ISentenceLexiconGenerator<DI, MR>>();

		private final Set<ShiftReduceUnaryParsingRule<MR>>		unaryRules					= new HashSet<ShiftReduceUnaryParsingRule<MR>>();

		public Builder(ICategoryServices<MR> categoryServices) {
			this.categoryServices = categoryServices;
		}

		public Builder<DI, MR> addParseRule(ShiftReduceBinaryParsingRule<MR> rule) {
			binaryRules.add(rule);
			return this;
		}

		public Builder<DI, MR> addParseRule(ShiftReduceUnaryParsingRule<MR> rule) {
			unaryRules.add(rule);
			return this;
		}

		public Builder<DI, MR> addSentenceLexicalGenerator(
				ISentenceLexiconGenerator<DI, MR> generator) {
			sentenceLexicalGenerators.add(generator);
			return this;
		}

		public Builder<DI, MR> addSloppyLexicalGenerator(
				ISentenceLexiconGenerator<DI, MR> sloppyGenerator) {
			sloppyLexicalGenerators.add(sloppyGenerator);
			return this;
		}

		@SuppressWarnings("unchecked")
		public ShiftReduce<DI, MR> build() {
			return new ShiftReduce<DI, MR>(maxStatesInBeam,
					binaryRules.toArray((ShiftReduceBinaryParsingRule<MR>[]) Array
							.newInstance(ShiftReduceBinaryParsingRule.class,
									binaryRules.size())),
					sentenceLexicalGenerators, sloppyLexicalGenerators,
					categoryServices, pruneLexicalCells, completeParseFilter,
					unaryRules.toArray((ShiftReduceUnaryParsingRule<MR>[]) Array
							.newInstance(ShiftReduceUnaryParsingRule.class,
									unaryRules.size())),
					categoryTransformation, lexicalRule, breakTies);
		}

		public Builder<DI, MR> setBreakTies(boolean breakTies) {
			this.breakTies = breakTies;
			return this;
		}

		public Builder<DI, MR> setCategoryTransformation(
				Function<Category<MR>, Category<MR>> categoryTransformation) {
			this.categoryTransformation = categoryTransformation;
			return this;
		}

		public Builder<DI, MR> setCompleteParseFilter(
				IFilter<Category<MR>> completeParseFilter) {
			this.completeParseFilter = completeParseFilter;
			return this;
		}

		public void setLexicalRule(ILexicalRule<MR> lexicalRule) {
			this.lexicalRule = lexicalRule;
		}

		public Builder<DI, MR> setMaxStatesInBeam(
				int maxNumberOfCellsInSpan) {
			this.maxStatesInBeam = maxNumberOfCellsInSpan;
			return this;
		}

		public Builder<DI, MR> setPruneLexicalCells(boolean pruneLexicalCells) {
			this.pruneLexicalCells = pruneLexicalCells;
			return this;
		}
	}

	public static class Creator<DI extends Sentence, MR> implements
			IResourceObjectCreator<ShiftReduce<DI, MR>> {

		private String	type;

		public Creator() {
			this("parser.shiftreduce");
		}

		public Creator(String type) {
			this.type = type;
		}

		@SuppressWarnings("unchecked")
		@Override
		public ShiftReduce<DI, MR> create(Parameters params,
				IResourceRepository repo) {
			final Builder<DI, MR> builder = new Builder<DI, MR>(
					(ICategoryServices<MR>) repo
							.get(ParameterizedExperiment.CATEGORY_SERVICES_RESOURCE));

			if (params.contains("breakTies")) {
				builder.setBreakTies(params.getAsBoolean("breakTies"));
			}

			if (params.contains("parseFilter")) {
				builder.setCompleteParseFilter((IFilter<Category<MR>>) repo
						.get(params.get("parseFilter")));
			}

			if (params.contains("beam")) {
				builder.setMaxStatesInBeam(params.getAsInteger("beam"));
			}

			if (params.contains("lex")) {
				builder.setLexicalRule((ILexicalRule<MR>) repo.get(params
						.get("lex")));
			}

			if (params.contains("pruneLexicalCells")) {
				builder.setPruneLexicalCells(params
						.getAsBoolean("pruneLexicalCells"));
			}

			for (final String id : params.getSplit("generators")) {
				builder.addSentenceLexicalGenerator((ISentenceLexiconGenerator<DI, MR>) repo
						.get(id));
			}

			for (final String id : params.getSplit("sloppyGenerators")) {
				builder.addSloppyLexicalGenerator((ISentenceLexiconGenerator<DI, MR>) repo
						.get(id));
			}

			NormalFormValidator nfValidator;
			if (params.contains("nfValidator")) {
				nfValidator = repo.get(params.get("nfValidator"));
			} else {
				nfValidator = null;
			}

			final String wordSkippingType = params.get("wordSkipping", "none");
			if (wordSkippingType.equals("simple")) {
				// Skipping lexical generator.
				builder.addSloppyLexicalGenerator(new SimpleWordSkippingLexicalGenerator<DI, MR>(
						(ICategoryServices<MR>) repo
								.get(ParameterizedExperiment.CATEGORY_SERVICES_RESOURCE)));

				// Skipping rules.
				final ForwardSkippingRule<MR> forwardSkip = new ForwardSkippingRule<MR>(
						(ICategoryServices<MR>) repo
								.get(ParameterizedExperiment.CATEGORY_SERVICES_RESOURCE));
				final BackwardSkippingRule<MR> backSkip = new BackwardSkippingRule<MR>(
						(ICategoryServices<MR>) repo
								.get(ParameterizedExperiment.CATEGORY_SERVICES_RESOURCE),
						false);

				// Add a normal form constraint to disallow unary steps after
				// skipping.
				final NormalFormValidator.Builder nfBuilder = new NormalFormValidator.Builder();
				if (nfValidator != null) {
					nfBuilder.addConstraints(nfValidator);
				}
				nfBuilder.addConstraint(new UnaryConstraint(SetUtils.createSet(
						forwardSkip.getName(), backSkip.getName())));
				nfValidator = nfBuilder.build();

				// Add the rules.
				addRule(builder, backSkip, nfValidator);
				addRule(builder, forwardSkip, nfValidator);
			} else if (wordSkippingType.equals("aggressive")) {
				// Skipping lexical generator.
				builder.addSloppyLexicalGenerator(new AggressiveWordSkippingLexicalGenerator<DI, MR>(
						(ICategoryServices<MR>) repo
								.get(ParameterizedExperiment.CATEGORY_SERVICES_RESOURCE)));
				// Skipping rules.
				final ForwardSkippingRule<MR> forwardSkip = new ForwardSkippingRule<MR>(
						(ICategoryServices<MR>) repo
								.get(ParameterizedExperiment.CATEGORY_SERVICES_RESOURCE));
				final BackwardSkippingRule<MR> backSkip = new BackwardSkippingRule<MR>(
						(ICategoryServices<MR>) repo
								.get(ParameterizedExperiment.CATEGORY_SERVICES_RESOURCE),
						true);

				// Add a normal form constraint to disallow unary steps after
				// skipping.
				final NormalFormValidator.Builder nfBuilder = new NormalFormValidator.Builder();
				if (nfValidator != null) {
					nfBuilder.addConstraints(nfValidator);
				}
				nfBuilder.addConstraint(new UnaryConstraint(SetUtils.createSet(
						forwardSkip.getName(), backSkip.getName())));
				nfValidator = nfBuilder.build();

				// Add the rules.
				addRule(builder, backSkip, nfValidator);
				addRule(builder, forwardSkip, nfValidator);
			}

			if (params.contains("transformation")) {
				builder.setCategoryTransformation((Function<Category<MR>, Category<MR>>) repo
						.get(params.get("transformation")));
			}

			for (final String id : params.getSplit("rules")) {
				final Object rule = repo.get(id);
				if (rule instanceof BinaryRuleSet) {
					for (final IBinaryParseRule<MR> singleRule : (BinaryRuleSet<MR>) rule) {
						addRule(builder, singleRule, nfValidator);
					}
				} else if (rule instanceof UnaryRuleSet) {
					for (final IUnaryParseRule<MR> singleRule : (UnaryRuleSet<MR>) rule) {
						addRule(builder, singleRule, nfValidator);
					}
				} else {
					addRule(builder, rule, nfValidator);
				}
			}

			return builder.build();
		}

		@Override
		public String type() {
			return type;
		}

		@Override
		public ResourceUsage usage() {
			return ResourceUsage
					.builder(type, ShiftReduce.class)
					.addParam(
							"breakTies",
							Boolean.class,
							"Breaks ties during pruning using the order of insertion to the queue. In a single-threaded parser, this is essentially deterministic (default: false)")
					.addParam("parseFilter", IFilter.class,
							"Filter to determine complete parses.")
					.addParam("beam", Integer.class,
							"Beam to use for cell pruning (default: 50).")
					.addParam("lex", ILexicalRule.class,
							"Lexical rule (default: simple generic rule)")
					.addParam("pruneLexicalCells", Boolean.class,
							"Prune lexical entries similarly to conventional categories (default: false)")
					.addParam(
							"wordSkipping",
							String.class,
							"Type of word skpping to use during sloppy inference: none, simple or aggressive (default: none)")
					.addParam("generators", ISentenceLexiconGenerator.class,
							"List of dynamic sentence lexical generators.")
					.addParam("sloppyGenerators",
							ISentenceLexiconGenerator.class,
							"List of dynamic sentence lexical generators for sloppy inference.")
					.addParam(
							"transformation",
							Function.class,
							"Transformation to be applied to each category before it's added to the chart (default: none).")
					.addParam("rules", IBinaryParseRule.class,
							"Binary parsing rules.").build();
		}

		@SuppressWarnings("unchecked")
		private void addRule(Builder<DI, MR> builder, Object rule,
				NormalFormValidator nfValidator) {
			if (rule instanceof IBinaryParseRule) {
				builder.addParseRule(new ShiftReduceBinaryParsingRule<MR>(
						(IBinaryParseRule<MR>) rule, nfValidator));
			} else if (rule instanceof IUnaryParseRule) {
				builder.addParseRule(new ShiftReduceUnaryParsingRule<MR>(
						(IUnaryParseRule<MR>) rule, nfValidator));
			} else if (rule instanceof ShiftReduceBinaryParsingRule) {
				builder.addParseRule((ShiftReduceBinaryParsingRule<MR>) rule);
			} else if (rule instanceof ShiftReduceUnaryParsingRule) {
				builder.addParseRule((ShiftReduceUnaryParsingRule<MR>) rule);
			} else {
				throw new IllegalArgumentException("Invalid rule class: "
						+ rule);
			}
		}

	}
}
