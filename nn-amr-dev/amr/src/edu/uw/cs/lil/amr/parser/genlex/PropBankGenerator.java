package edu.uw.cs.lil.amr.parser.genlex;

import java.util.Collection;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;

import edu.cornell.cs.nlp.spf.base.token.TokenSeq;
import edu.cornell.cs.nlp.spf.ccg.categories.Category;
import edu.cornell.cs.nlp.spf.ccg.lexicon.ILexicon;
import edu.cornell.cs.nlp.spf.ccg.lexicon.ILexiconImmutable;
import edu.cornell.cs.nlp.spf.ccg.lexicon.LexicalEntry;
import edu.cornell.cs.nlp.spf.ccg.lexicon.factored.lambda.FactoredLexicalEntry;
import edu.cornell.cs.nlp.spf.ccg.lexicon.factored.lambda.FactoringServices;
import edu.cornell.cs.nlp.spf.ccg.lexicon.factored.lambda.Lexeme;
import edu.cornell.cs.nlp.spf.ccg.lexicon.factored.lambda.LexicalTemplate;
import edu.cornell.cs.nlp.spf.data.situated.sentence.SituatedSentence;
import edu.cornell.cs.nlp.spf.explat.IResourceRepository;
import edu.cornell.cs.nlp.spf.explat.ParameterizedExperiment.Parameters;
import edu.cornell.cs.nlp.spf.explat.resources.IResourceObjectCreator;
import edu.cornell.cs.nlp.spf.explat.resources.usage.ResourceUsage;
import edu.cornell.cs.nlp.spf.genlex.ccg.template.GenerationRepository;
import edu.cornell.cs.nlp.spf.genlex.ccg.template.GenerationRepositoryWithConstants;
import edu.cornell.cs.nlp.spf.mr.lambda.LogicalConstant;
import edu.cornell.cs.nlp.spf.mr.lambda.LogicalExpression;
import edu.cornell.cs.nlp.spf.parser.ISentenceLexiconGenerator;
import edu.cornell.cs.nlp.utils.collections.MapUtils;
import edu.cornell.cs.nlp.utils.log.ILogger;
import edu.cornell.cs.nlp.utils.log.LoggerFactory;
import edu.uw.cs.lil.amr.data.AMRMeta;
import edu.uw.cs.lil.amr.lambda.AMRServices;
import edu.uw.cs.lil.amr.util.wordnet.WordNetServices;

/**
 * PropBank dynamic generator for {@link LexicalEntry}s. Given a word, tries to
 * retrieve the relevant predicates from PropBank and generate entries on the
 * fly.
 *
 * @author Kenton Lee
 * @author Yoav Artzi
 */
@Deprecated
public class PropBankGenerator implements
		ISentenceLexiconGenerator<SituatedSentence<AMRMeta>, LogicalExpression> {

	public static final ILogger	LOG					= LoggerFactory
			.create(PropBankGenerator.class);
	private static final long	serialVersionUID	= 4115646139027926146L;
	/**
	 * Conservative generation. If 'true', generate entries only for the first
	 * "-01" PropBank frame. If 'false', generate entries for all retrieved
	 * predicates.
	 */
	private final boolean		conservative;

	private final GenerationRepository	genRepo;
	private final String				origin;

	public PropBankGenerator(boolean conservative, GenerationRepository genRepo,
			String origin) {
		this.conservative = conservative;
		this.genRepo = genRepo;
		this.origin = origin;
	}

	@Override
	public Set<LexicalEntry<LogicalExpression>> generateLexicon(
			SituatedSentence<AMRMeta> sample) {

		// Generate entries for each word. Only process single tokens.
		final Set<LexicalEntry<LogicalExpression>> entries = new HashSet<>();
		final TokenSeq tokens = sample.getTokens();
		final int len = tokens.size();
		for (int i = 0; i < len; ++i) {
			final String posTag = sample.getState().getTags().get(i);
			final Set<LogicalConstant> predicates = retrievePropBankPredicates(
					tokens.get(i), posTag);
			final GenerationRepositoryWithConstants repoWithConsts = genRepo
					.setConstants(predicates);
			for (final Lexeme lexeme : repoWithConsts.generate(
					tokens.sub(i, i + 1), 1, MapUtils.createSingletonMap(
							LexicalEntry.ORIGIN_PROPERTY, origin))) {
				for (final LexicalTemplate template : genRepo.getTemplates()) {
					final Category<LogicalExpression> category = template
							.apply(lexeme);
					if (category != null) {
						final Map<String, String> entryProperties = MapUtils
								.createSingletonMap(
										LexicalEntry.ORIGIN_PROPERTY,
										origin + "-" + posTag + "-"
												+ category.getSyntax());
						final LexicalEntry<LogicalExpression> entry = new LexicalEntry<>(
								lexeme.getTokens(), category, true,
								entryProperties);
						LOG.debug("Generated: %s", entry);
						entries.add(entry);
					}
				}
			}
		}

		return entries;
	}

	/**
	 * Given a token, returns potential PropBank frames.
	 */
	private Set<LogicalConstant> retrievePropBankPredicates(String token,
			String posTag) {
		return AMRServices
				.getPropBankFrames(WordNetServices.getLemma(token, posTag))
				.stream().filter(frame -> !conservative || frame.getId() == 1)
				.map(frame -> LogicalConstant.create(frame.getConstantText(),
						AMRServices.getTypingPredicateType(), true, true))
				.collect(Collectors.toSet());
	}

	public static class Creator
			implements IResourceObjectCreator<PropBankGenerator> {

		private final String type;

		public Creator() {
			this("dyngen.amr.propbank");
		}

		public Creator(String type) {
			this.type = type;
		}

		@Override
		public PropBankGenerator create(Parameters params,
				IResourceRepository repo) {

			// Create the generation repository from a seed lexicon.
			final ILexiconImmutable<LogicalExpression> lexicon = repo
					.get(params.get("lexicon"));
			final Collection<LexicalEntry<LogicalExpression>> lexicalEntries = lexicon
					.toCollection();
			final Set<LexicalTemplate> templates = new HashSet<>();
			final Set<String> attributes = new HashSet<>();
			for (final LexicalEntry<LogicalExpression> entry : lexicalEntries) {
				final FactoredLexicalEntry factored = FactoringServices
						.factor(entry);
				// Only take templates that require one constant.
				if (factored.getLexeme().getConstants().size() == 1) {
					templates.add(factored.getTemplate());
					for (final String attribute : factored.getLexeme()
							.getAttributes()) {
						attributes.add(attribute);
					}
				}
			}
			final GenerationRepository genRepo = new GenerationRepository(
					templates, attributes);

			return new PropBankGenerator(
					params.getAsBoolean("conservative", false), genRepo,
					params.get("origin", "dyn-propbank"));
		}

		@Override
		public String type() {
			return type;
		}

		@Override
		public ResourceUsage usage() {
			return ResourceUsage.builder(type, PropBankGenerator.class)
					.setDescription(
							"PropBank dynamic generator for lexical entries. Given a word, tries to retrieve the relevant predicates from PropBank and generate entries on the fly.")
					.addParam("conservative", Boolean.class,
							"Conservative generation. If 'true', generate entries only for the first \"-01\" PropBank frame. If 'false', generate entries for all retrieved predicates (default: false)")
					.addParam("lexicon", ILexicon.class,
							"Source lexicon for template and attributes")
					.addParam("origin", String.class,
							"Origin label (default: dyn-propbank)")
					.build();
		}

	}

}
