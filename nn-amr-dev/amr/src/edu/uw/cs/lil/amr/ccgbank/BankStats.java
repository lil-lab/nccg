package edu.uw.cs.lil.amr.ccgbank;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.io.Serializable;

import edu.cornell.cs.nlp.spf.ccg.categories.syntax.Syntax;
import edu.cornell.cs.nlp.spf.explat.IResourceRepository;
import edu.cornell.cs.nlp.spf.explat.ParameterizedExperiment.Parameters;
import edu.cornell.cs.nlp.spf.explat.resources.IResourceObjectCreator;
import edu.cornell.cs.nlp.spf.explat.resources.usage.ResourceUsage;
import edu.cornell.cs.nlp.utils.log.ILogger;
import edu.cornell.cs.nlp.utils.log.LoggerFactory;
import it.unimi.dsi.fastutil.objects.Object2IntOpenHashMap;
import uk.ac.ed.easyccg.syntax.Category;

/**
 * CCG treebank statistics.
 *
 * @author Yoav Artzi
 */
@Deprecated
public class BankStats implements Serializable {

	public static final ILogger					LOG					= LoggerFactory
			.create(BankStats.class);
	private static final long					serialVersionUID	= 1248090107735452047L;
	private final Object2IntOpenHashMap<Syntax>	counts;

	private BankStats(Object2IntOpenHashMap<Syntax> counts) {
		this.counts = counts;
		LOG.info("Init %s :: #syntax=%d", getClass(), counts.size());
	}

	/**
	 * Read {@link BankStats} from a TSV file.
	 */
	public static BankStats read(File file) {
		try (BufferedReader reader = new BufferedReader(new FileReader(file))) {
			String line = null;
			final Object2IntOpenHashMap<Syntax> counts = new Object2IntOpenHashMap<>();
			counts.defaultReturnValue(0);
			while ((line = reader.readLine()) != null) {

				final String[] split = line.split("\t");
				final int count = Integer.valueOf(split[0]);
				final Syntax root = CcgBankServices
						.toSyntax(Category.valueOf(split[1]));
						// final Syntax[] children = new Syntax[split.length -
						// 2];
						// for (int i = 2; i < split.length; ++i) {
						// children[i - 2] = CcgBankServices.toSyntax(Category
						// .valueOf(split[i]));
						// }

				// Rewrite the categories to all possible Syntax forms. TODO
				if (root != null) {
					for (final Syntax rootRewrite : CcgBankServices
							.rewrite(root, true)) {
						counts.addTo(rootRewrite.stripAttributes(), count);
					}
				}

			}
			return new BankStats(counts);
		} catch (final IOException e) {
			throw new RuntimeException(e);
		}
	}

	public int getCount(Syntax syntax) {
		return counts.getInt(syntax);
	}

	public static class Creator implements IResourceObjectCreator<BankStats> {

		private final String type;

		public Creator() {
			this("ccgbank.stats");
		}

		public Creator(String type) {
			this.type = type;
		}

		@Override
		public BankStats create(Parameters params, IResourceRepository repo) {
			return BankStats.read(params.getAsFile("file"));
		}

		@Override
		public String type() {
			return type;
		}

		@Override
		public ResourceUsage usage() {
			return ResourceUsage.builder(type, BankStats.class)
					.setDescription("CCG Bank statistics")
					.addParam("file", File.class,
							"TSV file with parsing operation counts")
					.build();
		}

	}

}
