package edu.uw.cs.lil.amr.util.wordnet;

import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import java.util.stream.StreamSupport;

import org.yawni.wordnet.POS;
import org.yawni.wordnet.RelationType;
import org.yawni.wordnet.WordNet;
import org.yawni.wordnet.WordSense;

import edu.cornell.cs.nlp.utils.log.ILogger;
import edu.cornell.cs.nlp.utils.log.LoggerFactory;

/**
 * Various WordNet services that are used throughout the system. This service
 * class must be initialized after {@link WordNetServices}.
 *
 * @author Kenton Lee
 */
@Deprecated
public class WordNetServices {
	public static final ILogger LOG = LoggerFactory
			.create(WordNetServices.class);

	private static WordNetServices INSTANCE;

	private final Map<String, String>	lemmaCache;
	private final Map<String, POS>		tagMap;
	private final WordNet				wn;

	private WordNetServices() {
		wn = WordNet.getInstance();
		lemmaCache = new ConcurrentHashMap<>();
		tagMap = new ConcurrentHashMap<>();
		tagMap.put("VB", POS.VERB);
		tagMap.put("NN", POS.NOUN);
		tagMap.put("JJ", POS.ADJ);
		tagMap.put("RB", POS.ADV);
	}

	public static List<WordSense> getDerivationallyRelatedWords(String w,
			String tag) {
		return INSTANCE.wn.lookupWordSenses(w, getWordNetPOS(tag)).stream()
				.flatMap(ws -> ws
						.getRelationTargets(RelationType.DERIVATIONALLY_RELATED)
						.stream())
				.flatMap(args -> stream(args)).collect(Collectors.toList());
	}

	public static String getLemma(String word) {
		return getLemma(word, null);
	}

	public static String getLemma(String word, String tag) {
		final String lowerWord = word.toLowerCase();
		if (!INSTANCE.lemmaCache.containsKey(lowerWord)) {
			INSTANCE.lemmaCache.put(lowerWord,
					INSTANCE.wn
							.lookupBaseForms(word.toLowerCase(),
									getWordNetPOS(tag))
							.stream().findFirst().orElse(lowerWord));
		}
		return INSTANCE.lemmaCache.get(lowerWord).toLowerCase();
	}

	public static POS getWordNetPOS(String tag) {
		return tag == null || tag.length() < 2 ? POS.ALL
				: INSTANCE.tagMap.getOrDefault(
						tag.toUpperCase().substring(0, 2), POS.ALL);
	}

	public static void setInstance(WordNetServices instance) {
		INSTANCE = instance;
	}

	private static <DI> Stream<DI> stream(Iterable<DI> data) {
		return StreamSupport.stream(data.spliterator(), false);
	}

	public static class Builder {
		public WordNetServices build() {
			return new WordNetServices();
		}
	}
}