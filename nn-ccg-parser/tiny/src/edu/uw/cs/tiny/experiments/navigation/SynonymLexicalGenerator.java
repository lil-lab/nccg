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
package edu.uw.cs.tiny.experiments.navigation;

import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import edu.uw.cs.lil.tiny.ccg.lexicon.ILexicon;
import edu.uw.cs.lil.tiny.ccg.lexicon.LexicalEntry;
import edu.uw.cs.lil.tiny.data.sentence.Sentence;
import edu.uw.cs.lil.tiny.genlex.ccg.ILexiconGenerator;
import edu.uw.cs.lil.tiny.mr.lambda.LogicalExpression;
import edu.uw.cs.lil.tiny.mr.lambda.visitor.ReplaceExpression;
import edu.uw.cs.lil.tiny.parser.ISentenceLexiconGenerator;
import edu.uw.cs.utils.collections.CollectionUtils;

public class SynonymLexicalGenerator implements
		ISentenceLexiconGenerator<LogicalExpression> {
	
	// this is the full lexicon that will be used to expand input words
	// TODO [yoav] [urgent] get rid of lexicon here -- need to change interface?
	private final ILexicon<LogicalExpression>			lexicon;
	
	private final Map<List<String>, LogicalExpression>	numbers;
	
	//
	// lists of synonym sets, where each phrase is a list of words
	private final Set<Set<List<String>>>				synSets;
	
	public SynonymLexicalGenerator(ILexicon<LogicalExpression> lex,
			Set<Set<List<String>>> synSets,
			Map<List<String>, LogicalExpression> numbers) {
		this.lexicon = lex;
		this.synSets = synSets;
		this.numbers = numbers;
	}
	
	public Set<LexicalEntry<LogicalExpression>> expandNumbers(List<String> words) {
		final Set<LexicalEntry<LogicalExpression>> newEntries = new HashSet<LexicalEntry<LogicalExpression>>();
		if (numbers.containsKey(words)) {
			for (final Map.Entry<List<String>, LogicalExpression> numberPair : numbers
					.entrySet()) {
				final List<String> numberPhrase = numberPair.getKey();
				if (!numberPhrase.equals(words)) {
					for (final LexicalEntry<LogicalExpression> lex : lexicon
							.getLexEntries(numberPhrase)) {
						newEntries.add(new LexicalEntry<LogicalExpression>(
								words, lex.getCategory()
										.cloneWithNewSemantics(
												ReplaceExpression.of(
														lex.getCategory()
																.getSem(),
														numberPair.getValue(),
														numbers.get(words))),
								ILexiconGenerator.GENLEX_LEXICAL_ORIGIN));
					}
				}
			}
		}
		return newEntries;
	}
	
	public Set<LexicalEntry<LogicalExpression>> expandPhrase(
			List<String> originalPhrase, Set<List<String>> synSet) {
		final Set<LexicalEntry<LogicalExpression>> newEntries = new HashSet<LexicalEntry<LogicalExpression>>();
		for (final List<String> newPhrase : synSet) {
			if (!newPhrase.equals(originalPhrase)) {
				for (final LexicalEntry<LogicalExpression> lex : lexicon
						.getLexEntries(newPhrase)) {
					newEntries.add(new LexicalEntry<LogicalExpression>(
							originalPhrase, lex.getCategory(),
							ILexiconGenerator.GENLEX_LEXICAL_ORIGIN));
				}
			}
		}
		return newEntries;
	}
	
	public Set<LexicalEntry<LogicalExpression>> generateLexicon(
			Sentence sample, Sentence evidence) {
		final Set<LexicalEntry<LogicalExpression>> newEntries = new HashSet<LexicalEntry<LogicalExpression>>();
		
		// for each possible phrase in the sentence
		final List<String> sentence = sample.getTokens();
		for (int i = 0; i < sentence.size(); i++) {
			for (int j = i + 1; j <= sentence.size(); j++) {
				final List<String> phrase = CollectionUtils.subList(sentence,
						i, j);
				
				// for each synset that contains the phrase
				for (final Set<List<String>> synSet : synSets) {
					if (synSet.contains(phrase)) {
						
						// duplicate all of the lexical entries for the other
						// phrases in the synSet
						newEntries.addAll(expandPhrase(phrase, synSet));
						
					}
				}
				
				// expand the cardinals (first, second, third, etc.) by
				// generalizing unknown cases from lexical items for the known
				// ones
				newEntries.addAll(expandNumbers(phrase));
				
			}
		}
		
		return newEntries;
	}
	
}
