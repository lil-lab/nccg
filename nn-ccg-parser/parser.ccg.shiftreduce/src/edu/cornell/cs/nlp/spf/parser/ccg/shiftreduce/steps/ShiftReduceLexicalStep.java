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
package edu.cornell.cs.nlp.spf.parser.ccg.shiftreduce.steps;

import java.io.Serializable;

import edu.cornell.cs.nlp.spf.ccg.categories.Category;
import edu.cornell.cs.nlp.spf.ccg.lexicon.LexicalEntry;
import edu.cornell.cs.nlp.spf.parser.ccg.ILexicalParseStep;
import edu.cornell.cs.nlp.spf.parser.ccg.rules.RuleName;

public class ShiftReduceLexicalStep<MR> extends AbstractShiftReduceStep<MR> implements
		ILexicalParseStep<MR>, Serializable {

	private static final long serialVersionUID = 2982523642276651260L;

	private int						hashCode;

	private final LexicalEntry<MR>	lexicalEntry;

	public ShiftReduceLexicalStep(Category<MR> root, LexicalEntry<MR> lexicalEntry,
			boolean isFullParse, int start, int end) {
		this(root, lexicalEntry, isFullParse, LEXICAL_DERIVATION_STEP_RULENAME,
				start, end);
		this.hashCode = calcHashCode();
	}

	public ShiftReduceLexicalStep(LexicalEntry<MR> lexicalEntry, boolean isFullParse,
			int start, int end) {
		this(lexicalEntry.getCategory(), lexicalEntry, isFullParse,
				LEXICAL_DERIVATION_STEP_RULENAME, start, end);
	}

	private ShiftReduceLexicalStep(Category<MR> root, LexicalEntry<MR> lexicalEntry,
			boolean isFullParse, RuleName ruleName, int start, int end) {
		super(root, ruleName, isFullParse, start, end);
		this.lexicalEntry = lexicalEntry;
	}

	@Override
	public boolean equals(Object obj) {
		if (this == obj) {
			return true;
		}
		if (!super.equals(obj)) {
			return false;
		}
		if (getClass() != obj.getClass()) {
			return false;
		}
		@SuppressWarnings("unchecked")
		final ShiftReduceLexicalStep<MR> other = (ShiftReduceLexicalStep<MR>) obj;
		if (lexicalEntry == null) {
			if (other.lexicalEntry != null) {
				return false;
			}
		} else if (!lexicalEntry.equals(other.lexicalEntry)) {
			return false;
		}
		return true;
	}

	@Override
	public LexicalEntry<MR> getLexicalEntry() {
		return lexicalEntry;
	}

	@Override
	public int hashCode() {
		return hashCode;
	}

	@Override
	public String toString(boolean verbose, boolean recursive) {
		return new StringBuilder("[").append(getStart()).append("-")
				.append(getEnd()).append(" :: ").append(getRuleName())
				.append(" :: ").append(lexicalEntry).append("{")
				.append(lexicalEntry.getOrigin()).append("}]").toString();
	}

	private int calcHashCode() {
		final int prime = 31;
		int result = super.hashCode();
		result = prime * result
				+ (lexicalEntry == null ? 0 : lexicalEntry.hashCode());
		return result;
	}

}
