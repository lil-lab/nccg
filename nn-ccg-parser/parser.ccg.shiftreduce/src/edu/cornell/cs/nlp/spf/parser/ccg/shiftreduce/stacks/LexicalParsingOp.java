package edu.cornell.cs.nlp.spf.parser.ccg.shiftreduce.stacks;

import edu.cornell.cs.nlp.spf.ccg.categories.Category;
import edu.cornell.cs.nlp.spf.parser.ParsingOp;
import edu.cornell.cs.nlp.spf.parser.ccg.rules.RuleName;
import edu.cornell.cs.nlp.spf.parser.ccg.rules.SentenceSpan;
import edu.cornell.cs.nlp.utils.log.ILogger;
import edu.cornell.cs.nlp.utils.log.LoggerFactory;
import edu.cornell.cs.nlp.spf.ccg.lexicon.LexicalEntry;

/** ParsingOp do not have lexical entry. This is required for embedding parsing op. 
 * This class solves this by extending ParsingOp */
public class LexicalParsingOp<MR> extends ParsingOp<MR> {//implements Serializable {

	private static final long serialVersionUID = 8132249799963130300L;
	//private static final long serialVersionUID = -3163301753689552425L;
	private LexicalEntry<MR> lexicalEntry;
	public static final ILogger	LOG = LoggerFactory.create(LexicalParsingOp.class);
	
	public LexicalParsingOp(Category<MR> category, SentenceSpan span, RuleName ruleName, 
							   LexicalEntry<MR> lexicalEntry) {
		super(category, span, ruleName);
		//not doing checks such as lexicalEntry is not null iff its a lexical rule
		this.lexicalEntry = lexicalEntry;
	}
	
	public LexicalEntry<MR> getEntry() {
		return this.lexicalEntry;
	}
	
	@Override
	public int hashCode() {
		final int prime = 31;
		int result = super.hashCode();
		result = prime * result + (this.lexicalEntry == null ? 0 : this.lexicalEntry.hashCode());
		return result;
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
		final LexicalParsingOp<MR> other = (LexicalParsingOp<MR>) obj;
		if (!this.lexicalEntry.equals(other.lexicalEntry)) {
			return false;
		}
		return true;
	}
	
	@Override
	public String toString() {
		return "Lexical " + super.toString();
	}
}
