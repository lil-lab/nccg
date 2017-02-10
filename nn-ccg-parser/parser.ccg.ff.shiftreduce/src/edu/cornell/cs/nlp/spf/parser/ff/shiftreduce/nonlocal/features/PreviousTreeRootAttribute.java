package edu.cornell.cs.nlp.spf.parser.ff.shiftreduce.nonlocal.features;

import java.util.Set;
import java.util.stream.Collectors;

import edu.cornell.cs.nlp.spf.base.hashvector.IHashVector;
import edu.cornell.cs.nlp.spf.ccg.categories.Category;
import edu.cornell.cs.nlp.spf.ccg.categories.syntax.ComplexSyntax;
import edu.cornell.cs.nlp.spf.ccg.categories.syntax.Syntax;
import edu.cornell.cs.nlp.spf.parser.ccg.IParseStep;
import edu.cornell.cs.nlp.spf.parser.ccg.shiftreduce.stacks.DerivationState;

public class PreviousTreeRootAttribute<MR> implements AbstractNonLocalFeature<MR> {

	private static final long serialVersionUID = 5678911130871266285L;
	private final static String LASTTAG = "PREVROOTATTRIB";
	private final static String SNDLASTTAG = "SNDPREVROOTATTRIB";
	
	@Override
	public void add(DerivationState<MR> state, IParseStep<MR> parseStep, IHashVector features, String[] buffer,
			int bufferIndex, String[] tags) {
		
		final Category<MR> last, sndLast;
		
		if(state.getRightCategory() != null) {
			last = state.getRightCategory();
			sndLast = state.getLeftCategory();
		} else {
			if(state.getLeftCategory() != null) {
				last = state.getLeftCategory();
				sndLast = null;
			} else {
				last = null;
				sndLast = null;
			}
		}
		
		final String lastAttrib;
		if(last == null) {
			lastAttrib = "null";
		} else {
			lastAttrib = this.getAttribute(last.getSyntax());
		}
		
		final String sndLastAttrib;
		if(sndLast == null) {
			sndLastAttrib = "null";
		} else {
			sndLastAttrib = this.getAttribute(sndLast.getSyntax());
		}
		
		features.add(LASTTAG, lastAttrib, 1.0);
		features.add(SNDLASTTAG, sndLastAttrib, 1.0);
	}
	
	private String getAttribute(Syntax syntax) {
		
		final String attrib;
		
		// Concrete attribute features.
		if (syntax instanceof ComplexSyntax) {
			attrib = "complex";
		} else {
			final Set<String> attributes = syntax.getAttributes();
			if (attributes.isEmpty()) {
				attrib = "noattrib";
			} else {
				attrib = attributes.stream().sorted().collect(Collectors.joining("+"));
			}
		}
		
		final String attribWithVar;

		// Agreement variable features.
		if (syntax.hasAttributeVariable()) {
			attribWithVar = attrib + "+var";
		} else {
			attribWithVar = attrib;
		}
		
		return attribWithVar;
	}
}
