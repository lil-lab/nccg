package edu.uw.cs.lil.amr.parser.rules.coordination;

import java.io.ObjectStreamException;

import edu.cornell.cs.nlp.spf.ccg.categories.syntax.Syntax;
import edu.cornell.cs.nlp.spf.ccg.categories.syntax.Syntax.SimpleSyntax;
import edu.cornell.cs.nlp.utils.log.ILogger;
import edu.cornell.cs.nlp.utils.log.LoggerFactory;

/**
 * @author Yoav Artzi
 */
public class CoordinationSyntax extends SimpleSyntax {
	
	private static final long serialVersionUID = 8131546819143284621L;

	public static final ILogger	LOG					= LoggerFactory
			.create(CoordinationSyntax.class);

	private final Syntax		coordinatedSyntax;

	public CoordinationSyntax(Syntax coordinatedSyntax) {
		super(Syntax.C.getLabel() + "{" + coordinatedSyntax.toString() + "}");
		this.coordinatedSyntax = coordinatedSyntax;
	}

	public Syntax getCoordinatedSyntax() {
		return coordinatedSyntax;
	}
	
	@Override
	public String toString() {
		return Syntax.C.getLabel() + "{" + coordinatedSyntax.toString() + "}";
	}
	
	@Override
	protected Object readResolve() throws ObjectStreamException {
		return this;
	}

}
