package edu.uw.cs.lil.amr.wordnet;

import static org.junit.Assert.assertEquals;

import org.junit.Test;

import edu.uw.cs.lil.amr.util.wordnet.WordNetServices;

@Deprecated
public class WordNetServicesTest {
	static {
		WordNetServices.setInstance(new WordNetServices.Builder().build());
	}

	@Test
	public void testLemmatizer() {
		assertEquals("Testing 'running'", "run",
				WordNetServices.getLemma("running", "VB"));
		assertEquals("Testing 'were'", "be",
				WordNetServices.getLemma("were", "VB"));
		assertEquals("Testing 'humans'", "human",
				WordNetServices.getLemma("humans", "NN"));
		assertEquals("Testing 'notarealword'", "notarealword",
				WordNetServices.getLemma("notarealword", "NOTAREALPOS"));
		assertEquals("Testing 'was'", "be",
				WordNetServices.getLemma("was", "VB"));
		assertEquals("Testing 'meeting'", "meet",
				WordNetServices.getLemma("meeting", "VB"));
	}

	@Test
	public void testLemmatizerNoTag() {
		assertEquals("Testing 'running'", "run",
				WordNetServices.getLemma("running"));
		assertEquals("Testing 'were'", "be", WordNetServices.getLemma("were"));
		assertEquals("Testing 'humans'", "human",
				WordNetServices.getLemma("humans"));
		assertEquals("Testing 'notarealword'", "notarealword",
				WordNetServices.getLemma("notarealword"));
		assertEquals("Testing 'was'", "be", WordNetServices.getLemma("was"));
		assertEquals("Testing 'meeting'", "meet",
				WordNetServices.getLemma("meeting"));
	}
}
