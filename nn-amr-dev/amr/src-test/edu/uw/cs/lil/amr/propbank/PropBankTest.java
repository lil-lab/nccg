package edu.uw.cs.lil.amr.propbank;

import static org.junit.Assert.assertEquals;

import java.io.File;
import java.io.IOException;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import org.junit.Test;

import edu.uw.cs.lil.amr.util.propbank.PropBank;
import edu.uw.cs.lil.amr.util.wordnet.WordNetServices;

@Deprecated
public class PropBankTest {
	private final PropBank propBank;

	public PropBankTest() throws IOException {
		WordNetServices.setInstance(new WordNetServices.Builder().build());
		this.propBank = new PropBank(new File("../resources/propbank"));
	}

	@Test
	public void test() {
		doTest("testing", "test", "test-01");
		doTest("jumped", "jump", "jump-01", "jump-02", "jump-03", "jump-04",
				"jump-05", "jump-06", "jump-07");
	}

	private void doTest(String input, String expectedLemma,
			String... expectedFrameIds) {
		final String lemma = WordNetServices.getLemma(input);
		assertEquals("Lemma of " + input, expectedLemma, lemma);
		assertEquals("Frames of " + input,
				Stream.of(expectedFrameIds).collect(Collectors.toSet()),
				propBank.getFrames(lemma).stream().map(f -> f.getConstantText())
						.collect(Collectors.toSet()));
	}
}
