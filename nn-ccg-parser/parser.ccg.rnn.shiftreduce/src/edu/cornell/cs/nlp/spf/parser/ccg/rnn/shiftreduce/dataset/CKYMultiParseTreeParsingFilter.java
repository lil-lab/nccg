package edu.cornell.cs.nlp.spf.parser.ccg.rnn.shiftreduce.dataset;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.ObjectInput;
import java.io.ObjectInputStream;
import java.io.ObjectOutput;
import java.io.ObjectOutputStream;
import java.io.OutputStream;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.IdentityHashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;
import java.util.function.Predicate;

import com.google.common.base.Joiner;

import edu.cornell.cs.nlp.spf.ccg.categories.Category;
import edu.cornell.cs.nlp.spf.ccg.lexicon.LexicalEntry;
import edu.cornell.cs.nlp.spf.parser.ParsingOp;
import edu.cornell.cs.nlp.spf.parser.ccg.IOverloadedParseStep;
import edu.cornell.cs.nlp.spf.parser.ccg.IParseStep;
import edu.cornell.cs.nlp.spf.parser.ccg.cky.CKYDerivation;
import edu.cornell.cs.nlp.spf.parser.ccg.cky.chart.Cell;
import edu.cornell.cs.nlp.spf.parser.ccg.cky.steps.CKYLexicalStep;
import edu.cornell.cs.nlp.spf.parser.ccg.cky.steps.IWeightedCKYStep;
import edu.cornell.cs.nlp.spf.parser.ccg.cky.steps.WeightedCKYLexicalStep;
import edu.cornell.cs.nlp.spf.parser.ccg.cky.steps.WeightedCKYParseStep;
import edu.cornell.cs.nlp.spf.parser.ccg.rules.OverloadedRuleName;
import edu.cornell.cs.nlp.spf.parser.ccg.rules.RuleName;
import edu.cornell.cs.nlp.spf.parser.ccg.rules.SentenceSpan;
import edu.cornell.cs.nlp.spf.parser.ccg.rules.UnaryRuleName;
import edu.cornell.cs.nlp.spf.parser.ccg.shiftreduce.stacks.DerivationState;
import edu.cornell.cs.nlp.spf.parser.ccg.shiftreduce.stacks.LexicalParsingOp;
import edu.cornell.cs.nlp.utils.log.ILogger;
import edu.cornell.cs.nlp.utils.log.LogLevel;
import edu.cornell.cs.nlp.utils.log.LoggerFactory;

/** A Composite Parsing Filter that allows working with several parsing filter. This is an expensive filter.
 * 
 *  @author Dipendra Misra 
 */
public class CKYMultiParseTreeParsingFilter<MR> implements Predicate<ParsingOp<MR>>, Serializable {
	
	private static final long serialVersionUID = 2726181468372885504L;

	public static final ILogger	LOG = LoggerFactory.create(CKYMultiParseTreeParsingFilter.class);
	
	/** Category which is being derived */
	private final Category<MR> category;
	
	/** List of list of ordered steps in parse tree. These can possibly be exponential
	 * in the number of words in the sentence. */
	private final List<List<ParsingOp<MR>>> parseTrees;
	
	/** Length of sentence*/
	private final int n;
	
	/** Current cursor */
	private int cursor;
	
	/** Early updates enabled */
	private final boolean isEarlyUpdateFilter;
	private final IdentityHashMap<DerivationState<MR>, Boolean> states;
	
	/** List of mapping of current derivation states to list of parse trees. This helps the multi parse tree filter
	 * know which derivation state is trying to derive which parse tree. List of integer for a given dstate 
	 * corresponds to those parse trees which are consistent with the dstate, till the given cursor. */
	private final Map<DerivationState<MR>, List<Integer>> map;
	
	/** List of new mapping of derivation states to cursor */
	private final Map<DerivationState<MR>, List<Integer>> newMap;
	
	public CKYMultiParseTreeParsingFilter(CKYDerivation<MR> bestDerivation,
			Predicate<ParsingOp<MR>> supervisedPruningFilter, int n) {
		
		Cell<MR> bestDerivationCell = bestDerivation.getCell();
		this.category = bestDerivationCell.getCategory();
	
		this.cursor = 0;
		this.n = n;
		
		//TODO: move this 1000 up and set it as a constant modifiable from the inc files.
		List<List<ParsingOp<MR>>> allParseTrees = this.createFilterForAllParseTree(bestDerivationCell);
		LOG.info("There are %s many viterbi parse trees in CKY", allParseTrees.size());
		this.parseTrees = new ArrayList<List<ParsingOp<MR>>>(allParseTrees.subList(0, Math.min(1000, allParseTrees.size())));
											//this.createFilterForAllParseTree(bestDerivationCell);
		
		this.map = new IdentityHashMap<DerivationState<MR>, List<Integer>>();
		this.newMap = new IdentityHashMap<DerivationState<MR>, List<Integer>>();
		
		this.isEarlyUpdateFilter = false;
		this.states = null;
		
		if(LOG.getLogLevel() == LogLevel.DEBUG) {
			this.printParseTrees();
		}
		
		LOG.info("Multi CKY Filter. n = %s, number of parse trees %s", n, this.parseTrees.size());
	}
	
	/** Constructor below is used for early updates when we want to create data
	 * without having the complete parse tree from CKY. As input, we are given 
	 * cells from left to right i.e. cell_{i+1}'s start = cell_{i}'s end + 1 */
	public CKYMultiParseTreeParsingFilter(List<Cell<MR>> cells, int n) {
		
		this.category = null; //A single category is not derived
	
		this.cursor = 0;
		this.n = n;
		
		// For testing, currently working with only 1 parse tree
		List<ParsingOp<MR>> tree = new ArrayList<ParsingOp<MR>>();
		
		for(Cell<MR> cell: cells) {
			List<List<ParsingOp<MR>>> cellTrees = this.createFilterForAllParseTree(cell);
			tree.addAll(cellTrees.get(0)); //arbitrarily selecting on tree to add
		}
		
		this.parseTrees = new ArrayList<List<ParsingOp<MR>>>();
		this.parseTrees.add(tree);
		
		this.map = new IdentityHashMap<DerivationState<MR>, List<Integer>>();
		this.newMap = new IdentityHashMap<DerivationState<MR>, List<Integer>>();
		
		this.isEarlyUpdateFilter = true;
		this.states = new IdentityHashMap<DerivationState<MR>, Boolean>();
		
		if(LOG.getLogLevel() == LogLevel.DEBUG) {
			this.printParseTrees();
		}
		
		LOG.info("Multi CKY Filter with early updates"
				+ ". n = %s, number of cells %s, number of parse trees %s", n, cells.size(), this.parseTrees.size());
	}
	
	public void shiftParsingOpSpan() {
		
		if(this.parseTrees.size() > 1) {
			throw new RuntimeException("Works only for one tree");
		}
		
		List<ParsingOp<MR>> tree = this.parseTrees.get(0);
		List<ParsingOp<MR>> newTree = new ArrayList<ParsingOp<MR>>();
		
		int shiftBy = this.getStart();
		int senLen = this.getEnd() - shiftBy;
		
		for(ParsingOp<MR> op: tree) {
			
			final SentenceSpan span = op.getSpan();
			final SentenceSpan newSpan = new SentenceSpan(span.getStart() - shiftBy, span.getEnd() - shiftBy, senLen);
			final ParsingOp<MR> newOp;
			
			if(op instanceof LexicalParsingOp) {
				newOp = new LexicalParsingOp<MR>(op.getCategory(), newSpan, op.getRule(), ((LexicalParsingOp<MR>) op).getEntry());
			} else {
				newOp = new ParsingOp<MR>(op.getCategory(), newSpan, op.getRule()); 
			}
			
			newTree.add(newOp);
		}
		
		this.parseTrees.clear();
		this.parseTrees.add(newTree);
	}
	
	public List<List<ParsingOp<MR>>> getParseTrees() {
		return this.parseTrees;
	}
	
	public void trim() {
		List<ParsingOp<MR>> e = this.parseTrees.get(0);
		this.parseTrees.clear();
		this.parseTrees.add(e);
	}
	
	public Category<MR> getCategory() {
		return this.category;
	}
	
	public int numParseTrees() {
		return this.parseTrees.size();
	}

	@Override
	public boolean test(ParsingOp<MR> op) {
		throw new RuntimeException("Operation not supported. Use test(ParsingOp<MR>, DerivationState<MR>");
	}
	
	public boolean test(ParsingOp<MR> op, DerivationState<MR> parent) {
		
		List<Integer> ix = this.map.get(parent);
		
		if(ix == null) {
			if(this.cursor != 0) {

				int hashCode = parent.hashCode();
				LOG.info("Parent hashcode %s", hashCode);
				LOG.info("Parent Left Category %s",  parent.getLeftCategory());
				for(Entry<DerivationState<MR>, List<Integer>> e: this.map.entrySet()) {
					if(e.getKey().hashCode() == hashCode) {
						LOG.info("Hashcode matched. Equality %s", e.getKey().equals(parent));
					}
				}
				
				throw new RuntimeException("Test: Possible Bug or Incorrect Usage. Parent not registered. Cursor " + this.cursor);
			}
			
//			int j = 0;
			for(List<ParsingOp<MR>> parseTree: this.parseTrees) {
				ParsingOp<MR> gTruth = parseTree.get(0);

//				LOG.info("Comparing \n \t %s \n \t %s", gTruth, op);
				if(gTruth.equals(op)) {
//					LOG.debug("\n gtruth %s\n and op %s", gTruth, op);
//					if(this.parseTrees.size() > 1) 
//						LOG.info("Cursor %s. Parent %s. Passed %s op %s", this.cursor, parent.hashCode(), j, op);
					return true;
				}
//				j++;
			}
			
			return false;
		}
		
		for(Integer i: ix) {
			
			if(this.cursor >= this.parseTrees.get(i).size()) {
				continue;
			}
			
			final ParsingOp<MR> gTruth = this.parseTrees.get(i).get(this.cursor);
			
//			LOG.info("Comparing \n \t %s \n \t %s", gTruth, op);
			if(gTruth.equals(op)) {
//				LOG.debug("\n gtruth %s\n and op %s", gTruth, op);
//				if(this.parseTrees.size() > 1) 
//					LOG.info("Cursor %s. Parent %s. Passed %s op %s", this.cursor, parent.hashCode(), i, op);
				return true;
			}
		}
			
		return false;
	}
	
	@SuppressWarnings("unchecked")
	public void checkTest(Category<MR> c, Category<MR> other) {
		
		final String fileName = "./ctest.ser";
		
		//Serialize it
		try (
			      OutputStream file = new FileOutputStream(fileName);
			      OutputStream buffer = new BufferedOutputStream(file);
			      ObjectOutput output = new ObjectOutputStream(buffer);
			) {
			      output.writeObject(c);
			} catch(IOException e) {
			      throw new RuntimeException("Dataset Filters could not be saved. Exception " + e);
			}
		
		
		//Deserialize it
		final Category<MR> c1;
		try (
			    InputStream file = new FileInputStream(fileName);
			    InputStream buffer = new BufferedInputStream(file);
			    ObjectInput input = new ObjectInputStream (buffer);
			) {
				c1 = (Category<MR>)input.readObject();
			} catch(Exception e) {
				throw new RuntimeException("Could not deserialize AMR parsing filter. Error: " + e);
			}
		
		if(!c.equals(c1)) {
			LOG.info("Category Testing %s and %s fails and also %s", c, c1, c1.equals(other));
			System.exit(0);
		}

		LOG.info("Cataegory Check Test Pass");
	}
	
	@SuppressWarnings("unchecked")
	public void checkTest(ParsingOp<MR> op) {

		final String fileName = "./test.ser";
		
		//Serialize it
		try (
			      OutputStream file = new FileOutputStream(fileName);
			      OutputStream buffer = new BufferedOutputStream(file);
			      ObjectOutput output = new ObjectOutputStream(buffer);
			) {
			      output.writeObject(op);
			} catch(IOException e) {
			      throw new RuntimeException("Dataset Filters could not be saved. Exception " + e);
			}
		
		
		//Deserialize it
		final ParsingOp<MR> op1;
		try (
			    InputStream file = new FileInputStream(fileName);
			    InputStream buffer = new BufferedInputStream(file);
			    ObjectInput input = new ObjectInputStream (buffer);
			) {
				op1 = (ParsingOp<MR>)input.readObject();
			} catch(Exception e) {
				throw new RuntimeException("Could not deserialize AMR parsing filter. Error: " + e);
			}
		
		if(!op.equals(op1)) {
			LOG.info("Testing Equality %s and %s %s", op.equals(op1), op, op1);
			System.exit(0);
		}
				
		LOG.info("Check Test Pass");
	}
	
	public boolean isEarlyUpdateFilter() {
		return this.isEarlyUpdateFilter;
	}
	
	/** This is a ugly hack which we do for handling early updates. 
	 * Basically if a dstate has reached then end of a list of parsing op
	 * representing a dstate in itself; then we stop and add it to a special
	 * states list. This way, we can get these states representing "good" partial
	 * derivations of parse tree. */
	public void addIfEnd(DerivationState<MR> dstate, List<Integer> indices) {
		
		if(!this.isEarlyUpdateFilter) {
			return;
		}
		
		for(Integer i: indices) {	
			if(this.parseTrees.get(i).size() == this.cursor + 1) {
				
				synchronized (this.states) {
					this.states.put(dstate, true);
				}
				break;
			}
		}
	}
	
	public Set<DerivationState<MR>> getEarlyUpdateStates() {
		return this.states.keySet();
	}
	
	public int getStart() {
		return this.parseTrees.get(0).get(0).getSpan().getStart();
	}
	
	public int getEnd() {
		List<ParsingOp<MR>> headTree = this.parseTrees.get(0);
		return headTree.get(headTree.size() - 1).getSpan().getEnd();
	}
 	
	/** Register a new derivation state */
	public void register(ParsingOp<MR> op, DerivationState<MR> parent, DerivationState<MR> dNew) {
		
		List<Integer> ix = this.map.get(parent);
		if(ix == null) {
			
			if(this.cursor != 0) {
				throw new RuntimeException("Register: Possible Bug or Incorrect Usage. Parent not registered " + this.cursor);
			}
			
			List<Integer> newIx = new LinkedList<Integer>();
			boolean found = false;
			
			int j = 0;
			for(List<ParsingOp<MR>> parseTree: this.parseTrees) {
				ParsingOp<MR> gTruth = parseTree.get(this.cursor);
				
				if(gTruth.equals(op)) {
					found = true;
					newIx.add(j);
				}
				j++;
			}
			
			synchronized(this.newMap) {
				if(this.newMap.containsKey(dNew)) {
					throw new RuntimeException("dNew already contained");
				}
				this.newMap.put(dNew, newIx);
			}
			
			this.addIfEnd(dNew, newIx);
			
			if(this.parseTrees.size() > 1) {
				LOG.debug("Cursor %s. Parent %s. State %s. Registering to %s op %s", this.cursor, parent.hashCode(),
									dNew.hashCode(), Joiner.on(", ").join(newIx), op);
			}
		
			if(!found) {
				throw new RuntimeException("Bug. Register cannot deduce how dNew passed the test.");
			}
			
		} else {
			
			List<Integer> newIx = new LinkedList<Integer>();
			for(Integer j: ix) {
				
				ParsingOp<MR> gTruth = this.parseTrees.get(j).get(this.cursor);
				
				if(gTruth.equals(op)) {
					newIx.add(j);
				}
			}
			
			synchronized(this.newMap) {
				if(this.newMap.containsKey(dNew)) {
					throw new RuntimeException("dNew already contained");
				}
				this.newMap.put(dNew, newIx);
			}
			
			this.addIfEnd(dNew, newIx);
			
			if(this.parseTrees.size() > 1) {
				LOG.debug("Cursor %s. Parent %s. State %s. Registering to %s op %s", this.cursor, parent.hashCode(),
										dNew.hashCode(), Joiner.on(", ").join(newIx), op);
			}
		}
	}
	
	/** Creates filter for every parse tree */
	private List<List<ParsingOp<MR>>> createFilterForAllParseTree(Cell<MR> cell) {
		
		List<IWeightedCKYStep<MR>> viterbiSteps = cell.getViterbiSteps();
		
		List<List<ParsingOp<MR>>> parseTrees = new LinkedList<List<ParsingOp<MR>>>();
				
		for(IWeightedCKYStep<MR> viterbiStep: viterbiSteps) {
			
			List<ParsingOp<MR>> stepParsingOp = getOrderedParsingOpFromStep(viterbiStep);
			
			//get the child cell of this step
			int numChild = viterbiStep.numChildren();
			if(numChild == 0) {
				parseTrees.add(stepParsingOp);
			} else if(numChild == 1) {
				List<List<ParsingOp<MR>>> subTrees = 
								this.createFilterForAllParseTree(viterbiStep.getChildCell(0));
				for(List<ParsingOp<MR>> orderedParsingOp: subTrees) {
					orderedParsingOp.addAll(stepParsingOp);
				}	
				parseTrees.addAll(subTrees);
				
			} else if(numChild == 2) {
				
				Cell<MR> leftCell = viterbiStep.getChildCell(0);
				Cell<MR> rightCell = viterbiStep.getChildCell(1);
				
				final List<List<ParsingOp<MR>>> leftOrdered, rightOrdered;
				
				if(leftCell.getStart() < rightCell.getStart()) {
					leftOrdered = this.createFilterForAllParseTree(leftCell);
					rightOrdered = this.createFilterForAllParseTree(rightCell);
				} else {
					leftOrdered = this.createFilterForAllParseTree(rightCell);
					rightOrdered = this.createFilterForAllParseTree(leftCell);
				}
				
				//merge the two list of trees in inorder style
				for(List<ParsingOp<MR>> left: leftOrdered) {
					for(List<ParsingOp<MR>>  right: rightOrdered) {
						List<ParsingOp<MR>> orderedParsingOp = new LinkedList<ParsingOp<MR>>();
						orderedParsingOp.addAll(left);
						orderedParsingOp.addAll(right);
						orderedParsingOp.addAll(stepParsingOp);
						
						parseTrees.add(orderedParsingOp);
					}
				}
					
			} else if (numChild > 2) {
				throw new RuntimeException("Only handling binary parse trees");
			}
			
		}
		
		return parseTrees;
	}
	
	private List<ParsingOp<MR>> getOrderedParsingOpFromStep(IWeightedCKYStep<MR> step) {
		
		RuleName ruleName = step.getRuleName();
		
		List<ParsingOp<MR>> parsingOps = new LinkedList<ParsingOp<MR>>();
		
		if(step.getRuleName() == null)
			throw new RuntimeException("Null rule name");
		
		SentenceSpan span = new SentenceSpan(step.getStart(), step.getEnd() + 1, this.n); //span in SR notation
		
		if (ruleName instanceof OverloadedRuleName) {
			//Special case: when unary rules are overloaded with lexical or binary rules
			
			final IParseStep<MR> parseStep;
			
			if(step instanceof WeightedCKYLexicalStep) {
				parseStep = ((WeightedCKYLexicalStep<MR>)step).getStep();
			} else if(step instanceof WeightedCKYParseStep) {
				parseStep = ((WeightedCKYParseStep<MR>)step).getStep();
			} else {
				throw new RuntimeException("CKY step that is neither lexical step nor parse step.");
			}
			
			IOverloadedParseStep<MR> overloadedStep = (IOverloadedParseStep<MR>)parseStep;
			
			UnaryRuleName unaryRule = ((OverloadedRuleName) ruleName).getUnaryRule();
			RuleName baseRule = ((OverloadedRuleName) ruleName).getOverloadedRuleName();
			
			final ParsingOp<MR> baseOp;
			if(overloadedStep instanceof CKYLexicalStep) {
				@SuppressWarnings("unchecked")
				LexicalEntry<MR> lexicalEntry = ((CKYLexicalStep<MR>)overloadedStep).getLexicalEntry();
				baseOp = new LexicalParsingOp<MR>(overloadedStep.getIntermediate(), span, baseRule, lexicalEntry);
			} else {
				baseOp = new ParsingOp<MR>(overloadedStep.getIntermediate(), span, baseRule);
			}
			
			ParsingOp<MR> unaryOp = new ParsingOp<MR>(step.getRoot(), span, unaryRule);
			
			parsingOps.add(baseOp);
			parsingOps.add(unaryOp);
		} else {
		
			final ParsingOp<MR> op;
			if(step instanceof WeightedCKYLexicalStep) {
				LexicalEntry<MR> lexicalEntry = ((WeightedCKYLexicalStep<MR>)step).getLexicalEntry();
				op = new LexicalParsingOp<MR>(step.getRoot(), span, ruleName, lexicalEntry);
			} else {
				op = new ParsingOp<MR>(step.getRoot(), span, ruleName);
			}
			
			parsingOps.add(op);
		}
		
		return parsingOps;
	}
	
	private void printParseTrees() {

		int ix = 0;
		LOG.debug("Number of parse trees %s", this.parseTrees.size());
		
		for(List<ParsingOp<MR>> parseTree: this.parseTrees) {
			LOG.debug("Parse Tree %s {", ++ix);
			for(ParsingOp<MR> parsingOp: parseTree) {
				LOG.debug(parsingOp);
			}
			LOG.debug("}, ");
		}
	}
	
	public String getIndexOfState(DerivationState<MR> dstate) {
		
		//Complete parse trees need not be registered so we check their parent
		DerivationState<MR> parent = dstate.getParent();
		
		List<Integer> ix = this.map.get(parent);
		if(ix == null) {
			ix = this.newMap.get(parent);
			if(ix == null) {
				return null;
			}
		}
		
		return Joiner.on(", ").join(ix);
	}
	
	public void incrementCursor() {
		this.cursor++;
		this.map.clear();
		this.map.putAll(this.newMap);
		this.newMap.clear();
		LOG.debug("/------------ Cursor updated to %s --------- (%s, %s)/", this.cursor, this.map.size(), this.newMap.size());
		for(Entry<DerivationState<MR>, List<Integer>> e: this.map.entrySet()) {
//			if(e.getValue().size() > 1) {
//				LOG.info("%s is currently at %s", e.getKey().getDebugHashCode(), Joiner.on(", ").join(e.getValue()));
//			}
			LOG.debug("Tree %s", e.getKey().getDebugHashCode());
			for(int tree: e.getValue()) {
				final ParsingOp<MR> action;
				if(this.cursor == this.parseTrees.get(tree).size()) {
					action = null;
				} else {
					action = this.parseTrees.get(tree).get(this.cursor);
				}
				LOG.debug(">>>> %s, steps left %s, next action %s", tree, this.parseTrees.get(tree).size() - this.cursor, action);
			}
		}
	}
	
	public void clearCursor() {
		this.cursor = 0;
		this.map.clear();
		this.newMap.clear();
		
		if(this.states != null) {
			this.states.clear();
		}
	}
	
	public String toString() {
		
		StringBuilder s = new StringBuilder();
		s.append("cursor = " + this.cursor + ", n = " + this.n + ", num trees = " + this.parseTrees.size() + "\n");
		for(List<ParsingOp<MR>> tree: this.parseTrees) {
			s.append("{");
			for(ParsingOp<MR> decision: tree) {
				s.append(decision + "\n");
			}
			s.append("} ");
		}
		
		return s.toString();
	}
}
