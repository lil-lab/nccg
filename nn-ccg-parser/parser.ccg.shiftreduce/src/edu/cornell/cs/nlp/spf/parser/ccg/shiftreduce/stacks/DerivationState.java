package edu.cornell.cs.nlp.spf.parser.ccg.shiftreduce.stacks;

import java.util.List;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.Serializable;
import java.util.LinkedHashSet;
import java.util.LinkedList;

import edu.cornell.cs.nlp.spf.base.hashvector.HashVectorFactory;
import edu.cornell.cs.nlp.spf.base.hashvector.IHashVector;
import edu.cornell.cs.nlp.spf.ccg.categories.Category;
import edu.cornell.cs.nlp.spf.ccg.lexicon.LexicalEntry;
import edu.cornell.cs.nlp.spf.parser.ParsingOp;
import edu.cornell.cs.nlp.spf.parser.RuleUsageTriplet;
import edu.cornell.cs.nlp.spf.parser.ccg.rules.LexicalResult;
import edu.cornell.cs.nlp.spf.parser.ccg.rules.RuleName;
import edu.cornell.cs.nlp.spf.parser.ccg.rules.SentenceSpan;
import edu.cornell.cs.nlp.spf.parser.ccg.shiftreduce.steps.IWeightedShiftReduceStep;
import edu.cornell.cs.nlp.spf.parser.ccg.shiftreduce.steps.ShiftReduceLexicalStep;
import edu.cornell.cs.nlp.utils.composites.Pair;
import edu.cornell.cs.nlp.utils.log.ILogger;
import edu.cornell.cs.nlp.utils.log.LogLevel;
import edu.cornell.cs.nlp.utils.log.LoggerFactory;

/**
* Derivation state: ordered sequence of trees containing a partial parse
*
* @author Dipendra Misra
*/
public class DerivationState<MR> implements Serializable {
	
	private static final long serialVersionUID = -7370456068410995260L;

	/** number of words consumed by this state. Same as sum of all leaves of all the trees*/
	public int wordsConsumed;  
	
	/** log-likelihood score of this derivation state*/
	public double score;
	
	/** linear score for this derivation state*/
	public double linearScore;
	
	/** average of all the features computed at various points while constructing this state from scratch */
	private IHashVector avgFeatures;
	
	/** hashCode and the debugHashCode. HashCode only depends upon the root of tree segments in this derivation state
	 *  This is useful during packing in HashMaps. DebugHashCode has more information and covers the entire state.
	 *  This is useful for debugging. */
	
	private int hashCode, debugHashCode;
	
	public static final ILogger	LOG	= LoggerFactory.create(DerivationState.class);
	
	/** Certain constraints about the data-structure: 
	 * 1. if can never happen that right is not null while left is null
	 * 2. if left is not-null and right is null then there is only one node in the entire
	 *    segment arced by this state i.e. lenRoot = 1 */
	
	private Category<MR> left, right;
	private SentenceSpan leftSpan, rightSpan;
	private RuleName ruleName, leftRuleName; //ruleName is same as rightRuleName
	private ShiftReduceRuleNameSet<MR> leftSet, rightSet;
	private LexicalEntry<MR> lexicalEntry;
	protected DerivationState<MR> parent, nextLeft;
	private int lenRoot;
	private IWeightedShiftReduceStep<MR> step;
	private boolean isBinary;
	
	/** list of actions that were considered to create this dstate.
	 * These actions are explicitly stored for optimization purposes. */
	private List<ParsingOp<MR>> possibleActions;
	
	/** list of features of actions that were considered to create this dstate.
	 * These actions are explicitly stored for optimization purposes. */
	private List<IHashVector> possibleActionsFeatures;
	
	private IHashVector stateFeature;
	
	/** Persistent embeddings used by recurrent neural networks for encoding parsing operations 
	 * that constructed this derivation state and the categories in the derivation state
	 * itself*/
	
	private PersistentEmbeddings stateEmbedding;
	private PersistentEmbeddings parsingOpEmbedding;
	
	/** encoding of the derivation state, action history and buffer. Used for optimization */
	private INDArray encoding;
	
	/** A flag used for debugging. A tainted derivation state is one which should not have passed
	 * through a dataset creator pruning filter but it did. It is used for parser check to find 
	 * early errors. By default all derivation state are not tainted.*/
	public boolean tainted;
	
	/** which child is this of its parent. Use for debugging purpose. to be removed in prod version.  */
	public int childIndex;
	
	/** number of trees */
	public int numTree;
	
	/** packed state: If this derivation state was packed during parsing then this variable
	 * refers to that packed state. */
	private PackedState<MR> packedState;
	
	/** Perceptron feature*/
	private INDArray perceptronFeature;
	public double neuralModelScore;
	
	public DerivationState() {
		this.wordsConsumed = 0;
		this.score = 0.0;
		this.linearScore = 0.0;
		this.avgFeatures = HashVectorFactory.create();
		this.hashCode = -1;
		this.debugHashCode = -1;
		
		this.left = null;
		this.right = null;
		this.leftSpan = null;
		this.rightSpan = null;
		this.ruleName = null;
		this.leftRuleName = null;
		this.leftSet = null;
		this.rightSet = null;
		this.lexicalEntry = null;
		this.parent = null;
		this.nextLeft = null;
		this.lenRoot = 0;
		this.step = null;
		this.isBinary = false;
		
		this.stateEmbedding = null;
		this.parsingOpEmbedding = null;
		this.encoding = null;
		
		this.tainted = false;
		this.childIndex = -1;
		this.numTree = 0;
		this.perceptronFeature = null;
		this.neuralModelScore = 0.0;
	}
	
	public ShiftReduceRuleNameSet<MR> returnLastNonTerminal() {
		return (this.rightSet != null) ? this.rightSet : this.leftSet;
	}
	
	public SentenceSpan returnLastSentenceSpan() {
		return (this.right != null) ? this.rightSpan : this.leftSpan;
	}
	
	public ShiftReduceRuleNameSet<MR> return2ndLastNonTerminal() {
		return (this.rightSet != null) ? this.leftSet : null;
	}
	
	public SentenceSpan return2ndLastSentenceSpan() {
		return (this.right != null) ? this.leftSpan : null;		
	}
	
	public List<Category<MR>> returnBothCategories() {
		List<Category<MR>> categories = new LinkedList<Category<MR>>();
		if(left != null)
			categories.add(left);
		if(right != null)
			categories.add(right);
		
		return categories;
	}
	
	public void initializeRuleNameSet() {
		if(this.left != null) {
			this.leftSet = new ShiftReduceRuleNameSet<MR>(this.leftRuleName, this.left);
		}
		
		if(this.right != null) {
			this.rightSet = new ShiftReduceRuleNameSet<MR>(this.ruleName, this.right);
		}
	}
	
	public int lenRoot() {
		return this.lenRoot;
	}
	
	public Category<MR> getLeftCategory() {
		return this.left;
	}
	
	public Category<MR> getRightCategory() {
		return this.right;
	}
	
	public SentenceSpan getLeftSpan() {
		return this.leftSpan;
	}
	
	public SentenceSpan getRightSpan() {
		return this.rightSpan;
	}
	
	public ShiftReduceRuleNameSet<MR> getLeftSet() {
		return this.leftSet;
	}
	
	public ShiftReduceRuleNameSet<MR> getRightSet() {
		return this.rightSet;
	}
	
	public void setPackedState(PackedState<MR> packedState) {
		this.packedState = packedState;
	}
	
	public void unpack() {
		this.packedState = null;
	}
	
	public void appendPerceptronFeature(INDArray feat) {
		
		if(this.perceptronFeature == null) {	
			this.perceptronFeature = feat;
		} else {
			this.perceptronFeature = Nd4j.concat(1, this.perceptronFeature, feat);
		}
	}
	
	public INDArray getPerceptronFeature() {
		return this.perceptronFeature;
	}
	
	public DerivationStateHorizontalIterator<MR> horizontalIterator() {
		return new DerivationStateHorizontalIterator<MR>(this);
	}
	
	public DerivationStateVerticalIterator<MR> verticalIterator() {
		return new DerivationStateVerticalIterator<MR>(this);
	}
	
	public void defineStep(IWeightedShiftReduceStep<MR> step_) {
		this.step = step_;
	}
	
	/*public void defineStep() {
		if(this.ruleName.equals(ShiftReduceLexicalStep.LEXICAL_DERIVATION_STEP_RULENAME)) {
			this.step = new WeightedShiftReduceLexicalStep<MR>(this.score);
		}
		else {
			this.step = new WeightedShiftReduceParseStep<MR>(this.score);	
		}
	}*/
	
	public IWeightedShiftReduceStep<MR> returnStep() {
		return this.step;
	}
	
	public ParsingOp<MR> returnParsingOp() {
		RuleName ruleName = this.ruleName;
		SentenceSpan span = this.returnLastSentenceSpan();
		Category<MR> categ = this.returnLastNonTerminal().getCategory();
		
		if(this.lexicalEntry == null) {
			return new ParsingOp<MR>(categ, span, ruleName);
		} else {
			return new LexicalParsingOp<MR>(categ, span, ruleName, this.lexicalEntry);
		}
	}
	
	public boolean isUnary() {
		if(!this.ruleName.equals(ShiftReduceLexicalStep.LEXICAL_DERIVATION_STEP_RULENAME) && !this.isBinary)
			return true;
		return false;
	}
	
	public List<IWeightedShiftReduceStep<MR>> returnSteps() {
		DerivationState<MR> iter = this;
		List<IWeightedShiftReduceStep<MR>> steps = new LinkedList<IWeightedShiftReduceStep<MR>>();
		
		while(iter != null) {
			if(iter.step != null)
				steps.add(iter.step);
			
			iter = iter.parent;
		}
		
		return steps;
	}
	
	/** Returns a list of parsing ops used to create this state. This list is returned in 
	 * with most recent parsing op first in the list. LexicalParseRule are used to represent 
	 * lexical step. */
	public List<ParsingOp<MR>> returnParsingOps() {
		DerivationState<MR> iter = this;
		List<ParsingOp<MR>> parsingOps = new LinkedList<ParsingOp<MR>>();
		
		while(iter != null) {
			if(iter.step != null) {
				SentenceSpan span = iter.rightSpan;
				if(iter.rightSpan == null)
					span = iter.leftSpan;
				
				final ParsingOp<MR> op;
				if(iter.lexicalEntry != null) { //is a lexical step
					
					assert iter.ruleName.equals(ShiftReduceLexicalStep.LEXICAL_DERIVATION_STEP_RULENAME);
					
					op = new LexicalParsingOp<MR>(iter.step.getRoot(), span, 
										iter.step.getRuleName(), iter.lexicalEntry);
				} else {
					op = new ParsingOp<MR>(iter.step.getRoot(), span, iter.step.getRuleName());
				}
				parsingOps.add(op);
			}
			
			iter = iter.parent;
		}
		
		return parsingOps;
	}
	
	//iteratively computes all the rule triplets used
	public List<RuleUsageTriplet> returnAllRuleUsageTriplet() {
		List<RuleUsageTriplet> rules = new LinkedList<RuleUsageTriplet>();
		DerivationState<MR> iter = this;
		while(iter != null)
		{
			if(iter.ruleName != null)
			{
				List<Pair<Integer, Integer>> nodeRuleList = new LinkedList<Pair<Integer, Integer>>();
				
				if(iter.leftSpan != null)
					nodeRuleList.add(Pair.of(iter.leftSpan.getStart(), iter.leftSpan.getEnd()));
				if(iter.rightSpan != null)
					nodeRuleList.add(Pair.of(iter.rightSpan.getStart(), iter.rightSpan.getEnd()));
				
				RuleUsageTriplet nodeRule = new RuleUsageTriplet(iter.ruleName, nodeRuleList);
				rules.add(nodeRule);
			}
			iter = iter.parent;
		}
		
		return rules;
	}
	
	//iteratively computes lexical entries
	public LinkedHashSet<LexicalEntry<MR>> returnLexicalEntries() {
		final LinkedHashSet<LexicalEntry<MR>> result = new LinkedHashSet<LexicalEntry<MR>>();
		
		DerivationState<MR> iter = this;
		while(iter != null)
		{
			if(iter.lexicalEntry != null)
				result.add(iter.lexicalEntry);
			
			iter = iter.parent;
		}
		
		return result;
	}
	
	/** This function is used for shifting lexical results. For normal lexical entry shifting, see the other result. */
	public DerivationState<MR> shift(LexicalResult<MR> lexicalResult, int words, SentenceSpan span) {
		DerivationState<MR> nDState = new DerivationState<MR>();
		nDState.parent = this;
		nDState.wordsConsumed = this.wordsConsumed + words; 
		nDState.lenRoot = this.lenRoot + 1;
		
		if(this.left == null) {
			nDState.left = lexicalResult.getResultCategory();
			nDState.leftSpan = span;
			//next line is a bug fix. 
			nDState.leftRuleName = ShiftReduceLexicalStep.LEXICAL_DERIVATION_STEP_RULENAME;
		}
		else if(this.right == null) {
			nDState.left = this.left;
			nDState.leftSpan = this.leftSpan;
			nDState.leftRuleName = this.ruleName;
			nDState.right = lexicalResult.getResultCategory();
			nDState.rightSpan = span;
		}
		else {
			nDState.left = this.right;
			nDState.leftSpan = this.rightSpan;
			nDState.leftRuleName = this.ruleName;
			nDState.right = lexicalResult.getResultCategory();
			nDState.rightSpan = span;
			nDState.nextLeft = this;
		}
		
		assert this.left != null || this.right == null;
		//TODO this returns the unmodified lexical entry and might result in a bug where lexical entries are used 
		nDState.lexicalEntry = lexicalResult.getEntry(); 
		nDState.ruleName = ShiftReduceLexicalStep.LEXICAL_DERIVATION_STEP_RULENAME;
		
		nDState.initializeRuleNameSet();
		assert nDState.ruleName != null;
		
		nDState.numTree = this.numTree + 1;
		
		return nDState;
	}


	public DerivationState<MR> shift(LexicalEntry<MR> lexicalEntry, int words, SentenceSpan span) {
		DerivationState<MR> nDState = new DerivationState<MR>();
		nDState.parent = this;
		nDState.wordsConsumed = this.wordsConsumed + words; 
		nDState.lenRoot = this.lenRoot + 1;
		
		if(this.left == null) {
			nDState.left = lexicalEntry.getCategory();
			nDState.leftSpan = span;
			//next line is a bug fix. 
			nDState.leftRuleName = ShiftReduceLexicalStep.LEXICAL_DERIVATION_STEP_RULENAME;
		}
		else if(this.right == null) {
			nDState.left = this.left;
			nDState.leftSpan = this.leftSpan;
			nDState.leftRuleName = this.ruleName;
			nDState.right = lexicalEntry.getCategory();
			nDState.rightSpan = span;
		}
		else {
			nDState.left = this.right;
			nDState.leftSpan = this.rightSpan;
			nDState.leftRuleName = this.ruleName;
			nDState.right = lexicalEntry.getCategory();
			nDState.rightSpan = span;
			nDState.nextLeft = this;
		}
		
		assert this.left != null || this.right == null;
		nDState.lexicalEntry = lexicalEntry;
		nDState.ruleName = ShiftReduceLexicalStep.LEXICAL_DERIVATION_STEP_RULENAME;
		
		nDState.initializeRuleNameSet();
		assert nDState.ruleName != null;
		
		nDState.numTree = this.numTree + 1;
		
		return nDState;
	}
	
	public DerivationState<MR> reduceUnaryRule(RuleName ruleName, Category<MR> newCategory) {
		DerivationState<MR> nDState = new DerivationState<MR>();
		nDState.parent = this;
		nDState.wordsConsumed = this.wordsConsumed; 
		nDState.lenRoot = this.lenRoot;
		
		assert this.left != null;
		
		if(this.right == null) {
			nDState.left = newCategory;
			nDState.leftSpan = this.leftSpan;
			//next line is a bug fix
			nDState.leftRuleName = ruleName;//this.leftRuleName;
		}
		else {
			nDState.left = this.left;
			nDState.leftSpan = this.leftSpan;
			nDState.leftRuleName = this.leftRuleName;
			nDState.right = newCategory;
			nDState.rightSpan = this.rightSpan;
		}
		
		nDState.ruleName = ruleName;
		nDState.nextLeft = this.nextLeft;
		assert this.left != null || this.right == null;
		nDState.initializeRuleNameSet();
		assert nDState.ruleName != null;
		
		nDState.numTree = this.numTree;
		
		return nDState;
	}
	
	public Pair<Category<MR>, SentenceSpan> findLeftMostState() {
		if(this.nextLeft == null)
			return null;
		return Pair.of(this.nextLeft.left, this.nextLeft.leftSpan);
	}
	
	public DerivationState<MR> reduceBinaryRule(RuleName ruleName, Category<MR> newCategory, 
												SentenceSpan joined) {
		DerivationState<MR> nDState = new DerivationState<MR>();
		nDState.parent = this;
		nDState.wordsConsumed = this.wordsConsumed; 
		nDState.lenRoot = this.lenRoot - 1;
		nDState.isBinary = true;
		
		assert this.left != null && this.right != null;
		
		Pair<Category<MR>, SentenceSpan> left_ = this.findLeftMostState();
		
		if(left_ == null) {
			nDState.left = newCategory;
			nDState.leftSpan = joined;
			nDState.nextLeft = null;
			//next line a bug fix
			nDState.leftRuleName = ruleName; // --- null;//ruleName;
		}
		else {
			nDState.left = left_.first(); 
			nDState.leftSpan = left_.second();
			nDState.right = newCategory;
			nDState.rightSpan = joined;
			nDState.nextLeft = this.nextLeft.nextLeft;
			nDState.leftRuleName = this.nextLeft.leftRuleName;
		}
		
		nDState.ruleName = ruleName;
		
		assert this.left != null || this.right == null;	
		//System.err.println("Binary rule is "+ruleName);
		nDState.initializeRuleNameSet();
		assert nDState.ruleName != null;
		
		nDState.numTree = this.numTree - 1;
		
		return nDState;
	}

	public void print(ILogger log) {
		log.debug("Derivation State with score: %s and %s tree segments"
				  +"; categories are left: %s, right: %s", this.score, this.lenRoot(),
				  this.left != null ? this.left : "", this.right != null ? this.right : "");
	}
	
	private void calcHashCode() {
		/* hashcode should be same for root equivalent system
		 * if a derivation state has child [left1, right1] and if its nextLeft
		 * has children [left2, right2] then left1 == right2 always. */
		
//		DerivationState<MR> iter = this;
		int prime = 31;
		this.hashCode = 1; //was 0 earlier
		
		///////////////////
//		this.hashCode = this.hashCode * prime + this.wordsConsumed;
//		this.hashCode = this.hashCode * prime + this.lenRoot;
		
		if(this.right != null)
			this.hashCode = this.hashCode * prime + this.right.hashCode();
		
		if(this.left != null)
			this.hashCode = this.hashCode * prime + this.left.hashCode();

		if(this.nextLeft != null)
			this.hashCode = this.hashCode * prime + this.nextLeft.hashCode();
//		if(this.parent != null)
//			this.hashCode = this.hashCode * prime + this.parent.hashCode();
		///////////////////
		
//		while(iter != null) {
//			if(iter.right != null)
//				this.hashCode = this.hashCode * prime + iter.right.hashCode();
//			
//			if(iter.nextLeft == null && iter.left != null)
//				this.hashCode = this.hashCode * prime + iter.left.hashCode();
//			
//			iter = iter.nextLeft;
//		}
	}
	
	public void calcDebugHashCode() {
		
		if(LOG.getLogLevel() != LogLevel.DEBUG)
			return;
		
		int prime = 31;
		this.debugHashCode = 1;
		this.debugHashCode = prime * this.debugHashCode + this.wordsConsumed;
		
		if(this.parent != null) {
			if(this.parent.debugHashCode == -1)
				this.parent.calcDebugHashCode();
			
			this.debugHashCode = prime * this.debugHashCode + this.parent.getDebugHashCode(); //this can be an expensive call
		}
		
		if(this.left != null)
			this.debugHashCode = prime * this.debugHashCode + this.left.hashCode();
		
		if(this.right != null)
			this.debugHashCode = prime * this.debugHashCode + this.right.hashCode();
		
		if(this.step != null)
			this.debugHashCode = prime * this.debugHashCode + this.step.hashCode();
		
		long scoreBits = Double.doubleToLongBits(this.score);
		this.debugHashCode = prime * this.debugHashCode + (int) (scoreBits ^ (scoreBits >>> 32));
		
		this.debugHashCode = prime * this.debugHashCode + this.avgFeatures.hashCode();
	}
	
	public int hashCode() {
		if(this.hashCode == -1)
			this.calcHashCode();
		
		return this.hashCode;
	}
	
	@Override
	public boolean equals(Object obj) {
		if(obj.getClass()!= this.getClass())
			return false;
		
		@SuppressWarnings("unchecked")
		DerivationState<MR> dstate = (DerivationState<MR>)obj;
		
		if(dstate.hashCode() != this.hashCode()) 
			return false;
		
		return this.rootsEqual(dstate);
	}
	
	/** To-Do
	 * Try to remove or reduce the access levels of functions below as much as possible. */
	public DerivationState<MR> getParent() {
		return this.parent;
	}
		
	public int getDebugHashCode() {
		return this.debugHashCode;
	}
	
	public IHashVector getFeatures() {
		return this.avgFeatures;
	}
	
	public List<ParsingOp<MR>> possibleActions() {
		return this.possibleActions;
	}
	
	public List<IHashVector> possibleActionFeatures() {
		return this.possibleActionsFeatures;
	}
	
	public IHashVector stateFeature() {
		return this.stateFeature;
	}
	
	public void setStatePersistentEmbedding(PersistentEmbeddings stateEmbedding) {
		this.stateEmbedding = stateEmbedding;
	}
	
	public void setParsingOpPersistentEmbedding(PersistentEmbeddings parsingOpEmbedding) {
		this.parsingOpEmbedding = parsingOpEmbedding;
	}
	
	public PersistentEmbeddings getStatePersistentEmbedding() {
		return this.stateEmbedding;
	}
	
	public PersistentEmbeddings getParsingOpPersistentEmbedding() {
		return this.parsingOpEmbedding;
	}
	
	public void setPossibleActions(List<ParsingOp<MR>> possibleActions) {
		if(this.possibleActions != null) {
			throw new RuntimeException("Can only set it once");
		}
		this.possibleActions = possibleActions;
	}
	
	public void setPossibleActionsFeatures(List<IHashVector> possibleActionsFeatures) {
		if(this.possibleActionsFeatures != null) {
			throw new RuntimeException("Can only set it once");
		}
		this.possibleActionsFeatures = possibleActionsFeatures;
	}
	
	public void setStateFeature(IHashVector stateFeature) {
		if(this.stateFeature != null) {
			throw new RuntimeException("Can only set it once");
		}
		this.stateFeature = stateFeature;
	}
	
	public void setEncoding(INDArray encoding) {
		this.encoding = encoding;
	}
	
	public INDArray getEncoding() {
		return this.encoding;
	}
	
	/** two derivation states are root-equal if they have consumed 
	 *  the same number of words and have the same number and categories of 
	 *  roots of tree segments. A derivation state can be replaced by its root-equivalent
	 *  without modifying the parse tree. Root-Equivalance is an equivalance relation. */
	private boolean rootsEqual(DerivationState<MR> dstate) {
		
		if(this.wordsConsumed != dstate.wordsConsumed || this.lenRoot != dstate.lenRoot)
			return false;
		
		// Short circuit
		// if both state were created using lexical step then simply check the last category
		// and if they have the same parent (object identity not equality).
		if(this.returnStep().getRuleName() == ShiftReduceLexicalStep.LEXICAL_DERIVATION_STEP_RULENAME && 
		   dstate.returnStep().getRuleName() == ShiftReduceLexicalStep.LEXICAL_DERIVATION_STEP_RULENAME) {
			
			if(this.parent.packedState == null || dstate.parent.packedState == null) {
				throw new RuntimeException("Packed state cannot be null");
			}
			
			if(this.parent.packedState == dstate.parent.packedState) {
				if(this.returnStep().getRoot().equals(dstate.returnStep().getRoot())) {
					return true;
				}
			}
			
			return false;
		}
		
		if(this.parent.packedState == dstate.parent.packedState) {
			if(this.returnStep().getRoot().equals(dstate.returnStep().getRoot())) {
				return true;
			} else {
				return false;
			}
		}
		
		
		DerivationState<MR> iter1 = this;
		DerivationState<MR> iter2 = dstate;
		
		while(iter1 != null) {
			/* this code is not optimized. one entry is being used multiple times */
			if(iter2 == null)
				return false;
			
			if(iter1.hashCode() != iter2.hashCode()) {
				return false;
			}
			
			if(iter1.right == null && iter2.right != null || iter1.right != null && iter2.right == null)
				return false;
			
			if(iter1.nextLeft == null && iter2.nextLeft != null || iter1.nextLeft != null && iter2.nextLeft == null) 
				return false;
			
			if(iter1.left == null && iter2.left != null || iter1.left != null && iter2.left == null)
				return false;
			
			if(iter1.right != null && iter2.right != null && !iter1.right.equals(iter2.right))
					return false;
			
			if(iter1.nextLeft == null) {
				if(iter1.left != null && iter2.left != null && !iter1.left.equals(iter2.left))
					return false;
			}
			
			iter1 = iter1.nextLeft;
			iter2 = iter2.nextLeft;
		}
		
		if(iter2 != null)
			return false;
		
		return true;
	}
	
	public void computeFeatures() {
		DerivationState<MR> iter = this;
		
		while(iter != null) {
			if(iter.step != null && iter.step.getStepFeatures() != null) {
				//null features are used by models with no features such as Neural Network model
				this.avgFeatures = this.avgFeatures.addTimes(1, iter.step.getStepFeatures());
			}
			//this repeats computation, can do it in an efficient fashion
			iter = iter.parent;
		}
	}

}