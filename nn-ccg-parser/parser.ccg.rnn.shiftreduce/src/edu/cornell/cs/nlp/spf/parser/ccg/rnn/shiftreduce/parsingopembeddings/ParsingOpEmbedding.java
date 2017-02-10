package edu.cornell.cs.nlp.spf.parser.ccg.rnn.shiftreduce.parsingopembeddings;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.ObjectInput;
import java.io.ObjectInputStream;
import java.io.ObjectOutput;
import java.io.ObjectOutputStream;
import java.io.OutputStream;
import java.io.PrintWriter;
import java.io.UnsupportedEncodingException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.ListIterator;
import java.util.Map;
import java.util.Map.Entry;
import java.util.stream.IntStream;
import java.util.stream.StreamSupport;
import java.util.Set;
import java.util.Spliterator;
import java.util.Spliterators;

import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.atomic.AtomicBoolean;

import org.json.JSONObject;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.Sqrt;
import org.nd4j.linalg.api.ops.impl.transforms.Tanh;
import org.nd4j.linalg.api.ops.impl.transforms.TanhDerivative;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

import com.google.common.base.Joiner;

import edu.cornell.cs.nlp.spf.ccg.lexicon.ILexiconImmutable;
import edu.cornell.cs.nlp.spf.ccg.lexicon.LexicalEntry;
import edu.cornell.cs.nlp.spf.ccg.lexicon.factored.lambda.FactoredLexicalEntry;
import edu.cornell.cs.nlp.spf.ccg.lexicon.factored.lambda.FactoringServices;
import edu.cornell.cs.nlp.spf.ccg.lexicon.factored.lambda.Lexeme;
import edu.cornell.cs.nlp.spf.ccg.lexicon.factored.lambda.LexicalTemplate;
import edu.cornell.cs.nlp.spf.mr.lambda.LogicalExpression;
import edu.cornell.cs.nlp.spf.parser.ParsingOp;
import edu.cornell.cs.nlp.spf.parser.ccg.rnn.shiftreduce.categoryembeddings.CategoryEmbedding;
import edu.cornell.cs.nlp.spf.parser.ccg.rnn.shiftreduce.categoryembeddings.CategoryEmbeddingResult;
import edu.cornell.cs.nlp.spf.parser.ccg.rnn.shiftreduce.dataset.CompositeDataPoint;
import edu.cornell.cs.nlp.spf.parser.ccg.rnn.shiftreduce.dataset.CompositeDataPointDecision;
import edu.cornell.cs.nlp.spf.parser.ccg.rnn.shiftreduce.embeddings.AbstractEmbedding;
import edu.cornell.cs.nlp.spf.parser.ccg.rnn.shiftreduce.embeddings.EmbedWordBuffer;
import edu.cornell.cs.nlp.spf.parser.ccg.rnn.shiftreduce.learning.Helper;
import edu.cornell.cs.nlp.spf.parser.ccg.rnn.shiftreduce.learning.LearningRateStats;
import edu.cornell.cs.nlp.spf.parser.ccg.rnn.shiftreduce.recursive.Tree;
import edu.cornell.cs.nlp.spf.parser.ccg.rules.RuleName;
import edu.cornell.cs.nlp.spf.parser.ccg.shiftreduce.stacks.LexicalParsingOp;
import edu.cornell.cs.nlp.spf.parser.ccg.shiftreduce.steps.ShiftReduceLexicalStep;
import edu.cornell.cs.nlp.utils.log.ILogger;
import edu.cornell.cs.nlp.utils.log.LoggerFactory;


/** This class embeds a parsing action
 *  @author Dipendra Misra
 *  */
public class ParsingOpEmbedding<MR> implements AbstractEmbedding {
	
	public static final ILogger	LOG = LoggerFactory.create(ParsingOpEmbedding.class);

	/** object for embedding categories which are the result of parsing step */
	private final CategoryEmbedding<MR> categEmbedding;
	
	/** object for embedding words in lexical step */
	private final EmbedWordBuffer embedWordBuffer;
	final private List<RuleName> ruleNames;
	private final int numRules;
	private final int dim;
	
	private final double learningRate;
	private final double regularizer;
	
	/** Embedding of action i.e. rule name
	 *  A prior ordering over the rule based on ruleIndex
	 *  is assumed. The datastructures use the same ordering.
	 */
	private final int actionDim;
	private final List<INDArray> actionEmbedding;
	private final List<INDArray> gradActionEmbedding;
	private final List<INDArray> adaGradSumSquareGradientAction;
	
	/** Embedding of lexical entries */
	private final int lexicalEntryDim;
	private Map<LexicalEntry<MR>, INDArray> lexicalEntryEmbedding;
	private final Map<LexicalEntry<MR>, INDArray> gradLexicalEntryEmbedding;
	private final Map<LexicalEntry<MR>, INDArray> adaGradSumSquareGradientLexicalEntry;
	private final Set<LexicalEntry<MR>> modifiedLexicalEntries; 
	
	/** If a lexical entry is generated dynamically then it may not have its embedding
	 * in the list of lexical entry. In this case, we embed the lexical entry using origin of the
	 * dynamic entries. The key in the next set of datastructures represent the origin. */
	private Map<String, INDArray> dynamicOriginEmbedding;
	private final Map<String, INDArray> gradDynamicOriginEmbedding;
	private final Map<String, INDArray> adaGradSumSquareGradientDynamicOrigin;
	private final Collection<String> modifiedDynamicOrigin;
	
	/** Embedding of lexemes */
	private final int lexemeDim;
	private Map<Lexeme, INDArray> lexemeEmbedding;
	private final Map<Lexeme, INDArray> gradLexemeEmbedding;
	private final Map<Lexeme, INDArray> adaGradSumSquareGradientLexeme;
	
	/** Embedding of template */
	private final int templateDim;
	private Map<LexicalTemplate, INDArray> templateEmbedding;
	private final Map<LexicalTemplate, INDArray> gradTemplateEmbedding;
	private final Map<LexicalTemplate, INDArray> adaGradSumSquareGradientTemplate;
	
	/** padding used by lexical embeddings in Reduce operations instead of 0s */
	private final int lexicalEmbeddingDim;
	private final boolean usedPaddedVector;
	private INDArray paddedLexicalEmbedding;
	private final INDArray paddedLexicalEmbeddingGrad;
	private final INDArray adaGradSumSquarePaddedLexicalEmbedding;
	
	/** One layer MLP for squashing the lexical entries */
	private final boolean doSquashing;
	private final INDArray W, b;
	private final INDArray gradW, gradb, adaGradSumSquareGradW, adaGradSumSquareGradb;
	
	/** Statistics on learning rate for different embeddings*/
	private final LearningRateStats learningRateStatsActionRule;
	private final LearningRateStats learningRateStatsLexeme;
	private final LearningRateStats learningRateStatsTemplate;
	private final LearningRateStats learningRateStatsPaddedVector;
	
	/** If true then use lexical entry embedding else use lexeme and template embedding */
	private final boolean useLexicalEntry;
	
	/** Project Matrix to shared space*/
	private final boolean useSharedSpace;
	private final INDArray Wshift, Wreduce;
	private final INDArray gradWshift, gradWreduce;
	private final INDArray adaGradSumSquareWshift, adaGradSumSquareWreduce;
	
	/** Store the parameters that have been used, so that only these parameters are updated in 
	 * that iteration. Make sure these collection are thread-safe and are cleared after update. */
	private final Set<Integer> updatedActionRule;
	private final Set<LexicalEntry<MR>> updatedLexicalEntry;
	private final Set<Lexeme> updatedLexeme;
	private final Set<LexicalTemplate> updatedTemplate;
	private final AtomicBoolean updatedPaddedVector;
	
	/** Variables used for gradient checks*/
	public double empiricalAction, empiricalTemplate;
	private LexicalTemplate template;
	private final boolean doGradientChecks;
	
	/** Mean activation and gradient for debugging */
	private final Map<String, Double> meanActivations;
	private final Map<String, Double> meanGradients;
	
	public INDArray getActionVector() {
		return this.actionEmbedding.get(0);
	}
	
	public INDArray getTemplateVector() {
		this.template = this.templateEmbedding.entrySet().iterator().next().getKey();
		return this.templateEmbedding.get(this.template);		
	}
	/////////////////
	
	public ParsingOpEmbedding(CategoryEmbedding<MR> categEmbedding,
			EmbedWordBuffer embedWordBuffer, 
			List<RuleName> ruleNames, double learningRate, double l2) {

		this.categEmbedding = categEmbedding;
		this.embedWordBuffer = embedWordBuffer;
		this.ruleNames = ruleNames;
		this.numRules = ruleNames.size();
		
		this.actionDim = 10;//20;//10; 3;this.numRules;
		this.learningRate = learningRate;
		this.regularizer = l2;
		
		this.learningRateStatsActionRule = new LearningRateStats();
		this.learningRateStatsLexeme = new LearningRateStats();
		this.learningRateStatsTemplate = new LearningRateStats();
		this.learningRateStatsPaddedVector = new LearningRateStats();
		
		this.meanActivations = new HashMap<String, Double>();
		this.meanGradients = new HashMap<String, Double>();
		
		this.actionEmbedding = new ArrayList<INDArray>();
		this.gradActionEmbedding = new ArrayList<INDArray>();
		this.adaGradSumSquareGradientAction = new ArrayList<INDArray>();
		
		double epsilon = 2*Math.sqrt(6/(double)(this.actionDim + 1));
		
		for(int i = 0; i < this.numRules; i++) {
			INDArray vec = Nd4j.rand(new int[]{1, this.actionDim});
			vec.subi(0.5).muli(epsilon);
			this.actionEmbedding.add(vec);
			
			INDArray grad = Nd4j.zeros(this.actionDim);
			this.gradActionEmbedding.add(grad);
			this.adaGradSumSquareGradientAction.add(Nd4j.zeros(this.actionDim).addi(0.00001));
		}
		
		this.useLexicalEntry = false;
		
		this.lexicalEntryDim = 20;//40; 20; 10;
		this.lexicalEntryEmbedding = new HashMap<LexicalEntry<MR>, INDArray>();
		this.gradLexicalEntryEmbedding = new HashMap<LexicalEntry<MR>, INDArray>();
		this.adaGradSumSquareGradientLexicalEntry = new HashMap<LexicalEntry<MR>, INDArray>();
		this.modifiedLexicalEntries = new HashSet<LexicalEntry<MR>>();
		
		this.dynamicOriginEmbedding = new HashMap<String, INDArray>();
		this.gradDynamicOriginEmbedding = new HashMap<String, INDArray>();
		this.adaGradSumSquareGradientDynamicOrigin = new HashMap<String, INDArray>();
		this.modifiedDynamicOrigin = new ConcurrentLinkedQueue<String>();
		
		this.lexemeDim = 25;//15;
		this.lexemeEmbedding = new HashMap<Lexeme, INDArray>();
		this.gradLexemeEmbedding = new HashMap<Lexeme, INDArray>();
		this.adaGradSumSquareGradientLexeme = new HashMap<Lexeme, INDArray>();
		
		this.templateDim = 10;//5;
		this.templateEmbedding = new HashMap<LexicalTemplate, INDArray>();
		this.gradTemplateEmbedding = new HashMap<LexicalTemplate, INDArray>();
		this.adaGradSumSquareGradientTemplate = new HashMap<LexicalTemplate, INDArray>();
		
		if(this.useLexicalEntry) {
			//Add null entry into the lexical entry embedding that is used when the actual lexical entry is not present
			double epsilonLexicalEntry = 2*Math.sqrt(6/(double)(this.lexicalEntryDim + 1));
			INDArray vec = Nd4j.rand(new int[]{1, this.lexicalEntryDim});
			vec.subi(0.5).muli(epsilonLexicalEntry);
			this.lexicalEntryEmbedding.put(null, vec);
			
			INDArray gradLexicalEntry = Nd4j.zeros(this.lexicalEntryDim);
			this.gradLexicalEntryEmbedding.put(null, gradLexicalEntry);
			this.adaGradSumSquareGradientLexicalEntry.put(null, Nd4j.zeros(this.lexicalEntryDim).addi(0.00001));
			this.lexicalEmbeddingDim = this.lexicalEntryDim;
		} else {
			//Add null entry for lexeme and template
			double epsilonLexeme = 2*Math.sqrt(6/(double)(this.lexemeDim + 1));
			INDArray vecLexeme = Nd4j.rand(new int[]{1, this.lexemeDim});
			vecLexeme.subi(0.5).muli(epsilonLexeme);
			this.lexemeEmbedding.put(null, vecLexeme);
			
			INDArray gradLexeme = Nd4j.zeros(this.lexemeDim);
			this.gradLexemeEmbedding.put(null, gradLexeme);
			this.adaGradSumSquareGradientLexeme.put(null, Nd4j.zeros(this.lexemeDim).addi(0.00001));
			
			double epsilonTemplate = 2*Math.sqrt(6/(double)(this.templateDim + 1));
			INDArray vecTemplate = Nd4j.rand(new int[]{1, this.templateDim});
			vecTemplate.subi(0.5).muli(epsilonTemplate);
			this.templateEmbedding.put(null, vecTemplate);
			
			INDArray gradTemplate = Nd4j.zeros(this.templateDim);
			this.gradTemplateEmbedding.put(null, gradTemplate);
			this.adaGradSumSquareGradientTemplate.put(null, Nd4j.zeros(this.templateDim).addi(0.00001));
			this.lexicalEmbeddingDim = this.lexemeDim + this.templateDim;//  + this.embedWordBuffer.tunableWordDim();
		}
		
		this.updatedActionRule = Collections.synchronizedSet(new HashSet<Integer>());
		this.updatedLexicalEntry = Collections.synchronizedSet(new HashSet<LexicalEntry<MR>>());
		this.updatedLexeme = Collections.synchronizedSet(new HashSet<Lexeme>());
		this.updatedTemplate = Collections.synchronizedSet(new HashSet<LexicalTemplate>());
		this.updatedPaddedVector = new AtomicBoolean(false);
		
		this.useSharedSpace = false;
		this.doSquashing = false;
		
		if(this.useSharedSpace) {
			this.dim = 80; //dimension of the common projected space
		} else {
			if(!this.doSquashing) {
				this.dim = this.actionDim + categEmbedding.getDimension() + this.lexicalEmbeddingDim;
			} else {
				this.dim = 50; //squashing dimension
			}
		}
		
		double epsilonLexicalEmbedding = 2*Math.sqrt(6/(double)(this.lexicalEmbeddingDim + 1));
		this.usedPaddedVector = true;
		this.paddedLexicalEmbedding = Nd4j.rand(new int[]{1, this.lexicalEmbeddingDim}).subi(0.5).muli(epsilonLexicalEmbedding);
		this.paddedLexicalEmbeddingGrad = Nd4j.zeros(this.lexicalEmbeddingDim);
		this.adaGradSumSquarePaddedLexicalEmbedding = Nd4j.zeros(this.lexicalEmbeddingDim).addi(0.00001);
		
		final int col = this.actionDim + categEmbedding.getDimension() + this.lexicalEmbeddingDim;
		
		////////////////////////////////////
		INDArray Wshift = Nd4j.rand(new int[]{this.dim, categEmbedding.getDimension() + this.lexicalEmbeddingDim});
		double epsilonWShift = 2*Math.sqrt(6/(double)(Wshift.size(0) + Wshift.size(1)));
		Wshift.subi(0.5).muli(epsilonWShift);
		this.Wshift = Wshift;
		this.gradWshift = Nd4j.zeros(Wshift.shape());
		this.adaGradSumSquareWshift = Nd4j.zeros(Wshift.shape()).addi(0.00001);
		
		INDArray Wreduce = Nd4j.rand(new int[]{this.dim, categEmbedding.getDimension() + this.actionDim});
		double epsilonWreduce = 2*Math.sqrt(6/(double)(Wreduce.size(0) + Wreduce.size(1)));
		Wreduce.subi(0.5).muli(epsilonWreduce);
		this.Wreduce = Wreduce;
		this.gradWreduce = Nd4j.zeros(Wreduce.shape());
		this.adaGradSumSquareWreduce = Nd4j.zeros(Wreduce.shape()).addi(0.00001);
		////////////////////////////////////
		
		//Initialize Squashing parameters
		if(this.doSquashing) {
			double epsilonW = 2*Math.sqrt(6.0/(double)(this.dim + col));
			this.W = Nd4j.rand(new int[]{this.dim, col}); 
			this.W.subi(0.5).muli(epsilonW);
			
			double epsilonb = 2*Math.sqrt(6.0/(double)(this.dim + 1));
			this.b = Nd4j.rand(new int[]{this.dim, 1}); 
			this.b.subi(0.5).muli(epsilonb);
			
			this.gradW = Nd4j.zeros(this.dim, col);
			this.gradb = Nd4j.zeros(this.dim, 1);
			
			this.adaGradSumSquareGradW = Nd4j.zeros(this.dim, col).addi(0.00001); 
			this.adaGradSumSquareGradb = Nd4j.zeros(this.dim, 1).addi(0.00001);
		}  else {
			this.W = null;
			this.b = null;
			this.gradW = null;
			this.gradb = null;
			this.adaGradSumSquareGradW = null;
			this.adaGradSumSquareGradb = null;
		}
		
		this.doGradientChecks = false;
		
		LOG.info("ParsingOpEmbedding. Dimensionality %s [action %s, category %s, lexical-embedding %s],", 
				this.dim, this.actionDim, this.categEmbedding.getDimension(), this.lexicalEmbeddingDim);
		if(!this.useLexicalEntry) {
			LOG.info(".. Lexeme Dimension %s, Template Dimension %s", this.lexemeDim, this.templateDim);
		}
		LOG.info(".. Using LexicalEntry %s, Using Squashing %s, Using padded vector %s,  Using shared space %s, "
				+ " Using word embedding %s", this.useLexicalEntry,  this.doSquashing, this.usedPaddedVector,
				this.useSharedSpace, this.embedWordBuffer.tunableWordDim());
	}
	
	/** Logs the action embedding in a json file in the given folder */
	public void logEmbedding(String folderName) throws FileNotFoundException,
													  	     UnsupportedEncodingException {
		
		//log action embedding
		PrintWriter writer = new PrintWriter(folderName+"/parsing_action.json", "UTF-8");
		
		writer.write("{\"num_rules\": \"" + this.numRules + "\", \n");
		writer.write("\"padded_vector\": \"" + Helper.printFullVector(this.paddedLexicalEmbedding) + "\", \n");
		writer.write("\"W_shift\": [" + Helper.printFullMatrix(this.Wshift) + "], \n");
		writer.write("\"W_reduce\": [" + Helper.printFullMatrix(this.Wreduce) + "], \n");
		
		ListIterator<INDArray> it = this.actionEmbedding.listIterator();
		
		while(it.hasNext()) {
			writer.write("\"rules_" + it.nextIndex() + "\": \"" + Helper.printFullVector(it.next())
						+ "\", \n");
		}
		
		writer.write("}");
		
		writer.flush();
		writer.close();
		
		//store lexical entry embeddings
		if(this.useLexicalEntry) {
			try (
				OutputStream file = new FileOutputStream(folderName + "/lexical_entries.ser");
				OutputStream buffer = new BufferedOutputStream(file);
				ObjectOutput output = new ObjectOutputStream(buffer);
			) {
				output.writeObject(this.lexicalEntryEmbedding);
			} catch(IOException ex) {
				throw new RuntimeException("Cannot store serializable data");
			}
			
			//save modified lexical entries
			PrintWriter modifiedLexicalEntriesWriter = new PrintWriter(folderName+"/modified_lexical_entries.txt", "UTF-8");
			for(LexicalEntry<MR> entry: this.modifiedLexicalEntries) {
				if(entry == null) {
					modifiedLexicalEntriesWriter.write("null\n Lexeme: null, Template: null");
				} else {
					@SuppressWarnings("unchecked")
					FactoredLexicalEntry factoring = FactoringServices.factor((LexicalEntry<LogicalExpression>)entry);
					modifiedLexicalEntriesWriter.write(entry+"\n Lexeme: "+factoring.getLexeme()+", Template: "+factoring.getTemplate());
				}
			}
			modifiedLexicalEntriesWriter.flush();
			modifiedLexicalEntriesWriter.close();
			
			//Store origin embedding 
			PrintWriter originWriter = new PrintWriter(folderName+"/dynamic_origin.json", "UTF-8");
			
			originWriter.write("{\n");
			for(Entry<String, INDArray> e: this.dynamicOriginEmbedding.entrySet()) {
				originWriter.write("\"" + e.getKey() + "\": \"" + Helper.printFullVector(e.getValue())
							+ "\", \n");
			}
			
			originWriter.write("}");
			
			originWriter.flush();
			originWriter.close();
			
			//save modified origin
			PrintWriter modifiedOriginWriter = new PrintWriter(folderName+"/modified_origin.txt", "UTF-8");
			modifiedOriginWriter.write("Modified " + this.modifiedDynamicOrigin.size() + " out of " +
										this.dynamicOriginEmbedding.size()+"\n");
			for(String origin: this.modifiedDynamicOrigin) {
				modifiedOriginWriter.write(origin+"\n");
			}
			modifiedOriginWriter.flush();
			modifiedOriginWriter.close();
			
		} else {
			try (
					OutputStream file = new FileOutputStream(folderName + "/lexeme.ser");
					OutputStream buffer = new BufferedOutputStream(file);
					ObjectOutput output = new ObjectOutputStream(buffer);
			) {
				output.writeObject(this.lexemeEmbedding);
			} catch(IOException ex) {
				throw new RuntimeException("Cannot store serializable data");
			}
			
			try (
				OutputStream file = new FileOutputStream(folderName + "/template.ser");
				OutputStream buffer = new BufferedOutputStream(file);
				ObjectOutput output = new ObjectOutputStream(buffer);
			) {
				output.writeObject(this.templateEmbedding);
			} catch(IOException ex) {
				throw new RuntimeException("Cannot store serializable data");
			}
		}
	} 
	
	/** Bootstraps the action embedding from the folder */
	@SuppressWarnings("unchecked")
	public void bootstrapEmbedding(String folderName) {
		
		//bootstrap action embedding
		Path actionJsonPath = Paths.get(folderName + "/parsing_action.json");
		String actionJsonString;
		
		try {
			actionJsonString = Joiner.on("\r\n").join(Files.readAllLines(actionJsonPath));
		} catch (IOException e) {
			throw new RuntimeException("Could not read from parsing_action.json. Error: "+e);
		}
		
		JSONObject parsingActionObj = new JSONObject(actionJsonString);
		
		int numRules =  Integer.parseInt(parsingActionObj.getString("num_rules"));
		
		if(this.numRules != numRules) {
			throw new RuntimeException("Number of rules don't match. This.numRules = "+this.numRules+" found "+numRules);
		}
		
		String paddedVecString = parsingActionObj.getString("padded_vector");
		INDArray newVec = Helper.toVector(paddedVecString);
		Nd4j.copy(newVec, this.paddedLexicalEmbedding);
		
		for(int i = 0; i < numRules; i++) {
			String indarrayString = parsingActionObj.getString("rules_"+i);
			INDArray vec = Helper.toVector(indarrayString);
			INDArray current = this.actionEmbedding.get(i);
			
			if(vec.size(0) != current.size(0) && vec.size(1) != current.size(1)) {
				throw new RuntimeException("Sizes dont match while bootstrapping action embeddings.");
			}
			
			for(int j = 0; j < vec.size(0); j++) {
				for(int k = 0; k < vec.size(1); k++) {
					current.putScalar(new int[]{j,  k}, vec.getDouble(new int[]{j, k}));
				}
			}
		}
		
		if(this.useLexicalEntry) {
			try(
			     InputStream file = new FileInputStream(folderName + "/lexical_entries.ser");
			     InputStream buffer = new BufferedInputStream(file);
			     ObjectInput input = new ObjectInputStream (buffer);
			) {
				 this.lexicalEntryEmbedding = (HashMap<LexicalEntry<MR>, INDArray>)input.readObject();
			} catch(Exception e) {
				throw new RuntimeException("Could not deserialize lexical entries. Error: " + e);
		    }
			
			//bootstrap dynamic lexical entry origin
			Path originJsonPath = Paths.get(folderName + "/dynamic_origin.json");
			String originJsonString;
			
			try {
				originJsonString = Joiner.on("\r\n").join(Files.readAllLines(originJsonPath));
			} catch (IOException e) {
				throw new RuntimeException("Could not read from dynamic_origin.json. Error: "+e);
			}
			
			JSONObject originObj = new JSONObject(originJsonString);
			Iterator<?> it = originObj.keys();
			while(it.hasNext()) {
				String key = (String)it.next();
				String indarrayString = originObj.getString(key);
				INDArray vec = Helper.toVector(indarrayString);
				
				if(vec.size(0) != 1 && vec.size(1) != this.lexicalEntryDim) {
					throw new RuntimeException("Sizes dont match while bootstrapping dynamic origin embeddings.");
				}
				
				this.dynamicOriginEmbedding.put(key, vec);
				
				INDArray gradLexicalEntry = Nd4j.zeros(this.lexicalEntryDim);
				this.gradDynamicOriginEmbedding.put(key, gradLexicalEntry);
				this.adaGradSumSquareGradientDynamicOrigin.put(key,
											Nd4j.zeros(this.lexicalEntryDim).addi(0.00001));
			}
		} else {
			try(
			     InputStream file = new FileInputStream(folderName + "/lexeme.ser");
			     InputStream buffer = new BufferedInputStream(file);
			     ObjectInput input = new ObjectInputStream (buffer);
			) {
				 this.lexemeEmbedding = (HashMap<Lexeme, INDArray>)input.readObject();
			} catch(Exception e) {
				throw new RuntimeException("Could not deserialize lexical entries. Error: " + e);
			}
			
			try(
			     InputStream file = new FileInputStream(folderName + "/template.ser");
			     InputStream buffer = new BufferedInputStream(file);
			     ObjectInput input = new ObjectInputStream (buffer);
			) {
				 this.templateEmbedding = (HashMap<LexicalTemplate, INDArray>)input.readObject();
			} catch(Exception e) {
				throw new RuntimeException("Could not deserialize lexical entries. Error: " + e);
		    }
		}
		
		this.lexicalEntryEmbedding = Collections.unmodifiableMap(this.lexicalEntryEmbedding);
	}
	
	/** Induce lexical entry embeddings */
	public void induceLexicalEntryEmbedding(ILexiconImmutable<MR> lexicon) {
		
		Collection<LexicalEntry<MR>> listLexicalEntries = lexicon.toCollection();
		final double epsilonLexicalEntry = 2*Math.sqrt(6/(double)(this.lexicalEntryDim + 1));
		final double epsilonLexeme = 2*Math.sqrt(6/(double)(this.lexemeDim + 1));
		final double epsilonTemplate =  2*Math.sqrt(6/(double)(this.templateDim + 1));
		
		for(LexicalEntry<MR> lexicalEntry: listLexicalEntries) {
			
			if(this.useLexicalEntry) {
				INDArray vec = Nd4j.rand(new int[]{1, this.lexicalEntryDim});
				vec.subi(0.5).muli(epsilonLexicalEntry);
				this.lexicalEntryEmbedding.put(lexicalEntry, vec);
				
				INDArray grad = Nd4j.zeros(this.lexicalEntryDim);
				this.gradLexicalEntryEmbedding.put(lexicalEntry, grad);
				this.adaGradSumSquareGradientLexicalEntry.put(lexicalEntry, Nd4j.zeros(this.lexicalEntryDim).addi(0.00001));
				
				if(!this.dynamicOriginEmbedding.containsKey(lexicalEntry.getOrigin())) {
					INDArray dynamicVec = Nd4j.rand(new int[]{1, this.lexicalEntryDim});
					dynamicVec.subi(0.5).muli(epsilonLexicalEntry);
					this.dynamicOriginEmbedding.put(lexicalEntry.getOrigin(), dynamicVec);
					
					INDArray gradLexicalEntry = Nd4j.zeros(this.lexicalEntryDim);
					this.gradDynamicOriginEmbedding.put(lexicalEntry.getOrigin(), gradLexicalEntry);
					this.adaGradSumSquareGradientDynamicOrigin.put(lexicalEntry.getOrigin(),
												Nd4j.zeros(this.lexicalEntryDim).addi(0.00001));
				}
			} else {
				@SuppressWarnings("unchecked")
				LexicalEntry<LogicalExpression> expLexicalEntry = (LexicalEntry<LogicalExpression>)lexicalEntry;
				FactoredLexicalEntry factored = FactoringServices.factor(expLexicalEntry);
				
				Lexeme lexeme = factored.getLexeme();
				INDArray vecLexeme = Nd4j.rand(new int[]{1, this.lexemeDim});
				vecLexeme.subi(0.5).muli(epsilonLexeme);
				this.lexemeEmbedding.put(lexeme, vecLexeme);
				
				INDArray gradLexeme = Nd4j.zeros(this.lexemeDim);
				this.gradLexemeEmbedding.put(lexeme, gradLexeme);
				this.adaGradSumSquareGradientLexeme.put(lexeme, Nd4j.zeros(this.lexemeDim).addi(0.00001));
				
				LexicalTemplate template = factored.getTemplate();
				INDArray vecTemplate = Nd4j.rand(new int[]{1, this.templateDim});
				vecTemplate.subi(0.5).muli(epsilonTemplate);
				this.templateEmbedding.put(template, vecTemplate);
				
				INDArray gradTemplate = Nd4j.zeros(this.templateDim);
				this.gradTemplateEmbedding.put(template, gradTemplate);
				this.adaGradSumSquareGradientTemplate.put(template, Nd4j.zeros(this.templateDim).addi(0.00001));
			}
			
			//Make the hashmaps unmodifiable
			//The lexical entries are becoming null for some reason in multi-threading environment
			//even though they are populated once before doing get. We need to understand this but for now
			//we use unmodifable maps to get around.
			//this.lexicalEntryEmbedding = Collections.unmodifiableMap(this.lexicalEntryEmbedding);
		}
		
		if(this.useLexicalEntry) {
			LOG.info("Number of lexical entries %s from lexicon", this.lexicalEntryEmbedding.size());
		} else {
			LOG.info("Number of lexeme %s from lexicon", this.lexemeEmbedding.size());
			LOG.info("Number of template %s from lexicon", this.templateEmbedding.size());	
		}
	}
	
	/** Induce origin of dynamic lexical entry as well as template from processed dataset. */
	public void induceDynamicOriginAndTemplate(List<CompositeDataPoint<MR>> dataPoint) {
		
		final double epsilonLexicalEntry = 2*Math.sqrt(6/(double)(this.lexicalEntryDim + 1));
		final double epsilonLexeme = 2*Math.sqrt(6/(double)(this.lexemeDim + 1));
		final double epsilonTemplate =  2*Math.sqrt(6/(double)(this.templateDim + 1));
	
		for(CompositeDataPoint<MR> point: dataPoint) {
			List<CompositeDataPointDecision<MR>> decisions = point.getDecisions();
			for(CompositeDataPointDecision<MR> decision: decisions) {
				List<ParsingOp<MR>> actions = decision.getPossibleActions();
				for(ParsingOp<MR> action: actions) {
					if(action instanceof LexicalParsingOp) {
						LexicalEntry<MR> lexicalEntry = ((LexicalParsingOp<MR>)action).getEntry();
						
						if(!this.dynamicOriginEmbedding.containsKey(lexicalEntry.getOrigin())) {
							INDArray dynamicVec = Nd4j.rand(new int[]{1, this.lexicalEntryDim});
							dynamicVec.subi(0.5).muli(epsilonLexicalEntry);
							this.dynamicOriginEmbedding.put(lexicalEntry.getOrigin(), dynamicVec);
							
							INDArray gradLexicalEntry = Nd4j.zeros(this.lexicalEntryDim);
							this.gradDynamicOriginEmbedding.put(lexicalEntry.getOrigin(), gradLexicalEntry);
							this.adaGradSumSquareGradientDynamicOrigin.put(lexicalEntry.getOrigin(),
														Nd4j.zeros(this.lexicalEntryDim).addi(0.00001));
						}
						
						@SuppressWarnings("unchecked")
						LexicalEntry<LogicalExpression> expLexicalEntry = (LexicalEntry<LogicalExpression>)lexicalEntry;
						FactoredLexicalEntry factored = FactoringServices.factor(expLexicalEntry);
						
						Lexeme lexeme = factored.getLexeme();
						if(!this.lexemeEmbedding.containsKey(lexeme)) {
							
							INDArray vecLexeme = Nd4j.rand(new int[]{1, this.lexemeDim});
							vecLexeme.subi(0.5).muli(epsilonLexeme);
							this.lexemeEmbedding.put(lexeme, vecLexeme);
							
							INDArray gradLexeme = Nd4j.zeros(this.lexemeDim);
							this.gradLexemeEmbedding.put(lexeme, gradLexeme);
							this.adaGradSumSquareGradientLexeme.put(lexeme, Nd4j.zeros(this.lexemeDim).addi(0.00001));
						}
						
						LexicalTemplate template = factored.getTemplate();
						if(!this.templateEmbedding.containsKey(template)) {
							
							INDArray vecTemplate = Nd4j.rand(new int[]{1, this.templateDim});
							vecTemplate.subi(0.5).muli(epsilonTemplate);
							this.templateEmbedding.put(template, vecTemplate);
							
							INDArray gradTemplate = Nd4j.zeros(this.templateDim);
							this.gradTemplateEmbedding.put(template, gradTemplate);
							this.adaGradSumSquareGradientTemplate.put(template, Nd4j.zeros(this.templateDim).addi(0.00001));
						}
					}
				}
			}
		}
		
		if(this.useLexicalEntry) {
			LOG.info("Number of lexical entries %s from processed data", this.lexicalEntryEmbedding.size());
		} else {
			LOG.info("Number of lexeme %s from processed data", this.lexemeEmbedding.size());
			LOG.info("Number of template %s from processed data", this.templateEmbedding.size());	
		}
	}
	
	@Override
	public int getDimension() {
		return this.dim;
	}
	
	@Override
	public Object getEmbedding(Object obj) {
		
		if(!(obj instanceof ParsingOp))
			throw new RuntimeException("Object Class should be of type: ParsingOp<MR>");
		
		@SuppressWarnings("unchecked")
		final ParsingOp<MR> parsingOp = (ParsingOp<MR>)obj;
		
		return this.getEmbedding(parsingOp);
	}
	
	public ParsingOpEmbeddingResult getEmbedding(ParsingOp<MR> op) {
		
		if(op instanceof LexicalParsingOp) {
			LexicalParsingOp<MR> lexicalParsingOp = (LexicalParsingOp<MR>)op;
			return this.getEmbedding(op, lexicalParsingOp.getEntry());
		}
		
		if(op.getRule().equals(ShiftReduceLexicalStep.LEXICAL_DERIVATION_STEP_RULENAME)) { //shift step
			throw new RuntimeException("Lexical Entry Parsing Op must be declared using LexicalParsingOp");
		}
		
		int ruleIndex = this.ruleNames.indexOf(op.getRule());
		INDArray opActionEmbedding = this.actionEmbedding.get(ruleIndex);
		
		CategoryEmbeddingResult categResult = this.categEmbedding.getCategoryEmbedding(op.getCategory());
		INDArray cembed = categResult.getEmbedding();
		
		//////////////////////////// return W.concat = embedding
		if(this.useSharedSpace) {
			INDArray concat =  Nd4j.concat(1, cembed, opActionEmbedding);
			INDArray embedding = this.Wreduce.mmul(concat.transpose()).transpose();
			return new ParsingOpEmbeddingResult(ruleIndex, embedding, categResult, null, null, concat);
		}
		////////////////////////////
		
		final INDArray lexicalEmbedding;
		if(this.usedPaddedVector) { //use a learnable padding vector
			lexicalEmbedding = this.paddedLexicalEmbedding;
		} else { //else pad by 0
			lexicalEmbedding = Nd4j.zeros(this.lexicalEmbeddingDim);
		}
		
		INDArray concat = Nd4j.concat(1, opActionEmbedding, cembed, lexicalEmbedding);
		
		if(this.doSquashing) {
			concat.transposei();
			INDArray preOutput = this.W.mmul(concat).addi(this.b).transpose();
			INDArray embedding = Nd4j.getExecutioner()
										.execAndReturn(new Tanh(preOutput.dup()));
			return new ParsingOpEmbeddingResult(ruleIndex, embedding, categResult, null, preOutput, concat);
		} else {
			return new ParsingOpEmbeddingResult(ruleIndex, concat, categResult, null, null, null);
		}
	}
	
	/** Embed a parsing operation that represents a lexical step */
	private ParsingOpEmbeddingResult getEmbedding(ParsingOp<MR> op, LexicalEntry<MR> lexicalEntry) {
		
		if(!op.getRule().equals(ShiftReduceLexicalStep.LEXICAL_DERIVATION_STEP_RULENAME)) {
			throw new RuntimeException("Function only applicable for shift operations");
		}
		
		if(lexicalEntry == null) {
			throw new RuntimeException("Lexical Entry cannot be null for shift rules");
		}
		
		int ruleIndex = this.ruleNames.indexOf(op.getRule());
		INDArray opActionEmbedding = this.actionEmbedding.get(ruleIndex);
		
		CategoryEmbeddingResult categResult = this.categEmbedding.getCategoryEmbedding(lexicalEntry/*op*/.getCategory());
		INDArray cembed_ = categResult.getEmbedding();
		
		if(this.useLexicalEntry) {
		
			final LexicalEntry<MR> mappedLexicalEntry;
			final INDArray lexicalEntryEmbedding;
			
			if(this.lexicalEntryEmbedding.containsKey(lexicalEntry)) {
				mappedLexicalEntry = lexicalEntry;
				lexicalEntryEmbedding = this.lexicalEntryEmbedding.get(lexicalEntry);
			} else if (lexicalEntry.isDynamic() && this.dynamicOriginEmbedding.containsKey(lexicalEntry.getOrigin())) {
				mappedLexicalEntry = lexicalEntry;
				lexicalEntryEmbedding = this.dynamicOriginEmbedding.get(lexicalEntry.getOrigin());
			} else {
				mappedLexicalEntry = null;
				lexicalEntryEmbedding = this.lexicalEntryEmbedding.get(null);
				if(lexicalEntryEmbedding == null)
					throw new RuntimeException("Getting null lexical entry embedding fails.");
			}
			
			if(lexicalEntryEmbedding == null)
				throw new RuntimeException("lex entry embedding is null. LexicalEntry: " + lexicalEntry);
			
			//////////////////////
			if(this.useSharedSpace) { 
				INDArray concat =  Nd4j.concat(1, cembed_, lexicalEntryEmbedding);
				INDArray embedding = this.Wshift.mmul(concat.transpose()).transpose();
				return new ParsingOpEmbeddingResult(ruleIndex, embedding, categResult, mappedLexicalEntry, null, concat);
			}
			//////////////////////
			
			INDArray concat = Nd4j.concat(1, opActionEmbedding, cembed_, lexicalEntryEmbedding);
			if(this.doSquashing) {
				concat.transposei();
				INDArray preOutput = this.W.mmul(concat).addi(this.b).transpose();
				INDArray embedding = Nd4j.getExecutioner()
											.execAndReturn(new Tanh(preOutput.dup()));
				
				return new ParsingOpEmbeddingResult(ruleIndex, embedding,
													categResult, mappedLexicalEntry, preOutput, concat);
			} else {
				return new ParsingOpEmbeddingResult(ruleIndex, concat,
													categResult, mappedLexicalEntry, null, null);
			}
		} else {
			
			final INDArray lexemeEmbedding, lexicalTemplateEmbedding;
			
			@SuppressWarnings("unchecked")
			LexicalEntry<LogicalExpression> expLexicalEntry = (LexicalEntry<LogicalExpression>)lexicalEntry;
			FactoredLexicalEntry factoring = FactoringServices.factor(expLexicalEntry);
			
			Lexeme lexeme = factoring.getLexeme();
			if(this.lexemeEmbedding.containsKey(lexeme)) {
				lexemeEmbedding = this.lexemeEmbedding.get(lexeme);
			} else {
				lexemeEmbedding = this.lexemeEmbedding.get(null);
			}
			
			if(lexemeEmbedding == null)
				throw new RuntimeException("lexeme embedding is null");
			
			LexicalTemplate lexicalTemplate = factoring.getTemplate();
			if(this.templateEmbedding.containsKey(lexicalTemplate)) {
				lexicalTemplateEmbedding = this.templateEmbedding.get(lexicalTemplate);
			} else {
				lexicalTemplateEmbedding = this.templateEmbedding.get(null);
				LOG.warn("Unseen Template %s", lexicalTemplate);
			}
			
			//////////////////////
			if(this.useSharedSpace) { 
				INDArray concat =  Nd4j.concat(1, cembed_, lexemeEmbedding, lexicalTemplateEmbedding);
				INDArray embedding = this.Wshift.mmul(concat.transpose()).transpose();
				return new ParsingOpEmbeddingResult(ruleIndex, embedding, categResult, lexicalEntry, null, concat);
			}
			//////////////////////
			
			///// Word embedding 
//			INDArray headWordEmbedding = this.embedWordBuffer.
//									  getTunableWordEmbedding(lexicalEntry.getTokens().get(0).toString());
			////////////////////
			
			INDArray concat = Nd4j.concat(1, opActionEmbedding, cembed_, 
											lexemeEmbedding, lexicalTemplateEmbedding);//, headWordEmbedding);

			if(this.doSquashing) {
				concat.transposei();
				INDArray preOutput = this.W.mmul(concat).addi(this.b).transpose();
				INDArray embedding = Nd4j.getExecutioner()
										.execAndReturn(new Tanh(preOutput.dup()));
				return new ParsingOpEmbeddingResult(ruleIndex, embedding,
									categResult, lexicalEntry, preOutput, concat);
			} else {
				return new ParsingOpEmbeddingResult(ruleIndex, concat,
									categResult, lexicalEntry, null, null);
			}
		}
	}
	
	public void backPropSharedSpace(INDArray errorY, ParsingOpEmbeddingResult parseOpResult) {
		
		if(this.doSquashing) {
			throw new RuntimeException("Do Squashing should be closed");
		}
		
		int shiftRuleLexicalIndex = this.ruleNames.indexOf(ShiftReduceLexicalStep.LEXICAL_DERIVATION_STEP_RULENAME);
		
		//y = Wx, we are given error=dl/dy; we first compute dl/dx
		INDArray W, gradW;
		if(parseOpResult.ruleIndex() == shiftRuleLexicalIndex) { //shift rule
			W = this.Wshift;
			gradW = this.gradWshift;
		} else {
			W = this.Wreduce;
			gradW = this.gradWreduce;
		}
		
		//add to gradient of W
		INDArray newGradW = errorY.transpose().mmul(parseOpResult.getX());
		synchronized(gradW) {
			gradW.addi(newGradW);
		}
		INDArray error = errorY.mmul(W);
		
		final CategoryEmbeddingResult categResult = parseOpResult.getCategoryResult();
		if(categResult != null) { //backpropagate through the category 
		
			final Tree syntacticTree = categResult.getSyntacticTree();
			final Tree semanticsTree = categResult.getSemanticTree();
			
			INDArray errorCategory = error.get(NDArrayIndex.interval(0, this.categEmbedding.getDimension()));
			this.categEmbedding.backprop(syntacticTree, semanticsTree, errorCategory);
		}
		
		//update lexical embedding
		final int pad = this.categEmbedding.getDimension();
		
		if(parseOpResult.ruleIndex() == shiftRuleLexicalIndex) { //shift rule
			if(this.useLexicalEntry) { //update the lexical entry embedding
			
				INDArray errorLexicalEntry = error.get(NDArrayIndex.interval(pad, pad + this.lexicalEntryDim));
			
				@SuppressWarnings("unchecked")
				LexicalEntry<MR> opLexicalEntry = (LexicalEntry<MR>) parseOpResult.getLexicalEntry();
				
				final INDArray gradLexicalEntry;
				if(opLexicalEntry == null || this.lexicalEntryEmbedding.containsKey(opLexicalEntry)) {
					gradLexicalEntry = this.gradLexicalEntryEmbedding.get(opLexicalEntry);
				} else {
					gradLexicalEntry = this.gradDynamicOriginEmbedding.get(opLexicalEntry.getOrigin());
					this.modifiedDynamicOrigin.add(opLexicalEntry.getOrigin());
				}
			
				assert gradLexicalEntry != null;
			
				synchronized(gradLexicalEntry) {
					gradLexicalEntry.addi(errorLexicalEntry);
				}
			} else { //update the lexeme and template embedding
				
				@SuppressWarnings("unchecked")
				LexicalEntry<LogicalExpression> opLexicalEntry = (LexicalEntry<LogicalExpression>) parseOpResult.getLexicalEntry();
				FactoredLexicalEntry factoring = FactoringServices.factor(opLexicalEntry);
				
				INDArray errorLexeme = error.get(NDArrayIndex.interval(pad, pad + this.lexemeDim));
				
				Lexeme lexeme = factoring.getLexeme(); 
				final INDArray gradLexeme;
				if(this.gradLexemeEmbedding.containsKey(lexeme)) {
					gradLexeme = this.gradLexemeEmbedding.get(lexeme);	
				} else {
					gradLexeme = this.gradLexemeEmbedding.get(null);
				}
				
				assert gradLexeme != null;
			
				synchronized(gradLexeme) {
					gradLexeme.addi(errorLexeme);
				}
				
				INDArray errorLexicalTemplate = error.get(NDArrayIndex.interval(pad	+ this.lexemeDim, 
																	pad + this.lexemeDim + this.templateDim));
				
				LexicalTemplate template = factoring.getTemplate();
				INDArray gradLexicalTemplate;
				if(this.gradTemplateEmbedding.containsKey(template)) {
					gradLexicalTemplate = this.gradTemplateEmbedding.get(template);
				} else {
					gradLexicalTemplate = this.gradTemplateEmbedding.get(null);
				}
			
				assert gradLexicalTemplate != null;
						
				synchronized(gradLexicalTemplate) {
					gradLexicalTemplate.addi(errorLexicalTemplate);
				}
			}
		} else { //reduce rule
			INDArray errorAction = error.get(NDArrayIndex.interval(pad, pad + this.actionDim));
			INDArray actionGrad = this.gradActionEmbedding.get(parseOpResult.ruleIndex());
			this.updatedActionRule.add(parseOpResult.ruleIndex());
			
			synchronized(actionGrad) {
				actionGrad.addi(errorAction);
			}
		}
	}
	
	public void backProp(INDArray error, ParsingOpEmbeddingResult parseOpResult) {
		
		if(this.useSharedSpace) {
			this.backPropSharedSpace(error, parseOpResult);
			return;
		}
		
		if(this.doSquashing) {
			
			INDArray nonLinearDerivative = Nd4j.getExecutioner()
					.execAndReturn(new TanhDerivative(parseOpResult.getPreOutput().dup()));
			INDArray x = parseOpResult.getX().transpose();
			INDArray nonLinearDerivativeTranspose = nonLinearDerivative.transpose();
			
			INDArray dydx = Nd4j.zeros(this.W.shape());
			IntStream.range(0, this.W.shape()[1]).parallel().unordered()
					.forEach(col -> dydx.putColumn(col, nonLinearDerivativeTranspose.mul(this.W.getColumn(col))));
			
			INDArray errorTimesNonLinearDerivative = error.mul(nonLinearDerivative).transpose();
			
			synchronized(this.gradW) {
				this.gradW.addi(errorTimesNonLinearDerivative.mmul(x));
			}

			synchronized(this.gradb) {
				this.gradb.addi(errorTimesNonLinearDerivative);
			}
			
			error = error.mmul(dydx);	//dloss/dx
		}
		
		INDArray errorAction = error.get(NDArrayIndex.interval(0, this.actionDim));
		INDArray actionGrad = this.gradActionEmbedding.get(parseOpResult.ruleIndex());
		this.updatedActionRule.add(parseOpResult.ruleIndex());
		
		synchronized(actionGrad) {
			actionGrad.addi(errorAction);
		}
		
		final CategoryEmbeddingResult categResult = parseOpResult.getCategoryResult();
		
		if(categResult != null) { //backpropagate through the category 
		
			final Tree syntacticTree = categResult.getSyntacticTree();
			final Tree semanticsTree = categResult.getSemanticTree();
			
			INDArray errorCategory = error.get(NDArrayIndex.interval(this.actionDim,
												this.actionDim + this.categEmbedding.getDimension()));
			this.categEmbedding.backprop(syntacticTree, semanticsTree, errorCategory);
		}
		
		//update lexical embedding
		int shiftRuleLexicalIndex = this.ruleNames.indexOf(ShiftReduceLexicalStep.LEXICAL_DERIVATION_STEP_RULENAME);
		final int pad = this.actionDim + this.categEmbedding.getDimension();
		
		if(parseOpResult.ruleIndex() == shiftRuleLexicalIndex) { //shift rule
			if(this.useLexicalEntry) { //update the lexical entry embedding
			
				INDArray errorLexicalEntry = error.get(NDArrayIndex.interval(pad, pad + this.lexicalEntryDim));
			
				@SuppressWarnings("unchecked")
				LexicalEntry<MR> opLexicalEntry = (LexicalEntry<MR>) parseOpResult.getLexicalEntry();
				
				final INDArray gradLexicalEntry;
				if(opLexicalEntry == null || this.lexicalEntryEmbedding.containsKey(opLexicalEntry)) {
					gradLexicalEntry = this.gradLexicalEntryEmbedding.get(opLexicalEntry);
				} else {
					gradLexicalEntry = this.gradDynamicOriginEmbedding.get(opLexicalEntry.getOrigin());
					this.modifiedDynamicOrigin.add(opLexicalEntry.getOrigin());
				}
			
				assert gradLexicalEntry != null;
			
				synchronized(gradLexicalEntry) {
					gradLexicalEntry.addi(errorLexicalEntry);
				}
				
				this.updatedLexicalEntry.add(opLexicalEntry);
				
			} else { //update the lexeme and template embedding
				
				@SuppressWarnings("unchecked")
				LexicalEntry<LogicalExpression> opLexicalEntry = (LexicalEntry<LogicalExpression>) parseOpResult.getLexicalEntry();
				FactoredLexicalEntry factoring = FactoringServices.factor(opLexicalEntry);
				
				INDArray errorLexeme = error.get(NDArrayIndex.interval(pad, pad + this.lexemeDim));
				
				Lexeme lexeme = factoring.getLexeme(); 
				final INDArray gradLexeme;
				if(this.gradLexemeEmbedding.containsKey(lexeme)) {
					gradLexeme = this.gradLexemeEmbedding.get(lexeme);	
					this.updatedLexeme.add(lexeme);
				} else {
					gradLexeme = this.gradLexemeEmbedding.get(null);
					this.updatedLexeme.add(null);
				}
				
				synchronized(gradLexeme) {
					gradLexeme.addi(errorLexeme);
				}
				
				INDArray errorLexicalTemplate = error.get(NDArrayIndex.interval(pad	+ this.lexemeDim, 
																	pad + this.lexemeDim + this.templateDim));
				
				LexicalTemplate template = factoring.getTemplate();
				INDArray gradLexicalTemplate;
				if(this.gradTemplateEmbedding.containsKey(template)) {
					gradLexicalTemplate = this.gradTemplateEmbedding.get(template);
					this.updatedTemplate.add(template);
				} else {
					gradLexicalTemplate = this.gradTemplateEmbedding.get(null);
					this.updatedTemplate.add(null);
				}
						
				synchronized(gradLexicalTemplate) {
					gradLexicalTemplate.addi(errorLexicalTemplate);
				}
				
//				final int wordDim = this.embedWordBuffer.tunableWordDim();
//				INDArray errorHeadWord = error.get(NDArrayIndex.interval(pad + this.lexemeDim + this.templateDim, 
//														pad + this.lexemeDim + this.templateDim + wordDim));
//				
//				String headWord = opLexicalEntry.getTokens().get(0).toString();
//				this.embedWordBuffer.addTunableWordEmbeddingGrad(headWord, errorHeadWord);
			}
		} else { //reduce rule
			if(this.usedPaddedVector) {
				INDArray paddingError = error.get(NDArrayIndex.interval(pad, pad + this.paddedLexicalEmbedding.size(1)));
				this.updatedPaddedVector.set(true);
				synchronized(this.paddedLexicalEmbeddingGrad) {
					this.paddedLexicalEmbeddingGrad.addi(paddingError);
				}
			}
		}
	}
	
	/** Update the vector given the gradient and adagrad history. Given gradient
	 * will update the AdaGrad history. All the parameters must not be used by other threads.
	 * Warning: This function modifies the gradient value and adagrad history. */
	public void update(INDArray vec, INDArray grad, INDArray sumSquareGrad, LearningRateStats learningRateStats,
								Object label, String type) {
		
		double normGrad = grad.normmaxNumber().doubleValue();
		if(normGrad == 0) {
			LOG.warn("Parsing Op: 0 gradient found: " + label);
		}
		
		//// Code below is for debugging
		final double meanActivation = Helper.meanAbs(vec);
		final double meanGradient = Helper.meanAbs(grad);
		
		synchronized(this.meanActivations) {
			if(this.meanActivations.containsKey(type)) {
				double oldVal = this.meanActivations.get(type);
				this.meanActivations.put(type, meanActivation + oldVal);
			} else {
				this.meanActivations.put(type, meanActivation);
			}
		}
		
		synchronized(this.meanGradients) {
			if(this.meanGradients.containsKey(type)) {
				double oldVal = this.meanGradients.get(type);
				this.meanGradients.put(type, meanGradient + oldVal);
			} else {
				this.meanGradients.put(type, meanGradient);
			}
		}
		///////
		
		//Add regularizer
		grad.addi(vec.mul(this.regularizer));
		
		//not performing clipping
		
		INDArray squaredGrad = grad.mul(grad);
		
		if(squaredGrad.normmaxNumber().doubleValue() == 0) {
			LOG.warn("Parsing Op: 0 gradient^2 found: " + label + " gradient was " + normGrad);
		}
		
		
		//update AdaGrad history
		sumSquareGrad.addi(squaredGrad/*grad.mul(grad)*/);
		
		//Update the vectors
		INDArray invertedLearningRate = Nd4j.getExecutioner()
											.execAndReturn(new Sqrt(sumSquareGrad.dup()))
											.divi(this.learningRate);
		
		double minLearningRate = /*grad.minNumber().doubleValue();*/1.0/(invertedLearningRate.maxNumber().doubleValue());
		double maxLearningRate = /*grad.maxNumber().doubleValue();*/1.0/(invertedLearningRate.minNumber().doubleValue());
		
		synchronized(learningRateStats) {
			learningRateStats.min(minLearningRate);
			learningRateStats.max(maxLearningRate);
		}
	
		vec.subi(grad.div(invertedLearningRate));
	}
	
	public void updateParameters() {
		
		this.learningRateStatsActionRule.unset();
		this.learningRateStatsLexeme.unset();
		this.learningRateStatsTemplate.unset();
		this.learningRateStatsPaddedVector.unset();
		
		if(this.doGradientChecks) {
			LOG.info("Gradient Check. Empirical Action %s. Estimate Action %s", this.empiricalAction, 
													this.gradActionEmbedding.get(0).getDouble(new int[]{0, 0}));
			LOG.info("Gradient Check. Empirical Template %s. Estimate Template %s", this.empiricalTemplate, 
													this.gradTemplateEmbedding.get(this.template).getDouble(new int[]{0, 0}));
		}
		
//		Iterator<INDArray> itVec = this.actionEmbedding.iterator();
//		Iterator<INDArray> itGrad = this.gradActionEmbedding.iterator();
//		Iterator<INDArray> itSumSquare = this.adaGradSumSquareGradientAction.iterator();
//		
//		while(itVec.hasNext()) {
//			assert itGrad.hasNext() && itSumSquare.hasNext();
//			
//			INDArray vec = itVec.next();
//			INDArray grad = itGrad.next();
//			INDArray sumSquareGrad = itSumSquare.next();
//			
//			this.update(vec, grad, sumSquareGrad);
//		}
		
		StreamSupport.stream(Spliterators
				.spliterator(this.updatedActionRule, Spliterator.IMMUTABLE), true).unordered()
				.forEach(ruleIndex-> {
					INDArray vec = this.actionEmbedding.get(ruleIndex);
					INDArray grad = this.gradActionEmbedding.get(ruleIndex);
					INDArray sumSquareGrad = this.adaGradSumSquareGradientAction.get(ruleIndex);
				
					this.update(vec, grad, sumSquareGrad, this.learningRateStatsActionRule, ruleIndex, "Parsing-Rule");
				});
		
		if(this.useLexicalEntry) {
			
			StreamSupport.stream(Spliterators
					.spliterator(this.updatedLexicalEntry, Spliterator.IMMUTABLE), true).unordered()
					.forEach(lexicalEntry-> {
						INDArray vec = this.lexicalEntryEmbedding.get(lexicalEntry);
						INDArray grad = this.gradLexicalEntryEmbedding.get(lexicalEntry);
						INDArray sumSquareGrad = this.adaGradSumSquareGradientLexicalEntry.get(lexicalEntry);
					
						this.update(vec, grad, sumSquareGrad, null, lexicalEntry, "Lexical-Entry");
					});
			
			StreamSupport.stream(Spliterators
					.spliterator(this.dynamicOriginEmbedding.entrySet(), Spliterator.IMMUTABLE), true).unordered()
					.forEach(entrySet-> {
						INDArray vec = entrySet.getValue();
						INDArray grad = this.gradDynamicOriginEmbedding.get(entrySet.getKey());
						INDArray sumSquareGrad = this.adaGradSumSquareGradientDynamicOrigin.get(entrySet.getKey());
					
						this.update(vec, grad, sumSquareGrad, null, entrySet.getKey(), "Dynamic-Origin");
					});
			
		} else {
			
			StreamSupport.stream(Spliterators
					.spliterator(this.updatedLexeme, Spliterator.IMMUTABLE), true).unordered()
					.forEach(lexeme-> {
						INDArray vec = this.lexemeEmbedding.get(lexeme);
						INDArray grad = this.gradLexemeEmbedding.get(lexeme);
						INDArray sumSquareGrad = this.adaGradSumSquareGradientLexeme.get(lexeme);
					
						this.update(vec, grad, sumSquareGrad, this.learningRateStatsLexeme, lexeme, "Lexeme");
					});
			
			StreamSupport.stream(Spliterators
					.spliterator(this.updatedTemplate, Spliterator.IMMUTABLE), true).unordered()
					.forEach(lexicalTemplate-> {
						INDArray vec = this.templateEmbedding.get(lexicalTemplate);
						INDArray grad = this.gradTemplateEmbedding.get(lexicalTemplate);
						INDArray sumSquareGrad = this.adaGradSumSquareGradientTemplate.get(lexicalTemplate);
					
						this.update(vec, grad, sumSquareGrad, this.learningRateStatsTemplate, template, "Template");
					});
		}
		
//		LOG.info("Parsing Op. Learning Rate Stats Action Rule %s", this.learningRateStatsActionRule);
//		LOG.info("Parsing Op. Learning Rate Stats Lexeme %s", this.learningRateStatsLexeme);
//		LOG.info("Parsing Op. Learning Rate Stats Template %s", this.learningRateStatsTemplate);
		
		if(this.usedPaddedVector && this.updatedPaddedVector.get()) {
			this.update(this.paddedLexicalEmbedding, this.paddedLexicalEmbeddingGrad,
							this.adaGradSumSquarePaddedLexicalEmbedding, this.learningRateStatsPaddedVector,
							"padded_vector", "padded_vector");
			LOG.info("Parsing Op. Learning Rate Stats PaddedVector %s", this.learningRateStatsPaddedVector);
		}
		
		if(this.useSharedSpace) {
			this.update(this.Wshift, this.gradWshift, this.adaGradSumSquareWshift, null, "W_shift", "W_shift");
			this.update(this.Wreduce, this.gradWreduce, this.adaGradSumSquareWreduce, null, "W_reduce", "W_reduce");
		}
		
		//update W,b
		if(this.doSquashing) {
			this.update(this.W, this.gradW, this.adaGradSumSquareGradW, null, "Squash_W", "Squash_W");
			this.update(this.b, this.gradb, this.adaGradSumSquareGradb, null, "Squash_b", "Squash_b");
		}
		
		/////////////
		final Map<String, Integer> counts = new HashMap<String, Integer>();

		counts.put("Parsing-Rule", this.updatedActionRule.size());
		counts.put("Lexeme", this.updatedLexeme.size());
		counts.put("Template", this.updatedTemplate.size());
		if(this.usedPaddedVector && this.updatedPaddedVector.get()) {
			counts.put("padded_vector", 1);
		} else {
			counts.put("padded_vector", 0);
		}
		
		for(Entry<String, Double> e: this.meanActivations.entrySet()) {
			Integer i = counts.get(e.getKey());
			if(i == null || i == 0) {
				LOG.info("Activation:: %s  NA", e.getKey());
			} else {
				LOG.info("Activation:: %s  %s", e.getKey(), e.getValue()/(double)i);
			}
		}
		
		this.meanActivations.clear();
		
		for(Entry<String, Double> e: this.meanGradients.entrySet()) {
			Integer i = counts.get(e.getKey());
			if(i == null || i == 0) {
				LOG.info("Gradient:: %s  NA", e.getKey());
			} else {
				LOG.info("Gradient:: %s  %s", e.getKey(), e.getValue()/(double)i);
			}
		}
		
		this.meanGradients.clear();
		/////////////
	}
	
	public void flushGradients() {
		
//		for(INDArray grad: this.gradActionEmbedding) {
//			grad.muli(0);
//		}
		
		StreamSupport.stream(Spliterators
				.spliterator(this.updatedActionRule, Spliterator.IMMUTABLE), true).unordered()
				.forEach(ruleIndex-> {
					INDArray grad = this.gradActionEmbedding.get(ruleIndex);
					grad.muli(0);
				});
		
		this.updatedActionRule.clear();
		
		if(this.useLexicalEntry) {
		
			StreamSupport.stream(Spliterators
					.spliterator(this.updatedLexicalEntry, Spliterator.IMMUTABLE), true).unordered()
					.forEach(lexicalEntry-> {
						INDArray grad = this.gradLexicalEntryEmbedding.get(lexicalEntry);
						grad.muli(0);
					});
			
			StreamSupport.stream(Spliterators
					.spliterator(this.gradDynamicOriginEmbedding.values(), Spliterator.IMMUTABLE), true).unordered()
					.forEach(grad-> {
						grad.muli(0);
					});
			
			this.modifiedLexicalEntries.addAll(this.updatedLexicalEntry);
			this.updatedLexicalEntry.clear();
		} else {
			
			StreamSupport.stream(Spliterators
					.spliterator(this.updatedLexeme, Spliterator.IMMUTABLE), true).unordered()
					.forEach(lexeme-> {
						INDArray grad = this.gradLexemeEmbedding.get(lexeme);
						grad.muli(0);
					});
			
			this.updatedLexeme.clear();
			
			StreamSupport.stream(Spliterators
					.spliterator(this.updatedTemplate, Spliterator.IMMUTABLE), true).unordered()
					.forEach(lexicalTemplate-> {
						INDArray grad = this.gradTemplateEmbedding.get(lexicalTemplate);
						grad.muli(0);
					});
			
			this.updatedTemplate.clear();
		}
		
		if(this.usedPaddedVector && this.updatedPaddedVector.get()) {
			this.paddedLexicalEmbeddingGrad.muli(0);
			this.updatedPaddedVector.set(false);
		}
		
		if(this.doSquashing) {
			this.gradW.muli(0);
			this.gradb.muli(0);
		}
		
		if(this.useSharedSpace) {
			this.gradWshift.muli(0);
			this.gradWreduce.muli(0);
		}
	}

}
