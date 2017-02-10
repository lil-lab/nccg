package edu.cornell.cs.nlp.spf.parser.ccg.rnn.shiftreduce.embeddings;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.io.PrintWriter;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.HashMap;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;
import java.util.ListIterator;
import java.util.Map;
import java.util.Spliterator;
import java.util.Spliterators;
import java.util.Stack;
import java.util.Map.Entry;
import java.util.stream.StreamSupport;

import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.GravesLSTM;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import edu.cornell.cs.nlp.spf.ccg.categories.Category;
import edu.cornell.cs.nlp.spf.parser.ccg.rnn.shiftreduce.categoryembeddings.CategoryEmbedding;
import edu.cornell.cs.nlp.spf.parser.ccg.rnn.shiftreduce.categoryembeddings.CategoryEmbeddingResult;
import edu.cornell.cs.nlp.spf.parser.ccg.rnn.shiftreduce.learning.Helper;
import edu.cornell.cs.nlp.spf.parser.ccg.shiftreduce.stacks.DerivationState;
import edu.cornell.cs.nlp.spf.parser.ccg.shiftreduce.stacks.DerivationStateHorizontalIterator;
import edu.cornell.cs.nlp.spf.parser.ccg.shiftreduce.stacks.PersistentEmbeddings;
import edu.cornell.cs.nlp.utils.composites.Pair;
import edu.cornell.cs.nlp.utils.log.ILogger;
import edu.cornell.cs.nlp.utils.log.LogLevel;
import edu.cornell.cs.nlp.utils.log.LoggerFactory;

/** Embeds parser state */
public class EmbedParserState<MR> implements AbstractEmbedding, AbstractRecurrentNetworkHelper {
	
	public static final ILogger LOG = LoggerFactory.create(EmbedParserState.class);
	
	private final RecurrentSequenceEmbedding parserStateSeqEmbed;
	private final CategoryEmbedding<MR> embedCategory;
	private final MultiLayerNetwork net;
	private final int nIn, nOut;
	private final double learningRate;
	private final double l2;
	private final int seed;
	private Stack<CategoryEmbeddingResult> categoryResults;
	
	public double norm = 0;
	public int term = 0;
	
	/** Mean input activations*/
	private double meanInputActivations;
	
	public EmbedParserState(CategoryEmbedding<MR> categEmbedding, double learningRate, double l2, 
							int seed) {
		this.nIn = categEmbedding.getDimension();
		this.nOut = 20;//40;//20;
		this.learningRate = learningRate;
		this.l2 = l2;
		this.seed = seed;
		
		this.meanInputActivations = 0.0;
		
		this.net = this.buildRecurrentNetwork(this.nIn, this.nOut);
		
		this.parserStateSeqEmbed = new  RecurrentSequenceEmbedding(this.net, this.nIn, this.nOut);
		this.embedCategory = categEmbedding;
		this.categoryResults = new Stack<CategoryEmbeddingResult>();
		LOG.setCustomLevel(LogLevel.INFO);
		LOG.info("Embed Parser State nIn %s nOut %s", this.nIn, this.nOut);
	}
	
	public MultiLayerNetwork buildRecurrentNetwork(int nIn, int nOut) {
		
		int lstmLayerSize = 100;
		//Set up network configuration:
		MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
					.updater(org.deeplearning4j.nn.conf.Updater.ADAM)
					.optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).iterations(1)
					.learningRate(this.learningRate*2.0)
					.momentum(0.0)
					.seed(this.seed)
					.regularization(true)
					.l2(this.l2)
					.list(2)
					.layer(0, new GravesLSTM.Builder().nIn(nIn).nOut(lstmLayerSize)
							.updater(Updater.ADAM).dropOut(0.0)
							.activation("hardtanh").weightInit(WeightInit.XAVIER).build())
					.layer(1, new GravesLSTM.Builder().nIn(lstmLayerSize).nOut(nOut)
							.updater(Updater.ADAM).dropOut(0.0)
							.activation("hardtanh").weightInit(WeightInit.XAVIER).build())
					.pretrain(false).backprop(true)
					.build();
		
		MultiLayerNetwork net_ = new MultiLayerNetwork(conf);
		net_.init();
		
		return net_;
	}
	
	/** Dumps the configuration and parameters of the network in a file inside the given folder */
	public void logNetwork(String folderName) {
		
		try {
			
			PrintWriter writer = new PrintWriter(folderName + "/parser_state_conf.json", "UTF-8");
			
			writer.write(this.net.getLayerWiseConfigurations().toJson());
			writer.flush();
			writer.close();
			
		}  catch (IOException e) {
			throw new RuntimeException("Could not dump the parser state conf: "+e);
		}
		
		try {
			OutputStream fos = Files.newOutputStream(
					Paths.get(folderName + "/parser_state_param.bin"));
	        DataOutputStream dos = new DataOutputStream(fos);
		    
			Nd4j.write(this.net.params(), dos);
			dos.flush();
		    dos.close();
		    
		} catch (IOException e) {
			throw new RuntimeException("Could not dump the parser state params: "+e);
		}
	}
	
	/** Bootstraps the network with the parameters in the parser_state_param.bin
	 *  inside the given folder */
	public void bootstrapNetworkParam(String folderName) {
		
		//final String confFile = folderName+"/action_history_conf.json";
		final String paramFile = folderName+"/parser_state_param.bin";
		
		/*MultiLayerConfiguration confFromJson = MultiLayerConfiguration.fromJson(
					FileUtils.readFileToString(new File(confFile)));*/
		
		try {
		
			DataInputStream dis = new DataInputStream(new FileInputStream(paramFile));
			INDArray newParams = Nd4j.read(dis);
	
			dis.close();
			//MultiLayerNetwork savedNetwork = new MultiLayerNetwork(confFromJson);
			this.net.init();
			this.net.setParameters(newParams);
			
		} catch(IOException e) {
			throw new RuntimeException("Could not read the parser state param: "+e);
		}
	}
	
	public void clearCategoryResults() {
		this.categoryResults.clear();
	}
	
	@Override
	public int getDimension() {
		return this.nOut;
	}
	
	@Override
	public Object getEmbedding(Object obj) {
	
		if(!(obj instanceof DerivationState))
			throw new RuntimeException("Object type should be: DerivationState<MR>");
		
		@SuppressWarnings("unchecked")
		final DerivationState<MR> dstate = (DerivationState<MR>)obj;
		
		return this.getEmbedding(dstate);
	}
	
	@Override
	public Object getAllTopLayerEmbedding(Object obj) {
	
		if(!(obj instanceof DerivationState))
			throw new RuntimeException("Object type should be: DerivationState<MR>");
		
		@SuppressWarnings("unchecked")
		final DerivationState<MR> dstate = (DerivationState<MR>)obj;
		
		return this.getAllTopLayerEmbedding(dstate);
	}
	
	public void reclone() {
		this.parserStateSeqEmbed.reclone();
	}
	
	public void printTimeStamps() {
		this.parserStateSeqEmbed.printTimeStamps();
	}
	
	/** returns embedding of a set current states with the same rnn state activations */
	@SuppressWarnings("unchecked")
	public List<RecurrentTimeStepOutput> getAllEmbedding(
										List<Pair<PersistentEmbeddings, DerivationState<MR>>> stateMap) {
	
		final int batchSize = stateMap.size();
		
		if(batchSize == 0) {
			return new LinkedList<RecurrentTimeStepOutput>();
		}
		
		INDArray input = Nd4j.zeros(new int[]{batchSize, this.nIn, 1});
		
		final int numLayers = this.net.getnLayers();
		Map<String, INDArray>[] rnnStates = null;
		int ex = 0;
		
		for(Pair<PersistentEmbeddings, DerivationState<MR>> e: stateMap) {
			
			DerivationState<MR> dstate = e.second();
			//embed the rightmost category using rnn and given previous rnn state
			
			Category<MR> rightmost = dstate.getRightCategory();
			if(rightmost == null) {
				rightmost = dstate.getLeftCategory();
			}
			
			if(rightmost == null) { //degenerate case, the first derivation state
				//return 0 or maybe not reach here. Settle this later.
				throw new RuntimeException("Should not reach here. Case not handled.");
			}
			
			CategoryEmbeddingResult categoryResult = embedCategory.getCategoryEmbedding(rightmost);
			//this.categoryResults.push(categoryResult);
	
			INDArray categoryEmbedding = categoryResult.getEmbedding();
			for(int i = 0; i < this.nIn; i++)
				input.putScalar(new int[]{ex, i, 0}, categoryEmbedding.getDouble(i)); //check this line
			
			Map<String, INDArray>[] rnnState = e.first().getRNNState();
			
			if(rnnState != null) { //if rnn state is null then by default rnn state is all 0s
				
				if(rnnStates == null) {
					rnnStates = new HashMap[numLayers];
					for(int l=0; l<numLayers; l++) {
						rnnStates[l] = new HashMap<String, INDArray>();
					}
				}
				
				for(int l=0; l<numLayers; l++) {
					Map<String, INDArray> layerState = rnnState[l];
					
					for(Entry<String, INDArray> entry: layerState.entrySet()) {
						String key = entry.getKey();
						INDArray ar = entry.getValue();
						INDArray state = rnnStates[l].get(key);
						
						int size = ar.size(1);
						
						if(state == null) {
							state = Nd4j.zeros(new int[]{batchSize, size});
							rnnStates[l].put(key, state);
						}
						
						//modify state using ar
						for(int j=0; j<size; j++) {
							state.putScalar(new int[]{ex, j}, ar.getDouble(j));
						}
					}
				}
			}
			
			ex++;
		}
		
		return this.parserStateSeqEmbed.getAllEmbedding(input, rnnStates);
	}
	
	/** Special case embedding */
	public Pair<RecurrentTimeStepOutput, PersistentEmbeddings> 
					getEmbedding(Category<MR> left, Category<MR> right) {
	
		CategoryEmbeddingResult leftCategoryResult = embedCategory.getCategoryEmbedding(left);
		//this.categoryResults.push(categoryResult);
		INDArray leftInput = Nd4j.zeros(new int[]{1, this.nIn, 1});
		
		INDArray arrLeft = leftCategoryResult.getEmbedding();
		for(int i = 0; i < this.nIn; i++)
			leftInput.putScalar(new int[]{0, i, 0}, arrLeft.getDouble(i));
		
		RecurrentTimeStepOutput leftOutput = this.parserStateSeqEmbed.getEmbedding(leftInput, null);
		
		CategoryEmbeddingResult rightCategoryResult = embedCategory.getCategoryEmbedding(right);
		//this.categoryResults.push(categoryResult);
		INDArray rightInput = Nd4j.zeros(new int[]{1, this.nIn, 1});
		
		INDArray arrRight = rightCategoryResult.getEmbedding();
		for(int i = 0; i < this.nIn; i++)
			rightInput.putScalar(new int[]{0, i, 0}, arrRight.getDouble(i)); //check this line
		
		PersistentEmbeddings parentPersistentEmbedding = new PersistentEmbeddings(null, null);
		PersistentEmbeddings persistentEmbedding = new PersistentEmbeddings(leftOutput.getRNNState(),
																				parentPersistentEmbedding);
		
		RecurrentTimeStepOutput timeStepOutput = this.parserStateSeqEmbed
														.getEmbedding(rightInput, leftOutput.getRNNState());
		
		return Pair.of(timeStepOutput, persistentEmbedding);
	}
	
	/** returns embedding of the current state */
	public RecurrentTimeStepOutput getEmbedding(DerivationState<MR> dstate, 
								Map<String, INDArray>[] rnnState) {
		
		//embed the rightmost category using rnn and given previous rnn state
		
		Category<MR> rightmost = dstate.getRightCategory();
		if(rightmost == null) {
			rightmost = dstate.getLeftCategory();
		}
		
		if(rightmost == null) { //degenerate case, the first derivation state
			//return 0 or maybe not reach here. Settle this later.
			throw new RuntimeException("Should not reach here. Case not handled.");
		}
		
		CategoryEmbeddingResult categoryResult = embedCategory.getCategoryEmbedding(rightmost);
				//this.categoryResults.push(categoryResult);
		INDArray input = Nd4j.zeros(new int[]{1, this.nIn, 1});
	
		INDArray arr = categoryResult.getEmbedding();
		for(int i = 0; i < this.nIn; i++)
			input.putScalar(new int[]{0, i, 0}, arr.getDouble(i)); //check this line
		
		return this.parserStateSeqEmbed.getEmbedding(input, rnnState);
	}
	
	/** returns embedding of the current state */
	public INDArray getEmbedding(DerivationState<MR> dstate) {
		
		if(dstate == null) //edge case
			return Nd4j.zeros(this.getDimension());
		
		Stack<INDArray> rootCategories = new Stack<INDArray>();
		DerivationStateHorizontalIterator<MR> hiter = dstate.horizontalIterator();
		
		boolean first = true;
		
		while(hiter.hasNext()) {
			DerivationState<MR> current = hiter.next();
			
			if(first) {
				first = false;
				
				if(current.getRightCategory() != null) {
					CategoryEmbeddingResult categoryResult = embedCategory.getCategoryEmbedding(
													current.getRightCategory());
					this.categoryResults.push(categoryResult);
					rootCategories.push(categoryResult.getEmbedding());
				}
			}
			
			if(current.getLeftCategory() != null) {
				CategoryEmbeddingResult categoryResult = embedCategory.getCategoryEmbedding(
											current.getLeftCategory());
				this.categoryResults.push(categoryResult);
				rootCategories.push(categoryResult.getEmbedding());
			}
		}
		
		INDArray input = Nd4j.zeros(new int[]{1, this.nIn, rootCategories.size()});
		
		ListIterator<INDArray> it = rootCategories.listIterator();
		
		int n = rootCategories.size();
		
		while(it.hasNext()) {
			int node = it.nextIndex();
			INDArray arr = it.next();
			for(int i = 0; i < this.nIn; i++) {
				//input.putScalar(new int[]{0, i, node}, arr.getDouble(i)); //check this line
				//Rightmost category is rightmost in the time-series
				input.putScalar(new int[]{0, i, n - node - 1}, arr.getDouble(i)); //check this line
			}
		}
		
		return this.parserStateSeqEmbed.getEmbedding(input);
	}
	
	
	/** returns embedding of all prefixes of current state */
	public INDArray[] getAllTopLayerEmbedding(DerivationState<MR> dstate) {
		
		if(dstate == null) { //edge case
			INDArray[] prefixEmbedding = new INDArray[1];
			prefixEmbedding[0] = Nd4j.zeros(new int[]{1, this.getDimension(), 0});
			return prefixEmbedding; 
		}
		
		Stack<INDArray> rootCategories = new Stack<INDArray>();
		DerivationStateHorizontalIterator<MR> hiter = dstate.horizontalIterator();
		
		boolean first = true;
		
		while(hiter.hasNext()) {
			DerivationState<MR> current = hiter.next();
			
			if(first) {
				first = false;
				
				if(current.getRightCategory() != null) {
					CategoryEmbeddingResult categoryResult = embedCategory.getCategoryEmbedding(
													current.getRightCategory());
					this.categoryResults.push(categoryResult);
					rootCategories.push(categoryResult.getEmbedding());
				}
			}
			
			if(current.getLeftCategory() != null) {
				CategoryEmbeddingResult categoryResult = embedCategory.getCategoryEmbedding(
											current.getLeftCategory());
				this.categoryResults.push(categoryResult);
				rootCategories.push(categoryResult.getEmbedding());
			}
		}
		
		INDArray input = Nd4j.zeros(new int[]{1, this.nIn, rootCategories.size()});
		//INDArray subMatrix = input.tensorAlongDimension(0, 2, 1);
		
		ListIterator<INDArray> it = rootCategories.listIterator();
		
		int n = rootCategories.size();
		
		while(it.hasNext()) {
			int node = it.nextIndex();
			INDArray arr = it.next();
			for(int i = 0; i < this.nIn; i++) {
				//Rightmost category is rightmost in the time-series
				input.putScalar(new int[]{0, i, n - node - 1}, arr.getDouble(i)); //check this line
			}
			//subMatrix.putColumn(n - node - 1, arr);
		}
		
		this.meanInputActivations = Helper.meanAbs(input);
		
		return this.parserStateSeqEmbed.getAllTopLayerEmbeddings(input);
	}
	
	@Override
	public void backprop(INDArray error) {
		
		if(this.categoryResults.size() == 0) {
			return; //Nothing to do
		}
		
		// create a error feedback for the sequence
		final int dim = this.getDimension();
		assert error.size(1) == dim;
		
		INDArray paddedError = Nd4j.zeros(new int[]{1, dim, this.categoryResults.size()});
		
		for(int i = 0; i< dim; i++) {
			//paddedError.putScalar(i, error.getDouble(i));
			paddedError.putScalar(new int[]{0, i, this.categoryResults.size() - 1}, error.getDouble(i));
		}
		
		List<INDArray> errorTimeSeries = this.parserStateSeqEmbed.backprop(paddedError);
		
		//RNN grad output
		if(LOG.getLogLevel() == LogLevel.DEBUG) {
			
			String s  = "{ ";
			for(INDArray errorNode: errorTimeSeries) {
				s = s + ", {" + Helper.printVector(errorNode)+"}";
			}
			LOG.debug("Parse State output is %s }" , s);
		}
		
		//propage the loss
		assert errorTimeSeries.size() == this.categoryResults.size();
		
		Iterator<CategoryEmbeddingResult> it = this.categoryResults.iterator();
		int i = 0;
		int n = errorTimeSeries.size();
		
		LOG.debug("[[ ------ Embed Parse State ");
		
		List<Pair<CategoryEmbeddingResult, INDArray>> backpropCategories = 
					new LinkedList<Pair<CategoryEmbeddingResult, INDArray>>();
		while(it.hasNext()) {
			backpropCategories.add(Pair.of(it.next(), errorTimeSeries.get(n - i - 1)));
			this.norm = this.norm + errorTimeSeries.get(n - i - 1).normmaxNumber().doubleValue();
			this.term++;
			i++;
		}
		
		StreamSupport.stream(Spliterators
				.spliterator(backpropCategories, Spliterator.IMMUTABLE), true)
				.forEach(p->this.embedCategory.backprop(p.first().getSyntacticTree(),
														p.first().getSemanticTree(),
														p.second()));
		
		LOG.debug(" Embed Parse Statr ------ ]]");
		this.categoryResults.clear();
	}
	
	@Override
	public void backprop(INDArray[] error) {
		
		assert error.length == this.categoryResults.size() + 1;
		
		if(this.categoryResults.size() == 0) {
			return; //Nothing to do
		}
		
		// create a error feedback for the sequence
		final int dim = this.getDimension();
		
		INDArray paddedError = Nd4j.zeros(new int[]{1, dim, this.categoryResults.size()});
		//INDArray subMatrix = paddedError.tensorAlongDimension(0, 2, 1);
		
		for(int j = 1; j < error.length; j++) {
			
			for(int i = 0; i< dim; i++) {
				paddedError.putScalar(new int[]{0, i, j - 1}, error[j].getDouble(i));
			}
			//subMatrix.putColumn(j - 1, error[j]);
		}
		
		List<INDArray> errorTimeSeries = this.parserStateSeqEmbed.backprop(paddedError);
		
		LOG.info("Activation:: Parser-State-Input %s Time %s", this.meanInputActivations, System.currentTimeMillis());
		LOG.info("Gradient:: Parser-State-Recurrent-Param %s Time %s", this.parserStateSeqEmbed.getMeanGradients(), System.currentTimeMillis());
		LOG.info("Gradient:: Parser-State-Input %s Time %s", this.parserStateSeqEmbed.getMeanInputGradients(), System.currentTimeMillis());
		
		//RNN grad output
		if(LOG.getLogLevel() == LogLevel.DEBUG) {
			
			String s  = "{ ";
			for(INDArray errorNode: errorTimeSeries) {
				s = s + ", {" + Helper.printVector(errorNode)+"}";
			}
			LOG.debug("Parse State output is %s }" , s);
		}
		
		//propage the loss
		assert errorTimeSeries.size() == this.categoryResults.size();
		
		Iterator<CategoryEmbeddingResult> it = this.categoryResults.iterator();
		int i = 0;
		int n = errorTimeSeries.size();
		
		LOG.debug("[[ ------ Embed Parse State ");
		
		List<Pair<CategoryEmbeddingResult, INDArray>> backpropCategories = 
					new LinkedList<Pair<CategoryEmbeddingResult, INDArray>>();
		while(it.hasNext()) {
			backpropCategories.add(Pair.of(it.next(), errorTimeSeries.get(n - i - 1)));
			this.norm = this.norm + errorTimeSeries.get(n - i - 1).normmaxNumber().doubleValue();
			this.term++;
			i++;
		}
		
		StreamSupport.stream(Spliterators
				.spliterator(backpropCategories, Spliterator.IMMUTABLE), true)
				.forEach(p->this.embedCategory.backprop(p.first().getSyntacticTree(),
														p.first().getSemanticTree(),
														p.second()));
		
		LOG.debug(" Embed Parse Statr ------ ]]");
		this.categoryResults.clear();
	}

}
