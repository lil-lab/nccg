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
import java.util.Map;
import java.util.Spliterator;
import java.util.Spliterators;
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

import edu.cornell.cs.nlp.spf.parser.ParsingOp;
import edu.cornell.cs.nlp.spf.parser.ccg.rnn.shiftreduce.learning.Helper;
import edu.cornell.cs.nlp.spf.parser.ccg.rnn.shiftreduce.parsingopembeddings.ParsingOpEmbedding;
import edu.cornell.cs.nlp.spf.parser.ccg.rnn.shiftreduce.parsingopembeddings.ParsingOpEmbeddingResult;
import edu.cornell.cs.nlp.spf.parser.ccg.shiftreduce.stacks.PersistentEmbeddings;
import edu.cornell.cs.nlp.utils.composites.Pair;
import edu.cornell.cs.nlp.utils.log.ILogger;
import edu.cornell.cs.nlp.utils.log.LogLevel;
import edu.cornell.cs.nlp.utils.log.LoggerFactory;

/** Embeds the list of parsing operations performed by the parser */
public class EmbedActionHistory<MR> implements AbstractEmbedding, AbstractRecurrentNetworkHelper {

	public static final ILogger LOG = LoggerFactory.create(EmbedActionHistory.class);
	
	private final RecurrentSequenceEmbedding parsingOpSeqEmbed;
	private final ParsingOpEmbedding<MR> parsingOpEmbed;
	private final MultiLayerNetwork net;
	private final int nIn, nOut;
	private final double learningRate;
	private final double l2;
	private final int seed;
	private List<ParsingOpEmbeddingResult> parsingOpEmbeddingResult;
	
	public double norm = 0;
	public int term = 0;
	
	private double meanInputActivations;
	
	public EmbedActionHistory(ParsingOpEmbedding<MR> parsingOpEmbed, double learningRate, double l2, 
							  int seed) {
		this.nIn = parsingOpEmbed.getDimension();
		this.nOut = 25;//50;//25;
		this.learningRate = learningRate;
		this.l2 = l2;
		this.seed = seed;
		
		this.meanInputActivations = 0.0;
		
		this.net = this.buildRecurrentNetwork(this.nIn, this.nOut);
		
		this.parsingOpSeqEmbed = new RecurrentSequenceEmbedding(this.net, this.nIn, this.nOut);
		this.parsingOpEmbed = parsingOpEmbed;
		this.parsingOpEmbeddingResult = new LinkedList<ParsingOpEmbeddingResult>();
		LOG.setCustomLevel(LogLevel.INFO);
		LOG.info("Embed Action History RNN: nIN %s, nOut %s", this.nIn, this.nOut); 
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
				
		MultiLayerNetwork net = new MultiLayerNetwork(conf);
		net.init();
		
		return net;
	}
	
	public void reclone() {
		this.parsingOpSeqEmbed.reclone();
	}
	
	public void clearParsingOpEmbeddingResult() {
		this.parsingOpEmbeddingResult.clear();
	}
	
	public void printTimeStamps() {
		this.parsingOpSeqEmbed.printTimeStamps();
	}
	
	/** Dumps the configuration and parameters of the network in a file inside the given folder */
	public void logNetwork(String folderName) {
		
		try {
			
			PrintWriter writer = new PrintWriter(folderName + "/action_history_conf.json", "UTF-8");
			
			writer.write(this.net.getLayerWiseConfigurations().toJson());
			writer.flush();
			writer.close();
			
		}  catch (IOException e) {
			throw new RuntimeException("Could not dump the action history conf: "+e);
		}
		
		try {
			OutputStream fos = Files.newOutputStream(
					Paths.get(folderName + "/action_history_param.bin"));
	        DataOutputStream dos = new DataOutputStream(fos);
		    
			Nd4j.write(this.net.params(), dos);
			dos.flush();
		    dos.close();
		    
		} catch (IOException e) {
			throw new RuntimeException("Could not dump the action history params: "+e);
		}
	}
	
	/** Bootstraps the network with the parameters in the action_history_param.bin
	 *  inside the given folder */
	public void bootstrapNetworkParam(String folderName) {
		
		//final String confFile = folderName+"/action_history_conf.json";
		final String paramFile = folderName+"/action_history_param.bin";
		
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
			throw new RuntimeException("Could not read the action history param: "+e);
		}
	}
	
	@Override
	public int getDimension() {
		return this.nOut;
	}
	
	@Override
	public Object getEmbedding(Object obj) {
	
		if(!(obj instanceof List))
			throw new RuntimeException("Object type should be: List<ParsingOp<MR>>");
		
		@SuppressWarnings("unchecked")
		final List<ParsingOp<MR>> parsingSteps = (List<ParsingOp<MR>>)obj;
		
		return this.getEmbedding(parsingSteps);
	}
	
	@Override
	public Object getAllTopLayerEmbedding(Object obj) {
		
		if(!(obj instanceof List))
			throw new RuntimeException("Object type should be: List<ParsingOp<MR>>");
		
		@SuppressWarnings("unchecked")
		final List<ParsingOp<MR>> parsingSteps = (List<ParsingOp<MR>>)obj;
		
		return this.getAllTopLayerEmbedding(parsingSteps);	
	}

	/** gets the embedding of list of current histories given a rnn state */
	@SuppressWarnings("unchecked")
	public List<RecurrentTimeStepOutput> getAllEmbedding(
										List<Pair<PersistentEmbeddings, ParsingOp<MR>>> parsingOpMap) {
		
		final int batchSize = parsingOpMap.size();
		
		if(batchSize == 0) {
			return new LinkedList<RecurrentTimeStepOutput>();
		}
		
		INDArray input = Nd4j.zeros(new int[]{batchSize, this.nIn, 1});
		
		final int numLayers = this.net.getnLayers();
		Map<String, INDArray>[] rnnStates = null; 
		int ex = 0;
		
		for(Pair<PersistentEmbeddings, ParsingOp<MR>> e: parsingOpMap) {
	
			ParsingOp<MR> parsingOp = e.second();
			ParsingOpEmbeddingResult result = parsingOpEmbed.getEmbedding(parsingOp); 
			//this.parsingOpEmbeddingResult.add(result);
			INDArray stepEmbed_ = result.getEmbedding();
			
			for(int i=0; i< this.nIn; i++) {
				input.putScalar(new int[]{ex, i, 0}, stepEmbed_.getDouble(i));
			}
	
			Map<String, INDArray>[] rnnState = e.first().getRNNState();
			
			if(rnnState != null) { //if null then by default the rnn state is all 0s
				
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
		
		return this.parsingOpSeqEmbed.getAllEmbedding(input, rnnStates);
	}
	
	/** gets the embedding of the current history */
	public RecurrentTimeStepOutput getEmbedding(ParsingOp<MR> parsingStep, 
												Map<String, INDArray>[] rnnState) {
		
		final int dim = this.parsingOpEmbed.getDimension();
		INDArray input = Nd4j.zeros(new int[]{1, dim, 1});
		
		ParsingOpEmbeddingResult result = this.parsingOpEmbed.getEmbedding(parsingStep); 
		//this.parsingOpEmbeddingResult.add(result);
		INDArray embedStep = result.getEmbedding();
		
		for(int i=0; i<dim; i++) {
			input.putScalar(new int[]{0, i, 0}, embedStep.getDouble(i));
		}
	
		return this.parsingOpSeqEmbed.getEmbedding(input, rnnState);
	}
	
	/** gets the embedding of the current history */
	public INDArray getEmbedding(List<ParsingOp<MR>> parsingSteps) {
		
		final int dim = this.parsingOpEmbed.getDimension();
		INDArray input = Nd4j.zeros(new int[]{1, dim, parsingSteps.size()});
		
		int count = 0;
		for(ParsingOp<MR> e: parsingSteps) {
			ParsingOpEmbeddingResult result = this.parsingOpEmbed.getEmbedding(e); 
			this.parsingOpEmbeddingResult.add(result);
			INDArray stepEmbed_ = result.getEmbedding();
			
			for(int i=0; i<dim; i++) {
				input.putScalar(new int[]{0, i, count}, stepEmbed_.getDouble(i));
			}
			
			count++;
		}
		
		return this.parsingOpSeqEmbed.getEmbedding(input);
	}
	
	/** gets the embedding of all prefixes of the current history */
	public INDArray[] getAllTopLayerEmbedding(List<ParsingOp<MR>> parsingSteps) {
		
		final int dim = this.parsingOpEmbed.getDimension();
		INDArray input = Nd4j.zeros(new int[]{1, dim, parsingSteps.size()});
		//INDArray subMatrix = input.tensorAlongDimension(0, 2, 1);
		
		int count = 0;
		for(ParsingOp<MR> e: parsingSteps) {
			ParsingOpEmbeddingResult result = this.parsingOpEmbed.getEmbedding(e); 
			this.parsingOpEmbeddingResult.add(result);
			INDArray stepEmbed = result.getEmbedding();
			
			for(int i=0; i<dim; i++) {
				input.putScalar(new int[]{0, i, count}, stepEmbed.getDouble(i));
			}
			
			//subMatrix.putColumn(count, stepEmbed);
			
			count++;
		}
		
		this.meanInputActivations = Helper.meanAbs(input);
		
		return this.parsingOpSeqEmbed.getAllTopLayerEmbeddings(input);
	}
	
	@Override
	public void backprop(INDArray error) {
		
		if(this.parsingOpEmbeddingResult.size() == 0) { //nothing to do
			return;
		}
		
		// create a error feedback for the sequence
		final int dim = this.getDimension();
		assert error.size(1) == dim;
		
		INDArray paddedError = Nd4j.zeros(new int[]{1, dim, this.parsingOpEmbeddingResult.size()});
		
		for(int i = 0; i < dim; i++) {
			//paddedError.putScalar(i, error.getDouble(i));
			paddedError.putScalar(new int[]{0, i, this.parsingOpEmbeddingResult.size()-1},
								  error.getDouble(i));
		}
		
		List<INDArray> errorTimeSeries = this.parsingOpSeqEmbed.backprop(paddedError);
		
		//RNN grad output
		if(LOG.getLogLevel() == LogLevel.DEBUG) {
		
			String s  = "{ ";
			for(INDArray errorNode: errorTimeSeries) {
				s = s + ", {" + Helper.printVector(errorNode)+"}";
			}
			LOG.debug("Action History is %s }" , s);
		}
		
		Iterator<ParsingOpEmbeddingResult> it = this.parsingOpEmbeddingResult.iterator();
		LOG.debug("[[ ------ Embed Action History ");
		
		List<Pair<ParsingOpEmbeddingResult, INDArray>> backpropParsingOp = 
						new LinkedList<Pair<ParsingOpEmbeddingResult, INDArray>>();
		
		for(int i = 0; i < errorTimeSeries.size(); i++) {
			assert it.hasNext();
			backpropParsingOp.add(Pair.of(it.next(), errorTimeSeries.get(i)));
			this.norm = this.norm + errorTimeSeries.get(i).normmaxNumber().doubleValue();
			this.term++;
		}
		
		StreamSupport.stream(Spliterators
						.spliterator(backpropParsingOp, Spliterator.IMMUTABLE), true)
						.forEach(p->this.parsingOpEmbed.backProp(p.second(), p.first()));
		
		LOG.debug(" Embed Action History ------- ]]");
		this.parsingOpEmbeddingResult.clear();
	}
	
	/** backprops the error given for the entire time series.*/
	@Override
	public void backprop(INDArray[] error) {
		
		assert error.length == this.parsingOpEmbeddingResult.size() + 1; 
		
		if(this.parsingOpEmbeddingResult.size() == 0) { //nothing to do
			return;
		}
		
		// create a error feedback for the sequence
		final int dim = this.getDimension();
		
		INDArray paddedError = Nd4j.zeros(new int[]{1, dim, this.parsingOpEmbeddingResult.size()});
		//INDArray subMatrix = paddedError.tensorAlongDimension(0, 2, 1);
		
		for(int j = 1; j < error.length; j++) {
			
			for(int i = 0; i < dim; i++) {
				paddedError.putScalar(new int[]{0, i, j - 1}, error[j].getDouble(i));
			}
			
			//subMatrix.putColumn(j - 1, error[j]);
		}
		
		List<INDArray> errorTimeSeries = this.parsingOpSeqEmbed.backprop(paddedError);

		LOG.info("Activation:: Action-History-Input %s Time %s", this.meanInputActivations, System.currentTimeMillis());
		LOG.info("Gradient:: Action-History-Recurrent-Param %s Time %s", this.parsingOpSeqEmbed.getMeanGradients(), System.currentTimeMillis());
		LOG.info("Gradient:: Action-History-Input %s Time %s", this.parsingOpSeqEmbed.getMeanInputGradients(),  System.currentTimeMillis());
		
		
		//RNN grad output
		if(LOG.getLogLevel() == LogLevel.DEBUG) {
		
			String s  = "{ ";
			for(INDArray errorNode: errorTimeSeries) {
				s = s + ", {" + Helper.printVector(errorNode)+"}";
			}
			LOG.debug("Action History is %s }" , s);
		}
		
		Iterator<ParsingOpEmbeddingResult> it = this.parsingOpEmbeddingResult.iterator();
		LOG.debug("[[ ------ Embed Action History ");
		
		List<Pair<ParsingOpEmbeddingResult, INDArray>> backpropParsingOp = 
						new LinkedList<Pair<ParsingOpEmbeddingResult, INDArray>>();
		
		for(int i = 0; i < errorTimeSeries.size(); i++) {
			assert it.hasNext();
			backpropParsingOp.add(Pair.of(it.next(), errorTimeSeries.get(i)));
			this.norm = this.norm + errorTimeSeries.get(i).normmaxNumber().doubleValue();
			this.term++;
		}
		
		StreamSupport.stream(Spliterators
						.spliterator(backpropParsingOp, Spliterator.IMMUTABLE), true)
						.forEach(p->this.parsingOpEmbed.backProp(p.second(), p.first()));
		
		LOG.debug(" Embed Action History ------- ]]");
		this.parsingOpEmbeddingResult.clear();
	}
}
