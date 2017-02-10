package edu.cornell.cs.nlp.spf.parser.ccg.rnn.shiftreduce.categoryembeddings;
/*
import java.util.HashMap;
import java.util.Map.Entry;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.Tanh;
import org.nd4j.linalg.factory.Nd4j;

import edu.cornell.cs.nlp.utils.log.ILogger;
import edu.cornell.cs.nlp.utils.log.LoggerFactory;
import jcuda.*;
import jcuda.jcublas.*;*/

//File currently removed as it is not needed
//and requires unncessary lib file management
//not marking for deprecation as it maybe needed later
public class GPUServices {

	/*  public static final ILogger LOG = LoggerFactory.create(GPUServices.class);
	
	  private final int n;	  
	  private HashMap<String, Pointer> pointerMap;
	  private final float alpha;
	  private final float beta;
	  
	  private final Pointer pt, dResult;
	  private final float[] result;
	  
	  private long computeTime = 0;
	  
	  public GPUServices(int n) {
		  	// Initialize JCublas 
		    JCublas.cublasInit();
		    this.pointerMap = new HashMap<String, Pointer>();
		    this.alpha = 1.0f;
		    this.beta = 0.0f;
		    this.n = n;
		    
		    result = new float[this.n];
		    
		    pt = new Pointer();
		    JCublas.cublasAlloc(2 * n, Sizeof.FLOAT, pt);
		    
		    dResult = new Pointer();
		    JCublas.cublasAlloc(n, Sizeof.FLOAT, dResult);
		    JCublas.cublasSetVector(this.n, Sizeof.FLOAT, Pointer.to(result), 1, dResult, 1);
		    
		    computeTime = 0;
	  }
	  
	  public void store(String label, float data[]) {
		  Pointer pt = new Pointer();
		  this.pointerMap.put(label, pt);
		  
		  int size = data.length;
		  JCublas.cublasAlloc(size, Sizeof.FLOAT, pt);
		  JCublas.cublasSetVector(size, Sizeof.FLOAT, Pointer.to(data), 1, pt, 1);
	  }
	  
	  public boolean contains(String label) {
		  return this.pointerMap.containsKey(label);
	  }
	  
	  /** multiply the vector with the matrix represented by label */
	/*  public float[] matrixVectorMul(String label, float vec[]) {
		  
//		  Pointer pt = new Pointer();
		  int size = vec.length;
//		  JCublas.cublasAlloc(size, Sizeof.FLOAT, pt);
		  JCublas.cublasSetVector(size, Sizeof.FLOAT, Pointer.to(vec), 1, pt, 1);
		  
		  Pointer dWeight = this.pointerMap.get(label);
		  
//		  float[] result = new float[this.n];
//		  Pointer dResult = new Pointer();
//		  JCublas.cublasAlloc(this.n, Sizeof.FLOAT, dResult);
//		  JCublas.cublasSetVector(this.n, Sizeof.FLOAT, Pointer.to(result), 1, dResult, 1);
		  
		  long start = System.currentTimeMillis();
		  JCublas.cublasSgemm('n', 'n', this.n, 1, 2*this.n, alpha, dWeight, this.n, pt, 2*this.n, beta, dResult, this.n);
		  this.computeTime = this.computeTime  + System.currentTimeMillis() - start;
		  JCublas.cublasGetVector(this.n, Sizeof.FLOAT, dResult, 1, Pointer.to(result), 1);
		  
//		  JCublas.cublasFree(pt);
		  
		  return result;
	  }
	  
	  public void clean() {
		  for(Entry<String, Pointer> e: this.pointerMap.entrySet()) {
			  JCublas.cublasFree(e.getValue());
		  }
	  }
	  
	  public void shutdown() {
		  JCublas.cublasShutdown();
	  }
	  
	  public void test2() {
		  
		INDArray W = Nd4j.rand(n, 2*n);
		INDArray b = Nd4j.rand(n, 1);
		INDArray left = Nd4j.rand(1, n);
		INDArray right = Nd4j.rand(1, n);
		  
		//perform composition
		long time1 = System.currentTimeMillis();
		INDArray concat = Nd4j.concat(1, left, right).transpose();
		long time2 = System.currentTimeMillis();
		LOG.info("concat shape %s %s", concat.shape()[0], concat.shape()[1]);
		long time3 = System.currentTimeMillis();
		INDArray transformed = W.mmul(concat).addi(b).transposei();
		
		//next operation does not create new copy of transformed
		long time4 = System.currentTimeMillis();
		@SuppressWarnings("unused")
		INDArray nonLinear = Nd4j.getExecutioner().execAndReturn(new Tanh(transformed.dup()));
		long time5 = System.currentTimeMillis();
		
		LOG.info("Time stamps %s %s %s %s", (time2 - time1), (time3 - time2), 
					(time4 - time3), (time5 - time4));		
	  }
	  
	  public void test() {
		  
		  long totalTime1 = 0, totalTime2 = 0;
		  int N = 1000;
		  int correct = 0;
		  double tolerance = 0.0001;
		  
		  for(int iter = 1; iter <= N; iter ++) {
		  
			  INDArray W = Nd4j.rand(this.n, 2 * this.n);
			  INDArray v = Nd4j.rand(2 * this.n, 1);
			  
			  //Direct multiplication
			  long start1 = System.currentTimeMillis();
			  INDArray z = W.mmul(v);
			  long end1 = System.currentTimeMillis();
			  LOG.debug("Direct multiplication %s", z);
			  totalTime1 = totalTime1 +  end1 - start1;
			  
			  //GPU based multiplication
			  float[] WData = W.dup('f').data().asFloat();
			  this.store("W", WData);
			  
			  long start2 = System.currentTimeMillis();
			  float[] vData = v.data().asFloat();
			  float[] uData = this.matrixVectorMul("W", vData);
			  
			  INDArray u = Nd4j.create(uData).transposei();
			  long end2 = System.currentTimeMillis();
			  LOG.debug("GPU based multiplication %s", u);
			  totalTime2 = totalTime2 +  end2 - start2;
			  
			  boolean testPassed = true;
			  if(z.shape()[0] != u.shape()[0] || z.shape()[1] != u.shape()[1]) {
				  LOG.info("Failed. Shape mismatch");
				  testPassed = false;
				  continue;
			  }
			  
			  for(int i = 0; i < z.shape()[0]; i++) {
				  for(int j = 0; j < z.shape()[1]; j++) {
					  double eps = Math.abs(z.getDouble(new int[]{i, j}) - 
							  					u.getDouble(new int[]{i, j}));
					  if(eps > tolerance) {
						  LOG.info("Failed. Difference %s", eps);
						  testPassed = false;
						  break;
					  }
				  }
				  if(!testPassed) {
					  break;
				  }
			  }
			  
			  if(testPassed) {
				  correct++;
			  }
		  }
		  
		  double avg1 = totalTime1/(double)N, avg2 = totalTime2/(double)N;
		  double avg3 = this.computeTime/(double)N;
		  LOG.info("Test Results Passed %s / %s",correct ,N);
		  LOG.info("Total Time Stats: CPU %s, GPU %s, GPU compute %s", totalTime1, 
				  										totalTime2, this.computeTime);
		  LOG.info("Avg Time Stats: CPU %s, GPU %s, GPU-compute %s", avg1, avg2, avg3);
	  }*/
}
 