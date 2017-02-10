package edu.cornell.cs.nlp.spf.parser.ccg.rnn.shiftreduce.learning;

import java.util.Arrays;
import java.util.LinkedList;
import java.util.List;
import java.util.ListIterator;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.Sqrt;
import org.nd4j.linalg.factory.Nd4j;

import com.google.common.base.Splitter;

/** Class containing simple static functions performing operations on INDArray
 * @author Dipendra Misra
 *  */
public class Helper {
	
	public static Splitter splitter = Splitter.on(',').omitEmptyStrings().trimResults();

	public static INDArray printVector(INDArray v) {
		return v;
	}
	
	public static String printFullVector(INDArray v) {
		
		int[] shape = v.shape();
		int dim = Math.max(shape[0], shape[1]); 
		
		StringBuilder s = new StringBuilder();
		for(int i = 0; i < dim; i++) {
			
			if(i > 0 && dim > 1) {
				s.append(", ");
			}
			
			double u = v.getDouble(i);
			if(Double.isNaN(u)) {
				s.append("NaN");
			} else if(Double.isInfinite(u)) {
				s.append("Infinity");
			} else s.append(u);
		}
		
		return s.toString();
	}
	
	/** Returns vector in CSV format val1,val2,val3,....,valk*/
	public static String printVectorToCSV(INDArray v) {
		
		int[] shape = v.shape();
		int dim = Math.max(shape[0], shape[1]); 
		
		StringBuilder s = new StringBuilder();
		for(int i = 0; i < dim; i++) {
			
			double u = v.getDouble(i);
			s.append(u);
			
			if(i < dim - 1) {
				s.append(",");
			}
		}
		
		return s.toString();
	}
	
	public static INDArray getXavierInitiliazation(int row, int col) {
		
		double epsilon = 2 * Math.sqrt(6.0/(double)(row + col));
		INDArray v = Nd4j.rand(new int[]{row, col});
		v.subi(0.5).muli(epsilon);
		
		return v;
	}
	
	/** Update this vector using AdaGrad */
	public static void updateVector(INDArray vec, INDArray grad, INDArray sumSquareGrad,
								double l2, double learningRate) {
		
		//Add regularizer
		grad.addi(vec.mul(l2));
		
		//not performing clipping
		
		INDArray squaredGrad = grad.mul(grad);
		
		//update AdaGrad history
		sumSquareGrad.addi(squaredGrad);
		
		//Update the vectors
		INDArray invertedLearningRate = Nd4j.getExecutioner()
											.execAndReturn(new Sqrt(sumSquareGrad.dup()))
											.divi(learningRate);
	
		vec.subi(grad.div(invertedLearningRate));	
	}
	
	public static String printMatrix(INDArray v) {
	
			return v.toString();
	}
	
	public static String printFullMatrix(INDArray v) {
		
		int[] shape = v.shape();
		
		StringBuilder s = new StringBuilder();
		
		for(int i = 0; i < shape[0]; i++) {
			s.append("\"");
			for(int j = 0; j < shape[1]; j++) {
				double u = v.getDouble(new int[]{i, j});
				
				if(j != 0 && shape[1] > 1) {
					s.append(", ");
				}
				
				if(Double.isNaN(u)) {
					s.append("NaN");
				} else if(Double.isInfinite(u)) {
					s.append("Infinity");
				} else s.append(u);
			}
			s.append("\", \n");
		}
		
		return s.toString();
	}
	
	public static double squaredFrobeniusNorm(INDArray v) {
		
		int[] shape = v.shape();
		String s = "{";
		double norm = 0;
		for(int i = 0; i < shape[0]; i++) {
			s = s + " {";
			for(int j = 0; j < shape[1]; j++) {
				double u = v.getDouble(new int[]{i, j});
				norm = norm + u*u;
				s = s + u + " ( " + norm + " ), ";
			}
			s = s + " }";
		}
		
		s = s + "}";
		//System.out.println("Squared " + s);
		return norm;
	}
	
	/** Takes a string of the form "a,b,c,d,e,.." and outputs an INDArray vector */
	public static INDArray toVector(String s) {
		
		Iterable<String> split = Helper.splitter.split(s);
		List<Double> values = new LinkedList<Double>();
		
		for(String value: split) {
			values.add(Double.parseDouble(value));
		}
		
		INDArray vec = Nd4j.zeros(values.size());
		
		ListIterator<Double> it = values.listIterator();
		
		while(it.hasNext()) {
			vec.putScalar(it.nextIndex(), it.next());
		}
		
		return vec;
	}
	
	/** Takes a string representing rows joined by | and outputs an INDArray matrix */
	public static INDArray toMatrix(String s) {
		
		s = s.replaceAll("\"", "");
		String[] rows = s.split("#");
		final int numRow = rows.length;
		
		//find column
		Iterable<String> firstRow = Helper.splitter.split(rows[0]);
		int numCol = 0;
		
		for(@SuppressWarnings("unused") String w: firstRow) {
			numCol++;
		}
		
		INDArray matrix = Nd4j.zeros(new int[]{numRow, numCol});
		
		int rowIndex = 0;
		
		for(int colIndex = 0; colIndex < rows.length; colIndex++) {
		
			Iterable<String> split = Helper.splitter.split(rows[colIndex]);
			List<Double> values = new LinkedList<Double>();
			
			int thisCol = 0;
			for(String value: split) {
				values.add(Double.parseDouble(value));
				thisCol++;
			}
			
			if(numCol != thisCol) {
				throw new IllegalStateException("All matrix rows should have same number of columns. Found "+numCol+" and "+thisCol);
			}
			
			ListIterator<Double> it = values.listIterator();
			
			while(it.hasNext()) {
				matrix.putScalar(new int[]{rowIndex, it.nextIndex()}, it.next());
			}
			
			rowIndex++;
		}
		
		return matrix;
	}
	
	/**  Calculates mean of INDArrays of same dimension */
	public static double mean(INDArray[] embedding) {
		
		if(embedding.length == 0) {
			return 0.0;
		}
		
		double mean = 0;
		
		for(int i = 0; i < embedding.length; i++) {
			mean = mean + Helper.meanAbs(embedding[i]);
		}
		
		mean = mean/(double)embedding.length;
		
		return mean;
	}
	
	/** Calculates mean of absolute values in INDArray */
	public static double meanAbs(INDArray n) {
		final int len = n.length();
		
		if(len == 0) {
			return 0.0;
		}
		
		return n.norm1Number().doubleValue()/(double)len;
	}
	
	public static double getMean(List<Integer> ls) {
		
		if(ls.size() == 0) return 0.0;
		
		double sum = 0.0;
		for(Integer l: ls) {
			sum = sum + l;
		}
		return sum/(double)ls.size();
	}
	
	public static double getStd(List<Integer> ls, Double mean) {
		
		if(ls.size() == 0) return 0.0;
		
		double sum = 0.0;
		for(Integer l: ls) {
			sum = sum + (l - mean) * (l - mean);
		}
		return Math.sqrt(sum/(double)ls.size());
	}
	
	public static String printHist(List<Integer> ls) {
		
		StringBuilder s = new StringBuilder();
		
		if(ls.size() == 0) return s.toString();
		
		double max = ls.get(0);
		for(Integer l :ls) {
			if(l > max) {
				max = l;
			}
		}
		
		int slice = (int)Math.ceil(max/50.0) + 1;
		
		int[] hist = new int[slice];
		Arrays.fill(hist, 0);
		
		for(Integer l: ls) {
			
			int sliceL = (int)Math.floor(l/50.0);
			hist[sliceL]++;
		}
		
		int pad = 0;
		for(int i = 0; i < slice; i++) {
			s.append(pad + "=" + (pad + 50) + " => " + hist[i] + "\n");
			pad = pad + 50;
		}
		
		return s.toString();
	}
	
	public static void main(String[] args) throws Exception {
		
		INDArray x = Nd4j.rand(new int[]{1, 10});
		INDArray y = Nd4j.rand(new int[]{3, 10});
		
		System.out.println("x is " + x);
		System.out.println("y is " + y);
		
		Nd4j.copy(x, y);
		
		System.out.println("x is " + x);
		System.out.println("y is " + y);
		
		
		
	}

}
