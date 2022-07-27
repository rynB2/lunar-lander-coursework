package coursework;

import java.lang.reflect.Field;
import java.util.ArrayList;
import java.util.Random;
import model.LunarParameters;
import model.NeuralNetwork;
import model.LunarParameters.DataSet;

public class Parameters {
 
	/**
	 * These parameter values can be changed 
	 * You may add other Parameters as required to this class 
	 * 
	 */

	//operators and activation to be used

	public static String activationFunction = "leakyrelu";  //can be relu, tanh (default), step, none, ncu, leakyrelu
	public static String selectionOperator = "random"; //can be tournament, random



	//other params
	private static int numHidden = 20; //default = 5
	public static int numGenes = calculateNumGenes();
	public static double minGene = -3; // default = -3
	public static double maxGene = +3; //default = +3

	public static int popSize = 50; //default 100

	//tournament params
	public static int tournamentSize = Math.round(Math.round(.3*popSize)); //default .3


	public static int maxEvaluations = 20000;

	// Parameters for crossover
	// Rate = probability of switching
	// crossoverChance = probability of any crossover happening
	public static double crossoverChance = 0.6;
	
	// Parameters for mutation 
	// Rate = probability of changing a gene
	// Change = the maximum +/- adjustment to the gene value
	public static double mutateRate = 0.06; // mutation rate for mutation operator, default = 0.04 
	public static double mutateChange = 0.1; // delta change for mutation operator, default = 0.1 
	
	//Random number generator used throughout the application
	public static long seed = System.currentTimeMillis();
	public static Random random = new Random(seed);

	//set the NeuralNetwork class here to use your code from the GUI
	public static Class neuralNetworkClass = ExampleEvolutionaryAlgorithm.class;







































	
	/**
	 * Do not change any methods that appear below here.
	 * 
	 */
	
	public static int getNumGenes() {					
		return numGenes;
	}

	
	private static int calculateNumGenes() {
		int num = (NeuralNetwork.numInput * numHidden) + (numHidden * NeuralNetwork.numOutput) + numHidden + NeuralNetwork.numOutput;
		return num;
	}

	public static int getNumHidden() {
		return numHidden;
	}
	
	public static void setHidden(int nHidden) {
		numHidden = nHidden;
		numGenes = calculateNumGenes();		
	}

	public static String printParams() {
		String str = "";
		for(Field field : Parameters.class.getDeclaredFields()) {
			String name = field.getName();
			Object val = null;
			try {
				val = field.get(null);
			} catch (IllegalArgumentException | IllegalAccessException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
			str += name + " \t" + val + "\r\n";
			
		}
		return str;
	}
	
	public static void setDataSet(DataSet dataSet) {
		LunarParameters.setDataSet(dataSet);
	}
	
	public static DataSet getDataSet() {
		return LunarParameters.getDataSet();
	}
	
	public static void main(String[] args) {
		printParams();
	}
}
