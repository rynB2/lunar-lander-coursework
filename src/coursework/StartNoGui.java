package coursework;

import model.Fitness;
import model.LunarParameters;
import model.LunarParameters.DataSet;
import model.NeuralNetwork;
import model.StringIO;

import java.util.ArrayList;
import java.util.stream.IntStream;

/**
 * Example of how to to run the {@link ExampleEvolutionaryAlgorithm} without the need for the GUI
 * This allows you to conduct multiple runs programmatically 
 * The code runs faster when not required to update a user interface
 *
 */
public class StartNoGui {

	public static void main(String[] args) {
		ArrayList<Double> trainFitnessList = new ArrayList<>();
		ArrayList<Double> testFitnessList = new ArrayList<>();
		for(int i = 0; i < 10; i++) {
			//Set the data set for training
			Parameters.setDataSet(DataSet.Training);


			//Create a new Neural Network Trainer Using the above parameters
			NeuralNetwork nn = new ExampleEvolutionaryAlgorithm();

			//train the neural net (Go and have a coffee)
			nn.run();


			/**
			 * The last File Saved to the Output Directory will contain the best weights /
			 * Parameters and Fitness on the Training Set
			 *
			 * We can used the trained NN to Test on the test Set
			 */
			Parameters.setDataSet(DataSet.Training);
			double train_fitness = Fitness.evaluate(nn);
			trainFitnessList.add(train_fitness);
			//System.out.println("TRAIN: " + Parameters.getDataSet() + " " + train_fitness);

			Parameters.setDataSet(DataSet.Test);
			double test_fitness = Fitness.evaluate(nn);
			testFitnessList.add(test_fitness);
			//System.out.println("TEST: " + Parameters.getDataSet() + " " + test_fitness);
		}
		System.out.println("training fitnesses:" + trainFitnessList);
		double average = trainFitnessList.stream().mapToDouble(d -> d).average().orElse(0.0);
		System.out.println("average: " + average);

		System.out.println("test fitnesses:" + testFitnessList);

		double average2 = testFitnessList.stream().mapToDouble(d -> d).average().orElse(0.0);
		System.out.println("average2: " + average2);
		/*
		String str = "";

		str = str + Parameters.printParams();
		str = str + "Training Fitness: " + trainFitnessList + ", \r\n";
		str = str + "Test Fitness: " + testFitnessList + ", \r\n";
		String filePrefix = "C:\\Users\\rynml\\Documents\\University\\Computational Intelligence\\Experiments\\initialise" + "-" + "seeded";
		StringIO.writeStringToFile(filePrefix + ".txt", str, false);
		System.out.println(str);

		/*
		 * Or We can reload the NN from the file generated during training and test it on a data set 
		 * We can supply a filename or null to open a file dialog 
		 * Note that files must be in the project root and must be named *-n.txt
		 * where "n" is the number of hidden nodes
		 * ie  1518461386696-5.txt was saved at timestamp 1518461386696 and has 5 hidden nodes
		 * Files are saved automatically at the end of training
		 *  

		
		ExampleEvolutionaryAlgorithm nn2 = ExampleEvolutionaryAlgorithm.loadNeuralNetwork("1518446327913-5.txt");
		Parameters.setDataSet(DataSet.Random);
		double fitness2 = Fitness.evaluate(nn2);
		System.out.println("Fitness on " + Parameters.getDataSet() + " " + fitness2);
		 */
	}
}
