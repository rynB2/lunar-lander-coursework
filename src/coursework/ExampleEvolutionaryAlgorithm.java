package coursework;

import java.lang.reflect.Array;
import java.util.*;

import model.Fitness;
import model.Individual;
import model.LunarParameters.DataSet;
import model.NeuralNetwork;

import javax.swing.*;
/**
 * Implements a basic Evolutionary Algorithm to train a Neural Network
 * 
 * 
 * 
 */
public class ExampleEvolutionaryAlgorithm extends NeuralNetwork {

	/**
	 * The Main Evolutionary Loop
	 */
	@Override
	public void run() {		
		//Initialise a population of Individuals with random weights
		population = initialise(); //random best

		//Record a copy of the best Individual in the population
		best = getBest(population);
		System.out.println("Best From Initialisation " + best);

		/**
		 * main EA processing loop
		 */		
		while (evaluations < Parameters.maxEvaluations) {  

			/**
			 * this is a skeleton EA - you need to add the methods.
			 * You can also change the EA if you want 
			 * You must set the best Individual at the end of a run
			 * 
			 */
			// Select 2 Individuals from the current population.
			Individual parent1 = select(); //tournament selection
			Individual parent2 = select();

			// Generate a child by crossover.
			ArrayList<Individual> children;
			if (Parameters.random.nextDouble() < Parameters.crossoverChance) {
				children = onePointCrossover(parent1, parent2); //onePointCrossover, uniformCrossover, twoPointCrossover
			}
			else {
				children = reproduce(parent1, parent2);
			}
			//mutate the offspring
			gaussianMutate(children); //shuffleMutate, randomMutate, gaussianMutate
			// Evaluate the children
			evaluateIndividuals(children);
			// Replace children in population
			replaceWorst(children); //replaceTournament, replaceWorst

			// check to see if the best has improved
			best = getBest(population);
			// Implemented in NN class. 
			outputStats();
			//Increment number of completed generations			
		}

		//save the trained network to disk
		saveNeuralNetwork();
	}


	/**
	 * Sets the fitness of the individuals passed as parameters (whole population)
	 * 
	 */
	private void evaluateIndividuals(ArrayList<Individual> individuals) {
		for (Individual individual : individuals) {
			individual.fitness = Fitness.evaluate(individual, this);
		}
	}


	/**
	 * Returns a copy of the best individual in the population
	 * 
	 */
	private Individual getBest(ArrayList<Individual> individuals) {
		best = null;
		for (Individual individual : individuals) {
			if (best == null) {
				best = individual.copy();
			} else if (individual.fitness < best.fitness) {
				best = individual.copy();
			}
		}
		return best;
	}


	/**
	 * Generates a randomly initialised population
	 * 
	 */
	private ArrayList<Individual> initialise() {
		population = new ArrayList<>();
		for (int i = 0; i < Parameters.popSize; ++i) {
			//chromosome weights are initialised randomly in the constructor
			Individual individual = new Individual();
			population.add(individual);
		}

		evaluateIndividuals(population);
		return population;
	}

	//initialises population with good members
	private ArrayList<Individual> seededInitialise() {
		population = new ArrayList<>();
		for (int i = 0; i < (Parameters.popSize - 2); ++i) {
			//chromosome weights are initialised randomly in the constructor
			Individual individual = new Individual();
			population.add(individual);
		}

		Individual seededIndividual1 = new Individual();
		Individual seededIndividual2 = new Individual();

		double[] seeded1 = {-1.9630567723237415,5.436270280848185,5.37717812028134,2.6814007625718483,2.8662797662723873,-4.253456732971072,-1.0714430739555283,-2.966470012859272,-4.975935374855008,6.980070218616101,3.1847752012647863,13.253799616677401,9.817986188323223,-3.118829788236396,1.965025735930037,-10.70732863923389,2.104768528144124,-3.021594980221506,-2.046172011817077,6.089165552157418,-0.6822246085163974,-0.4329604307536272,-2.0694129695638246,-2.7218222698731287,0.13394566893475268,2.4535874437528404,0.17272711724659828,0.4018053301696545,-0.44266869445149704,2.6969133898821465,-2.4664617770163746,0.26371644312437237,-3.9977319930167163,4.554982390376859,-1.980230272387697,2.0465185919407753,1.8167645700994246,-0.3473117208854717,-2.657492359669676,1.9531774655211622,0.29019623928464,-1.562944636753623,-0.6765024824359995,-0.6451648710319882,3.5930933939883922,5.085400687761323,1.538117017674005,-0.765758590059348};
		double[] seeded2 = {5.269921925961341,2.2356952990130936,-2.0161233451610174,-0.43554892009712587,-3.2598980444708463,-0.5797213942822368,3.247207176122573,-3.2956917331237543,5.39979230674117,-2.055156151651407,4.920964952925648,5.632686041271315,-3.925506317982424,-2.021329279090711,-10.066695797285327,0.3040170231768977,-4.803144751698731,-10.638370230946581,3.823842730731203,1.771752988085094,0.20859660702340488,-1.9752677875123836,0.1830192591888602,0.7602188592409,0.5029418192269062,0.5275782167667712,0.8305591799360383,1.6596849196195784,0.7163616216512443,1.4238863926712342,7.288528995676854,-3.8314977213727928,-1.2708268814047534,0.808475338896399,-3.2828330472947664,-0.0075945155672423326,0.6368114276785732,1.0432689836784543,-2.6095038417028316,-0.3412215416144293,0.514343696100048,2.3836170332096955,-2.1400479286815273,2.0219020188704118,0.31579955452118824,-1.2299111555566526,-5.4189388563910015,2.346513252836611};

		seededIndividual1.chromosome = seeded1;
		seededIndividual2.chromosome = seeded2;

		population.add(seededIndividual1);
		population.add(seededIndividual2);
		evaluateIndividuals(population);
		return population;
	}

	/**
	 * Selection --
	 * 
	 *
	 *
	 */

	 //randomly selects a member of population
	public Individual randomSelect(){
			Individual parent = population.get(Parameters.random.nextInt(Parameters.popSize));

			return parent.copy();
	}


	//selects from best of n members randomly chosen
	public Individual tournamentSelect(){

		ArrayList<Individual> tournamentParticipants = new ArrayList<>();
		Collections.shuffle(population);

		for(int i = 0; i < Parameters.tournamentSize; i++){
			tournamentParticipants.add(population.get(i));
		}
		return getBest(tournamentParticipants);
	}


	private Individual select() {
		if (Parameters.selectionOperator == "random") return randomSelect();
		else if (Parameters.selectionOperator == "tournament") return tournamentSelect();
		else System.out.println("typo?");
		return randomSelect();
	}

	/**
	 * Crossover
	 * 
	 *
	 *
	 */


	private ArrayList<Individual> reproduce(Individual parent1, Individual parent2) {
		ArrayList<Individual> children = new ArrayList<>();
		children.add(parent1.copy());
		children.add(parent2.copy());			
		return children;
	}

	private ArrayList<Individual> onePointCrossover(Individual parent1, Individual parent2) {
		ArrayList<Individual> children = new ArrayList<>();
		Individual child1 = parent1.copy();
		Individual child2 = parent2.copy();

		int crossoverPoint = Parameters.random.nextInt(Parameters.numGenes);

		for(int i = 0; i < Parameters.numGenes; i++){

			if (i < crossoverPoint){
				double temp = child1.chromosome[i];
				child1.chromosome[i] = child2.chromosome[i];
				child2.chromosome[i] = temp;
			}
		}
		children.add(child1);
		children.add(child2);
		return children;
	}

	private ArrayList<Individual> twoPointCrossover(Individual parent1, Individual parent2){
		ArrayList<Individual> children = new ArrayList<>();
		Individual child1 = parent1.copy();
		Individual child2 = parent2.copy();

		int firstCrossoverPoint = Parameters.random.nextInt(Parameters.numGenes);
		int secondCrossoverPoint = Parameters.random.nextInt(Parameters.numGenes);

		if (firstCrossoverPoint == secondCrossoverPoint){
			if (firstCrossoverPoint == 0){
				secondCrossoverPoint++;
			} else {
				firstCrossoverPoint--;
			}
		}

		if (secondCrossoverPoint < firstCrossoverPoint) {
			int temp = firstCrossoverPoint;
			firstCrossoverPoint = secondCrossoverPoint;
			secondCrossoverPoint = temp;
		}

		for (int i = 0; i < Parameters.numGenes; i++) {
			if (i < firstCrossoverPoint || i > secondCrossoverPoint);
			else{
				double temp = child1.chromosome[i];
				child1.chromosome[i] = child2.chromosome[i];
				child2.chromosome[i] = temp;
			}
		}

		children.add(child1);
		children.add(child2);
		return children;
	}

	private ArrayList<Individual> uniformCrossover(Individual parent1, Individual parent2){
		ArrayList<Individual> children = new ArrayList<>();
		Individual child1 = parent1.copy();
		Individual child2 = parent2.copy();

		for (int i = 0; i < Parameters.numGenes; i++) {
			//if (Parameters.random.nextDouble() < Parameters.uniformCrossoverRate){
				if (Parameters.random.nextBoolean()) {
					double temp = child1.chromosome[i];
					child1.chromosome[i] = child2.chromosome[i];
					child2.chromosome[i] = temp;
				}
			//}
		}
		children.add(child1);
		children.add(child2);

		return children;
	}

	
	/**
	 * Mutation
	 * 
	 * 
	 */
	private void randomMutate(ArrayList<Individual> individuals) {
		for(Individual individual : individuals) {
			for (int i = 0; i < individual.chromosome.length; i++) {
				if (Parameters.random.nextDouble() < Parameters.mutateRate) {
					if (Parameters.random.nextBoolean()) {
						individual.chromosome[i] += (Parameters.mutateChange);
					} else {
						individual.chromosome[i] -= (Parameters.mutateChange);
					}
				}
			}
		}		
	}

	private void shuffleMutate(ArrayList<Individual> individuals){
		for (Individual individual : individuals){

			if (Parameters.random.nextDouble() < Parameters.mutateRate){
				double[] chromosomeArray;
				chromosomeArray = individual.chromosome;

				for (int i = chromosomeArray.length - 1; i > 0; i--) {
					int index = Parameters.random.nextInt(i);
					// swap
					double temp = chromosomeArray[index];
					chromosomeArray[index] = chromosomeArray[i];
					chromosomeArray[i] = temp;
				}
				individual.chromosome = chromosomeArray;
			}

		}
	}
	
	private void gaussianMutate(ArrayList<Individual> individuals) {
		for(Individual individual : individuals) {
			for (int i = 0; i < individual.chromosome.length; i++) {
				if (Parameters.random.nextDouble() < Parameters.mutateRate) {
					//calculate mean
					ArrayList<Double> stdDeviation = new ArrayList<>();
					double total = 0;
					for (int j = 0; j < individual.chromosome.length; j++) {
						total += individual.chromosome[j];
						stdDeviation.add(individual.chromosome[j]);
					}

					double mean = total/individual.chromosome.length;
					//calculate std. deviation
					double stddev = 0;
					for (int j = 0; j < stdDeviation.size(); j++) {
						stddev = stddev + ((stdDeviation.get(j) - mean) * (stdDeviation.get(j) - mean));
					}
					double squaredDiffMean = (stddev) / (stdDeviation.size());
					double standardDev = (Math.sqrt(squaredDiffMean));

					//actual mutation
					if (Parameters.random.nextBoolean()) {
						individual.chromosome[i] += (Parameters.random.nextGaussian()*standardDev+mean);
					} else {
						individual.chromosome[i] -= (Parameters.random.nextGaussian()*standardDev+mean);
					}
				}
			}
		}
	}
	/**
	 * 
	 * Replacement
	 * 
	 */
	private void replaceWorst(ArrayList<Individual> individuals) {   //tournament replacement (better diversity)
		for(Individual individual : individuals) {
			int idx = getWorstIndex(population);
			population.set(idx, individual);
		}		
	}

	private void replaceTournament(ArrayList<Individual> individuals){
		for(Individual individual : individuals) {
			ArrayList<Individual> tournamentParticipants = new ArrayList<>();
			Collections.shuffle(population);

			for (int i = 0; i < Parameters.tournamentSize; i++) {
				tournamentParticipants.add(population.get(i));
			}

			int worst_idx = getWorstIndex(tournamentParticipants);
			population.set(worst_idx, individual);
		}
	}
	

	/**
	 * Returns the index of the worst member of the population
	 * @return
	 *
	private int getWorstIndex() {
		Individual worst = null;
		int idx = -1;
		for (int i = 0; i < population.size(); i++) {
			Individual individual = population.get(i);
			if (worst == null) {
				worst = individual;
				idx = i;
			} else if (individual.fitness > worst.fitness) {
				worst = individual;
				idx = i; 
			}
		}
		return idx;
	}
	*/
	private int getWorstIndex(ArrayList<Individual> individuals) {
		Individual worst = null;
		int idx = -1;
		for (int i = 0; i < individuals.size(); i++) {
			Individual individual = individuals.get(i);
			if (worst == null) {
				worst = individual;
				idx = i;
			} else if (individual.fitness > worst.fitness) {
				worst = individual;
				idx = i;
			}
		}
		return idx;
	}

	@Override
	public double activationFunction(double x) {
		String activationChoice = Parameters.activationFunction;
		if (activationChoice == "tanh") {
			if (x < -20.0) {
				return -1.0;
			} else if (x > 20.0) {
				return 1.0;
			}
			return Math.tanh(x);
		}
		if (activationChoice == "leakyrelu"){
			if (x > 0) return x;
			else return 0.3*x; //https://www.tensorflow.org/api_docs/python/tf/keras/layers/LeakyReLU 0.3 used as default in keras
		}
		if (activationChoice == "relu") {
			if (x > 0) return x;
			else return 0;
		}
		if (activationChoice == "step"){
			if (x > 0) return 1;
			if (x < 0) return -1;
		}
		if (activationChoice == "ncu"){ //https://arxiv.org/pdf/2204.02921.pdf
			return x - Math.pow(x, 3);
		}
		if (activationChoice == "none") {
			return x;
		}

		return -1;
	}

}
