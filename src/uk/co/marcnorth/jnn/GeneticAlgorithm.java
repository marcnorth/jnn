package uk.co.marcnorth.jnn;

import java.util.Arrays;
import java.util.Random;
import java.util.concurrent.CountDownLatch;

import org.ejml.simple.SimpleMatrix;

public class GeneticAlgorithm {
  
  /**
   * The number of heighest scoring networks to keep for the next generation
   */
  private int numKeep = 4;
  
  /**
   * The number of heighest scoring networks to clone (copy with mutations)
   */
  private int numClone = 4;
  
  /**
   * The number of heighest scoring networks to breed
   */
  private int numBreed = 4;
  
  /**
   * Percentage of weight/biases to randomly change when mutating
   */
  private double mutationRate = 0.3;
  
	private int currentGeneration = 0;
	private NeuralNetwork[] networks;
	private NeuralNetworkTask task;
	
	public GeneticAlgorithm(NeuralNetwork[] networks, NeuralNetworkTask task, int numKeep, int numClone, int numBreed, double mutationRate) {
	  
	  this.networks = networks;
	  this.task = task;
	  
	  this.numKeep = numKeep;
	  this.numClone = numClone;
	  this.numBreed = numBreed;
	  this.mutationRate = mutationRate;
	  
	}
	
	public void runGenerations(int n) {
	  
	  for (int i = 0; i < n; i++) {
	    
	    this.nextGeneration();
	    
	  }
	  
	}
	
	private void nextGeneration() {
	  
	  this.currentGeneration++;
	  
	  CountDownLatch latch = new CountDownLatch(this.networks.length);
	  
	  final NeuralNetworkTaskInstance[] taskInstances = new NeuralNetworkTaskInstance[this.networks.length];
	  
	  // Run the task on each network
	  for (int i = 0; i < this.networks.length; i++) {
	    
	    taskInstances [i] = new NeuralNetworkTaskInstance(this.task, this.networks[i], latch);
	    
	    new Thread(taskInstances[i]).start();
	    
	  }
	  
    try {
      
      latch.await();
      
    } catch (InterruptedException e) {
      
      // TODO Auto-generated catch block
      e.printStackTrace();
      
    }
    
    // Sort instances by score
    Arrays.sort(taskInstances);
    
    NeuralNetworkTaskInstance[] taskInstancesOrdered = new NeuralNetworkTaskInstance[taskInstances.length];
    
    for (int i = 0; i < taskInstances.length; i++)
      taskInstancesOrdered[i] = taskInstances[taskInstances.length - i - 1];
    
    double maxScore = taskInstancesOrdered[0].getScore();
    
    this.createNewGeneration(taskInstancesOrdered);
    
	}
	
	/**
	 * Replaces this.networks with a new generation based on given scores
	 * @param scores
	 */
	private void createNewGeneration(NeuralNetworkTaskInstance[] taskInstances) {
	  
	  NeuralNetwork[] nextGeneration = new NeuralNetwork[this.networks.length];
	  
	  // Keep top networks for next generation
	  for (int i = 0; i < this.numKeep; i++) {
	    
	    nextGeneration[i] = taskInstances[i].getNetwork();
	    
	  }
	  
	  // Clone networks for next generation
	  for (int i = 0; i < this.numClone; i++) {
	    
	    int index = this.numKeep + i;
	    
	    nextGeneration[index] = this.mutate(taskInstances[i].getNetwork());
	    
	  }
	  
	  // Breed networks for next generation
	  Random r = new Random();
	  
	  for (int i = 0; i < this.numBreed; i++) {
	    
	    int index = this.numKeep + this.numClone + i;
	    
	    // Select two networks from the networks being kept
      int parent1 = r.nextInt(this.numKeep);
	    int parent2;
      
      do {
        
        parent2 = r.nextInt(this.numKeep);
        
      } while (parent2 == parent1 || this.numKeep < 2);
      
	    nextGeneration[index] = this.breed(taskInstances[parent1].getNetwork(), taskInstances[parent2].getNetwork());
	    
	  }
	  
	  // Fill rest of next generation with random networks
	  for (int i = 0; i < nextGeneration.length - this.numKeep - this.numClone - this.numBreed; i++) {
	    
	    int index = this.numKeep + this.numClone + this.numBreed + i;
	    
	    nextGeneration[index] = new NeuralNetwork(this.networks[0].getLayerSizes(), NeuralNetwork.Init.RANDOM);
	    
	  }
	  
	  this.networks = nextGeneration;
	  
	}
	
	/**
	 * Breeds two networks, then mutates and returns child
	 * @return
	 */
	private NeuralNetwork breed(NeuralNetwork nn1, NeuralNetwork nn2) {
	  
    int numActiveLayers = nn1.getNumActiveLayers();

    SimpleMatrix[] weights = new SimpleMatrix[numActiveLayers];
    SimpleMatrix[] biases = new SimpleMatrix[numActiveLayers];
    
    Random rand = new Random();
    
    for (int l = 0; l < numActiveLayers; l++) {
      
      SimpleMatrix[] layerWeights = {
          nn1.getWeightsForLayer(l + 1),
          nn2.getWeightsForLayer(l + 1),
      };
      
      SimpleMatrix[] layerBiases = {
          nn1.getBiasesForLayer(l + 1),
          nn2.getBiasesForLayer(l + 1),
      };
      
      weights[l] = new SimpleMatrix(layerWeights[0].numRows(), layerWeights[0].numCols());
      biases[l] = new SimpleMatrix(layerBiases[0].numRows(), layerBiases[0].numCols());
      
      // For each weight/bias select from either network at random
      for (int r = 0; r < weights[l].numRows(); r++) {
        
        for (int c = 0; c < weights[l].numCols(); c++) {
          
          int nnIndex = rand.nextInt(layerWeights.length);
          
          weights[l].set(r, c, layerWeights[nnIndex].get(r, c));
          
        }
        
      }
      
      for (int r = 0; r < biases[l].numRows(); r++) {
        
        for (int c = 0; c < biases[l].numCols(); c++) {
          
          int nnIndex = rand.nextInt(layerBiases.length);
          
          biases[l].set(r, c, layerBiases[nnIndex].get(r, c));
          
        }
        
      }
      
    }

    NeuralNetwork child = new NeuralNetwork(weights, biases);
    
    return this.mutate(child);
    
	}
	
	/**
	 * Returns a mutated copy of a network
	 * Randomly changes weights/biases of a network based on mutation rate
	 * @param nn
	 */
	private NeuralNetwork mutate(NeuralNetwork nn) {
	  
	  int numActiveLayers = nn.getNumActiveLayers();

    SimpleMatrix[] weights = new SimpleMatrix[numActiveLayers];
    SimpleMatrix[] biases = new SimpleMatrix[numActiveLayers];
	  
    for (int l = 0; l < numActiveLayers; l++) {
      
      SimpleMatrix layerWeights = nn.getWeightsForLayer(l + 1);
      SimpleMatrix layerBiases = nn.getBiasesForLayer(l + 1);
      
      // Loop through and mutate
      this.mutateMatrix(layerWeights, this.mutationRate);
      
      weights[l] = layerWeights;
      biases[l] = layerBiases;
      
    }
    
    return new NeuralNetwork(weights, biases);
    
	}
	
	private void mutateMatrix(SimpleMatrix m, double mutationRate) {
	  
	  Random rand = new Random();
	  
	  for (int r = 0; r < m.numRows(); r++)
      for (int c = 0; c < m.numCols(); c++)
        if (rand.nextDouble() < mutationRate)
          m.set(r, c, rand.nextDouble() * 2 - 1);
	  
	}
	
	/**
	 * An instance of a task being run with a specific network
	 */
	private class NeuralNetworkTaskInstance implements Runnable, Comparable<NeuralNetworkTaskInstance> {
	  
	  private NeuralNetworkTask task;
	  private NeuralNetwork nn;
	  private CountDownLatch latch;
	  private double score;
	  
	  public NeuralNetworkTaskInstance(NeuralNetworkTask task, NeuralNetwork nn, CountDownLatch latch) {
      
      this.task = task;
      this.nn = nn;
      
      this.latch = latch;
      
    }
	  
	  public double getScore() {
	    
	    return this.score;
	    
	  }
	  
	  public NeuralNetwork getNetwork() {
	    
	    return this.nn;
	    
	  }
	  
    @Override
    public void run() {
      
      this.score = this.task.runTask(nn);
      
      this.latch.countDown();
      
    }

    @Override
    public int compareTo(NeuralNetworkTaskInstance other) {
      
      double diff = this.score - other.getScore();
      
      if (diff > 0)
        return 1;
      else if (diff < 0)
        return -1;
      else
        return 0;
      
    }
	  
	}
	
}
