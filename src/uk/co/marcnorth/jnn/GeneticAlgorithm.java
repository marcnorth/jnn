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
	
	public GeneticAlgorithm(NeuralNetwork[] networks, NeuralNetworkTask task) {
	  
	  this.networks = networks;
	  this.task = task;
	  
	}
	
	public void runGenerations(int n) {
	  
	  for (int i = 0; i < n; i++) {
	    
	    this.nextGeneration();
	    
	  }
	  
	}
	
	private void nextGeneration() {
	  
	  this.currentGeneration++;
	  
	  CountDownLatch latch = new CountDownLatch(this.networks.length);
	  
	  final NeuralNetworkTaskInstance[] taskScores = new NeuralNetworkTaskInstance[this.networks.length];
	  
	  // Run the task on each network
	  for (int i = 0; i < this.networks.length; i++) {
	    
	    taskScores[i] = new NeuralNetworkTaskInstance(this.task, this.networks[i], latch);
	    
	    new Thread(taskScores[i]).start();
	    
	  }
	  
    try {
      
      latch.await();
      
    } catch (InterruptedException e) {
      
      // TODO Auto-generated catch block
      e.printStackTrace();
      
    }
    
    this.createNewGeneration(taskScores);
    
	}
	
	/**
	 * Replaces this.networks with a new generation based on given scores
	 * @param scores
	 */
	private void createNewGeneration(NeuralNetworkTaskInstance[] taskInstances) {
	  
    // Sort instances by score
	  Arrays.sort(taskInstances);
	  
	  NeuralNetwork[] nextGeneration = new NeuralNetwork[this.networks.length];
	  
	  // Keep top networks for next generation
	  System.arraycopy(this.networks, 0, nextGeneration, 0, this.numKeep);
	  
	  // Clone networks for next generation
	  for (int i = 0; i < this.numClone; i++) {
	    
	    int index = this.numKeep + i;
	    
	    nextGeneration[index] = this.mutate(taskInstances[i].getNetwork());
	    
	  }
	  
    for (int i = 0; i < nextGeneration.length; i++)
      System.out.println(nextGeneration[i]);
    
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
