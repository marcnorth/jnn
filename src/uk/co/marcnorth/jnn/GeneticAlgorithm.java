package uk.co.marcnorth.jnn;

import java.util.ArrayList;
import java.util.List;

public class GeneticAlgorithm {
  
  private final Class<? extends NeuralNetworkTask> taskClass;
  private final int[] networkLayerSizes;
  private final int numberOfNetworks;
  private final int numberToKeepBetweenGenerations;
  private final int numberToCloneBetweenGenerations;
  private final int numberToBreedBetweenGenerations;
  private final double mutationRate;
  private final List<GeneticAlgorithmListener> listeners = new ArrayList<>();
  private Generation currentGeneration;
  private int currentGenerationNumber = 0;
  private double highestScoreOfAnyGeneration = Double.MIN_VALUE;
  
	public GeneticAlgorithm(
	  Class<? extends NeuralNetworkTask> taskClass,
	  int[] networkLayerSizes,
	  int numberOfNetworks,
	  int numberToKeepBetweenGenerations,
	  int numberToCloneBetweenGenerations,
	  int numberToBreedBetweenGenerations,
	  double mutationRate
	) {
	  if (numberToKeepBetweenGenerations + numberToCloneBetweenGenerations + numberToBreedBetweenGenerations > numberOfNetworks)
      throw new IllegalArgumentException("numberOfNetworks can not be smaller that the number of networks being passed to the next generation");
    this.taskClass = taskClass;
	  this.networkLayerSizes = networkLayerSizes;
	  this.numberOfNetworks = numberOfNetworks;
    this.numberToKeepBetweenGenerations = numberToKeepBetweenGenerations;
    this.numberToCloneBetweenGenerations = numberToCloneBetweenGenerations;
    this.numberToBreedBetweenGenerations = numberToBreedBetweenGenerations;
    this.mutationRate = mutationRate;
	}

  public int getCurrentGenerationNumber() {
    return currentGenerationNumber;
  }
  
  public int getNumberOfNetworks() {
    return numberOfNetworks;
  }
  
	public void addListener(GeneticAlgorithmListener listener) {
	  listeners.add(listener);
	}

  public double getHighestScoreOfAnyGeneration() {
    return highestScoreOfAnyGeneration;
  }
  
	public void runGenerations(int numberOfGenerations) {
	  for (int i = 0; i < numberOfGenerations; i++)
	    runGeneration();
	}
	
	private void runGeneration() {
    currentGenerationNumber++;
	  triggerListenersOnGenerationStart();
	  createNextGeneration();
	  currentGeneration.start();
	  currentGeneration.await();
	  highestScoreOfAnyGeneration = Math.max(
	    highestScoreOfAnyGeneration,
	    currentGeneration.getHighestScore()
	  );
	  triggerListenersOnGenerationEnd();
	}
	
	private void createNextGeneration() {
	  if (currentGeneration == null) {
	    currentGeneration = Generation.createGenerationWithRandomNetworks(
	      taskClass,
	      numberOfNetworks,
	      networkLayerSizes
	    );
	  } else {
	    currentGeneration = currentGeneration.createNextGeneration(
	      numberOfNetworks,
        numberToKeepBetweenGenerations,
        numberToCloneBetweenGenerations,
        numberToBreedBetweenGenerations,
        mutationRate
	    );
	  }
	}
	
  private void triggerListenersOnGenerationStart() {
    for (GeneticAlgorithmListener listener : this.listeners)
      listener.onGenerationStart();
  }

  private void triggerListenersOnGenerationEnd() {
    for (GeneticAlgorithmListener listener : this.listeners)
      listener.onGenerationEnd();
  }
  
}