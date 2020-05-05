package uk.co.marcnorth.jnn;

import java.lang.reflect.InvocationTargetException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;
import java.util.concurrent.CountDownLatch;
import javafx.util.Pair;

public class Generation {
  
  private final Class<? extends NeuralNetworkTask> taskClass;
  private final int[] networkLayerSizes;
  private final NeuralNetwork[] networks;
  private final List<Pair<NeuralNetwork, Double>> scores = Collections.synchronizedList(new ArrayList<>());
  private boolean hasStarted = false;
  private boolean sortedScores = false;
  private final CountDownLatch finishedCountDown;
  
  private Generation(Class<? extends NeuralNetworkTask> taskClass, NeuralNetwork[] networks, int[] networkLayerSizes) {
    this.taskClass = taskClass;
    this.networks = networks;
    this.networkLayerSizes = networkLayerSizes;
    this.finishedCountDown = new CountDownLatch(networks.length);
  }
  
  public Class<? extends NeuralNetworkTask> getTaskClass() {
    return taskClass;
  }
  
  public void start() {
    if (hasStarted)
      throw new RuntimeException("Generation has already started");
    hasStarted = true;
    for (NeuralNetwork network : networks)
      startTask(network);
  }
  
  private void startTask(NeuralNetwork network) {
    new Thread(() -> {
      try {
        NeuralNetworkTask task = this.taskClass.getDeclaredConstructor().newInstance();
        scores.add(new Pair<>(
          network,
          task.getNetworkScore(network)
        ));
        finishedCountDown.countDown();
      } catch (InstantiationException | IllegalAccessException | IllegalArgumentException | InvocationTargetException
          | NoSuchMethodException | SecurityException e) {
        e.printStackTrace();
        throw new RuntimeException("Starting task failed");
      }
    }).start();
  }
  
  public void await() {
    try {
      finishedCountDown.await();
    } catch (InterruptedException e) {
      throw new RuntimeException("Unexpected interrupt");
    }
  }
  
  public double getHighestScore() {
    sortScores();
    return scores.size() > 0 ? scores.get(0).getValue() : null;
  }
  
  private void sortScores() {
    if (sortedScores)
      return;
    await();
    Collections.sort(scores, (networkScore1, networkScore2) -> {
      double difference = networkScore1.getValue() - networkScore2.getValue();
      if (difference > 0)
        return -1;
      else if (difference < 0)
        return 1;
      else
        return 0;
    });
    sortedScores = true;
  }
  
  public Generation createNextGeneration(
    int numberOfNetworks,
    int numberToKeepBetweenGenerations,
    int numberToCloneBetweenGenerations,
    int numberToBreedBetweenGenerations,
    double mutationRate
  ) {
    sortScores();
    
    NeuralNetwork[] networks = new NeuralNetwork[numberOfNetworks];
    Random random = new Random();
    int index = 0;
    
    for (int i = 0; i < numberToKeepBetweenGenerations; i++) {
      networks[index] = getNetworkByRank(i);
      index++;
    }
    
    for (int i = 0; i < numberToCloneBetweenGenerations; i++) {
      networks[index] = new NeuralNetworkMutator(getNetworkByRank(i))
        .mutate(mutationRate)
        .getNetwork();
      index++;
    }
    
    for (int i = 0; i < numberToBreedBetweenGenerations; i++) {
      NeuralNetwork parent1 = getNetworkByRank(random.nextInt(numberToKeepBetweenGenerations));
      NeuralNetwork parent2 = getNetworkByRank(random.nextInt(numberToKeepBetweenGenerations));
      networks[index] = new NeuralNetworkMutator(parent1)
        .crossoverWith(parent2)
        .getNetwork();
      index++;
    }
    
    while (index < numberOfNetworks) {
      networks[index] = new NeuralNetwork(networkLayerSizes, NeuralNetwork.Init.RANDOM);
      index++;
    }
    
    return new Generation(taskClass, networks, networkLayerSizes);
  }
  
  private NeuralNetwork getNetworkByRank(int rank) {
    sortScores();
    return scores.get(rank).getKey();
  }
  
  public static Generation createGenerationWithRandomNetworks(Class<? extends NeuralNetworkTask> taskClass, int numberOfNetworks, int[] networkLayerSizes) {
    NeuralNetwork[] networks = new NeuralNetwork[numberOfNetworks];
    for (int i = 0; i < networks.length; i++)
      networks[i] = new NeuralNetwork(networkLayerSizes, NeuralNetwork.Init.RANDOM);
    return new Generation(taskClass, networks, networkLayerSizes);
    
  }
  
}