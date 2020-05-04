package uk.co.marcnorth.jnn;

import java.util.Random;
import org.ejml.simple.SimpleMatrix;
import javafx.util.Pair;

public class NeuralNetworkMutator {

  private final int numberOfActiveLayers;
  private final SimpleMatrix[] weights;
  private final SimpleMatrix[] biases;
  private final Random rng = new Random();
  
  public NeuralNetworkMutator(NeuralNetwork network) {
    this.numberOfActiveLayers = network.getNumActiveLayers();
    Pair<SimpleMatrix[], SimpleMatrix[]> weightsAndBiases = getWeightsAndBiasesFromNetwork(network);
    this.weights = weightsAndBiases.getKey();
    this.biases = weightsAndBiases.getValue();
  }
  
  public NeuralNetwork getNetwork() {
    return new NeuralNetwork(weights, biases);
  }
  
  public NeuralNetworkMutator crossoverWith(NeuralNetwork network) {
    Pair<SimpleMatrix[], SimpleMatrix[]> otherWeightsAndBiases = getWeightsAndBiasesFromNetwork(network);
    SimpleMatrix[] otherWeights = otherWeightsAndBiases.getKey();
    SimpleMatrix[] otherBiases = otherWeightsAndBiases.getValue();
    for (int i = 0; i < numberOfActiveLayers; i++) {
      weights[i] = crossoverMatrices(weights[i], otherWeights[i]);
      biases[i] = crossoverMatrices(biases[i], otherBiases[i]);
    }
    return this;
  }
  
  private SimpleMatrix crossoverMatrices(SimpleMatrix matrix1, SimpleMatrix matrix2) {
    SimpleMatrix child = new SimpleMatrix(matrix1.numRows(), matrix1.numCols());
    for (int row = 0; row < matrix1.numRows(); row++) {
      for (int column = 0; column < matrix1.numCols(); column++) {
        child.set(
          row,
          column,
          rng.nextDouble() < 0.5 ? matrix1.get(row, column) : matrix2.get(row, column)
        );
      }
    }
    return child;
  }
  
  public NeuralNetworkMutator mutate(double mutationRate) {
    for (int i = 0; i < numberOfActiveLayers; i++) {
      mutateMatrix(weights[i], mutationRate);
      mutateMatrix(biases[i], mutationRate);
    }
    return this;
  }
  
  private void mutateMatrix(SimpleMatrix m, double mutationRate) {
    for (int row = 0; row < m.numRows(); row++)
      for (int column = 0; column < m.numCols(); column++)
        if (rng.nextDouble() < mutationRate)
          m.set(row, column, rng.nextDouble() * 2 - 1);
  }
  
  private Pair<SimpleMatrix[], SimpleMatrix[]> getWeightsAndBiasesFromNetwork(NeuralNetwork network) {
    int numberOfActiveLayers = network.getNumActiveLayers();
    SimpleMatrix[] weights = new SimpleMatrix[numberOfActiveLayers];
    SimpleMatrix[] biases = new SimpleMatrix[numberOfActiveLayers];
    for (int i = 0; i < numberOfActiveLayers; i++) {
      weights[i] = network.getWeightsForActiveLayer(i);
      biases[i] = network.getBiasesForActiveLayer(i);
    }
    return new Pair<SimpleMatrix[], SimpleMatrix[]>(weights, biases);
  }
  
}