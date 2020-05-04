package uk.co.marcnorth.jnn;

import java.util.Random;

import org.ejml.simple.SimpleMatrix;

import uk.co.marcnorth.jnn.NeuralNetwork.Init;

public class Layer {
  
  private final SimpleMatrix weights;
  private final SimpleMatrix biases;
  
  public Layer(int numNodes, int previousLayerNumNodes, Init init) {
    this.biases = new SimpleMatrix(numNodes, 1);
    this.weights = new SimpleMatrix(numNodes, previousLayerNumNodes);
    if (init == Init.RANDOM)
      initializeRandomParameters();
  }
  
  public Layer(SimpleMatrix weights, SimpleMatrix biases) {
    if (!biases.isVector())
      throw new RuntimeException("Biases should be a vector");
    this.weights = new SimpleMatrix(weights);
    this.biases = new SimpleMatrix(biases);
  }
  
  private void initializeRandomParameters() {
    Random rng = new Random();

    for (int row = 0; row < weights.numRows(); row++)
      for (int column = 0; column < weights.numCols(); column++)
        weights.set(row, column, rng.nextDouble() * 2 - 1);
    
    for (int row = 0; row < biases.numRows(); row++)
      for (int column = 0; column < biases.numCols(); column++)
        biases.set(row, column, rng.nextDouble() * 2 - 1);
  }
  
  public SimpleMatrix getWeights() {
    return weights;
  }

  public SimpleMatrix getBiases() {
    return biases;
  }
  
  public SimpleMatrix feedForward(SimpleMatrix inputs) {
    return activate(weights.mult(inputs).plus(biases));
  }
  
  private SimpleMatrix activate(SimpleMatrix m) {
    for (int r = 0; r < m.numRows(); r++)
      m.set(r, 0, Math.tanh(m.get(r, 0)));
    return m;
  }
  
}