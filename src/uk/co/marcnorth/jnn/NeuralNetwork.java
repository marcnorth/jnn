package uk.co.marcnorth.jnn;

import org.ejml.simple.SimpleMatrix;

public class NeuralNetwork {
  
  public enum Init {
    ZERO,
    RANDOM,
  }
  private final Layer[] activeLayers;
  private final int inputSize;
  private final int[] layerSizes;
  
  public NeuralNetwork (int[] layerSizes) {
    this(layerSizes, Init.ZERO);
  }
  
  public NeuralNetwork(int[] layerSizes, Init init) {
    if (layerSizes.length < 2)
      throw new RuntimeException("Network must have at least two layers");
    if (layerSizes[0] <= 0)
      throw new RuntimeException("Network input size must be greater than 0");
    this.layerSizes = layerSizes;
  	this.inputSize = layerSizes[0];
  	this.activeLayers = new Layer[layerSizes.length - 1];
  	for (int i = 1; i < layerSizes.length; i++)
  		this.activeLayers[i-1] = new Layer(layerSizes[i], layerSizes[i-1], init);
  }
  
  public NeuralNetwork(double[][][] weights, double[][][] biases) {
    SimpleMatrix[] matrixWeights = new SimpleMatrix[weights.length];
    SimpleMatrix[] matrixBiases = new SimpleMatrix[biases.length];
    for (int i = 0; i < weights.length; i++)
      matrixWeights[i] = new SimpleMatrix(weights[i]);
    for (int i = 0; i < biases.length; i++)
      matrixWeights[i] = new SimpleMatrix(biases[i]);
    inputSize = matrixWeights[0].numCols();
    layerSizes = new int[weights.length + 1];
    layerSizes[0] = this.inputSize;
    activeLayers = new Layer[biases.length];
    setWeightsAndBiases(matrixWeights, matrixBiases);
  }
  
  public NeuralNetwork(SimpleMatrix[] weights, SimpleMatrix[] biases) {
    inputSize = weights[0].numCols();
    layerSizes = new int[weights.length + 1];
    layerSizes[0] = this.inputSize;
    activeLayers = new Layer[biases.length];
    setWeightsAndBiases(weights, biases);
  }
  
  private void setWeightsAndBiases(SimpleMatrix[] weights, SimpleMatrix[] biases) {
    int layerSize;
    int previousLayerSize = inputSize;
    for (int i = 0; i < this.activeLayers.length; i++) {
      layerSize = weights[i].numRows();
      layerSizes[i+1] = layerSize;
      if (weights[i].numCols() != previousLayerSize)
        throw new RuntimeException("Layer " + (i+1) + " weights columns (" + weights[i].numCols() + ") does not match previous layer size: " + previousLayerSize);
      if (biases[i].numRows() != layerSize)
        throw new RuntimeException("Layer " + (i+1) + " biases rows (" + biases[i].numRows() + ") does not match layer size: " + layerSize);
      activeLayers[i] = new Layer(weights[i], biases[i]);
      previousLayerSize = layerSize;
    }
  }
  
  public int getInputSize() {
    return this.inputSize;
  }
  
  public int[] getLayerSizes() {
    return this.layerSizes;
  }
  
  public int getNumActiveLayers() {
    return this.activeLayers.length;
  }
  
  public SimpleMatrix getWeightsForActiveLayer(int layerIndex) {
  	return new SimpleMatrix(activeLayers[layerIndex].getWeights());
  }
  
  public SimpleMatrix getBiasesForActiveLayer(int layerIndex) {
  	return new SimpleMatrix(activeLayers[layerIndex].getBiases());
  }
  
  public SimpleMatrix feedForward(SimpleMatrix inputs) {
  	if (!inputs.isVector())
  		throw new RuntimeException("Inputs must be a vector");
  	if (inputs.numRows() != inputSize)
  		throw new RuntimeException("Inputs length (" + inputs.numRows() + ") must match network input size (" + inputSize + ")");
  	SimpleMatrix values = inputs;
  	for (int i = 0; i < activeLayers.length; i++)
  		values = activeLayers[i].feedForward(values);
  	return values;
  }
  
  public double[] feedForward(double[] inputs) {
    double[][] formattedInputs = new double[inputs.length][1];
    for (int i = 0; i < inputs.length; i++)
      formattedInputs[i][0] = inputs[i];
    SimpleMatrix outputs = feedForward(new SimpleMatrix(formattedInputs));
    double[] outputColumn = new double[outputs.numRows()];
    for (int i = 0; i < outputs.numRows(); i++)
      outputColumn[i] = outputs.get(i, 0);
    return outputColumn;
  }
  
}