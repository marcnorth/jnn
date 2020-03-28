package uk.co.marcnorth.jnn;

import org.ejml.simple.SimpleMatrix;

public class NeuralNetwork {
	
	/**
	 * The hidden layers and output layer
	 */
	private Layer[] activeLayers;
	
	private int inputSize;
	
	public NeuralNetwork(int[] layerSizes) {
		
		if (layerSizes.length < 2)
			throw new RuntimeException("Network must have at least two layers");
		
		// Set input size
		this.setInputSize(layerSizes[0]);
		
		// Create hidden/output layers
		this.activeLayers = new Layer[layerSizes.length - 1];
		
		for (int i = 1; i < layerSizes.length; i++) {
			
			this.activeLayers[i-1] = new Layer(layerSizes[i], layerSizes[i-1]);
			
		}
		
	}
	
	/**
	 * Feeds the given inputs through the network
	 * @param inputs
	 * @return Output of the output layer
	 */
	public SimpleMatrix feedForward(SimpleMatrix inputs) {

		if (!inputs.isVector())
			throw new RuntimeException("Inputs must be a vector");

		if (inputs.numRows() != this.inputSize)
			throw new RuntimeException("Inputs length (" + inputs.numRows() + ") must match network input size (" + this.inputSize + ")");
		
		SimpleMatrix values = inputs;
		
		// Feed forward through each layer
		for (int i = 0; i < this.activeLayers.length; i++) {
			
			values = this.activeLayers[i].feedForward(values);
			
		}
		
		return values;
		
	}
	
	private void setInputSize(int size) {
		
		if (size <= 0)
			throw new RuntimeException("Network input size must be greater than 0");
		
		this.inputSize = size;
		
	}
	
	private class Layer {
		
		private SimpleMatrix[] weights;
		private SimpleMatrix biases;
		
		/**
		 * @param numNodes The number of nodes in this layer
		 * @param previousLayerNumNodes The number of nodes in the previous layer (i.e. how many inputs each node in this layer will have)
		 */
		public Layer(int numNodes, int previousLayerNumNodes) {
			
			this.biases = new SimpleMatrix(numNodes, 1);
			
			this.weights = new SimpleMatrix[numNodes];
			
			for (int i = 0; i < numNodes; i++)
				this.weights[i] = new SimpleMatrix(numNodes, previousLayerNumNodes);
			
		}
		
		public SimpleMatrix feedForward(SimpleMatrix inputs) {
			
			return inputs;
			
		}
		
	}
	
}