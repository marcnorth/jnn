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
		if (layerSizes[0] <= 0)
			throw new RuntimeException("Network input size must be greater than 0");
		
		this.inputSize = layerSizes[0];
		
		// Create hidden/output layers
		this.activeLayers = new Layer[layerSizes.length - 1];
		
		for (int i = 1; i < layerSizes.length; i++) {
			
			this.activeLayers[i-1] = new Layer(layerSizes[i], layerSizes[i-1]);
			
		}
		
	}
	
	public NeuralNetwork(SimpleMatrix[] weights, SimpleMatrix[] biases) {
		
		this.inputSize = weights[0].numCols();

		// Create hidden/output layers
		this.activeLayers = new Layer[biases.length];
		
		int layerSize;
		int previousLayerSize = this.inputSize;
		
		for (int i = 0; i < this.activeLayers.length; i++) {
			
			layerSize = weights[i].numRows();

			if (weights[i].numCols() != previousLayerSize)
				throw new RuntimeException("Layer " + (i+1) + " weights columns (" + weights[i].numCols() + ") does not match previous layer size: " + previousLayerSize);

			if (biases[i].numRows() != layerSize)
				throw new RuntimeException("Layer " + (i+1) + " biases rows (" + biases[i].numRows() + ") does not match layer size: " + layerSize);
			
			this.activeLayers[i] = new Layer(weights[i], biases[i]);
			
		}
		
	}
	
	/**
	 * @param layerIndex The layer to get the weights of (zero-based including the input layer, which doesn't really exist, so the first active layer index is 1)
	 * @return A copy of the weights matrix
	 */
	public SimpleMatrix getWeightsForLayer(int layerIndex) {
		
		return new SimpleMatrix(this.activeLayers[layerIndex - 1].weights);
		
	}

	/**
	 * @param layerIndex The layer to get the biases of (zero-based including the input layer, which doesn't really exist, so the first active layer index is 1)
	 * @return A copy of the biases matrix
	 */
	public SimpleMatrix getBiasesForLayer(int layerIndex) {
		
		return new SimpleMatrix(this.activeLayers[layerIndex - 1].biases);
		
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
	
	/**
	 * Prints information about the network
	 */
	public void print() {
		
		System.out.printf("%d", this.inputSize);
		
		for (int i = 0; i < this.activeLayers.length; i++)
			System.out.printf(", %d", this.activeLayers[i].size);
		
		System.out.println();
		
	}
	
	private class Layer {
		
		private int size;
		private SimpleMatrix weights;
		private SimpleMatrix biases;
		
		/**
		 * @param numNodes The number of nodes in this layer
		 * @param previousLayerNumNodes The number of nodes in the previous layer (i.e. how many inputs each node in this layer will have)
		 */
		public Layer(int numNodes, int previousLayerNumNodes) {
			
			this.size = numNodes;
			
			this.biases = new SimpleMatrix(numNodes, 1);
			this.weights = new SimpleMatrix(numNodes, previousLayerNumNodes);
			
		}
		
		public Layer(SimpleMatrix weights, SimpleMatrix biases) {
			
			if (!biases.isVector())
				throw new RuntimeException("Biases should be a vector");
			
			this.size = weights.numRows();
			
			this.weights = new SimpleMatrix(weights);
			this.biases = new SimpleMatrix(biases);
			
		}
		
		public SimpleMatrix feedForward(SimpleMatrix inputs) {
			
			this.weights.print();
			this.weights.mult(inputs).print();
			this.biases.print();
			
			return this.weights.mult(inputs).plus(this.biases);
			
		}
		
	}
	
}