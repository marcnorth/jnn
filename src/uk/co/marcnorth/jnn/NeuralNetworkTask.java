package uk.co.marcnorth.jnn;

public interface NeuralNetworkTask {
	
  /**
   * Runs the task with the given network and returns the network's score
   * @param network
   * @return The network's score for the task
   */
	public double runTask(NeuralNetwork network);
	
}
