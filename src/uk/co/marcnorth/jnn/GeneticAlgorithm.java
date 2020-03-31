package uk.co.marcnorth.jnn;

import java.util.concurrent.CountDownLatch;

public class GeneticAlgorithm {
  
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
	  
	  final CountDownLatch latch = new CountDownLatch(this.networks.length);
	  
	  // Run the task on each network
	  for (int i = 0; i < this.networks.length; i++) {
	    
	    Runnable run = new Runnable() {

        @Override
        public void run() {
           
          //this.task.runTask(this.networks[i]);
          
          latch.countDown();
          
        }
        
	    };
	    
	    new Thread(run).start();
	    
	  }
	  
    try {
      
      latch.await();
      
    } catch (InterruptedException e) {
      
      // TODO Auto-generated catch block
      e.printStackTrace();
      
    }
    
	}
	
}
