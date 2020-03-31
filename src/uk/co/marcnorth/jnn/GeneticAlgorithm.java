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
	  
	  CountDownLatch latch = new CountDownLatch(this.networks.length);
	  
	  final double[] scores = new double[this.networks.length];
	  
	  // Run the task on each network
	  for (int i = 0; i < this.networks.length; i++) {
	    
	    final int index = i;
	    
	    Runnable run = new Runnable() {
	      
	      private NeuralNetwork nn;
	      private NeuralNetworkTask task;
	      
	      public Runnable init(NeuralNetworkTask task, NeuralNetwork nn) {
	        
	        this.task = task;
	        this.nn = nn;
	        
	        return this;
	        
	      }
	      
        @Override
        public void run() {
           
          scores[index] = this.task.runTask(this.nn);
          
          latch.countDown();
          
        }
        
	    }.init(this.task, this.networks[i]);
	    
	    new Thread(run).start();
	    
	  }
	  
    try {
      
      latch.await();
      
    } catch (InterruptedException e) {
      
      // TODO Auto-generated catch block
      e.printStackTrace();
      
    }
    
    for (int i = 0; i < scores.length; i++)
      System.out.println(scores[i]);
    
	}
	
}
