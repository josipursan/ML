# Learning rate schedulers  
https://neptune.ai/blog/how-to-choose-a-learning-rate-scheduler  
  
  
-a *learning rate schedule* is a predefined framework that adjusts the learning rate between epochs  
-there are two common approaches to learning rate schedules :  
&nbsp;&nbsp;&nbsp;-constant learning rate - we initialize a learning rate and don't change it during training  
&nbsp;&nbsp;&nbsp;-learning rate decay - we select an initial training rate, and then gradually reduce it in accordance to the chosen learning rate scheduler  
  
-learning rate decay scheduler can further be divided into other categories :  
&nbsp;&nbsp;&nbsp;-exponential decay - reduces the learning rate exponentially over time  
&nbsp;&nbsp;&nbsp;-step decay - reduces the learning rate in steps after a fixed number of epochs  
&nbsp;&nbsp;&nbsp;-cyclic learning rates (CLR) - alternates between lower and higher learning rates during training; very useful for exploration and escaping local minima regions  
&nbsp;&nbsp;&nbsp;
