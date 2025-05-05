# Dataset batching  
  
https://machinelearningmastery.com/difference-between-a-batch-and-an-epoch/  
  
  
-batch size - hyperparameter of gradient descent that controls the number of training examples to work through  
before the model's internal parameters are updated  
-number of epochs - hyperparameter of gradient descent that controls the number of complete passes through the training dataset  
  
-a training dataset can be divided into one or more batches  
-at the end of each batch, the predictions (of that batch) are compared to the expected output variables and an error is calculated  
&nbsp;&nbsp;&nbsp;-using this error model is improved, ie. moved further along the error gradient  
  
-usually common reasons are used for mini-batching :  
&nbsp;&nbsp;&nbsp;-memory efficiency - massive datasets are more easily handled if split into  
&nbsp;&nbsp;&nbsp;-speed - smaller batches are easier and faster to handle when training a network  
  
-but what is it about mini-batching that it improves the learning process?  
&nbsp;&nbsp;&nbsp;-gradient approximation - mini-batching provides a good trade-off by approximating the gradient efficiently while still capturing enough variability to generalize well  
&nbsp;&nbsp;&nbsp;-noise injection - each mini-batch possesses enough noise in it to help the optimization process by providing enough info for a better generalization  
&nbsp;&nbsp;&nbsp;-parallelization - mini-batches nicely allow for parallelization using modern hardware  
&nbsp;&nbsp;&nbsp;-learning stability - mini-batches smooth out extreme gradient fluctuations that might occur with single-sample updates
