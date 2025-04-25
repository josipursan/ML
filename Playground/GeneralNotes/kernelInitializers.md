# What are kernel intializers?  
  
-kernel initializers are used to set the starting weights before the training process begins  
  
-here are some of the key benefits of kernel initialization :  
&nbsp;&nbsp;&nbsp;-aids optimization stability  
&nbsp;&nbsp;&nbsp;-can reduce the time it takes to converge  
&nbsp;&nbsp;&nbsp;-avoid symmetry - symmetry usually causes neurons to learn identical patterns, which isn't something we want  

https://datascience.stackexchange.com/questions/37378/what-are-kernel-initializers-and-what-is-their-significance :  
-the neural network must start with some weights, which are then iteratively updated and made better  
-the term *kernel initializer* is just a different way of saying *statistical distribution*  
-whichever statistical distribution we choose gets used to generate values, thus populating the initially empty matrix  
  
https://machinelearningmastery.com/weight-initialization-for-deep-learning-neural-networks/ : 
-generally speaking, initializing a matrix with some small value (e.g. [-1, 1], [0.5, 1], ...) will work well enough  
-despite this, more tailored approaches have been developed over the years  
-various kernel initialization methods have been developed, each tailored to some subset of activation functions  
  
-for example, **he** initialization is recommended for ReLu  
-he initialization is computed as a random number, with a Gaussian probability distribution with a mean of 0.0, and a standard deviation of sqrt(2/n), where n is the number of inputs to the node  
