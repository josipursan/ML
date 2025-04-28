# Gradient clipping  
  
-gradient clipping is a method used to prevent model gradients from becoming too large during backpropagation  
-backpropagation computes the gradients of cost functions w.r.t. the weights and biases in the network  
  
-gradient clipping is used to control the *exploding gradient problem*  
&nbsp;&nbsp;&nbsp;-if the gradients of the network's loss w.r.t. to the weights becomes to large, any further weight updates are bound to be too large, thus causing the network to start oscilating and diverging  
  
-in networks using backpropagation, gradients can accumulate over iterations, leading to very large weight updates, hence the loss function oscilation/divergence  
  
-gradient clipping involes setting a *threshold* value, and then scaling the gradients down to this value if they exceed it  
