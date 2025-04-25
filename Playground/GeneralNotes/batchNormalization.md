# Batch normalization  
 
https://towardsdatascience.com/batch-norm-explained-visually-how-it-works-and-why-neural-networks-need-it-b18919692739/  
 
-batch normalization is an essential tool  
-batch normalization, often referd to as *batch norm*, is just another network layer  
-it helps stabilize the network  
  
-generally speaking, when inputing data to a network we will scale and normalize it, thus reducing
vast range differences between variable, enabling gradient descent to run smoother  
-because we've scaled and normalized the values, any adjustments that need to be made to weights are more likely
to properly contribute to the overall improvement of the model, instead of possibly inducing oscillatory behaviour  
-features on different scales will take longer (if ever) to converge  
  
-notice that the paragraph above refers **ONLY** for the input to the whole network  
-then what about normalization between hidden layers?  
-activations of a hidden layer are simply the outputs of the hidden layer before it  
-if we scaled and normalized the initial dataset, why not scale and normalize the activations in hidden layers as well? -if we are able to normalize activations from each previous layer, then grad desc will converge much better  
  
-batch norm is just another network layer  
-it has its own parameters :  
&nbsp;&nbsp;&nbsp;-two learnable parameters : *beta* and *gamma*  
&nbsp;&nbsp;&nbsp;-two non-learnable parameters : *mean moving average* and *variance moving average*  
-batch norm layer algorithm :  
&nbsp;&nbsp;&nbsp;-activations from the previous layer are passed to the batch norm layer  
&nbsp;&nbsp;&nbsp;-for each activation vector, separately, compute mean and variance  
&nbsp;&nbsp;&nbsp;-normalize the activation values, centering them around 0  
&nbsp;&nbsp;&nbsp;-scale and shift - normalized values are, element-wise, multiplied by gamma, and then added beta  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;-this allows batch norm to scale (to different variance) and shift (to different mean)  
&nbsp;&nbsp;&nbsp;-exponential mean average and variance are compute during each pass, and stored for later usage  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;-the saved mean and variance get used during inference  
  
-batch norm generally gets placed before or after activation function  
-this is why instead of using layer notation like this *Dense(8, activation='relu', kernel_initializer = tf.keras.initializers.HeNormal())* you will have to use the "deconstructed" approach to creating the NN  
-example :  
```python
import tensorflow as tf

model = tf.keras.Sequential([
    tf.keras.layers.Dense(8, kernel_initializer=tf.keras.initializers.HeNormal()),
    tf.keras.layers.BatchNormalization(),  # Batch norm added as a separate layer
    tf.keras.layers.ReLU()                 # Apply activation separately
])
```  
-above you can see that you have more control over what, and when, gets done in each layer  
  
  
## What is the reasoning behind "scale and shift" step during batch norm?  
-it allows the model to learn optimal representations after normalization  
  
1. restoring model capacity  
-normalization rescales inputs to have zero mean and unit variance  
-although this stabilizes the learning process, it can constrin network's ability to represent complex patterns  
-scale and shift parameters (ie. beta and gamma) reintroduce flexibility  
  
2. avoiding over-normalization  
-without scaling and shifting, the layer's outputs might always remain restricted to a standard distribution, thus  
eliminating, or reducing, network's ability to learn complex relationships  
  
3. gamma and beta are trainable  
-since gamma and beta are trainable parameters, meaning the network learns during training how to optimally scale and shift, we are ensuring the network is sufficiently "smart" to figure out patterns, yet also not prone to oscillating or diverging
