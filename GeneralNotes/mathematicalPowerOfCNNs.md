# Mathematical power of CNNs  
What is it about CNNs, mathematically speaking, that makes them perform so much better than any other approach?  
  
1 : https://www.linkedin.com/pulse/understanding-convolutional-neural-networks-cnns-deep-ahmed-rxuke  
2 : https://www.reddit.com/r/MachineLearning/comments/emby2v/d_why_are_cnns_so_much_better_than_other/  
3 : https://ujjwalkarn.me/2016/08/11/intuitive-explanation-convnets/  
4 : https://www.youtube.com/watch?v=YFL-MI5xzgg  
  
## Here is what Microsoft's Copilot says  
CNNs possess a number of elegant mathematical principles allowing them to efficientily caputer and transform spatial data  
  
1. Local connectivity and receptive fields  
Instead of connecting every neuron to all inputs, CNNs use local receptive fields.  
Mathematically, this means that each neuron computes a weighted sum over a small spatial patch of the input image.  
This localized approach (due to the usage of a convolution window) mirrors the structure of an image, where meaningful patterns (consequently information) reside in small regions.  
Such locality reduces the number of parameters and enhances the network's ability to learn spatial hierarchies  
  
2. Convolution operator (Linear transformation)  
Heart of CNN is the *convolution operator*.  
Convolution is a linera transformation.  
It extracts local features by computing weighted sums over spatially correlated pixels.  
Convolution operations is *shift equivariant* - equivariance indicates that a change in input features results in an equivalent, expected, and predictable, translation of outputs.  
Shift equivariance is critical for handling images where identical, or similar, objects can appear anywhere.  
  
3. Weight sharing  
A single filter is applied across the entire image.  
This means the same set of weights is used to process different regions.  
This encodes the assumption that features like edges and curves are useful regardless of position in the image.  
  
4. Non-Linear activation functions  
While the convolution itself is linear, stacking these operations non-linear operations enables the network to learn complex patterns and decision boundaries.  
  
5. Pooling operations  
Pooling functions further enhance the network's capabilities.  
They perform a form of **downsampling**.  
Downsampling is the reduction of resolution of feature maps. Critial information is preserved.  
This reduces the computational complexity, but also introduces invariance to small translations and distortions (ie. small changes in the position, shape and orientation of ROI).  
Mathematicall speaking, pooling is a non-linear subsampling operation that summarizes regions, reinforcing network's robustness to spatial variance.  
  
6. Hierarchical feature extraction  
Layers of convolutions, activations, and pooling operations build a hierarchical representation of the input.  
Early layers might capture simple features such as edges or basic textures.  
Deeper layers combine these features into increasingly abstract representations (shapes, objects, scenes).  
Mathematically speaking, this is nothing other than a **composition of functions**.  
Each layer applies a transformation, and the successive compositions (*layers*) allow the network to model highly comples, non-linear mappings between the raw input and the final output.  
  
7. Fourier perspective and Efficiency  
In the Fourier domain, a convolution in the spatial domain translates to a point-wise multiplication.  
We can simply perform two Fourier transforms, multiply the results element-wise, and then apply inverse Fourier transform to return to the spatial domain.  