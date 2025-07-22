#  Improving Deep Neural Network performance : Hyperparameter tuning, Regularization and Optimization  
  
##  Train/Dev/Test sets
https://www.youtube.com/watch?v=1waHlpKiNyY&list=PLkDaE6sCZn6Hn0vK8co82zjQtt3T2Nkqc&index=2  
  
-never forget : **constructing a model is a highly iterative process - it is ok to sometimes feel you are running around in circles, or to feel uncomfortable because you are trying out a lot of different things with the network**  
  
-initial dataset is usually split into three subsets :  
&nbsp;&nbsp;&nbsp;-training set - usually the largest one  
&nbsp;&nbsp;&nbsp;-cross validation, development set, dev set, hold-out set - all names for the same thing  
&nbsp;&nbsp;&nbsp;-test set  
  
-we train a model on the training set and then check its performance on the dev set  
&nbsp;&nbsp;&nbsp;-when we find the best model using the approach above we then test this model on the test set  
  
-modern, big data, era is characterized by large amounts of data (one million, and more, individual entries)  
&nbsp;&nbsp;&nbsp;-when dealing with such datasets it is common practice to use 98% of the data set for training, 1% for dev set, and 1% for test set  
&nbsp;&nbsp;&nbsp;-this is because the point of dev set is to provide a quick evaluation reference point for the performance of the most recently generated model, hence it makes no sense to waste a lot of compute time  
  
## Bias and variance  
https://www.youtube.com/watch?v=SjQyLhQIXSM&list=PLkDaE6sCZn6Hn0vK8co82zjQtt3T2Nkqc&index=2  
  
-high bias - underfit (the model is highly convinced its representation of data is accurate, but it isn't)  
-high variance - overfit (the model is too tightly coupled to the data, ie. represents the dataset step by step instead of generalizing it)  
  
-two key values for understanding bias and variance : *training set error* and *dev set error*  
-low training set error and high dev set error --> **high variance** ("high" dev set error means, in this context, an error value that is significanly greater than the aforementioned set, which is training set in this case)  
&nbsp;&nbsp;&nbsp;-why high variance? Because the model does not generalize well to the dev set  
  
-if a model has comparable performance both on the training set and the dev set, which is considerably worse than the SoTA performance (e.g. human performance when recognizing cats or dogs), then we can conclude that the model is **high bias** (ie. poor model of the data)  
  
-both **high bias** and **high variance** is a possible occurence at the same time - ie. the model is performing poorly on training set, and even worse on the dev set  
  
## Basic recipe for ML  
https://www.youtube.com/watch?v=C1N_PDHuJ6Q&list=PLkDaE6sCZn6Hn0vK8co82zjQtt3T2Nkqc&index=3  
  
You have trained a model.

Is the model high bias?  
If yes :  
&nbsp;&nbsp;&nbsp;-try a bigger network  
&nbsp;&nbsp;&nbsp;-try adding more layers  
&nbsp;&nbsp;&nbsp;-try a different network architecture  
  
Once you've rectified high bias issues, check if the model is high variance.  
If yes :  
&nbsp;&nbsp;&nbsp;-get more data  
&nbsp;&nbsp;&nbsp;-regularize  
&nbsp;&nbsp;&nbsp;-try a different network architecture  
  
## Regularization  
https://www.youtube.com/watch?v=6g0t3Phly2M&list=PLkDaE6sCZn6Hn0vK8co82zjQtt3T2Nkqc&index=4  
  
-regularization is one of the primary techniques used to combat high variance (overfit) models  
-remember the logistic regression cost function :  
$J(w,b) = \frac{1}{m} \sum_{i=1}^{m}\alpha(y^{hat.(i)}, y^{(i)}) + \frac{\lambda}{2m}||w||^{2}_{2}$  
  
L2 regularization : $||w||^{2}_{2} = \sum_{j=1}^{n_{x}}w_{j}^{2} = w^{T}w$  
&nbsp;&nbsp;&nbsp;-ie. the euclidian norm  
  
-*b* usually isn't regularized since *w* parameters have a significantly greater impact on a model being overfit  
  
- $\lambda$ is the regularization parameter  
  
-L1 regularization is also often used : $\frac{\lambda}{2m}\sum_{i=1}^{n_{x}}|w| = \frac{\lambda}{2m}||w||_{1}$  
  
-if L1 regularization is used *w* will be *sparse* - this means *w* vector will have a lot of zeros in it  
&nbsp;&nbsp;&nbsp;-this can help with compressing a model, although professor mentioned in his experience it doesn't help that much  
  
-for a NN with a number of layers the cost function has a bit different notation simply because we are dealing with higher dimensional structures :  
$J(w^{[1]}, b^{[1]}, w^{[2]}, b^{[2]}, ..., w^{[L]}, b^{[L]}) = \frac{1}{m}\sum_{i=1}^{n}\alpha(y^{hat(i)}, y^{(i)}) + \frac{\lambda}{2m}\sum_{l=1}^{L}||w^{[l]}|^{2}$  
  
-Frobenius norm : $||w^{[l]}||^{2} = \sum_{i=1}^{n[l]}\sum_{j=1}^{n[l-1]} (w_{ij}^{[l]})^{2}$  
&nbsp;&nbsp;&nbsp;- *i* axis is, I guess, considered the dimension of the neurons in layer *l*, while axis *j* should probably represent number of layers  
&nbsp;&nbsp;&nbsp;-HOWEVER, this doesn't really matter - understanding what each part is important, from which you can figure out what the inputs should be, also taking into account your dataset  
  
-Frobenius norm - matrix norm of an *m* x *n* matrix **A** is defined as the square root of the sum of the absolute squares of its elements  
  
## Dropout regularization  
https://www.youtube.com/watch?v=D8PJAL-MZv8&list=PLkDaE6sCZn6Hn0vK8co82zjQtt3T2Nkqc&index=6  
  
-dropout regularization means we go layer by layer and set the probability of eliminating one, or more, neurons from each layer  
  
-for each layer, given the number of neurons in it, generate a vector $R_{v}$ of random values [0,1], each corresponding to the randomly generated probability of a neuron, whose index matches that of the ranomdly generated value in $R_{v}$, being eliminated  
&nbsp;&nbsp;&nbsp;-if a ranomdly generated value from $R_{v}$ is less than $keepThreshold$, neuron gets eliminated (dropped out)  
  
-another approach is to generate a vector of random values so that there is a $keepProbability$ chance of the random value being 1 (ie. keep the neuron), and $1-keepProbability$ chance of the random value being 0 (ie. drop the neuron)  
&nbsp;&nbsp;&nbsp;-e.g. we set $keepProbability=0.9$, meaning there is a 0.9 probability the randomly generated value is 1, thus keeping the neuron, and 0.1 probability we generate 0, therefore dropping the neuron  
  
-once we have these randomly generated dropout masks we need to multiply, element-wise, the mask with weights for the NN - this is how we turn off certain neurons  
  
-inverted dropout - since we are dropping a chosen amount of neurons (e.g. 20%) this means we have to scale the remaining ones to make up for the dropped ones  
&nbsp;&nbsp;&nbsp;-why? Because testing the performance of model using dev set and test sets will not be representative