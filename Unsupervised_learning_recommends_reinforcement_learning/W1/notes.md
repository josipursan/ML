# Week 1  
## What is clustering?  
https://www.coursera.org/learn/unsupervised-learning-recommenders-reinforcement-learning/lecture/zoCuG/what-is-clustering  
  
-clustering looks at data, and finds data points that are similar, related to each other  
 
## k-means intuition  
https://www.coursera.org/learn/unsupervised-learning-recommenders-reinforcement-learning/lecture/xS8nN/k-means-intuition  
  
-initially k-means will take a random guess at where the *cluster centroids* might be  
-then it will iterate over all available data points to check which cluster centroid is closer to each data point  
-whichever cluster centroid is closest to the currently observed data point gets assigned this data point  
  
-then k-means recomputes the centroids - it does this by computing the average location of all points that have been assigned to their centroid, and moving the centroid to this average location  
-after this step we again interate over all data points, checking which cluster centroid is closest to each data point - this is an iterative process  
  
-k-means usually converges - it reaches a point where the position of cluster centroids can not be refined any more  
  
## k-means algorithm  
https://www.coursera.org/learn/unsupervised-learning-recommenders-reinforcement-learning/lecture/GwgDo/k-means-algorithm  
  
-first we randomly initialize *K* cluster centroids : $\mu_{1}, \mu_{2}, ..., \mu_{K}$ (*K* represents how many clusters we want to create, therefore how many cluster centroids we need)  
repeat {  
&nbsp;&nbsp;&nbsp;#Assign points to cluster centroids  
&nbsp;&nbsp;&nbsp;for *i* = 1 to *m*  
&nbsp;&nbsp;&nbsp; $c^{(i)}$ := index (from 1 to K) of cluster centroid closest to $x^{(i)}$  
  
&nbsp;&nbsp;&nbsp;#Move the cluster centroids  
&nbsp;&nbsp;&nbsp;for k = 1 to *K*   #*for all existing clusters we will compute their new clister centroids based on the total average position of all data points assigned to each observed cluster in the loop above*   
&nbsp;&nbsp;&nbsp; $\mu_{k}$ := average (mean) of all points assigned to cluster k  
}  
  
-assigning data points to cluster centroids, mathematically speaking, means computing which cluster centroid is closest to each data point, and then assigning the data point to that cluster centroid  
  
-*m* - number of training examples  
  
-how do we check which cluster centroid is closest to each point/training example?  
&nbsp;&nbsp;&nbsp;-we simply compute the distances, ie. the **L2 norm** :  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; $||x^{(i)} - \mu_{k}||^{2}$  
&nbsp;&nbsp;&nbsp;-in addition to using the above written L2 norm, we are searching for cluster centroid, *k*, that minimizes the L2 norm, ie. yields the smallest distance between the observed data point and the available cluster centroids (or to be even more blunt - we are searching for a cluster centroid (*k*) closest to the observed data  point)   
  
-if a cluster is empty because no points have been assigned to it, it usually gets eliminated  
  
### k-means for clusters that are not well separated  
-k-means does not necessarily have to be used only for datasets that have well separated groups of data points  
  
## Optimization objective  
https://www.coursera.org/learn/unsupervised-learning-recommenders-reinforcement-learning/lecture/f5G5k/optimization-objective  
  
- $c^{(i)}$ - index of clusters (1, 2, ..., *K*) to which example $x^{(i)}$ is currently assigned  
- $\mu_{k}$ - cluster centroid k  
- $\mu_{c^{(i)}}$ - cluster centroid of cluster to which example $x^{i}$ has been assigned  

-notation examples :  
&nbsp;&nbsp;&nbsp;- $x^{(10)}$ - training example 10  
&nbsp;&nbsp;&nbsp;- $c^{(10)}$ - cluster centroid to which the tenth training example has been assigned  
&nbsp;&nbsp;&nbsp;- $\mu_{c}^{(10)}$ - location of the cluster centroid to which $x^{(10)}$ has been assigned  
  
**Cost function**  
$J(c^{(1)}, ..., c^{(m)}, \mu_{1}, ..., \mu_{K}) = \frac{1}{m}\sum_{i=1}^{m}||x^{(i)} - \mu_{c^{i}}||^{2}$  
  
-average of sum, from 1 to m, of the squared distance between every training example *i* and the location of the cluster centroid to which the training examle $x^{(i)}$ has been assigned  
  
-the above shown equation often gets called **distortion**  
  
-due to the nature of the k-means algorithm, especially the cost function it uses, cost function/distortion reduces for each iteration of k-means algorithm, converging at a cluster centroid points that can't be positioned any better due to the distribution of datapoints in space  
  
-cost function should NEVER go up - if it does, it usually indicates a bug in code  
  
-additionally, since k-means converges, once you reach an iteration that has the same cost function value as the previous one we can conclude the algorithm has converged, and there is no point in running further iterations  
  
## Initializing k-means  
https://www.coursera.org/learn/unsupervised-learning-recommenders-reinforcement-learning/lecture/lw9LD/initializing-k-means  
  
-high level overview of k-means algorithm :  
&nbsp;&nbsp;&nbsp;-step 0 : randomly initialize K cluster centroids $\mu_{1}$, $\mu_{2}$, ..., $\mu_{k}$  
&nbsp;&nbsp;&nbsp;-step 1 : assign points to cluster centroids  
&nbsp;&nbsp;&nbsp;-step 2 : recompute positions of cluster centroids  
&nbsp;&nbsp;&nbsp;-repeat step 1 and step 2 until convergence is reached  
  
-random initialization is a crucial step, as the initial centroid positions have the ability to heavily influence what the end result will be  
  
-different initial cluster centroid positions can result in a vastly different datapoint grouping  
  
*Random initialization*  
-randomly pick *K* training examples (*K* being the number of clusters we want to create)  
-set cluster centroids, $\mu_{1}$, $\mu_{2}$, $\mu_{k}$, equal to *K* randomly picked training examples  
-run k-means algorithm  
  
-often times we will run k-means algorithm for a number of different random initializations due to their high level of importance  
  
-here is a screenshot showing how three different random initializations result in different groupings once the algorithm converges : 
<p style="text-align: center">
    <img src="./screenshots/w1_kmeans_random_initialization.png"/>
</p>  
  
-when running k-algorithm multiple times we will use the end cost function to determine which run provides us with the best clustering  
  
## Choosing the number of clusters  
https://www.coursera.org/learn/unsupervised-learning-recommenders-reinforcement-learning/lecture/LK4Zn/choosing-the-number-of-clusters  
  
-the "right" number clusters is more often, than not, quite ambiguous  
-some people might see 2 clusters where others see 4 clusters  
-data often does not give a clear answer how many clusters there are  
  
-academic literature defines a couple of different, rigorous, methods for choosing K, one of them being *elbow method*  
&nbsp;&nbsp;&nbsp;-K is chosen W.R.T. the decrease of cost function, ie. where we find a noticeable elbow in the cost function decrease, we choose this K value  
  
-choosing the right value *K* usually boils down to appropriate knowledge of what the dataset represents, and the right, conscious choice, of one of the possible K values that have been used to visualize the dataset  
  
# Anomaly detection  
## Finding unusual events  
https://www.coursera.org/learn/unsupervised-learning-recommenders-reinforcement-learning/lecture/1FML2/finding-unusual-events  
  
-one of the most common ways to conduct anomaly detection is via *density estimation*  
### Density estimation  
-you are given a dataset  
-the first thing your algorithm will do is figure out what are the values of the features representing our dataset that have *high probability* and *low probability* of being seen in the dataset  
-once we have defined the density areas in our dataset (ie. areas where it is highly probable the newly given example is not anomalous, areas where it is more likely it is anomalous, and areas where it is highly likely it is an anomalous example), we can feed new data through our model to try and determine whether they are anomalous  
  
## Gaussian (normal) distribution  
https://www.coursera.org/learn/unsupervised-learning-recommenders-reinforcement-learning/lecture/1pURx/gaussian-normal-distribution  
  
Signore Gauss (hehe) :  
$p(x) = \frac{1}{\sqrt{2\pi}\sigma}e^{\frac{-(x-\mu)^{2}}{2\sigma^{2}}}$  
  
-probability of *x* is determined by a Gaussian  
-a Gaussian is defined by its mean ($\mu$) and variance ($\sigma^{2}$, ie. std.dev.)  
  
-value of $\mu$ defines where top of the curve is, while variance (ie.std.dev.) defines how tight or wide the bell shape is - lower variance (std.dev.) value produces a tighter bell, while higher values produces wider bells  
  
-always remember : *area under the curve (bell) always equals to 1*  
  
## Anomaly detection algorithm  
https://www.coursera.org/learn/unsupervised-learning-recommenders-reinforcement-learning/lecture/nZcu2/anomaly-detection-algorithm  
  
-imagine you have a training set : ${\vec{x}^{(1)}, \vec{x}^{(2)}, ..., \vec{x}^{(m)}}$  
-each example $\vec{x}^{(i)}$ has *n* features (so each example is a vector of scalars)  
  
-here is how probability would look like for *n* features :  
$p(\vec{x}) = p(x_{1}; \mu_{1}, \sigma_{1}^{2}) * p(x_{2}; \mu_{2}, \sigma_{2}^{2}) * p(x_{3}; \mu_{3}, \sigma_{3}^{2}) * ... * p(x_{n}; \mu_{n}, \sigma_{n}^{2})$  

-a different, more succint, way of writing the equation above :  
$p(x) = \prod_{j=1}^{n} = p(x_{j}; \mu_{j}, \sigma_{j}^{2})$  
  
-the above written equation, following strict mathematical rules, can only be used if variables are statistically **independent**  
-however, in real world, variables describing a system are rarely independent  
-despite this, the above used equation will usually be more than good enough  
  
**How do we go about creating a general anomaly detection algorithm?**  
1. choose *n* features that you think might be indicative of anomalous behaviour  
&nbsp;&nbsp;&nbsp;&nbsp;-I wouldn't really write this like that - this means you are hunting only for variables that describe anomalies, however it might be easier to try and model what is "good" behaviour, and then catch if any new datapoints fall out of that range  
  
2. fit parameters $\mu$ and $\sigma$ - ie. figure out what the Gaussian is for each variable  
&nbsp;&nbsp;&nbsp; $\mu_{j} = \frac{1}{m}\sum_{i = 1}^{m}x_{j}^{i}$  
&nbsp;&nbsp;&nbsp; $\sigma_{j}^{2} = \frac{1}{m}\sum_{i=1}^{m}(x_{j}^{(i)} - \mu_{j})^{2}$  
  
3. given a new example *x*, compute *p(x)*  
&nbsp;&nbsp;&nbsp; $p(x) = \prod_{j = 1}^{n}p(x_{j}; \mu_{j}, \sigma_{j}^{2}) = \prod_{j = 1}^{n}\frac{1}{\sqrt{2\pi}\sigma_{j}}exp(-\frac{(x_{j} - \mu_{j})^{2}}{2\sigma_{j}^{2s}})$  
&nbsp;&nbsp;&nbsp;&nbsp; -**REMEMBER** - $x_{j}$ and $\sigma_{j}^{2}$ in the equation above are value compute in step 2., ie. we are using the already existing Gaussian curves for the variables to try and see how the new datapoint fits there (we are basically checking what kind of probability value will it return W.R.T. Gaussian distribution of the variables that describe our system)  
  
4. if $p(x) < \epsilon$, we have an anomaly  
  
## Developing and evaluating an anomaly detection algorithm  
https://www.coursera.org/learn/unsupervised-learning-recommenders-reinforcement-learning/lecture/WyzeY/developing-and-evaluating-an-anomaly-detection-system  
  
-for the sake of evaluation we will actually need a bit of labeled data, ie. tagged as anomalous or not anomalous  
  
-training set should be composed exclusively of datapoints that **AREN'T** anomalous  
-cross validation and test sets should mainly be made up of non-anomalous datapoints, but should also contain some anomalous datapoints  
  
-depending on how many anomalous examples you have (usually not a lot), you might choose to skip using the CV dataset altogether, and instead use training set, and place all of the anomalous examples in the CV set  
&nbsp;&nbsp;&nbsp;-note that all of the above is just nomenclature - you take a dataset, place only normal datapoints in it and then use it to train the network; then you take another, smaller, dataset, place all of the anomalous examples in it, and scrutinize the model generated by the initially created dataset  
  
-bear in mind that $\epsilon$ must be iteratively tuned to make it as good as possible at thresholding between a normal and anomalous datapoint  
  
## Anomaly detection vs. supervised learning  
https://www.coursera.org/learn/unsupervised-learning-recommenders-reinforcement-learning/lecture/eLO9Y/anomaly-detection-vs-supervised-learning  
  
-since anomaly detection actually requires some number of labeled datapoints (not a lot, but still some), one might wonder what is the difference between supervised learning and anomaly detection (unsupervised learning)  
  
**Anomaly detection**  
-operates on datasets with very large number of negative examples (non-anomalous) and a pretty small number of positive examples (anomalous)  
-when there is a number of different types of anomalies  
-examples of common applications :  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;-fraud detection  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;-finding new, previously unseen, manufacturing defects  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;-monitoring machines in a data center for anomalies  
  
  
**Supervised learning**  
-operates on datasets that have a lot of both positive and negative examples  
-requires enough positive examples to be able to properly model these examples in the future (relies on the fact that positive examples are likely to be similar in the future)  
-examples of common applications :  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;-email spam classification  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;-finding known, previously seen, manufacturing defects  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;-weather prediction  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;-disease classification  
  
## Choosing what features to use  
https://www.coursera.org/learn/unsupervised-learning-recommenders-reinforcement-learning/lecture/7MOXj/choosing-what-features-to-use  
  
-making sure you are using gaussian features for your anomaly detection algorithm is one of the steps to making sure algorithm performs well  
  
-however, even non-gaussian features can be changed a bit to make them a little bit more gaussian  
  
