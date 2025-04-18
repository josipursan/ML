# Supervised ML  

# Week 1  
## Supervised vs Unsunpervised ML
-**supervised** ML algos learn x to y mappings  
  
-the algorithm learns from being given the right answer (the correct label y)  
  
-by initially training the network on what the right answer is, once this network is given a new, never before seen input value x, it can accurately predict output label y  
  
-**regression** - predicting a value from inifinitely many possible outputs  
  
-**classification** - differs from regression in the sense that we are usually trying to predict a very limit set of output values  
&nbsp;&nbsp;&nbsp;-terms *class* and *category* are often used interchangeably when talking about classification problems  
&nbsp;&nbsp;&nbsp;-classification algorithms predict *categories*, usually a relatively small number of categories  
&nbsp;&nbsp;&nbsp;-classification algorithms use boundary lines to demarcate borders between different categories  
  
-**unsupervised learning** - the most widely used form of ML  
&nbsp;&nbsp;&nbsp;-in unsupervised learning we are given data that is not associated to any label y - in essence, the algorithm is not given any right answers as in supervised learning  
&nbsp;&nbsp;&nbsp;-e.g. algorithm is not told whether a space rocket launch was successful, it is only given various launch parameters  
&nbsp;&nbsp;&nbsp;-the job of unsupervised learning algos is to find some **structure**, a **pattern**  
&nbsp;&nbsp;&nbsp;-it is called unsupervised because we are not supervising the algorithm to give some right or wrong answer  
&nbsp;&nbsp;&nbsp;-some important unsupervise ML algos : clustering, anomaly detection, dimensionality reduction  
  
## Linear regression part 1  
-what is linear regression? Basically fitting a straight line to some dataset  
  
**Useful terminology**  
-**training set** - data set used to train the model  
-**x** - input variable; also called *feature* (e.g. weight of aircraft, maximum amount of fuel, ...)  
-**y** - output variable; also called *target* (ie. the value we are trying to predict; e.g. maximum range of aircraft given some load it is carrying, maximum takeoff weight with the given atmospheric conditions)  
-**m** - total number of training examples (you can think of this as the total number of entries in a table)  
-**(x,y)** - single training example  
-**(x<sup>i</sup>, y<sup>i</sup>)** - i<sup>th</sup> training example (ie. some i<sup>th</sup> row in the data table containing a specific example)  
  
-when we run the training set (which contains both the features and the targets), our learning algorithm will produce some function *f* (this function often times gets called hypothesis)  
-the job of *f* is to take some new input *x* and output and estimate, prediction, y<sup>^</sup> (*y hat*)  
-the function *f* also gets called the *model* - input to the model is called *feature*, and the output is the *estimate* (or *prediction*)  
  
So, how will we choose our function *f*?  
-for linear regression with one variable : **f(x) = f<sub>w,b</sub>(x) = wx + b**  
&nbsp;&nbsp;&nbsp;-what does "...with one variable" mean? It simply means the model we are working with has only one feature *x*  
&nbsp;&nbsp;&nbsp;-another name is *univariate linear regression*  

## Cost function formula  
-a cost function tells us how well the model is doing  

- - -  
<span style="color:yellow">A minor digression : </span>  
&nbsp;&nbsp;&nbsp;-remember that a linear equation is defined as **f(x) = f<sub>w,b</sub>(x) = wx + b**  
&nbsp;&nbsp;&nbsp;-**w**, and **b** are called *parameters* of the ML model  
&nbsp;&nbsp;&nbsp;-*parameters* of the model are the variables you can adjust during training to improve the model  
&nbsp;&nbsp;&nbsp;-*parameters* can often be called *coefficients*, or *weights*  
- - -  
  
-the problem we are facing when training a model is (in case of linear regression) **how do we choose parameters *w*,*b*, so that the predicted value y-hat (y^) is as close as possible to y<sup>real</sup> for all (x<sup>i</sup>, y<sup>real</sup>)? pairs**  
  
-to do this we will measure how well a line fits the training data - this is a **cost function!**  
  
-first we will comapare y-hat to y - this is called the **error** :  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;y<sup>hat</sup> - y  
  
-we want to compute the square error, and we will natuarally do it for all available training example :  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;(y<sup>hat<sup>i</sup></sup> - y<sup>i</sup>)<sup>2</sup>  
  
-considering this is done for all training set examples, the cost function would behave unreliably in the sense that with more training set examples, cost function would inevitably always increase, thus negating the useful information it is carrying  
&nbsp;&nbsp;&nbsp;-therefore, we will always compute the average  
-finally, we have the final expression :  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; J(w,b) = $\frac{{1}}{2m}$ $\sum_{i=1}^{m} $(y<sup>hat<sup>i</sup></sup> - y<sup>i</sup>)<sup>2</sup>  
  
-why is the averaging done using `2m` for division?  
&nbsp;&nbsp;&nbsp;-it makes some of our calculations a bit cleaner, but it can also be done using only `m` for division  
  
-this function is actually something you are well acquainted to : **square errror cost function!**  
  
-squared error cost function is by far the most used one for regression problems  
  
## Cost function intuition  
-our goal is to minimize our cost function : `min(J(w,b))`  
  
-we have some trained model which gave us some kind of curve or a straigth line based on the data it trained on  
-the point of cost function is to play dumb a bit, and say "I wonder what y-hat value I'd get if I inputed x<sup>i</sup> value in my model"  
-naturally the model will spit out some value y-hat - a *prediction*  
-however, we were playing dumb a bit - the parameter feature x<sup>i</sup> we inputed already has some known, true, value y<sup>i</sup>  
-then we can determine the difference between the true value and the predicted value, using cost function, to determine how our model is behaving compared to what the real data is  
  

## Visualization examples  
-no notes taken  
-when implementing your cost function visualizer, remember that there is `Optional Lab : Cost function` if you need to take a look or need inspiration  
  
- - -
## Gradient descent  
-**gradient descent** is an algorithm which enables us to systematically find optimal model parameters  
-it is one of the most popular optimization algorithms in ML overall, not just for linear regression  
  
-algorithm outline :  
&nbsp;&nbsp;&nbsp;-start with some *w*,*b* (most often you start with 0)  
&nbsp;&nbsp;&nbsp;-keep changin *w*, *b* to reduce J(w,b)  
&nbsp;&nbsp;&nbsp;-by changing values of parameters, settle at, or near, minimum J(w,b) value  
&nbsp;&nbsp;&nbsp;-remember : depending on how the cost function is defined, and what the nature is of your model, you might have more than one minimum  
  
### Implementing gradient descent  
-now we will outline, step by step, what each iteration of gradient descent does  
  
w = w - $\alpha$ $\frac{\partial }{\partial w}$ J(w,b)  
What does the equation above mean?  
&nbsp;&nbsp;&nbsp;It can literally be intepreted like this :  
&nbsp;&nbsp;&nbsp;&nbsp;-update the the value w, by taking the current value w, and adjusting it by a small amount (the section right of the subtraction sign)  
  
-what is $\alpha$?  
&nbsp;&nbsp;&nbsp;- $\alpha$ is the **learning rate**  
&nbsp;&nbsp;&nbsp;- it is usually a small positive number  
&nbsp;&nbsp;&nbsp;- it basically shows how big our step is in each loop of gradient descent  
  
-the derivative part basically tells in which direction we want to move when moving from current position  
  
-the expression for b is relatively similar  
  
b = b - $\alpha$ $\frac{\partial}{\partial b} J(w,b)$  
  
-these two equations are repated until our algorithm converges  
-what does convergence indicate here? Basically that we've reached a point where parameters *w* and *b* do not change that much anymore with each additional step  
  
-**IMPORTANT** : parameters *w* and *b* must be updated/computed **simultaneously**  
&nbsp;&nbsp;&nbsp;-why? Because, once we've computed the new, updated, parameter *w*, we still need to compute our new parameter *b*, but this must be done using the *w* value which was used when calculating the new parameter *w*  
&nbsp;&nbsp;&nbsp;-if you immediately updated *w* with the newly computed value, you would be computing the new value *b* using the modified value *w*  
&nbsp;&nbsp;&nbsp;-think of it like this : cost function freezes during each iteration; only once you've computed all new parameter values can you plug them back into the cost function  
  
-gradient descent should be implemented using the *simultaneous update* described above  
&nbsp;&nbsp;&nbsp;-however, even if you accidentally mess up, and don't update the parameters simultaneously, the gradient descent algorithm will probably work more or less ok (it probably adds a small error, or prolongs the seeking time)  

## Gradient descent intuition  
-a derivative, generally speaking, indicates the rate of change  
-in our case, we are basically looking, at a given point of some model parameter (e.g. *w*, or *b*) what the derivative of J(w,b) will be W.R.T. say parameter *w*  
-if the derivative is positive (meaning the tangent in that points points upwards towards area of the first quadrant), further steps of the observed parameter to the right will lead to an increase of J(w,b) (and vice versa also applies)  
-**note** : what you described in the remark above is the exact reason why you have subtraction in the term used to compute updated *w* and *b* values for each gradient descent iteration :  
w = w **-** $\alpha$ $\frac{\partial }{\partial w}$ J(w,b)  
w = w - $\alpha \cdot (positiveValueFromPartialDerivative)$  --> this moves *w* leftwards!  
&nbsp;&nbsp;&nbsp;-this means that if the partial derivative indicates a positive change, we actually do not want that positive change - if this positive step was present in the equation used to compute updated parameters our parameters would diverge,
not converge to the smallest possible cost function value!  
  
## Learning rate  
-learning rate has a big impact on the learning process  
  
-if $\alpha$ is too small, each partial derivative result will be multiplied by this immensely small $\alpha$ value  
&nbsp;&nbsp;&nbsp;-eventually gradient descent will reach the necessary results, but in doing so it will waste more time than necessary because each
step will be extremely slow  
&nbsp;&nbsp;&nbsp;-gradient descent will be too slow  

if $\alpha$ is too large, the coarseness of each step may cause your parameter to oscillate  
&nbsp;&nbsp;&nbsp;-even if oscillating, it might end up reaching the end of gradient descent, however, it might simply continue continue oscillating, never converging to a desired value  
  
-what if, the cost function we are evaluating our parameters on, is not a squared error cost function, but instead
some funky curve, containing one global minimum and multiple local minimums  
&nbsp;&nbsp;&nbsp;-let us assume your gradient descent algorithm has reached a local minimum - however this is not the global minimum, which we'd like to reach  
&nbsp;&nbsp;&nbsp;-further gradient descent iterations do nothing  
  
-note that, even though learning rate is fixed, the partial derivative (ie. rate of change) decreases as we approach local minimum  
  
## Gradient descent for linear regression  
**NOTE** : in this lecture/video everything gets assembled - squared error cost function will be used for our linear regression model with gradient descent  
This allows us to train the linear regression model to fit a straight line to some data.  
  
**Linear regression model** : f<sub>w,b</sub>(x) = wx + b  
  
**Cost function** : J(w,b) = $\frac{1}{2m}$ $\sum_{i=1}^{m} $(f<sub>w,b</sub>(x<sup>i</sup>) - y<sup>i</sup>)<sup>2</sup>  
  
**Gradient descent algorithm** : already explained, won't inflate notes rewriting it again  
&nbsp;&nbsp;&nbsp;-however, I will note the partial derivatives here  
&nbsp;&nbsp;&nbsp;-until convergence is reached we repeat :   
  
&nbsp;&nbsp;&nbsp; $\frac{\partial}{\partial w}$ J(w,b) = $\frac{1}{m}$ $\sum_{i=1}^{m} $(f<sub>w,b</sub>(x<sup>i</sup>) - y<sup>i</sup>) x<sup>i</sup>  
&nbsp;&nbsp;&nbsp; $\frac{\partial}{\partial b}$ J(w,b) = $\frac{1}{m}$ $\sum_{i=1}^{m} $(f<sub>w,b</sub>(x<sup>i</sup>) - y<sup>i</sup>)  
  
-squared error cost will always result in a "bowl" shaped cost function (mathematically correct term would be **convex function**, therefore a *convex* shape)  
-however, cost functions won't always be squared error costs - often times you might get wild, funky 3D plots having multiple local minimums  
  
## Running gradient descent  
-batch gradient descent - each step of gradient descent uses **all* available training examples  
-*Lab : Gradient descent* has their implementation of gradient descent