# Supervised ML  

# Week 1  
## Supervised vs Unsunpervised ML
-**supervised** ML algos learn x to y mappings  
  
-the algorithm learns from being given the right answer (the correct label y)  
  
-by initially training the network on what the right answer is, once this network is given a new, never before seen input value x, it can accurately predict output label y  
  
-**regression** - predicting a value from inifinitely many possible outputs  
  
-**classification** - differs from regression in the sense that we are usually trying to predict a very limit set of output values  
&nbsp;&nbsp;&nbsp;-terms *class* and *category* are often used interchangeably when talking about classification problems  
&nbsp;&nbsp;&nbsp;-classification algorithms predict *categories*, usually a relatively number of categories
&nbsp;&nbsp;&nbsp;-classification algorithms use boundary lines to demarcate borders between different categories  
  
-**unsupervised learning** - the most widely used form of ML  
&nbsp;&nbsp;&nbsp;-in unsupervised learning we are given data that is not associated to any label y - in essence, the algorithm is not given any right answers as in supervised learning  
&nbsp;&nbsp;&nbsp;-e.g. algorithm is not told whether a space rocket launch was successful, it is only given various launch parameters  
&nbsp;&nbsp;&nbsp;-the job of unsupervised learning algos is to find some **structure**, a **pattern**  
&nbsp;&nbsp;&nbsp;-it is called unsupervised because we are not supervising the algorithm to give some right or wrong answer  
&nbsp;&nbsp;&nbsp;-some important unsupervise ML algos : clustering, anomaly detection, dimensionality reduction  
  
## Linaer regression part 1  
-what is linear regression? Basically fitting a straight line to some dataset  
  
**Useful terminology**  
-**training set** - data set used to train the model  
-**x** - input variable; also called *feature* (e.g. weight of aircraft, maximum amount of fuel, ...)  
-**y** - output variable; also called *target* (ie. the value we are trying to predict; e.g. maximum range of aircraft given some load it is carrying, maximum takeoff weight with the given atmospheric conditions)  
-**m** - total number of training examples (you can think of this as the total number of entries in a table)  
-**(x,y)** - single training example  
-**(x<sup>i</sup>, y<sup>i</sup>)** - i<sup>th</sup> training example (ie. some i<sup>th</sup> row in the data table containing a specific example)  
  
-when we run the training set (which contains both the features and the targets), our learning algorithm will produce some function *f* (this function often times gets called hypothesis)  
-the job of *f* is to take some new input *x* and output and estimate, prediction, called y<sup>^</sup> (*y hat*)  
-the function *f* also gets called the *model* - input to the model is called *feature*, and the output is the *estimate* (or *prediction*)  
  
So, how will we choose our function *f*?  
-for linear regression with one variable : **f(x) = f<sub>w,b</sub>(x) = wx + b**  
&nbsp;&nbsp;&nbsp;-what does "...with one variable" mean? It simply means the model we are working with has only one feature *x*  
&nbsp;&nbsp;&nbsp;-another name is *univariate linear regression*  

TO DO : `Optional Lab : Model representation`