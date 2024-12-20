# Linear regression gradient descent

-*x* - input variable (also called feature : aircraft weight, range, payload, ...)  
-*y* - output variable (also called target, ie. the value we are trying to predict based on *x*)  
-*m* - total number of training examples  
-*(x,y)* - single training example  
-$(x^{i}, y^{i})$ - i-th training examples (i-th row in training data matrix)  

Model : $f_{w,b}(x) = wx + b$  
Cost function : $J(w,b) = \frac{1}{2m}\sum_{i = 1}^{m}(f_{w,b}(x^{i}) - y^{i})^{2}$  

repeat until hypothesis is satisfactory {  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; $w = w - \alpha\frac{\partial}{\partial{w}}J(w,b) = w - \alpha * [\frac{1}{m}\sum_{i = 1}^{m}(f_{w,b}(x^{i}) - y^{i})x^{i}]$  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; $b = b - \alpha\frac{\partial}{\partial{b}}J(w,b) = b - \alpha * [\frac{1}{m}\sum_{i=1}^{m}(f_{w,b}(x^{i}) - y^{i})]$  
}  
  
-$\alpha$ - learning rate/step  
&nbsp;&nbsp;&nbsp;-if too small, each partial derivative won't contribute enough to change w/b parameters quickly enough (ie. more iterations will be needed to reach properly optimized hypothesis)  
&nbsp;&nbsp;&nbsp;-if too large, parameters will likely end up oscilating  
  
# Regularized linear regression  
-general idea of regularization is the idea of creating a properly fitting model using a regularization parameter that penalizes all model parameters (consequently features) to avoid overfitting or underfitting models  
  
-$\lambda$ - regularization parameter  
  
Regularized cost function for lin.reg. : $J(\vec{w}, b) = \frac{1}{2m}\sum_{i = 1}^{m}(f_{\vec{w}, b}(\vec{x}^{i}) - y^{i})^{2} + \frac{\lambda}{2m}\sum_{j = 1}^{n}w_{j}^{2}$  
  
repeat until hypothesis is satisfactory {  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; $w_{j} = w_{j} - \alpha[\frac{1}{m}\sum_{i=1}^{m}((f_{w,b}(x^{i}) - y^{i})x_{j}^{i}) + \frac{\lambda}{m}w_{j}]$  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; $b = b - \alpha\frac{1}{m}\sum_{i=1}^{m}(f_{w,b}(x^{i}) - y^{i})$  
}  
  
# Multivariable linear regression  
-previosly we've demonstrated modeling **univariate** linear regression (one parameter) : $f_{w,b}(x) = y = wx + b$  
-here we will demonstarted modeling **multivariable** linaer regression (more than one parameter) : $f_{w,b}(x) = y = w_{1}x_{1} + w_{2}x_{2} + w_{3}x_{3} + ... + w_{n}x_{n} + b$  
  
-w and x are now (row) vectors  
-*n* - number of features  
  
$x_{j}$ - represents the j-th feature of a chosen row vector  
$\vec{x}^{i}$ - row vector containing all of the features found in the i<sup>th</sup> training example  
$x_{j}^{i}$ - value of the j<sup>th</sup> feature in i<sup>th</sup> training example ($x_{3}^{2}$ is the value of third feature found in second row of training matrix)  
  
-because we are dealing with vectors, we will use dot product of vectors to speed up things :  $f_{\vec{w},b}(\vec{x}) = \vec{w} \cdot \vec{x} + b$  
  
Cost function : $J(w,b) = \frac{1}{2m}\sum_{i=1}^{m}(f_{w,b}(x^{i}) - y^{i})^{2}$  
repeat until hypothesis is satisfactory {  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; $w_{n} = w_{n} - \alpha\frac{1}{m}(f_{w,b}(\vec{x})^{i} - y^{i})x_{n}^{i}$  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; $b = b - \alpha\frac{1}{m}\sum_{i=1}^{m}(f_{w,b}(\vec{x})^{i} - y^{i})$  
}  
  
NOTE : for better understanding of $w_{n}$ update term check out :  
https://github.com/josipursan/ML/blob/main/Supervised%20Machine%20Learning%3A%20Regression%20and%20Classification/W2/multiple_variable_linear_regression/explainedSolution.md  

# Logistic regression
-used for classification  
-when we have discrete states *y* can take on, logistic regression is used  
-**linear** regression operates in a discrete domain, while **logistic** regression operates in a discrete domain  
  
-**sigmoid** function is very important for logistic regression because running data on it allows us to split it into classes/labels  
  
Sigmoid : $g(z) = \frac{1}{1+e^{-z}}$  
$z = \vec{w} \cdot \vec{x} + b$ (same expression as for linear regression)  
Logistic regression model : $f_{\vec{w},b}(\vec{x}) = g(z) = g(\vec{w} \cdot \vec{x} + b) = \frac{1}{1+e^{\vec{w} \cdot \vec{x} + b}}$  
Loss function : $L(f_{\vec{w},b}(\vec{x^{i}}, y^{i}) = -y^{(i)} log(f_{\vec{w},b}(\vec{x^{i}})) - (1-y^{(i)})log(1 - f_{\vec{w},b}(\vec{x^{i}}))$  

  
-take a look at $z$ equation written couple of lines above - we can use this to determine the decision boundary, ie. figure out where the border between `class = 0` and `class = 1` exactly is  
&nbsp;&nbsp;&nbsp;&nbsp;-this is done by solving the *z* equation for $z = 0$ :  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;$\vec{w} \cdot \vec{x} + b = 0$  
-this is exactly what we are doing by training our model - figuring out the best possible *w* and *b* parameters, thus enabling us to have the most appropriately trained model by figuring out exactly where the decision boundary lies  

# Regularized logistic regression  
-don't forget - n is the number of features, m is the number of training examples (each training examples is made up of n features)  

Logistic regression model : $f_{\vec{w},b}(\vec{x}) = g(z) = g(\vec{w} \cdot \vec{x} + b) = \frac{1}{1+e^{\vec{w} \cdot \vec{x} + b}}$  
Regularized cost function : $J(\vec{w}, b) = -\frac{1}{m} \sum_{i = 1}^{m}[y^{(i)} log(f_{\vec{w},b}(\vec{x^{i}})) + (1-y^{(i)})log(1 - f_{\vec{w},b}(\vec{x^{i}})] + \frac{\lambda}{2m}\sum_{j=1}^{n}w_{j}^{2}$  
repeat until satisfactory hypothesis {  
&nbsp;&nbsp;&nbsp;&nbsp; $w_{j} = w_{j} - \alpha\frac{\partial}{\partial{w_{j}}}J(\vec{w}, b) = w_{j} - \alpha[\frac{1}{m}\sum_{i=1}^{m}(f_{\vec{w}, b}(\vec{x}^{i}) - y^{i})x_{j}^{i}] + \frac{\lambda}{2m}\sum_{j=1}^{n}w_{j}^{2}$  
&nbsp;&nbsp;&nbsp;&nbsp; $b = b - \alpha\frac{\partial}{\partial{b}}J(\vec{w}, b) = b - \alpha[\frac{1}{m}\sum_{i=1}^{m}(f_{\vec{w}, b}(\vec{x}^{i}) - y^{i})x_{j}^{i}]$  
}  
  
-non-regularized implementation of logistic regression is without the regularization term, everything else remains the same  


# Overfit/underfit models  
-**underfit** models - high bias models (because these models carry a high level of confidence that the data is properly modeled)  
  
-**overfit** models - high variance (because a small change in training data results in a drastically different end model)  
&nbsp;&nbsp;&nbsp;-tightly coupled to training data  
&nbsp;&nbsp;&nbsp;-usually models with a lot of higher order polynomials  
  
# Automatic convergence test  
-instead of running model training for a predefined number of iterations, or until a satisfactory model cost is reached, you can use **automatic convergence test**  
-this method simply checks whether cost decrease from in iteration *i* is less than some predefined threshold  
-this threshold is called *epsilon* - $\epsilon$  
-however, setting up epsilon also represents a joruney of its own - what if it is too small? what if it is too large? ...  