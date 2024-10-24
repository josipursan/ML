# Supervised Machine Learning: Regression and Classification  
  
# Week 2 - Regression with multiple input variables  
-the goal of W2 is to make our linear regression much faster and simpler  
-we will start this week off with `Linear regression with multiple input variables`  
  
## Multiple features  
-our example for W1 had one input variable (feature *x* )and one output variable (*y*) - using square area of the house (*x*) we were predicting its price (*y*) : `f(x) = y = wx+b`  
  
-what if instead of having only one input feature (square area) we actually had more input features : number of bedrooms, number of floors, age of house in years  
  
-we will name these features as follows :  
&nbsp;&nbsp;&nbsp;-house area (size) - x<sub>1</sub>  
&nbsp;&nbsp;&nbsp;-number of bedrooms - x<sub>2</sub>  
&nbsp;&nbsp;&nbsp;-number of floors - x<sub>3</sub>  
&nbsp;&nbsp;&nbsp;-age in years - x<sub>4</sub>  
  
-x<sub>j</sub> = represents the j<sup>th</sup> feature  
-`n` - number of features  
$\vec{x}$ <sup>i</sup> = features of the i<sup>th</sup> training example (vector containing all features of the i<sup>th</sup> training example)  
&nbsp;&nbsp;&nbsp;-example : $\vec{x}$<sup>2</sup> - vector of features for the second training example (ie. the second row in table)  
-x<sub>j</sub><sup>(i)</sup> - value of feature *j* in the i<sup>th</sup> training example  
&nbsp;&nbsp;&nbsp;-example : x<sub>3</sub><sup>(2)</sup> refers to the value of third feature in the second training example; third feature is *number of floors*, and the second training example refers to the second table row, meaning we are referring to value **2**  
  
-model we used for linear regression : f<sub>w,b</sub>(x) = wx + b  
-**new model for linear regression with multiple input variables (house example)** :  
&nbsp;&nbsp;&nbsp;f<sub>w,b</sub>(x) = w<sub>1</sub>x<sub>1</sub> + w<sub>2</sub>x<sub>2</sub> + w<sub>3</sub>x<sub>3</sub> + w<sub>4</sub>x<sub>4</sub> + b  
  
-general expression for linear regression with multiple input variables :  
&nbsp;&nbsp;&nbsp;f<sub>w,b</sub>(x) = w<sub>1</sub>x<sub>1</sub> + w<sub>2</sub>x<sub>2</sub> + ... + w<sub>n</sub>x<sub>n</sub> + b  
  
-now we will simplify the expression above  
-all `w` parameters can be written as a row vector :  
&nbsp;&nbsp;&nbsp; $\vec{w}$ = [w<sub>1</sub>, w<sub>2</sub>, ..., w<sub>n</sub>]  
-all `x` inputs can be written as a row vector :  
&nbsp;&nbsp;&nbsp; $\vec{x}$ = [x<sub>1</sub>, x<sub>2</sub>, ..., x<sub>n</sub>]  
-`b` is just a scalar  
  
-now the model can be written more succintly :  
&nbsp;&nbsp;&nbsp;f<sub>w,b</sub>($\vec{x}$) = $\vec{x}\cdot\vec{w}$ + b  
&nbsp;&nbsp;&nbsp;-the `dot` product refers to the action of multiplying corresponding pairs of values between the used vectors;  
&nbsp;&nbsp;&nbsp;-example : for $\vec{w}$ and $\vec{x}$ we will multiply elements positionally, namely w<sub>1</sub> * x<sub>1</sub>, w<sub>2</sub> * x<sub>2</sub>, w<sub>3</sub> * x<sub>3</sub>, etc.  
NOTE : in the expression above, `w` in subscript of `f` should also have a vector above it, but it wouldn't render successfully when `\vec` was in `sub` tag  
  
-this model is called **multiple linear regression** (NOT multivariate linear regression)  
  
  
## Vectorization part 1  
-makes the code shorter, simpler and executes quicker  
  
- $\vec{w}$ = [w<sub>1</sub>, w<sub>2</sub>, w<sub>3</sub>]  
- $\vec{x}$ = [x<sub>1</sub>, x<sub>2</sub>, x<sub>3</sub>]  
-`b` is a scalar  
  
-in python, using numpy :  
&nbsp;&nbsp;&nbsp;`w = np.array([1.0, 2.5, -3.3])`  
&nbsp;&nbsp;&nbsp;`x = np.array([10, 20, 30])`  
&nbsp;&nbsp;&nbsp;`b = 4`  
  
-implementing our multiple linear regression model without vectorization, using python :  
&nbsp;&nbsp;&nbsp;`f = w[0] * x[0] + w[1] * x[1] + w[2] * x[2] + b`  
&nbsp;&nbsp;&nbsp;-this kind of implemnentation is not good because it requires a lot of manual manipulation should length of vectors `w` and `x` change  
  
-another option, without vectorization is using a `for` loop, which is a lot better than the hardcoded example above  
  
-now we will implement our mutiple linear regression model WITH vectorization :  
&nbsp;&nbsp;&nbsp;`f = np.dot(w,x) + b`  
&nbsp;&nbsp;&nbsp;-when `n` is large, vectorized approach will always run much faster than the 2 other approaches outlined above  
  
## Vectorization part 2  
-let us imagine we are running grad desc  
-we have 16 `w` params because we have 16 input variables  
-because we have 16 input variables, we will also have 16 derivative terms to compute  
-imagine we have already computed all of these derivative terms, and that they are stored in a row vector called `d` (`w`is in a row vector as well) :  
&nbsp;&nbsp;&nbsp; $\vec{w}$ = [w<sub>1</sub>, w<sub>2</sub>, w<sub>3</sub>, ..., w<sub>16</sub>]  
&nbsp;&nbsp;&nbsp; $\vec{d}$ = [d<sub>1</sub>, d<sub>2</sub>, d<sub>3</sub>, ..., d<sub>16</sub>]  
  
-remember that the update term for `w` when running grad desc is w<sub>j</sub> = w<sub>j</sub> - $\alpha$*d<sub>j</sub> for j = 1,...,16  
  
-implementing this in python without vectorization would mean we would have to approach one by one each parameter `w` and compute its new value :  
&nbsp;&nbsp;&nbsp;w<sub>1</sub> = w<sub>1</sub> - $\alpha$*d<sub>1</sub>  
&nbsp;&nbsp;&nbsp;w<sub>2</sub> = w<sub>2</sub> - $\alpha$*d<sub>2</sub>  
&nbsp;&nbsp;&nbsp;w<sub>3</sub> = w<sub>3</sub> - $\alpha$*d<sub>3</sub>  
&nbsp;&nbsp;&nbsp;...  
&nbsp;&nbsp;&nbsp;w<sub>16</sub> = w<sub>16</sub> - $\alpha$*d<sub>16</sub>  
&nbsp;&nbsp;&nbsp;-in python we would write this using a for loop :  
&nbsp;&nbsp;&nbsp;&nbsp;`for j in range(0, 16) : `  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`w[j] = w[j] - alpha*d[j]`  
  
-using vectorization, we can use can clean, pretty line of code, which also makes the code maintenance easier :  
&nbsp;&nbsp;&nbsp; $\vec{w}$ = $\vec{w}$ - 0.1* $\vec{d}$  
-in python : `w = w - 0.1*d`  
  
-using vectorized implementation will result in a huge difference in run time and efficiency  
  
## Gradient descent for multiple linear regression  
  
-`b` is just a number  
-$\vec{w}$ = [w<sub>1</sub>, w<sub>2</sub>, ..., w<sub>n</sub>]  
-model : f<sub>w,b</sub> = $\vec{w} \cdot \vec{x}$ + b  
-cost function : J($\vec{w}$, b)  
-grad desc algo :  
&nbsp;&nbsp;&nbsp;&nbsp;repeat until convergence {  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;w<sub>j</sub> = w<sub>j</sub> - $\alpha \frac{\partial}{\partial w_j}$ J($\vec{w}$, b)  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;b = w<sub>j</sub> - $\alpha \frac{\partial}{\partial b}$ J($\vec{w}$, b)  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;}  
  
-remember :  
&nbsp;&nbsp;&nbsp;-superscript `i` represents all features for the i<sup>th</sup> training example  
&nbsp;&nbsp;&nbsp;-subscript `j` represents the j<sup>th</sup> feature out of all available features

-in case we have n=>2 features, update terms for grad desc will change only a little bit :  
&nbsp;&nbsp;&nbsp;&nbsp;repeat until convergence {  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; w<sub>n</sub> = w<sub>n</sub> - $\alpha \frac{1}{m} \sum_{i=1}^m$(f<sub>w,b</sub>($\vec{x}$<sup>i</sup>) - y<sup>i</sup>) $\cdot$ x<sub>n</sub><sup>i</sup>  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; b = b - $\alpha \frac{1}{m} \sum_{i=1}^m$(f<sub>w,b</sub>($\vec{x}$<sup>i</sup>) - y<sup>i</sup>)  
  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; simultaneously update  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; w<sub>j</sub> (for j = 1,..., n) and b  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;}  
  
-for each existing feature (all `j`) we have to compute their respective parameters `w`  
  
### An alternative to gradient descent  
-normal equation can be used only for linear regression  
-using normal equation we can avoid solving for w,b parameters using iteration  
  
-some disadvantages of normal equation are that this approach does not generalize to other learning algorithms, it is slow if we have a lot of features (more than 10000), 