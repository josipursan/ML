# Week 3
In week 3 we will be learning about classification.  
This week covers category prediction using logistic regression, as well as what overfitting is and how it is handled (regularization).  

## Logistic regression  
-**sigmoid function** is important for our logistic regression implementation  
&nbsp;&nbsp;&nbsp;-also called **logistic function**  
&nbsp;&nbsp;&nbsp;-sigmoid expression :  $g(z) = \frac{1}{1+e^{-z}}$ &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;0 < g(z) < 1  
  
-how will we use sigmoid to build logistic regression?  
&nbsp;&nbsp;&nbsp;-remember that we defined a linear equation as f<sub>w,b</sub>($\vec{x}$) = $\vec{w} \cdot \vec{x} + b$  
&nbsp;&nbsp;&nbsp;-everything right of the equals sign can be called *z* : z = $\vec{w} \cdot \vec{x} + b$  
&nbsp;&nbsp;&nbsp;-now this whole expression *z* becomes what gets inserted into *z* in our sigmoid expression, thus getting the **logistic regression model**  
  
&nbsp;&nbsp;&nbsp;- f<sub>w,b</sub>($\vec{x}$) = g(z) = g($\vec{w} \cdot \vec{x} + b$) = $\frac{1}{1+e^-{(\vec{w} \cdot \vec{x} + b)}}$  
  
&nbsp;&nbsp;&nbsp;-it inputs a feature x, or a set of features, and outputs a number between 0 and 1 (ie. class 0 or class 1)  
  
### Interpretation of logistic regression output  
-logistic regression output value can be though of as the probability that class is 1 (or that it is not 0, or vice versa, or however you posit it)  
  
- f<sub>w,b</sub>($\vec{x}$) = P(y = 1 | $\vec{x};\vec{w}, b$)  
&nbsp;&nbsp;&nbsp;&nbsp;-translated in english : Probability that y is 1, given input $\vec{x}$ and parameters $\vec{w}$,b  
  
`Optional lab : Sigmoid function and logistic regression` is more of an exploratory lab for what sigmid is and how it behaves - check it out later if you get stuck on some code implementations  
  
## Decision boundary  
-the `z` term we are using in our model can be used to determine where the `decision boundary` is  
&nbsp;&nbsp;&nbsp;-to be more precise, we check out when our term `z` will be equal to zero to figure out where the decision boundary is  
&nbsp;&nbsp;&nbsp;-if the models we are working for are modeled by some well known and described mathematical equations (such as a line, circle, etc.), by rearranging the equation we can figure out where exactly the boundary between y = 1 and y = 0 lies  
  
`Optional lab : Decision boundary` has a nice example of how logistic regression, specifically decision boundary works - check it out if later in the course you get stuck on some implementations  
  
## Cost function for logistic regression
-cost functions, generally speaking, give us a way of measuring how well a specific set of parameters fits the training data, enabling us to choose best possible parameters  
  
-*m* - number of training examples  
-*n* - total number of features (each training example, ie. row in table, has *n* features)  
-x<sub>n</sub> - n-th feature  
-*y* - target (0 or 1 in case of logistic regression)  
-logistic model is defined by this equation :  
&nbsp;&nbsp;&nbsp;$f_{\vec{w},b}(\vec{x}) = \frac{1}{1+e^{-(\vec{w} \cdot \vec{x} + b)}}$  
  
-recall that for linear regression we used the **squared error cost** :  
&nbsp;&nbsp;&nbsp; J($\vec{w}, b$) = $\frac{1}{m} \sum_{i=1}^m \frac{1}{2}(f_{\vec{w}, b}(\vec{x^{i}}) - y^{i})^{2}$  
  
&nbsp;&nbsp;&nbsp;-if you try using this cost function for logistic regression, you will end up with a cost function that is highly non-convex, ie. it has a lot of local minima, which is not what we want  
&nbsp;&nbsp;&nbsp;-we want a different cost function, one that will be convex and with only one, global, minimum if possible  
  
-**remember** : the loss function tells you how well you are doing on that one training example - by summing up all of the losses of all of the training examples we get the cost function (which measures how well you are doing on the whole training set)  

-let's again take a look at the squared error cost expression :  
&nbsp;&nbsp;&nbsp; J($\vec{w}, b$) = $\frac{1}{m} \sum_{i=1}^m \frac{1}{2}(f_{\vec{w}, b}(\vec{x^{i}}) - y^{i})^{2}$  
&nbsp;&nbsp;&nbsp;&nbsp;-the chunk under summation operator will be called *loss*:  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; L($f_{\vec{w},b}(\vec{x^{i}}, y^{i})) =  \frac{1}{2} (f_{\vec{w}, b}(\vec{x^{i}}) - y^{i})^{2}$  
  
&nbsp;&nbsp;&nbsp;-the loss functions input f(x) and the true label y, and tells us how well we are doing on that example  
&nbsp;&nbsp;&nbsp;-so, what kind of format/expression will we use for the loss function when doing logistic regression  
&nbsp;&nbsp;&nbsp;&nbsp;-we will actually split it into two cases, and therefore two expressions, based on the true label  
&nbsp;&nbsp;&nbsp;&nbsp;if y<sup>(i)</sup> = 1 ---> $-log(f_{\vec{w}, b}(\vec{x^{i}}))$  
&nbsp;&nbsp;&nbsp;&nbsp;if y<sup>(i)</sup> = 0 ---> $-log(1 - f_{\vec{w}, b}(\vec{x^{i}}))$  
  
-let us first consider the loss function for the case y<sup>(i)</sup> = 1 :  
&nbsp;&nbsp;&nbsp;-because the output of logistic regression is always between 0 and 1, we can focus only on the part of the curve between 0 and 1  


&nbsp;&nbsp;&nbsp;-if our model/algorithm predicts a probability close to 1, and the true label is 1, then the loss is 0, or very small, therefore we can infer that our model performed well (notice the graph below which is visually showing what is exaplained in this paragraph)  
&nbsp;&nbsp;&nbsp;- another way to write what is explained here is : As $f_{\vec{w}, b}(\vec{x^{i}})$ --> 1 then loss --> 0  
&nbsp;&nbsp;&nbsp;-continuing on with our y = 1 curve, if $f_{\vec{w}, b}(\vec{x^{i}})$ --> 0, then loss increases significantly because we move leftwards on our curve  
  
&nbsp;&nbsp;&nbsp;remember : y axis represents how big the loss is  
  
!["Graphs showing how loss works when observing label y=1 curve"](./screenshots/Logistic_loss_function_1.png "Graphs showing how loss works when observing label y=1 curve")  
  
_ _ _ _ _ _ _ _  
  
-let us now consider the loss function for y<sup>(i)</sup> = 0 case  
&nbsp;&nbsp;&nbsp;-as with the previous example, y axis shows the amount of loss for each $f_{\vec{w}, b}(\vec{x^{i}})$ value  
&nbsp;&nbsp;&nbsp;-when our prediction is close to zero, loss for that prediction will be close to zero because our f value lies very close to the axis intersection where loss is minimal  
&nbsp;&nbsp;&nbsp;-the larger the value f is (ie. more to the right), the greater the loss  
&nbsp;&nbsp;&nbsp;-visual example shown below  
  
!["Graphs showing how loss works when observing label y=0 curve"](./screenshots/Logistic_loss_function_2.png "Graphs showing how loss works when observing label y=0 curve")  
  
-these two separate expressions, one for each label, give us a convex cost function, thus making sure cost function runs fully and without any hiccups  
  
