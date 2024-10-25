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
  
TO DO : `Optional lab : Sigmoid function and logistic regression`  