This .md file is used to step by step explain how update terms for *w* and *b* will be computed.

These are the update terms :  
  
&nbsp;&nbsp;&nbsp;repeat until convergence {  
  
&nbsp;&nbsp;&nbsp;&nbsp;w<sub>j</sub> = w<sub>j</sub> - $\alpha \frac{\partial J(w,b)}{\partial w_j} for j=0,...,n-1$  
  
&nbsp;&nbsp;&nbsp;&nbsp;b = b - $\alpha \frac{\partial J(w,b)}{\partial b}$  
&nbsp;&nbsp;&nbsp;&nbsp;}  
  
Partial derivatives can be simplified :  
  
$\frac{\partial J(w,b)}{\partial w_j} = \frac{1}{m} \sum_{i=0}^{m-1} (f_{w,b}(x^{(i)}) - y^{(i)}) x_{(j)}^{(i)}$  
  
$\frac{\partial J(w,b)}{\partial b} = \frac{1}{m} \sum_{i=0}^{m-1}(f_{w,b}(x^{(i)}) - y^{(i)})$  
  
-n - number of features  
-m - number of training examples  
  
-i - i<sup>th</sup> training example (ie. one row of data table, composed of multiple different columns)  
-j - j<sup>th</sup> feature within the i<sup>th</sup> training example  
  
-what is this expression? $(f_{w,b}(x^{(i)}) - y^{(i)})$  
&nbsp;&nbsp;&nbsp;-this is the error term
