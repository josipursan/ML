# Advanced Learning Algorithms - W4  
  
## Decision tree model  
https://www.coursera.org/learn/advanced-learning-algorithms/lecture/HFvPH/decision-tree-model  
  
-very popular in competitions and general product applications  
  
## Learning process  
https://www.coursera.org/learn/advanced-learning-algorithms/lecture/5ysdd/learning-process  
  
-learning tree gets constructed by choosing on which features the algorithm will split  
-after splitting for a chosen/sufficient amount of times, we can create leaf nodes, ie. decision nodes which declare "this is cat", "this is dog", etc...  
  
-1. how do we choose which features to split on?  
&nbsp;&nbsp;&nbsp;-our goal is to always maximize **purity** - purity refers to the degree to which splitting separates wanted from non-wanted categories, ie. whether the split enables us to have only wanted category  
  
-2. how/when do we stop splitting?  
&nbsp;&nbsp;&nbsp;-we stop splitting when a node is 100% one class (no impurities)  
&nbsp;&nbsp;&nbsp;-we stop splitting when splitting a node will result in the tree exceeding a predefined maximum depth  
&nbsp;&nbsp;&nbsp;-we stop splitting when improvements in purity score are below a threshold  
&nbsp;&nbsp;&nbsp;-we stop splitting when the number of examples in a node is below a threshold  

