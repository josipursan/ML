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
  
## Measuring purity  
https://www.coursera.org/learn/advanced-learning-algorithms/lecture/6jL2z/measuring-purity  
  
-entropy is used as a measure of impurity  
  
-$p_{1}$ denotes the fraction of examples that are cats  
-$p_{0}$ denotes the fraction of examples that are NOT cats ($p_{0} = 1 - p_{1}$)  
-remember good old Shannon : https://en.wikipedia.org/wiki/Entropy_(information_theory)  

-remember that entropy is basically an inverted parabola  
-x axis is where our $p_{1}$ values is located  
-y axis represents the entropy (impurity) for the given $p_{1}$ value  
  
-what is the worst scenario? Having an equal split of classes, ie. having half cats and half dogs - this yields greatest entropy, ie. greatest level of impurity  
  
  
Entropy : $H(p_{1}) = -p_{1}log_{2}(p_{1}) - p_{0}log_{2}(p_{0}) = -p_{1}log_{2}(p_{1}) - (1-p_{1})log_{2}(1-p_{1})$  
  
## Choosing a split : information gain  
https://www.coursera.org/learn/advanced-learning-algorithms/lecture/ZSbs2/choosing-a-split-information-gain  
  
-now we will learn how entropy gets used to choose how to split/on what to split  
  
<p style="text-align: center">
    <img src="./screenshots/w4_choosing_split_information_gain.png"/>
</p>  
  
-observe the screenshot attached above  
  
-there are three different features we can split on : **ear shape**, **face shape**, and **whiskers**  
-there are 10 subjects in total that we can test for these features, of which 5 are cats and 5 are dogs  
  
-occurence, ratio, fraction, or however you want to call it, of cats for each branch is represented with $p$ value  
-entropy for each $p$ is also visible in the screenshot above  
  
-how will we, using this information, choose which feature to split on?  
-we will use the **weighted averages** of entropy  
**WHY?**  
-because it matters how much examples went into each branch  
&nbsp;&nbsp;&nbsp;&nbsp;-a branch that has a high entropy (ie. high impurity) is worse than a branch with only few examples yet comparable entropy (impurity)  
&nbsp;&nbsp;&nbsp;&nbsp;-but why does this matter? Because it is far more important to have low entropy (impurity) in a branch that handles a lot of examples!  
  
&nbsp;&nbsp;&nbsp;&nbsp;-however, computing entropy using weighted averages is not the end  
&nbsp;&nbsp;&nbsp;&nbsp;-we will actually compute the difference between entropy at the root node and the weighted average entropy of the branches  
&nbsp;&nbsp;&nbsp;&nbsp;-note that entropy of root node in our case is 1 (highest possible) - why? Because of the 10 subjects we are examining, 5 are cats and 5 are dogs, therefore $p(rootNode) = \frac{5}{10} = 0.5$ (maximum impurity)  
  
&nbsp;&nbsp;&nbsp;&nbsp;-this difference between root node entropy and the weighted average entropy of branches is called **information gain**  
  
-information gain computation is shown below :  
<p style="text-align: center">
    <img src="./screenshots/w4_information_gain_comptation.png"/>
</p>

-given the information gain for splitting on ear shape (0.28), face shape (0.03) and whiskers (0.12), we would choose to split on ear shape since it provides the greatest information gain  
    
-**information gain** - measures reduction in entropy, ie. how much information we gained by choosing a specific feature to split on (for which this information gain computation was done)  
  
-information gain gets used as an improvement threshold - if splitting does not yield an information gain greater than some predefined threshold, we will stop splitting to avoid creating an overburdeneds/overly complex tree  
  
### Information gain - general notation/implementation  
  
-let us now write what the general implementation of information gain looks like (splitting on *ear shape* is used as an example)  
  
$p_{1}^{left}$ - number of examples in the left subtree that really are cats ($\frac{4}{5} = 0.8$)  
$w^{left}$ - fraction of examples, out of all of the examples at the root node, that went to the left subbranch ()$\frac{5}{10}$  
  
-same logic applies for the right subbranch  
$p_{1}^{right} = \frac{1}{5} = 0.2$  
$w^{right} = \frac{5}{10}$  
  
$p_{1}^{root}$ - fraction of examples that are positive (ie. examples that **ARE** cats) in the root node  
  
$informationGain = H(p_{1}^{root}) - (w^{left}H(p_{1}^{left}) + w^{right}H(p_{1}^{right}))$  
  
-what is $w^{left/right}$? It is basically the weighting factor, ie. how many of any examples (whether true class or false class) ended up in the observed subbranch  
  
## Putting it all together  
https://www.wikipedia.org/https://www.coursera.org/learn/advanced-learning-algorithms/lecture/ZSbs2/choosing-a-split-information-gain   
  
-start with all examples at the root node  
-compute information gain for all possible features, and pick the one with the highest information gain  
-split the dataset according to selected feature, thus creating left and right branches of the tree  
-keep repeating process explained in second and third hyphen above until stopping criteria is met :  
&nbsp;&nbsp;&nbsp;&nbsp;-a node is 100% one class  
&nbsp;&nbsp;&nbsp;&nbsp;-splitting a node will result in tree exceeding maximum depth  
&nbsp;&nbsp;&nbsp;&nbsp;-information gain from additiona splits is below a set threshold  
&nbsp;&nbsp;&nbsp;&nbsp;-number of examples in a node is below a threshold  
  
