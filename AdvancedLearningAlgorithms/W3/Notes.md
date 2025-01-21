# Advanced learning algorithms - Week 3  
https://www.coursera.org/learn/advanced-learning-algorithms/home/week/3  
  
-when the developed model has unacceptably large errors when making predictions, there is a number of things we can try to fix the model :  
&nbsp;&nbsp;&nbsp; -get more training examples  
&nbsp;&nbsp;&nbsp; -try a smaller set of feature on which you will train the model  
&nbsp;&nbsp;&nbsp; -try adding more features  
&nbsp;&nbsp;&nbsp; -try adding polynomial features  
&nbsp;&nbsp;&nbsp; -try decreasing/increasing $\lambda$ (regularization term)  
  
-this week covers **diagnostics**  
&nbsp;&nbsp; -by *diagnostics* we actually refer to various tests that can be conducted to determine what might, or might not, help improve the performance of our algorithm  
  
## Evaluating a model  
https://www.coursera.org/learn/advanced-learning-algorithms/lecture/26yGi/evaluating-a-model  
  
-do not use all of the data to train the model  
&nbsp;&nbsp;&nbsp; -split the starting dataset into **training set** and **test set**  
&nbsp;&nbsp;&nbsp; -model's parameters will be trained using the **training set**  
&nbsp;&nbsp;&nbsp; -performance of the generated model is then tested on the **test set**  
  
&nbsp;&nbsp;&nbsp; $m_{train}$ - number of training examples in the whole dataset  
&nbsp;&nbsp;&nbsp; $m_{test}$ - number of test examples in the whole dataset  
  
&nbsp;&nbsp;&nbsp; -once we have fit the model we will compute **training error** and **test error** (linear regression example) :  
<p style="text-align: center">
    <img src="./screenshots/evaluating_a_model.png"/>
</p>  
&nbsp;&nbsp;&nbsp; -why is this beneficial? If our model performs very well on our training data (low training error), but it performs very poorly on our test data (test error is very high), it means our model is not generalizing well (e.g. model is overfit, ie. emulates training data very tightly)  
  
## Model selection and training/cross validation/test sets  
https://www.coursera.org/learn/advanced-learning-algorithms/lecture/zqXm6/model-selection-and-training-cross-validation-test-sets  
  
-using all of the data from the input dataset to train the model, and then using that same data to determine the error of our model is not representative of the actual generalization error, which is most likely higher than the error computed as described above  
  
-splitting the data into a **training** and a **test* set, training the model using the **training set**, and then computing the error of the model using the **test set** is a step forward  
  
-a further refinement is to actually split the input dataset into 3 subsets - instead of splitting it only into **training set** and **test set**, we will introduce one more subset : **cross validation set**  
  
$m_{cv}$ - number of cross validation examples  
  
-cross-validation set is also called just *validation set*, or *development set*, or *dev set*  
  
<p style="text-align: center">
    <img src="./screenshots/model_selection_training_crossValidation_test_sets.png"/>
</p>

-when evaluating multiple models, we will use *cross validation error* to determine which model we should choose  
&nbsp;&nbsp;&nbsp; -generalization error is now estimated using the *test set error*  


  

