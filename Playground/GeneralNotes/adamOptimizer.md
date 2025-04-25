# Adam optimizer  
  
I am writing this primarily because I wanted to find out if *Adam* performs a grid search on its own for best
possible alpha.  
  
Guess not : https://stackoverflow.com/questions/60871794/how-does-the-keras-adam-optimizer-learning-rate-hyper-parameter-relate-to-indivi  
*"As for the varying learning rate (or update), you can see the last equation (it uses m_t and v_t, these are updated in the loop) but the alpha stays fixed in the whole algorithm. This is the keras learning rate that we have to provide."* 
