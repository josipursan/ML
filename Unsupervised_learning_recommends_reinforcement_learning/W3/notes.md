# Reinforcement learning  
## The return in reinforcement learning systems  
https://www.coursera.org/learn/unsupervised-learning-recommenders-reinforcement-learning/lecture/5SCL1/the-return-in-reinforcement-learning  
  
-discount factor ($\gamma$) - makes our reinforcement learning algorithm *impatient*  
&nbsp;&nbsp;&nbsp;-by *impatient* we mean that the first reward gets the full credit, a little bit less credit to the reward from the second step, even less credit to the reward given by the third step etc.  
&nbsp;&nbsp;&nbsp;-the more steps you need the more penalization is enforced by the discount factor  
&nbsp;&nbsp;&nbsp;-in many reinforcement learning algorithms the discount factor is usually a value pretty close to 1  
&nbsp;&nbsp;&nbsp;-lower discount factor values (closer to zero) heavily penalize future rewards (ie. heavily penalizes more steps), and vice versa  
  
## Making decisions : policies in reinforcement learning  
https://www.coursera.org/learn/unsupervised-learning-recommenders-reinforcement-learning/lecture/BsteY/making-decisions-policies-in-reinforcement-learning  
  
-in reinforcement learning our goal is to come up with a function/policy ($\pi$) whose job it is to take any state (*s*) as an input and map it to some action (*a*) that it wants us to take  
  
## Review of key concepts  
https://www.coursera.org/learn/unsupervised-learning-recommenders-reinforcement-learning/lecture/fvSTb/review-of-key-concepts  
  
-Markov Decision Process (MDP) - the future depends only on the current state, and not on anything that might have occured prior to reaching the current state  
&nbsp;&nbsp;&nbsp;-in other words, the future depends only on where you are now, not on how you got here  
  
## State-action value function definition  
https://www.coursera.org/learn/unsupervised-learning-recommenders-reinforcement-learning/lecture/FEU97/state-action-value-function-definition  
  
