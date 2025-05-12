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
  
-state activation function is the key quantityy reinforcement learning algorithms are trying to compute  
  
-typical representation of a *state value activation function* : $Q(s,a) = x$  
&nbsp;&nbsp;&nbsp;-*s* - state the system is currently in  
&nbsp;&nbsp;&nbsp;-*a* - action you might choose to do in state *s*  
&nbsp;&nbsp;&nbsp;-*x* - return if you start in state *s* and take action *a* (once), and then behave optimally after that   
  
-the issue with the description above is how do we know what is *optimal* behaviour?  
&nbsp;&nbsp;&nbsp;-do not worry about this for now, one of later videos will provide more context and sense  
  
-let's look at a policy example :  
<p style="text-align: center">
    <img src="./screenshots/reinforcementLearning_stateActionValueFunction.png"/>
</p>  
-this is a pretty good policy : in states 2,3,4 you move left, and go right from state 5  
  
-this is actually the optimal policy for our Mars rover example when discount factor ($\gamma$)  
  
-imagine we are in state 2, and choose to go right : $Q(2, right)$  
-since we are in state 2, and chose to go right, we end up in state 3  
-looking at the optimal policy shown above we can see that the optimal behaviour from state 3 is to go left  
-therefore we move left, ending up again in state 2  
-optimal behaviour from state 2 is to move left, ending up in state 1, the highest rewarding one, ending our traversal since we've reached optimal state  
-all in all, our traversal is made up of reward for the starting step, reward for the first step (from 2 to 3), reward for second step (from 3 to 2) and the final step (from 2 to 1) :  
$ 0 + (0.5)0 + (0.5)^{2}0 + (0.5)^{3}100 = 12.5 $  
  
-now imagine we are in state 2, and we choose to go left : $Q(2, left)$  
-we start in state 2 (reward is 0) and move left, leading us to state 1 (reward 100) where we finish our movement  
$0 + (0.5)100 = 50$
  
-here is a screenshot showing what the computed moves look like for all states :  
<p style="text-align: center">
    <img src="./screenshots/reinforcementLearning_stateActionValueFunction_2.png"/>
</p>  
  
-values shown above tell you what is the value, how good it is to do a certain action *a* when in state *s* (and then behave optimally after the action *a*)  
  
-state-action value function is also called *Q-function* (is also sometimes referred to as Q* or *optimal Q function*)  
  
-the best possible return from state is $maxQ(s,a)$  
  
-the best possible action in state *s* is the action *a* that gives $maxQ(s,a)$  
  
-when you compute all of the possible rewards from all states you can determine what the optimal *Q* function is (optimal movements are shown on the first screenshot above)  
  
## Bellman equation  
https://www.coursera.org/learn/unsupervised-learning-recommenders-reinforcement-learning/lecture/3Wpee/bellman-equation  
  
-Bellman equation is one of the most important equations in unsupervised learning, enabling us to compute actions *a* depending on states *s*  
  
-*s* - current state  
-*R(s)* - reward of current state  
-*a* - current action (the action taken when in state *s*)  
-*s'* - state you get to after taking action *a* (*s'* is read as *s prime*)  
-*a'* - action that you take in state *s'*  
  
**Bellman equation**  
$Q(s,a) = R(s) + \gamma max_{a}Q(s', a')$  
  
-the return under the used set of assumptions (*s* and *a*) is equal to reward of current state (*R(s)*) plus the discount factor gamma ($\gamma$) times maximum value, over all possible actions *a'*, of the new state we just got to ( *Q(s', a')* )  
  
-an example of Bellman's equation follows :  
&nbsp;&nbsp;&nbsp;&nbsp;-imagine we are in state 2 and have chosen action *a* to be *right*  
&nbsp;&nbsp;&nbsp;&nbsp;-imagine another example : we are in state 4 and choose to go *left*  
&nbsp;&nbsp;&nbsp;&nbsp;-the screenshot shown below details all of the computations, as well as visualizations  

<p style="text-align: center">
    <img src="./screenshots/bellmanEquation.png"/>
</p>  
  
-Bellman equation, when in terminal states, simplifies to $Q(s,a) = R(s)$ (the expression in the top right section of the screenshot above)  
  
-Bellman equation can be broken down into two parts :  
&nbsp;&nbsp;&nbsp;- $R(s)$ - the reward you get right away (in literature also referred to as *immediate reward*)  
&nbsp;&nbsp;&nbsp;- $max_{a'}Q(s', a')$ - return from behaving optimally starting from state *s'*  
  
## Stochastic environment  
https://www.coursera.org/learn/unsupervised-learning-recommenders-reinforcement-learning/lecture/rL525/random-stochastic-environment-optional  
  
-in real life, the outcome is not always completely reliable due to everchanging circumstances, as well as any unknown variables affecting the system we are designing  
  
-for random, stochastic, environments we need a generalization of the reinforcement learning framework discussed up to this point  
  
-consequences of actions are now no longer modeled as deterministic events, but rather probabilistic  
-continuing on with our Mars rover example, choosing to move left (action *a*) from current state *s* no longer results in a definitive move to the state left of our current state  
&nbsp;&nbsp;&nbsp;-now we have two probabilities : a probability that we actually will move to left, and a probability that we will move to the right  
  
-considering we are dealing with a probabilistic environment now, our goal isn't anymore to maximize the return, but to rather maximize the average return value of all of the possible event chains our system can experience  
  
-this average is referred to as *expected return* (ie. average)  
  
