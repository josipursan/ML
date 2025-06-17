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
  
## Learning the state value function  
https://www.coursera.org/learn/unsupervised-learning-recommenders-reinforcement-learning/lecture/EH7Zf/learning-the-state-value-function  
  
-the heart of our reinforcement learning algorithm will be a deep learning network that will learn how various inputs translate to outputs,  
ie. it will learn *Q(s,a)*  
  
-the idea of using our neural network for lunar lander is to run some input (state *s*) through the network to compute which of the actions is most probable  
-the lunar lander can only do 4 things : nothing, fire left thruster, fire right thruster and fire main thruster  
  
-based on the starting state *s*, the goal of our network is to determine which of the above stated actions is the most likely to be the right one  
  
-you may notice this is equivalent to multilabel supervised learning - the *things* lunar lander does (nothing, fire left, fire right, or fire main thruster) are possible output labels, while  
the current states of the lunar lander are *x* inputs to the network  
  
-considering we don't have a dataset for the lunar lander, our initial phase will focus on simply running the lunar lander and noting what happens (*y*) for various actions (*x*)  
-depending on how many examples we want, we will end up with a number of tuples, each made up of values *s*, *a*, *R(s)* and *s'*  
&nbsp;&nbsp;&nbsp;&nbsp;-*s* - starting state  
&nbsp;&nbsp;&nbsp;&nbsp;-*a* - action taken in initial state  
&nbsp;&nbsp;&nbsp;&nbsp;-*R(s)* - reward for taking action *a* in state *s*  
&nbsp;&nbsp;&nbsp;&nbsp;-*s'* - new state reached after action *a* was taken in state *s*  
  
-however, instead of just walking around with these tuples, and putting them raw through the network, we will actually combine *s* and *a* values into one value, value *x*, and *x*'s corresponding
value *y* will be computed using Bellman equation (the whole part right of the *equals* sign in the Bellman equation can be considered as *y*)  
  
-you might wonder how is *y* computed using Bellman equation since we have no idea what the *Q* functions is - well, when we are only at the stage of acquiring training examples, we can pretty much just guess
what the *Q* function is  
  
-below is a screenshot showing everything described above :  
<p style="text-align: center">
    <img src="./screenshots/learning_the_state_value_function.png"/>
</p  
  
-now we will, mostly in pseudocode, lay out how our learning algorithm for unsupervised learning looks like  
  
*Learning algorithm*  
Initialize NN randomly as guess of *Q(s,a)*  
Repeat {  
&nbsp;&nbsp;&nbsp;Take actions in the lunar lander to collect training examples  
&nbsp;&nbsp;&nbsp;Store 10000 most recent training tuples (this approach is also called *replay buffer*)  
  
&nbsp;&nbsp;&nbsp;Train neural network :  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Create training set of 10000 training examples using  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;x = (s,a); y = R(s) + $\gamma max_{a'}$Q(s', a')  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Train $Q_{new}$ such that $Q_{new}(s,a)$ approximately equals y  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Set $Q = Q_{new}$  
}  
  
-the above described algorithm is also called **DQN** algorithm - **D**eep **Q** **N**etwork  
&nbsp;&nbsp;&nbsp;-it is called this because we are using deep learning networks to learn/tune a model to learn  
the *Q* function  
  
-the above described algorithm can be significantly improved (in the next videos)  
  
## Algorithm refinement : Improved NN architecture  
https://www.coursera.org/learn/unsupervised-learning-recommenders-reinforcement-learning/lecture/hpmUe/algorithm-refinement-improved-neural-network-architecture  
  
-problem of the previously explained DQN architecture lies in the fact that the network outputs probability only of one possible lunar lander action (left thruster, right thruster, main thruster, nothing) - this means we have to run this 4 times to check probability of each possible lunar lander action  
  
-this "issue" can be rectified by introducing 4 neurons in the output layer, instead of previously only one, and using the softmax activation function  
  
## Algorithm refinement : $\epsilon$-greedy policy  
https://www.coursera.org/learn/unsupervised-learning-recommenders-reinforcement-learning/lecture/GyBzo/algorithm-refinement-greedy-policy  
  
-how should we choose actions while still learning?  
  
-there are two obvious approaches :  
&nbsp;&nbsp;&nbsp;Option 1  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Pick the action *a* that maximizes Q(s,a)  
  
&nbsp;&nbsp;&nbsp;Option 2  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;With probability 0.95, pick the action *a* that maximizes Q(s,a)  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;With probability 0.05, pick an action *a* randomly  
  
-Option 2 is the better of the two - why?  
&nbsp;&nbsp;&nbsp;-because the random action introduces non-linearity, randomness, it allows the network to explore whether some random action, the model might consider not interesting in normal circumstances, could greatly benefit the behaviour of our model  
&nbsp;&nbsp;&nbsp;-this random pick is often called *Exploration step*  
  
&nbsp;&nbsp;&nbsp;-action that maximizes Q(s,a) is often called *Greedy action/step*  
  
-Option 2 is also known as $\epsilon$*-greedy policy*  
-in our example, epsilon ($\epsilon$) is 0.05, ie. the probability of doing the random thing  
  
-a trick often used is to start with $\epsilon$ being high (taking random actions a lot of the time), and then gradually decrease it so that with time we are relying more on our improving estimate of the *Q* function  
  
-professor mentioned reinforcement learning algorithms seem a bit more finicky, in terms of choosing the hyperparameters, than supervised learning algorithms are  
  
## Algorithm refinement : Mini-batch refinement and soft updates  
https://www.coursera.org/learn/unsupervised-learning-recommenders-reinforcement-learning/lecture/TsaXj/algorithm-refinement-mini-batch-and-soft-updates-optional  
  
-let's take a look back at first week of *Supervised Learning* course : https://github.com/josipursan/ML/blob/main/Supervised%20Machine%20Learning%3A%20Regression%20and%20Classification/W1/Notes.md#gradient-descent-for-linear-regression  
 
-notice that every step of gradient descent requires computing the average for as many examples as there are in the dataset  
-you can notice this will significantly slow down gradient descent  
-the idea of **mini-batching** is to use only a part of the dataset - now on every step of grad.desc. we have to compute the average term only for a significantly smaller subset of the initial dataset, therefore speeding up grad.desc.  
  
-mini-batching can often times exhibit oscillatory behaviour in terms of finding the global minimum (ie. the smallest possible loss, therefore the best model), however, when you observe the general path of mini-batching on the gradient descent hyperplane you will notice it moves towards the global minimum, albeit a bit wonky  
-despite this wonkiness, lower computational cost of mini-batching is a significant benefit  
  
-as for **soft updates**, remember how we previously set $Q = Q_{new}$ when we explained how reinforcement algorithm works  
-this can be a bit upsetting as this new *Q* function can cause an abrupt change in the behaviour of the model due to significantly different *Q* function  
-instead of this possibly rough update, we will update the parameters making up *Q* functions using ratios :  
&nbsp;&nbsp;&nbsp; $w = 0.01w_{new} + 0.99w$  
&nbsp;&nbsp;&nbsp; $b = 0.01b_{new} + 0.99b$  
  
-these ratios (0.01 and 0.99) are hyperparameters that can be controlled  
-soft-update contributes to non-oscillatory, converging, behaviour  
  
  

