"""
Implementation presented here covers what was done in this chapter : https://github.com/josipursan/ML/blob/main/Supervised%20Machine%20Learning%3A%20Regression%20and%20Classification/Notes.md#running-gradient-descent

You already implemented cost function computation for linear eq.
Now you will make it do gradient descent, ie. run until it converges as close as possible to the data it wants to model/mimic.

========================

Steps :

    1. Generate "random" data - look up your cost function computation solution for inspiration
    2. Run gradient descent on the data
        2.1. Start out with some linear eq assumption
        2.2. Check linear eq assumption cost
        2.3. Do the partial derivatives - if partial derivative leads to reduction in cost function (ie. updated w and b parameters lead to a more precise linear eq assumption), go to 2.2.
            2.3.1 if partial derivative increases cost function, exit

    Remarks : 
        -start out with an ordinary linear equation for the original data, then move on to more complex equations - when using a simple linear equation for your real data, gradient descent should converge to exactly that linaer equation
        -plot the gradient descent in some fashion which will allow you to plot how gradient descent worked to find the optimal model
"""

"""
Update w,b parameters expressions

w_updated = w - alpha * ( (1/m)*sum_per_i( (f(x_i)-y_i)*x_i  ) )
b_updated = w - alpha * ( (1/m)*sum_per_i( (f(x_i)-y_i)  ) )

These expressions can be broken down a bit
First in a for loop, for all available values, do this : 
    
    w_summation_temp += (f(x_i) - y_i)*x_i [FOR ALL i ELEMENTS]
    
    Once all available elemens are summed up : 
    w_summation_temp = alpha * w_summaton_temp

Then do all of this for b.
Only then update w and b simultaneously.
"""

import random
import time

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

random.seed(time.time())

def f(x):
    return ((3*x)+8)  # y = f(x) = wx + b

def optimize():
    alpha = 0.0001

    x_vals = np.arange(1.0, 15.0, 0.95) # generate values in range 1-15, using 0.95 as step between values
    y_vals = f(x_vals)

    # generate random w,b parameters to start gradient descent
    w = random.uniform(-100.0, 100.0)
    b = random.uniform(-100.0, 100.0)
    print("Randomly generated initial w,b pair : {},{}\n".format(w,b))

    allCosts = []
    all_w_b_hypothesis = []
    all_y_vals = []
    all_w_b_hypothesis.append([w,b])
    print("all_w_b_hypothesis before running graddesc : {}\n".format(all_w_b_hypothesis))
    
    while True:
        # Compute y_vals with the latest model hypothesis
        y_vals_prediction = w*x_vals+b
        all_y_vals.append(y_vals_prediction)
        """ plt.plot(x_vals, y_vals_prediction, color="orange")
        plt.legend(loc="upper left")
        plt.show() """

        # Compute the cost function for the current model hypothesis
        sum_cost_func = 0
        for el in range(0, len(y_vals_prediction)):
            sum_cost_func += (y_vals_prediction[el] - y_vals[el])**2
        cost_func = (1/(2*(len(y_vals_prediction)))) * sum_cost_func
        allCosts.append(cost_func)

        if(cost_func < 0.01):
            print("\n\n\nLast computed cost is below set threshold!\nCost : {}\nExiting!\n\n".format(cost_func))
            plt.figure(1)
            plt.plot(x_vals, y_vals, label="Original data", color="red")    # let's check out our real model
            plt.plot(x_vals, y_vals_prediction, linestyle='dotted', color='teal', linewidth=3, label="BEST FIT")
            
            indices_random_five_models_to_plot = []
            for i in range(0, 5):
                indices_random_five_models_to_plot.append(random.randrange(len(allCosts)))
            for model in indices_random_five_models_to_plot:
                plt.plot(x_vals, all_y_vals[model], marker='x', label = "model_" + str(model), linewidth=1, linestyle='none')
            plt.xlabel("x_vals")
            plt.ylabel("y_vals")
            plt.legend(loc="upper left")

            # 3D plot of cost function W.R.T w,b changes
            plt.figure(2)
            ax = plt.axes(projection='3d')
            ax.scatter3D([item[0] for item in all_w_b_hypothesis], [item[1] for item in all_w_b_hypothesis], allCosts, label="Cost function W.R.T w,b")          #w,b,allCosts
            plt.legend(loc="upper left")
            x_y_z_3dPlot_label_font_size = 10
            ax.set_xlabel("w", fontsize = x_y_z_3dPlot_label_font_size)
            ax.set_ylabel("b", fontsize = x_y_z_3dPlot_label_font_size)
            ax.set_zlabel("cost", labelpad=10, fontsize = x_y_z_3dPlot_label_font_size)
            plt.show()
            break

        # Update the terms
        w_summation_temp = 0
        b_summation_temp = 0
        for el in range(0, len(y_vals_prediction)):
            w_summation_temp += (y_vals_prediction[el] - y_vals[el])*x_vals[el]
            b_summation_temp += (y_vals_prediction[el] - y_vals[el])

        if(len(allCosts) < 2):
            print("len(allCosts) = {}\tCost for {},{} : {}\n".format(len(allCosts), w, b, cost_func))
        else:
            print("Cost for {},{} : {}\nCost for previous w,b ({}) : {}\n".format(w, b, cost_func, all_w_b_hypothesis[-2], allCosts[-2]))   # why -2? Because we've already inserted our new values - therefore to retrieve the value which was last in list before we inserted the new elements, we need to grab the penultimate elements from lists (previously last element)
        
        w_summation_temp = alpha*w_summation_temp
        b_summation_temp = alpha*b_summation_temp
        w = w - w_summation_temp
        b = b - b_summation_temp
        all_w_b_hypothesis.append([w,b])
        
        print("New w,b hypothesis : {}\n".format(all_w_b_hypothesis[-1]))
        #input("Continue?\n\n")  # here only to make the debugging easier
        continue
        
    print("Numer of attempted w_b hypothesis : {}\nNumber of allCosts : {}\nNumber of all_y_vals : {}\nLast model hypothesis : w,b = {}\nOriginal model : y = 3x+8\n".format(len(all_w_b_hypothesis), len(allCosts), len(all_y_vals), all_w_b_hypothesis[-1]))
optimize()