"""
ref_1 : https://github.com/josipursan/ML/blob/main/Supervised%20Machine%20Learning%3A%20Regression%20and%20Classification/W1/gradient_descent_linear_regression/solutionExplanations.md

To run grad desc, you will iteratively have to improve w and b parameters using equations (2) and (3) from ref_1.

For each w and b pair you will run x labels through it (the current w and b pair effectively represents our best guess for the real f(x)).
By running all x labels through your f(x) guess you will get y_hat values - these are our assumptions.
Using true y_labels, and our latest y_hat labels, we comput the error of our model, ie. the cost function.

If cost of our latest model is below a certain predefined threshold, or if its change is less than epsilon (automatic convergence test), we decalre our last hypothesis correct.
If cost of our latest model is not below a certain predefined threshold, or if its change is greater than epsilon, we declare our model is NOT good enough - now our next move is to update w and b parameters.

_________________________________________________________________________
cost of the latest model prediction
    in a for loop iterating over all available training examples (m) using variable i:
        summation_term = (y_hat - y_real)**2
    
    summation_term = (1/(2m)) * summation_term

_________________________________________________________________________
w and b update
    in a for loop iterating over all training examples using variable i:
        summation_term_w += (y_hat[i] - y_real[i])*x_label[i]
        summation_term_b += (y_hat[i] - y_real[i])

    summation_term_w = (1/m)*summation_term_w
    summation_term_b = (1/m)*summation_term_b

    summation_term_w = alpha * summation_term_w
    summation_term_b = alpha * summation_term_b

    w = w - summation_term_w
    b = b - summation_term_b
________________________________________________________________________

1. Create a data set
    -randomly create x labels
    -create a random f(x) which will be used to generate y labels

2. starting point : w = 0, b = 0

3. our starting hypothesis is f(x) = 0x+0

4. run x_labels through our hypothesis

5. compute cost function for our latest hypothesis

6. does our latest hypothesis yield a cost below a threshold/epsilon we are satisfied with (ie. model behaves good enough)?

    6.1. if yes, exit (plot some stuff to show it, printf some stuff to show how much attempts it took, etc.)
        6.1.1. exit

    6.2 if no, update the w and b parameters
        
        6.2.1 w,b updates
            for all y_vals indexed using index i:
                dj_db += (y_hat[i] - y_real[i])
                dj_dw += (y_hat[i] - y_real[i]) * x[i]

            dj_dw = (1/m) * dj_dw
            dj_db = (1/m) * dj_db   /*Comment : at this point you could also return these values - it is just a matter of style and code organziation*/

            w = w - alpha * dj_dw
            b = b - alpha * dj_db

        6.2.2. Now we have new w and b parameters, ie. a new hypothesis.

        6.2.3. Go to 4.
"""

import sys
import matplotlib.pyplot as plt
import numpy as np

# Global lists used to store all attempts for debugging and stats
all_costs = []
all_w_guesses = []
all_b_guesses = []
all_y_hat = []

def cost_function(y_vals, y_hat):
    m = len(y_vals)

    model_cost = 0
    for i in range(0, len(y_vals)):
        model_cost += (y_hat_latest_model[i] - y_vals[i])**2
    model_cost = (1/(2*m)) * model_cost

    return model_cost


def f(x):
    return ((3*x)+8)

# 1.
x_vals = np.arange(1.0, 15.0, 0.95)
y_vals = f(x_vals)

m = len(y_vals)
alpha = 0.001

# 2.
w = 0
b = 0
all_w_guesses.append(w)
all_b_guesses.append(b)

while True:
    # 4.
    y_hat_latest_model = w*x_vals + b
    all_y_hat.append(y_hat_latest_model)

    # 5.
    latest_cost = cost_function(y_vals, y_hat_latest_model)
    all_costs.append(latest_cost)

    # 6. and 6.1. and 6.1.1.
    if latest_cost < 0.001:
        print("Latest cost : {}".format(latest_cost))
        print("Latest hypothesis : {}x + {}".format(w, b))
        print("Number of iterations : {}".format(len(all_costs)))
        
        plt.figure(1)
        plt.plot(x_vals, y_vals, label="Original data", color="red")
        plt.plot(x_vals, y_hat_latest_model, label="Trained model", linestyle='dotted', color='teal')
        plt.xlabel("x_vals")
        plt.ylabel("y_vals (y_hat)")
        plt.legend(loc="upper left")

        plt.figure(2)
        plt.plot([i for i in all_costs], all_costs, label='Cost function')
        plt.xlabel("Number of iterations")
        plt.ylabel("Cost")
        plt.legend(loc="upper left")
        plt.show()
        sys.exit()

    # 6.2. and 6.2.1.
    dj_dw = 0
    dj_db = 0
    for i in range(len(y_vals)):
        dj_db += (y_hat_latest_model[i] - y_vals[i])
        dj_dw += (y_hat_latest_model[i] - y_vals[i])*x_vals[i]

    dj_dw = (1/m)*dj_dw
    dj_db = (1/m)*dj_db

    w = w - alpha*dj_dw
    b = b - alpha*dj_db
    all_w_guesses.append(w)
    all_b_guesses.append(b)

    # Now again step 4.






