"""
ref_1 : https://github.com/josipursan/ML/blob/main/Supervised%20Machine%20Learning%3A%20Regression%20and%20Classification/W2/multiple_variable_linear_regression/explainedSolution.md

6 equations are listed in ref_1.
Relevant equations for this .py file are (1), (2), (4), (6).

(1) - hypothesis
(2) - cost function for hypothesis (ie. how far off the hypothesis is to the real model)
(4) - w update expression, ie. fully written out equation used when w needs to be updated
(6) - b update expression, ie. fully written out equation used when b needs to be updated

How does this differ from gradient descent for linear regression from W1?
    There we had only one input variable - x1.
    Now we have 4 input variables - x1, x2, x3, x4

We start out with defined training example, ie. training data :
    x_vals = [[...], [...], [...], ...]
    y_vals = [..., ..., ..., ]

x_vals is a m x n sized matrix (m = number of rows; n = number of columns).
y_vals is usually a row vector, each value representing what the true model output is for the given row in x_vals.

We want to use these training examples to establish the best possible model, so that in future we could use this model for new x input values to forecast a relevant and representative value y.

We will run our model for a certain number of iteration : number_of_iterations;

Start out with default w and b predictions : 0.

If we haven't done all of the for loop iterations, input our latest w and b predictions to grad_desc algorithm.

Run our x_vals training examples through the latest model prediction/hypothesis -  this gives us y_hat values, ie. the y labels our assumed model spits out.

Compute the cost of our latest model - this means we are checking how much the y_hat labels (or in other words, labels spit out by the model using our latest w and b predictions) differ from our true y labels.
    The bigger the cost, the greater the error of our model.

Now we have to update our w and b predictions
    In a for loop, iterate over all rows in y_true and y_hat. Do the error term subtraction : y_hat - y_true
        In a for loop iterating over all columns, compute what the partial derivative of J W.R.T to w_j is, where j represents the current for loop iteration over columns.
        
            *NOTE : dj_dw_j is a row vector whose dimension corresponds to the number of columns in x_vals. Why? Each column corresponds to one input variable/model parameter. We are interested how much of an error each of our predicted w parameters is making, hence the nested for loop iterating over columns.*

        Once all summation terms are done, they have to be multiplied by (1/m) term - m represents the total number of training examples.

    To make our new w and b predictions, result of the operation done in the line above must be multiplied by alpha.

    Finally, to make our new w and b predictions, we subtract the above computed value from w and b values used up to this point
"""

import numpy as np
import matplotlib.pyplot as plt

allCosts = []
all_w_guesses = []
all_b_guesses = []

def compute_cost(w, b, x_vals, y_vals):
    m,n = x_vals.shape

    summation_term = 0
    for i in range(m):
        f_wb = np.dot(w, x_vals[i]) + b
        summation_term += (f_wb - y_vals[i])**2
    cost = (summation_term/(2*m))

    allCosts.append(cost)
    return cost


def multiple_variable_linear_regression(w, b, x_vals, y_vals, alpha):
    m,n = x_vals.shape

    latest_model_cost = compute_cost(w, b, x_vals, y_vals)

    dj_dw_j = np.zeros((n, ))
    dj_db = 0

    for i in range(m):
        f_wb = np.dot(w, x_vals[i]) + b
        error_term = f_wb - y_vals[i]
        dj_db += error_term
        for j in range(n):
            dj_dw_j[j] += error_term*x_vals[i][j]

    dj_dw_j = dj_dw_j/m
    dj_db = dj_db/m

    w = w - alpha*dj_dw_j   # We immediately update w and b values and return these updated values
    b = b - alpha*dj_db
    all_w_guesses.append(w)
    all_b_guesses.append(b)

    return w,b


def main():
    x_vals = np.array([[2104, 5, 1, 45], [1416, 3, 2, 40], [852, 2, 1, 35]])
    y_vals = np.array([460, 232, 178])

    number_of_iterations = 500
    alpha = 0.0000001
    
    print(x_vals)
    print(y_vals)

    # Initializing w and b assumptions
    w = np.zeros((x_vals.shape[1],))
    b = 0

    print(w)

    for i in range(number_of_iterations):
        w,b = multiple_variable_linear_regression(w, b, x_vals, y_vals, alpha)
    
    print("Number of iterations : {}\nAlpha : {}\nLast w guess : {}\nLast b guess : {}\nCost for last model : {}\n".format(number_of_iterations, alpha, all_w_guesses[-1], all_b_guesses[-1], allCosts[-1]))

    plt.plot([i for i in range(len(allCosts))], allCosts)
    plt.xlabel("Number of iterations")
    plt.ylabel("Cost")
    plt.title("Cost W.R.T to number of iterations")
    plt.legend(loc="upper left")
    plt.show()

main()