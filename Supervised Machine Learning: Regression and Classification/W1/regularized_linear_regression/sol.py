"""
This sol.py file implements regularized linear regression.

Reference document ref_1 : https://github.com/josipursan/ML/blob/main/Supervised%20Machine%20Learning%3A%20Regression%20and%20Classification/W3/Notes.md 
"""

import numpy as np
import matplotlib.pyplot as plt

all_costs = []
all_w_guesses = []
all_b_guesses = []

def f(x):
    m,n =  x.shape

    y_vals = np.zeros(x.shape[0])   # y_vals is a row vector whose dimension corresponds to the number of rows in x. Why? Because each subset can be considered a dataset of its own

    w = np.array([0.54, 1.33, 4.69, 8.11, 7.65])    # these are the real w parameters that will be used to generate y values from the given inputs (array x) - it is this values we will be trying to model!
    b = 8

    for i in range(m):  # for all existing rows ...
        y_vals[i] = np.dot(w, x[i]) + b # compute the output value y_vals[i] for the given set of inputs defined by x[i]

    return y_vals


def compute_cost(x_vals, y_vals, w, b, lam_bda):
    m,n = x_vals.shape
    cost = 0
    regularization_term = 0

    for i in range(m):
        f_wb = np.dot(w, x_vals[i]) + b
        cost += (f_wb - y_vals[i])**2
    cost = cost/(2*m)

    for j in range(n):
        regularization_term += w[i]**2
    regularization_term = (lam_bda/(2*m))*regularization_term

    cost += regularization_term
    all_costs.append(cost)

def gradient_descent(x_vals, y_vals, w, b, alpha, lam_bda):
    m,n = x_vals.shape

    dj_dw_j = np.zeros([x_vals.shape[1], ]) # n, 
    dj_db = 0

    for i in range(m):
        f_wb = np.dot(w, x_vals[i]) + b
        error_term = f_wb - y_vals[i]

        dj_db += error_term

        for j in range(n):
            dj_dw_j[j] += error_term*x_vals[i][j]
    
    dj_db = dj_db/m
    dj_dw_j = dj_dw_j/m

    for j in range(n):
        dj_dw_j[j] += (lam_bda/m)*w[j]
    
    w_updated = w - alpha*dj_dw_j
    b_updated = b - alpha*dj_db
    all_w_guesses.append(w_updated)
    all_b_guesses.append(b_updated)

    compute_cost(x_vals, y_vals, w_updated, b_updated, lam_bda)

    return w_updated, b_updated


def main():
    x_vals = np.array([[2104, 5, 1, 45, 9], [1416, 3, 2, 40, 4], [852, 2, 1, 35, 11]])
    y_vals = f(x_vals)

    m,n = x_vals.shape

    number_of_iterations = 1000000
    alpha = 0.00000001
    lam_bda = 0.001

    w = np.zeros([x_vals.shape[1], ])
    b = 0

    for i in range(number_of_iterations):
        w,b = gradient_descent(x_vals, y_vals, w, b, alpha, lam_bda)

    print("Number of iterations : {}\nAlpha : {}\nlambda : {}\nmReal model : y = wx + b = 0.54x1 + 1.33x2 + 4.69x3 + 8.11x4 + 7.65x5 + 8\nLast w and b guess : {} + {}\nLast model hypothesis cost : {}\n".format(number_of_iterations, alpha, lam_bda, all_w_guesses[-1], all_b_guesses[-1], all_costs[-1]))
    
    plt.plot([i for i in range(len(all_costs))], all_costs)
    plt.xlabel("Number of iterations")
    plt.ylabel("Cost")
    plt.title("Hypothesis cost W.R.T number of iterations")
    plt.legend(loc="upper left")
    plt.show()

main()