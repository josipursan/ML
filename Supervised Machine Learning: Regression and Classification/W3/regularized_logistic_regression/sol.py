"""
ref_1 : https://github.com/josipursan/ML/blob/main/Supervised%20Machine%20Learning%3A%20Regression%20and%20Classification/W3/Notes.md 
Document containing necessary equations, as well as explanations used in this .py file.
Specific chapter in ref_1 : https://github.com/josipursan/ML/blob/main/Supervised%20Machine%20Learning%3A%20Regression%20and%20Classification/W3/Notes.md#regularized-logistic-regression 
"""

import numpy as np
import math
import matplotlib.pyplot as plt

all_costs = []
all_w_attempts = []
all_b_attempts = []

def sigmoid(z):
    return (1/(1 + math.exp(-z)))

def compute_cost(x_vals, y_vals, w, b, lam_bda):
    m,n = x_vals.shape

    cost = 0
    for i in range(m):
        z_i = np.dot(w, x_vals[i]) + b
        f_wb = sigmoid(z_i)
        cost += y_vals[i]*np.log(f_wb) + (1-y_vals[i])*np.log(1-f_wb)
    
    regularization_term = 0
    for j in range(n):
        regularization_term += w[j]**2
    
    regularization_term = (lam_bda/(2*m))*regularization_term

    cost = -(cost/m) + regularization_term
    all_costs.append(cost)


def regularized_gradient_descent(x_vals, y_vals, w, b, lam_bda, alpha):
    m,n = x_vals.shape

    dj_dw_j = np.zeros((x_vals.shape[1], ))
    dj_db = 0

    for i in range(m):
        z_i = np.dot(w, x_vals[i]) + b
        f_wb = sigmoid(z_i)
        error_term = f_wb - y_vals[i]

        dj_db += error_term

        for j in range(n):
            dj_dw_j[j] += error_term*x_vals[i][j] + (lam_bda/(2*m))*w[j]
    
    dj_db = dj_db/m
    dj_dw_j = dj_dw_j/m

    w = w - alpha*dj_dw_j
    b = b - alpha*dj_db
    all_w_attempts.append(w)
    all_b_attempts.append(b)

    compute_cost(x_vals, y_vals, w, b, lam_bda)

    return w,b

def main():
    x_vals = np.array([[1.1, 0.8], [2.54, 1.11], [9.54, 3.33], [5.17, 4.44], [1, 1], [4, 4]])
    y_vals = np.array([1, 1, 0, 1, 0, 1])

    alpha = 1e-5
    lam_bda = 1e-5

    number_of_iterations = 50000

    w = np.zeros((x_vals.shape[1], ))
    b = 0

    for i in range(number_of_iterations):
        w, b = regularized_gradient_descent(x_vals, y_vals, w, b, lam_bda, alpha)

    print("Last attempted w + b : {} + {}\nCost of last hypothesis : {}\nalpha : {}\nlambda : {}\niterations : {}\n".format(all_w_attempts[-1], all_b_attempts[-1], all_costs[-1], alpha, lam_bda, number_of_iterations))
    
    plt.plot([i for i in range(len(all_costs))], all_costs)
    plt.xlabel("Number of iterations")
    plt.ylabel("Cost")
    plt.title("Cost W.R.T. number of iterations")
    plt.show()


main()