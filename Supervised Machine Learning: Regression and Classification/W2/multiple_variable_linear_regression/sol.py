import numpy as np
import matplotlib.pyplot as plt

all_costs = []
all_w_vec_predictions = []
all_b_predictions = []

"""
Each row of x_vals matrix represents one training example, or in other words, each row represents one house described by all of the available features.
Matrix is of m,n dimensions - m rows, n columns

x^i - i-th vector in x_vals (ie. one row) (note that i and m represent the same thing in this context)
x_j^i - represents j-th feature in the i-th training example (note that [i,j] and [m,n] indexation represent the same thing)
"""

"""
Function : compute_cost()

Parameters : x_vals, y_vals, w_vec, b
    Why these parameters?
    x_vals is necessary to compute y_hat for the new w and b predictions
    y_vals is necessary to compute the error of y_hat values in comparison to y_vals (ie. how far off y_hat are from the real, true data that is y_vals)
    w_vec is a row vector made up of the w predictions (so it is essentially the new w parameter used to generate new y_hat values)
    b is simply the b prediction used for generating new y_hat values

Return : computed cost
"""
def compute_cost(x_vals, y_vals, w_vec, b):
    m,n = x_vals.shape

    cost = 0
    for i in range(m):
        f_wb = np.dot(x_vals[i], w_vec) + b
        cost += (f_wb - y_vals[i])**2
    cost = (cost/(2*m))
    return cost

"""
Function : gradient_descent()

Parameters : x_vals, y_vals, w_vec, b
    Why these parameters?
    x_vals is necessary to compute the error term in the derivative expression
    y_vals is, like x_vals, necessary to compute the derivative expression
    w_vec is necessary to compute the y_hat values of the newly adjusted model
    b is necessary to compute the y_hat values of the newly adjusted model

Steps : 
    1. gradient_descent() is called with the updated w_vec and b values
    2. Compute the cost using the latest model parameters
    3. Update the w_vec and b terms to keep reducing the cost (break this down : first do the summation part of the derivative, then multiply this with 1/m, then multiply this by alpha, and only after this
        you can update w_vec and b simultaneously)

        3.1. Remember! f_wb = np.dot(w_vec, x_vals[i]) + b
            In a for loop, traversing rows of matrix x_vals, do the f_wb computation
            In a for loop nested in the loop above, traversing n columns of each row in matrix x_vals, multiply f_wb like this : f_wb += f_wb*x[i,j]
            When both loops are done, summation is done, and the end result needs to be multiplied by 1/m
            This gives you your end result for the whole derivative term

        3.2. Result from 3.1. gets multiplied by alpha

        3.3. End action is to simultaneously update w_vec and b (after this move you would run grad_desc again if necessary/if user has requested more iterations)

Return : nothing
"""

def gradient_descent(x_vals, y_vals, w_vec, b):
    m,n = x_vals.shape
    alpha = 5.0e-7

    cost_for_given_w_vec_b = compute_cost(x_vals, y_vals, w_vec, b)
    all_costs.append(cost_for_given_w_vec_b)    # used to just save all costs in one place

    dj_dw_j = np.zeros((n,))    # Why is dj_dw_j declared like this? There are as much of w parameters as there are input variables - 4 in our case
    dj_db = 0
    for i in range(m):
        f_wb = np.dot(x_vals[i], w_vec) + b # Computing what our proposed model will give for outputs
        error_term = f_wb - y_vals[i]   # For each row (each house in this example) we compute how much the latest proposed model differs from the real output data;
        #for j in range(n):
            dj_dw_j[j] = dj_dw_j[j] + error_term * x_vals[i,j]
        dj_db += error_term

    dj_dw_j = (dj_dw_j/m)
    dj_db = (dj_db/m)

    w_vec_temp = w_vec-(alpha*dj_dw_j)
    b_temp = b-(alpha*dj_db)

    all_w_vec_predictions.append(w_vec_temp)
    all_b_predictions.append(b_temp)

def multiple_variable_linear_regression():
    x_vals = np.array([[2104, 5, 1, 45],[1416, 3, 2, 40],[852, 2, 1, 35]])
    y_vals = np.array([460, 232, 178])

    # Let's check out the shape of our arrays, as well as their contents
    print("x_vals shape : {}\nx_vals : {}\n\n".format(x_vals.shape, x_vals))
    print("y_vals shape : {}\ny_vals : {}\n\n".format(y_vals.shape, y_vals))

    number_of_iterations = 1000
    initial_w_assumption = np.zeros((x_vals.shape[1],)) # Why x_vals.shape[1] for the array shape? Because vector w has dimension 4, equaling that of n dimension in x_vals, or in other words number of columns == dim w
    b_assumption = 0
    all_w_vec_predictions.append(initial_w_assumption)
    all_b_predictions.append(b_assumption)

    for i in range(number_of_iterations):
        gradient_descent(x_vals, y_vals, all_w_vec_predictions[-1], all_b_predictions[-1])  # Why pass these two global arrays used to store all w and b values? Because they will always contain the latest entries. Instead of having a couple of more variables, you can simply grab the last element added to the array

    print("Last cost after {} iteratons : {}\n".format(number_of_iterations, all_costs[-1]))
    print("Last w prediction : {}\nLast b prediction : {}\n".format(all_w_vec_predictions[-1], all_b_predictions[-1]))
    y_predicted = np.dot(x_vals, all_w_vec_predictions[-1]) + all_b_predictions[-1]
    print("Real y_vals : {}\nPredicted y : {}\n".format(y_vals, y_predicted))
    plt.plot([i for i in range(number_of_iterations)], all_costs)
    plt.title("Cost function W.R.T number of iterations", loc="center")
    plt.xlabel("number of iterations")
    plt.ylabel("cost function")
    plt.show()

multiple_variable_linear_regression()