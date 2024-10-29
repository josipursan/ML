"""
Loss for single data point in logistic regression is defined as : 
    loss = -ylog(f_w_b(x)) - (1-y)log(1-f_w_b(x))

f_w_b is defined as : f_w_b = g(z)

z = w DOT x + b

g(z) = 1/(1 + e^(-z))

What is the point of this code?
    Run cost function on randomly generated w and b values to see how well it performs compared to the real data.

1. Generate a couple of random w and b values (remember that w will be a matrix, each row representing w values applied to each individual row in x_train matrix)
2. Run cost function on all of these randomly generated w,b values
    2.1. To compute the cost you must iterate over all of the examples (x_train) computing the loss for each individual example, and then summing it up at the end.

Another way to write x_train array : 

              0.5  1.5
               1     1
              1.5  0.5
    x_train =  3    0.5
               2     2
               1    2.5

x_train is a matrix made up of m x n elements - m represents the number of rows (each row being an individual example), n represents the number of columns.
Out intermediary array used to store costs for each row will be of m dimension.
"""

import numpy as np
import math
import random
import matplotlib.pyplot as plt

all_f_w_b_values = []
all_z_i_values = []
all_final_costs = []
cost_for_each_individual_row = []

def sigmoid(x):
    return (1/(1 + math.exp(-x)))

"""
Function : compute_cost()

Parameters : x_train, y_train, w_vec_prediction, b_prediction
    Why these parameters?
    x_train is needed to run our model guess and see how far off he results are this model gives us
    y_train represents the true, real values against which we will be checking how well our model is performing
    w_vec_prediction is a vector of assumed parameter values
    b_prediction is a scalar; last assumption for b value in our model

Returns : total cost for the given w_vec_prediction and b_prediction parameters
"""

def compute_cost(x_train, y_train, w_vec_prediction, b_prediction):
    cost = 0
    cost_for_each_row = []

    m,n = x_train.shape

    for i in range(m):  # for each available row (ie. example) in x_train ...
        z_i = np.dot(x_train[i], w_vec_prediction) + b_prediction
        f_w_b = sigmoid(z_i)
        cost += -y_train[i]*np.log(f_w_b) - (1-y_train[i])*np.log(1 - f_w_b)

        cost_for_each_row.append(-y_train[i]*np.log(f_w_b) - (1-y_train[i])*np.log(1 - f_w_b))
        all_f_w_b_values.append(f_w_b)
        all_z_i_values.append(z_i)
    
    cost_for_each_individual_row.append(cost_for_each_row)

    cost = cost/m
    all_final_costs.append(cost)
    return cost

def run_logistic_regression_cost():
    x_train = np.array([[0.5, 1.5], [1, 1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])
    y_train = np.array([0, 0, 0, 1, 1, 1])
    

    """
    Now we will generate 10 r andom w and b predictions.
    Remember! : each vector w must be of n dimension because each row in x_train has 2 columns, therefore each w vector needs two w paramters to predict those two columns.
    
    What is the point of these 10 random w and b predictions? They are simulating grad desc attempts, ie. pretending we have various different models, except these models probably won't be
    improving, instead they are here just to visualize how the cost function works for logistic regression.
    """
    number_of_random_predictions = 10
    w_vec_assumptions = np.zeros((number_of_random_predictions, x_train.shape[1]))
    b_assumptions = np.zeros((number_of_random_predictions, ))

    for i in range(number_of_random_predictions):   # for each row in x_train ...
        for j in range(x_train.shape[1]): #for each column in x_train generate a random w parameter for the x_train[i][j]-th element
            w_vec_assumptions[i][j] = random.uniform(-5.0, 5.0)
        b_assumptions[i] = random.uniform(-5.0, 5.0)

    # Why are you setting these values manually here when these indices have already been populated by the random.uniform() calls above? Because w = (1,1) and b = 3 were one example of parameters used in the online lab, so I0ve
    # chosen to put them here just to double check how my cost function implementation is behaving compared to the implementation shown in course labs - cost compuation is running ok!
    w_vec_assumptions[0][0] = 1
    w_vec_assumptions[0][1] = 1
    b_assumptions[0] = -3

    print("w_vec_assumptions : \n{}\nw_vec_assumptions shape : {}\n".format(w_vec_assumptions, w_vec_assumptions.shape))
    print("b_assumptions : \n{}\nb_assumptions shape : {}\n".format(b_assumptions, b_assumptions.shape))

    for i in range(number_of_random_predictions):
        returned_cost = compute_cost(x_train, y_train, w_vec_assumptions[i], b_assumptions[i])

        print("Current w : {}\nCurrent b : {}\nCost for given w and b : {}\n".format(w_vec_assumptions[i], b_assumptions[i], returned_cost))
    
    plt.figure(1)
    ax = plt.axes(projection='3d')
    ax.scatter3D([item[0] for item in w_vec_assumptions], b_assumptions, all_final_costs, label="Cost function W.R.T w1,b")
    plt.legend(loc="upper left")
    x_y_z_3dPlot_label_font_size = 10
    ax.set_xlabel("w1", fontsize = x_y_z_3dPlot_label_font_size)
    ax.set_ylabel("b", fontsize = x_y_z_3dPlot_label_font_size)
    ax.set_zlabel("cost", labelpad=10, fontsize = x_y_z_3dPlot_label_font_size)
    plt.show()

    plt.figure(2)
    ax = plt.axes(projection='3d')
    ax.scatter3D([item[1] for item in w_vec_assumptions], b_assumptions, all_final_costs, label="Cost function W.R.T w2,b")
    plt.legend(loc="upper left")
    x_y_z_3dPlot_label_font_size = 10
    ax.set_xlabel("w2", fontsize = x_y_z_3dPlot_label_font_size)
    ax.set_ylabel("b", fontsize = x_y_z_3dPlot_label_font_size)
    ax.set_zlabel("cost", labelpad=10, fontsize = x_y_z_3dPlot_label_font_size)
    plt.show()

run_logistic_regression_cost()


