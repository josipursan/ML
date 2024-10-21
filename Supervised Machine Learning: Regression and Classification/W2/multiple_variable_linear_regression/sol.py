"""
This sol.py implements multiple variable linear regression with grad desc.
Primary source of information for ML, during this time period, was Andrew Ng's Supervised ML course on Coursera : https://www.coursera.org/learn/machine-learning

My notes about what is implemented here : https://github.com/josipursan/ML/blob/main/Supervised%20Machine%20Learning%3A%20Regression%20and%20Classification/W2/Notes.md 

================================================================================

In this sol.py we will work on the housing data.
Housing data set consists of : house area (x1), number of bedrooms (x2), number of floors (x3), age in years (x4).
As you can see, we have 4 input features, instead of the only one we had when we were doing the univariate linear regression.


Necessary expression/equations : 

    Model : 
        f = w_vec DOT x_vec + b     (_vec represents it is a row vector; DOT represents the two operands are to be multiplied using DOT product operation)

    Update terms until convergence : 

        w = w - alpha*(partial derivative of J(w_vec, b) W.R.T w_vec) 
            NOTE : because the partial derivative is here being done W.R.T w_vec, you will have effectively have a vector of derivative values, meaning this term can be rewritten as : 
                w = w - alpha*d_vec
                e.g. : w = w - 0.1*[0.1, 1.5, 1.1, 2.9]

                So, the wisest move here would probably be to first get all of the derivatives in a vector - then 

        b = b - alpha*(partial derivative of J(w_vec, b) W.R.T to b)

        Expanding these terms a bit more, to drop the ambiguity of the partial derivative, we get : 

            w = w - alpha*(1/m)*(f(x_vec) - y)*x_n^i for i in range 1,...,m
            b = b - alpha(1/m)*(f(x_vec) - y) for i in range 1,...,m

        Cost is defined same way as before : 
            J(w,b) = (1/2m)*(f(x) - y)**2 for i in range 0,..., m-1


How will we implement this?
    We will approach this similarily to the univariate linear regression with grad desc implementation.

    1. Define some random, "real" model consisting of 4 input variables, to which you will assign some freely chosen parameters w.
    2. Generate set of x_vals
    3. Use generated x_vals, and the previously defined "real" model to generate y_vals - x_vals and y_vals are the real data, they represent data we will try to model

    4. Given the number of input variables, declare and define w row vector of appropriate dimension
    5. Initialize parameters w to 0
    6. Enter grad desc iterative process
        6.1. 
"""

import numpy as np
import matplotlib.pyplot as plt

def f(x):
    w = [0.6, 5.35, 4.3, 2.42]
    return_values = np.zeros((x.shape[0], x.shape[1]))
    for i in range(return_values.shape[0]):
        return_values[i] = np.dot(x[i], w) + 8.9
    return return_values

def multiple_variable_linear_regression_gradient_descent():
    # Let's generate a random "real" model that we will want to model
    number_of_training_examples = 4
    #x_vals = np.empty(shape=(number_of_training_examples, 4)) # 4,4 shape means 4 rows and 4 columns (4 rows represent distinct training examples; 4 columns represent features of each training example)
    x_vals = np.random.rand(number_of_training_examples, 4)
    print(x_vals)
    y_vals = f(x_vals)
    print("\n{}".format(y_vals))
    plt.scatter(x_vals, y_vals)
    plt.show()
    

    # Now we will init parameters used for grad desc
    w_vec = np.zeros(4)
    b = 0.0
    print("\nw_vec : {}\n".format(w_vec))

    alpha = 0.01

    all_y_hat_attempts = []
    all_costs = []
    all_w_attempts = []
    all_b_attempts = []
    
    all_w_attempts.append(w_vec)
    all_b_attempts.append(b)
    
    # Now we can start grad desc
    while True:
        """
            compute y_hat values for the given parameters
            compute cost function
            if cost function below wanted threshold (ie. the model is good enough), exit
        """
        y_hat_values_temp = np.zeros((x.shape[0], x.shape[1]))

        temp_cost = 0
        for i in range(x.shape[0]):
            y_hat_values_temp[i] = np.dot(x[i], w_vec) + b
            temp_cost += (y_hat_values_temp[i] - y_vals[i])**2
        temp_cost = (temp_cost)/(2*x.shape[0])  # x.shape[0] is m, ie. the number of training examples
        all_costs.append(temp_cost)

        if(temp_cost < 1):
            print("New cost below wanted threshold!\nQuitting!")
            break
        
        # If we have not reached the threshold, meaning our model is still not good enough, we need to get new w and b params

        
    print("Last cost : {}\nLast used parameters --> w : {} b : {}\n".format(all_costs[-1], all_w_attempts[-1], all_b_attempts[-1]))


multiple_variable_linear_regression_gradient_descent()
