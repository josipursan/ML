import numpy as np
import math
import time
import matplotlib.pyplot as plt
from ucimlrepo import fetch_ucirepo 
  
# fetch dataset 
#magic_gamma_telescope = fetch_ucirepo(id=159)
  
# data (as pandas dataframes) 
#X = magic_gamma_telescope.data.features
#y = magic_gamma_telescope.data.targets

#print("All features\nX : {}\n".format(X))
#print("Example of grabbing some features from the pandas df\nX.fLength : {}\n\nX.fAsym : {}\nX.fSize : {}\n".format(X.fLength, X.fAsym, X.fSize))
#print("\nAll target values : {}\n".format(y))

# metadata 
#print(magic_gamma_telescope.metadata) 
  
# variable information 
#print(magic_gamma_telescope.variables)


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

def y_vals_convert_to_binary(y_vals):
    m,n = y_vals.shape

    converted_y_vals = np.zeros((m, n))

    for i in range(m):
        if y_vals[i] == 'h':
            converted_y_vals[i] = 1
        if y_vals[i] == 'g':
            converted_y_vals[i] = 0

    return converted_y_vals

def main():
    alpha = 1e-5
    lam_bda = 1e-5
    number_of_iterations = 1000

    magic_gamma_telescope = fetch_ucirepo(id=159)
    x_pandas_df = magic_gamma_telescope.data.features
    y_pandas_df = magic_gamma_telescope.data.targets

    # converting pandas dataframe to numpy array. Why? Because I like it better when running grad desc.
    x_vals = x_pandas_df.to_numpy()
    y_vals = y_pandas_df.to_numpy()

    print("x_pandas_df : {}\n".format(x_pandas_df))
    #print("y_pandas_df : {}\n".format(y_pandas_df))

    #print("x_vals : {}\nx_vals shape : {}\n".format(x_vals, x_vals.shape))
    #print("y_vals : {}\ny_vals shape : {}\n".format(y_vals, y_vals.shape))

    y_vals = y_vals_convert_to_binary(y_vals)

    w = np.zeros((x_vals.shape[1], ))
    b = 0

    for i in range(number_of_iterations):
        print("iteration : {}\n".format(i))
        w,b = regularized_gradient_descent(x_vals, y_vals, w,  b, lam_bda, alpha)
    
    print("Last attempted w + b : {} + {}\nCost of last hypothesis : {}\nalpha : {}\nlambda : {}\niterations : {}\n".format(all_w_attempts[-1], all_b_attempts[-1], all_costs[-1], alpha, lam_bda, number_of_iterations))
    
    plt.plot([i for i in range(len(all_costs))], all_costs)
    plt.xlabel("Number of iterations")
    plt.ylabel("Cost")
    plt.title("Cost W.R.T. number of iterations")
    plt.show()

main()



