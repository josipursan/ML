from ucimlrepo import fetch_ucirepo 

import time
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import pandas as pd
# fetch dataset 
heart_disease = fetch_ucirepo(id=45) 
  
# data (as pandas dataframes) 
X = heart_disease.data.features 
y = heart_disease.data.targets 


features_only_list = X.columns.tolist()
del features_only_list[-1]

X = X.to_numpy()
y = y.to_numpy()
rows_to_remove = []

# This is just data cleanup due to some NaN values 
print("X dim : {}\n".format(X.shape))
for i in range(0, X.shape[0]):
    for j in range(0, X.shape[1]):
        if np.isnan(X[i][j]):
            print("X[{}][{}] is NaN : {}".format(i, j, X[i][j]))
            rows_to_remove.append(i)
            break

X = np.delete(X, (rows_to_remove), axis = 0)
y = np.delete(y, (rows_to_remove), axis = 0) 
print("Row containing NaN : {}\nX dim after removing offending rows : {}\n".format(rows_to_remove, X.shape))

print("Pre normalization : {}\n".format(X))
tf.keras.utils.normalize(X)
print("\nAfter normalization : {}\n".format(X))

# variable information 
print(heart_disease.variables) 
print("\n\nX: \n{}".format(X[0:10]))  #example of accessing one column in X : X.age
print("y : {}\n".format(y[0:10]))


NN_model = Sequential(
    [
        Dense(12, activation='relu'),     #input layer
        Dense(12, activation='relu'),    #hidden layer
        Dense(5, activation='softmax'),     #output layer
    ]
)

NN_model.compile(
    loss = tf.keras.losses.SparseCategoricalCrossentropy(),
    optimizer=tf.keras.optimizers.Adam(0.01),
)

NN_model.fit(
    X, y,
    epochs=500
)

# Now we test our model (although here it is done using the same dataset used to train the NN)
NN_output_probabilities = NN_model.predict(X)
for i in range(X.shape[0]):  # for every data row in X ...
    for j in range(len(features_only_list)): # for every feature available in each data row of X...
        print("{} : {}".format(features_only_list[j], X[i][j]))
    
    val = np.where(NN_output_probabilities[i] == max(NN_output_probabilities[i]))
    print("NN_output_probabilities[{}] : {}\nHeart condition classification : {}\ny : {}\n\n".format(i, NN_output_probabilities[i], NN_output_probabilities[i][val[0]], y[i]))

for i in range(0,10):
    print("NN_output_probabilities : {}\ny : {}\n".format(NN_output_probabilities[i], y[i]))