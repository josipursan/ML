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
        Dense(8, activation='relu'),    #hidden layer
        #Dense(10, activation='relu'),    #hidden layer
        #Dense(9, activation='relu'),    #hidden layer
        #Dense(8, activation='relu'),    #hidden layer
        #Dense(7, activation='relu'),    #hidden layer
        #Dense(6, activation='relu'),    #hidden layer
        Dense(5, activation='linear'),     #output layer
    ]
)

NN_model.compile(
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=tf.keras.optimizers.Adam(0.01),
)

NN_model.fit(X, y, epochs=500)

# Now we try predicting
logits = NN_model(X)
softmaxed_output = tf.nn.softmax(logits)
print("softmaxed_output : {}\n".format(softmaxed_output))

matches_counter = 0
for i in range(len(softmaxed_output)):
    index_of_max_probability = np.where(softmaxed_output[i] == max(softmaxed_output[i]))
    if(y[i] == index_of_max_probability[0]):    # if the label y class, and class predicted by model match, increment match counter
        matches_counter += 1

print("Class matches between y label and model output : {}  Percentage : {}\n".format(matches_counter, matches_counter/len(y)*100))
