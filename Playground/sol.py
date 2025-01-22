from ucimlrepo import fetch_ucirepo 

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

#X = X.drop(['ca'], axis=1)

X = X.to_numpy()
y = y.to_numpy()
rows_to_remove = []

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

tf.keras.utils.normalize(X)

# variable information 
print(heart_disease.variables) 
print(heart_disease)
print("\n\nX: \n{}".format(X))  #example of accessing one column in X : X.age
print("\n\ny : \n{}".format(y))

NN_model = Sequential(
    [
        Dense(12, activation='relu'),     #input layer
        Dense(8, activation='relu'),    #hidden layer
        Dense(5, activation='softmax'),     #output layer
    ]
)

NN_model.compile(
    loss = tf.keras.losses.SparseCategoricalCrossentropy(),
    optimizer=tf.keras.optimizers.Adam(0.01),
)

NN_model.fit(
    X, y,
    epochs=10
)

# Now we test our model (although here it is done using the same dataset used to train the NN)
NN_output_probabilities = NN_model.predict(X)
print("\nNN_output_probabilities[0:5] : \n{}\n".format(NN_output_probabilities[0:5]))