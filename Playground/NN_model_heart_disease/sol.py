from ucimlrepo import fetch_ucirepo 

import time
import math
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

print("Example of how I will slice the initial dataset to get training set, test set and cv set\n")
print("X.shape : {}\nX.shape : {}\n".format(X.shape, X[X.shape[0]-1]))
print("0.6*X.shape[0] : {}\n0.2*X.shape[0] : {}\n0.2*X.shape[0] : {}\n".format(math.floor(0.6*X.shape[0]), math.floor(0.2*X.shape[0]), math.floor(0.2*X.shape[0])))
print("X[0:{}]\nX[{}:{}]\nX[{}:{}]\n".format(math.floor(0.6*X.shape[0]), math.floor(0.6*X.shape[0])+1, math.floor(0.6*X.shape[0]) + math.floor(0.2*X.shape[0]), math.floor(0.6*X.shape[0]) + math.floor(0.2*X.shape[0]) + 1, X.shape[0]-1))


training_set = X[0:math.floor(0.6*X.shape[0])]  # get first 60% of elements from X; why math.floor()? To make sure end index 0.6*X.shape[0] returns is a whole number, and also to ensure the indices won't overrun max range of array
test_set = X[(math.floor(0.6*X.shape[0]) + 1) : (math.floor(0.6*X.shape[0]) + math.floor(0.2*X.shape[0]))]
cv_set = X[(math.floor(0.6*X.shape[0]) + math.floor(0.2*X.shape[0]) + 1 ):(X.shape[0] - 1)]

y_training_set = y[0:math.floor(0.6*y.shape[0])]
y_test_set = y[(math.floor(0.6*y.shape[0]) + 1):((math.floor(0.6*y.shape[0]) + (math.floor(0.2*y.shape[0]))))]
y_cv_set = y[((math.floor(0.6*y.shape[0]) + (math.floor(0.2*y.shape[0]) + 1))):(y.shape[0] - 1)]

# variable information 
print(heart_disease.variables) 
print("\n\nX: \n{}".format(X[0:10]))  #example of accessing one column in X : X.age
print("y : {}\n".format(y[0:10]))


NN_model = Sequential(
    [
        Dense(12, activation='relu'),     #input layer
        Dense(11, activation='relu'),    #hidden layer
        Dense(10, activation='relu'),    #hidden layer
        Dense(9, activation='relu'),    #hidden layer
        Dense(8, activation='relu'),    #hidden layer
        Dense(7, activation='relu'),    #hidden layer
        Dense(6, activation='relu'),    #hidden layer
        Dense(5, activation='linear'),     #output layer
    ]
)

NN_model.compile(
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=tf.keras.optimizers.Adam(0.01),
)

NN_model.fit(training_set, y_training_set, epochs=500)

# Now we try predicting
logits = NN_model(test_set)
softmaxed_output = tf.nn.softmax(logits)

matches_counter = 0
for i in range(len(softmaxed_output)):
    index_of_max_probability = np.where(softmaxed_output[i] == max(softmaxed_output[i]))
    if(y_test_set[i] == index_of_max_probability[0]):    # if the label y class, and class predicted by model match, increment match counter
        matches_counter += 1

print("Len test_set : {}\nLen y_test_set : {}\n".format(len(test_set), len(y_test_set)))
print("TEST_SET | Class matches between y label and model output : {}  Percentage : {}\n".format(matches_counter, matches_counter/len(y_test_set)*100))

logits_cv_test = NN_model(cv_set)
cv_set_softmaxed_model_output = tf.nn.softmax(logits_cv_test)
cv_test_matches = 0
for i in range(len(cv_set_softmaxed_model_output)):
    index_of_max_probability_cv = np.where(cv_set_softmaxed_model_output[i] == max(cv_set_softmaxed_model_output[i]))
    if(y_cv_set[i] == index_of_max_probability_cv[0]):
        cv_test_matches += 1
print("CV_SET | Class matches between y label and model output : {}  Percentage : {}\n".format(cv_test_matches, cv_test_matches/len(y_cv_set)*100))

# Now we test our model (although here it is done using the same dataset used to train the NN)
""" NN_output_probabilities = NN_model.predict(X)
for i in range(X.shape[0]):  # for every data row in X ...
    for j in range(len(features_only_list)): # for every feature available in each data row of X...
        print("{} : {}".format(features_only_list[j], X[i][j]))
    
    val = np.where(NN_output_probabilities[i] == max(NN_output_probabilities[i]))
    print("NN_output_probabilities[{}] : {}\nHeart condition classification : {}\ny : {}\n\n".format(i, NN_output_probabilities[i], val[0], y[i]))

matches_counter = 0
for i in range(len(NN_output_probabilities)):
    index_of_max_probability = np.where(NN_output_probabilities[i] == max(NN_output_probabilities[i]))
    if(y[i] == index_of_max_probability[0]):    # if the label y class, and class predicted by model match, increment match counter
        matches_counter += 1

print("Class matches between y label and model output : {}  Percentage : {}\n".format(matches_counter, matches_counter/len(y)*100)) """