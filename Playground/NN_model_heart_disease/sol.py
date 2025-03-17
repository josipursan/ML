from ucimlrepo import fetch_ucirepo 

import time
import math
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import pandas as pd
import json
# fetch dataset 
heart_disease = fetch_ucirepo(id=45) 


def the_ol_trusworthy(X, y, features_only_list):
    X_rows, X_cols = X.shape
    number_of_new_examples = 100
    #clean_X = np.empty(shape=(0, X.shape[1]))
    # What is clean_X numpy array? It is an array containing only those rows from X whose y label is 1,2,3, or 4 - we do not want to generate examples with y label 0!
    dict__min_max_per_variable = {}

    print("y shape : {}\nX shape : {}\n".format(y.shape, X.shape))

    """ for i in range(len(y)):
        if(y[i] != 0):
            clean_X = np.vstack((clean_X, X[i])) """

    #print("clean_X : {}\ndim : {}\n".format(clean_X, clean_X.shape))  # clean_X must have X.shape[0] - X_where_y_equals rows (e.g. X has 297 rows, of which 160 have y=0, hence clean_X must have 137 rows)

    min_values_columns = np.min(X, axis = 0)
    max_values_columns = np.max(X, axis = 0)

    print("min : {}\nmax : {}\n".format(min_values_columns, max_values_columns))

    for example in range(number_of_new_examples):
        current_example = np.empty(shape=(1, X_cols))

        for i in range(X_cols):
            random_value = np.random.uniform(min_values_columns[i], max_values_columns[i])
            current_example[0][i] = random_value

        X = np.vstack((X, current_example))
        y = np.append(y, math.floor(np.random.uniform(np.min(y), np.max(y))))

    return X, y

# data (as pandas dataframes) 
X = heart_disease.data.features 
y = heart_disease.data.targets

features_only_list = X.columns.tolist()
del features_only_list[-1]

print("features_only_list : {}\n".format(features_only_list))

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

print("X : {}\n".format(X[20]))
rows, cols = X.shape
print("X rows/cols : {} {}\n".format(rows, cols))
generated_Data_X, generated_y = the_ol_trusworthy(X, y, features_only_list)
print("generated_Data_X : {}\n[:10] : {}\ndims : {}\n".format(generated_Data_X, generated_Data_X[:10], generated_Data_X.shape))
print("generated_y = {}\n".format(generated_y))
print("min_generated_y : {}\nmax_generated_y : {}\n".format(np.min(generated_y), np.max(generated_y)))
time.sleep(5)


print("Pre normalization : {}\n".format(X))
tf.keras.utils.normalize(X)
print("\nAfter normalization : {}\n".format(X))

print("Example of how I will slice the initial dataset to get training set, test set and cv set\n")
print("X.shape : {}\nX.shape : {}\n".format(X.shape, X[X.shape[0]-1]))
print("0.6*X.shape[0] : {}\n0.2*X.shape[0] : {}\n0.2*X.shape[0] : {}\n".format(math.floor(0.6*X.shape[0]), math.floor(0.2*X.shape[0]), math.floor(0.2*X.shape[0])))
print("X[0:{}]\nX[{}:{}]\nX[{}:{}]\n".format(math.floor(0.6*X.shape[0]), math.floor(0.6*X.shape[0])+1, math.floor(0.6*X.shape[0]) + math.floor(0.2*X.shape[0]), math.floor(0.6*X.shape[0]) + math.floor(0.2*X.shape[0]) + 1, X.shape[0]-1))


TRAINING_SET_SIZE = 0.6
TEST_SET_SIZE = 0.15
CV_SET_SIZE = 0.15

training_set = generated_Data_X[0:math.floor(TRAINING_SET_SIZE*generated_Data_X.shape[0])]  # get first 60% of elements from X; why math.floor()? To make sure end index 0.6*X.shape[0] returns is a whole number, and also to ensure the indices won't overrun max range of array
test_set = generated_Data_X[(math.floor(TRAINING_SET_SIZE*generated_Data_X.shape[0]) + 1) : (math.floor(TRAINING_SET_SIZE*generated_Data_X.shape[0]) + math.floor(TEST_SET_SIZE*generated_Data_X.shape[0]))]
cv_set = generated_Data_X[(math.floor(TRAINING_SET_SIZE*generated_Data_X.shape[0]) + math.floor(CV_SET_SIZE*generated_Data_X.shape[0]) + 1):(generated_Data_X.shape[0] - 1)]

y_training_set = generated_y[0:math.floor(TRAINING_SET_SIZE*generated_y.shape[0])]
y_test_set = generated_y[(math.floor(TRAINING_SET_SIZE*generated_y.shape[0]) + 1):((math.floor(TRAINING_SET_SIZE*generated_y.shape[0]) + (math.floor(TEST_SET_SIZE*generated_y.shape[0]))))]
y_cv_set = generated_y[((math.floor(TRAINING_SET_SIZE*generated_y.shape[0]) + (math.floor(CV_SET_SIZE*generated_y.shape[0]) + 1))):(generated_y.shape[0] - 1)]

# variable information 
print(heart_disease.variables) 
print("\n\nX: \n{}".format(X[0:10]))  #example of accessing one column in X : X.age
print("y : {}\n".format(y[0:10]))

regularization_terms = np.arange(0.01, 0.1, 0.01, dtype="float64")

regTerm = 0.0025
NN_model = Sequential(
    [
        Dense(13, activation='relu', kernel_initializer='ones', activity_regularizer = tf.keras.regularizers.L2(l2=regTerm)),     #input layer
        Dense(12, activation='relu'),    #hidden layer
        Dense(11, activation='relu'),    #hidden layer
        Dense(10, activation='relu'),    #hidden layer
        Dense(9, activation='relu'),    #hidden layer
        #Dense(8, activation='relu'),    #hidden layer
        #Dense(7, activation='relu'),    #hidden layer
        #Dense(6, activation='relu'),    #hidden layer
        Dense(5, activation='linear'),     #output layer
    ]
)

NN_model.compile(
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=tf.keras.optimizers.Adam(0.0057),
)

NN_model.fit(training_set, y_training_set, epochs=250)

logits_training_set = NN_model(training_set)
softmaxed_training_set_predictions = tf.nn.softmax(logits_training_set)

#Yes, yes, repeated code, I know...
matches_training_set = 0
for i in range(len(softmaxed_training_set_predictions)):
    index_max_probability_class_training_set = np.where(softmaxed_training_set_predictions[i] == max(softmaxed_training_set_predictions[i]))
    if(y_training_set[i] == index_max_probability_class_training_set[0]):
        matches_training_set += 1


# Now we try predicting
logits = NN_model(test_set)
softmaxed_output = tf.nn.softmax(logits)

matches_counter = 0
for i in range(len(softmaxed_output)):
    index_of_max_probability = np.where(softmaxed_output[i] == max(softmaxed_output[i]))
    if(y_test_set[i] == index_of_max_probability[0]):    # if the label y class, and class predicted by model match, increment match counter
        matches_counter += 1

logits_cv_test = NN_model(cv_set)
cv_set_softmaxed_model_output = tf.nn.softmax(logits_cv_test)
cv_test_matches = 0
for i in range(len(cv_set_softmaxed_model_output)):
    index_of_max_probability_cv = np.where(cv_set_softmaxed_model_output[i] == max(cv_set_softmaxed_model_output[i]))
    if(y_cv_set[i] == index_of_max_probability_cv[0]):
        cv_test_matches += 1

print("TRAINING SET | Class matches between y label and model output : {}  Percentage : {}\n".format(matches_training_set, matches_training_set/len(y_training_set)*100))
print("CV_SET | Class matches between y label and model output : {}  Percentage : {}\n".format(cv_test_matches, cv_test_matches/len(y_cv_set)*100))
print("TEST_SET | Class matches between y label and model output : {}  Percentage : {}\n".format(matches_counter, matches_counter/len(y_test_set)*100))
print("NN_model losses : {}\n".format(NN_model.losses))
    

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