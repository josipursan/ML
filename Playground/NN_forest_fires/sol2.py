from ucimlrepo import fetch_ucirepo, list_available_datasets
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import datetime
import time
import math

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

def column_to_categorical(column_to_convert_to_categorical):
    old_to_new_column_encoding = {}
    frequency_counts = column_to_convert_to_categorical.value_counts()
    
    counter = 0
    for key, value in frequency_counts.items():
        old_to_new_column_encoding[key] = counter
        counter += 1
    
    """ json_object = json.dumps(old_to_new_column_encoding, indent=4)
    with open(column_to_convert_to_categorical.name + "_old_to_new_column_encoding" + str(int(time.time())) + ".json", "w") as outfile:
        outfile.write(json_object) """

    categorical_column = pd.DataFrame()
    for i in range(len(column_to_convert_to_categorical)):
        # Inital new column assignment shown below was causing pandas to throw a warning concerning the possibility I am replacing only a slice of the original column https://pandas.pydata.org/docs/reference/api/pandas.errors.SettingWithCopyWarning.html
        #column_to_convert_to_categorical[i] = old_to_new_column_encoding[column_to_convert_to_categorical[i]] 
        categorical_column.at[i, column_to_convert_to_categorical.name + "_category"] = old_to_new_column_encoding[column_to_convert_to_categorical[i]]

    return categorical_column

# https://en.wikipedia.org/wiki/Feature_scaling#Mean_normalization
def mean_normalization(pandas_df_for_scaling):
    min_values_in_each_column = pandas_df_for_scaling.min()
    max_values_in_each_column = pandas_df_for_scaling.max()

    merged_min_max_values = pd.concat([min_values_in_each_column, max_values_in_each_column], axis=1)
    merged_min_max_values = merged_min_max_values.T # Why the transposition? Because we want to align the structure of this matrix to pandas_df_for_scaling

    mean_normalized_values = pd.DataFrame()

    for column in pandas_df_for_scaling:
        mean_normalized_values[column] = (pandas_df_for_scaling[column]-pandas_df_for_scaling[column].mean(axis=0))/(merged_min_max_values[column][1] - merged_min_max_values[column][0])

    return mean_normalized_values

def min_max_scaling(pandas_df_for_scaling):
    min_values_in_each_column = pandas_df_for_scaling.min()
    max_values_in_each_column = pandas_df_for_scaling.max()
    print("min : \n{}\nmax : \n{}\n".format(min_values_in_each_column, max_values_in_each_column))

    merged_df = pd.concat([min_values_in_each_column, max_values_in_each_column], axis=1)
    merged_df = merged_df.T

    scaled_df = pd.DataFrame()
    for column in pandas_df_for_scaling:
        scaled_df[column] = (pandas_df_for_scaling[column]-merged_df[column][0])/(merged_df[column][1] - merged_df[column][0])

    """ 
    Another, more manual, way of applying min-max scaling
    scaled_df = pd.DataFrame()
    for column in pandas_df_for_scaling:
        scaled_column = []
        current_column = pandas_df_for_scaling[column].values.tolist()
        for element in current_column:
            scaled_column.append((element-merged_df[column][0])/(merged_df[column][1] - merged_df[column][0]))
        scaled_df[column] = scaled_column"""

    return scaled_df

def create_report_file():
    if os.path.isfile("./reportFile.txt"): # if file is found it means it is a leftover from previous rename, therefore rename it using current unix timestamp
        os.rename("reportFile.txt", "reportFile_" + str(time.time()) + ".txt")

    with open("reportFile.txt", "w") as currentReportFile:
        currentReportFile.write("Report file\n")
        currentReportFile.write(str(datetime.datetime.now()) + "\n")

def report_file_write(stringToWrite):
    if not(os.path.isfile("./reportFile.txt")): # if file is not found something has gone wrong
        print("ERROR : reportFile.txt has been moved")

    with open("reportFile.txt", "a") as currentReportFile:
        currentReportFile.write(stringToWrite)

def close_report_file():
    if not(os.path.isfile("./reportFile.txt")): # if file is not found something has gone wrong
        print("ERROR : reportFile.txt has been moved")
        sys.exit()
    report_file_write("\n\n\n----------------------------------")
    os.rename("reportFile.txt", "reportFile_" + str(time.time()) + ".txt")

def trainingset_devset_testset_split(X, y):
    TRAINING_SET_SIZE = 0.8
    CV_SET_SIZE = 0.15#dev set
    TEST_SET_SIZE = 0.05

    inputData = X.iloc[:, 4:] #slicing operation to ignore first 4 columns (x,y,month,day)

    #print("Mean normalized : \n{}\n".format(inputData))
    #inputData.drop(inputData.columns[[0,1,2,3]], axis = 1, inplace=True)    # removing columns for X coordinates, Y coordinates, day and month - for now they seem irrelevant
    #print("Input data final : \n{}\nInput data format : {}\n".format(inputData, len(inputData)))
    report_file_write("\nInput data rows : {}\n".format(len(inputData)))
    report_file_write("\nTRAINING_SET_SIZE : {}\nCV_SET_SIZE : {}\nTEST_SET_SIZE : {}\n".format(TRAINING_SET_SIZE, CV_SET_SIZE, TEST_SET_SIZE))

    #shuffled_inputData = inputData.sample(frac=1).reset_index(drop=True) # a clever way to shuffle a Pandas df. All credits to : https://stackoverflow.com/a/34879805
    training_set = inputData[0:math.floor(TRAINING_SET_SIZE*len(inputData))]
    training_set_targets = y[0:math.floor(TRAINING_SET_SIZE*len(inputData))]
    dev_set = inputData[math.floor(TRAINING_SET_SIZE*len(inputData)):len(training_set) + math.floor(CV_SET_SIZE*len(inputData))]
    dev_set_targets = y[math.floor(TRAINING_SET_SIZE*len(inputData)):len(training_set) + math.floor(CV_SET_SIZE*len(inputData))]
    test_set = inputData[len(training_set) + math.floor(CV_SET_SIZE*len(inputData)):len(inputData) - 1]
    test_set_targets = y[len(training_set) + math.floor(CV_SET_SIZE*len(inputData)):len(inputData) - 1]

    return training_set, training_set_targets, dev_set, dev_set_targets, test_set, test_set_targets #ugly and wasteful return

# ---------------------------------------------------------------------------------------------------------

create_report_file()
forest_fires = fetch_ucirepo(id=162)
X = forest_fires.data.features
y = forest_fires.data.targets

print("X : \n{}\n\ny : \n{}\n\n".format(X, y))
#report_file_write("Month column : \n{}\n".format(X['month']))
#report_file_write("Day colum : \n{}\n".format(X['day']))

# Converting month and day columns to categorical values, ie. 0-6 for days and 0-11 for months
month_categorical = column_to_categorical(X['month'])
day_categorical = column_to_categorical(X['day'])
X['month'] = month_categorical
X['day'] = day_categorical

log1p_X = np.log1p(X)
log1p_y = np.log1p(y)
mean_normalized_X = mean_normalization(X)
mean_normalized_y = mean_normalization(y)
mean_normalized_log1p_X = np.log1p(mean_normalized_X)
mean_normalized_log1p_y = np.log1p(mean_normalized_y)
min_max_scaled_X = min_max_scaling(X)
min_max_scaled_y = min_max_scaling(y)
min_max_scaled_log1p_X = np.log1p(min_max_scaled_X)
min_max_scaled_log1p_y = np.log1p(min_max_scaled_y)

training_set, training_set_targets, dev_set, dev_set_targets, test_set, test_set_targets = trainingset_devset_testset_split(min_max_scaled_log1p_X, min_max_scaled_log1p_y)
close_report_file()
""" # inverse min max scaling example
array = np.array([0.58439621, 0.81262134, 0.231262134, 0.191])
#scaled from 100 to 250
minimo = 100
maximo = 250
array * minimo + (maximo - minimo)
"""

NN_model = Sequential([
    Dense(8, activation='relu', kernel_initializer=tf.keras.initializers.HeNormal()),
    Dense(32, activation='relu', kernel_initializer=tf.keras.initializers.HeNormal()),
    Dense(16, activation='relu', kernel_initializer=tf.keras.initializers.HeNormal()),
    Dense(1, activation='linear')
])

NN_model.compile(
    loss=tf.keras.losses.MeanSquaredError(),
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0000015)
)

fit_result = NN_model.fit(training_set, training_set_targets, epochs=5000, validation_data=(dev_set, dev_set_targets))
training_loss = fit_result.history['loss']
validation_loss = fit_result.history['val_loss']
plt.plot(training_loss, label="Training loss", color='b')
plt.plot(validation_loss, label="Validation loss", color='r')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.yticks(np.arange(0, 0.01, 0.001))
plt.ylim(0, 0.015)
plt.grid()
plt.legend()
plt.show()
