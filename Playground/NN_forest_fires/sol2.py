from ucimlrepo import fetch_ucirepo, list_available_datasets
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import datetime
import time
import math
import sys

import sklearn
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.preprocessing import MinMaxScaler

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
    print("\n\nmin : \n{}\nmax : \n{}\n".format(min_values_in_each_column, max_values_in_each_column))

    merged_df = pd.concat([min_values_in_each_column, max_values_in_each_column], axis=1)
    merged_df = merged_df.T

    scaled_df = pd.DataFrame()
    for column in pandas_df_for_scaling:
        if merged_df[column][1] == 0 and merged_df[column][0] == 0:
            scaled_df[column] = 0
        else:
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
    TRAINING_SET_SIZE = 0.75
    CV_SET_SIZE = 0.10
    TEST_SET_SIZE = 0.15

    inputData = X.iloc[:, 4:] #slicing operation to ignore first 4 columns (x,y,month,day)
    inputData = inputData.iloc[:, :-1]  #dropping last column (rain) due to its high zero count

    training_set = inputData[0:math.floor(TRAINING_SET_SIZE*len(inputData))]
    training_set_targets = y[0:math.floor(TRAINING_SET_SIZE*len(inputData))]
    dev_set = inputData[math.floor(TRAINING_SET_SIZE*len(inputData)):len(training_set) + math.floor(CV_SET_SIZE*len(inputData))]
    dev_set_targets = y[math.floor(TRAINING_SET_SIZE*len(inputData)):len(training_set) + math.floor(CV_SET_SIZE*len(inputData))]
    test_set = inputData[len(training_set) + math.floor(CV_SET_SIZE*len(inputData)):len(inputData) - 1]
    test_set_targets = y[len(training_set) + math.floor(CV_SET_SIZE*len(inputData)):len(inputData) - 1]

    return training_set, training_set_targets, dev_set, dev_set_targets, test_set, test_set_targets #ugly and wasteful return

def dataset_shuffle_create_subsets_and_save():
    forest_fires = fetch_ucirepo(id=162)
    X = forest_fires.data.features
    y = forest_fires.data.targets

    print("X : \n{}\n\ny : \n{}\n\n".format(X, y))

    # Converting month and day columns to categorical values, ie. 0-6 for days and 0-11 for months
    month_categorical = column_to_categorical(X['month'])
    day_categorical = column_to_categorical(X['day'])
    X['month'] = month_categorical
    X['day'] = day_categorical

    combined = pd.concat([X, y], axis=1)
    combined_shuffled = combined.sample(frac=1, random_state=42).reset_index(drop=True)
    X_shuffled = combined_shuffled[X.columns]
    y_shuffled = combined_shuffled[y.name] if isinstance(y, pd.Series) else combined_shuffled[y.columns]

    training_set, training_set_targets, dev_set, dev_set_targets, test_set, test_set_targets = trainingset_devset_testset_split(X_shuffled, y_shuffled)
    training_set.to_csv('training_set.csv', index=False)
    training_set_targets.to_csv('training_set_targets.csv', index=False)
    dev_set.to_csv('dev_set.csv', index=False)
    dev_set_targets.to_csv('dev_set_targets.csv', index=False)
    test_set.to_csv('test_set.csv', index=False)
    test_set_targets.to_csv('test_set_targets.csv', index=False)

# ---------------------------------------------------------------------------------------------------------

#dataset_shuffle_create_subsets_and_save()
training_set = pd.read_csv('./training_set.csv')
training_set_targets = pd.read_csv('./training_set_targets.csv')
dev_set = pd.read_csv('./dev_set.csv')
dev_set_targets = pd.read_csv('./dev_set_targets.csv')
test_set = pd.read_csv('./test_set.csv')
test_set_targets = pd.read_csv('./test_set_targets.csv')

print("training_set : \n{}\ntraining_set_targets : \n{}\n".format(training_set, training_set_targets))

# training_set and training_set_targets
""" scaler_training = MinMaxScaler()
scaler_training.fit(training_set)
min_max_training_set = scaler_training.transform(training_set)
min_max_log1p_training_set = np.log1p(min_max_training_set) #training_set fully transformed

scaler_training_targets = MinMaxScaler()
scaler_training_targets.fit(training_set_targets)
min_max_training_targets = scaler_training_targets.transform(training_set_targets)
min_max_log1p_training_targets = np.log1p(min_max_training_targets) #training targets fully transformed

# dev_set and dev_set_targets
scaler_dev = MinMaxScaler()
scaler_dev.fit(dev_set)
min_max_dev = scaler_dev.transform(dev_set)
min_max_log1p_dev = np.log1p(min_max_dev)   #dev_set fully transformed

scaler_dev_targets = MinMaxScaler()
scaler_dev_targets.fit(dev_set_targets)
min_max_dev_targets = scaler_dev_targets.transform(dev_set_targets)
min_max_log1p_dev_targets = np.log1p(min_max_dev_targets)   #dev set targets fully transformed

# test_set and test_set_targets
scaler_test = MinMaxScaler()
scaler_test.fit(test_set)
min_max_scaler = scaler_test.transform(test_set)
min_max_log1p_test = np.log1p(min_max_scaler)   # test set fully transformed

scaler_test_targets = MinMaxScaler()
scaler_test_targets.fit(test_set_targets)
min_max_test_targets = scaler_test_targets.transform(test_set_targets)
min_max_log1p_test_targets = np.log1p(min_max_test_targets) #test set targets fully transformed """

log1p_training_set = np.log1p(training_set)
log1p_training_set_targets = np.log1p(training_set_targets)
log1p_dev_set = np.log1p(dev_set)
log1p_dev_set_targets = np.log1p(dev_set_targets)
log1p_test_set = np.log1p(test_set)
log1p_test_set_targets = np.log1p(test_set_targets)


"""
min_max_training_set = min_max_scaling(training_set)
min_max_log1p_training_set = np.log1p(min_max_training_set)
min_max_training_set_targets = min_max_scaling(training_set_targets)
min_max_log1p_training_set_targets = np.log1p(min_max_training_set_targets)

min_max_dev_set = min_max_scaling(dev_set)
min_max_log1p_dev_set = np.log1p(min_max_dev_set)
min_max_dev_set_targets = min_max_scaling(dev_set_targets)
min_max_log1p_dev_set_targets = np.log1p(min_max_dev_set_targets)

min_max_test_set = min_max_scaling(test_set)
min_max_log1p_test_set = np.log1p(min_max_test_set)
scaler_test_set_targets = MinMaxScaler()
scaler_test_set_targets.fit(test_set_targets)
min_max_test_set_targets = scaler_test_set_targets.transform(test_set_targets)
min_max_log1p_test_set_targets = np.log1p(min_max_test_set_targets) """


""" # inverse min max scaling example
array = np.array([0.58439621, 0.81262134, 0.231262134, 0.191])
#scaled from 100 to 250
minimo = 100
maximo = 250
array * minimo + (maximo - minimo)
"""

plateau_change_LR = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.65,
    patience=40,
    cooldown=20,
    min_lr = 0.00000003
)

early_stopping_callback = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    mode='min',
    restore_best_weights=True,
    min_delta=0.0002,
    patience=500
)


NN_model = Sequential([
    Dense(7, activation='leaky_relu', kernel_initializer=tf.keras.initializers.HeNormal()),   # 7 input neurons because 'rain' column has been dropped due to high zero count
    Dense(64, activation='leaky_relu', kernel_initializer=tf.keras.initializers.HeNormal()),
    tf.keras.layers.Dropout(0.2),
    Dense(16, activation='leaky_relu', kernel_initializer=tf.keras.initializers.HeNormal()),
    tf.keras.layers.Dropout(0.2),
    Dense(1, activation='linear')
])

NN_model.compile(
    loss=tf.keras.losses.MeanAbsoluteError(),
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001)
)

#min_max_log1p_training_set, min_max_log1p_training_targets, min_max_log1p_dev, min_max_log1p_dev_targets, min_max_log1p_test, min_max_log1p_test_targets
fit_result = NN_model.fit(log1p_training_set, log1p_training_set_targets, epochs=500, validation_data=(log1p_dev_set, log1p_dev_set_targets))
NN_model.save('lastModel.keras')
training_loss = fit_result.history['loss']
validation_loss = fit_result.history['val_loss']
plt.plot(training_loss, label="Training loss", color='b')
plt.plot(validation_loss, label="Validation loss", color='r')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.yticks(np.arange(0, 0.075, 0.0025))
plt.ylim(0, 0.075)
plt.grid()
plt.legend()
plt.show()


predictionResults = NN_model.predict(log1p_test_set)
expm1_predictionResults = np.expm1(predictionResults)
recovered_predictionResults = expm1_predictionResults
#recovered_predictionResults = scaler_test_targets.inverse_transform(expm1_predictionResults)
print("test_set_targets : \n{}\npredictions : \n{}\n".format(test_set_targets, recovered_predictionResults))
plt.plot([i for i in range(len(recovered_predictionResults))], recovered_predictionResults, label='Prediction results', color='r', marker='.', linestyle='none')
plt.plot([i for i in range((len(test_set_targets)))], test_set_targets, label='Y label', color='b', marker='.', linestyle='none')
plt.xlabel('Individual datapoint')
plt.ylabel('Burned area')
plt.legend()
plt.show()
print("\nMSE : {}\nR_squared : {}\n".format( sklearn.metrics.mean_squared_error(recovered_predictionResults, test_set_targets), sklearn.metrics.r2_score(recovered_predictionResults, test_set_targets) ))

""" predictionResults = NN_model.predict(min_max_test_set)
expm1_predictionResults = np.expm1(predictionResults)
recoveredPredictionResults = scaler_test_set_targets.inverse_transform(expm1_predictionResults)
print("\nMSE : {}\nR_squared : {}\n".format( sklearn.metrics.mean_squared_error(recoveredPredictionResults, test_set_targets), sklearn.metrics.r2_score(recoveredPredictionResults, test_set_targets) ))
plt.plot([i for i in range(len(recoveredPredictionResults))], recoveredPredictionResults, label='Predictions', color='r', linestyle='none')
plt.plot([i for i in range(len(test_set_targets))], test_set_targets, label='Y target (ground truth)', color='g', linestyle='none')
plt.xlabel('Individual datapoint')
plt.ylabel('Burned area')
plt.legend()
plt.show() """