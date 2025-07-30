from ucimlrepo import fetch_ucirepo, list_available_datasets
import matplotlib
import matplotlib.pyplot as plt
import json
import time
import pandas as pd
import math
import numpy as np
import random
import json
import sys
import configparser
import os
import datetime
import time
import statistics

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout


seed_value = 11

################### Dataset cleanup ##########################

'''
Name : column_to_categorical()

Parameters : column_to_convert_to_categorical - this is the column we want to convert to a categorical variable

Functionality? : intended to convert columns whose values are not numerical, but are instead alphanumeric
    e.g. months are listed as jan, feb, mar, ...
        This function converts months from their ascii representation to a numerical representation

How? : it does this by counting the frequencies of each unique value in the original column,
    which is done using pandas function values_counts()
    This provides us with the information how many different unique values there are.
    Then we take the keys from the value_counts() output, and assign values, starting from 0.

Return value : returns a column which now contains numerical representations of the values found in original column
    Also creates a JSON file which shows the mapping between the original values and the new values.
    This json file is named like this : ORIGINAL_COLUMN_NAME_old_to_new_column_encoding_UNIX_TIMESTAMP.json
                        e.g. : month_old_to_new_column_encoding_120492928.json
'''
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

######################## End ###############################


################# Feature scaling and normalization functions #################
'''
More info about feature scaling : 
https://github.com/josipursan/ML/blob/main/Supervised%20Machine%20Learning%3A%20Regression%20and%20Classification/W2/Notes.md
https://en.wikipedia.org/wiki/Feature_scaling
https://www.geeksforgeeks.org/ml-feature-scaling-part-2/

Note that scaling a feature, and normalizing it, are not interchangeable operations.
Scaling means simply shifting the data range to some other data range.
Normalization changes the distribution of the data. (well yes, but actually no, but yes... it helps stabilize the distribution of the data further exposing relationships between datapoints)
'''

# Max val scaling divides all values in column by the max value found in that column, thus scaling to [0,1] range
def maximum_value_scaling(pandas_df_for_scaling):
    max_values_in_each_column = pandas_df_for_scaling.max()

    for key, value in max_values_in_each_column.items():
        pandas_df_for_scaling[key] /= value
    
    return pandas_df_for_scaling

# https://www.geeksforgeeks.org/ml-feature-scaling-part-2/
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
########################## End ##############################

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

def generate_colors(NBestModels):
    colors = []
    for i in range(NBestModels):
        current_color = []
        for j in range(3):
            current_color.append(random.uniform(0.01, 0.99))
        colors.append(current_color)
    return colors

def get_dataset_preprocess_dataset():
    forest_fires = fetch_ucirepo(id=162)
    X = forest_fires.data.features
    y = forest_fires.data.targets
    y = np.log1p(y['area']) # aaaaaaaa... never cut the branch you are sitting on - revisit this once NN works ok

    #print(forest_fires.metadata)
    # variable information
    print(forest_fires.variables)
    print("\nFeatures dataframe length : {}\nTarget dataframe length : {}\n\n".format(len(X), len(y)))

    cleaned_up_X = X.copy()

    categorized_month_column = column_to_categorical(cleaned_up_X['month'])
    categorized_day_column = column_to_categorical(cleaned_up_X['day'])
    cleaned_up_X['month'] = categorized_month_column
    cleaned_up_X['day'] = categorized_day_column
    #maximum_value_scaling(cleaned_up_X)
    #min_max_scaling(cleaned_up_X)
    inputData = mean_normalization(cleaned_up_X.iloc[:, 4:]) #ignoring the first 4 columns with this slicing operation : cleaned_up_X.iloc[:, 4:])

    return inputData, y

def trainingset_devset_testset_split():
    inputData, y = get_dataset_preprocess_dataset()

    #print("Mean normalized : \n{}\n".format(inputData))
    #inputData.drop(inputData.columns[[0,1,2,3]], axis = 1, inplace=True)    # removing columns for X coordinates, Y coordinates, day and month - for now they seem irrelevant
    print("Input data final : \n{}\nInput data format : {}\n".format(inputData, len(inputData)))

    TRAINING_SET_SIZE = 0.80
    TEST_SET_SIZE = 0
    CV_SET_SIZE = 0.20
    shuffled_inputData = inputData.sample(frac=1).reset_index(drop=True) # a clever way to shuffle a Pandas df. All credits to : https://stackoverflow.com/a/34879805
    training_set = shuffled_inputData[0:math.floor(TRAINING_SET_SIZE*len(shuffled_inputData))]
    training_set_targets = y[0:math.floor(TRAINING_SET_SIZE*len(shuffled_inputData))]
    dev_set = shuffled_inputData[math.floor(TRAINING_SET_SIZE*len(shuffled_inputData)):len(training_set) + math.floor(CV_SET_SIZE*len(shuffled_inputData))]
    dev_set_targets = y[math.floor(TRAINING_SET_SIZE*len(shuffled_inputData)):len(training_set) + math.floor(CV_SET_SIZE*len(shuffled_inputData))]
    test_set = shuffled_inputData[len(training_set) + math.floor(CV_SET_SIZE*len(shuffled_inputData)):len(shuffled_inputData) - 1]
    test_set_targets = y[len(training_set) + math.floor(CV_SET_SIZE*len(shuffled_inputData)):len(shuffled_inputData) - 1]

    return training_set, training_set_targets, dev_set, dev_set_targets, test_set, test_set_targets

'''
Three additional approaches to splitting the dataset into train, dev and train sets

1. TRAINING_SET_SIZE, TEST_SET_SIZE, and CV_SET_SIZE are variables defining what percentage of starting dataset will be dedicated to each subset (TRAINING, TEST, CV).
    training_set_indices, test_set_indices, and cv_set_indices are lists defining, based on TRAINING_SET_SIZE, TEST_SET_SIZE, and CV_SET_SIZE, what indices from the subsets will be used.
    Note that this approach WILL result in a random number of indices being shared by the subsets.

    training_set_indices = np.random.choice(inputData.shape[0], size=math.floor(TRAINING_SET_SIZE*inputData.shape[0]), replace=False)
    test_set_indices = np.random.choice(inputData.shape[0], size=math.floor(TEST_SET_SIZE*inputData.shape[0]), replace=False)
    cv_set_indices = np.random.choice(inputData.shape[0], size=math.floor(CV_SET_SIZE*inputData.shape[0]), replace=False)

    training_set = inputData.iloc[training_set_indices]
    y_for_training_set = y.iloc[training_set_indices]
    test_set = inputData.iloc[test_set_indices]
    y_for_test_set = y.iloc[test_set_indices]
    cv_set = inputData.iloc[cv_set_indices]
    y_for_cv_Set = y.iloc[cv_set_indices]

- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

2. Second approach defines starting and ending indices for each subset based on the wanted sizes of subsets.
    Going out of bounds can be problematic as proper precautions do not exist, besides the hardocded -2 xD

    print("Dataset dimensions : {}\n{}\t{}\n{}\t{}\n{}\t{}\n".format(inputData.shape, "0", math.floor(TRAINING_SET_SIZE*inputData.shape[0]), 
                                            math.floor(TRAINING_SET_SIZE*inputData.shape[0]) + 1, 
                                            math.floor(TRAINING_SET_SIZE*inputData.shape[0]) + 1 + math.floor(TEST_SET_SIZE*inputData.shape[0]), 
                                            math.floor(TRAINING_SET_SIZE*inputData.shape[0]) + 1 + math.floor(TEST_SET_SIZE*inputData.shape[0]) + 1, 
                                            math.floor(TRAINING_SET_SIZE*inputData.shape[0]) + 1 + math.floor(TEST_SET_SIZE*inputData.shape[0]) + 1 + math.floor(CV_SET_SIZE*inputData.shape[0]) - 2))

- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

3. Just use the built in stuff, don't reinvent the wheel.
    https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.sample.html

    Function "sample", based on parameter frac, that we pass to it, returns a fraction of the initial dataframe, after it has been shuffled.

    training_set = inputData.sample(frac=TRAINING_SET_SIZE)
    test_set = inputData.sample(frac=TEST_SET_SIZE)
    cv_set = inputData.sample(frac=CV_SET_SIZE)
    print("training_set shape : {}\ntest_set shape : {}\ncv_set shape : {}\n".format(training_set.shape, test_set.shape, cv_set.shape))
'''


class CustomThresholdCallback(tf.keras.callbacks.Callback):
    def __init__(self, threshold):
        super().__init__()
        self.threshold = threshold

    def on_epoch_end(self, epoch, logs=None):
        if logs.get('loss') is not None and logs.get('loss') < self.threshold:
            print("Stopping training! Loss below threshold {}\n".format(self.threshold))
            self.model.stop_training = True

#tf.random.set_seed(seed_value)


lr_callback_plateau = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor = 0.93,
    min_delta = 10,
    cooldown = 10,
    min_lr = 0.00033
)

l2_regularizer = tf.keras.regularizers.L2(l2=0.015)

def generate_random_network_architectures():
    allGeneratedArchitectures = []

    generatorConfig = configparser.ConfigParser()
    generatorConfig.read('configOptions_randomArchitectures.ini')
    #print(generatorConfig['DEFAULT']['DifferentArchitectures'])
    #print("All keys : {}\n".format(dict(generatorConfig.items('DEFAULT'))))

    DifferentArchitectures = int(generatorConfig['DEFAULT']['DifferentArchitectures'])
    MinHiddenLayers = int(generatorConfig['DEFAULT']['MinHiddenLayers'])
    MaxHiddenLayers = int(generatorConfig['DEFAULT']['MaxHiddenLayers'])
    MinNeurons = int(generatorConfig['DEFAULT']['MinNeurons'])
    MaxNeurons = int(generatorConfig['DEFAULT']['MaxNeurons'])
    MinAlpha = float(generatorConfig['DEFAULT']['MinAlpha'])
    MaxAlpha = float(generatorConfig['DEFAULT']['MaxAlpha'])
    InputLayerNeurons = int(generatorConfig['DEFAULT']['InputLayerNeurons'])
    OutputLayerNeurons = int(generatorConfig['DEFAULT']['OutputLayerNeurons'])
    OutputLayerActivationFunction = generatorConfig['DEFAULT']['OutputLayerActivationFunction']
    ActivationFunction = generatorConfig['DEFAULT']['ActivationFunction']
    Epochs = int(generatorConfig['DEFAULT']['Epochs'])

    for current_architecture in range(DifferentArchitectures):
        curr_arch = {}

        curr_arch['architecture_number'] = current_architecture
        curr_arch['total_layers'] = random.randint(MinHiddenLayers, MaxHiddenLayers) + 2  #generating random number of layers in range MinHiddenLayers,max_hidden layers ; INCREMENT BY TWO BECAUSE HIDDEN LAYERS ARE SANDWICHED BETWEEN INPUT AND OUTPUT LAYER
        
        neuron_structure = []
        for layer in range(int(curr_arch['total_layers'])):
            neuron_structure.append(random.randint(MinNeurons, MaxNeurons))
        neuron_structure[0] = InputLayerNeurons   # set number of neurons for input and output layers - these are defined by the number of inputs and the desired output from the NN - we do not want random values here
        neuron_structure[-1] = OutputLayerNeurons
        curr_arch['neuron_structure'] = neuron_structure

        curr_arch['alpha'] = random.uniform(MinAlpha, MaxAlpha)
        curr_arch['ActivationFunction'] = ActivationFunction
        curr_arch['OutputLayerActivationFunction'] = OutputLayerActivationFunction
        curr_arch['epochs'] = Epochs

        allGeneratedArchitectures.append(curr_arch)

    with open("generated_architectures.json", "a") as outfile:
        json.dump(allGeneratedArchitectures, outfile, indent=2)

def train_randomly_generated_network_architectures():
    create_report_file()

    training_set, training_set_targets, dev_set, dev_set_targets, test_set, test_set_targets = trainingset_devset_testset_split()
    generate_random_network_architectures()
    
    all_model_losses = []
    all_model_validation_losses = []

    generatorConfig = configparser.ConfigParser()
    generatorConfig.read('configOptions_randomArchitectures.ini')
    NBestModels = int(generatorConfig['DEFAULT']['NBestModels'])  #get the N best performing models (in terms of loss) for plotting
    AverageWindow_training_vs_validation = int(generatorConfig['DEFAULT']['AverageWindow_training_vs_validation'])  # value determining size of the windows used to compute average for training loss and validation loss in order to provide a smoother approximation of the values for the later computation of their difference, which is used to determin N best performing models

    with open("generated_architectures.json", "r") as architecturesFile:
        network_architectures = json.load(architecturesFile)

    for current_model in range(len(network_architectures)): # len(network_architectures) == DifferentArchitectures
        NN_model_test = Sequential()
        for current_layer in range(network_architectures[current_model]['total_layers']):
            if current_layer == network_architectures[current_model]['total_layers'] - 1:   # we have reached iteration where output layer is defined
                NN_model_test.add(Dense(network_architectures[current_model]['neuron_structure'][current_layer], network_architectures[current_model]['OutputLayerActivationFunction']))
            else:
                NN_model_test.add(Dense(network_architectures[current_model]['neuron_structure'][current_layer], network_architectures[current_model]['ActivationFunction']))
        NN_model_test.compile(
            loss = tf.keras.losses.MeanSquaredError(),
            optimizer=tf.keras.optimizers.Adam(learning_rate=network_architectures[current_model]['alpha'])
        )
        
        model_history = NN_model_test.fit(training_set, training_set_targets, epochs = network_architectures[current_model]['epochs'], validation_data=(dev_set, dev_set_targets))
        all_model_losses.append(model_history.history['loss'])
        all_model_validation_losses.append(model_history.history['val_loss'])


    '''
    model_diffs_validation_training_loss - a list holding differences between the last value of validation_loss and last value of training_loss for each model
        -is used to determine NBestModels (ie. the models with smallest difference between the last value of validation_loss and last value of training_loss)

    n_best_models_indices - a list made up of NBestModels (e.g. 5) indices, ie. the indices representing models with the smallest diff between validaton and training losses

    model_diffs_average_last_X_values - a list holding differences between the average (last X values) of validation_loss and average (last X values) value of training_loss for each model
    '''
    #model_diffs_validation_training_loss = []
    model_diffs_average_last_X_values = []
    for current_model in range(len(network_architectures)):
        #model_diffs_validation_training_loss.append(abs(all_model_losses[current_model][-1] - all_model_validation_losses[current_model][-1]))
        model_diffs_average_last_X_values.append(abs(statistics.mean(all_model_losses[current_model][-AverageWindow_training_vs_validation:]) - statistics.mean(all_model_validation_losses[current_model][-AverageWindow_training_vs_validation:])))
    #print("model_diffs_validation_training_loss : {}\n".format(model_diffs_validation_training_loss))
    report_file_write("(Average last 20 values training loss) - (Average last 20 values validation loss) : \n{}\n".format(model_diffs_average_last_X_values))
    #print("model_diffs_average_last_X_values : {}\n".format(model_diffs_average_last_X_values))

    n_best_models_indices = []
    for n in range(NBestModels):
        n_best_models_indices.append(model_diffs_average_last_X_values.index(min(model_diffs_average_last_X_values)))
        report_file_write("\nModel {} last {} values average training loss - last {} values average validation loss : {}\n".format(n_best_models_indices[-1], AverageWindow_training_vs_validation, AverageWindow_training_vs_validation, min(model_diffs_average_last_X_values)))
        #Now set the latest found minimum value in model_diffs_validation_training_loss to some outrageously big value so that we can find the next min value in the next iteration
        model_diffs_average_last_X_values[n_best_models_indices[-1]] = 1000

    #print("Best performing models : {}\n".format(n_best_models_indices))
    report_file_write("\nBest performing models : \n{}\n".format(n_best_models_indices))
    for model in n_best_models_indices:
        #print("Model {} : {}\n".format(model, network_architectures[model]))
        report_file_write("\nModel {} : \n{}".format(model, network_architectures[model]))

    close_report_file()

    colors = generate_colors(NBestModels)
    markers = [".", "v", "^", "<", ">", "1", "2", "3", "4", "8", "s", "p", "*", "h", "d"]   # 15 different markers - this means 15 best models is maximum we can look to find, which should be enough
    for current_model in n_best_models_indices:
        plt.plot(all_model_losses[current_model], label="Training loss model_"+str(current_model), color=colors[n_best_models_indices.index(current_model)], linestyle='none', marker = markers[n_best_models_indices.index(current_model)])
        plt.plot(all_model_validation_losses[current_model], label="Validation loss model_"+str(current_model), color=colors[n_best_models_indices.index(current_model)], linestyle='none', marker = markers[n_best_models_indices.index(current_model)])
    plt.title("Training vs validation loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.yticks(range(1,4))
    plt.legend()
    plt.show()

train_randomly_generated_network_architectures()