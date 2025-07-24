from ucimlrepo import fetch_ucirepo, list_available_datasets
import matplotlib.pyplot as plt
import json
import time
import pandas as pd
import math
import numpy as np
import random
import json
import sys

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
Normalization changes the distribution of the data.
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



#list_available_datasets()
# fetch dataset 
forest_fires = fetch_ucirepo(id=162) 
  
# data (as pandas dataframes) 
X = forest_fires.data.features 
y = forest_fires.data.targets
y = np.log1p(y['area']) # aaaaaaaa... never cut the branch you are sitting on - revisit this once NN works ok


  
# metadata 
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
#print("Mean normalized : \n{}\n".format(inputData))
#inputData.drop(inputData.columns[[0,1,2,3]], axis = 1, inplace=True)    # removing columns for X coordinates, Y coordinates, day and month - for now they seem irrelevant
print("Input data final : \n{}\nInput data format : {}\n".format(inputData, len(inputData)))

TRAINING_SET_SIZE = 0.80
TEST_SET_SIZE = 0
CV_SET_SIZE = 0.20
'''
What does the block of code below do?

TRAINING_SET_SIZE, TEST_SET_SIZE, and CV_SET_SIZE are variables defining what percentage of starting dataset will be dedicated to each subset (TRAINING, TEST, CV).

training_set_indices, test_set_indices, and cv_set_indices are lists defining, based on TRAINING_SET_SIZE, TEST_SET_SIZE, and CV_SET_SIZE, what indices from the subsets will be used.
Note that this approach WILL result in a random number of indices being shared by the subsets.

A different approach is shown below this code block. '''

""" training_set_indices = np.random.choice(inputData.shape[0], size=math.floor(TRAINING_SET_SIZE*inputData.shape[0]), replace=False)
test_set_indices = np.random.choice(inputData.shape[0], size=math.floor(TEST_SET_SIZE*inputData.shape[0]), replace=False)
cv_set_indices = np.random.choice(inputData.shape[0], size=math.floor(CV_SET_SIZE*inputData.shape[0]), replace=False)

training_set = inputData.iloc[training_set_indices]
y_for_training_set = y.iloc[training_set_indices]
test_set = inputData.iloc[test_set_indices]
y_for_test_set = y.iloc[test_set_indices]
cv_set = inputData.iloc[cv_set_indices]
y_for_cv_Set = y.iloc[cv_set_indices]
 """#print("TRAINING_SET_SIZE : {}\nTEST_SET_SIZE : {}\nCV_SET_SIZE : {}\ntraining_set_indices : {}\ntest_set_indices : {}\ncv_set_indices : {}\n".format(TRAINING_SET_SIZE, TEST_SET_SIZE, CV_SET_SIZE, np.sort(training_set_indices), np.sort(test_set_indices), np.sort(cv_set_indices)))


'''
This is a different approach to creating subsets from the initial dataset - here we define starting and ending indices for each subset based on the wanted sizes of subsets.
Going out of bounds can be problematic as proper precautions do not exist, besides the hardocded -2 xD

print("Dataset dimensions : {}\n{}\t{}\n{}\t{}\n{}\t{}\n".format(inputData.shape, "0", math.floor(TRAINING_SET_SIZE*inputData.shape[0]), 
                                            math.floor(TRAINING_SET_SIZE*inputData.shape[0]) + 1, 
                                            math.floor(TRAINING_SET_SIZE*inputData.shape[0]) + 1 + math.floor(TEST_SET_SIZE*inputData.shape[0]), 
                                            math.floor(TRAINING_SET_SIZE*inputData.shape[0]) + 1 + math.floor(TEST_SET_SIZE*inputData.shape[0]) + 1, 
                                            math.floor(TRAINING_SET_SIZE*inputData.shape[0]) + 1 + math.floor(TEST_SET_SIZE*inputData.shape[0]) + 1 + math.floor(CV_SET_SIZE*inputData.shape[0]) - 2))
'''

'''
Another, even simpler way, is shown below.
Here we use an integrated pandas functionality, called "sample" : https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.sample.html

Function "sample", based on parameter frac, that we pass to it, returns a fraction of the initial dataframe, after it has been shuffled.

training_set = inputData.sample(frac=TRAINING_SET_SIZE)
test_set = inputData.sample(frac=TEST_SET_SIZE)
cv_set = inputData.sample(frac=CV_SET_SIZE)
print("training_set shape : {}\ntest_set shape : {}\ncv_set shape : {}\n".format(training_set.shape, test_set.shape, cv_set.shape))
'''

training_set = inputData[0:math.floor(TRAINING_SET_SIZE*len(inputData))]
training_set_targets = y[0:math.floor(TRAINING_SET_SIZE*len(inputData))]
dev_set = inputData[math.floor(TRAINING_SET_SIZE*len(inputData)):len(training_set) + math.floor(CV_SET_SIZE*len(inputData))]
dev_set_targets = y[math.floor(TRAINING_SET_SIZE*len(inputData)):len(training_set) + math.floor(CV_SET_SIZE*len(inputData))]
test_set = inputData[len(training_set) + math.floor(CV_SET_SIZE*len(inputData)):len(inputData) - 1]
test_set_targets = y[len(training_set) + math.floor(CV_SET_SIZE*len(inputData)):len(inputData) - 1]
#print("Training set ending index : {}".format(math.floor(TRAINING_SET_SIZE*len(inputData))))
#print("Dev set start/end index : {}/{}".format(math.floor(TRAINING_SET_SIZE*len(inputData)), len(training_set) + math.floor(CV_SET_SIZE*len(inputData))))
#print("Test set start/end index : {}/{}".format(len(training_set) + math.floor(CV_SET_SIZE*len(inputData)), len(inputData) - 1))
#print("train_targets : {}\ndev_targets : {}\ntest_targets : {}\n".format(training_set_targets, dev_set_targets, test_set_targets))

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

#### A simple grid search implementation
different_architectures = 20 # how many different NN architecture we want to create
min_hidden_layers = 1
max_hidden_layers = 6
activation_function = 'relu'    #use same names used by TF : https://www.tensorflow.org/api_docs/python/tf/keras/activations
output_layer_activation_function = 'linear'
input_layer_neurons = 8     #how many neurons the input layer should have
output_layer_neurons = 1    #how many neurons the output layer should have
min_neurons = 2
max_neurons = 64
min_alpha = 0.0003
max_alpha = 0.0008
epo_chs = 250
all_training_losses = []
all_validation_losses = []

def generate_random_network_architectures(different_architectures, activation_function, output_layer_activation_function, input_layer_neurons, output_layer_neurons, min_hidden_layers, max_hidden_layers, min_neurons, max_neurons, min_alpha, max_alpha, epo_chs):
    allGeneratedArchitectures = []
    for current_architecture in range(different_architectures):
        curr_arch = {}

        curr_arch['architecture_number'] = current_architecture
        curr_arch['total_layers'] = random.randint(min_hidden_layers, max_hidden_layers) + 2  #generating random number of layers in range min_hidden_layers,max_hidden layers ; INCREMENT BY TWO BECAUSE HIDDEN LAYERS ARE SANDWICHED BETWEEN INPUT AND OUTPUT LAYER
        
        neuron_structure = []
        for layer in range(int(curr_arch['total_layers'])):
            neuron_structure.append(random.randint(min_neurons, max_neurons))
        neuron_structure[0] = input_layer_neurons   # set number of neurons for input and output layers - these are defined by the number of inputs and the desired output from the NN - we do not want random values here
        neuron_structure[-1] = output_layer_neurons
        curr_arch['neuron_structure'] = neuron_structure

        curr_arch['alpha'] = random.uniform(min_alpha, max_alpha)
        curr_arch['activation_function'] = activation_function
        curr_arch['output_layer_activation_function'] = output_layer_activation_function
        curr_arch['epochs'] = epo_chs

        allGeneratedArchitectures.append(curr_arch)

    with open("generated_architectures.json", "a") as outfile:
        json.dump(allGeneratedArchitectures, outfile, indent=2)

generate_random_network_architectures(different_architectures, activation_function, output_layer_activation_function, input_layer_neurons, output_layer_neurons, min_hidden_layers, max_hidden_layers, min_neurons, max_neurons, min_alpha, max_alpha, epo_chs)

with open("generated_architectures.json", "r") as architecturesFile:
    network_architectures = json.load(architecturesFile)


all_model_losses = []
all_model_validation_losses = []

for current_model in range(len(network_architectures)): # len(network_architectures) == different_architectures
    NN_model_test = Sequential()
    for current_layer in range(network_architectures[current_model]['total_layers']):
        if current_layer == network_architectures[current_model]['total_layers'] - 1:   # we have reached iteration where output layer is defined
            NN_model_test.add(Dense(network_architectures[current_model]['neuron_structure'][current_layer], network_architectures[current_model]['output_layer_activation_function']))
        else:
            NN_model_test.add(Dense(network_architectures[current_model]['neuron_structure'][current_layer], network_architectures[current_model]['activation_function']))
    NN_model_test.compile(
        loss = tf.keras.losses.MeanSquaredError(),
        optimizer=tf.keras.optimizers.Adam(learning_rate=network_architectures[current_model]['alpha'])
    )
    
    model_history = NN_model_test.fit(training_set, training_set_targets, epochs = network_architectures[current_model]['epochs'], validation_data=(dev_set, dev_set_targets))
    all_model_losses.append(model_history.history['loss'])
    all_model_validation_losses.append(model_history.history['val_loss'])


color_map = ['orange', 'red', 'green', 'blue', 'olive', 'pink', 'silver', 'tan', 'lime', 'lavender', 'lightcyan', 'black', 'plum', 'gold', 'tomato', 'steelblue', 'cyan', 'gainsboro', 'indigo', 'palegoldenrod']
for current_model in range(len(network_architectures)):
    plt.plot(all_model_losses[current_model], label="Training loss model_"+str(current_model), color=color_map[current_model])
    plt.plot(all_model_validation_losses[current_model], label="Validation loss model_"+str(current_model), color=color_map[current_model], linestyle='dotted')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()






""" NN_model = Sequential(
    [
        Dense(8, activation='relu', kernel_initializer = tf.keras.initializers.HeNormal()),
        Dense(16, activation='relu', kernel_initializer = tf.keras.initializers.HeNormal(), kernel_regularizer=l2_regularizer),
        #Dropout(0.1),
        Dense(32, activation='relu', kernel_initializer = tf.keras.initializers.HeNormal(), kernel_regularizer=l2_regularizer),
        Dense(64, activation='relu', kernel_initializer = tf.keras.initializers.HeNormal(), kernel_regularizer=l2_regularizer),
        Dense(32, activation='relu', kernel_initializer = tf.keras.initializers.HeNormal(), kernel_regularizer=l2_regularizer),
        Dense(16, activation='relu', kernel_initializer = tf.keras.initializers.HeNormal(), kernel_regularizer=l2_regularizer),
        #Dropout(0.15),
        #Dense(32, activation='relu', kernel_initializer = tf.keras.initializers.HeNormal()),
        #Dense(16, activation='relu', kernel_initializer = tf.keras.initializers.HeNormal()),
        #Dense(64, activation='relu', kernel_initializer = tf.keras.initializers.HeNormal(), kernel_regularizer=l2_regularizer),
        #Dropout(0.1),
        Dense(1, activation='linear')     #output layer
    ]
)

NN_model.compile(
    loss = tf.keras.losses.MeanSquaredError(),
    optimizer=tf.keras.optimizers.Adam(learning_rate = 0.00073) # 0.00033; learning_rate = 0.00075 is pretty good with 8,64,48,16,1 network
)



model_history = NN_model.fit(training_set, training_set_targets, epochs = 750, validation_data=(dev_set, dev_set_targets)) #, batch_size = 32
training_loss = model_history.history['loss']
#training_accuracy = model_history.history['acc']
validation_loss = model_history.history['val_loss']
#validation_accuracy = model_history.history['val_acc']
plt.plot(training_loss, label="Training loss", color='b')
#plt.plot(training_accuracy[-1000::], label="Training accuracy -1000", color='b', marker='o')
plt.plot(validation_loss, label="Validation loss", color='r')
#plt.plot(validation_accuracy[-1000::], label="Validation accuracy -1000", color='r', marker='o')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

print("Training loss last 20 epoch values : {}\n".format(training_loss[-20::])) """
#print("model_history.history : {}".format(model_history.history[-100::]))

#training_set_predictions = NN_model(training_set)
#print("Training set predictions : {}\n".format(training_set_predictions))