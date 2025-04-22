from ucimlrepo import fetch_ucirepo, list_available_datasets
import matplotlib.pyplot as plt
import json
import time

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
    
    json_object = json.dumps(old_to_new_column_encoding, indent=4)
    with open(column_to_convert_to_categorical.name + "_old_to_new_column_encoding" + str(int(time.time())) + ".json", "w") as outfile:
        outfile.write(json_object)

    for i in range(len(column_to_convert_to_categorical)):
        column_to_convert_to_categorical[i] = old_to_new_column_encoding[column_to_convert_to_categorical[i]]

    return column_to_convert_to_categorical

######################## End ###############################


################# Feature scaling functions #################
#https://github.com/josipursan/ML/blob/main/Supervised%20Machine%20Learning%3A%20Regression%20and%20Classification/W2/Notes.md

# Max val scaling divides all values in column by the max value found in that column, thus scaling to [0,1] range
def maximum_value_scaling(pandas_df_for_scaling):
    max_values_in_each_column = pandas_df_for_scaling.max()

    for key, value in max_values_in_each_column.items():
        pandas_df_for_scaling[key] /= value
    
    return pandas_df_for_scaling
    
########################## End ##############################



#list_available_datasets()
# fetch dataset 
forest_fires = fetch_ucirepo(id=162) 
  
# data (as pandas dataframes) 
X = forest_fires.data.features 
y = forest_fires.data.targets 
  
# metadata 
print(forest_fires.metadata) 
  
# variable information 
print(forest_fires.variables)

cleaned_up_X = X.copy()

categorized_month_column = column_to_categorical(cleaned_up_X['month'])
categorized_day_column = column_to_categorical(cleaned_up_X['day'])

cleaned_up_X['month'] = categorized_month_column
cleaned_up_X['day'] = categorized_day_column
print(cleaned_up_X)

maximum_value_scaling(cleaned_up_X)


