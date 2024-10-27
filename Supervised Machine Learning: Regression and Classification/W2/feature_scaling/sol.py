"""
This .py is just a quick demo of feature scaling approaches shown in the lectures.

There are three ideas : 
    1. Divide by maximum value
    2. Mean normalization
    3. Z-score normalization

Considering all of these are probably available through a simple function call, I will actually break them down completely.
All of the additions, and really anything requiring you to iterate over an array will be fully implemented. Doing a simple sum_vector() call too boring.
Existing functions will be used to double check my implementations.
"""

import numpy as np

def get_max(array_of_values):
    found_max = -1
    max_el_index = 0

    for el in array_of_values:
        if el > found_max:
            found_max = el
            max_el_index = np.where(array_of_values == found_max)
    
    print("Max value in array is : {}\nIndex is : {}\n".format(found_max, max_el_index[0]))

    return found_max, max_el_index

def get_min(array_of_values) : 
    min_value = array_of_values[0]
    min_value_index = 0

    for el in array_of_values:
        if el < min_value:
            min_value = el
            min_value_index = np.where(array_of_values == min_value)

    print("Min value in array is : {}\nIndex is : {}\n".format(min_value, min_value_index))

    return min_value, min_value_index

def divide_by_max(array_to_scale):
    print("\n\n=======================Normalize dividing by max=======================\n")
    max_val, max_val_index = get_max(array_to_scale)
    
    print("Max value in array is : {}\nIndex is : {}\n".format(max_val, max_val_index[0]))

    scaled_array = np.zeros(array_to_scale.shape[0])    # Creating an array where scaled values will be stored; size must match size of the array which has been passed to the function

    for i in range(array_to_scale.shape[0]):
        scaled_array[i] = array_to_scale[i]/max_val
    
    print("Original array : {}\nScaled array : {}\n".format(array_to_scale, scaled_array))

    return scaled_array

def mean_normalization(array_to_scale):
    print("\n\n====================Mean normalization====================\n")
    # First we have to find the average of given array
    average = 0
    number_of_elements = array_to_scale.shape[0]

    for i in range(number_of_elements):
        average += array_to_scale[i]
    
    average /= number_of_elements

    print("Number of elements : {}\nAverage : {}\nNumpy build in mean function : {}\n".format(number_of_elements, average, np.mean(array_to_scale)))

    normalized_array = np.zeros(number_of_elements)

    max_value, _ = get_max(array_to_scale)
    min_value, _ = get_min(array_to_scale)

    for i in range(number_of_elements):
        normalized_array[i] = ( (array_to_scale[i] - average) / (max_value - min_value) )

    print("Given array : {}\nNormalized array : {}\n".format(array_to_scale, normalized_array))

    return normalized_array

def z_score_normalization(array_to_scale):
    print("\n\n============================Z score normalization========================\n")

    # For z score normalization we will need average and std.dev

    average = 0
    number_of_elements = array_to_scale.shape[0]

    for i in range(number_of_elements):
        average += array_to_scale[i]
    
    average /= number_of_elements

    print("Number of elements : {}\nAverage : {}\nNumpy built in average : {}\n".format(number_of_elements, average, np.mean(array_to_scale)))

    standard_deviation = 0

    for i in range(number_of_elements):
        standard_deviation += (array_to_scale[i] - average)**2

    standard_deviation = standard_deviation/number_of_elements
    standard_deviation = (standard_deviation)**(0.5)

    z_score_normalized = np.zeros(number_of_elements)

    for i in range(number_of_elements):
        z_score_normalized[i] = ((array_to_scale[i] - average)/standard_deviation)

    print("Number of elements : {}\nAverage : {}\nNumpy mean : {}\nStd dev : {}\nNumpy std dev : {}\n".format(number_of_elements, average, np.mean(array_to_scale), standard_deviation, np.std(array_to_scale)))
    print("Given array : {}\nZ Score normalized array : {}\n".format(array_to_scale, z_score_normalized))

    return z_score_normalized


def main():
    my_array = np.array([1, 5, 2, 8, 29.0, 33.54, 32, 31, 30, 54, 46, 21, 29, 38.9, 28.5])
    scaled_array = divide_by_max(my_array)

    mean_normalized_array = mean_normalization(my_array)

    z_score_normalized = z_score_normalization(my_array)

main()