from ucimlrepo import fetch_ucirepo, list_available_datasets
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

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

forest_fires = fetch_ucirepo(id=162)
X = forest_fires.data.features
y = forest_fires.data.targets



print("X : {}\n\ny : {}\n".format(X, y))
print(X.describe())
for col in X:
    if col == 'month' or col == 'day':
        continue
    avg = X[col].mean()
    power_fourth_sum = 0
    square_square_sum = 0
    for i in range(len(X[col])):
        power_fourth_sum += (X[col][i] - avg)**4
        square_square_sum += (X[col][i] - avg)**2
    square_square_sum **=2

    n, m = X.shape
    #https://www.scribbr.com/statistics/kurtosis/
    kurtosis = (n*(n+1)/((n-1)*(n-2)*(n-3)))*((power_fourth_sum)/(square_square_sum))-3*((n-1)**2/((n-2)*(n-3)))
    print("Col {}  kurtosis {}\n".format(col, kurtosis))
