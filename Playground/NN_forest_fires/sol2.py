from ucimlrepo import fetch_ucirepo, list_available_datasets
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

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

# ---------------------------------------------------------------------------------------------------------

#create_report_file()
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

print("log1p_X : \n{}\n".format(log1p_X))
print("log1p_y : \n{}\n".format(log1p_y))
print("mean_normalized_X : \n{}\nmean_normalized_y : \n{}\n".format(mean_normalized_X, mean_normalized_y))
print("mean_normalized_log1p_X : \n{}\nmean_normalized_log1p_y : \n{}\n".format(mean_normalized_log1p_X, mean_normalized_log1p_y))

for column in log1p_X:
    fig = plt.figure(figsize=(18,9))
    ax1 = fig.add_subplot(511)
    ax2 = fig.add_subplot(512)
    ax3 = fig.add_subplot(513)
    #ax4 = fig.add_subplot(614)
    ax5 = fig.add_subplot(514)
    ax6 = fig.add_subplot(515)


    ax1.title.set_text("{} histogram".format(column))
    ax2.title.set_text("log1p({}) histogram".format(column))
    ax3.title.set_text("mean_normalized_{} histogram".format(column))
    #ax4.title.set_text("log1p(mean_normalized_{}) histogram".format(column))
    ax5.title.set_text("min_max_scaled_{} histogram".format(column))
    ax6.title.set_text("log1p(min_max_scaled_{}) histogram".format(column))

    ax1.set(xlabel="Measurement bin value", ylabel="Number of values")
    ax2.set(xlabel="Measurement bin value", ylabel="Number of values")
    ax3.set(xlabel="Measurement bin value", ylabel="Number of values")
    #ax4.set(xlabel="Measurement bin value", ylabel="Number of values")
    ax5.set(xlabel="Measurement bin value", ylabel="Number of values")
    ax6.set(xlabel="Measurement bin value", ylabel="Number of values")

    ax1.hist(X[column], bins=80)
    ax2.hist(log1p_X[column], bins=80)
    ax3.hist(mean_normalized_X[column], bins=80)
    #ax4.hist(mean_normalized_log1p_X[column], bins=80)
    ax5.hist(min_max_scaled_X[column], bins=80)
    ax6.hist(min_max_scaled_log1p_X[column], bins=80)


    fig.tight_layout()
    #fig.subplots_adjust( left=None, bottom=None,  right=None, top=None, wspace=None, hspace=None)
    #plt.show()
    plt.savefig('{}_VS_log1p({})_VS_mean_normalized_{}_VS_min_max_scaled_{}_VS_log1p(min_max_scaled_{})HISTOGRAMS.png'.format(column, column, column, column, column), bbox_inches='tight')