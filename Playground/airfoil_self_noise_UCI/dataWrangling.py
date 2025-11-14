from ucimlrepo import fetch_ucirepo
import matplotlib.pyplot as plt
import numpy as np
import time
import os
import shutil

def createDirectory(plotsDir):
    if os.path.isdir(plotsDir):
        newName_oldPlotsDir = plotsDir + str(time.time())
        print("Directory {} already exists!\nNew name for the old directory : {}\n".format(plotsDir, newName_oldPlotsDir))
        shutil.move(plotsDir, newName_oldPlotsDir)
    try:
        os.mkdir(plotsDir)
    except Exception as e:
        print("Exception caught when trying to create directory {}\nError : {}\n".format(plotsDir, e))

# fetch dataset
airfoil_self_noise = fetch_ucirepo(id=291)

# data (as pandas dataframes)
X = airfoil_self_noise.data.features
y = airfoil_self_noise.data.targets

statsFile = open("./stats_analysis_dump.txt", "w")
statsFile.write("stats_analysis_dump\nTIMESTAMP : {}\n\n".format(time.time()))
statsFile.write("===============================================\n\n")
X_columns = list(X.columns.values)

for col in X_columns:
    plotsDir = "./" + col + "_plots"
    createDirectory(plotsDir)

    tailVerdict = "EMPTY"
    excessiveKurtosis = "EMPTY"
    minVal = X[col].min()
    maxVal = X[col].max()
    mean = X.loc[:, col].mean()
    median = X.loc[:, col].median()
    variance = X[col].var()
    stddev = X[col].std()
    skew = X.loc[:, col].skew()
    kurtosis = X.loc[:, col].kurtosis()

    if mean > median :
        tailVerdict = "mean > median --> long right tail, positive skew"
    else:
        tailVerdict = "mean < median --> long left tail, negative skew"

    excessiveKurtosis = 3 - kurtosis
    if excessiveKurtosis > 0:
        kurtosisVerdict = "excessiveKurtosis > 0, leptokurtosis"
    elif excessiveKurtosis == 0:
        kurtosisVerdict = "excessiveKurtosis == 0, mesokurtosis"
    elif excessiveKurtosis < 0:
        kurtosisVerdict = "excessiveKurtosis < 0, platykurtosis"
    
    statsFile.write("{} stats :\n\tmin : {}\n\tmax : {}\t\n\tmean : {}\n\tmedian : {}\n\tvariance : {}\n\tstddev : {}\n\tskew : {}\n\tkurtosis : {}\n\texcessiveKurtosis : {}\n\ttailVerdict : {}\n\t{}\n".format(col, minVal, maxVal, mean, median, variance, stddev, skew, kurtosis, excessiveKurtosis, tailVerdict, kurtosisVerdict))
    statsFile.write("- - - - - - - - - - - - -\n\n")

    plt.hist(X[col])
    plt.title(col + ' histogram')
    plt.xlabel('Frequency bins')
    plt.ylabel('Bin count')
    plt.savefig(plotsDir + "/" + col + '_histogram.png')
    plt.clf()

    plt.scatter([i for i in range(0, len(X[col]))], X[col])
    plt.title(col + ' scatter plot')
    plt.xlabel('Individual value')
    plt.ylabel(col)
    plt.savefig(plotsDir + "/" + col + '_scatter.png')
    plt.clf()

    plt.violinplot(X[col])
    plt.title(col + ' violin plot')
    plt.xlabel(col)
    plt.ylabel(col + ' values')
    plt.savefig(plotsDir + "/" + col + '_violin.png')
    plt.clf()

statsFile.close()
