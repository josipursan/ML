from ucimlrepo import fetch_ucirepo
import matplotlib.pyplot as plt
import numpy as np
import time

# fetch dataset
airfoil_self_noise = fetch_ucirepo(id=291)

# data (as pandas dataframes)
X = airfoil_self_noise.data.features
y = airfoil_self_noise.data.targets

# variable information
print(airfoil_self_noise.variables)

print(X)
print("\n\n")
print(y)

#X['frequency'].plot(kind='hist', bins=20)

minMax = [X['frequency'].min().item(), X['frequency'].max().item()]
#print("\nPD DF describe : {}\n".format(X.describe()))
#agg_res = X.select_dtypes(include="number").agg(["min", "max", "avg", "median", "kurtosis", "skew"]) # one column is messing up stuff here because it is a 'Series' object, even though I am forcing dtypes=number - too lazy to add handling for this, which is why I'll write my own function for per column analysis lol
plt.hist(X['frequency'], bins=20)
plt.title("Frequency histogram")
plt.xlabel("Frequency bin")
plt.ylabel("Bin count")
#plt.show()

statsFile = open("./stats_analysis_dump.txt", "w")
statsFile.write("stats_analysis_dump\nTIMESTAMP : {}\n\n".format(time.time()))
statsFile.write("===============================================\n\n")
X_columns = list(X.columns.values)

for col in X_columns:
    mean = X.loc[:, col].mean()
    median = X.loc[:, col].median()
    skew = X.loc[:, col].skew()
    statsFile.write("{} stats :\n\tmean : {}\n\tmedian : {}\n\tskew : {}\n".format(col, mean, median, skew))
    statsFile.write("- - - - - - - - - - - - -\n\n")

    #statsFile.write("{}\n\tmean : {}\n".format(col, X.loc[:, col].mean()))
    #print("{} mean : {}\n".format(col, X.loc[:, col].mean()))
    
statsFile.close()
