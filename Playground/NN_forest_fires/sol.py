from ucimlrepo import fetch_ucirepo, list_available_datasets
import matplotlib.pyplot as plt

  
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

print("X.FFMC : \n{}\n".format(X['FFMC']))

plt.hist(X['FFMC'], bins = 75)
plt.title("FFMC distribution (75 bins)")
plt.xlabel("FFMC values")
plt.ylabel("Frequency")
plt.show()

plt.hist(X['DMC'], bins = 75)
plt.title("DMC distribution (75 bins)")
plt.xlabel("DMC values")
plt.ylabel("Frequency")
plt.show()

plt.hist(X['DC'], bins = 75)
plt.title("DC distribution (75 bins)")
plt.xlabel("DC values")
plt.ylabel("Frequency")
plt.show()

plt.hist(X['ISI'], bins = 75)
plt.title("ISI distribution (75 bins)")
plt.xlabel("ISI values")
plt.ylabel("Frequency")
plt.show()

plt.hist(X["temp"], bins = 100)
plt.title("temp distribution (100 bins)")
plt.xlabel("temp values")
plt.ylabel("Frequency")
plt.show()

plt.hist(X["RH"], bins = 100)
plt.title("RH distribution (100 bins)")
plt.xlabel("RH values")
plt.ylabel("Frequency")
plt.show()

plt.hist(X["wind"], bins = 100)
plt.title("wind distribution (100 bins)")
plt.xlabel("wind values")
plt.ylabel("Frequency")
plt.show()

plt.plot(X["rain"], 'o')
plt.title("rain distribution")
plt.xlabel("rain values")
plt.ylabel("mm/m2")
plt.show()

plt.plot(y["area"], 'o')
plt.title("area distribution")
plt.xlabel("area entry")
plt.ylabel("ha")
plt.show()