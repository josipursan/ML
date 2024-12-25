from ucimlrepo import fetch_ucirepo 
  
# fetch dataset 
magic_gamma_telescope = fetch_ucirepo(id=159) 
  
# data (as pandas dataframes) 
X = magic_gamma_telescope.data.features 
y = magic_gamma_telescope.data.targets 

print("All features\nX : {}\n".format(X))
print("Example of grabbing some features from the pandas df\nX.fLength : {}\n\nX.fAsym : {}\nX.fSize : {}\n".format(X.fLength, X.fAsym, X.fSize))
print("\nAll target values : {}\n".format(y))

# metadata 
#print(magic_gamma_telescope.metadata) 
  
# variable information 
#print(magic_gamma_telescope.variables)