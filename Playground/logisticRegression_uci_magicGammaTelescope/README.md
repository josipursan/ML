UCI dataset : MAGIC Gamma Telescope  
https://archive.ics.uci.edu/dataset/159/magic+gamma+telescope  
  
Dataset can be imported using *ucimlrepo* pip package :  
```python
pip install ucimlrepo  
  
from ucimlrepo import fetch_ucirepo  
  
# fetch dataset  
magic_gamma_telescope = fetch_ucirepo(id=159)  
  
# data (as pandas dataframes)  
X = magic_gamma_telescope.data.features  
y = magic_gamma_telescope.data.targets  
  
# metadata  
print(magic_gamma_telescope.metadata)  
  
# variable information  
print(magic_gamma_telescope.variables)  
```  
  
## About the dataset  
This dataset is a synthetic dataset simulating registration of high energy gamma particles in a ground-based  
atmospheric Cherenkov gamma telescope.  
More about the Cherenkov telescope array : https://en.wikipedia.org/wiki/Cherenkov_Telescope_Array  
  
## What is the goal of this mini project?  
We will basically train a model on the given dataset so that the trained model provides best possible predictions for the given inputs.
Inputs are defined by the dataset.
Trained model will effectively replicate what the parameters are of the model used by the initial creator of this dataset.  
  
## Deliverables?  
*sol.py* - main .py file handling both data ingestion, data wrangling (where/if necessary), model training, model evaluation, plotting, as well as saving training stats.  
  
*model_cost.png* - plot showing how cost's dependency to number of iterations  
  
*model_stats.txt* - .txt file containing relevant stats such as number of iterations, model parameters, model cost, number of model errors (when running on y_vals).  
