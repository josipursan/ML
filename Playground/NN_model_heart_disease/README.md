# General project and dataset info
Just a quick, simple README.md explaining what I did in this project, any possible caveats, as well as anything related to the used dataset.  
  
## Dataset  
UCI ML Repo - Heart Disease dataset :    
https://archive.ics.uci.edu/dataset/45/heart+disease  
  
Based on the dataset description, it looks like this UCI dataset is similar, or identical, to this one on Kaggle : https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset  
  
Original database used to construct this dataset is made up of 76 different attributes/measured variables.  
However, dataset is made up of only 14 of the initial 76 variables (13 features (X), and 1 target (y)) :  
&nbsp;&nbsp;&nbsp; age (*integer*) - age of the patient  
&nbsp;&nbsp;&nbsp; sex (*categorical*) - male or female  
&nbsp;&nbsp;&nbsp; cp (*categorical) - chest pain; can have 4 possible values (probably increasing in intensity from 0 to 3)   
&nbsp;&nbsp;&nbsp; trestbps (*integer*) - resting blood pressure (in mmHg)  
&nbsp;&nbsp;&nbsp; chol (*integer*) - serum cholesterol (mg/dl)  
&nbsp;&nbsp;&nbsp; fbs (*categorical*) - fasting blood sugar; categorical variable because its value declares whether a patinet has *fbs* below or over 120mg/dl (I guess if above 120mg/dl this value of *fbs* is set to 1, else set to 0)  
&nbsp;&nbsp;&nbsp; restecg (*categorical*) - resting ecg results; categorical variable, possible values are 0,1,2; I'm guessing 0 denotes no abnormality, 1 indicates some abnormality/issue, and 2 indicates a definitive, or big, abnormality on ecg  
&nbsp;&nbsp;&nbsp; thalach (*integer*) - maximum achieved HR  
&nbsp;&nbsp;&nbsp; exang (*categorical*) - exercise induced angina; possible categories : 0, 1  
&nbsp;&nbsp;&nbsp; oldpeak (*integer*) - ST depression induced by exercise, relative to rest;  
&nbsp;&nbsp;&nbsp; slope (*categorical*) - slope of the peak ST segment during exercise; looks like possible categories are 0, 1, 2  
&nbsp;&nbsp;&nbsp; ca (*integer*) - (ca = coronary angiogram?); variable used to define number of major vessels colored by fluoroscopy; possible values are 0,1,2,3,4, so I guess this can also be considered a categorical variable  
&nbsp;&nbsp;&nbsp; thal (*categorical*) - don't know what this means; categorical variable, possible values being 1 (normal), 2 (fixed defect), 3 (reversable defect)  
&nbsp;&nbsp;&nbsp; target (*integer/categorical*) - target label used to define whether a patient represented by the row of values listed above has, or has not, a heart condition; possible values are 0, 1, 2, 3, 4. Value *0* indicates no heart conditions are present. Values *1*, *2*, *3*, *4* probably indicate increasing severity of found heart condition.
  
Looks like the original (76 variable) dataset contains some other, interesting, inputs such as smoker (*y/n*), number of cigarettes (*integer*), ...  
  
**NOTE**  
Of the above listed variables, for now I will **ignore** *thal* considering there dataset initial examination of the dataset docs didn't provide an adequate explanation of its meaning - it looks like it has to do something with heartrate. If I later change my mind about, I'll add another comment like this.  
  
## Goal of this project  
The goal of this project is to use TF to implement a simple NN that will be trained on the data provided by the above described dataset.  
  
How many layers, and nodes per layer, do you need to implement?  
&nbsp;&nbsp;&nbsp; As a rule of thumb, refer to https://stats.stackexchange.com/questions/181/how-to-choose-the-number-of-hidden-layers-and-nodes-in-a-feedforward-neural-netw  
&nbsp;&nbsp;&nbsp; -number of nodes in input layer needs to match number of attributes  
&nbsp;&nbsp;&nbsp; -one hidden layer, whose number of neurons is some value between the number of neurons in the input layer, and the number of neurons in the output layer, is sufficient  
&nbsp;&nbsp;&nbsp; -output layer needs only one neuron - currently I intend to run the NN in regression mode, ie. interpreting the output value as the probability of heart disease  
&nbsp;&nbsp;&nbsp; -you can also try running NN in *machine mode*, ie. applying softmax to the result of the output layer, transforming the output value into either 0 or 1  
