# UCI dataset
https://archive.ics.uci.edu/datasets
## Forest fires
https://archive.ics.uci.edu/dataset/162/forest+fires  
  
`Forest fires` dataset is made up of various relevant parameters which should facilitate better prediction of forest fire spread rates.  
  
|  Var. name | Data type  | Variable description  |
|---|---|---|
| X  | int  | x-axis coordinate in the Montesinho park map (1-9)   |
| Y  | int  | y-axis coordinate in the Montesinho park map (1-9)  |
| month  | categorical  | month of the year; 'jan' to 'dec'  |
| day  | categorical  | day of the week; 'mon' to 'sun'   |
| FFMC  | continous  | FFMC index from the FWI system; (18.7 to 96.20)  |
| DMC  | int   | DMC index from the FWI system; (1.1 to 291.3)   |
| DC  | continuous  | DC index from the FWI system; (7.9 to 860.6)  |
| ISI  | continuous  | ISI index from the FWI sytem; (0.0 to 56.10)  |
| temp  | int  | temperature; (2.2 to 33.30)  |
| RH  | continuous  | relative humidity; (15.0 to 100)  |
| wind  | int   | wind speed; (0.40 to 9.40)  |
| rain  | int  | rain; (0.0 to 6.4)  |
| area  | integer  | burned area of the forest; 0.0 to 1090.84 |

*area* is the **TARGET**.  
All other variables are **features**.  
  
**FFMC** - Fine Fuel Moisture Code  
&nbsp;&nbsp;&nbsp;-a numberic rating of the moisture content of litter and other cured fine fuels  
&nbsp;&nbsp;&nbsp;-this code is an indicator of the relative ease of ignition and flamability of fine fuel  
  
**DMC** - Duff Moisture Code  
&nbsp;&nbsp;&nbsp;-a numeric rating of the average moisture content of loosely compacted organic layers of moderate depth  
&nbsp;&nbsp;&nbsp;-this code gives an indication of fuel consumption in moderate duff layers and medium-size woody material  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;-duff layer - a deep, dense layer of partially decomposed pine needles  
  
**DC** - Drought Code  
&nbsp;&nbsp;&nbsp;-a numeric rating of the average moisture content of deep, compact organic layers  
&nbsp;&nbsp;&nbsp;-this code is a useful indicator of seasonal drought effects on forest fuels and the amount of smoldering in deep duff layers and large logs  
  
**ISI** - Initial Spread Index  
&nbsp;&nbsp;&nbsp;-a numeric rating of the expected rate of fire spread  
&nbsp;&nbsp;&nbsp;-based on wind speed and FFMC  
&nbsp;&nbsp;&nbsp;-ISI does not take fuel type into account, hence actual spread rates vary between fuel types for the same ISI value  
  
### What is the goal of this mini project?  
1. Get the NN to perform as good as possible on this data (above 90% accuracy, preferrably on all data sets)  
2. Create a graph showing influence of variables on the resulting burned area.  

### How will the above be achieved?  
**As for 1.**  
&nbsp;&nbsp;&nbsp;-scale or normalize the data before feeding it to NN  
&nbsp;&nbsp;&nbsp;-optimize as much as possible :  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;-try out inputing different variable subsets to the NN  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;-vary the number of layers  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;-optimize hyperparameters  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;-optimize number of epochs  
  
**As for 2.**  
&nbsp;&nbsp;&nbsp;-once you have an optimized NN performing well on all data sets, plot 3D plots showing how burned area depends on two other, chosen, variables  
&nbsp;&nbsp;&nbsp;-note that you will have to plot multiple 3D plots  
  
*Initial comments:*  
-FFMC, DMC, DC and ISI look like they can nicely be scaled to some reasonable ranges  
-month and day might be a bit problematic if handled as is ('jan', 'feb', ...) - preprocess this either to two categorical variables (one for months and one for day), or one hot encode it to binary variables  
  
### 09.04.2025. - Comments about stock data distributions  
Burned area distribution is highly skewed, overwhelming majority of the dataset sits at, or about, 0.  
Large value range, will probably benefit from scaling, and even normalization.
  
DC distribution is split, right hand portion being bigger than the left hand one. Left hand portion encompasses values 0-180, while right hand portion starts about 580. Some values are sprinkled between 180 and 580.  
Will benefit from scaling.  
  
DMC distribution is relatively cohesive, pretty Gaussian, some stragglers on the far right.  
Needs scaling.  
  
FFMC distribution is quite skewed, although most of the values are concentrated from 80 to ~95.  
Definitely scale, consider normalization.  
  
ISI distribution looks ok, quite Gaussian, scaling needs to be done.  
  
Rain distribution is basically a set composed of only few entries != 0. Needs to be scaled.  
  
RH distribution definitely needs scaling, Gaussian-ish, but quite flat.  
  
temp distribution is a nice Gaussian distribution, a little tail on the right, as well as whiskers on the top. Scaling is a must.
  
Wind distribution is quite binned. Maybe I messed up the binning? Anyways, scaling needs to be done to get a nicer range. Histo is quite sparse, although a bit Gaussian looking with a tail on the right.