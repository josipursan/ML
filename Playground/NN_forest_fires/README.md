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
&nbsp;&nbsp;&nbsp;-a numeric rating of the moisture content of litter and other cured fine fuels  
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
