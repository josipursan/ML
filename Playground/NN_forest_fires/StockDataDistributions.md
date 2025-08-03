# Stock data distributions
A dive into the stock data - histogram, scatter plots, value ranges, distribution shapes, and the necessary preprocessing  
steps to make the model train and perform the best it can.  
  
|Var.name| min | max |        mean        |median|     skew      |      variance     |      std.dev     |        kurtosis      |
|--------|-----|-----|--------------------|------|---------------|-------------------|------------------|----------------------|
X        |  1  |  9  |4.669245647969052   |  4.0 |0.8677 (right) |5.353567840702922  |2.3137778287257666|-3.01752799091 (platy)|
Y        |  1  |  9  |4.299806576402321   |  4.0 |0.7312 (right) |1.5126550012744964 |1.2299004029898097|-3.01753772923 (platy)|
FFMC     |18.7 | 96.2|90.64468085106384   |  91.6|-0.5191 (left) |30.471623783605477 |5.520110848851269 |-3.01778427949 (platy)|
DMC      | 1.1 |291.3|110.87234042553192  | 108.3|0.1204 (right) |4101.951888504041  |64.04648224925425 |-3.01753316320 (platy)|
DC       | 7.9 |860.6|547.9400386847195   | 664.2|-1.4059 (left) |61536.83546744037  |248.06619170584364|-3.01753147285 (platy)|
ISI      | 0.0 |56.1 |9.021663442940039   |   8.4|0.4090 (right) |20.78883211131603  |4.559477175216039 |-3.01761298568 (platy)|
temp     | 2.2 |33.3 |18.88916827852998   |  19.3|-0.2122 (left) |33.71689795030963  |5.806625349573505 |-3.01753290534 (platy)|
RH       | 15  | 100 |44.28820116054158   |  42.0|0.4207 (right) |266.25980237806067 |16.31746923937841 |-3.01753403966 (platy)|
wind     | 0.4 | 9.4 |4.017601547388782   |   4.0|0.0294 (right) |3.2100190424782213 |1.7916526009464617|-3.01753259796 (platy)|
rain     | 0.0 | 6.4 |0.021663442940038684|   0  |0.2195 (right) |0.08759180123851079|0.295959120890894 |-3.01911469054 (platy)|

If mean > median, you have a long right tail (positive skew)  
If mean < median, you have a long left tail (negative skew) (these are rule-of-thumb estimates, skew and kurtosis should always be computed)  
  
Skew - represents distribution's degree of asymmetry  
&nbsp;&nbsp;&nbsp;-a distribution may be right or left skewed  
&nbsp;&nbsp;&nbsp;-right-skew is also called positive skew because the tail extends in the positive direction of the x axis  
&nbsp;&nbsp;&nbsp;-left skew is also called negative skew because the tail extens in the neative direction of the x axis  
**Pearson's second skeweness coefficient (median skeweness)**  
$skew = \frac{3(mean - median)}{std.dev}$  
&nbsp;&nbsp;&nbsp;-close to 0, or 0, indicates no or little skew  
&nbsp;&nbsp;&nbsp;-positive value indicates a right skewed (positive skew) distribution  
&nbsp;&nbsp;&nbsp;-negative value indicates a left skewed (negative skew) distribution  
&nbsp;&nbsp;&nbsp;-there are other formulas for computing skeweness  
  
Kurtosis - a measure of tailedness of the distribution  
&nbsp;&nbsp;&nbsp;-indicates how fat or thin the tails are, ie. how much of the dataset is in the tails  
&nbsp;&nbsp;&nbsp;-kurtosis is often called *excessive kurtosis* because any kind of deviation from the kurtosis of normal distribution, which is 3, can be considered excessive kurtosis :  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; $excessiveKurtosis = 3 - kurtosis$  
&nbsp;&nbsp;&nbsp;-there are 3 kinds of kurtosis - mesokurtosis, platykurtosis, leptokurtosis  
&nbsp;&nbsp;&nbsp;-mesokurtosis (*meso* is greek for *middle, intermediate*) - normal tailedness, not excessive tailedness (normal distribution is an example of mesokurtic distribution)  
&nbsp;&nbsp;&nbsp;-platykurtosis (*platy* is greek for *flat, broad*) - flattened, squished, normal distribution; a distribution yielding a *negatve excessive kurtosis* value is platykurtic; visually they look quite plateauish on top  
&nbsp;&nbsp;&nbsp;-leptokurosis (*lepto* is greek for *slight, thin, fine*) - characterized by a very thin, peaky top, and very fat tails; distributions yielding a *positive excessive kurtosis* value are leptokurtic  

Variance - how much values of data set vary from each other  
Std.dev - how much values of data set vary from the mean  
Both show dataset spread.  
  
## X - xaxis coordinate in the park  
Histogram showing *X* variable distribution :  
<p style="text-align: center">
    <img src="./stock_data_distributions/nicerPlots/X_coord_histo.png"/>
</p>  
  
  
Scatter plot showing *X* values for each datapoint :  
<p style="text-align: center">
    <img src="./stock_data_distributions/nicerPlots/X_coord_scatter.png"/>
</p>  
  
Variable *X* takes on values [1-9].  
|Var.name| min | max |        mean        |median|     skew      |      variance     |      std.dev     |        kurtosis      |
|--------|-----|-----|--------------------|------|---------------|-------------------|------------------|----------------------|
X        |  1  |  9  |4.669245647969052   |  4.0 |0.8677 (right) |5.353567840702922  |2.3137778287257666|-3.01752799091 (platy)|  
  
Variable *X* has a small range of possible values.  
Both histo and scatter show existing values are pretty uniformly spread out.  
Mean and median are pretty close to eachother.  
A very slight right (positive) skew exists.  
Kurtosis confirms this further, inidcating a platy distribution - flat, wide top (peak) and thin tails.  
  
Still a bit on the edge whether this variable should be used at all.  
If used, scale it and check the model behaves. Only then consider some transformation, although I don't think it is necessary considering histo and scatter plots.  
  
## Y - yaxis cordinate in park  
Histogram showing *Y* variable distribution :  
<p style="text-align: center">
    <img src="./stock_data_distributions/nicerPlots/Y_coord_histo.png"/>
</p>  
  
  
Scatter plot showing *Y* values for each datapoint :  
<p style="text-align: center">
    <img src="./stock_data_distributions/nicerPlots/Y_coord_scatter.png"/>
</p>  
  
A drastically different situation compared to variable *X*.  
  
