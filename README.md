
# Interactions - Lab

## Introduction

In this lab, you'll explore interactions in the Boston Housing data set.

## Objectives

You will be able to:
- Understand what interactions are
- Understand how to accommodate for interactions in regression

## Build a baseline model 

You'll use a couple of built-in functions, which we imported for you below.


```python
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.datasets import load_boston
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
```

Import the Boston data set using `load_boston()`. We won't bother to preprocess the data in this lab. If you still want to build a model in the end, you can do that, but this lab will just focus on finding meaningful insights in interactions and how they can improve $R^2$ values.


```python
regression = LinearRegression()
boston = load_boston()
```

Create a baseline model which includes all the variables in the Boston housing data set to predict the house prices. The use 10-fold cross-validation and report the mean $R^2$ value as the baseline $R^2$.


```python
## code here
crossvalidation = KFold(n_splits=10,shuffle=True, random_state=42)
baseline = np.mean(cross_val_score(regression, X=boston.data, y=boston.target, scoring='r2', cv=crossvalidation))
```


```python
baseline
```




    0.717068752774469



## See how interactions improve your baseline

Next, create all possible combinations of interactions, loop over them and add them to the baseline model one by one to see how they affect the R^2. We'll look at the 3 interactions which have the biggest effect on our R^2, so print out the top 3 combinations.

You will create a for loop to loop through all the combinations of 2 predictors. You can use `combinations` from itertools to create a list of all the pairwise combinations. To find more info on how this is done, have a look [here](https://docs.python.org/2/library/itertools.html).


```python
from itertools import combinations
combinations = list(combinations(boston.feature_names, 2))
```


```python
df = pd.DataFrame(boston.data, columns=boston.feature_names)
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CRIM</th>
      <th>ZN</th>
      <th>INDUS</th>
      <th>CHAS</th>
      <th>NOX</th>
      <th>RM</th>
      <th>AGE</th>
      <th>DIS</th>
      <th>RAD</th>
      <th>TAX</th>
      <th>PTRATIO</th>
      <th>B</th>
      <th>LSTAT</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.00632</td>
      <td>18.0</td>
      <td>2.31</td>
      <td>0.0</td>
      <td>0.538</td>
      <td>6.575</td>
      <td>65.2</td>
      <td>4.0900</td>
      <td>1.0</td>
      <td>296.0</td>
      <td>15.3</td>
      <td>396.90</td>
      <td>4.98</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.02731</td>
      <td>0.0</td>
      <td>7.07</td>
      <td>0.0</td>
      <td>0.469</td>
      <td>6.421</td>
      <td>78.9</td>
      <td>4.9671</td>
      <td>2.0</td>
      <td>242.0</td>
      <td>17.8</td>
      <td>396.90</td>
      <td>9.14</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.02729</td>
      <td>0.0</td>
      <td>7.07</td>
      <td>0.0</td>
      <td>0.469</td>
      <td>7.185</td>
      <td>61.1</td>
      <td>4.9671</td>
      <td>2.0</td>
      <td>242.0</td>
      <td>17.8</td>
      <td>392.83</td>
      <td>4.03</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.03237</td>
      <td>0.0</td>
      <td>2.18</td>
      <td>0.0</td>
      <td>0.458</td>
      <td>6.998</td>
      <td>45.8</td>
      <td>6.0622</td>
      <td>3.0</td>
      <td>222.0</td>
      <td>18.7</td>
      <td>394.63</td>
      <td>2.94</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.06905</td>
      <td>0.0</td>
      <td>2.18</td>
      <td>0.0</td>
      <td>0.458</td>
      <td>7.147</td>
      <td>54.2</td>
      <td>6.0622</td>
      <td>3.0</td>
      <td>222.0</td>
      <td>18.7</td>
      <td>396.90</td>
      <td>5.33</td>
    </tr>
  </tbody>
</table>
</div>




```python
## code to find top 3 interactions by R^2 value here
interaction_scores = []
for combo in combinations:
    df["interaction"] = df[combo[0]]*df[combo[1]]
    crossvalidation = KFold(n_splits=10,shuffle=True, random_state=42)
    interaction_score = np.mean(cross_val_score(regression, X=df.values, y=boston.target, 
                                                scoring='r2', cv=crossvalidation))
    if interaction_score > baseline:
        interaction_scores.append((combo[0], combo[1], interaction_score))
```


```python
interaction_scores = sorted(interaction_scores, key=lambda inter: inter[2], reverse=True)
interaction_scores[:3]
```




    [('RM', 'LSTAT', 0.7826199503986451),
     ('RM', 'TAX', 0.7717836595567208),
     ('RM', 'RAD', 0.7661601104295918)]



## Look at the top 3 interactions: "RM" as a confounding factor

The top three interactions seem to involve "RM", the number of rooms as a confounding variable for all of them. Let's have a look at interaction plots for all three of them. This exercise will involve:

- splitting our data up in 3 groups: one for houses with a few rooms, one for houses with a "medium" amount of rooms, one for a high amount of rooms.
- Create a function `build_interaction_rm`. This function takes an argument `varname` (which can be set equal to the column name as a string) and a column `description` (which describes the variable or varname, to be included on the x-axis of the plot). The function outputs a plot that uses "RM" as a confounding factor. 

We split the data set for high, medium and low amount of rooms for you.


```python
df_target = pd.DataFrame(boston.target, columns=['target'])
alldata = pd.concat([df, df_target], axis=1)
alldata.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CRIM</th>
      <th>ZN</th>
      <th>INDUS</th>
      <th>CHAS</th>
      <th>NOX</th>
      <th>RM</th>
      <th>AGE</th>
      <th>DIS</th>
      <th>RAD</th>
      <th>TAX</th>
      <th>PTRATIO</th>
      <th>B</th>
      <th>LSTAT</th>
      <th>interaction</th>
      <th>target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.00632</td>
      <td>18.0</td>
      <td>2.31</td>
      <td>0.0</td>
      <td>0.538</td>
      <td>6.575</td>
      <td>65.2</td>
      <td>4.0900</td>
      <td>1.0</td>
      <td>296.0</td>
      <td>15.3</td>
      <td>396.90</td>
      <td>4.98</td>
      <td>1976.5620</td>
      <td>24.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.02731</td>
      <td>0.0</td>
      <td>7.07</td>
      <td>0.0</td>
      <td>0.469</td>
      <td>6.421</td>
      <td>78.9</td>
      <td>4.9671</td>
      <td>2.0</td>
      <td>242.0</td>
      <td>17.8</td>
      <td>396.90</td>
      <td>9.14</td>
      <td>3627.6660</td>
      <td>21.6</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.02729</td>
      <td>0.0</td>
      <td>7.07</td>
      <td>0.0</td>
      <td>0.469</td>
      <td>7.185</td>
      <td>61.1</td>
      <td>4.9671</td>
      <td>2.0</td>
      <td>242.0</td>
      <td>17.8</td>
      <td>392.83</td>
      <td>4.03</td>
      <td>1583.1049</td>
      <td>34.7</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.03237</td>
      <td>0.0</td>
      <td>2.18</td>
      <td>0.0</td>
      <td>0.458</td>
      <td>6.998</td>
      <td>45.8</td>
      <td>6.0622</td>
      <td>3.0</td>
      <td>222.0</td>
      <td>18.7</td>
      <td>394.63</td>
      <td>2.94</td>
      <td>1160.2122</td>
      <td>33.4</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.06905</td>
      <td>0.0</td>
      <td>2.18</td>
      <td>0.0</td>
      <td>0.458</td>
      <td>7.147</td>
      <td>54.2</td>
      <td>6.0622</td>
      <td>3.0</td>
      <td>222.0</td>
      <td>18.7</td>
      <td>396.90</td>
      <td>5.33</td>
      <td>2115.4770</td>
      <td>36.2</td>
    </tr>
  </tbody>
</table>
</div>




```python
rm = np.asarray(df[["RM"]]).reshape(len(df[["RM"]]))
```


```python
high_rm = alldata[rm > np.percentile(rm, 67)]
med_rm = alldata[(rm > np.percentile(rm, 33)) & (rm <= np.percentile(rm, 67))]
low_rm = alldata[rm <= np.percentile(rm, 33)]
```

Create `build_interaction_rm`.


```python
def build_interaction_rm(varname, description):
    regression_h = LinearRegression()
    regression_m = LinearRegression()
    regression_l = LinearRegression()
    regression_h.fit(high_rm[varname].values.reshape(-1, 1), high_rm["target"])
    regression_m.fit(med_rm[varname].values.reshape(-1, 1), med_rm["target"])
    regression_l.fit(low_rm[varname].values.reshape(-1, 1), low_rm["target"])

    # Make predictions using the testing set
    pred_high = regression_h.predict(high_rm[varname].values.reshape(-1, 1))
    pred_med = regression_m.predict(med_rm[varname].values.reshape(-1, 1))
    pred_low = regression_l.predict(low_rm[varname].values.reshape(-1, 1))

    # The coefficients
    print(regression_h.coef_)
    print(regression_m.coef_)
    print(regression_l.coef_)

    # Plot outputs
    plt.figure(figsize=(12,7))
    plt.scatter(high_rm[varname], high_rm["target"],  color='blue', alpha = 0.3, label = "more rooms")
    plt.scatter(med_rm[varname], med_rm["target"],  color='red', alpha = 0.3, label = "medium rooms")
    plt.scatter(low_rm[varname], low_rm["target"],  color='orange', alpha = 0.3, label = "low amount of rooms")

    plt.plot(low_rm[varname], pred_low,  color='orange', linewidth=2)
    plt.plot(med_rm[varname], pred_med,  color='red', linewidth=2)
    plt.plot(high_rm[varname], pred_high,  color='blue', linewidth=2)
    plt.ylabel("house value")
    plt.xlabel(description)
    plt.legend()
```

Next, use build_interaction_rm with the three variables that came out with the highest effect on $R^2$


```python
# first plot
build_interaction_rm('RM', 'LSTAT')
```

    [13.97372526]
    [2.23033521]
    [2.3655758]



![png](index_files/index_28_1.png)



```python
# second plot
build_interaction_rm('RM', 'TAX')
```

    [13.97372526]
    [2.23033521]
    [2.3655758]



![png](index_files/index_29_1.png)



```python
# third plot
build_interaction_rm('RM', 'RAD')
```

    [13.97372526]
    [2.23033521]
    [2.3655758]



![png](index_files/index_30_1.png)


## Build a final model including all three interactions at once

Use 10-fold crossvalidation.


```python
# code here
final_df = df.copy()
for interaction_score in interaction_scores[:3]:
    c1 = final_df[interaction_score[0]]
    c2 = final_df[interaction_score[1]]
    col_name = "{}_{}".format(interaction_score[0], interaction_score[1])
    
    final_df[col_name] = c1*c2
final_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CRIM</th>
      <th>ZN</th>
      <th>INDUS</th>
      <th>CHAS</th>
      <th>NOX</th>
      <th>RM</th>
      <th>AGE</th>
      <th>DIS</th>
      <th>RAD</th>
      <th>TAX</th>
      <th>PTRATIO</th>
      <th>B</th>
      <th>LSTAT</th>
      <th>interaction</th>
      <th>RM_LSTAT</th>
      <th>RM_TAX</th>
      <th>RM_RAD</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.00632</td>
      <td>18.0</td>
      <td>2.31</td>
      <td>0.0</td>
      <td>0.538</td>
      <td>6.575</td>
      <td>65.2</td>
      <td>4.0900</td>
      <td>1.0</td>
      <td>296.0</td>
      <td>15.3</td>
      <td>396.90</td>
      <td>4.98</td>
      <td>1976.5620</td>
      <td>32.74350</td>
      <td>1946.200</td>
      <td>6.575</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.02731</td>
      <td>0.0</td>
      <td>7.07</td>
      <td>0.0</td>
      <td>0.469</td>
      <td>6.421</td>
      <td>78.9</td>
      <td>4.9671</td>
      <td>2.0</td>
      <td>242.0</td>
      <td>17.8</td>
      <td>396.90</td>
      <td>9.14</td>
      <td>3627.6660</td>
      <td>58.68794</td>
      <td>1553.882</td>
      <td>12.842</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.02729</td>
      <td>0.0</td>
      <td>7.07</td>
      <td>0.0</td>
      <td>0.469</td>
      <td>7.185</td>
      <td>61.1</td>
      <td>4.9671</td>
      <td>2.0</td>
      <td>242.0</td>
      <td>17.8</td>
      <td>392.83</td>
      <td>4.03</td>
      <td>1583.1049</td>
      <td>28.95555</td>
      <td>1738.770</td>
      <td>14.370</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.03237</td>
      <td>0.0</td>
      <td>2.18</td>
      <td>0.0</td>
      <td>0.458</td>
      <td>6.998</td>
      <td>45.8</td>
      <td>6.0622</td>
      <td>3.0</td>
      <td>222.0</td>
      <td>18.7</td>
      <td>394.63</td>
      <td>2.94</td>
      <td>1160.2122</td>
      <td>20.57412</td>
      <td>1553.556</td>
      <td>20.994</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.06905</td>
      <td>0.0</td>
      <td>2.18</td>
      <td>0.0</td>
      <td>0.458</td>
      <td>7.147</td>
      <td>54.2</td>
      <td>6.0622</td>
      <td>3.0</td>
      <td>222.0</td>
      <td>18.7</td>
      <td>396.90</td>
      <td>5.33</td>
      <td>2115.4770</td>
      <td>38.09351</td>
      <td>1586.634</td>
      <td>21.441</td>
    </tr>
  </tbody>
</table>
</div>




```python
## code here
crossvalidation = KFold(n_splits=10,shuffle=True, random_state=42)
final_score = np.mean(cross_val_score(regression, X=final_df.values, y=boston.target, scoring='r2', cv=crossvalidation))
final_score
```




    0.7833909250720847



Our $R^2$ has increased considerably! Let's have a look in statsmodels to see if all these interactions are significant.


```python
# code here
import statsmodels.api as sm
df_inter_sm = sm.add_constant(final_df)
model = sm.OLS(boston.target,df_inter_sm)
results = model.fit()

results.summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>            <td>y</td>        <th>  R-squared:         </th> <td>   0.815</td> 
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.809</td> 
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>   126.6</td> 
</tr>
<tr>
  <th>Date:</th>             <td>Sun, 18 Nov 2018</td> <th>  Prob (F-statistic):</th> <td>1.70e-166</td>
</tr>
<tr>
  <th>Time:</th>                 <td>11:57:55</td>     <th>  Log-Likelihood:    </th> <td> -1413.1</td> 
</tr>
<tr>
  <th>No. Observations:</th>      <td>   506</td>      <th>  AIC:               </th> <td>   2862.</td> 
</tr>
<tr>
  <th>Df Residuals:</th>          <td>   488</td>      <th>  BIC:               </th> <td>   2938.</td> 
</tr>
<tr>
  <th>Df Model:</th>              <td>    17</td>      <th>                     </th>     <td> </td>    
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>     <td> </td>    
</tr>
</table>
<table class="simpletable">
<tr>
       <td></td>          <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>const</th>       <td>  -16.7964</td> <td>    7.613</td> <td>   -2.206</td> <td> 0.028</td> <td>  -31.756</td> <td>   -1.837</td>
</tr>
<tr>
  <th>CRIM</th>        <td>   -0.1616</td> <td>    0.028</td> <td>   -5.737</td> <td> 0.000</td> <td>   -0.217</td> <td>   -0.106</td>
</tr>
<tr>
  <th>ZN</th>          <td>    0.0173</td> <td>    0.012</td> <td>    1.459</td> <td> 0.145</td> <td>   -0.006</td> <td>    0.041</td>
</tr>
<tr>
  <th>INDUS</th>       <td>    0.0947</td> <td>    0.053</td> <td>    1.793</td> <td> 0.074</td> <td>   -0.009</td> <td>    0.198</td>
</tr>
<tr>
  <th>CHAS</th>        <td>    2.6094</td> <td>    0.740</td> <td>    3.527</td> <td> 0.000</td> <td>    1.156</td> <td>    4.063</td>
</tr>
<tr>
  <th>NOX</th>         <td>  -13.4938</td> <td>    3.275</td> <td>   -4.120</td> <td> 0.000</td> <td>  -19.929</td> <td>   -7.059</td>
</tr>
<tr>
  <th>RM</th>          <td>   10.6938</td> <td>    0.992</td> <td>   10.782</td> <td> 0.000</td> <td>    8.745</td> <td>   12.643</td>
</tr>
<tr>
  <th>AGE</th>         <td>    0.0073</td> <td>    0.011</td> <td>    0.633</td> <td> 0.527</td> <td>   -0.015</td> <td>    0.030</td>
</tr>
<tr>
  <th>DIS</th>         <td>   -0.9516</td> <td>    0.175</td> <td>   -5.453</td> <td> 0.000</td> <td>   -1.294</td> <td>   -0.609</td>
</tr>
<tr>
  <th>RAD</th>         <td>    0.6607</td> <td>    0.478</td> <td>    1.383</td> <td> 0.167</td> <td>   -0.278</td> <td>    1.599</td>
</tr>
<tr>
  <th>TAX</th>         <td>    0.0344</td> <td>    0.025</td> <td>    1.401</td> <td> 0.162</td> <td>   -0.014</td> <td>    0.083</td>
</tr>
<tr>
  <th>PTRATIO</th>     <td>   -0.6962</td> <td>    0.113</td> <td>   -6.153</td> <td> 0.000</td> <td>   -0.919</td> <td>   -0.474</td>
</tr>
<tr>
  <th>B</th>           <td>    0.0127</td> <td>    0.007</td> <td>    1.795</td> <td> 0.073</td> <td>   -0.001</td> <td>    0.027</td>
</tr>
<tr>
  <th>LSTAT</th>       <td>    1.2734</td> <td>    0.253</td> <td>    5.041</td> <td> 0.000</td> <td>    0.777</td> <td>    1.770</td>
</tr>
<tr>
  <th>interaction</th> <td>   -0.0004</td> <td>    0.000</td> <td>   -1.165</td> <td> 0.245</td> <td>   -0.001</td> <td>    0.000</td>
</tr>
<tr>
  <th>RM_LSTAT</th>    <td>   -0.2904</td> <td>    0.041</td> <td>   -7.135</td> <td> 0.000</td> <td>   -0.370</td> <td>   -0.210</td>
</tr>
<tr>
  <th>RM_TAX</th>      <td>   -0.0074</td> <td>    0.004</td> <td>   -1.870</td> <td> 0.062</td> <td>   -0.015</td> <td>    0.000</td>
</tr>
<tr>
  <th>RM_RAD</th>      <td>   -0.0620</td> <td>    0.078</td> <td>   -0.792</td> <td> 0.428</td> <td>   -0.216</td> <td>    0.092</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>252.490</td> <th>  Durbin-Watson:     </th> <td>   1.083</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.000</td>  <th>  Jarque-Bera (JB):  </th> <td>2472.595</td>
</tr>
<tr>
  <th>Skew:</th>          <td> 1.945</td>  <th>  Prob(JB):          </th> <td>    0.00</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td>13.107</td>  <th>  Cond. No.          </th> <td>2.40e+05</td>
</tr>
</table><br/><br/>Warnings:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.<br/>[2] The condition number is large, 2.4e+05. This might indicate that there are<br/>strong multicollinearity or other numerical problems.



What is your conclusion here?


```python
# formulate your conclusion
```

## Summary

You now understand how to include interaction effects in your model!
