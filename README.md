# Berkeley Haas - Professional Certificate in AIML - Practical Application 2 - What drives the price of a car?

## Background
This project amins to understand the factors that make a car more or less expensive.  This analysis done provides  recommendations to a used car dealership -- as to what consumers value in a used car. The dataset came from Kaggle and contains information of 426K used car sales data. 

In this project the work I have done is split into 3 parts.

### Part 1 - Preprocessing of the data
The raw data has vehicle sales information of 426K vehicles with 18 data points  for each vehicle sold:
The raw data set had the following features for the vehicles sold:
'year','manufacturer','model','condition','cylinders', 'fuel', 'odometer', 'title_status', 'transmission','VIN','drive','size','type','paint_color

During this data processing, I have worked to:

* Handle Missing Values
* Handle Duplicated Data
* Handle Outliers
* Feature Transformation
* Feature Encoding
* Handle Class Imbalance
* Feature Selection & Feature Extraction

#### Outlier Removal
To prevent skewness and ensure robust analysis, I removed the bottom 5% and top 5% of values in the price feature, effectively eliminating outliers.

After cleansing the data using the steps above, the cleaner dataset now has:
369516 vehicles sales data. 

### Part 2 Modeling training

Here in Part 2, I have transformed the features  we tested our data to train to machine learning model and evaluated it. 

### Part 3 -  Data Tranformation and Encoding
Here I am performing data preprocessing for machine learning pipeline using ColumnTransformer from sklearn. 

Standardized numerical features by scaling them to have zero mean and unit variance using StandardScaler().

Encoded categorical variables using OneHotEncoding, converting each category into a binary vector.

Fit the transformations to X_train (learns scaling parameters for numerical data and identifies unique categories for OneHotEncoding).

Used the same transformations learned from X_train and applied them to X_test (without refitting).

All numerical features are standardized.
All categorical features are one-hot encoded.
With this the data is now ready for machine learning models that require numerical input.

Applying StandardScaler and OneHotEncoding created a combination of Input features to form a complex set of categorical features. 

### Numerical Features
['year', 'odometer']

### Categorical Features

['manufacturer_acura' 'manufacturer_alfa-romeo'
 'manufacturer_aston-martin' 'manufacturer_audi' 'manufacturer_bmw'
 'manufacturer_buick' 'manufacturer_cadillac' 'manufacturer_chevrolet'
 'manufacturer_chrysler' 'manufacturer_datsun' 'manufacturer_dodge'
 'manufacturer_ferrari' 'manufacturer_fiat' 'manufacturer_ford'
 'manufacturer_gmc' 'manufacturer_harley-davidson' 'manufacturer_honda'
 'manufacturer_hyundai' 'manufacturer_infiniti' 'manufacturer_jaguar'
 'manufacturer_jeep' 'manufacturer_kia' 'manufacturer_land rover'
 'manufacturer_lexus' 'manufacturer_lincoln' 'manufacturer_mazda'
 'manufacturer_mercedes-benz' 'manufacturer_mercury' 'manufacturer_mini'
 'manufacturer_mitsubishi' 'manufacturer_morgan' 'manufacturer_nissan'
 'manufacturer_pontiac' 'manufacturer_porsche' 'manufacturer_ram'
 'manufacturer_rover' 'manufacturer_saturn' 'manufacturer_subaru'
 'manufacturer_tesla' 'manufacturer_toyota' 'manufacturer_unknown'
 'manufacturer_volkswagen' 'manufacturer_volvo' 'condition_excellent'
 'condition_fair' 'condition_good' 'condition_like new' 'condition_new'
 'condition_salvage' 'condition_unknown' 'cylinders_10 cylinders'
 'cylinders_12 cylinders' 'cylinders_3 cylinders' 'cylinders_4 cylinders'
 'cylinders_5 cylinders' 'cylinders_6 cylinders' 'cylinders_8 cylinders'
 'cylinders_other' 'fuel_diesel' 'fuel_electric' 'fuel_gas' 'fuel_hybrid'
 'fuel_other' 'title_status_clean' 'title_status_lien'
 'title_status_missing' 'title_status_parts only' 'title_status_rebuilt'
 'title_status_salvage' 'transmission_automatic' 'transmission_manual'
 'transmission_other' 'drive_4wd' 'drive_fwd' 'drive_rwd' 'drive_unknown'
 'type_SUV' 'type_bus' 'type_convertible' 'type_coupe' 'type_hatchback'
 'type_mini-van' 'type_offroad' 'type_other' 'type_pickup' 'type_sedan'
 'type_truck' 'type_unknown' 'type_van' 'type_wagon' 'paint_color_black'
 'paint_color_blue' 'paint_color_brown' 'paint_color_custom'
 'paint_color_green' 'paint_color_grey' 'paint_color_orange'
 'paint_color_purple' 'paint_color_red' 'paint_color_silver'
 'paint_color_unknown' 'paint_color_white' 'paint_color_yellow' 'state_ak'
 'state_al' 'state_ar' 'state_az' 'state_ca' 'state_co' 'state_ct'
 'state_dc' 'state_de' 'state_fl' 'state_ga' 'state_hi' 'state_ia'
 'state_id' 'state_il' 'state_in' 'state_ks' 'state_ky' 'state_la'
 'state_ma' 'state_md' 'state_me' 'state_mi' 'state_mn' 'state_mo'
 'state_ms' 'state_mt' 'state_nc' 'state_nd' 'state_ne' 'state_nh'
 'state_nj' 'state_nm' 'state_nv' 'state_ny' 'state_oh' 'state_ok'
 'state_or' 'state_pa' 'state_ri' 'state_sc' 'state_sd' 'state_tn'
 'state_tx' 'state_ut' 'state_va' 'state_vt' 'state_wa' 'state_wi'
 'state_wv' 'state_wy']

The total number of features after applying Scaling and OneHotEncodiing stands as 156. 

#### Creation and running performance of different models

 I created and tested the model performance for Ridge Regression.
 

### Part 3 Summary and Evaluation of Model performance

I used Ridge Regression, primarily because of its faster execution time, given very large dataset

Ridge Regression is a linear regression model that uses L2 regularization to reduce overfitting by minimizing the magnitude of model coefficients.

The objective of this evaluation was twofold:

Achieve a balance between Mean Squared Error (MSE), ensuring accurate predictions while minimizing overfitting.
Identify the coefficients of each feature, enabling the analysis of their impact on car prices.
By comparing the performance of these models, we can determine which one provides the most insightful and accurate predictions, ultimately informing data-driven decisions.

Model performance was assessed using:
Mean Squared Error (MSE): measures prediction accuracy
Score: Evaluates the model's ability to make accurate predictions.
Time/Performance: Measures the computational efficiency and speed of the model.

The result of the evaluation can be found in the following table:

|Model | Best Score	| Training Mean Squared Error (MSE) | Testing Mean Squared Error (MSE) | Elapsed Time   |
|------|------------|--------------------------|--------|--------|
|Ridge |Score: 0.46720	| 69895156.25	   | 69704668.40        |30 seconds      |


The following table summarizes the Positive and Negative influence of features on price of a Vehicle

NOTE: negative value reduces price, postive value increases price

#### Feature Coefficients

| Feature                          | Coefficient      |
|----------------------------------|------------------|
| Tesla                            | 10358.76        |
| Datsun                           | 9398.11         |
| Porsche                          | 6442.74         |
| Transmission (Other)             | 5340.24         |
| Alfa-Romeo                       | 5176.76         |
| Cylinders (12 Cylinders)         | 5101.62         |
| Cylinders (5 Cylinders)          | -4062.89        |
| Mercury                          | -4168.76        |
| Mitsubishi                       | -4259.24        |
| Harley-Davidson                  | -4796.26        |
| Condition (Fair)                 | -5173.44        |
| Fiat                             | -5862.20        |
| Saturn                           | -5987.10        |
 
### Evaluation
#### Telsa, Datsun, Porsche and Alfa-romeo manufacturer brands command higher selling prices
#### Saturn, Fiat, Harley Davidson, Mitsubishi and Mecury manfacturer brands 
#### Vehicles with 12 Cylinders seem to command higher selling price.

<h2>Coefficients Insights</h2>
<p>Coefficients represent the change in the predicted outcome for a one-unit change in the feature, while holding all other features constant. They indicate:</p>
<p>
    <strong>Coefficients:</strong> How much do individual features impact the prediction?
</p>
<div style="text-align:center;">
    <img src="data/coefficients.png" />
</div>

<ul>
    <li><strong>Luxury brands</strong>: Positive coefficients for brands like Tesla, Porsche and Alfa-romeo command higher prices than other luxery cars.</li>
    <li><strong>Datsun</strong>: High positive coefficient suggest elevated demand and prices, presenting lucrative opportunities.</li>
    <li><strong>Saturn</strong>: Negative coefficient indicate weaker demand and lesser price impacting revenue stream.</li>
</ul>


<h3>Common Benefits</h3>

<ul>
    <li>Data-driven pricing: Utilise coefficients to inform pricing strategies.</li>
    <li>Inventory optimisation: Stock vehicles with in-demand features.</li>
    <li>Targeted marketing: Focus on specific customer segments.</li>
</ul>


<p>Car dealerships can refine their business strategies, enhance customer satisfaction, and boost profitability by  leveraging these insights</p>

<h2>Conclusion</h2>
<p>
    This analysis offers crucial insights that help used car dealers enhance their pricing strategies and manage inventory more effectively. By identifying how key features influence car prices, dealers can make data-driven decisions to improve their business operations.
</p>
</div>
</div>


## Link to Notebook:
[Explore Coupon Acceptance Factors by Customers](https://github.com/nbajam/BH-PCAIML-MOD11-PAA2/blob/main/bh_pcaiml_mod11_prac_assign.ipynb)

## Data Used:
[In-Vehicle Coupon Recommendation](https://archive.ics.uci.edu/dataset/603/in+vehicle+coupon+recommendation)

