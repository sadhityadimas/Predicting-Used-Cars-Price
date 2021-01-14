# Predicting-Used-Cars-Price

## Predicting used cars price using machine learning model ensemble method Random Forest Regressor
* the dataset is taken from kaggle uploaded by Avi Kasliwal
* link to file https://www.kaggle.com/avikasliwal/used-cars-price-prediction?select=train-data.csv
* The data taken from various state in India
* Currency used is rupees (in LAKH or per 100.000 rupees)

## Library Used:
* Pandas
* Numpy
* Matplotlib
* Scikit Learn

## Features Used
* Year
* Kilometers_Driven
* Fuel_Type
* Transmission
* Owner_Type
* Mileage(kmpl)
* Engine(CC)
* Power(bhp)
* Seats

## Methods
### Data Cleaning
* Check and clean the NAN values; drop new_price columns as it containts more than 80% NAN, and drop 38 rows
* original rows 6019, after Drop NAN 5975 Rows. Only 0.6% loss
* Check and Change the Dtypes
* Performing String Parsing on Mileage, Engine and Power Features
* Taking only the Float value and put the respective information value in features attribute (e.g, Power in BHP)

### Preprocessing 
* Perfroming Train-Test split on training data
* Performing ONE HOT ENCODING on the categorical features
* There are 3 categorical columns, and all have cardinality lower than 10.

### Building and Training the Model
* ML model used : Random Forest Regressor (random state 0 and n_estimator 100)
* Checking model accuracy using MAE and Mean Squared Error approach
* MAE score 1.688 and MSE score 12.47

### Predicting the Test data 
* Doing the same method as train data and put the prediction result in CSV pair it with the cars name respectively

