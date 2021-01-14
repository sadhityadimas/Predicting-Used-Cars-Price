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
### Original Train Data
|    | Name                             | Location   |   Year |   Kilometers_Driven | Fuel_Type   | Transmission   | Owner_Type   |   Mileage(kmpl) |   Engine(CC) |   Power(bhp) |   Seats |   Price |
|---:|:---------------------------------|:-----------|-------:|--------------------:|:------------|:---------------|:-------------|----------------:|-------------:|-------------:|--------:|--------:|
|  0 | Maruti Wagon R LXI CNG           | Mumbai     |   2010 |               72000 | CNG         | Manual         | First        |           26.6  |          998 |        58.16 |       5 |    1.75 |
|  1 | Hyundai Creta 1.6 CRDi SX Option | Pune       |   2015 |               41000 | Diesel      | Manual         | First        |           19.67 |         1582 |       126.2  |       5 |   12.5  |
|  2 | Honda Jazz V                     | Chennai    |   2011 |               46000 | Petrol      | Manual         | First        |           18.2  |         1199 |        88.7  |       5 |    4.5  |
|  3 | Maruti Ertiga VDI                | Chennai    |   2012 |               87000 | Diesel      | Manual         | First        |           20.77 |         1248 |        88.76 |       7 |    6    |
|  4 | Audi A4 New 2.0 TDI Multitronic  | Coimbatore |   2013 |               40670 | Diesel      | Automatic      | Second       |           15.2  |         1968 |       140.8  |       5 |   17.74 |

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

### Training Data After Cleaning and Preprocessing ready to be use to train the model
|    | Name                             | Location   |   Year |   Kilometers_Driven | Fuel_Type   | Transmission   | Owner_Type   |   Mileage(kmpl) |   Engine(CC) |   Power(bhp) |   Seats |   Price |
|---:|:---------------------------------|:-----------|-------:|--------------------:|:------------|:---------------|:-------------|----------------:|-------------:|-------------:|--------:|--------:|
|  0 | Maruti Wagon R LXI CNG           | Mumbai     |   2010 |               72000 | CNG         | Manual         | First        |           26.6  |          998 |        58.16 |       5 |    1.75 |
|  1 | Hyundai Creta 1.6 CRDi SX Option | Pune       |   2015 |               41000 | Diesel      | Manual         | First        |           19.67 |         1582 |       126.2  |       5 |   12.5  |
|  2 | Honda Jazz V                     | Chennai    |   2011 |               46000 | Petrol      | Manual         | First        |           18.2  |         1199 |        88.7  |       5 |    4.5  |
|  3 | Maruti Ertiga VDI                | Chennai    |   2012 |               87000 | Diesel      | Manual         | First        |           20.77 |         1248 |        88.76 |       7 |    6    |
|  4 | Audi A4 New 2.0 TDI Multitronic  | Coimbatore |   2013 |               40670 | Diesel      | Automatic      | Second       |           15.2  |         1968 |       140.8  |       5 |   17.74 |


### Building and Training the Model
* ML model used : Random Forest Regressor (random state 0 and n_estimator 100)
* Checking model accuracy using MAE and Mean Squared Error approach
* MAE score 1.688 and MSE score 12.47

### Predicting the Test data 
* Doing the same method as train data and put the prediction result in CSV pair it with the cars name respectively

### Results Preview
|    | Name                                      |   Price_Prediction |
|---:|:------------------------------------------|-------------------:|
|  0 | Maruti Alto K10 LXI CNG                   |             3.3695 |
|  1 | Maruti Alto 800 2016-2019 LXI             |             2.3234 |
|  2 | Toyota Innova Crysta Touring Sport 2.4 MT |            17.6951 |
|  3 | Hyundai i20 Magna                         |             4.5116 |
|  4 | Mahindra XUV500 W8 2WD                    |            12.9034 |

