# Predicting-Used-Cars-Price

## Predicting used cars price using machine learning model ensemble method Random Forest Regressor
* the dataset is taken from kaggle uploaded by Avi Kasliwal
* link to file https://www.kaggle.com/avikasliwal/used-cars-price-prediction?select=train-data.csv
* The data was taken from various state in India
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

* 6019 rows × 13 columns

### Data Cleaning
* Check and clean the NAN values; drop new_price columns as it containts more than 80% NAN, and drop 38 rows
* original rows 6019, after Drop NAN 5975 Rows. Only 0.6% loss
* Check and Change the Dtypes
* Performing String Parsing on Mileage, Engine and Power Features
* Taking only the Float value and put the respective information value in features attribute (e.g, Power in BHP)


### Preprocessing 
* Performing Initial data analysis on the training dataset
* Perfroming Train-Test split on training data
* Performing ONE HOT ENCODING on the categorical features
* There are 3 categorical columns, and all have cardinality lower than 10.

## Initial Data Analysis

### Percentage of Automatic Cars VS Manual Car in the dataset

![](/images/manualvstransmissionpie.png)

* More Manual Cars than Automatic

### Manual VS Automatic Car's Owner Comparison

![](/images/manualvstransmissionownertype.png)

* in both categories, most cars are owned by the first owner

### Cars Year Distribution

![](/images/yeardistribution.png)

* Year distribution are skewed to the left, more cars are from around 2012-2017

### Km Driven Distribution and Boxplot

![](/images/kilometerdistribution.png)

![](/images/kilometerdboxplot.png)

* This doesn't seems right, it means there are outliers that will impact our model

* Dropping Outliers by taking only Km driven Below 150.000

![](/images/kilometerdistribution4.png)

* Looks good and 5872-5707 = 165 rows are drop, This should improve the model. comparison score later

### Power, Engine, and Mileage Distribution

![](/images/3hist.png)

* There are outliers on these Feature as well, but not as impactful as in KM Driven


## Model Building


### Training Data After Cleaning and Preprocessing ready to be use to train the model

|    |   Year |   Kilometers_Driven |   Mileage(kmpl) |   Engine(CC) |   Power(bhp) |   Seats |   0 |   1 |   2 |   3 |   4 |   5 |   6 |   7 |   8 |   9 |
|---:|-------:|--------------------:|----------------:|-------------:|-------------:|--------:|----:|----:|----:|----:|----:|----:|----:|----:|----:|----:|
|  0 |   2016 |               53886 |           25.17 |          799 |        53.3  |       5 |   0 |   0 |   0 |   1 |   0 |   1 |   1 |   0 |   0 |   0 |
|  1 |   2017 |              100400 |           25    |         1396 |        69    |       5 |   0 |   1 |   0 |   0 |   0 |   1 |   1 |   0 |   0 |   0 |
|  2 |   2017 |                9500 |           14.62 |         1999 |       149.92 |       5 |   0 |   0 |   0 |   1 |   1 |   0 |   1 |   0 |   0 |   0 |
|  3 |   2017 |               10000 |           19.27 |         2143 |       167.62 |       5 |   0 |   1 |   0 |   0 |   1 |   0 |   1 |   0 |   0 |   0 |
|  4 |   2016 |               31000 |           17.05 |         1995 |       190    |       5 |   0 |   1 |   0 |   0 |   1 |   0 |   1 |   0 |   0 |   0 |



### Building and Training the Model 1
* ML model used : Random Forest Regressor (random state 0 and n_estimator 100)
* Checking model accuracy using MAE and Mean Squared Error approach
* MAE score 1.688 and MSE score 12.47

### Predicting the Test data 1
* Doing the same method as train data and put the prediction result in CSV pair it with the cars name respectively

### Results Preview 1

|    | Name                                      |   Price_Prediction |
|---:|:------------------------------------------|-------------------:|
|  0 | Maruti Alto K10 LXI CNG                   |             3.3695 |
|  1 | Maruti Alto 800 2016-2019 LXI             |             2.3234 |
|  2 | Toyota Innova Crysta Touring Sport 2.4 MT |            17.6951 |
|  3 | Hyundai i20 Magna                         |             4.5116 |
|  4 | Mahindra XUV500 W8 2WD                    |            12.9034 |


### Building and Training the Model 2
* Removing Outliers in KM Driven
* ML model used : Random Forest Regressor (random state 0 and n_estimator 100)
* Checking model accuracy using MAE and Mean Squared Error approach
* MAE score 1.6024 and MSE score 9.799

### Predicting the Test data 2
* Doing the same method as train data and put the prediction result in CSV pair it with the cars name respectively

### Results Preview 2

|    | Name                                      |   Price_Prediction |
|---:|:------------------------------------------|-------------------:|
|  0 | Maruti Alto K10 LXI CNG                   |             3.2125 |
|  1 | Maruti Alto 800 2016-2019 LXI             |             2.3972 |
|  2 | Toyota Innova Crysta Touring Sport 2.4 MT |            17.1875 |
|  3 | Hyundai i20 Magna                         |             4.581  |
|  4 | Mahindra XUV500 W8 2WD                    |            12.0833 |

