#!/bin/bash

python run.py analytics-zoo train/nyc_taxi_outliers_level_5.csv 2064 timestamp value 60
python run.py azure_automl train/nyc_taxi_outliers_level_5.csv 2064 timestamp value 60
python run.py tpot_automl train/nyc_taxi_outliers_level_5.csv 2064 timestamp value 60
python run.py analytics-zoo train/nyc_taxi_outliers_level_4.csv 2064 timestamp value 60
python run.py azure_automl train/nyc_taxi_outliers_level_4.csv 2064 timestamp value 60
python run.py tpot_automl train/nyc_taxi_outliers_level_4.csv 2064 timestamp value 60
python run.py analytics-zoo train/nyc_taxi_outliers_level_3.csv 2064 timestamp value 60
python run.py azure_automl train/nyc_taxi_outliers_level_3.csv 2064 timestamp value 60
python run.py tpot_automl train/nyc_taxi_outliers_level_3.csv 2064 timestamp value 60
python run.py analytics-zoo train/nyc_taxi_outliers_level_2.csv 2064 timestamp value 60
python run.py azure_automl train/nyc_taxi_outliers_level_2.csv 2064 timestamp value 60
python run.py tpot_automl train/nyc_taxi_outliers_level_2.csv 2064 timestamp value 60
python run.py analytics-zoo train/nyc_taxi_outliers_level_1.csv 2064 timestamp value 60
python run.py azure_automl train/nyc_taxi_outliers_level_1.csv 2064 timestamp value 60
python run.py tpot_automl train/nyc_taxi_outliers_level_1.csv 2064 timestamp value 60
