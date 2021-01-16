#!/bin/bash

python run.py analytics-zoo train/nyc_taxi_outliers_level_5.csv 2064 timestamp value 120
python run.py analytics-zoo train/nyc_taxi_outliers_level_4.csv 2064 timestamp value 120
python run.py analytics-zoo train/nyc_taxi_outliers_level_5.csv 2064 timestamp value 120
