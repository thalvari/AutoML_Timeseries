#!/bin/bash

python run.py prophet train/temperature_outliers_level_0.csv 2223 datetime Montreal 360
python run.py prophet train/temperature_outliers_level_1.csv 2223 datetime Montreal 360
python run.py prophet train/temperature_outliers_level_2.csv 2223 datetime Montreal 360
python run.py prophet train/temperature_outliers_level_3.csv 2223 datetime Montreal 360
python run.py prophet train/temperature_outliers_level_4.csv 2223 datetime Montreal 360
python run.py prophet train/temperature_outliers_level_5.csv 2223 datetime Montreal 360
