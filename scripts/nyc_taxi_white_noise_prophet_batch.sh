#!/bin/bash

python run.py prophet train/nyc_taxi_white_noise_level_0.csv 2064 timestamp value 120
python run.py prophet train/nyc_taxi_white_noise_level_1.csv 2064 timestamp value 120
python run.py prophet train/nyc_taxi_white_noise_level_2.csv 2064 timestamp value 120
python run.py prophet train/nyc_taxi_white_noise_level_3.csv 2064 timestamp value 120
python run.py prophet train/nyc_taxi_white_noise_level_4.csv 2064 timestamp value 120
python run.py prophet train/nyc_taxi_white_noise_level_5.csv 2064 timestamp value 120
