#!/bin/bash

python run.py analytics-zoo train/nyc_energy_white_noise_level_1.csv 200 timeStamp demand
python run.py azure_automl train/nyc_energy_white_noise_level_1.csv 200 timeStamp demand
python run.py tpot_automl train/nyc_energy_white_noise_level_1.csv 200 timeStamp demand
python run.py tpot_automl train/nyc_energy_white_noise_level_1.csv 200 timeStamp demand
