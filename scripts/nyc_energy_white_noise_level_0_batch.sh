#!/bin/bash

python run2.py analytic_zoo train/nyc_energy_white_noise_level_0.csv 200 timeStamp demand
python run2.py azure_automl train/nyc_energy_white_noise_level_0.csv 200 timeStamp demand
python run2.py tpot_automl train/nyc_energy_white_noise_level_0.csv 200 timeStamp demand
python run2.py analytic_zoo train/nyc_energy_white_noise_level_1.csv 200 timeStamp demand
python run2.py azure_automl train/nyc_energy_white_noise_level_1.csv 200 timeStamp demand
python run2.py tpot_automl train/nyc_energy_white_noise_level_1.csv 200 timeStamp demand
python run2.py analytic_zoo train/nyc_energy_white_noise_level_2.csv 200 timeStamp demand
python run2.py azure_automl train/nyc_energy_white_noise_level_3.csv 200 timeStamp demand
python run2.py tpot_automl train/nyc_energy_white_noise_level_2.csv 200 timeStamp demand
python run2.py analytic_zoo train/nyc_energy_white_noise_level_3.csv 200 timeStamp demand
python run2.py azure_automl train/nyc_energy_white_noise_level_3.csv 200 timeStamp demand
python run2.py tpot_automl train/nyc_energy_white_noise_level_3.csv 200 timeStamp demand
python run2.py analytic_zoo train/nyc_energy_white_noise_level_4.csv 200 timeStamp demand
python run2.py azure_automl train/nyc_energy_white_noise_level_4.csv 200 timeStamp demand
python run2.py tpot_automl train/nyc_energy_white_noise_level_4.csv 200 timeStamp demand
python run2.py analytic_zoo train/nyc_energy_white_noise_level_5.csv 200 timeStamp demand
python run2.py azure_automl train/nyc_energy_white_noise_level_5.csv 200 timeStamp demand
python run2.py tpot_automl train/nyc_energy_white_noise_level_5.csv 200 timeStamp demand
