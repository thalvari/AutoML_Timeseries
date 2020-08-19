import argparse
import logging
import multiprocessing as mp
import os
import random as rn
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from azureml.core.experiment import Experiment
from azureml.core.workspace import Workspace
from azureml.train.automl import AutoMLConfig

seed = 42
os.environ["PYTHONHASHSEED"] = str(seed)
rn.seed(seed)
np.random.seed(seed)


def main(args):
    train_path = args.train_path
    pred_path = args.pred_path
    n_pred = args.n_pred
    dt = args.dt
    target = args.target

    df_train = pd.read_csv(train_path)
    df_train[dt] = pd.to_datetime(df_train[dt])

    time_series_settings = {
        "time_column_name": dt,
        "max_horizon": "auto",
        "target_lags": "auto",
        "target_rolling_window_size": "auto",
        "featurization": "auto"
    }
    automl_config = AutoMLConfig(task="forecasting", training_data=df_train, label_column_name=target,
                                 max_cores_per_iteration=-1, enable_early_stopping=True,
                                 n_cross_validations=5, verbosity=logging.INFO, **time_series_settings)
    ws = Workspace.from_config()
    experiment = Experiment(ws, "experiment")
    best_run, fitted_model = experiment.submit(automl_config, show_output=True).get_output()
    print(fitted_model.steps)

    x_pred = pd.date_range(df_train[dt].iloc[-1], periods=n_pred+1, freq=pd.infer_freq(df_train[dt]))[1:]
    y_pred = fitted_model.forecast(pd.DataFrame({dt: x_pred}))[0]
    df_pred = pd.DataFrame({dt: x_pred, target: y_pred})
    df_pred.to_csv(pred_path, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("train_path")
    parser.add_argument("pred_path")
    parser.add_argument("n_pred", type=int)
    parser.add_argument("dt", type=str)
    parser.add_argument("target", type=str)
    args = parser.parse_args()
    main(args)
