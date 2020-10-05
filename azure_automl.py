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


def main(train_path, pred_path, n_pred, dt, target, time_limit_min):
    df_train = pd.read_csv(train_path)
    df_train[dt] = pd.to_datetime(df_train[dt])

    time_series_settings = {
        "time_column_name": dt,
        "max_horizon": n_pred,
        "target_lags": "auto",
        "target_rolling_window_size": "auto"
    }
    automl_config = AutoMLConfig(task="forecasting", training_data=df_train, label_column_name=target,
                                 n_cross_validations=5, max_cores_per_iteration=-1, path=os.environ["SCRATCH"],
                                 experiment_timeout_minutes=time_limit_min, ensemble_download_models_timeout_sec=3600,
                                 **time_series_settings)
    ws = Workspace.from_config()
    experiment = Experiment(ws, "experiment")
    best_run, fitted_model = experiment.submit(automl_config, show_output=True).get_output()

    print("Best pipeline:")
    try:
        ensemble = vars(fitted_model.steps[1][1])["_wrappedEnsemble"]
        print(ensemble.__class__)
        steps = ensemble.estimators_
    except:
        steps = fitted_model.steps
    best_pipeline = ""
    for i, step in enumerate(steps):
        best_pipeline += f"{i}. {str(step)}\n"
    print(best_pipeline)

    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_colwidth', -1)
    print(fitted_model.named_steps["timeseriestransformer"].get_engineered_feature_names())
    featurization_summary = fitted_model.named_steps["timeseriestransformer"].get_featurization_summary()
    print(pd.DataFrame.from_records(featurization_summary))

    x_pred = pd.date_range(df_train[dt].iloc[-1], periods=n_pred+1, freq=pd.infer_freq(df_train[dt]))[1:]
    y_pred = fitted_model.forecast(forecast_destination=x_pred[-1])[0]
#     y_pred = fitted_model.forecast(pd.DataFrame({dt: x_pred}))[0]

    df_pred = pd.DataFrame({dt: x_pred, target: y_pred})
    df_pred.to_csv(pred_path, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("train_path", type=str)
    parser.add_argument("pred_path", type=str)
    parser.add_argument("n_pred", type=int)
    parser.add_argument("dt", type=str)
    parser.add_argument("target", type=str)
    parser.add_argument("time_limit_min", type=int)
    args = parser.parse_args()
    
    main(args.train_path, args.pred_path, args.n_pred, args.dt, args.target, args.time_limit_min)
