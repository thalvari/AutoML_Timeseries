import argparse
import multiprocessing as mp
import os
import random as rn
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import optuna
import pandas as pd
from fbprophet import Prophet
from fbprophet.diagnostics import cross_validation, performance_metrics


def prophet_objective(trial, df, n_pred):
    changepoint_prior_scale = trial.suggest_float("changepoint_prior_scale", 0.001, 0.5, log=True)
    seasonality_prior_scale = trial.suggest_float("seasonality_prior_scale", 0.01, 10, log=True)
    holidays_prior_scale = trial.suggest_float("holidays_prior_scale", 0.01, 10, log=True)
    seasonality_mode = trial.suggest_categorical("seasonality_mode", ["additive", "multiplicative"])
#     changepoint_range = trial.suggest_float("changepoint_range", 0.5, 0.95)
    m = Prophet(changepoint_prior_scale=changepoint_prior_scale,
                seasonality_prior_scale=seasonality_prior_scale,
                holidays_prior_scale=holidays_prior_scale, seasonality_mode=seasonality_mode)
#                 holidays_prior_scale=holidays_prior_scale, seasonality_mode=seasonality_mode,
#                 changepoint_range=changepoint_range)
#     for col in df.columns:
#         if col not in ["ds", "y"]:
#             m.add_regressor(col)
    m.fit(df)
    freq = pd.infer_freq(df["ds"])
#     df_cv = cross_validation(m, horizon=pd.Timedelta(n_pred*freq), period=pd.Timedelta(freq),
#                              initial=pd.Timedelta((len(df)-n_pred-5)*freq), parallel="processes")
    df_cv = cross_validation(m, horizon=pd.Timedelta(n_pred*freq), period=pd.Timedelta(n_pred*freq),
                             initial=pd.Timedelta((n_pred-1)*freq), parallel="processes")
    df_p = performance_metrics(df_cv, rolling_window=1)
    mape = np.mean(df_p.mape)
    return mape


def main(train_path, pred_path, n_pred, dt, target, time_limit_min):
    df_train = pd.read_csv(train_path)
    df_train["ds"] = pd.to_datetime(df_train.pop(dt))
    df_train["y"] = df_train.pop(target)

    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: prophet_objective(trial, df_train, n_pred), timeout=60*time_limit_min)
    print(study.best_params)

    d_pred = {"ds": pd.date_range(df_train["ds"].iloc[-1], periods=n_pred+1, freq=pd.infer_freq(df_train["ds"]))[1:]}
#     for col in df_train.columns:
#         if col not in ["ds", "y"]:
#             m = Prophet()
#             m.fit(pd.DataFrame({"ds": df_train["ds"], "y": df_train[col]}))
#             d_pred[col] = m.predict(pd.DataFrame({"ds": d_pred["ds"]}))["yhat"]

    m = Prophet(**study.best_params)
#     for col in df_train.columns:
#         if col not in ["ds", "y"]:
#             m.add_regressor(col)
    m.fit(df_train)

    df_pred = m.predict(pd.DataFrame(d_pred))[["ds", "yhat"]]
    df_pred = df_pred.rename(columns={"ds": dt, "yhat": target})
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
