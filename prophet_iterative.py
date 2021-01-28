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


def main(train_path, pred_path, n_pred, dt, target, time_limit_min):
    df_train = pd.read_csv(train_path)
    df_train["ds"] = pd.to_datetime(df_train.pop(dt))
    df_train["y"] = df_train.pop(target)

    d_pred = {"ds": pd.date_range(df_train["ds"].iloc[-1], periods=n_pred+1, freq=pd.infer_freq(df_train["ds"]))[1:]}
    for col in df_train.columns:
        if col not in ["ds", "y"]:
            m = Prophet()
            m.fit(pd.DataFrame({"ds": df_train["ds"], "y": df_train[col]}))
            d_pred[col] = m.predict(pd.DataFrame({"ds": d_pred["ds"]}))["yhat"]

    m = Prophet()
    for col in df_train.columns:
        if col not in ["ds", "y"]:
            m.add_regressor(col)
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
