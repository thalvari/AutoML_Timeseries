import argparse
import multiprocessing as mp
import os
import random as rn
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from tpot import TPOTRegressor

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
    
    x_train = df_train[dt].values.reshape(-1, 1)
    y_train = df_train[target].values.reshape(-1, 1)

    tpot = TPOTRegressor(cv=TimeSeriesSplit(n_splits=5), n_jobs=-1, random_state=seed, verbosity=2)
    tpot.fit(x_train, y_train)

    x_pred = pd.date_range(df_train[dt].iloc[-1], periods=n_pred+1, freq=pd.infer_freq(df_train[dt]))[1:]
    y_pred = tpot.predict(x_pred.values.reshape(-1, 1))
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
