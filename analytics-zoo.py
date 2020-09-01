import argparse
import multiprocessing as mp
import os
import random as rn
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from zoo import init_spark_on_local
from zoo.automl.config.recipe import BayesRecipe
from zoo.automl.regression.time_sequence_predictor import TimeSequencePredictor
from zoo.ray import RayContext


def main(train_path, pred_path, n_pred, dt, target, time_limit_min):
    os.environ["TRIALRUNNER_WALLTIME_LIMIT"] = str(time_limit_min*60)

    df_train = pd.read_csv(train_path)
    df_train[dt] = pd.to_datetime(df_train[dt])

    sc = init_spark_on_local(cores=mp.cpu_count(), spark_log_level="ERROR")
    ray_ctx = RayContext(sc=sc)
    ray_ctx.init()

    tsp = TimeSequencePredictor(dt_col=dt, target_col=target, future_seq_len=n_pred)
    pipeline = tsp.fit(df_train, resources_per_trial={"cpu": 4},
                       recipe=BayesRecipe(num_samples=10000, training_iteration=10000))

    df_pred = pipeline.predict(df_train)
    x_pred = pd.date_range(df_pred.iloc[-1][0], periods=n_pred, freq=pd.infer_freq(df_train[dt]))
    y_pred = df_pred.iloc[-1][1:]
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
