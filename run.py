import argparse
import os
import warnings
from pathlib import Path
from shlex import split
from subprocess import PIPE, run
from sys import exit
from tempfile import mktemp

warnings.filterwarnings("ignore")


def run_cmd(automl_name, train_path, n_pred, dt, target, time_limit_min, lang="python", verbose=False):
    unique_id = Path(mktemp(prefix="")).stem
    log_path = f"logs/{Path(train_path).stem}_{automl_name}_{time_limit_min}_min_{unique_id}.log"
    pred_path = f"pred/{Path(train_path).stem}_{automl_name}_{time_limit_min}_min_{unique_id}.csv"

    if lang == "python":
        cmd = f"conda run -n {automl_name} bash -c 'python -u {automl_name}.py " \
              f"{train_path} {pred_path} {n_pred} {dt} {target} {time_limit_min} > {log_path} 2>&1'"
    if verbose:
        print(f"[cmd] {cmd}")
    
    print(f"[log] {log_path}")
    p = run(split(cmd), stderr=PIPE, text=True)
    if p.stderr.strip():
        exit(f"[crashed] {log_path}")
    else:
        print(f"[ok] {pred_path}")
    
    return pred_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("automl_name", type=str)
    parser.add_argument("train_path", type=str)
    parser.add_argument("n_pred", type=int)
    parser.add_argument("dt", type=str)
    parser.add_argument("target", type=str)
    parser.add_argument("time_limit_min", type=int)
    args = parser.parse_args()

    run_cmd(args.automl_name, args.train_path, args.n_pred, args.dt, args.target, args.time_limit_min)
