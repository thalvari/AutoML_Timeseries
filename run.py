import argparse
import os
import warnings
from pathlib import Path
from shlex import split
from subprocess import PIPE, run
from sys import exit
from tempfile import mkstemp, mktemp

warnings.filterwarnings("ignore")


def run_cmd(automl_name, train_path, n_pred, dt, target, lang="python", verbose=False):
    log_dir = f"{os.environ['TMPDIR']}/logs"
    Path(log_dir).mkdir(exist_ok=True)
    log_path = mkstemp(prefix=f"{Path(train_path).stem}_{automl_name}_", suffix=".log",
                        dir=log_dir)[1]
    pred_path = mktemp(prefix=f"{Path(train_path).stem}_{automl_name}_", suffix=".csv",
                        dir="pred")
    if lang == "python":
        cmd = f"conda run -n {automl_name} bash -c 'python -u {automl_name}.py " \
              f"{train_path} {pred_path} {n_pred} {dt} {target} > {log_path} 2>&1'"
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
    args = parser.parse_args()

    automl_name = args.automl_name
    train_path = args.train_path
    n_pred = args.n_pred
    dt = args.dt
    target = args.target

    run_cmd(automl_name, train_path, n_pred, dt, target)
