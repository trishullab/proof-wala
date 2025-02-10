#!/usr/bin/env python3

import sys
root_dir = f"{__file__.split('proof_wala')[0]}"
if root_dir not in sys.path:
    sys.path.append(root_dir)
import logging
import os
os.environ["HYDRA_FULL_ERROR"] = "1"
import ray
import hydra
import time
from ray.runtime_env import RuntimeEnv
from proof_wala.main.config import ExperimentType, parse_config
from proof_wala.main.run_training import train_experiment
from proof_wala.main.run_token_count import count_tokens

@hydra.main(config_path="config", config_name="experiment", version_base="1.2")
def main(cfg):
    print("To run this experiment on multiple GPUs, use the following command:")
    print(f"torchrun --nproc-per-node <num-proc-per-node> {__file__}")
    print(f"Working directory: {os.getcwd()}")
    print(f"Config: {cfg}")
    experiment = parse_config(cfg)
    time_now = time.strftime("%Y%m%d-%H%M%S")
    if experiment.expertiment_type == ExperimentType.Training:
        train_experiment(experiment, time_now)
    elif experiment.expertiment_type == ExperimentType.TokenCount:
        os.environ["COMET_MODE"] = "DISABLED"
        count_tokens(experiment)
    else:
        raise Exception(f"Invalid experiment type: {experiment.expertiment_type}")
    pass

def run():
    # Start the ray cluster
    os.environ["PYTHONPATH"] = f"{root_dir}:{os.environ.get('PYTHONPATH', '')}"
    environ = os.environ.copy()
    runtime_env = RuntimeEnv(
        env_vars=environ
    )
    ray.init(
        num_cpus=10, 
        object_store_memory=50*2**30, 
        _memory=50*2**30, 
        logging_level=logging.ERROR, 
        ignore_reinit_error=False, 
        log_to_driver=False, 
        configure_logging=False,
        _system_config={"metrics_report_interval_ms": 10**8},
        runtime_env=runtime_env)
    main()

if __name__ == "__main__":
    run()