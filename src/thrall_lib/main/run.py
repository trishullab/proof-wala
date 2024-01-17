#!/usr/bin/env python3

import sys
root_dir = f"{__file__.split('thrall_lib')[0]}"
if root_dir not in sys.path:
    sys.path.append(root_dir)
import logging
import os
os.environ["HYDRA_FULL_ERROR"] = "1"
import ray
import hydra
from thrall_lib.main.config import ExperimentType, parse_config
from thrall_lib.main.run_training import train_experiment

@hydra.main(config_path="config", config_name="experiment", version_base="1.2")
def main(cfg):
    print("To run this experiment on multiple GPUs, use the following command:")
    print(f"Working directory: {os.getcwd()}")
    print(f"Config: {cfg}")
    experiment = parse_config(cfg)
    print(f"torchrun --nproc-per-node <num-proc-per-node> {__file__}")
    if experiment.expertiment_type == ExperimentType.Training:
        train_experiment(experiment)
    else:
        raise Exception(f"Invalid experiment type: {experiment.expertiment_type}")
    pass

if __name__ == "__main__":
    # Start the ray cluster
    ray.init(
        num_cpus=10, 
        object_store_memory=50*2**30, 
        _memory=50*2**30, 
        logging_level=logging.ERROR, 
        ignore_reinit_error=False, 
        log_to_driver=False, 
        configure_logging=False,
        _system_config={"metrics_report_interval_ms": 10**8})
    main()