#!/usr/bin/env python3

import sys
root_dir = f"{__file__.split('proof_wala')[0]}"
if root_dir not in sys.path:
    sys.path.append(root_dir)
import json
import os
import time
import uuid
from itp_interface.tools.training_data import TrainingData
from itp_interface.tools.log_utils import setup_logger
from proof_wala.llm_helpers.model import Model
from proof_wala.main.config import Experiment, ExperimentType

def train_experiment(experiment: Experiment, time_now: str = None):
    assert experiment.expertiment_type == ExperimentType.Training, f"Experiment type must be {ExperimentType.Training}, but is {experiment.expertiment_type}"
    # assert experiment.training_data_settings.training_data_dir is not None, f"Training data directory must be specified"
    time_now = time.strftime("%Y%m%d-%H%M%S") if time_now is None else time_now
    training_dataset_log_dir = os.path.join(experiment.training_data_settings.training_data_log_dir, time_now)
    os.makedirs(training_dataset_log_dir, exist_ok=True)
    eval_dataset_log_dir = os.path.join(experiment.training_data_settings.eval_data_log_dir, time_now) if experiment.training_data_settings.eval_data_log_dir is not None else None
    if eval_dataset_log_dir is not None:
        os.makedirs(eval_dataset_log_dir, exist_ok=True)
    test_dataset_log_dir = os.path.join(experiment.training_data_settings.test_data_log_dir, time_now) if experiment.training_data_settings.test_data_log_dir is not None else None
    if test_dataset_log_dir is not None:
        os.makedirs(test_dataset_log_dir, exist_ok=True)
    should_train = experiment.training_data_settings.training_data_dir is not None
    should_load_eval = experiment.training_data_settings.eval_data_dir is not None or experiment.training_data_settings.evals is not None
    should_load_test = experiment.training_data_settings.test_data_dir is not None
    should_split_train_eval = experiment.training_settings.train_eval_split and should_train
    if should_split_train_eval:
        assert not should_load_eval, f"Cannot split train and eval data if eval data is loaded from {experiment.training_data_settings.eval_data_dir}"
    # Load the training data
    gpuid_local = os.environ.get("LOCAL_RANK", None)
    gpuid_global = os.environ.get("RANK", None)
    gpuid_unique = str(uuid.uuid4())
    gpuid = f"{gpuid_local}-{gpuid_global}-{gpuid_unique}"
    hf_training_dataset = None
    training_dataset_type = experiment.training_data_settings.training_dataset_type.get_class()
    if should_train:
        training_data = TrainingData(experiment.training_data_settings.training_data_dir, experiment.training_data_settings.training_meta_filename, logger=setup_logger(f"TrainingData-{gpuid}", os.path.join(training_dataset_log_dir, f"training_data-{gpuid}.log")))
        with training_dataset_type(training_data, **experiment.training_data_settings.training_dataset_args) as training_dataset:
            hf_training_dataset = training_dataset.get_hf_dataset()
            if should_split_train_eval:
                hf_train_test_split = hf_training_dataset.train_test_split(test_size=experiment.training_settings.eval_percentage, seed=experiment.training_settings.training_args.data_seed)
                hf_training_dataset = hf_train_test_split["train"]
                hf_eval_dataset = hf_train_test_split["test"]
            else:
                hf_eval_dataset = None
            if experiment.training_settings.train_percentage < 1.0 and not should_split_train_eval:
                # Sample train_percentage of the dataset
                hf_training_dataset = hf_training_dataset.shuffle(seed=experiment.training_settings.training_args.data_seed).select(range(int(len(hf_training_dataset) * experiment.training_settings.train_percentage)))
    # Load the eval data
    if should_load_eval:
        if experiment.training_data_settings.evals is None:
            eval_data = TrainingData(experiment.training_data_settings.eval_data_dir, experiment.training_data_settings.eval_meta_filename, logger=setup_logger("EvalData-{gpuid}", os.path.join(eval_dataset_log_dir, f"eval_data-{gpuid}.log")))
            with training_dataset_type(eval_data, **experiment.training_data_settings.training_dataset_args) as eval_dataset:
                hf_eval_dataset = eval_dataset.get_hf_dataset()
        else:
            for eval_settings in experiment.training_data_settings.evals:
                os.makedirs(eval_settings.eval_data_log_dir, exist_ok=True) # Create the log directories
            eval_datasets = {
                eval_settings.eval_name: TrainingData(eval_settings.eval_data_dir, eval_settings.eval_meta_filename, logger=setup_logger(f"EvalData-{eval_settings.eval_name}-{gpuid}", os.path.join(eval_settings.eval_data_log_dir, f"eval_data-{eval_settings.eval_name}-{gpuid}.log")))
                for eval_settings in experiment.training_data_settings.evals
            }
            hf_eval_datasets = {}
            for eval_name, eval_data in eval_datasets.items():
                with training_dataset_type(eval_data, **experiment.training_data_settings.training_dataset_args) as eval_dataset:
                    hf_eval_datasets[eval_name] = eval_dataset.get_hf_dataset()
            hf_eval_dataset = hf_eval_datasets
    else:
        hf_eval_dataset = hf_eval_dataset if should_split_train_eval else None
    if hf_eval_dataset is not None:
        if experiment.training_settings.eval_percentage < 1.0 and not should_split_train_eval:
            # Sample eval_percentage of the dataset
            if isinstance(hf_eval_dataset, dict):
                for eval_name, eval_dataset in hf_eval_dataset.items():
                    hf_eval_dataset[eval_name] = eval_dataset.shuffle(seed=experiment.training_settings.training_args.data_seed).select(range(int(len(eval_dataset) * experiment.training_settings.eval_percentage)))
            else:
                hf_eval_dataset = hf_eval_dataset.shuffle(seed=experiment.training_settings.training_args.data_seed).select(range(int(len(hf_eval_dataset) * experiment.training_settings.eval_percentage)))
    # Load the test data
    if should_load_test:
        test_data = TrainingData(experiment.training_data_settings.test_data_dir, experiment.training_data_settings.test_meta_filename, logger=setup_logger("TestData-{gpuid}", os.path.join(test_dataset_log_dir, f"test_data-{gpuid}.log")))
        with training_dataset_type(test_data, **experiment.training_data_settings.training_dataset_args) as test_dataset:
            hf_test_dataset = test_dataset.get_hf_dataset()
    else:
        hf_test_dataset = None
    if hf_test_dataset is not None:
        if experiment.training_settings.test_percentage < 1.0:
            # Sample test_percentage of the dataset
            hf_test_dataset = hf_test_dataset.shuffle(seed=experiment.training_settings.training_args.data_seed).select(range(int(len(hf_test_dataset) * experiment.training_settings.test_percentage)))
    model_log_dir = os.path.join(experiment.model_settings.logging_dir, time_now)
    os.makedirs(model_log_dir, exist_ok=True)
    model_args = experiment.model_settings.model_args
    if "token" not in model_args or model_args["token"] is None:
        # try to get from .secrets/huggingface_token.json
        if os.path.exists(".secrets/huggingface_token.json"):
            with open(".secrets/huggingface_token.json", "r") as f:
                model_args["token"] = json.load(f)["token"]
    if "comet_experiment" not in model_args or model_args["comet_experiment"] is None:
        model_args["comet_experiment"] = f"{time_now}-{experiment.name}"
    # Train the model
    if experiment.training_settings.training_args.output_dir is None or experiment.training_settings.training_args.output_dir == "":
        experiment.training_settings.training_args.output_dir = os.path.join(model_log_dir, experiment.name)
    else:
        experiment.training_settings.training_args.output_dir = os.path.join(experiment.training_settings.training_args.output_dir, experiment.name)
    os.makedirs(experiment.training_settings.training_args.output_dir, exist_ok=True)
    model = Model(
        experiment.model_settings.name_or_path, 
        experiment.training_settings.training_args, 
        model_log_dir,
        **experiment.model_settings.model_args)
    training_data_formatter_callback = experiment.training_data_settings.training_data_formatter_type.get_class()
    with model:
        if hf_training_dataset is None:
            hf_training_dataset = hf_test_dataset
        if hf_eval_dataset is not None:
            eval_set = hf_eval_dataset
        else:
            eval_set = None
        model.train(
            time_now,
            training_data_formatter_callback(),
            hf_training_dataset, 
            eval_set)
        if hf_test_dataset is not None:
            model.training_args.do_train = False
            model.training_args.do_eval = True
            model.train(
                time_now,
                training_data_formatter_callback(),
                hf_training_dataset, 
                hf_test_dataset)