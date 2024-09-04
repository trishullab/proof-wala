#!/usr/bin/env python3

import os
import sys
root_dir = f"{__file__.split('proof_wala')[0]}"
if root_dir not in sys.path:
    sys.path.append(root_dir)
import json
import time
import typing
import ray
import logging
from torch.utils.data import Dataset, DataLoader
from itp_interface.tools.training_data import TrainingData
from itp_interface.tools.log_utils import setup_logger
from proof_wala.llm_helpers.model import Model, TrainingDataFormatterCallback
from proof_wala.main.config import Experiment, ExperimentType

@ray.remote
def tokenize_and_count(experiment: Experiment, prompts: typing.List[str], completions: typing.List[str]) -> int:
    token_count = 0
    model = Model(
        experiment.model_settings.name_or_path, 
        experiment.training_settings.training_args, 
        log_folder=None,
        **experiment.model_settings.model_args)
    tokenizer, tokenizer_args = model.get_tokenizer()
    for prompt, completion in zip(prompts, completions):
        full_text = f"{prompt}{completion}"
        tokenized_input = model.tokenize(tokenizer, [full_text], tokenizer_args)
        token_count += sum([len(x) for x in tokenized_input["input_ids"]])
    return token_count

def token_count(experiment: Experiment, dataset: Dataset, formatter_callback: TrainingDataFormatterCallback, logger: logging.Logger) -> int:
    root_dir = f"{__file__.split('proof_wala')[0]}"
    os.environ["PYTHONPATH"] = f"{root_dir}:{os.environ.get('PYTHONPATH', '')}"
    batch_size = experiment.training_settings.training_args.train_batch_size
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    all_prompts = []
    all_completions = []
    tokenization_results = []
    total_tokens = 0
    for batch_idx, dataset in enumerate(dataloader):
        prompts_and_completions = formatter_callback.get_prompt_and_completion(dataset)
        batch_prompts = [x[0] for x in prompts_and_completions]
        batch_completions = [x[1] for x in prompts_and_completions]
        all_prompts.extend(batch_prompts)
        all_completions.extend(batch_completions)
        tokenization_results.append(tokenize_and_count.remote(experiment, batch_prompts, batch_completions))
        if len(tokenization_results) >= 30:
            tokenization_results : typing.List[int] = ray.get(tokenization_results)
            total_tokens += sum(tokenization_results)
            tokenization_results = []
            logger.info(f"Processed {batch_idx * batch_size + len(batch_prompts)} examples")
            logger.info(f"Total tokens so far: {total_tokens}")
    tokenization_results : typing.List[int] = ray.get(tokenization_results)
    total_tokens += sum(tokenization_results)
    logger.info(f"Processed {batch_idx * batch_size + len(batch_prompts)} examples")
    logger.info(f"Total tokens so far: {total_tokens}")
    return total_tokens

def count_tokens(experiment: Experiment):
    assert experiment.expertiment_type == ExperimentType.TokenCount, f"Experiment type must be {ExperimentType.Training}, but is {experiment.expertiment_type}"
    assert experiment.training_data_settings.training_data_dir is not None, f"Training data directory must be specified"
    time_now = time.strftime("%Y%m%d-%H%M%S")
    training_dataset_log_dir = os.path.join(experiment.training_data_settings.training_data_log_dir, time_now)
    os.makedirs(training_dataset_log_dir, exist_ok=True)
    eval_dataset_log_dir = os.path.join(experiment.training_data_settings.eval_data_log_dir, time_now) if experiment.training_data_settings.eval_data_log_dir is not None else None
    if eval_dataset_log_dir is not None:
        os.makedirs(eval_dataset_log_dir, exist_ok=True)
    test_dataset_log_dir = os.path.join(experiment.training_data_settings.test_data_log_dir, time_now) if experiment.training_data_settings.test_data_log_dir is not None else None
    if test_dataset_log_dir is not None:
        os.makedirs(test_dataset_log_dir, exist_ok=True)
    should_load_eval = experiment.training_data_settings.eval_data_dir is not None
    should_load_test = experiment.training_data_settings.test_data_dir is not None
    should_split_train_eval = experiment.training_settings.train_eval_split
    if should_split_train_eval:
        assert not should_load_eval, f"Cannot split train and eval data if eval data is loaded from {experiment.training_data_settings.eval_data_dir}"
    # Load the training data
    training_data = TrainingData(experiment.training_data_settings.training_data_dir, experiment.training_data_settings.training_meta_filename, logger=setup_logger("TrainingData", os.path.join(training_dataset_log_dir, "training_data.log")))
    training_dataset_type = experiment.training_data_settings.training_dataset_type.get_class()
    with training_dataset_type(training_data, **experiment.training_data_settings.training_dataset_args) as training_dataset:
        training_data.logger.info(f"Loading training data from {experiment.training_data_settings.training_data_dir}")
        hf_training_dataset = training_dataset.get_hf_dataset()
        training_data.logger.info(f"Training data loaded with {len(hf_training_dataset)} examples")
        if should_split_train_eval:
            training_data.logger.info(f"Splitting training data into train and eval with eval percentage: {experiment.training_settings.eval_percentage}")
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
        eval_data = TrainingData(experiment.training_data_settings.eval_data_dir, experiment.training_data_settings.eval_meta_filename, logger=setup_logger("EvalData", os.path.join(eval_dataset_log_dir, "eval_data.log")))
        with training_dataset_type(eval_data, **experiment.training_data_settings.training_dataset_args) as eval_dataset:
            hf_eval_dataset = eval_dataset.get_hf_dataset()
    else:
        hf_eval_dataset = hf_eval_dataset if should_split_train_eval else None
    if hf_eval_dataset is not None:
        if experiment.training_settings.eval_percentage < 1.0 and not should_split_train_eval:
            # Sample eval_percentage of the dataset
            hf_eval_dataset = hf_eval_dataset.shuffle(seed=experiment.training_settings.training_args.data_seed).select(range(int(len(hf_eval_dataset) * experiment.training_settings.eval_percentage)))
    # Load the test data
    if should_load_test:
        test_data = TrainingData(experiment.training_data_settings.test_data_dir, experiment.training_data_settings.test_meta_filename, logger=setup_logger("TestData", os.path.join(test_dataset_log_dir, "test_data.log")))
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
    training_data.logger.info(f"Counting tokens in training dataset")
    training_data_formatter_callback = experiment.training_data_settings.training_data_formatter_type.get_class()
    total_tokens = token_count(experiment, hf_training_dataset, training_data_formatter_callback(), training_data.logger)
    training_data.logger.info(f"Total tokens in training dataset: {total_tokens}")