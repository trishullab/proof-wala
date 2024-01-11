#!/usr/bin/env python3

import sys
root_dir = f"{__file__.split('thrall_lib')[0]}"
if root_dir not in sys.path:
    sys.path.append(root_dir)
import os
import time
from itp_interface.tools.training_data import TrainingData
from itp_interface.tools.log_utils import setup_logger
from itp_interface.tools.ray_utils import RayUtils
from transformers import TrainingArguments, IntervalStrategy, SchedulerType
from thrall_lib.llm_helpers.model import Model
from thrall_lib.llm_helpers.theorem_proving_training_dataset import TheoremProvingTrainingDataset
from thrall_lib.itp.training_data_formatter import BasicTrainingDataFormatterCallback


def train(
    model_name: str,
    training_data_folder: str, 
    meta_filename: str,
    logging_folder: str,
    training_args: TrainingArguments,
    training_data_formatter_callback=BasicTrainingDataFormatterCallback(),
    eval_data_folder: str = None,
    train_percentage: float = 1.0,
    eval_percentage: float = 1.0,
    **kwargs):
    os.makedirs(logging_folder, exist_ok=True)
    code_logs = f"{logging_folder}/code_logs"
    model_logs = f"{logging_folder}/model_logs"
    os.makedirs(code_logs, exist_ok=True)
    os.makedirs(model_logs, exist_ok=True)
    model = Model(model_name, training_args, model_logs, **kwargs)
    with model:
        training_data = TrainingData(training_data_folder, meta_filename, logger=setup_logger("TrainingData", f"{code_logs}/training_data.log"))
        with TheoremProvingTrainingDataset(training_data) as training_dataset:
            dataset = training_dataset.get_hf_dataset()
            if train_percentage < 1.0:
                # Sample train_percentage of the dataset
                dataset = dataset.shuffle(seed=training_args.data_seed).select(range(int(len(dataset) * train_percentage)))
            # Split dataset into training and evaluation 10% evaluation
            if eval_data_folder is not None:
                hf_train_test_split = dataset.train_test_split(test_size=0.1, seed=training_args.seed)
                hf_training_dataset = hf_train_test_split["train"]
                hf_eval_dataset = hf_train_test_split["test"]
            else:
                hf_training_dataset = dataset
                hf_eval_dataset = None
        if eval_data_folder is not None:
            eval_data = TrainingData(eval_data_folder, meta_filename, logger=setup_logger("TestData", f"{code_logs}/test_data.log"))
            with TheoremProvingTrainingDataset(eval_data) as test_dataset:
                hf_eval_dataset = test_dataset.get_hf_dataset()
                if eval_percentage < 1.0:
                    # Sample test_percentage of the dataset
                    hf_eval_dataset = hf_eval_dataset.shuffle(seed=training_args.data_seed).select(range(int(len(hf_eval_dataset) * eval_percentage)))
        model.train(training_data_formatter_callback, hf_training_dataset, hf_eval_dataset)
        pass

if __name__ == "__main__":
    import json
    RayUtils.init_ray(num_of_cpus=10, object_store_memory_in_gb=100, memory_in_gb=50)
    model_name = "meta-llama/Llama-2-7b-hf"
    with open(".secrets/huggingface_token.json", "r") as f:
        hf_token = json.load(f)["token"]
    time_str = time.strftime("%Y%m%d-%H%M%S")
    seq_length = 3720
    new_model_name = "thrall-Llama-2-7b-hf-compcert-{seq_length}-001-001".format(seq_length=seq_length)
    test_folder = "/mnt/sdd1/amthakur/data/compcert/test/20240105-021512/"
    train_folder = "/mnt/sdd1/amthakur/data/compcert/train/20240105-021649/"
    train_percentage=1.0
    eval_percentage=0.1
    eval_steps = 500
    save_steps = 500
    num_train_epochs = 3
    train_batch_size_per_device = 2
    eval_batch_size_per_device = 2
    # seq_length = 1024
    # new_model_name = "thrall-Llama-2-7b-hf-toycoq-{seq_length}-001-001".format(seq_length=seq_length)
    # test_folder = ".log/train"
    # train_folder = ".log/train"
    # train_percentage=1.0
    # eval_percentage=0.1
    # eval_steps = 6
    # save_steps = 6
    # num_train_epochs = 6
    # train_batch_size_per_device = 10
    # eval_batch_size_per_device = 10
    train(
        model_name, 
        train_folder, 
        "local.meta.json", 
        f".log/run_training/{model_name}/logs/{time_str}",
        TrainingArguments(
            output_dir=f".log/run_training/{new_model_name}",
            overwrite_output_dir=False,
            do_train=True,
            do_eval=True,
            do_predict=False,
            evaluation_strategy=IntervalStrategy.STEPS,
            auto_find_batch_size=True,
            #per_device_train_batch_size=train_batch_size_per_device,
            per_device_eval_batch_size=eval_batch_size_per_device,
            gradient_accumulation_steps=1,
            learning_rate=2e-4,
            weight_decay=0.001,
            max_grad_norm=0.3, # Gradient clipping
            num_train_epochs=num_train_epochs,
            max_steps=-1,
            lr_scheduler_type=SchedulerType.COSINE,
            warmup_ratio=0.03,
            warmup_steps=0,
            logging_dir=f".log/run_training/{new_model_name}/training_logs/{time_str}",
            logging_strategy=IntervalStrategy.STEPS,
            logging_steps=1,
            logging_first_step=True,
            save_strategy=IntervalStrategy.STEPS,
            save_steps=save_steps,
            save_total_limit=3,
            # no_cuda=False,
            # use_cpu=False,
            seed=42,
            data_seed=42,
            bf16=False,
            fp16=False,
            eval_steps=eval_steps,
            load_best_model_at_end=True,
            metric_for_best_model="exact_match",
            greater_is_better=True,
            optim="paged_adamw_32bit", # Important for LoRA
            group_by_length=True, # helps in saving memory
            # resume_from_checkpoint=None,
            gradient_checkpointing=True
        ),
        eval_data_folder=test_folder,
        train_percentage=train_percentage,
        eval_percentage=eval_percentage,
        token=hf_token,
        max_seq_length=3720,
        comet_experiment=f"{time_str}-{new_model_name}")