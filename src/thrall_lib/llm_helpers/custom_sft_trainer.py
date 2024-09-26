#!/usr/bin/env python3

import torch
import torch.nn as nn
import typing
from logging import Logger
from torch.utils.data import DataLoader, Dataset
from typing import List, Union, Optional, Callable, Dict, Tuple
from transformers import PreTrainedModel, TrainingArguments, PreTrainedTokenizerBase, TrainerCallback, DataCollator
from transformers.trainer_utils import EvalLoopOutput, EvalPrediction, denumpify_detensorize, find_executable_batch_size
from trl import SFTTrainer
from transformers.trainer_callback import TrainerState
from transformers.trainer import logger as hf_trainer_logger
from transformers.integrations.deepspeed import deepspeed_init
from transformers import DataCollator

class GenerateEvalSFTTrainer(SFTTrainer):
    def __init__(
        self,
        model: Union[PreTrainedModel, nn.Module, str] = None,
        args: TrainingArguments = None,
        data_collator: Optional[DataCollator] = None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        model_init: Optional[Callable[[], PreTrainedModel]] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
        preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        peft_config: Optional["PeftConfig"] = None,
        dataset_text_field: Optional[str] = None,
        packing: Optional[bool] = False,
        formatting_func: Optional[Callable] = None,
        max_seq_length: Optional[int] = None,
        infinite: Optional[bool] = None,
        num_of_sequences: Optional[int] = 1024,
        chars_per_token: Optional[float] = 3.6,
        dataset_num_proc: Optional[int] = None,
        dataset_batch_size: int = 1000,
        neftune_noise_alpha: Optional[float] = None,
        model_init_kwargs: Optional[Dict] = None,
        dataset_kwargs: Optional[Dict] = None,
        generative_compute_metrics: Optional[Callable[[str, TrainerState, List[str], List[str], List[List[str]]], Dict]] = None,
        generate_callback: Optional[Callable[[Dataset], Tuple[List[str], List[str], List[List[str]]]]] = None,
        logger: Logger = None,
    ):
        assert (generative_compute_metrics is None) == (generate_callback is None), "Both generative_compute_metrics and generate_callback must be defined or not defined"
        args.prediction_loss_only = True
        args.remove_unused_columns = False
        args.include_inputs_for_metrics = True
        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            model_init=model_init,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
            peft_config=peft_config,
            dataset_text_field=dataset_text_field,
            packing=packing,
            formatting_func=formatting_func,
            max_seq_length=max_seq_length,
            infinite=infinite,
            num_of_sequences=num_of_sequences,
            chars_per_token=chars_per_token,
            dataset_num_proc=dataset_num_proc,
            dataset_batch_size=dataset_batch_size,
            neftune_noise_alpha=neftune_noise_alpha,
            model_init_kwargs=model_init_kwargs,
            dataset_kwargs=dataset_kwargs
        )
        self.original_training_dataset = train_dataset
        self.original_eval_dataset = eval_dataset
        self.generative_compute_metrics = generative_compute_metrics
        self.generate_callback = generate_callback
        self.generate_batch_size = self.args.eval_batch_size
        self.logger = logger if logger is not None else hf_trainer_logger
        self._agg_metrics = {}
        self._num_evals = 1 if type(eval_dataset) is not dict else len(eval_dataset)
        self._metric_agg_idx = 0
        pass

    def get_eval_dataloader(self, eval_dataset: typing.Union[Dataset, None] = None) -> DataLoader:
        return None

    def evaluation_loop(self, 
            dataloader: DataLoader, 
            description: str, 
            prediction_loss_only: typing.Union[bool, None] = None, 
            ignore_keys: typing.Union[List[str], None] = None, 
            metric_key_prefix: str = "eval") -> EvalLoopOutput:
        """
        Prediction/evaluation loop, shared by `Trainer.evaluate()` and `Trainer.predict()`.

        Works both with or without labels.
        """
        args = self.args

        # NEW: no logits are returned, so we need to change the code here
        prediction_loss_only = prediction_loss_only if prediction_loss_only is not None else args.prediction_loss_only

        # if eval is called w/o train, handle model prep here
        if self.is_deepspeed_enabled and self.deepspeed is None:
            _, _ = deepspeed_init(self, num_training_steps=0, inference=True)

        model = self._wrap_model(self.model, training=False, dataloader=dataloader)

        if len(self.accelerator._models) == 0 and model is self.model:
            model = (
                self.accelerator.prepare(model)
                if self.is_deepspeed_enabled
                else self.accelerator.prepare_model(model, evaluation_mode=True)
            )

            if self.is_fsdp_enabled:
                self.model = model

            # for the rest of this function `model` is the outside model, whether it was wrapped or not
            if model is not self.model:
                self.model_wrapped = model

            # backward compatibility
            if self.is_deepspeed_enabled:
                self.deepspeed = self.model_wrapped

        # if full fp16 or bf16 eval is wanted and this ``evaluation`` or ``predict`` isn't called
        # while ``train`` is running, cast it to the right dtype first and then put on device
        if not self.is_in_train:
            if args.fp16_full_eval:
                model = model.to(dtype=torch.float16, device=args.device)
            elif args.bf16_full_eval:
                model = model.to(dtype=torch.bfloat16, device=args.device)

        model.eval()

        self.callback_handler.eval_dataloader = dataloader
        # Do this before wrapping.
        if args.past_index >= 0:
            self._past = None


        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of the evaluation loop
            delattr(self, "_past")

        all_preds : List[List[str]] = []
        all_labels : List[str] = []
        all_inputs : List[str] = []
        if self.generate_callback is not None:
            if isinstance(self.original_eval_dataset, dict):
                assert metric_key_prefix.startswith("eval_")
                dataset_name = metric_key_prefix[len("eval_"):]
                original_eval_dataset = self.original_eval_dataset[dataset_name]
            else:
                assert metric_key_prefix == "eval"
                dataset_name = "eval"
                original_eval_dataset = self.original_eval_dataset
            _eval_cnt = 0
            def _eval_callback(batch_size: int):
                self.logger.info(f"***** Running {description} *****")
                self.logger.info(f"  Batch size = {batch_size}")
                nonlocal _eval_cnt
                current_eval_cnt = 0
                dataloader_params = {
                    "batch_size": batch_size,
                    "num_workers": self.args.dataloader_num_workers,
                    "pin_memory": self.args.dataloader_pin_memory,
                }
                if not isinstance(original_eval_dataset, torch.utils.data.IterableDataset):
                    dataloader_params["drop_last"] = self.args.dataloader_drop_last
                new_dataloader = DataLoader(original_eval_dataset, **dataloader_params)
                for step, inputs in enumerate(new_dataloader):
                    if current_eval_cnt + len(inputs) < _eval_cnt:
                        current_eval_cnt += len(inputs)
                        continue
                    try:
                        prompts, completions, generated = self.generate_callback(inputs)
                        all_inputs.extend(prompts)
                        all_labels.extend(completions)
                        all_preds.extend(generated)
                        _eval_cnt += len(prompts)
                        current_eval_cnt += len(inputs)
                        self.logger.info(f"Evaluated: {_eval_cnt} / {len(original_eval_dataset)}")
                    except Exception as e:
                        self.logger.error(f"Error evaluating: {e}")
                        self.logger.error(f"Skipping: {_eval_cnt} / {len(original_eval_dataset)}")
                return batch_size
            generation_evaluation = find_executable_batch_size(_eval_callback, starting_batch_size=self.generate_batch_size, auto_find_batch_size=self.args.auto_find_batch_size)
            self.generate_batch_size = generation_evaluation()

        # Metrics!
        if self.generative_compute_metrics is not None:
            assert self.generate_callback is not None
            if args.include_inputs_for_metrics:
                metrics = self.generative_compute_metrics(dataset_name, self.state, all_inputs, all_labels, all_preds)
        else:
            metrics = {}

        # To be JSON-serializable, we need to remove numpy types or zero-d tensors
        metrics = denumpify_detensorize(metrics)

        if hasattr(self, "jit_compilation_time"):
            metrics[f"{metric_key_prefix}_jit_compilation_time"] = self.jit_compilation_time

        # Prefix all keys with metric_key_prefix + '_'
        for key in list(metrics.keys()):
            if self._num_evals >= 1:
                if self._metric_agg_idx == 0:
                    self._agg_metrics[key] = 0
                    self._agg_metrics["eval_count"] = 0
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)
                if self._num_evals >= 1:
                    if self._agg_metrics["eval_count"] == 0 and _eval_cnt == 0:
                        self._agg_metrics["eval_count"] = 1e-6 # Avoid division by zero
                    self._agg_metrics[key] = (self._agg_metrics[key] * self._agg_metrics['eval_count'] + metrics[f"{metric_key_prefix}_{key}"] * _eval_cnt) / (self._agg_metrics['eval_count'] + _eval_cnt)
                    self._agg_metrics['eval_count'] += _eval_cnt
                    metrics[f"eval_{key}"] = self._agg_metrics[key]
                    metrics[f"{metric_key_prefix}_count"] = _eval_cnt
                    metrics[f"eval_agg_count"] = self._agg_metrics['eval_count']

        all_labels = None
        all_preds = None
        num_samples = _eval_cnt
        self._metric_agg_idx = (self._metric_agg_idx + 1) % self._num_evals
        return EvalLoopOutput(predictions=all_preds, label_ids=all_labels, metrics=metrics, num_samples=num_samples)
    
if __name__ == "__main__":
    trainer = GenerateEvalSFTTrainer()
