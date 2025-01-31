#!/usr/bin/env python3

import sys

root_dir = f"{__file__.split('proof_wala')[0]}"
if root_dir not in sys.path:
    sys.path.append(root_dir)
import os
import typing
import omegaconf
from dataclasses import dataclass, field
from dataclasses_json import dataclass_json
from enum import Enum
from transformers import TrainingArguments
from proof_wala.llm_helpers.theorem_proving_training_dataset import TheoremProvingTrainingDataset
from proof_wala.itp.training_data_formatter import BasicTrainingDataFormatterCallback
from proof_wala.itp.copra_training_data_formatter import CopraPromptTrainingDataFormatter, CopraTrainingDataset
from proof_wala.itp.codet5_training_data_formatter import CodeT5PromptTrainingDataFormatter, CodeT5TrainingDataset
from proof_wala.itp.proof_model_training_data_formatter import ProofModelTrainingDataset, ProofModelPromptTrainingDataFormatter

class ExperimentType(Enum):
    Training = "Training"
    Inferencing = "Inferencing"
    Evaluation = "Evaluation"
    TokenCount = "TokenCount"

    def __str__(self):
        return self.value

class TrainingDatasetType(Enum):
    TheoremProvingTrainingDataset = "TheoremProvingTrainingDataset"
    CopraTrainingDataset = "CopraTrainingDataset"
    CodeT5TrainingDataset = "CodeT5TrainingDataset"
    ProofModelTrainingDataset = "ProofModelTrainingDataset" 

    def __str__(self):
        return self.value
    
    def get_class(self):
        if self == TrainingDatasetType.TheoremProvingTrainingDataset:
            return TheoremProvingTrainingDataset
        elif self == TrainingDatasetType.CopraTrainingDataset:
            return CopraTrainingDataset
        elif self == TrainingDatasetType.CodeT5TrainingDataset:
            return CodeT5TrainingDataset
        elif self == TrainingDatasetType.ProofModelTrainingDataset:
            return ProofModelTrainingDataset
        else:
            raise Exception(f"Invalid training dataset type: {self}")
    
class TrainingDataFormatterType(Enum):
    BasicTrainingDataFormatterCallback = "BasicTrainingDataFormatterCallback"
    CopraPromptTrainingDataFormatter = "CopraPromptTrainingDataFormatter"
    CodeT5PromptTrainingDataFormatter = "CodeT5PromptTrainingDataFormatter"
    ProofModelPromptTrainingDataFormatter = "ProofModelPromptTrainingDataFormatter"

    def __str__(self):
        return self.value
    
    def get_class(self):
        if self == TrainingDataFormatterType.BasicTrainingDataFormatterCallback:
            return BasicTrainingDataFormatterCallback
        elif self == TrainingDataFormatterType.CopraPromptTrainingDataFormatter:
            return CopraPromptTrainingDataFormatter
        elif self == TrainingDataFormatterType.CodeT5PromptTrainingDataFormatter:
            return CodeT5PromptTrainingDataFormatter
        elif self == TrainingDataFormatterType.ProofModelPromptTrainingDataFormatter:
            return ProofModelPromptTrainingDataFormatter
        else:
            raise Exception(f"Invalid training data formatter type: {self}")

@dataclass_json
@dataclass
class ModelSettings(object):
    name_or_path: str
    logging_dir: str
    model_args: dict = field(default_factory=dict)

@dataclass_json
@dataclass
class EvalSettings(object):
    eval_name: str
    eval_data_dir: str
    eval_meta_filename: str
    eval_data_log_dir: str

@dataclass_json
@dataclass
class TrainingDataSettings(object):
    training_data_dir: str
    training_meta_filename: str
    training_data_log_dir: str
    training_dataset_type: TrainingDatasetType
    training_data_formatter_type: TrainingDataFormatterType
    training_dataset_args: dict = field(default_factory=dict)   
    eval_data_dir: typing.Optional[str] = None
    eval_meta_filename: typing.Optional[str] = None
    eval_data_log_dir: typing.Optional[str] = None
    test_data_dir: typing.Optional[str] = None
    test_meta_filename: typing.Optional[str] = None
    test_data_log_dir: typing.Optional[str] = None
    evals : typing.Optional[typing.List[EvalSettings]] = None

@dataclass_json
@dataclass
class TrainingSettings(object):
    train_eval_split: bool = False
    train_percentage: float = 1.0
    eval_percentage: float = 1.0
    test_percentage: float = 1.0
    training_args: TrainingArguments = field(default_factory=TrainingArguments)

@dataclass_json
@dataclass
class Experiment(object):
    name : str
    expertiment_type: ExperimentType
    model_settings: ModelSettings
    training_data_settings: TrainingDataSettings
    training_settings: TrainingSettings

def recursive_replace_keywords(cfg, key_word: str, replace_word: str):
    if isinstance(cfg, omegaconf.dictconfig.DictConfig) or isinstance(cfg, dict):
        keys = [key for key in cfg] # to avoid immutable dict error
        for key in keys:
            value = cfg[key]
            if isinstance(value, str):
                cfg[key] = value.replace(key_word, replace_word)
            elif isinstance(value, omegaconf.dictconfig.DictConfig) or \
                isinstance(value, omegaconf.listconfig.ListConfig) or \
                isinstance(value, dict) or \
                isinstance(value, list):
                recursive_replace_keywords(value, key_word, replace_word)
    elif isinstance(cfg, omegaconf.listconfig.ListConfig) or isinstance(cfg, list):
        for i in range(len(cfg)):
            value = cfg[i]
            if isinstance(value, str):
                cfg[i] = value.replace(key_word, replace_word)
            elif isinstance(value, omegaconf.dictconfig.DictConfig) or \
                isinstance(value, omegaconf.listconfig.ListConfig) or \
                isinstance(value, dict) or \
                isinstance(value, list):
                recursive_replace_keywords(value, key_word, replace_word)
    else:
        raise Exception(f"Invalid type: {type(cfg)}")


def parse_config(cfg) -> Experiment:
    if "ROOT" in os.environ:
        root = os.environ["ROOT"]
    else:
        root = None
    if root is not None:
        # Replace all the <root> placeholders in all the paths in all the setting
        recursive_replace_keywords(cfg, "<root>", root)
    name = cfg["name"]
    experiment_type = ExperimentType(cfg["experiment_type"])
    model_settings : ModelSettings = ModelSettings.from_dict(cfg["model_settings"])
    if model_settings.model_args is None:
        model_settings.model_args = {}
    training_data_settings : TrainingDataSettings = TrainingDataSettings.from_dict(cfg["training_data_settings"])
    if training_data_settings.training_dataset_args is None:
        training_data_settings.training_dataset_args = {}
    training_settings = TrainingSettings.from_dict(cfg["training_settings"])
        
    experiment = Experiment(
        name=name,
        expertiment_type=experiment_type,
        model_settings=model_settings,
        training_data_settings=training_data_settings,
        training_settings=training_settings
    )
    return experiment