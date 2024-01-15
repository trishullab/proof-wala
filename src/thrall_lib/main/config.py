#!/usr/bin/env python3

import sys

root_dir = f"{__file__.split('thrall_lib')[0]}"
if root_dir not in sys.path:
    sys.path.append(root_dir)
import typing
from dataclasses import dataclass, field
from dataclasses_json import dataclass_json
from enum import Enum
from transformers import TrainingArguments
from thrall_lib.llm_helpers.theorem_proving_training_dataset import TheoremProvingTrainingDataset
from thrall_lib.itp.training_data_formatter import BasicTrainingDataFormatterCallback
from thrall_lib.itp.copra_training_data_formatter import CopraPromptTrainingDataFormatter, CopraTrainingDataset

class ExperimentType(Enum):
    Training = "Training"
    Inferencing = "Inferencing"
    Evaluation = "Evaluation"

    def __str__(self):
        return self.value

class TrainingDatasetType(Enum):
    TheoremProvingTrainingDataset = "TheoremProvingTrainingDataset"
    CopraTrainingDataset = "CopraTrainingDataset"

    def __str__(self):
        return self.value
    
    def get_class(self):
        if self == TrainingDatasetType.TheoremProvingTrainingDataset:
            return TheoremProvingTrainingDataset
        elif self == TrainingDatasetType.CopraTrainingDataset:
            return CopraTrainingDataset
        else:
            raise Exception(f"Invalid training dataset type: {self}")
    
class TrainingDataFormatterType(Enum):
    BasicTrainingDataFormatterCallback = "BasicTrainingDataFormatterCallback"
    CopraPromptTrainingDataFormatter = "CopraPromptTrainingDataFormatter"

    def __str__(self):
        return self.value
    
    def get_class(self):
        if self == TrainingDataFormatterType.BasicTrainingDataFormatterCallback:
            return BasicTrainingDataFormatterCallback
        elif self == TrainingDataFormatterType.CopraPromptTrainingDataFormatter:
            return CopraPromptTrainingDataFormatter
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


def parse_config(cfg) -> Experiment:
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