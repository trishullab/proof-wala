#!/usr/bin/env python3

import sys

root_dir = f"{__file__.split('proof_wala')[0]}"
if root_dir not in sys.path:
    sys.path.append(root_dir)
import typing
import os
import json
import ray
import omegaconf
from dataclasses import dataclass, field
from dataclasses_json import dataclass_json
from enum import Enum
from itp_interface.rl.proof_tree import ProofSearchResult
from itp_interface.rl.proof_action import ProofAction
from itp_interface.rl.simple_proof_env import ProofEnvReRankStrategy
from proof_wala.search.search import SearchAlgorithm
from proof_wala.proof_search.search_driver import ProofPathTracer
from proof_wala.search.beam_search import BeamSearch
from proof_wala.search.best_first_search import BestFirstSearch
from proof_wala.proof_search.llm_tactic_generator import NegLoglikelihoodMinimizingHeuristic

class SettingType(Enum):
    Agent = "Agent"
    GptF = "GptF"

    def __str__(self):
        return self.value

@dataclass_json
@dataclass
class EnvSettings(object):
    name: str
    retrieval_strategy: ProofEnvReRankStrategy

@dataclass_json
@dataclass
class EvalSettings(object):
    name: str
    use_hammer: bool
    model_name: str
    is_seq2seq: bool
    max_proof_depth: int = 50
    max_seq_length: int = 2048
    timeout_in_secs: int = 60
    proof_retries: int = 1
    character_per_token: float = 3.6
    width: int = 64
    max_tokens_per_action: int = 175
    do_sample: bool = True
    top_k: int = 20
    stop_tokens: typing.List[str] = field(default_factory=lambda: ["[END]"])
    padding: bool = True
    return_full_text: bool = False
    compute_probabilities: bool = True
    checkpoint_dir: str = ".log/checkpoints"
    should_checkpoint: bool = False
    temperature: float = 0.0
    max_history_messages: int = 0
    proof_dump_dir: str = ".log/proofs/proof-dump-"
    use_human_readable_proof_context: bool = True
    sample: float = 1.0
    sample_seed: int = 0xf00
    use_example_retrieval: bool = False
    always_use_useful_theorem_retrieval: bool = False
    num_goal_per_prompt: typing.Optional[int] = None
    search_strategy: str = "BeamSearch"
    search_params: typing.Dict[str, typing.Any] = field(default_factory=lambda: {"width": 64, "max_new_tokens": 175, "temperature": 0.75, "do_sample": True, "top_k": 64, "padding": True, "return_full_text": False, "compute_probabilities": True})
    proof_search_heuristic: str = "NegLoglikelihoodMinimizingHeuristic"
    model_parallelism: int = 4
    do_lemmas_discovery: bool = True
    proof_tracer: typing.Optional[ProofPathTracer] = None

    @property
    def should_capture_traces_while_proof_search(self):
        return self.proof_tracer is not None

    def get_search_algo(self):
        if self.search_strategy == "BeamSearch":
            return BeamSearch(**self.search_params)
        elif self.search_strategy == "BestFirstSearch":
            return BestFirstSearch(**self.search_params)
        else:
            raise ValueError(f"Unknown search strategy {self.search_strategy}")

    def get_proof_search_heuristic(self):
        if self.proof_search_heuristic == "NegLoglikelihoodMinimizingHeuristic":
            return NegLoglikelihoodMinimizingHeuristic()
        else:
            raise ValueError(f"Unknown proof search heuristic {self.proof_search_heuristic}")

@dataclass_json
@dataclass
class EvalFile(object):
    path: str
    theorems: typing.Union[str, typing.List[str]]
    max_retry_attempts_limits: typing.Dict[str, int]
    max_time_limits_in_secs: typing.Dict[str, float]

@dataclass_json
@dataclass
class EvalDataset(object):
    project: str
    files: typing.List[EvalFile]
    negation: bool

@dataclass_json
@dataclass
class EvalBenchmark(object):
    name: str
    num_files: int
    language: ProofAction.Language
    datasets: typing.List[EvalDataset]
    few_shot_data_path_for_retrieval: str = None
    few_shot_metadata_filename_for_retrieval: str = None
    dfs_data_path_for_retrieval: str = None
    dfs_metadata_filename_for_retrieval: str = None
    timeout_per_theorem_in_secs: int = 720
    setup_cmds: typing.List[str] = field(default_factory=list)

@dataclass_json
@dataclass
class Experiments(object):
    env_settings: EnvSettings
    eval_settings: EvalSettings
    benchmark: EvalBenchmark

@dataclass_json
@dataclass
class EvalRunCheckpointInfo(object):
    checkpoint_file: str
    logging_dirs: typing.List[str]
    proof_dump_dir: str
    theorem_maps: typing.Dict[str, typing.Dict[str, bool]]

    def add_path_to_maps(self, path: str):
        if path not in self.theorem_maps:
            self.theorem_maps[path] = {}

    def add_theorem_to_maps(self, path: str, theorem: str, success: bool):
        self.theorem_maps[path][theorem] = success
        # First create a temporary copy of the checkpoint file
        # This is to avoid memory issues when writing to the file
        checkpoint_file = self.checkpoint_file + ".tmp"
        with open(checkpoint_file, "w") as f:
            f.write(self.to_json(indent=4))
        # Then move the temporary file to the original checkpoint file
        os.rename(checkpoint_file, self.checkpoint_file)
        # with open(self.checkpoint_file, "w") as f:
        #     f.write(self.to_json(indent=4))
    
@dataclass_json
@dataclass
class EvalProofResults(object):
    path: str
    theorem_map: typing.Dict[str, typing.Dict[str, ProofSearchResult]]

    def add_path_to_maps(self, path: str):
        if path not in self.theorem_map:
            self.theorem_map[path] = {}
    
    def add_theorem_to_maps(self, path: str, theorem: str, proof_result: ProofSearchResult):
        self.theorem_map[path][theorem] = proof_result
        # First create a temporary copy of the proof results file
        # This is to avoid memory issues when writing to the file
        proof_results_file = self.path + ".tmp"
        with open(proof_results_file, "w") as f:
            f.write(self.to_json(indent=4))
        # Then move the temporary file to the original proof results file
        os.rename(proof_results_file, self.path)
        # with open(self.path, "w") as f:
        #     f.write(self.to_json(indent=4))

@ray.remote
class EvalProofResultsActor(object):
    def __init__(self, eval_results: EvalProofResults):
        self.eval_results = eval_results
    
    def get_proof_results(self):
        return self.eval_results

    def get_theorem_map(self):
        return self.eval_results.theorem_map

    def add_path_to_maps(self, path: str):
        self.eval_results.add_path_to_maps(path)
    
    def add_theorem_to_maps(self, path: str, theorem: str, proof_result: ProofSearchResult):
        self.eval_results.add_theorem_to_maps(path, theorem, proof_result)

@ray.remote
class EvalRunCheckpointInfoActor(object):
    def __init__(self, eval_checkpoint: EvalRunCheckpointInfo):
        self.eval_checkpoint = eval_checkpoint
    
    def get_checkpoint_info(self):
        return self.eval_checkpoint

    def add_path_to_maps(self, path: str):
        self.eval_checkpoint.add_path_to_maps(path)
    
    def add_theorem_to_maps(self, path: str, theorem: str, success: bool):
        self.eval_checkpoint.add_theorem_to_maps(path, theorem, success)

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


def parse_config(cfg):
    if "ROOT" in os.environ:
        root = os.environ["ROOT"]
    else:
        root = None
    if root is not None:
        # Replace all the <root> placeholders in all the paths in all the setting
        recursive_replace_keywords(cfg, "<root>", root)
    env_settings_cfg = cfg["env_settings"]
    env_settings = EnvSettings(
        name=env_settings_cfg["name"],
        retrieval_strategy=ProofEnvReRankStrategy(env_settings_cfg["retrieval_strategy"]))
    eval_settings_cfg = cfg["eval_settings"]
    eval_settings = EvalSettings(
        name=eval_settings_cfg["name"],
        use_hammer=eval_settings_cfg["use_hammer"],
        model_name=eval_settings_cfg["model_name"],
        is_seq2seq=eval_settings_cfg["is_seq2seq"],
        max_proof_depth=eval_settings_cfg["max_proof_depth"],
        max_seq_length=eval_settings_cfg["max_seq_length"],
        timeout_in_secs=eval_settings_cfg["timeout_in_secs"],
        proof_retries=eval_settings_cfg["proof_retries"],
        character_per_token=eval_settings_cfg["character_per_token"],
        width=eval_settings_cfg["width"],
        max_tokens_per_action=eval_settings_cfg["max_tokens_per_action"],
        temperature=eval_settings_cfg["temperature"],
        do_sample=eval_settings_cfg["do_sample"],
        top_k=eval_settings_cfg["top_k"],
        stop_tokens=eval_settings_cfg["stop_tokens"],
        padding=eval_settings_cfg["padding"],
        return_full_text=eval_settings_cfg["return_full_text"],
        compute_probabilities=eval_settings_cfg["compute_probabilities"],
        checkpoint_dir=eval_settings_cfg["checkpoint_dir"],
        should_checkpoint=eval_settings_cfg["should_checkpoint"],
        proof_dump_dir=eval_settings_cfg["proof_dump_dir"],
        use_human_readable_proof_context=eval_settings_cfg["use_human_readable_proof_context"],
        sample=eval_settings_cfg["sample"],
        sample_seed=eval_settings_cfg["sample_seed"],
        use_example_retrieval=eval_settings_cfg["use_example_retrieval"],
        always_use_useful_theorem_retrieval=eval_settings_cfg["always_use_useful_theorem_retrieval"],
        num_goal_per_prompt=eval_settings_cfg["num_goal_per_prompt"],
        search_strategy=eval_settings_cfg["search_strategy"],
        search_params=eval_settings_cfg["search_params"],
        proof_search_heuristic=eval_settings_cfg["proof_search_heuristic"],
        model_parallelism=eval_settings_cfg["model_parallelism"],
        do_lemmas_discovery=eval_settings_cfg["do_lemmas_discovery"] if "do_lemmas_discovery" in eval_settings_cfg else True,
        proof_tracer=ProofPathTracer.from_dict(eval_settings_cfg["proof_tracer"]) if "proof_tracer" in eval_settings_cfg else None
    )
    benchmark_cfg = cfg["benchmark"]
    datasets_cfg = benchmark_cfg["datasets"]
    eval_datasets = []
    for dataset_cfg in datasets_cfg:
        files_cfg = list(dataset_cfg["files"])
        eval_files = []
        for file_cfg in files_cfg:
            theorems = None
            if type(file_cfg["theorems"]) == str:
                theorems = file_cfg["theorems"]
            else:
                theorems = list(file_cfg["theorems"])
            if "max_retry_attempts_limits" not in file_cfg:
                max_retry_attempts_limits = {}
            else:
                max_retry_attempts_limits = file_cfg["max_retry_attempts_limits"]
            if "max_time_limits_in_secs" not in file_cfg:
                max_time_limits_in_secs = {}
            else:
                max_time_limits_in_secs = file_cfg["max_time_limits_in_secs"]
            eval_files.append(EvalFile(
                path=file_cfg["path"],
                theorems=theorems,
                max_retry_attempts_limits=max_retry_attempts_limits,
                max_time_limits_in_secs=max_time_limits_in_secs))
        eval_datasets.append(EvalDataset(
            project=dataset_cfg["project"],
            files=eval_files,
            negation=dataset_cfg["negation"] if "negation" in dataset_cfg else False))
    language = ProofAction.Language(benchmark_cfg["language"])
    benchmark = EvalBenchmark(
        name=benchmark_cfg["name"],
        num_files=benchmark_cfg["num_files"],
        language=language,
        datasets=eval_datasets,
        few_shot_data_path_for_retrieval=benchmark_cfg["few_shot_data_path_for_retrieval"],
        few_shot_metadata_filename_for_retrieval=benchmark_cfg["few_shot_metadata_filename_for_retrieval"],
        dfs_data_path_for_retrieval=benchmark_cfg["dfs_data_path_for_retrieval"],
        dfs_metadata_filename_for_retrieval=benchmark_cfg["dfs_metadata_filename_for_retrieval"],
        timeout_per_theorem_in_secs=benchmark_cfg["timeout_per_theorem_in_secs"] if "timeout_per_theorem_in_secs" in benchmark_cfg else 720)
    return Experiments(env_settings=env_settings, eval_settings=eval_settings, benchmark=benchmark)