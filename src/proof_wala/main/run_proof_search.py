#!/usr/bin/env python3
import os
os.environ["COMET_MODE"] = "DISABLED"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys
root_dir = f"{__file__.split('proof_wala')[0]}"
if root_dir not in sys.path:
    sys.path.append(root_dir)
import hydra
import copy
import logging
import random
import time
import math
import typing
import ray
import multiprocessing
import unicodedata
import re
import uuid
import random
multiprocessing.set_start_method('spawn', force=True)
from datetime import datetime
from proof_wala.tools.ray_utils import clean_ray_logs
from proof_wala.llm_helpers.model import Model
from proof_wala.proof_search.search_driver import ProofSearchDriver
from proof_wala.proof_search.llm_tactic_generator import CodeT5PromptFormatter, CodeT5ResponseParser, LlmProofActionGenerator
from proof_wala.main.eval_config import EnvSettings, EvalBenchmark, EvalDataset, EvalFile, EvalProofResults, EvalProofResultsActor, EvalRunCheckpointInfoActor, EvalSettings, Experiments, EvalRunCheckpointInfo, parse_config
from itp_interface.tools.log_utils import setup_logger
from itp_interface.rl.proof_tree import ProofSearchResult
from itp_interface.rl.simple_proof_env import ProofEnv, ProofAction
from itp_interface.tools.proof_exec_callback import ProofExecutorCallback
from itp_interface.tools.dynamic_coq_proof_exec import DynamicProofExecutor as DynamicCoqProofExecutor
from itp_interface.tools.dynamic_lean_proof_exec import DynamicProofExecutor as DynamicLeanProofExecutor
from itp_interface.tools.lean4_sync_executor import get_all_theorems_in_file as get_all_theorems_in_file_lean4, get_fully_qualified_theorem_name as get_fully_qualified_theorem_name_lean4, get_theorem_name_resembling as get_theorem_name_resembling_lean4

def slugify(value, allow_unicode=False):
    """
    Taken from https://github.com/django/django/blob/master/django/utils/text.py
    Convert to ASCII if 'allow_unicode' is False. Convert spaces or repeated
    dashes to single dashes. Remove characters that aren't alphanumerics,
    underscores, or hyphens. Convert to lowercase. Also strip leading and
    trailing whitespace, dashes, and underscores.
    """
    value = str(value)
    if allow_unicode:
        value = unicodedata.normalize('NFKC', value)
    else:
        value = unicodedata.normalize('NFKD', value).encode('ascii', 'ignore').decode('ascii')
    value = re.sub(r'[^\w\s-]', '', value.lower())
    return re.sub(r'[-\s]+', '-', value).strip('-_')

def check_query_limit_reached(max_query_limit: int) -> typing.Callable[[int, typing.Dict[str, typing.Any]], bool]:
    def _check_query_limit_reached(steps: int, info: typing.Dict[str, typing.Any]):
        return info["queries"] >= max_query_limit
    return _check_query_limit_reached

def query_limit_info_message(max_query_limit: int) -> typing.Callable[[int, typing.Dict[str, typing.Any]], str]:
    def _query_limit_info_message(steps: int, info: typing.Dict[str, typing.Any]):
        return f"Step {info['queries']}/{max_query_limit} (Actual steps: {steps})"
    return _query_limit_info_message

def get_all_lemmas(coq_proof_exec_callback: ProofExecutorCallback, logger: logging.Logger):
    lemmas_to_prove = []
    if coq_proof_exec_callback.language == ProofAction.Language.LEAN4:
        lemmas_to_prove = get_all_theorems_in_file_lean4(coq_proof_exec_callback.file_path, use_cache=True)
        lemmas_to_prove = [get_fully_qualified_theorem_name_lean4(lemma) for lemma in lemmas_to_prove]
    else:
        with coq_proof_exec_callback.get_proof_executor() as main_executor:
            if isinstance(main_executor, DynamicLeanProofExecutor):
                main_executor.run_all_without_exec()
                lemmas_to_prove = main_executor.find_all_theorems_names()
            elif isinstance(main_executor, DynamicCoqProofExecutor):
                while not main_executor.execution_complete:
                    assert not main_executor.is_in_proof_mode(), "main_executor must not be in proof mode"
                    _ = list(main_executor.run_till_next_lemma_return_exec_stmt())
                    if main_executor.execution_complete:
                        break
                    lemma_name = main_executor.get_lemma_name_if_running()
                    if lemma_name is None:
                        _ = list(main_executor.run_to_finish_lemma_return_exec())
                        if main_executor.execution_complete:
                            break
                    else:
                        logger.info(f"Discovered lemma: {lemma_name}")
                        lemmas_to_prove.append(lemma_name)
                        main_executor.run_to_finish_lemma()
            else:
                raise Exception(f"Unsupported proof executor: {main_executor}")
    logger.info(f"Discovered {len(lemmas_to_prove)} lemmas")
    return lemmas_to_prove

class _Get_All_Lemmas:
    def __call__(self, ret_dict, path, get_all_lemmas_proof_exec_callback, logger: logging.Logger):
        try:
            ret_dict["lemmas"] = get_all_lemmas(get_all_lemmas_proof_exec_callback, logger)
        except:
            logger.exception(f"Exception occurred while getting all lemmas in file: {path}")

@ray.remote
def discover_lemmas_in_benchmark(
    skip_files_in_checkpoint: bool,
    eval_benchmark: EvalBenchmark, 
    dataset: EvalDataset, 
    eval_checkpoint_info: EvalRunCheckpointInfoActor,
    eval_settings: EvalSettings,
    eval_proof_results: EvalProofResultsActor,
    logfile: str):
    log_id = str(uuid.uuid4())
    logger = setup_logger(f"discover_lemmas_in_benchmark_{log_id}", logfile, format='%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s - %(message)s')
    if eval_settings.do_lemmas_discovery:
        for file in dataset.files:
            path = os.path.join(dataset.project, file.path)
            theorem_maps : typing.Dict[str, typing.Dict[str, ProofSearchResult]] = ray.get(eval_proof_results.get_theorem_map.remote())
            if skip_files_in_checkpoint and path in theorem_maps:
                logger.info(f"Skipping the file: {path} as it was already attempted before.")
                continue
            ray.get(eval_checkpoint_info.add_path_to_maps.remote(path))
            ray.get(eval_proof_results.add_path_to_maps.remote(path))
            get_all_lemmas_proof_exec_callback = ProofExecutorCallback(
                project_folder=dataset.project,
                file_path=path,
                language=eval_benchmark.language,
                use_hammer=False, # We don't need hammer for this
                timeout_in_secs=eval_settings.timeout_in_secs,
                use_human_readable_proof_context=eval_settings.use_human_readable_proof_context,
                suppress_error_log=True,
                always_use_retrieval=False,
                logger=logger,
                setup_cmds=eval_benchmark.setup_cmds)
            manager = multiprocessing.Manager()
            return_dict = manager.dict()
            file_time_out = eval_settings.timeout_in_secs * eval_settings.max_proof_depth * 50
            logger.info(f"Getting all lemmas in file: {path} with timeout: {file_time_out} seconds")
            p = multiprocessing.Process(target=_Get_All_Lemmas(), args=(return_dict, path, get_all_lemmas_proof_exec_callback, logger))
            p.start()
            p.join(file_time_out)
            if p.is_alive():
                p.kill()
                p.join()
            p.close()
            if "lemmas" not in return_dict:
                logger.info(f"Failed to get all lemmas in file: {path}, moving on to the next file.")
                continue
            lemmas_to_prove = list(return_dict["lemmas"])
            if isinstance(file.theorems, str) and file.theorems == "*":
                file.theorems = list(lemmas_to_prove)
                file.theorems.sort() # sort to ensure one order when no theorems are specified
            elif isinstance(file.theorems, list):
                logger.info(f"Discovered {len(lemmas_to_prove)} lemmas in file: {path}")
                if get_all_lemmas_proof_exec_callback.language == ProofAction.Language.LEAN4:
                    logger.info("Converting theorems to fully qualified names for Lean 4.")
                    actual_theorem_name = [get_theorem_name_resembling_lean4(path, lemma, use_cache=True) for lemma in file.theorems]
                    for _lemma in actual_theorem_name:
                        logger.info(f"Converted lemma: {_lemma}")
                    for _lemma in lemmas_to_prove:
                        logger.info(f"Discovered lemma: {_lemma}")
                else:
                    actual_theorem_name = file.theorems
                if not dataset.negation:
                    # Check all theorems which can be proved
                    intersection = set(actual_theorem_name).intersection(lemmas_to_prove)
                    for _lemma in intersection:
                        logger.info(f"Intersection lemma: {_lemma}")
                    # Arrange them in the order of the file.theorems
                    file.theorems = [x for x in actual_theorem_name if x in intersection]
                else:
                    logger.info("Dataset is negation dataset. So, removing the theorems which are mentioned in the dataset.")
                    # Take the difference of the theorems which can be proved
                    difference = set(lemmas_to_prove).difference(file.theorems)
                    for lemma in difference:
                        logger.info(f"Lemma: {lemma} is not in the dataset.")
                    difference = list(difference)
                    difference.sort()
                    file.theorems = difference
            else:
                raise ValueError(f"Invalid theorems: {file.theorems}")
            logger.info(f"Discovered {len(file.theorems)} lemmas to prove in {path}")
            logger.info(f"Lemmas to prove in file {path}: \n{file.theorems}")
            if eval_settings.sample < 1.0:
                sample_size = math.ceil(len(file.theorems) * eval_settings.sample)
                logger.info(f"Sampling {sample_size} lemmas from {len(file.theorems)} lemmas in file {path}")
                file.theorems = list(random.sample(file.theorems, sample_size))
                logger.info(f"Sampled lemmas to prove in file {path}: \n{file.theorems}")
    else:
        logger.info("Skipping lemmas discovery")
    clean_ray_logs(logger)
    return dataset

def eval_dataset_once(
    model: Model, 
    proof_attempts_done: bool, 
    skip_files_in_checkpoint: bool, 
    track_time: bool,
    eval_benchmark: EvalBenchmark, 
    dataset: EvalDataset, 
    eval_checkpoint_info: EvalRunCheckpointInfoActor,
    eval_settings: EvalSettings,
    env_settings: EnvSettings,
    eval_proof_results: EvalProofResultsActor,
    time_budget_tracker: typing.Dict[str, typing.Dict[str, float]],
    attempt_idx: int,
    logger: logging.Logger,
    reshuffle: bool = False):
    if proof_attempts_done:
        return True
    any_proof_attempted = False
    if eval_settings.proof_tracer is not None:
        eval_settings.proof_tracer.reset_training_data(logger)
    if reshuffle:
        dataset_files = list(dataset.files)
        # Reshuffle the files
        random.shuffle(dataset_files)
    else:
        dataset_files = dataset.files
    current_attempt_idx = attempt_idx
    logger.info(f"Starting evaluation for dataset: {dataset.project}, attempt: {attempt_idx + 1}")
    for file in dataset_files:
        clean_ray_logs(logger)
        path = os.path.join(dataset.project, file.path)
        if track_time and path not in time_budget_tracker:
            if len(file.max_time_limits_in_secs) > 0:
                time_budget_tracker[path] = copy.deepcopy(file.max_time_limits_in_secs)
            else:
                time_budget_tracker[path] = {}
        proof_dump_file_name = os.path.join(eval_settings.proof_dump_dir, f"{path.replace('/', '_')}.txt")
        theorem_maps : typing.Dict[str, typing.Dict[str, ProofSearchResult]] = ray.get(eval_proof_results.get_theorem_map.remote())
        if skip_files_in_checkpoint and path in theorem_maps:
            logger.info(f"Skipping the file: {path} as it was already attempted before.")
            # The proof result for this file is already in the proof results
            # So we just log the proof result
            for lemma_name, proof_res_chkpt in theorem_maps[path].items():
                logger.info(f"Dumping proof search result:\n{proof_res_chkpt}")
                logger.info(f"Prover for lemma: {lemma_name} in file {path} completed.")
            continue
        if not os.path.exists(proof_dump_file_name):
            with open(proof_dump_file_name, "w") as f:
                f.write(f"File: {path}\n")
                f.write(f"Dataset:\n {dataset.to_json(indent=4)}\n")
                f.write(f"Evaluation Settings:\n {eval_settings.to_json(indent=4)}\n")
        ray.get(eval_checkpoint_info.add_path_to_maps.remote(path))
        ray.get(eval_proof_results.add_path_to_maps.remote(path))
        proof_exec_callback = ProofExecutorCallback(
            project_folder=dataset.project,
            file_path=path,
            language=eval_benchmark.language,
            use_hammer=eval_settings.use_hammer,
            timeout_in_secs=eval_settings.timeout_in_secs,
            use_human_readable_proof_context=eval_settings.use_human_readable_proof_context,
            suppress_error_log=True,
            always_use_retrieval=eval_settings.always_use_useful_theorem_retrieval,
            logger=logger,
            setup_cmds=eval_benchmark.setup_cmds,
            enable_search=False) # TODO: Make this configurable
        if reshuffle:
            file_theorems = list(file.theorems)
            random.shuffle(file_theorems)
        else:
            file_theorems = file.theorems
        for lemma_name in file_theorems:
            no_proof_res = ProofSearchResult(
                None, 
                False, 
                lemma_name, 
                [], 
                -1, 
                -1, 
                possible_failed_paths=-1, 
                num_of_backtracks=-1, 
                is_timeout=False, 
                is_inference_exhausted=False, 
                longest_success_path=-1,
                additional_info={},
                language=eval_benchmark.language)
            try:
                if track_time and lemma_name not in time_budget_tracker[path]:
                    time_budget_tracker[path][lemma_name] = eval_benchmark.timeout_per_theorem_in_secs
                if track_time and time_budget_tracker[path][lemma_name] <= 0:
                    logger.info(f"Time budget exhausted for lemma: {lemma_name} in file {path} so skipping it.")
                    continue
                logger.info(f"Will check if attempt possible for lemma: {lemma_name} [# {attempt_idx + 1}/{eval_settings.proof_retries}]")

                max_seq_length = eval_settings.max_seq_length
                character_per_token = eval_settings.character_per_token
                # TODO: Move no_steps to eval_settings
                prompt_formatter = CodeT5PromptFormatter(max_token_in_prompt=max_seq_length, character_per_token=character_per_token, no_steps=True)
                response_parser = CodeT5ResponseParser()
                width = eval_settings.width
                max_new_tokens=eval_settings.max_tokens_per_action
                temperature=eval_settings.temperature # Nucleus sampling
                do_sample=eval_settings.do_sample # Nucleus sampling
                top_k=eval_settings.top_k # Nucleus sampling
                stop_tokens=eval_settings.stop_tokens
                padding=eval_settings.padding
                return_full_text=eval_settings.return_full_text
                compute_probabilities=eval_settings.compute_probabilities
                # Get theorem_maps again because the previous attempt can be overwriiten
                theorem_maps: typing.Dict[str, typing.Dict[str, ProofSearchResult]] = ray.get(eval_proof_results.get_theorem_map.remote())
                proof_res_chkpt = theorem_maps.get(path, {}).get(lemma_name, None)
                max_retry_attempts = file.max_retry_attempts_limits.get(lemma_name, eval_settings.proof_retries)
                if proof_res_chkpt is None or (not proof_res_chkpt.proof_found and proof_res_chkpt.additional_info["attempt_idx"] < attempt_idx): #max_retry_attempts - 1):
                    any_proof_attempted = True
                    manager = multiprocessing.Manager()
                    return_dict = manager.dict()
                    def _run_prover(ret_dict):
                        try:
                            env = ProofEnv(f"basic_proof_env_{lemma_name}", 
                                proof_exec_callback, 
                                lemma_name, 
                                retrieval_strategy=env_settings.retrieval_strategy, 
                                max_proof_depth=eval_settings.max_proof_depth, 
                                always_retrieve_thms=eval_settings.always_use_useful_theorem_retrieval, 
                                logger=logger)
                            algo = eval_settings.get_search_algo()
                            generator = LlmProofActionGenerator(
                                model, prompt_formatter, response_parser, width, logger,
                                max_new_tokens=max_new_tokens, temperature=temperature, do_sample=do_sample, top_k=top_k,
                                stop_tokens=stop_tokens, padding=padding, 
                                return_full_text=return_full_text, 
                                compute_probabilities=compute_probabilities)
                            proof_search_heuristic = eval_settings.get_proof_search_heuristic()
                            search_driver = ProofSearchDriver(
                                algo,
                                generator,
                                proof_search_heuristic,
                                width=eval_settings.width,
                                logger=logger,
                                tracer=eval_settings.proof_tracer)
                            start_node, tree_node, proof_res = search_driver.search_proof(
                                env,
                                timeout_in_secs=eval_benchmark.timeout_per_theorem_in_secs)
                            logger.info("Proof search finished.")
                            if start_node is not None:
                                proof_paths = []
                                if proof_res.proof_found:
                                    proof_paths = algo.reconstruct_all_paths(start_node, tree_node)
                                dot = algo.visualize_search(start_node, show=False, mark_paths=proof_paths)
                                lemma_file_name = slugify(lemma_name, allow_unicode=False)
                                tree_dump_folder = os.path.join(eval_settings.proof_dump_dir, "proof_trees", lemma_file_name)
                                try:
                                    # if lemma_file_name can be a valid file name
                                    os.makedirs(tree_dump_folder, exist_ok=True)
                                except:
                                    logger.exception(f"Failed to create the proof tree dump folder for lemma: {lemma_name} in file {path}")
                                    lemma_name_hash = hash(lemma_name)
                                    lemma_file_name = lemma_name_hash
                                    tree_dump_folder = os.path.join(eval_settings.proof_dump_dir, "proof_trees", lemma_file_name)
                                    os.makedirs(tree_dump_folder, exist_ok=True)
                                    logger.info("\nUsing hash of lemma name:" +
                                        f"\n {lemma_name} to create the folder for dumping the proof tree in file {file.path}" +
                                        f"\n The hash is: {lemma_name_hash}")
                                time_stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
                                tree_dump_path = os.path.join(tree_dump_folder, f"{lemma_file_name}_{time_stamp}")
                                dot.render(tree_dump_path, format='svg', quiet=True)
                            ret_dict["proof_res"] = proof_res
                            ret_dict["attempted_success"] = True
                            ret_dict["service_down"] = False
                            additional_info = proof_res_chkpt.additional_info if proof_res_chkpt is not None else {'attempt_idx': 0}
                            proof_res.additional_info = additional_info
                            logger.info(f"Dumping proof search result:\n{proof_res}")
                            logger.info(f"Prover for lemma: {lemma_name} in file {path} completed.")
                        except:
                            logger.exception(f"Exception occurred while proving lemma: {lemma_name} in file {path}")
                            ret_dict["attempted_success"] = False
                            ret_dict["service_down"] = False

                    should_retry = True
                    max_retry = 4 # This retry is only when for some mysterious reason the llama service goes down
                    if proof_res_chkpt is not None:
                        current_attempt_idx = proof_res_chkpt.additional_info["attempt_idx"]
                        logger.info(f"Previous attempt for proving lemma: {lemma_name} in file {path} was: {attempt_idx}")
                        logger.info(f"Previous proof search result:\n{proof_res_chkpt}")
                    logger.info(f"Attempt {attempt_idx + 1 if proof_res_chkpt is None else current_attempt_idx + 2} for proving lemma: {lemma_name} in file {path}")
                    while should_retry and max_retry > 0:
                        # Run the prover with a timeout
                        timeout = min(eval_settings.timeout_in_secs * eval_settings.max_proof_depth * 1.25, eval_benchmark.timeout_per_theorem_in_secs)
                        if track_time and time_budget_tracker[path][lemma_name] < timeout:
                            timeout = time_budget_tracker[path][lemma_name]
                        logger.info(f"Running the prover agent for lemma: {lemma_name} with timeout: {timeout} seconds")
                        tic_start = time.time()
                        _run_prover(return_dict)
                        toc_end = time.time()
                        if track_time:
                            time_budget_tracker[path][lemma_name] -= (toc_end - tic_start)
                        if track_time and time_budget_tracker[path][lemma_name] <= 0:
                            logger.info(f"Time budget exhausted for lemma: {lemma_name} in file {path}")
                            proof_attempt_idx = (proof_res_chkpt.additional_info["attempt_idx"] + 1) if proof_res_chkpt is not None and "attempt_idx" in proof_res_chkpt.additional_info else current_attempt_idx
                            proof_res_chkpt = copy.deepcopy(no_proof_res)
                            proof_res_chkpt.is_timeout = True
                            proof_res_chkpt.proof_time_in_secs = toc_end - tic_start
                            proof_res_chkpt.additional_info["attempt_idx"] = proof_attempt_idx
                            ray.get(eval_proof_results.add_theorem_to_maps.remote(path, lemma_name, proof_res_chkpt))
                            ray.get(eval_checkpoint_info.add_theorem_to_maps.remote(path, lemma_name, False))
                            should_retry = False
                        elif "attempted_success" not in return_dict:
                            logger.info(f"Prover Agent for lemma: {lemma_name} in file {path} got killed as it timed out.")
                            proof_attempt_idx = (proof_res_chkpt.additional_info["attempt_idx"] + 1) if proof_res_chkpt is not None and "attempt_idx" in proof_res_chkpt.additional_info else current_attempt_idx
                            proof_res_chkpt = copy.deepcopy(no_proof_res)
                            proof_res_chkpt.is_timeout = True
                            proof_res_chkpt.proof_time_in_secs = toc_end - tic_start
                            proof_res_chkpt.additional_info["attempt_idx"] = proof_attempt_idx
                            ray.get(eval_proof_results.add_theorem_to_maps.remote(path, lemma_name, proof_res_chkpt))
                            ray.get(eval_checkpoint_info.add_theorem_to_maps.remote(path, lemma_name, False))
                            should_retry = False
                        elif not return_dict["attempted_success"]:
                            if not return_dict["service_down"] or \
                                (eval_settings.model_name is not None and \
                                len(eval_settings.model_name) != 0) or \
                                max_retry <= 1:
                                logger.info(f"Failed to prove lemma: {lemma_name} in file {path}")
                                proof_attempt_idx = (proof_res_chkpt.additional_info["attempt_idx"] + 1) if proof_res_chkpt is not None and "attempt_idx" in proof_res_chkpt.additional_info else current_attempt_idx
                                proof_res_chkpt = copy.deepcopy(no_proof_res)
                                proof_res_chkpt.is_timeout = True
                                proof_res_chkpt.proof_time_in_secs = toc_end - tic_start
                                proof_res_chkpt.additional_info["attempt_idx"] = proof_attempt_idx
                                ray.get(eval_proof_results.add_theorem_to_maps.remote(path, lemma_name, proof_res_chkpt))
                                ray.get(eval_checkpoint_info.add_theorem_to_maps.remote(path, lemma_name, False))
                                should_retry = False
                        else:
                            logger.info(f"Prover for lemma: {lemma_name} in file {path} completed.")
                            proof_attempt_idx = (proof_res_chkpt.additional_info["attempt_idx"] + 1) if proof_res_chkpt is not None and "attempt_idx" in proof_res_chkpt.additional_info else current_attempt_idx
                            proof_res_chkpt : ProofSearchResult = return_dict["proof_res"]
                            proof_res_chkpt.additional_info["attempt_idx"] = proof_attempt_idx
                            ray.get(eval_proof_results.add_theorem_to_maps.remote(path, lemma_name, proof_res_chkpt))
                            ray.get(eval_checkpoint_info.add_theorem_to_maps.remote(path, lemma_name, True))
                            should_retry = False
                        return_dict.clear()
                        max_retry -= 1
                else:
                    proof_res_attempt_idx = proof_res_chkpt.additional_info["attempt_idx"]
                    if proof_res_attempt_idx == attempt_idx:
                        logger.info(f"Dumping proof search result:\n{proof_res_chkpt}")
                        logger.info(f"Prover for lemma: {lemma_name} in file {path} completed.")
                    else:
                        logger.info(f"Skipping the attempt for proving lemma: {lemma_name} in file {path} as it was already attempted before.")
            except:
                logger.exception(f"Exception occurred while proving lemma: {lemma_name} in file {path}")
                proof_res_chkpt = copy.deepcopy(no_proof_res)
                proof_res_chkpt.is_timeout = True
                proof_res_chkpt.additional_info["attempt_idx"] = attempt_idx
                proof_res_chkpt.additional_info["total_queries"] = 0
                proof_res_chkpt.proof_time_in_secs = 0
                ray.get(eval_proof_results.add_theorem_to_maps.remote(path, lemma_name, proof_res_chkpt))
                ray.get(eval_checkpoint_info.add_theorem_to_maps.remote(path, lemma_name, False))
                # eval_proof_results.add_theorem_to_maps(path, lemma_name, proof_res_chkpt)
                # eval_checkpoint_info.add_theorem_to_maps(path, lemma_name, False)
    
    if eval_settings.proof_tracer is not None:
        eval_settings.proof_tracer.save()
    proof_attempts_done = not any_proof_attempted
    return proof_attempts_done

@ray.remote
def eval_dataset_once_with_retries(
    time_budget_tracker: typing.Dict[str, typing.Dict[str, float]],
    attempt_idx: int,
    model_path: str,
    is_seq2seq: bool,
    proof_attempts_done: bool, 
    skip_files_in_checkpoint: bool, 
    track_time: bool,
    eval_benchmark: EvalBenchmark, 
    dataset: EvalDataset, 
    eval_checkpoint_info: EvalRunCheckpointInfoActor,
    eval_settings: EvalSettings,
    env_settings: EnvSettings,
    eval_proof_results: EvalProofResultsActor,
    log_dir: str,
    device_id: typing.Optional[int] = None):
    # time_budget_tracker = {}
    if device_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
    model = Model(model_path, is_seq2seq=is_seq2seq, use_lora=False)
    model.__enter__()
    logfile = os.path.join(log_dir, f"eval_dataset_multiple_attempts.log")
    logger_id = str(uuid.uuid4())
    logger = setup_logger(f"eval_dataset_multiple_attempts_{logger_id}", logfile, format='%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s - %(message)s')
    retry_cnt = 0
    reshuffle = False
    attempt_success = False
    eval_result = False
    while retry_cnt < 100 and not attempt_success:
        try:
            logger.info(f"(Retry Number #: {retry_cnt + 1}) Restarting evaluation for dataset: {dataset.project}")
            eval_result = eval_dataset_once(
                model,
                proof_attempts_done,
                skip_files_in_checkpoint,
                track_time,
                eval_benchmark,
                dataset,
                eval_checkpoint_info,
                eval_settings,
                env_settings,
                eval_proof_results,
                time_budget_tracker,
                attempt_idx,
                logger,
                reshuffle=reshuffle)
            attempt_success = True
        except:
            logger.exception(f"Exception occurred while evaluating dataset: {dataset.project}")
            if retry_cnt < 98:
                logger.info(f"Next Retry Count: {retry_cnt + 2}")
            else:
                logger.info(f"Final Retry Count: {retry_cnt + 2}")
            attempt_success = False
            reshuffle = True
            random_seed = random.randint(0, 1000000)
            logger.info(f"Setting random seed to: {random_seed}")
            random.seed(random_seed)
        if eval_result:
            break
        retry_cnt += 1
    return eval_result

@ray.remote
def eval_dataset_multiple_attempts(
    model_path: str,
    is_seq2seq: bool,
    proof_attempts_done: bool, 
    skip_files_in_checkpoint: bool, 
    track_time: bool,
    eval_benchmark: EvalBenchmark, 
    dataset: EvalDataset, 
    eval_checkpoint_info: EvalRunCheckpointInfoActor,
    eval_settings: EvalSettings,
    env_settings: EnvSettings,
    eval_proof_results: EvalProofResultsActor,
    log_dir: str,
    device_id: typing.Optional[int] = None):
    time_budget_tracker = {}
    # if device_id is not None:
    #     os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
    # model = Model(model_path, is_seq2seq=is_seq2seq, use_lora=False)
    # model.__enter__()
    logfile = os.path.join(log_dir, f"eval_dataset_multiple_attempts.log")
    logger = setup_logger("eval_dataset_multiple_attempts", logfile, format='%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s - %(message)s')
    logger.info(f"Starting evaluation for dataset: {dataset.project}")
    for file in dataset.files:
        logger.info(f"File in dataset: {file.path}")
        logger.info(f"Theorems in file: {file.theorems}")
    for attempt_idx in range(eval_settings.proof_retries):
        proof_attempts_done_remote = eval_dataset_once_with_retries.remote(
            time_budget_tracker,
            attempt_idx,
            model_path,
            is_seq2seq,
            proof_attempts_done, 
            skip_files_in_checkpoint, 
            track_time,
            eval_benchmark, 
            dataset, 
            eval_checkpoint_info,
            eval_settings,
            env_settings,
            eval_proof_results,
            log_dir,
            device_id)
        proof_attempts_done = ray.get(proof_attempts_done_remote)
        proof_attempts_done = proof_attempts_done and attempt_idx == eval_settings.proof_retries - 1
        if proof_attempts_done:
            logger.info('=' * 80)
            logger.info(f"Finished all proof attempts for dataset: {dataset.project}")
            logger.info('=' * 80)
        else:
            logger.info('=' * 80)
            logger.info(f"Proof attempt (# {attempt_idx + 1}/{eval_settings.proof_retries}) done for dataset: {dataset.project}")
            logger.info('=' * 80)

def eval_dataset(env_settings: EnvSettings, eval_benchmark: EvalBenchmark, dataset: EvalDataset, eval_settings: EvalSettings, eval_checkpoint_info: EvalRunCheckpointInfo, eval_proof_results: EvalProofResults, logger: logging.Logger = None):
    logger = logger if logger else logging.getLogger(__name__)
    skip_files_in_checkpoint = False if "SKIP_FILES_IN_CHECKPOINT" not in os.environ else bool(os.environ["SKIP_FILES_IN_CHECKPOINT"])
    follow_seed = False if "FOLLOW_SEED" not in os.environ else bool(os.environ["FOLLOW_SEED"])

    if eval_settings.proof_retries > 1:
        assert eval_settings.temperature > 0.0, "Proof retries is only supported for temperature > 0.0"

    proof_attempts_done = False
    if "STRICT_TIME_BUDGET_ACCROSS_ATTEMPTS" in os.environ and bool(os.environ["STRICT_TIME_BUDGET_ACCROSS_ATTEMPTS"]):
        track_time = True
        logger.info(f"Strict time budget across attempts is enabled. Proofs will not be attempted beyond {eval_benchmark.timeout_per_theorem_in_secs} seconds.")
    else:
        track_time = False
    model_path = eval_settings.model_name
    is_seq2seq = eval_settings.is_seq2seq
    # get CUDA_VISIBLE_DEVICES
    gpus = []
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        num_gpus = len(os.environ["CUDA_VISIBLE_DEVICES"].split(","))
        max_model_parallelism = min(num_gpus, eval_settings.model_parallelism)
        gpus = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
    else:
        max_model_parallelism = 1
    # Split the datset into max_model_parallelism parts
    dataset_chunks = []
    files_per_chunk = math.ceil(len(dataset.files) / max_model_parallelism)
    for i in range(max_model_parallelism):
        start = i * files_per_chunk
        end = min((i + 1) * files_per_chunk, len(dataset.files))
        files = dataset.files[start:end]
        if len(files) == 0:
            break
        dataset_chunks.append(EvalDataset(project=dataset.project, files=dataset.files[start:end], negation=dataset.negation))
    remotes = []
    # Add proof_wala to python path
    root_dir = f"{__file__.split('proof_wala')[0]}"
    os.environ["PYTHONPATH"] = f"{root_dir}:{os.environ.get('PYTHONPATH', '')}"
    eval_checkpoint_info_actor = EvalRunCheckpointInfoActor.remote(eval_checkpoint_info)
    eval_proof_results_actor = EvalProofResultsActor.remote(eval_proof_results)
    if follow_seed:
        random.seed(eval_settings.sample_seed)
    # First discover all the lemmas that has to be proved
    for i, dataset in enumerate(dataset_chunks):
        logdir = eval_checkpoint_info.logging_dirs[-1]
        logfile = os.path.join(logdir, f"discover_lemmas_in_benchmark_{i}.log")
        remotes.append(discover_lemmas_in_benchmark.remote(
            skip_files_in_checkpoint,
            eval_benchmark,
            dataset,
            eval_checkpoint_info_actor,
            eval_settings,
            eval_proof_results_actor,
            logfile))
    discovered_dataset_chunks : typing.List[EvalDataset] = ray.get(remotes)
    # No negation datasets are allowed now as we have already discovered the lemmas
    new_dataset_chunks = [EvalDataset(project=dataset.project, files=[], negation=False) for _ in range(max_model_parallelism)]
    # Redistribute the lemmas to the dataset chunks
    new_dataset_chunk_idx = 0
    new_dataset_chunk_path = [{} for _ in range(max_model_parallelism)]
    if follow_seed:
        seed = eval_settings.sample_seed
    else:
        seed = int(time.time()) % 1000000
    logger.info(f"Shuffling the discovered lemmas with seed: {seed}")
    random.seed(seed)
    random.shuffle(discovered_dataset_chunks)
    for i in range(len(discovered_dataset_chunks)):
        files = list(discovered_dataset_chunks[i].files)
        random.shuffle(files)
        for file in files:
            assert isinstance(file.theorems, list), f"Invalid theorems: {file.theorems}"
            thm_lst = list(file.theorems)
            random.shuffle(thm_lst)
            for thm in thm_lst:
                if file.path not in new_dataset_chunk_path[new_dataset_chunk_idx]:
                    new_file = EvalFile(file.path, [thm], 
                        file.max_retry_attempts_limits, 
                        file.max_time_limits_in_secs)
                    new_dataset_chunks[new_dataset_chunk_idx].files.append(new_file)
                    new_dataset_chunk_path[new_dataset_chunk_idx][file.path] = len(new_dataset_chunks[new_dataset_chunk_idx].files) - 1
                else:
                    new_file_idx = new_dataset_chunk_path[new_dataset_chunk_idx][file.path]
                    new_dataset_chunks[new_dataset_chunk_idx].files[new_file_idx].theorems.append(thm)
                new_dataset_chunk_idx = (new_dataset_chunk_idx + 1) % max_model_parallelism
    dataset_chunks = new_dataset_chunks

    for i, dataset in enumerate(dataset_chunks):
        gpu_id = gpus[i] if i < len(gpus) else None
        if gpu_id is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
        log_dir = eval_checkpoint_info.logging_dirs[-1]
        log_dir = os.path.join(log_dir, f"gpu_{gpu_id}_{i}")
        os.makedirs(log_dir, exist_ok=True)
        remotes.append(eval_dataset_multiple_attempts.remote(
            model_path,
            is_seq2seq,
            proof_attempts_done,
            skip_files_in_checkpoint,
            track_time,
            eval_benchmark,
            dataset,
            eval_checkpoint_info_actor,
            eval_settings,
            env_settings,
            eval_proof_results_actor,
            log_dir,
            gpu_id))
    ray.get(remotes)
    eval_checkpoint_info = ray.get(eval_checkpoint_info_actor.get_checkpoint_info.remote())
    eval_proof_results = ray.get(eval_proof_results_actor.get_proof_results.remote())
    return eval_checkpoint_info, eval_proof_results

def measure_success(benchmark : EvalBenchmark, eval_settings : EvalSettings, eval_proof_results: EvalProofResults, logger: logging.Logger = None):
    success_count = 0
    proofs_dump_file = os.path.join(eval_settings.proof_dump_dir, "benchmark_proof_results.txt")
    proof_dump_file_exists = os.path.exists(proofs_dump_file)
    open_mode = "a" if proof_dump_file_exists else "w"
    with open(proofs_dump_file, open_mode) as f:
        if not proof_dump_file_exists:
            f.write(f"Settings: \n{eval_settings.to_json(indent=4)}\n")
            f.write(f"Benchmark: \n{benchmark.to_json(indent=4)}\n")
        for path, proofs in eval_proof_results.theorem_map.items():
            for lemma_name, proof_res in proofs.items():
                if proof_res.proof_found:
                    success_count += 1
                    logger.info(f"Proof found for lemma: {lemma_name} in file {path}")
                else:
                    logger.info(f"Proof not found for lemma: {lemma_name} in file {path}")
                f.write(f"Lemma: {lemma_name}\n")
                f.write(f"File: {path}\n")
                f.write(f"Proof/Incomplete proof: \n{proof_res}\n")
        total_attempted = sum([len(x) for _, x in eval_proof_results.theorem_map.items()])
        logger.info(f"Success rate: {success_count}/{total_attempted} = {success_count/total_attempted} for benchmark: {benchmark.name}")
        f.write(f"Success rate: {success_count}/{total_attempted} = {success_count/total_attempted} for benchmark: {benchmark.name}\n")

def eval_benchmark(experiment: Experiments, log_dir: str, logger: logging.Logger = None):
    trial_cnt = 1
    env_settings = experiment.env_settings
    eval_settings = experiment.eval_settings
    benchmark = experiment.benchmark
    checkpoint_dir = experiment.eval_settings.checkpoint_dir
    eval_settings.checkpoint_dir = os.path.join(checkpoint_dir, benchmark.name, eval_settings.name)
    os.makedirs(eval_settings.checkpoint_dir, exist_ok=True)
    # Load the checkpoint file if it exists
    checkpoint_file = os.path.join(eval_settings.checkpoint_dir, "checkpoint_info.json")
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, "r") as f:
            checkpoint_info: EvalRunCheckpointInfo = EvalRunCheckpointInfo.from_json(f.read())
        eval_settings.proof_dump_dir = checkpoint_info.proof_dump_dir
        checkpoint_info.logging_dirs.append(log_dir)
    else:
        time_now = time.strftime("%Y%m%d-%H%M%S")
        eval_settings.proof_dump_dir = os.path.join(eval_settings.proof_dump_dir, benchmark.name, time_now)
        os.makedirs(eval_settings.proof_dump_dir, exist_ok=True)
        checkpoint_info = EvalRunCheckpointInfo(
            checkpoint_file=checkpoint_file,
            proof_dump_dir=eval_settings.proof_dump_dir, 
            logging_dirs=[log_dir], 
            theorem_maps={})
    eval_proof_file = os.path.join(eval_settings.proof_dump_dir, "proof_results.json")
    if os.path.exists(eval_proof_file):
        with open(eval_proof_file, "r") as f:
            eval_proof_results: EvalProofResults = EvalProofResults.from_json(f.read())
    else:
        eval_proof_results = EvalProofResults(
            path=os.path.join(eval_settings.proof_dump_dir, "proof_results.json"),
            theorem_map={})
    while trial_cnt > 0:
        try:
            logger = logger or logging.getLogger(__name__)
            for dataset in benchmark.datasets:
                checkpoint_info, eval_proof_results = eval_dataset(env_settings, benchmark, dataset, eval_settings, checkpoint_info, eval_proof_results, logger=logger)
            measure_success(benchmark, eval_settings, eval_proof_results, logger=logger)
            trial_cnt = 0
        except:
            trial_cnt -= 1
            logger.exception(f"Exception occurred. Retrying {trial_cnt} more times.")
            if trial_cnt <= 0:
                raise
            else:
                time.sleep(10)
    logger.info(f"Finished running experiment: \n{experiment.to_json(indent=4)}")

@hydra.main(config_path="config", config_name="eval_simple_lean_test_multilingual", version_base="1.2")
def main(cfg):
    experiment = parse_config(cfg)
    log_dif_prefix = ".log/evals/benchmark" if "log_dir" not in cfg else cfg.log_dir
    log_dir = os.path.join(log_dif_prefix, experiment.benchmark.name, time.strftime("%Y%m%d-%H%M%S"))
    # log_dir = ".log/evals/benchmark/{}/{}".format(experiment.benchmark.name, time.strftime("%Y%m%d-%H%M%S"))
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "eval.log")
    logger = setup_logger(__name__, log_path, logging.INFO, format='%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s - %(message)s')
    logger.info(f"Pid: {os.getpid()}")
    logger.info(f"Running Experiment: {experiment.to_json(indent=4)}")
    eval_benchmark(experiment, log_dir, logger=logger)
    pass

def run_proof_search():
    # RayUtils.init_ray(num_of_cpus=20, object_store_memory_in_gb=50, memory_in_gb=1)
    # Start the ray cluster
    os.environ["PYTHONPATH"] = f"{root_dir}:{os.environ.get('PYTHONPATH', '')}"
    from filelock import FileLock
    import json
    os.makedirs(".log/locks", exist_ok=True)
    ray_was_started = False
    print("Starting run_proof_search Pid: ", os.getpid())
    with FileLock(".log/locks/ray.lock"):
        if os.path.exists(".log/ray/session_latest"):
            with open(".log/ray/session_latest", "r") as f:
                ray_session = f.read()
                ray_session = json.loads(ray_session)
            ray_address = ray_session["address"]
            ray.init(address=ray_address)
            print("Ray was already started")
        else:
            ray_session = ray.init(
                num_cpus=8, 
                object_store_memory=150*2**30, 
                _memory=150*2**30, 
                logging_level=logging.CRITICAL, 
                ignore_reinit_error=False, 
                log_to_driver=False, 
                configure_logging=False,
                _system_config={"metrics_report_interval_ms": 3*10**8})
            with open(".log/ray/session_latest", "w") as f:
                f.write(json.dumps(ray_session))
            ray_was_started = True
            print("Ray was started")
            print("Ray session: ", ray_session)
    main()

if __name__ == "__main__":
    run_proof_search()