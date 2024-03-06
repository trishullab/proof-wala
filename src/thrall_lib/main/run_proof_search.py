#!/usr/bin/env python3
import os
os.environ["COMET_MODE"] = "DISABLED"
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys
root_dir = f"{__file__.split('thrall_lib')[0]}"
if root_dir not in sys.path:
    sys.path.append(root_dir)
import hydra
import copy
import logging
import random
import time
import math
import typing
import multiprocessing
multiprocessing.set_start_method('spawn', force=True)
from datetime import datetime
from thrall_lib.llm_helpers.model import Model
from thrall_lib.proof_search.search_driver import ProofSearchDriver
from thrall_lib.proof_search.llm_tactic_generator import CodeT5PromptFormatter, CodeT5ResponseParser, LlmProofActionGenerator
from thrall_lib.main.eval_config import EnvSettings, EvalBenchmark, EvalDataset, EvalProofResults, EvalSettings, Experiments, EvalRunCheckpointInfo, parse_config
from itp_interface.tools.log_utils import setup_logger
from itp_interface.rl.proof_tree import ProofSearchResult
from itp_interface.rl.simple_proof_env import ProofEnv, ProofAction
from itp_interface.tools.proof_exec_callback import ProofExecutorCallback
from itp_interface.tools.dynamic_coq_proof_exec import DynamicProofExecutor as DynamicCoqProofExecutor
from itp_interface.tools.dynamic_lean_proof_exec import DynamicProofExecutor as DynamicLeanProofExecutor

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

def eval_dataset(env_settings: EnvSettings, eval_benchmark: EvalBenchmark, dataset: EvalDataset, eval_settings: EvalSettings, eval_checkpoint_info: EvalRunCheckpointInfo, eval_proof_results: EvalProofResults, logger: logging.Logger = None):
    logger = logger if logger else logging.getLogger(__name__)
    skip_files_in_checkpoint = False if "SKIP_FILES_IN_CHECKPOINT" not in os.environ else bool(os.environ["SKIP_FILES_IN_CHECKPOINT"])

    if eval_settings.proof_retries > 1:
        assert eval_settings.temperature > 0.0, "Proof retries is only supported for temperature > 0.0"

    proof_attempts_done = False
    if "STRICT_TIME_BUDGET_ACCROSS_ATTEMPTS" in os.environ and bool(os.environ["STRICT_TIME_BUDGET_ACCROSS_ATTEMPTS"]):
        track_time = True
        logger.info(f"Strict time budget across attempts is enabled. Proofs will not be attempted beyond {eval_benchmark.timeout_per_theorem_in_secs} seconds.")
    else:
        track_time = False
    time_budget_tracker = {}
    model_path = eval_settings.model_name
    is_seq2seq = eval_settings.is_seq2seq
    model = Model(model_path, is_seq2seq=is_seq2seq, use_lora=False)
    model.__enter__()
    for attempt_idx in range(eval_settings.proof_retries):
        if proof_attempts_done:
            break
        any_proof_attempted = False
        for file in dataset.files:
            path = os.path.join(dataset.project, file.path)
            if track_time and path not in time_budget_tracker:
                if len(file.max_time_limits_in_secs) > 0:
                    time_budget_tracker[path] = copy.deepcopy(file.max_time_limits_in_secs)
                else:
                    time_budget_tracker[path] = {}
            proof_dump_file_name = os.path.join(eval_settings.proof_dump_dir, f"{path.replace('/', '_')}.txt")
            if skip_files_in_checkpoint and path in eval_checkpoint_info.theorem_maps:
                logger.info(f"Skipping the file: {path} as it was already attempted before.")
                # The proof result for this file is already in the checkpoint
                if path in eval_proof_results.theorem_map:
                    # The proof result for this file is already in the proof results
                    # So we just log the proof result
                    for lemma_name, proof_res_chkpt in eval_proof_results.theorem_map[path].items():
                        logger.info(f"Dumping proof search result:\n{proof_res_chkpt}")
                        logger.info(f"Prover for lemma: {lemma_name} in file {path} completed.")
                    continue
            if not os.path.exists(proof_dump_file_name):
                with open(proof_dump_file_name, "w") as f:
                    f.write(f"File: {path}\n")
                    f.write(f"Dataset:\n {dataset.to_json(indent=4)}\n")
                    f.write(f"Evaluation Settings:\n {eval_settings.to_json(indent=4)}\n")
            eval_checkpoint_info.add_path_to_maps(path)
            eval_proof_results.add_path_to_maps(path)
            proof_exec_callback = ProofExecutorCallback(
                project_folder=dataset.project,
                file_path=path,
                language=eval_benchmark.language,
                use_hammer=eval_settings.use_hammer,
                timeout_in_secs=eval_settings.timeout_in_secs,
                use_human_readable_proof_context=eval_settings.use_human_readable_proof_context,
                suppress_error_log=True,
                always_use_retrieval=eval_settings.always_use_useful_theorem_retrieval,
                logger=logger)
            get_all_lemmas_proof_exec_callback = ProofExecutorCallback(
                project_folder=dataset.project,
                file_path=path,
                language=eval_benchmark.language,
                use_hammer=False, # We don't need hammer for this
                timeout_in_secs=eval_settings.timeout_in_secs,
                use_human_readable_proof_context=eval_settings.use_human_readable_proof_context,
                suppress_error_log=True,
                always_use_retrieval=False,
                logger=logger)
            # class _Get_All_Lemmas:
            #     def __call__(self, ret_dict, logger: logging.Logger):
            #         try:
            #             ret_dict["lemmas"] = get_all_lemmas(get_all_lemmas_proof_exec_callback, logger)
            #         except:
            #             logger.exception(f"Exception occurred while getting all lemmas in file: {path}")
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
            lemmas_to_prove = return_dict["lemmas"]
            if isinstance(file.theorems, str) and file.theorems == "*":
                file.theorems = list(lemmas_to_prove)
                file.theorems.sort() # sort to ensure one order when no theorems are specified
            elif isinstance(file.theorems, list):
                # Check all theorems which can be proved
                intersection = set(file.theorems).intersection(lemmas_to_prove)
                # Arrange them in the order of the file.theorems
                file.theorems = [x for x in file.theorems if x in intersection]
            else:
                raise ValueError(f"Invalid theorems: {file.theorems}")
            logger.info(f"Discovered {len(file.theorems)} lemmas to prove in {path}")
            logger.info(f"Lemmas to prove in file {path}: \n{file.theorems}")
            if eval_settings.sample < 1.0:
                sample_size = math.ceil(len(file.theorems) * eval_settings.sample)
                logger.info(f"Sampling {sample_size} lemmas from {len(file.theorems)} lemmas in file {path}")
                random.seed(eval_settings.sample_seed)
                file.theorems = list(random.sample(file.theorems, sample_size))
                logger.info(f"Sampled lemmas to prove in file {path}: \n{file.theorems}")
            for lemma_name in file.theorems:
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
                    logger.info(f"Attempting to prove lemma: {lemma_name}")

                    max_seq_length = eval_settings.max_seq_length
                    character_per_token = eval_settings.character_per_token
                    prompt_formatter = CodeT5PromptFormatter(max_token_in_prompt=max_seq_length, character_per_token=character_per_token)
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
                    proof_res_chkpt = eval_proof_results.theorem_map.get(path, {}).get(lemma_name, None)
                    max_retry_attempts = file.max_retry_attempts_limits.get(lemma_name, eval_settings.proof_retries)
                    if proof_res_chkpt is None or (not proof_res_chkpt.proof_found and proof_res_chkpt.additional_info["attempt_idx"] < max_retry_attempts - 1):
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
                                    width=eval_settings.width)
                                start_node, tree_node, proof_res = search_driver.search_proof(
                                    env,
                                    timeout_in_secs=eval_benchmark.timeout_per_theorem_in_secs)
                                proof_paths = []
                                if proof_res.proof_found:
                                    proof_paths = algo.reconstruct_all_paths(start_node, tree_node)
                                dot = algo.visualize_search(start_node, show=False, mark_paths=proof_paths)
                                tree_dump_folder = os.path.join(eval_settings.proof_dump_dir, "proof_trees", lemma_name)
                                os.makedirs(tree_dump_folder, exist_ok=True)
                                time_stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
                                tree_dump_path = os.path.join(tree_dump_folder, f"{lemma_name}_{time_stamp}")
                                dot.render(tree_dump_path, format='svg', quiet=True)
                                ret_dict["proof_res"] = proof_res
                                ret_dict["attempted_success"] = True
                                ret_dict["service_down"] = False
                                with env:
                                    for td in proof_res.proof_steps:
                                        env.step(ProofAction(ProofAction.ActionType.RUN_TACTIC, env.language, tactics=td.proof_steps))
                                    assert env.done
                                    additional_info = proof_res_chkpt.additional_info if proof_res_chkpt is not None else {'attempt_idx': 0}
                                    env.dump_proof(additional_info=additional_info)
                            except:
                                logger.exception(f"Exception occurred while proving lemma: {lemma_name} in file {path}")
                                ret_dict["attempted_success"] = False
                                ret_dict["service_down"] = False

                        should_retry = True
                        max_retry = 4 # This retry is only when for some mysterious reason the llama service goes down
                        logger.info(f"Attempt {attempt_idx + 1} for proving lemma: {lemma_name} in file {path}")
                        while should_retry and max_retry > 0:
                            # Run the prover with a timeout
                            timeout = min(eval_settings.timeout_in_secs * eval_settings.max_proof_depth * 1.25, eval_benchmark.timeout_per_theorem_in_secs)
                            if track_time and time_budget_tracker[path][lemma_name] < timeout:
                                timeout = time_budget_tracker[path][lemma_name]
                            logger.info(f"Running the prover agent for lemma: {lemma_name} with timeout: {timeout} seconds")
                            #p = multiprocessing.Process(target=_run_prover, args=(return_dict,))
                            tic_start = time.time()
                            _run_prover(return_dict)
                            # p.start()
                            # p.join(timeout)
                            # if p.is_alive():
                            #     p.kill()
                            #     p.join()
                            # p.close()
                            toc_end = time.time()
                            if track_time:
                                time_budget_tracker[path][lemma_name] -= (toc_end - tic_start)
                            if track_time and time_budget_tracker[path][lemma_name] <= 0:
                                logger.info(f"Time budget exhausted for lemma: {lemma_name} in file {path}")
                                proof_attempt_idx = (proof_res_chkpt.additional_info["attempt_idx"] + 1) if proof_res_chkpt is not None and "attempt_idx" in proof_res_chkpt.additional_info else attempt_idx
                                proof_res_chkpt = copy.deepcopy(no_proof_res)
                                proof_res_chkpt.is_timeout = True
                                proof_res_chkpt.proof_time_in_secs = toc_end - tic_start
                                proof_res_chkpt.additional_info["attempt_idx"] = proof_attempt_idx
                                eval_proof_results.add_theorem_to_maps(path, lemma_name, proof_res_chkpt)
                                eval_checkpoint_info.add_theorem_to_maps(path, lemma_name, False)
                                should_retry = False
                            elif "attempted_success" not in return_dict:
                                logger.info(f"Prover Agent for lemma: {lemma_name} in file {path} got killed as it timed out.")
                                proof_attempt_idx = (proof_res_chkpt.additional_info["attempt_idx"] + 1) if proof_res_chkpt is not None and "attempt_idx" in proof_res_chkpt.additional_info else attempt_idx
                                proof_res_chkpt = copy.deepcopy(no_proof_res)
                                proof_res_chkpt.is_timeout = True
                                proof_res_chkpt.proof_time_in_secs = toc_end - tic_start
                                proof_res_chkpt.additional_info["attempt_idx"] = proof_attempt_idx
                                eval_proof_results.add_theorem_to_maps(path, lemma_name, proof_res_chkpt)
                                eval_checkpoint_info.add_theorem_to_maps(path, lemma_name, False)
                                should_retry = False
                            elif not return_dict["attempted_success"]:
                                if not return_dict["service_down"] or \
                                    (eval_settings.model_name is not None and \
                                    len(eval_settings.model_name) != 0) or \
                                    max_retry <= 1:
                                    logger.info(f"Failed to prove lemma: {lemma_name} in file {path}")
                                    proof_attempt_idx = (proof_res_chkpt.additional_info["attempt_idx"] + 1) if proof_res_chkpt is not None and "attempt_idx" in proof_res_chkpt.additional_info else attempt_idx
                                    proof_res_chkpt = copy.deepcopy(no_proof_res)
                                    proof_res_chkpt.is_timeout = True
                                    proof_res_chkpt.proof_time_in_secs = toc_end - tic_start
                                    proof_res_chkpt.additional_info["attempt_idx"] = proof_attempt_idx
                                    eval_proof_results.add_theorem_to_maps(path, lemma_name, proof_res_chkpt)
                                    eval_checkpoint_info.add_theorem_to_maps(path, lemma_name, False)
                                    should_retry = False
                            else:
                                logger.info(f"Prover for lemma: {lemma_name} in file {path} completed.")
                                proof_attempt_idx = (proof_res_chkpt.additional_info["attempt_idx"] + 1) if proof_res_chkpt is not None and "attempt_idx" in proof_res_chkpt.additional_info else attempt_idx
                                proof_res_chkpt : ProofSearchResult = return_dict["proof_res"]
                                proof_res_chkpt.additional_info["attempt_idx"] = proof_attempt_idx
                                eval_proof_results.add_theorem_to_maps(path, lemma_name, proof_res_chkpt)
                                eval_checkpoint_info.add_theorem_to_maps(path, lemma_name, True)
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
                    eval_proof_results.add_theorem_to_maps(path, lemma_name, proof_res_chkpt)
                    eval_checkpoint_info.add_theorem_to_maps(path, lemma_name, False)
        proof_attempts_done = not any_proof_attempted
    pass

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
                eval_dataset(env_settings, benchmark, dataset, eval_settings, checkpoint_info, eval_proof_results, logger=logger)
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

@hydra.main(config_path="config", config_name="eval_test_experiment", version_base="1.2")
def main(cfg):
    experiment = parse_config(cfg)
    log_dir = ".log/evals/benchmark/{}/{}".format(experiment.benchmark.name, time.strftime("%Y%m%d-%H%M%S"))
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "eval.log")
    logger = setup_logger(__name__, log_path, logging.INFO, '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger.info(f"Pid: {os.getpid()}")
    logger.info(f"Running Experiment: {experiment.to_json(indent=4)}")
    eval_benchmark(experiment, log_dir, logger=logger)
    pass

if __name__ == "__main__":
    # RayUtils.init_ray(num_of_cpus=20, object_store_memory_in_gb=50, memory_in_gb=1)
    main()