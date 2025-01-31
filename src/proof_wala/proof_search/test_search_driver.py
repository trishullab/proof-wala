#!/usr/bin/env python3

import sys
root_dir = f"{__file__.split('proof_wala')[0]}"
if root_dir not in sys.path:
    sys.path.append(root_dir)
import typing
import logging
import numpy as np
import os
from datetime import datetime
from proof_wala.proof_search.search_driver import ProofActionGenerator, ProofSearhHeuristic, ProofSearchDriver, ProofStateInfo, ProofPathTracer
from itp_interface.rl.simple_proof_env import ProofState, ProofAction, ProofEnv
from itp_interface.rl.simpl_proof_env_pool import replicate_proof_env
from itp_interface.tools.proof_exec_callback import ProofExecutorCallback
from itp_interface.tools.training_data_format import TrainingDataMetadataFormat
from itp_interface.tools.training_data import TrainingData
from proof_wala.search.search import Node, SearchAlgorithm, Edge
from proof_wala.search.best_first_search import BestFirstSearch
from proof_wala.search.beam_search import BeamSearch
os.environ["CUDA_VISIBLE_DEVICES"] = "5"

class RandomCoqProofActionGenerator(ProofActionGenerator):
    def __init__(self, width: int = 4):
        super().__init__(width)
        pass

    def get_proof_end_for_language(self, language: ProofAction.Language) -> ProofAction:
        return ProofAction(ProofAction.ActionType.RUN_TACTIC, ProofAction.Language.COQ, tactics=['Qed.'])
    
    def generate_actions(self, state_info: ProofStateInfo, k: int = None) -> typing.List[typing.Tuple[float, ProofAction]]:
        state : ProofState = state_info.proof_state
        done: bool = state_info.done
        assert state.language == ProofAction.Language.COQ, "Only Coq is supported"
        if done:
            return []
        elif len(state.proof_tree) == 0:
            tactics = {
                "intros.": 0.3,
                "intro.": 0.5,
                "simpl.": 0.20
            }
        elif len(state.training_data_format.start_goals) == 0:
            tactics = {
                "Qed.": 1.0
            }
        else:
            tactics = {
                "Qed.": 0.2,
                "intro.": 0.15,
                "intros.": 0.10,
                "trivial.": 0.15,
                "reflexivity.": 0.10,
                "auto.": 0.15,
                "firstorder.": 0.15,
            }
        # Generate random actions
        actions = []
        # Sample actions based on the probabilities
        k = k if k else self.width
        for _ in range(k):
            action = np.random.choice(list(tactics.keys()), p=list(tactics.values()))
            actions.append((tactics[action], ProofAction(ProofAction.ActionType.RUN_TACTIC, state.language, tactics=[action])))
        return actions

class ProofFoundHeuristic(ProofSearhHeuristic):
    def __init__(self):
        pass

    def __call__(self, parent_node: Node, edge: Edge, node: Node) -> float:
        state_info : ProofStateInfo = node.other_data
        state : ProofState = state_info.proof_state
        if state_info.done or len(state.training_data_format.start_goals) == 0:
            return -100.0
        else:
            return node.score

def test_search_algorithm(
        exp_name: str,
        algo: SearchAlgorithm, 
        proof_env: ProofEnv,
        proof_search_heuristic: ProofFoundHeuristic,
        action_generator: ProofActionGenerator,
        search_width: int = 4, 
        attempt_count: int = 1,
        timeout_in_secs: float = 60,
        trace_settings: ProofPathTracer = None,
        extract_original_proof: bool = False) -> typing.Callable[[Node], typing.List[Node]]:
    #tracemalloc.start()
    proof_found = False
    tree_dump_folder = f".log/proof_search/{exp_name}"
    time_now = datetime.now().strftime("%Y%m%d_%H%M%S")
    tree_dump_folder = os.path.join(tree_dump_folder, time_now)
    os.makedirs(tree_dump_folder, exist_ok=True)
    max_attempts = attempt_count
    final_proof_res = None
    while attempt_count > 0 and not proof_found:
        print(f"Attempt {max_attempts - attempt_count + 1} of {max_attempts}")
        search_driver = ProofSearchDriver(
            algo,
            action_generator,
            proof_search_heuristic,
            width=search_width,
            tracer=trace_settings)
        replicated_env = replicate_proof_env(proof_env)
        start_node, tree_node, proof_res = search_driver.search_proof(
            proof_env,
            timeout_in_secs=timeout_in_secs,
            extract_original_proofs=extract_original_proof)
        final_proof_res = proof_res
        proof_paths = []
        if proof_res.proof_found:
            proof_paths = algo.reconstruct_all_paths(start_node, tree_node)
            print(f"{len(proof_paths)} proof(s) found in {max_attempts - attempt_count + 1} attempt(s)")
            if len(proof_paths) == 0:
                print("Proof path is empty, something went wrong!")
                print(proof_res)
            else:
                # Run the proof steps and check if the proof is finished
                with replicated_env:
                    for td in proof_res.proof_steps:
                        replicated_env.step(ProofAction(ProofAction.ActionType.RUN_TACTIC, proof_env.language, tactics=td.proof_steps))
                    assert replicated_env.done
                proof_found = True
        dot = algo.visualize_search(start_node, show=False, mark_paths=proof_paths)
        dot.render(os.path.join(tree_dump_folder, f"proof_search_{max_attempts - attempt_count + 1}"), format='png', quiet=True)
        attempt_count -= 1
    print(final_proof_res)

if __name__ == '__main__':
    import ray
    ray.init(num_cpus=10, num_gpus=1, ignore_reinit_error=True)
    proof_exec_callback = ProofExecutorCallback(
        project_folder=".",
        file_path="src/proof_wala/data/proofs/coq/simple2/thms.v"
    )
    theorem_names = [
        "nat_add_comm",
        "double_neg",
        "trival_implication",
        "modus_ponens",
        "modus_tollens",
        "disjunctive_syllogism",
        "contrapositive",
        "nat_zero_add",
        "nat_add_zero",
        "nat_add_succ",
        "nat_succ_add"
    ]
    width = 10
    tracer = ProofPathTracer(
        collect_traces=True,
        folder=".log/traces",
        training_meta_filename="traces_meta.json",
        training_metadata=TrainingDataMetadataFormat(
            data_filename_prefix="traces_data",
            lemma_ref_filename_prefix="traces_lemma_ref"
        ),
        max_parallelism=4
    )
    os.makedirs(tracer.folder, exist_ok=True)
    tracer.reset_training_data()
    for search_aglo in [BestFirstSearch(), BeamSearch(3)]:
        algo_name = search_aglo.__class__.__name__
        print(f"Running tests for {algo_name}")
        for theorem_name in theorem_names:
            print(f"Trying to prove {theorem_name}")
            language = ProofAction.Language.COQ
            always_retrieve_thms = False
            logger = logging.getLogger(__name__)
            env = ProofEnv("test", proof_exec_callback, theorem_name, max_proof_depth=10, always_retrieve_thms=always_retrieve_thms, logger=logger)
            test_search_algorithm(
                exp_name=os.path.join(algo_name, theorem_name),
                algo=search_aglo,
                proof_env=env,
                proof_search_heuristic=ProofFoundHeuristic(),
                action_generator=RandomCoqProofActionGenerator(width),
                search_width=width,
                attempt_count=2,
                timeout_in_secs=60,
                trace_settings=tracer,
                extract_original_proof=True)
            print('-' * 80)
        print('=' * 80)
    tracer.save()