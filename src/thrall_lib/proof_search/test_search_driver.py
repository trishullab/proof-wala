#!/usr/bin/env python3

import sys
root_dir = f"{__file__.split('thrall_lib')[0]}"
if root_dir not in sys.path:
    sys.path.append(root_dir)
import typing
import logging
import numpy as np
import os
from datetime import datetime
from thrall_lib.proof_search.search_driver import ProofActionGenerator, ProofSearhHeuristic, ProofSearchDriver
from itp_interface.rl.simple_proof_env import ProofState, ProofAction, ProofEnv
from itp_interface.tools.proof_exec_callback import ProofExecutorCallback
from itp_interface.tools.dynamic_coq_proof_exec import DynamicProofExecutor as DynamicCoqExecutor
from thrall_lib.search.search import Node, SearchAlgorithm
from thrall_lib.search.best_first_search import BestFirstSearch
from thrall_lib.search.beam_search import BeamSearch

class RandomCoqProofActionGenerator(ProofActionGenerator):
    def __init__(self, width: int = 4):
        super().__init__(width)
        pass
    
    def generate_actions(self, state: ProofState, k: int = None) -> typing.List[typing.Tuple[float, ProofAction]]:
        assert state.language == ProofAction.Language.COQ, "Only Coq is supported"
        if state.training_data_format.goal_description == DynamicCoqExecutor.NotInProofModeDescription:
            return []
        else:
            if len(state.proof_tree) == 0:
                tactics = {
                    "intros.": 0.3,
                    "intro.": 0.5,
                    "simpl.": 0.20
                }
            elif state.training_data_format.goal_description == DynamicCoqExecutor.ProofFinishedDescription:
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

    def __call__(self, node: Node) -> float:
        if node.name == DynamicCoqExecutor.NotInProofModeDescription:
            return float("-inf")
        else:
            return len(node.name)

def test_search_algorithm(
        exp_name: str,
        algo: SearchAlgorithm, 
        proof_env: ProofEnv,
        proof_search_heuristic: ProofFoundHeuristic,
        action_generator: ProofActionGenerator,
        search_width: int = 4, 
        attempt_count: int = 1,
        timeout_in_secs: float = 60) -> typing.Callable[[Node], typing.List[Node]]:
    #tracemalloc.start()
    proof_found = False
    tree_dump_folder = f".log/proof_search/{exp_name}"
    time_now = datetime.now().strftime("%Y%m%d_%H%M%S")
    tree_dump_folder = os.path.join(tree_dump_folder, time_now)
    os.makedirs(tree_dump_folder, exist_ok=True)
    max_attempts = attempt_count
    final_proof_res = None
    while attempt_count > 0 and not proof_found:
        search_driver = ProofSearchDriver(
            algo,
            action_generator,
            proof_search_heuristic,
            width=search_width)
        start_node, tree_node, proof_res = search_driver.search_proof(
            proof_env,
            timeout_in_secs=timeout_in_secs)
        final_proof_res = proof_res
        proof_paths = []
        if proof_res.proof_found:
            proof_path = algo.reconstruct_path(start_node, tree_node)
            proof_paths = [proof_path]
            # Run the proof steps and check if the proof is finished
            with proof_env:
                for td in proof_res.proof_steps:
                    proof_env.step(ProofAction(ProofAction.ActionType.RUN_TACTIC, proof_env.language, tactics=td.proof_steps))
                assert proof_env.done
                proof_found = True
        dot = algo.visualize_search(start_node, show=False, mark_paths=proof_paths)
        dot.render(os.path.join(tree_dump_folder, f"proof_search_{max_attempts - attempt_count + 1}"), format='png', quiet=True)
        attempt_count -= 1
    print(final_proof_res)

if __name__ == '__main__':
    proof_exec_callback = ProofExecutorCallback(
        project_folder=".",
        file_path="src/thrall_lib/data/proofs/coq/simple2/thms.v"
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
    for search_aglo in [BestFirstSearch(), BeamSearch(2)]:
        algo_name = search_aglo.__class__.__name__
        for theorem_name in theorem_names:
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
                attempt_count=5,
                timeout_in_secs=60)