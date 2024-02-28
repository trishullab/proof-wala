#!/usr/bin/env python3

import sys
root_dir = f"{__file__.split('thrall_lib')[0]}"
if root_dir not in sys.path:
    sys.path.append(root_dir)
import typing
import logging
from abc import ABC, abstractmethod
from itp_interface.rl.simple_proof_env import ProofEnv, ProofState, ProofAction
from itp_interface.tools.training_data_format import Goal, TrainingDataFormat
from itp_interface.tools.dynamic_coq_proof_exec import DynamicProofExecutor as DynamicCoqExecutor
from itp_interface.rl.proof_tree import ProofSearchResult
from thrall_lib.search.search import SearchAlgorithm, Node, Edge
from thrall_lib.tools.proof_env_replicator import ProofEnvPool

def convert_state_to_string(state: ProofState) -> str:
    goals = state.training_data_format.start_goals
    if len(goals) == 0:
        return state.training_data_format.goal_description
    else:
        all_lines = []
        goal_len = len(goals)
        for idx, goal in enumerate(state.training_data_format.start_goals):
            all_lines.append(f"({idx+1}/{goal_len})")
            for hyp in goal.hypotheses:
                all_lines.append(hyp)
            if goal.goal is not None:
                all_lines.append(f"⊢ {goal.goal}")
        return '\n'.join(all_lines)

def end_state_string(env: ProofEnv) -> str:
    if env.language == ProofAction.Language.COQ:
        return DynamicCoqExecutor.NotInProofModeDescription
    else:
        raise NotImplementedError("Lean not supported yet")


class ProofActionGenerator(ABC):
    def __init__(self, width: int = 4):
        assert width > 0, "Width must be greater than 0"
        self.width = width
        pass
    
    @abstractmethod
    def generate_actions(self, state: ProofState) -> typing.List[typing.Tuple[float, ProofAction]]:
        pass

class ProofSearchBranchGenerator(ABC):
    def __init__(self, 
        envs : typing.List[ProofEnv], 
        proof_action_generator: ProofActionGenerator):
        assert proof_action_generator is not None, "Proof action generator cannot be None"
        assert envs is not None, "Environments cannot be None"
        assert len(envs) > 0, "Environments must contain at least one environment"
        assert len(envs) == proof_action_generator.width, "Number of environments must match the width of the proof action generator"
        self.proof_action_generator = proof_action_generator
        self.envs = envs
        pass

    def __call__(self, node: Node) -> typing.List[Node]:
        # Call the proof action generator to generate actions for each environment
        if node.other_data is None:
            return [] # This is kind of dead end
        state: ProofState = node.other_data
        actions_scores = self.proof_action_generator.generate_actions(state)
        # Run the actions in each environment
        for i, (_, action) in enumerate(actions_scores):
            self.envs[i].step(action)
        nodes = []
        for i, env in enumerate(self.envs):
            state = env.state
            score, action = actions_scores[i]
            state_name = convert_state_to_string(state)
            nodes.append(Node(state_name, score, state))
        return nodes
    
class ProofSearhHeuristic(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def __call__(self, node: Node) -> float:
        pass

class ProofSearchDriver:
    def __init__(self, 
        search_algorithm: SearchAlgorithm,
        proof_action_generator: ProofActionGenerator,
        proof_search_heuristic: ProofSearhHeuristic,
        width: int = 4,
        logger: logging.Logger = None):
        assert search_algorithm is not None, "Search algorithm cannot be None"
        assert proof_action_generator is not None, "Proof action generator cannot be None"
        assert proof_search_heuristic is not None, "Proof search heuristic cannot be None"
        assert width > 0, "Width must be greater than 0"
        self.search_algorithm = search_algorithm
        self.proof_action_generator = proof_action_generator
        self.logger = logger if logger is not None else logging.getLogger(__name__)
        self.width = width
        self.proof_search_heuristic = proof_search_heuristic

    def search_proof(self, env: ProofEnv, timeout_in_secs: int = 60) -> typing.Tuple[Node, Node, ProofSearchResult]:
        pool = ProofEnvPool(env, self.width)
        with pool:
            envs = [pool.get(_) for _ in range(self.width)]
            start_state = envs[0].state
            # the lower the score the better
            start_goal = Node(convert_state_to_string(start_state), float('inf'), start_state)
            end_goal = Node(end_state_string(envs[0]), float('-inf'))
            branch_generator = ProofSearchBranchGenerator(envs, self.proof_action_generator)
            tree_node, found, time_taken = self.search_algorithm.search(
                start_goal,
                end_goal,
                self.proof_search_heuristic, 
                branch_generator, 
                parallel_count=self.width,
                timeout_in_secs=timeout_in_secs)
        if found:
            found_env = None
            for _env in envs:
                if _env.done:
                    found_env = _env
                    break
            found_env.dump_proof()
            # Reconstruction of the path to the root node
            # proof_path = self.search_algorithm.reconstruct_path(start_goal, tree_node)
            # proof_state : typing.List[ProofState] = [node.other_data for node in proof_path]
            # proof_steps = [TrainingDataFormat(proof_steps=state.training_data_format.proof_steps) for state in proof_state]
            return start_goal, tree_node, found_env.proof_search_res
        else:
            return start_goal, tree_node, ProofSearchResult(
                    None, 
                    False, 
                    env.lemma_name, 
                    [], 
                    time_taken, 
                    -1, 
                    possible_failed_paths=-1, 
                    num_of_backtracks=-1, 
                    is_timeout=False, 
                    is_inference_exhausted=False, 
                    longest_success_path=-1,
                    additional_info={},
                    language=env.language)