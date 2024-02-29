#!/usr/bin/env python3

import sys
root_dir = f"{__file__.split('thrall_lib')[0]}"
if root_dir not in sys.path:
    sys.path.append(root_dir)
import typing
import logging
from abc import ABC, abstractmethod
from itp_interface.rl.simple_proof_env import ProofEnv, ProofState, ProofAction, ProofEnvInfo, ProofTree
from itp_interface.tools.dynamic_coq_proof_exec import DynamicProofExecutor as DynamicCoqExecutor
from itp_interface.rl.proof_tree import ProofSearchResult
from thrall_lib.search.search import SearchAlgorithm, Node, Edge
from thrall_lib.tools.proof_env_replicator import ProofEnvPool
from collections import namedtuple
 
# Declaring namedtuple()
ProofStateInfo = namedtuple('ProofStateInfo', ['proof_state', 'done', 'info', 'env_idx'])

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
    def generate_actions(self, state_info: ProofStateInfo, k: int = None) -> typing.List[typing.Tuple[float, ProofAction]]:
        pass

class ProofSearchBranchGenerator(ABC):
    def __init__(self, 
        envs : typing.List[ProofEnv],
        env_to_state_map: typing.List[int],
        state_to_id_map: typing.Dict[str, int], 
        proof_action_generator: ProofActionGenerator):
        assert proof_action_generator is not None, "Proof action generator cannot be None"
        assert envs is not None, "Environments cannot be None"
        assert len(envs) > 0, "Environments must contain at least one environment"
        assert len(envs) >= proof_action_generator.width, "Number of environments must match the width of the proof action generator"
        self.proof_action_generator = proof_action_generator
        self.envs = envs
        self.env_to_state_map = env_to_state_map
        self.state_id_to_env_map = {}
        self.state_to_state_id_map = state_to_id_map
        for env_idx, state_idx in enumerate(env_to_state_map):
            if state_idx not in self.state_id_to_env_map:
                self.state_id_to_env_map[state_idx] = []
            self.state_id_to_env_map[state_idx].append(env_idx)
        pass

    def __call__(self, node: Node) -> typing.Tuple[typing.List[Node], typing.List[Edge]]:
        # Call the proof action generator to generate actions for each environment
        if node.other_data is None:
            return [] # This is kind of dead end
        state_info : ProofStateInfo = node.other_data
        state: ProofState = state_info.proof_state
        state_str = convert_state_to_string(state)
        state_idx = self.state_to_state_id_map.get(state_str, -1)
        assert state_idx != -1, f"State {state_str} not found in the state to state id map"
        env_idxs : typing.List[int] = self.state_id_to_env_map.get(state_idx, [])
        if len(env_idxs) == 0:
            # This means that there is some backtracking and somewhere the original path got lost
            # Reset the environment to the state
            env_idx: int = state_info.env_idx
            env = self.envs[env_idx]
            proof_tree : ProofTree = state.proof_tree
            actions_till_state : typing.List[ProofAction] = proof_tree.actions
            env.reset()
            for action in actions_till_state:
                env.step(action)
            env_idxs.append(env_idx)
        assert len(env_idxs) > 0, f"No environments found for state {state_str}"

        actions_scores = self.proof_action_generator.generate_actions(state_info, k=len(env_idxs))
        nodes = []
        edges = []
        for i, env_idx in enumerate(env_idxs):
            env = self.envs[env_idx]
            step_tuple = env.step(actions_scores[i][1])
            if len(step_tuple) == 6:
                _, _, next_state, _, done, info  = step_tuple
            elif len(step_tuple) == 4: # This is because of bug in itp_interface which returns 4 elements when proof is done
                next_state, _, done, info = step_tuple
            else:
                raise ValueError(f"Step tuple must contain 4 or 6 elements, but contains {len(step_tuple)} = {step_tuple}")
            proof_state_info = ProofStateInfo(next_state, done, info, env_idx)
            score, action = actions_scores[i]
            state_name = convert_state_to_string(next_state)
            if state_name not in self.state_to_state_id_map:
                self.state_to_state_id_map[state_name] = len(self.state_to_state_id_map)
            edges.append(Edge('\n'.join(action.kwargs['tactics']), score, action))
            nodes.append(Node(state_name, score, proof_state_info))
        return nodes, edges
    
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
            start_state_info = ProofStateInfo(start_state, False, None, 0)
            env_to_state_map = [0 for _ in range(self.width)]
            start_state_str = convert_state_to_string(start_state)
            state_to_id_map = {start_state_str: 0}
            # the lower the score the better
            start_goal = Node(start_state_str, float('inf'), start_state_info)
            end_goal = Node(end_state_string(envs[0]), float('-inf'))
            branch_generator = ProofSearchBranchGenerator(envs, env_to_state_map, state_to_id_map, self.proof_action_generator)
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