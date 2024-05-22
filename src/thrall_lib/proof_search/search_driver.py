#!/usr/bin/env python3

import sys
root_dir = f"{__file__.split('thrall_lib')[0]}"
if root_dir not in sys.path:
    sys.path.append(root_dir)
import typing
import logging
import os
import time
import enum
from abc import ABC, abstractmethod
from itp_interface.tools.training_data_format import TrainingDataFormat, TrainingDataMetadataFormat
from itp_interface.tools.training_data import TrainingData
from itp_interface.rl.simple_proof_env import ProofEnv, ProofState, ProofAction, ProofTree, ProgressState
from itp_interface.rl.simpl_proof_env_pool import ProofEnvPool, replicate_proof_env
from itp_interface.tools.dynamic_coq_proof_exec import DynamicProofExecutor as DynamicCoqExecutor
from itp_interface.rl.proof_tree import ProofSearchResult
from thrall_lib.search.search import SearchAlgorithm, Node, Edge
from collections import namedtuple
from dataclasses import dataclass
from dataclasses_json import dataclass_json
 
# Declaring namedtuple()
ProofStateInfo = namedtuple('ProofStateInfo', ['proof_state', 'done', 'info', 'env_idx'])

class SearchPolicy(enum.Enum):
    DISCARD_FAILED_NODES = 1
    KEEP_FAILED_NODES = 2


@dataclass_json
@dataclass
class ProofPathTracer:
    collect_traces: bool
    folder: str
    training_meta_filename: str
    training_metadata: TrainingDataMetadataFormat
    max_parallelism: int

    def reset_training_data(self, logger: logging.Logger = None):
        self.training_data = TrainingData(
            self.folder, 
            self.training_meta_filename, 
            self.training_metadata, 
            self.max_parallelism, 
            logger)
    
    def trace(self, training_data_format: TrainingDataFormat):
        if self.collect_traces and hasattr(self, 'training_data'):
            self.training_data.merge(training_data_format)
    
    def save(self):
        if self.collect_traces and hasattr(self, 'training_data'):
            self.training_data.save()

def convert_state_to_string(state: typing.Union[ProofState, TrainingDataFormat]) -> str:
    tdf = state.training_data_format if isinstance(state, ProofState) else state
    goals = tdf.start_goals
    if len(goals) == 0:
        return tdf.goal_description
    else:
        all_lines = []
        goal_len = len(goals)
        for idx, goal in enumerate(tdf.start_goals):
            all_lines.append(f"({idx+1}/{goal_len})")
            for hyp in goal.hypotheses:
                all_lines.append(hyp)
            if goal.goal is not None:
                all_lines.append(f"⊢ {goal.goal}")
        return '\n'.join(all_lines)

def end_state_string(language: ProofAction.Language) -> str:
    if language == ProofAction.Language.COQ:
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
        width: int,
        envs : ProofEnvPool,
        env_to_state_map: typing.List[int],
        state_to_id_map: typing.Dict[str, int], 
        proof_action_generator: ProofActionGenerator,
        theorem_name: str,
        file_path: str,
        project_id: str,
        logger: logging.Logger,
        tracer: typing.Optional[ProofPathTracer] = None,
        original_proofs: typing.Optional[typing.List[typing.List[TrainingDataFormat]]] = None,
        search_policy: SearchPolicy = SearchPolicy.DISCARD_FAILED_NODES):
        assert proof_action_generator is not None, "Proof action generator cannot be None"
        assert envs is not None, "Environments cannot be None"
        assert envs.pool_size > 0, "Environments must contain at least one environment"
        self.width = width
        self.proof_action_generator = proof_action_generator
        self.envs = envs
        self.env_to_state_map = env_to_state_map
        self.state_id_to_env_map : typing.Dict[int, typing.Set[int]] = {}
        self.state_to_state_id_map = state_to_id_map
        self.tracer = tracer if tracer is not None else ProofPathTracer(False, "", "", TrainingDataMetadataFormat(), 1)
        self.original_proofs = original_proofs if original_proofs is not None else [[]]
        self.theorem_name = theorem_name
        self.file_path = file_path
        self.project_id = project_id
        self.original_proofs_state_names = set()
        self.state_action_map = set()
        self.logger = logger
        for proof in self.original_proofs:
            for state in proof:
                self.original_proofs_state_names.add(convert_state_to_string(state))
        for env_idx, state_idx in enumerate(env_to_state_map):
            if state_idx not in self.state_id_to_env_map:
                self.state_id_to_env_map[state_idx] = set()
            self.state_id_to_env_map[state_idx].add(env_idx)
        self.search_policy = search_policy
        pass

    def get_unused_envs(self):
        # Get all the environments for the start state
        return list(self.state_id_to_env_map[0]) if 0 in self.state_id_to_env_map else []

    def reset_envs(self, env_idxs: int, actions_till_states : typing.List[typing.List[ProofAction]], force_reset: bool = True):
        if force_reset:
            self.envs.reset(env_idxs)
        for env_idx in env_idxs:
            env_state_idx = self.env_to_state_map[env_idx]
            assert env_state_idx in self.state_id_to_env_map, f"Env state idx {env_state_idx} not found in the state id to env map"
            assert env_idx in self.state_id_to_env_map[env_state_idx], f"Env idx {env_idx} not found in the state id to env map for state {env_state_idx}"
            self.state_id_to_env_map[env_state_idx].remove(env_idx)
        new_env_state_idx = None
        # Remove the env_idx from the old state id
        actions_zipped : typing.List[typing.List[ProofAction]]= []
        env_idxs_zipped : typing.List[typing.List[int]] = []
        for _idx, actions in enumerate(actions_till_states):
            assigned_env = env_idxs[_idx]
            for __idx, action in enumerate(actions):
                if len(actions_zipped) <= __idx:
                    actions_zipped.append([])
                if len(env_idxs_zipped) <= __idx:
                    env_idxs_zipped.append([])
                env_idxs_zipped[__idx].append(assigned_env)
                actions_zipped[__idx].append(action)
        for actions_flat, env_idxs_flat in zip(actions_zipped, env_idxs_zipped):
            self.envs.step(actions_flat, env_idxs_flat)
        env_states = self.envs.get_state(env_idxs)
        for env_state, env_idx in zip(env_states, env_idxs):
            env_state_str = convert_state_to_string(env_state)
            if env_state_str not in self.state_to_state_id_map:
                new_env_state_idx = len(self.state_to_state_id_map)
                self.state_to_state_id_map[env_state_str] = new_env_state_idx
            else:
                new_env_state_idx = self.state_to_state_id_map[env_state_str]
            self.env_to_state_map[env_idx] = new_env_state_idx
            if new_env_state_idx not in self.state_id_to_env_map:
                self.state_id_to_env_map[new_env_state_idx] = set()
            self.state_id_to_env_map[new_env_state_idx].add(env_idx)

    def _check_if_within_timeout(self, start_time: float, timeout_in_secs: float, logger: logging.Logger) -> bool:
        timedout = time.time() - start_time >= timeout_in_secs
        if timedout:
            logger.info("Search timed out, returning without generating more nodes")
        return timedout

    def __call__(self, node: Node, timeout_in_secs: float) -> typing.Tuple[typing.List[Node], typing.List[Edge]]:
        # Call the proof action generator to generate actions for each environment
        if node.other_data is None or timeout_in_secs <= 0:
            return [], [] # This is kind of dead end
        # for _key, _valu in self.state_to_state_id_map.items():
        #     self.logger.info(f"State_to_state_id_map: {_valu} -> \n{_key}")
        # for _key, _valu in self.state_id_to_env_map.items():
        #     self.logger.info(f"State_id_to_env_map: {_key} -> {_valu}")
        node_gen_start_time = time.time()
        state_info : ProofStateInfo = node.other_data
        state: ProofState = state_info.proof_state
        state_str = convert_state_to_string(state)
        state_idx = self.state_to_state_id_map.get(state_str, -1)
        assert state_idx != -1, f"State {state_str} not found in the state to state id map"
        actions_scores = self.proof_action_generator.generate_actions(state_info, k=self.width)
        nodes = []
        edges = []
        if self._check_if_within_timeout(node_gen_start_time, timeout_in_secs, self.logger):
            return nodes, edges
        while len(actions_scores) > 0:
            env_idxs : typing.List[int] = list(self.state_id_to_env_map.get(state_idx, set()))
            if len(env_idxs) < len(actions_scores):

                # This means that there is some backtracking and somewhere the original path got lost
                # Reset the environment to the state
                env_idx: int = state_info.env_idx
                proof_tree : ProofTree = state.proof_tree
                actions_till_state : typing.List[ProofAction] = proof_tree.actions
                free_envs = self.get_unused_envs()
                diff = len(actions_scores) - len(env_idxs)
                if len(free_envs) == 0:
                    self.reset_envs([env_idx], [actions_till_state], force_reset=True)
                    env_idxs.append(env_idx)
                else:
                    to_reset = free_envs[:diff] # We already have sufficient free envs so no need to reset all
                    diff = len(to_reset)
                    self.reset_envs(to_reset, [actions_till_state for _ in range(diff)], force_reset=False)
                    env_idxs.extend(to_reset)
            assert len(env_idxs) > 0, f"No environments found for state {state_str}"

            actions_to_run = []
            for _idx in range(min(len(env_idxs), len(actions_scores))):
                env_idx = env_idxs[_idx]
                score, action = actions_scores.pop()
                actions_to_run.append((env_idx, score, action))
            env_idxs = [env_idx for env_idx, _, _ in actions_to_run]
            actions = [action for _, _, action in actions_to_run]
            action_start_time = time.time()
            results = self.envs.step(actions, env_idxs)
            action_end_time = time.time()
            self.logger.info(f"Finished executing {len(actions)} actions parallely in {action_end_time - action_start_time} seconds.")
            for idx, result in enumerate(results):
                if len(result) == 6:
                    _, _, next_state, _, done, info  = result
                elif len(result) == 4: # This is because of bug in itp_interface which returns 4 elements when proof is done
                    next_state, _, done, info = result
                else:
                    raise ValueError(f"Step tuple must contain 4 or 6 elements, but contains {len(result)} = {result}")
                proof_state_info = ProofStateInfo(next_state, done, info, env_idxs[idx])
                state_name = convert_state_to_string(next_state)
                action_str = '\n'.join(actions[idx].kwargs['tactics'])
                new_state_id = None
                if not done and (state_name, action_str) not in self.state_action_map:
                    self.state_action_map.add((state_name, action_str))
                    on_proof_path = state_name in self.original_proofs_state_names
                    additional_info = {
                        'done': done,
                        'progress': info.progress,
                        'error_message': info.error_message,
                        'distance_from_root': node.distance_from_root + 1,
                        'score': actions_to_run[idx][1],
                        'on_proof_path': on_proof_path
                    }
                    training_data_format = TrainingDataFormat(
                        goal_description=next_state.training_data_format.goal_description,
                        start_goals=state.training_data_format.start_goals,
                        proof_steps=actions[idx].kwargs['tactics'],
                        end_goals=next_state.training_data_format.start_goals,
                        addition_state_info=additional_info,
                        file_path=self.file_path,
                        project_id=self.project_id,
                        theorem_name=self.theorem_name
                    )
                    self.tracer.trace(training_data_format)
                if state_name not in self.state_to_state_id_map:
                    new_state_id = len(self.state_to_state_id_map)
                    self.state_to_state_id_map[state_name] = new_state_id
                else:
                    new_state_id = self.state_to_state_id_map[state_name]
                self.env_to_state_map[env_idxs[idx]] = new_state_id
                # Remove the env_idx from the old state id
                if env_idxs[idx] in self.state_id_to_env_map[state_idx]:
                    self.state_id_to_env_map[state_idx].remove(env_idxs[idx])
                if new_state_id not in self.state_id_to_env_map:
                    self.state_id_to_env_map[new_state_id] = set()
                self.state_id_to_env_map[new_state_id].add(env_idxs[idx])
                if len(next_state.training_data_format.start_goals) == 0:
                    # We found a very good action so we should signal the search to stop, regardless of the search heuristic
                    _temp_list = list(actions_to_run[idx])
                    _temp_list[1] = -100
                    actions_to_run[idx] = tuple(_temp_list)
                # TODO: Don't add the node if the action failed and the search policy is to discard failed nodes
                if (info.progress != ProgressState.FAILED and info.progress != ProgressState.STATE_UNCHANGED) or \
                    self.search_policy != SearchPolicy.DISCARD_FAILED_NODES:
                    edges.append(Edge('\n'.join(actions[idx].kwargs['tactics']), actions_to_run[idx][1], actions_to_run[idx][2]))
                    nodes.append(Node(state_name, actions_to_run[idx][1], proof_state_info))
            if self._check_if_within_timeout(node_gen_start_time, timeout_in_secs, self.logger):
                return nodes, edges
        node_gen_end_time = time.time()
        self.logger.info(f"Finished executing {len(nodes)} branches in {node_gen_end_time - node_gen_start_time} seconds")
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
        logger: logging.Logger = None,
        tracer: typing.Optional[ProofPathTracer] = None,
        search_policy: SearchPolicy = SearchPolicy.DISCARD_FAILED_NODES):
        assert search_algorithm is not None, "Search algorithm cannot be None"
        assert proof_action_generator is not None, "Proof action generator cannot be None"
        assert proof_search_heuristic is not None, "Proof search heuristic cannot be None"
        assert width > 0, "Width must be greater than 0"
        self.search_algorithm = search_algorithm
        self.proof_action_generator = proof_action_generator
        self.logger = logger if logger is not None else logging.getLogger(__name__)
        self.width = width
        self.proof_search_heuristic = proof_search_heuristic
        self.env_count = 8 * self.width # We need more environments to run in parallel without waiting
        self.tracer = tracer if tracer is not None else ProofPathTracer(False, "", "", TrainingDataMetadataFormat(), 1)
        self.search_policy = search_policy

    def search_proof(
        self, 
        env: ProofEnv, 
        timeout_in_secs: int = 60,
        dump_file_name: str = None,
        additional_info: dict = {},
        extract_original_proofs: bool = False,
        original_proofs: typing.Optional[typing.List[typing.List[TrainingDataFormat]]] = None
    ) -> typing.Tuple[Node, Node, ProofSearchResult]:
        if extract_original_proofs and original_proofs is None:
            env_cpy = replicate_proof_env(env)
            env_cpy1 = replicate_proof_env(env)
            original_proof = []
            with env_cpy:
                with env_cpy1:
                    done = False
                    while not done:
                        env_cpy._dynamic_proof_executor.run_next()
                        stmt = env_cpy._dynamic_proof_executor.current_stmt
                        proof_steps = [stmt]
                        tdf = env_cpy1.state.training_data_format
                        res = env_cpy1.step(
                            ProofAction(
                            ProofAction.ActionType.RUN_TACTIC, 
                            env_cpy.language, 
                            tactics=proof_steps))
                        if len(res) == 6:
                            _, _, _, _, done, _  = res
                        elif len(res) == 4: # This is because of bug in itp_interface which returns 4 elements when proof is done
                            _, _, done, _ = res
                        if not done:
                            tdf.proof_steps = proof_steps
                            original_proof.append(tdf)
            original_proofs = [original_proof]

        pool = ProofEnvPool(self.env_count, proof_env=env, logger=self.logger)
        with pool:
            start_state = pool.get_state([0])[0]
            start_state_info = ProofStateInfo(start_state, False, None, 0)
            env_to_state_map = [0 for _ in range(self.env_count)]
            start_state_str = convert_state_to_string(start_state)
            state_to_id_map = {start_state_str: 0}
            # the lower the score the better
            start_goal = Node(start_state_str, 100, start_state_info)
            language : ProofAction.Language = pool._get_attr('language', [0])[0]
            end_goal = Node(end_state_string(language), -100)
            theorem_name = env.lemma_name
            file_path = env.dynamic_proof_executor_callback.file_path
            project_id = env.dynamic_proof_executor_callback.project_folder 
            branch_generator = ProofSearchBranchGenerator(
                self.width, 
                pool, 
                env_to_state_map, 
                state_to_id_map, 
                self.proof_action_generator, 
                theorem_name,
                file_path,
                project_id,
                self.logger,
                self.tracer,
                original_proofs)
            tree_node, found, time_taken = self.search_algorithm.search(
                start_goal,
                end_goal,
                self.proof_search_heuristic, 
                branch_generator, 
                parallel_count=self.width,
                timeout_in_secs=timeout_in_secs,
                logger=self.logger)
            lemma_name = pool._get_attr('_lemma_name_with_stmt', [0])[0]
            full_path = env.dynamic_proof_executor_callback.file_path
            if found:
                last_state: ProofState = tree_node.other_data.proof_state
                proof_tree: ProofTree = last_state.proof_tree
                actions_till_state: typing.List[ProofAction] = proof_tree.tactics
                proof_steps = [TrainingDataFormat(proof_steps=tactic.proof_steps) for _, tactic in actions_till_state]
                proof_search_res = ProofSearchResult(
                            full_path,
                            True, 
                            lemma_name,
                            proof_steps, 
                            time_taken, 
                            -1, 
                            possible_failed_paths=-1, 
                            num_of_backtracks=-1, 
                            is_timeout=False, 
                            is_inference_exhausted=False, 
                            longest_success_path=-1,
                            additional_info=additional_info,
                            language=language)
            else:
                proof_search_res = ProofSearchResult(
                    full_path, 
                    False, 
                    lemma_name, 
                    [], 
                    time_taken, 
                    -1, 
                    possible_failed_paths=-1, 
                    num_of_backtracks=-1, 
                    is_timeout=time_taken >= timeout_in_secs, 
                    is_inference_exhausted=False, 
                    longest_success_path=-1,
                    additional_info=additional_info,
                    language=language)
        if dump_file_name is not None:
            opening_mode = 'a' if os.path.exists(dump_file_name) else 'w'
            with open(dump_file_name, opening_mode) as f:
                if opening_mode == 'a':
                    f.write("\n\n")
                f.write(str(proof_search_res))
        return start_goal, tree_node, proof_search_res