import copy
import logging
import typing
import ray
from itp_interface.rl.simple_proof_env import ProofEnv, ProofAction, ProofEnvActor, ProofState, ProofEnvInfo
from itp_interface.tools.proof_exec_callback import ProofExecutorCallback


def replicate_proof_env(proof_env: ProofEnv, logger: typing.Optional[logging.Logger] = None) -> ProofEnv:
    new_proof_env = copy.deepcopy(proof_env)
    new_proof_env.logger = logger if logger else logging.getLogger(__name__)
    return new_proof_env

class ProofEnvPool(object):
    def __init__(self, 
            pool_size: int = 1,
            proof_env_actors: typing.List[ProofEnvActor] = None,
            proof_env: ProofEnv = None, 
            logger: typing.Optional[logging.Logger] = None):
        """
        Keeps a pool of proof environments to be used in parallel,
        and replenishes them as needed. It keeps extra environments
        in a garbage collection list to be used when the pool is
        replenished.
        """
        assert pool_size > 0 or len(proof_env_actors) > 0, "Pool size must be greater than 0"
        self._current_index = 0
        self._callback = None
        self._logger = logger if logger else logging.getLogger(__name__)
        if proof_env_actors is None:
            self.pool_size = pool_size
            self._frozeen_env = replicate_proof_env(proof_env, self._logger) # This is like a frozen copy we never change it
            self._proof_env_pool : typing.List[ProofEnvActor] = [
                ProofEnvActor.remote(
                    name=self._frozeen_env.name,
                    dynamic_proof_executor_callback=self._frozeen_env.dynamic_proof_executor_callback,
                    lemma_name=self._frozeen_env.lemma_name,
                    retrieval_strategy=self._frozeen_env.retrieve_strategy,
                    max_proof_depth=self._frozeen_env.max_proof_depth,
                    always_retrieve_thms=self._frozeen_env._always_retrieve_thms,
                    logger=None
                )
                for _ in range(self.pool_size)
            ]
        else:
            self.pool_size = len(proof_env_actors)
            self._actual_pool_size = len(proof_env_actors)
            self._frozeen_env = None
            self._proof_env_pool = proof_env_actors            
        self._is_initialized = False
    
    def __enter__(self):
        self._is_initialized = True
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        self._is_initialized = False
        cleanup_remotes = [proof_env_actor.cleanup.remote() for proof_env_actor in self._proof_env_pool]
        ray.get(cleanup_remotes)
        
    def step(self, actions: typing.List[ProofAction], idxs: typing.List[int] = None) -> typing.List[typing.Tuple[ProofState, ProofAction, ProofState, float, bool, ProofEnvInfo]]:
        assert self._is_initialized, "Pool must be initialized before stepping"
        assert len(actions) == len(self._proof_env_pool) or (idxs is not None and len(actions) == len(idxs)), \
            "Number of actions must match the number of proof environments"
        if idxs is None:
            idxs = range(len(actions))
        return ray.get([self._proof_env_pool[idx].step.remote(actions[idx]) for idx in idxs])

    def get_pool(self, idxs: typing.List[int]) -> 'ProofEnvPool':
        assert self._is_initialized, "Pool must be initialized before getting"
        assert len(idxs) > 0, "Must provide at least one index"
        return ProofEnvPool( 
            proof_env_actors=[self._proof_env_pool[idx] for idx in idxs], 
            logger=self._logger)
    
    def reset(self, idxs: typing.List[int]) -> typing.List[ProofState]:
        assert self._is_initialized, "Pool must be initialized before resetting"
        assert len(idxs) > 0, "Must provide at least one index"
        return ray.get([self._proof_env_pool[idx].reset.remote() for idx in idxs])

    def get_state(self, idxs: int) -> typing.List[ProofState]:
        assert self._is_initialized, "Pool must be initialized before getting"
        assert len(idxs) > 0, "Must provide at least one index"
        return ray.get([self._proof_env_pool[idx].get_state.remote() for idx in idxs])
    
    def get_done(self, idxs: int) -> typing.List[bool]:
        assert self._is_initialized, "Pool must be initialized before getting"
        assert len(idxs) > 0, "Must provide at least one index"
        return ray.get([self._proof_env_pool[idx].get_done.remote() for idx in idxs])

if __name__ == '__main__':
    proof_exec_callback = ProofExecutorCallback(
        project_folder=".",
        file_path="src/thrall_lib/data/proofs/coq/simple2/thms.v"
    )
    theorem_name = "trival_implication"
    language = ProofAction.Language.COQ
    always_retrieve_thms = False
    logger = logging.getLogger(__name__)
    env = ProofEnv("test", proof_exec_callback, theorem_name, max_proof_depth=10, always_retrieve_thms=always_retrieve_thms, logger=logger)
    pool = ProofEnvPool(4, proof_env=env, logger=logger)
    with env:
        with pool:
            resps = pool.step([
                ProofAction(ProofAction.ActionType.RUN_TACTIC, language, tactics=["trivial."]),
                ProofAction(ProofAction.ActionType.RUN_TACTIC, language, tactics=["auto."]),
                ProofAction(ProofAction.ActionType.RUN_TACTIC, language, tactics=["intro."]),
                ProofAction(ProofAction.ActionType.RUN_TACTIC, language, tactics=["firstorder."])
            ])
            state = pool.get_state([0])[0]
            pool.step([
                ProofAction(ProofAction.ActionType.RUN_TACTIC, language, tactics=["Qed."]),
            ], [0])
            assert pool.get_done([0])[0]
            pool.reset([0])
            assert not pool.get_done([0])[0]
            _, _, next_state, _, done, _ = pool.step([
                ProofAction(ProofAction.ActionType.RUN_TACTIC, language, tactics=["firstorder. Qed."]),
            ], [0])[0]
            assert done
            assert pool.get_done([0])[0]
            pool.reset([0])
            assert not pool.get_done([0])[0]
            pass