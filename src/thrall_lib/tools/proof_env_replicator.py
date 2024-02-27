import copy
import logging
import typing
import multiprocessing
from itp_interface.rl.simple_proof_env import ProofEnv, ProofAction
from itp_interface.tools.proof_exec_callback import ProofExecutorCallback

class ParallelResetCallback(object):
    def __init__(self):
        pass

    def __call__(self, sync_dict, idx: int):
        env :ProofEnv = sync_dict[idx]
        env.reset()
        # add back to the dictionary
        sync_dict[idx] = env

def replicate_proof_env(proof_env: ProofEnv, logger: typing.Optional[logging.Logger] = None) -> ProofEnv:
    new_proof_env = copy.deepcopy(proof_env)
    new_proof_env.logger = logger if logger else logging.getLogger(__name__)
    return new_proof_env

class ProofEnvPool(object):
    def __init__(self, 
            proof_env: ProofEnv, 
            pool_size: int,
            logger: typing.Optional[logging.Logger] = None):
        """
        Keeps a pool of proof environments to be used in parallel,
        and replenishes them as needed. It keeps extra environments
        in a garbage collection list to be used when the pool is
        replenished.
        """
        assert pool_size > 0, "Pool size must be greater than 0"
        self.pool_size = pool_size
        self._actual_pool_size = max(int(self.pool_size * 1.5), self.pool_size + 1)
        self._current_index = 0
        self._callback = None
        self._logger = logger if logger else logging.getLogger(__name__)
        self._frozeen_env = replicate_proof_env(proof_env, self._logger) # This is like a frozen copy we never change it
        self._proof_env_pool = []
        self._proof_envs = []
        self._gc_envs = []
        self._is_initialized = False
    
    def _parallely_reset(self, size: int, pool: list[ProofEnv]):
        # Initialize the pool
        mp_pool = multiprocessing.Pool(processes=size)
        mp_manager = multiprocessing.Manager()
        sync_dict = mp_manager.dict()
        for idx, env in enumerate(pool):
            sync_dict[idx] = env
        reset_callback = ParallelResetCallback()
        mp_pool.starmap(reset_callback, [(sync_dict, idx) for idx in range(size)])
        for _idx in range(size):
            pool[_idx] = sync_dict[_idx]
        mp_pool.close()
    
    def _sequential_reset(self, size: int, pool: list[ProofEnv]):
        assert len(pool) == size, "Pool size must be equal to the size of the pool"
        for idx in range(size):
            pool[idx].reset()
    
    def __enter__(self):
        self._proof_env_pool = [replicate_proof_env(self._frozeen_env, self._logger) for _ in range(self._actual_pool_size)]
        if self._frozeen_env.language == ProofAction.Language.LEAN:
            self._parallely_reset(self._actual_pool_size, self._proof_env_pool)
        else:
            # Parallel reset is not supported because pickling those
            # environments is not supported
            self._sequential_reset(self._actual_pool_size, self._proof_env_pool)
        for _ in range(self.pool_size):
            self._proof_envs.append(self._proof_env_pool.pop())
        self._is_initialized = True
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        for env in self._proof_env_pool:
            env.__exit__(exc_type, exc_value, traceback)
        for env in self._gc_envs:
            env.__exit__(exc_type, exc_value, traceback)
        for env in self._proof_envs:
            env.__exit__(exc_type, exc_value, traceback)
        self._proof_env_pool = []
        self._is_initialized = False
    
    def get(self, idx: int) -> ProofEnv:
        assert self._is_initialized, "Pool must be initialized"
        assert idx < self.pool_size, "Index must be less than pool size"
        return self._proof_envs[idx]
    
    def replenish(self, idx: int) -> ProofEnv:
        assert self._is_initialized, "Pool must be initialized"
        assert idx < self.pool_size, "Index must be less than pool size"
        if len(self._proof_env_pool) == 0:
            new_envs = [replicate_proof_env(self._frozeen_env, self._logger) for _ in range(len(self._gc_envs))]
            if self._frozeen_env.language == ProofAction.Language.LEAN:
                self._parallely_reset(len(new_envs), new_envs)
            else:
                self._sequential_reset(len(new_envs), new_envs)
            self._proof_env_pool.extend(new_envs)
            for env in self._gc_envs:
                env.__exit__(None, None, None)
            self._gc_envs = []
        self._gc_envs.append(self._proof_envs[idx])
        self._proof_envs[idx] = self._proof_env_pool.pop()

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
    pool = ProofEnvPool(env, 4, logger)
    with env:
        with pool:
            env1 = pool.get(0)
            env2 = pool.get(1)
            env3 = pool.get(2)
            env4 = pool.get(3)
            env1.step(ProofAction(ProofAction.ActionType.RUN_TACTIC, language, tactics=["trivial."]))
            env2.step(ProofAction(ProofAction.ActionType.RUN_TACTIC, language, tactics=["auto."]))
            env3.step(ProofAction(ProofAction.ActionType.RUN_TACTIC, language, tactics=["intro."]))
            env4.step(ProofAction(ProofAction.ActionType.RUN_TACTIC, language, tactics=["firstorder."]))
            env1.step(ProofAction(ProofAction.ActionType.RUN_TACTIC, language, tactics=["Qed."]))
            assert env1.done
            pool.replenish(0)
            env1 = pool.get(0)
            assert not env1.done
            env1.step(ProofAction(ProofAction.ActionType.RUN_TACTIC, language, tactics=["firstorder. Qed."]))
            assert env1.done
            pool.replenish(0)
            env1 = pool.get(0)
            assert not env1.done
            env1.step(ProofAction(ProofAction.ActionType.RUN_TACTIC, language, tactics=["auto. Qed."]))
            assert env1.done
            pool.replenish(0)
            env1 = pool.get(0)
            assert not env1.done
            pass